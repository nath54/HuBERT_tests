"""
Official Training Pipeline for NaVox Model Part 1: Acoustic Conformer MoE (Model_A2P).
Supports all model tiers (tiny, small, base, large, ultra), automatic gradient checkpointing,
JSONL metrics tracking, and real-time training curve rendering.
"""

# Import Modules
import argparse
import datetime
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import editdistance
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

# Root path resolution
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from src.model_config import ModelConfig, ModelA2PConfig
from src.model1_a2p import Model_A2P
from src.weights_manager import WeightsManager
from src.training.train_logger import TrainLogger
from src.training.dataset import NaVoxAudioDataset, NaVoxCollator, MultilingualWeightedSampler
from src.tokenizers import PhonemeTokenizer, RomanCharTokenizer


class NaVoxAcousticTrainer:
    """
    Modular, Object-Oriented Trainer for NaVox Acoustic Models (A2P).
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initialize Trainer with CLI arguments.
        """
        self.args: argparse.Namespace = args
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Initialize Logger
        self.logger: TrainLogger = TrainLogger(
            log_dir=args.log_dir,
            run_name=args.run_name,
            model_tier=args.tier,
        )

        # 2. Tokenizers
        self.tokenizer: PhonemeTokenizer = PhonemeTokenizer()
        self.rom_tokenizer: RomanCharTokenizer = RomanCharTokenizer()

        # 3. Model Configuration & Instantiation
        self.logger.log(f"🏗️  Building NaVox ModelConfig for Tier: `{args.tier.upper()}`")
        tier_method = getattr(ModelConfig, args.tier.lower(), None)
        if tier_method is None:
            raise ValueError(f"Unknown tier: `{args.tier}`. Choose from 'tiny', 'small', 'base', 'large', 'ultra'.")
        
        self.model_config: ModelConfig = tier_method(device_model_a2p="CUDA" if torch.cuda.is_available() else "CPU")
        if args.gradient_checkpointing:
            self.model_config.a2p.gradient_checkpointing = True

        # Instantiate Model 1 (A2P)
        self.model: Model_A2P = Model_A2P(self.model_config).to(self.device)
        self.total_params: int = sum(p.numel() for p in self.model.parameters())
        self.logger.log(f"   • Total Parameters: {self.total_params / 1e6:.2f} M")
        self.logger.log(f"   • Conformer Layers: {self.model_config.a2p.num_layers} | Hidden Dim: {self.model_config.a2p.embed_dim}")
        self.logger.log(f"   • DeepSeek MoE: {self.model_config.a2p.use_deepseek_moe} ({self.model_config.a2p.num_routed_experts} Routed Experts, Top-{self.model_config.a2p.top_k})")
        self.logger.log(f"   • Gradient Checkpointing: {self.model_config.a2p.gradient_checkpointing}")

        # 4. Warm-Start from Checkpoint if requested
        if args.warm_start:
            self._warm_start(args.warm_start)

        # 5. Datasets & DataLoaders
        self.logger.log(f"📦 Loading training dataset: {args.train_manifest}")
        self.train_dataset: NaVoxAudioDataset = NaVoxAudioDataset(
            manifest_path=args.train_manifest,
            max_duration=args.max_duration,
        )
        self.val_dataset: NaVoxAudioDataset = NaVoxAudioDataset(
            manifest_path=args.val_manifest,
            max_duration=args.max_duration,
        )
        self.logger.log(f"   • Train Samples: {len(self.train_dataset):,} | Val Samples: {len(self.val_dataset):,}")

        self.collator: NaVoxCollator = NaVoxCollator(
            phoneme_tokenizer=self.tokenizer,
            pad_id=self.model_config.a2p.pad_id,
            blank_id=self.model_config.a2p.blank_id,
            space_id=self.model_config.a2p.space_id,
        )

        # Balanced Multilingual Samplers (35% French, 25% English, 20% Spanish, 20% Italian)
        lang_weights = {"FR": 0.35, "EN": 0.25, "ES": 0.20, "IT": 0.20}
        self.sampler: MultilingualWeightedSampler = MultilingualWeightedSampler(
            samples=self.train_dataset.samples,
            language_weights=lang_weights,
            seed=42,
        )
        self.logger.log(f"   • Balanced Multilingual Training Sampler Active: {lang_weights}")

        val_sampler: MultilingualWeightedSampler = MultilingualWeightedSampler(
            samples=self.val_dataset.samples,
            language_weights={"FR": 0.25, "EN": 0.25, "ES": 0.25, "IT": 0.25},
            seed=42,
        )

        self.train_loader: DataLoader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            sampler=self.sampler,
            num_workers=args.num_workers,
            collate_fn=self.collator,
            pin_memory=True,
        )
        self.val_loader: DataLoader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            collate_fn=self.collator,
        )

        # 6. Optimizer & LR Scheduler
        if getattr(args, "optimizer", "adafactor").lower() == "adafactor":
            from transformers.optimization import Adafactor
            self.logger.log("⚡ Optimizer: Adafactor (Factored 2nd moments, < 0.5 GB state memory)")
            self.optimizer = Adafactor(
                self.model.parameters(),
                lr=args.lr,
                eps=(1e-30, 1e-3),
                clip_threshold=args.max_grad_norm,
                decay_rate=-0.8,
                beta1=None,
                weight_decay=args.weight_decay,
                relative_step=False,
                scale_parameter=False,
                warmup_init=False,
            )
        else:
            self.logger.log("⚡ Optimizer: AdamW")
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
                betas=(0.9, 0.98),
                eps=1e-8,
            )

        total_opt_steps = max(1, args.max_steps // args.grad_accum)
        warmup_opt_steps = max(1, args.warmup_steps // args.grad_accum)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_opt_steps,
            num_training_steps=total_opt_steps,
        )

        # Stop time handling
        self.stop_timestamp: Optional[float] = None
        if getattr(args, "max_hours", None) is not None and args.max_hours > 0:
            self.stop_timestamp = time.time() + args.max_hours * 3600.0
            self.logger.log(f"⏰ Auto-stop enabled: will stop in {args.max_hours:.2f} hours")
        elif getattr(args, "stop_time", None) is not None:
            now = datetime.datetime.now()
            target_hour, target_min = map(int, args.stop_time.split(":"))
            target_dt = now.replace(hour=target_hour, minute=target_min, second=0, microsecond=0)
            if target_dt <= now:
                target_dt += datetime.timedelta(days=1)
            self.stop_timestamp = target_dt.timestamp()
            hours_remaining = (self.stop_timestamp - time.time()) / 3600.0
            self.logger.log(f"⏰ Target stop time configured: {target_dt.strftime('%Y-%m-%d %H:%M:%S')} ({hours_remaining:.2f} hours from now)")

        # Best metrics tracking
        self.best_per: float = 999.0
        self.step: int = 0
        os.makedirs(args.output_dir, exist_ok=True)

    def _warm_start(self, ckpt_path: str) -> None:
        """Loads compatible weights into the model."""
        if not os.path.isfile(ckpt_path):
            self.logger.log(f"⚠️  Warm start checkpoint not found: `{ckpt_path}`, starting from scratch.")
            return

        self.logger.log(f"🔄 Warm-starting from: `{ckpt_path}`")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        sd = ckpt.get("acoustic_state_dict", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))

        # Filter out mismatched tensor sizes
        model_sd = self.model.state_dict()
        filtered_sd = {}
        for k, v in sd.items():
            clean_k = k.replace("module.", "").replace("backbone.", "")
            if clean_k in model_sd and model_sd[clean_k].shape == v.shape:
                filtered_sd[clean_k] = v

        missing, unexpected = self.model.load_state_dict(filtered_sd, strict=False)
        self.logger.log(f"   • Loaded {len(filtered_sd)}/{len(model_sd)} parameters cleanly.")

    def train(self) -> None:
        """
        Executes the main training loop with multi-task loss and gradient accumulation.
        """
        self.logger.log(f"🔥 Starting Official NaVox-{self.args.tier.upper()} Training...")
        self.logger.log(f"   • Max Steps: {self.args.max_steps:,} | Batch Size: {self.args.batch_size} (Grad Accum: {self.args.grad_accum})")
        self.logger.log(f"   • Effective Batch Size: {self.args.batch_size * self.args.grad_accum} utterances")

        self.model.train()
        accum_loss = 0.0
        start_time = time.time()
        train_iter = iter(self.train_loader)

        while self.step < self.args.max_steps:
            if self.stop_timestamp is not None and time.time() >= self.stop_timestamp:
                self.logger.log(
                    f"⏰ Target stop time reached ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})! "
                    f"Wrapping up pilot training cleanly at step {self.step}..."
                )
                break

            self.step += 1

            # Fetch next batch (cycle endlessly)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Move tensors to GPU
            audio = batch["audio"].to(self.device, non_blocking=True)
            audio_lens = batch["audio_lengths"].to(self.device, non_blocking=True)
            targets = batch["targets"].to(self.device, non_blocking=True)
            target_lens = batch["target_lengths"].to(self.device, non_blocking=True)

            # Forward Training Pass
            out = self.model.forward_training(
                audio=audio,
                audio_lengths=audio_lens,
                targets=targets,
                target_lengths=target_lens,
            )

            loss = out["loss"] / self.args.grad_accum
            loss.backward()
            accum_loss += out["loss"].item()

            # Optimizer Step
            if self.step % self.args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # Periodic Terminal Progress Log
            if self.step % self.args.log_every == 0:
                avg_loss = accum_loss / self.args.log_every
                accum_loss = 0.0
                lr_curr = self.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start_time
                steps_per_sec = self.step / max(1.0, elapsed)
                eta_sec = (self.args.max_steps - self.step) / max(0.01, steps_per_sec)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))

                self.logger.log(
                    f"Step {self.step:>6}/{self.args.max_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr_curr:.2e} | "
                    f"{steps_per_sec:.2f} step/s | "
                    f"ETA: {eta_str}"
                )
                self.logger.log_train_step(self.step, {
                    "loss": avg_loss,
                    "lr": lr_curr,
                    "steps_per_sec": steps_per_sec,
                })

            # Validation Step
            if self.step % self.args.eval_every == 0:
                self.evaluate()
                self.model.train()

        # Run final evaluation after training finishes or reaches stop_time
        if self.step % self.args.eval_every != 0:
            self.evaluate()
        self.logger.log("🎉 Pilot Training Completed Successfully!")
        self.logger.close()

    @torch.no_grad()
    def evaluate(self) -> None:
        """
        Executes validation, computes PER, Space Accuracy, and renders progress curves.
        """
        self.model.eval()
        self.logger.log(f"⏳ Running Validation Evaluation at Step {self.step}...")

        total_val_loss = 0.0
        total_phon_dist = 0
        total_phon_ref = 0
        total_spaces_correct = 0
        total_spaces_ref = 0

        lang_stats: Dict[str, Dict[str, int]] = {}
        samples_inspection: List[Dict[str, Any]] = []

        val_batches = 0
        max_batches = self.args.max_val_batches or len(self.val_loader)

        for b_idx, batch in enumerate(self.val_loader):
            if b_idx >= max_batches:
                break
            val_batches += 1

            audio = batch["audio"].to(self.device)
            audio_lens = batch["audio_lengths"].to(self.device)
            targets = batch["targets"].to(self.device)
            target_lens = batch["target_lengths"].to(self.device)

            out = self.model.forward_training(
                audio=audio,
                audio_lengths=audio_lens,
                targets=targets,
                target_lengths=target_lens,
            )
            total_val_loss += out["loss"].item()

            # Greedy inference for phoneme evaluation
            inf_out = self.model.forward_inference(audio=audio)
            pred_tokens_list = inf_out["pred_tokens"]

            for i in range(len(pred_tokens_list)):
                ref_toks = targets[i, :target_lens[i]].cpu().tolist()
                hyp_toks = pred_tokens_list[i]
                lang = batch["languages"][i]

                # Phoneme Error Rate (PER)
                dist = editdistance.eval(ref_toks, hyp_toks)
                total_phon_dist += dist
                total_phon_ref += len(ref_toks)

                if lang not in lang_stats:
                    lang_stats[lang] = {"dist": 0, "ref": 0}
                lang_stats[lang]["dist"] += dist
                lang_stats[lang]["ref"] += len(ref_toks)

                # Space accuracy (token 8)
                ref_spaces = sum(1 for t in ref_toks if t == self.model_config.a2p.space_id)
                hyp_spaces = sum(1 for t in hyp_toks if t == self.model_config.a2p.space_id)
                total_spaces_correct += max(0, min(ref_spaces, hyp_spaces))
                total_spaces_ref += max(1, ref_spaces)

                # Collect a few samples for inspection
                if len(samples_inspection) < 10:
                    samples_inspection.append({
                        "lang": lang,
                        "gt_text": batch["texts"][i],
                        "pred_text": "",  # Model 1 produces phonemes
                        "gt_phonemes": self.tokenizer.decode(ref_toks) if hasattr(self.tokenizer, "decode") else str(ref_toks),
                        "pred_phonemes": self.tokenizer.decode(hyp_toks) if hasattr(self.tokenizer, "decode") else str(hyp_toks),
                        "per": (dist / max(1, len(ref_toks))) * 100.0,
                    })

        avg_loss = total_val_loss / max(1, val_batches)
        overall_per = (total_phon_dist / max(1, total_phon_ref)) * 100.0
        space_acc = (total_spaces_correct / max(1, total_spaces_ref)) * 100.0

        lang_breakdown = {
            lang: {"per": (m["dist"] / max(1, m["ref"])) * 100.0}
            for lang, m in lang_stats.items()
        }

        # Log & Render Dashboard Curves
        val_metrics = {
            "loss": avg_loss,
            "per": overall_per,
            "acoustic_per": overall_per,
            "space_acc": space_acc,
            "languages": lang_breakdown,
        }
        self.logger.log_val_step(self.step, val_metrics, samples=samples_inspection)

        # Checkpoint Saving
        if overall_per < self.best_per:
            self.best_per = overall_per
            best_ckpt_path = os.path.join(self.args.output_dir, f"best_navox_{self.args.tier}_a2p.pt")
            torch.save({
                "step": self.step,
                "model_state_dict": self.model.state_dict(),
                "config": self.model_config,
                "per": overall_per,
                "space_acc": space_acc,
            }, best_ckpt_path)
            self.logger.log(f"🏆 New Best Checkpoint Saved! (PER: {overall_per:.2f}%) -> `{best_ckpt_path}`")

        # Save latest checkpoint
        latest_ckpt_path = os.path.join(self.args.output_dir, f"latest_navox_{self.args.tier}_a2p.pt")
        torch.save({
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "config": self.model_config,
            "per": overall_per,
        }, latest_ckpt_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Official NaVox A2P Acoustic Training Suite")
    parser.add_argument("--tier", type=str, default="ultra", choices=["tiny", "small", "base", "large", "ultra"])
    parser.add_argument("--optimizer", type=str, default="adafactor", choices=["adafactor", "adamw"], help="Optimizer type")
    parser.add_argument("--train_manifest", type=str, default="data/full_multilingual_train_400k.json")
    parser.add_argument("--val_manifest", type=str, default="data/full_multilingual_val_4k.json")
    parser.add_argument("--output_dir", type=str, default="checkpoints/navox_ultra_full")
    parser.add_argument("--log_dir", type=str, default="logs/navox_ultra_full")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--warm_start", type=str, default="checkpoints/navox_ultra_pilot/best_navox_ultra_a2p.pt")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8.0e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=80000)
    parser.add_argument("--max_duration", type=float, default=14.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--max_val_batches", type=int, default=60)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--stop_time", type=str, default=None, help="Target stop time in HH:MM (e.g. '10:00'). Gracefully finishes at that time.")
    parser.add_argument("--max_hours", type=float, default=None, help="Maximum training wall-clock hours before stopping cleanly.")

    args = parser.parse_args()
    trainer = NaVoxAcousticTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
