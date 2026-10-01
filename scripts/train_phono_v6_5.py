#!/usr/bin/env python3
"""Training script for Phono-V6.5 Hierarchical Word-to-Byte Recursive Denoising Decoder."""

import argparse
import copy
from datetime import datetime
import json
import math
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio.transforms as T
import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.registry import ModelRegistry
from src.models.phono_v6_5_hierarchical_decoder import (
    HierarchicalByteConfig,
    PhonoV65HierarchicalByteDecoder,
    SLOT_SILENCE,
    SLOT_WORD,
    SLOT_EOS,
    SLOT_SPACE,
)
from src.data.byte_tokenizer import ByteTokenizer


class MultilingualByteDataset(Dataset):
    """Loads audio and encodes text hierarchically into word-byte sequences."""

    def __init__(
        self,
        manifest_path: str,
        tokenizer: ByteTokenizer,
        max_duration_sec: float = 20.0,
        min_duration_sec: float = 1.0,
        max_words_per_utt: int = 64,
        max_bytes_per_word: int = 24,
    ):
        self.tokenizer = tokenizer
        self.max_duration_sec = max_duration_sec
        self.min_duration_sec = min_duration_sec
        self.max_words_per_utt = max_words_per_utt
        self.max_bytes_per_word = max_bytes_per_word

        print(f"📖 Loading manifest: {manifest_path}...")
        with open(manifest_path, "r", encoding="utf-8") as f:
            raw_samples = json.load(f)

        self.samples = [
            s for s in raw_samples
            if min_duration_sec <= s.get("duration", 0) <= max_duration_sec
            and len(s.get("transcript", "").strip()) > 0
        ]
        print(f"✅ Filtered {len(self.samples):,} valid samples from {len(raw_samples):,} total ({manifest_path})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        item = self.samples[idx]
        wav_np, sr = sf.read(item["audio_path"], dtype="float32")
        wav = torch.from_numpy(wav_np)
        if wav.ndim > 1:
            wav = wav.mean(dim=-1)

        if sr != 16000:
            resampler = T.Resample(sr, 16000)
            wav = resampler(wav.unsqueeze(0)).squeeze(0)

        raw_text = item.get("transcript", "").strip()
        word_byte_seqs = self.tokenizer.encode_words(raw_text)[:self.max_words_per_utt]

        return {
            "id": item.get("id", f"sample_{idx}"),
            "audio": wav,
            "audio_len": wav.shape[-1],
            "word_byte_seqs": word_byte_seqs,
            "num_words": len(word_byte_seqs),
            "text": raw_text,
        }


class HierarchicalByteCollateFn:
    """Collation function padding audio, word slots, and byte sequences with macro slot targets."""

    def __init__(self, tokenizer: ByteTokenizer, max_bytes_per_word: int = 24):
        self.tokenizer = tokenizer
        self.max_bytes_per_word = max_bytes_per_word

    def __call__(self, batch: List[Dict[str, any]]) -> Dict[str, any]:
        B = len(batch)
        audio_lens = torch.tensor([item["audio_len"] for item in batch], dtype=torch.long)
        max_audio_len = audio_lens.max().item()

        word_counts = [max(1, item["num_words"]) for item in batch]
        max_words = max(word_counts)
        total_slots = max_words + 1  # Extra slot for explicit EOS target

        # Determine max bytes across all words in this batch
        max_bytes = 0
        for item in batch:
            for w in item["word_byte_seqs"]:
                max_bytes = max(max_bytes, len(w))
        max_bytes = min(max_bytes + 1, self.max_bytes_per_word)  # +1 for BOS

        padded_audio = torch.zeros(B, max_audio_len, dtype=torch.float32)
        padded_input_bytes = torch.full((B, total_slots, max_bytes), self.tokenizer.pad_id, dtype=torch.long)
        padded_target_bytes = torch.full((B, total_slots, max_bytes), self.tokenizer.pad_id, dtype=torch.long)
        slot_targets = torch.full((B, total_slots), SLOT_SILENCE, dtype=torch.long)

        for i, item in enumerate(batch):
            a_len = item["audio_len"]
            padded_audio[i, :a_len] = item["audio"]

            w_count = item["num_words"]
            for w_idx, w_bytes in enumerate(item["word_byte_seqs"]):
                # Truncate if exceeds limit
                cur_w = w_bytes[:max_bytes - 1]
                # Teacher forcing: input is [BOS, b1, b2, ...], target is [b1, b2, ..., EOW]
                in_seq = [self.tokenizer.bos_id] + cur_w
                tgt_seq = cur_w
                k = len(tgt_seq)

                padded_input_bytes[i, w_idx, :k] = torch.tensor(in_seq[:k], dtype=torch.long)
                padded_target_bytes[i, w_idx, :k] = torch.tensor(tgt_seq, dtype=torch.long)
                slot_targets[i, w_idx] = SLOT_WORD

            if w_count < total_slots:
                slot_targets[i, w_count] = SLOT_EOS

        return {
            "audio": padded_audio,
            "audio_lens": audio_lens,
            "input_byte_ids": padded_input_bytes,
            "target_byte_ids": padded_target_bytes,
            "slot_targets": slot_targets,
            "num_words": torch.tensor(word_counts, dtype=torch.long),
            "texts": [item["text"] for item in batch],
        }


def evaluate_v6_5(
    decoder: PhonoV65HierarchicalByteDecoder,
    backbone: nn.Module,
    tokenizer: ByteTokenizer,
    val_loader: DataLoader,
    device: torch.device,
    max_eval_batches: int = 15,
) -> Dict[str, any]:
    """Evaluate Phono-V6.5 on validation split."""
    decoder.eval()
    backbone.eval()

    total_loss = 0.0
    total_samples = 0
    total_tokens = 0
    correct_tokens = 0
    predictions = []
    references = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_eval_batches:
                break
            audio = batch["audio"].to(device)
            audio_lens = batch["audio_lens"].to(device)
            in_bytes = batch["input_byte_ids"].to(device)
            tgt_bytes = batch["target_byte_ids"].to(device)
            num_words = batch["num_words"].to(device)
            slot_targets = batch.get("slot_targets")
            if slot_targets is not None:
                slot_targets = slot_targets.to(device)
            refs = batch["texts"]

            # Frozen acoustic extraction
            out_bb = backbone(audio=audio, audio_lengths=audio_lens)
            acoustic_mem = out_bb["hidden_state"]
            mem_lens = out_bb["input_lengths"]

            with torch.amp.autocast("cuda"):
                out_dec = decoder(
                    acoustic_memory=acoustic_mem,
                    memory_lengths=mem_lens,
                    num_words=num_words,
                    input_byte_ids=in_bytes,
                    target_byte_ids=tgt_bytes,
                    slot_targets=slot_targets,
                )
                loss = out_dec["loss"]
                logits = out_dec["logits"]

            total_loss += loss.item() * audio.size(0)
            total_samples += audio.size(0)

            # Token byte accuracy
            flat_preds = logits.view(-1, tokenizer.vocab_size).argmax(dim=-1)
            flat_tgts = tgt_bytes.view(-1)
            mask = flat_tgts != tokenizer.pad_id
            if mask.sum() > 0:
                correct_tokens += (flat_preds[mask] == flat_tgts[mask]).sum().item()
                total_tokens += mask.sum().item()

            # Autoregressive generation on the first batch
            if batch_idx == 0:
                gen_word_bytes = decoder.generate(
                    acoustic_memory=acoustic_mem,
                    memory_lengths=mem_lens,
                    max_words=num_words.max().item(),
                    max_bytes_per_word=16,
                )
                for i in range(min(5, len(refs))):
                    pred_str = tokenizer.decode_words(gen_word_bytes[i])
                    predictions.append(pred_str)
                    references.append(refs[i].lower())

    avg_loss = total_loss / max(1, total_samples)
    byte_acc = (correct_tokens / max(1, total_tokens)) * 100.0

    wer = 100.0
    if predictions and references:
        try:
            wer = round(float(jiwer.wer(references, predictions)) * 100.0, 2)
        except Exception:
            wer = 100.0

    return {
        "val_loss": round(avg_loss, 4),
        "val_acc": round(byte_acc, 2),
        "val_wer": wer,
        "sample_pred": predictions[0] if predictions else "",
        "sample_ref": references[0] if references else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Train Phono-V6.5 Hierarchical Word-to-Byte Decoder")
    parser.add_argument("--backbone_ckpt", type=str, default="checkpoints/phono_v6_4_gated_diffusion/medium/v6_4_960h/best_checkpoint.pt")
    parser.add_argument("--train_manifest", type=str, default="data/librispeech/benchmark_train_960h.json")
    parser.add_argument("--val_manifest", type=str, default="data/librispeech/benchmark_val.json")
    parser.add_argument("--tier", type=str, default="medium", choices=["medium", "large"], help="Architecture tier (medium: 16 experts top-2, large: 32 experts top-4)")
    parser.add_argument("--micro_num_experts", type=int, default=16, help="Experts in micro byte head (default: 16)")
    parser.add_argument("--micro_moe_top_k", type=int, default=2, help="Top-K sparse routing in micro head (default: 2)")
    parser.add_argument("--disable_macro_fastpath", action="store_true", default=False, help="Disable Macro Fast-Path routing")
    parser.add_argument("--steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per step")
    parser.add_argument("--accum_steps", type=int, default=2, help="Gradient accumulation steps (effective batch 16)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Linear warmup steps")
    parser.add_argument("--eval_interval", type=int, default=500, help="Steps between validation evaluations")
    parser.add_argument("--max_duration_sec", type=float, default=20.0, help="Max duration filter")
    parser.add_argument("--output_dir", type=str, default="checkpoints/phono_v6_5_hierarchical/medium")
    parser.add_argument("--only_save_best", action="store_true", default=True, help="Only save best checkpoint to protect SSD")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("🚀 PHONO-V6.5: HIERARCHICAL WORD-TO-BYTE RECURSIVE DENOISING DECODER")
    print(f"Device: {device} | Tier: {args.tier.upper()} | Total Steps: {args.steps} | Batch Size: {args.batch_size} (Effective: {args.batch_size * args.accum_steps})")
    print(f"Backbone: {args.backbone_ckpt}")
    print(f"Output: {output_dir}")
    print("=" * 75)

    # 1. Byte Tokenizer
    tokenizer = ByteTokenizer()
    print(f"✅ Universal ByteTokenizer initialized (Vocab: {tokenizer.vocab_size} tokens, 100% UTF-8 coverage).")

    # 2. Frozen Acoustic Backbone
    print(f"📦 Loading frozen acoustic backbone from {args.backbone_ckpt}...")
    backbone_ckpt = torch.load(args.backbone_ckpt, map_location=device, weights_only=False)
    arch = backbone_ckpt.get("arch", "phono_v6_4_gated_diffusion")
    backbone_cls = ModelRegistry.get_entry(arch)["model_cls"]
    backbone = backbone_cls(backbone_ckpt["config"]).to(device)
    backbone.load_state_dict(backbone_ckpt["model_state_dict"])
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    print(f"🔒 Frozen acoustic backbone initialized ({arch}, ~83.88M parameters).")

    # 3. Phono-V6.5 Model Configured by Tier
    enable_fastpath = not args.disable_macro_fastpath
    if args.tier == "large":
        config = HierarchicalByteConfig.large(
            byte_vocab_size=tokenizer.vocab_size,
            enable_macro_fastpath=enable_fastpath,
        )
    else:
        config = HierarchicalByteConfig.medium(
            byte_vocab_size=tokenizer.vocab_size,
            micro_num_experts=args.micro_num_experts,
            micro_moe_top_k=args.micro_moe_top_k,
            enable_macro_fastpath=enable_fastpath,
        )

    decoder = PhonoV65HierarchicalByteDecoder(config).to(device)
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    emb_size_kb = decoder.micro_byte_head.byte_embedding.weight.numel() * 2 / 1024
    print(f"✨ Phono-V6.5-MoE Initialized: {decoder_params:,} trainable parameters ({decoder_params/1e6:.2f}M).")
    print(f"🧠 Micro Byte Head: {config.micro_layers} Layers, {config.micro_num_experts} Experts (Top-{config.micro_moe_top_k} Routing), d={config.micro_dim}.")
    print(f"⚡ Macro Fast-Path Router: {'Enabled (Instant EOS Early-Exit + Silence Bypass)' if config.enable_macro_fastpath else 'Disabled'}")
    print(f"💾 Byte Embedding Table: {emb_size_kb:.1f} KB in FP16 (vs 10.2 MB in fixed word models).")

    # 4. Data Loaders
    train_dataset = MultilingualByteDataset(
        manifest_path=args.train_manifest,
        tokenizer=tokenizer,
        max_duration_sec=args.max_duration_sec,
    )
    val_dataset = MultilingualByteDataset(
        manifest_path=args.val_manifest,
        tokenizer=tokenizer,
        max_duration_sec=args.max_duration_sec,
    )

    collate_fn = HierarchicalByteCollateFn(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 5. Optimizer and LR Scheduler
    optimizer = torch.optim.AdamW(
        [p for p in decoder.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=1e-2,
    )
    scaler = torch.amp.GradScaler("cuda")

    def lr_lambda(current_step: int) -> float:
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        progress = float(current_step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 6. Training Loop
    best_val_wer = float("inf")
    best_val_loss = float("inf")
    step = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    print("\n🚀 Starting Phono-V6.5 Hierarchical Training Loop...")
    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break

            decoder.train()
            audio = batch["audio"].to(device, non_blocking=True)
            audio_lens = batch["audio_lens"].to(device, non_blocking=True)
            in_bytes = batch["input_byte_ids"].to(device, non_blocking=True)
            tgt_bytes = batch["target_byte_ids"].to(device, non_blocking=True)
            num_words = batch["num_words"].to(device, non_blocking=True)
            slot_targets = batch.get("slot_targets")
            if slot_targets is not None:
                slot_targets = slot_targets.to(device, non_blocking=True)
            refs = batch["texts"]

            # Frozen acoustic extraction
            with torch.no_grad():
                out_bb = backbone(audio=audio, audio_lengths=audio_lens)
                acoustic_mem = out_bb["hidden_state"]
                mem_lens = out_bb["input_lengths"]

            with torch.amp.autocast("cuda"):
                out_dec = decoder(
                    acoustic_memory=acoustic_mem,
                    memory_lengths=mem_lens,
                    num_words=num_words,
                    input_byte_ids=in_bytes,
                    target_byte_ids=tgt_bytes,
                    slot_targets=slot_targets,
                )
                loss = out_dec["loss"] / args.accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            step += 1

            # Free batch references immediately
            batch_acc = out_dec["byte_acc"].item()
            raw_loss = loss.item() * args.accum_steps
            diff_val = out_dec["diff_loss"].item()
            aux_val = out_dec["aux_loss"].item()

            # Compute live training batch WER every 10 steps
            live_wer_str = "N/A"
            if step % 10 == 0 or step == 1:
                with torch.no_grad():
                    logits = out_dec["logits"]
                    preds_ids = logits.argmax(dim=-1)  # [B, L, K]
                    pred_texts = []
                    for b_idx in range(min(4, audio.size(0))):
                        utt_words = []
                        for w_idx in range(num_words[b_idx].item()):
                            b_seq = preds_ids[b_idx, w_idx].tolist()
                            w_str = tokenizer.decode(b_seq, skip_special=True)
                            if w_str:
                                utt_words.append(w_str)
                        pred_texts.append(" ".join(utt_words))
                    
                    sub_refs = [refs[b_idx].lower() for b_idx in range(len(pred_texts))]
                    try:
                        w_val = jiwer.wer(sub_refs, pred_texts) * 100.0
                        live_wer_str = f"{w_val:5.1f}%"
                    except Exception:
                        live_wer_str = "N/A"

            del audio, audio_lens, in_bytes, tgt_bytes, num_words, acoustic_mem, mem_lens
            if step % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Logging with Accuracy and WER
            if step % 10 == 0 or step == 1:
                cur_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                steps_per_sec = step / max(1e-5, elapsed)
                remaining_sec = (args.steps - step) / max(1e-5, steps_per_sec)
                eta_min = int(remaining_sec / 60)
                print(
                    f"[V6.5 Step {step:5d}/{args.steps}] Loss: {raw_loss:.4f} | "
                    f"Acc: {batch_acc:5.1f}% | WER: {live_wer_str} | "
                    f"Diff: {diff_val:.4f} | Aux: {aux_val:.3f} | "
                    f"LR: {cur_lr:.2e} | Speed: {steps_per_sec:.2f} it/s | ETA: {eta_min}m"
                )

            # Validation & Checkpointing
            if step % args.eval_interval == 0 or step == args.steps:
                print(f"\n--- [V6.5 Step {step}] Evaluating on Validation Set ---")
                eval_metrics = evaluate_v6_5(
                    decoder=decoder,
                    backbone=backbone,
                    tokenizer=tokenizer,
                    val_loader=val_loader,
                    device=device,
                    max_eval_batches=15,
                )
                print(
                    f"📊 Step {step} Validation -> Loss: {eval_metrics['val_loss']:.4f} | "
                    f"Byte Acc: {eval_metrics['val_acc']:.2f}% | WER: {eval_metrics['val_wer']:.2f}%"
                )
                if eval_metrics["sample_pred"]:
                    print(f"   • Ref : {eval_metrics['sample_ref'][:70]}...")
                    print(f"   • Pred: {eval_metrics['sample_pred'][:70]}...")

                if eval_metrics["val_wer"] < best_val_wer or eval_metrics["val_loss"] < best_val_loss:
                    if eval_metrics["val_wer"] < best_val_wer:
                        best_val_wer = eval_metrics["val_wer"]
                    if eval_metrics["val_loss"] < best_val_loss:
                        best_val_loss = eval_metrics["val_loss"]

                    best_ckpt_path = output_dir / "best_checkpoint.pt"
                    save_payload = {
                        "step": step,
                        "arch": "phono_v6_5_hierarchical_decoder",
                        "config": config,
                        "model_state_dict": decoder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "val_acc": eval_metrics["val_acc"],
                        "val_wer": best_val_wer,
                        "saved_at": time.time(),
                    }
                    torch.save(save_payload, best_ckpt_path)
                    print(f"🌟 [New Best Model Saved] Step {step} achieved lowest Val WER: {best_val_wer:.2f}% (Loss: {best_val_loss:.4f}) -> {best_ckpt_path.name}")

                # Save latest checkpoint
                latest_ckpt_path = output_dir / "checkpoint_latest.pt"
                save_payload = {
                    "step": step,
                    "arch": "phono_v6_5_hierarchical_decoder",
                    "config": config,
                    "model_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": eval_metrics["val_loss"],
                    "val_acc": eval_metrics["val_acc"],
                    "val_wer": eval_metrics["val_wer"],
                    "saved_at": time.time(),
                }
                torch.save(save_payload, latest_ckpt_path)
                print(f"💾 Checkpoint saved at step {step} -> {latest_ckpt_path.name}\n")

    print("\n" + "=" * 75)
    print("🏁 PHONO-V6.5 TRAINING COMPLETED!")
    print(f"Best Val Loss: {best_val_loss:.4f} | Best Val WER: {best_val_wer:.2f}%")
    print(f"Best model saved to: {output_dir / 'best_checkpoint.pt'}")
    print("=" * 75)


if __name__ == "__main__":
    main()
