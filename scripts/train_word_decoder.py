#!/usr/bin/env python3
"""Training script for Phoneme-to-Word Denoising Decoder on top of frozen V6.4 acoustic representations."""

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
from src.models.word_denoising_decoder import WordDecoderConfig, WordDenoisingDecoder
from src.data.word_tokenizer import WordTokenizer


class LibriSpeechWordDataset(Dataset):
    """Loads LibriSpeech audio and tokenizes transcripts with WordTokenizer."""

    def __init__(
        self,
        manifest_path: str,
        tokenizer: WordTokenizer,
        max_duration_sec: float = 20.0,
        min_duration_sec: float = 1.0,
    ):
        self.tokenizer = tokenizer
        self.max_duration_sec = max_duration_sec
        self.min_duration_sec = min_duration_sec

        print(f"📖 Loading manifest: {manifest_path}...")
        with open(manifest_path, "r", encoding="utf-8") as f:
            raw_samples = json.load(f)

        # Filter by duration and valid transcript
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

        # Tokenize with BOS and EOS
        raw_text = item.get("transcript", "").strip().lower()
        tokens = self.tokenizer.encode(raw_text, add_bos=True, add_eos=True)

        return {
            "id": item.get("id", f"sample_{idx}"),
            "audio": wav,
            "audio_len": wav.shape[-1],
            "input_word_ids": torch.tensor(tokens[:-1], dtype=torch.long),  # [BOS, w1, w2, ...]
            "target_word_ids": torch.tensor(tokens[1:], dtype=torch.long),  # [w1, w2, ..., EOS]
            "word_len": len(tokens) - 1,
            "text": raw_text,
        }


class WordCollateFn:
    """Dynamic padding collation for audio and word sequences."""

    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict[str, any]]) -> Dict[str, any]:
        B = len(batch)
        audio_lens = torch.tensor([item["audio_len"] for item in batch], dtype=torch.long)
        word_lens = torch.tensor([item["word_len"] for item in batch], dtype=torch.long)

        max_audio_len = audio_lens.max().item()
        max_word_len = word_lens.max().item()

        padded_audio = torch.zeros(B, max_audio_len, dtype=torch.float32)
        padded_input_ids = torch.full((B, max_word_len), self.pad_id, dtype=torch.long)
        padded_target_ids = torch.full((B, max_word_len), self.pad_id, dtype=torch.long)

        for i, item in enumerate(batch):
            a_len = item["audio_len"]
            w_len = item["word_len"]
            padded_audio[i, :a_len] = item["audio"]
            padded_input_ids[i, :w_len] = item["input_word_ids"]
            padded_target_ids[i, :w_len] = item["target_word_ids"]

        return {
            "audio": padded_audio,
            "audio_lens": audio_lens,
            "input_word_ids": padded_input_ids,
            "target_word_ids": padded_target_ids,
            "word_lens": word_lens,
            "texts": [item["text"] for item in batch],
        }


def evaluate_word_decoder(
    decoder: WordDenoisingDecoder,
    backbone: nn.Module,
    tokenizer: WordTokenizer,
    val_loader: DataLoader,
    device: torch.device,
    max_eval_batches: int = 20,
) -> Dict[str, float]:
    """Evaluate Word Denoising Decoder on validation batches."""
    decoder.eval()
    backbone.eval()

    total_loss = 0.0
    total_samples = 0
    predictions = []
    references = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_eval_batches:
                break
            audio = batch["audio"].to(device)
            audio_lens = batch["audio_lens"].to(device)
            input_ids = batch["input_word_ids"].to(device)
            target_ids = batch["target_word_ids"].to(device)
            refs = batch["texts"]

            # Extract frozen acoustic representations
            out_backbone = backbone(audio=audio, audio_lengths=audio_lens)
            acoustic_mem = out_backbone["hidden_state"]
            mem_lens = out_backbone["input_lengths"]

            with torch.amp.autocast("cuda"):
                out_dec = decoder(
                    word_ids=input_ids,
                    acoustic_memory=acoustic_mem,
                    memory_lengths=mem_lens,
                    target_word_ids=target_ids,
                )
                loss = out_dec["loss"]

            total_loss += loss.item() * audio.size(0)
            total_samples += audio.size(0)

            # Autoregressive generation on the first batch
            if batch_idx == 0:
                gen_ids = decoder.generate(acoustic_mem, memory_lengths=mem_lens, max_len=60)
                for i in range(min(5, len(refs))):
                    pred_tokens = gen_ids[i].tolist()
                    pred_str = tokenizer.decode(pred_tokens, skip_special=True)
                    predictions.append(pred_str)
                    references.append(refs[i])

    avg_loss = total_loss / max(1, total_samples)

    wer = 100.0
    if predictions and references:
        try:
            wer = round(float(jiwer.wer(references, predictions)) * 100.0, 2)
        except Exception:
            wer = 100.0

    return {
        "val_loss": round(avg_loss, 4),
        "val_wer": wer,
        "sample_pred": predictions[0] if predictions else "",
        "sample_ref": references[0] if references else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Train Phoneme-to-Word Denoising Decoder")
    parser.add_argument("--backbone_ckpt", type=str, default="checkpoints/phono_v6_4_gated_diffusion/medium/v6_4_960h/best_checkpoint.pt")
    parser.add_argument("--vocab_path", type=str, default="data/word_vocab_10k.json")
    parser.add_argument("--train_manifest", type=str, default="data/librispeech/benchmark_train_960h.json")
    parser.add_argument("--val_manifest", type=str, default="data/librispeech/benchmark_val.json")
    parser.add_argument("--steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per step")
    parser.add_argument("--accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Linear warmup steps")
    parser.add_argument("--eval_interval", type=int, default=500, help="Steps between validation evaluations")
    parser.add_argument("--max_duration_sec", type=float, default=20.0, help="Max duration filter")
    parser.add_argument("--output_dir", type=str, default="checkpoints/word_denoising_decoder/medium")
    parser.add_argument("--only_save_best", action="store_true", default=True, help="Only save best checkpoint to protect SSD")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 WORD DENOISING DECODER TRAINING (MoE + Latent Diffusion)")
    print(f"Device: {device} | Total Steps: {args.steps} | Batch Size: {args.batch_size} (Accum: {args.accum_steps})")
    print(f"Backbone: {args.backbone_ckpt}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # 1. Tokenizer
    tokenizer = WordTokenizer(args.vocab_path)
    print(f"✅ Loaded WordTokenizer with {tokenizer.vocab_size} tokens from {args.vocab_path}")

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

    # 3. Word Denoising Decoder
    decoder_cfg = WordDecoderConfig(
        vocab_size=tokenizer.vocab_size,
        word_embed_dim=512,
        acoustic_embed_dim=512,
        decoder_layers=4,
        decoder_heads=8,
        decoder_ffn_dim=1536,
        num_experts=4,
        moe_top_k=2,
        cross_attn_band_width=32,
        use_word_diffusion=True,
    )
    decoder = WordDenoisingDecoder(decoder_cfg).to(device)
    decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"✨ Word Denoising Decoder initialized: {decoder_params:,} trainable parameters ({decoder_params/1e6:.2f}M).")

    # 4. Datasets and Loaders
    train_dataset = LibriSpeechWordDataset(
        manifest_path=args.train_manifest,
        tokenizer=tokenizer,
        max_duration_sec=args.max_duration_sec,
    )
    val_dataset = LibriSpeechWordDataset(
        manifest_path=args.val_manifest,
        tokenizer=tokenizer,
        max_duration_sec=args.max_duration_sec,
    )

    collate_fn = WordCollateFn(pad_id=tokenizer.pad_id)
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

    # 5. Optimizer, Scaler, and LR Scheduler
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
    best_val_loss = float("inf")
    best_val_wer = float("inf")
    step = 0
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)

    print("\n🚀 Starting training loop...")
    while step < args.steps:
        for batch in train_loader:
            if step >= args.steps:
                break

            decoder.train()
            audio = batch["audio"].to(device, non_blocking=True)
            audio_lens = batch["audio_lens"].to(device, non_blocking=True)
            input_ids = batch["input_word_ids"].to(device, non_blocking=True)
            target_ids = batch["target_word_ids"].to(device, non_blocking=True)

            # Frozen acoustic representations
            with torch.no_grad():
                out_backbone = backbone(audio=audio, audio_lengths=audio_lens)
                acoustic_mem = out_backbone["hidden_state"]
                mem_lens = out_backbone["input_lengths"]

            with torch.amp.autocast("cuda"):
                out_dec = decoder(
                    word_ids=input_ids,
                    acoustic_memory=acoustic_mem,
                    memory_lengths=mem_lens,
                    target_word_ids=target_ids,
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
            del audio, audio_lens, input_ids, target_ids, acoustic_mem, mem_lens
            if step % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Logging
            if step % 10 == 0 or step == 1:
                cur_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                steps_per_sec = step / max(1e-5, elapsed)
                remaining_sec = (args.steps - step) / max(1e-5, steps_per_sec)
                eta_min = int(remaining_sec / 60)
                raw_loss = loss.item() * args.accum_steps
                print(
                    f"[Step {step:5d}/{args.steps}] Loss: {raw_loss:.4f} | "
                    f"Diff: {out_dec['diff_loss'].item():.4f} | Aux: {out_dec['aux_loss'].item():.3f} | "
                    f"LR: {cur_lr:.2e} | Speed: {steps_per_sec:.2f} it/s | ETA: {eta_min}m"
                )

            # Evaluation & Checkpoint
            if step % args.eval_interval == 0 or step == args.steps:
                print(f"\n--- [Step {step}] Evaluating on Validation Set ---")
                eval_metrics = evaluate_word_decoder(
                    decoder=decoder,
                    backbone=backbone,
                    tokenizer=tokenizer,
                    val_loader=val_loader,
                    device=device,
                    max_eval_batches=20,
                )
                print(f"📊 Step {step} Validation -> Loss: {eval_metrics['val_loss']:.4f} | WER: {eval_metrics['val_wer']:.2f}%")
                if eval_metrics["sample_pred"]:
                    print(f"   • Ref : {eval_metrics['sample_ref'][:70]}...")
                    print(f"   • Pred: {eval_metrics['sample_pred'][:70]}...")

                if eval_metrics["val_loss"] < best_val_loss:
                    best_val_loss = eval_metrics["val_loss"]
                    best_val_wer = eval_metrics["val_wer"]
                    best_ckpt_path = output_dir / "best_checkpoint.pt"
                    save_payload = {
                        "step": step,
                        "config": decoder_cfg,
                        "model_state_dict": decoder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "val_wer": best_val_wer,
                        "saved_at": time.time(),
                    }
                    torch.save(save_payload, best_ckpt_path)
                    print(f"🌟 [New Best Model Saved] Lowest Val Loss: {best_val_loss:.4f} -> {best_ckpt_path.name}")

                # Save latest checkpoint
                latest_ckpt_path = output_dir / "checkpoint_latest.pt"
                save_payload = {
                    "step": step,
                    "config": decoder_cfg,
                    "model_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": eval_metrics["val_loss"],
                    "saved_at": time.time(),
                }
                torch.save(save_payload, latest_ckpt_path)
                print(f"💾 Checkpoint saved at step {step} -> {latest_ckpt_path.name}\n")

    print("\n" + "=" * 70)
    print("🏁 WORD DENOISING DECODER TRAINING COMPLETED!")
    print(f"Best Val Loss: {best_val_loss:.4f} | Best Val WER: {best_val_wer:.2f}%")
    print(f"Best model saved to: {output_dir / 'best_checkpoint.pt'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
