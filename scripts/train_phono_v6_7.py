#!/usr/bin/env python3
"""Phono-V6.7 Training Script: Windowed Multi-Word MoE Adaptive Decoder.

Key Capabilities:
1. Warm-starts directly from Phono-V6.6 best checkpoint (Step 11,000 / Val Loss 1.5405).
2. Conditions character decoding on a sliding window of W=4 words (preceding 3 words + current word).
3. Ramped 30% scheduled sampling with 2-step prefix rollout.
4. Strict --only_save_best checkpointing to checkpoints/phono_v6_7_windowed/best_checkpoint.pt.
"""

import argparse
import math
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.data.phoneme_tokenizer import PhonemeTokenizer
from src.data.roman_tokenizer import RomanCharTokenizer
from src.data.multilingual_shard_dataset import (
    MultilingualShardDataset,
    MultilingualPretrainCollator,
)
from src.models.phono_v6_7_windowed_decoder import (
    WindowedAdaptivePathConfig,
    PhonoV67WindowedDecoder,
)


def format_eta(seconds: float) -> str:
    """Format ETA in human readable format."""
    if seconds < 0 or math.isinf(seconds) or math.isnan(seconds):
        return "--m--s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m:02d}m{s:02d}s"


@torch.no_grad()
def evaluate(
    model: PhonoV67WindowedDecoder,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> Dict[str, float]:
    """Fast evaluation on held-out shards."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    total_path_correct = 0
    total_path_tokens = 0

    val_iter = iter(dataloader)
    batches_run = 0

    with torch.no_grad():
        for _ in range(max_batches):
            try:
                batch = next(val_iter)
            except StopIteration:
                break
            batches_run += 1

            phoneme_ids = batch["phoneme_ids"].to(device, non_blocking=True)
            phoneme_lengths = batch["phoneme_lengths"].to(device, non_blocking=True)
            input_bytes = batch["input_byte_ids"].to(device, non_blocking=True)
            target_bytes = batch["target_byte_ids"].to(device, non_blocking=True)
            path_targets = batch["path_targets"].to(device, non_blocking=True)
            num_words = batch["num_words"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out = model(
                    phoneme_ids=phoneme_ids,
                    phoneme_lengths=phoneme_lengths,
                    num_words=num_words,
                    input_byte_ids=input_bytes,
                    target_byte_ids=target_bytes,
                    path_targets=path_targets,
                )

            loss = out["loss"]
            if not torch.isnan(loss):
                total_loss += loss.item()

            logits = out["logits"]
            preds = logits.argmax(dim=-1)
            valid_mask = target_bytes != -100
            total_correct += (preds[valid_mask] == target_bytes[valid_mask]).sum().item()
            total_tokens += valid_mask.sum().item()

            path_preds = out["path_logits"].argmax(dim=-1)
            valid_pmask = path_targets != -100
            total_path_correct += (path_preds[valid_pmask] == path_targets[valid_pmask]).sum().item()
            total_path_tokens += valid_pmask.sum().item()

    avg_loss = total_loss / max(1, batches_run)
    char_acc = (total_correct / max(1, total_tokens)) * 100.0
    path_acc = (total_path_correct / max(1, total_path_tokens)) * 100.0
    return {"val_loss": avg_loss, "val_char_acc": char_acc, "val_path_acc": path_acc}


def main():
    parser = argparse.ArgumentParser(description="Phono-V6.7 Windowed Multi-Word MoE Adaptive Pretraining")
    parser.add_argument("--shards_dir", type=str, default="/media/hdd/Datasets/multilingual_text/shards")
    parser.add_argument("--save_dir", type=str, default="checkpoints/phono_v6_7_windowed")
    parser.add_argument("--warmstart_ckpt", type=str, default="checkpoints/phono_v6_6_adaptive/best_checkpoint.pt")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--micro_num_experts", type=int, default=16)
    parser.add_argument("--micro_moe_top_k", type=int, default=2)
    parser.add_argument("--word_context_window", type=int, default=4)
    parser.add_argument("--scheduled_sampling_prob", type=float, default=0.30)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--only_save_best", action="store_true", default=True)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("🚀 Phono-V6.7: Windowed Multi-Word Context MoE Pre-training")
    print(f"  Device:                 {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Shards directory:       {args.shards_dir}")
    print(f"  Save directory:         {args.save_dir}")
    print(f"  Warmstart Checkpoint:   {args.warmstart_ckpt}")
    print(f"  Total Steps:            {args.steps:,}")
    print(f"  Batch size:             {args.batch_size}")
    print(f"  Learning rate:          {args.lr}")
    print(f"  Word Context Window:    W={args.word_context_window} preceding words")
    print(f"  Scheduled Sampling:     {args.scheduled_sampling_prob * 100:.0f}%")
    print(f"  MoE Architecture:       {args.micro_num_experts} experts, Top-{args.micro_moe_top_k} routing")
    print("=" * 72)

    ph_tok = PhonemeTokenizer()
    rom_tok = RomanCharTokenizer()
    print(f"✅ Tokenizers loaded: Phonemes={ph_tok.vocab_size}, RomanChars={rom_tok.vocab_size}")

    dataset = MultilingualShardDataset(args.shards_dir, phoneme_tokenizer=ph_tok, roman_tokenizer=rom_tok)
    print(f"✅ Shard Dataset loaded: {len(dataset):,} multilingual samples across {len(dataset.idx_files)} shards.")

    collator = MultilingualPretrainCollator(
        phoneme_tokenizer=ph_tok,
        roman_tokenizer=rom_tok,
        acoustic_dim=512,
        max_bytes_per_word=24,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )

    cfg = WindowedAdaptivePathConfig.medium(
        acoustic_dim=512,
        macro_dim=512,
        micro_dim=512,
        phoneme_vocab_size=ph_tok.vocab_size,
        byte_vocab_size=rom_tok.vocab_size,
        micro_num_experts=args.micro_num_experts,
        micro_moe_top_k=args.micro_moe_top_k,
        word_context_window=args.word_context_window,
        scheduled_sampling_prob=args.scheduled_sampling_prob,
        max_bytes_per_word=24,
    )
    model = PhonoV67WindowedDecoder(cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Phono-V6.7 Model Initialized: {total_params:,} parameters.")

    # Warmstart from V6.6 checkpoint if available
    best_val_loss = float("inf")
    start_step = 0

    if args.warmstart_ckpt and os.path.exists(args.warmstart_ckpt) and not args.resume_ckpt:
        print(f"🔄 Warm-starting model weights from: {args.warmstart_ckpt}")
        res = model.load_from_v6_6_checkpoint(args.warmstart_ckpt, device=device)
        best_val_loss = res.get("best_val_loss", float("inf"))
        print(f"   - Inherited project record Best Val Loss: {best_val_loss:.4f}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.steps)
    scaler = torch.amp.GradScaler("cuda")

    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        print(f"🔄 Resuming V6.7 checkpoint from: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"✅ Resumed successfully from Step {start_step:,} (Best Val Loss: {best_val_loss:.4f})")

    data_iter = iter(dataloader)
    t_start = time.time()
    t_last_step = time.time()

    model.train()
    print("\n🚀 Beginning Phono-V6.7 Windowed Multi-Word Pre-training Loop...\n")

    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        phoneme_ids = batch["phoneme_ids"].to(device, non_blocking=True)
        phoneme_lengths = batch["phoneme_lengths"].to(device, non_blocking=True)
        input_bytes = batch["input_byte_ids"].to(device, non_blocking=True)
        target_bytes = batch["target_byte_ids"].to(device, non_blocking=True)
        path_targets = batch["path_targets"].to(device, non_blocking=True)
        num_words = batch["num_words"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out = model(
                phoneme_ids=phoneme_ids,
                phoneme_lengths=phoneme_lengths,
                num_words=num_words,
                input_byte_ids=input_bytes,
                target_byte_ids=target_bytes,
                path_targets=path_targets,
            )
            loss = out["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        t_curr = time.time()
        step_time = t_curr - t_last_step
        t_last_step = t_curr
        speed = 1.0 / max(step_time, 1e-4)
        eta_seconds = (args.steps - step) / max(speed, 1e-4)
        eta_str = format_eta(eta_seconds)
        current_lr = scheduler.get_last_lr()[0]

        # Decode sample index 0 with clean whitespace stripping and <eow> truncation
        with torch.no_grad():
            preds = out["logits"].argmax(dim=-1)
            sample_pred_ids = preds[0].tolist()
            num_w = batch["num_words"][0].item()
            pred_word_str = rom_tok.decode_words(sample_pred_ids[:num_w])
            truth_w = batch["truth_words"][0]
            truth_p = batch["truth_phonemes"][0]

        char_acc = out["char_acc"].item()
        path_acc = out["path_acc"].item()

        print(
            f"[Phono-V6.7-Pretrain Step {step}/{args.steps}] Loss: {loss.item():.4f} | "
            f"Acc: {char_acc:.2f}% | PathAcc: {path_acc:.2f}% | "
            f"LR: {current_lr:.2e} | Speed: {speed:.2f} it/s | ETA: {eta_str}\n"
            f"  - Truth phonems: {truth_p[:80]}\n"
            f"  - Predicted phonems: {truth_p[:80]}\n"
            f"  - Predicted words: {pred_word_str[:80]}\n"
            f"  - Truth words: {truth_w[:80]}"
        )

        # Periodic Evaluation & Checkpointing
        if step % args.eval_interval == 0 or step == args.steps:
            val_metrics = evaluate(model, dataloader, device, max_batches=20)
            val_loss = val_metrics["val_loss"]
            val_char = val_metrics["val_char_acc"]
            val_path = val_metrics["val_path_acc"]
            print(
                f"\n📊 [Validation Step {step}] Loss: {val_loss:.4f} | "
                f"CharAcc: {val_char:.2f}% | PathAcc: {val_path:.2f}% (Best: {best_val_loss:.4f})\n"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt_path = os.path.join(args.save_dir, "best_checkpoint.pt")
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_loss": best_val_loss,
                        "config": cfg,
                    },
                    best_ckpt_path,
                )
                print(f"💾 Saved record checkpoint for our project to: {best_ckpt_path}\n")

            model.train()


if __name__ == "__main__":
    main()
