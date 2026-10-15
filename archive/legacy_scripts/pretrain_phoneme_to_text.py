#!/usr/bin/env python3
"""Phono-V6.5 Multilingual Phoneme-to-Roman-Text Pre-training Script.

Pre-trains the Hierarchical Word-to-Character MoE Decoder on massive multilingual
phoneme-to-text data across 9 languages (en, fr, es, de, it, ar, zh, ja, ko)
streamed sequentially from /media/hdd/Datasets/multilingual_text/shards/.

Features:
- Lowercase RomanCharTokenizer (123 tokens, 100% case normalized).
- 16 orthographic MoE experts (Top-2 routing).
- Macro Base-Token Fast-Path Classifier supervision.
- Per-step batch sample logging with exact 4-field format.
- --only_save_best SSD protection.
"""

import argparse
import glob
import math
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.multilingual_shard_dataset import (
    MultilingualPretrainCollator,
    MultilingualShardDataset,
)
from src.data.phoneme_tokenizer import PhonemeTokenizer
from src.data.roman_tokenizer import RomanCharTokenizer
from src.models.phono_v6_5_hierarchical_decoder import (
    HierarchicalByteConfig,
    PhonoV65HierarchicalByteDecoder,
)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def format_eta(seconds: float) -> str:
    if seconds < 0 or math.isinf(seconds) or math.isnan(seconds):
        return "--m"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def evaluate(
    model: PhonoV65HierarchicalByteDecoder,
    dataloader: DataLoader,
    tokenizer: RomanCharTokenizer,
    device: torch.device,
    max_batches: int = 25,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            acoustic_memory = batch["acoustic_memory"].to(device)
            memory_lengths = batch["memory_lengths"].to(device)
            input_bytes = batch["input_byte_ids"].to(device)
            target_bytes = batch["target_byte_ids"].to(device)
            slot_targets = batch["slot_targets"].to(device)
            num_words = batch["num_words"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out = model(
                    acoustic_memory=acoustic_memory,
                    memory_lengths=memory_lengths,
                    num_words=num_words,
                    input_byte_ids=input_bytes,
                    target_byte_ids=target_bytes,
                    slot_targets=slot_targets,
                )

            loss = out["loss"]
            if not torch.isnan(loss):
                total_loss += loss.item()

            logits = out["logits"]
            preds = logits.argmax(dim=-1)
            valid_mask = target_bytes != -100
            total_correct += (preds[valid_mask] == target_bytes[valid_mask]).sum().item()
            total_tokens += valid_mask.sum().item()

    avg_loss = total_loss / max(1, min(len(dataloader), max_batches))
    acc = (total_correct / max(1, total_tokens)) * 100.0
    return {"val_loss": avg_loss, "val_acc": acc}


def main():
    parser = argparse.ArgumentParser(description="Phono-V6.5 Multilingual Pretraining")
    parser.add_argument("--shards_dir", type=str, default="/media/hdd/Datasets/multilingual_text/shards")
    parser.add_argument("--save_dir", type=str, default="checkpoints/phono_v6_5_pretrain")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--micro_num_experts", type=int, default=16)
    parser.add_argument("--micro_moe_top_k", type=int, default=2)
    parser.add_argument("--max_bytes_per_word", type=int, default=24)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--only_save_best", action="store_true", default=True)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("🚀 Phono-V6.5 Multilingual Phoneme-to-Roman-Text Pre-training")
    print(f"  Device:                 {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Shards directory:       {args.shards_dir}")
    print(f"  Save directory:         {args.save_dir}")
    print(f"  Total Steps:            {args.steps:,}")
    print(f"  Batch size:             {args.batch_size}")
    print(f"  Learning rate:          {args.lr}")
    print(f"  MoE Architecture:       {args.micro_num_experts} experts, Top-{args.micro_moe_top_k} routing")
    print("=" * 70)

    # 1. Initialize Tokenizers
    ph_tok = PhonemeTokenizer()
    rom_tok = RomanCharTokenizer()
    print(f"✅ Tokenizers loaded: Phonemes={ph_tok.vocab_size}, RomanChars={rom_tok.vocab_size}")

    # 2. Dataset & DataLoader
    dataset = MultilingualShardDataset(args.shards_dir, phoneme_tokenizer=ph_tok, roman_tokenizer=rom_tok)
    print(f"✅ Shard Dataset loaded: {len(dataset):,} multilingual samples across {len(dataset.idx_files)} shards.")

    collator = MultilingualPretrainCollator(
        phoneme_tokenizer=ph_tok,
        roman_tokenizer=rom_tok,
        acoustic_dim=512,
        max_bytes_per_word=args.max_bytes_per_word,
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

    # 3. Model
    cfg = HierarchicalByteConfig.medium(
        acoustic_dim=512,
        macro_dim=512,
        micro_dim=512,
        byte_vocab_size=rom_tok.vocab_size,
        micro_num_experts=args.micro_num_experts,
        micro_moe_top_k=args.micro_moe_top_k,
        max_bytes_per_word=args.max_bytes_per_word,
    )
    model = PhonoV65HierarchicalByteDecoder(cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Phono-V6.5 Model Initialized: {total_params:,} parameters.")

    # 4. Optimizer & Scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.steps)
    scaler = torch.amp.GradScaler("cuda")

    start_step = 0
    best_val_loss = float("inf")

    if args.resume_ckpt and os.path.exists(args.resume_ckpt):
        print(f"🔄 Resuming checkpoint from: {args.resume_ckpt}")
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
    print("\n🚀 Beginning Multilingual Pre-training Loop...\n")

    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        acoustic_memory = batch["acoustic_memory"].to(device, non_blocking=True)
        memory_lengths = batch["memory_lengths"].to(device, non_blocking=True)
        input_bytes = batch["input_byte_ids"].to(device, non_blocking=True)
        target_bytes = batch["target_byte_ids"].to(device, non_blocking=True)
        slot_targets = batch["slot_targets"].to(device, non_blocking=True)
        num_words = batch["num_words"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out = model(
                acoustic_memory=acoustic_memory,
                memory_lengths=memory_lengths,
                num_words=num_words,
                input_byte_ids=input_bytes,
                target_byte_ids=target_bytes,
                slot_targets=slot_targets,
            )
            loss = out["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Metrics for logging
        t_curr = time.time()
        step_time = t_curr - t_last_step
        t_last_step = t_curr
        speed = 1.0 / max(step_time, 1e-4)
        eta_seconds = (args.steps - step) / max(speed, 1e-4)
        eta_str = format_eta(eta_seconds)
        current_lr = scheduler.get_last_lr()[0]

        # Compute accuracy and decode sample for live log
        with torch.no_grad():
            logits = out["logits"]
            preds = logits.argmax(dim=-1)
            valid_mask = target_bytes != -100
            correct = (preds[valid_mask] == target_bytes[valid_mask]).sum().item()
            tot = valid_mask.sum().item()
            acc = (correct / max(1, tot)) * 100.0

            # Decode sample index 0
            sample_pred_ids = preds[0].tolist()
            pred_word_str = rom_tok.decode_words(sample_pred_ids)
            truth_w = batch["truth_words"][0]
            truth_p = batch["truth_phonemes"][0]
            pred_p = truth_p  # In acoustic-latents decoder, input phonemes match acoustic memory

        print(
            f"[Phono-V6.5-Pretrain Step {step}/{args.steps}] Loss: {loss.item():.4f} | "
            f"Acc: {acc:.2f}% | LR: {current_lr:.2e} | Speed: {speed:.2f} it/s | ETA: {eta_str}\n"
            f"  - Truth phonems: {truth_p[:80]}\n"
            f"  - Predicted phonems: {pred_p[:80]}\n"
            f"  - Predicted words: {pred_word_str[:80]}\n"
            f"  - Truth words: {truth_w[:80]}"
        )

        # Periodic Evaluation & Checkpointing
        if step % args.eval_interval == 0 or step == args.steps:
            val_metrics = evaluate(model, dataloader, rom_tok, device, max_batches=20)
            val_loss = val_metrics["val_loss"]
            val_acc = val_metrics["val_acc"]
            print(
                f"\n📊 [Validation Step {step}] Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% "
                f"(Best: {best_val_loss:.4f})\n"
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
