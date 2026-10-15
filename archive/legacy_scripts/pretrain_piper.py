#!/usr/bin/env python3
"""HuBERT Self-Supervised Pre-training using On-The-Fly Streaming Piper Voices (MADGen).

Streams multi-speaker synthetic speech directly in RAM (0 bytes of disk space for audio),
extracts k-means acoustic unit pseudo-labels, and trains HuBERT with masked unit prediction.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.config import HuBERTConfig
from src.models.hubert_pretrain import HuBERTForPreTraining
from src.data.streaming_piper import (
    PiperVoiceManager,
    ProceduralTextSampler,
    AcousticUnitExtractor,
    PiperStreamingDataset,
    collate_pretrain_batch,
)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.05):
    """Cosine learning rate scheduler with linear warmup."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description="Pre-train HuBERT with on-the-fly streaming Piper voices.")
    parser.add_argument("--steps", type=int, default=100, help="Total pre-training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--warmup-steps", type=int, default=15, help="Warmup steps")
    parser.add_argument("--num-clusters", type=int, default=100, help="Number of k-means acoustic clusters")
    parser.add_argument("--mask-prob", type=float, default=0.08, help="Mask span probability")
    parser.add_argument("--mask-length", type=int, default=10, help="Consecutive frames per mask span")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every", type=int, default=25, help="Save checkpoint every N steps")
    parser.add_argument("--checkpoint-out", type=str, default="checkpoints/hubert_piper_pretrained.pt")
    parser.add_argument("--log-out", type=str, default="logs/pretrain_history.json")
    args = parser.parse_args()

    print("=" * 85)
    print("      🧪 HUBERT SELF-SUPERVISED PRE-TRAINING: STREAMING PIPER VOICES (0-DISK)")
    print("=" * 85)
    print(f"Device:               {args.device}")
    print(f"Total Steps:          {args.steps}")
    print(f"Batch Size:           {args.batch_size}")
    print(f"Peak Learning Rate:   {args.lr}")
    print(f"Acoustic Clusters:    {args.num_clusters}")
    print(f"Audio Disk Usage:     0 BYTES (Generated in RAM on-the-fly)")
    print("-" * 85)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 1. Initialize On-The-Fly Streaming Data Pipeline
    print("[1/3] Initializing Piper voice manager & procedural text sampler...")
    voice_mgr = PiperVoiceManager()
    text_sampler = ProceduralTextSampler()
    unit_extractor = AcousticUnitExtractor(num_clusters=args.num_clusters)

    dataset = PiperStreamingDataset(
        voice_manager=voice_mgr,
        text_sampler=text_sampler,
        unit_extractor=unit_extractor,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_pretrain_batch,
    )

    # 2. Warm-start K-Means Acoustic Clusters with first few samples
    print("[2/3] Calibrating k-means acoustic codebook from initial in-memory speech...")
    warmup_features = []
    for _ in range(8):
        text = text_sampler.sample_sentence()
        wf, _, _ = voice_mgr.synthesize_to_tensor_16k(text)
        mfcc = unit_extractor.compute_mfcc_features(wf)
        warmup_features.append(mfcc)
    unit_extractor.warm_start_clusters(warmup_features)

    # 3. Initialize Model & Optimizer
    print("[3/3] Instantiating HuBERT pre-training model...")
    config = HuBERTConfig(
        encoder_layers=4,
        encoder_heads=4,
        encoder_embed_dim=256,
        encoder_ffn_dim=1024,
    )
    model = HuBERTForPreTraining(config=config, num_clusters=args.num_clusters).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.device == "cuda"))

    # Training state
    history = []
    total_audio_sec = 0.0
    start_time = time.time()

    print("\n" + "=" * 85)
    print(f"{'Step':<8} | {'Loss':<8} | {'Mask Acc (%)':<14} | {'LR':<10} | {'RAM Audio':<12} | {'Disk Space':<12} | {'Time'}")
    print("-" * 85)

    step = 0
    data_iter = iter(dataloader)

    while step < args.steps:
        step += 1
        t_step0 = time.time()

        batch = next(data_iter)
        audio = batch["audio"].to(args.device)
        targets = batch["target_clusters"].to(args.device)
        batch_dur = sum(batch["durations"])
        total_audio_sec += batch_dur

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(args.device == "cuda")):
            outputs = model(
                audio=audio,
                target_clusters=targets,
                mask_prob=args.mask_prob,
                mask_length=args.mask_length,
            )
            loss = outputs["loss"]
            acc = outputs["accuracy"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        acc_pct = float(acc.item() * 100.0) if acc is not None else 0.0
        loss_val = float(loss.item())

        history.append({
            "step": step,
            "loss": round(loss_val, 4),
            "masked_accuracy_pct": round(acc_pct, 2),
            "lr": round(current_lr, 7),
            "cumulative_audio_sec": round(total_audio_sec, 2),
            "cumulative_audio_hours": round(total_audio_sec / 3600.0, 4),
            "disk_bytes_used": 0,
            "active_voices": batch["voices"],
            "sample_prompt": batch["texts"][0],
        })

        if step % 5 == 0 or step == 1 or step == args.steps:
            print(
                f"{step:<8} | {loss_val:<8.4f} | {acc_pct:<13.2f}% | {current_lr:<10.2e} | {total_audio_sec:<9.1f}s | {'0.00 MB':<12} | {elapsed:.1f}s"
            )

        if step % args.save_every == 0 or step == args.steps:
            model.save_pretrained_backbone(args.checkpoint_out)
            with open(args.log_out, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "total_steps": step,
                        "cumulative_audio_hours": round(total_audio_sec / 3600.0, 4),
                        "disk_space_used_mb": 0.0,
                        "history": history,
                    },
                    f,
                    indent=2,
                )

    total_time = time.time() - start_time
    total_hours = total_audio_sec / 3600.0
    print("=" * 85)
    print("                      🎉 PRE-TRAINING COMPLETED!")
    print("=" * 85)
    print(f"Total Pre-training Steps:      {step}")
    print(f"Synthesized Speech in RAM:     {total_audio_sec:.1f} seconds ({total_hours:.3f} hours)")
    print(f"Physical Disk Space for Audio: 0 BYTES (Saved ~{total_audio_sec * 32 / 1024 / 1024:.2f} MB on disk)")
    print(f"Final Masked Unit Accuracy:    {acc_pct:.2f}% (vs. 1.0% random baseline)")
    print(f"Pre-trained Model Saved:       {args.checkpoint_out}")
    print(f"History Log Saved:             {args.log_out}\n")


if __name__ == "__main__":
    main()
