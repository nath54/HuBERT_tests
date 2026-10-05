#!/usr/bin/env python3
"""Training script for Phono-V7.2 Real-Time Streaming Speech Model.

Features:
- Two-Level Hierarchical Router:
  - Level 1: Binary Gate Router (SPECIAL/PAUSE vs SPEECH WORD)
  - Level 2: Partitioned MoE Expert Routing (Short 0..4, Med 5..10, Long 11..15)
- Continuous Word Length Guidance and Strict Horizon Capping:
  - max_k = min(head_bound, ceil(k_hat) + 1)
- Band-Causal Macro Attention (K=8 words) and Shift-Invariant Base Query (q_base).
- Warm-starts from Phono-V7.1 checkpoint.
"""

import argparse
import json
import os
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import AudioUtterance
from src.data.multilingual_audio_dataset import (
    MultilingualAudioDataset,
    MultilingualBalancedBatchSampler,
    MultilingualSpeechCollator,
    RomanCharTokenizer,
)
from src.data.phoneme_tokenizer import PhonemeTargetExtractor
from src.models.phono_v7_1_speech_model import PhonoV71SpeechConfig
from src.models.phono_v7_2_speech_model import PhonoV72SpeechModel


def compute_edit_distance(ref: List[int], hyp: List[int]) -> int:
    """Standard Levenshtein distance."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m]


@torch.no_grad()
def evaluate_streaming(
    model: PhonoV72SpeechModel,
    dataloader: DataLoader,
    device: torch.device,
    roman_tok: RomanCharTokenizer,
    ph_extractor: PhonemeTargetExtractor,
    max_batches: int = 15,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_enc_loss = 0.0
    total_dec_loss = 0.0
    total_path_acc = 0.0
    total_char_correct = 0
    total_char_tokens = 0
    total_phoneme_ed = 0
    total_phoneme_ref_len = 0
    batches_run = 0
    sample_info = None

    val_iter = iter(dataloader)
    for b_num in range(max_batches):
        try:
            batch = next(val_iter)
        except StopIteration:
            break
        batches_run += 1

        audio = batch["audio"].to(device, non_blocking=True)
        audio_lengths = batch["audio_lengths"].to(device, non_blocking=True)
        phoneme_targets = batch["phoneme_targets"].to(device, non_blocking=True)
        phoneme_lengths = batch["phoneme_lengths"].to(device, non_blocking=True)
        num_words = batch["num_words"].to(device, non_blocking=True)
        input_bytes = batch["input_byte_ids"].to(device, non_blocking=True)
        target_bytes = batch["target_byte_ids"].to(device, non_blocking=True)
        path_targets = batch["path_targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out = model(
                audio=audio,
                audio_lengths=audio_lengths,
                phoneme_targets=phoneme_targets,
                phoneme_lengths=phoneme_lengths,
                num_words=num_words,
                input_byte_ids=input_bytes,
                target_byte_ids=target_bytes,
                path_targets=path_targets,
                target_lengths=target_lengths,
            )

        loss = out["loss"]
        total_loss += loss.item()
        total_enc_loss += out["ctc_loss"].item()
        total_dec_loss += out["decoder_loss"].item()
        total_path_acc += out["path_acc"].item()

        targets_flat = target_bytes.reshape(-1)
        valid_mask = targets_flat != -100
        if valid_mask.any():
            total_char_correct += int(out["char_acc"].item() * valid_mask.sum().item())
            total_char_tokens += int(valid_mask.sum().item())

        # Encoder Greedy Decoding for PER
        ctc_logits = out["ctc_logits"]
        pred_ids = ctc_logits.argmax(dim=-1)
        for i in range(audio.shape[0]):
            p_len = phoneme_lengths[i].item()
            gt_ph = phoneme_targets[i, :p_len].tolist()
            raw_pred = pred_ids[i].tolist()
            collapsed = []
            prev = None
            for tok in raw_pred:
                if tok != prev:
                    if tok != 1:  # Blank token
                        collapsed.append(tok)
                    prev = tok
            ed = compute_edit_distance(gt_ph, collapsed)
            total_phoneme_ed += ed
            total_phoneme_ref_len += max(1, len(gt_ph))

            if sample_info is None and i == 0:
                # Decode greedy words with partitioned expert heads & horizon capping
                max_w = int(num_words[i].item()) if num_words is not None else 16
                decoded_word_tokens = model.decoder.decode_greedy(
                    acoustic_memory=out.get("memory_lengths", out["ctc_logits"]),
                    max_words=max_w,
                    ctc_logits=ctc_logits[i : i + 1],
                )
                pred_text_words = []
                for w in decoded_word_tokens:
                    chars = [roman_tok.id_to_char.get(c, "") for c in w if c not in (roman_tok.bos_id, roman_tok.eow_id, roman_tok.eos_id, roman_tok.pad_id)]
                    pred_text_words.append("".join(chars))

                sample_info = {
                    "lang": batch["languages"][i],
                    "gt_ph": " ".join([ph_extractor.tokenizer.id_to_token.get(p, "") for p in gt_ph]),
                    "pred_ph": " ".join([ph_extractor.tokenizer.id_to_token.get(p, "") for p in collapsed]),
                    "gt_text": batch["transcripts"][i],
                    "pred_text": " ".join(pred_text_words),
                }

    avg_loss = total_loss / max(1, batches_run)
    avg_enc = total_enc_loss / max(1, batches_run)
    avg_dec = total_dec_loss / max(1, batches_run)
    avg_path_acc = (total_path_acc / max(1, batches_run)) * 100.0
    char_acc = (total_char_correct / max(1, total_char_tokens)) * 100.0 if total_char_tokens > 0 else 0.0
    per = (total_phoneme_ed / max(1, total_phoneme_ref_len)) * 100.0

    return {
        "val_loss": avg_loss,
        "enc_loss": avg_enc,
        "dec_loss": avg_dec,
        "path_acc": avg_path_acc,
        "char_acc": char_acc,
        "per": per,
        "sample": sample_info,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Phono-V7.2 Partitioned Streaming Character Model")
    parser.add_argument("--librispeech_train", type=str, default="data/librispeech/librispeech_train_100h.json")
    parser.add_argument("--librispeech_val", type=str, default="data/librispeech/benchmark_val.json")
    parser.add_argument("--mls_italian", type=str, default="/media/hdd/Datasets/mls/mls_italian")
    parser.add_argument("--mls_spanish", type=str, default="/media/hdd/Datasets/mls/mls_spanish")
    parser.add_argument("--mls_french", type=str, default="/media/hdd/Datasets/mls/mls_french")
    parser.add_argument("--cv_french", type=str, default="/media/hdd/Datasets/common_voice/french/cv-corpus-27.0-2026-09-11/fr")

    parser.add_argument("--batch_size", type=int, default=2, help="Micro batch size across languages")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps (effective BS = 16)")
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--decoder_lr", type=float, default=3e-4)
    parser.add_argument("--band_window", type=int, default=8, help="Sliding band causal attention window in words")
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/phono_v7_2/streaming")
    parser.add_argument("--warm_start_v7_1", type=str, default="checkpoints/phono_v7_1/streaming/best_checkpoint.pt")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--smoke_test", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Phono-V7.2 Partitioned MoE Streaming Training on: {device}", flush=True)

    # 1. Load Datasets
    print("\n📚 Loading Multilingual Datasets...", flush=True)
    train_utts: List[AudioUtterance] = []
    val_utts: List[AudioUtterance] = []

    # English
    with open(args.librispeech_train, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            train_utts.append(AudioUtterance(audio_path=item["audio_path"], transcript=item["transcript"], duration=item.get("duration", 0.0), lang="en"))
    with open(args.librispeech_val, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data[:400]:
            val_utts.append(AudioUtterance(audio_path=item["audio_path"], transcript=item["transcript"], duration=item.get("duration", 0.0), lang="en"))

    # Italian
    it_trans_path = os.path.join(args.mls_italian, "train/transcripts.txt")
    if os.path.exists(it_trans_path):
        with open(it_trans_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    p = os.path.join(args.mls_italian, "train/audio", parts[0].replace("_", "/").rsplit("/", 1)[0], f"{parts[0]}.flac")
                    if os.path.exists(p):
                        train_utts.append(AudioUtterance(audio_path=p, transcript=parts[1], duration=0.0, lang="it"))
                        if len(train_utts) >= 15000:
                            break

    # Spanish
    es_trans_path = os.path.join(args.mls_spanish, "train/transcripts.txt")
    if os.path.exists(es_trans_path):
        with open(es_trans_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    p = os.path.join(args.mls_spanish, "train/audio", parts[0].replace("_", "/").rsplit("/", 1)[0], f"{parts[0]}.flac")
                    if os.path.exists(p):
                        train_utts.append(AudioUtterance(audio_path=p, transcript=parts[1], duration=0.0, lang="es"))
                        if len(train_utts) >= 30000:
                            break

    # French
    fr_trans_path = os.path.join(args.mls_french, "train/transcripts.txt")
    if os.path.exists(fr_trans_path):
        with open(fr_trans_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    p = os.path.join(args.mls_french, "train/audio", parts[0].replace("_", "/").rsplit("/", 1)[0], f"{parts[0]}.flac")
                    if os.path.exists(p):
                        train_utts.append(AudioUtterance(audio_path=p, transcript=parts[1], duration=0.0, lang="fr"))
                        if len(train_utts) >= 45000:
                            break

    print(f"Loaded {len(train_utts)} train utterances and {len(val_utts)} val utterances.", flush=True)

    roman_tok = RomanCharTokenizer()
    ph_extractor = PhonemeTargetExtractor()

    train_dataset = MultilingualAudioDataset(
        train_utts,
        roman_tokenizer=roman_tok,
        phoneme_extractor=ph_extractor,
        max_duration_seconds=20.0,
    )
    val_dataset = MultilingualAudioDataset(
        val_utts,
        roman_tokenizer=roman_tok,
        phoneme_extractor=ph_extractor,
        max_duration_seconds=20.0,
    )

    collator = MultilingualSpeechCollator(roman_tok)
    train_sampler = MultilingualBalancedBatchSampler(
        train_utts,
        batch_size=args.batch_size,
        languages=["en", "it", "es", "fr"],
        seed=args.seed,
    )
    val_sampler = MultilingualBalancedBatchSampler(
        val_utts,
        batch_size=args.batch_size,
        languages=["en", "it", "es", "fr"],
        seed=args.seed + 1,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    # 2. Model Configuration & Warm-Start
    config = PhonoV71SpeechConfig.medium()
    config.band_window_words = args.band_window
    config.encoder_learning_rate = args.encoder_lr
    config.decoder_learning_rate = args.decoder_lr

    model = PhonoV72SpeechModel(config).to(device)

    # Warm-start
    if args.warm_start_v7_1 and os.path.exists(args.warm_start_v7_1):
        print(f"🔗 Warm-starting Phono-V7.2 from {args.warm_start_v7_1}...", flush=True)
        stats = model.warm_start_from_v7_1(args.warm_start_v7_1)
        print(f"   Transferred: {stats['transferred']} tensors (Missing: {stats['missing']}, Unexpected: {stats['unexpected']})", flush=True)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_per = float("inf")

    # Optimizer with differential learning rates
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {"params": model.decoder.parameters(), "lr": args.decoder_lr},
        ],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler(device="cuda", enabled=(device.type == "cuda"))

    print(f"\n⚡ Starting Phono-V7.2 Training ({args.max_steps} steps, Accum: {args.grad_accum})...", flush=True)
    step = 0
    start_time = time.time()
    train_iter = iter(train_loader)

    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_enc_loss = 0.0
        accum_dec_loss = 0.0
        accum_lev_loss = 0.0
        accum_align_loss = 0.0
        accum_len_loss = 0.0
        accum_char_acc = 0.0
        accum_cover = 0.0
        accum_near = 0.0
        accum_headroom = 0.0
        accum_per = 0.0

        for _ in range(args.grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            audio = batch["audio"].to(device, non_blocking=True)
            audio_lengths = batch["audio_lengths"].to(device, non_blocking=True)
            phoneme_targets = batch["phoneme_targets"].to(device, non_blocking=True)
            phoneme_lengths = batch["phoneme_lengths"].to(device, non_blocking=True)
            num_words = batch["num_words"].to(device, non_blocking=True)
            input_bytes = batch["input_byte_ids"].to(device, non_blocking=True)
            target_bytes = batch["target_byte_ids"].to(device, non_blocking=True)
            path_targets = batch["path_targets"].to(device, non_blocking=True)
            target_lengths = batch["target_lengths"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out = model(
                    audio=audio,
                    audio_lengths=audio_lengths,
                    phoneme_targets=phoneme_targets,
                    phoneme_lengths=phoneme_lengths,
                    num_words=num_words,
                    input_byte_ids=input_bytes,
                    target_byte_ids=target_bytes,
                    path_targets=path_targets,
                    target_lengths=target_lengths,
                )
                loss = out["loss"] / args.grad_accum

            scaler.scale(loss).backward()
            accum_loss += out["loss"].item() / args.grad_accum
            accum_enc_loss += out["ctc_loss"].item() / args.grad_accum
            accum_dec_loss += out["char_loss"].item() / args.grad_accum
            accum_lev_loss += out["levenshtein_loss"].item() / args.grad_accum
            accum_align_loss += out["path_loss"].item() / args.grad_accum
            accum_len_loss += out["length_loss"].item() / args.grad_accum
            accum_char_acc += out["char_acc"].item() / args.grad_accum
            accum_headroom += out.get("length_headroom", torch.tensor(0.0)).item() / args.grad_accum

            # Compute coverage and PER
            k_hat = out.get("k_hat", None)
            if k_hat is not None:
                tl = target_lengths[:, : k_hat.shape[1]]
                valid_mask = tl > 0
                if valid_mask.any():
                    pred_horizon = torch.ceil(k_hat[valid_mask]) + 1.0
                    true_len = tl[valid_mask]
                    accum_cover += (pred_horizon >= true_len).float().mean().item() / args.grad_accum
                    accum_near += ((k_hat[valid_mask] - true_len).abs() <= 1.0).float().mean().item() / args.grad_accum

            # Batch PER estimate
            ctc_logits = out["ctc_logits"]
            pred_ids = ctc_logits.argmax(dim=-1)
            b_ed, b_ref = 0, 0
            for i in range(audio.shape[0]):
                p_len = phoneme_lengths[i].item()
                gt_ph = phoneme_targets[i, :p_len].tolist()
                raw_pred = pred_ids[i].tolist()
                collapsed = []
                prev = None
                for tok in raw_pred:
                    if tok != prev:
                        if tok != 1:
                            collapsed.append(tok)
                        prev = tok
                b_ed += compute_edit_distance(gt_ph, collapsed)
                b_ref += max(1, len(gt_ph))
            accum_per += (b_ed / max(1, b_ref)) * 100.0 / args.grad_accum

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        step += 1

        # Cosine LR warmup/decay
        if step < args.warmup_steps:
            enc_lr = args.encoder_lr * (step / args.warmup_steps)
            dec_lr = args.decoder_lr * (step / args.warmup_steps)
        else:
            progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
            cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            enc_lr = max(1e-6, args.encoder_lr * cos_decay)
            dec_lr = max(1e-5, args.decoder_lr * cos_decay)

        optimizer.param_groups[0]["lr"] = enc_lr
        optimizer.param_groups[1]["lr"] = dec_lr

        if step % args.log_every == 0 or step == 1:
            rate = step / max(1e-6, time.time() - start_time)
            eta_h = (args.max_steps - step) / max(1e-6, rate) / 3600.0
            print(
                f"Step {step:6d}/{args.max_steps} | Loss: {accum_loss:.4f} (Enc: {accum_enc_loss:.3f}, Dec: {accum_dec_loss:.3f}, Lev: {accum_lev_loss:.3f}, Align: {accum_align_loss:.3f}, Len: {accum_len_loss:.3f}) | CharAcc: {accum_char_acc*100:5.1f}% | PER: {accum_per:5.1f}% | Headroom: {accum_headroom:+5.2f}c | Cover: {accum_cover*100:5.1f}% | NearAcc: {accum_near*100:5.1f}% | LR: [E:{enc_lr:.1e}, D:{dec_lr:.1e}] | Rate: {rate:.2f} st/s | ETA: {eta_h:.1f}h",
                flush=True,
            )

        if step % args.eval_every == 0:
            print(f"\n🧪 Evaluating Phono-V7.2 streaming model at step {step}...", flush=True)
            val_metrics = evaluate_streaming(model, val_loader, device, roman_tok, ph_extractor)
            print(
                f"  Validation Loss:    {val_metrics['val_loss']:.4f} (Enc: {val_metrics['enc_loss']:.3f} [PER: {val_metrics['per']:.2f}%], Dec: {val_metrics['dec_loss']:.3f}) | PathAcc: {val_metrics['path_acc']:.1f}% | CharAcc: {val_metrics['char_acc']:.1f}%",
                flush=True,
            )
            sample = val_metrics.get("sample", None)
            if sample:
                flags = {"en": "🇬🇧 EN", "it": "🇮🇹 IT", "es": "🇪🇸 ES", "fr": "🇫🇷 FR"}
                print(f"  Validation Sample [{flags.get(sample['lang'], sample['lang'].upper())}]:", flush=True)
                print(f"    • GT Phonemes:   {sample['gt_ph'][:200]}", flush=True)
                print(f"    • Pred Phonemes: {sample['pred_ph'][:200]}", flush=True)
                print(f"    • GT Text:       {sample['gt_text'][:200]}", flush=True)
                print(f"    • Pred Text:     {sample['pred_text'][:200]}", flush=True)

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                best_ckpt_path = os.path.join(args.checkpoint_dir, "best_checkpoint.pt")
                print(f"  ⭐ New best validation loss! Saving to {best_ckpt_path}...", flush=True)
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": config,
                        "val_loss": best_val_loss,
                        "val_per": val_metrics["per"],
                    },
                    best_ckpt_path,
                )
            model.train()


if __name__ == "__main__":
    main()
