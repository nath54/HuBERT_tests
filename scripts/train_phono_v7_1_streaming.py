#!/usr/bin/env python3
"""Phono-V7.1 Real-Time Streaming Character Training Script.

Architectural Innovations:
1. Band-Causal Macro Attention (K=8 words):
   - Local sliding causal attention eliminates indefinite error snowballing.
   - Constant O(K * L) compute and memory per word.
2. Shift-Invariant Query Formulation:
   - Shared base query + relative distance embeddings.
   - Slot 100 has identical trained capacity to Slot 0.
   - Unlimited live streaming capability without a static slot ceiling.
3. Event-Driven Online CTC Speech Slicing:
   - Causal speech energy peak detector (1 - P(blank)) with space/silence transitions.
   - Absorbs pauses, breaths, and speech tempo variations with zero temporal drift.
4. Macro Latent History Noise Injection:
   - Eliminates exposure bias and forces the decoder to anchor on acoustic evidence.
5. Continuous Duration-Aware Length Prediction + Asymmetric Loss + Soft-Levenshtein:
   - Heavily penalizes under-prediction while granting free positive headroom.
"""

import argparse
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import editdistance
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.data.phoneme_tokenizer import PhonemeTokenizer
from src.data.roman_tokenizer import RomanCharTokenizer
from src.data.target_extractors import PhonemeTargetExtractor
from src.data.multilingual_audio_dataset import (
    AudioUtterance,
    MultilingualAudioDataset,
    MultilingualBalancedBatchSampler,
    MultilingualSpeechCollator,
    load_librispeech_manifest,
    load_mls_manifest,
    load_common_voice_manifest,
)
from src.models.phono_v7_1_speech_model import (
    PhonoV71SpeechConfig,
    PhonoV71SpeechModel,
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


def decode_ctc_phoneme_ids(logits: torch.Tensor, blank_id: int = 1, pad_id: int = 0) -> List[int]:
    """Greedy CTC collapse into token ID list."""
    preds = logits.argmax(dim=-1).cpu().tolist()
    collapsed = []
    prev = None
    for p in preds:
        if p != prev:
            if p != blank_id and p != pad_id:
                collapsed.append(p)
            prev = p
    return collapsed


def decode_ctc_phonemes(logits: torch.Tensor, tokenizer: PhonemeTokenizer, blank_id: int = 1, pad_id: int = 0) -> str:
    """Greedy CTC collapse and decode into readable IPA phonemes."""
    collapsed = decode_ctc_phoneme_ids(logits, blank_id=blank_id, pad_id=pad_id)
    return tokenizer.decode(collapsed, skip_special=True)


@torch.no_grad()
def evaluate(
    model: PhonoV71SpeechModel,
    dataloader: DataLoader,
    roman_tok: RomanCharTokenizer,
    ph_tokenizer: PhonemeTokenizer,
    device: torch.device,
    max_batches: int = 15,
) -> Dict[str, float]:
    """Evaluation on held-out multilingual speech batches with PER and CER computation."""
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
        if not torch.isnan(loss):
            total_loss += loss.item()
            total_enc_loss += out["enc_loss"].item()
            total_dec_loss += out["dec_loss"].item()
            total_path_acc += out["path_acc"].item()

        logits = out["logits"]
        preds = logits.argmax(dim=-1)
        tb_sliced = target_bytes[:, : logits.shape[1]]
        valid_mask = tb_sliced != -100
        total_char_correct += (preds[valid_mask] == tb_sliced[valid_mask]).sum().item()
        total_char_tokens += valid_mask.sum().item()

        # Compute PER
        enc_log = out.get("ctc_logits")
        if enc_log is not None and phoneme_targets is not None:
            for b_i in range(audio.shape[0]):
                ph_len = phoneme_lengths[b_i].item()
                gt_ids = phoneme_targets[b_i, :ph_len].cpu().tolist()
                gt_clean = [p for p in gt_ids if p not in (0, 1)]
                pred_ids = decode_ctc_phoneme_ids(enc_log[b_i])
                ed = editdistance.eval(gt_clean, pred_ids)
                total_phoneme_ed += ed
                total_phoneme_ref_len += max(1, len(gt_clean))

        if sample_info is None and len(batch.get("transcripts", [])) > 0:
            b_idx = random.randint(0, len(batch["languages"]) - 1)
            lang = batch["languages"][b_idx]
            lang_flag = {"en": "🇬🇧 EN", "it": "🇮🇹 IT", "es": "🇪🇸 ES", "fr": "🇫🇷 FR"}.get(lang, lang.upper())
            gt_text = batch["transcripts"][b_idx]
            preds_bytes = logits[b_idx].argmax(dim=-1).cpu().tolist()
            num_w = batch["num_words"][b_idx].item()
            pred_text = roman_tok.decode_words(preds_bytes[: min(num_w, len(preds_bytes))])

            gt_ph_len = batch["phoneme_lengths"][b_idx].item()
            gt_ph_tokens = batch["phoneme_targets"][b_idx, :gt_ph_len].cpu().tolist()
            gt_phonemes = ph_tokenizer.decode(gt_ph_tokens, skip_special=True)
            pred_phonemes = decode_ctc_phonemes(enc_log[b_idx], ph_tokenizer) if enc_log is not None else "N/A"

            sample_info = {
                "flag": lang_flag,
                "gt_phonemes": gt_phonemes,
                "pred_phonemes": pred_phonemes,
                "gt_text": gt_text,
                "pred_text": pred_text,
            }

    n = max(1, batches_run)
    val_char_acc = (total_char_correct / max(1, total_char_tokens)) * 100.0
    val_per = (total_phoneme_ed / max(1, total_phoneme_ref_len)) * 100.0

    return {
        "val_loss": total_loss / n,
        "val_enc_loss": total_enc_loss / n,
        "val_dec_loss": total_dec_loss / n,
        "val_path_acc": total_path_acc / n,
        "val_char_acc": val_char_acc,
        "val_per": val_per,
        "sample_info": sample_info,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Phono-V7.1 Real-Time Streaming Character Model")
    parser.add_argument("--librispeech_train", type=str, default="data/librispeech/librispeech_train_100h.json")
    parser.add_argument("--librispeech_val", type=str, default="data/librispeech/benchmark_val.json")
    parser.add_argument("--mls_italian", type=str, default="/media/hdd/Datasets/mls/mls_italian")
    parser.add_argument("--mls_spanish", type=str, default="/media/hdd/Datasets/mls/mls_spanish")
    parser.add_argument("--mls_french", type=str, default="/media/hdd/Datasets/mls/mls_french")
    parser.add_argument("--cv_french", type=str, default="/media/hdd/Datasets/common_voice/french/cv-corpus-27.0-2026-09-11/fr")

    parser.add_argument("--batch_size", type=int, default=2, help="Micro batch size across languages (e.g. 2)")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps (effective BS = 16)")
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--decoder_lr", type=float, default=3e-4)
    parser.add_argument("--band_window", type=int, default=8, help="Sliding band causal attention window in words")
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/phono_v7_1/streaming")
    parser.add_argument("--warm_start_v7", type=str, default="checkpoints/phono_v7/char/best_checkpoint.pt")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--smoke_test", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Phono-V7.1 Real-Time Streaming Training initialized on device: {device}", flush=True)

    # 1. Load Multilingual Manifests
    print("\n📚 Loading Multilingual Datasets...", flush=True)
    train_utts: List[AudioUtterance] = []
    val_utts: List[AudioUtterance] = []

    # English
    en_train = load_librispeech_manifest(args.librispeech_train)
    en_val = load_librispeech_manifest(args.librispeech_val)
    print(f"  • English (LibriSpeech): {len(en_train)} train, {len(en_val)} val")
    train_utts.extend(en_train)
    val_utts.extend(en_val)

    # Italian
    it_train = load_mls_manifest(args.mls_italian, lang="it", split="train", max_samples=25000)
    it_val = load_mls_manifest(args.mls_italian, lang="it", split="dev", max_samples=500)
    print(f"  • Italian (MLS): {len(it_train)} train, {len(it_val)} val")
    train_utts.extend(it_train)
    val_utts.extend(it_val)

    # Spanish
    es_train = load_mls_manifest(args.mls_spanish, lang="es", split="train", max_samples=25000)
    es_val = load_mls_manifest(args.mls_spanish, lang="es", split="dev", max_samples=500)
    print(f"  • Spanish (MLS): {len(es_train)} train, {len(es_val)} val")
    train_utts.extend(es_train)
    val_utts.extend(es_val)

    # French
    fr_train = load_mls_manifest(args.mls_french, lang="fr", split="train", max_samples=15000)
    fr_cv = load_common_voice_manifest(args.cv_french, lang="fr", split="train", max_samples=10000)
    fr_train.extend(fr_cv)
    fr_val = load_mls_manifest(args.mls_french, lang="fr", split="dev", max_samples=500)
    print(f"  • French (MLS + CV): {len(fr_train)} train, {len(fr_val)} val")
    train_utts.extend(fr_train)
    val_utts.extend(fr_val)

    print(f"  🌟 Combined: {len(train_utts)} train utterances, {len(val_utts)} val utterances")

    # 2. Tokenizers and Datasets
    ph_tokenizer = PhonemeTokenizer()
    ph_extractor = PhonemeTargetExtractor(ph_tokenizer)
    roman_tok = RomanCharTokenizer()

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

    # 3. Model Configuration & Warm-Start
    config = PhonoV71SpeechConfig.medium()
    config.band_window_words = args.band_window
    config.encoder_learning_rate = args.encoder_lr
    config.decoder_learning_rate = args.decoder_lr

    model = PhonoV71SpeechModel(config)
    model = model.to(device)

    start_step = 0
    best_val_loss = float("inf")
    optimizer_state = None

    if args.resume_from and Path(args.resume_from).is_file():
        print(f"\n🔄 Resuming checkpoint from: {args.resume_from}...", flush=True)
        ckpt = torch.load(args.resume_from, map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        start_step = ckpt.get("step", 0)
        best_val_loss = ckpt.get("val_loss", float("inf"))
        optimizer_state = ckpt.get("optimizer_state_dict")
        print(f"  • Successfully resumed from step {start_step} (best_val_loss={best_val_loss:.4f})", flush=True)
    elif args.warm_start_v7 and Path(args.warm_start_v7).is_file():
        print(f"\n🔄 Warm-starting Phono-V7.1 from V7 record: {args.warm_start_v7}...", flush=True)
        ws_res = model.warm_start_from_v7(args.warm_start_v7)
        print(f"  • Transferred {ws_res['transferred']} tensors (unexpected: {ws_res['unexpected']}, missing: {ws_res['missing']})", flush=True)

    # 4. Optimization Setup
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.encoder_lr, "weight_decay": 0.01},
            {"params": decoder_params, "lr": args.decoder_lr, "weight_decay": 0.01},
        ],
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            print("  • Optimizer state restored successfully", flush=True)
        except Exception as e:
            print(f"  ⚠️ Could not restore optimizer state ({e}), starting fresh AdamW", flush=True)

    max_steps = (start_step + 10) if args.smoke_test else args.max_steps
    sched_max_steps = (start_step + 500) if args.smoke_test else args.max_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=sched_max_steps,
    )
    if start_step > 0:
        for _ in range(start_step):
            lr_scheduler.step()

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_checkpoint.pt"

    print(f"\n🔥 Starting Phono-V7.1 Streaming Training ({max_steps} steps, start_step={start_step}, band_window={config.band_window_words} words)...", flush=True)
    start_time = time.time()
    step = start_step
    accum_step = 0

    model.train()
    train_iter = iter(train_loader)

    while step < max_steps:
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
            raw_loss = out["loss"]

        if torch.isnan(raw_loss):
            print(f"⚠️ Warning: NaN loss at step {step}, skipping update.")
            optimizer.zero_grad()
            continue

        loss = raw_loss / args.grad_accum
        scaler.scale(loss).backward()
        accum_step += 1

        if accum_step % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_after >= scale_before:
                lr_scheduler.step()
            optimizer.zero_grad()
            step += 1

            # Logging
            if step % args.log_every == 0 or step == 1:
                elapsed = time.time() - start_time
                rate = step / max(elapsed, 1e-5)
                eta = format_eta((max_steps - step) / max(rate, 1e-5))
                cur_enc_lr = optimizer.param_groups[0]["lr"]
                cur_dec_lr = optimizer.param_groups[1]["lr"]

                # Real-time batch PER
                step_per = 0.0
                if out.get("ctc_logits") is not None and phoneme_targets is not None:
                    batch_ed = 0
                    batch_ref = 0
                    for b_i in range(audio.shape[0]):
                        ph_len = phoneme_lengths[b_i].item()
                        gt_ids = phoneme_targets[b_i, :ph_len].cpu().tolist()
                        gt_clean = [p for p in gt_ids if p not in (0, 1)]
                        pred_ids = decode_ctc_phoneme_ids(out["ctc_logits"][b_i])
                        batch_ed += editdistance.eval(gt_clean, pred_ids)
                        batch_ref += max(1, len(gt_clean))
                    step_per = (batch_ed / max(1, batch_ref)) * 100.0

                lev_val = out["lev_loss"].item() if "lev_loss" in out else 0.0
                align_val = out["align_loss"].item() if "align_loss" in out else 0.0
                len_val = out["length_loss"].item() if "length_loss" in out else 0.0
                headroom_val = out["length_headroom"].item() if "length_headroom" in out else 0.0
                near_val = out["near_acc"].item() if "near_acc" in out else 0.0
                print(
                    f"Step {step:6d}/{max_steps} | "
                    f"Loss: {raw_loss.item():.4f} (Enc: {out['enc_loss'].item():.3f}, Dec: {out['dec_loss'].item():.3f}, Lev: {lev_val:.3f}, Align: {align_val:.3f}, Len: {len_val:.3f}) | "
                    f"CharAcc: {out['char_acc'].item():5.1f}% | "
                    f"PER: {step_per:5.1f}% | "
                    f"Headroom: {headroom_val:+.2f}c | "
                    f"Cover: {out['path_acc'].item():5.1f}% | "
                    f"NearAcc: {near_val:5.1f}% | "
                    f"LR: [E:{cur_enc_lr:.1e}, D:{cur_dec_lr:.1e}] | "
                    f"Rate: {rate:.2f} st/s | ETA: {eta}",
                    flush=True,
                )

                # Inspect a random sample
                b_idx = random.randint(0, len(batch["languages"]) - 1)
                lang = batch["languages"][b_idx]
                lang_flag = {"en": "🇬🇧 EN", "it": "🇮🇹 IT", "es": "🇪🇸 ES", "fr": "🇫🇷 FR"}.get(lang, lang.upper())

                gt_text = batch["transcripts"][b_idx]
                preds_bytes = out["logits"][b_idx].argmax(dim=-1).cpu().tolist()
                num_w = batch["num_words"][b_idx].item()
                pred_text = roman_tok.decode_words(preds_bytes[: min(num_w, len(preds_bytes))])

                gt_ph_len = batch["phoneme_lengths"][b_idx].item()
                gt_ph_tokens = batch["phoneme_targets"][b_idx, :gt_ph_len].cpu().tolist()
                gt_phonemes = ph_tokenizer.decode(gt_ph_tokens, skip_special=True)
                pred_phonemes = decode_ctc_phonemes(out["ctc_logits"][b_idx], ph_tokenizer) if out.get("ctc_logits") is not None else "N/A"

                print(f"  Sample [{lang_flag}]:", flush=True)
                print(f"    • GT Phonemes:   {gt_phonemes}", flush=True)
                print(f"    • Pred Phonemes: {pred_phonemes}", flush=True)
                print(f"    • GT Text:       {gt_text}", flush=True)
                print(f"    • Pred Text:     {pred_text}", flush=True)

            # Evaluation & Checkpointing
            if step > 0 and (step % args.eval_every == 0 or step == max_steps) and not args.smoke_test:
                print(f"\n🧪 Evaluating streaming model at step {step}...", flush=True)
                val_metrics = evaluate(
                    model,
                    val_loader,
                    roman_tok=roman_tok,
                    ph_tokenizer=ph_tokenizer,
                    device=device,
                )
                v_loss = val_metrics["val_loss"]
                print(
                    f"  Validation Loss:    {v_loss:.4f} "
                    f"(Enc: {val_metrics['val_enc_loss']:.3f} [PER: {val_metrics['val_per']:.2f}%], Dec: {val_metrics['val_dec_loss']:.3f}) | "
                    f"PathAcc: {val_metrics['val_path_acc']:.1f}% | "
                    f"CharAcc: {val_metrics['val_char_acc']:.1f}%",
                    flush=True,
                )
                v_info = val_metrics.get("sample_info")
                if v_info:
                    print(f"  Validation Sample [{v_info['flag']}]:", flush=True)
                    print(f"    • GT Phonemes:   {v_info['gt_phonemes']}", flush=True)
                    print(f"    • Pred Phonemes: {v_info['pred_phonemes']}", flush=True)
                    print(f"    • GT Text:       {v_info['gt_text']}", flush=True)
                    print(f"    • Pred Text:     {v_info['pred_text']}", flush=True)

                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    print(f"  ⭐ New best validation loss! Saving best checkpoint to {best_ckpt_path}...", flush=True)
                    torch.save(
                        {
                            "step": step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "config": config,
                            "val_loss": v_loss,
                            "val_metrics": val_metrics,
                        },
                        best_ckpt_path,
                    )
                print(flush=True)
                model.train()

    print(f"\n🎉 Completed {step} steps in {time.time() - start_time:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
