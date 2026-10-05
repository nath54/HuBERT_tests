#!/usr/bin/env python3
"""Phono-V7 Hierarchical Two-Level Character Training Script with Double Loss.

Architectural Highlights:
1. Conformer Acoustic Backbone (Model 1):
   - Conformer depthwise separable convolution (k=31) captures local co-articulation.
   - Initialized from our 5.87% PER checkpoint with zero identity regression at step 0.
   - Supervised by CTC Phoneme Loss (Double Loss).
2. Guaranteed Monotonic Temporal Alignment (Axes 1 & 2):
   - Proportional monotonic time slicing on actual valid duration T_valid (Axe 1).
   - Dynamic local refinement around CTC phoneme energy peaks (Axe 2).
   - Continuous 32-frame speech slices extracted via extract_ctc_monotonic_slices for each word slot.
3. Two-Level Hierarchical Character Decoder (Model 2 & 3):
   - Macro Word Layer with causal self-attention over sentence context.
   - 4-Path length-adaptive routing (Special, Short, Medium, Long) + Diffusion Refiner.
   - Micro Recursive Character Head with Dual Cross-Attention (intra-word continuous speech + multi-word sliding window).
   - 122 character alphabet: completely immune to Zipfian word collapse!
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
from src.models.phono_v7_speech_model import (
    PhonoV7SpeechConfig,
    PhonoV7SpeechModel,
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
    model: PhonoV7SpeechModel,
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
    parser = argparse.ArgumentParser(description="Train Phono-V7 Two-Level Character Model with Double Loss")
    parser.add_argument("--librispeech_train", type=str, default="data/librispeech/librispeech_train_100h.json")
    parser.add_argument("--librispeech_val", type=str, default="data/librispeech/benchmark_val.json")
    parser.add_argument("--mls_italian", type=str, default="/media/hdd/Datasets/mls/mls_italian")
    parser.add_argument("--mls_spanish", type=str, default="/media/hdd/Datasets/mls/mls_spanish")
    parser.add_argument("--mls_french", type=str, default="/media/hdd/Datasets/mls/mls_french")
    parser.add_argument("--cv_french", type=str, default="/media/hdd/Datasets/common_voice/french/cv-corpus-27.0-2026-09-11/fr")

    parser.add_argument("--batch_size", type=int, default=2, help="Micro batch size across languages (e.g. 2)")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps (effective BS = 16)")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--encoder_lr", type=float, default=1e-5, help="Fine-tuning LR for 5.87% PER backbone")
    parser.add_argument("--decoder_lr", type=float, default=3e-4, help="Primary LR for Hierarchical Character Decoder")
    parser.add_argument("--ctc_loss_weight", type=float, default=0.5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/phono_v7/char")
    parser.add_argument("--encoder_checkpoint", type=str, default="checkpoints/phono_v6_4_gated_diffusion/medium/v6_4_960h/best_checkpoint.pt")
    parser.add_argument("--decoder_checkpoint", type=str, default="checkpoints/phono_v6_7_speech/medium/best_checkpoint.pt")
    parser.add_argument("--levenshtein_weight", type=float, default=0.2, help="Soft-Levenshtein loss weight")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume training from an existing Phono-V7 checkpoint")
    parser.add_argument("--smoke_test", action="store_true", help="Run 10 steps to verify memory and gradients then exit")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Phono-V7 Hierarchical Character Training initialized on device: {device}", flush=True)

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
    fr_train_mls = load_mls_manifest(args.mls_french, lang="fr", split="train", max_samples=15000)
    fr_val = load_mls_manifest(args.mls_french, lang="fr", split="dev", max_samples=500)
    fr_train_cv = load_common_voice_manifest(args.cv_french, lang="fr", split="train", max_samples=10000)
    fr_train = fr_train_mls + fr_train_cv
    print(f"  • French (MLS + CV): {len(fr_train)} train, {len(fr_val)} val")
    train_utts.extend(fr_train)
    val_utts.extend(fr_val)

    print(f"  🌟 Combined: {len(train_utts)} train utterances, {len(val_utts)} val utterances", flush=True)

    # 2. Create Tokenizers & Datasets
    ph_tokenizer = PhonemeTokenizer()
    ph_extractor = PhonemeTargetExtractor(ph_tokenizer)
    roman_tok = RomanCharTokenizer()

    train_dataset = MultilingualAudioDataset(train_utts, roman_tokenizer=roman_tok, phoneme_extractor=ph_extractor, max_duration_seconds=20.0)
    val_dataset = MultilingualAudioDataset(val_utts, roman_tokenizer=roman_tok, phoneme_extractor=ph_extractor, max_duration_seconds=20.0)

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

    # 3. Initialize Phono-V7 Model & Warm Start
    config = PhonoV7SpeechConfig.medium()
    config.ctc_loss_weight = args.ctc_loss_weight
    config.levenshtein_loss_weight = args.levenshtein_weight
    config.encoder_learning_rate = args.encoder_lr
    config.decoder_learning_rate = args.decoder_lr

    model = PhonoV7SpeechModel(config)
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
        if missing_keys:
            print(f"  • Note: New parameters cleanly initialized: {len(missing_keys)} tensors ({missing_keys[:3]}...)", flush=True)
    else:
        print("\n🔄 Warm-starting from pretrained checkpoints...", flush=True)
        ws_res = model.warm_start_from_checkpoints(
            encoder_checkpoint_path=args.encoder_checkpoint,
            decoder_checkpoint_path=args.decoder_checkpoint,
        )
        if ws_res["encoder"]:
            print(f"  • Encoder (V6.4 Gated Diffusion): {ws_res['encoder']['transferred']} tensors transferred (missing: {ws_res['encoder']['missing']}, unexpected: {ws_res['encoder']['unexpected']})")
        if ws_res["decoder"]:
            print(f"  • Decoder (V6.7 Windowed Decoder): {ws_res['decoder']['transferred']} tensors transferred (missing: {ws_res['decoder']['missing']}, unexpected: {ws_res['decoder']['unexpected']})")

    # 4. Dual Parameter Group Optimization with Clean Extension for Length Predictor
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    length_params = [p for p in model.decoder.length_predictor.parameters() if p.requires_grad]
    length_param_ids = set(id(p) for p in length_params)
    decoder_base_params = [p for n, p in model.decoder.named_parameters() if p.requires_grad and id(p) not in length_param_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.encoder_lr, "weight_decay": 0.01},
            {"params": decoder_base_params, "lr": args.decoder_lr, "weight_decay": 0.01},
        ],
        betas=(0.9, 0.98),
        eps=1e-8,
    )
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            print("  • Optimizer state restored successfully (378 encoder + 530 decoder base params)", flush=True)
        except Exception as e:
            print(f"  ⚠️ Could not restore optimizer state ({e}), re-initializing AdamW", flush=True)

    # Add new duration-aware word length predictor params to optimizer
    optimizer.add_param_group({"params": length_params, "lr": args.decoder_lr, "weight_decay": 0.01})

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

    print(f"\n🔥 Starting Phono-V7 Two-Level Character Training ({max_steps} steps, start_step={start_step}, grad_accum={args.grad_accum}, lev_weight={config.levenshtein_loss_weight})...", flush=True)
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
                print(
                    f"Step {step:6d}/{max_steps} | "
                    f"Loss: {raw_loss.item():.4f} (Enc: {out['enc_loss'].item():.3f}, Dec: {out['dec_loss'].item():.3f}, Lev: {lev_val:.3f}, Align: {align_val:.3f}, Len: {len_val:.3f}) | "
                    f"CharAcc: {out['char_acc'].item():5.1f}% | "
                    f"PER: {step_per:5.1f}% | "
                    f"Headroom: {headroom_val:+.2f}c | "
                    f"PathAcc: {out['path_acc'].item():5.1f}% | "
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
                print(f"\n🧪 Evaluating at step {step}...", flush=True)
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

    print(f"\n🎉 Completed {step} steps in {time.time() - start_time:.1f}s!")
    if not args.smoke_test:
        print(f"Best Validation Loss: {best_val_loss:.4f} saved at {best_ckpt_path}")


if __name__ == "__main__":
    main()
