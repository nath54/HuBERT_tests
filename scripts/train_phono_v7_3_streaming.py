"""Phono-V7.3 Real-Time Streaming Speech Model Training Script.

Features:
- Multi-Scale Dilated Conformer convolutions (2.48s acoustic context).
- Decoupled Word Boundary Gate head with 4x space loss penalty.
- Recursive 2-Pass Phoneme Head with causal depthwise recurrent phonotactic refinement.
- Boundary-guided online slicing with 2-frame silence confirmation and 4-frame min burst protection.
- Level-1 Binary Gate & Level-2 Partitioned MoE Decoder with dynamic length horizon capping.
- 4-way balanced multilingual streaming across English, Italian, Spanish, French.
- 100% Zero-perturbation warm-start from V7.1 or V7.2 checkpoints.
"""

import argparse
import difflib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import editdistance
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from src.models.phono_v7_1_speech_model import PhonoV71SpeechConfig
from src.models.phono_v7_3_speech_model import PhonoV73SpeechModel


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


def compute_spacing_errors(gt_tokens: List[int], pred_tokens: List[int], space_token: int = 8) -> Dict[str, Any]:
    """Computes phoneme spacing error breakdown:
    - total_gt: number of true space delimiters in GT
    - correct: correctly predicted spaces
    - missed: omitted spaces (causing word mergers)
    - extra: falsely inserted spaces (causing word splits/fractures)
    - total_errors: missed + extra
    - acc: space accuracy percentage
    """
    matcher = difflib.SequenceMatcher(None, gt_tokens, pred_tokens)
    missed_spaces = 0
    extra_spaces = 0
    correct_spaces = 0
    total_gt_spaces = sum(1 for t in gt_tokens if t == space_token)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        gt_chunk = gt_tokens[i1:i2]
        pred_chunk = pred_tokens[j1:j2]
        if tag == "equal":
            correct_spaces += sum(1 for t in gt_chunk if t == space_token)
        elif tag == "delete":
            missed_spaces += sum(1 for t in gt_chunk if t == space_token)
        elif tag == "insert":
            extra_spaces += sum(1 for t in pred_chunk if t == space_token)
        elif tag == "replace":
            gt_sp = sum(1 for t in gt_chunk if t == space_token)
            pred_sp = sum(1 for t in pred_chunk if t == space_token)
            matched = min(gt_sp, pred_sp)
            correct_spaces += matched
            missed_spaces += max(0, gt_sp - pred_sp)
            extra_spaces += max(0, pred_sp - gt_sp)

    total_errors = missed_spaces + extra_spaces
    acc = (correct_spaces / max(1, total_gt_spaces)) * 100.0 if total_gt_spaces > 0 else 100.0
    return {
        "total_gt": total_gt_spaces,
        "correct": correct_spaces,
        "missed": missed_spaces,
        "extra": extra_spaces,
        "total_errors": total_errors,
        "acc": acc,
    }


@torch.no_grad()
def evaluate(
    model: PhonoV73SpeechModel,
    dataloader: DataLoader,
    roman_tok: RomanCharTokenizer,
    ph_tokenizer: PhonemeTokenizer,
    device: torch.device,
    max_eval_batches: int = 40,
) -> Dict[str, Any]:
    """Evaluate Phono-V7.3 on held-out validation data."""
    model.eval()
    total_loss = 0.0
    total_enc_loss = 0.0
    total_dec_loss = 0.0
    total_bnd_loss = 0.0
    total_path_acc = 0.0
    total_char_correct = 0
    total_char_tokens = 0
    total_phoneme_ed = 0
    total_phoneme_ref_len = 0
    total_spc_gt = 0
    total_spc_correct = 0
    total_spc_missed = 0
    total_spc_extra = 0
    batches_run = 0
    sample_info = None

    for b_idx_loop, batch in enumerate(dataloader):
        if b_idx_loop >= max_eval_batches:
            break
        batches_run += 1

        audio = batch["audio"].to(device)
        audio_lengths = batch["audio_lengths"].to(device)
        phoneme_targets = batch.get("phoneme_targets")
        phoneme_lengths = batch.get("phoneme_lengths")
        num_words = batch.get("num_words")
        input_bytes = batch.get("input_byte_ids")
        target_bytes = batch.get("target_byte_ids")
        path_targets = batch.get("path_targets")
        target_lengths = batch.get("target_lengths")

        if phoneme_targets is not None:
            phoneme_targets = phoneme_targets.to(device)
            phoneme_lengths = phoneme_lengths.to(device)
        if num_words is not None:
            num_words = num_words.to(device)
        if input_bytes is not None:
            input_bytes = input_bytes.to(device)
            target_bytes = target_bytes.to(device)
        if path_targets is not None:
            path_targets = path_targets.to(device)
        if target_lengths is not None:
            target_lengths = target_lengths.to(device)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
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
            total_bnd_loss += out.get("boundary_loss", torch.tensor(0.0)).item()
            total_path_acc += out["path_acc"].item()

        logits = out["logits"]
        if logits is not None and target_bytes is not None:
            preds = logits.argmax(dim=-1)
            tb_sliced = target_bytes[:, : logits.shape[1], : logits.shape[2]]
            valid_mask = tb_sliced != -100
            total_char_correct += (preds[valid_mask] == tb_sliced[valid_mask]).sum().item()
            total_char_tokens += valid_mask.sum().item()

        # Compute PER and spacing errors
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

                spc_stat = compute_spacing_errors(gt_clean, pred_ids, space_token=8)
                total_spc_gt += spc_stat["total_gt"]
                total_spc_correct += spc_stat["correct"]
                total_spc_missed += spc_stat["missed"]
                total_spc_extra += spc_stat["extra"]

        if sample_info is None and len(batch.get("transcripts", [])) > 0 and logits is not None:
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

            sample_spc = compute_spacing_errors(
                [t for t in gt_ph_tokens if t not in (0, 1)],
                decode_ctc_phoneme_ids(enc_log[b_idx]) if enc_log is not None else [],
                space_token=8,
            )
            sample_info = {
                "flag": lang_flag,
                "gt_phonemes": gt_phonemes,
                "pred_phonemes": pred_phonemes,
                "gt_text": gt_text,
                "pred_text": pred_text,
                "spc_info": sample_spc,
            }

    n = max(1, batches_run)
    val_char_acc = (total_char_correct / max(1, total_char_tokens)) * 100.0
    val_per = (total_phoneme_ed / max(1, total_phoneme_ref_len)) * 100.0
    val_spc_acc = (total_spc_correct / max(1, total_spc_gt)) * 100.0 if total_spc_gt > 0 else 100.0

    return {
        "val_loss": total_loss / n,
        "val_enc_loss": total_enc_loss / n,
        "val_dec_loss": total_dec_loss / n,
        "val_bnd_loss": total_bnd_loss / n,
        "val_path_acc": total_path_acc / n,
        "val_char_acc": val_char_acc,
        "val_per": val_per,
        "val_spc_acc": val_spc_acc,
        "val_spc_missed": total_spc_missed,
        "val_spc_extra": total_spc_extra,
        "val_spc_gt": total_spc_gt,
        "sample_info": sample_info,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Phono-V7.3 Real-Time Streaming Speech Model")
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

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/phono_v7_3/streaming")
    parser.add_argument("--warm_start_v7_2", type=str, default="checkpoints/phono_v7_2/streaming/best_checkpoint.pt")
    parser.add_argument("--warm_start_v7_1", type=str, default="checkpoints/phono_v7_1/streaming/best_checkpoint.pt")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--smoke_test", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Phono-V7.3 Multi-Scale & Boundary Gate Streaming Training initialized on device: {device}", flush=True)

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

    model = PhonoV73SpeechModel(config)
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
    elif args.warm_start_v7_2 and Path(args.warm_start_v7_2).is_file():
        print(f"\n🔄 Warm-starting Phono-V7.3 from V7.2 record: {args.warm_start_v7_2}...", flush=True)
        ws_res = model.warm_start_from_v7_2(args.warm_start_v7_2)
        print(f"  • Transferred {ws_res['transferred']} tensors (skipped: {ws_res['skipped']}, missing: {ws_res['missing']})", flush=True)
    elif args.warm_start_v7_1 and Path(args.warm_start_v7_1).is_file():
        print(f"\n🔄 Warm-starting Phono-V7.3 from V7.1 record: {args.warm_start_v7_1}...", flush=True)
        ws_res = model.warm_start_from_v7_2(args.warm_start_v7_1)
        print(f"  • Transferred {ws_res['transferred']} tensors (skipped: {ws_res['skipped']}, missing: {ws_res['missing']})", flush=True)

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
            print(f"  ⚠️ Could not restore optimizer state: {e}", flush=True)

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

    print(f"\n🔥 Starting Phono-V7.3 Streaming Training ({max_steps} steps, start_step={start_step}, band_window={config.band_window_words} words)...", flush=True)
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
        phoneme_targets = batch.get("phoneme_targets")
        phoneme_lengths = batch.get("phoneme_lengths")
        num_words = batch.get("num_words")
        input_bytes = batch.get("input_byte_ids")
        target_bytes = batch.get("target_byte_ids")
        path_targets = batch.get("path_targets")
        target_lengths = batch.get("target_lengths")

        if phoneme_targets is not None:
            phoneme_targets = phoneme_targets.to(device, non_blocking=True)
            phoneme_lengths = phoneme_lengths.to(device, non_blocking=True)
        if num_words is not None:
            num_words = num_words.to(device, non_blocking=True)
        if input_bytes is not None:
            input_bytes = input_bytes.to(device, non_blocking=True)
            target_bytes = target_bytes.to(device, non_blocking=True)
        if path_targets is not None:
            path_targets = path_targets.to(device, non_blocking=True)
        if target_lengths is not None:
            target_lengths = target_lengths.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
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
            loss = raw_loss / args.grad_accum

        if torch.isnan(loss):
            print(f"⚠️ NaN loss detected at step {step + 1}, skipping backward step!", flush=True)
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        accum_step += 1

        if accum_step % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            step += 1

            # Logging
            if step % args.log_every == 0 or step == 1:
                elapsed = time.time() - start_time
                rate = step / max(elapsed, 1e-5)
                eta = format_eta((max_steps - step) / max(rate, 1e-5))
                cur_enc_lr = optimizer.param_groups[0]["lr"]
                cur_dec_lr = optimizer.param_groups[1]["lr"]

                # Real-time batch PER & spacing errors
                step_per = 0.0
                batch_gt_spc = 0
                batch_spc_miss = 0
                batch_spc_ext = 0
                batch_spc_corr = 0
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

                        spc = compute_spacing_errors(gt_clean, pred_ids, space_token=8)
                        batch_gt_spc += spc["total_gt"]
                        batch_spc_miss += spc["missed"]
                        batch_spc_ext += spc["extra"]
                        batch_spc_corr += spc["correct"]

                    step_per = (batch_ed / max(1, batch_ref)) * 100.0

                total_spc_err = batch_spc_miss + batch_spc_ext
                spc_acc = (batch_spc_corr / max(1, batch_gt_spc)) * 100.0 if batch_gt_spc > 0 else 100.0

                lev_val = out["lev_loss"].item() if "lev_loss" in out else 0.0
                align_val = out["align_loss"].item() if "align_loss" in out else 0.0
                len_val = out["length_loss"].item() if "length_loss" in out else 0.0
                bnd_val = out["boundary_loss"].item() if "boundary_loss" in out else 0.0
                headroom_val = out["length_headroom"].item() if "length_headroom" in out else 0.0
                print(
                    f"Step {step:6d}/{max_steps} | "
                    f"Loss: {raw_loss.item():.4f} (Enc: {out['enc_loss'].item():.3f}, Dec: {out['dec_loss'].item():.3f}, Bnd: {bnd_val:.3f}, Lev: {lev_val:.3f}, Len: {len_val:.3f}) | "
                    f"CharAcc: {out['char_acc'].item():5.1f}% | "
                    f"PER: {step_per:5.1f}% | "
                    f"SpcErr: {total_spc_err:2d} (M:{batch_spc_miss}, E:{batch_spc_ext}) | "
                    f"SpcAcc: {spc_acc:5.1f}% | "
                    f"Headroom: {headroom_val:+.2f}c | "
                    f"Cover: {out['path_acc'].item():5.1f}% | "
                    f"LR: [E:{cur_enc_lr:.1e}, D:{cur_dec_lr:.1e}] | "
                    f"Rate: {rate:.2f} st/s | ETA: {eta}",
                    flush=True,
                )

                # Inspect a random sample
                b_idx = random.randint(0, len(batch["languages"]) - 1)
                lang = batch["languages"][b_idx]
                lang_flag = {"en": "🇬🇧 EN", "it": "🇮🇹 IT", "es": "🇪🇸 ES", "fr": "🇫🇷 FR"}.get(lang, lang.upper())

                gt_text = batch["transcripts"][b_idx]
                preds_bytes = out["logits"][b_idx].argmax(dim=-1).cpu().tolist() if out.get("logits") is not None else []
                num_w = batch["num_words"][b_idx].item()
                pred_text = roman_tok.decode_words(preds_bytes[: min(num_w, len(preds_bytes))]) if preds_bytes else "N/A"
                if not pred_text and preds_bytes:
                    pred_text = "<empty>"

                gt_ph_len = batch["phoneme_lengths"][b_idx].item()
                gt_ph_tokens = batch["phoneme_targets"][b_idx, :gt_ph_len].cpu().tolist()
                gt_phonemes = ph_tokenizer.decode(gt_ph_tokens, skip_special=True)
                pred_phonemes = decode_ctc_phonemes(out["ctc_logits"][b_idx], ph_tokenizer) if out.get("ctc_logits") is not None else "N/A"

                sample_spc = compute_spacing_errors(
                    [t for t in gt_ph_tokens if t not in (0, 1)],
                    decode_ctc_phoneme_ids(out["ctc_logits"][b_idx]) if out.get("ctc_logits") is not None else [],
                    space_token=8,
                )

                print(f"  Sample [{lang_flag}]:", flush=True)
                print(f"    • GT Phonemes:   {gt_phonemes}", flush=True)
                print(f"    • Pred Phonemes: {pred_phonemes}", flush=True)
                print(
                    f"    • Spacing Stats: GT Spaces: {sample_spc['total_gt']} | "
                    f"Errors: {sample_spc['total_errors']} (Missed: {sample_spc['missed']} merges, Extra: {sample_spc['extra']} splits) | "
                    f"Space Accuracy: {sample_spc['acc']:.1f}%",
                    flush=True,
                )
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
                    f"(Enc: {val_metrics['val_enc_loss']:.3f} [PER: {val_metrics['val_per']:.2f}%], Dec: {val_metrics['val_dec_loss']:.3f}, Bnd: {val_metrics['val_bnd_loss']:.3f}) | "
                    f"PathAcc: {val_metrics['val_path_acc']:.1f}% | "
                    f"CharAcc: {val_metrics['val_char_acc']:.1f}% | "
                    f"SpcAcc: {val_metrics.get('val_spc_acc', 0.0):.1f}% (Missed: {val_metrics.get('val_spc_missed', 0)}, Extra: {val_metrics.get('val_spc_extra', 0)})",
                    flush=True,
                )
                v_info = val_metrics.get("sample_info")
                if v_info is not None:
                    v_spc = v_info.get("spc_info", {})
                    print(f"  Val Sample [{v_info['flag']}]:", flush=True)
                    print(f"    • GT Phonemes:   {v_info['gt_phonemes']}", flush=True)
                    print(f"    • Pred Phonemes: {v_info['pred_phonemes']}", flush=True)
                    if v_spc:
                        print(
                            f"    • Spacing Stats: GT Spaces: {v_spc.get('total_gt', 0)} | "
                            f"Errors: {v_spc.get('total_errors', 0)} (Missed: {v_spc.get('missed', 0)}, Extra: {v_spc.get('extra', 0)}) | "
                            f"Space Accuracy: {v_spc.get('acc', 0.0):.1f}%",
                            flush=True,
                        )
                    print(f"    • GT Text:       {v_info['gt_text']}", flush=True)
                    print(f"    • Pred Text:     {v_info['pred_text']}", flush=True)

                if v_loss < best_val_loss:
                    print(f"  🌟 New Project Record! Validation loss dropped: {best_val_loss:.4f} -> {v_loss:.4f}", flush=True)
                    best_val_loss = v_loss
                    torch.save(
                        {
                            "step": step,
                            "val_loss": best_val_loss,
                            "val_per": val_metrics["val_per"],
                            "val_char_acc": val_metrics["val_char_acc"],
                            "val_path_acc": val_metrics["val_path_acc"],
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "config": config,
                        },
                        best_ckpt_path,
                    )
                    print(f"  💾 Saved best checkpoint to: {best_ckpt_path}", flush=True)

                model.train()

    if args.smoke_test:
        print(f"\n🎉 Completed {max_steps} steps in {time.time() - start_time:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
