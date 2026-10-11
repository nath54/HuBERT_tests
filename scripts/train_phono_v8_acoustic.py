#!/usr/bin/env python3
"""Phono-V8.0 Acoustic Conformer Training with DeepSeek MoE & Engram Memory.

Features:
1. Pure Isolated Module 1 Acoustic Training (Decoders 2 & 3 completely detached)
2. 12-Layer Conformer Backbone with DeepSeek Mixture of Experts (1 Shared + 8 Routed, Top-2)
3. Multi-Order Phoneme Engram Conditional Memory (2-gram, 3-gram, 4-gram with 8 hash heads)
4. Intermediate Layer 4 Draft Phoneme Emission & Supervision
5. Gold-Standard Verbatim Dataset Training on Sanitized Multilingual Audio
6. Zero Language Conditioning Vectors (Strictly Language-Agnostic)
7. Non-Destructive Warm-Start from Phono-V7.5 Acoustic Weights
"""

import argparse
import difflib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import editdistance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
from src.models.phono_v8_acoustic import (
    PhonoV8AcousticBackbone,
    warm_start_phono_v8_from_v75,
)
from src.models.phono_v7_3_speech_model import decode_calibrated_phonemes


def decode_ctc_phoneme_ids(logits: torch.Tensor, blank_id: int = 1, pad_id: int = 0) -> List[int]:
    preds = logits.argmax(dim=-1).detach().cpu().tolist()
    collapsed = []
    prev = None
    for p in preds:
        if p != prev:
            if p != blank_id and p != pad_id:
                collapsed.append(p)
            prev = p
    return collapsed


def compute_spacing_errors(gt_tokens: List[int], pred_tokens: List[int], space_token: int = 8) -> Dict[str, Any]:
    matcher = difflib.SequenceMatcher(None, gt_tokens, pred_tokens)
    missed_spaces = 0
    extra_spaces = 0
    correct_spaces = 0
    total_gt_spaces = sum(1 for t in gt_tokens if t == space_token)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for t in gt_tokens[i1:i2]:
                if t == space_token:
                    correct_spaces += 1
        elif tag == "delete":
            for t in gt_tokens[i1:i2]:
                if t == space_token:
                    missed_spaces += 1
        elif tag == "insert":
            for t in pred_tokens[j1:j2]:
                if t == space_token:
                    extra_spaces += 1
        elif tag == "replace":
            gt_spc = sum(1 for t in gt_tokens[i1:i2] if t == space_token)
            pr_spc = sum(1 for t in pred_tokens[j1:j2] if t == space_token)
            matched_spc = min(gt_spc, pr_spc)
            correct_spaces += matched_spc
            missed_spaces += max(0, gt_spc - pr_spc)
            extra_spaces += max(0, pr_spc - gt_spc)

    acc = (correct_spaces / max(1, total_gt_spaces)) * 100.0 if total_gt_spaces > 0 else 100.0
    return {
        "accuracy": acc,
        "missed": missed_spaces,
        "extra": extra_spaces,
        "correct": correct_spaces,
        "total_gt": total_gt_spaces,
    }


def calibrate_predictions_for_eval(
    pred_tokens: List[int],
    boundary_probs: Optional[torch.Tensor] = None,
    space_token: int = 8,
    space_threshold: float = 0.50,
) -> List[int]:
    """Applies word boundary calibration to align spacing."""
    if not pred_tokens:
        return pred_tokens
    cleaned = []
    prev = None
    for t in pred_tokens:
        if t == space_token and prev == space_token:
            continue
        cleaned.append(t)
        prev = t
    while cleaned and cleaned[0] == space_token:
        cleaned.pop(0)
    while cleaned and cleaned[-1] == space_token:
        cleaned.pop()
    return cleaned


@torch.no_grad()
def evaluate_acoustic(
    model: PhonoV8AcousticBackbone,
    val_loader: DataLoader,
    ph_tokenizer: PhonemeTokenizer,
    device: torch.device,
    token_to_macro: torch.Tensor,
    max_eval_batches: int = 25,
    chunk_eval: bool = False,
) -> Dict[str, Any]:
    model.eval()
    total_ed = 0
    total_ref_len = 0
    total_ed_cal = 0
    total_spc_correct = 0
    total_spc_gt = 0
    total_loss = 0.0
    num_batches = 0

    per_lang_ed: Dict[str, int] = {}
    per_lang_ed_cal: Dict[str, int] = {}
    per_lang_len: Dict[str, int] = {}
    val_sample_records: Dict[str, Dict[str, Any]] = {}

    for b_idx, batch in enumerate(val_loader):
        if b_idx >= max_eval_batches:
            break

        audio = batch["audio"].to(device)
        audio_lengths = batch["audio_lengths"].to(device)
        phoneme_targets = batch["phoneme_targets"].to(device)
        phoneme_lengths = batch["phoneme_lengths"].to(device)
        macro_targets = token_to_macro[phoneme_targets.clamp(0, 111)]

        if chunk_eval and audio.shape[1] > 128000:
            chunk_out = model.forward_sliding_chunks(audio, chunk_seconds=8.0, overlap_seconds=2.0)
            logits = chunk_out["logits"]
            boundary_logits = chunk_out["boundary_logits"]
            loss_val = 0.0
        else:
            out = model(
                audio=audio,
                audio_lengths=audio_lengths,
                targets=phoneme_targets,
                target_lengths=phoneme_lengths,
                macro_targets=macro_targets,
                macro_lengths=phoneme_lengths,
            )
            logits = out["logits"]
            boundary_logits = out["boundary_logits"]
            loss_val = out["loss"].item()

        total_loss += loss_val
        num_batches += 1

        b_probs = F.softmax(boundary_logits, dim=-1)

        for i in range(len(audio)):
            l_i = batch["languages"][i]
            p_toks = decode_ctc_phoneme_ids(logits[i])
            g_toks = phoneme_targets[i][: phoneme_lengths[i]].cpu().tolist()

            cal_toks = decode_calibrated_phonemes(logits[i], blank_penalty=0.4, min_tokens_between_spaces=1)

            ed_i = editdistance.eval(p_toks, g_toks)
            ed_cal_i = editdistance.eval(cal_toks, g_toks)
            ref_i = max(1, len(g_toks))

            spc_res = compute_spacing_errors(g_toks, p_toks, space_token=8)

            total_ed += ed_i
            total_ed_cal += ed_cal_i
            total_ref_len += ref_i
            total_spc_correct += spc_res["correct"]
            total_spc_gt += spc_res["total_gt"]

            per_lang_ed[l_i] = per_lang_ed.get(l_i, 0) + ed_i
            per_lang_ed_cal[l_i] = per_lang_ed_cal.get(l_i, 0) + ed_cal_i
            per_lang_len[l_i] = per_lang_len.get(l_i, 0) + ref_i

            if l_i not in val_sample_records:
                val_sample_records[l_i] = {
                    "lang": l_i,
                    "text": batch["transcripts"][i],
                    "gt": ph_tokenizer.decode(g_toks, skip_special=True),
                    "pred_greedy": ph_tokenizer.decode(p_toks, skip_special=True),
                    "pred_calib": ph_tokenizer.decode(cal_toks, skip_special=True),
                    "per_greedy": (ed_i / ref_i) * 100.0,
                    "per_calib": (ed_cal_i / ref_i) * 100.0,
                    "spc_acc": spc_res["accuracy"],
                }

    val_per = (total_ed / max(1, total_ref_len)) * 100.0
    val_per_cal = (total_ed_cal / max(1, total_ref_len)) * 100.0
    spc_acc = (total_spc_correct / max(1, total_spc_gt)) * 100.0 if total_spc_gt > 0 else 100.0

    val_per_by_lang = {l: (per_lang_ed[l] / max(1, per_lang_len[l])) * 100.0 for l in per_lang_ed}
    val_per_by_lang_cal = {l: (per_lang_ed_cal[l] / max(1, per_lang_len[l])) * 100.0 for l in per_lang_ed_cal}

    return {
        "val_per": val_per,
        "val_per_cal": val_per_cal,
        "val_spc_acc": spc_acc,
        "val_loss": total_loss / max(1, num_batches),
        "val_per_by_lang": val_per_by_lang,
        "val_per_by_lang_cal": val_per_by_lang_cal,
        "samples": val_sample_records,
    }


def format_eta(seconds: float) -> str:
    if seconds < 0 or seconds > 1e7:
        return "N/A"
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hrs > 0:
        return f"{hrs}h{mins:02d}m{secs:02d}s"
    return f"{mins:02d}m{secs:02d}s"


def main():
    parser = argparse.ArgumentParser(description="Train Phono-V8.0 Acoustic Conformer with DeepSeek MoE & Engram")
    parser.add_argument("--clean_train_manifest", type=str, default="data/clean_multilingual_train.json")
    parser.add_argument("--clean_val_manifest", type=str, default="data/clean_multilingual_val.json")
    parser.add_argument("--librispeech_train", type=str, default="data/librispeech/librispeech_train_460h.json")
    parser.add_argument("--warm_start_ckpt", type=str, default="checkpoints/phono_v7_5_acoustic/best_acoustic.pt")
    parser.add_argument("--output_dir", type=str, default="checkpoints/phono_v8_acoustic")
    parser.add_argument("--max_steps", type=int, default=120000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8.0e-5)
    parser.add_argument("--min_lr", type=float, default=5.0e-6)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_duration_seconds", type=float, default=14.0)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--chunk_eval", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("🚀 PHONO-V8.0: ACOUSTIC CONFORMER WITH DEEPSEEK MoE & PHONEME ENGRAM")
    print("=" * 80)
    print(f"📦 Warm-Start Checkpoint : {args.warm_start_ckpt}")
    print(f"📁 Checkpoint Output Dir : {out_dir}")
    print(f"⚡ Device                 : {device.type.upper()}")
    print(f"🎯 Total Steps           : {args.max_steps}")
    print(f"🔥 Micro-Batch / Accum   : {args.batch_size} / {args.grad_accum} (Effective BS = {args.batch_size * args.grad_accum})")
    print(f"📈 Learning Rate         : {args.lr} -> {args.min_lr}")

    print("\n📚 Loading Gold-Standard Verbatim Datasets...", flush=True)
    train_utts: List[AudioUtterance] = []
    val_utts: List[AudioUtterance] = []

    # 1. Load Training Manifest
    if os.path.isfile(args.clean_train_manifest):
        print(f"  ✨ Using sanitized training manifest: {args.clean_train_manifest}")
        with open(args.clean_train_manifest, "r", encoding="utf-8") as f:
            train_raw = json.load(f)
        train_utts = [
            AudioUtterance(
                id=item["id"],
                audio_path=item["audio_path"],
                transcript=item["transcript"],
                lang=item["lang"],
                duration=float(item.get("duration", 0.0)),
            )
            for item in train_raw
        ]
    else:
        print(f"  • Fallback: Loading English LibriSpeech 460h...")
        train_utts = load_librispeech_manifest(args.librispeech_train, max_samples=40000)

    # 2. Load Clean Validation Manifest
    if os.path.isfile(args.clean_val_manifest):
        print(f"  ✨ Using gold-standard clean validation: {args.clean_val_manifest}")
        with open(args.clean_val_manifest, "r", encoding="utf-8") as f:
            val_raw = json.load(f)
        val_utts = [
            AudioUtterance(
                id=item["id"],
                audio_path=item["audio_path"],
                transcript=item["transcript"],
                lang=item["lang"],
                duration=float(item.get("duration", 0.0)),
            )
            for item in val_raw
        ]
    else:
        print(f"  • Fallback: using default validation manifest...")
        with open("data/multilingual_val_patched.json", "r", encoding="utf-8") as f:
            val_raw = json.load(f)
        val_utts = [
            AudioUtterance(id=x["id"], audio_path=x["audio_path"], transcript=x["transcript"], lang=x["lang"])
            for x in val_raw
        ]

    print(f"  🌟 Combined Corpus: {len(train_utts)} train utterances, {len(val_utts)} val utterances\n", flush=True)

    ph_tokenizer = PhonemeTokenizer()
    ph_extractor = PhonemeTargetExtractor(ph_tokenizer)
    roman_tok = RomanCharTokenizer()

    train_dataset = MultilingualAudioDataset(
        train_utts,
        roman_tokenizer=roman_tok,
        phoneme_extractor=ph_extractor,
        max_duration_seconds=args.max_duration_seconds,
    )
    val_dataset = MultilingualAudioDataset(
        val_utts,
        roman_tokenizer=roman_tok,
        phoneme_extractor=ph_extractor,
        max_duration_seconds=args.max_duration_seconds,
    )

    batch_sampler = MultilingualBalancedBatchSampler(
        train_dataset.utterances,
        batch_size=args.batch_size,
        languages=["en", "fr", "es", "it"],
        drop_last=True,
        seed=args.seed,
    )
    collator = MultilingualSpeechCollator(roman_tokenizer=roman_tok)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    val_sampler = MultilingualBalancedBatchSampler(
        val_dataset.utterances,
        batch_size=args.batch_size,
        languages=["en", "fr", "es", "it"],
        drop_last=False,
        seed=args.seed,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collator,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    token_to_macro = ph_tokenizer.get_articulatory_mapping().to(device)

    print("🏗️  Initializing Phono-V8.0 Acoustic Backbone (Module 1)...", flush=True)
    acoustic_model = PhonoV8AcousticBackbone()

    start_step = 0
    saved_val_per = None
    latest_v8_path = os.path.join(out_dir, "latest_v8_acoustic.pt")
    if args.resume and os.path.isfile(latest_v8_path):
        print(f"🔄 Resuming Phono-V8 checkpoint from {latest_v8_path}...")
        ckpt = torch.load(latest_v8_path, map_location="cpu", weights_only=False)
        acoustic_model.load_state_dict(ckpt["acoustic_state_dict"])
        start_step = int(ckpt.get("step", 0))
        saved_val_per = ckpt.get("val_per")
        print(f"📍 Resuming training from step {start_step} (saved best val PER: {saved_val_per:.2f}%)...")
    elif os.path.isfile(args.warm_start_ckpt):
        print(f"🔄 Warm-starting Phono-V8 from V7.5 weights: {args.warm_start_ckpt}...")
        transferred, fresh = warm_start_phono_v8_from_v75(acoustic_model, args.warm_start_ckpt, device=torch.device("cpu"))
        print(f"✅ Warm-start complete: transferred {transferred} tensors into DeepSeek MoE & Conformer ({fresh} fresh Engram/Router tensors)")
    else:
        print(f"⚠️ Warning: checkpoint {args.warm_start_ckpt} not found! Initializing randomly.")

    acoustic_model = acoustic_model.to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    total_params = sum(p.numel() for p in acoustic_model.parameters())
    active_params = total_params - (6 * 8 * (512 * 2048 * 2))  # Routed MoE top-2 savings
    print(f"📊 Total Model Capacity: {total_params / 1e6:.2f}M | Active Compute per Frame: ~{active_params / 1e6:.2f}M\n")

    optimizer = torch.optim.AdamW(
        acoustic_model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01,
    )
    total_opt_steps = max(1, args.max_steps // args.grad_accum)
    warmup_opt_steps = max(1, args.warmup_steps // args.grad_accum)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_opt_steps,
        num_training_steps=total_opt_steps,
    )
    if start_step > 0:
        opt_steps_done = start_step // args.grad_accum
        for _ in range(opt_steps_done):
            scheduler.step()
        print(f"⏩ Advanced LR scheduler to step {start_step} (opt step {opt_steps_done}, LR={scheduler.get_last_lr()[0]:.2e})")

    flag_map = {"en": "🇬🇧 EN", "fr": "🇫🇷 FR", "es": "🇪🇸 ES", "it": "🇮🇹 IT"}
    latest_train_samples: Dict[str, Dict[str, Any]] = {}
    log_cycle_idx = 0

    print(f"🔬 Running clean validation baseline audit before training (resuming at step {start_step})...")
    init_val = evaluate_acoustic(
        acoustic_model, val_loader, ph_tokenizer, device, token_to_macro, max_eval_batches=25, chunk_eval=args.chunk_eval
    )
    init_score = min(init_val["val_per"], init_val.get("val_per_cal", init_val["val_per"]))
    best_val_per = min(init_score, saved_val_per) if saved_val_per is not None else init_score
    init_bd = " | ".join([
        f"{flag_map.get(l, l.upper())}: {p:4.1f}% (cal: {init_val.get('val_per_by_lang_cal', {}).get(l, p):4.1f}%)"
        for l, p in init_val.get("val_per_by_lang", {}).items()
    ])
    print(
        f"  📊 Step {start_step} Baseline | Val PER: \033[1;31m{init_val['val_per']:5.2f}%\033[0m (Calibrated: \033[1;32m{init_val.get('val_per_cal', init_val['val_per']):5.2f}%\033[0m) | "
        f"Space Acc: {init_val['val_spc_acc']:5.2f}% | Val Loss: {init_val['val_loss']:.4f} | Best Record: {best_val_per:.2f}%"
    )
    if init_bd:
        print(f"     🌍 Multilingual Baseline: {init_bd}\n", flush=True)

    print(f"🔥 Launching Phono-V8.0 Training ({start_step} -> {args.max_steps} steps)...\n")
    acoustic_model.train()
    optimizer.zero_grad()

    global_step = start_step
    running_loss = 0.0
    running_ctc = 0.0
    running_draft = 0.0
    running_macro = 0.0
    running_moe = 0.0
    running_per_ed = 0
    running_per_len = 0
    running_spc_correct = 0
    running_spc_gt = 0

    train_iter = iter(train_loader)
    t_start = time.time()

    while global_step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        audio = batch["audio"].to(device)
        audio_lengths = batch["audio_lengths"].to(device)
        phoneme_targets = batch["phoneme_targets"].to(device)
        phoneme_lengths = batch["phoneme_lengths"].to(device)
        macro_targets = token_to_macro[phoneme_targets.clamp(0, 111)]

        # Dynamic Acoustic Augmentation
        if acoustic_model.training:
            if torch.rand(1).item() < 0.35:
                noise_scale = torch.empty(audio.shape[0], 1, device=device).uniform_(0.001, 0.005)
                audio = audio + torch.randn_like(audio) * noise_scale
            if torch.rand(1).item() < 0.35:
                gain = torch.empty(audio.shape[0], 1, device=device).uniform_(0.85, 1.15)
                audio = (audio * gain).clamp(-1.0, 1.0)

        out = acoustic_model(
            audio=audio,
            audio_lengths=audio_lengths,
            targets=phoneme_targets,
            target_lengths=phoneme_lengths,
            macro_targets=macro_targets,
            macro_lengths=phoneme_lengths,
        )

        loss = out["loss"]
        ctc_loss = out["ctc_loss"]
        draft_ctc_loss = out.get("draft_ctc_loss", torch.tensor(0.0, device=device))
        macro_loss = out.get("macro_ctc_loss", torch.tensor(0.0, device=device))
        aux_moe_loss = out.get("aux_moe_loss", torch.tensor(0.0, device=device))

        (loss / args.grad_accum).backward()

        running_loss += loss.item()
        running_ctc += ctc_loss.item()
        running_draft += draft_ctc_loss.item()
        running_macro += macro_loss.item()
        running_moe += aux_moe_loss.item()

        logits = out["logits"]
        for i in range(len(audio)):
            l_i = batch["languages"][i]
            p_toks = decode_ctc_phoneme_ids(logits[i])
            g_toks = phoneme_targets[i][: phoneme_lengths[i]].cpu().tolist()
            ed_i = editdistance.eval(p_toks, g_toks)
            ref_i = max(1, len(g_toks))
            spc_i = compute_spacing_errors(g_toks, p_toks, space_token=8)

            latest_train_samples[l_i] = {
                "lang": l_i,
                "text": batch["transcripts"][i],
                "gt_ph": ph_tokenizer.decode(g_toks, skip_special=True),
                "pred_ph": ph_tokenizer.decode(p_toks, skip_special=True),
                "per": (ed_i / ref_i) * 100.0,
                "spc_res": spc_i,
            }

            running_per_ed += ed_i
            running_per_len += ref_i
            running_spc_correct += spc_i["correct"]
            running_spc_gt += spc_i["total_gt"]

        if (global_step + 1) % args.grad_accum == 0 or (global_step + 1) == args.max_steps:
            torch.nn.utils.clip_grad_norm_(acoustic_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        global_step += 1

        if global_step % args.log_every == 0 or global_step == 1:
            t_now = time.time()
            elapsed_sec = t_now - t_start
            steps_done = max(1, global_step - start_step)
            rate = steps_done / max(1e-4, elapsed_sec)
            rem_sec = (args.max_steps - global_step) / max(1e-4, rate)
            eta_str = format_eta(rem_sec)

            avg_loss = running_loss / max(1, args.log_every if global_step > 1 else 1)
            avg_ctc = running_ctc / max(1, args.log_every if global_step > 1 else 1)
            avg_draft = running_draft / max(1, args.log_every if global_step > 1 else 1)
            avg_macro = running_macro / max(1, args.log_every if global_step > 1 else 1)
            avg_moe = running_moe / max(1, args.log_every if global_step > 1 else 1)
            train_per = (running_per_ed / max(1, running_per_len)) * 100.0
            spc_acc = (running_spc_correct / max(1, running_spc_gt)) * 100.0 if running_spc_gt > 0 else 100.0
            cur_lr = scheduler.get_last_lr()[0]

            lang_cycle = ["en", "fr", "es", "it"]
            target_lang = lang_cycle[log_cycle_idx % len(lang_cycle)]
            log_cycle_idx += 1

            sample = latest_train_samples.get(target_lang)
            if sample is None and len(latest_train_samples) > 0:
                sample = next(iter(latest_train_samples.values()))

            s_lang = sample["lang"] if sample else batch["languages"][0]
            flag = flag_map.get(s_lang, s_lang.upper())

            print(
                f"Step {global_step}/{args.max_steps} | Loss: {avg_loss:.4f} (CTC: {avg_ctc:.4f}, Draft: {avg_draft:.4f}, Macro: {avg_macro:.4f}, MoE: {avg_moe:.4f}) | "
                f"Train PER: {train_per:5.2f}% | SpcAcc: {spc_acc:5.2f}% | LR: {cur_lr:.2e} | Rate: {rate:4.1f} st/s | ETA: {eta_str}"
            )
            if sample is not None:
                spc_r = sample["spc_res"]
                print(
                    f"  🔍 [Step {global_step} Sample ({flag})] | PER: {sample['per']:4.1f}% | "
                    f"Space Acc: {spc_r['accuracy']:4.1f}% (Missed: {spc_r['missed']}, Extra: {spc_r['extra']})\n"
                    f"    • GT Text      : {sample['text'][:140]}\n"
                    f"    • GT Phonemes  : {sample['gt_ph'][:140]}\n"
                    f"    • Pred Phonemes: {sample['pred_ph'][:140]}"
                )

            running_loss = 0.0
            running_ctc = 0.0
            running_draft = 0.0
            running_macro = 0.0
            running_moe = 0.0
            running_per_ed = 0
            running_per_len = 0
            running_spc_correct = 0
            running_spc_gt = 0

        # Periodic Validation & Checkpointing
        if global_step % args.eval_every == 0:
            print("\n" + "─" * 80)
            print(f"⏳ Running Clean Validation at Step {global_step}...")
            val_res = evaluate_acoustic(
                acoustic_model,
                val_loader,
                ph_tokenizer,
                device,
                token_to_macro,
                max_eval_batches=30,
                chunk_eval=args.chunk_eval,
            )

            cal_score = val_res.get("val_per_cal", val_res["val_per"])
            is_best = cal_score < best_val_per
            record_str = " 🌟 NEW BEST PHONO-V8 RECORD!" if is_best else ""
            if is_best:
                best_val_per = cal_score

            bd_str = " | ".join([
                f"{flag_map.get(l, l.upper())}: {p:4.1f}% (cal: {val_res.get('val_per_by_lang_cal', {}).get(l, p):4.1f}%)"
                for l, p in val_res.get("val_per_by_lang", {}).items()
            ])
            print(
                f"  📊 Val Results: PER: \033[1;31m{val_res['val_per']:5.2f}%\033[0m (Calibrated: \033[1;32m{cal_score:5.2f}%\033[0m) | "
                f"Space Acc: {val_res['val_spc_acc']:5.2f}% | Loss: {val_res['val_loss']:.4f}{record_str}\n"
                f"     🌍 Multilingual PER: {bd_str}"
            )
            print("─" * 80)

            for l_k, s_v in val_res["samples"].items():
                s_flag = flag_map.get(l_k, l_k.upper())
                print(
                    f"  🔍 [Val {s_flag}] | Greedy PER: {s_v['per_greedy']:4.1f}% | Cal PER: {s_v['per_calib']:4.1f}% | Spc Acc: {s_v['spc_acc']:4.1f}%\n"
                    f"    • GT Text      : {s_v['text'][:140]}\n"
                    f"    • GT Phonemes  : {s_v['gt'][:140]}\n"
                    f"    • Pred (Greedy): {s_v['pred_greedy'][:140]}\n"
                    f"    • Pred (Calib) : {s_v['pred_calib'][:140]}"
                )
            print("─" * 80 + "\n")

            # Save Checkpoints
            save_payload = {
                "step": global_step,
                "acoustic_state_dict": acoustic_model.state_dict(),
                "val_per": cal_score,
                "best_val_per": best_val_per,
                "args": vars(args),
            }
            torch.save(save_payload, latest_v8_path)
            if is_best:
                best_v8_path = os.path.join(out_dir, "best_v8_acoustic.pt")
                torch.save(save_payload, best_v8_path)
                print(f"  💾 Saved New Best Record Checkpoint -> {best_v8_path}\n")

            acoustic_model.train()

    print(f"\n🎉 Phono-V8.0 Acoustic Training Completed across {args.max_steps} steps!")


if __name__ == "__main__":
    main()
