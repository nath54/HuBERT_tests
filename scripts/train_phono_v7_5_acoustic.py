#!/usr/bin/env python3
"""Phono-V7.5 Isolated Acoustic-to-Phoneme Training Script.

Trains strictly Module 1 (Acoustic Conformer Backbone + Recursive Phoneme Head + Boundary Gate)
WITHOUT touching Model Parts 2 & 3 (FastTextPhonemeWordEncoder & OverlappingLengthMoEDecoder).
"""

import argparse
import difflib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import editdistance
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from src.models.phono_v7_3_speech_model import (
    PhonoV73SpeechModel,
    PhonoV73AcousticBackbone,
    decode_calibrated_phonemes,
)


def format_eta(seconds: float) -> str:
    if seconds < 0 or math.isinf(seconds) or math.isnan(seconds):
        return "--m--s"
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"


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


def decode_ctc_phonemes(logits: torch.Tensor, tokenizer: PhonemeTokenizer, blank_id: int = 1, pad_id: int = 0) -> str:
    collapsed = decode_ctc_phoneme_ids(logits, blank_id=blank_id, pad_id=pad_id)
    return tokenizer.decode(collapsed, skip_special=True)


def compute_spacing_errors(gt_tokens: List[int], pred_tokens: List[int], space_token: int = 8) -> Dict[str, Any]:
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


def load_acoustic_backbone_weights(
    model: PhonoV73AcousticBackbone,
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, int]:
    """Cleanly warm-starts PhonoV73AcousticBackbone with automatic tensor splicing
    from 56/64-class checkpoints into 104/112-class architecture."""
    model_dict = model.state_dict()
    transferred = 0
    spliced = 0

    for k, v in state_dict.items():
        target_k = None
        if k in model_dict:
            target_k = k
        elif k.startswith("encoder.") and k[len("encoder.") :] in model_dict:
            target_k = k[len("encoder.") :]
        elif f"encoder.{k}" in model_dict:
            target_k = f"encoder.{k}"

        if target_k is None:
            continue

        if model_dict[target_k].shape == v.shape:
            model_dict[target_k].copy_(v)
            transferred += 1
        elif "base_head.weight" in target_k and model_dict[target_k].ndim == v.ndim:
            min_c = min(model_dict[target_k].shape[0], v.shape[0])
            model_dict[target_k][:min_c].copy_(v[:min_c])
            transferred += 1
            spliced += 1
        elif "base_head.bias" in target_k and model_dict[target_k].ndim == v.ndim:
            min_c = min(model_dict[target_k].shape[0], v.shape[0])
            model_dict[target_k][:min_c].copy_(v[:min_c])
            transferred += 1
            spliced += 1
        elif "refiner_in.weight" in target_k and model_dict[target_k].ndim == v.ndim:
            t_w = model_dict[target_k]
            t_w[:, :512, :].copy_(v[:, :512, :])
            min_p = min(t_w.shape[1] - 512, v.shape[1] - 512)
            t_w[:, 512 : 512 + min_p, :].copy_(v[:, 512 : 512 + min_p, :])
            transferred += 1
            spliced += 1
        elif "refiner_out.weight" in target_k and model_dict[target_k].ndim == v.ndim:
            min_c = min(model_dict[target_k].shape[0], v.shape[0])
            model_dict[target_k][:min_c].copy_(v[:min_c])
            transferred += 1
            spliced += 1
        elif "refiner_out.bias" in target_k and model_dict[target_k].ndim == v.ndim:
            min_c = min(model_dict[target_k].shape[0], v.shape[0])
            model_dict[target_k][:min_c].copy_(v[:min_c])
            transferred += 1
            spliced += 1

    model.load_state_dict(model_dict)
    return {"transferred": transferred, "spliced": spliced}


@torch.no_grad()
def evaluate_acoustic(
    acoustic_model: PhonoV73AcousticBackbone,
    dataloader: DataLoader,
    ph_tokenizer: PhonemeTokenizer,
    device: torch.device,
    token_to_macro: torch.Tensor,
    max_eval_batches: int = 30,
    chunk_eval: bool = False,
) -> Dict[str, Any]:
    acoustic_model.eval()
    total_loss = 0.0
    total_ctc_loss = 0.0
    total_macro_loss = 0.0
    total_bnd_loss = 0.0
    total_phoneme_ed = 0
    total_phoneme_ed_cal = 0
    total_phoneme_ref_len = 0
    total_spc_gt = 0
    total_spc_correct = 0
    total_spc_missed = 0
    total_spc_extra = 0
    batches_run = 0

    per_by_lang: Dict[str, Dict[str, int]] = {l: {"ed": 0, "ed_cal": 0, "ref": 0} for l in ["en", "fr", "es", "it"]}
    samples_by_lang: Dict[str, Any] = {}

    for b_idx, batch in enumerate(dataloader):
        if b_idx >= max_eval_batches:
            break
        batches_run += 1

        audio = batch["audio"].to(device)
        audio_lengths = batch["audio_lengths"].to(device)
        phoneme_targets = batch["phoneme_targets"].to(device)
        phoneme_lengths = batch["phoneme_lengths"].to(device)
        macro_targets = token_to_macro[phoneme_targets.clamp(0, 111)]

        out = acoustic_model(
            audio=audio,
            audio_lengths=audio_lengths,
            targets=phoneme_targets,
            target_lengths=phoneme_lengths,
            macro_targets=macro_targets,
            macro_lengths=phoneme_lengths,
        )

        loss = out["loss"]
        if not torch.isnan(loss):
            total_loss += loss.item()
            total_ctc_loss += out["ctc_loss"].item()
            total_macro_loss += out.get("macro_ctc_loss", torch.tensor(0.0)).item()
            total_bnd_loss += out.get("boundary_loss", torch.tensor(0.0)).item()

        logits = out["logits"]
        for i in range(len(audio)):
            lang = batch["languages"][i]
            a_len = audio_lengths[i].item()
            if chunk_eval and a_len > 128000:
                chunk_res = acoustic_model.forward_sliding_chunks(audio[i : i + 1, :a_len])
                eval_logits = chunk_res["logits"][0]
            else:
                eval_logits = logits[i]

            pred_tokens = decode_ctc_phoneme_ids(eval_logits)
            pred_tokens_cal = decode_calibrated_phonemes(eval_logits, blank_penalty=0.4, min_tokens_between_spaces=1)

            gt_tokens = phoneme_targets[i][: phoneme_lengths[i]].cpu().tolist()
            ed = editdistance.eval(pred_tokens, gt_tokens)
            ed_cal = editdistance.eval(pred_tokens_cal, gt_tokens)
            ref_len = max(1, len(gt_tokens))
            total_phoneme_ed += ed
            total_phoneme_ed_cal += ed_cal
            total_phoneme_ref_len += ref_len

            if lang in per_by_lang:
                per_by_lang[lang]["ed"] += ed
                per_by_lang[lang]["ed_cal"] += ed_cal
                per_by_lang[lang]["ref"] += ref_len

            spc_res = compute_spacing_errors(gt_tokens, pred_tokens, space_token=8)
            total_spc_gt += spc_res["total_gt"]
            total_spc_correct += spc_res["correct"]
            total_spc_missed += spc_res["missed"]
            total_spc_extra += spc_res["extra"]

            if lang not in samples_by_lang:
                samples_by_lang[lang] = {
                    "lang": lang,
                    "gt_text": batch["transcripts"][i],
                    "gt_phonemes": ph_tokenizer.decode(gt_tokens, skip_special=True),
                    "pred_phonemes": ph_tokenizer.decode(pred_tokens, skip_special=True),
                    "pred_phonemes_cal": ph_tokenizer.decode(pred_tokens_cal, skip_special=True),
                    "per": (ed / ref_len) * 100.0,
                    "per_cal": (ed_cal / ref_len) * 100.0,
                    "spc_info": spc_res,
                }

    n = max(1, batches_run)
    val_per = (total_phoneme_ed / max(1, total_phoneme_ref_len)) * 100.0
    val_per_cal = (total_phoneme_ed_cal / max(1, total_phoneme_ref_len)) * 100.0
    val_spc_acc = (total_spc_correct / max(1, total_spc_gt)) * 100.0 if total_spc_gt > 0 else 100.0

    lang_per_dict = {}
    lang_per_dict_cal = {}
    for l, d in per_by_lang.items():
        if d["ref"] > 0:
            lang_per_dict[l] = (d["ed"] / max(1, d["ref"])) * 100.0
            lang_per_dict_cal[l] = (d["ed_cal"] / max(1, d["ref"])) * 100.0

    return {
        "val_loss": total_loss / n,
        "val_ctc_loss": total_ctc_loss / n,
        "val_macro_loss": total_macro_loss / n,
        "val_bnd_loss": total_bnd_loss / n,
        "val_per": val_per,
        "val_per_cal": val_per_cal,
        "val_spc_acc": val_spc_acc,
        "val_spc_missed": total_spc_missed,
        "val_spc_extra": total_spc_extra,
        "val_per_by_lang": lang_per_dict,
        "val_per_by_lang_cal": lang_per_dict_cal,
        "samples_by_lang": samples_by_lang,
    }


def main():
    parser = argparse.ArgumentParser(description="Isolated Training for Phono-V7.5 Acoustic-to-Phoneme Model")
    parser.add_argument("--librispeech_train", type=str, default="data/librispeech/benchmark_train_960h.json")
    parser.add_argument("--librispeech_val", type=str, default="data/librispeech/benchmark_val.json")
    parser.add_argument("--mls_italian", type=str, default="/media/hdd/Datasets/mls/mls_italian")
    parser.add_argument("--mls_spanish", type=str, default="/media/hdd/Datasets/mls/mls_spanish")
    parser.add_argument("--mls_french", type=str, default="/media/hdd/Datasets/mls/mls_french")
    parser.add_argument("--cv_french", type=str, default="/media/hdd/Datasets/common_voice/french/cv-corpus-27.0-2026-09-11/fr")
    parser.add_argument(
        "--val_clean_manifest",
        type=str,
        default="data/multilingual_val_clean.json",
        help="Path to sanitized multilingual validation manifest",
    )

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (e.g. 4 on TITAN X)")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_duration_seconds", type=float, default=14.0, help="Max audio duration in seconds")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=1.0e-5, help="Minimum learning rate")
    parser.add_argument("--max_steps", type=int, default=45000, help="Total training steps")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Learning rate warmup steps")
    parser.add_argument("--eval_every", type=int, default=250, help="Evaluate every N steps")
    parser.add_argument("--log_every", type=int, default=20, help="Log metrics and sample every N steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_eval", action="store_true", default=True, help="Use sliding multi-window chunk evaluation")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume training step count and scheduler from checkpoint if step metadata exists",
    )

    parser.add_argument(
        "--warm_start_ckpt",
        type=str,
        default="checkpoints/phono_v7_5_acoustic/latest_acoustic.pt"
        if os.path.exists("checkpoints/phono_v7_5_acoustic/latest_acoustic.pt")
        else ("checkpoints/phono_v7_5_acoustic/best_acoustic.pt"
              if os.path.exists("checkpoints/phono_v7_5_acoustic/best_acoustic.pt")
              else "checkpoints/phono_v7_3/streaming/best_checkpoint.pt"),
        help="Path to initial acoustic checkpoint to warm start from",
    )
    parser.add_argument("--max_samples_en", type=int, default=60000, help="Max English train utterances")
    parser.add_argument("--max_samples_fr", type=int, default=50000, help="Max French train utterances")
    parser.add_argument("--max_samples_es", type=int, default=50000, help="Max Spanish train utterances")
    parser.add_argument("--max_samples_it", type=int, default=35000, help="Max Italian train utterances")
    parser.add_argument("--max_char_len", type=int, default=180, help="Max transcript char len for zero truncation")
    parser.add_argument("--it_patched_manifest", type=str, default="data/it_train_patched.json", help="Path to patched Italian transcripts")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/phono_v7_5_acoustic",
        help="Directory to save acoustic checkpoints",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🎙️  Phono-V7.5 Isolated Acoustic-to-Phoneme Training Suite")
    print("=" * 80)
    print(f"📦 Warm-Start Checkpoint : {args.warm_start_ckpt}")
    print(f"📁 Checkpoint Output Dir : {out_dir}")
    print(f"⚡ Device                 : {device.type.upper()}")
    print(f"🎯 Total Steps           : {args.max_steps}")
    print(f"🔥 Micro-Batch / Accum   : {args.batch_size} / {args.grad_accum} (Effective BS = {args.batch_size * args.grad_accum})")
    print(f"📈 Learning Rate         : {args.lr} -> {args.min_lr}")

    print("\n📚 Loading Multilingual Speech Datasets...", flush=True)
    train_utts: List[AudioUtterance] = []
    val_utts: List[AudioUtterance] = []

    en_train_p = args.librispeech_train if os.path.exists(args.librispeech_train) else "data/librispeech/librispeech_train_100h.json"
    en_train = load_librispeech_manifest(en_train_p, max_samples=args.max_samples_en)
    en_val = load_librispeech_manifest(args.librispeech_val, max_samples=400)
    print(f"  • English (LibriSpeech) : {len(en_train)} train, {len(en_val)} raw val")
    train_utts.extend(en_train)

    fr_train = load_mls_manifest(args.mls_french, lang="fr", split="train", max_samples=args.max_samples_fr, max_char_len=args.max_char_len)
    if os.path.isdir(args.cv_french):
        fr_cv = load_common_voice_manifest(args.cv_french, lang="fr", split="train", max_samples=15000)
        fr_train.extend(fr_cv)
    fr_val = load_mls_manifest(args.mls_french, lang="fr", split="dev", max_samples=400, max_char_len=args.max_char_len)
    print(f"  • French (MLS + CV)     : {len(fr_train)} train, {len(fr_val)} raw val")
    train_utts.extend(fr_train)

    es_train = load_mls_manifest(args.mls_spanish, lang="es", split="train", max_samples=args.max_samples_es, max_char_len=args.max_char_len)
    es_val = load_mls_manifest(args.mls_spanish, lang="es", split="dev", max_samples=400, max_char_len=args.max_char_len)
    print(f"  • Spanish (MLS)         : {len(es_train)} train, {len(es_val)} raw val")
    train_utts.extend(es_train)

    it_train = load_mls_manifest(args.mls_italian, lang="it", split="train", max_samples=args.max_samples_it, max_char_len=args.max_char_len)
    if args.it_patched_manifest and os.path.isfile(args.it_patched_manifest):
        with open(args.it_patched_manifest, "r", encoding="utf-8") as f:
            patched_it_list = json.load(f)
        it_patched_map = {item["id"]: item["transcript"] for item in patched_it_list}
        patched_applied = 0
        for u in it_train:
            if u.id in it_patched_map:
                u.transcript = it_patched_map[u.id]
                patched_applied += 1
        print(f"  ✨ Injected {patched_applied} acoustically-verified patched Italian transcripts!")
    it_val = load_mls_manifest(args.mls_italian, lang="it", split="dev", max_samples=400, max_char_len=args.max_char_len)
    print(f"  • Italian (MLS)         : {len(it_train)} train, {len(it_val)} raw val")
    train_utts.extend(it_train)

    if args.val_clean_manifest and os.path.isfile(args.val_clean_manifest):
        print(f"  ✨ Using sanitized validation manifest: {args.val_clean_manifest}")
        with open(args.val_clean_manifest, "r", encoding="utf-8") as f:
            clean_data = json.load(f)
        val_utts = [
            AudioUtterance(
                id=item["id"],
                audio_path=item["audio_path"],
                transcript=item["transcript"],
                lang=item["lang"],
                duration=float(item.get("duration", 0.0)),
            )
            for item in clean_data
        ]
        print(f"  • Sanitized Multilingual Validation: {len(val_utts)} utterances (zero unexpanded digit/truncation noise)")
    else:
        val_utts.extend(en_val)
        val_utts.extend(fr_val)
        val_utts.extend(es_val)
        val_utts.extend(it_val)

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

    print("🏗️  Initializing Acoustic Backbone (Module 1)...", flush=True)
    temp_model = PhonoV73SpeechModel()
    acoustic_model = temp_model.encoder
    del temp_model.decoder
    del temp_model
    import gc
    gc.collect()

    start_step = 0
    saved_val_per = None
    if os.path.isfile(args.warm_start_ckpt):
        print(f"🔄 Warm-starting from {args.warm_start_ckpt}...")
        ckpt = torch.load(args.warm_start_ckpt, map_location="cpu", weights_only=False)
        sd = ckpt.get("acoustic_state_dict", ckpt.get("model_state_dict", ckpt))
        res = load_acoustic_backbone_weights(acoustic_model, sd)
        print(f"✅ Warm-start complete: transferred {res['transferred']} tensors (spliced {res['spliced']} tensors for expanded 112 vocab)")
        if args.resume and "step" in ckpt:
            start_step = int(ckpt["step"])
            saved_val_per = ckpt.get("val_per")
            print(f"📍 Resuming training from step {start_step} (saved best val PER: {saved_val_per:.2f}%)...")
    else:
        print(f"⚠️ Warning: checkpoint {args.warm_start_ckpt} not found! Initializing randomly.")

    acoustic_model = acoustic_model.to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    param_count = sum(p.numel() for p in acoustic_model.parameters() if p.requires_grad)
    print(f"📊 Active Acoustic Parameters: {param_count / 1e6:.2f}M (Decoders 2 & 3 completely detached)\n")

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

    print(f"🔬 Running validation audit before training (resuming at step {start_step})...")
    init_val = evaluate_acoustic(acoustic_model, val_loader, ph_tokenizer, device, token_to_macro, max_eval_batches=20, chunk_eval=args.chunk_eval)
    init_score = min(init_val["val_per"], init_val.get("val_per_cal", init_val["val_per"]))
    best_val_per = min(init_score, saved_val_per) if saved_val_per is not None else init_score
    init_bd = " | ".join([f"{flag_map.get(l, l.upper())}: {p:4.1f}% (cal: {init_val.get('val_per_by_lang_cal', {}).get(l, p):4.1f}%)" for l, p in init_val.get("val_per_by_lang", {}).items()])
    print(
        f"  📊 Step {start_step} Baseline | Val PER: \033[1;31m{init_val['val_per']:5.2f}%\033[0m (Calibrated: \033[1;32m{init_val.get('val_per_cal', init_val['val_per']):5.2f}%\033[0m) | "
        f"Space Acc: {init_val['val_spc_acc']:5.2f}% | Val Loss: {init_val['val_loss']:.4f} | Best Record: {best_val_per:.2f}%"
    )
    if init_bd:
        print(f"     🌍 Multilingual Baseline: {init_bd}\n", flush=True)

    print(f"🔥 Resuming Isolated Acoustic Training ({start_step} -> {args.max_steps} steps)...\n")
    acoustic_model.train()
    optimizer.zero_grad()

    global_step = start_step
    running_loss = 0.0
    running_ctc = 0.0
    running_macro = 0.0
    running_bnd = 0.0
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

        # Dynamic Acoustic Augmentation (Audio Noise & Gain Perturbation)
        if acoustic_model.training:
            # 1. Additive Gaussian noise with 35% probability (SNR 25-40 dB)
            if torch.rand(1).item() < 0.35:
                noise_scale = torch.empty(audio.shape[0], 1, device=device).uniform_(0.001, 0.005)
                audio = audio + torch.randn_like(audio) * noise_scale
            # 2. Random gain perturbation (+/- 2.0 dB) with 35% probability
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
        macro_loss = out.get("macro_ctc_loss", torch.tensor(0.0, device=device))
        boundary_loss = out.get("boundary_loss", torch.tensor(0.0, device=device))

        (loss / args.grad_accum).backward()

        running_loss += loss.item()
        running_ctc += ctc_loss.item()
        running_macro += macro_loss.item()
        running_bnd += boundary_loss.item()

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
            rate = global_step / max(1e-4, elapsed_sec)
            rem_sec = (args.max_steps - global_step) / max(1e-4, rate)
            eta_str = format_eta(rem_sec)

            avg_loss = running_loss / max(1, args.log_every if global_step > 1 else 1)
            avg_ctc = running_ctc / max(1, args.log_every if global_step > 1 else 1)
            avg_macro = running_macro / max(1, args.log_every if global_step > 1 else 1)
            avg_bnd = running_bnd / max(1, args.log_every if global_step > 1 else 1)
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
            s_flag = flag_map.get(s_lang, s_lang.upper())
            s_text = sample["text"] if sample else batch["transcripts"][0]
            s_gt_ph = sample["gt_ph"] if sample else ""
            s_pred_ph = sample["pred_ph"] if sample else ""
            s_per = sample["per"] if sample else 0.0
            s_spc = sample["spc_res"] if sample else {"acc": 100.0, "missed": 0, "extra": 0}

            print(
                f"Step {global_step:>5d}/{args.max_steps} | "
                f"Loss: {avg_loss:6.4f} (CTC: {avg_ctc:6.4f}, Macro: {avg_macro:6.4f}, Bnd: {avg_bnd:6.4f}) | "
                f"Train PER: \033[1;33m{train_per:5.2f}%\033[0m | "
                f"SpcAcc: {spc_acc:5.2f}% | "
                f"LR: {cur_lr:.2e} | "
                f"Rate: {rate:4.1f} st/s | "
                f"ETA: {eta_str}",
                flush=True,
            )
            print(f"  🔍 [Step {global_step} Sample ({s_flag})] | PER: {s_per:4.1f}% | Space Acc: {s_spc['acc']:4.1f}% (Missed: {s_spc['missed']}, Extra: {s_spc['extra']})")
            print(f"    • GT Text      : {s_text}")
            print(f"    • GT Phonemes  : {s_gt_ph}")
            print(f"    • Pred Phonemes: {s_pred_ph}", flush=True)

            running_loss = 0.0
            running_ctc = 0.0
            running_macro = 0.0
            running_bnd = 0.0
            running_per_ed = 0
            running_per_len = 0
            running_spc_correct = 0
            running_spc_gt = 0

        if global_step % args.eval_every == 0 or global_step == args.max_steps:
            print(f"\n{'─'*80}")
            print(f"⏳ Running Validation at Step {global_step}...")
            val_res = evaluate_acoustic(acoustic_model, val_loader, ph_tokenizer, device, token_to_macro, max_eval_batches=30, chunk_eval=args.chunk_eval)
            val_per = val_res["val_per"]
            val_per_cal = val_res.get("val_per_cal", val_per)
            val_spc = val_res["val_spc_acc"]
            val_loss = val_res["val_loss"]

            eval_score = min(val_per, val_per_cal)
            is_best = eval_score < best_val_per
            best_tag = "🌟 NEW BEST ACOUSTIC RECORD!" if is_best else ""
            print(
                f"  📊 Val Results: PER: \033[1;36m{val_per:5.2f}%\033[0m (Calibrated: \033[1;32m{val_per_cal:5.2f}%\033[0m) | "
                f"Space Acc: {val_spc:5.2f}% | Loss: {val_loss:6.4f} {best_tag}"
            )
            lang_breakdown = " | ".join(
                [f"{flag_map.get(l, l.upper())}: {p:4.1f}% (cal: {val_res.get('val_per_by_lang_cal', {}).get(l, p):4.1f}%)"
                 for l, p in val_res.get("val_per_by_lang", {}).items()]
            )
            if lang_breakdown:
                print(f"     🌍 Multilingual PER: {lang_breakdown}")
            print(f"{'─'*80}")

            for l_key in ["en", "fr", "es", "it"]:
                s_l = val_res.get("samples_by_lang", {}).get(l_key)
                if s_l:
                    cal_str = f" | Cal PER: {s_l.get('per_cal', s_l['per']):4.1f}%" if "per_cal" in s_l else ""
                    print(f"  🔍 [Val {flag_map.get(l_key, l_key.upper())}] | Greedy PER: {s_l['per']:4.1f}%{cal_str} | Spc Acc: {s_l['spc_info']['acc']:4.1f}%")
                    print(f"    • GT Text      : {s_l['gt_text']}")
                    print(f"    • GT Phonemes  : {s_l['gt_phonemes']}")
                    print(f"    • Pred (Greedy): {s_l['pred_phonemes']}")
                    if "pred_phonemes_cal" in s_l:
                        print(f"    • Pred (Calib) : {s_l['pred_phonemes_cal']}")
            print(f"{'─'*80}\n", flush=True)

            full_sd = {}
            for k, v in acoustic_model.state_dict().items():
                full_sd[f"encoder.{k}"] = v.cpu()

            ckpt_payload = {
                "step": global_step,
                "val_per": val_per,
                "val_per_cal": val_per_cal,
                "val_spc_acc": val_spc,
                "val_loss": val_loss,
                "model_state_dict": full_sd,
                "acoustic_state_dict": {k: v.cpu() for k, v in acoustic_model.state_dict().items()},
            }

            torch.save(ckpt_payload, out_dir / "latest_acoustic.pt")

            if is_best:
                best_val_per = eval_score
                torch.save(ckpt_payload, out_dir / "best_acoustic.pt")
                print(f"  💾 Saved new best acoustic checkpoint: {out_dir / 'best_acoustic.pt'} (Best PER: {best_val_per:.2f}%)\n", flush=True)

            acoustic_model.train()

    print("\n" + "=" * 80)
    print(f"🎉 Isolated Acoustic Training Complete! Best Validation PER: {best_val_per:.2f}%")
    print(f"💾 Checkpoints saved to: {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
