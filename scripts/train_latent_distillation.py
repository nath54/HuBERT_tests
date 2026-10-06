"""Train Fast Text/Phoneme Word Encoder via Latent Space Distillation.

Distills the stabilized acoustic word manifold z_word from Phono-V7.3/V7.5
into a lightweight 2-layer FastTextPhonemeWordEncoder operating at 50x speed.
"""

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

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
from src.models.phono_v7_3_speech_model import PhonoV73SpeechModel
from src.models.phono_v7_1_speech_model import PhonoV71SpeechConfig
from src.models.text_phoneme_encoder import FastTextPhonemeWordEncoder
from src.models.latent_distillation import LatentSpaceDistillationLoss


def main():
    parser = argparse.ArgumentParser(description="Train Latent Space Distillation")
    parser.add_argument("--librispeech_train", type=str, default="data/librispeech/librispeech_train_100h.json")
    parser.add_argument("--librispeech_val", type=str, default="data/librispeech/benchmark_val.json")
    parser.add_argument("--mls_italian", type=str, default="/media/hdd/Datasets/mls/mls_italian")
    parser.add_argument("--mls_spanish", type=str, default="/media/hdd/Datasets/mls/mls_spanish")
    parser.add_argument("--mls_french", type=str, default="/media/hdd/Datasets/mls/mls_french")
    parser.add_argument("--cv_french", type=str, default="/media/hdd/Datasets/common_voice/french/cv-corpus-27.0-2026-09-11/fr")

    parser.add_argument("--teacher_ckpt", type=str, default="checkpoints/phono_v7_3/streaming/best_checkpoint.pt")
    parser.add_argument("--save_dir", type=str, default="checkpoints/latent_distillation")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Latent Space Distillation on: {device}", flush=True)

    # 1. Dataset setup
    phono_tok = PhonemeTokenizer()
    rom_tok = RomanCharTokenizer()
    extractor = PhonemeTargetExtractor(phono_tok)

    train_utts: List[AudioUtterance] = []
    val_utts: List[AudioUtterance] = []

    if Path(args.librispeech_train).is_file():
        en_train = load_librispeech_manifest(args.librispeech_train)
        en_val = load_librispeech_manifest(args.librispeech_val)
        train_utts.extend(en_train[:15000] if not args.smoke_test else en_train[:50])
        val_utts.extend(en_val[:500] if not args.smoke_test else en_val[:20])

    if Path(args.mls_italian).is_dir():
        it_train = load_mls_manifest(args.mls_italian, lang="it", split="train", max_samples=10000 if not args.smoke_test else 50)
        it_val = load_mls_manifest(args.mls_italian, lang="it", split="dev", max_samples=200 if not args.smoke_test else 20)
        train_utts.extend(it_train)
        val_utts.extend(it_val)

    if Path(args.mls_spanish).is_dir():
        es_train = load_mls_manifest(args.mls_spanish, lang="es", split="train", max_samples=10000 if not args.smoke_test else 50)
        es_val = load_mls_manifest(args.mls_spanish, lang="es", split="dev", max_samples=200 if not args.smoke_test else 20)
        train_utts.extend(es_train)
        val_utts.extend(es_val)

    if Path(args.mls_french).is_dir():
        fr_train = load_mls_manifest(args.mls_french, lang="fr", split="train", max_samples=10000 if not args.smoke_test else 50)
        fr_val = load_mls_manifest(args.mls_french, lang="fr", split="dev", max_samples=200 if not args.smoke_test else 20)
        train_utts.extend(fr_train)
        val_utts.extend(fr_val)

    print(f"📚 Loaded {len(train_utts)} train utterances, {len(val_utts)} val utterances", flush=True)

    train_ds = MultilingualAudioDataset(train_utts, roman_tokenizer=rom_tok, phoneme_extractor=extractor)
    val_ds = MultilingualAudioDataset(val_utts, roman_tokenizer=rom_tok, phoneme_extractor=extractor)

    collator = MultilingualSpeechCollator(rom_tok)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=1)

    # 2. Load Teacher Acoustic Model
    print(f"\n🔄 Loading frozen Teacher model from: {args.teacher_ckpt}...", flush=True)
    teacher_cfg = PhonoV71SpeechConfig.medium()
    teacher = PhonoV73SpeechModel(teacher_cfg).to(device)
    if Path(args.teacher_ckpt).is_file():
        ckpt = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
        teacher.load_state_dict(ckpt["model_state_dict"], strict=False)
        print("  • Teacher weights loaded successfully", flush=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # 3. Create Student Fast Text/Phoneme Encoder
    student = FastTextPhonemeWordEncoder(
        d_macro=teacher_cfg.decoder_config.macro_dim,
        d_model=256,
        nhead=4,
        num_layers=2,
    ).to(device)
    student.train()
    print(f"  • Student encoder created ({sum(p.numel() for p in student.parameters()) / 1e6:.2f}M params)", flush=True)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-2)
    max_steps = 15 if args.smoke_test else args.max_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=max_steps)
    distill_loss_fn = LatentSpaceDistillationLoss(mse_weight=1.0, cos_weight=0.5, nce_weight=0.1)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    print(f"\n🔥 Starting Latent Space Distillation Training ({max_steps} steps)...", flush=True)
    step = 0
    train_iter = iter(train_loader)

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        audio = batch["audio"].to(device, non_blocking=True)
        audio_lengths = batch["audio_lengths"].to(device, non_blocking=True)
        num_words = batch.get("num_words")
        input_bytes = batch.get("input_byte_ids")
        target_bytes = batch.get("target_byte_ids")
        target_lengths = batch.get("target_lengths")

        if num_words is not None:
            num_words = num_words.to(device, non_blocking=True)
        if input_bytes is not None:
            input_bytes = input_bytes.to(device, non_blocking=True)
            target_bytes = target_bytes.to(device, non_blocking=True)
        if target_lengths is not None:
            target_lengths = target_lengths.to(device, non_blocking=True)

        # Teacher acoustic forward (no grad)
        with torch.no_grad():
            t_out = teacher(
                audio=audio,
                audio_lengths=audio_lengths,
                num_words=num_words,
                input_byte_ids=input_bytes,
                target_byte_ids=target_bytes,
                target_lengths=target_lengths,
            )
            z_teacher = t_out.get("z_word")

        if z_teacher is None or input_bytes is None:
            continue

        B, L, D = z_teacher.shape
        K = input_bytes.shape[2]

        flat_bytes = input_bytes[:, :L].reshape(B * L, K)
        flat_teacher = z_teacher.reshape(B * L, D)

        valid_words = (flat_bytes[:, 1] != 0)  # non-pad words
        if not valid_words.any():
            continue

        # Student forward
        z_student = student(byte_ids=flat_bytes[valid_words])

        loss, metrics = distill_loss_fn(z_student, flat_teacher[valid_words])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        step += 1

        if step % 10 == 0 or args.smoke_test:
            print(
                f"Step {step:4d}/{max_steps} | DistillLoss: {metrics['distill_loss']:.4f} | "
                f"MSE: {metrics['mse']:.4f} | CosSim: {metrics['cos_sim']:.3f} | NCE: {metrics['nce']:.3f}",
                flush=True,
            )

        if (step % args.eval_every == 0 or step == max_steps) and not args.smoke_test:
            if metrics["distill_loss"] < best_loss:
                best_loss = metrics["distill_loss"]
                torch.save(
                    {"step": step, "student_state_dict": student.state_dict(), "metrics": metrics},
                    save_dir / "best_student.pt",
                )
                print(f"  💾 Saved best student checkpoint (CosSim={metrics['cos_sim']:.3f})", flush=True)

    print(f"\n🎉 Distillation completed successfully!", flush=True)


if __name__ == "__main__":
    main()
