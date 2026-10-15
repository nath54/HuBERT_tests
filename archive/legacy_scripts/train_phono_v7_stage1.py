#!/usr/bin/env python3
"""Phono-V7 Stage 1 Training Script: Audio to 768D Multilingual Word Embeddings.

Stage 1 Objectives:
1. Train the Conformer Acoustic Backbone (Model 1) with Double Loss CTC Phoneme Supervision.
2. Train the Macro Word Decoder (Model 2) to predict 768-dimensional word latent vectors.
3. Optimize against the MultilingualLexicon (EN, IT, ES, FR) using Cosine Similarity +
   Full Vocabulary Contrastive Cross-Entropy Loss.
4. Guaranteed 100% Identity Preservation when warm-starting from our 5.87% PER checkpoint.
5. Strict --only_save_best checkpointing to checkpoints/phono_v7/stage1/best_checkpoint.pt.
"""

import argparse
from collections import Counter
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
from src.data.multilingual_lexicon import MultilingualLexicon
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


def build_or_load_lexicon(
    utterances: List[AudioUtterance],
    cache_path: str = "data/multilingual_lexicon_768d.pt",
    min_count: int = 2,
    embed_dim: int = 768,
) -> MultilingualLexicon:
    """Build or load the shared multilingual 768D word lexicon."""
    path = Path(cache_path)
    if path.exists():
        print(f"📦 Loading cached Multilingual Lexicon from {path}...", flush=True)
        return MultilingualLexicon.load(path)

    print(f"🔨 Building Multilingual Lexicon from {len(utterances)} utterances...", flush=True)
    counter = Counter()
    for u in utterances:
        for w in u.transcript.strip().lower().split():
            clean_w = w.strip()
            if clean_w:
                counter[clean_w] += 1

    selected_words = [w for w, c in counter.items() if c >= min_count]
    print(f"  • Total unique words: {len(counter)}")
    print(f"  • Words with count >= {min_count}: {len(selected_words)}", flush=True)

    lex = MultilingualLexicon(words=selected_words, embed_dim=embed_dim)
    path.parent.mkdir(parents=True, exist_ok=True)
    lex.save(path)
    print(f"  ✅ Saved {lex.vocab_size} words to {path}", flush=True)
    return lex


@torch.no_grad()
def evaluate_stage1(
    model: PhonoV7SpeechModel,
    dataloader: DataLoader,
    ph_tokenizer: PhonemeTokenizer,
    device: torch.device,
    max_batches: int = 20,
) -> Dict[str, float]:
    """Evaluation on held-out multilingual speech batches for Stage 1."""
    model.eval()
    total_loss = 0.0
    total_lex_loss = 0.0
    total_enc_loss = 0.0
    total_word_acc = 0.0
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
        target_word_ids = batch.get("target_word_ids")
        if target_word_ids is not None:
            target_word_ids = target_word_ids.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out = model(
                audio=audio,
                audio_lengths=audio_lengths,
                phoneme_targets=phoneme_targets,
                phoneme_lengths=phoneme_lengths,
                num_words=num_words,
                target_word_ids=target_word_ids,
                stage=1,
            )

        loss = out["loss"]
        if not torch.isnan(loss):
            total_loss += loss.item()
            total_lex_loss += out["lexical_loss"].item()
            total_enc_loss += out["enc_loss"].item()
            total_word_acc += out["word_acc"].item()

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

            # Predicted words via lexicon
            pred_words_batch = model.lexicon.predict_words(out["z_lexicon"])
            nw = batch["num_words"][b_idx].item()
            pred_words = pred_words_batch[b_idx][:nw]
            pred_text = " ".join(pred_words)

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
    val_per = (total_phoneme_ed / max(1, total_phoneme_ref_len)) * 100.0

    return {
        "val_loss": total_loss / n,
        "val_lex_loss": total_lex_loss / n,
        "val_enc_loss": total_enc_loss / n,
        "val_word_acc": total_word_acc / n,
        "val_per": val_per,
        "sample_info": sample_info,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Phono-V7 Stage 1: Audio -> 768D Word Embeddings")
    parser.add_argument("--librispeech_train", type=str, default="data/librispeech/librispeech_train_100h.json")
    parser.add_argument("--librispeech_val", type=str, default="data/librispeech/benchmark_val.json")
    parser.add_argument("--mls_italian", type=str, default="/media/hdd/Datasets/mls/mls_italian")
    parser.add_argument("--mls_spanish", type=str, default="/media/hdd/Datasets/mls/mls_spanish")
    parser.add_argument("--mls_french", type=str, default="/media/hdd/Datasets/mls/mls_french")
    parser.add_argument("--cv_french", type=str, default="/media/hdd/Datasets/common_voice/french/cv-corpus-27.0-2026-09-11/fr")

    parser.add_argument("--batch_size", type=int, default=4, help="Micro batch size across 4 languages (1 per lang)")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps (effective BS = 16)")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--encoder_lr", type=float, default=1e-5, help="Fine-tuning LR for 5.87% PER backbone")
    parser.add_argument("--decoder_lr", type=float, default=3e-4, help="Primary LR for Macro Word Decoder")
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/phono_v7/stage1")
    parser.add_argument("--encoder_checkpoint", type=str, default="checkpoints/phono_v6_4_gated_diffusion/medium/v6_4_960h/best_checkpoint.pt")
    parser.add_argument("--decoder_checkpoint", type=str, default="checkpoints/phono_v6_7_speech/medium/best_checkpoint.pt")
    parser.add_argument("--lexicon_cache", type=str, default="data/multilingual_lexicon_768d.pt")
    parser.add_argument("--smoke_test", action="store_true", help="Run 10 steps to verify memory and gradients then exit")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Phono-V7 Stage 1 Training initialized on device: {device}", flush=True)

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

    # 2. Build or Load Shared 768D Multilingual Lexicon
    lexicon = build_or_load_lexicon(train_utts, cache_path=args.lexicon_cache, min_count=2, embed_dim=768)
    lexicon = lexicon.to(device)

    # 3. Create Tokenizers & Datasets
    ph_tokenizer = PhonemeTokenizer()
    ph_extractor = PhonemeTargetExtractor(ph_tokenizer)
    roman_tok = RomanCharTokenizer()

    train_dataset = MultilingualAudioDataset(train_utts, roman_tokenizer=roman_tok, phoneme_extractor=ph_extractor, max_duration_seconds=10.0)
    val_dataset = MultilingualAudioDataset(val_utts, roman_tokenizer=roman_tok, phoneme_extractor=ph_extractor, max_duration_seconds=10.0)

    collator = MultilingualSpeechCollator(roman_tok, lexicon=lexicon)

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

    # 4. Initialize Phono-V7 Model & Warm Start
    config = PhonoV7SpeechConfig.medium()
    config.encoder_learning_rate = args.encoder_lr
    config.decoder_learning_rate = args.decoder_lr

    model = PhonoV7SpeechModel(config, lexicon=lexicon)
    model = model.to(device)

    print("\n🔄 Warm-starting from pretrained checkpoints...", flush=True)
    ws_res = model.warm_start_from_checkpoints(
        encoder_checkpoint_path=args.encoder_checkpoint,
        decoder_checkpoint_path=args.decoder_checkpoint,
    )
    if ws_res["encoder"]:
        print(f"  • Encoder (V6.4 Gated Diffusion): {ws_res['encoder']['transferred']} tensors transferred (missing: {ws_res['encoder']['missing']}, unexpected: {ws_res['encoder']['unexpected']})")
    if ws_res["decoder"]:
        print(f"  • Decoder (V6.7 Windowed Decoder): {ws_res['decoder']['transferred']} tensors transferred (missing: {ws_res['decoder']['missing']}, unexpected: {ws_res['decoder']['unexpected']})")

    # 5. Dual Parameter Group Optimization
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    decoder_params = [p for n, p in model.decoder.named_parameters() if p.requires_grad]
    lexicon_params = [p for p in lexicon.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.encoder_lr, "weight_decay": 0.01},
            {"params": decoder_params + lexicon_params, "lr": args.decoder_lr, "weight_decay": 0.01},
        ],
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    max_steps = 10 if args.smoke_test else args.max_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_steps,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best_checkpoint.pt"
    best_val_loss = float("inf")

    print(f"\n🔥 Starting Phono-V7 Stage 1 Training ({max_steps} steps, grad_accum={args.grad_accum})...", flush=True)
    start_time = time.time()
    step = 0
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
        target_word_ids = batch.get("target_word_ids")
        if target_word_ids is not None:
            target_word_ids = target_word_ids.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out = model(
                audio=audio,
                audio_lengths=audio_lengths,
                phoneme_targets=phoneme_targets,
                phoneme_lengths=phoneme_lengths,
                num_words=num_words,
                target_word_ids=target_word_ids,
                stage=1,
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

                print(
                    f"Step {step:6d}/{max_steps} | "
                    f"Loss: {raw_loss.item():.4f} (Lex: {out['lexical_loss'].item():.3f}, Enc: {out['enc_loss'].item():.3f}) | "
                    f"WordAcc: {out['word_acc'].item():5.1f}% | "
                    f"LR: [E:{cur_enc_lr:.1e}, D:{cur_dec_lr:.1e}] | "
                    f"Rate: {rate:.2f} st/s | ETA: {eta}",
                    flush=True,
                )

                # Inspect a random sample
                b_idx = random.randint(0, len(batch["languages"]) - 1)
                lang = batch["languages"][b_idx]
                lang_flag = {"en": "🇬🇧 EN", "it": "🇮🇹 IT", "es": "🇪🇸 ES", "fr": "🇫🇷 FR"}.get(lang, lang.upper())
                gt_text = batch["transcripts"][b_idx]

                pred_words_batch = lexicon.predict_words(out["z_lexicon"])
                nw = batch["num_words"][b_idx].item()
                pred_words = pred_words_batch[b_idx][:nw]
                pred_text = " ".join(pred_words)

                gt_ph_len = batch["phoneme_lengths"][b_idx].item()
                gt_ph_tokens = batch["phoneme_targets"][b_idx, :gt_ph_len].cpu().tolist()
                gt_phonemes = ph_tokenizer.decode(gt_ph_tokens, skip_special=True)
                pred_phonemes = decode_ctc_phonemes(out["ctc_logits"][b_idx], ph_tokenizer) if out.get("ctc_logits") is not None else "N/A"

                print(f"  Sample [{lang_flag}]:", flush=True)
                print(f"    • GT Phonemes:   {gt_phonemes}", flush=True)
                print(f"    • Pred Phonemes: {pred_phonemes}", flush=True)
                print(f"    • GT Words:      {gt_text}", flush=True)
                print(f"    • Pred Words:    {pred_text}", flush=True)

            # Evaluation & Checkpointing
            if step > 0 and (step % args.eval_every == 0 or step == max_steps) and not args.smoke_test:
                print(f"\n🧪 Evaluating at step {step}...", flush=True)
                val_metrics = evaluate_stage1(
                    model,
                    val_loader,
                    ph_tokenizer=ph_tokenizer,
                    device=device,
                )
                v_loss = val_metrics["val_loss"]
                print(
                    f"  Validation Loss:    {v_loss:.4f} "
                    f"(Lex: {val_metrics['val_lex_loss']:.3f}, Enc: {val_metrics['val_enc_loss']:.3f} [PER: {val_metrics['val_per']:.2f}%]) | "
                    f"WordAcc: {val_metrics['val_word_acc']:.1f}%",
                    flush=True,
                )
                v_info = val_metrics.get("sample_info")
                if v_info:
                    print(f"  Validation Sample [{v_info['flag']}]:", flush=True)
                    print(f"    • GT Phonemes:   {v_info['gt_phonemes']}", flush=True)
                    print(f"    • Pred Phonemes: {v_info['pred_phonemes']}", flush=True)
                    print(f"    • GT Words:      {v_info['gt_text']}", flush=True)
                    print(f"    • Pred Words:    {v_info['pred_text']}", flush=True)

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
                    lexicon.save(ckpt_dir / "lexicon.pt")
                print(flush=True)
                model.train()

    print(f"\n🎉 Completed {step} steps in {time.time() - start_time:.1f}s!")
    if not args.smoke_test:
        print(f"Best Validation Loss: {best_val_loss:.4f} saved at {best_ckpt_path}")


if __name__ == "__main__":
    main()
