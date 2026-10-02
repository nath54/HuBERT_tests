#!/usr/bin/env python3
"""Phono-V6.7 Speech Training Script: Joint Continuous Audio-to-Text Model with Double Loss.

Key Architecture Mechanics:
1. Double Loss Objective:
   - First Model (PhonoV6.4 Gated Diffusion): Penalized by CTC phoneme loss + diffusion loss
   - Second Model (PhonoV6.7 Windowed Decoder): Penalized by character cross-entropy + 4-path length loss + MoE aux loss
   - L_total = L_decoder + lambda_ctc * L_phoneme_ctc
2. Balanced 4-Way Multilingual Batch Mixing:
   - Every single batch contains an equal proportion of:
     [English (LibriSpeech), Italian (MLS), Spanish (MLS), French (MLS + Common Voice)]
   - Prevents MoE router collapse and catastrophic forgetting.
3. Dual Learning Rates:
   - Encoder: LR = 1e-5 (fine-tuning project record 5.87% PER backbone)
   - Decoder: LR = 3e-4 (primary training of windowed MoE decoder)
4. Strict --only_save_best checkpointing to checkpoints/phono_v6_7_speech/best_checkpoint.pt.
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from src.models.phono_v6_7_speech_model import (
    PhonoV67SpeechConfig,
    PhonoV67SpeechModel,
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


def decode_ctc_phonemes(
    logits: torch.Tensor,
    tokenizer: PhonemeTokenizer,
    blank_id: int = 1,
    pad_id: int = 0,
) -> str:
    """Greedy CTC collapse and decode into readable IPA phonemes."""
    preds = logits.argmax(dim=-1).cpu().tolist()
    collapsed = []
    prev = None
    for p in preds:
        if p != prev:
            if p != blank_id and p != pad_id:
                collapsed.append(p)
            prev = p
    return tokenizer.decode(collapsed, skip_special=True)


@torch.no_grad()
def evaluate(
    model: PhonoV67SpeechModel,
    dataloader: DataLoader,
    roman_tok: RomanCharTokenizer,
    ph_tokenizer: PhonemeTokenizer,
    device: torch.device,
    max_batches: int = 15,
) -> Dict[str, float]:
    """Fast evaluation on held-out multilingual speech batches."""
    model.eval()
    total_loss = 0.0
    total_enc_loss = 0.0
    total_dec_loss = 0.0
    total_path_acc = 0.0
    total_char_correct = 0
    total_char_tokens = 0
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
            enc_log = out.get("refined_logits") if out.get("refined_logits") is not None else out.get("enc_logits")
            pred_phonemes = decode_ctc_phonemes(enc_log[b_idx], ph_tokenizer) if enc_log is not None else "N/A"
            sample_info = {
                "flag": lang_flag,
                "gt_phonemes": gt_phonemes,
                "pred_phonemes": pred_phonemes,
                "gt_text": gt_text,
                "pred_text": pred_text,
            }

    count = max(batches_run, 1)
    return {
        "val_loss": total_loss / count,
        "val_enc_loss": total_enc_loss / count,
        "val_dec_loss": total_dec_loss / count,
        "val_path_acc": total_path_acc / count,
        "val_char_acc": (total_char_correct / max(total_char_tokens, 1)) * 100.0,
        "sample_info": sample_info,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train Phono-V6.7 Joint Speech Model with Double Loss.")
    parser.add_argument("--batch_size", type=int, default=4, help="Micro batch size across 4 languages (1 per lang)")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps (effective batch size = batch_size * grad_accum = 16)")
    parser.add_argument("--encoder_lr", type=float, default=1e-5, help="Fine-tuning learning rate for PhonoV6.4 backbone")
    parser.add_argument("--decoder_lr", type=float, default=3e-4, help="Primary learning rate for PhonoV6.7 decoder")
    parser.add_argument("--ctc_loss_weight", type=float, default=0.5, help="Weight of CTC phoneme loss in double loss")
    parser.add_argument("--max_steps", type=int, default=20000, help="Total training steps")
    parser.add_argument("--warmup_steps", type=int, default=500, help="LR warmup steps")
    parser.add_argument("--eval_every", type=int, default=200, help="Evaluation frequency in steps")
    parser.add_argument("--log_every", type=int, default=1, help="Logging frequency in steps (default: 1 for every step)")
    parser.add_argument("--max_samples_per_lang", type=int, default=None, help="Cap samples per language (useful for testing)")
    parser.add_argument("--max_duration", type=float, default=10.0, help="Max audio clip duration in seconds")
    parser.add_argument("--output_dir", type=str, default="checkpoints/phono_v6_7_speech/medium")
    parser.add_argument("--only_save_best", action="store_true", default=True, help="Strictly keep only best checkpoint")
    parser.add_argument("--encoder_ckpt", type=str, default="checkpoints/phono_v6_4_gated_diffusion/medium/v6_4_960h/best_checkpoint.pt")
    parser.add_argument("--decoder_ckpt", type=str, default="checkpoints/phono_v6_6_adaptive/best_checkpoint.pt")
    parser.add_argument("--smoke_test", action="store_true", help="Run 10 steps smoke test and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("🚀 Phono-V6.7 Joint Continuous Speech Training: Double Loss & 4-Way Mixing")
    print(f"  Device:         {device}")
    print(f"  Batch size:     {args.batch_size} (4-way balanced: {args.batch_size // 4} per language)")
    print(f"  Encoder LR:     {args.encoder_lr}")
    print(f"  Decoder LR:     {args.decoder_lr}")
    print(f"  CTC Weight:     {args.ctc_loss_weight}")
    print(f"  Max Steps:      {args.max_steps}")
    print(f"  Output Dir:     {args.output_dir}")
    print("=" * 70)

    # 1. Load Multilingual Utterances across 4 Languages
    print("\n📂 Indexing Multilingual Datasets...")
    all_train_utterances: List[AudioUtterance] = []
    all_val_utterances: List[AudioUtterance] = []

    # 🇬🇧 English: LibriSpeech 100h
    en_train_json = "data/librispeech/librispeech_train_100h.json"
    en_val_json = "data/librispeech/benchmark_val.json"
    if os.path.exists(en_train_json):
        en_train = load_librispeech_manifest(en_train_json, max_samples=args.max_samples_per_lang)
        all_train_utterances.extend(en_train)
        print(f"  🇬🇧 English (LibriSpeech): {len(en_train):,} train samples")
    if os.path.exists(en_val_json):
        all_val_utterances.extend(load_librispeech_manifest(en_val_json, max_samples=100))

    # 🇮🇹 Italian: OpenSLR MLS Italian
    it_train = load_mls_manifest("/media/hdd/Datasets/mls/mls_italian", lang="it", split="train", max_samples=args.max_samples_per_lang)
    it_val = load_mls_manifest("/media/hdd/Datasets/mls/mls_italian", lang="it", split="dev", max_samples=100)
    all_train_utterances.extend(it_train)
    all_val_utterances.extend(it_val)
    print(f"  🇮🇹 Italian (MLS):        {len(it_train):,} train samples")

    # 🇪🇸 Spanish: OpenSLR MLS Spanish
    es_train = load_mls_manifest("/media/hdd/Datasets/mls/mls_spanish", lang="es", split="train", max_samples=args.max_samples_per_lang)
    es_val = load_mls_manifest("/media/hdd/Datasets/mls/mls_spanish", lang="es", split="dev", max_samples=100)
    all_train_utterances.extend(es_train)
    all_val_utterances.extend(es_val)
    print(f"  🇪🇸 Spanish (MLS):        {len(es_train):,} train samples")

    # 🇫🇷 French: OpenSLR MLS French (+ Common Voice fallback)
    fr_train = load_mls_manifest("/media/hdd/Datasets/mls/mls_french", lang="fr", split="train", max_samples=args.max_samples_per_lang)
    if not fr_train:
        fr_train = load_common_voice_manifest("/media/hdd/Datasets/common_voice/french/cv-corpus-27.0-2026-09-11/fr", lang="fr", split="train", max_samples=args.max_samples_per_lang)
    fr_val = load_mls_manifest("/media/hdd/Datasets/mls/mls_french", lang="fr", split="dev", max_samples=100)
    if not fr_val:
        fr_val = load_common_voice_manifest("/media/hdd/Datasets/common_voice/french/cv-corpus-27.0-2026-09-11/fr", lang="fr", split="dev", max_samples=100)
    all_train_utterances.extend(fr_train)
    all_val_utterances.extend(fr_val)
    print(f"  🇫🇷 French (MLS/CV):      {len(fr_train):,} train samples")

    print(f"  Total Indexed Train Samples: {len(all_train_utterances):,}")
    print(f"  Total Indexed Val Samples:   {len(all_val_utterances):,}")

    # 2. Build Datasets & Balanced Samplers
    roman_tok = RomanCharTokenizer()
    ph_tokenizer = PhonemeTokenizer()
    ph_extractor = PhonemeTargetExtractor(tokenizer=ph_tokenizer)

    train_dataset = MultilingualAudioDataset(
        all_train_utterances,
        roman_tokenizer=roman_tok,
        phoneme_extractor=ph_extractor,
        max_duration_seconds=args.max_duration,
    )
    val_dataset = MultilingualAudioDataset(
        all_val_utterances,
        roman_tokenizer=roman_tok,
        phoneme_extractor=ph_extractor,
        max_duration_seconds=args.max_duration,
    )

    collator = MultilingualSpeechCollator(roman_tokenizer=roman_tok)

    train_sampler = MultilingualBalancedBatchSampler(
        all_train_utterances,
        batch_size=args.batch_size,
        languages=["en", "it", "es", "fr"],
    )
    val_sampler = MultilingualBalancedBatchSampler(
        all_val_utterances,
        batch_size=min(args.batch_size, 8),
        languages=["en", "it", "es", "fr"],
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

    # 3. Build Model & Warm-Start
    print("\n🧠 Initializing Phono-V6.7 Speech Model...")
    config = PhonoV67SpeechConfig.medium(ctc_loss_weight=args.ctc_loss_weight)
    model = PhonoV67SpeechModel(config).to(device)

    # Count parameters
    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    total_params = enc_params + dec_params
    print(f"  Model Parameters: {total_params:,} (Encoder: {enc_params:,} | Decoder: {dec_params:,})")

    # Warm-start from best checkpoints
    print("\n🔥 Warm-Starting Weights:")
    ws_res = model.warm_start_from_checkpoints(
        encoder_checkpoint_path=args.encoder_ckpt if os.path.exists(args.encoder_ckpt) else None,
        decoder_checkpoint_path=args.decoder_ckpt if os.path.exists(args.decoder_ckpt) else None,
    )
    if ws_res["encoder"]:
        print(f"  Encoder: Transferred {ws_res['encoder']['transferred']} tensors from {args.encoder_ckpt}")
    if ws_res["decoder"]:
        print(f"  Decoder: Transferred {ws_res['decoder']['transferred']} tensors from {args.decoder_ckpt}")

    # 4. Optimizers with Dual Learning Rates
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr, "weight_decay": 0.01},
            {"params": model.decoder.parameters(), "lr": args.decoder_lr, "weight_decay": 0.01},
        ],
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = out_dir / "best_checkpoint.pt"

    best_val_loss = float("inf")
    step = 0
    accum_step = 0
    start_time = time.time()
    train_iter = iter(train_loader)

    print("\n🏁 Starting Training Loop...")
    model.train()
    optimizer.zero_grad()

    max_steps = 10 if args.smoke_test else args.max_steps

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
            scaler.step(optimizer)
            scaler.update()
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
                    f"Loss: {raw_loss.item():.4f} (Enc: {out['enc_loss'].item():.3f}, Dec: {out['dec_loss'].item():.3f}) | "
                    f"PathAcc: {out['path_acc'].item():5.1f}% | "
                    f"LR: [E:{cur_enc_lr:.1e}, D:{cur_dec_lr:.1e}] | "
                    f"Rate: {rate:.2f} steps/s | ETA: {eta}",
                    flush=True,
                )

                # Inspect a random sample from the batch
                b_idx = random.randint(0, len(batch["languages"]) - 1)
                lang = batch["languages"][b_idx]
                lang_flag = {"en": "🇬🇧 EN", "it": "🇮🇹 IT", "es": "🇪🇸 ES", "fr": "🇫🇷 FR"}.get(lang, lang.upper())

                # Ground truth & predicted text
                gt_text = batch["transcripts"][b_idx]
                preds_bytes = out["logits"][b_idx].argmax(dim=-1).cpu().tolist()
                num_w = batch["num_words"][b_idx].item()
                pred_text = roman_tok.decode_words(preds_bytes[: min(num_w, len(preds_bytes))])

                # Ground truth & predicted phonemes
                gt_ph_len = batch["phoneme_lengths"][b_idx].item()
                gt_ph_tokens = batch["phoneme_targets"][b_idx, :gt_ph_len].cpu().tolist()
                gt_phonemes = ph_tokenizer.decode(gt_ph_tokens, skip_special=True)

                enc_logits = out.get("refined_logits") if out.get("refined_logits") is not None else out.get("enc_logits")
                if enc_logits is not None:
                    pred_phonemes = decode_ctc_phonemes(enc_logits[b_idx], ph_tokenizer)
                else:
                    pred_phonemes = "N/A"

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
                    f"(Enc: {val_metrics['val_enc_loss']:.3f}, Dec: {val_metrics['val_dec_loss']:.3f}) | "
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
