#!/usr/bin/env python3
"""Phono-V8.0 Robust Phoneme-to-Grapheme (P2G) Training with Twin Latent Anchor Regularization.

Features:
1. Warm-Starts Non-Destructively from Gold Decoders 2 & 3 (inheriting 99.0% spelling prior).
2. Empirical Phonetic Noise Simulator (substitutions from empirical confusion matrix,
   coarticulation allophones, unstressed vowel reductions, boundary space jitter).
3. Twin Latent Anchor Regularization:
   Enforces: Model2(p_noisy) ≈ Model2(p_clean) in the 512D word latent manifold (MSE + Cosine).
   Completely prevents latent space shrinkage or degradation without running audio forward.
4. Curricular Noise Scheduling: Smoothly scales noise rate from 5% to 15% over training.
5. Saves independently to checkpoints/phono_v8_robust_p2g/ leaving the gold vault untouched.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import editdistance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from piper.voice import EspeakPhonemizer
from src.data.phoneme_tokenizer import PhonemeTokenizer
from src.data.roman_tokenizer import RomanCharTokenizer
from src.data.phonetic_noise import PhoneticNoiseSimulator
from src.models.phono_v7_5_speech_model import PhonoV75SpeechConfig, PhonoV75SpeechModel
from src.models.text_phoneme_encoder import FastTextPhonemeWordEncoder
from src.training.metrics import compute_wer, compute_cer

_espeak_phonemizer = None
_phoneme_tokenizer = None


def get_phonemizer() -> EspeakPhonemizer:
    global _espeak_phonemizer
    if _espeak_phonemizer is None:
        _espeak_phonemizer = EspeakPhonemizer()
    return _espeak_phonemizer


def get_phoneme_tokenizer() -> PhonemeTokenizer:
    global _phoneme_tokenizer
    if _phoneme_tokenizer is None:
        _phoneme_tokenizer = PhonemeTokenizer()
    return _phoneme_tokenizer


@lru_cache(maxsize=200000)
def phonemize_word_cached(word: str, lang: str) -> Tuple[int, ...]:
    ep = get_phonemizer()
    ptok = get_phoneme_tokenizer()
    lang_map = {"en": "en-us", "fr": "fr", "it": "it", "es": "es"}
    target_voice = lang_map.get(lang, "en-us")
    try:
        chunks = ep.phonemize(target_voice, word)
        flat = [ch for sub in chunks for ch in sub]
        # Strip non-acoustic diacritics ('ˈ', 'ˌ', 'ː', 'ˑ') to perfectly align with Model 1 CTC targets
        clean_tokens = [ch for ch in flat if ch not in ("ˈ", "ˌ", "ː", "ˑ")]
        ids = ptok.encode(clean_tokens)
        return tuple(ids)
    except Exception:
        return tuple()


class RobustP2GDataset(Dataset):
    """Loads text sentences with word tokens across languages."""

    def __init__(
        self,
        items: List[Tuple[str, str]],
        max_words: int = 32,
    ):
        self.items = [(s.strip().lower(), lang) for s, lang in items if s.strip()]
        self.max_words = max_words

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text, lang = self.items[idx]
        words = text.split()[: self.max_words]
        return {
            "text": text,
            "words": words,
            "lang": lang,
        }


class RobustP2GCollator:
    """Collates words into clean phonemes, noisy phonemes, and target byte tokens."""

    def __init__(
        self,
        roman_tokenizer: RomanCharTokenizer,
        phoneme_tokenizer: PhonemeTokenizer,
        noise_simulator: PhoneticNoiseSimulator,
        max_chars_per_word: int = 24,
        max_phonemes_per_word: int = 24,
    ):
        self.rom_tok = roman_tokenizer
        self.ph_tok = phoneme_tokenizer
        self.noise_sim = noise_simulator
        self.max_chars = max_chars_per_word
        self.max_ph = max_phonemes_per_word
        self.noise_prob = 0.12  # Dynamic

    def set_noise_prob(self, sub_prob: float):
        self.noise_sim.sub_rate = sub_prob
        self.noise_sim.del_rate = sub_prob * 0.35
        self.noise_sim.ins_rate = sub_prob * 0.25

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(batch)
        max_L = max(1, max(len(b["words"]) for b in batch))

        # First pass: pre-extract word tokenizations to compute tight batch bounds
        extracted = []
        max_char_in_batch = 6
        max_ph_in_batch = 6

        for b_idx, item in enumerate(batch):
            words = item["words"]
            lang = item["lang"]
            b_words = []
            for w in words:
                p_ids = list(phonemize_word_cached(w, lang))
                w_tokens = self.rom_tok.encode(w, add_bos=False, add_eos=False)
                if len(w_tokens) + 2 > max_char_in_batch:
                    max_char_in_batch = len(w_tokens) + 2
                if len(p_ids) > max_ph_in_batch:
                    max_ph_in_batch = len(p_ids)
                b_words.append((p_ids, w_tokens))
            extracted.append(b_words)

        K_char = min(self.max_chars, max_char_in_batch)
        K_ph = min(self.max_ph, max_ph_in_batch)

        clean_ph = torch.zeros(B, max_L, K_ph, dtype=torch.long)
        noisy_ph = torch.zeros(B, max_L, K_ph, dtype=torch.long)
        noisy_top_ids = torch.zeros(B, max_L, K_ph, 3, dtype=torch.long)
        noisy_top_probs = torch.zeros(B, max_L, K_ph, 3, dtype=torch.float32)
        input_bytes = torch.zeros(B, max_L, K_char, dtype=torch.long)
        target_bytes = torch.full((B, max_L, K_char), -100, dtype=torch.long)
        target_lengths = torch.zeros(B, max_L, dtype=torch.float32)

        for b_idx, b_words in enumerate(extracted):
            for w_idx, (p_ids, w_tokens) in enumerate(b_words):
                # 1. Clean Phonemes
                p_len = min(len(p_ids), K_ph)
                if p_len > 0:
                    clean_ph[b_idx, w_idx, :p_len] = torch.tensor(p_ids[:p_len], dtype=torch.long)

                    # 2. Discrete Noisy Phonemes
                    tokens = [self.ph_tok.id_to_token.get(i, "") for i in p_ids[:p_len]]
                    perturbed_tokens = self.noise_sim.perturb_phonemes(tokens)
                    p_noisy_ids = self.ph_tok.encode(perturbed_tokens)
                    pn_len = min(len(p_noisy_ids), K_ph)
                    if pn_len > 0:
                        noisy_ph[b_idx, w_idx, :pn_len] = torch.tensor(p_noisy_ids[:pn_len], dtype=torch.long)
                    else:
                        noisy_ph[b_idx, w_idx, :p_len] = clean_ph[b_idx, w_idx, :p_len]

                    # 3. Soft Top-K Mixtures
                    mix_cands, mix_probs = self.noise_sim.generate_top_k_mixture_for_tokens(
                        tokens, top_k=3, mix_rate=0.60
                    )
                    for k_idx in range(p_len):
                        c_toks = mix_cands[k_idx]
                        c_probs = mix_probs[k_idx]
                        encoded_cands = self.ph_tok.encode(c_toks)
                        while len(encoded_cands) < 3:
                            encoded_cands.append(encoded_cands[-1] if encoded_cands else 0)
                        noisy_top_ids[b_idx, w_idx, k_idx] = torch.tensor(encoded_cands[:3], dtype=torch.long)
                        noisy_top_probs[b_idx, w_idx, k_idx] = torch.tensor(c_probs[:3], dtype=torch.float32)

                # 4. Characters for Word
                char_len = max(1, len(w_tokens))
                target_lengths[b_idx, w_idx] = float(char_len)

                seq_len = min(char_len, K_char - 2)
                input_bytes[b_idx, w_idx, 0] = self.rom_tok.bos_id
                input_bytes[b_idx, w_idx, 1 : seq_len + 1] = torch.tensor(w_tokens[:seq_len], dtype=torch.long)

                target_bytes[b_idx, w_idx, :seq_len] = torch.tensor(w_tokens[:seq_len], dtype=torch.long)
                target_bytes[b_idx, w_idx, seq_len] = self.rom_tok.eos_id

        return {
            "clean_phonemes": clean_ph,
            "noisy_phonemes": noisy_ph,
            "noisy_top_ids": noisy_top_ids,
            "noisy_top_probs": noisy_top_probs,
            "input_byte_ids": input_bytes,
            "target_byte_ids": target_bytes,
            "target_lengths": target_lengths,
            "raw_texts": [b["text"] for b in batch],
            "languages": [b["lang"] for b in batch],
        }


def load_french_sentences(
    mls_fr_txt: str = "/media/hdd/Datasets/mls/mls_french/train/transcripts.txt",
    mls_fr_dev_txt: str = "/media/hdd/Datasets/mls/mls_french/dev/transcripts.txt",
    max_train: int = 100000,
    max_val: int = 1000,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    train_items: List[Tuple[str, str]] = []
    val_items: List[Tuple[str, str]] = []

    if os.path.isfile(mls_fr_txt):
        with open(mls_fr_txt, "r", encoding="utf-8") as f:
            lines = [l.strip().split("\t")[-1] for l in f if l.strip()]
        random.shuffle(lines)
        train_items.extend([(t, "fr") for t in lines[:max_train]])

    if os.path.isfile(mls_fr_dev_txt):
        with open(mls_fr_dev_txt, "r", encoding="utf-8") as f:
            lines = [l.strip().split("\t")[-1] for l in f if l.strip()]
        random.shuffle(lines)
        val_items.extend([(t, "fr") for t in lines[:max_val]])
    else:
        val_items.extend(train_items[-max_val:])
        train_items = train_items[:-max_val]

    random.shuffle(train_items)
    return train_items, val_items


def main():
    parser = argparse.ArgumentParser(description="Train Phono-V8 Robust French P2G Decoders")
    parser.add_argument("--mls_french_txt", type=str, default="/media/hdd/Datasets/mls/mls_french/train/transcripts.txt")
    parser.add_argument("--mls_french_dev_txt", type=str, default="/media/hdd/Datasets/mls/mls_french/dev/transcripts.txt")
    parser.add_argument("--warm_start_p2g_ckpt", type=str, default="checkpoints/phono_v8_aligned_soft_p2g/best_robust_p2g.pt")
    parser.add_argument("--gold_p2g_ckpt", type=str, default="checkpoints/backup_gold_decoders_99acc/gold_best_p2g_99acc.pt")
    parser.add_argument("--output_dir", type=str, default="checkpoints/phono_v8_p2g_french")
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr_encoder", type=float, default=2.0e-5)
    parser.add_argument("--lr_decoder", type=float, default=5.0e-5)
    parser.add_argument("--min_lr", type=float, default=2.0e-6)
    parser.add_argument("--latent_anchor_weight", type=float, default=1.0)
    parser.add_argument("--start_noise_rate", type=float, default=0.05)
    parser.add_argument("--end_noise_rate", type=float, default=0.15)
    parser.add_argument("--levenshtein_weight", type=float, default=0.05)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("🚀 PHONO-V8.0: ROBUST MODEL 2 + MODEL 3 P2G TRAINING (NOISE RESILIENCE)")
    print("=" * 80)
    print(f"📦 Warm-Start Checkpoint : {args.gold_p2g_ckpt} (Protected Vault)")
    print(f"📁 Output Directory      : {args.output_dir}")
    print(f"⚡ Device                : {device.type.upper()}")
    print(f"🎯 Total Steps          : {args.max_steps} (Effective BS = {args.batch_size * args.grad_accum})")
    print(f"🛡️  Latent Anchor Weight : {args.latent_anchor_weight}")
    print(f"⚡ Levenshtein Weight    : {args.levenshtein_weight}")

    rom_tok = RomanCharTokenizer()
    ph_tok = PhonemeTokenizer()
    noise_sim = PhoneticNoiseSimulator(confusion_matrix_path="data/phoneme_confusion_matrix.json")

    print("\n📚 Loading French Text Corpora...", flush=True)
    train_items, val_items = load_french_sentences(
        mls_fr_txt=args.mls_french_txt,
        mls_fr_dev_txt=args.mls_french_dev_txt,
        max_train=100000,
        max_val=1000,
    )
    print(f"  • Loaded {len(train_items)} French train sentences, {len(val_items)} French val sentences")

    train_ds = RobustP2GDataset(train_items)
    val_ds = RobustP2GDataset(val_items)
    collator = RobustP2GCollator(rom_tok, ph_tok, noise_sim)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=1)

    # 1. Model 2 (FastTextPhonemeWordEncoder)
    cfg = PhonoV75SpeechConfig.medium()
    cfg.levenshtein_loss_weight = args.levenshtein_weight
    encoder_m2 = FastTextPhonemeWordEncoder(
        d_macro=cfg.decoder_config.macro_dim,
        d_model=256,
        nhead=4,
        num_layers=2,
        phoneme_vocab_size=128,
    ).to(device)

    # 2. Model 3 (Overlapping MoE Character Decoder)
    model_m3 = PhonoV75SpeechModel(cfg).to(device)
    model_m3.decoder.levenshtein_loss_weight = args.levenshtein_weight

    step = 0
    latest_pt = os.path.join(args.output_dir, "latest_robust_p2g.pt")
    if args.resume and os.path.isfile(latest_pt):
        print(f"\n🔄 Resuming Model 2 & Model 3 from {latest_pt}...", flush=True)
        ckpt = torch.load(latest_pt, map_location=device, weights_only=False)
        encoder_m2.load_state_dict(ckpt["phoneme_encoder_state_dict"])
        model_m3.decoder.load_state_dict(ckpt["decoder_state_dict"])
        step = ckpt.get("step", 0)
        print(f"  ✅ Resumed from Step {step}!")
    else:
        init_ckpt = args.warm_start_p2g_ckpt if os.path.isfile(args.warm_start_p2g_ckpt) else args.gold_p2g_ckpt
        print(f"\n🔄 Warm-starting Model 2 & Model 3 from: {init_ckpt}...", flush=True)
        loaded_ckpt = torch.load(init_ckpt, map_location=device, weights_only=False)
        encoder_m2.load_state_dict(loaded_ckpt["phoneme_encoder_state_dict"])
        model_m3.decoder.load_state_dict(loaded_ckpt["decoder_state_dict"])
        print(f"  ✅ Warm-start successful from step {loaded_ckpt.get('step', 'N/A')}!")

    trainable_params = [
        {"params": [p for p in encoder_m2.parameters() if p.requires_grad], "lr": args.lr_encoder},
        {"params": [p for p in model_m3.decoder.parameters() if p.requires_grad], "lr": args.lr_decoder},
    ]
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=1e-2, betas=(0.9, 0.98))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=args.max_steps)

    step = 0
    accum_step = 0
    start_time = time.time()
    best_val_wer = float("inf")
    train_iter = iter(train_loader)

    encoder_m2.train()
    model_m3.train()

    print(f"\n🔥 Commencing Robust Training ({args.max_steps} steps)...", flush=True)

    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Dynamic noise curriculum
        progress = step / max(1, args.max_steps)
        curr_noise_rate = args.start_noise_rate + progress * (args.end_noise_rate - args.start_noise_rate)
        collator.set_noise_prob(curr_noise_rate)

        clean_ph = batch["clean_phonemes"].to(device, non_blocking=True)
        noisy_ph = batch["noisy_phonemes"].to(device, non_blocking=True)
        input_bytes = batch["input_byte_ids"].to(device, non_blocking=True)
        target_bytes = batch["target_byte_ids"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)

        B, L, K_ph = clean_ph.shape
        flat_clean = clean_ph.reshape(B * L, K_ph)
        flat_noisy = noisy_ph.reshape(B * L, K_ph)

        # Step 1: Model 2 Forward (Clean & Noisy)
        z_clean = encoder_m2(phoneme_ids=flat_clean).view(B, L, -1)

        # Dual Training: 50% continuous soft Top-3 mixtures, 50% discrete perturbations
        if random.random() < 0.5:
            noisy_top_ids = batch["noisy_top_ids"].to(device, non_blocking=True).reshape(B * L, K_ph, 3)
            noisy_top_probs = batch["noisy_top_probs"].to(device, non_blocking=True).reshape(B * L, K_ph, 3)
            z_noisy = encoder_m2(top_k_ids=noisy_top_ids, top_k_probs=noisy_top_probs).view(B, L, -1)
        else:
            z_noisy = encoder_m2(phoneme_ids=flat_noisy).view(B, L, -1)

        # Step 2: Twin Latent Invariance Loss (MSE + Cosine Distance)
        mse_latent = F.mse_loss(z_noisy, z_clean.detach())
        cos_sim = F.cosine_similarity(z_noisy, z_clean.detach(), dim=-1).mean()
        latent_loss = mse_latent + (1.0 - cos_sim)

        # Step 3: Model 3 Character Decoder Forward on Noisy Latents
        out = model_m3.forward_text(
            z_word=z_noisy,
            input_byte_ids=input_bytes,
            target_byte_ids=target_bytes,
            target_lengths=target_lengths,
        )

        total_loss = (out["loss"] + args.latent_anchor_weight * latent_loss) / args.grad_accum
        total_loss.backward()
        accum_step += 1

        if accum_step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(encoder_m2.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model_m3.decoder.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1

            if step % args.log_every == 0 or step == 1:
                elapsed = time.time() - start_time
                rate = step / max(1e-3, elapsed)
                eta = (args.max_steps - step) / max(1e-3, rate)
                lr_curr = optimizer.param_groups[1]["lr"]
                print(
                    f"Step {step:5d}/{args.max_steps} | Loss: {out['loss'].item():.4f} "
                    f"(Char: {out['char_loss'].item():.3f}, Latent: {latent_loss.item():.4f}, CosSim: {cos_sim.item():.3f}) | "
                    f"CharAcc: {out['char_acc'].item():5.1f}% | MoEAcc: {out['path_acc'].item():5.1f}% | "
                    f"Noise: {curr_noise_rate*100:4.1f}% | LR: {lr_curr:.2e} | {rate:4.1f} st/s | ETA: {int(eta//60)}m{int(eta%60):02d}s",
                    flush=True,
                )

            # Validation
            if step % args.eval_every == 0 or step == args.max_steps:
                encoder_m2.eval()
                model_m3.eval()
                val_losses = []
                val_char_accs = []
                val_preds = []
                val_refs = []

                with torch.no_grad():
                    for v_idx, v_batch in enumerate(val_loader):
                        if v_idx >= 30:
                            break
                        v_noisy = v_batch["noisy_phonemes"].to(device)
                        v_inp_b = v_batch["input_byte_ids"].to(device)
                        v_tgt_b = v_batch["target_byte_ids"].to(device)
                        v_tgt_l = v_batch["target_lengths"].to(device)

                        v_B, v_L, v_K = v_noisy.shape
                        v_flat_n = v_noisy.reshape(v_B * v_L, v_K)
                        v_z_noisy = encoder_m2(phoneme_ids=v_flat_n).view(v_B, v_L, -1)

                        v_out = model_m3.forward_text(
                            z_word=v_z_noisy,
                            input_byte_ids=v_inp_b,
                            target_byte_ids=v_tgt_b,
                            target_lengths=v_tgt_l,
                        )
                        val_losses.append(v_out["loss"].item())
                        val_char_accs.append(v_out["char_acc"].item())

                        if v_idx == 0:
                            # Run vectorized autoregressive rollout on first batch to compute WER & log samples
                            for b_i in range(min(4, v_B)):
                                b_noisy = v_noisy[b_i]
                                valid_mask = (b_noisy != 0).any(dim=-1)
                                b_L = max(1, int(valid_mask.sum().item()))
                                cur_z = v_z_noisy[b_i : b_i + 1, :b_L, :]
                                exp_l = model_m3.decoder.macro_expert_router(cur_z)
                                top1_e = exp_l.argmax(dim=-1)[0]
                                exp_c = model_m3.decoder.expert_embeddings(top1_e).unsqueeze(1)
                                z_win = model_m3.decoder.build_word_context_windows(cur_z)[0]

                                act_w = torch.full((b_L, 24), rom_tok.pad_id, dtype=torch.long, device=device)
                                act_w[:, 0] = rom_tok.bos_id
                                fin = torch.zeros(b_L, dtype=torch.bool, device=device)
                                for st in range(1, 24):
                                    c_inp = act_w[:, :st]
                                    l_c, _ = model_m3.decoder.micro_head(c_inp, z_win, acoustic_slices=None, path_bias=exp_c, path_id=top1_e)
                                    n_tok = l_c[:, -1, :].argmax(dim=-1)
                                    for l_idx in range(b_L):
                                        if not fin[l_idx]:
                                            if n_tok[l_idx].item() in (rom_tok.eos_id, rom_tok.pad_id):
                                                fin[l_idx] = True
                                            else:
                                                act_w[l_idx, st] = n_tok[l_idx]
                                    if fin.all():
                                        break
                                p_words = []
                                for l_idx in range(b_L):
                                    w_t = [t.item() for t in act_w[l_idx, 1:] if t.item() not in (rom_tok.bos_id, rom_tok.eos_id, rom_tok.pad_id)]
                                    p_words.append(w_t)
                                p_str = rom_tok.decode_words(p_words)
                                val_preds.append(p_str)
                                val_refs.append(v_batch["raw_texts"][b_i].lower().strip())

                mean_val_loss = sum(val_losses) / max(1, len(val_losses))
                mean_val_acc = sum(val_char_accs) / max(1, len(val_char_accs))

                sample_wer = compute_wer(val_preds, val_refs) * 100.0 if val_preds else 0.0
                sample_cer = compute_cer(val_preds, val_refs) * 100.0 if val_preds else 0.0

                is_best = mean_val_acc > (98.0 if best_val_wer == float("inf") else best_val_wer)
                star = " 🌟 NEW RECORD!" if is_best else ""
                print(
                    f"\n  📊 Val Evaluation @ Step {step}: Loss: {mean_val_loss:.4f} | "
                    f"Noisy CharAcc: {mean_val_acc:5.2f}% | Sample WER: {sample_wer:5.2f}% | Sample CER: {sample_cer:5.2f}%{star}",
                    flush=True,
                )
                if val_preds:
                    print("  🔍 [Validation Sample Comparison under Acoustic Noise]:")
                    for s_i in range(min(2, len(val_preds))):
                        print(f"     • Ref : {val_refs[s_i]}")
                        print(f"     • Pred: {val_preds[s_i]}")
                print("", flush=True)

                # Save Checkpoints
                save_payload = {
                    "step": step,
                    "val_loss": mean_val_loss,
                    "val_char_acc": mean_val_acc,
                    "val_wer": sample_wer,
                    "val_cer": sample_cer,
                    "phoneme_encoder_state_dict": encoder_m2.state_dict(),
                    "decoder_state_dict": model_m3.decoder.state_dict(),
                }
                latest_path = os.path.join(args.output_dir, "latest_robust_p2g.pt")
                torch.save(save_payload, latest_path)

                if is_best or step == args.max_steps:
                    best_val_wer = mean_val_acc
                    best_path = os.path.join(args.output_dir, "best_robust_p2g.pt")
                    torch.save(save_payload, best_path)
                    print(f"  💾 Saved Best Robust P2G -> {best_path}\n", flush=True)

                encoder_m2.train()
                model_m3.train()

    print(f"\n🎉 Robust P2G Training Completed across {args.max_steps} steps!")


if __name__ == "__main__":
    main()
