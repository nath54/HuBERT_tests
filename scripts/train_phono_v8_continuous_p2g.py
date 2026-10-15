#!/usr/bin/env python3
"""Phono-V8 Space-Free Continuous Phoneme-to-Word Segmenter Multilingual Training.

Features:
1. Space-Free Continuous Phoneme Input:
   Feeds full sentences as uninterrupted phoneme streams without acoustic spaces.
2. Learned End-Of-Word (EOW) Boundary Head:
   Supervised with binary cross-entropy to predict word boundaries using bidirectional phonemic context.
3. Multilingual Balanced Training:
   Trained across English, French, Spanish, and Italian text corpora.
4. Inherits Prior from Gold Decoders 2 & 3.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from functools import lru_cache
from piper.voice import EspeakPhonemizer
from src.data.phoneme_tokenizer import PhonemeTokenizer
from src.data.roman_tokenizer import RomanCharTokenizer
from src.data.phonetic_noise import PhoneticNoiseSimulator
from src.models.phono_v7_5_speech_model import PhonoV75SpeechConfig, PhonoV75SpeechModel
from src.models.text_phoneme_encoder import ContinuousPhonemeWordSegmenter
from src.inference.character_beam_search import beam_search_character_rollout
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
        clean_tokens = [ch for ch in flat if ch not in ("ˈ", "ˌ", "ː", "ˑ")]
        ids = ptok.encode(clean_tokens)
        return tuple(ids)
    except Exception:
        return tuple()


class ContinuousP2GDataset(Dataset):
    """Loads multilingual text sentences and extracts continuous phonemes with binary EOW labels."""

    def __init__(
        self,
        items: List[Tuple[str, str]],
        max_words: int = 24,
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


class ContinuousP2GCollator:
    """Collates sentences into space-free continuous phonemes, EOW boundary labels, and word byte tokens."""

    def __init__(
        self,
        roman_tokenizer: RomanCharTokenizer,
        phoneme_tokenizer: PhonemeTokenizer,
        noise_simulator: PhoneticNoiseSimulator,
        max_chars_per_word: int = 24,
        max_total_phonemes: int = 192,
    ):
        self.rom_tok = roman_tokenizer
        self.ph_tok = phoneme_tokenizer
        self.noise_sim = noise_simulator
        self.max_chars = max_chars_per_word
        self.max_total_ph = max_total_phonemes

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(batch)
        max_L = max(1, max(len(b["words"]) for b in batch))

        batch_continuous_ids = []
        batch_boundary_labels = []
        batch_word_spans = []
        batch_extracted_words = []

        for item in batch:
            words = item["words"]
            lang = item["lang"]

            sent_ph_ids: List[int] = []
            sent_b_labels: List[float] = []
            spans: List[Tuple[int, int]] = []
            b_words = []

            for w in words:
                p_ids = list(phonemize_word_cached(w, lang))
                if not p_ids:
                    continue
                w_tokens = self.rom_tok.encode(w, add_bos=False, add_eos=False)
                b_words.append((p_ids, w_tokens))

                start = len(sent_ph_ids)
                sent_ph_ids.extend(p_ids)
                end = len(sent_ph_ids)
                spans.append((start, end))

                # Boundary labels: 0 for interior phonemes, 1.0 at End-Of-Word (EOW)
                sent_b_labels.extend([0.0] * (len(p_ids) - 1) + [1.0])

            batch_continuous_ids.append(sent_ph_ids)
            batch_boundary_labels.append(sent_b_labels)
            batch_word_spans.append(spans)
            batch_extracted_words.append(b_words)

        max_T = max(1, min(self.max_total_ph, max(len(p) for p in batch_continuous_ids)))
        continuous_phonemes = torch.zeros(B, max_T, dtype=torch.long)
        boundary_targets = torch.zeros(B, max_T, dtype=torch.float32)
        pad_mask = torch.ones(B, max_T, dtype=torch.bool)

        for b_idx in range(B):
            p_ids = batch_continuous_ids[b_idx][:max_T]
            b_lbls = batch_boundary_labels[b_idx][:max_T]
            if p_ids:
                continuous_phonemes[b_idx, : len(p_ids)] = torch.tensor(p_ids, dtype=torch.long)
                boundary_targets[b_idx, : len(b_lbls)] = torch.tensor(b_lbls, dtype=torch.float32)
                pad_mask[b_idx, : len(p_ids)] = False

        # Character targets for Model 3
        max_words_in_batch = max(1, max(len(w) for w in batch_extracted_words))
        max_char_in_batch = 6
        for b_words in batch_extracted_words:
            for _, w_toks in b_words:
                if len(w_toks) + 2 > max_char_in_batch:
                    max_char_in_batch = len(w_toks) + 2
        K_char = min(self.max_chars, max_char_in_batch)

        input_bytes = torch.zeros(B, max_words_in_batch, K_char, dtype=torch.long)
        target_bytes = torch.full((B, max_words_in_batch, K_char), -100, dtype=torch.long)
        target_lengths = torch.zeros(B, max_words_in_batch, dtype=torch.float32)

        for b_idx, b_words in enumerate(batch_extracted_words):
            for w_idx, (p_ids, w_tokens) in enumerate(b_words):
                char_len = max(1, len(w_tokens))
                target_lengths[b_idx, w_idx] = float(char_len)
                seq_len = min(char_len, K_char - 2)
                input_bytes[b_idx, w_idx, 0] = self.rom_tok.bos_id
                input_bytes[b_idx, w_idx, 1 : seq_len + 1] = torch.tensor(w_tokens[:seq_len], dtype=torch.long)
                target_bytes[b_idx, w_idx, :seq_len] = torch.tensor(w_tokens[:seq_len], dtype=torch.long)
                target_bytes[b_idx, w_idx, seq_len] = self.rom_tok.eos_id

        return {
            "continuous_phonemes": continuous_phonemes,
            "boundary_targets": boundary_targets,
            "pad_mask": pad_mask,
            "word_spans": batch_word_spans,
            "extracted_words": batch_extracted_words,
            "input_byte_ids": input_bytes,
            "target_byte_ids": target_bytes,
            "target_lengths": target_lengths,
            "raw_texts": [b["text"] for b in batch],
            "languages": [b["lang"] for b in batch],
        }


def load_multilingual_sentences(
    clean_manifest: str = "data/clean_multilingual_train.json",
    ls_train: str = "data/librispeech/librispeech_train_460h.json",
    mls_fr_txt: str = "/media/hdd/Datasets/mls/mls_french/train/transcripts.txt",
    mls_it_txt: str = "/media/hdd/Datasets/mls/mls_italian/train/transcripts.txt",
    mls_es_txt: str = "/media/hdd/Datasets/mls/mls_spanish/train/transcripts.txt",
    max_per_lang: int = 75000,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    train_items: List[Tuple[str, str]] = []
    val_items: List[Tuple[str, str]] = []

    # 1. English
    if os.path.isfile(ls_train):
        with open(ls_train, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts = [d.get("transcript", d.get("text", "")) for d in data if len(d.get("transcript", "").split()) >= 2]
        random.shuffle(texts)
        train_items.extend([(t, "en") for t in texts[:max_per_lang]])
        val_items.extend([(t, "en") for t in texts[max_per_lang : max_per_lang + 500]])

    # 2. French
    if os.path.isfile(mls_fr_txt):
        with open(mls_fr_txt, "r", encoding="utf-8") as f:
            texts = [l.strip().split("\t")[-1] for l in f if len(l.strip().split("\t")[-1].split()) >= 2]
        random.shuffle(texts)
        train_items.extend([(t, "fr") for t in texts[:max_per_lang]])
        val_items.extend([(t, "fr") for t in texts[max_per_lang : max_per_lang + 500]])

    # 3. Italian
    if os.path.isfile(mls_it_txt):
        with open(mls_it_txt, "r", encoding="utf-8") as f:
            texts = [l.strip().split("\t")[-1] for l in f if len(l.strip().split("\t")[-1].split()) >= 2]
        random.shuffle(texts)
        train_items.extend([(t, "it") for t in texts[:max_per_lang]])
        val_items.extend([(t, "it") for t in texts[max_per_lang : max_per_lang + 500]])

    # 4. Spanish
    if os.path.isfile(mls_es_txt):
        with open(mls_es_txt, "r", encoding="utf-8") as f:
            texts = [l.strip().split("\t")[-1] for l in f if len(l.strip().split("\t")[-1].split()) >= 2]
        random.shuffle(texts)
        train_items.extend([(t, "es") for t in texts[:max_per_lang]])
        val_items.extend([(t, "es") for t in texts[max_per_lang : max_per_lang + 500]])

    random.shuffle(train_items)
    return train_items, val_items


def main():
    parser = argparse.ArgumentParser(description="Train Phono-V8 Space-Free Continuous Segmenter")
    parser.add_argument("--clean_manifest", type=str, default="data/clean_multilingual_train.json")
    parser.add_argument("--warm_start_p2g_ckpt", type=str, default="checkpoints/phono_v8_aligned_soft_p2g/best_robust_p2g.pt")
    parser.add_argument("--output_dir", type=str, default="checkpoints/phono_v8_continuous_p2g")
    parser.add_argument("--max_steps", type=int, default=80000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr_encoder", type=float, default=3.0e-5)
    parser.add_argument("--lr_decoder", type=float, default=5.0e-5)
    parser.add_argument("--min_lr", type=float, default=2.0e-6)
    parser.add_argument("--boundary_loss_weight", type=float, default=2.0)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--sample_every", type=int, default=25, help="Interval for visual sample inspections")
    parser.add_argument("--freeze_decoder", action="store_true", default=True, help="Strictly freeze Model 3 character decoder")
    parser.add_argument("--no_freeze_decoder", dest="freeze_decoder", action="store_false", help="Allow co-fine-tuning of Model 3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("🚀 PHONO-V8: SPACE-FREE CONTINUOUS PHONEME-TO-WORD SEGMENTER (MULTILINGUAL)")
    print("=" * 80)
    print(f"📦 Warm-Start Checkpoint : {args.warm_start_p2g_ckpt}")
    print(f"📁 Output Directory      : {args.output_dir}")
    print(f"⚡ Device                : {device.type.upper()}")
    print(f"🎯 Total Steps          : {args.max_steps} (Effective BS = {args.batch_size * args.grad_accum})")
    print(f"✂️  Boundary Loss Weight  : {args.boundary_loss_weight}")

    rom_tok = RomanCharTokenizer()
    ph_tok = PhonemeTokenizer()
    noise_sim = PhoneticNoiseSimulator(confusion_matrix_path="data/phoneme_confusion_matrix.json")

    print("\n📚 Loading Multilingual Text Corpora...", flush=True)
    train_items, val_items = load_multilingual_sentences(args.clean_manifest, max_per_lang=25000)
    print(f"  • Loaded {len(train_items)} train sentences, {len(val_items)} val sentences across languages")

    train_ds = ContinuousP2GDataset(train_items)
    val_ds = ContinuousP2GDataset(val_items)
    collator = ContinuousP2GCollator(rom_tok, ph_tok, noise_sim)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=1)

    # 1. Model 2: Continuous Segmenter
    cfg = PhonoV75SpeechConfig.medium()
    segmenter_m2 = ContinuousPhonemeWordSegmenter(
        d_macro=cfg.decoder_config.macro_dim,
        d_model=256,
        nhead=4,
        num_layers=2,
        phoneme_vocab_size=128,
    ).to(device)

    # 2. Model 3: MoE Character Decoder
    model_m3 = PhonoV75SpeechModel(cfg).to(device)

    step = 0
    latest_pt = os.path.join(args.output_dir, "latest_continuous_p2g.pt")
    if args.resume and os.path.isfile(latest_pt):
        print(f"\n🔄 Resuming Continuous Segmenter & Model 3 from {latest_pt}...", flush=True)
        ckpt = torch.load(latest_pt, map_location="cpu", weights_only=False)
        segmenter_m2.load_state_dict(ckpt["segmenter_state_dict"])
        model_m3.decoder.load_state_dict(ckpt["decoder_state_dict"])
        step = ckpt.get("step", 0)
        print(f"  ✅ Resumed from Step {step}!")
    elif os.path.isfile(args.warm_start_p2g_ckpt):
        print(f"\n🔄 Warm-starting from: {args.warm_start_p2g_ckpt}...", flush=True)
        loaded = torch.load(args.warm_start_p2g_ckpt, map_location="cpu", weights_only=False)
        m2_state = dict(loaded["phoneme_encoder_state_dict"])
        # If pos_embedding size differs, copy available weights and initialize rest
        if "pos_embedding.weight" in m2_state and m2_state["pos_embedding.weight"].shape != segmenter_m2.pos_embedding.weight.shape:
            old_pe = m2_state["pos_embedding.weight"]
            with torch.no_grad():
                n_copy = min(old_pe.shape[0], segmenter_m2.pos_embedding.weight.shape[0])
                segmenter_m2.pos_embedding.weight[:n_copy].copy_(old_pe[:n_copy])
            del m2_state["pos_embedding.weight"]
        segmenter_m2.load_state_dict(m2_state, strict=False)
        model_m3.decoder.load_state_dict(loaded["decoder_state_dict"])
        print(f"  ✅ Inherited Transformer encoder & MoE decoder weights from step {loaded.get('step', 'N/A')}!")

    if args.freeze_decoder:
        print("  ❄️ Strictly FREEZING Model 3 (MoE Character Decoder)! Gradients flow exclusively to Model 2.", flush=True)
        model_m3.decoder.eval()
        for p in model_m3.decoder.parameters():
            p.requires_grad = False
        trainable_params = [
            {"params": [p for p in segmenter_m2.parameters() if p.requires_grad], "lr": args.lr_encoder},
        ]
    else:
        print("  🔥 Co-fine-tuning Model 3 alongside Model 2.", flush=True)
        trainable_params = [
            {"params": [p for p in segmenter_m2.parameters() if p.requires_grad], "lr": args.lr_encoder},
            {"params": [p for p in model_m3.decoder.parameters() if p.requires_grad], "lr": args.lr_decoder},
        ]

    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=1e-2, betas=(0.9, 0.98))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=400, num_training_steps=args.max_steps)

    step = 0
    accum_step = 0
    best_val_f1 = 0.0
    train_iter = iter(train_loader)

    segmenter_m2.train()
    if args.freeze_decoder:
        model_m3.decoder.eval()
    else:
        model_m3.train()

    print(f"\n🔥 Commencing Space-Free Continuous Multilingual Training ({args.max_steps} steps)...\n", flush=True)

    while step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        cont_ph = batch["continuous_phonemes"].to(device)
        b_targets = batch["boundary_targets"].to(device)
        pad_m = batch["pad_mask"].to(device)
        in_bytes = batch["input_byte_ids"].to(device)
        tgt_bytes = batch["target_byte_ids"].to(device)
        tgt_lens = batch["target_lengths"].to(device)
        spans_batch = batch["word_spans"]

        h, b_logits = segmenter_m2.forward_continuous(phoneme_ids=cont_ph)

        # 1. Boundary Loss (only over non-pad positions, balanced pos_weight=1.0)
        valid_mask = ~pad_m
        loss_boundary = bce_loss_fn(b_logits[valid_mask], b_targets[valid_mask])

        # 2. Model 2 Word Phoneme Encoding (Producing optimal z_word latents for Model 3)
        B = cont_ph.shape[0]
        max_words = in_bytes.shape[1]
        z_words = torch.zeros(B, max_words, segmenter_m2.d_macro, device=device)

        words_to_encode = []
        word_coords = []
        for b_idx in range(B):
            spans = spans_batch[b_idx][:max_words]
            for w_idx, (s_idx, e_idx) in enumerate(spans):
                if e_idx <= cont_ph.shape[1] and e_idx > s_idx:
                    words_to_encode.append(cont_ph[b_idx, s_idx:e_idx])
                    word_coords.append((b_idx, w_idx))

        if words_to_encode:
            max_p = max(len(w) for w in words_to_encode)
            batch_w = torch.zeros(len(words_to_encode), max_p, dtype=torch.long, device=device)
            for i, w in enumerate(words_to_encode):
                batch_w[i, : len(w)] = w
            
            # Model 2 Word Encoder Forward (Phonemes -> 512D Word Latents)
            z_encoded = segmenter_m2(phoneme_ids=batch_w, pad_id=0)  # [N_words, d_macro]
            for i, (b_idx, w_idx) in enumerate(word_coords):
                z_words[b_idx, w_idx] = z_encoded[i]

        # 3. Model 3 Character Decoder Loss
        out_m3 = model_m3.forward_text(
            z_word=z_words,
            input_byte_ids=in_bytes,
            target_byte_ids=tgt_bytes,
            target_lengths=tgt_lens,
        )
        loss_char = out_m3["loss"]
        char_acc_t = out_m3.get("char_acc", out_m3.get("char_accuracy", torch.tensor(0.0)))
        char_acc = char_acc_t.item() if isinstance(char_acc_t, torch.Tensor) else float(char_acc_t)

        total_loss = args.boundary_loss_weight * loss_boundary + loss_char
        loss_scaled = total_loss / args.grad_accum
        loss_scaled.backward()
        accum_step += 1

        if accum_step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(segmenter_m2.parameters(), 1.0)
            if not args.freeze_decoder:
                torch.nn.utils.clip_grad_norm_(model_m3.decoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % args.log_every == 0 or step == 1:
                with torch.no_grad():
                    pred_b = (torch.sigmoid(b_logits[valid_mask]) > 0.5).float()
                    tgt_b = b_targets[valid_mask]
                    space_acc = (pred_b == tgt_b).float().mean().item() * 100.0
                    true_pos = ((pred_b == 1.0) & (tgt_b == 1.0)).sum().item()
                    prec = true_pos / max(1.0, (pred_b == 1.0).sum().item()) * 100.0
                    rec = true_pos / max(1.0, (tgt_b == 1.0).sum().item()) * 100.0
                    bound_f1 = 2 * prec * rec / max(1e-4, prec + rec)

                print(
                    f"Step {step:5d}/{args.max_steps} | Loss: {total_loss.item():.4f} (Bound: {loss_boundary.item():.4f}, Char: {loss_char.item():.4f}) | "
                    f"SpaceAcc: {space_acc:5.1f}% (Rec: {rec:4.1f}%, Prec: {prec:4.1f}%) | Bound F1: {bound_f1:5.1f}% | CharAcc: {char_acc:5.1f}% | LR: {scheduler.get_last_lr()[0]:.2e}"
                )

            # Visual Sample Inspection
            if step % args.sample_every == 0 or step == 1:
                segmenter_m2.eval()
                with torch.no_grad():
                    sample_z, sample_spans = segmenter_m2.segment_and_encode_stream(
                        phoneme_ids=cont_ph[0:1], threshold=0.5
                    )
                    if sample_z.shape[1] > 0:
                        pred_word_tokens = beam_search_character_rollout(
                            model_m3=model_m3,
                            z_words=sample_z,
                            rom_tok=rom_tok,
                            trie=None,
                            beam_width=1,
                            device=device,
                        )
                        pred_words = [rom_tok.decode(w) for w in pred_word_tokens]
                        pred_text = " ".join([w for w in pred_words if w.strip()])
                    else:
                        pred_text = ""
                    gt_text = batch["raw_texts"][0]
                    lang_code = batch["languages"][0]
                    flag_map = {"en": "🇬🇧 EN", "fr": "🇫🇷 FR", "es": "🇪🇸 ES", "it": "🇮🇹 IT"}
                    sample_wer = compute_wer([pred_text], [gt_text]) * 100.0 if pred_text else 100.0
                    sample_cer = compute_cer([pred_text], [gt_text]) * 100.0 if pred_text else 100.0

                    cur_spans = sample_spans[0]
                    sliced_strs = [
                        "".join([ph_tok.id_to_token.get(int(cont_ph[0, idx]), "") for idx in range(s, e)])
                        for s, e in cur_spans
                    ]
                    slices_preview = " ".join([f"/{s}/" for s in sliced_strs[:10]])
                    if len(sliced_strs) > 10:
                        slices_preview += f" ... (+{len(sliced_strs) - 10} words)"

                    gt_words_count = len(batch["word_spans"][0])
                    print("\n" + "═" * 80)
                    print(
                        f"🔍 [Step {step:5d} Sample Inspection ({flag_map.get(lang_code, lang_code.upper())})] | "
                        f"WER: {sample_wer:5.1f}% | CER: {sample_cer:5.1f}% | SpaceAcc: {space_acc:5.1f}%"
                    )
                    print(f"   • GT Text      : {gt_text}")
                    print(f"   • Pred Text    : {pred_text}")
                    print(f"   • Sliced Words : {slices_preview} (Detected: {len(cur_spans)} words | GT: {gt_words_count} words)")
                    print("═" * 80 + "\n")
                segmenter_m2.train()

            # Periodic Validation
            if step % args.eval_every == 0:
                print(f"\n⏳ Running Multilingual Evaluation @ Step {step}...")
                segmenter_m2.eval()
                val_f1s = []
                val_space_accs = []
                val_wers = []
                val_cers = []

                with torch.no_grad():
                    for v_idx, v_batch in enumerate(val_loader):
                        if v_idx >= 25:
                            break
                        v_cont = v_batch["continuous_phonemes"].to(device)
                        v_tgt_b = v_batch["boundary_targets"].to(device)
                        v_pad = v_batch["pad_mask"].to(device)

                        v_h, v_b_logits = segmenter_m2.forward_continuous(phoneme_ids=v_cont)
                        v_valid = ~v_pad
                        v_pred_b = (torch.sigmoid(v_b_logits[v_valid]) > 0.5).float()
                        v_tb = v_tgt_b[v_valid]
                        tp = ((v_pred_b == 1.0) & (v_tb == 1.0)).sum().item()
                        p = tp / max(1.0, (v_pred_b == 1.0).sum().item())
                        r = tp / max(1.0, (v_tb == 1.0).sum().item())
                        f = (2 * p * r / max(1e-4, p + r)) * 100.0
                        val_f1s.append(f)
                        val_space_accs.append((v_pred_b == v_tb).float().mean().item() * 100.0)

                        # End-to-end stream decode on sample 0 of first 5 batches
                        if v_idx < 5:
                            sample_z, sample_spans = segmenter_m2.segment_and_encode_stream(
                                phoneme_ids=v_cont[0:1], threshold=0.5
                            )
                            if sample_z.shape[1] > 0:
                                pred_word_tokens = beam_search_character_rollout(
                                    model_m3=model_m3,
                                    z_words=sample_z,
                                    rom_tok=rom_tok,
                                    trie=None,
                                    beam_width=1,
                                    device=device,
                                )
                                pred_words = [rom_tok.decode(w) for w in pred_word_tokens]
                                pred_text = " ".join([w for w in pred_words if w.strip()])
                            else:
                                pred_text = ""
                            v_gt = v_batch["raw_texts"][0]
                            val_wers.append(compute_wer([pred_text], [v_gt]) * 100.0)
                            val_cers.append(compute_cer([pred_text], [v_gt]) * 100.0)
                            if v_idx == 0:
                                v_lang = v_batch["languages"][0]
                                flag_map = {"en": "🇬🇧 EN", "fr": "🇫🇷 FR", "es": "🇪🇸 ES", "it": "🇮🇹 IT"}
                                print(f"  🔍 [Stream Decode Sample 0 ({flag_map.get(v_lang, v_lang.upper())})]:")
                                print(f"     • GT Text  : {v_gt}")
                                print(f"     • Seg Words: {pred_text}")

                avg_f1 = sum(val_f1s) / max(1, len(val_f1s))
                avg_space_acc = sum(val_space_accs) / max(1, len(val_space_accs))
                avg_wer = sum(val_wers) / max(1, len(val_wers))
                avg_cer = sum(val_cers) / max(1, len(val_cers))
                print(
                    f"  📊 Val Evaluation: Boundary F1: {avg_f1:5.1f}% | SpaceAcc: {avg_space_acc:5.1f}% | "
                    f"Stream WER: {avg_wer:5.1f}% | CER: {avg_cer:5.1f}%\n"
                )

                # Save Checkpoints
                save_data = {
                    "step": step,
                    "segmenter_state_dict": segmenter_m2.state_dict(),
                    "decoder_state_dict": model_m3.decoder.state_dict(),
                    "val_boundary_f1": avg_f1,
                    "args": vars(args),
                }
                torch.save(save_data, latest_pt)
                if avg_f1 > best_val_f1:
                    best_val_f1 = avg_f1
                    best_pt = os.path.join(args.output_dir, "best_continuous_p2g.pt")
                    torch.save(save_data, best_pt)
                    print(f"  💾 Saved New Best Continuous P2G -> {best_pt}\n")

                segmenter_m2.train()
                if args.freeze_decoder:
                    model_m3.decoder.eval()
                else:
                    model_m3.train()

    print(f"\n🎉 Space-Free Continuous Multilingual Training Completed across {args.max_steps} steps!")


if __name__ == "__main__":
    main()
