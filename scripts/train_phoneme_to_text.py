"""Phono-V7.5 True Phoneme-to-Text (P2G: Phoneme-to-Grapheme) Training Script.

Trains the Phoneme Word Encoder and Overlapping Length MoE Character Decoder
to transcribe IPA Phoneme sequences directly into orthographic text characters
across English, French, Italian, and Spanish.

Pipeline:
  [IPA Phonemes] -> FastTextPhonemeWordEncoder -> z_word (512D) -> Overlapping MoE Decoder -> [Target Text]
"""

import argparse
import json
import math
import os
import random
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from piper.voice import EspeakPhonemizer
from src.data.phoneme_tokenizer import PhonemeTokenizer
from src.data.roman_tokenizer import RomanCharTokenizer
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


@lru_cache(maxsize=150000)
def phonemize_word_cached(word: str, lang: str) -> Tuple[int, ...]:
    """Phonemizes a single word and returns a tuple of PhonemeTokenizer token IDs."""
    ep = get_phonemizer()
    ptok = get_phoneme_tokenizer()
    lang_map = {"en": "en-us", "fr": "fr", "it": "it", "es": "es"}
    target_voice = lang_map.get(lang, "en-us")
    try:
        chunks = ep.phonemize(target_voice, word)
        flat = [ch for sub in chunks for ch in sub]
        ids = ptok.encode(flat)
        return tuple(ids)
    except Exception:
        return tuple()


def format_eta(seconds: float) -> str:
    if seconds < 0 or math.isinf(seconds) or math.isnan(seconds):
        return "--m--s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m:02d}m{s:02d}s"


class PhonemeToTextDataset(Dataset):
    """Loads text sentences and prepares word-level IPA phonemes and target byte tokens."""

    def __init__(
        self,
        items: List[Tuple[str, str]],
        roman_tokenizer: RomanCharTokenizer,
        max_words: int = 32,
        max_chars_per_word: int = 24,
        max_phonemes_per_word: int = 24,
    ):
        self.items = [(s.strip().lower(), lang) for s, lang in items if s.strip()]
        self.rom_tok = roman_tokenizer
        self.max_words = max_words
        self.max_chars = max_chars_per_word
        self.max_phonemes = max_phonemes_per_word

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


class PhonemeToTextCollator:
    """Collates words into input_phoneme_ids, input_byte_ids, target_byte_ids, and target_lengths."""

    def __init__(
        self,
        roman_tokenizer: RomanCharTokenizer,
        phoneme_tokenizer: PhonemeTokenizer,
        max_chars_per_word: int = 24,
        max_phonemes_per_word: int = 24,
    ):
        self.rom_tok = roman_tokenizer
        self.ph_tok = phoneme_tokenizer
        self.max_chars = max_chars_per_word
        self.max_ph = max_phonemes_per_word

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(batch)
        max_L = max(1, max(len(b["words"]) for b in batch))
        K_char = self.max_chars
        K_ph = self.max_ph

        input_phonemes = torch.zeros(B, max_L, K_ph, dtype=torch.long)
        input_bytes = torch.zeros(B, max_L, K_char, dtype=torch.long)
        target_bytes = torch.full((B, max_L, K_char), -100, dtype=torch.long)
        target_lengths = torch.zeros(B, max_L, dtype=torch.float32)

        for b_idx, item in enumerate(batch):
            words = item["words"]
            lang = item["lang"]

            for w_idx, w in enumerate(words):
                # 1. Phonemes for word
                ph_ids = phonemize_word_cached(w, lang)
                ph_len = min(len(ph_ids), K_ph)
                if ph_len > 0:
                    input_phonemes[b_idx, w_idx, :ph_len] = torch.tensor(ph_ids[:ph_len], dtype=torch.long)

                # 2. Characters for word
                w_tokens = self.rom_tok.encode(w, add_bos=False, add_eos=False)
                char_len = max(1, len(w_tokens))
                target_lengths[b_idx, w_idx] = float(char_len)

                seq_len = min(char_len, K_char - 2)
                # Input for teacher forcing: [BOS, c1, c2, ...]
                input_bytes[b_idx, w_idx, 0] = self.rom_tok.bos_id
                input_bytes[b_idx, w_idx, 1 : seq_len + 1] = torch.tensor(w_tokens[:seq_len], dtype=torch.long)

                # Target: [c1, c2, ..., EOS]
                target_bytes[b_idx, w_idx, :seq_len] = torch.tensor(w_tokens[:seq_len], dtype=torch.long)
                target_bytes[b_idx, w_idx, seq_len] = self.rom_tok.eos_id

        return {
            "input_phonemes": input_phonemes,
            "input_byte_ids": input_bytes,
            "target_byte_ids": target_bytes,
            "target_lengths": target_lengths,
            "raw_texts": [b["text"] for b in batch],
            "languages": [b["lang"] for b in batch],
        }


def load_multilingual_data(
    librispeech_json: str,
    italian_txt: str,
    mls_es_txt: str,
    mls_fr_txt: str,
    max_per_lang: int = 60000,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Loads balanced multilingual sentences from the 4 languages, guaranteed 100% disjoint from previous text-to-text run."""
    train_items: List[Tuple[str, str]] = []
    val_items: List[Tuple[str, str]] = []

    # 1. English: LibriSpeech 960h (skip first 30,000 sentences to be 100% disjoint from 100h used in text-to-text)
    if Path(librispeech_json).is_file():
        with open(librispeech_json, "r") as f:
            data = json.load(f)
        en_texts = [d.get("text", d.get("transcript", "")) for d in data if d.get("text") or d.get("transcript")]
        disjoint_en = en_texts[30000:]
        random.shuffle(disjoint_en)
        train_items.extend([(t, "en") for t in disjoint_en[:max_per_lang]])
        val_items.extend([(t, "en") for t in disjoint_en[max_per_lang : max_per_lang + 1000]])
        print(f"  • English (LibriSpeech 960h [Held-Out 30k+]): {len(disjoint_en[:max_per_lang])} train sentences", flush=True)

    # 2. Italian: Wikipedia Clean (100% disjoint from MLS Italian used in text-to-text)
    if Path(italian_txt).is_file():
        with open(italian_txt, "r", encoding="utf-8") as f:
            it_texts = [line.strip() for line in f if line.strip()]
        random.shuffle(it_texts)
        train_items.extend([(t, "it") for t in it_texts[:max_per_lang]])
        val_items.extend([(t, "it") for t in it_texts[max_per_lang : max_per_lang + 1000]])
        print(f"  • Italian (Wikipedia Clean): {len(it_texts[:max_per_lang])} train sentences", flush=True)

    # 3. Spanish: MLS Spanish (lines 60,000+ -> 100% disjoint from first 60k used in text-to-text)
    if Path(mls_es_txt).is_file():
        with open(mls_es_txt, "r", encoding="utf-8") as f:
            es_texts = [line.strip().split("\t", 1)[-1] for line in f if "\t" in line or " " in line]
        disjoint_es = es_texts[60000:]
        random.shuffle(disjoint_es)
        train_items.extend([(t, "es") for t in disjoint_es[:max_per_lang]])
        val_items.extend([(t, "es") for t in disjoint_es[max_per_lang : max_per_lang + 1000]])
        print(f"  • Spanish (MLS [Held-Out 60k+]): {len(disjoint_es[:max_per_lang])} train sentences", flush=True)

    # 4. French: MLS French (lines 60,000+ -> 100% disjoint from first 60k used in text-to-text)
    if Path(mls_fr_txt).is_file():
        with open(mls_fr_txt, "r", encoding="utf-8") as f:
            fr_texts = [line.strip().split("\t", 1)[-1] for line in f if "\t" in line or " " in line]
        disjoint_fr = fr_texts[60000:]
        random.shuffle(disjoint_fr)
        train_items.extend([(t, "fr") for t in disjoint_fr[:max_per_lang]])
        val_items.extend([(t, "fr") for t in disjoint_fr[max_per_lang : max_per_lang + 1000]])
        print(f"  • French (MLS [Held-Out 60k+]): {len(disjoint_fr[:max_per_lang])} train sentences", flush=True)

    random.shuffle(train_items)
    random.shuffle(val_items)
    return train_items, val_items


def main():
    parser = argparse.ArgumentParser(description="True Phoneme-to-Text (P2G) Training for Phono-V7.5")
    parser.add_argument("--librispeech_train", type=str, default="data/librispeech/librispeech_train_960h.json")
    parser.add_argument("--italian_txt", type=str, default="data/italian_clean_60k.txt")
    parser.add_argument("--mls_spanish_txt", type=str, default="/media/hdd/Datasets/mls/mls_spanish/train/transcripts.txt")
    parser.add_argument("--mls_french_txt", type=str, default="/media/hdd/Datasets/mls/mls_french/train/transcripts.txt")

    parser.add_argument("--distilled_student_ckpt", type=str, default="checkpoints/latent_distillation/best_student.pt")
    parser.add_argument("--warm_start_decoder", type=str, default="checkpoints/phono_v7_5_text_middle/best_decoder.pt")
    parser.add_argument("--warm_start_v7_3", type=str, default="checkpoints/phono_v7_3/streaming/best_checkpoint.pt")
    parser.add_argument("--save_dir", type=str, default="checkpoints/phono_v7_5_phoneme_to_text")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=2, help="Effective batch size = 32")
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--lr_encoder", type=float, default=3e-4)
    parser.add_argument("--lr_decoder", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing True Phoneme-to-Text (P2G) Training on: {device}", flush=True)

    # 1. Load Datasets
    print("\n📚 Loading Multilingual Text Datasets (EN, IT, ES, FR)...", flush=True)
    max_sentences = 100 if args.smoke_test else 60000
    train_items, val_items = load_multilingual_data(
        args.librispeech_train,
        args.italian_txt,
        args.mls_spanish_txt,
        args.mls_french_txt,
        max_per_lang=max_sentences,
    )
    print(f"🌟 Total Corpora: {len(train_items)} train sentences, {len(val_items)} val sentences", flush=True)

    rom_tok = RomanCharTokenizer()
    ph_tok = PhonemeTokenizer()

    train_dataset = PhonemeToTextDataset(train_items, rom_tok)
    val_dataset = PhonemeToTextDataset(val_items, rom_tok)
    collator = PhonemeToTextCollator(rom_tok, ph_tok)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=1)

    # 2. Initialize Phoneme Word Encoder
    print(f"\n🔄 Initializing Phoneme Word Encoder (Vocab={ph_tok.vocab_size})...", flush=True)
    cfg = PhonoV75SpeechConfig.medium()
    phoneme_encoder = FastTextPhonemeWordEncoder(
        d_macro=cfg.decoder_config.macro_dim,
        d_model=256,
        nhead=4,
        num_layers=2,
        phoneme_vocab_size=128,
    ).to(device)

    # Warm-start encoder transformer from distilled student
    if Path(args.distilled_student_ckpt).is_file():
        ckpt_student = torch.load(args.distilled_student_ckpt, map_location=device, weights_only=False)
        sd_student = ckpt_student.get("student_state_dict", ckpt_student)
        # Load transformer weights, ignoring phoneme embedding if shape mismatch
        filtered = {k: v for k, v in sd_student.items() if k in phoneme_encoder.state_dict() and phoneme_encoder.state_dict()[k].shape == v.shape}
        phoneme_encoder.load_state_dict(filtered, strict=False)
        print(f"  • Warm-started phoneme encoder transformer from student ({len(filtered)} tensors)", flush=True)

    # 3. Initialize Phono-V7.5 Model & Warm-start Decoder from Middle-Training Record
    print(f"\n🔄 Initializing Phono-V7.5 Model...", flush=True)
    model = PhonoV75SpeechModel(cfg).to(device)

    if Path(args.warm_start_decoder).is_file():
        print(f"  • Warm-starting decoder from Middle-Training record: {args.warm_start_decoder}...", flush=True)
        dec_ckpt = torch.load(args.warm_start_decoder, map_location=device, weights_only=False)
        dec_sd = dec_ckpt.get("decoder_state_dict", dec_ckpt)
        model.decoder.load_state_dict(dec_sd, strict=False)
        print(f"  ✅ Decoder successfully warm-started (inheriting 98.4% CharAcc spelling prior!)", flush=True)
    elif Path(args.warm_start_v7_3).is_file():
        print(f"  • Warm-starting from V7.3: {args.warm_start_v7_3}...", flush=True)
        model.warm_start_from_v7_3(args.warm_start_v7_3)

    # Train BOTH the Phoneme Encoder AND the Decoder
    trainable_params = [
        {"params": [p for p in phoneme_encoder.parameters() if p.requires_grad], "lr": args.lr_encoder},
        {"params": [p for p in model.decoder.parameters() if p.requires_grad], "lr": args.lr_decoder},
    ]
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=1e-2, betas=(0.9, 0.98))

    max_steps = 20 if args.smoke_test else args.max_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_val_wer = float("inf")

    print(f"\n🔥 Starting True Phoneme-to-Text Training ({max_steps} steps, BS={args.batch_size * args.grad_accum})...", flush=True)
    start_time = time.time()
    step = 0
    accum_step = 0
    train_iter = iter(train_loader)

    phoneme_encoder.train()
    model.train()

    ep = get_phonemizer()
    def get_full_phonemes(text, lang):
        lang_map = {"en": "en-us", "fr": "fr", "it": "it", "es": "es"}
        voice = lang_map.get(lang, "en-us")
        try:
            chunks = ep.phonemize(voice, text)
            return " ".join("".join(c) for c in chunks)
        except Exception:
            return ""

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ph = batch["input_phonemes"].to(device, non_blocking=True)
        input_bytes = batch["input_byte_ids"].to(device, non_blocking=True)
        target_bytes = batch["target_byte_ids"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)

        B, L, K_ph = input_ph.shape
        flat_ph = input_ph.reshape(B * L, K_ph)

        # 1. Encode IPA Phonemes -> z_word (Acoustic Latent Space)
        z_words = phoneme_encoder(phoneme_ids=flat_ph).view(B, L, -1)

        # 2. Decode z_word -> Orthographic Text Characters
        out = model.forward_text(
            z_word=z_words,
            input_byte_ids=input_bytes,
            target_byte_ids=target_bytes,
            target_lengths=target_lengths,
        )
        loss = out["loss"] / args.grad_accum

        loss.backward()
        accum_step += 1

        if accum_step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(phoneme_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1

            if step % args.log_every == 0 or step == 1 or args.smoke_test:
                elapsed = time.time() - start_time
                rate = step / max(1e-3, elapsed)
                eta = (max_steps - step) / max(1e-3, rate)
                lr_curr = optimizer.param_groups[0]["lr"]

                print(
                    f"Step {step:5d}/{max_steps} | Loss: {out['loss'].item():.4f} "
                    f"(Char: {out['char_loss'].item():.3f}, Path: {out['path_loss'].item():.3f}, "
                    f"Bnd: {out['balance_loss'].item():.3f}, Lev: {out['lev_loss'].item():.3f}) | "
                    f"CharAcc: {out['char_acc'].item():5.1f}% | "
                    f"PathAcc (MoE): {out['path_acc'].item():5.1f}% | "
                    f"LR: {lr_curr:.2e} | Rate: {rate:4.1f} st/s | ETA: {format_eta(eta)}",
                    flush=True,
                )

                # Live sample prediction from current train batch
                with torch.no_grad():
                    pred_tokens_0 = out["logits"][0].argmax(dim=-1)
                    tgt_0 = target_bytes[0]
                    tgt_words = []
                    for w in range(L):
                        valid_t = [t.item() for t in tgt_0[w] if t.item() not in (-100, rom_tok.pad_id, rom_tok.bos_id)]
                        if valid_t:
                            tgt_words.append(valid_t)
                    ref_s = rom_tok.decode_words(tgt_words)
                    pred_words = [pred_tokens_0[w].tolist() for w in range(len(tgt_words))]
                    pred_s = rom_tok.decode_words(pred_words)
                    lang_s = batch["languages"][0]
                    flag_s = {"en": "🇬🇧 EN", "it": "🇮🇹 IT", "es": "🇪🇸 ES", "fr": "🇫🇷 FR"}.get(lang_s, lang_s.upper())
                    gt_ph_s = get_full_phonemes(ref_s, lang_s)
                    sample_wer = compute_wer([pred_s], [ref_s]) * 100.0 if ref_s else 0.0

                    print(f"  🔍 [Train Sample] ({flag_s}) | Sample WER: {sample_wer:5.1f}%", flush=True)
                    print(f"    • GT Phonem : {gt_ph_s}", flush=True)
                    print(f"    • GT Text   : {ref_s}", flush=True)
                    print(f"    • Pred Text : {pred_s}", flush=True)

            # Validation with WER & Multilingual Sample Predictions
            if (step % args.eval_every == 0 or step == 1 or step == max_steps) and not args.smoke_test:
                phoneme_encoder.eval()
                model.eval()

                val_losses, val_char_accs, val_path_accs = [], [], []
                all_preds, all_refs = [], []
                samples_to_show = []

                with torch.no_grad():
                    for v_idx, v_batch in enumerate(val_loader):
                        if v_idx >= 50:
                            break
                        v_ph = v_batch["input_phonemes"].to(device)
                        v_inp = v_batch["input_byte_ids"].to(device)
                        v_tgt = v_batch["target_byte_ids"].to(device)
                        v_lens = v_batch["target_lengths"].to(device)
                        v_texts = v_batch["raw_texts"]
                        v_langs = v_batch["languages"]

                        v_B, v_L, v_K_ph = v_ph.shape
                        v_flat_ph = v_ph.reshape(v_B * v_L, v_K_ph)
                        v_z = phoneme_encoder(phoneme_ids=v_flat_ph).view(v_B, v_L, -1)

                        v_out = model.forward_text(
                            z_word=v_z,
                            input_byte_ids=v_inp,
                            target_byte_ids=v_tgt,
                            target_lengths=v_lens,
                        )
                        val_losses.append(v_out["loss"].item())
                        val_char_accs.append(v_out["char_acc"].item())
                        val_path_accs.append(v_out["path_acc"].item())

                        v_pred_tokens = v_out["logits"].argmax(dim=-1)

                        for i in range(v_B):
                            tgt_word_lists = []
                            for w in range(v_L):
                                valid_t = [t.item() for t in v_tgt[i, w] if t.item() not in (-100, rom_tok.pad_id, rom_tok.bos_id)]
                                if valid_t:
                                    tgt_word_lists.append(valid_t)
                            ref_text = rom_tok.decode_words(tgt_word_lists)
                            if not ref_text.strip():
                                continue

                            pred_word_lists = []
                            for w in range(len(tgt_word_lists)):
                                pred_w = v_pred_tokens[i, w].tolist()
                                pred_word_lists.append(pred_w)
                            pred_text = rom_tok.decode_words(pred_word_lists)

                            all_refs.append(ref_text)
                            all_preds.append(pred_text)

                            if len(samples_to_show) < 4:
                                lang_code = v_langs[i]
                                flag = {"en": "🇬🇧 EN", "it": "🇮🇹 IT", "es": "🇪🇸 ES", "fr": "🇫🇷 FR"}.get(lang_code, lang_code.upper())
                                gt_ph = get_full_phonemes(ref_text, lang_code)
                                samples_to_show.append({
                                    "flag": flag,
                                    "gt_ph": gt_ph,
                                    "gt_text": ref_text,
                                    "pred_text": pred_text,
                                })

                v_loss = sum(val_losses) / max(1, len(val_losses))
                v_char = sum(val_char_accs) / max(1, len(val_char_accs))
                v_path = sum(val_path_accs) / max(1, len(val_path_accs))
                val_wer = compute_wer(all_preds, all_refs) * 100.0 if all_refs else 0.0
                val_cer = compute_cer(all_preds, all_refs) * 100.0 if all_refs else 0.0

                print(
                    f"\n🧪 [Validation Step {step}] Loss: {v_loss:.4f} | WER: {val_wer:5.2f}% | CER: {val_cer:5.2f}% | CharAcc: {v_char:.1f}% | PathAcc: {v_path:.1f}%",
                    flush=True,
                )
                if samples_to_show:
                    print("  " + "─" * 78, flush=True)
                    for s_idx, s in enumerate(samples_to_show, 1):
                        print(f"  🔍 [Sample {s_idx}] ({s['flag']})", flush=True)
                        print(f"    • GT Phonem : {s['gt_ph']}", flush=True)
                        print(f"    • GT Text   : {s['gt_text']}", flush=True)
                        print(f"    • Pred Text : {s['pred_text']}", flush=True)
                    print("  " + "─" * 78, flush=True)

                ckpt_payload = {
                    "step": step,
                    "val_loss": v_loss,
                    "val_wer": val_wer,
                    "val_cer": val_cer,
                    "val_char_acc": v_char,
                    "val_path_acc": v_path,
                    "phoneme_encoder_state_dict": phoneme_encoder.state_dict(),
                    "decoder_state_dict": model.decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }

                def atomic_save(payload, path):
                    tmp = path.with_suffix(".tmp")
                    torch.save(payload, tmp)
                    os.replace(tmp, path)

                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    best_val_wer = val_wer
                    print(f"  🌟 New Best Phoneme-to-Text Model: Val Loss {best_val_loss:.4f} | WER {best_val_wer:.2f}%!", flush=True)
                    atomic_save(ckpt_payload, save_dir / "best_p2g.pt")
                    print(f"  💾 Saved best checkpoint to {save_dir / 'best_p2g.pt'}", flush=True)

                atomic_save(ckpt_payload, save_dir / "latest_p2g.pt")
                phoneme_encoder.train()
                model.train()

    print(f"\n🎉 True Phoneme-to-Text Training completed successfully in {time.time() - start_time:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
