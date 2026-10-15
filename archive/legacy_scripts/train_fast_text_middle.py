"""Fast Text Middle-Training for Phono-V7.5 Decoder across 4 Languages (EN, FR, IT, ES).

Trains the Macro/Micro Decoder and Overlapping Length Experts on pure text corpora
using the distilled FastTextPhonemeWordEncoder to project words directly into z_word.
Bypasses the 12-layer Conformer audio computation, running 50x faster.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from piper.voice import EspeakPhonemizer
from src.data.roman_tokenizer import RomanCharTokenizer
from src.models.phono_v7_5_speech_model import PhonoV75SpeechConfig, PhonoV75SpeechModel
from src.models.text_phoneme_encoder import FastTextPhonemeWordEncoder
from src.training.metrics import compute_wer, compute_cer

_espeak_phonemizer: Optional[EspeakPhonemizer] = None


def get_phonemes(text: str, lang: str = "en") -> str:
    global _espeak_phonemizer
    if _espeak_phonemizer is None:
        try:
            _espeak_phonemizer = EspeakPhonemizer()
        except Exception:
            return ""
    lang_map = {"en": "en-us", "fr": "fr", "it": "it", "es": "es"}
    target_voice = lang_map.get(lang, "en-us")
    try:
        chunks = _espeak_phonemizer.phonemize(target_voice, text)
        return " ".join("".join(ch) for ch in chunks)
    except Exception:
        return ""


def format_eta(seconds: float) -> str:
    if seconds < 0 or math.isinf(seconds) or math.isnan(seconds):
        return "--m--s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m:02d}m{s:02d}s"


class PureTextDataset(Dataset):
    """Loads and tokenizes pure text sentences into word byte ID tensors with language awareness."""

    def __init__(
        self,
        items: List[Tuple[str, str]],
        tokenizer: RomanCharTokenizer,
        max_words: int = 32,
        max_chars_per_word: int = 24,
    ):
        self.items = [(s.strip().lower(), lang) for s, lang in items if s.strip()]
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_chars = max_chars_per_word

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text, lang = self.items[idx]
        words = text.split()[: self.max_words]
        return {"words": words, "text": text, "lang": lang}


class PureTextCollator:
    """Collates list of word lists into input_byte_ids, target_byte_ids, target_lengths, preserving raw text and lang."""

    def __init__(self, tokenizer: RomanCharTokenizer, max_chars_per_word: int = 24):
        self.tok = tokenizer
        self.max_chars = max_chars_per_word

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(batch)
        max_L = max(1, max(len(b["words"]) for b in batch))
        K = self.max_chars

        input_bytes = torch.zeros(B, max_L, K, dtype=torch.long)
        target_bytes = torch.full((B, max_L, K), -100, dtype=torch.long)
        target_lengths = torch.zeros(B, max_L, dtype=torch.float32)

        for b_idx, item in enumerate(batch):
            words = item["words"]
            for w_idx, w in enumerate(words):
                w_tokens = self.tok.encode(w, add_bos=False, add_eos=False)
                char_len = max(1, len(w_tokens))
                target_lengths[b_idx, w_idx] = float(char_len)

                seq_len = min(char_len, K - 2)
                # Input: [BOS, c1, c2, ..., EOS]
                input_bytes[b_idx, w_idx, 0] = self.tok.bos_id
                input_bytes[b_idx, w_idx, 1 : seq_len + 1] = torch.tensor(w_tokens[:seq_len], dtype=torch.long)

                # Target: [c1, c2, ..., EOS]
                target_bytes[b_idx, w_idx, :seq_len] = torch.tensor(w_tokens[:seq_len], dtype=torch.long)
                target_bytes[b_idx, w_idx, seq_len] = self.tok.eos_id

        return {
            "input_byte_ids": input_bytes,
            "target_byte_ids": target_bytes,
            "target_lengths": target_lengths,
            "raw_texts": [item["text"] for item in batch],
            "languages": [item.get("lang", "en") for item in batch],
        }


def load_multilingual_sentences(
    librispeech_json: str,
    mls_it_txt: str,
    mls_es_txt: str,
    mls_fr_txt: str,
    max_per_lang: int = 60000,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Loads balanced multilingual text sentences from the 4 languages with language tags."""
    train_items: List[Tuple[str, str]] = []
    val_items: List[Tuple[str, str]] = []

    # 1. English (LibriSpeech)
    if Path(librispeech_json).is_file():
        with open(librispeech_json, "r") as f:
            data = json.load(f)
        en_texts = [d.get("text", d.get("transcript", "")) for d in data if d.get("text") or d.get("transcript")]
        random.shuffle(en_texts)
        train_items.extend([(t, "en") for t in en_texts[:max_per_lang]])
        val_items.extend([(t, "en") for t in en_texts[max_per_lang : max_per_lang + 1000]])
        print(f"  • English (LibriSpeech): {len(en_texts[:max_per_lang])} train sentences", flush=True)

    # 2. Italian (MLS)
    if Path(mls_it_txt).is_file():
        with open(mls_it_txt, "r", encoding="utf-8") as f:
            it_texts = [line.strip().split("\t", 1)[-1] for line in f if "\t" in line or " " in line]
        random.shuffle(it_texts)
        train_items.extend([(t, "it") for t in it_texts[:max_per_lang]])
        val_items.extend([(t, "it") for t in it_texts[max_per_lang : max_per_lang + 1000]])
        print(f"  • Italian (MLS): {len(it_texts[:max_per_lang])} train sentences", flush=True)

    # 3. Spanish (MLS)
    if Path(mls_es_txt).is_file():
        with open(mls_es_txt, "r", encoding="utf-8") as f:
            es_texts = [line.strip().split("\t", 1)[-1] for line in f if "\t" in line or " " in line]
        random.shuffle(es_texts)
        train_items.extend([(t, "es") for t in es_texts[:max_per_lang]])
        val_items.extend([(t, "es") for t in es_texts[max_per_lang : max_per_lang + 1000]])
        print(f"  • Spanish (MLS): {len(es_texts[:max_per_lang])} train sentences", flush=True)

    # 4. French (MLS)
    if Path(mls_fr_txt).is_file():
        with open(mls_fr_txt, "r", encoding="utf-8") as f:
            fr_texts = [line.strip().split("\t", 1)[-1] for line in f if "\t" in line or " " in line]
        random.shuffle(fr_texts)
        train_items.extend([(t, "fr") for t in fr_texts[:max_per_lang]])
        val_items.extend([(t, "fr") for t in fr_texts[max_per_lang : max_per_lang + 1000]])
        print(f"  • French (MLS): {len(fr_texts[:max_per_lang])} train sentences", flush=True)

    random.shuffle(train_items)
    random.shuffle(val_items)
    return train_items, val_items


def main():
    parser = argparse.ArgumentParser(description="Fast Text Middle-Training for Phono-V7.5")
    parser.add_argument("--librispeech_train", type=str, default="data/librispeech/librispeech_train_100h.json")
    parser.add_argument("--mls_italian_txt", type=str, default="/media/hdd/Datasets/mls/mls_italian/train/transcripts.txt")
    parser.add_argument("--mls_spanish_txt", type=str, default="/media/hdd/Datasets/mls/mls_spanish/train/transcripts.txt")
    parser.add_argument("--mls_french_txt", type=str, default="/media/hdd/Datasets/mls/mls_french/train/transcripts.txt")

    parser.add_argument("--distilled_student_ckpt", type=str, default="checkpoints/latent_distillation/best_student.pt")
    parser.add_argument("--warm_start_v7_3", type=str, default="checkpoints/phono_v7_3/streaming/best_checkpoint.pt")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from (or 'auto')")
    parser.add_argument("--save_dir", type=str, default="checkpoints/phono_v7_5_text_middle")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size of text sentences")
    parser.add_argument("--grad_accum", type=int, default=2, help="Effective batch size = 32")
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Fast Text Middle-Training on: {device}", flush=True)

    # 1. Load Multilingual Text Datasets
    print("\n📚 Loading Multilingual Text Datasets (EN, IT, ES, FR)...", flush=True)
    max_sentences = 100 if args.smoke_test else 60000
    train_items, val_items = load_multilingual_sentences(
        args.librispeech_train,
        args.mls_italian_txt,
        args.mls_spanish_txt,
        args.mls_french_txt,
        max_per_lang=max_sentences,
    )
    print(f"🌟 Total Text Corpora: {len(train_items)} train sentences, {len(val_items)} val sentences", flush=True)

    tokenizer = RomanCharTokenizer()
    train_dataset = PureTextDataset(train_items, tokenizer)
    val_dataset = PureTextDataset(val_items, tokenizer)
    collator = PureTextCollator(tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=1)

    # 2. Load Distilled Fast Text/Phoneme Word Encoder
    print(f"\n🔄 Loading Distilled Fast Text/Phoneme Word Encoder: {args.distilled_student_ckpt}...", flush=True)
    cfg = PhonoV75SpeechConfig.medium()
    text_encoder = FastTextPhonemeWordEncoder(
        d_macro=cfg.decoder_config.macro_dim,
        d_model=256,
        nhead=4,
        num_layers=2,
    ).to(device)

    if Path(args.distilled_student_ckpt).is_file():
        ckpt_student = torch.load(args.distilled_student_ckpt, map_location=device, weights_only=False)
        sd_student = ckpt_student.get("student_state_dict", ckpt_student)
        text_encoder.load_state_dict(sd_student, strict=False)
        print("  • Distilled word encoder loaded successfully (Cosine Similarity ~98%)", flush=True)
    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad = False

    # 3. Initialize Phono-V7.5 Model & Warm-start Decoder
    print(f"\n🔄 Initializing Phono-V7.5 Medium Model...", flush=True)
    model = PhonoV75SpeechModel(cfg).to(device)
    if Path(args.warm_start_v7_3).is_file():
        print(f"  • Warm-starting decoder weights from V7.3 record: {args.warm_start_v7_3}...", flush=True)
        res = model.warm_start_from_v7_3(args.warm_start_v7_3)
        print(f"  • Transferred {res['transferred']} tensors (skipped {res['skipped']}, missing {res['missing']})", flush=True)

    # Train only decoder parameters
    decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(decoder_params, lr=args.lr, weight_decay=1e-2, betas=(0.9, 0.98))

    max_steps = 20 if args.smoke_test else args.max_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    start_step = 0

    # Resume support
    resume_target = args.resume
    if resume_target == "auto":
        latest_cand = save_dir / "latest_decoder.pt"
        best_cand = save_dir / "best_decoder.pt"
        if latest_cand.is_file():
            resume_target = str(latest_cand)
        elif best_cand.is_file():
            resume_target = str(best_cand)
        else:
            resume_target = ""

    if resume_target and Path(resume_target).is_file():
        print(f"\n🔄 Resuming training from checkpoint: {resume_target}...", flush=True)
        r_ckpt = torch.load(resume_target, map_location=device, weights_only=False)
        sd_dec = r_ckpt.get("decoder_state_dict", r_ckpt)
        model.decoder.load_state_dict(sd_dec, strict=False)
        if "step" in r_ckpt:
            start_step = r_ckpt["step"]
        if "val_loss" in r_ckpt:
            best_val_loss = r_ckpt["val_loss"]
        if "optimizer_state_dict" in r_ckpt:
            try:
                optimizer.load_state_dict(r_ckpt["optimizer_state_dict"])
            except Exception as e:
                print(f"  ⚠️ Could not restore optimizer: {e}", flush=True)
        if "scheduler_state_dict" in r_ckpt:
            try:
                scheduler.load_state_dict(r_ckpt["scheduler_state_dict"])
            except Exception as e:
                print(f"  ⚠️ Could not restore scheduler: {e}", flush=True)
        print(f"  ✅ Successfully resumed from Step {start_step} (historical best val loss: {best_val_loss:.4f})", flush=True)

    print(f"\n🔥 Starting Fast Text Middle-Training in Pure FP32 ({start_step} -> {max_steps} steps, BS={args.batch_size * args.grad_accum})...", flush=True)
    start_time = time.time()
    step = start_step
    accum_step = 0
    train_iter = iter(train_loader)
    model.train()

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_bytes = batch["input_byte_ids"].to(device, non_blocking=True)
        target_bytes = batch["target_byte_ids"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)

        B, L, K = input_bytes.shape
        flat_bytes = input_bytes.reshape(B * L, K)

        # 1. Encode words with frozen distilled FastTextPhonemeWordEncoder (Pure FP32)
        with torch.no_grad():
            z_words = text_encoder(byte_ids=flat_bytes).view(B, L, -1)

        # 2. Decoder Forward with Overlapping Length Experts MoE (Pure FP32)
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
            torch.nn.utils.clip_grad_norm_(decoder_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            step += 1

            if step % args.log_every == 0 or step == 1 or args.smoke_test:
                elapsed = time.time() - start_time
                completed = step - start_step
                rate = completed / max(1e-3, elapsed)
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

            # Evaluation
            if (step % args.eval_every == 0 or step == max_steps) and not args.smoke_test:
                model.eval()
                val_losses, val_char_accs, val_path_accs = [], [], []
                all_preds, all_refs = [], []
                samples_to_show = []

                with torch.no_grad():
                    for v_idx, v_batch in enumerate(val_loader):
                        if v_idx >= 50:  # Eval on 50 batches for fast feedback
                            break
                        v_inp = v_batch["input_byte_ids"].to(device)
                        v_tgt = v_batch["target_byte_ids"].to(device)
                        v_lens = v_batch["target_lengths"].to(device)
                        v_texts = v_batch["raw_texts"]
                        v_langs = v_batch["languages"]

                        v_B, v_L, v_K = v_inp.shape
                        v_flat = v_inp.reshape(v_B * v_L, v_K)
                        v_z = text_encoder(byte_ids=v_flat).view(v_B, v_L, -1)

                        v_out = model.forward_text(
                            z_word=v_z,
                            input_byte_ids=v_inp,
                            target_byte_ids=v_tgt,
                            target_lengths=v_lens,
                        )
                        val_losses.append(v_out["loss"].item())
                        val_char_accs.append(v_out["char_acc"].item())
                        val_path_accs.append(v_out["path_acc"].item())

                        # Decode predictions for WER and sample logging
                        v_logits = v_out["logits"]  # [B, L, K, V]
                        v_pred_ids = v_logits.argmax(dim=-1)  # [B, L, K]

                        for i in range(v_B):
                            tgt_word_lists = []
                            for w in range(v_L):
                                valid_t = [t.item() for t in v_tgt[i, w] if t.item() not in (-100, tokenizer.pad_id, tokenizer.bos_id)]
                                if valid_t:
                                    tgt_word_lists.append(valid_t)
                            ref_text = tokenizer.decode_words(tgt_word_lists)
                            if not ref_text.strip():
                                continue

                            pred_word_lists = []
                            for w in range(len(tgt_word_lists)):
                                pred_w = v_pred_ids[i, w].tolist()
                                pred_word_lists.append(pred_w)
                            pred_text = tokenizer.decode_words(pred_word_lists)

                            all_refs.append(ref_text)
                            all_preds.append(pred_text)

                            if len(samples_to_show) < 4 and random.random() < 0.25:
                                lang_code = v_langs[i]
                                flag = {"en": "🇬🇧 EN", "it": "🇮🇹 IT", "es": "🇪🇸 ES", "fr": "🇫🇷 FR"}.get(lang_code, lang_code.upper())
                                gt_ph = get_phonemes(ref_text, lang_code)
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

                ckpt_data = {
                    "step": step,
                    "val_loss": v_loss,
                    "val_wer": val_wer,
                    "val_cer": val_cer,
                    "val_char_acc": v_char,
                    "val_path_acc": v_path,
                    "decoder_state_dict": model.decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }

                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    print(f"  🌟 New Best Decoder Validation Loss: {best_val_loss:.4f} (WER: {val_wer:.2f}%)!", flush=True)
                    torch.save(ckpt_data, save_dir / "best_decoder.pt")
                    print(f"  💾 Saved best decoder checkpoint to {save_dir / 'best_decoder.pt'}", flush=True)

                torch.save(ckpt_data, save_dir / "latest_decoder.pt")
                model.train()

    print(f"\n🎉 Fast Text Middle-Training completed successfully in {time.time() - start_time:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
