#!/usr/bin/env python3
"""Build Full Multilingual SOTA Dataset (400,000 samples, 100k per language).

Languages:
- English (EN): 100,000 samples from LibriSpeech train-clean
- French (FR): 100,000 samples from MLS French (with sentence-level liaisons)
- Spanish (ES): 100,000 samples from MLS Spanish
- Italian (IT): 100,000 samples from MLS Italian (100% of 59.6k, balanced-oversampled)

Features:
- Full-sentence phonemization preserving all natural liaisons and elisions.
- Token 8 = '<space>' preserved to provide acoustic macro-spaces for Model 1.
- Validates audio file existence.
- Generates train (400k) and validation (4k) manifests.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from piper.voice import EspeakPhonemizer
from src.data.phoneme_tokenizer import PhonemeTokenizer


def phonemize_sentence(ep: EspeakPhonemizer, ph_tok: PhonemeTokenizer, text: str, voice: str) -> Tuple[List[str], List[int]]:
    try:
        chunks = ep.phonemize(voice, text)
        flat = [ch for sub in chunks for ch in sub]
        # Strip non-acoustic stress and length diacritics
        clean_tokens = [ch for ch in flat if ch not in ("ˈ", "ˌ", "ː", "ˑ")]
        p_ids = ph_tok.encode(clean_tokens)
        return clean_tokens, p_ids
    except Exception:
        return [], []


def load_mls_samples(mls_root: str, partition: str = "train", max_samples: int = 100000, target_oversample: int = 100000) -> List[Dict[str, str]]:
    tsv_path = os.path.join(mls_root, partition, "transcripts.txt")
    if not os.path.isfile(tsv_path):
        print(f"  ❌ Missing {tsv_path}")
        return []

    raw_items = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            sid = parts[0].strip()
            txt = parts[1].strip()
            if len(txt.split()) < 2:
                continue

            id_parts = sid.split("_")
            if len(id_parts) < 2:
                continue
            audio_path = os.path.join(mls_root, partition, "audio", id_parts[0], id_parts[1], f"{sid}.flac")
            raw_items.append({"id": sid, "audio_path": audio_path, "transcript": txt})

    # Shuffle
    random.shuffle(raw_items)

    # Filter to existing files
    valid_items = []
    for item in raw_items:
        if os.path.isfile(item["audio_path"]):
            valid_items.append(item)
            if partition == "train" and len(valid_items) >= max_samples:
                break
            elif partition != "train" and len(valid_items) >= 1000:
                break

    if partition == "train" and len(valid_items) < target_oversample and len(valid_items) > 0:
        print(f"  ℹ️ Oversampling from {len(valid_items)} to {target_oversample} samples...")
        unique_count = len(valid_items)
        oversampled = list(valid_items)
        idx = 0
        while len(oversampled) < target_oversample:
            src = valid_items[idx % unique_count]
            oversampled.append({
                "id": f"{src['id']}_rep{len(oversampled)}",
                "audio_path": src["audio_path"],
                "transcript": src["transcript"],
            })
            idx += 1
        valid_items = oversampled

    return valid_items


def main():
    parser = argparse.ArgumentParser(description="Build Full 400k-Sample Multilingual SOTA Dataset")
    parser.add_argument("--mls_dir", type=str, default="/media/hdd/Datasets/mls")
    parser.add_argument("--librispeech_train", type=str, default="data/librispeech/librispeech_train_460h.json")
    parser.add_argument("--librispeech_test", type=str, default="data/librispeech/librispeech_test_clean.json")
    parser.add_argument("--output_train", type=str, default="data/full_multilingual_train_400k.json")
    parser.add_argument("--output_val", type=str, default="data/full_multilingual_val_4k.json")
    parser.add_argument("--samples_per_lang", type=int, default=100000)
    parser.add_argument("--val_per_lang", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    ep = EspeakPhonemizer()
    ph_tok = PhonemeTokenizer()

    print("=" * 80)
    print("🌍 BUILDING FULL MULTILINGUAL SOTA DATASET (400,000 SAMPLES | 100k PER LANGUAGE)")
    print("=" * 80)
    print(f"📁 MLS Root           : {args.mls_dir}")
    print(f"📁 LibriSpeech Train  : {args.librispeech_train}")
    print(f"🎯 Output Train       : {args.output_train} (400,000 samples)")
    print(f"🎯 Output Val         : {args.output_val} (4,000 samples)")
    print("=" * 80)

    # 1. English
    print("\n[1/4] Loading English (LibriSpeech)...")
    with open(args.librispeech_train, "r", encoding="utf-8") as f:
        en_raw = json.load(f)
    random.shuffle(en_raw)
    en_train_items = [
        {"id": d.get("id", str(i)), "audio_path": d["audio_path"], "transcript": d.get("transcript", d.get("text", "")), "lang": "en"}
        for i, d in enumerate(en_raw)
        if os.path.isfile(d.get("audio_path", "")) and len(d.get("transcript", d.get("text", "")).split()) >= 2
    ][:args.samples_per_lang]
    print(f"  • Selected {len(en_train_items)} English train items")

    with open(args.librispeech_test, "r", encoding="utf-8") as f:
        en_val_raw = json.load(f)
    en_val_items = [
        {"id": d.get("id", str(i)), "audio_path": d["audio_path"], "transcript": d.get("transcript", d.get("text", "")), "lang": "en"}
        for i, d in enumerate(en_val_raw)
        if os.path.isfile(d.get("audio_path", ""))
    ][:args.val_per_lang]
    print(f"  • Selected {len(en_val_items)} English val items")

    # 2. French
    print("\n[2/4] Loading French (MLS French with Liaisons)...")
    fr_train_raw = load_mls_samples(os.path.join(args.mls_dir, "mls_french"), partition="train", max_samples=args.samples_per_lang, target_oversample=args.samples_per_lang)
    for item in fr_train_raw:
        item["lang"] = "fr"
    fr_train_items = fr_train_raw[:args.samples_per_lang]
    print(f"  • Selected {len(fr_train_items)} French train items")

    fr_val_raw = load_mls_samples(os.path.join(args.mls_dir, "mls_french"), partition="test", max_samples=args.val_per_lang, target_oversample=args.val_per_lang)
    for item in fr_val_raw:
        item["lang"] = "fr"
    fr_val_items = fr_val_raw[:args.val_per_lang]
    print(f"  • Selected {len(fr_val_items)} French val items")

    # 3. Spanish
    print("\n[3/4] Loading Spanish (MLS Spanish)...")
    es_train_raw = load_mls_samples(os.path.join(args.mls_dir, "mls_spanish"), partition="train", max_samples=args.samples_per_lang, target_oversample=args.samples_per_lang)
    for item in es_train_raw:
        item["lang"] = "es"
    es_train_items = es_train_raw[:args.samples_per_lang]
    print(f"  • Selected {len(es_train_items)} Spanish train items")

    es_val_raw = load_mls_samples(os.path.join(args.mls_dir, "mls_spanish"), partition="test", max_samples=args.val_per_lang, target_oversample=args.val_per_lang)
    for item in es_val_raw:
        item["lang"] = "es"
    es_val_items = es_val_raw[:args.val_per_lang]
    print(f"  • Selected {len(es_val_items)} Spanish val items")

    # 4. Italian
    print("\n[4/4] Loading Italian (MLS Italian Balanced-Oversampled)...")
    it_train_raw = load_mls_samples(
        os.path.join(args.mls_dir, "mls_italian"),
        partition="train",
        max_samples=args.samples_per_lang,
        target_oversample=args.samples_per_lang,
    )
    for item in it_train_raw:
        item["lang"] = "it"
    it_train_items = it_train_raw[:args.samples_per_lang]
    print(f"  • Selected {len(it_train_items)} Italian train items (1:1:1:1 balanced)")

    it_val_raw = load_mls_samples(os.path.join(args.mls_dir, "mls_italian"), partition="test", max_samples=args.val_per_lang)
    for item in it_val_raw:
        item["lang"] = "it"
    it_val_items = it_val_raw[:args.val_per_lang]
    print(f"  • Selected {len(it_val_items)} Italian val items")

    # Combine and shuffle
    all_train = en_train_items + fr_train_items + es_train_items + it_train_items
    all_val = en_val_items + fr_val_items + es_val_items + it_val_items
    random.shuffle(all_train)
    random.shuffle(all_val)

    print(f"\n⚡ Pre-Phonemizing Full Dataset ({len(all_train)} train + {len(all_val)} val sentences)...")
    lang_voice_map = {"en": "en-us", "fr": "fr", "es": "es", "it": "it"}

    def process_split(items: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
        processed = []
        t0 = time.time()
        for idx, item in enumerate(items):
            lang = item["lang"]
            voice = lang_voice_map.get(lang, "en-us")
            tokens, p_ids = phonemize_sentence(ep, ph_tok, item["transcript"], voice)
            if p_ids and len(p_ids) >= 3:
                processed.append({
                    "id": item["id"],
                    "audio_path": item["audio_path"],
                    "transcript": item["transcript"],
                    "lang": lang,
                    "phonemes": " ".join(tokens),
                    "phoneme_ids": p_ids,
                })
            if (idx + 1) % 50000 == 0 or idx + 1 == len(items):
                elapsed = time.time() - t0
                rate = (idx + 1) / max(1e-3, elapsed)
                print(f"  • [{split_name}] Processed {idx + 1:,} / {len(items):,} ({rate:.1f} sent/s)")
        return processed

    print("  Processing Validation Set...")
    val_processed = process_split(all_val, "VAL")
    print(f"  Saving validation manifest -> {args.output_val} ({len(val_processed)} samples)...")
    with open(args.output_val, "w", encoding="utf-8") as f:
        json.dump(val_processed, f, ensure_ascii=False)

    print("  Processing Training Set...")
    train_processed = process_split(all_train, "TRAIN")
    print(f"  Saving training manifest -> {args.output_train} ({len(train_processed)} samples)...")
    with open(args.output_train, "w", encoding="utf-8") as f:
        json.dump(train_processed, f, ensure_ascii=False)

    # Summary Stats
    lang_counts = {}
    space_counts = 0
    total_tokens = 0
    for p in train_processed:
        l = p["lang"]
        lang_counts[l] = lang_counts.get(l, 0) + 1
        space_counts += p["phoneme_ids"].count(8)
        total_tokens += len(p["phoneme_ids"])

    print("\n" + "=" * 80)
    print("✅ MULTILINGUAL DATASET MANIFEST GENERATION COMPLETE!")
    print("=" * 80)
    print(f"📊 Total Train Samples : {len(train_processed):,}")
    print(f"📊 Total Val Samples   : {len(val_processed):,}")
    print(f"🌐 Language Balance    : {lang_counts}")
    print(f"✂️  Total Acoustic Spaces (Token 8) : {space_counts:,} ({space_counts/max(1, total_tokens)*100:.1f}% of all tokens)")
    print("=" * 80)


if __name__ == "__main__":
    main()
