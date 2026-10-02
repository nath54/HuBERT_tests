#!/usr/bin/env python3
"""High-Throughput Multilingual Romanization, Phonemization, and Binary Sharding Pipeline.

Processes raw text corpus from /media/hdd/Datasets/multilingual_text/raw/ across 9 languages:
- English (en), French (fr), Spanish (es), German (de), Italian (it)
- Arabic (ar) -> Latin Transliteration
- Chinese (zh) -> Pinyin (Tone3 numbers)
- Japanese (ja) -> Hepburn Romaji
- Korean (ko) -> Revised Romanization

Outputs balanced, zero-copy memory-mappable binary shards (.bin + .idx + .json)
to /media/hdd/Datasets/multilingual_text/shards/.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import glob
import gzip
import json
import os
import random
import re
import struct
import sys
import time
from typing import Dict, Iterator, List, Optional, Tuple

import pypinyin
from pykakasi import kakasi
import pyarabic.trans as ar_trans
from korean_romanizer.romanizer import Romanizer
from piper.voice import EspeakPhonemizer

from src.data.phoneme_tokenizer import PhonemeTokenizer
from src.data.roman_tokenizer import RomanCharTokenizer

LANG_CONFIGS = {
    "en": {"espeak": "en-us", "name": "English", "id": 0},
    "fr": {"espeak": "fr", "name": "French", "id": 1},
    "es": {"espeak": "es", "name": "Spanish", "id": 2},
    "de": {"espeak": "de", "name": "German", "id": 3},
    "it": {"espeak": "it", "name": "Italian", "id": 4},
    "ar": {"espeak": "ar", "name": "Arabic", "id": 5},
    "zh": {"espeak": "cmn", "name": "Chinese", "id": 6},
    "ja": {"espeak": "ja", "name": "Japanese", "id": 7},
    "ko": {"espeak": "ko", "name": "Korean", "id": 8},
}

CLEAN_REGEX = re.compile(r"<[^>]+>|\[[^\]]+\]|\{[^\}]+\}|https?://\S+")


def clean_raw_line(line: str) -> str:
    """Strip markup, formatting tags, and excess whitespace."""
    line = CLEAN_REGEX.sub(" ", line)
    line = line.strip()
    return line


class MultilingualProcessor:
    def __init__(self):
        self.phonemizer = EspeakPhonemizer()
        self.phoneme_tok = PhonemeTokenizer()
        self.roman_tok = RomanCharTokenizer()
        self.kakasi = kakasi()

    def romanize(self, lang: str, text: str) -> str:
        """Convert native script into Romanized form for non-Latin languages."""
        if lang == "zh":
            return " ".join(pypinyin.lazy_pinyin(text, style=pypinyin.Style.TONE3))
        elif lang == "ja":
            return " ".join([x["hepburn"] for x in self.kakasi.convert(text)])
        elif lang == "ko":
            return Romanizer(text).romanize()
        elif lang == "ar":
            return ar_trans.utf82latin(text)
        return text

    def process_sentence(
        self, lang: str, text: str, min_words: int = 2, max_words: int = 48
    ) -> Optional[bytes]:
        """Convert a sentence into a packed binary sample."""
        text = clean_raw_line(text)
        if len(text) < 3 or len(text) > 400:
            return None

        # 1. Acoustic Phonemization (from native text)
        cfg = LANG_CONFIGS[lang]
        try:
            ph_chunks = self.phonemizer.phonemize(cfg["espeak"], text)
        except Exception:
            return None

        flat_ph = " ".join("".join(s) for s in ph_chunks).strip()
        if not flat_ph:
            return None

        ph_ids = self.phoneme_tok.encode(flat_ph)
        if not ph_ids or len(ph_ids) > 512:
            return None

        # 2. Orthographic Target Tokenization (Romanized lowercase)
        try:
            roman_text = self.romanize(lang, text)
            word_tokens = self.roman_tok.encode_words(roman_text)
        except Exception:
            return None

        if len(word_tokens) < min_words or len(word_tokens) > max_words:
            return None

        # 3. Pack into binary record:
        # Format:
        # [1B lang_id] [2B num_ph] [num_ph * 1B ph_ids] [2B num_words]
        # For each word: [1B word_len] [word_len * 1B word_tokens]
        lang_id = cfg["id"]
        ph_header = struct.pack("<H", len(ph_ids))
        ph_bytes = bytes(ph_ids)
        w_header = struct.pack("<H", len(word_tokens))
        w_data = bytearray()
        for w in word_tokens:
            w_data.append(len(w))
            w_data.extend(w)

        return bytes([lang_id]) + ph_header + ph_bytes + w_header + bytes(w_data)


def stream_language_lines(raw_dir: str, lang: str) -> Iterator[str]:
    """Interleave lines across all sources (tatoeba, wiki, books, opensub) for a language."""
    files = sorted(glob.glob(os.path.join(raw_dir, f"*_{lang}.txt.gz")))
    if not files:
        return

    # Prioritize Tatoeba -> Wiki -> Books -> OpenSubtitles
    priority_order = ["tatoeba", "wiki", "books", "opensub"]
    ordered_files = []
    for p in priority_order:
        for f in files:
            if p in os.path.basename(f) and f not in ordered_files:
                ordered_files.append(f)
    for f in files:
        if f not in ordered_files:
            ordered_files.append(f)

    for fpath in ordered_files:
        try:
            with gzip.open(fpath, "rt", encoding="utf-8", errors="ignore") as gz:
                for line in gz:
                    line = line.strip()
                    if line:
                        yield line
        except Exception as e:
            print(f"Warning: error reading {fpath}: {e}", file=sys.stderr)


def write_shard(
    output_dir: str,
    shard_idx: int,
    samples: List[bytes],
    lang_counts: Dict[str, int],
):
    """Write binary shard, index offsets, and metadata JSON."""
    os.makedirs(output_dir, exist_ok=True)
    bin_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.bin")
    idx_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.idx")
    meta_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.json")

    offsets = []
    current_offset = 0

    with open(bin_path, "wb") as f_bin:
        for s in samples:
            offsets.append((current_offset, len(s)))
            f_bin.write(s)
            current_offset += len(s)

    with open(idx_path, "wb") as f_idx:
        for off, length in offsets:
            f_idx.write(struct.pack("<QI", off, length))

    total_bytes = current_offset
    meta = {
        "shard_idx": shard_idx,
        "num_samples": len(samples),
        "total_bytes": total_bytes,
        "languages": lang_counts,
    }
    with open(meta_path, "w", encoding="utf-8") as f_meta:
        json.dump(meta, f_meta, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Multilingual Sharding Pipeline")
    parser.add_argument("--raw_dir", type=str, default="/media/hdd/Datasets/multilingual_text/raw")
    parser.add_argument("--output_dir", type=str, default="/media/hdd/Datasets/multilingual_text/shards")
    parser.add_argument("--samples_per_shard", type=int, default=50000)
    parser.add_argument("--max_samples_per_lang", type=int, default=250000)
    parser.add_argument("--min_words", type=int, default=2)
    parser.add_argument("--max_words", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("🌍 Multilingual Romanized Phoneme-to-Text Sharder")
    print(f"  Raw directory:          {args.raw_dir}")
    print(f"  Output directory:       {args.output_dir}")
    print(f"  Samples per shard:      {args.samples_per_shard:,}")
    print(f"  Max samples per lang:   {args.max_samples_per_lang:,}")
    print(f"  Target languages (9):   {', '.join(LANG_CONFIGS.keys())}")
    print("=" * 70)

    processor = MultilingualProcessor()

    # Create line generators for each language
    lang_iterators = {
        lang: stream_language_lines(args.raw_dir, lang)
        for lang in LANG_CONFIGS
    }

    shard_idx = 0
    current_shard_samples: List[bytes] = []
    current_shard_lang_counts: Dict[str, int] = {lang: 0 for lang in LANG_CONFIGS}
    total_processed = 0
    total_lang_counts: Dict[str, int] = {lang: 0 for lang in LANG_CONFIGS}

    active_langs = set(LANG_CONFIGS.keys())
    t0 = time.time()
    last_log_time = t0

    while active_langs:
        # Round-robin cycle across all active languages to ensure perfect balance
        for lang in list(active_langs):
            try:
                line = next(lang_iterators[lang])
            except StopIteration:
                active_langs.remove(lang)
                continue

            rec = processor.process_sentence(
                lang, line, min_words=args.min_words, max_words=args.max_words
            )
            if rec is None:
                continue

            current_shard_samples.append(rec)
            current_shard_lang_counts[lang] += 1
            total_lang_counts[lang] += 1
            total_processed += 1

            if total_lang_counts[lang] >= args.max_samples_per_lang:
                active_langs.discard(lang)

            if len(current_shard_samples) >= args.samples_per_shard:
                # Shuffle the shard to interleave languages and sentence lengths
                random.shuffle(current_shard_samples)
                write_shard(
                    args.output_dir,
                    shard_idx,
                    current_shard_samples,
                    current_shard_lang_counts,
                )
                dt = time.time() - t0
                speed = total_processed / max(dt, 1e-3)
                sz_mb = sum(len(s) for s in current_shard_samples) / (1024 * 1024)
                print(
                    f"📦 [Shard {shard_idx:05d}] Wrote {len(current_shard_samples):,} samples "
                    f"({sz_mb:.1f} MB) | Total: {total_processed:,} | Speed: {speed:.1f} s/s"
                )
                shard_idx += 1
                current_shard_samples = []
                current_shard_lang_counts = {l: 0 for l in LANG_CONFIGS}

        # Periodic status check
        if time.time() - last_log_time > 10.0:
            last_log_time = time.time()
            dt = time.time() - t0
            speed = total_processed / max(dt, 1e-3)
            active_str = ", ".join(f"{l}:{total_lang_counts[l]}" for l in LANG_CONFIGS)
            print(f"⏳ Progress: {total_processed:,} samples ({speed:.1f} s/s) | Active: {len(active_langs)} langs [{active_str}]")

    # Flush final remainder shard
    if current_shard_samples:
        random.shuffle(current_shard_samples)
        write_shard(
            args.output_dir,
            shard_idx,
            current_shard_samples,
            current_shard_lang_counts,
        )
        sz_mb = sum(len(s) for s in current_shard_samples) / (1024 * 1024)
        print(
            f"📦 [Shard {shard_idx:05d} (Final)] Wrote {len(current_shard_samples):,} samples "
            f"({sz_mb:.1f} MB) | Total: {total_processed:,}"
        )
        shard_idx += 1

    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print(f"🎉 Multilingual Sharding Complete in {total_time/60:.2f} minutes!")
    print(f"  Total shards generated: {shard_idx}")
    print(f"  Total valid samples:    {total_processed:,}")
    for lang, cnt in total_lang_counts.items():
        print(f"    - {lang.upper()} ({LANG_CONFIGS[lang]['name']}): {cnt:,} samples")
    print("=" * 70)


if __name__ == "__main__":
    main()
