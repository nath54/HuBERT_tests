#!/usr/bin/env python3
"""
Prepare a Strictly Held-Out 10,000 Sentence Test Dataset (5k English + 5k French).
Extracts clean literary sentences and computes their ground-truth Piper IPA phonemes.
Saves to data/test_sets/held_out_10k_test.json and registers exclusion hashes for training.
"""

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from piper.voice import PiperVoice

OUTPUT_DIR = Path("data/test_sets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_FILE = OUTPUT_DIR / "held_out_10k_test.json"
HASHES_FILE = OUTPUT_DIR / "held_out_hashes.json"

EN_VOICE_PATH = "/home/nathan/github/MADGen/data/piper_voices/en/en_US/lessac/en_US-lessac-medium.onnx"
FR_VOICE_PATH = "/home/nathan/github/MADGen/data/piper_voices/fr/fr_FR/siwis/fr_FR-siwis-medium.onnx"


def clean_sentence(text: str) -> str:
    """Normalize text into clean, punctually sound sentence."""
    if not text:
        return ""
    text = re.sub(r"=+.*?=+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\s+", " ", text).strip().strip("\"'«»“”")
    if text and not text.endswith((".", "!", "?")):
        text += "."
    return text


def main():
    target_per_lang = 5000
    print("=" * 65)
    print("   PREPARING HELD-OUT 10,000 SENTENCE BENCHMARK TEST DATASET")
    print(f"   Target: {target_per_lang:,} English + {target_per_lang:,} French sentences")
    print("=" * 65)

    print("[1/4] Loading Piper phonetic models for IPA ground-truth generation...")
    voice_en = PiperVoice.load(EN_VOICE_PATH)
    voice_fr = PiperVoice.load(FR_VOICE_PATH)
    print("Piper phonetic models ready.")

    print("\n[2/4] Streaming clean literary sentences from Opus Books (EN-FR)...")
    ds = load_dataset("Helsinki-NLP/opus_books", "en-fr", split="train", streaming=True)

    en_samples: List[Dict] = []
    fr_samples: List[Dict] = []
    hashes = set()

    t0 = time.time()
    for item in ds:
        trans = item.get("translation", {})
        raw_en = trans.get("en", "")
        raw_fr = trans.get("fr", "")

        c_en = clean_sentence(raw_en)
        c_fr = clean_sentence(raw_fr)

        # Check English candidate
        if len(en_samples) < target_per_lang:
            words_en = c_en.split()
            if 6 <= len(words_en) <= 22 and re.search(r"[a-zA-Z]", c_en):
                h_en = hashlib.md5(c_en.lower().encode("utf-8")).hexdigest()
                if h_en not in hashes:
                    hashes.add(h_en)
                    # Generate Piper phonemes
                    try:
                        ph_list = voice_en.phonemize(c_en)
                        phonemes = "".join(ph_list[0]) if ph_list else ""
                    except Exception:
                        phonemes = ""
                    en_samples.append({
                        "id": f"test_en_{len(en_samples):05d}",
                        "lang": "en",
                        "text": c_en,
                        "phonemes": phonemes,
                        "num_words": len(words_en),
                    })

        # Check French candidate
        if len(fr_samples) < target_per_lang:
            words_fr = c_fr.split()
            if 6 <= len(words_fr) <= 22 and re.search(r"[a-zA-Zà-üÀ-Ü]", c_fr):
                h_fr = hashlib.md5(c_fr.lower().encode("utf-8")).hexdigest()
                if h_fr not in hashes:
                    hashes.add(h_fr)
                    # Generate Piper phonemes
                    try:
                        ph_list = voice_fr.phonemize(c_fr)
                        phonemes = "".join(ph_list[0]) if ph_list else ""
                    except Exception:
                        phonemes = ""
                    fr_samples.append({
                        "id": f"test_fr_{len(fr_samples):05d}",
                        "lang": "fr",
                        "text": c_fr,
                        "phonemes": phonemes,
                        "num_words": len(words_fr),
                    })

        if len(en_samples) % 500 == 0 and len(en_samples) > 0:
            elapsed = time.time() - t0
            print(f"Collected: {len(en_samples):,}/{target_per_lang:,} EN | {len(fr_samples):,}/{target_per_lang:,} FR ({elapsed:.1f}s)")

        if len(en_samples) >= target_per_lang and len(fr_samples) >= target_per_lang:
            break

    all_test_samples = en_samples + fr_samples
    print(f"\n[3/4] Successfully collected {len(all_test_samples):,} sentences with IPA phonemes.")

    print(f"[4/4] Writing test manifest to {TEST_FILE} and hashes to {HASHES_FILE}...")
    with open(TEST_FILE, "w", encoding="utf-8") as f:
        json.dump(all_test_samples, f, indent=2, ensure_ascii=False)

    with open(HASHES_FILE, "w", encoding="utf-8") as f:
        json.dump(list(hashes), f, indent=2)

    file_size_mb = TEST_FILE.stat().st_size / (1024 * 1024)
    print(f"Test dataset saved! Size: {file_size_mb:.2f} MB.")
    print("Sample EN:")
    print("  Text:    ", en_samples[0]["text"])
    print("  Phonemes:", en_samples[0]["phonemes"])
    print("Sample FR:")
    print("  Text:    ", fr_samples[0]["text"])
    print("  Phonemes:", fr_samples[0]["phonemes"])
    print("\n✅ Held-out 10,000 dataset ready. Training loops will strictly exclude all of these sentences.")


if __name__ == "__main__":
    main()
