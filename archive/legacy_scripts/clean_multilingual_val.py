#!/usr/bin/env python3
"""Audit and Clean Multilingual Validation Sets for Objective Phoneme Recognition.

Filters out:
1. Unexpanded digits (0-9) and currency/math symbols where spoken speech diverges from transcript.
2. Truncated audio or runaway transcripts (character rate outside [5.0, 30.0] chars/sec).
3. Extreme duration outliers (< 0.6s or > 20s).

Outputs clean validation manifest to data/multilingual_val_clean.json.
"""

import sys
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.multilingual_audio_dataset import (
    AudioUtterance,
    load_librispeech_manifest,
    load_mls_manifest,
)


DIGIT_RE = re.compile(r"\d")
SYMBOL_RE = re.compile(r"[\$€£%&#@\+=\<\>\\/]")


def audit_utterance(utt: AudioUtterance) -> Dict[str, Any]:
    text = utt.transcript.strip()
    reasons = []

    if DIGIT_RE.search(text):
        reasons.append("contains_digits")

    if SYMBOL_RE.search(text):
        reasons.append("contains_symbols")

    if len(text) < 3:
        reasons.append("too_short_text")

    # Check audio file and duration
    if not os.path.isfile(utt.audio_path):
        reasons.append("audio_missing")
        return {"valid": False, "reasons": reasons, "duration": 0.0, "rate": 0.0}

    try:
        info = sf.info(utt.audio_path)
        dur = info.duration
    except Exception:
        reasons.append("audio_unreadable")
        return {"valid": False, "reasons": reasons, "duration": 0.0, "rate": 0.0}

    if dur < 0.6:
        reasons.append("audio_too_short")
    elif dur > 20.0:
        reasons.append("audio_too_long")

    char_rate = len(text) / max(0.1, dur)
    if char_rate < 5.0:
        reasons.append(f"char_rate_too_low_{char_rate:.1f}")
    elif char_rate > 30.0:
        reasons.append(f"char_rate_too_high_{char_rate:.1f}")

    return {
        "valid": len(reasons) == 0,
        "reasons": reasons,
        "duration": dur,
        "rate": char_rate,
    }


def main():
    print("=" * 80)
    print("🔬 MULTILINGUAL VALIDATION SET SANITIZATION & AUDIT")
    print("=" * 80)

    # 1. English
    en_val = load_librispeech_manifest(
        "data/librispeech/librispeech_val.json" if os.path.exists("data/librispeech/librispeech_val.json") else "data/librispeech/dev_clean.json",
        max_samples=500,
    )
    # 2. French
    fr_val = load_mls_manifest("/media/hdd/Datasets/mls/mls_french", lang="fr", split="dev", max_samples=500)
    # 3. Spanish
    es_val = load_mls_manifest("/media/hdd/Datasets/mls/mls_spanish", lang="es", split="dev", max_samples=500)
    # 4. Italian
    it_val = load_mls_manifest("/media/hdd/Datasets/mls/mls_italian", lang="it", split="dev", max_samples=500)

    corpora = {
        "en": ("🇬🇧 English", en_val),
        "fr": ("🇫🇷 French", fr_val),
        "es": ("🇪🇸 Spanish", es_val),
        "it": ("🇮🇹 Italian", it_val),
    }

    clean_samples: List[Dict[str, Any]] = []

    for lang, (display_name, utts) in corpora.items():
        total = len(utts)
        valid_utts = []
        flagged_count = 0
        reasons_dist: Dict[str, int] = {}

        for u in utts:
            res = audit_utterance(u)
            if res["valid"]:
                valid_utts.append(u)
                clean_samples.append({
                    "id": u.id,
                    "audio_path": u.audio_path,
                    "transcript": u.transcript,
                    "lang": u.lang,
                    "duration": res["duration"],
                })
            else:
                flagged_count += 1
                for r in res["reasons"]:
                    r_key = r.split("_")[0] if "_" in r else r
                    reasons_dist[r_key] = reasons_dist.get(r_key, 0) + 1

        print(f"\n{display_name}: {total} examined -> {len(valid_utts)} passed ({len(valid_utts)/max(1, total)*100:.1f}%), {flagged_count} filtered")
        if reasons_dist:
            print(f"   Flags: {reasons_dist}")

    out_path = Path("data/multilingual_val_clean.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean_samples, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Clean validation manifest saved to: {out_path} ({len(clean_samples)} total utterances)")
    print("=" * 80)


if __name__ == "__main__":
    main()
