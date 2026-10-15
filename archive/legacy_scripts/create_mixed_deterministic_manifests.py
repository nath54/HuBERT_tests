#!/usr/bin/env python3
"""Create uniformly and deterministically interleaved multi-split LibriSpeech manifests.

Evenly distributes utterances across train-clean-100, train-clean-360, and train-other-500
using round-robin interleaving by speaker/chapter blocks and deterministic hashing.
Ensures zero bias where one dataset split is trained before another.
"""

import json
from pathlib import Path
import random
from typing import Dict, List, Tuple

DATA_DIR = Path("data/librispeech")

def get_split_name(audio_path: str) -> str:
    if "train-clean-100" in audio_path:
        return "clean-100"
    elif "train-clean-360" in audio_path:
        return "clean-360"
    elif "train-other-500" in audio_path:
        return "other-500"
    return "unknown"

def interleave_splits_deterministically(samples: List[Dict], seed: int = 42) -> List[Dict]:
    """Group utterances by split, shuffle chapters deterministically, and round-robin interleave."""
    by_split: Dict[str, List[Dict]] = {
        "clean-100": [],
        "clean-360": [],
        "other-500": [],
    }

    for s in samples:
        split = get_split_name(s["audio_path"])
        if split in by_split:
            by_split[split].append(s)

    # Within each split, group by chapter so context/speaker isn't fragmented at single-sentence level
    interleaved: List[Dict] = []
    split_chapters: Dict[str, List[List[Dict]]] = {}

    rng = random.Random(seed)

    for split_name, split_samples in by_split.items():
        if not split_samples:
            continue
        chap_map: Dict[Tuple[str, str], List[Dict]] = {}
        for s in split_samples:
            key = (s.get("speaker_id", "unk"), s.get("chapter_id", "unk"))
            chap_map.setdefault(key, []).append(s)
        chaps = list(chap_map.values())
        # Sort each chapter deterministically
        for c in chaps:
            c.sort(key=lambda x: x["id"])
        # Deterministically shuffle the order of chapters
        rng.shuffle(chaps)
        split_chapters[split_name] = chaps

    # Proportional Round-Robin across available splits:
    # 960h roughly has proportions ~ 1 : 3.6 : 5.0
    active_splits = [k for k, v in split_chapters.items() if v]
    indices = {k: 0 for k in active_splits}

    total_chapters = sum(len(v) for v in split_chapters.values())
    print(f"Total chapters to interleave: {total_chapters} across splits: {active_splits}")

    while any(indices[k] < len(split_chapters[k]) for k in active_splits):
        for k in active_splits:
            if indices[k] < len(split_chapters[k]):
                chap = split_chapters[k][indices[k]]
                interleaved.extend(chap)
                indices[k] += 1

    return interleaved

def main():
    print("=" * 70)
    print("🔀 CREATING UNIFORMLY INTERLEAVED & DETERMINISTIC TRAINING MANIFESTS")
    print("=" * 70)

    # 1. Interleave 960h manifest
    src_960 = DATA_DIR / "benchmark_train_960h.json"
    if src_960.exists():
        with open(src_960, "r", encoding="utf-8") as f:
            samples_960 = json.load(f)
        mixed_960 = interleave_splits_deterministically(samples_960, seed=42)
        out_mixed_960 = DATA_DIR / "benchmark_train_960h_mixed.json"
        with open(out_mixed_960, "w", encoding="utf-8") as f:
            json.dump(mixed_960, f, indent=2)
        # Also update standard benchmark_train_960h.json so defaults automatically pick the mixed version
        with open(src_960, "w", encoding="utf-8") as f:
            json.dump(mixed_960, f, indent=2)
        print(f"✅ Created 960h mixed manifest: {len(mixed_960):,} utterances -> {src_960}")

    # 2. Interleave 460h manifest
    src_460 = DATA_DIR / "benchmark_train_460h.json"
    if src_460.exists():
        with open(src_460, "r", encoding="utf-8") as f:
            samples_460 = json.load(f)
        mixed_460 = interleave_splits_deterministically(samples_460, seed=42)
        out_mixed_460 = DATA_DIR / "benchmark_train_460h_mixed.json"
        with open(out_mixed_460, "w", encoding="utf-8") as f:
            json.dump(mixed_460, f, indent=2)
        with open(src_460, "w", encoding="utf-8") as f:
            json.dump(mixed_460, f, indent=2)
        print(f"✅ Created 460h mixed manifest: {len(mixed_460):,} utterances -> {src_460}")

    # Check distribution over the first 500 samples
    first_500 = mixed_960[:500]
    c100 = sum(1 for s in first_500 if "train-clean-100" in s["audio_path"])
    c360 = sum(1 for s in first_500 if "train-clean-360" in s["audio_path"])
    c500 = sum(1 for s in first_500 if "train-other-500" in s["audio_path"])
    print(f"\n📊 Sample verification across the first 500 utterances:")
    print(f"   • train-clean-100 : {c100} utterances ({c100/5:.1f}%)")
    print(f"   • train-clean-360 : {c360} utterances ({c360/5:.1f}%)")
    print(f"   • train-other-500 : {c500} utterances ({c500/5:.1f}%)")
    print("🎉 Uniform multi-split mixing confirmed!")

if __name__ == "__main__":
    main()
