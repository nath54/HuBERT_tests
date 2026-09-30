#!/usr/bin/env python3
"""Update validation and training manifests for multi-split LibriSpeech (100h + 360h + 500h).

Extracts held-out validation chapters from train-clean-360 and train-other-500,
combines them with the existing benchmark_val.json, and ensures complete disjointness
from the updated training manifests.
"""

import json
from pathlib import Path
import random
from typing import Dict, List, Set, Tuple

DATA_DIR = Path("data/librispeech")

def sample_chapters(samples: List[Dict], target_hours: float, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    chapters: Dict[Tuple[str, str], List[Dict]] = {}
    for s in samples:
        key = (s.get("speaker_id", "unk"), s.get("chapter_id", "unk"))
        chapters.setdefault(key, []).append(s)

    chapter_keys = sorted(list(chapters.keys()))
    rng = random.Random(seed)
    rng.shuffle(chapter_keys)

    val_samples = []
    train_samples = []
    target_seconds = target_hours * 3600.0
    accumulated_sec = 0.0

    for k in chapter_keys:
        chap_utts = chapters[k]
        chap_sec = sum(u.get("duration", 0.0) for u in chap_utts)
        if accumulated_sec < target_seconds:
            val_samples.extend(chap_utts)
            accumulated_sec += chap_sec
        else:
            train_samples.extend(chap_utts)

    return val_samples, train_samples

def main():
    print("=" * 70)
    print("🔄 EXPANDING VALIDATION SUITE WITH CLEAN-360 AND OTHER-500 SAMPLES")
    print("=" * 70)

    # 1. Load existing benchmark_val.json
    val_100_path = DATA_DIR / "benchmark_val.json"
    with open(val_100_path, "r", encoding="utf-8") as f:
        val_100 = json.load(f)
    print(f"• Current Val Set (Clean-100) : {len(val_100):,} utts ({sum(s['duration'] for s in val_100)/3600:.2f}h)")

    # 2. Load 960h manifest
    manifest_960_path = DATA_DIR / "librispeech_train_960h.json"
    with open(manifest_960_path, "r", encoding="utf-8") as f:
        all_960 = json.load(f)

    # Separate into 360h and 500h splits
    samples_360 = [s for s in all_960 if "train-clean-360" in s["audio_path"]]
    samples_500 = [s for s in all_960 if "train-other-500" in s["audio_path"]]
    samples_100 = [s for s in all_960 if "train-clean-100" in s["audio_path"]]

    print(f"• Found Clean-360 candidates  : {len(samples_360):,} utts ({sum(s['duration'] for s in samples_360)/3600:.2f}h)")
    print(f"• Found Other-500 candidates  : {len(samples_500):,} utts ({sum(s['duration'] for s in samples_500)/3600:.2f}h)")

    # Sample ~5 hours from clean-360 and ~5 hours from other-500
    val_360, rem_360 = sample_chapters(samples_360, target_hours=5.0, seed=42)
    val_500, rem_500 = sample_chapters(samples_500, target_hours=5.0, seed=43)

    print(f"• Sampled from Clean-360      : {len(val_360):,} utts ({sum(s['duration'] for s in val_360)/3600:.2f}h)")
    print(f"• Sampled from Other-500      : {len(val_500):,} utts ({sum(s['duration'] for s in val_500)/3600:.2f}h)")

    # 3. Create comprehensive Multi-Split Validation Suite
    comprehensive_val = list(val_100) + list(val_360) + list(val_500)
    comprehensive_val.sort(key=lambda s: s["id"])

    val_ids: Set[str] = {s["id"] for s in comprehensive_val}
    val_speakers: Set[str] = {s["speaker_id"] for s in comprehensive_val}

    # Backup previous benchmark_val.json
    val_backup = DATA_DIR / "benchmark_val_clean100_only.json"
    if not val_backup.exists():
        import shutil
        shutil.copyfile(val_100_path, val_backup)
        print(f"Backed up Clean-100 val to {val_backup}")

    # Overwrite benchmark_val.json with the multi-split suite
    with open(val_100_path, "w", encoding="utf-8") as f:
        json.dump(comprehensive_val, f, indent=2)

    total_val_hours = sum(s["duration"] for s in comprehensive_val) / 3600.0
    print(f"\n🌟 Comprehensive Validation Suite Saved to {val_100_path}:")
    print(f"   • Total Utterances: {len(comprehensive_val):,}")
    print(f"   • Total Audio: {total_val_hours:.2f} hours")
    print(f"   • Composition: Clean-100 ({len(val_100):,}) + Clean-360 ({len(val_360):,}) + Other-500 ({len(val_500):,})")

    # 4. Create strictly disjoint training manifests (filtered of all val IDs)
    train_960_filtered = [s for s in all_960 if s["id"] not in val_ids]
    train_460_filtered = [s for s in train_960_filtered if "train-other-500" not in s["audio_path"]]

    with open(DATA_DIR / "benchmark_train_960h.json", "w", encoding="utf-8") as f:
        json.dump(train_960_filtered, f, indent=2)

    with open(DATA_DIR / "benchmark_train_460h.json", "w", encoding="utf-8") as f:
        json.dump(train_460_filtered, f, indent=2)

    print("\n✅ Filtered Training Manifests Created (Strictly Disjoint from Validation):")
    print(f"   • benchmark_train_460h.json: {len(train_460_filtered):,} utts ({sum(s['duration'] for s in train_460_filtered)/3600:.2f}h)")
    print(f"   • benchmark_train_960h.json: {len(train_960_filtered):,} utts ({sum(s['duration'] for s in train_960_filtered)/3600:.2f}h)")

    # Assert 0 overlap
    overlap = val_ids.intersection({s["id"] for s in train_960_filtered})
    assert len(overlap) == 0, f"Error: {len(overlap)} overlapping IDs found!"
    print("🔒 Verification Passed: Exact 0 overlap between validation and training manifests.")

if __name__ == "__main__":
    main()
