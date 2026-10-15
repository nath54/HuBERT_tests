"""Deterministic Train/Val Partitioning for Benchmark Protocol.

Splits data/librispeech/librispeech_train.json into:
- data/librispeech/benchmark_train.json (~95% of utterances)
- data/librispeech/benchmark_val.json (~5% of utterances)

Maintains deterministic speaker/chapter grouping so validation utterances
are unseen during training. Leaves test-clean untouched for unbiased evaluation.
"""

import json
from pathlib import Path
import random
from typing import Dict, List, Tuple


def create_benchmark_split(
    train_manifest_path: str = "data/librispeech/librispeech_train.json",
    output_dir: str = "data/librispeech",
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[Path, Path]:
    train_path = Path(train_manifest_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(f"Source manifest not found at: {train_path}")

    print(f"📖 Loading source manifest from: {train_path}")
    with open(train_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    total_samples = len(samples)
    print(f"Total utterances found: {total_samples:,}")

    # Group by (speaker_id, chapter_id) for clean acoustic separation
    chapters: Dict[Tuple[str, str], List[Dict]] = {}
    for s in samples:
        key = (s.get("speaker_id", "unknown"), s.get("chapter_id", "unknown"))
        chapters.setdefault(key, []).append(s)

    chapter_keys = sorted(list(chapters.keys()))
    rng = random.Random(seed)
    rng.shuffle(chapter_keys)

    target_val_samples = int(total_samples * val_ratio)
    val_samples = []
    train_samples = []

    for key in chapter_keys:
        chap_utts = chapters[key]
        if len(val_samples) < target_val_samples:
            val_samples.extend(chap_utts)
        else:
            train_samples.extend(chap_utts)

    # Sort deterministically by id
    train_samples.sort(key=lambda s: s["id"])
    val_samples.sort(key=lambda s: s["id"])

    # Verify zero overlap
    train_ids = {s["id"] for s in train_samples}
    val_ids = {s["id"] for s in val_samples}
    overlap = train_ids.intersection(val_ids)
    assert len(overlap) == 0, f"Critical error: {len(overlap)} overlapping IDs found between train and val!"

    train_out_path = out_dir / "benchmark_train.json"
    val_out_path = out_dir / "benchmark_val.json"

    with open(train_out_path, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2)

    with open(val_out_path, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2)

    train_hours = sum(s.get("duration", 0.0) for s in train_samples) / 3600.0
    val_hours = sum(s.get("duration", 0.0) for s in val_samples) / 3600.0

    print("✅ Partitioning Complete:")
    print(f"  • Train Set : {len(train_samples):,} utterances ({train_hours:.2f} hours) -> {train_out_path}")
    print(f"  • Val Set   : {len(val_samples):,} utterances ({val_hours:.2f} hours) -> {val_out_path}")
    print(f"  • Disjoint Check: 0 overlapping utterances.")

    return train_out_path, val_out_path


if __name__ == "__main__":
    create_benchmark_split()
