"""Parser and manifest generator for real LibriSpeech audio recordings."""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import soundfile as sf


def build_librispeech_manifest(
    root_dir: str = "data/raw/librispeech/LibriSpeech/dev-clean",
    output_dir: str = "data/librispeech",
    val_split: float = 0.15,
) -> Tuple[List[Dict], List[Dict]]:
    """Scan real LibriSpeech directory and create train/validation manifests."""
    root_path = Path(root_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Scanning LibriSpeech recordings in: {root_path}")
    trans_files = list(root_path.glob("*/*/*.trans.txt"))
    print(f"Found {len(trans_files)} chapter transcript files.")

    all_samples = []
    total_audio_seconds = 0.0

    for tf in trans_files:
        speaker_id = tf.parent.parent.name
        chapter_id = tf.parent.name
        with open(tf, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    utt_id, text = parts
                    flac_path = tf.parent / f"{utt_id}.flac"
                    if flac_path.exists():
                        # Read header for exact duration
                        info = sf.info(str(flac_path))
                        dur = float(info.duration)
                        total_audio_seconds += dur
                        all_samples.append({
                            "id": utt_id,
                            "audio_path": str(flac_path.resolve()),
                            "transcript": text.lower(),
                            "duration": round(dur, 2),
                            "speaker_id": speaker_id,
                            "chapter_id": chapter_id,
                        })

    # Sort deterministically
    all_samples.sort(key=lambda s: s["id"])
    total_hours = total_audio_seconds / 3600.0

    print(f"Total LibriSpeech Utterances: {len(all_samples):,} ({total_hours:.2f} hours of real human speech).")

    # Train / Val Split by speaker/chapters
    val_count = max(1, int(len(all_samples) * val_split))
    val_samples = all_samples[:val_count]
    train_samples = all_samples[val_count:]

    train_file = out_path / "librispeech_train.json"
    val_file = out_path / "librispeech_val.json"
    all_file = out_path / "librispeech_all.json"

    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2)
    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2)
    with open(all_file, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2)

    stats_file = out_path / "dataset_stats.json"
    stats = {
        "dataset_name": "LibriSpeech (dev-clean)",
        "total_utterances": len(all_samples),
        "train_utterances": len(train_samples),
        "val_utterances": len(val_samples),
        "total_audio_seconds": round(total_audio_seconds, 1),
        "total_audio_hours": round(total_hours, 2),
        "min_duration": min(s["duration"] for s in all_samples),
        "max_duration": max(s["duration"] for s in all_samples),
        "avg_duration": round(total_audio_seconds / len(all_samples), 2),
    }
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved LibriSpeech manifests to: {out_path}")
    return train_samples, val_samples
