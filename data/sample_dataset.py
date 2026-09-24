"""Dataset creation, generation, and downloading utilities."""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import soundfile as sf
import torch

from src.utils.audio import save_audio, synthesize_spoken_word


DEFAULT_VOCABULARY_WORDS = [
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
    "yes", "no", "hello", "world", "speech",
    "audio", "learn", "deep", "mind", "hubert",
]


def generate_synthetic_asr_dataset(
    output_dir: str = "data/raw/synthetic",
    num_train: int = 120,
    num_val: int = 30,
    sample_rate: int = 16000,
    words: Optional[List[str]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """Generate a clean synthetic ASR speech dataset with diverse acoustic pitch and formants.
    
    Creates WAV files and manifest JSON files for train and validation splits.
    """
    out_path = Path(output_dir)
    audio_dir = out_path / "wavs"
    audio_dir.mkdir(parents=True, exist_ok=True)

    word_pool = words or DEFAULT_VOCABULARY_WORDS

    train_samples = []
    val_samples = []

    print(f"Generating synthetic acoustic speech dataset in: {out_path}")

    # Generate single words and short combinations
    total_samples = num_train + num_val

    for i in range(total_samples):
        # Choose 1 or 2 words
        is_phrase = (i % 3 == 0)
        if is_phrase:
            w1 = random.choice(word_pool)
            w2 = random.choice(word_pool)
            transcript = f"{w1} {w2}"
            duration = random.uniform(1.2, 1.6)
            f0 = random.uniform(100.0, 220.0)  # Diverse speaker pitch
            # Synthesize 2 connected words
            wav1 = synthesize_spoken_word(w1, duration_s=duration / 2, sample_rate=sample_rate, f0=f0)
            wav2 = synthesize_spoken_word(w2, duration_s=duration / 2, sample_rate=sample_rate, f0=f0 * 0.95)
            # Add short silence between words
            silence = torch.zeros(int(0.1 * sample_rate))
            wav = torch.cat([wav1, silence, wav2], dim=0)
        else:
            transcript = random.choice(word_pool)
            duration = random.uniform(0.6, 0.9)
            f0 = random.uniform(110.0, 240.0)
            wav = synthesize_spoken_word(transcript, duration_s=duration, sample_rate=sample_rate, f0=f0)

        wav_filename = f"sample_{i:04d}.wav"
        wav_filepath = audio_dir / wav_filename
        save_audio(wav, wav_filepath, sample_rate=sample_rate)

        entry = {
            "id": f"synthetic_{i:04d}",
            "audio_path": str(wav_filepath),
            "transcript": transcript,
            "duration": float(len(wav) / sample_rate),
        }

        if i < num_train:
            train_samples.append(entry)
        else:
            val_samples.append(entry)

    # Save manifests
    train_manifest = out_path / "train_manifest.json"
    val_manifest = out_path / "val_manifest.json"

    with open(train_manifest, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, indent=2)
    with open(val_manifest, "w", encoding="utf-8") as f:
        json.dump(val_samples, f, indent=2)

    print(f"Generated {len(train_samples)} training and {len(val_samples)} validation samples.")
    return train_samples, val_samples


def load_manifest(manifest_path: str) -> List[Dict]:
    """Load sample metadata list from a JSON manifest."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)
