#!/usr/bin/env python3
"""Script to prepare ASR datasets (synthetic or LibriSpeech)."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.sample_dataset import generate_synthetic_asr_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare or synthesize ASR dataset.")
    parser.add_argument("--output_dir", type=str, default="data/raw/synthetic", help="Output directory for data")
    parser.add_argument("--num_train", type=int, default=150, help="Number of training samples")
    parser.add_argument("--num_val", type=int, default=30, help="Number of validation samples")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Audio sample rate (Hz)")
    args = parser.parse_args()

    print(f"Generating synthetic speech dataset: {args.num_train} train, {args.num_val} val...")
    generate_synthetic_asr_dataset(
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        sample_rate=args.sample_rate,
    )
    print(f"[Done] Dataset manifests created in {args.output_dir}")


if __name__ == "__main__":
    main()
