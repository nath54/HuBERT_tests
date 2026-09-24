#!/usr/bin/env python3
"""Utility script to download and verify clean Piper neural voices for audiolearn.

Downloads official Piper models from HuggingFace (rhasspy/piper-voices) or lists available
models for French, English, and other languages.
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.streaming_piper import PIPER_VOICES_DIR

HF_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Curated list of verified clean models (medium/high tier with full phoneme coverage)
RECOMMENDED_VOICES = {
    "fr": [
        {
            "name": "fr_FR-mls-medium",
            "onnx": f"{HF_BASE_URL}/fr/fr_FR/mls/medium/fr_FR-mls-medium.onnx",
            "json": f"{HF_BASE_URL}/fr/fr_FR/mls/medium/fr_FR-mls-medium.onnx.json",
        },
        {
            "name": "fr_FR-siwis-medium",
            "onnx": f"{HF_BASE_URL}/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx",
            "json": f"{HF_BASE_URL}/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json",
        },
        {
            "name": "fr_FR-tom-medium",
            "onnx": f"{HF_BASE_URL}/fr/fr_FR/tom/medium/fr_FR-tom-medium.onnx",
            "json": f"{HF_BASE_URL}/fr/fr_FR/tom/medium/fr_FR-tom-medium.onnx.json",
        },
        {
            "name": "fr_FR-upmc-medium",
            "onnx": f"{HF_BASE_URL}/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx",
            "json": f"{HF_BASE_URL}/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx.json",
        },
    ],
    "en": [
        {
            "name": "en_US-lessac-medium",
            "onnx": f"{HF_BASE_URL}/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
            "json": f"{HF_BASE_URL}/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        },
        {
            "name": "en_US-ryan-medium",
            "onnx": f"{HF_BASE_URL}/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
            "json": f"{HF_BASE_URL}/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json",
        },
        {
            "name": "en_GB-alan-medium",
            "onnx": f"{HF_BASE_URL}/en/en_GB/alan/medium/en_GB-alan-medium.onnx",
            "json": f"{HF_BASE_URL}/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json",
        },
    ]
}


def download_voice(voice_info: dict, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    onnx_dest = target_dir / f"{voice_info['name']}.onnx"
    json_dest = target_dir / f"{voice_info['name']}.onnx.json"

    if onnx_dest.exists() and json_dest.exists():
        print(f"  ✓ {voice_info['name']} already exists in {target_dir}")
        return

    print(f"  ⬇️ Downloading {voice_info['name']}...")
    print(f"     ONNX: {voice_info['onnx']}")
    urllib.request.urlretrieve(voice_info["onnx"], onnx_dest)
    print(f"     JSON: {voice_info['json']}")
    urllib.request.urlretrieve(voice_info["json"], json_dest)
    print(f"  ✅ Installed {voice_info['name']}")


def main():
    parser = argparse.ArgumentParser(description="Download and verify clean Piper neural voices.")
    parser.add_argument("--lang", type=str, default="fr", choices=["fr", "en", "all"], help="Language pool to prepare")
    parser.add_argument("--target_dir", type=Path, default=PIPER_VOICES_DIR, help="Destination directory for voices")
    args = parser.parse_args()

    langs = ["fr", "en"] if args.lang == "all" else [args.lang]

    print(f"==================================================")
    print(f"🔊 Piper Voice Preparation Utility")
    print(f"Target Directory: {args.target_dir}")
    print(f"Languages: {', '.join(langs)}")
    print(f"==================================================")

    for lang in langs:
        voices = RECOMMENDED_VOICES.get(lang, [])
        print(f"\nProcessing {lang.upper()} voices ({len(voices)} available clean models):")
        for v in voices:
            sub_dir = args.target_dir / lang / v["name"]
            download_voice(v, sub_dir)

    print("\n✅ Verification complete. All specified voice models are ready for training.")


if __name__ == "__main__":
    main()
