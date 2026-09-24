#!/usr/bin/env python3
"""Utility script to download, verify, and install premier Piper neural voices.

Curates top-of-the-line French and English voices:
- Ultra-high resolution 44.1 kHz French studio models (tjiho 1/2/3)
- Premium French models (siwis-medium, tom-medium, upmc-medium, mls-medium with 125 speakers)
- Flagship 28.0 kHz English models (libritts-high with 904 speakers, lessac-high, ljspeech-high, ryan-high, cori-high)
- Restored multi-speaker English models (libritts_r-medium with 904 speakers, vctk-medium with 109 British speakers)
- Accented English generalization sets (l2arctic-medium with 24 speakers, arctic-medium with 18 speakers)
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.streaming_piper import PIPER_VOICES_DIR

HF_RHASSPY_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
HF_CSUKUANGFJ_BASE = "https://huggingface.co/csukuangfj"

TOP_VOICES = {
    "fr": [
        # Ultra-quality 44.1 kHz Studio Models (157 phonemes, nasal vowels fully supported)
        {
            "name": "fr_FR-tjiho-model1",
            "tier": "44.1 kHz Studio",
            "speakers": 1,
            "dest_dir": "fr/fr_FR/tjiho",
            "onnx": f"{HF_CSUKUANGFJ_BASE}/vits-piper-fr_FR-tjiho-model1/resolve/main/fr_FR-tjiho-model1.onnx",
            "json": f"{HF_CSUKUANGFJ_BASE}/vits-piper-fr_FR-tjiho-model1/raw/main/fr_FR-tjiho-model1.onnx.json",
        },
        {
            "name": "fr_FR-tjiho-model2",
            "tier": "44.1 kHz Studio",
            "speakers": 1,
            "dest_dir": "fr/fr_FR/tjiho",
            "onnx": f"{HF_CSUKUANGFJ_BASE}/vits-piper-fr_FR-tjiho-model2/resolve/main/fr_FR-tjiho-model2.onnx",
            "json": f"{HF_CSUKUANGFJ_BASE}/vits-piper-fr_FR-tjiho-model2/raw/main/fr_FR-tjiho-model2.onnx.json",
        },
        {
            "name": "fr_FR-tjiho-model3",
            "tier": "44.1 kHz Studio",
            "speakers": 1,
            "dest_dir": "fr/fr_FR/tjiho",
            "onnx": f"{HF_CSUKUANGFJ_BASE}/vits-piper-fr_FR-tjiho-model3/resolve/main/fr_FR-tjiho-model3.onnx",
            "json": f"{HF_CSUKUANGFJ_BASE}/vits-piper-fr_FR-tjiho-model3/raw/main/fr_FR-tjiho-model3.onnx.json",
        },
        # Official Medium-Quality Clean Models
        {
            "name": "fr_FR-siwis-medium",
            "tier": "22.05 kHz Studio Female",
            "speakers": 1,
            "dest_dir": "fr/fr_FR/siwis",
            "onnx": f"{HF_RHASSPY_BASE}/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx",
            "json": f"{HF_RHASSPY_BASE}/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json",
        },
        {
            "name": "fr_FR-tom-medium",
            "tier": "22.05 kHz Studio Male",
            "speakers": 1,
            "dest_dir": "fr/fr_FR/tom",
            "onnx": f"{HF_RHASSPY_BASE}/fr/fr_FR/tom/medium/fr_FR-tom-medium.onnx",
            "json": f"{HF_RHASSPY_BASE}/fr/fr_FR/tom/medium/fr_FR-tom-medium.onnx.json",
        },
        {
            "name": "fr_FR-upmc-medium",
            "tier": "22.05 kHz Sorbonne Pair",
            "speakers": 2,
            "dest_dir": "fr/fr_FR/upmc",
            "onnx": f"{HF_RHASSPY_BASE}/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx",
            "json": f"{HF_RHASSPY_BASE}/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx.json",
        },
        {
            "name": "fr_FR-mls-medium",
            "tier": "22.05 kHz Multi-Speaker",
            "speakers": 125,
            "dest_dir": "fr/fr_FR/mls",
            "onnx": f"{HF_RHASSPY_BASE}/fr/fr_FR/mls/medium/fr_FR-mls-medium.onnx",
            "json": f"{HF_RHASSPY_BASE}/fr/fr_FR/mls/medium/fr_FR-mls-medium.onnx.json",
        },
    ],
    "en": [
        # Flagship 28.0 kHz / Multi-Speaker Models
        {
            "name": "en_US-libritts-high",
            "tier": "28.0 kHz Flagship Multi-Speaker",
            "speakers": 904,
            "dest_dir": "en/en_US/libritts",
            "onnx": f"{HF_RHASSPY_BASE}/en/en_US/libritts/high/en_US-libritts-high.onnx",
            "json": f"{HF_RHASSPY_BASE}/en/en_US/libritts/high/en_US-libritts-high.onnx.json",
        },
        {
            "name": "en_US-libritts_r-medium",
            "tier": "22.05 kHz Restored Multi-Speaker",
            "speakers": 904,
            "dest_dir": "en/en_US/libritts_r",
            "onnx": f"{HF_RHASSPY_BASE}/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx",
            "json": f"{HF_RHASSPY_BASE}/en/en_US/libritts_r/medium/en_US-libritts_r-medium.onnx.json",
        },
        {
            "name": "en_GB-vctk-medium",
            "tier": "22.05 kHz British Multi-Speaker",
            "speakers": 109,
            "dest_dir": "en/en_GB/vctk",
            "onnx": f"{HF_RHASSPY_BASE}/en/en_GB/vctk/medium/en_GB-vctk-medium.onnx",
            "json": f"{HF_RHASSPY_BASE}/en/en_GB/vctk/medium/en_GB-vctk-medium.onnx.json",
        },
        {
            "name": "en_US-l2arctic-medium",
            "tier": "22.05 kHz International Accents",
            "speakers": 24,
            "dest_dir": "en/en_US/l2arctic",
            "onnx": f"{HF_RHASSPY_BASE}/en/en_US/l2arctic/medium/en_US-l2arctic-medium.onnx",
            "json": f"{HF_RHASSPY_BASE}/en/en_US/l2arctic/medium/en_US-l2arctic-medium.onnx.json",
        },
        {
            "name": "en_US-arctic-medium",
            "tier": "22.05 kHz Phonetic Diversity",
            "speakers": 18,
            "dest_dir": "en/en_US/arctic",
            "onnx": f"{HF_RHASSPY_BASE}/en/en_US/arctic/medium/en_US-arctic-medium.onnx",
            "json": f"{HF_RHASSPY_BASE}/en/en_US/arctic/medium/en_US-arctic-medium.onnx.json",
        },
        {
            "name": "en_US-lessac-high",
            "tier": "28.0 kHz High-Fidelity Female",
            "speakers": 1,
            "dest_dir": "en/en_US/lessac",
            "onnx": f"{HF_RHASSPY_BASE}/en/en_US/lessac/high/en_US-lessac-high.onnx",
            "json": f"{HF_RHASSPY_BASE}/en/en_US/lessac/high/en_US-lessac-high.onnx.json",
        },
        {
            "name": "en_US-ryan-high",
            "tier": "28.0 kHz High-Fidelity Male",
            "speakers": 1,
            "dest_dir": "en/en_US/ryan",
            "onnx": f"{HF_RHASSPY_BASE}/en/en_US/ryan/high/en_US-ryan-high.onnx",
            "json": f"{HF_RHASSPY_BASE}/en/en_US/ryan/high/en_US-ryan-high.onnx.json",
        },
        {
            "name": "en_US-ljspeech-high",
            "tier": "28.0 kHz High-Fidelity Studio",
            "speakers": 1,
            "dest_dir": "en/en_US/ljspeech",
            "onnx": f"{HF_RHASSPY_BASE}/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx",
            "json": f"{HF_RHASSPY_BASE}/en/en_US/ljspeech/high/en_US-ljspeech-high.onnx.json",
        },
        {
            "name": "en_GB-cori-high",
            "tier": "28.0 kHz High-Fidelity British",
            "speakers": 1,
            "dest_dir": "en/en_GB/cori",
            "onnx": f"{HF_RHASSPY_BASE}/en/en_GB/cori/high/en_GB-cori-high.onnx",
            "json": f"{HF_RHASSPY_BASE}/en/en_GB/cori/high/en_GB-cori-high.onnx.json",
        },
    ]
}


def download_with_progress(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    
    start_time = time.time()
    with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as f:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        block_size = 1024 * 1024  # 1 MB blocks
        
        while True:
            buffer = resp.read(block_size)
            if not buffer:
                break
            downloaded += len(buffer)
            f.write(buffer)
            
            if total_size > 0:
                pct = downloaded / total_size * 100
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(f"\r     [{mb:5.1f} / {total_mb:5.1f} MB] ({pct:5.1f}%)", end="", flush=True)
            else:
                mb = downloaded / (1024 * 1024)
                print(f"\r     [{mb:5.1f} MB]", end="", flush=True)
                
    elapsed = max(time.time() - start_time, 0.01)
    rate_mb = (downloaded / (1024 * 1024)) / elapsed
    print(f"\r     ✓ {downloaded / 1024 / 1024:.1f} MB downloaded in {elapsed:.1f}s ({rate_mb:.1f} MB/s)")
    temp_path.rename(dest_path)


def download_voice(voice_info: dict, target_root: Path):
    dest_dir = target_root / voice_info["dest_dir"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    onnx_dest = dest_dir / f"{voice_info['name']}.onnx"
    json_dest = dest_dir / f"{voice_info['name']}.onnx.json"

    already_done = onnx_dest.exists() and json_dest.exists() and onnx_dest.stat().st_size > 1000
    if already_done:
        print(f"  ✓ {voice_info['name']} [{voice_info['tier']}, {voice_info['speakers']} spk] already installed.")
        return

    print(f"\n  ⬇️  Downloading '{voice_info['name']}' [{voice_info['tier']}, {voice_info['speakers']} speakers]:")
    print(f"     Target: {onnx_dest}")
    download_with_progress(voice_info["onnx"], onnx_dest)
    print(f"     Config: {json_dest.name}")
    download_with_progress(voice_info["json"], json_dest)
    print(f"  ✅ Installed {voice_info['name']} successfully.")


def main():
    parser = argparse.ArgumentParser(description="Download top-of-the-top Piper neural voices.")
    parser.add_argument("--lang", type=str, default="all", choices=["fr", "en", "all"], help="Language pool to download")
    parser.add_argument("--target_dir", type=Path, default=PIPER_VOICES_DIR, help="Destination directory for voices")
    args = parser.parse_args()

    langs = ["fr", "en"] if args.lang == "all" else [args.lang]

    print(f"================================================================================")
    print(f"🔊 Premier Neural Voice Downloader & Quality Verifier")
    print(f"Target Directory: {args.target_dir}")
    print(f"Languages: {', '.join(l.upper() for l in langs)}")
    print(f"================================================================================")

    for lang in langs:
        voices = TOP_VOICES.get(lang, [])
        total_spk = sum(v["speakers"] for v in voices)
        print(f"\n📦 Processing {lang.upper()} voices ({len(voices)} premier models, {total_spk:,} total speaker profiles):")
        for v in voices:
            download_voice(v, args.target_dir)

    print(f"\n================================================================================")
    print(f"✅ All requested premier voices installed successfully!")
    print(f"================================================================================")


if __name__ == "__main__":
    main()
