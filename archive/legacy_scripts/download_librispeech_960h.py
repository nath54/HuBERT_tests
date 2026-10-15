#!/usr/bin/env python3
"""Piped, streaming downloader and extractor for LibriSpeech 360h & 500h with auto-cleanup.

Protects disk space by downloading, extracting, and deleting archives sequentially.
"""

import os
import subprocess
import sys
from pathlib import Path
import json
import soundfile as sf

BASE_URL = "https://www.openslr.org/resources/12"
RAW_DIR = Path("data/raw/librispeech/LibriSpeech")
MANIFEST_DIR = Path("data/librispeech")

SPLITS = [
    ("train-clean-360", f"{BASE_URL}/train-clean-360.tar.gz"),
    ("train-other-500", f"{BASE_URL}/train-other-500.tar.gz"),
]

def check_disk_free_gb():
    stat = os.statvfs(".")
    free_bytes = stat.f_bavail * stat.f_frsize
    return free_bytes / (1024 ** 3)

def download_and_extract_split(split_name, url):
    dest_dir = RAW_DIR / split_name
    if dest_dir.exists() and any(dest_dir.glob("*/*/*.flac")):
        print(f"✅ Split '{split_name}' already exists at {dest_dir}. Skipping download.")
        return True

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = RAW_DIR.parent / f"{split_name}.tar.gz"

    free_gb = check_disk_free_gb()
    print(f"📊 Free disk space before {split_name}: {free_gb:.1f} GB")
    if free_gb < 35.0:
        print(f"⚠️ Insufficient disk space ({free_gb:.1f} GB < 35 GB threshold) to safely process {split_name}!")
        return False

    print(f"\n⬇️ Downloading {split_name} from {url}...")
    dl_cmd = ["curl", "-L", "-C", "-", url, "-o", str(tar_path), "--retry", "5", "--retry-delay", "3"]
    ret = subprocess.run(dl_cmd)
    if ret.returncode != 0:
        print(f"❌ Failed to download {split_name}!")
        if tar_path.exists():
            tar_path.unlink()
        return False

    print(f"\n📦 Extracting {split_name} directly to {RAW_DIR}...")
    ext_cmd = ["tar", "-xzf", str(tar_path), "-C", str(RAW_DIR.parent)]
    ret_ext = subprocess.run(ext_cmd)

    if tar_path.exists():
        print(f"🧹 Deleting archive {tar_path.name} to free disk space immediately...")
        tar_path.unlink()

    if ret_ext.returncode != 0:
        print(f"❌ Failed to extract {split_name}!")
        return False

    free_after = check_disk_free_gb()
    print(f"✅ Extracted {split_name} successfully! Current free disk space: {free_after:.1f} GB")
    return True

def scan_and_build_manifest(split_dirs, output_manifest_name):
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    all_samples = []
    total_audio_seconds = 0.0

    print(f"\n📑 Scanning directories for manifest {output_manifest_name}...")
    for s_name in split_dirs:
        s_path = RAW_DIR / s_name
        if not s_path.exists():
            print(f"⚠️ Directory {s_path} does not exist, skipping.")
            continue
        trans_files = list(s_path.glob("*/*/*.trans.txt"))
        print(f"   • {s_name}: found {len(trans_files)} transcript files.")
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

    all_samples.sort(key=lambda s: s["id"])
    total_hours = total_audio_seconds / 3600.0
    out_file = MANIFEST_DIR / output_manifest_name
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2)

    print(f"🎉 Generated {out_file}: {len(all_samples):,} utterances ({total_hours:.2f} hours)")
    return out_file

def main():
    print("=" * 65)
    print("🚀 SAFE SEQUENTIAL LIBRISPEECH 960h DOWNLOADER & EXTRACTOR")
    print("=" * 65)

    for split_name, url in SPLITS:
        ok = download_and_extract_split(split_name, url)
        if not ok:
            print(f"Aborting downstream steps due to failure in {split_name}.")
            break

    # Build 460h manifest (clean-100 + clean-360)
    scan_and_build_manifest(["train-clean-100", "train-clean-360"], "librispeech_train_460h.json")

    # Build full 960h manifest (clean-100 + clean-360 + other-500)
    scan_and_build_manifest(["train-clean-100", "train-clean-360", "train-other-500"], "librispeech_train_960h.json")

if __name__ == "__main__":
    main()
