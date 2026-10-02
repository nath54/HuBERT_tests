#!/usr/bin/env python3
"""Download Mozilla Common Voice datasets from Mozilla Data Collective (MDC).

Saves directly to secondary HDD: /media/hdd/Datasets/common_voice/
Uses API key from .mdc_token.txt or .mk_token.txt (never logged or printed).
"""

import os
import sys
from pathlib import Path
import datacollective

DATASETS = {
    "italian": {
        "id": "cmu5xagm300dyo1075qdhf20m",
        "name": "Common Voice Scripted Speech 27.0 - Italian",
        "url": "https://mozilladatacollective.com/datasets/cmu5xagm300dyo1075qdhf20m",
    },
    "french": {
        "id": "cmu5kub4600pkmh078n7n5ogl",
        "name": "Common Voice Scripted Speech 27.0 - French",
        "url": "https://mozilladatacollective.com/datasets/cmu5kub4600pkmh078n7n5ogl",
    },
    "spanish": {
        "id": "cmu5kitx700pemi073feoiqqu",
        "name": "Common Voice Scripted Speech 27.0 - Spanish",
        "url": "https://mozilladatacollective.com/datasets/cmu5kitx700pemi073feoiqqu",
    },
    "english": {
        "id": "cmu5jplf300nwmh07iqvk9leo",
        "name": "Common Voice Scripted Speech 27.0 - English",
        "url": "https://mozilladatacollective.com/datasets/cmu5jplf300nwmh07iqvk9leo",
    },
}

DEST_DIR = Path("/media/hdd/Datasets/common_voice")


def load_token() -> str:
    """Load MDC API key securely from local file."""
    for fn in [".mdc_token.txt", ".mk_token.txt"]:
        p = Path(__file__).resolve().parent.parent / fn
        if p.exists():
            token = p.read_text().strip()
            if token:
                return token
    raise FileNotFoundError("MDC token file (.mdc_token.txt or .mk_token.txt) not found.")


def main():
    token = load_token()
    os.environ["MDC_API_KEY"] = token
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🚀 Mozilla Common Voice (MDC) Downloader")
    print(f"  Destination: {DEST_DIR}")
    print("=" * 60)

    selected = sys.argv[1:] if len(sys.argv) > 1 else ["italian", "french", "spanish"]

    for lang in selected:
        info = DATASETS.get(lang.lower())
        if not info:
            print(f"⚠️  Unknown language: {lang}. Available: {list(DATASETS.keys())}")
            continue

        did = info["id"]
        name = info["name"]
        print(f"\n⬇️  Checking / Downloading {name} ({did})...")

        try:
            downloaded_path = datacollective.download_dataset(
                did,
                download_directory=str(DEST_DIR / lang),
                show_progress=True,
                overwrite_existing=False,
            )
            print(f"✅ Successfully downloaded {lang} to {downloaded_path}")
        except PermissionError:
            print(f"❌ Terms not yet accepted for {name}!")
            print(f"   Please open this URL in your browser and click 'Accept Terms':")
            print(f"   👉 {info['url']}")
        except Exception as e:
            print(f"❌ Failed to download {name}: {type(e).__name__}: {e}")

    print("\n🎉 Common Voice check complete!")


if __name__ == "__main__":
    main()
