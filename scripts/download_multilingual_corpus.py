"""High-speed Multilingual Corpus Downloader for Phoneme-to-Text Pretraining.

Targets 9 languages: English, French, Spanish, German, Italian, Arabic, Chinese, Japanese, Korean.
Storage location: /media/hdd/Datasets/multilingual_text/raw/ (protects system SSD).
Pillars:
1. OpenSubtitles (Spoken conversational dialogue & contractions)
2. Tatoeba (Daily conversational pairs & human translations)
3. Wikipedia (Factual knowledge & named entities)
4. OPUS Books (Narrative literature & complex syntax)
"""

import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple


DEST_DIR = Path("/media/hdd/Datasets/multilingual_text/raw")

DOWNLOAD_MANIFEST: List[Tuple[str, str, str]] = [
    # --- Pillar 1: OpenSubtitles (Conversational Dialogue) ---
    ("opensub_en.txt.gz", "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/en.txt.gz", "en"),
    ("opensub_fr.txt.gz", "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/fr.txt.gz", "fr"),
    ("opensub_es.txt.gz", "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/es.txt.gz", "es"),
    ("opensub_de.txt.gz", "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/de.txt.gz", "de"),
    ("opensub_it.txt.gz", "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/it.txt.gz", "it"),
    ("opensub_ar.txt.gz", "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/ar.txt.gz", "ar"),
    ("opensub_zh.txt.gz", "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/zh_cn.txt.gz", "zh"),
    ("opensub_ja.txt.gz", "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/ja.txt.gz", "ja"),
    ("opensub_ko.txt.gz", "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/ko.txt.gz", "ko"),

    # --- Pillar 2: Tatoeba (Conversational Pairs & Daily Speech) ---
    ("tatoeba_en.txt.gz", "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/mono/en.txt.gz", "en"),
    ("tatoeba_fr.txt.gz", "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/mono/fr.txt.gz", "fr"),
    ("tatoeba_es.txt.gz", "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/mono/es.txt.gz", "es"),
    ("tatoeba_de.txt.gz", "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/mono/de.txt.gz", "de"),
    ("tatoeba_it.txt.gz", "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/mono/it.txt.gz", "it"),
    ("tatoeba_ar.txt.gz", "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/mono/ar.txt.gz", "ar"),
    ("tatoeba_zh.txt.gz", "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/mono/cmn.txt.gz", "zh"),
    ("tatoeba_ja.txt.gz", "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/mono/ja.txt.gz", "ja"),
    ("tatoeba_ko.txt.gz", "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/mono/ko.txt.gz", "ko"),

    # --- Pillar 3: Wikipedia (Encyclopedic & Technical) ---
    ("wiki_en.txt.gz", "https://object.pouta.csc.fi/OPUS-Wikipedia/v1.0/mono/en.txt.gz", "en"),
    ("wiki_fr.txt.gz", "https://object.pouta.csc.fi/OPUS-Wikipedia/v1.0/mono/fr.txt.gz", "fr"),
    ("wiki_es.txt.gz", "https://object.pouta.csc.fi/OPUS-Wikipedia/v1.0/mono/es.txt.gz", "es"),
    ("wiki_de.txt.gz", "https://object.pouta.csc.fi/OPUS-Wikipedia/v1.0/mono/de.txt.gz", "de"),
    ("wiki_it.txt.gz", "https://object.pouta.csc.fi/OPUS-Wikipedia/v1.0/mono/it.txt.gz", "it"),
    ("wiki_ar.txt.gz", "https://object.pouta.csc.fi/OPUS-Wikipedia/v1.0/mono/ar.txt.gz", "ar"),

    # --- Pillar 4: Books (Literature & Narrative Prose) ---
    ("books_fr.txt.gz", "https://object.pouta.csc.fi/OPUS-Books/v1/mono/fr.txt.gz", "fr"),
    ("books_es.txt.gz", "https://object.pouta.csc.fi/OPUS-Books/v1/mono/es.txt.gz", "es"),
    ("books_de.txt.gz", "https://object.pouta.csc.fi/OPUS-Books/v1/mono/de.txt.gz", "de"),
    ("books_it.txt.gz", "https://object.pouta.csc.fi/OPUS-Books/v1/mono/it.txt.gz", "it"),
]


def download_file(filename: str, url: str, target_dir: Path) -> bool:
    dest_path = target_dir / filename
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"⏩ [Already Exists] {filename} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
        return True

    print(f"\n⬇️ Downloading: {filename} from {url}...")
    temp_path = target_dir / f"{filename}.part"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp, open(temp_path, "wb") as out_f:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            t0 = time.time()
            chunk_size = 4 * 1024 * 1024  # 4 MB chunks for fast sequential HDD writes

            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)

                elapsed = max(1e-5, time.time() - t0)
                speed_mb = (downloaded / (1024 * 1024)) / elapsed
                if total_size > 0:
                    pct = (downloaded / total_size) * 100.0
                    print(
                        f"\r   Progress: {downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB "
                        f"({pct:.1f}%) | {speed_mb:.2f} MB/s",
                        end="",
                        flush=True,
                    )
                else:
                    print(f"\r   Downloaded: {downloaded / (1024*1024):.1f} MB | {speed_mb:.2f} MB/s", end="", flush=True)

        temp_path.rename(dest_path)
        print(f"\n✅ Completed: {filename} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
        return True
    except Exception as e:
        print(f"\n❌ Error downloading {filename}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 75)
    print("🌍 MULTILINGUAL PRETRAINING CORPUS DOWNLOADER")
    print(f"Target Directory: {DEST_DIR} (HDD)")
    print(f"Total Targets: {len(DOWNLOAD_MANIFEST)} files across 9 languages")
    print("=" * 75)

    success_count = 0
    t_start = time.time()

    for filename, url, lang in DOWNLOAD_MANIFEST:
        ok = download_file(filename, url, DEST_DIR)
        if ok:
            success_count += 1

    total_time = (time.time() - t_start) / 60
    total_bytes = sum(f.stat().st_size for f in DEST_DIR.glob("*.gz"))
    print("\n" + "=" * 75)
    print(f"🎉 DOWNLOAD COMPLETE! {success_count}/{len(DOWNLOAD_MANIFEST)} files downloaded successfully.")
    print(f"Total Downloaded Size: {total_bytes / (1024*1024*1024):.2f} GB on {DEST_DIR}")
    print(f"Elapsed Time: {total_time:.1f} minutes")
    print("=" * 75)


if __name__ == "__main__":
    main()
