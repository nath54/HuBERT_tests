#!/usr/bin/env bash
# Resumable download script for Multilingual LibriSpeech (OpenSLR 94)
# Saves directly to secondary HDD: /media/hdd/Datasets/mls/

set -e

DEST_DIR="/media/hdd/Datasets/mls"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

LANGUAGES=(
    "italian"
    "spanish"
    "french"
)

echo "=========================================================="
echo "🚀 Starting Multilingual LibriSpeech (OpenSLR 94) Download"
echo "  Destination: $DEST_DIR"
echo "  Languages:   ${LANGUAGES[*]}"
echo "=========================================================="

for lang in "${LANGUAGES[@]}"; do
    FILE="mls_${lang}.tar.gz"
    if [ -f "$FILE" ]; then
        echo "⏭️  Already downloaded: $FILE (skipping)"
        continue
    fi
    URL="https://dl.fbaipublicfiles.com/mls/${FILE}"
    echo ""
    echo "⬇️  Downloading ${lang} (${FILE})..."
    wget -c --progress=dot:giga "$URL" -O "${FILE}.tmp"
    mv "${FILE}.tmp" "$FILE"
    echo "✅ Completed download: $FILE"
done

echo ""
echo "🎉 All Multilingual LibriSpeech datasets downloaded successfully!"
