#!/usr/bin/env python3
"""Automated Spoken Number & Roman Numeral Disambiguation Script.

Scans an audio dataset manifest for numbers and Roman numerals,
generates phonetic candidates (raw, cardinal, ordinal, gendered, years),
and uses acoustic model CTC forced alignment to select the candidate
that precisely matches what the speaker uttered in the audio.

Saves the patched manifest with clean, acoustically-verified transcripts.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.phoneme_tokenizer import PhonemeTokenizer
from src.data.spoken_number_patcher import SpokenNumberPatcher, find_number_spans
from src.data.target_extractors import PhonemeTargetExtractor
from src.models.phono_v7_3_speech_model import PhonoV73AcousticBackbone
from src.models.phono_variants import PhonoV64GatedDiffusionConfig
from scripts.train_phono_v7_5_acoustic import load_acoustic_backbone_weights


def load_model(ckpt_path: str, device: torch.device) -> PhonoV73AcousticBackbone:
    print(f"📦 Loading acoustic model from {ckpt_path} on {device}...")
    cfg = PhonoV64GatedDiffusionConfig()
    cfg.vocab_size = 112
    model = PhonoV73AcousticBackbone(cfg).to(device)

    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("acoustic_state_dict", ckpt.get("model_state_dict", ckpt))
        load_acoustic_backbone_weights(model, sd)
        print(f"✅ Loaded checkpoint successfully.")
    else:
        print(f"⚠️ Checkpoint not found at {ckpt_path}! Initializing default model.")

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Patch spoken numbers and Roman numerals via CTC forced alignment.")
    parser.add_argument("--input_manifest", type=str, default="data/multilingual_val_clean.json", help="Path to input manifest JSON.")
    parser.add_argument("--output_manifest", type=str, default="data/multilingual_val_patched.json", help="Path to output patched manifest JSON.")
    parser.add_argument("--model_ckpt", type=str, default="checkpoints/phono_v7_5_acoustic/latest_acoustic.pt", help="Path to model checkpoint.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference ('cpu' or 'cuda').")
    parser.add_argument("--min_gain", type=float, default=0.02, help="Minimum CTC loss reduction to accept a patch.")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to scan (for testing).")
    parser.add_argument("--num_threads", type=int, default=4, help="PyTorch CPU threads.")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.num_threads)

    print("=" * 80)
    print("🎙️ AUTOMATED SPOKEN NUMBER DISAMBIGUATION & FORCED ALIGNMENT PATCHER")
    print("=" * 80)
    print(f"  • Input Manifest : {args.input_manifest}")
    print(f"  • Output Manifest: {args.output_manifest}")
    print(f"  • Model CKPT     : {args.model_ckpt}")
    print(f"  • Device         : {device}")
    print(f"  • Min Loss Gain  : {args.min_gain:.4f}")
    print("=" * 80)

    # 1. Load manifest
    with open(args.input_manifest, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.max_samples is not None:
        data = data[: args.max_samples]

    print(f"\n📂 Manifest loaded: {len(data)} total utterances.")

    # 2. Identify candidates
    flagged_indices = []
    for idx, item in enumerate(data):
        txt = item.get("transcript", "")
        lang = item.get("lang", "en")
        spans = find_number_spans(txt, lang)
        if spans:
            flagged_indices.append((idx, spans))

    print(f"🔍 Found {len(flagged_indices)} utterances containing potential numbers or Roman numerals ({len(flagged_indices)/max(1, len(data))*100:.2f}%).")

    if not flagged_indices:
        print("✅ No numbers found in manifest! Copying manifest unchanged.")
        with open(args.output_manifest, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return

    # 3. Load model and patcher
    model = load_model(args.model_ckpt, device)
    tokenizer = PhonemeTokenizer()
    extractor = PhonemeTargetExtractor(tokenizer)
    patcher = SpokenNumberPatcher(model, tokenizer, extractor, device=device)

    patched_count = 0
    total_gain = 0.0
    t0 = time.time()

    print("\n⚡ Beginning CTC Forced Alignment Evaluation...")
    print("-" * 80)

    for step, (idx, spans) in enumerate(flagged_indices, 1):
        item = data[idx]
        uid = item.get("id", str(idx))
        lang = item.get("lang", "en")
        txt = item.get("transcript", "")
        audio_path = item.get("audio_path", "")

        res = patcher.patch_utterance(audio_path, txt, lang, gain_threshold=args.min_gain)

        if res["patched"]:
            patched_count += 1
            gain = res["loss_gain"]
            total_gain += gain
            data[idx]["transcript"] = res["patched_text"]
            data[idx]["original_transcript"] = txt
            data[idx]["patch_info"] = {
                "winner_label": res["best_label"],
                "raw_loss": res["raw_loss"],
                "best_loss": res["best_loss"],
                "gain": gain,
            }

            print(f"[{step}/{len(flagged_indices)}] 🎯 PATCHED [{lang.upper()}] Utterance {uid}")
            print(f"   • Original : \"{txt}\"")
            print(f"   • Patched  : \"{res['patched_text']}\"")
            print(f"   • Selection: {res['best_label']} (CTC Loss: {res['raw_loss']:.4f} -> {res['best_loss']:.4f}, Δ: -{gain:.4f})")
            print()
        else:
            best_lbl = res.get("best_label", "none")
            raw_loss = res.get("raw_loss", 0.0)
            best_loss = res.get("best_loss", 0.0)
            print(f"[{step}/{len(flagged_indices)}] ℹ️ Kept Raw [{lang.upper()}] Utterance {uid} (winner: {best_lbl}, raw_loss: {raw_loss:.4f}, best_loss: {best_loss:.4f})")

    elapsed = time.time() - t0
    avg_gain = total_gain / max(1, patched_count)

    print("=" * 80)
    print(f"📊 SUMMARY OF NUMBER PATCHING:")
    print(f"  • Examined Flagged : {len(flagged_indices)}")
    print(f"  • Successfully Patched: {patched_count} / {len(flagged_indices)} ({patched_count/max(1, len(flagged_indices))*100:.1f}%)")
    print(f"  • Average CTC Loss Gain: -{avg_gain:.4f}")
    print(f"  • Time Elapsed     : {elapsed:.2f} s ({elapsed/max(1, len(flagged_indices)):.2f} s/utt)")

    # Save output
    out_dir = Path(args.output_manifest).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_manifest, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Patched manifest saved to: {args.output_manifest}")
    print("=" * 80)


if __name__ == "__main__":
    main()
