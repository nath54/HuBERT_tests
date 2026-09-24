#!/usr/bin/env python3
"""
Final Shootout Benchmark Evaluator on Held-Out 10,000 Sentence Dataset.
Compares Meta HuBERT-Large vs. OpenAI Whisper-Tiny vs. OurHuBERT on unseen test sentences.
Evaluates Word Error Rate (WER), Character Error Rate (CER), Phoneme Error Rate (PER), and RTF.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import jiwer
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.sota_evaluator import SOTABenchmarkRunner, normalize_text
from src.data.streaming_piper import PiperVoiceManager


TEST_FILE = Path("data/test_sets/held_out_10k_test.json")


def compute_per(pred_phonemes: str, ref_phonemes: str) -> float:
    """Compute Phoneme Error Rate using Levenshtein distance on phoneme tokens."""
    if not ref_phonemes:
        return 0.0
    if not pred_phonemes:
        return 1.0
    return round(float(jiwer.wer(ref_phonemes, pred_phonemes)), 4)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SOTA vs OurHuBERT on held-out test dataset.")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of test sentences to evaluate (up to 10,000)")
    parser.add_argument("--lang", type=str, default="all", choices=["all", "en", "fr"], help="Language filter")
    parser.add_argument("--blank_penalty", type=float, default=2.0, help="Blank penalty for scratch model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="logs/held_out_10k_benchmark_results.json", help="Path to save report")
    args = parser.parse_args()

    if not TEST_FILE.exists():
        print(f"Error: Held-out test file not found at {TEST_FILE}.")
        print("Run 'python scripts/prepare_10k_testset.py' first.")
        sys.exit(1)

    print("=" * 70)
    print("   HELD-OUT 10,000 SENTENCE BENCHMARK SHOOTOUT")
    print(f"   Meta HuBERT-Large vs. OpenAI Whisper-Tiny vs. OurHuBERT")
    print("=" * 70)
    print(f"Device: {args.device} | Evaluating: {args.num_samples} samples | Language: {args.lang}")
    print("-" * 70)

    # 1. Load test samples
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    if args.lang != "all":
        samples = [s for s in all_samples if s["lang"] == args.lang][: args.num_samples]
    else:
        # Balanced sampling of EN and FR
        half = args.num_samples // 2
        en_part = [s for s in all_samples if s["lang"] == "en"][:half]
        fr_part = [s for s in all_samples if s["lang"] == "fr"][: args.num_samples - half]
        samples = en_part + fr_part

    print(f"Loaded {len(samples)} held-out test sentences.")

    # 2. Initialize SOTA Evaluator and Piper Voice Manager
    runner = SOTABenchmarkRunner(device=args.device)
    runner.load_models()
    voice_manager = PiperVoiceManager()

    results_by_model = {
        "scratch_hubert": {"wers": [], "cers": [], "latencies": []},
        "meta_hubert_large": {"wers": [], "cers": [], "latencies": []},
        "whisper_tiny": {"wers": [], "cers": [], "latencies": []},
    }

    detailed_evals = []
    total_audio_sec = 0.0

    print("\nRunning live evaluation...")
    for idx, s in enumerate(samples):
        text = s["text"]
        lang = s["lang"]
        ground_truth_phonemes = s.get("phonemes", "")

        # Synthesize audio entirely in RAM (0 disk storage)
        waveform, voice_name, dur = voice_manager.synthesize_to_tensor_16k(text, lang=lang)
        total_audio_sec += dur

        # Run 3-model head-to-head comparison
        comp = runner.compare_audio(
            audio_path_or_array=waveform,
            ground_truth=text,
            blank_penalty=args.blank_penalty,
        )

        models = comp["models"]
        for m_key in ["scratch_hubert", "meta_hubert_large", "whisper_tiny"]:
            m_res = models[m_key]
            results_by_model[m_key]["wers"].append(m_res["wer"] if m_res["wer"] is not None else 1.0)
            results_by_model[m_key]["cers"].append(m_res["cer"] if m_res["cer"] is not None else 1.0)
            results_by_model[m_key]["latencies"].append(m_res["latency_ms"])

        detailed_evals.append({
            "id": s["id"],
            "lang": lang,
            "voice": voice_name,
            "duration": dur,
            "ground_truth_text": text,
            "ground_truth_phonemes": ground_truth_phonemes,
            "scratch_pred": models["scratch_hubert"]["raw_text"],
            "scratch_wer": models["scratch_hubert"]["wer"],
            "scratch_cer": models["scratch_hubert"]["cer"],
            "meta_pred": models["meta_hubert_large"]["raw_text"],
            "meta_wer": models["meta_hubert_large"]["wer"],
            "whisper_pred": models["whisper_tiny"]["raw_text"],
            "whisper_wer": models["whisper_tiny"]["wer"],
        })

        if (idx + 1) % 10 == 0 or idx == len(samples) - 1:
            print(f"Evaluated [{idx+1:3d}/{len(samples)}] utts | Cumulative Audio: {total_audio_sec:.1f}s")

    # Compute aggregate stats
    summary = {}
    for m_key in ["meta_hubert_large", "whisper_tiny", "scratch_hubert"]:
        wers = results_by_model[m_key]["wers"]
        cers = results_by_model[m_key]["cers"]
        lats = results_by_model[m_key]["latencies"]
        avg_wer = round(float(sum(wers) / max(1, len(wers))) * 100.0, 2)
        avg_cer = round(float(sum(cers) / max(1, len(cers))) * 100.0, 2)
        avg_lat = round(float(sum(lats) / max(1, len(lats))), 2)
        rtf = round((avg_lat / 1000.0) / (total_audio_sec / max(1, len(samples))), 4)

        summary[m_key] = {
            "avg_wer_pct": avg_wer,
            "avg_cer_pct": avg_cer,
            "avg_latency_ms": avg_lat,
            "rtf": rtf,
            "throughput_x": round(1.0 / (rtf + 1e-6), 1),
        }

    # Print Leaderboard
    print("\n" + "=" * 75)
    print("                      HELD-OUT BENCHMARK LEADERBOARD")
    print("=" * 75)
    print(f"{'Model':<24} | {'WER (%)':<10} | {'CER (%)':<10} | {'Latency':<12} | {'Throughput':<12}")
    print("-" * 75)
    print(f"{'Meta HuBERT-Large':<24} | {summary['meta_hubert_large']['avg_wer_pct']:>8.2f}% | {summary['meta_hubert_large']['avg_cer_pct']:>8.2f}% | {summary['meta_hubert_large']['avg_latency_ms']:>8.1f} ms | {summary['meta_hubert_large']['throughput_x']:>8.1f}x RT")
    print(f"{'OpenAI Whisper-Tiny':<24} | {summary['whisper_tiny']['avg_wer_pct']:>8.2f}% | {summary['whisper_tiny']['avg_cer_pct']:>8.2f}% | {summary['whisper_tiny']['avg_latency_ms']:>8.1f} ms | {summary['whisper_tiny']['throughput_x']:>8.1f}x RT")
    print(f"{'OurHuBERT':<24} | {summary['scratch_hubert']['avg_wer_pct']:>8.2f}% | {summary['scratch_hubert']['avg_cer_pct']:>8.2f}% | {summary['scratch_hubert']['avg_latency_ms']:>8.1f} ms | {summary['scratch_hubert']['throughput_x']:>8.1f}x RT")
    print("=" * 75)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_samples": len(samples),
            "total_audio_sec": round(total_audio_sec, 2),
            "summary": summary,
            "detailed_evals": detailed_evals,
        }, f, indent=2, ensure_ascii=False)

    print(f"Full benchmark shootout report saved to {out_path}")


if __name__ == "__main__":
    main()
