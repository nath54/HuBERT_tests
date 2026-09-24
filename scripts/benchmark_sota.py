#!/usr/bin/env python3
"""CLI tool to benchmark Scratch HuBERT against Meta HuBERT and OpenAI Whisper on standard benchmarks."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.sota_evaluator import SOTABenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="Evaluate HuBERT against SOTA models on standard benchmarks.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/librispeech/librispeech_test_clean.json",
        help="Path to LibriSpeech manifest (test_clean, val, all)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=25,
        help="Number of samples to evaluate (0 for full set)",
    )
    parser.add_argument(
        "--blank-penalty",
        type=float,
        default=0.0,
        help="Blank token penalty for CTC decoding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device (cuda or cpu)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("      🏆 SOTA BENCHMARK EVALUATOR: HU-BERT vs META vs WHISPER")
    print("=" * 80)
    print(f"Dataset Manifest: {args.manifest}")
    print(f"Samples:          {args.samples}")
    print(f"Device:           {args.device}")
    print("-" * 80)

    runner = SOTABenchmarkRunner(device=args.device)
    report = runner.evaluate_benchmark(
        manifest_path=args.manifest,
        max_samples=args.samples,
        blank_penalty=args.blank_penalty,
    )

    print("\n" + "=" * 80)
    print("                      📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(
        f"{'Model':<25} | {'Params':<8} | {'Train Audio':<15} | {'WER (%)':<8} | {'CER (%)':<8} | {'Latency':<9} | {'RTF':<8} | {'Speedup'}"
    )
    print("-" * 105)

    for k, v in report["models_summary"].items():
        print(
            f"{v['name']:<25} | {v['parameters']:<8} | {v['training_data'][:15]:<15} | {v['avg_wer']:<8.2f} | {v['avg_cer']:<8.2f} | {v['avg_latency_ms']:<7.1f}ms | {v['rtf']:<8.4f} | {v['throughput_x']:.1f}x"
        )
    print("=" * 105)

    print("\n📝 Sample Side-by-Side Predictions:")
    for i, s in enumerate(report["detailed_samples"][:3]):
        print(f"\n--- [Sample #{i+1}: {s['id']} ({s['duration']:.2f}s)] ---")
        print(f"  Target:  {s['ground_truth']}")
        print(f"  Meta:    {s['meta_pred']}  (WER: {s['meta_wer']*100:.1f}%)")
        print(f"  Whisper: {s['whisper_pred']}  (WER: {s['whisper_wer']*100:.1f}%)")
        print(f"  Scratch: {s['scratch_pred'] or '[BLANK]'}  (WER: {s['scratch_wer']*100:.1f}%)")

    print("\nSaved full results to logs/sota_benchmark_results.json\n")


if __name__ == "__main__":
    main()
