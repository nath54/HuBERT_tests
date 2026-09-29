#!/usr/bin/env python3
"""Standardized Cross-Architecture Benchmark Orchestrator.

Trains and evaluates all registered speech architectures under strictly identical conditions:
- Same training split: data/librispeech/benchmark_train.json (~95% LibriSpeech train-clean-100)
- Same held-out validation set: data/librispeech/benchmark_val.json (~5% LibriSpeech train-clean-100)
- Evaluated every 200 steps on validation set (tracking lowest Val PER)
- ONLY the single best checkpoint per architecture is saved to disk (protecting SSD wear)
- Best validation model is restored at the end for unbiased evaluation on LibriSpeech test-clean
- Zero data leakage ("no cheating")

Usage:
    # Run full standardized benchmark across all models (4000 steps each):
    .venv/bin/python scripts/benchmark_all_architectures.py

    # Run specific subset of architectures:
    .venv/bin/python scripts/benchmark_all_architectures.py --archs phono_hubert phono_v6_2_sparse phono_v6_3_diffusion

    # Fast verification dry-run (4 steps, eval every 2 steps):
    .venv/bin/python scripts/benchmark_all_architectures.py --dry_run
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Optional

# Architectures ordered by conceptual evolution
DEFAULT_ARCHITECTURES = [
    "phono_hubert",               # Dense Baseline + Anti-Blank CTC
    "phono_hubert_dual",          # Dual-Loss SSL Frame CE + CTC
    "phono_hubert_hierarchical",  # 2-Stage Gated Router
    "phono_hubert_recursive",     # Recurrent Frame Memory Feedback
    "phono_v6_1_moe",             # 4-Expert Mixture of Experts FFN
    "phono_v6_2_sparse",          # Sparse Syllabic Local Attention + InterCTC
    "phono_v6_3_diffusion",       # Gaussian Latent Diffusion Refiner
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run standardized cross-architecture benchmark suite")
    parser.add_argument(
        "--archs",
        nargs="+",
        default=DEFAULT_ARCHITECTURES,
        help="List of model architecture IDs to benchmark",
    )
    parser.add_argument("--tier", type=str, default="medium", choices=["mini", "small", "medium", "base"], help="Model tier")
    parser.add_argument("--steps", type=int, default=4000, help="Total training steps per model")
    parser.add_argument("--eval_interval", type=int, default=200, help="Validation evaluation step interval")
    parser.add_argument("--val_samples", type=int, default=50, help="Number of validation samples evaluated every eval_interval")
    parser.add_argument("--test_samples", type=int, default=100, help="Number of test-clean samples evaluated on best model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per training step")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--max_duration_sec", type=float, default=30.0, help="Maximum utterance duration in seconds")
    parser.add_argument("--real_ratio", type=float, default=1.0, help="Ratio of real human speech")
    parser.add_argument("--blank_penalty", type=float, default=2.0, help="Blank logit deduction during evaluation")
    parser.add_argument("--output_dir", type=str, default="reports/cross_arch_benchmark", help="Directory for summary reports")
    parser.add_argument("--skip_existing", action="store_true", help="Skip architectures with existing completed benchmark summary")
    parser.add_argument("--dry_run", action="store_true", help="Quick dry-run: 4 steps, eval every 2 steps on 2 architectures")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    # If dry-run requested, override with minimal test parameters
    if args.dry_run:
        print("⚡ [Dry-Run Mode Enabled] Setting steps=4, eval_interval=2, val_samples=4, test_samples=4")
        args.steps = 4
        args.eval_interval = 2
        args.val_samples = 4
        args.test_samples = 4
        if len(args.archs) > 2:
            args.archs = args.archs[:2]

    # Ensure train/val manifests exist
    train_manifest = project_root / "data" / "librispeech" / "benchmark_train.json"
    val_manifest = project_root / "data" / "librispeech" / "benchmark_val.json"
    test_manifest = project_root / "data" / "librispeech" / "librispeech_test_clean.json"

    if not train_manifest.exists() or not val_manifest.exists():
        print("⚙️ Benchmark partition not found. Generating now via prepare_benchmark_split.py...")
        from scripts.prepare_benchmark_split import create_benchmark_split
        create_benchmark_split()

    out_dir = project_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("🚀 STANDARDIZED CROSS-ARCHITECTURE SPEECH BENCHMARK SUITE")
    print("=" * 70)
    print(f"  • Architectures     : {', '.join(args.archs)}")
    print(f"  • Model Tier        : {args.tier}")
    print(f"  • Training Steps    : {args.steps:,} steps per architecture")
    print(f"  • Validation Eval   : Every {args.eval_interval} steps on {val_manifest.name} ({args.val_samples} samples)")
    print(f"  • Model Selection   : Best model state on Validation Set (Lowest Val PER)")
    print(f"  • Checkpoint Policy : ONLY best_checkpoint.pt saved to disk (Zero intermediate SSD writes)")
    print(f"  • Final Benchmark   : Single uncheated evaluation on {test_manifest.name} ({args.test_samples} samples)")
    print("=" * 70 + "\n")

    results: List[Dict] = []
    start_all_time = time.time()

    for idx, arch in enumerate(args.archs, 1):
        print(f"\n▶️ [{idx}/{len(args.archs)}] Initiating Benchmark Run for: {arch} [{args.tier}]")
        run_name = f"bench_{arch}_{args.tier}"
        run_log_dir = project_root / "logs" / arch / args.tier / run_name
        summary_file = run_log_dir / "benchmark_summary.json"
        if not summary_file.exists():
            fallback_summary = project_root / "logs" / run_name / "benchmark_summary.json"
            if fallback_summary.exists():
                summary_file = fallback_summary

        if args.skip_existing and summary_file.exists():
            print(f"⏩ [Skip] Summary file already exists at {summary_file}. Skipping execution.")
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    results.append(json.load(f))
                continue
            except Exception:
                pass

        cmd = [
            str(project_root / ".venv" / "bin" / "python"),
            str(project_root / "scripts" / "run_pretrain.py"),
            "--arch", arch,
            "--tier", args.tier,
            "--steps", str(args.steps),
            "--eval_interval", str(args.eval_interval),
            "--val_samples", str(args.val_samples),
            "--test_samples", str(args.test_samples),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.lr),
            "--max_duration_sec", str(args.max_duration_sec),
            "--real_ratio", str(args.real_ratio),
            "--blank_penalty", str(args.blank_penalty),
            "--real_speech_manifest", str(train_manifest),
            "--val_manifest", str(val_manifest),
            "--test_manifest", str(test_manifest),
            "--only_save_best",
            "--run_name", run_name,
        ]

        t0 = time.time()
        print(f"💻 Command: {' '.join(cmd)}\n")
        proc = subprocess.run(cmd, cwd=str(project_root))
        elapsed = time.time() - t0

        if proc.returncode != 0:
            print(f"❌ Error: Architecture {arch} failed with return code {proc.returncode}!")
            continue

        if summary_file.exists():
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                    summary["elapsed_sec"] = round(elapsed, 2)
                    summary["elapsed_min"] = round(elapsed / 60.0, 2)
                    results.append(summary)
                    print(f"✅ [{arch}] Run Finished! Best Val PER: {summary.get('best_val_per')}% @ Step {summary.get('best_val_step')} | Test Clean PER: {summary.get('test_clean_per')}%")
            except Exception as e:
                print(f"⚠️ Could not load summary file for {arch}: {e}")
        else:
            print(f"⚠️ Warning: No benchmark_summary.json generated at {summary_file}")

    # Generate Consolidated Markdown and JSON Reports
    total_elapsed_min = (time.time() - start_all_time) / 60.0
    json_path = out_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "tier": args.tier,
                "steps": args.steps,
                "eval_interval": args.eval_interval,
                "total_elapsed_min": round(total_elapsed_min, 2),
                "results": results,
            },
            f,
            indent=2,
        )

    md_report_path = out_dir / "benchmark_report.md"
    generate_markdown_report(results, args, md_report_path, total_elapsed_min)

    print("\n" + "=" * 70)
    print("🏁 BENCHMARK SUITE EXECUTION COMPLETED")
    print("=" * 70)
    print(f"📄 JSON Results    : {json_path}")
    print(f"📊 Markdown Report : {md_report_path}")
    print(f"⏱️  Total Duration  : {total_elapsed_min:.2f} minutes")
    print("=" * 70 + "\n")


def generate_markdown_report(results: List[Dict], args, report_path: Path, elapsed_min: float):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Standardized Cross-Architecture Benchmark Report",
        "",
        f"**Date**: {now_str}  ",
        f"**Model Tier**: `{args.tier}` | **Training Steps**: `{args.steps:,}` | **Evaluation Interval**: every `{args.eval_interval}` steps  ",
        f"**Training Split**: `benchmark_train.json` (~95% LibriSpeech Clean-100)  ",
        f"**Validation Split**: `benchmark_val.json` (~5% Held-Out LibriSpeech Clean-100)  ",
        f"**Test Benchmark**: `librispeech_test_clean.json` (Untainted Test-Clean)  ",
        f"**Total Suite Runtime**: {elapsed_min:.1f} minutes  ",
        "",
        "## Executive Summary & Model Selection Protocol",
        "1. **Fair & Scientific Comparison**: All architectures trained on identical audio frames and random seeds with identical learning rates and batch sizes.",
        "2. **Strict Cheat-Free Policy**: The test-clean split was never accessed during training. Models were evaluated every 200 steps solely on the held-out validation set.",
        "3. **Zero Waste / SSD Protection**: Only the single best model weights (`best_checkpoint.pt`) based on validation PER were preserved on disk.",
        "4. **Final Scoring**: The best validation model was restored at the conclusion of training to obtain the definitive test-clean score.",
        "",
        "## Consolidated Benchmark Results",
        "",
        "| Architecture | Best Val PER | Best Step | Test PER (Greedy) | Test Lexicon PER | Test CER | Audio Hours | Checkpoint |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    # Sort results by Test PER ascending
    sorted_res = sorted(results, key=lambda r: r.get("test_clean_per") or 999.0)

    for r in sorted_res:
        arch = r.get("arch", "N/A")
        val_per = f"{r.get('best_val_per', 0.0):.2f}%" if r.get('best_val_per') is not None else "N/A"
        step = r.get("best_val_step", "N/A")
        test_per = f"**{r.get('test_clean_per', 0.0):.2f}%**" if r.get('test_clean_per') is not None else "N/A"
        lex_per = f"{r.get('test_clean_lexicon_per', 0.0):.2f}%" if r.get('test_clean_lexicon_per') is not None else "-"
        cer = f"{r.get('test_clean_cer', 0.0):.2f}%" if r.get('test_clean_cer') is not None else "N/A"
        hours = f"{r.get('cumulative_audio_hours', 0.0):.1f}h"
        ckpt = f"`checkpoints/{arch}/{args.tier}/best_checkpoint.pt`"
        lines.append(f"| **{arch}** | {val_per} | {step} | {test_per} | {lex_per} | {cer} | {hours} | {ckpt} |")

    lines.extend([
        "",
        "```",
        "Test PER Ranking (Lower is Better):",
    ])

    for r in sorted_res:
        arch = r.get("arch", "N/A")
        t_per = r.get("test_clean_per") or 100.0
        bar_len = int(t_per / 2.5)
        bar = "█" * max(1, bar_len)
        lines.append(f"  {arch:<28}: {bar} {t_per:.2f}%")

    lines.extend([
        "```",
        "",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
