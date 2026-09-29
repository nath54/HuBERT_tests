#!/usr/bin/env python3
"""Standardized Cross-Architecture Benchmark Orchestrator with Fault Tolerance.

Trains and evaluates all registered speech architectures under strictly identical conditions:
- Same training split: data/librispeech/benchmark_train.json (~95% LibriSpeech train-clean-100)
- Same held-out validation set: data/librispeech/benchmark_val.json (~5% LibriSpeech train-clean-100)
- Evaluated every 200 steps on validation set (tracking lowest Val PER)
- ONLY the single best checkpoint per architecture is saved to disk (protecting SSD wear)
- Best validation model is restored at the end for unbiased evaluation on LibriSpeech test-clean
- Fault-tolerant execution: If any model fails (e.g. CUDA OOM or runtime error), the failure
  and exact error message are recorded, GPU memory is reclaimed, and the runner automatically
  advances to the next architecture / variant / size without halting the entire benchmark!

Usage:
    # Run full benchmark across all models (medium tier, 4000 steps each):
    .venv/bin/python scripts/benchmark_all_architectures.py

    # Benchmark across multiple sizes/tiers:
    .venv/bin/python scripts/benchmark_all_architectures.py --tiers small medium base

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
from typing import Any, Dict, List, Optional

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
    parser.add_argument("--tier", type=str, default="medium", choices=["mini", "small", "medium", "base"], help="Model tier (default: medium)")
    parser.add_argument("--tiers", nargs="+", default=None, choices=["mini", "small", "medium", "base"], help="List of tiers to benchmark sequentially (e.g. --tiers small medium base)")
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


def reclaim_gpu_memory():
    """Attempt to reclaim GPU memory between runs."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    time.sleep(2)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    # Resolve target tiers
    tiers = args.tiers if args.tiers else [args.tier]

    # If dry-run requested, override with minimal test parameters
    if args.dry_run:
        print("⚡ [Dry-Run Mode Enabled] Setting steps=4, eval_interval=2, val_samples=4, test_samples=4")
        args.steps = 4
        args.eval_interval = 2
        args.val_samples = 4
        args.test_samples = 4
        tiers = [tiers[0]]
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

    total_runs = len(tiers) * len(args.archs)

    print("\n" + "=" * 70)
    print("🚀 STANDARDIZED CROSS-ARCHITECTURE SPEECH BENCHMARK SUITE")
    print("=" * 70)
    print(f"  • Architectures     : {', '.join(args.archs)}")
    print(f"  • Model Tiers/Sizes : {', '.join(tiers)}")
    print(f"  • Total Planned Runs: {total_runs} benchmark experiments")
    print(f"  • Training Steps    : {args.steps:,} steps per architecture")
    print(f"  • Validation Eval   : Every {args.eval_interval} steps on {val_manifest.name} ({args.val_samples} samples)")
    print(f"  • Model Selection   : Best model state on Validation Set (Lowest Val PER)")
    print(f"  • Checkpoint Policy : ONLY best_checkpoint.pt saved to disk (Zero intermediate SSD writes)")
    print(f"  • Fault Tolerance   : Failure / CUDA OOM recorded; auto-advances to next model")
    print(f"  • Final Benchmark   : Single uncheated evaluation on {test_manifest.name} ({args.test_samples} samples)")
    print("=" * 70 + "\n")

    results: List[Dict[str, Any]] = []
    start_all_time = time.time()
    run_idx = 0

    for tier in tiers:
        for arch in args.archs:
            run_idx += 1
            print(f"\n▶️ [{run_idx}/{total_runs}] Initiating Benchmark Run: {arch} [{tier}]")
            run_name = f"bench_{arch}_{tier}"
            run_log_dir = project_root / "logs" / arch / tier / run_name
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
                "--tier", tier,
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

            # Inspect run summary if generated by run_pretrain.py
            summary = None
            if summary_file.exists():
                try:
                    with open(summary_file, "r", encoding="utf-8") as f:
                        summary = json.load(f)
                except Exception:
                    summary = None

            run_failed = (proc.returncode != 0) or (summary is not None and summary.get("status") == "failed")

            if run_failed:
                # Classify the error type
                error_type = "Unknown Error"
                error_msg = f"Process exited with code {proc.returncode}"

                if summary and summary.get("status") == "failed":
                    error_type = summary.get("error_type", error_type)
                    error_msg = summary.get("error_message", error_msg)
                elif proc.returncode == 2:
                    error_type = "CUDA OOM (Out of Memory)"
                    error_msg = "PyTorch CUDA OutOfMemoryError encountered during execution"
                elif proc.returncode in (137, -9):
                    error_type = "Linux OOM Killer (SIGKILL)"
                    error_msg = "Process terminated by OS out-of-memory killer (RAM/VRAM limit exceeded)"

                failed_record = {
                    "arch": arch,
                    "tier": tier,
                    "status": "failed",
                    "error_type": error_type,
                    "error_message": error_msg,
                    "best_val_step": summary.get("best_val_step") if summary else None,
                    "best_val_per": summary.get("best_val_per") if summary else None,
                    "test_clean_per": None,
                    "test_clean_cer": None,
                    "test_clean_lexicon_per": None,
                    "test_clean_lexicon_wer": None,
                    "cumulative_audio_hours": summary.get("cumulative_audio_hours", 0.0) if summary else 0.0,
                    "elapsed_sec": round(elapsed, 2),
                    "elapsed_min": round(elapsed / 60.0, 2),
                }
                results.append(failed_record)

                print("\n" + "!" * 70)
                print(f"💥 [{arch} - {tier}] RUN FAILED: {error_type}")
                print(f"   Details: {error_msg}")
                print("   Recorded failure in benchmark report. Reclaiming GPU memory...")
                print("!" * 70 + "\n")

                reclaim_gpu_memory()
                print("⏭️  Advancing to next model architecture / size...\n")
                continue

            # Successful run
            if summary is not None:
                summary["tier"] = tier
                summary["elapsed_sec"] = round(elapsed, 2)
                summary["elapsed_min"] = round(elapsed / 60.0, 2)
                summary["status"] = "success"
                results.append(summary)
                print(f"✅ [{arch} - {tier}] Run Succeeded! Best Val PER: {summary.get('best_val_per')}% @ Step {summary.get('best_val_step')} | Test Clean PER: {summary.get('test_clean_per')}%")
            else:
                print(f"⚠️ Warning: Completed with code 0 but no benchmark_summary.json generated at {summary_file}")

            # Reclaim GPU memory before next run
            reclaim_gpu_memory()

    # Generate Consolidated Markdown and JSON Reports
    total_elapsed_min = (time.time() - start_all_time) / 60.0
    json_path = out_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "tiers": tiers,
                "steps": args.steps,
                "eval_interval": args.eval_interval,
                "total_elapsed_min": round(total_elapsed_min, 2),
                "total_runs": total_runs,
                "completed_runs": len([r for r in results if r.get("status") == "success"]),
                "failed_runs": len([r for r in results if r.get("status") == "failed"]),
                "results": results,
            },
            f,
            indent=2,
        )

    md_report_path = out_dir / "benchmark_report.md"
    generate_markdown_report(results, args, tiers, md_report_path, total_elapsed_min)

    print("\n" + "=" * 70)
    print("🏁 BENCHMARK SUITE EXECUTION COMPLETED")
    print("=" * 70)
    print(f"📄 JSON Results    : {json_path}")
    print(f"📊 Markdown Report : {md_report_path}")
    print(f"⏱️  Total Duration  : {total_elapsed_min:.2f} minutes")
    print("=" * 70 + "\n")


def generate_markdown_report(results: List[Dict], args, tiers: List[str], report_path: Path, elapsed_min: float):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    successful = [r for r in results if r.get("status") != "failed" and r.get("test_clean_per") is not None]
    failed = [r for r in results if r.get("status") == "failed"]

    lines = [
        "# Standardized Cross-Architecture Benchmark Report",
        "",
        f"**Date**: {now_str}  ",
        f"**Model Tiers**: `{', '.join(tiers)}` | **Training Steps**: `{args.steps:,}` | **Evaluation Interval**: every `{args.eval_interval}` steps  ",
        f"**Training Split**: `benchmark_train.json` (~95% LibriSpeech Clean-100)  ",
        f"**Validation Split**: `benchmark_val.json` (~5% Held-Out LibriSpeech Clean-100)  ",
        f"**Test Benchmark**: `librispeech_test_clean.json` (Untainted Test-Clean)  ",
        f"**Summary**: {len(successful)} Successful / {len(failed)} Failed / {len(results)} Total Runs  ",
        f"**Total Suite Runtime**: {elapsed_min:.1f} minutes  ",
        "",
        "## Executive Summary & Model Selection Protocol",
        "1. **Fair & Scientific Comparison**: All architectures trained on identical audio frames and random seeds with identical learning rates and batch sizes.",
        "2. **Strict Cheat-Free Policy**: The test-clean split was never accessed during training. Models were evaluated every 200 steps solely on the held-out validation set.",
        "3. **Zero Waste / SSD Protection**: Only the single best model weights (`best_checkpoint.pt`) based on validation PER were preserved on disk.",
        "4. **Fault Tolerance**: Any model encountering out-of-memory or errors is safely caught, documented with its exact error, and the suite automatically proceeds to the next model.",
        "5. **Final Scoring**: The best validation model was restored at the conclusion of training to obtain the definitive test-clean score.",
        "",
        "## Consolidated Benchmark Results",
        "",
        "| Architecture | Tier | Best Val PER | Best Step | Test PER (Greedy) | Test Lexicon PER | Test CER | Audio Hours | Status / Checkpoint |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    # Sort successful results by Test PER ascending, failed results at bottom
    sorted_successful = sorted(successful, key=lambda r: r.get("test_clean_per") or 999.0)

    for r in sorted_successful:
        arch = r.get("arch", "N/A")
        tier = r.get("tier", args.tier)
        val_per = f"{r.get('best_val_per', 0.0):.2f}%" if r.get('best_val_per') is not None else "N/A"
        step = r.get("best_val_step", "N/A")
        test_per = f"**{r.get('test_clean_per', 0.0):.2f}%**"
        lex_per = f"{r.get('test_clean_lexicon_per', 0.0):.2f}%" if r.get('test_clean_lexicon_per') is not None else "-"
        cer = f"{r.get('test_clean_cer', 0.0):.2f}%" if r.get('test_clean_cer') is not None else "N/A"
        hours = f"{r.get('cumulative_audio_hours', 0.0):.1f}h"
        ckpt = f"`checkpoints/{arch}/{tier}/best_checkpoint.pt`"
        lines.append(f"| **{arch}** | {tier} | {val_per} | {step} | {test_per} | {lex_per} | {cer} | {hours} | {ckpt} |")

    for r in failed:
        arch = r.get("arch", "N/A")
        tier = r.get("tier", args.tier)
        val_per = f"{r.get('best_val_per', 0.0):.2f}%" if r.get('best_val_per') is not None else "-"
        step = r.get("best_val_step") if r.get("best_val_step") is not None else "-"
        err_type = r.get("error_type", "FAILED")
        lines.append(f"| **{arch}** | {tier} | {val_per} | {step} | ❌ **FAILED** | - | - | 0.0h | `{err_type}` |")

    if successful:
        lines.extend([
            "",
            "```",
            "Test PER Ranking (Lower is Better):",
        ])
        for r in sorted_successful:
            arch = r.get("arch", "N/A")
            tier = r.get("tier", args.tier)
            t_per = r.get("test_clean_per") or 100.0
            bar_len = int(t_per / 2.5)
            bar = "█" * max(1, bar_len)
            label = f"{arch} [{tier}]"
            lines.append(f"  {label:<32}: {bar} {t_per:.2f}%")
        lines.extend([
            "```",
            "",
        ])

    if failed:
        lines.extend([
            "## ⚠️ Diagnostic Log for Failed Runs",
            "",
            "| Architecture | Tier | Failure Type | Diagnostic Error Message |",
            "| :--- | :---: | :--- | :--- |",
        ])
        for r in failed:
            arch = r.get("arch", "N/A")
            tier = r.get("tier", args.tier)
            err_type = r.get("error_type", "Unknown Error")
            err_msg = str(r.get("error_message", "N/A")).replace("\n", " ")[:120]
            lines.append(f"| **{arch}** | {tier} | **{err_type}** | `{err_msg}` |")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
