#!/usr/bin/env python3
"""Run Management CLI for AudioLearn.

Allows inspecting, listing, comparing, and deleting training runs across
model architectures and scaling tiers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.run_manager import RunManager


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """Render a clean ASCII table with padded columns."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    row_lines = [
        " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))
        for row in rows
    ]
    return f"{header_line}\n{sep_line}\n" + "\n".join(row_lines)


def cmd_list(args: argparse.Namespace) -> None:
    """List runs matching arch and tier filters."""
    runs = RunManager.list_runs(arch=args.arch, tier=args.tier)
    if not runs:
        print(f"No training runs found" + (f" for arch='{args.arch}', tier='{args.tier}'" if args.arch or args.tier else "") + ".")
        return

    headers = ["Run Name", "Arch", "Tier", "Status", "Step", "Loss", "PER", "Audio (h)", "Created"]
    rows = []
    for r in runs:
        summary = r.get("summary", {})
        step_str = f"{summary.get('step', 0)}/{summary.get('total_steps', '?')}"
        loss_val = f"{summary.get('loss', 0.0):.4f}" if summary.get("loss") is not None else "--"
        per_val = f"{summary.get('per', 0.0):.1f}%" if summary.get("per") is not None else "--"
        audio_h = f"{summary.get('cumulative_audio_hours', 0.0):.2f}h" if summary.get("cumulative_audio_hours") is not None else "--"

        created_str = r.get("created_at", "")
        if created_str and "T" in created_str:
            created_str = created_str.split("T")[0] + " " + created_str.split("T")[1][:5]

        rows.append([
            r["run_name"],
            r["arch"],
            r["tier"],
            r["status"],
            step_str,
            loss_val,
            per_val,
            audio_h,
            created_str or "--",
        ])

    print("=" * 80)
    print(f"📋 AUDIOLEARN TRAINING RUNS CATALOG ({len(runs)} runs)")
    print("=" * 80)
    print(format_table(headers, rows))
    print()


def cmd_show(args: argparse.Namespace) -> None:
    """Show detailed configuration and metrics for a specific run."""
    # Find matching run if arch/tier not fully specified
    runs = RunManager.list_runs(arch=args.arch, tier=args.tier)
    matched = [r for r in runs if r["run_name"] == args.run_name]
    if not matched:
        print(f"❌ Error: Run '{args.run_name}' not found.")
        sys.exit(1)

    target_meta = matched[0]
    details = RunManager.get_run_details(
        arch=target_meta["arch"],
        tier=target_meta["tier"],
        run_name=args.run_name,
    )
    if not details:
        print(f"❌ Error: Run details for '{args.run_name}' could not be loaded.")
        sys.exit(1)

    cfg = details.get("config", {})
    status = details.get("status_live", {})
    history = details.get("history", [])
    ckpts = details.get("checkpoints", [])

    print("=" * 80)
    print(f"🔍 RUN DETAILS: {args.run_name} [{details['arch'].upper()} - {details['tier'].upper()}]")
    print("=" * 80)
    print(f"Status:               {cfg.get('status', 'unknown')}")
    print(f"Created At:           {cfg.get('created_at', '--')}")
    print(f"Finished At:          {cfg.get('finished_at', '--')}")
    print(f"Git Commit:           {cfg.get('git_commit', '--')}")
    print(f"Launch Command:       {cfg.get('command', '--')}")
    print()

    print("--- ⚙️  Model & Training Hyperparameters ---")
    model_info = cfg.get("model", {})
    train_info = cfg.get("training_hyperparameters", {})
    mask_info = cfg.get("masking", {})
    data_info = cfg.get("data_pipeline", {})

    print(f"Parameters:           {model_info.get('trainable_parameters', 0):,} trainable ({model_info.get('total_parameters', 0):,} total)")
    print(f"Total Steps:          {train_info.get('steps', '--')} | Batch Size: {train_info.get('batch_size', '--')}")
    print(f"Learning Rate:        {train_info.get('learning_rate', '--')} (Warmup: {train_info.get('warmup_steps', 0)} steps)")
    print(f"CNN Feature Extractor: Frozen for first {train_info.get('freeze_cnn_steps', 0)} steps")
    print(f"Masking Mode:         {mask_info.get('masking_mode', '--')} (Prob: {mask_info.get('mask_prob', '--')}, Length: {mask_info.get('mask_length', '--')})")
    print(f"Data Pipeline:        Workers: {data_info.get('num_workers', '--')}, Buffer: {data_info.get('buffer_size', '--')}, Pool: {data_info.get('pool_size', '--')}")
    print()

    print("--- 💾 Saved Checkpoints ---")
    if ckpts:
        for c in ckpts:
            print(f"  • {c['filename']:<28} {c['size_mb']:>6.1f} MB  (modified: {c['modified_at']})")
    else:
        print("  (No checkpoints saved yet)")
    print()

    print("--- 🏆 Milestone History ---")
    if history:
        for h in history:
            print(f"  • Step {h.get('step', 0):4d} | Hours: {h.get('cumulative_audio_hours', 0.0):.2f}h | Loss: {h.get('pretrain_loss', 0.0):.4f} | "
                  f"PER: {h.get('librispeech_per', 100.0)}% | WER: {h.get('librispeech_wer', 100.0)}% | Pred: \"{h.get('sample_prediction', '')}\"")
    else:
        print("  (No milestone evaluations recorded yet)")
    print()


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare multiple runs side-by-side."""
    if len(args.run_names) < 2:
        print("Please provide at least 2 run names to compare.")
        sys.exit(1)

    runs = RunManager.list_runs(arch=args.arch, tier=args.tier)
    matched = {r["run_name"]: r for r in runs if r["run_name"] in args.run_names}

    missing = set(args.run_names) - set(matched.keys())
    if missing:
        print(f"⚠️  Warning: Runs not found: {', '.join(missing)}")

    compare_keys = [
        ("Architecture", lambda r, cfg: r["arch"]),
        ("Tier", lambda r, cfg: r["tier"]),
        ("Status", lambda r, cfg: r["status"]),
        ("Learning Rate", lambda r, cfg: cfg.get("training_hyperparameters", {}).get("learning_rate", "--")),
        ("Warmup Steps", lambda r, cfg: cfg.get("training_hyperparameters", {}).get("warmup_steps", "--")),
        ("Freeze CNN Steps", lambda r, cfg: cfg.get("training_hyperparameters", {}).get("freeze_cnn_steps", "--")),
        ("Masking Mode", lambda r, cfg: cfg.get("masking", {}).get("masking_mode", "--")),
        ("Mask Prob", lambda r, cfg: cfg.get("masking", {}).get("mask_prob", "--")),
        ("Current Step", lambda r, cfg: r.get("summary", {}).get("step", "--")),
        ("Loss", lambda r, cfg: f"{r.get('summary', {}).get('loss', 0.0):.4f}" if r.get("summary", {}).get("loss") is not None else "--"),
        ("PER", lambda r, cfg: f"{r.get('summary', {}).get('per', 0.0):.1f}%" if r.get("summary", {}).get("per") is not None else "--"),
        ("Audio Hours", lambda r, cfg: f"{r.get('summary', {}).get('cumulative_audio_hours', 0.0):.2f}h" if r.get("summary", {}).get("cumulative_audio_hours") is not None else "--"),
    ]

    headers = ["Metric / Parameter"] + [name for name in args.run_names if name in matched]
    rows = []

    for label, extractor in compare_keys:
        row = [label]
        for name in headers[1:]:
            r = matched[name]
            cfg = r.get("config", {})
            val = extractor(r, cfg)
            row.append(str(val))
        rows.append(row)

    print("=" * 80)
    print(f"⚖️  TRAINING RUNS COMPARISON")
    print("=" * 80)
    print(format_table(headers, rows))
    print()


def cmd_delete(args: argparse.Namespace) -> None:
    """Delete a run from disk and registry."""
    runs = RunManager.list_runs(arch=args.arch, tier=args.tier)
    matched = [r for r in runs if r["run_name"] == args.run_name]
    if not matched:
        print(f"❌ Error: Run '{args.run_name}' not found.")
        sys.exit(1)

    target = matched[0]
    if not args.yes:
        confirm = input(f"Are you sure you want to permanently delete run '{args.run_name}' ({target['arch']}/{target['tier']})? [y/N]: ")
        if confirm.lower() not in ("y", "yes"):
            print("Aborted.")
            return

    ok = RunManager.delete_run(
        arch=target["arch"],
        tier=target["tier"],
        run_name=args.run_name,
    )
    if ok:
        print(f"✅ Run '{args.run_name}' successfully deleted.")
    else:
        print(f"⚠️  No files were deleted for run '{args.run_name}'.")


def main():
    parser = argparse.ArgumentParser(description="AudioLearn Training Run Manager.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # list
    p_list = subparsers.add_parser("list", help="List all training runs")
    p_list.add_argument("--arch", type=str, default=None, help="Filter by model architecture")
    p_list.add_argument("--tier", type=str, default=None, help="Filter by model tier")
    p_list.set_defaults(func=cmd_list)

    # show
    p_show = subparsers.add_parser("show", help="Show full config and metrics of a run")
    p_show.add_argument("run_name", type=str, help="Name of the run")
    p_show.add_argument("--arch", type=str, default=None, help="Architecture filter")
    p_show.add_argument("--tier", type=str, default=None, help="Tier filter")
    p_show.set_defaults(func=cmd_show)

    # compare
    p_comp = subparsers.add_parser("compare", help="Compare two or more runs side-by-side")
    p_comp.add_argument("run_names", nargs="+", help="Names of the runs to compare")
    p_comp.add_argument("--arch", type=str, default=None, help="Architecture filter")
    p_comp.add_argument("--tier", type=str, default=None, help="Tier filter")
    p_comp.set_defaults(func=cmd_compare)

    # delete
    p_del = subparsers.add_parser("delete", help="Delete a training run")
    p_del.add_argument("run_name", type=str, help="Name of the run to delete")
    p_del.add_argument("--arch", type=str, default=None, help="Architecture filter")
    p_del.add_argument("--tier", type=str, default=None, help="Tier filter")
    p_del.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    p_del.set_defaults(func=cmd_delete)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
