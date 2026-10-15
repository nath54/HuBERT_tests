"""
Training Logger & Real-Time Curve Renderer for NaVox Speech Models.
Features:
- Structured JSONL metrics logging (`train_metrics.jsonl`)
- Human-readable terminal logs with step benchmarks
- High-resolution matplotlib training curves (`curves.png`) rendered at every validation step:
  * Panel 1: Multi-Task Losses (Total, CTC, Draft CTC, Macro Manner, MoE Balance)
  * Panel 2: Acoustic Phoneme Error Rate (PER %) & Space Boundary Accuracy (%)
  * Panel 3: Full End-to-End Word Error Rate (WER %) & Character Error Rate (CER %)
  * Panel 4: Per-Language Multilingual Breakdown (EN, FR, ES, IT)
- Verbatim sample inspection formatting (Ground truth vs. Predictions).
"""

# Import Modules
from dataclasses import dataclass, field
import datetime
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


class TrainLogger:
    """
    Dedicated Logger and Real-Time Dashboard Renderer for NaVox Training Runs.
    """

    def __init__(
        self,
        log_dir: str,
        run_name: Optional[str] = None,
        model_tier: str = "ultra",
    ) -> None:
        """
        Initialize TrainLogger.

        Args:
            log_dir (str): Base root logging directory.
            run_name (Optional[str]): Unique run identifier. If None, timestamped automatically.
            model_tier (str): Architecture tier ('tiny', 'small', 'base', 'large', 'ultra').
        """
        # Timestamp
        now_str: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name: str = run_name or f"navox_{model_tier}_{now_str}"
        self.run_dir: str = os.path.abspath(os.path.join(log_dir, self.run_name))
        os.makedirs(self.run_dir, exist_ok=True)

        # File paths
        self.metrics_jsonl_path: str = os.path.join(self.run_dir, "train_metrics.jsonl")
        self.console_log_path: str = os.path.join(self.run_dir, "console.log")
        self.curves_png_path: str = os.path.join(self.run_dir, "curves.png")
        self.samples_log_path: str = os.path.join(self.run_dir, "val_samples.log")

        # In-memory history for real-time plotting
        self.train_history: List[Dict[str, Any]] = []
        self.val_history: List[Dict[str, Any]] = []

        # Open console log file
        self.console_file = open(self.console_log_path, "a", encoding="utf-8")
        self.log(f"================================================================================")
        self.log(f"🚀 Initialized NaVox Training Run: {self.run_name}")
        self.log(f"📁 Log Directory: {self.run_dir}")
        self.log(f"📊 Live Curves: {self.curves_png_path}")
        self.log(f"================================================================================")

    def log(self, message: str) -> None:
        """Prints to stdout and flushes to console.log."""
        print(message, flush=True)
        self.console_file.write(message + "\n")
        self.console_file.flush()

    def log_train_step(self, step: int, metrics: Dict[str, Any]) -> None:
        """Records a training iteration step."""
        metrics["step"] = step
        metrics["type"] = "train"
        metrics["timestamp"] = datetime.datetime.now().isoformat()
        self.train_history.append(metrics)

        # Append to JSONL
        with open(self.metrics_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

    def log_val_step(
        self,
        step: int,
        val_metrics: Dict[str, Any],
        samples: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Records a validation step, renders fresh training curves, and formats sample inspections.
        """
        val_metrics["step"] = step
        val_metrics["type"] = "val"
        val_metrics["timestamp"] = datetime.datetime.now().isoformat()
        self.val_history.append(val_metrics)

        # Append to JSONL
        with open(self.metrics_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(val_metrics) + "\n")

        # Print formatted summary table to console
        self.log(f"\n" + "═" * 80)
        self.log(f"📊 [VAL EVALUATION @ STEP {step}]")
        self.log(
            f"   • Loss: {val_metrics.get('loss', 0.0):.4f} | "
            f"Acoustic PER: {val_metrics.get('acoustic_per', 0.0):.2f}% | "
            f"PER: {val_metrics.get('per', 0.0):.2f}% | "
            f"Space Acc: {val_metrics.get('space_acc', 0.0):.2f}%"
        )
        if "text_wer" in val_metrics:
            self.log(
                f"   🌟 E2E Full Pipeline: Text WER: {val_metrics.get('text_wer', 0.0):.2f}% | "
                f"Text CER: {val_metrics.get('text_cer', 0.0):.2f}%"
            )

        # Per-language breakdown if available
        lang_breakdown = val_metrics.get("languages", {})
        if lang_breakdown:
            lang_strs = [
                f"{lang}: PER {m.get('per', 0.0):.1f}% (WER: {m.get('wer', 0.0):.1f}%)"
                for lang, m in lang_breakdown.items()
            ]
            self.log(f"   🌍 Lang Breakdown: " + " | ".join(lang_strs))
        self.log("═" * 80)

        # Log inspection samples if provided
        if samples:
            self._log_samples(step, samples)

        # Render fresh matplotlib curves
        self.render_curves()

    def _log_samples(self, step: int, samples: List[Dict[str, Any]]) -> None:
        """Writes verbatim side-by-side text/phoneme inspection samples."""
        with open(self.samples_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n==================== STEP {step} SAMPLES ====================\n")
            for idx, s in enumerate(samples[:5]):  # Up to 5 samples
                lang = s.get("lang", "??")
                f.write(f"\n[{idx+1}] Language: {lang} | PER: {s.get('per', 0.0):.1f}% | WER: {s.get('wer', 0.0):.1f}%\n")
                f.write(f"  • GT Text   : {s.get('gt_text', '')}\n")
                f.write(f"  • Pred Text : {s.get('pred_text', '')}\n")
                f.write(f"  • GT Phon   : {s.get('gt_phonemes', '')}\n")
                f.write(f"  • Pred Phon : {s.get('pred_phonemes', '')}\n")

                # Also print the first 2 directly to console for instant visibility
                if idx < 2:
                    self.log(f"  🔍 [{lang}] GT   : {s.get('gt_text', '')[:100]}")
                    self.log(f"       PRED : {s.get('pred_text', '')[:100]}")
                    self.log(f"       PHON : {s.get('pred_phonemes', '')[:100]}")

    def render_curves(self) -> None:
        """
        Renders a 4-panel comprehensive dashboard plot of training progress to curves.png.
        """
        if not self.val_history:
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        fig.suptitle(f"NaVox Training Progress — {self.run_name}", fontsize=15, fontweight="bold")

        val_steps = [v["step"] for v in self.val_history]

        # ---------------- Panel 1: Loss Progression ----------------
        ax1 = axes[0, 0]
        ax1.set_title("Multi-Task Losses", fontsize=12, fontweight="bold")
        if self.train_history:
            tr_steps = [t["step"] for t in self.train_history if "loss" in t]
            tr_losses = [t["loss"] for t in self.train_history if "loss" in t]
            # Smooth train loss if long
            ax1.plot(tr_steps, tr_losses, label="Train Total Loss", alpha=0.35, color="gray", linewidth=1.0)
        
        val_losses = [v.get("loss", 0.0) for v in self.val_history]
        ax1.plot(val_steps, val_losses, label="Val Total Loss", marker="o", color="#1f77b4", linewidth=2.0)

        # Plot individual sub-losses if logged
        if "ctc_loss" in self.val_history[0]:
            ax1.plot(val_steps, [v.get("ctc_loss", 0.0) for v in self.val_history], label="CTC Phonemes", linestyle="--", color="#ff7f0e")
        if "draft_loss" in self.val_history[0]:
            ax1.plot(val_steps, [v.get("draft_loss", 0.0) for v in self.val_history], label="Layer Draft CTC", linestyle=":", color="#2ca02c")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right")

        # ---------------- Panel 2: Acoustic Accuracy & PER ----------------
        ax2 = axes[0, 1]
        ax2.set_title("Phoneme Error Rate (PER) & Space Accuracy", fontsize=12, fontweight="bold")
        val_per = [v.get("per", 0.0) for v in self.val_history]
        val_ac_per = [v.get("acoustic_per", v.get("per", 0.0)) for v in self.val_history]
        val_space_acc = [v.get("space_acc", 0.0) for v in self.val_history]

        ax2.plot(val_steps, val_per, label="Overall PER (%)", marker="s", color="#d62728", linewidth=2.0)
        ax2.plot(val_steps, val_ac_per, label="Acoustic PER (%)", marker="^", color="#9467bd", linewidth=1.5)
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(val_steps, val_space_acc, label="Space Boundary Acc (%)", marker="x", color="#2ca02c", linestyle="-.", linewidth=2.0)
        ax2_twin.set_ylabel("Space Accuracy (%)", color="#2ca02c")
        ax2_twin.set_ylim(50, 101)

        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Error Rate (%)")
        ax2.grid(True, linestyle="--", alpha=0.5)
        # Combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        # ---------------- Panel 3: E2E Full Pipeline Text WER & CER ----------------
        ax3 = axes[1, 0]
        ax3.set_title("End-to-End Multilingual Text WER & CER", fontsize=12, fontweight="bold")
        wer_vals = [v.get("text_wer", None) for v in self.val_history]
        cer_vals = [v.get("text_cer", None) for v in self.val_history]

        valid_wer_pairs = [(s, w) for s, w in zip(val_steps, wer_vals) if w is not None]
        if valid_wer_pairs:
            s_w, w_v = zip(*valid_wer_pairs)
            ax3.plot(s_w, w_v, label="Full Text WER (%)", marker="D", color="#ff7f0e", linewidth=2.5)
        valid_cer_pairs = [(s, c) for s, c in zip(val_steps, cer_vals) if c is not None]
        if valid_cer_pairs:
            s_c, c_v = zip(*valid_cer_pairs)
            ax3.plot(s_c, c_v, label="Full Text CER (%)", marker="v", color="#17becf", linewidth=1.8)

        ax3.set_xlabel("Steps")
        ax3.set_ylabel("Error Rate (%)")
        ax3.grid(True, linestyle="--", alpha=0.5)
        ax3.legend(loc="upper right")

        # ---------------- Panel 4: Multilingual Language Breakdown ----------------
        ax4 = axes[1, 1]
        ax4.set_title("Multilingual Acoustic PER Breakdown by Language", fontsize=12, fontweight="bold")
        lang_colors = {"EN": "#1f77b4", "FR": "#2ca02c", "ES": "#d62728", "IT": "#9467bd"}

        for lang, color in lang_colors.items():
            lang_pers = [
                v.get("languages", {}).get(lang, {}).get("per", None)
                for v in self.val_history
            ]
            pairs = [(s, p) for s, p in zip(val_steps, lang_pers) if p is not None]
            if pairs:
                xs, ys = zip(*pairs)
                ax4.plot(xs, ys, label=f"{lang} PER (%)", marker="o", color=color, linewidth=1.8)

        ax4.set_xlabel("Steps")
        ax4.set_ylabel("PER (%)")
        ax4.grid(True, linestyle="--", alpha=0.5)
        ax4.legend(loc="upper right")

        # Save figure cleanly
        plt.tight_layout()
        plt.savefig(self.curves_png_path, dpi=180)
        plt.close(fig)

    def close(self) -> None:
        """Closes console log file stream."""
        if hasattr(self, "console_file") and self.console_file:
            self.console_file.close()
