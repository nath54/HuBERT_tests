"""RunManager for speech model training.

Manages multiple training runs of the same model and tier, providing:
- Automatic sequential run naming (run_1, run_2, ...) or custom user naming.
- Full configuration persistence (saving train_config.json at run start).
- Isolated checkpoint and logging directories per run.
- Master catalog registry (runs_registry.json) tracking run statuses and metrics.
- Backwards compatibility mirrors for global status files and root checkpoints.
"""

from __future__ import annotations

import collections
import datetime
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


class RunManager:
    """Manages individual training runs for a specific architecture and scaling tier."""

    def __init__(
        self,
        arch: str,
        tier: str,
        run_name: Optional[str] = None,
        base_ckpt_dir: str | Path = "checkpoints",
        base_log_dir: str | Path = "logs",
        is_resume: bool = False,
    ):
        self.arch = str(arch).lower()
        self.tier = str(tier).lower()
        self.base_ckpt_dir = Path(base_ckpt_dir)
        self.base_log_dir = Path(base_log_dir)
        self.is_resume = is_resume

        # Tier root directories
        self.tier_ckpt_dir = self.base_ckpt_dir / self.arch / self.tier
        self.tier_log_dir = self.base_log_dir / self.arch / self.tier
        self.tier_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.tier_log_dir.mkdir(parents=True, exist_ok=True)

        self.registry_file = self.tier_ckpt_dir / "runs_registry.json"

        # Determine run name
        if run_name and run_name not in ("auto", "none", "None", ""):
            # Sanitize custom run name
            sanitized = re.sub(r"[^\w\.-]", "_", str(run_name).strip())
            self.run_name = sanitized
            self.is_auto_named = False
        else:
            self.is_auto_named = True
            if self.is_resume:
                # When resuming without a name, pick the latest existing run if one exists
                latest_existing = self._find_latest_existing_run_name()
                if latest_existing:
                    self.run_name = latest_existing
                else:
                    self.run_name = self._generate_next_run_name()
            else:
                self.run_name = self._generate_next_run_name()

        # Run-specific directories
        self.run_ckpt_dir = self.tier_ckpt_dir / self.run_name
        self.run_log_dir = self.tier_log_dir / self.run_name
        self.run_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.run_log_dir.mkdir(parents=True, exist_ok=True)

        self.config_file = self.run_ckpt_dir / "train_config.json"
        self.log_config_file = self.run_log_dir / "train_config.json"
        self.summary_file = self.run_ckpt_dir / "run_summary.json"

        # Backward compatibility root targets
        self.root_status_file = self.base_log_dir / f"{self.arch}_status_live.json"
        self.tier_status_file = self.base_log_dir / f"{self.arch}_{self.tier}_status_live.json"
        self.tier_history_file = self.base_log_dir / f"{self.arch}_{self.tier}_history.json"
        self.tier_step_history_file = self.base_log_dir / f"{self.arch}_{self.tier}_step_history.json"
        self.tier_latest_ckpt = self.tier_ckpt_dir / "checkpoint_latest.pt"

    def _generate_next_run_name(self) -> str:
        """Find highest existing integer in run_{N} and return run_{N+1}."""
        existing_numbers = []

        # Check subdirectories in checkpoint tier dir
        if self.tier_ckpt_dir.exists():
            for p in self.tier_ckpt_dir.iterdir():
                if p.is_dir():
                    m = re.match(r"^run_(\d+)$", p.name)
                    if m:
                        existing_numbers.append(int(m.group(1)))

        # Also check registry if present
        reg = self._load_registry()
        for r_name in reg.get("runs", {}).keys():
            m = re.match(r"^run_(\d+)$", r_name)
            if m:
                existing_numbers.append(int(m.group(1)))

        next_idx = max(existing_numbers, default=0) + 1
        return f"run_{next_idx}"

    def _find_latest_existing_run_name(self) -> Optional[str]:
        """Find the most recently modified run folder or latest run in registry."""
        reg = self._load_registry()
        runs_dict = reg.get("runs", {})
        if runs_dict:
            # Sort by updated_at or created_at
            sorted_runs = sorted(
                runs_dict.items(),
                key=lambda item: item[1].get("updated_at") or item[1].get("created_at") or "",
                reverse=True,
            )
            for r_name, _ in sorted_runs:
                if (self.tier_ckpt_dir / r_name).exists():
                    return r_name

        # Fallback: scan disk directories by mtime
        candidate_dirs = [
            p for p in self.tier_ckpt_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ]
        if candidate_dirs:
            candidate_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidate_dirs[0].name

        return None

    def _load_registry(self) -> Dict[str, Any]:
        """Load runs registry dictionary."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"arch": self.arch, "tier": self.tier, "runs": {}}

    def _save_registry(self, registry: Dict[str, Any]) -> None:
        """Write registry atomically."""
        try:
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2)
            # Mirror in log tier dir
            mirror_path = self.tier_log_dir / "runs_registry.json"
            with open(mirror_path, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            print(f"[RunManager Warning] Failed to update registry: {e}")

    @staticmethod
    def get_git_commit() -> Optional[str]:
        """Return the current Git commit SHA if in a git repo."""
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                return res.stdout.strip()
        except Exception:
            pass
        return None

    @classmethod
    def assemble_full_config(
        cls,
        arch: str,
        tier: str,
        run_name: str,
        model: torch.nn.Module,
        model_config: Any,
        target_extractor: Any,
        args: Any,
    ) -> Dict[str, Any]:
        """Assemble the complete training configuration dictionary with all parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Convert model_config to dict
        if hasattr(model_config, "to_dict"):
            cfg_dict = model_config.to_dict()
        elif hasattr(model_config, "__dict__"):
            cfg_dict = {k: v for k, v in model_config.__dict__.items() if not k.startswith("_")}
        else:
            cfg_dict = str(model_config)

        # Training hyperparams from args
        cli_args_dict = vars(args) if hasattr(args, "__dict__") else {}

        full_config = {
            "run_name": run_name,
            "arch": arch,
            "tier": tier,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "initialized",
            "command": " ".join(sys.argv),
            "git_commit": cls.get_git_commit(),
            "system_environment": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            },
            "model": {
                "arch": arch,
                "tier": tier,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "config": cfg_dict,
            },
            "training_hyperparameters": {
                "steps": getattr(args, "steps", 625),
                "batch_size": getattr(args, "batch_size", 8),
                "learning_rate": getattr(args, "lr", 1e-4),
                "warmup_steps": getattr(args, "warmup_steps", 0),
                "freeze_cnn_steps": getattr(args, "freeze_cnn_steps", 0),
                "optimizer": "AdamW",
                "weight_decay": 1e-2,
                "clip_grad_norm": 1.0,
                "amp_enabled": torch.cuda.is_available(),
                "eval_interval": getattr(args, "eval_interval", 50),
                "save_interval": getattr(args, "save_interval", 50),
                "probe_steps": getattr(args, "probe_steps", 25),
            },
            "masking": {
                "masking_mode": getattr(args, "masking_mode", "none"),
                "mask_prob": getattr(args, "mask_prob", 0.65),
                "mask_length": getattr(args, "mask_length", 10),
            },
            "data_pipeline": {
                "target_type": getattr(target_extractor, "target_type", "unknown"),
                "vocab_size": getattr(target_extractor, "vocab_size", 0),
                "num_workers": getattr(args, "num_workers", 4),
                "buffer_size": getattr(args, "buffer_size", 50),
                "watermark": getattr(args, "watermark", 25),
                "use_rolling_pool": getattr(args, "use_rolling_pool", True),
                "pool_size": getattr(args, "pool_size", 250),
            },
            "cli_arguments": cli_args_dict,
        }
        return full_config

    def init_run(self, full_config: Dict[str, Any]) -> Path:
        """Save train_config.json and register run in catalog."""
        full_config["run_name"] = self.run_name
        full_config["status"] = full_config.get("status", "running")
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        full_config.setdefault("created_at", now_iso)
        full_config["updated_at"] = now_iso

        # Save configuration in both ckpt and log folders
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(full_config, f, indent=2)
        with open(self.log_config_file, "w", encoding="utf-8") as f:
            json.dump(full_config, f, indent=2)

        # Register in master runs catalog
        registry = self._load_registry()
        if "runs" not in registry:
            registry["runs"] = {}

        t_hparams = full_config.get("training_hyperparameters", {})
        m_params = full_config.get("masking", {})

        registry["runs"][self.run_name] = {
            "run_name": self.run_name,
            "created_at": full_config.get("created_at", now_iso),
            "updated_at": full_config.get("updated_at", now_iso),
            "status": full_config.get("status", "running"),
            "checkpoint_dir": str(self.run_ckpt_dir),
            "log_dir": str(self.run_log_dir),
            "config_path": str(self.config_file),
            "latest_checkpoint": str(self.run_ckpt_dir / "checkpoint_latest.pt"),
            "summary": {
                "step": 0,
                "total_steps": t_hparams.get("steps", 0),
                "learning_rate": t_hparams.get("learning_rate", 0.0),
                "masking_mode": m_params.get("masking_mode", "none"),
                "warmup_steps": t_hparams.get("warmup_steps", 0),
                "freeze_cnn_steps": t_hparams.get("freeze_cnn_steps", 0),
            },
        }
        self._save_registry(registry)
        return self.config_file

    def get_checkpoint_paths(self, step: int) -> Tuple[Path, Path]:
        """Return (step_checkpoint_path, latest_checkpoint_path) for this run."""
        step_path = self.run_ckpt_dir / f"checkpoint_step_{step}.pt"
        latest_path = self.run_ckpt_dir / "checkpoint_latest.pt"
        return step_path, latest_path

    def on_checkpoint_saved(
        self,
        step: int,
        loss: Optional[float] = None,
        acc: Optional[float] = None,
        per: Optional[float] = None,
        audio_hours: Optional[float] = None,
    ) -> None:
        """Update run summary, master registry, and root latest_checkpoint link."""
        latest_path = self.run_ckpt_dir / "checkpoint_latest.pt"

        # Update root latest_checkpoint.pt mirror/copy for backwards compatibility
        try:
            shutil.copyfile(latest_path, self.tier_latest_ckpt)
        except Exception:
            pass

        # Update registry
        registry = self._load_registry()
        if "runs" in registry and self.run_name in registry["runs"]:
            entry = registry["runs"][self.run_name]
            entry["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            summary = entry.setdefault("summary", {})
            summary["step"] = step
            if loss is not None:
                summary["loss"] = loss
            if acc is not None:
                summary["accuracy"] = acc
            if per is not None:
                summary["per"] = per
            if audio_hours is not None:
                summary["cumulative_audio_hours"] = audio_hours
            self._save_registry(registry)

    def find_resume_checkpoint(self, requested_resume: Optional[str]) -> Optional[Path]:
        """Locate checkpoint to resume from for this run or legacy checkpoints."""
        if not requested_resume:
            return None

        # 1. Custom path provided
        if requested_resume not in ("auto", "latest", "", True):
            custom_path = Path(requested_resume)
            if custom_path.exists():
                return custom_path
            return None

        # 2. Check current run directory first
        run_candidates = [
            self.run_ckpt_dir / "checkpoint_latest.pt",
            self.run_ckpt_dir / "latest_checkpoint.pt",
        ]
        for cand in run_candidates:
            if cand.exists():
                return cand

        run_step_ckpts = sorted(list(self.run_ckpt_dir.glob("checkpoint_step_*.pt")), key=os.path.getmtime)
        if run_step_ckpts:
            return run_step_ckpts[-1]

        # 3. If auto-named run_1, check legacy tier root checkpoints
        root_candidates = [
            self.tier_ckpt_dir / "checkpoint_latest.pt",
            self.tier_ckpt_dir / "latest_checkpoint.pt",
        ]
        for cand in root_candidates:
            if cand.exists():
                return cand

        root_step_ckpts = sorted(list(self.tier_ckpt_dir.glob("checkpoint_step_*.pt")), key=os.path.getmtime)
        if root_step_ckpts:
            return root_step_ckpts[-1]

        return None

    def update_status(self, live_status: Dict[str, Any]) -> None:
        """Write live status to run directory and mirror to root files."""
        live_status["run_name"] = self.run_name

        # 1. Run-specific status
        run_status_file = self.run_log_dir / "status_live.json"
        try:
            with open(run_status_file, "w", encoding="utf-8") as f:
                json.dump(live_status, f, indent=2)
        except Exception:
            pass

        # 2. Backward compatibility mirrors
        try:
            with open(self.tier_status_file, "w", encoding="utf-8") as f:
                json.dump(live_status, f, indent=2)
            with open(self.root_status_file, "w", encoding="utf-8") as f:
                json.dump(live_status, f, indent=2)
        except Exception:
            pass

    def update_history(self, history: List[Dict[str, Any]]) -> None:
        """Write milestone history to run directory and mirror to root."""
        run_history_file = self.run_log_dir / "history.json"
        payload = {"arch": self.arch, "tier": self.tier, "run_name": self.run_name, "history": history}
        try:
            with open(run_history_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            with open(self.tier_history_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    def update_step_history(self, step_history: List[Dict[str, Any]]) -> None:
        """Write step history to run directory and mirror to root."""
        run_step_history_file = self.run_log_dir / "step_history.json"
        try:
            with open(run_step_history_file, "w", encoding="utf-8") as f:
                json.dump(step_history, f)
            with open(self.tier_step_history_file, "w", encoding="utf-8") as f:
                json.dump(step_history, f)
        except Exception:
            pass

    def finish_run(self, status: str = "completed", final_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Mark run as completed or interrupted and finalize summary."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Update train_config.json
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                cfg["status"] = status
                cfg["finished_at"] = now_iso
                if final_metrics:
                    cfg["final_metrics"] = final_metrics
                with open(self.config_file, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
                with open(self.log_config_file, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2)
            except Exception:
                pass

        # Update summary file
        summary_payload = {
            "run_name": self.run_name,
            "arch": self.arch,
            "tier": self.tier,
            "status": status,
            "finished_at": now_iso,
            "final_metrics": final_metrics or {},
        }
        try:
            with open(self.summary_file, "w", encoding="utf-8") as f:
                json.dump(summary_payload, f, indent=2)
        except Exception:
            pass

        # Update registry
        registry = self._load_registry()
        if "runs" in registry and self.run_name in registry["runs"]:
            registry["runs"][self.run_name]["status"] = status
            registry["runs"][self.run_name]["finished_at"] = now_iso
            if final_metrics:
                registry["runs"][self.run_name].setdefault("summary", {}).update(final_metrics)
            self._save_registry(registry)

    # --------------------------------------------------------------------------
    # Static & Class Query Utilities
    # --------------------------------------------------------------------------

    @classmethod
    def list_runs(
        cls,
        arch: Optional[str] = None,
        tier: Optional[str] = None,
        base_ckpt_dir: str | Path = "checkpoints",
    ) -> List[Dict[str, Any]]:
        """List all discovered runs matching the optional arch and tier filters."""
        base_ckpt_dir = Path(base_ckpt_dir)
        runs_list = []

        if not base_ckpt_dir.exists():
            return []

        arch_dirs = [base_ckpt_dir / arch] if arch else [p for p in base_ckpt_dir.iterdir() if p.is_dir()]

        for a_dir in arch_dirs:
            if not a_dir.exists() or not a_dir.is_dir():
                continue
            cur_arch = a_dir.name
            tier_dirs = [a_dir / tier] if tier else [p for p in a_dir.iterdir() if p.is_dir()]

            for t_dir in tier_dirs:
                if not t_dir.exists() or not t_dir.is_dir():
                    continue
                cur_tier = t_dir.name

                # Check runs_registry.json
                reg_file = t_dir / "runs_registry.json"
                reg_runs = {}
                if reg_file.exists():
                    try:
                        with open(reg_file, "r", encoding="utf-8") as f:
                            reg_runs = json.load(f).get("runs", {})
                    except Exception:
                        pass

                # Scan physical run directories
                run_dirs = [p for p in t_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]

                for r_dir in run_dirs:
                    r_name = r_dir.name
                    reg_entry = reg_runs.get(r_name, {})

                    cfg_file = r_dir / "train_config.json"
                    cfg = {}
                    if cfg_file.exists():
                        try:
                            with open(cfg_file, "r", encoding="utf-8") as f:
                                cfg = json.load(f)
                        except Exception:
                            pass

                    # Latest checkpoint check
                    latest_pt = r_dir / "checkpoint_latest.pt"
                    has_checkpoint = latest_pt.exists()

                    # Status live
                    run_log_dir = Path("logs") / cur_arch / cur_tier / r_name
                    live_status_file = run_log_dir / "status_live.json"
                    live_info = {}
                    if live_status_file.exists():
                        try:
                            with open(live_status_file, "r", encoding="utf-8") as f:
                                live_info = json.load(f)
                        except Exception:
                            pass

                    # Merge summary
                    summary = reg_entry.get("summary", {})
                    if live_info:
                        summary["step"] = live_info.get("step", summary.get("step", 0))
                        summary["loss"] = live_info.get("loss", summary.get("loss", 0.0))
                        summary["cumulative_audio_hours"] = live_info.get(
                            "cumulative_audio_hours", summary.get("cumulative_audio_hours", 0.0)
                        )

                    entry = {
                        "arch": cur_arch,
                        "tier": cur_tier,
                        "run_name": r_name,
                        "status": cfg.get("status") or reg_entry.get("status") or ("running" if live_info.get("is_running") else "unknown"),
                        "created_at": cfg.get("created_at") or reg_entry.get("created_at") or "",
                        "updated_at": reg_entry.get("updated_at") or "",
                        "checkpoint_dir": str(r_dir),
                        "has_checkpoint": has_checkpoint,
                        "latest_checkpoint": str(latest_pt) if has_checkpoint else None,
                        "config": cfg,
                        "summary": summary,
                    }
                    runs_list.append(entry)

        # Sort runs by created_at or run_name
        runs_list.sort(key=lambda item: item.get("created_at") or item.get("run_name"), reverse=True)
        return runs_list

    @classmethod
    def get_run_details(
        cls,
        arch: str,
        tier: str,
        run_name: str,
        base_ckpt_dir: str | Path = "checkpoints",
        base_log_dir: str | Path = "logs",
    ) -> Optional[Dict[str, Any]]:
        """Retrieve complete information, configuration, and telemetry for a specific run."""
        run_ckpt_dir = Path(base_ckpt_dir) / arch / tier / run_name
        run_log_dir = Path(base_log_dir) / arch / tier / run_name

        if not run_ckpt_dir.exists() and not run_log_dir.exists():
            return None

        # Load train_config.json
        cfg = {}
        for candidate_cfg in (run_ckpt_dir / "train_config.json", run_log_dir / "train_config.json"):
            if candidate_cfg.exists():
                try:
                    with open(candidate_cfg, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    break
                except Exception:
                    pass

        # Load status_live.json
        status_live = {}
        status_file = run_log_dir / "status_live.json"
        if status_file.exists():
            try:
                with open(status_file, "r", encoding="utf-8") as f:
                    status_live = json.load(f)
            except Exception:
                pass

        # Load history.json
        history = []
        history_file = run_log_dir / "history.json"
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f).get("history", [])
            except Exception:
                pass

        # Load step_history.json
        step_history = []
        step_history_file = run_log_dir / "step_history.json"
        if step_history_file.exists():
            try:
                with open(step_history_file, "r", encoding="utf-8") as f:
                    step_history = json.load(f)
            except Exception:
                pass

        # List checkpoints
        checkpoints = []
        latest_checkpoint = None
        if run_ckpt_dir.exists():
            for p in sorted(run_ckpt_dir.glob("checkpoint_*.pt"), key=os.path.getmtime):
                checkpoints.append({
                    "filename": p.name,
                    "path": str(p),
                    "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                    "modified_at": datetime.datetime.fromtimestamp(
                        p.stat().st_mtime, tz=datetime.timezone.utc
                    ).isoformat(),
                })
            latest_cand = run_ckpt_dir / "checkpoint_latest.pt"
            if latest_cand.exists():
                latest_checkpoint = str(latest_cand)
            elif checkpoints:
                latest_checkpoint = checkpoints[-1]["path"]

        return {
            "arch": arch,
            "tier": tier,
            "run_name": run_name,
            "config": cfg,
            "status_live": status_live,
            "history": history,
            "step_history": step_history,
            "checkpoints": checkpoints,
            "latest_checkpoint": latest_checkpoint,
            "checkpoint_dir": str(run_ckpt_dir),
            "log_dir": str(run_log_dir),
        }

    @classmethod
    def delete_run(
        cls,
        arch: str,
        tier: str,
        run_name: str,
        base_ckpt_dir: str | Path = "checkpoints",
        base_log_dir: str | Path = "logs",
    ) -> bool:
        """Remove a run's checkpoint directory, log directory, and registry entry."""
        run_ckpt_dir = Path(base_ckpt_dir) / arch / tier / run_name
        run_log_dir = Path(base_log_dir) / arch / tier / run_name

        deleted_something = False
        if run_ckpt_dir.exists():
            shutil.rmtree(run_ckpt_dir)
            deleted_something = True

        if run_log_dir.exists():
            shutil.rmtree(run_log_dir)
            deleted_something = True

        # Remove from registry
        reg_file = Path(base_ckpt_dir) / arch / tier / "runs_registry.json"
        if reg_file.exists():
            try:
                with open(reg_file, "r", encoding="utf-8") as f:
                    reg = json.load(f)
                if "runs" in reg and run_name in reg["runs"]:
                    del reg["runs"][run_name]
                    with open(reg_file, "w", encoding="utf-8") as f:
                        json.dump(reg, f, indent=2)
            except Exception:
                pass

        return deleted_something
