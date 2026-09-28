"""Unit tests for RunManager and multi-run training management."""

import json
import shutil
import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.training.run_manager import RunManager
from src.models.config import HuBERTConfig


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class DummyTargetExtractor:
    def __init__(self):
        self.target_type = "phoneme_tokens"
        self.vocab_size = 64


class DummyArgs:
    def __init__(self):
        self.steps = 500
        self.batch_size = 8
        self.lr = 0.0001
        self.warmup_steps = 50
        self.freeze_cnn_steps = 100
        self.masking_mode = "none"
        self.mask_prob = 0.0
        self.mask_length = 10
        self.num_workers = 4
        self.buffer_size = 20
        self.watermark = 10
        self.use_rolling_pool = True
        self.pool_size = 250
        self.eval_interval = 50
        self.save_interval = 25
        self.probe_steps = 25


@pytest.fixture
def temp_dirs():
    temp_ckpt = tempfile.mkdtemp()
    temp_log = tempfile.mkdtemp()
    yield Path(temp_ckpt), Path(temp_log)
    shutil.rmtree(temp_ckpt, ignore_errors=True)
    shutil.rmtree(temp_log, ignore_errors=True)


def test_automatic_run_naming(temp_dirs):
    ckpt_dir, log_dir = temp_dirs

    # First run -> run_1
    mgr1 = RunManager(arch="phono_hubert", tier="mini", base_ckpt_dir=ckpt_dir, base_log_dir=log_dir)
    assert mgr1.run_name == "run_1"
    assert mgr1.is_auto_named is True
    mgr1.init_run({"run_name": mgr1.run_name, "training_hyperparameters": {"steps": 100, "learning_rate": 1e-4}, "masking": {"masking_mode": "none"}})

    # Second run -> run_2
    mgr2 = RunManager(arch="phono_hubert", tier="mini", base_ckpt_dir=ckpt_dir, base_log_dir=log_dir)
    assert mgr2.run_name == "run_2"
    mgr2.init_run({"run_name": mgr2.run_name, "training_hyperparameters": {"steps": 100, "learning_rate": 1e-4}, "masking": {"masking_mode": "none"}})

    # Third run -> run_3
    mgr3 = RunManager(arch="phono_hubert", tier="mini", base_ckpt_dir=ckpt_dir, base_log_dir=log_dir)
    assert mgr3.run_name == "run_3"


def test_custom_run_naming(temp_dirs):
    ckpt_dir, log_dir = temp_dirs

    mgr = RunManager(
        arch="phono_hubert",
        tier="medium",
        run_name="my_experiment_clean_ctc",
        base_ckpt_dir=ckpt_dir,
        base_log_dir=log_dir,
    )
    assert mgr.run_name == "my_experiment_clean_ctc"
    assert mgr.is_auto_named is False

    # Name sanitization test
    mgr_dirty = RunManager(
        arch="phono_hubert",
        tier="medium",
        run_name="dirty name/test:1",
        base_ckpt_dir=ckpt_dir,
        base_log_dir=log_dir,
    )
    assert mgr_dirty.run_name == "dirty_name_test_1"


def test_full_config_assembly_and_persistence(temp_dirs):
    ckpt_dir, log_dir = temp_dirs

    model = DummyModel()
    model_cfg = HuBERTConfig(vocab_size=64)
    target_ext = DummyTargetExtractor()
    args = DummyArgs()

    mgr = RunManager(
        arch="phono_hubert",
        tier="small",
        run_name="full_config_test",
        base_ckpt_dir=ckpt_dir,
        base_log_dir=log_dir,
    )

    full_cfg = RunManager.assemble_full_config(
        arch="phono_hubert",
        tier="small",
        run_name=mgr.run_name,
        model=model,
        model_config=model_cfg,
        target_extractor=target_ext,
        args=args,
    )

    # Verify all expected sections are present
    assert full_cfg["run_name"] == "full_config_test"
    assert full_cfg["arch"] == "phono_hubert"
    assert full_cfg["tier"] == "small"
    assert "system_environment" in full_cfg
    assert "model" in full_cfg
    assert full_cfg["model"]["trainable_parameters"] == sum(p.numel() for p in model.parameters())
    assert "training_hyperparameters" in full_cfg
    assert full_cfg["training_hyperparameters"]["learning_rate"] == 0.0001
    assert full_cfg["training_hyperparameters"]["warmup_steps"] == 50
    assert full_cfg["training_hyperparameters"]["freeze_cnn_steps"] == 100
    assert full_cfg["masking"]["masking_mode"] == "none"
    assert full_cfg["data_pipeline"]["target_type"] == "phoneme_tokens"

    # Save configuration
    config_path = mgr.init_run(full_cfg)
    assert config_path.exists()

    # Load from disk and verify
    with open(config_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["run_name"] == "full_config_test"
    assert loaded["training_hyperparameters"]["freeze_cnn_steps"] == 100


def test_run_lifecycle_and_registry(temp_dirs):
    ckpt_dir, log_dir = temp_dirs

    mgr = RunManager(
        arch="phono_hubert",
        tier="mini",
        run_name="lifecycle_run",
        base_ckpt_dir=ckpt_dir,
        base_log_dir=log_dir,
    )

    config = {
        "run_name": mgr.run_name,
        "training_hyperparameters": {"steps": 100, "learning_rate": 1e-4, "warmup_steps": 10, "freeze_cnn_steps": 20},
        "masking": {"masking_mode": "specaugment"},
    }
    mgr.init_run(config)

    # Checkpoint paths
    step_pt, latest_pt = mgr.get_checkpoint_paths(50)
    assert step_pt.name == "checkpoint_step_50.pt"
    assert latest_pt.name == "checkpoint_latest.pt"

    # Create dummy checkpoint files
    torch.save({"step": 50}, latest_pt)
    torch.save({"step": 50}, step_pt)
    mgr.on_checkpoint_saved(step=50, loss=4.5, acc=42.0, per=38.5, audio_hours=0.5)

    # Status live and history
    mgr.update_status({"step": 50, "loss": 4.5, "is_running": True})
    mgr.update_history([{"step": 50, "librispeech_per": 38.5}])

    # Finish run
    mgr.finish_run(status="completed", final_metrics={"step": 100, "best_per": 35.0})

    # Test list_runs
    runs = RunManager.list_runs(arch="phono_hubert", tier="mini", base_ckpt_dir=ckpt_dir)
    assert len(runs) == 1
    assert runs[0]["run_name"] == "lifecycle_run"
    assert runs[0]["status"] == "completed"
    assert runs[0]["has_checkpoint"] is True

    # Test get_run_details
    details = RunManager.get_run_details(
        arch="phono_hubert",
        tier="mini",
        run_name="lifecycle_run",
        base_ckpt_dir=ckpt_dir,
        base_log_dir=log_dir,
    )
    assert details is not None
    assert details["run_name"] == "lifecycle_run"
    assert len(details["checkpoints"]) >= 1
    assert len(details["history"]) == 1

    # Test delete_run
    deleted = RunManager.delete_run(
        arch="phono_hubert",
        tier="mini",
        run_name="lifecycle_run",
        base_ckpt_dir=ckpt_dir,
        base_log_dir=log_dir,
    )
    assert deleted is True
    assert not (ckpt_dir / "phono_hubert" / "mini" / "lifecycle_run").exists()
    assert len(RunManager.list_runs(arch="phono_hubert", tier="mini", base_ckpt_dir=ckpt_dir)) == 0
