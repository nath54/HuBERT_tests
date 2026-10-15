"""Comprehensive Training and Evaluation Pipeline for HuBERT ASR."""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.hubert_asr import HuBERTForCTC
from src.data.tokenizer import CharacterTokenizer
from src.training.metrics import MetricTracker


class HuBERTASTTrainer:
    """Trainer for HuBERT ASR with mixed precision, gradient clipping, and logging."""

    def __init__(
        self,
        model: HuBERTForCTC,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        tokenizer: CharacterTokenizer,
        device: torch.device,
        scheduler: Optional[any] = None,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        use_amp: bool = True,
        max_grad_norm: float = 1.0,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        self.scheduler = scheduler
        self.use_amp = use_amp and device.type == "cuda"
        self.max_grad_norm = max_grad_norm

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_cer": [],
            "val_wer": [],
        }
        self.best_cer = float("inf")

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch."""
        self.model.train()
        tracker = MetricTracker()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for step, batch in enumerate(pbar):
            audio = batch["audio"].to(self.device)
            audio_lengths = batch["audio_lengths"].to(self.device)
            targets = batch["targets"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(
                    audio=audio,
                    audio_lengths=audio_lengths,
                    targets=targets,
                    target_lengths=target_lengths,
                )
                loss = outputs["loss"]

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Warning] Loss is NaN/Inf at epoch {epoch}, step {step}. Skipping step.")
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            batch_size = audio.size(0)
            tracker.update(loss.item(), batch_size=batch_size)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{tracker.avg_loss:.4f}"})

            global_step = epoch * len(self.train_loader) + step
            self.writer.add_scalar("Train/StepLoss", loss.item(), global_step)

        return tracker.avg_loss

    @torch.no_grad()
    def evaluate(self, epoch: int = 0) -> Dict[str, float]:
        """Run evaluation on validation split."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        tracker = MetricTracker()
        sample_preds = []
        sample_refs = []

        for batch in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
            audio = batch["audio"].to(self.device)
            audio_lengths = batch["audio_lengths"].to(self.device)
            targets = batch["targets"].to(self.device)
            target_lengths = batch["target_lengths"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(
                    audio=audio,
                    audio_lengths=audio_lengths,
                    targets=targets,
                    target_lengths=target_lengths,
                )
                loss = outputs["loss"]

            logits = outputs["logits"]
            out_lengths = outputs["output_lengths"]
            decoded_tokens_batch = self.model.decode_greedy(logits, lengths=out_lengths)

            batch_preds = [
                self.tokenizer.decode(tokens) for tokens in decoded_tokens_batch
            ]
            batch_refs = batch["texts"]

            tracker.update(
                loss.item(),
                batch_size=audio.size(0),
                predictions=batch_preds,
                references=batch_refs,
            )

            if len(sample_preds) < 5:
                sample_preds.extend(batch_preds[: 5 - len(sample_preds)])
                sample_refs.extend(batch_refs[: 5 - len(sample_refs)])

        val_metrics = {
            "val_loss": tracker.avg_loss,
            "val_cer": tracker.cer,
            "val_wer": tracker.wer,
        }

        # Print samples
        print(f"\n--- Validation Samples (Epoch {epoch}) ---")
        for i, (pred, ref) in enumerate(zip(sample_preds, sample_refs)):
            print(f"[{i+1}] Ref : '{ref}'")
            print(f"    Pred: '{pred}'")
        print("-------------------------------------------\n")

        return val_metrics

    def fit(self, num_epochs: int) -> Dict[str, List[float]]:
        """Run full training and evaluation loop across epochs."""
        print(f"Starting training on device: {self.device} (AMP: {self.use_amp})")
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            t_epoch_start = time.time()
            train_loss = self.train_epoch(epoch)
            self.history["train_loss"].append(train_loss)
            self.writer.add_scalar("Train/EpochLoss", train_loss, epoch)

            val_metrics = self.evaluate(epoch)
            val_loss = val_metrics.get("val_loss", 0.0)
            val_cer = val_metrics.get("val_cer", 1.0)
            val_wer = val_metrics.get("val_wer", 1.0)

            self.history["val_loss"].append(val_loss)
            self.history["val_cer"].append(val_cer)
            self.history["val_wer"].append(val_wer)

            self.writer.add_scalar("Val/EpochLoss", val_loss, epoch)
            self.writer.add_scalar("Val/CER", val_cer, epoch)
            self.writer.add_scalar("Val/WER", val_wer, epoch)

            elapsed = time.time() - t_epoch_start
            print(
                f"Epoch {epoch:02d}/{num_epochs:02d} [{elapsed:.1f}s] - "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val CER: {val_cer:.4f} | "
                f"Val WER: {val_wer:.4f}"
            )

            # Checkpoint best model
            if val_cer < self.best_cer:
                self.best_cer = val_cer
                self.save_checkpoint("best_model.pt", epoch, val_metrics)
                print(f"[*] New best model saved with CER: {val_cer:.4f}")

            # Save latest checkpoint
            self.save_checkpoint("latest_model.pt", epoch, val_metrics)

        total_elapsed = time.time() - start_time
        print(f"Training completed in {total_elapsed:.1f}s. Best CER: {self.best_cer:.4f}")

        # Save history log
        history_path = self.log_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint with optimizer and config."""
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.model.config,
                "metrics": metrics,
            },
            path,
        )

    def load_checkpoint(self, filename: str):
        """Load checkpoint weights into model."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {path} (Epoch {checkpoint['epoch']})")
