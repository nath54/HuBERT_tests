#!/usr/bin/env python3
"""Training script for HuBERT ASR."""

import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.config import HuBERTConfig
from src.models.hubert_asr import HuBERTForCTC
from src.data.tokenizer import CharacterTokenizer
from src.data.dataset import AudioASRDataset, AudioCollateFn
from src.data.augmentations import WaveformAugmenter
from src.training.trainer import HuBERTASTTrainer
from data.sample_dataset import generate_synthetic_asr_dataset, load_manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Train HuBERT ASR model.")
    parser.add_argument("--config", type=str, default="configs/train_asr.yaml", help="Path to training config")
    parser.add_argument("--model_config", type=str, default="configs/hubert_base.yaml", help="Path to model config")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load configurations
    with open(args.model_config, "r") as f:
        m_cfg = yaml.safe_load(f)["model"]

    with open(args.config, "r") as f:
        t_cfg = yaml.safe_load(f)

    # Initialize Tokenizer
    tokenizer = CharacterTokenizer()

    # Build HuBERT Configuration
    config = HuBERTConfig(
        sample_rate=m_cfg.get("sample_rate", 16000),
        conv_layers=[tuple(l) for l in m_cfg["conv_layers"]],
        conv_feature_dim=m_cfg.get("conv_feature_dim", 256),
        encoder_embed_dim=m_cfg.get("encoder_embed_dim", 256),
        encoder_layers=m_cfg.get("encoder_layers", 4),
        encoder_heads=m_cfg.get("encoder_heads", 4),
        encoder_ffn_dim=m_cfg.get("encoder_ffn_dim", 1024),
        dropout=m_cfg.get("dropout", 0.1),
        attention_dropout=m_cfg.get("attention_dropout", 0.1),
        pos_conv_kernel=m_cfg.get("pos_conv_kernel", 64),
        pos_conv_groups=m_cfg.get("pos_conv_groups", 16),
        vocab_size=tokenizer.vocab_size,
    )

    # Model
    model = HuBERTForCTC(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"HuBERT Model instantiated: {trainable_params:,} trainable parameters ({total_params:,} total).")

    # Dataset preparation
    data_dir = Path(t_cfg["data"]["dataset_dir"])
    train_manifest = data_dir / "train_manifest.json"
    val_manifest = data_dir / "val_manifest.json"

    if not train_manifest.exists() or not val_manifest.exists():
        print("Manifests not found. Auto-generating synthetic speech dataset...")
        train_samples, val_samples = generate_synthetic_asr_dataset(output_dir=str(data_dir))
    else:
        train_samples = load_manifest(str(train_manifest))
        val_samples = load_manifest(str(val_manifest))

    augmenter = WaveformAugmenter() if t_cfg["data"].get("augment", True) else None

    train_dataset = AudioASRDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        target_sample_rate=config.sample_rate,
        augmenter=augmenter,
        normalize_audio=t_cfg["data"].get("normalize_audio", True),
        max_duration_s=t_cfg["data"].get("max_duration_s", 3.0),
    )
    val_dataset = AudioASRDataset(
        samples=val_samples,
        tokenizer=tokenizer,
        target_sample_rate=config.sample_rate,
        augmenter=None,
        normalize_audio=t_cfg["data"].get("normalize_audio", True),
    )

    batch_size = args.batch_size or t_cfg["training"]["batch_size"]
    collate_fn = AudioCollateFn(pad_token_id=tokenizer.pad_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # 0 for immediate cross-platform safety
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Optimizer & Scheduler
    lr = args.lr or t_cfg["training"]["lr"]
    weight_decay = t_cfg["training"].get("weight_decay", 0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = args.epochs or t_cfg["training"]["epochs"]
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    # Trainer
    trainer = HuBERTASTTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=t_cfg["training"].get("checkpoint_dir", "checkpoints"),
        log_dir=t_cfg["training"].get("log_dir", "logs"),
        use_amp=t_cfg["training"].get("use_amp", True),
        max_grad_norm=t_cfg["training"].get("max_grad_norm", 1.0),
    )

    trainer.fit(num_epochs=epochs)
    print("Training process finished.")


if __name__ == "__main__":
    main()
