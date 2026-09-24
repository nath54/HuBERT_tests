#!/usr/bin/env python3
"""Train HuBERT ASR model on real LibriSpeech human voice recordings."""

import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.config import HuBERTConfig
from src.models.hubert_asr import HuBERTForCTC
from src.data.tokenizer import CharacterTokenizer
from src.data.dataset import AudioASRDataset, AudioCollateFn
from src.data.augmentations import WaveformAugmenter
from src.training.trainer import HuBERTASTTrainer
from data.sample_dataset import load_manifest


def main():
    parser = argparse.ArgumentParser(description="Train HuBERT ASR on real LibriSpeech recordings.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--max_train_samples", type=int, default=300, help="Train subset for fast turnaround")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 65)
    print("   TRAINING HUBERT ASR ON REAL LIBRISPEECH RECORDINGS")
    print("=" * 65)
    print(f"Device: {device} | Epochs: {args.epochs} | Batch size: {args.batch_size}")

    tokenizer = CharacterTokenizer()
    config = HuBERTConfig(
        vocab_size=tokenizer.vocab_size,
        encoder_layers=4,
        encoder_heads=4,
        encoder_embed_dim=256,
        encoder_ffn_dim=1024,
    )
    model = HuBERTForCTC(config).to(device)

    # Load real LibriSpeech manifests
    train_manifest = Path("data/librispeech/librispeech_train.json")
    val_manifest = Path("data/librispeech/librispeech_val.json")

    if not train_manifest.exists():
        from src.data.librispeech import build_librispeech_manifest
        build_librispeech_manifest()

    train_samples = load_manifest(str(train_manifest))[: args.max_train_samples]
    val_samples = load_manifest(str(val_manifest))[: 50]

    print(f"Loaded {len(train_samples):,} real human speech utterances for training.")
    print(f"Loaded {len(val_samples):,} real human speech utterances for validation.")

    # Filter out samples longer than 6 seconds for consistent batch GPU memory
    train_samples = [s for s in train_samples if s["duration"] <= 6.0]
    val_samples = [s for s in val_samples if s["duration"] <= 6.0]

    train_ds = AudioASRDataset(train_samples, tokenizer, target_sample_rate=16000, max_duration_s=6.0, augmenter=WaveformAugmenter())
    val_ds = AudioASRDataset(val_samples, tokenizer, target_sample_rate=16000, max_duration_s=6.0)

    collate_fn = AudioCollateFn(pad_token_id=tokenizer.pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    trainer = HuBERTASTTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        scheduler=scheduler,
        checkpoint_dir="checkpoints",
        log_dir="logs",
        use_amp=True,
    )

    trainer.fit(num_epochs=args.epochs)
    print("\n[Done] Training on real LibriSpeech speech finished.")


if __name__ == "__main__":
    main()
