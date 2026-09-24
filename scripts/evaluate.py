#!/usr/bin/env python3
"""Evaluation script for HuBERT ASR."""

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
from src.training.metrics import MetricTracker
from data.sample_dataset import load_manifest


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained HuBERT ASR model.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--manifest", type=str, default="data/raw/synthetic/val_manifest.json", help="Test manifest")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading checkpoint from: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    tokenizer = CharacterTokenizer()

    model = HuBERTForCTC(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    samples = load_manifest(args.manifest)
    dataset = AudioASRDataset(samples=samples, tokenizer=tokenizer, target_sample_rate=config.sample_rate)
    collate_fn = AudioCollateFn(pad_token_id=tokenizer.pad_id)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    tracker = MetricTracker()
    print(f"Evaluating {len(dataset)} samples on {device}...")

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            audio_lengths = batch["audio_lengths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            outputs = model(
                audio=audio,
                audio_lengths=audio_lengths,
                targets=targets,
                target_lengths=target_lengths,
            )
            loss = outputs["loss"]
            logits = outputs["logits"]
            out_lengths = outputs["output_lengths"]

            decoded_tokens_batch = model.decode_greedy(logits, lengths=out_lengths)
            preds = [tokenizer.decode(t) for t in decoded_tokens_batch]
            refs = batch["texts"]

            tracker.update(loss.item(), batch_size=audio.size(0), predictions=preds, references=refs)

    print("\n" + "=" * 50)
    print("           EVALUATION RESULTS")
    print("=" * 50)
    print(f" Average Loss: {tracker.avg_loss:.4f}")
    print(f" CER         : {tracker.cer * 100:.2f}%")
    print(f" WER         : {tracker.wer * 100:.2f}%")
    print("=" * 50)

    print("\nSample Predictions:")
    for i in range(min(8, len(tracker.all_predictions))):
        print(f"[{i+1}] Ref : '{tracker.all_references[i]}'")
        print(f"    Pred: '{tracker.all_predictions[i]}'")


if __name__ == "__main__":
    main()
