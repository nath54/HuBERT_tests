#!/usr/bin/env python3
"""CLI script to generate beginner-friendly dynamic visualizers and layer inspection dashboards."""

import argparse
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.config import HuBERTConfig
from src.models.hubert_asr import HuBERTForCTC
from src.data.tokenizer import CharacterTokenizer
from src.data.dataset import AudioASRDataset
from data.sample_dataset import load_manifest, generate_synthetic_asr_dataset
from src.xai.interactive_visualizer import HuBERTLayerInspector


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive layer inspection and audio processing diagrams.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to trained checkpoint")
    parser.add_argument("--manifest", type=str, default="data/raw/synthetic/val_manifest.json", help="Dataset manifest")
    parser.add_argument("--output_dir", type=str, default="outputs/visualizations", help="Output directory for plots")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CharacterTokenizer()
    ckpt_path = Path(args.checkpoint)

    if ckpt_path.exists():
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = HuBERTForCTC(ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("[Warning] Checkpoint not found. Instantiating fresh HuBERT model...")
        cfg = HuBERTConfig(vocab_size=tokenizer.vocab_size, encoder_layers=4, encoder_heads=4, encoder_embed_dim=256)
        model = HuBERTForCTC(cfg).to(device)

    model.eval()

    # Load dataset sample
    if not Path(args.manifest).exists():
        generate_synthetic_asr_dataset()
    samples = load_manifest(args.manifest)
    dataset = AudioASRDataset(samples, tokenizer, target_sample_rate=model.config.sample_rate)

    sample = dataset[0]
    audio = sample["audio"]
    transcript = sample["text"]
    print(f"Inspecting sample '{transcript}' ({len(audio)} audio samples, {len(audio)/16000:.2f} seconds)...")

    inspector = HuBERTLayerInspector(model, tokenizer, device)

    # 1. Plot comprehensive flow (Waveform -> Spectrogram -> Layers -> CTC Posteriorgram)
    flow_path = out_dir / "hubert_pipeline_flow.png"
    inspector.plot_comprehensive_flow(audio, transcript, save_path=str(flow_path))

    # 2. Plot receptive field growth & audio chunking logic
    rf_path = out_dir / "receptive_field_flow.png"
    inspector.plot_receptive_field_and_chunking(save_path=str(rf_path))

    # 3. Export full inspection JSON data for interactive browser tools
    inspection_data = inspector.inspect_sample(audio)
    json_path = out_dir / "layer_inspection_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(inspection_data, f, indent=2)
    print(f"Exported interactive layer inspection data to: {json_path}")

    print("\nVisualizations successfully generated in:", out_dir)


if __name__ == "__main__":
    main()
