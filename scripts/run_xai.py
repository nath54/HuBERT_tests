#!/usr/bin/env python3
"""Run the complete XAI Suite (Captum Gradients, NAPS, Trained Probes/Decoders) on HuBERT ASR."""

import argparse
import sys
from pathlib import Path
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.config import HuBERTConfig
from src.models.hubert_asr import HuBERTForCTC
from src.data.tokenizer import CharacterTokenizer
from src.data.dataset import AudioASRDataset
from data.sample_dataset import load_manifest, generate_synthetic_asr_dataset
from src.xai.captum_gradients import AudioGradientExplainer
from src.xai.naps import (
    NeuronActivationProfiler,
    NeuronSaliencyAnalyzer,
    ActivationPatcher,
)
from src.xai.probes import LayerwiseProbeTrainer
from src.xai.visualizer import (
    plot_waveform_attribution,
    plot_spectrogram_and_attribution,
    plot_neuron_selectivity_heatmap,
    plot_layer_probing_curve,
    plot_attention_map,
)


def main():
    parser = argparse.ArgumentParser(description="Run XAI Suite on HuBERT ASR.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--manifest", type=str, default="data/raw/synthetic/val_manifest.json", help="Data manifest")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save XAI plots and metrics")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    grad_dir = out_dir / "gradients"
    naps_dir = out_dir / "naps"
    probes_dir = out_dir / "probes"
    for d in (grad_dir, naps_dir, probes_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("      RUNNING HUBERT EXPLAINABLE AI (XAI) SUITE")
    print("=" * 60)
    print(f"Device: {device} | Output Directory: {out_dir}")

    # Load model and dataset
    tokenizer = CharacterTokenizer()
    ckpt_path = Path(args.checkpoint)

    if ckpt_path.exists():
        print(f"Loading weights from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = HuBERTForCTC(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("[Warning] Checkpoint not found. Initializing untrained HuBERT model for XAI demonstration...")
        config = HuBERTConfig(vocab_size=tokenizer.vocab_size, encoder_layers=4, encoder_heads=4, encoder_embed_dim=256)
        model = HuBERTForCTC(config).to(device)

    model.eval()

    # Load dataset
    if not Path(args.manifest).exists():
        generate_synthetic_asr_dataset()
    samples = load_manifest(args.manifest)
    dataset = AudioASRDataset(samples=samples, tokenizer=tokenizer, target_sample_rate=config.sample_rate)

    test_sample = dataset[0]
    audio_sample = test_sample["audio"]
    transcript = test_sample["text"]
    print(f"\nAnalyzing reference sample: '{transcript}' ({len(audio_sample)} audio samples)")

    # -------------------------------------------------------------
    # 1. Gradient-based Attribution using Captum
    # -------------------------------------------------------------
    print("\n[1/3] Running Gradient-based XAI with Captum...")
    explainer = AudioGradientExplainer(model, device)

    # Explain token at frame 10 (or middle frame)
    out = model(audio_sample.unsqueeze(0).to(device))
    logits = out["logits"]
    t_frames = logits.shape[1]
    target_frame = t_frames // 2
    pred_token_id = logits[0, target_frame].argmax().item()
    pred_char = tokenizer.id_to_char.get(pred_token_id, f"token_{pred_token_id}")

    print(f"  -> Attributing token '{pred_char}' (id={pred_token_id}) at frame {target_frame} / {t_frames}...")

    # Integrated Gradients
    ig_result = explainer.explain_token(
        audio=audio_sample,
        frame_idx=target_frame,
        token_idx=pred_token_id,
        method="integrated_gradients",
        n_steps=25,
    )
    ig_attr = ig_result["attributions"]

    # Saliency
    sal_result = explainer.explain_token(
        audio=audio_sample,
        frame_idx=target_frame,
        token_idx=pred_token_id,
        method="saliency",
    )
    sal_attr = sal_result["attributions"]

    # Save plots
    plot_waveform_attribution(
        audio_sample,
        ig_attr,
        title=f"Waveform Attribution for '{pred_char}' (Captum Integrated Gradients)",
        save_path=grad_dir / "integrated_gradients_waveform.png",
    )
    plot_spectrogram_and_attribution(
        audio_sample,
        ig_attr,
        title=f"Audio Spectrogram & Integrated Gradients for '{pred_char}'",
        save_path=grad_dir / "integrated_gradients_spectrogram.png",
    )
    plot_waveform_attribution(
        audio_sample,
        sal_attr,
        title=f"Waveform Saliency Map for '{pred_char}' (Gradient Saliency)",
        save_path=grad_dir / "saliency_waveform.png",
    )
    print(f"  [+] Saved Captum gradient plots to {grad_dir}")

    # -------------------------------------------------------------
    # 2. NAPS (Neuron Activation Profiles, Selectivity & Patching)
    # -------------------------------------------------------------
    print("\n[2/3] Running NAPS (Neuron Activation Profiles & Patching)...")
    profiler = NeuronActivationProfiler(model, device)
    profiles = profiler.collect_profiles(dataset, max_samples=30)
    selectivity = profiler.compute_selectivity(profiles, target_category="speech", baseline_category="silence")

    plot_neuron_selectivity_heatmap(
        selectivity_per_layer=selectivity,
        top_k=20,
        title="Neuron Activation Profiles: Speech vs Silence Selectivity",
        save_path=naps_dir / "neuron_selectivity_heatmap.png",
    )

    # Activation Patching / Causal scan
    patcher = ActivationPatcher(model, device)
    causal_impacts = patcher.layer_causal_importance_scan(audio_sample)
    print(f"  -> Causal layer ablation impacts (Logit shifts per layer): {causal_impacts}")

    # Neuron Saliency
    saliency_analyzer = NeuronSaliencyAnalyzer(model, device)
    sample_target = test_sample["target"]
    neuron_saliency = saliency_analyzer.compute_neuron_saliency(audio_sample, sample_target)
    print(f"  [+] Saved NAPS analysis and heatmaps to {naps_dir}")

    # -------------------------------------------------------------
    # 3. Trained Decoders & Diagnostic Probing
    # -------------------------------------------------------------
    print("\n[3/3] Training Diagnostic Probes and Reconstruction Decoders across layers...")
    probe_trainer = LayerwiseProbeTrainer(model, device)

    # Diagnostic Probing (predicting acoustic energy categories from each layer)
    probe_results = probe_trainer.train_diagnostic_probes(dataset, epochs=5, num_classes=4)
    plot_layer_probing_curve(
        layer_names=probe_results["layer_names"],
        accuracies=probe_results["accuracies"],
        title="Diagnostic Linear Probing Accuracy Across HuBERT Layers",
        save_path=probes_dir / "diagnostic_probing_curve.png",
    )

    # Inversion Decoders (measuring retention of acoustic spectrum)
    inversion_results = probe_trainer.train_reconstruction_decoders(dataset, epochs=5, n_mels=32)
    print(f"  -> Layer-wise Acoustic Reconstruction MSE: {inversion_results['reconstruction_mse']}")
    print(f"  [+] Saved Probing and Decoder plots to {probes_dir}")

    # Attention map visualization
    if out["attentions"] is not None and len(out["attentions"]) > 0:
        attn_matrix = out["attentions"][0][0]  # Layer 0, Batch 0
        plot_attention_map(
            attn_matrix,
            title="Layer 0 Self-Attention Weights",
            save_path=out_dir / "attention_layer0.png",
        )

    print("\n" + "=" * 60)
    print("      XAI SUITE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"All artifacts, figures, and plots are stored in: {out_dir}")


if __name__ == "__main__":
    main()
