"""Interactive and Beginner-Friendly Visualizations for HuBERT ASR & Audio Processing."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.models.hubert_asr import HuBERTForCTC
from src.data.tokenizer import CharacterTokenizer


class HuBERTLayerInspector:
    """Extracts, analyzes, and visualizes representations across every layer of HuBERT."""

    def __init__(self, model: HuBERTForCTC, tokenizer: CharacterTokenizer, device: torch.device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def inspect_sample(self, audio: torch.Tensor) -> Dict[str, any]:
        """Perform a full inspection pass recording intermediate states at every stage."""
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)

        # 1. Input audio stats
        raw_audio_np = audio.squeeze().cpu().numpy()
        sample_rate = self.model.config.sample_rate

        # 2. Step-by-step CNN Feature Extraction
        cnn_activations = []
        x_cnn = audio.unsqueeze(1) if audio.dim() == 2 else audio
        for i, conv_layer in enumerate(self.model.feature_extractor.layers):
            x_cnn = conv_layer(x_cnn)
            cnn_activations.append({
                "layer_idx": i,
                "shape": list(x_cnn.shape),
                "mean_act": float(x_cnn.abs().mean().item()),
                "sample_features": x_cnn[0, :8, :].cpu().numpy().tolist(),  # First 8 channels
            })

        # 3. Model forward pass
        outputs = self.model(audio, output_hidden_states=True, output_attentions=True)
        logits = outputs["logits"]  # (1, T_frames, vocab_size)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (T_frames, vocab_size)

        # 4. Hidden states across Transformer layers
        hidden_states = outputs["hidden_states"]  # L+1 tensors, each (1, T_frames, embed_dim)
        layer_summaries = []
        for l_idx, h in enumerate(hidden_states):
            h_np = h.squeeze(0).cpu().numpy()  # (T_frames, embed_dim)
            # Compute variance and energy across time
            temporal_energy = np.mean(h_np ** 2, axis=-1).tolist()
            feature_variance = np.var(h_np, axis=0)[:16].tolist()  # First 16 dims
            layer_summaries.append({
                "layer_idx": l_idx,
                "name": "CNN Projection" if l_idx == 0 else f"Transformer Layer {l_idx}",
                "temporal_energy": temporal_energy,
                "feature_variance": feature_variance,
                "shape": list(h_np.shape),
            })

        # 5. Attention Maps
        attentions = outputs["attentions"]  # List of L tensors, each (1, num_heads, T, T)
        attention_matrices = []
        if attentions is not None:
            for l_idx, attn in enumerate(attentions):
                attn_np = attn.squeeze(0).cpu().numpy()  # (num_heads, T, T)
                # Head 0 and head average
                head_avg = np.mean(attn_np, axis=0).tolist()
                attention_matrices.append({
                    "layer_idx": l_idx,
                    "num_heads": attn_np.shape[0],
                    "average_attention": head_avg,
                })

        # 6. CTC Posteriorgram and Token Alignment
        out_lengths = outputs["output_lengths"]
        decoded_tokens = self.model.decode_greedy(logits, lengths=out_lengths)[0]
        decoded_text = self.tokenizer.decode(decoded_tokens)

        # Argmax per frame
        argmax_ids = logits.squeeze(0).argmax(dim=-1).cpu().tolist()
        frame_emissions = []
        for t, token_id in enumerate(argmax_ids):
            char = self.tokenizer.id_to_char.get(token_id, "")
            is_blank = (token_id == self.model.config.blank_index)
            frame_emissions.append({
                "frame": t,
                "time_sec": round(t * 0.02, 3),  # 20ms per frame
                "token_id": token_id,
                "char": "_" if is_blank else char,
                "confidence": float(probs[t, token_id]),
                "is_blank": is_blank,
            })

        return {
            "sample_rate": sample_rate,
            "num_audio_samples": len(raw_audio_np),
            "duration_sec": round(len(raw_audio_np) / sample_rate, 3),
            "num_frames": probs.shape[0],
            "decoded_text": decoded_text,
            "cnn_layers": cnn_activations,
            "layer_summaries": layer_summaries,
            "attention_matrices": attention_matrices,
            "ctc_probabilities": probs.tolist(),
            "frame_emissions": frame_emissions,
            "vocab": self.tokenizer.vocab,
        }

    def plot_comprehensive_flow(
        self,
        audio: torch.Tensor,
        transcript: str,
        save_path: str = "outputs/visualizations/hubert_pipeline_flow.png",
    ):
        """Generate a 4-panel static diagram illustrating the pipeline from raw wave to CTC tokens."""
        data = self.inspect_sample(audio)
        wav = audio.squeeze().cpu().numpy()
        probs = np.array(data["ctc_probabilities"])  # (T, V)
        t_frames = probs.shape[0]

        fig, axes = plt.subplots(4, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [1.2, 1.2, 1.5, 1.2]})

        # Panel 1: Raw Input Audio Waveform (16kHz)
        time_wav = np.linspace(0, data["duration_sec"], len(wav))
        axes[0].plot(time_wav, wav, color="#2563EB", linewidth=1.0)
        axes[0].set_title(
            f"1. Input Audio Waveform [16 kHz Sampling] — Length: {len(wav):,} samples ({data['duration_sec']:.2f}s) — Transcript: '{transcript}'",
            fontweight="bold",
            fontsize=11,
            color="#1E293B",
        )
        axes[0].set_ylabel("Amplitude", fontsize=10)
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Panel 2: Audio Spectrogram (Frequency Formants & Energy)
        Pxx, freqs, bins, im = axes[1].specgram(wav, NFFT=512, Fs=data["sample_rate"], noverlap=256, cmap="magma")
        axes[1].set_title(
            "2. Acoustic Spectrogram (Acoustic Formants, Pitch F0 & Resonances over Time)",
            fontweight="bold",
            fontsize=11,
            color="#1E293B",
        )
        axes[1].set_ylabel("Frequency (Hz)", fontsize=10)

        # Panel 3: Layer Representation Dynamics (CNN vs Deep Transformer Layers)
        layer_energies = [np.array(l["temporal_energy"]) for l in data["layer_summaries"]]
        time_frames = np.linspace(0, data["duration_sec"], t_frames)
        colors = ["#9333EA", "#0284C7", "#059669", "#D97706", "#DC2626"]
        for i, energy in enumerate(layer_energies[:5]):
            name = data["layer_summaries"][i]["name"]
            axes[2].plot(time_frames, energy, label=name, color=colors[i % len(colors)], linewidth=1.6)

        axes[2].set_title(
            "3. Intermediate Layer Representation Energy across Time (Feature Abstraction Progression)",
            fontweight="bold",
            fontsize=11,
            color="#1E293B",
        )
        axes[2].set_ylabel("Representation Norm", fontsize=10)
        axes[2].legend(loc="upper right", fontsize=9, framealpha=0.8)
        axes[2].grid(True, linestyle="--", alpha=0.5)

        # Panel 4: CTC Posteriorgram & Output Sequence Emissions
        # Top 10 most active tokens across all frames
        active_token_indices = np.argsort(probs.sum(axis=0))[-10:][::-1]
        active_token_names = [data["vocab"][idx] for idx in active_token_indices]
        probs_active = probs[:, active_token_indices].T  # (10, T)

        cax = axes[3].imshow(probs_active, aspect="auto", cmap="Blues", origin="lower", extent=[0, data["duration_sec"], -0.5, 9.5])
        axes[3].set_yticks(range(len(active_token_indices)))
        axes[3].set_yticklabels(active_token_names, fontsize=10, fontweight="bold")
        axes[3].set_title(
            f"4. CTC Output Posteriorgram (Character Probabilities) -> Decoded: '{data['decoded_text']}'",
            fontweight="bold",
            fontsize=11,
            color="#1E293B",
        )
        axes[3].set_xlabel("Time (seconds)", fontsize=10, fontweight="bold")
        axes[3].set_ylabel("Tokens", fontsize=10)
        fig.colorbar(cax, ax=axes[3], orientation="vertical", pad=0.01, label="Probability")

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved comprehensive flow diagram to: {save_path}")

    def plot_receptive_field_and_chunking(
        self,
        sample_rate: int = 16000,
        save_path: str = "outputs/visualizations/receptive_field_flow.png",
    ):
        """Diagram showing how raw audio samples map through 7 CNN downsampling strides and long-audio chunking."""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Receptive field growth table
        conv_specs = self.model.config.conv_layers
        rf = 1
        stride_cum = 1
        strides_list = [1]
        rf_list = [1]
        layer_labels = ["Audio Wave"]

        for idx, (channels, kernel, stride) in enumerate(conv_specs):
            rf = rf + (kernel - 1) * stride_cum
            stride_cum *= stride
            rf_list.append(rf)
            strides_list.append(stride_cum)
            layer_labels.append(f"CNN {idx+1}\n(k={kernel}, s={stride})")

        # Plot RF and cumulative downsampling stride
        x = range(len(layer_labels))
        ax.plot(x, rf_list, marker="o", color="#2563EB", linewidth=2.5, markersize=8, label="Receptive Field (Audio Samples)")
        ax.plot(x, strides_list, marker="s", color="#059669", linewidth=2.5, markersize=8, label="Cumulative Temporal Stride (Downsample Factor)")

        for i in range(len(x)):
            ms_rf = (rf_list[i] / sample_rate) * 1000
            ax.annotate(
                f"{rf_list[i]} smp\n({ms_rf:.1f} ms)",
                (x[i], rf_list[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                fontweight="bold",
                color="#1E3A8A",
            )
            ax.annotate(
                f"x{strides_list[i]}",
                (x[i], strides_list[i]),
                textcoords="offset points",
                xytext=(0, -18),
                ha="center",
                fontsize=8,
                fontweight="bold",
                color="#065F46",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels, fontsize=9)
        ax.set_yscale("log")
        ax.set_title(
            "How Long Audio is Processed: CNN Receptive Field & Downsampling Growth (320x / 20ms Frame Rate)",
            fontsize=12,
            fontweight="bold",
            color="#0F172A",
        )
        ax.set_ylabel("Samples (Log Scale)", fontsize=10, fontweight="bold")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", fontsize=10)

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved receptive field visualization to: {save_path}")
