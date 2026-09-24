"""Visualization tools for XAI (Attributions, NAPS, Probing Curves, Attention)."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_waveform_attribution(
    waveform: torch.Tensor,
    attribution: torch.Tensor,
    sample_rate: int = 16000,
    title: str = "Audio Waveform Attribution (Captum Integrated Gradients)",
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot audio waveform with gradient attribution heatmap overlay."""
    wav = waveform.squeeze().cpu().numpy()
    attr = attribution.squeeze().cpu().numpy()

    # Align lengths if needed
    if len(attr) != len(wav):
        # Interpolate attr to waveform length
        attr_t = torch.tensor(attr).view(1, 1, -1)
        attr_interp = torch.nn.functional.interpolate(attr_t, size=len(wav), mode="linear")
        attr = attr_interp.squeeze().numpy()

    time_axis = np.arange(len(wav)) / sample_rate
    norm_attr = np.abs(attr) / (np.max(np.abs(attr)) + 1e-8)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Waveform
    ax1.plot(time_axis, wav, color="royalblue", alpha=0.8, label="Waveform")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Attribution Intensity
    ax2.plot(time_axis, attr, color="crimson", linewidth=1.2, label="Attribution")
    ax2.fill_between(time_axis, 0, attr, color="crimson", alpha=0.3)
    ax2.set_ylabel("Attribution Score")
    ax2.set_xlabel("Time (seconds)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_spectrogram_and_attribution(
    waveform: torch.Tensor,
    attribution: torch.Tensor,
    sample_rate: int = 16000,
    title: str = "Spectrogram vs Captum Attribution",
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot audio spectrogram side-by-side with temporal attribution."""
    wav = waveform.squeeze().cpu().numpy()
    attr = attribution.squeeze().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    # Spectrogram
    Pxx, freqs, bins, im = ax1.specgram(wav, NFFT=512, Fs=sample_rate, noverlap=256, cmap="viridis")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title(title, fontsize=13, fontweight="bold")

    # Temporal Attribution
    time_axis = np.linspace(0, len(wav) / sample_rate, len(attr))
    ax2.plot(time_axis, attr, color="darkorange", linewidth=1.5)
    ax2.fill_between(time_axis, 0, attr, color="darkorange", alpha=0.35)
    ax2.set_ylabel("Attribution")
    ax2.set_xlabel("Time (seconds)")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_neuron_selectivity_heatmap(
    selectivity_per_layer: Dict[int, np.ndarray],
    top_k: int = 25,
    title: str = "Neuron Activation Profiles (Selectivity Across Layers)",
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot heatmap of top selective neurons across layers."""
    layers = sorted(selectivity_per_layer.keys())
    # Extract top_k most selective neurons per layer
    matrix = []
    for l in layers:
        vec = selectivity_per_layer[l]
        top_indices = np.argsort(np.abs(vec))[-top_k:]
        matrix.append(vec[top_indices])

    heatmap_data = np.array(matrix)  # (num_layers, top_k)

    fig, ax = plt.subplots(figsize=(10, 5))
    cax = ax.imshow(heatmap_data, cmap="coolwarm", aspect="auto", interpolation="nearest")
    fig.colorbar(cax, ax=ax, label="Selectivity Index")

    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"Layer {l}" for l in layers])
    ax.set_xlabel(f"Top {top_k} Selective Neurons (Ranked)")
    ax.set_ylabel("Transformer Layer")
    ax.set_title(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_layer_probing_curve(
    layer_names: List[str],
    accuracies: List[float],
    title: str = "Layer-wise Diagnostic Probe Performance (Acoustic / Phonetic)",
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot diagnostic probe accuracy progression across HuBERT layers."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x_positions = range(len(layer_names))

    ax.plot(x_positions, [acc * 100 for acc in accuracies], marker="o", color="teal", linewidth=2.2, markersize=8)
    for i, acc in enumerate(accuracies):
        ax.annotate(
            f"{acc * 100:.1f}%",
            (x_positions[i], acc * 100),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontweight="bold",
            fontsize=9,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(layer_names)
    ax.set_xlabel("HuBERT Architecture Layer", fontweight="bold")
    ax.set_ylabel("Probe Classification Accuracy (%)", fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_attention_map(
    attention_matrix: torch.Tensor,
    title: str = "HuBERT Multi-Head Attention Map",
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot self-attention weights between time frames."""
    attn = attention_matrix.detach().cpu().numpy()
    if attn.ndim == 3:  # (num_heads, T, T)
        attn = attn.mean(axis=0)  # Average over heads

    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.imshow(attn, cmap="magma", aspect="auto", origin="lower")
    fig.colorbar(cax, ax=ax, label="Attention Weight")

    ax.set_xlabel("Key Time Frames", fontweight="bold")
    ax.set_ylabel("Query Time Frames", fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()
