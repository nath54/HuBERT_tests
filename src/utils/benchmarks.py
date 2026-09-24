"""Benchmarking, parameter counting, and computational scaling analysis for HuBERT."""

import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn

from src.models.config import HuBERTConfig
from src.models.hubert_asr import HuBERTForCTC


def get_model_parameters_breakdown(model: HuBERTForCTC) -> Dict[str, any]:
    """Compute detailed layer-by-layer parameter count and memory footprint."""
    breakdown = []

    # 1. Feature Extractor (CNN)
    cnn_params = sum(p.numel() for p in model.feature_extractor.parameters())
    breakdown.append({
        "component": "Feature Extractor (7-Layer CNN)",
        "params": cnn_params,
        "description": "Temporal 1D Convolutions with 320x downsampling factor",
    })

    # 2. Feature Projection
    proj_params = sum(p.numel() for p in model.feature_projection.parameters())
    breakdown.append({
        "component": "Feature Projection & LayerNorm",
        "params": proj_params,
        "description": "Projects conv channel dim to Transformer embedding dim",
    })

    # 3. Positional Conv Embedding
    pos_params = sum(p.numel() for p in model.encoder.pos_conv.parameters())
    breakdown.append({
        "component": "Convolutional Positional Embedding",
        "params": pos_params,
        "description": "Depthwise 1D Conv (kernel 64/128, groups 16)",
    })

    # 4. Transformer Layers
    total_transformer_params = 0
    num_layers = len(model.encoder.layers)
    for i, layer in enumerate(model.encoder.layers):
        attn_p = sum(p.numel() for p in layer.self_attn.parameters())
        ffn_p = sum(p.numel() for p in layer.ffn.parameters())
        ln_p = sum(p.numel() for p in layer.self_attn_layer_norm.parameters()) + sum(p.numel() for p in layer.final_layer_norm.parameters())
        layer_total = attn_p + ffn_p + ln_p
        total_transformer_params += layer_total
        breakdown.append({
            "component": f"Transformer Layer {i+1}",
            "params": layer_total,
            "description": f"MHSA: {attn_p:,} | FFN: {ffn_p:,} | Norms: {ln_p:,}",
        })

    # 5. CTC Output Head
    ctc_params = sum(p.numel() for p in model.ctc_head.parameters())
    breakdown.append({
        "component": "CTC Prediction Head",
        "params": ctc_params,
        "description": f"Linear projection to {model.config.vocab_size} character logits",
    })

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Memory footprint in MB
    fp32_mb = (total_params * 4) / (1024 * 1024)
    fp16_mb = (total_params * 2) / (1024 * 1024)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "fp32_memory_mb": round(fp32_mb, 2),
        "fp16_memory_mb": round(fp16_mb, 2),
        "components": breakdown,
    }


def benchmark_inference_speed(
    model: HuBERTForCTC,
    device: torch.device,
    durations: List[float] = [0.5, 1.0, 2.0, 5.0],
    num_runs: int = 5,
    sample_rate: int = 16000,
) -> List[Dict[str, any]]:
    """Measure live inference latency, Real-Time Factor (RTF), and throughput."""
    model.eval()
    results = []

    # Warmup
    dummy = torch.randn(1, 16000).to(device)
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    for dur in durations:
        n_samples = int(dur * sample_rate)
        audio = torch.randn(1, n_samples).to(device)

        latencies = []
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(audio)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

        avg_latency_ms = float(sum(latencies) / len(latencies))
        avg_latency_sec = avg_latency_ms / 1000.0

        # RTF = Latency (s) / Audio Duration (s)
        # An RTF < 1.0 means faster than real-time (e.g. 0.05 = 20x real-time speed)
        rtf = avg_latency_sec / dur
        throughput = dur / avg_latency_sec

        results.append({
            "audio_duration_sec": dur,
            "num_samples": n_samples,
            "latency_ms": round(avg_latency_ms, 2),
            "rtf": round(rtf, 4),
            "speedup_factor": round(throughput, 1),
            "throughput_audio_sec_per_sec": round(throughput, 1),
        })

    return results


def compute_scaling_analysis(
    config: HuBERTConfig,
    durations: List[float] = [1.0, 5.0, 10.0, 30.0, 60.0],
) -> List[Dict[str, any]]:
    """Compute mathematical scaling laws for CNN linear stage vs Transformer quadratic stage."""
    scaling_data = []

    for dur in durations:
        audio_samples = int(dur * config.sample_rate)
        frames = config.compute_output_length(audio_samples)

        # CNN FLOPs: O(T_audio * sum(kernel * channels_in * channels_out / stride))
        cnn_flops = 0
        in_c = config.in_channels
        curr_len = audio_samples
        for out_c, k, s in config.conv_layers:
            curr_len = (curr_len - k) // s + 1
            # 2 FLOPs per multiply-add
            cnn_flops += 2 * in_c * out_c * k * curr_len
            in_c = out_c

        # Transformer FLOPs:
        # Per layer:
        # QKV: 3 * 2 * frames * D^2
        # Attn scores: 2 * frames^2 * D  <-- Quadratic bottleneck!
        # Attn context: 2 * frames^2 * D <-- Quadratic bottleneck!
        # Out proj: 2 * frames * D^2
        # FFN: 2 * 2 * frames * D * FFN_DIM
        D = config.encoder_embed_dim
        FFN = config.encoder_ffn_dim
        L = config.encoder_layers

        attn_quadratic_flops = L * (4 * (frames ** 2) * D)
        linear_transformer_flops = L * (frames * (8 * (D ** 2) + 4 * D * FFN))
        transformer_flops = attn_quadratic_flops + linear_transformer_flops

        total_flops = cnn_flops + transformer_flops
        gflops = total_flops / 1e9

        pct_cnn = (cnn_flops / total_flops) * 100
        pct_transformer = (transformer_flops / total_flops) * 100

        scaling_data.append({
            "duration_sec": dur,
            "audio_samples": audio_samples,
            "downsampled_frames": frames,
            "gflops": round(gflops, 3),
            "pct_cnn_linear": round(pct_cnn, 1),
            "pct_transformer_quadratic": round(pct_transformer, 1),
            "bottleneck_stage": "CNN (Feature Extraction)" if pct_cnn > pct_transformer else "Transformer (Self-Attention)",
        })

    return scaling_data
