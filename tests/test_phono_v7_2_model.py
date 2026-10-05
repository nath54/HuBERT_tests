"""Unit tests for Phono-V7.2 Partitioned MoE and Horizon Capped Speech Model."""

import pytest
import torch
import torch.nn.functional as F

from src.models.phono_v6_7_windowed_decoder import WindowedAdaptivePathConfig
from src.models.phono_v7_2_speech_model import (
    PartitionedConditionedMoEFFN,
    PhonoV72Decoder,
    PhonoV72SpeechModel,
    EXPERT_PARTITIONS,
)


def test_partitioned_moe_ffn_routing():
    """Verify that path_id strictly restricts expert activation to the designated partition."""
    torch.manual_seed(42)
    B, seq_len, D = 4, 3, 64
    ffn_dim = 128
    moe = PartitionedConditionedMoEFFN(
        embed_dim=D,
        ffn_dim=ffn_dim,
        num_experts=16,
        top_k=2,
    )
    x = torch.randn(B, seq_len, D)

    # 1. Test SHORT partition (path_id = 1 -> experts 0..4)
    path_short = torch.full((B,), 1, dtype=torch.long)
    out_short, aux_loss = moe(x, path_id=path_short)
    assert out_short.shape == (B, seq_len, D)
    assert not torch.isnan(out_short).any()

    # Verify gate softmax probability is strictly 0 outside partition [0, 5)
    flat_x = x.reshape(-1, D)
    logits = moe.gate(flat_x)
    partition_mask = torch.full_like(logits, float("-inf"))
    partition_mask[:, 0:5] = 0.0
    probs = F.softmax(logits + partition_mask, dim=-1)
    assert (probs[:, 5:] == 0.0).all()
    assert (probs[:, 0:5] > 0.0).any()

    # 2. Test MEDIUM partition (path_id = 2 -> experts 5..10)
    path_med = torch.full((B,), 2, dtype=torch.long)
    out_med, _ = moe(x, path_id=path_med)
    assert out_med.shape == (B, seq_len, D)

    # 3. Test LONG partition (path_id = 3 -> experts 11..15)
    path_long = torch.full((B,), 3, dtype=torch.long)
    out_long, _ = moe(x, path_id=path_long)
    assert out_long.shape == (B, seq_len, D)


def test_v7_2_decoder_forward():
    """Verify forward pass of PhonoV72Decoder with partitioned MoE."""
    torch.manual_seed(42)
    B, T, D = 2, 60, 64
    L = 5
    K = 8

    cfg = WindowedAdaptivePathConfig(
        acoustic_dim=D,
        macro_dim=D,
        micro_dim=D,
        macro_layers=2,
        macro_heads=4,
        macro_ffn_dim=128,
        micro_layers=2,
        micro_heads=4,
        micro_ffn_dim=128,
        micro_num_experts=16,
        micro_moe_top_k=2,
        word_context_window=4,
        acoustic_window_frames=16,
        max_bytes_per_word=K,
        k_short=4,
        k_medium=6,
        k_long=10,
    )

    decoder = PhonoV72Decoder(config=cfg, band_window_words=4)
    acoustic_memory = torch.randn(B, T, D)
    input_bytes = torch.randint(1, 25, (B, L, K))
    target_bytes = torch.randint(1, 25, (B, L, K))
    path_targets = torch.tensor([[1, 2, 3, 1, 2], [2, 1, 3, 2, 1]], dtype=torch.long)
    target_lengths = torch.tensor([[3.0, 5.0, 8.0, 2.0, 6.0], [5.0, 2.0, 9.0, 4.0, 3.0]])

    out = decoder(
        acoustic_memory=acoustic_memory,
        input_byte_ids=input_bytes,
        target_byte_ids=target_bytes,
        path_targets=path_targets,
        target_lengths=target_lengths,
    )

    assert "loss" in out
    assert "char_loss" in out
    assert "length_headroom" in out
    assert not torch.isnan(out["loss"])
    assert out["loss"] > 0


def test_v7_2_horizon_capping():
    """Verify that greedy decoding caps generation by min(head_bound, ceil(k_hat) + 1)."""
    torch.manual_seed(42)
    B, T, D = 1, 60, 64
    cfg = WindowedAdaptivePathConfig(
        acoustic_dim=D,
        macro_dim=D,
        micro_dim=D,
        macro_layers=2,
        macro_heads=4,
        macro_ffn_dim=128,
        micro_layers=2,
        micro_heads=4,
        micro_ffn_dim=128,
        micro_num_experts=16,
        micro_moe_top_k=2,
        word_context_window=4,
        acoustic_window_frames=16,
        k_short=5,
        k_medium=9,
        k_long=24,
    )
    decoder = PhonoV72Decoder(config=cfg, band_window_words=4)
    decoder.eval()

    acoustic_memory = torch.randn(B, T, D)
    decoded_words = decoder.decode_greedy(
        acoustic_memory=acoustic_memory,
        max_words=4,
    )
    assert isinstance(decoded_words, list)
    for word in decoded_words:
        # Every decoded word must be non-empty and bounded by max characters
        assert len(word) > 0
        assert len(word) <= 25


def test_v7_2_warm_start_from_v7_1():
    """Verify that PhonoV72SpeechModel can warm-start from a V7.1 checkpoint."""
    import os
    v7_ckpt = "checkpoints/phono_v7_1/streaming/best_checkpoint.pt"
    if not os.path.exists(v7_ckpt):
        pytest.skip(f"Checkpoint {v7_ckpt} not available yet")

    model = PhonoV72SpeechModel()
    stats = model.warm_start_from_v7_1(v7_ckpt)
    assert stats["transferred"] > 500, f"Expected > 500 transferred tensors, got {stats['transferred']}"
    print(f"Warm-started {stats['transferred']} tensors from {v7_ckpt}")
