"""Tests for Phono-V6.7 Windowed Multi-Word MoE Adaptive Decoder."""

import pytest
import torch

from src.data.roman_tokenizer import RomanCharTokenizer
from src.models.phono_v6_6_adaptive_decoder import AdaptivePathConfig, PhonoV66AdaptiveDecoder
from src.models.phono_v6_7_windowed_decoder import (
    WindowedAdaptivePathConfig,
    WindowedMicroRecursiveHead,
    PhonoV67WindowedDecoder,
)
from src.models.registry import ModelRegistry


def test_windowed_adaptive_path_config():
    cfg = WindowedAdaptivePathConfig.medium()
    assert cfg.macro_dim == 512
    assert cfg.micro_dim == 512
    assert cfg.micro_num_experts == 16
    assert cfg.word_context_window == 6
    assert cfg.scheduled_sampling_prob == 0.30

    cfg_large = WindowedAdaptivePathConfig.large()
    assert cfg_large.macro_dim == 768
    assert cfg_large.micro_num_experts == 32
    assert cfg_large.word_context_window == 6


def test_build_word_context_windows():
    cfg = WindowedAdaptivePathConfig(
        macro_dim=64,
        micro_dim=32,
        word_context_window=6,
    )
    model = PhonoV67WindowedDecoder(cfg)

    B, L, D = 2, 5, 64
    z_word = torch.randn(B, L, D)
    windows = model.build_word_context_windows(z_word)

    assert windows.shape == (B, L, 6, D)


def test_windowed_micro_recursive_head():
    cfg = WindowedAdaptivePathConfig(
        micro_dim=32,
        macro_dim=64,
        micro_layers=1,
        micro_heads=2,
        micro_ffn_dim=64,
        byte_vocab_size=122,
        micro_num_experts=4,
        micro_moe_top_k=1,
        word_context_window=6,
    )
    head = WindowedMicroRecursiveHead(cfg)

    N, K, W = 4, 8, 6
    input_ids = torch.randint(0, 122, (N, K))
    zw_window = torch.randn(N, W, cfg.macro_dim)
    path_bias = torch.randn(N, 1, cfg.micro_dim)

    logits, aux_loss = head(input_ids, zw_window, path_bias=path_bias)
    assert logits.shape == (N, K, 122)
    assert aux_loss >= 0.0


def test_phono_v6_7_forward_and_backward():
    tok = RomanCharTokenizer()
    cfg = WindowedAdaptivePathConfig(
        acoustic_dim=64,
        macro_dim=64,
        micro_dim=32,
        macro_layers=1,
        macro_heads=2,
        macro_ffn_dim=64,
        micro_layers=1,
        micro_heads=2,
        micro_ffn_dim=64,
        byte_vocab_size=tok.vocab_size,
        micro_num_experts=4,
        micro_moe_top_k=1,
        max_bytes_per_word=16,
        word_context_window=6,
    )
    model = PhonoV67WindowedDecoder(cfg)

    B, T, L, K = 2, 20, 4, 16
    acoustic = torch.randn(B, T, cfg.acoustic_dim)
    input_ids = torch.randint(0, tok.vocab_size, (B, L, K))
    target_ids = torch.randint(0, tok.vocab_size, (B, L, K))
    path_targets = torch.tensor([[1, 2, 3, 0], [2, 1, 0, 0]], dtype=torch.long)

    out = model(
        acoustic_memory=acoustic,
        input_byte_ids=input_ids,
        target_byte_ids=target_ids,
        path_targets=path_targets,
    )

    assert "loss" in out
    assert out["loss"] is not None
    assert out["logits"].shape == (B, L, K, tok.vocab_size)
    assert out["path_logits"].shape == (B, L, 4)
    assert out["path_acc"] >= 0.0
    assert out["char_acc"] >= 0.0

    # Backward pass verification
    out["loss"].backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0.0


def test_phono_v6_7_dynamic_generate():
    tok = RomanCharTokenizer()
    cfg = WindowedAdaptivePathConfig(
        acoustic_dim=64,
        macro_dim=64,
        micro_dim=32,
        macro_layers=1,
        macro_heads=2,
        macro_ffn_dim=64,
        micro_layers=1,
        micro_heads=2,
        micro_ffn_dim=64,
        byte_vocab_size=tok.vocab_size,
        micro_num_experts=4,
        micro_moe_top_k=1,
        k_short=4,
        k_medium=6,
        k_long=10,
        word_context_window=6,
    )
    model = PhonoV67WindowedDecoder(cfg)

    acoustic = torch.randn(1, 24, cfg.acoustic_dim)
    words = model.generate(acoustic, max_words=3)

    assert isinstance(words, list)
    for w in words:
        assert isinstance(w, list)
        assert len(w) <= cfg.k_long


def test_phono_v6_7_warmstart_from_v6_6():
    cfg_66 = AdaptivePathConfig(
        acoustic_dim=64,
        macro_dim=64,
        micro_dim=32,
        macro_layers=1,
        macro_heads=2,
        macro_ffn_dim=64,
        micro_layers=1,
        micro_heads=2,
        micro_ffn_dim=64,
        byte_vocab_size=122,
        micro_num_experts=4,
        micro_moe_top_k=1,
    )
    model_66 = PhonoV66AdaptiveDecoder(cfg_66)

    cfg_67 = WindowedAdaptivePathConfig(
        acoustic_dim=64,
        macro_dim=64,
        micro_dim=32,
        macro_layers=1,
        macro_heads=2,
        macro_ffn_dim=64,
        micro_layers=1,
        micro_heads=2,
        micro_ffn_dim=64,
        byte_vocab_size=122,
        micro_num_experts=4,
        micro_moe_top_k=1,
        word_context_window=6,
    )
    model_67 = PhonoV67WindowedDecoder(cfg_67)

    # Transfer state dict from V6.6
    sd_66 = model_66.state_dict()
    missing, unexpected = model_67.load_state_dict(sd_66, strict=False)

    # Only word_bos_embedding and word_relative_pos_embedding should be missing
    assert unexpected == []
    assert set(missing) == {"word_bos_embedding", "word_relative_pos_embedding.weight"}


def test_phono_v6_7_registry():
    entry = ModelRegistry.get_entry("phono_v6_7_windowed_decoder")
    assert entry["model_cls"] is PhonoV67WindowedDecoder
    assert entry["config_cls"] is WindowedAdaptivePathConfig
