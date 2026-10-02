"""Unit tests for Phono-V6.6: 4-Path Length-Adaptive Routing & Conditioned MoE Decoder."""

import pytest
import torch

from src.data.roman_tokenizer import RomanCharTokenizer
from src.models.phono_v6_6_adaptive_decoder import (
    AdaptivePathConfig,
    ConditionedMoEFeedForwardNetwork,
    MicroAdaptiveRecursiveHead,
    PhonoV66AdaptiveDecoder,
)
from src.models.registry import ModelRegistry


def test_adaptive_path_config():
    cfg = AdaptivePathConfig.medium(byte_vocab_size=123)
    assert cfg.num_paths == 4
    assert cfg.micro_num_experts == 16
    assert cfg.micro_moe_top_k == 2
    assert cfg.k_short == 5
    assert cfg.k_medium == 9
    assert cfg.k_long == 24
    assert cfg.byte_vocab_size == 123

    cfg_large = AdaptivePathConfig.large()
    assert cfg_large.micro_num_experts == 32
    assert cfg_large.micro_moe_top_k == 4
    assert cfg_large.macro_dim == 768


def test_conditioned_moe_ffn():
    B, seq_len, D = 2, 8, 128
    ffn = ConditionedMoEFeedForwardNetwork(
        embed_dim=D, ffn_dim=256, num_experts=8, top_k=2
    )

    x = torch.randn(B, seq_len, D)
    path_bias = torch.randn(B, 1, D)

    # Without path bias
    out1, aux1 = ffn(x)
    assert out1.shape == (B, seq_len, D)
    assert aux1 >= 0.0

    # With path bias
    out2, aux2 = ffn(x, path_bias=path_bias)
    assert out2.shape == (B, seq_len, D)
    assert aux2 >= 0.0


def test_micro_adaptive_recursive_head():
    cfg = AdaptivePathConfig(
        macro_dim=128,
        micro_dim=64,
        micro_layers=2,
        micro_heads=2,
        micro_ffn_dim=128,
        byte_vocab_size=123,
        micro_num_experts=8,
        micro_moe_top_k=2,
    )
    head = MicroAdaptiveRecursiveHead(cfg)

    # Weight tying verification
    assert head.lm_head.weight is head.char_embedding.weight

    N, K = 4, 8
    input_ids = torch.randint(0, 123, (N, K))
    zw = torch.randn(N, 1, cfg.macro_dim)
    path_bias = torch.randn(N, 1, cfg.micro_dim)

    logits, aux_loss = head(input_ids, zw, path_bias=path_bias)
    assert logits.shape == (N, K, 123)
    assert aux_loss >= 0.0


def test_phono_v6_6_forward_and_backward():
    tok = RomanCharTokenizer()
    cfg = AdaptivePathConfig(
        acoustic_dim=128,
        macro_dim=128,
        micro_dim=64,
        macro_layers=2,
        macro_heads=2,
        macro_ffn_dim=128,
        micro_layers=2,
        micro_heads=2,
        micro_ffn_dim=128,
        byte_vocab_size=tok.vocab_size,
        micro_num_experts=8,
        micro_moe_top_k=2,
        max_bytes_per_word=16,
    )
    model = PhonoV66AdaptiveDecoder(cfg)

    B, T, L, K = 2, 40, 4, 16
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

    # Test backward pass
    out["loss"].backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0.0


def test_phono_v6_6_dynamic_generate():
    tok = RomanCharTokenizer()
    cfg = AdaptivePathConfig(
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
    )
    model = PhonoV66AdaptiveDecoder(cfg)

    acoustic = torch.randn(1, 24, cfg.acoustic_dim)
    words = model.generate(acoustic, max_words=3)

    assert isinstance(words, list)
    for w in words:
        assert isinstance(w, list)
        # Any generated word cannot exceed the longest horizon K_long
        assert len(w) <= cfg.k_long


def test_registry_integration():
    catalog = ModelRegistry.list_models()
    model_ids = [m["id"] for m in catalog]
    assert "phono_v6_6_adaptive_decoder" in model_ids

    cfg = AdaptivePathConfig(
        acoustic_dim=64,
        macro_dim=64,
        micro_dim=32,
        macro_layers=1,
        macro_heads=2,
        macro_ffn_dim=64,
        micro_layers=1,
        micro_heads=2,
        micro_ffn_dim=64,
        byte_vocab_size=123,
    )
    model = ModelRegistry.build_model("phono_v6_6_adaptive_decoder", config=cfg)
    assert isinstance(model, PhonoV66AdaptiveDecoder)
