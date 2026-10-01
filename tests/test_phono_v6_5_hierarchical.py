"""Unit tests for Phono-V6.5 Hierarchical Word-to-Byte Recursive Denoising Decoder."""

import pytest
import torch

from src.data.byte_tokenizer import ByteTokenizer
from src.models.phono_v6_5_hierarchical_decoder import (
    HierarchicalByteConfig,
    MicroRecursiveByteHead,
    PhonoV65HierarchicalByteDecoder,
)


def test_byte_tokenizer_multilingual():
    tok = ByteTokenizer()
    assert tok.vocab_size == 261

    test_sentences = [
        "he hoped there would be stew for dinner",
        "le renard brun rapide saute par-dessus le chien endormi",
        "el rápido zorro marrón salta sobre el perro perezoso",
        "der schnelle braune Fuchs springt über den faulen Hund",
        "你好世界，这是一个测试",
        "مرحبا بالعالم",
        "안녕하세요 세계",
        "Edge computing 🚀 AudioLearn 100% 🎯",
    ]

    for sent in test_sentences:
        enc = tok.encode(sent, add_bos=True, add_eos=True)
        assert enc[0] == tok.bos_id
        assert enc[-1] == tok.eos_id
        dec = tok.decode(enc, skip_special=True)
        assert dec == sent, f"Mismatch: '{sent}' != '{dec}'"

    # Test word-level encoding
    words_enc = tok.encode_words("hello world test")
    assert len(words_enc) == 3
    for w_seq in words_enc:
        assert w_seq[-1] == tok.eow_id


def test_micro_recursive_byte_head():
    cfg = HierarchicalByteConfig(
        macro_dim=128,
        micro_dim=64,
        micro_layers=1,
        micro_heads=2,
        micro_ffn_dim=128,
        byte_vocab_size=261,
    )
    head = MicroRecursiveByteHead(cfg)

    # Verify weight tying
    assert head.lm_head.weight is head.byte_embedding.weight

    N, K = 4, 8
    byte_ids = torch.randint(0, 261, (N, K))
    word_latents = torch.randn(N, 1, cfg.macro_dim)

    logits = head(byte_ids, word_latents)
    assert logits.shape == (N, K, 261)


def test_phono_v6_5_hierarchical_forward():
    cfg = HierarchicalByteConfig(
        macro_dim=64,
        acoustic_dim=128,
        macro_layers=2,
        macro_heads=4,
        macro_ffn_dim=128,
        micro_dim=64,
        micro_layers=1,
        micro_heads=2,
        micro_ffn_dim=128,
        cross_attn_band_width=8,
        use_word_diffusion=True,
    )
    model = PhonoV65HierarchicalByteDecoder(cfg)

    B, T, L, K = 2, 30, 4, 6
    acoustic = torch.randn(B, T, cfg.acoustic_dim)
    in_bytes = torch.randint(0, 261, (B, L, K))
    tgt_bytes = torch.randint(0, 261, (B, L, K))

    out = model(acoustic_memory=acoustic, input_byte_ids=in_bytes, target_byte_ids=tgt_bytes)
    assert "loss" in out
    assert "logits" in out
    assert "word_latents" in out
    assert "byte_acc" in out
    assert "diff_loss" in out
    assert "aux_loss" in out
    assert not torch.isnan(out["loss"])
    assert out["word_latents"].shape == (B, L, cfg.macro_dim)
    assert out["logits"].shape == (B, L, K, cfg.byte_vocab_size)


def test_phono_v6_5_hierarchical_generate():
    cfg = HierarchicalByteConfig(
        macro_dim=64,
        acoustic_dim=128,
        macro_layers=2,
        macro_heads=4,
        macro_ffn_dim=128,
        micro_dim=64,
        micro_layers=1,
        micro_heads=2,
        micro_ffn_dim=128,
        cross_attn_band_width=8,
    )
    model = PhonoV65HierarchicalByteDecoder(cfg)
    model.eval()

    B, T = 2, 40
    acoustic = torch.randn(B, T, cfg.acoustic_dim)

    gen_results = model.generate(acoustic_memory=acoustic, max_words=3, max_bytes_per_word=6)
    assert len(gen_results) == B
    assert len(gen_results[0]) <= 3


def test_phono_v6_5_memory_footprint():
    cfg = HierarchicalByteConfig()
    model = PhonoV65HierarchicalByteDecoder(cfg)
    
    emb_bytes = model.micro_byte_head.byte_embedding.weight.numel() * 2  # FP16
    assert emb_bytes < 150 * 1024, f"Embedding table exceeds 150 KB limit: {emb_bytes / 1024:.2f} KB"
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params < 45_000_000, f"Total parameters exceed budget: {total_params}"
