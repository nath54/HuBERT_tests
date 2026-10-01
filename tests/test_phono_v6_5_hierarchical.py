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

    logits, aux_loss = head(byte_ids, word_latents)
    assert logits.shape == (N, K, 261)
    assert aux_loss.item() >= 0.0


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
    # Medium Tier: 16 experts in micro head
    cfg_med = HierarchicalByteConfig.medium()
    model_med = PhonoV65HierarchicalByteDecoder(cfg_med)

    emb_bytes = model_med.micro_byte_head.byte_embedding.weight.numel() * 2  # FP16
    assert emb_bytes < 300 * 1024, f"Embedding table exceeds 300 KB limit: {emb_bytes / 1024:.2f} KB"

    params_med = sum(p.numel() for p in model_med.parameters() if p.requires_grad)
    assert 130_000_000 < params_med < 160_000_000, f"Medium params {params_med} unexpected"
    assert cfg_med.micro_num_experts == 16
    assert cfg_med.micro_moe_top_k == 2


def test_phono_v6_5_large_tier():
    # Large Tier: 32 fine-grained experts with Top-4 routing
    cfg_large = HierarchicalByteConfig.large()
    model_large = PhonoV65HierarchicalByteDecoder(cfg_large)

    assert cfg_large.micro_num_experts == 32
    assert cfg_large.micro_moe_top_k == 4
    assert cfg_large.micro_ffn_dim == 768  # Fine-grained
    assert cfg_large.macro_dim == 768

    params_large = sum(p.numel() for p in model_large.parameters() if p.requires_grad)
    assert 400_000_000 < params_large < 500_000_000, f"Large params {params_large} unexpected"


def test_phono_v6_5_macro_fastpath():
    from src.models.phono_v6_5_hierarchical_decoder import SLOT_SILENCE, SLOT_WORD, SLOT_EOS

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
        enable_macro_fastpath=True,
    )
    model = PhonoV65HierarchicalByteDecoder(cfg)
    assert model.macro_slot_classifier is not None

    B, T, L, K = 2, 30, 4, 6
    acoustic = torch.randn(B, T, cfg.acoustic_dim)
    in_bytes = torch.randint(0, 261, (B, L, K))
    tgt_bytes = torch.randint(0, 261, (B, L, K))
    slot_targets = torch.tensor([[SLOT_WORD, SLOT_WORD, SLOT_EOS, SLOT_SILENCE],
                                 [SLOT_WORD, SLOT_EOS, SLOT_SILENCE, SLOT_SILENCE]], dtype=torch.long)

    out = model(
        acoustic_memory=acoustic,
        input_byte_ids=in_bytes,
        target_byte_ids=tgt_bytes,
        slot_targets=slot_targets,
    )

    assert "slot_logits" in out
    assert "slot_loss" in out
    assert out["slot_logits"].shape == (B, L, cfg.num_slot_classes)
    assert out["slot_loss"].item() >= 0.0

    # Test Early Exit in generate()
    # Force slot classifier to predict SLOT_EOS at step 1
    model.eval()
    with torch.no_grad():
        # Set weights such that SLOT_EOS always wins
        model.macro_slot_classifier.weight.zero_()
        model.macro_slot_classifier.bias.zero_()
        model.macro_slot_classifier.bias[SLOT_EOS] = 100.0  # Force EOS

        gen_out = model.generate(acoustic_memory=acoustic, max_words=10)
        # Since first slot immediately triggers EOS, 0 words should be emitted
        assert len(gen_out[0]) == 0

