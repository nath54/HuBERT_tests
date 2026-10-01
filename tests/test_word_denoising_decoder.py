import torch
import pytest
from src.models.word_denoising_decoder import (
    WordDecoderConfig,
    BandedCrossAttention,
    MoEWordDecoderLayer,
    WordDenoisingDecoder,
)


def test_banded_cross_attention():
    query_dim = 128
    key_dim = 256
    layer = BandedCrossAttention(query_dim=query_dim, key_dim=key_dim, num_heads=4, band_width=10)
    
    query = torch.randn(2, 8, query_dim)
    key_val = torch.randn(2, 40, key_dim)
    
    out = layer(query, key_val)
    assert out.shape == query.shape


def test_moe_word_decoder_layer():
    cfg = WordDecoderConfig(
        word_embed_dim=128,
        acoustic_embed_dim=256,
        decoder_heads=4,
        decoder_ffn_dim=256,
        num_experts=4,
        moe_top_k=2,
        cross_attn_band_width=10,
    )
    layer = MoEWordDecoderLayer(cfg)
    
    x = torch.randn(2, 8, cfg.word_embed_dim)
    mem = torch.randn(2, 30, cfg.acoustic_embed_dim)
    out, aux_loss = layer(x, mem)
    assert out.shape == x.shape
    assert aux_loss.item() >= 0.0


def test_word_denoising_decoder_forward():
    cfg = WordDecoderConfig(
        vocab_size=1000,
        word_embed_dim=64,
        acoustic_embed_dim=128,
        decoder_layers=2,
        decoder_heads=4,
        decoder_ffn_dim=128,
        num_experts=4,
        moe_top_k=2,
        cross_attn_band_width=8,
        use_word_diffusion=True,
    )
    decoder = WordDenoisingDecoder(cfg)
    
    B, L, T = 2, 6, 24
    word_ids = torch.randint(5, 1000, (B, L))
    acoustic_latents = torch.randn(B, T, cfg.acoustic_embed_dim)
    
    # Forward with targets
    out = decoder(word_ids, acoustic_latents, target_word_ids=word_ids)
    assert 'logits' in out
    assert 'loss' in out
    assert 'diff_loss' in out
    assert 'aux_loss' in out
    assert 'contrastive_loss' in out
    assert out['logits'].shape == (B, L, cfg.vocab_size)
    assert not torch.isnan(out['loss'])

    # Forward inference without targets
    decoder.eval()
    infer_out = decoder(word_ids, acoustic_latents)
    assert 'logits' in infer_out
    assert infer_out['logits'].shape == (B, L, cfg.vocab_size)

    # Autoregressive generation
    gen_tokens = decoder.generate(acoustic_latents, max_len=10)
    assert gen_tokens.shape[0] == B
    assert gen_tokens.shape[1] <= 11
    assert (gen_tokens[:, 0] == cfg.bos_token_id).all()

