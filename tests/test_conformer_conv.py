"""Unit tests for Conformer Convolution Module and Layers."""

import pytest
import torch
from src.models.conformer_layers import ConformerConvModule, ConformerTransformerLayer, ConformerMoETransformerEncoder


def test_conformer_conv_module_shapes_and_zero_init():
    B, T, D = 2, 50, 64
    x = torch.randn(B, T, D)
    conv_mod = ConformerConvModule(embed_dim=D, kernel_size=31)

    # Test initial zero contribution for seamless warm start
    out = conv_mod(x)
    assert out.shape == (B, T, D)
    assert torch.allclose(out, torch.zeros_like(out)), "Conformer conv output should be exactly 0 initially"


def test_conformer_conv_module_backward():
    B, T, D = 2, 30, 32
    x = torch.randn(B, T, D, requires_grad=True)
    conv_mod = ConformerConvModule(embed_dim=D, kernel_size=15)

    # Give pointwise_conv2 some initial weights so gradients flow through
    torch.nn.init.normal_(conv_mod.pointwise_conv2.weight, std=0.1)

    out = conv_mod(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert conv_mod.depthwise_conv.weight.grad is not None
    assert conv_mod.depthwise_conv.weight.grad.abs().sum() > 0


def test_conformer_transformer_layer():
    B, T, D = 2, 20, 64
    x = torch.randn(B, T, D)
    layer = ConformerTransformerLayer(embed_dim=D, num_heads=4, ffn_dim=128, kernel_size=15)

    out, attn_w, intermediate = layer(x)
    assert out.shape == (B, T, D)
    assert intermediate.shape == (B, T, 128)


def test_conformer_moe_encoder():
    encoder = ConformerMoETransformerEncoder(
        embed_dim=64,
        num_layers=2,
        num_heads=2,
        ffn_dim=128,
        num_experts=2,
        top_k=1,
        kernel_size=5,
    )
    x = torch.randn(2, 20, 64)
    out = encoder(x)
    assert "last_hidden_state" in out
    assert out["last_hidden_state"].shape == (2, 20, 64)
    assert out["aux_loss"].dim() == 0
