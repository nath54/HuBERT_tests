"""Conformer Convolution Module and Layer Definitions for Phono-V7.

Implements standard depthwise separable convolution blocks for speech processing
(Gulati et al., 2020) to capture local co-articulation patterns and acoustic transitions:
- Pointwise Conv1d: projects dimension D -> 2*D
- Gated Linear Unit (GLU): activates half the channels
- 1D Depthwise Conv: kernel size = 31 frames (~620ms receptive field)
- BatchNorm1d
- Swish activation (SiLU)
- Pointwise Conv1d: projects back to D
- Dropout

Crucially, the final pointwise projection is initialized to ZERO, ensuring that when
integrated into a pretrained Transformer, the initial convolutional contribution is exactly 0,
enabling 100% seamless, non-destructive warm-start from pretrained checkpoints.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.transformer import MultiHeadSelfAttention, FeedForwardNetwork


class ConformerConvModule(nn.Module):
    """Depthwise separable convolution module with GLU and LayerNorm."""

    def __init__(
        self,
        embed_dim: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for symmetric same-length padding"
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2

        self.layer_norm = nn.LayerNorm(embed_dim)
        # Pointwise Conv 1: D -> 2*D
        self.pointwise_conv1 = nn.Conv1d(embed_dim, 2 * embed_dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)

        # Depthwise Conv: D -> D with groups=D
        self.depthwise_conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=embed_dim,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(embed_dim)
        self.activation = nn.SiLU()

        # Pointwise Conv 2: D -> D
        self.pointwise_conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

        # Zero-initialize the final projection for seamless warm-start!
        nn.init.zeros_(self.pointwise_conv2.weight)
        if self.pointwise_conv2.bias is not None:
            nn.init.zeros_(self.pointwise_conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            out: (B, T, D)
        """
        # 1. Pre-LN
        h = self.layer_norm(x)

        # 2. Transpose for Conv1d: (B, D, T)
        h = h.transpose(1, 2)

        # 3. Pointwise Conv + GLU
        h = self.pointwise_conv1(h)
        h = self.glu(h)  # (B, D, T)

        # 4. Depthwise Conv + BatchNorm + Swish
        h = self.depthwise_conv(h)
        h = self.batch_norm(h)
        h = self.activation(h)

        # 5. Pointwise Conv 2 + Dropout
        h = self.pointwise_conv2(h)
        h = self.dropout(h)

        # 6. Transpose back: (B, T, D)
        return h.transpose(1, 2)


class MultiScaleDilatedConformerConvModule(nn.Module):
    """Multi-Scale Dilated Depthwise Separable Convolution for Phono-V7.3.

    Expands temporal receptive field from 620ms to 2,420ms without increasing parameters:
    - Branch 1: kernel=15, dilation=1 -> 300ms (sharp phone transitions)
    - Branch 2: kernel=31, dilation=2 -> 1,220ms (intra-word coarticulation)
    - Branch 3: kernel=31, dilation=4 -> 2,420ms (inter-word cadence & prosody)
    """

    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Pointwise Conv 1: D -> 2*D
        self.pointwise_conv1 = nn.Conv1d(embed_dim, 2 * embed_dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)

        # Split embed_dim into 3 branches
        d1 = embed_dim // 3
        d2 = embed_dim // 3
        d3 = embed_dim - (d1 + d2)
        self.branch_dims = (d1, d2, d3)

        # Branch 1: k=15, d=1 (pad = 7)
        self.conv_b1 = nn.Conv1d(d1, d1, kernel_size=15, padding=7, dilation=1, groups=d1, bias=False)
        # Branch 2: k=31, d=2 (pad = 30)
        self.conv_b2 = nn.Conv1d(d2, d2, kernel_size=31, padding=30, dilation=2, groups=d2, bias=False)
        # Branch 3: k=31, d=4 (pad = 60)
        self.conv_b3 = nn.Conv1d(d3, d3, kernel_size=31, padding=60, dilation=4, groups=d3, bias=False)

        self.batch_norm = nn.BatchNorm1d(embed_dim)
        self.activation = nn.SiLU()

        # Pointwise Conv 2: D -> D
        self.pointwise_conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

        # Zero-initialize final projection for seamless warm-start!
        nn.init.zeros_(self.pointwise_conv2.weight)
        if self.pointwise_conv2.bias is not None:
            nn.init.zeros_(self.pointwise_conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer_norm(x)
        h = h.transpose(1, 2)  # (B, D, T)
        h = self.pointwise_conv1(h)
        h = self.glu(h)  # (B, D, T)

        # Multi-scale branch split & conv
        d1, d2, d3 = self.branch_dims
        h1 = self.conv_b1(h[:, :d1, :])
        h2 = self.conv_b2(h[:, d1 : d1 + d2, :])
        h3 = self.conv_b3(h[:, d1 + d2 :, :])
        h = torch.cat([h1, h2, h3], dim=1)

        h = self.batch_norm(h)
        h = self.activation(h)
        h = self.pointwise_conv2(h)
        h = self.dropout(h)
        return h.transpose(1, 2)


class ConformerTransformerLayer(nn.Module):
    """Transformer Layer augmented with Conformer Depthwise Separable Convolution."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        kernel_size: int = 31,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)

        self.conv_module = ConformerConvModule(
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.ffn = FeedForwardNetwork(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        # 1. Multi-Head Self-Attention
        residual = x
        normed = self.self_attn_layer_norm(x)
        attn_out, attn_weights = self.self_attn(
            normed,
            key_padding_mask=key_padding_mask,
            return_attn_weights=return_attn_weights,
        )
        x = residual + self.dropout(attn_out)

        # 2. Conformer Convolution Module
        x = x + self.conv_module(x)

        # 3. Feed-Forward Network
        residual = x
        normed = self.final_layer_norm(x)
        ffn_out, intermediate_neurons = self.ffn(normed)
        x = residual + ffn_out

        return x, attn_weights, intermediate_neurons


class ConformerMoETransformerLayer(nn.Module):
    """Transformer Layer with Multi-Head Self-Attention, Conformer Conv Module, and MoE FFN.

    Maintains 100% parameter name compatibility with MoETransformerLayer from V6.1-V6.4,
    enabling non-destructive weight transfer while adding depthwise convolution.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        kernel_size: int = 31,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_multi_scale_conv: bool = False,
    ):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)

        if use_multi_scale_conv:
            self.conv_module = MultiScaleDilatedConformerConvModule(
                embed_dim=embed_dim,
                dropout=dropout,
            )
        else:
            self.conv_module = ConformerConvModule(
                embed_dim=embed_dim,
                kernel_size=kernel_size,
                dropout=dropout,
            )

        from src.models.phono_variants import MoEFeedForwardNetwork
        self.moe_ffn = MoEFeedForwardNetwork(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        # 1. Multi-Head Self-Attention
        residual = x
        normed = self.self_attn_layer_norm(x)
        attn_out, attn_weights = self.self_attn(
            normed,
            key_padding_mask=key_padding_mask,
            return_attn_weights=return_attn_weights,
        )
        x = residual + self.dropout(attn_out)

        # 2. Conformer Depthwise Convolution Module
        x = x + self.conv_module(x)

        # 3. MoE Feed-Forward Network
        residual = x
        normed = self.final_layer_norm(x)
        ffn_out, aux_loss = self.moe_ffn(normed)
        x = residual + self.dropout(ffn_out)

        return x, attn_weights, aux_loss


class ConformerMoETransformerEncoder(nn.Module):
    """Transformer Contextual Encoder with Positional Convolutions and Conformer MoE Layers."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        kernel_size: int = 31,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        pos_conv_kernel: int = 128,
        pos_conv_groups: int = 16,
        use_multi_scale_conv: bool = False,
    ):
        super().__init__()
        from src.models.transformer import ConvolutionalPositionalEmbedding
        self.pos_conv = ConvolutionalPositionalEmbedding(
            embed_dim,
            kernel_size=pos_conv_kernel,
            groups=min(pos_conv_groups, embed_dim),
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            ConformerMoETransformerLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                num_experts=num_experts,
                top_k=top_k,
                kernel_size=kernel_size,
                dropout=dropout,
                attention_dropout=attention_dropout,
                use_multi_scale_conv=use_multi_scale_conv,
            )
            for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        pos = self.pos_conv(x)
        x = x + pos
        x = self.layer_norm(x)
        x = self.dropout(x)

        hidden_states = [x] if output_hidden_states else None
        attentions = [] if output_attentions else None
        total_aux_loss = torch.tensor(0.0, device=x.device)

        for layer in self.layers:
            x, attn_weights, aux_loss = layer(
                x,
                key_padding_mask=key_padding_mask,
                return_attn_weights=output_attentions,
            )
            total_aux_loss = total_aux_loss + aux_loss
            if output_hidden_states:
                hidden_states.append(x)
            if output_attentions and attn_weights is not None:
                attentions.append(attn_weights)

        x = self.final_layer_norm(x)

        return {
            "last_hidden_state": x,
            "hidden_states": hidden_states,
            "attentions": attentions,
            "aux_loss": total_aux_loss,
        }

