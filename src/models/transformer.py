"""Transformer Encoder with Convolutional Positional Embeddings for HuBERT."""

import math
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalPositionalEmbedding(nn.Module):
    """Convolutional Positional Embedding as used in wav2vec 2.0 / HuBERT."""

    def __init__(self, embed_dim: int, kernel_size: int = 128, groups: int = 16):
        super().__init__()
        self.conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )
        self.conv = nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, T, C)
        Returns:
            Positional embedding tensor of shape (B, T, C)
        """
        # (B, T, C) -> (B, C, T)
        x_conv = x.transpose(1, 2)
        x_conv = self.conv(x_conv)
        # Handle odd kernel size padding mismatch if needed
        if x_conv.size(2) > x.size(1):
            x_conv = x_conv[:, :, : x.size(1)]
        x_conv = self.activation(x_conv)
        return x_conv.transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention returning attention weights for interpretability."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling  # (B, H, T, T)

        if key_padding_mask is not None:
            # key_padding_mask: (B, T), True indicates padding
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs_dropped = self.dropout(attn_probs)

        out = torch.matmul(attn_probs_dropped, v)  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)

        return out, (attn_probs if return_attn_weights else None)


class FeedForwardNetwork(nn.Module):
    """Feed-Forward Network exposing intermediate neuron activations."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

        # Storage for hooked neuron activations (useful for NAPS)
        self.last_intermediate_activation = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            out: (B, T, embed_dim)
            intermediate: (B, T, ffn_dim) before fc2
        """
        h = self.fc1(x)
        h = self.activation(h)
        self.last_intermediate_activation = h  # Cache for probing / NAPS
        h_dropped = self.dropout1(h)
        out = self.fc2(h_dropped)
        out = self.dropout2(out)
        return out, h


class HuBERTTransformerLayer(nn.Module):
    """Pre-LN Transformer Layer with explicit intermediate hook points."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        self.ffn = FeedForwardNetwork(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        # Pre-LN Self-Attention
        residual = x
        normed = self.self_attn_layer_norm(x)
        attn_out, attn_weights = self.self_attn(
            normed,
            key_padding_mask=key_padding_mask,
            return_attn_weights=return_attn_weights,
        )
        x = residual + self.dropout(attn_out)

        # Pre-LN FFN
        residual = x
        normed = self.final_layer_norm(x)
        ffn_out, intermediate_neurons = self.ffn(normed)
        x = residual + ffn_out

        return x, attn_weights, intermediate_neurons


class HuBERTEncoder(nn.Module):
    """Stack of HuBERT Transformer layers with pos-conv embedding."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        pos_conv_kernel: int = 128,
        pos_conv_groups: int = 16,
    ):
        super().__init__()
        self.pos_conv = ConvolutionalPositionalEmbedding(
            embed_dim=embed_dim,
            kernel_size=pos_conv_kernel,
            groups=pos_conv_groups,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                HuBERTTransformerLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
    ) -> Dict[str, any]:
        """
        Args:
            x: (B, T, embed_dim)
            key_padding_mask: (B, T)
        Returns:
            Dictionary with:
                - 'last_hidden_state': (B, T, embed_dim)
                - 'hidden_states': List of all layer outputs (L+1 tensors)
                - 'attentions': List of attention weight tensors (if requested)
                - 'neuron_activations': List of intermediate FFN activations (L tensors)
        """
        # Add convolutional positional embeddings
        pos_emb = self.pos_conv(x)
        x = x + pos_emb
        x = self.layer_norm(x)
        x = self.dropout(x)

        hidden_states = [x] if output_hidden_states else None
        attentions = [] if output_attentions else None
        neuron_activations = []

        for layer in self.layers:
            x, attn_weights, ffn_intermediate = layer(
                x,
                key_padding_mask=key_padding_mask,
                return_attn_weights=output_attentions,
            )
            if output_hidden_states:
                hidden_states.append(x)
            if output_attentions:
                attentions.append(attn_weights)
            neuron_activations.append(ffn_intermediate)

        return {
            "last_hidden_state": x,
            "hidden_states": hidden_states,
            "attentions": attentions,
            "neuron_activations": neuron_activations,
        }
