"""
Conformer Convolution Module, MoE Layers, and Contextual Encoder.
Implements:
- MultiHeadSelfAttention: Standard multi-head self-attention with q/k/v/out projections
- DeepSeekMoEFeedForwardNetwork: 1 Shared Expert + N Routed Experts with top-k gating
- ConformerMoETransformerLayer: Self-Attention + MultiScale Dilated Conv + DeepSeek MoE FFN
- ConformerEngramMoEEncoder: 12-layer contextual encoder with Draft Head and Engram Memory
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .conv_modules import ConformerConvModule, MultiScaleDilatedConformerConvModule
from .embeddings import ConvolutionalPositionalEmbedding
from .engram_memory import PhonemeEngramMemory


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention returning attention weights."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.head_dim: int = embed_dim // num_heads
        self.scaling: float = self.head_dim ** -0.5

        self.q_proj: nn.Linear = nn.Linear(embed_dim, embed_dim)
        self.k_proj: nn.Linear = nn.Linear(embed_dim, embed_dim)
        self.v_proj: nn.Linear = nn.Linear(embed_dim, embed_dim)
        self.out_proj: nn.Linear = nn.Linear(embed_dim, embed_dim)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs_dropped = self.dropout(attn_probs)

        out = torch.matmul(attn_probs_dropped, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)

        return out, (attn_probs if return_attn_weights else None)


class DeepSeekMoEFeedForwardNetwork(nn.Module):
    """DeepSeek-style Mixture of Experts with 1 Shared Expert and N Routed Experts."""

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_routed_experts: int = 8,
        top_k: int = 2,
        num_shared_experts: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim: int = embed_dim
        self.ffn_dim: int = ffn_dim
        self.num_routed_experts: int = num_routed_experts
        self.top_k: int = min(top_k, num_routed_experts)
        self.num_shared_experts: int = num_shared_experts

        if num_shared_experts > 0:
            self.shared_expert: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim),
                nn.Dropout(dropout),
            )
            nn.init.zeros_(self.shared_expert[3].weight)
            nn.init.zeros_(self.shared_expert[3].bias)
        else:
            self.shared_expert = None

        self.router: nn.Linear = nn.Linear(embed_dim, num_routed_experts)

        self.routed_experts: nn.ModuleList = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_routed_experts)
        ])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B, T, C = x.shape
        x_flat = x.view(-1, C)

        shared_out = self.shared_expert(x_flat) if self.shared_expert is not None else 0.0

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        routed_out = torch.zeros_like(x_flat)
        batch_idx = torch.arange(x_flat.shape[0], device=x.device)
        for e_id in range(self.num_routed_experts):
            mask = (topk_indices == e_id)
            token_mask = mask.any(dim=-1)
            idx = batch_idx[token_mask]
            if idx.shape[0] > 0:
                w = (topk_probs * mask.float()).sum(dim=-1, keepdim=True)[token_mask]
                expert_out = self.routed_experts[e_id](x_flat[idx])
                routed_out.index_add_(0, idx, w * expert_out)

        density = (router_probs > (1.0 / self.num_routed_experts)).float().mean(dim=0)
        aux_loss = self.num_routed_experts * torch.sum(density * router_probs.mean(dim=0))

        total_out = shared_out + routed_out
        return total_out.view(B, T, C), aux_loss


class ConformerMoETransformerLayer(nn.Module):
    """
    Conformer Transformer Layer with Self-Attention, Multi-Scale Dilated Conv, and MoE FFN.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        kernel_size: int = 31,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_multi_scale_conv: bool = True,
        use_deepseek_moe: bool = True,
        num_shared_experts: int = 1,
    ) -> None:
        super().__init__()
        self.self_attn: MultiHeadSelfAttention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
        )
        self.self_attn_layer_norm: nn.LayerNorm = nn.LayerNorm(embed_dim)

        if use_multi_scale_conv:
            self.conv_module: nn.Module = MultiScaleDilatedConformerConvModule(
                embed_dim=embed_dim,
                kernel_size=kernel_size,
                dropout=dropout,
            )
        else:
            self.conv_module = ConformerConvModule(
                embed_dim=embed_dim,
                kernel_size=kernel_size,
                dropout=dropout,
            )

        if use_deepseek_moe:
            self.moe_ffn: nn.Module = DeepSeekMoEFeedForwardNetwork(
                embed_dim=embed_dim,
                ffn_dim=ffn_dim,
                num_routed_experts=num_experts,
                top_k=top_k,
                num_shared_experts=num_shared_experts,
                dropout=dropout,
            )
        else:
            self.moe_ffn = DeepSeekMoEFeedForwardNetwork(
                embed_dim=embed_dim,
                ffn_dim=ffn_dim,
                num_routed_experts=num_experts,
                top_k=top_k,
                num_shared_experts=0,
                dropout=dropout,
            )
        self.final_layer_norm: nn.LayerNorm = nn.LayerNorm(embed_dim)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor]:
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

        # 3. MoE Feed-Forward Network
        residual = x
        normed = self.final_layer_norm(x)
        ffn_out, aux_loss = self.moe_ffn(normed)
        x = residual + self.dropout(ffn_out)

        return x, attn_weights, aux_loss


class ConformerEngramMoEEncoder(nn.Module):
    """
    12-layer Conformer Contextual Encoder with Positional Convolutions,
    DeepSeek MoE Layers, Draft Phoneme Head, and Multi-Scale Engram Conditional Memory.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        num_experts: int = 8,
        top_k: int = 2,
        use_deepseek_moe: bool = True,
        num_shared_experts: int = 1,
        kernel_size: int = 31,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        pos_conv_kernel: int = 128,
        pos_conv_groups: int = 16,
        use_multi_scale_conv: bool = True,
        vocab_size: int = 112,
        draft_layer_idx: int = 3,
        engram_layers: Tuple[int, ...] = (4, 7),
        engram_heads: int = 4,
        engram_memory_dim: int = 128,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim: int = embed_dim
        self.vocab_size: int = vocab_size
        self.draft_layer_idx: int = draft_layer_idx
        self.engram_layers: Set[int] = set(engram_layers)
        self.gradient_checkpointing: bool = gradient_checkpointing

        self.pos_conv: ConvolutionalPositionalEmbedding = ConvolutionalPositionalEmbedding(
            embed_dim,
            kernel_size=pos_conv_kernel,
            groups=min(pos_conv_groups, embed_dim),
        )
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(embed_dim)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

        self.layers: nn.ModuleList = nn.ModuleList([
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
                use_deepseek_moe=use_deepseek_moe,
                num_shared_experts=num_shared_experts,
            )
            for _ in range(num_layers)
        ])

        self.draft_head: nn.Linear = nn.Linear(embed_dim, vocab_size)

        self.engram_modules: nn.ModuleDict = nn.ModuleDict({
            f"layer_{l_idx}": PhonemeEngramMemory(
                embed_dim=embed_dim,
                vocab_size=vocab_size,
                num_heads=engram_heads,
                memory_dim=engram_memory_dim,
                dropout=dropout,
            )
            for l_idx in engram_layers
        })

        self.final_layer_norm: nn.LayerNorm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        teacher_tokens: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        pos = self.pos_conv(x)
        x = x + pos
        x = self.layer_norm(x)
        x = self.dropout(x)

        hidden_states = [x] if output_hidden_states else None
        attentions = [] if output_attentions else None
        total_aux_loss = torch.tensor(0.0, device=x.device)
        draft_logits = None
        draft_tokens = None
        engram_gates = {}

        for l_idx, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                def custom_layer_forward(layer_mod, x_in, mask):
                    return layer_mod(x_in, key_padding_mask=mask, return_attn_weights=False)
                x, attn_weights, aux_loss = checkpoint(
                    custom_layer_forward,
                    layer,
                    x,
                    key_padding_mask,
                    use_reentrant=False,
                )
            else:
                x, attn_weights, aux_loss = layer(
                    x,
                    key_padding_mask=key_padding_mask,
                    return_attn_weights=output_attentions,
                )
            total_aux_loss = total_aux_loss + aux_loss

            if l_idx == self.draft_layer_idx:
                draft_logits = self.draft_head(x)
                draft_tokens = torch.argmax(draft_logits, dim=-1)

            if l_idx in self.engram_layers and draft_tokens is not None:
                mod_name = f"layer_{l_idx}"
                if mod_name in self.engram_modules:
                    x, gate = self.engram_modules[mod_name](x, draft_tokens)
                    engram_gates[mod_name] = gate

            if output_hidden_states:
                hidden_states.append(x)
            if output_attentions and attn_weights is not None:
                attentions.append(attn_weights)

        x = self.final_layer_norm(x)

        return {
            "last_hidden_state": x,
            "draft_logits": draft_logits,
            "engram_gates": engram_gates,
            "hidden_states": hidden_states,
            "attentions": attentions,
            "aux_loss": total_aux_loss,
        }
