"""
Modular Conformer Block (Macaron Architecture).
Combines:
1. Feed-Forward Module 1 (half-step Macaron style: 0.5 * FFN1)
2. Multi-Head Self-Attention Module with RoPE relative positional encoding
3. Depthwise Separable Convolution Module
4. Feed-Forward Module 2 (half-step Macaron style: 0.5 * FFN2 or DeepSeek MoE)
5. Final LayerNorm with residual additions
"""

# Import Modules
from typing import Optional, Tuple
import torch
from torch import nn, Tensor

from .attention import MultiHeadSelfAttention
from .conv_modules import ConformerConvModule
from .ffn import FeedForwardNetwork, DeepSeekMoEFFN


class ConformerBlock(nn.Module):
    """
    Standard Conformer Layer (Gulati et al., 2020) with Macaron-style dual FFN.
    Combines self-attention for global temporal context with depthwise convolution
    for local acoustic transitions. Supports DeepSeek MoE in the second FFN stage.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        use_deepseek_moe: bool = True,
        num_routed_experts: int = 8,
        top_k: int = 2,
        num_shared_experts: int = 1,
        use_rope: bool = True,
    ) -> None:
        """
        Initialize ConformerBlock.

        Args:
            embed_dim (int): Model hidden dimension D (512).
            num_heads (int): Number of attention heads H (8).
            ffn_dim (int): FFN inner hidden dimension (2048).
            conv_kernel_size (int): Temporal kernel size for depthwise convolution (31).
            dropout (float): Dropout probability.
            use_deepseek_moe (bool): Whether to use DeepSeek MoE in FFN2.
            num_routed_experts (int): Number of routed MoE experts (8).
            top_k (int): Number of active experts per token (2).
            num_shared_experts (int): Number of shared experts (1).
            use_rope (bool): Whether to use Rotary Position Embedding in MHSA.
        """
        # Parent initialization
        super().__init__()
        self.embed_dim: int = embed_dim
        self.use_deepseek_moe: bool = use_deepseek_moe

        # 1. Macaron FFN 1 (Pre-LN)
        self.ffn1_norm: nn.LayerNorm = nn.LayerNorm(embed_dim)
        self.ffn1: FeedForwardNetwork = FeedForwardNetwork(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        # 2. Multi-Head Self-Attention (Pre-LN)
        self.self_attn_norm: nn.LayerNorm = nn.LayerNorm(embed_dim)
        self.self_attn: MultiHeadSelfAttention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope,
        )

        # 3. Convolution Module (Internal Pre-LN handled in module)
        self.conv_module: ConformerConvModule = ConformerConvModule(
            embed_dim=embed_dim,
            kernel_size=conv_kernel_size,
            dropout=dropout,
        )

        # 4. Macaron FFN 2 (Pre-LN) - Standard FFN or DeepSeek MoE
        self.ffn2_norm: nn.LayerNorm = nn.LayerNorm(embed_dim)
        if use_deepseek_moe:
            self.ffn2: nn.Module = DeepSeekMoEFFN(
                embed_dim=embed_dim,
                ffn_dim=ffn_dim,
                num_routed_experts=num_routed_experts,
                top_k=top_k,
                num_shared_experts=num_shared_experts,
                dropout=dropout,
            )
        else:
            self.ffn2 = FeedForwardNetwork(
                embed_dim=embed_dim,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )

        # 5. Final LayerNorm
        self.final_norm: nn.LayerNorm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        """
        Forward pass through Macaron Conformer block.

        Args:
            x (Tensor): Input tensor of shape [B, T, D].
                        Type: torch.float32.
                        Role: Sequence frame features entering Conformer layer.
            key_padding_mask (Optional[Tensor]): Boolean padding mask of shape [B, T].
            return_attn_weights (bool): Whether to return self-attention weights.

        Returns:
            Tuple[Tensor, Optional[Tensor], Tensor]:
                - out: Transformed sequence features [B, T, D].
                - attn_weights: Attention matrix [B, H, T, T] if requested else None.
                - aux_loss: Scalar MoE load-balancing loss [1] (0.0 if MoE disabled).
        """
        # --- Stage 1: Macaron FFN 1 (0.5x scaling) ---
        # Pre-LN: [B, T, D]
        h1: Tensor = self.ffn1_norm(x)
        # FFN1 forward: [B, T, D]
        ffn1_out: Tensor = self.ffn1(h1)
        # Residual addition with 0.5 Macaron factor: [B, T, D]
        x = x + 0.5 * ffn1_out

        # --- Stage 2: Multi-Head Self-Attention ---
        # Pre-LN: [B, T, D]
        h2: Tensor = self.self_attn_norm(x)
        # MHSA forward: [B, T, D]
        attn_out, attn_weights = self.self_attn(
            h2,
            key_padding_mask=key_padding_mask,
            return_weights=return_attn_weights,
        )
        # Residual addition: [B, T, D]
        x = x + attn_out

        # --- Stage 3: Conformer Convolution Module ---
        # Conv module has internal LayerNorm and residual zero-init
        conv_out: Tensor = self.conv_module(x)
        # Residual addition: [B, T, D]
        x = x + conv_out

        # --- Stage 4: Macaron FFN 2 / DeepSeek MoE (0.5x scaling) ---
        # Pre-LN: [B, T, D]
        h4: Tensor = self.ffn2_norm(x)
        if self.use_deepseek_moe:
            # DeepSeek MoE returns (output, aux_loss): [B, T, D] and [1]
            ffn2_out, aux_loss = self.ffn2(h4)
        else:
            ffn2_out = self.ffn2(h4)
            aux_loss = torch.tensor(0.0, device=x.device)

        # Residual addition with 0.5 Macaron factor: [B, T, D]
        x = x + 0.5 * ffn2_out

        # --- Stage 5: Final LayerNorm ---
        out: Tensor = self.final_norm(x)

        return out, attn_weights, aux_loss
