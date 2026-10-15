"""
Modular Multi-Head Attention Modules for Speech and Phoneme Processing.
Includes:
- MultiHeadSelfAttention: Scaled dot-product self-attention with RoPE support
- BandCausalSelfAttention: Local band-causal self-attention for word-level context
- AttentionPooling: Query-guided attention pooling for sequence-to-vector compression
"""

# Import Modules
import math
from typing import Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .embeddings import RotaryEmbedding


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) (Vaswani et al., 2017).
    Projects input representations into H parallel attention subspaces,
    computes scaled dot-product attention scores, and optionally integrates
    Rotary Position Embeddings (RoPE) for relative sequence awareness.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_rope: bool = True,
        max_seq_len: int = 4096,
    ) -> None:
        """
        Initialize MultiHeadSelfAttention.

        Args:
            embed_dim (int): Model hidden dimension D (e.g. 512).
            num_heads (int): Number of attention heads H (e.g. 8).
            dropout (float): Attention map dropout probability.
            use_rope (bool): Whether to apply Rotary Position Embeddings to Q and K.
            max_seq_len (int): Maximum precomputed RoPE sequence length.
        """
        # Parent initialization
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.head_dim: int = embed_dim // num_heads
        self.scaling: float = 1.0 / math.sqrt(self.head_dim)

        # Q, K, V Projections: [B, T, D] -> [B, T, D]
        self.q_proj: nn.Linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj: nn.Linear = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj: nn.Linear = nn.Linear(embed_dim, embed_dim, bias=True)

        # Output projection: [B, T, D] -> [B, T, D]
        self.out_proj: nn.Linear = nn.Linear(embed_dim, embed_dim, bias=True)

        # Attention dropout
        self.dropout: nn.Dropout = nn.Dropout(dropout)

        # Optional RoPE Rotary Embedding
        self.use_rope: bool = use_rope
        if use_rope:
            self.rope: Optional[RotaryEmbedding] = RotaryEmbedding(dim=self.head_dim, max_seq_len=max_seq_len)
        else:
            self.rope = None

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass computing multi-head self-attention.

        Args:
            x (Tensor): Input sequence tensor.
                        Shape: [B, T, D].
                        Type: torch.float32.
                        Role: Sequence representations entering self-attention.
            key_padding_mask (Optional[Tensor]): Boolean mask indicating padded frames.
                        Shape: [B, T]. True = ignore frame, False = keep frame.
            attn_mask (Optional[Tensor]): 2D causal/band mask of shape [T, T] or [B, T, T].
            return_weights (bool): Whether to return attention probability maps.

        Returns:
            Tuple[Tensor, Optional[Tensor]]:
                - Output tensor: Shape [B, T, D], Type torch.float32.
                - Attention weights: Shape [B, H, T, T] if return_weights else None.
        """
        # Batch size B, Sequence length T, Hidden dimension D
        b_sz, seq_len, _ = x.shape

        # 1. Project Q, K, V: [B, T, D] -> [B, T, D]
        q: Tensor = self.q_proj(x)
        k: Tensor = self.k_proj(x)
        v: Tensor = self.v_proj(x)

        # 2. Reshape into multi-head format: [B, T, D] -> [B, H, T, d_k]
        q = q.view(b_sz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b_sz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b_sz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE rotation to Q and K if enabled
        if self.use_rope and self.rope is not None:
            q, k = self.rope.apply_rope(q, k)

        # 4. Scaled dot-product attention scores: [B, H, T, d_k] x [B, H, d_k, T] -> [B, H, T, T]
        scores: Tensor = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # 5. Apply key padding mask if provided: [B, 1, 1, T]
        if key_padding_mask is not None:
            # Expand mask to broadcast over heads and query positions
            # Dimension: [B, 1, 1, T], Type: torch.bool
            mask_expanded: Tensor = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded, -1e9)

        # 6. Apply attention structure mask if provided: [T, T]
        if attn_mask is not None:
            scores = scores + attn_mask

        # 7. Softmax attention probabilities: [B, H, T, T]
        probs: Tensor = F.softmax(scores, dim=-1)
        probs_dropped: Tensor = self.dropout(probs)

        # 8. Weight values: [B, H, T, T] x [B, H, T, d_k] -> [B, H, T, d_k]
        context: Tensor = torch.matmul(probs_dropped, v)

        # 9. Concat heads and project out: [B, H, T, d_k] -> [B, T, D]
        context = context.transpose(1, 2).contiguous().view(b_sz, seq_len, self.embed_dim)
        out: Tensor = self.out_proj(context)

        return out, (probs if return_weights else None)


class BandCausalSelfAttention(nn.Module):
    """
    Band-Causal Local Self-Attention for Streaming Word Contexts.
    Restricts attention to a sliding window of preceding words (default: K=8 words),
    ensuring strictly causal and computationally bounded streaming decoding.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        band_window: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize BandCausalSelfAttention.

        Args:
            embed_dim (int): Channel dimension D (512).
            num_heads (int): Number of heads H (8).
            band_window (int): Number of causal preceding words visible (8).
            dropout (float): Dropout probability.
        """
        # Parent initialization
        super().__init__()
        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.band_window: int = band_window

        # Base self-attention module
        self.attn: MultiHeadSelfAttention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=False,
        )

    def _build_band_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """
        Constructs additive band-causal mask where position i attends only to max(0, i-K)..i.

        Args:
            seq_len (int): Number of words L.
            device (torch.device): Device on which to allocate the mask.

        Returns:
            Tensor: Additive mask of shape [seq_len, seq_len] with 0.0 on valid slots and -inf on invalid slots.
        """
        # Indices: [seq_len, 1] and [1, seq_len]
        q_idx: Tensor = torch.arange(seq_len, device=device).unsqueeze(1)
        k_idx: Tensor = torch.arange(seq_len, device=device).unsqueeze(0)

        # Delta: q_idx - k_idx
        # Valid if: 0 <= (q_idx - k_idx) <= band_window
        diff: Tensor = q_idx - k_idx
        valid_mask: Tensor = (diff >= 0) & (diff <= self.band_window)

        # Additive mask: 0.0 for valid, -1e9 for blocked
        mask: Tensor = torch.full((seq_len, seq_len), -1e9, device=device)
        mask.masked_fill_(valid_mask, 0.0)
        return mask

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass applying band-causal self-attention.

        Args:
            x (Tensor): Input tensor of shape [B, L, D].
            key_padding_mask (Optional[Tensor]): Padding mask of shape [B, L].

        Returns:
            Tensor: Attended representations of shape [B, L, D].
        """
        seq_len: int = x.shape[1]
        band_mask: Tensor = self._build_band_causal_mask(seq_len, device=x.device)
        out, _ = self.attn(x, key_padding_mask=key_padding_mask, attn_mask=band_mask)
        return out


class AttentionPooling(nn.Module):
    """
    Attention-Weighted Sequence-to-Vector Pooling (Lee et al., 2019).
    Compresses variable-length token sequences (e.g. K phonemes in a word)
    into a single fixed-size continuous latent vector z_word via a learned query vector.
    """

    def __init__(self, embed_dim: int = 256) -> None:
        """
        Initialize AttentionPooling.

        Args:
            embed_dim (int): Hidden dimension D of input tokens (e.g. 256).
        """
        # Parent initialization
        super().__init__()

        # Learned Query Projection: projects D -> 1 score per token
        # Parameter: [D, 1], Type: torch.float32, Role: scores token importance
        self.query_proj: nn.Linear = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, x: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass pooling tokens into a single latent vector.

        Args:
            x (Tensor): Token sequence embeddings.
                        Shape: [B, K, D], where K is number of tokens in the word.
                        Type: torch.float32.
                        Role: Intra-word phoneme representations.
            pad_mask (Optional[Tensor]): Boolean mask indicating padded positions.
                        Shape: [B, K]. True = pad token, False = valid token.

        Returns:
            Tensor: Pooled word latent representation.
                    Shape: [B, D].
                    Type: torch.float32.
                    Role: Unified word latent vector z_word.
        """
        # Compute raw alignment scores: [B, K, 1] -> [B, K]
        scores: Tensor = self.query_proj(x).squeeze(-1)

        # Mask padding tokens with large negative value
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, -1e9)

        # Softmax weights over token axis K: [B, K, 1]
        weights: Tensor = F.softmax(scores, dim=-1).unsqueeze(-1)

        # Weighted sum across tokens: [B, K, D] * [B, K, 1] -> sum over K -> [B, D]
        pooled: Tensor = (x * weights).sum(dim=1)
        return pooled
