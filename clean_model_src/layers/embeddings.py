"""
Modular Embedding Layers for Speech Transformers and Conformers.
Includes:
- ConvolutionalPositionalEmbedding (Relative Local Context)
- RotaryEmbedding (RoPE for Long-Context Attention)
- LearnablePositionalEmbedding (Static Absolute Slots)
"""

# Import Modules
import math
from typing import Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class ConvolutionalPositionalEmbedding(nn.Module):
    """
    Convolutional Relative Positional Embedding (Mohamed et al., 2019 / Gulati et al., 2020).
    Uses a 1D grouped convolution with weight normalization to inject shift-invariant
    positional information into frame representations without hard sequence limits.
    """

    def __init__(
        self,
        embed_dim: int,
        kernel_size: int = 128,
        groups: int = 16,
    ) -> None:
        """
        Initialize ConvolutionalPositionalEmbedding.

        Args:
            embed_dim (int): Channel / hidden embedding dimension (e.g., 512).
            kernel_size (int): Temporal kernel window (e.g., 128 frames = ~2.5s).
            groups (int): Number of grouped convolution channels (e.g., 16).
        """
        # Parent initialization
        super().__init__()

        # Embed dim: [1], Type: int, Role: channel dimension D
        self.embed_dim: int = embed_dim

        # Kernel size: [1], Type: int, Role: context span
        self.kernel_size: int = kernel_size

        # Groups: [1], Type: int, Role: grouped conv parameter reduction
        self.groups: int = min(groups, embed_dim)

        # 1D Convolution: input [B, D, T], output [B, D, T]
        # Padding: (kernel_size // 2) ensures symmetric same-length temporal alignment
        self.conv: nn.Conv1d = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=self.groups,
        )

        # Weight normalization for numerical training stability
        self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass applying convolutional positional embeddings.

        Args:
            x (Tensor): Input tensor of shape [B, T, D].
                        Dimension: Batch size B, Time frames T, Hidden dimension D.
                        Type: torch.float32 or torch.float16.

        Returns:
            Tensor: Positional bias tensor of shape [B, T, D].
        """
        # Shape: [B, T, D]
        b_sz, seq_len, dim = x.shape

        # Transpose for Conv1d: [B, T, D] -> [B, D, T]
        # Type: Tensor, Role: format for 1D convolution over temporal axis
        h: Tensor = x.transpose(1, 2)

        # Convolve: [B, D, T] -> [B, D, T + padding]
        # Type: Tensor, Role: relative spatial kernel response
        pos: Tensor = self.conv(h)

        # Slice to ensure exact sequence length T matching input
        # Shape: [B, D, T]
        pos = pos[:, :, :seq_len]

        # Transpose back: [B, D, T] -> [B, T, D]
        out: Tensor = pos.transpose(1, 2)
        return out


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) (Su et al., 2021).
    Applies continuous complex-number rotation to Query and Key vectors in self-attention,
    naturally encoding relative distances with full TFLite / ONNX export compatibility.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
    ) -> None:
        """
        Initialize RotaryEmbedding.

        Args:
            dim (int): Dimension of each attention head (d_k = D // num_heads).
            max_seq_len (int): Maximum precomputed cache length.
            base (float): Frequency scaling base (standard: 10000.0).
        """
        # Parent initialization
        super().__init__()

        # Dimension per head: [1], Type: int, Role: must be even for 2D rotational pairs
        assert dim % 2 == 0, f"RoPE dimension must be even, got {dim}"
        self.dim: int = dim
        self.max_seq_len: int = max_seq_len
        self.base: float = base

        # Inverse frequencies: [dim // 2], Type: torch.float32, Role: geometric frequency scale
        theta: Tensor = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("theta", theta, persistent=False)

        # Precompute initial cosine and sine cache: [max_seq_len, dim]
        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, seq_len: int) -> None:
        """
        Precomputes cos and sin tables up to seq_len.

        Args:
            seq_len (int): Sequence length to cache.
        """
        # Temporal indices: [seq_len], Type: torch.float32, Role: frame positions 0..T-1
        t_pos: Tensor = torch.arange(seq_len, dtype=torch.float32, device=self.theta.device)

        # Outer product: [seq_len, dim // 2], Type: torch.float32, Role: phase angles
        freqs: Tensor = torch.outer(t_pos, self.theta)

        # Repeat along channel to cover full head dim: [seq_len, dim]
        freqs_full: Tensor = torch.cat([freqs, freqs], dim=-1)

        # Cache cos and sin: [seq_len, dim]
        self.register_buffer("cos_cached", freqs_full.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_full.sin(), persistent=False)

    def _rotate_half(self, x: Tensor) -> Tensor:
        """
        Rotates half the hidden dimensions: [-x2, x1].

        Args:
            x (Tensor): Input tensor of shape [..., dim].

        Returns:
            Tensor: Rotated tensor of identical shape [..., dim].
        """
        # Split into halves along last dimension
        # x1, x2: [..., dim // 2]
        x1: Tensor = x[..., : self.dim // 2]
        x2: Tensor = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: Tensor, seq_len: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """
        Get cos and sin embeddings for a given sequence length.

        Args:
            x (Tensor): Query or Key tensor of shape [B, num_heads, T, head_dim].
            seq_len (Optional[int]): Sequence length T. If None, derived from x.shape[2].

        Returns:
            Tuple[Tensor, Tensor]: cos and sin tensors shaped [1, 1, T, head_dim].
        """
        # Length T: [1], Type: int
        t: int = seq_len if seq_len is not None else x.shape[2]

        # Expand cache dynamically if input exceeds precomputed bound
        if t > self.cos_cached.shape[0]:
            self._precompute_cache(max(t, self.cos_cached.shape[0] * 2))

        # Slice cos and sin up to current length: [1, 1, T, head_dim]
        # Dimension: matches broadcasting for [B, num_heads, T, head_dim]
        cos: Tensor = self.cos_cached[:t, :].unsqueeze(0).unsqueeze(0)
        sin: Tensor = self.sin_cached[:t, :].unsqueeze(0).unsqueeze(0)
        return cos, sin

    def apply_rope(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Applies rotary positional rotation to Query and Key tensors.

        Args:
            q (Tensor): Query tensor [B, num_heads, T, head_dim].
            k (Tensor): Key tensor [B, num_heads, T, head_dim].

        Returns:
            Tuple[Tensor, Tensor]: Rotated Query and Key tensors of identical shape.
        """
        # Fetch cos, sin: [1, 1, T, head_dim]
        cos, sin = self.forward(q)

        # Apply 2D complex rotation: (x * cos) + (rotate_half(x) * sin)
        # q_rot, k_rot: [B, num_heads, T, head_dim]
        q_rot: Tensor = (q * cos) + (self._rotate_half(q) * sin)
        k_rot: Tensor = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot


class LearnablePositionalEmbedding(nn.Module):
    """
    Standard Learnable Absolute Positional Embedding.
    Maps discrete position indices [0..max_len-1] to continuous embedding vectors.
    Commonly used in text decoders and word-slot query initializations.
    """

    def __init__(self, max_seq_len: int, embed_dim: int) -> None:
        """
        Initialize LearnablePositionalEmbedding.

        Args:
            max_seq_len (int): Maximum position index supported (e.g. 64 or 512).
            embed_dim (int): Channel dimension D (e.g. 512).
        """
        # Parent initialization
        super().__init__()

        # Max sequence length: [1], Type: int
        self.max_seq_len: int = max_seq_len

        # Embed dim: [1], Type: int
        self.embed_dim: int = embed_dim

        # Parameter: [1, max_seq_len, embed_dim], Type: torch.float32, Role: learned slot bias
        self.embedding: nn.Parameter = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02
        )

    def forward(self, seq_len: int) -> Tensor:
        """
        Retrieve learned positional embeddings for positions [0..seq_len-1].

        Args:
            seq_len (int): Length of the sequence.

        Returns:
            Tensor: Positional embeddings of shape [1, seq_len, embed_dim].
        """
        assert seq_len <= self.max_seq_len, (
            f"Requested seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
        )
        # Sliced tensor: [1, seq_len, embed_dim]
        return self.embedding[:, :seq_len, :]
