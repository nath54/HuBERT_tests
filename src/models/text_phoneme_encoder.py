"""Fast Lightweight Text/Phoneme Word Encoder for Cross-Modal Latent Distillation.

Maps written text words (character byte IDs) and/or phoneme sequences
directly to the acoustic word latent space (z_word) of Phono-V7.3/V7.5.
Operates 50x faster than the full 12-layer acoustic Conformer.
"""

from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Attention-weighted pooling across sequence tokens, respecting pad masks."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, S, D] token embeddings
            pad_mask: [B, S] True for padding positions
        Returns:
            pooled: [B, D] word embedding
        """
        scores = self.query(x).squeeze(-1)  # [B, S]
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask, -1e9)
        attn = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, S, 1]
        pooled = (x * attn).sum(dim=1)  # [B, D]
        return pooled


class FastTextPhonemeWordEncoder(nn.Module):
    """Lightweight 2-layer Transformer encoder mapping words/phonemes to z_word latent space.

    Supports:
    - Byte modality (orthographic characters 0..122)
    - Phoneme modality (acoustic tokens 0..63)
    - Latency: < 1-2 ms on GPU for hundreds of words
    """

    def __init__(
        self,
        d_macro: int = 512,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        byte_vocab_size: int = 128,
        phoneme_vocab_size: int = 64,
        max_seq_len: int = 32,
    ):
        super().__init__()
        self.d_macro = d_macro
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Modality embeddings
        self.byte_embedding = nn.Embedding(byte_vocab_size, d_model, padding_idx=0)
        self.phoneme_embedding = nn.Embedding(phoneme_vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Modality type indicators
        self.modality_bias = nn.Embedding(2, d_model)  # 0: byte/text, 1: phoneme

        # Lightweight Transformer stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Attention pooling
        self.pooling = AttentionPooling(d_model)

        # Projection into acoustic latent space (z_word)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_macro),
            nn.GELU(),
            nn.LayerNorm(d_macro),
            nn.Linear(d_macro, d_macro),
        )

    def forward(
        self,
        byte_ids: Optional[torch.Tensor] = None,
        phoneme_ids: Optional[torch.Tensor] = None,
        pad_id: int = 0,
    ) -> torch.Tensor:
        """Encode either character bytes or phonemes into acoustic latent word vectors.

        Args:
            byte_ids: [N, K] batch of word character token IDs
            phoneme_ids: [N, P] batch of word phoneme token IDs
            pad_id: padding token index (default 0)
        Returns:
            z_pred: [N, d_macro] estimated word latent vectors
        """
        if byte_ids is not None:
            tokens = byte_ids
            emb = self.byte_embedding(tokens)
            mod_type = 0
        elif phoneme_ids is not None:
            tokens = phoneme_ids
            emb = self.phoneme_embedding(tokens)
            mod_type = 1
        else:
            raise ValueError("Must provide either byte_ids or phoneme_ids")

        N, S = tokens.shape
        device = tokens.device
        S = min(S, self.max_seq_len)
        tokens = tokens[:, :S]
        emb = emb[:, :S]

        positions = torch.arange(S, device=device).unsqueeze(0).expand(N, -1)
        pos_emb = self.pos_embedding(positions)
        mod_emb = self.modality_bias(torch.full((N, S), mod_type, device=device, dtype=torch.long))

        h = emb + pos_emb + mod_emb
        pad_mask = tokens == pad_id

        # Check for all-padding words to prevent PyTorch MultiheadAttention all-masked row NaN
        non_pad_counts = (tokens != pad_id).sum(dim=1)
        is_all_pad = non_pad_counts == 0
        safe_pad_mask = pad_mask.clone()
        if is_all_pad.any():
            safe_pad_mask[is_all_pad, 0] = False

        # Transformer encoding
        h = self.transformer(h, src_key_padding_mask=safe_pad_mask)
        h = self.norm(h)

        # Sequence-level pooling into a single vector per word
        word_h = self.pooling(h, pad_mask=safe_pad_mask)

        # Project to target latent dimension (z_word)
        z_pred = self.proj(word_h)
        if is_all_pad.any():
            z_pred[is_all_pad] = 0.0
        return z_pred
