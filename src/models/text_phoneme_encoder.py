"""Fast Lightweight Text/Phoneme Word Encoder for Cross-Modal Latent Distillation.

Maps written text words (character byte IDs) and/or phoneme sequences
directly to the acoustic word latent space (z_word) of Phono-V7.3/V7.5.
Operates 50x faster than the full 12-layer acoustic Conformer.
"""

from typing import List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.phoneme_tokenizer import PHONEME_VOCAB_SIZE


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
        phoneme_vocab_size: int = PHONEME_VOCAB_SIZE,
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
        top_k_ids: Optional[torch.Tensor] = None,
        top_k_probs: Optional[torch.Tensor] = None,
        pad_id: int = 0,
    ) -> torch.Tensor:
        """Encode either character bytes, discrete phonemes, or Top-K soft phoneme mixtures into word latent vectors.

        Args:
            byte_ids: [N, K] batch of word character token IDs
            phoneme_ids: [N, P] batch of word phoneme token IDs
            top_k_ids: [N, P, K_cand] Top-K phoneme candidate token IDs per position
            top_k_probs: [N, P, K_cand] Softmax probabilities for each candidate
            pad_id: padding token index (default 0)
        Returns:
            z_pred: [N, d_macro] estimated word latent vectors
        """
        if byte_ids is not None:
            tokens = byte_ids
            emb = self.byte_embedding(tokens)
            mod_type = 0
        elif top_k_ids is not None:
            tokens = top_k_ids[:, :, 0]
            k_embs = self.phoneme_embedding(top_k_ids)  # [N, P, K_cand, d_model]
            if top_k_probs is not None:
                probs = top_k_probs.unsqueeze(-1)  # [N, P, K_cand, 1]
                emb = (k_embs * probs).sum(dim=2)  # [N, P, d_model]
            else:
                emb = k_embs[:, :, 0, :]
            mod_type = 1
        elif phoneme_ids is not None:
            tokens = phoneme_ids
            emb = self.phoneme_embedding(tokens)
            mod_type = 1
        else:
            raise ValueError("Must provide either byte_ids, phoneme_ids, or top_k_ids")

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


class ContinuousPhonemeWordSegmenter(FastTextPhonemeWordEncoder):
    """Phono-V8 Space-Free Continuous Phoneme-to-Word Segmenter.

    Receives an uninterrupted continuous stream of phonemes (without acoustic spaces),
    predicts lexical End-Of-Word (EOW) boundaries, dynamically slices the hidden representations,
    and pools each word span into the canonical 512D word latent vector (z_word).
    """

    def __init__(
        self,
        d_macro: int = 512,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        phoneme_vocab_size: int = PHONEME_VOCAB_SIZE,
        max_seq_len: int = 256,
        silence_token_id: int = 3,
    ):
        super().__init__(
            d_macro=d_macro,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            phoneme_vocab_size=phoneme_vocab_size,
            max_seq_len=max_seq_len,
        )
        self.silence_token_id = silence_token_id

        # Lexical End-Of-Word (EOW) Boundary Classifier Head
        self.boundary_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward_continuous(
        self,
        phoneme_ids: Optional[torch.Tensor] = None,
        top_k_ids: Optional[torch.Tensor] = None,
        top_k_probs: Optional[torch.Tensor] = None,
        pad_id: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes continuous phonemes into per-token contextual hidden states and boundary logits.

        Args:
            phoneme_ids: [B, T] discrete phoneme tokens
            top_k_ids: [B, T, K_cand] Top-K phoneme candidate tokens
            top_k_probs: [B, T, K_cand] Softmax probabilities
            pad_id: padding index
        Returns:
            h: [B, T, d_model] contextual hidden representations
            boundary_logits: [B, T] unnormalized log-odds for End-Of-Word (EOW) at each position
        """
        if top_k_ids is not None:
            tokens = top_k_ids[:, :, 0]
            k_embs = self.phoneme_embedding(top_k_ids)
            if top_k_probs is not None:
                probs = top_k_probs.unsqueeze(-1)
                emb = (k_embs * probs).sum(dim=2)
            else:
                emb = k_embs[:, :, 0, :]
        elif phoneme_ids is not None:
            tokens = phoneme_ids
            emb = self.phoneme_embedding(tokens)
        else:
            raise ValueError("Must provide either phoneme_ids or top_k_ids")

        B, T = tokens.shape
        device = tokens.device
        T_clamped = min(T, self.max_seq_len)
        tokens = tokens[:, :T_clamped]
        emb = emb[:, :T_clamped]

        positions = torch.arange(T_clamped, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(positions)
        mod_emb = self.modality_bias(torch.full((B, T_clamped), 1, device=device, dtype=torch.long))

        h = emb + pos_emb + mod_emb
        pad_mask = tokens == pad_id

        non_pad_counts = (tokens != pad_id).sum(dim=1)
        is_all_pad = non_pad_counts == 0
        safe_pad_mask = pad_mask.clone()
        if is_all_pad.any():
            safe_pad_mask[is_all_pad, 0] = False

        h = self.transformer(h, src_key_padding_mask=safe_pad_mask)
        h = self.norm(h)  # [B, T, d_model]

        boundary_logits = self.boundary_head(h).squeeze(-1)  # [B, T]
        return h, boundary_logits

    def segment_and_encode_stream(
        self,
        phoneme_ids: Optional[torch.Tensor] = None,
        top_k_ids: Optional[torch.Tensor] = None,
        top_k_probs: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        max_word_len: int = 16,
        pad_id: int = 0,
    ) -> Tuple[torch.Tensor, List[List[Tuple[int, int]]]]:
        """Splits a continuous stream of phonemes into word vectors without acoustic spaces.

        Args:
            phoneme_ids: [1, T] continuous phonemes (or top_k_ids [1, T, K])
            threshold: boundary probability threshold (default 0.5)
            max_word_len: fallback maximum phoneme span per word
        Returns:
            z_words: [1, L, d_macro] sequence of 512D word latents ready for Model 3
            word_spans: list of (start_idx, end_idx) for each segmented word
        """
        h, b_logits = self.forward_continuous(
            phoneme_ids=phoneme_ids,
            top_k_ids=top_k_ids,
            top_k_probs=top_k_probs,
            pad_id=pad_id,
        )
        probs = torch.sigmoid(b_logits[0])  # [T]
        toks = phoneme_ids[0] if phoneme_ids is not None else top_k_ids[0, :, 0]
        T = toks.shape[0]

        spans: List[Tuple[int, int]] = []
        start_idx = 0

        for t in range(T):
            token_id = int(toks[t].item())
            if token_id == pad_id:
                break

            # Rule 1: Explicit Acoustic Silence is guaranteed EOW
            is_silence = (token_id == self.silence_token_id)
            # Rule 2: Neural Boundary Detector triggers EOW
            is_boundary = bool(probs[t].item() > threshold)
            # Rule 3: Forced cut if span length reaches max_word_len
            span_len = t - start_idx + 1
            is_forced = span_len >= max_word_len

            if (is_boundary or is_silence or is_forced or t == T - 1) and span_len > 0:
                # Exclude pure silence from word span
                end_pos = t if is_silence else (t + 1)
                if end_pos > start_idx:
                    spans.append((start_idx, end_pos))
                start_idx = t + 1

        if not spans:
            z_words = torch.zeros((1, 0, self.d_macro), device=h.device)
            return z_words, [spans]

        # Pool each word span into z_word [L, d_macro]
        word_latents = []
        for s_idx, e_idx in spans:
            span_h = h[0, s_idx:e_idx].unsqueeze(0)  # [1, span_len, d_model]
            pooled_w = self.pooling(span_h)           # [1, d_model]
            z_w = self.proj(pooled_w)                 # [1, d_macro]
            word_latents.append(z_w)

        z_words = torch.cat(word_latents, dim=0).unsqueeze(0)  # [1, L, d_macro]
        return z_words, [spans]

