"""Progressive Speech Transformer Variants for SOTA Phoneme Recognition (< 10% PER).

5 Progressive Levers Built upon the Winning PhonoHuBERT-Hierarchical Architecture:
- Variant 1 (phono_v1_frontend): Pretrained Meta HuBERT 7-layer 1D CNN feature extractor (formant-tuned).
- Variant 2 (phono_v2_specaugment): Variant 1 + dynamic acoustic time & frequency SpecAugment.
- Variant 3 (phono_v3_hybrid): Variant 2 + hybrid real human speech & synthetic data streaming.
- Variant 4 (phono_v4_scaled): Variant 3 + deep scaled transformer capacity and extended schedule.
- Variant 5 (phono_v5_beam): Variant 4 + CTC Prefix Beam Search decoding with phonotactic constraints.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.phono_hubert_hierarchical import (
    PhonoHuBERTHierarchicalConfig,
    PhonoHuBERTHierarchicalForPreTraining,
)


def load_pretrained_cnn_weights(feature_extractor: nn.Module, weights_path: Optional[str] = None) -> bool:
    """Load pretrained 7-layer temporal 1D CNN weights from disk."""
    p = Path(weights_path) if weights_path else Path("data/pretrained_cnn_ls960.pt")
    if p.exists():
        try:
            state_dict = torch.load(p, map_location="cpu", weights_only=True)
            feature_extractor.load_state_dict(state_dict, strict=False)
            print(f"✅ [Pretrained CNN] Loaded 7-layer acoustic filter weights from {p}")
            return True
        except Exception as e:
            print(f"⚠️ [Pretrained CNN] Failed to load {p}: {e}")
            return False
    return False


# ==============================================================================
# VARIANT 1: Pretrained Acoustic Front-End (Formant-Tuned)
# ==============================================================================
@dataclass
class PhonoV1FrontendConfig(PhonoHuBERTHierarchicalConfig):
    """Variant 1: Pretrained 1D CNN Feature Extractor from Meta HuBERT (960h)."""
    pretrained_cnn: bool = True
    pretrained_cnn_path: str = "data/pretrained_cnn_ls960.pt"


class PhonoV1FrontendForPreTraining(PhonoHuBERTHierarchicalForPreTraining):
    """Hierarchical Speech Transformer with Pretrained 7-Layer CNN Feature Extractor."""

    def __init__(self, config: Optional[PhonoV1FrontendConfig] = None):
        super().__init__(config=config or PhonoV1FrontendConfig())
        if getattr(self.config, "pretrained_cnn", True):
            load_pretrained_cnn_weights(self.feature_extractor, getattr(self.config, "pretrained_cnn_path", None))


# ==============================================================================
# VARIANT 2: Acoustic SpecAugment & Context Masking
# ==============================================================================
@dataclass
class PhonoV2SpecAugmentConfig(PhonoV1FrontendConfig):
    """Variant 2: Variant 1 + Active Time and Frequency SpecAugment."""
    masking_mode: str = "specaugment"
    mask_prob: float = 0.20             # 20% of time frames masked
    mask_length: int = 5                # 5 frames = 100ms contiguous acoustic span
    channel_mask_prob: float = 0.10     # 10% of hidden channels masked


class PhonoV2SpecAugmentForPreTraining(PhonoV1FrontendForPreTraining):
    """Hierarchical Speech Transformer with Pretrained CNN + Built-in SpecAugment."""

    def _get_feat_extract_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Compute downsampled acoustic frame length from raw audio length."""
        return torch.tensor([
            self.config.compute_output_length(int(l.item())) for l in input_lengths
        ], device=input_lengths.device, dtype=torch.long)

    def apply_specaugment(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply SpecAugment time-span and channel masking to projected acoustic features."""
        B, T, D = features.shape
        masked = features.clone()
        mask = torch.zeros((B, T), dtype=torch.bool, device=features.device)

        if not self.training:
            return masked, mask

        # 1. Time Span Masking (100ms blocks)
        mask_prob = getattr(self.config, "mask_prob", 0.20)
        mask_len = getattr(self.config, "mask_length", 5)
        num_mask = int(mask_prob * T / max(1, mask_len))
        for b in range(B):
            if num_mask > 0 and T > mask_len:
                starts = torch.randint(0, T - mask_len + 1, (num_mask,), device=features.device)
                for s in starts:
                    masked[b, s : s + mask_len] = 0.0
                    mask[b, s : s + mask_len] = True

        # 2. Channel Masking per sample
        ch_prob = getattr(self.config, "channel_mask_prob", 0.10)
        ch_count = max(1, int(D * ch_prob))
        for b in range(B):
            ch_idx = torch.randperm(D, device=features.device)[:ch_count]
            masked[b, :, ch_idx] = 0.0

        return masked, mask

    def forward(
        self,
        audio: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        frame_targets: Optional[torch.Tensor] = None,
        frame_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass with dynamic SpecAugment."""
        cnn_features = self.feature_extractor(audio)
        projected = self.feature_projection(cnn_features)
        B, T_frames, D = projected.shape

        if audio_lengths is not None:
            input_lengths = self._get_feat_extract_output_lengths(audio_lengths)
        else:
            input_lengths = torch.full((B,), T_frames, device=audio.device, dtype=torch.long)

        # Apply SpecAugment in training mode
        features, mask = self.apply_specaugment(projected)

        encoder_res = self.encoder(features)
        hidden_state = encoder_res["last_hidden_state"]

        # Hierarchical Head Factoring
        state_logits = self.state_router(hidden_state)
        phoneme_logits = self.phoneme_head(hidden_state)
        composite_log_probs = self.compute_composite_log_probs(state_logits, phoneme_logits)

        loss = None
        ctc_loss_val = 0.0
        router_loss_val = 0.0
        accuracy = None

        if targets is not None:
            ctc_log_probs = composite_log_probs.transpose(0, 1).float()
            if target_lengths is None:
                target_lengths = torch.full((B,), targets.shape[1], device=audio.device, dtype=torch.long)

            ctc_loss = self.ctc_loss_fn(ctc_log_probs, targets, input_lengths, target_lengths)
            ctc_loss_val = float(ctc_loss.item())

            # Anti-blank regularization
            blank_penalty_weight = getattr(self.config, "blank_penalty_weight", 0.0)
            blank_threshold = getattr(self.config, "blank_threshold", 0.25)
            if blank_penalty_weight > 0.0 and self.training:
                probs = composite_log_probs.exp()
                blank_probs = probs[:, :, self.config.blank_token_id]
                mask_frames = torch.arange(T_frames, device=audio.device).unsqueeze(0) < input_lengths.unsqueeze(1)
                valid_blank_probs = blank_probs[mask_frames]
                mean_blank_prob = valid_blank_probs.mean() if valid_blank_probs.numel() > 0 else blank_probs.mean()
                blank_loss = blank_penalty_weight * torch.relu(mean_blank_prob - blank_threshold) ** 2
                ctc_loss = ctc_loss + blank_loss

            loss = ctc_loss * self.config.ctc_loss_weight

            # Sequence accuracy
            with torch.no_grad():
                preds_list = self.decode_greedy(composite_log_probs, lengths=input_lengths)
                matches = 0.0
                valid_count = 0
                for b in range(B):
                    ref_seq = targets[b, : int(target_lengths[b])].tolist()
                    pred_seq = preds_list[b]
                    if len(ref_seq) > 0:
                        try:
                            import editdistance
                            dist = editdistance.eval(pred_seq, ref_seq)
                            s_acc = max(0.0, 1.0 - (dist / max(1, len(ref_seq))))
                        except Exception:
                            s_acc = 1.0 if pred_seq == ref_seq else 0.0
                        matches += s_acc
                        valid_count += 1
                accuracy = torch.tensor((matches / max(1, valid_count)) * 100.0, device=audio.device)

        return {
            "loss": loss,
            "ctc_loss": ctc_loss_val,
            "accuracy": accuracy,
            "logits": composite_log_probs,
            "state_logits": state_logits,
            "phoneme_logits": phoneme_logits,
            "input_lengths": input_lengths,
            "output_lengths": input_lengths,
            "mask": mask,
        }


# ==============================================================================
# VARIANT 3: Hybrid Real Human + Synthetic Data
# ==============================================================================
@dataclass
class PhonoV3HybridConfig(PhonoV2SpecAugmentConfig):
    """Variant 3: Variant 2 + Hybrid Real Human & Synthetic Data Compatibility."""
    hybrid_training: bool = True
    librispeech_manifest: str = "data/librispeech/librispeech_train.json"


class PhonoV3HybridForPreTraining(PhonoV2SpecAugmentForPreTraining):
    """Hierarchical Speech Transformer optimized for Hybrid Real/Synthetic Data."""
    pass


# ==============================================================================
# VARIANT 4: Scaled Capacity & Extended Budget
# ==============================================================================
@dataclass
class PhonoV4ScaledConfig(PhonoV3HybridConfig):
    """Variant 4: Variant 3 + Deep Scaled Multi-Head Attention Backbone."""
    encoder_layers: int = 8
    encoder_heads: int = 8
    encoder_embed_dim: int = 512
    encoder_ffn_dim: int = 2048
    dropout: float = 0.1
    attention_dropout: float = 0.1


class PhonoV4ScaledForPreTraining(PhonoV3HybridForPreTraining):
    """Deep Scaled Hierarchical Speech Transformer."""
    pass


# ==============================================================================
# VARIANT 5: Prefix Beam Search & Phonotactic Decoding
# ==============================================================================
@dataclass
class PhonoV5BeamConfig(PhonoV4ScaledConfig):
    """Variant 5: Variant 4 + CTC Prefix Beam Search Decoding with Phonotactic Prior."""
    beam_width: int = 24
    length_penalty_alpha: float = 0.05
    lm_weight_beta: float = 0.20
    lm_model_path: Optional[str] = "data/phonotactic_3gram.json"


class PhonoV5BeamForPreTraining(PhonoV4ScaledForPreTraining):
    """Hierarchical Speech Transformer with Prefix Beam Search Decoding and Phonotactic Prior."""

    def __init__(self, config: Optional[PhonoV5BeamConfig] = None):
        super().__init__(config=config or PhonoV5BeamConfig())
        self._phonotactic_lm = None
        self._load_phonotactic_lm()

    def _load_phonotactic_lm(self):
        lm_path = getattr(self.config, "lm_model_path", "data/phonotactic_3gram.json")
        if lm_path:
            p = Path(lm_path)
            if p.exists():
                try:
                    import json
                    with open(p, "r", encoding="utf-8") as f:
                        self._phonotactic_lm = json.load(f)
                except Exception:
                    self._phonotactic_lm = None

    def _score_lm_token(self, prefix: Tuple[int, ...], token: int) -> float:
        """Compute log P(token | prefix) under precomputed phonotactic n-gram prior."""
        import math
        if self._phonotactic_lm is None:
            return 0.0
        counts = self._phonotactic_lm.get("counts", {})
        context_counts = self._phonotactic_lm.get("context_counts", {})
        vocab_size = self._phonotactic_lm.get("vocab_size", 65)
        p_list = list(prefix)
        c1 = str(p_list[-2]) if len(p_list) >= 2 else "<s>"
        c2 = str(p_list[-1]) if len(p_list) >= 1 else "<s>"
        ctx = f"{c1}_{c2}"
        cnt = counts.get(ctx, {}).get(str(token), 0)
        tot = context_counts.get(ctx, 0)
        return math.log((cnt + 0.05) / (tot + 0.05 * vocab_size))

    def decode_beam(
        self,
        logits_or_audio: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        beam_width: Optional[int] = None,
        length_penalty_alpha: Optional[float] = None,
        lm_weight_beta: Optional[float] = None,
    ) -> List[List[int]]:
        """CTC Prefix Beam Search with non-speech filtering, repeat collapsing, length scoring, and phonotactic prior."""
        import math

        width = beam_width or getattr(self.config, "beam_width", 16)
        alpha = length_penalty_alpha if length_penalty_alpha is not None else getattr(self.config, "length_penalty_alpha", 0.05)
        beta = lm_weight_beta if lm_weight_beta is not None else getattr(self.config, "lm_weight_beta", 0.15)
        blank_id = self.config.blank_token_id
        non_speech_ids = {
            self.config.blank_token_id,
            self.config.pad_token_id,
            self.config.silence_token_id,
            self.config.noise_token_id,
        }

        if logits_or_audio.dim() == 3:
            log_probs = logits_or_audio
        else:
            self.eval()
            with torch.no_grad():
                audio = logits_or_audio
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                out = self.forward(audio=audio)
                log_probs = out["logits"]

        def _logaddexp(a: float, b: float) -> float:
            if a == -float("inf"):
                return b
            if b == -float("inf"):
                return a
            return max(a, b) + math.log1p(math.exp(-abs(a - b)))

        B, T, V = log_probs.shape
        NEG_INF = -float("inf")
        batch_results = []

        for b in range(B):
            curr_T = int(lengths[b]) if lengths is not None else T
            # prefix -> (p_blank, p_non_blank, cumulative_lm_score)
            beam = {(): (0.0, NEG_INF, 0.0)}

            for t in range(curr_T):
                frame_log_probs = log_probs[b, t].tolist()
                next_beam = {}
                top_tokens = torch.topk(log_probs[b, t], min(16, V)).indices.tolist()

                for prefix, (p_b, p_nb, p_lm) in beam.items():
                    p_total = _logaddexp(p_b, p_nb)

                    # 1. Blank emission
                    p_blank_emit = frame_log_probs[blank_id]
                    curr_b, curr_nb, curr_lm = next_beam.get(prefix, (NEG_INF, NEG_INF, p_lm))
                    next_beam[prefix] = (_logaddexp(curr_b, p_total + p_blank_emit), curr_nb, p_lm)

                    # 2. Non-blank token emissions
                    for c in top_tokens:
                        if c == blank_id:
                            continue
                        p_c = frame_log_probs[c]

                        if len(prefix) > 0 and c == prefix[-1]:
                            # Repeated token:
                            # Produces an extended prefix ONLY if previous state was blank (separated by blank)
                            token_lm = self._score_lm_token(prefix, c) if (self._phonotactic_lm and beta > 0) else 0.0
                            new_prefix = prefix + (c,)
                            nb_b, nb_nb, nlm = next_beam.get(new_prefix, (NEG_INF, NEG_INF, p_lm + token_lm))
                            next_beam[new_prefix] = (nb_b, _logaddexp(nb_nb, p_b + p_c), p_lm + token_lm)

                            # If previous state was non-blank, it collapses into the same prefix
                            curr_b, curr_nb, curr_lm = next_beam.get(prefix, (NEG_INF, NEG_INF, p_lm))
                            next_beam[prefix] = (curr_b, _logaddexp(curr_nb, p_nb + p_c), p_lm)
                        else:
                            if c in non_speech_ids:
                                new_prefix = prefix
                                token_lm = 0.0
                            else:
                                new_prefix = prefix + (c,)
                                token_lm = self._score_lm_token(prefix, c) if (self._phonotactic_lm and beta > 0) else 0.0
                            nb_b, nb_nb, nlm = next_beam.get(new_prefix, (NEG_INF, NEG_INF, p_lm + token_lm))
                            next_beam[new_prefix] = (nb_b, _logaddexp(nb_nb, p_total + p_c), p_lm + token_lm)

                # Prune beam with combined score: log(p_ctc) + beta * log(p_lm) + alpha * length
                sorted_items = sorted(
                    next_beam.items(),
                    key=lambda item: _logaddexp(item[1][0], item[1][1]) + beta * item[1][2] + alpha * len(item[0]),
                    reverse=True,
                )
                beam = dict(sorted_items[:width])

            if beam:
                best_prefix = max(
                    beam.keys(),
                    key=lambda p: _logaddexp(beam[p][0], beam[p][1]) + beta * beam[p][2] + alpha * len(p),
                )
                batch_results.append(list(best_prefix))
            else:
                batch_results.append([])

        return batch_results


# ==============================================================================
# NOVEL MODULES: MoE, Sparse Local Attention, & Diffusion Refinement
# ==============================================================================

class MoEFeedForwardNetwork(nn.Module):
    """Mixture of Experts Feed-Forward Network with Top-K Gating and Load Balancing."""

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.router = nn.Linear(embed_dim, num_experts)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        x_flat = x.view(-1, C)

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Vectorized zero-sync expert dispatch using index_add_
        out_flat = torch.zeros_like(x_flat)
        batch_idx = torch.arange(x_flat.shape[0], device=x.device)
        for e_id in range(self.num_experts):
            mask = (topk_indices == e_id)
            token_mask = mask.any(dim=-1)
            idx = batch_idx[token_mask]
            if idx.shape[0] > 0:
                w = (topk_probs * mask.float()).sum(dim=-1, keepdim=True)[token_mask]
                expert_out = self.experts[e_id](x_flat[idx])
                out_flat.index_add_(0, idx, w * expert_out)

        # Switch-Transformer auxiliary load-balancing loss
        density = (router_probs > (1.0 / self.num_experts)).float().mean(dim=0)
        aux_loss = self.num_experts * torch.sum(density * router_probs.mean(dim=0))

        return out_flat.view(B, T, C), aux_loss


class SparseLocalSelfAttention(nn.Module):
    """Local Syllabic Window Multi-Head Self-Attention with cached band-diagonal masking."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.scaling = self.head_dim ** -0.5
        self._cached_window_mask = None

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _get_window_mask(self, T: int, device: torch.device) -> torch.Tensor:
        if (
            self._cached_window_mask is None
            or self._cached_window_mask.shape[-1] != T
            or self._cached_window_mask.device != device
        ):
            indices = torch.arange(T, device=device)
            distance = (indices.unsqueeze(0) - indices.unsqueeze(1)).abs()
            self._cached_window_mask = (distance > self.window_size).unsqueeze(0).unsqueeze(0)
        return self._cached_window_mask

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # Symmetric local window mask (|i - j| <= window_size) using cached tensor
        window_mask = self._get_window_mask(T, x.device)
        attn_scores = attn_scores.masked_fill(window_mask, float("-inf"))

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs_dropped = self.dropout(attn_probs)

        out = torch.matmul(attn_probs_dropped, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)

        return out, (attn_probs if return_attn_weights else None)


class MoETransformerLayer(nn.Module):
    """Transformer Layer with optional Sparse Window Attention and MoE FFN."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        use_sparse_attention: bool = False,
        window_size: int = 16,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        if use_sparse_attention:
            self.self_attn = SparseLocalSelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                dropout=attention_dropout,
            )
        else:
            from src.models.transformer import MultiHeadSelfAttention
            self.self_attn = MultiHeadSelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=attention_dropout,
            )

        self.moe_ffn = MoEFeedForwardNetwork(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
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
        residual = x
        normed = self.self_attn_layer_norm(x)
        attn_out, attn_weights = self.self_attn(
            normed,
            key_padding_mask=key_padding_mask,
            return_attn_weights=return_attn_weights,
        )
        x = residual + self.dropout(attn_out)

        residual = x
        normed = self.final_layer_norm(x)
        ffn_out, aux_loss = self.moe_ffn(normed)
        x = residual + self.dropout(ffn_out)

        return x, attn_weights, aux_loss


class MoETransformerEncoder(nn.Module):
    """Transformer Contextual Encoder with Positional Convolutions and MoE Layers."""

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        use_sparse_attention: bool = False,
        window_size: int = 16,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        pos_conv_kernel: int = 128,
        pos_conv_groups: int = 16,
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
            MoETransformerLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                num_experts=num_experts,
                top_k=top_k,
                use_sparse_attention=use_sparse_attention,
                window_size=window_size,
                dropout=dropout,
                attention_dropout=attention_dropout,
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
        intermediate_layers: Optional[List[int]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        pos = self.pos_conv(x)
        x = x + pos
        x = self.layer_norm(x)
        x = self.dropout(x)

        total_aux_loss = torch.tensor(0.0, device=x.device)
        intermediate_states = {}
        for idx, layer in enumerate(self.layers):
            layer_num = idx + 1
            x, _, layer_aux = layer(x, key_padding_mask=key_padding_mask, return_attn_weights=output_attentions)
            total_aux_loss = total_aux_loss + layer_aux
            if intermediate_layers and layer_num in intermediate_layers:
                intermediate_states[layer_num] = x

        x = self.final_layer_norm(x)
        return {
            "last_hidden_state": x,
            "intermediate_states": intermediate_states,
            "aux_loss": total_aux_loss,
        }


# ==============================================================================
# VARIANT 6.1: Mixture of Experts Speech Transformer
# ==============================================================================
@dataclass
class PhonoV61MoEConfig(PhonoV5BeamConfig):
    """Variant 6.1: PhonoV5 + Top-K Gated Mixture of Experts FFN."""
    num_experts: int = 4
    moe_top_k: int = 2
    moe_aux_loss_weight: float = 0.01


class PhonoV61MoEForPreTraining(PhonoV5BeamForPreTraining):
    """Speech Transformer with Top-K Mixture of Experts FFN Backbone."""

    def __init__(self, config: Optional[PhonoV61MoEConfig] = None):
        super().__init__(config=config or PhonoV61MoEConfig())
        self.encoder = MoETransformerEncoder(
            embed_dim=self.config.encoder_embed_dim,
            num_layers=self.config.encoder_layers,
            num_heads=self.config.encoder_heads,
            ffn_dim=self.config.encoder_ffn_dim,
            num_experts=getattr(self.config, "num_experts", 4),
            top_k=getattr(self.config, "moe_top_k", 2),
            use_sparse_attention=False,
            dropout=self.config.dropout,
            attention_dropout=self.config.attention_dropout,
            pos_conv_kernel=self.config.pos_conv_kernel,
            pos_conv_groups=self.config.pos_conv_groups,
        )

    def forward(
        self,
        audio: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        frame_targets: Optional[torch.Tensor] = None,
        frame_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        cnn_features = self.feature_extractor(audio)
        projected = self.feature_projection(cnn_features)
        B, T_frames, D = projected.shape

        if audio_lengths is not None:
            input_lengths = self._get_feat_extract_output_lengths(audio_lengths)
        else:
            input_lengths = torch.full((B,), T_frames, device=audio.device, dtype=torch.long)

        # Apply SpecAugment in training mode
        features, mask = self.apply_specaugment(projected)

        encoder_res = self.encoder(features)
        hidden_state = encoder_res["last_hidden_state"]
        aux_loss = encoder_res.get("aux_loss", torch.tensor(0.0, device=audio.device))

        # Hierarchical Head Factoring
        state_logits = self.state_router(hidden_state)
        phoneme_logits = self.phoneme_head(hidden_state)
        composite_log_probs = self.compute_composite_log_probs(state_logits, phoneme_logits)

        loss = None
        ctc_loss_val = 0.0
        accuracy = None

        if targets is not None:
            ctc_log_probs = composite_log_probs.transpose(0, 1).float()
            if target_lengths is None:
                target_lengths = torch.full((B,), targets.shape[1], device=audio.device, dtype=torch.long)

            ctc_loss = self.ctc_loss_fn(ctc_log_probs, targets, input_lengths, target_lengths)
            ctc_loss_val = float(ctc_loss.item())

            # Anti-blank regularization
            blank_penalty_weight = getattr(self.config, "blank_penalty_weight", 0.0)
            blank_threshold = getattr(self.config, "blank_threshold", 0.25)
            if blank_penalty_weight > 0.0 and self.training:
                probs = composite_log_probs.exp()
                blank_probs = probs[:, :, self.config.blank_token_id]
                mask_frames = torch.arange(T_frames, device=audio.device).unsqueeze(0) < input_lengths.unsqueeze(1)
                valid_blank_probs = blank_probs[mask_frames]
                mean_blank_prob = valid_blank_probs.mean() if valid_blank_probs.numel() > 0 else blank_probs.mean()
                blank_loss = blank_penalty_weight * torch.relu(mean_blank_prob - blank_threshold) ** 2
                ctc_loss = ctc_loss + blank_loss

            aux_weight = getattr(self.config, "moe_aux_loss_weight", 0.01)
            loss = ctc_loss * self.config.ctc_loss_weight + aux_weight * aux_loss

            with torch.no_grad():
                preds_list = self.decode_greedy(composite_log_probs, lengths=input_lengths)
                matches = 0.0
                valid_count = 0
                for b in range(B):
                    ref_seq = targets[b, : int(target_lengths[b])].tolist()
                    pred_seq = preds_list[b]
                    if len(ref_seq) > 0:
                        try:
                            import editdistance
                            dist = editdistance.eval(pred_seq, ref_seq)
                            s_acc = max(0.0, 1.0 - (dist / max(1, len(ref_seq))))
                        except Exception:
                            s_acc = 1.0 if pred_seq == ref_seq else 0.0
                        matches += s_acc
                        valid_count += 1
                accuracy = torch.tensor((matches / max(1, valid_count)) * 100.0, device=audio.device)

        return {
            "loss": loss,
            "ctc_loss": ctc_loss_val,
            "aux_loss": float(aux_loss.item()) if hasattr(aux_loss, "item") else 0.0,
            "accuracy": accuracy,
            "logits": composite_log_probs,
            "state_logits": state_logits,
            "phoneme_logits": phoneme_logits,
            "input_lengths": input_lengths,
            "output_lengths": input_lengths,
            "mask": mask,
        }


# ==============================================================================
# VARIANT 6.2: MoE + Sparse Local Syllabic Attention + Intermediate CTC Early Exit
# ==============================================================================
@dataclass
class PhonoV62SparseConfig(PhonoV61MoEConfig):
    """Variant 6.2: Variant 6.1 + Sparse Local Syllabic Attention + Intermediate CTC Early Exit."""
    use_sparse_attention: bool = True
    sparse_window_size: int = 16
    enable_intermediate_ctc: bool = True
    inter_ctc_layers: Tuple[int, ...] = (4, 8)
    inter_ctc_loss_weight: float = 0.25
    early_exit_threshold: float = 0.95


class PhonoV62SparseForPreTraining(PhonoV61MoEForPreTraining):
    """Speech Transformer with MoE FFN, Sparse Syllabic Attention, and Intermediate CTC Early Exit."""

    def __init__(self, config: Optional[PhonoV62SparseConfig] = None):
        super().__init__(config=config or PhonoV62SparseConfig())
        self.encoder = MoETransformerEncoder(
            embed_dim=self.config.encoder_embed_dim,
            num_layers=self.config.encoder_layers,
            num_heads=self.config.encoder_heads,
            ffn_dim=self.config.encoder_ffn_dim,
            num_experts=getattr(self.config, "num_experts", 4),
            top_k=getattr(self.config, "moe_top_k", 2),
            use_sparse_attention=getattr(self.config, "use_sparse_attention", True),
            window_size=getattr(self.config, "sparse_window_size", 16),
            dropout=self.config.dropout,
            attention_dropout=self.config.attention_dropout,
            pos_conv_kernel=self.config.pos_conv_kernel,
            pos_conv_groups=self.config.pos_conv_groups,
        )

    def forward(
        self,
        audio: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        frame_targets: Optional[torch.Tensor] = None,
        frame_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        cnn_features = self.feature_extractor(audio)
        projected = self.feature_projection(cnn_features)
        B, T_frames, D = projected.shape

        if audio_lengths is not None:
            input_lengths = self._get_feat_extract_output_lengths(audio_lengths)
        else:
            input_lengths = torch.full((B,), T_frames, device=audio.device, dtype=torch.long)

        # Apply SpecAugment in training mode
        features, mask = self.apply_specaugment(projected)

        enable_inter = getattr(self.config, "enable_intermediate_ctc", True)
        inter_layers = list(getattr(self.config, "inter_ctc_layers", (4, 8))) if enable_inter else None

        encoder_res = self.encoder(features, intermediate_layers=inter_layers)
        hidden_state = encoder_res["last_hidden_state"]
        intermediate_states = encoder_res.get("intermediate_states", {})
        aux_loss = encoder_res.get("aux_loss", torch.tensor(0.0, device=audio.device))

        # Hierarchical Head Factoring for Final Layer
        state_logits = self.state_router(hidden_state)
        phoneme_logits = self.phoneme_head(hidden_state)
        composite_log_probs = self.compute_composite_log_probs(state_logits, phoneme_logits)

        loss = None
        ctc_loss_val = 0.0
        inter_loss_val = 0.0
        accuracy = None

        if targets is not None:
            ctc_log_probs = composite_log_probs.transpose(0, 1).float()
            if target_lengths is None:
                target_lengths = torch.full((B,), targets.shape[1], device=audio.device, dtype=torch.long)

            final_ctc_loss = self.ctc_loss_fn(ctc_log_probs, targets, input_lengths, target_lengths)
            ctc_loss_val = float(final_ctc_loss.item())

            # Intermediate CTC Multi-Task Loss across Layers (e.g. Layer 4 & Layer 8)
            inter_ctc_total = torch.tensor(0.0, device=audio.device)
            if enable_inter and intermediate_states:
                for l_idx, h_inter in intermediate_states.items():
                    s_logits_l = self.state_router(h_inter)
                    p_logits_l = self.phoneme_head(h_inter)
                    c_log_probs_l = self.compute_composite_log_probs(s_logits_l, p_logits_l).transpose(0, 1).float()
                    l_loss = self.ctc_loss_fn(c_log_probs_l, targets, input_lengths, target_lengths)
                    inter_ctc_total = inter_ctc_total + l_loss
                inter_ctc_total = inter_ctc_total / max(1, len(intermediate_states))
                inter_loss_val = float(inter_ctc_total.item())

            # Anti-blank regularization
            blank_penalty_weight = getattr(self.config, "blank_penalty_weight", 0.0)
            blank_threshold = getattr(self.config, "blank_threshold", 0.25)
            if blank_penalty_weight > 0.0 and self.training:
                probs = composite_log_probs.exp()
                blank_probs = probs[:, :, self.config.blank_token_id]
                mask_frames = torch.arange(T_frames, device=audio.device).unsqueeze(0) < input_lengths.unsqueeze(1)
                valid_blank_probs = blank_probs[mask_frames]
                mean_blank_prob = valid_blank_probs.mean() if valid_blank_probs.numel() > 0 else blank_probs.mean()
                blank_loss = blank_penalty_weight * torch.relu(mean_blank_prob - blank_threshold) ** 2
                final_ctc_loss = final_ctc_loss + blank_loss

            aux_weight = getattr(self.config, "moe_aux_loss_weight", 0.01)
            inter_weight = getattr(self.config, "inter_ctc_loss_weight", 0.25)
            loss = (
                final_ctc_loss * self.config.ctc_loss_weight
                + aux_weight * aux_loss
                + inter_weight * inter_ctc_total
            )

            with torch.no_grad():
                preds_list = self.decode_greedy(composite_log_probs, lengths=input_lengths)
                matches = 0.0
                valid_count = 0
                for b in range(B):
                    ref_seq = targets[b, : int(target_lengths[b])].tolist()
                    pred_seq = preds_list[b]
                    if len(ref_seq) > 0:
                        try:
                            import editdistance
                            dist = editdistance.eval(pred_seq, ref_seq)
                            s_acc = max(0.0, 1.0 - (dist / max(1, len(ref_seq))))
                        except Exception:
                            s_acc = 1.0 if pred_seq == ref_seq else 0.0
                        matches += s_acc
                        valid_count += 1
                accuracy = torch.tensor((matches / max(1, valid_count)) * 100.0, device=audio.device)

        return {
            "loss": loss,
            "ctc_loss": ctc_loss_val,
            "inter_ctc_loss": inter_loss_val,
            "aux_loss": float(aux_loss.item()) if hasattr(aux_loss, "item") else 0.0,
            "accuracy": accuracy,
            "logits": composite_log_probs,
            "hidden_state": hidden_state,
            "intermediate_states": intermediate_states,
            "state_logits": state_logits,
            "phoneme_logits": phoneme_logits,
            "input_lengths": input_lengths,
            "output_lengths": input_lengths,
            "mask": mask,
        }

    def decode_early_exit(
        self,
        audio: torch.Tensor,
        confidence_threshold: Optional[float] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Adaptive layer-skipping inference using intermediate CTC confidence gating."""
        threshold = confidence_threshold or getattr(self.config, "early_exit_threshold", 0.95)
        self.eval()
        with torch.no_grad():
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            B = audio.shape[0]

            out = self.forward(audio=audio, audio_lengths=lengths)
            T_frames = out["logits"].shape[1]
            inter_states = out.get("intermediate_states", {})

            # Frame-level dynamic exit across layers: 4 -> 8 -> 12
            final_probs = out["logits"].exp()
            exit_layers_used = torch.full((B, T_frames), 12.0, device=audio.device)
            selected_probs = final_probs.clone()

            # Process intermediate layers in reverse order: layer 8 then layer 4
            for l_num in sorted(inter_states.keys(), reverse=True):
                h_l = inter_states[l_num]
                s_logits = self.state_router(h_l)
                p_logits = self.phoneme_head(h_l)
                log_p = self.compute_composite_log_probs(s_logits, p_logits)
                probs = log_p.exp()
                max_p = probs.max(dim=-1).values
                confident = max_p >= threshold

                selected_probs[confident] = probs[confident]
                exit_layers_used[confident] = float(l_num)

            selected_log_probs = (selected_probs + 1e-8).log()
            decoded_tokens = self.decode_greedy(selected_log_probs, lengths=out["input_lengths"])
            avg_exit_layer = float(exit_layers_used.mean().item())
            compute_saved_pct = round((1.0 - (avg_exit_layer / 12.0)) * 100.0, 1)

            return {
                "tokens": decoded_tokens,
                "avg_exit_layer": avg_exit_layer,
                "compute_saved_pct": compute_saved_pct,
            }


# ==============================================================================
# VARIANT 6.3: MoE + Sparse Attention + Sliding Window Latent Diffusion Refiner
# ==============================================================================
class GaussianNoiseScheduler(nn.Module):
    """Generates spatio-temporal Gaussian-modulated noise schedules for latent diffusion.

    The noise strength follows a bell-shaped Gaussian envelope centered at frame tau:
        sigma(t; tau) = sigma_scale * exp(- (t - tau)^2 / (2 * w^2))
    Outside the local receptive window (|t - tau| > 2w), noise rapidly decays to zero.
    """

    def __init__(self, default_window_width: int = 16, default_noise_max: float = 0.8):
        super().__init__()
        self.default_window_width = default_window_width
        self.default_noise_max = default_noise_max

    def compute_noise_map(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        centers: Optional[torch.Tensor] = None,
        noise_scales: Optional[torch.Tensor] = None,
        window_width: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates a spatial-temporal Gaussian noise map of shape [B, T, 1]."""
        w = float(window_width or self.default_window_width)
        time_steps = torch.arange(seq_len, device=device, dtype=torch.float32)  # [T]

        if centers is None:
            centers = torch.randint(0, max(1, seq_len), (batch_size,), device=device, dtype=torch.float32)
        else:
            centers = centers.to(device=device, dtype=torch.float32)

        if noise_scales is None:
            noise_scales = torch.empty(batch_size, device=device).uniform_(0.1, self.default_noise_max)
        else:
            noise_scales = noise_scales.to(device=device, dtype=torch.float32)

        # Gaussian kernel: [B, T]
        diff = time_steps.unsqueeze(0) - centers.unsqueeze(1)
        exponent = -0.5 * (diff / (w + 1e-6)) ** 2
        gaussian_kernel = torch.exp(exponent)
        noise_map = (noise_scales.unsqueeze(1) * gaussian_kernel).unsqueeze(-1)  # [B, T, 1]

        return noise_map, centers, noise_scales


class LatentDiffusionRefiner(nn.Module):
    """Lightweight 1D residual convolutional refiner with FiLM noise conditioning.

    Refines noisy frame embeddings Z_sigma back toward clean acoustic latents Z_0.
    Uses depthwise-separable 1D convolutions and FiLM modulation on the noise map.
    """

    def __init__(self, embed_dim: int = 512, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        h_dim = hidden_dim or embed_dim
        self.in_proj = nn.Linear(embed_dim, h_dim)

        # FiLM projection: noise scale sigma -> scale (gamma) and shift (beta)
        self.noise_mlp = nn.Sequential(
            nn.Linear(1, h_dim // 2),
            nn.SiLU(),
            nn.Linear(h_dim // 2, 2 * h_dim),
        )

        # 2 residual blocks with depthwise-separable 1D convs
        self.conv1 = nn.Conv1d(h_dim, h_dim, kernel_size=5, padding=2, groups=h_dim)
        self.pw_conv1 = nn.Conv1d(h_dim, h_dim, kernel_size=1)
        self.norm1 = nn.LayerNorm(h_dim)

        self.conv2 = nn.Conv1d(h_dim, h_dim, kernel_size=5, padding=2, groups=h_dim)
        self.pw_conv2 = nn.Conv1d(h_dim, h_dim, kernel_size=1)
        self.norm2 = nn.LayerNorm(h_dim)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(h_dim, embed_dim)

        # Zero-initialize the final projection so the refiner starts as an identity map
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z: torch.Tensor, noise_map: torch.Tensor) -> torch.Tensor:
        """Forward pass for latent refinement.

        Args:
            z: [B, T, D] frame representations
            noise_map: [B, T, 1] spatial-temporal noise scale
        Returns:
            z_hat: [B, T, D] refined clean representation estimate
        """
        h = self.in_proj(z)

        # FiLM parameters: gamma and beta of shape [B, T, H]
        film_params = self.noise_mlp(noise_map)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)

        # Residual Block 1
        res = h
        h_conv = h.transpose(1, 2)
        h_conv = self.act(self.pw_conv1(self.conv1(h_conv))).transpose(1, 2)
        h_conv = self.dropout(h_conv)
        h = self.norm1(res + h_conv)
        h = h * (1.0 + gamma) + beta

        # Residual Block 2
        res = h
        h_conv = h.transpose(1, 2)
        h_conv = self.act(self.pw_conv2(self.conv2(h_conv))).transpose(1, 2)
        h_conv = self.dropout(h_conv)
        h = self.norm2(res + h_conv)

        # Zero-init residual output addition
        z_hat = z + self.out_proj(h)
        return z_hat


@dataclass
class PhonoV63DiffusionConfig(PhonoV62SparseConfig):
    """Variant 6.3: MoE + Sparse Attention + Sliding Window Latent Diffusion Refiner."""
    diffusion_window_width: int = 16       # Gaussian half-width in frames (~320ms)
    diffusion_noise_max: float = 0.8       # Maximum noise level for training perturbation
    diffusion_loss_weight: float = 1.0     # Multi-task weight for Gaussian MSE diffusion loss
    refined_ctc_loss_weight: float = 0.5   # Multi-task weight for refined phoneme CTC loss
    diffusion_inference_steps: int = 3     # Iterative denoising steps per window during decoding
    diffusion_window_stride: int = 16      # Stride between consecutive Gaussian sliding windows
    diffusion_steps: int = 4               # Backward compatibility
    diffusion_window_size: int = 64        # Backward compatibility
    diffusion_stride: int = 32             # Backward compatibility
    diffusion_confidence_threshold: float = 0.35


class PhonoV63DiffusionForPreTraining(PhonoV62SparseForPreTraining):
    """Hierarchical MoE Sparse Transformer with Sliding Window Latent Diffusion Refiner."""

    def __init__(self, config: PhonoV63DiffusionConfig):
        super().__init__(config)
        self.noise_scheduler = GaussianNoiseScheduler(
            default_window_width=getattr(config, "diffusion_window_width", 16),
            default_noise_max=getattr(config, "diffusion_noise_max", 0.8),
        )
        self.latent_refiner = LatentDiffusionRefiner(
            embed_dim=self.config.encoder_embed_dim,
            dropout=self.config.dropout,
        )

    def forward(
        self,
        audio: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        frame_targets: Optional[torch.Tensor] = None,
        frame_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Compute baseline forward pass through acoustic backbone & intermediate CTC
        out = super().forward(
            audio=audio,
            targets=targets,
            target_lengths=target_lengths,
            audio_lengths=audio_lengths,
            frame_targets=frame_targets,
            frame_lengths=frame_lengths,
            **kwargs,
        )

        hidden_state = out.get("hidden_state")
        input_lengths = out.get("input_lengths")

        if targets is not None and self.training and hidden_state is not None:
            B, T_frames, D = hidden_state.shape
            if target_lengths is None:
                target_lengths = torch.full((B,), targets.shape[1], device=audio.device, dtype=torch.long)

            # 1. Sample spatio-temporal Gaussian noise schedule
            noise_map, centers, noise_scales = self.noise_scheduler.compute_noise_map(
                batch_size=B,
                seq_len=T_frames,
                device=audio.device,
                window_width=getattr(self.config, "diffusion_window_width", 16),
            )

            # 2. Perturb latents with Gaussian-modulated Gaussian noise
            eps = torch.randn_like(hidden_state)
            z_noisy = hidden_state + noise_map * eps

            # 3. Latent Diffusion Denoising Step
            z_clean_est = self.latent_refiner(z_noisy, noise_map)

            # 4. Gaussian-weighted Diffusion Reconstruction Loss
            diff_sq = (z_clean_est - hidden_state) ** 2
            weight_norm = (noise_map.sum() * D) + 1e-6
            diff_loss = (noise_map * diff_sq).sum() / weight_norm

            # 5. Refined CTC Phoneme Loss on recovered latents
            ref_s_logits = self.state_router(z_clean_est)
            ref_p_logits = self.phoneme_head(z_clean_est)
            ref_composite_log_probs = self.compute_composite_log_probs(ref_s_logits, ref_p_logits)
            ref_ctc_log_probs = ref_composite_log_probs.transpose(0, 1).float()
            ref_ctc_loss = self.ctc_loss_fn(ref_ctc_log_probs, targets, input_lengths, target_lengths)

            # 6. Combined Multi-Task Objective
            diff_w = getattr(self.config, "diffusion_loss_weight", 1.0)
            ref_w = getattr(self.config, "refined_ctc_loss_weight", 0.5)
            out["loss"] = out["loss"] + (diff_w * diff_loss) + (ref_w * ref_ctc_loss)
            out["diff_loss"] = float(diff_loss.item())
            out["refined_ctc_loss"] = float(ref_ctc_loss.item())
            out["refined_logits"] = ref_composite_log_probs
        else:
            out["diff_loss"] = 0.0
            out["refined_ctc_loss"] = 0.0
            out["refined_logits"] = out.get("logits")

        return out

    def decode_sliding_diffusion(
        self,
        audio: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        window_width: Optional[int] = None,
        window_stride: Optional[int] = None,
    ) -> List[List[int]]:
        """Trailing Sliding Gaussian Window Diffusion Decoding.

        Sweeps a Gaussian noise window across time. Inside the window, latents are iteratively
        refined. As frames exit the trailing edge of the window, their latents are finalized
        and decoded into phoneme tokens.
        """
        self.eval()
        with torch.no_grad():
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            B = audio.shape[0]

            out = self.forward(audio=audio, audio_lengths=lengths)
            hidden_states = out["hidden_state"]  # [B, T_frames, D]
            input_lengths = out["input_lengths"]

            w = float(window_width or getattr(self.config, "diffusion_window_width", 16))
            stride = int(window_stride or getattr(self.config, "diffusion_window_stride", 16))
            steps = int(num_steps or getattr(self.config, "diffusion_inference_steps", 3))

            batch_results = []
            for b in range(B):
                T_b = int(input_lengths[b].item())
                z_curr = hidden_states[b, :T_b].clone().unsqueeze(0)  # [1, T_b, D]
                device = z_curr.device

                # Sliding Gaussian window sweep along the temporal axis
                for tau in range(0, T_b + int(w), stride):
                    t_start = max(0, int(tau - 2 * w))
                    t_end = min(T_b, int(tau + 2 * w))
                    if t_start >= t_end:
                        continue

                    active_z = z_curr[:, t_start:t_end]  # [1, T_active, D]
                    active_T = active_z.shape[1]
                    time_indices = torch.arange(t_start, t_end, device=device, dtype=torch.float32)

                    # Iterative refinement within the active Gaussian window
                    for k in range(steps):
                        sigma_k = getattr(self.config, "diffusion_noise_max", 0.8) * ((steps - k) / steps)
                        diff = time_indices - tau
                        noise_map = sigma_k * torch.exp(-0.5 * (diff / (w + 1e-6)) ** 2)
                        noise_map = noise_map.view(1, active_T, 1)

                        refined_active = self.latent_refiner(active_z, noise_map)
                        active_z = 0.5 * active_z + 0.5 * refined_active

                    z_curr[:, t_start:t_end] = active_z

                # Trailing decode: emit phonemes once frames have exited the sliding window
                s_logits = self.state_router(z_curr)
                p_logits = self.phoneme_head(z_curr)
                log_probs = self.compute_composite_log_probs(s_logits, p_logits)
                decoded = self.decode_greedy(log_probs, lengths=torch.tensor([T_b], device=device))[0]
                batch_results.append(decoded)

            return batch_results

    def decode_diffusion(
        self,
        logits_or_audio: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
    ) -> List[List[int]]:
        """Unified decoding dispatcher supporting both audio and precomputed logits."""
        if logits_or_audio.dim() == 3:
            # 3D logit tensor: use beam search on logits
            return self.decode_beam(logits_or_audio, lengths=lengths)
        return self.decode_sliding_diffusion(logits_or_audio, lengths=lengths, num_steps=num_steps)


# ==============================================================================
# VARIANT 6.4: Confidence-Gated Latent Diffusion with Deep Refiner
# ==============================================================================

class DeepLatentDiffusionRefiner(nn.Module):
    """3-Block Residual Convolutional Refiner with FiLM noise conditioning.

    Extends the 2-block LatentDiffusionRefiner with an additional residual block
    for stronger denoising capacity on ambiguous frames, plus a wider kernel for
    better temporal context.
    """

    def __init__(self, embed_dim: int = 512, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        h_dim = hidden_dim or embed_dim
        self.in_proj = nn.Linear(embed_dim, h_dim)

        # FiLM projection: noise scale sigma -> scale (gamma) and shift (beta)
        self.noise_mlp = nn.Sequential(
            nn.Linear(1, h_dim // 2),
            nn.SiLU(),
            nn.Linear(h_dim // 2, 2 * h_dim),
        )

        # 3 residual blocks with depthwise-separable 1D convs
        # conv1 & conv2 match LatentDiffusionRefiner (kernel 5) for seamless warm-start from V6.3
        self.conv1 = nn.Conv1d(h_dim, h_dim, kernel_size=5, padding=2, groups=h_dim)
        self.pw_conv1 = nn.Conv1d(h_dim, h_dim, kernel_size=1)
        self.norm1 = nn.LayerNorm(h_dim)

        self.conv2 = nn.Conv1d(h_dim, h_dim, kernel_size=5, padding=2, groups=h_dim)
        self.pw_conv2 = nn.Conv1d(h_dim, h_dim, kernel_size=1)
        self.norm2 = nn.LayerNorm(h_dim)

        # Additional residual block for fine detail and deeper denoising capacity
        self.conv3 = nn.Conv1d(h_dim, h_dim, kernel_size=3, padding=1, groups=h_dim)
        self.pw_conv3 = nn.Conv1d(h_dim, h_dim, kernel_size=1)
        self.norm3 = nn.LayerNorm(h_dim)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(h_dim, embed_dim)

        # Zero-initialize the final projection so the refiner starts as identity
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z: torch.Tensor, noise_map: torch.Tensor) -> torch.Tensor:
        """Forward pass for latent refinement.

        Args:
            z: [B, T, D] frame representations
            noise_map: [B, T, 1] spatial-temporal noise scale
        Returns:
            z_hat: [B, T, D] refined clean representation estimate
        """
        h = self.in_proj(z)

        # FiLM parameters: gamma and beta of shape [B, T, H]
        film_params = self.noise_mlp(noise_map)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)

        # Residual Block 1 (wide kernel=7 for broad context)
        res = h
        h_conv = h.transpose(1, 2)
        h_conv = self.act(self.pw_conv1(self.conv1(h_conv))).transpose(1, 2)
        h_conv = self.dropout(h_conv)
        h = self.norm1(res + h_conv)
        h = h * (1.0 + gamma) + beta  # FiLM conditioning

        # Residual Block 2 (medium kernel=5)
        res = h
        h_conv = h.transpose(1, 2)
        h_conv = self.act(self.pw_conv2(self.conv2(h_conv))).transpose(1, 2)
        h_conv = self.dropout(h_conv)
        h = self.norm2(res + h_conv)

        # Residual Block 3 (narrow kernel=3 for fine detail)
        res = h
        h_conv = h.transpose(1, 2)
        h_conv = self.act(self.pw_conv3(self.conv3(h_conv))).transpose(1, 2)
        h_conv = self.dropout(h_conv)
        h = self.norm3(res + h_conv)

        # Zero-init residual output addition
        z_hat = z + self.out_proj(h)
        return z_hat


@dataclass
class PhonoV64GatedDiffusionConfig(PhonoV63DiffusionConfig):
    """Variant 6.4: Confidence-Gated Latent Diffusion with Deep Refiner.

    Key innovation: Frames where CTC is already confident (high top-1 margin)
    bypass diffusion entirely, preserving crisp probability spikes. Only ambiguous
    frames receive iterative Gaussian denoising through a deeper 3-block refiner.

    This solves V6.3's weakness of smearing already-confident CTC peaks.
    """
    # Confidence gating thresholds
    gate_confidence_high: float = 0.8     # Frames with margin > this bypass diffusion entirely
    gate_confidence_low: float = 0.3      # Frames with margin < this get full diffusion
    # Between low and high: linear interpolation (smooth transition)

    # Deep refiner uses 3 conv blocks instead of 2
    use_deep_refiner: bool = True

    # Confidence-weighted diffusion loss: weight diffusion loss inversely with confidence
    confidence_weighted_loss: bool = True

    # Learnable gating: If True, uses a small MLP over hidden states and margin to learn the gating map
    learnable_gating: bool = True
    gating_hidden_dim: int = 128


class PhonoV64GatedDiffusionForPreTraining(PhonoV63DiffusionForPreTraining):
    """MoE Sparse Transformer with Confidence-Gated Latent Diffusion Refiner.

    Adaptive gating mechanism:
    - High-confidence frames: bypass diffusion, preserve crisp spikes
    - Low-confidence frames: full diffusion refinement
    - Medium-confidence frames: smooth learned or linear interpolation
    - Trainable Gating MLP: learns optimal non-linear boundary between CTC backbone and diffusion
    """

    def __init__(self, config: PhonoV64GatedDiffusionConfig):
        super().__init__(config)

        # Replace the 2-block refiner with a deeper 3-block refiner
        if getattr(config, "use_deep_refiner", True):
            self.latent_refiner = DeepLatentDiffusionRefiner(
                embed_dim=self.config.encoder_embed_dim,
                dropout=self.config.dropout,
            )

        # Trainable Gating MLP: takes [hidden_state (D), margin (1), max_prob (1)] -> gate in [0, 1]
        self.learnable_gating = getattr(config, "learnable_gating", True)
        if self.learnable_gating:
            g_dim = getattr(config, "gating_hidden_dim", 128)
            in_dim = self.config.encoder_embed_dim + 2  # D + margin + top1_prob
            self.gate_mlp = nn.Sequential(
                nn.Linear(in_dim, g_dim),
                nn.GELU(),
                nn.Linear(g_dim, 1),
                nn.Sigmoid(),
            )
            # Initialize gate_mlp bias so it starts with initial bias towards 0.5 (balanced)
            nn.init.constant_(self.gate_mlp[-2].bias, 0.0)

        # Running statistics for monitoring gating behavior
        self.register_buffer("_gate_bypass_ema", torch.tensor(0.0))
        self.register_buffer("_gate_partial_ema", torch.tensor(0.0))
        self.register_buffer("_gate_full_ema", torch.tensor(0.0))

    def _compute_confidence_gate(
        self,
        hidden_state: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-frame confidence gate from CTC logit margin or learnable MLP.

        Args:
            hidden_state: [B, T, D] encoder hidden states
            input_lengths: [B] valid frame counts

        Returns:
            gate: [B, T, 1] in [0, 1] where 0 = bypass diffusion, 1 = full diffusion
        """
        # 1. Compute CTC logits and probabilities (no gradients needed through head for gating)
        with torch.no_grad():
            s_logits = self.state_router(hidden_state)
            p_logits = self.phoneme_head(hidden_state)
            log_probs = self.compute_composite_log_probs(s_logits, p_logits)
            probs = log_probs.exp()  # [B, T, V]
            top2_probs, _ = probs.topk(2, dim=-1)  # [B, T, 2]
            top1_prob = top2_probs[:, :, :1]        # [B, T, 1]
            margin = top1_prob - top2_probs[:, :, 1:2]  # [B, T, 1]

        if getattr(self, "learnable_gating", False) and hasattr(self, "gate_mlp"):
            # Differentiable learned gate conditioned on acoustic context + margin + top-1 prob
            g_in = torch.cat([hidden_state, margin, top1_prob], dim=-1)  # [B, T, D + 2]
            gate = self.gate_mlp(g_in)  # [B, T, 1] in [0, 1]
        else:
            with torch.no_grad():
                high = getattr(self.config, "gate_confidence_high", 0.8)
                low = getattr(self.config, "gate_confidence_low", 0.3)
                gate = 1.0 - (margin - low).clamp(0.0, high - low) / max(high - low, 1e-6)

        # Mask out padding frames (set gate to 0 = no diffusion on padding)
        B, T = gate.shape[:2]
        frame_mask = torch.arange(T, device=gate.device).unsqueeze(0) < input_lengths.unsqueeze(1)
        gate = gate * frame_mask.unsqueeze(-1).float()

        # Update running statistics for monitoring
        if self.training:
            with torch.no_grad():
                decay = getattr(self.config, "gate_ema_decay", 0.99)
                valid_gate = gate[frame_mask.unsqueeze(-1).expand_as(gate)]
                if valid_gate.numel() > 0:
                    bypass_frac = (valid_gate < 0.1).float().mean()
                    partial_frac = ((valid_gate >= 0.1) & (valid_gate <= 0.9)).float().mean()
                    full_frac = (valid_gate > 0.9).float().mean()
                    self._gate_bypass_ema.mul_(decay).add_((1 - decay) * bypass_frac)
                    self._gate_partial_ema.mul_(decay).add_((1 - decay) * partial_frac)
                    self._gate_full_ema.mul_(decay).add_((1 - decay) * full_frac)

        return gate

    def forward(
        self,
        audio: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        frame_targets: Optional[torch.Tensor] = None,
        frame_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Run the V6.2 Sparse backbone forward (CTC + MoE + InterCTC)
        # We call V6.2's forward directly, skipping V6.3's unconditional diffusion
        out = PhonoV62SparseForPreTraining.forward(
            self,
            audio=audio,
            targets=targets,
            target_lengths=target_lengths,
            audio_lengths=audio_lengths,
            frame_targets=frame_targets,
            frame_lengths=frame_lengths,
            **kwargs,
        )

        hidden_state = out.get("hidden_state")
        input_lengths = out.get("input_lengths")

        if targets is not None and self.training and hidden_state is not None:
            B, T_frames, D = hidden_state.shape
            if target_lengths is None:
                target_lengths = torch.full((B,), targets.shape[1], device=audio.device, dtype=torch.long)

            # 1. Compute per-frame confidence gate
            gate = self._compute_confidence_gate(hidden_state, input_lengths)  # [B, T, 1]

            # 2. Sample spatio-temporal Gaussian noise schedule
            noise_map, centers, noise_scales = self.noise_scheduler.compute_noise_map(
                batch_size=B,
                seq_len=T_frames,
                device=audio.device,
                window_width=getattr(self.config, "diffusion_window_width", 16),
            )

            # 3. Gate the noise: only inject noise on ambiguous frames
            gated_noise_map = noise_map * gate  # [B, T, 1]

            # 4. Perturb latents with confidence-gated Gaussian noise
            eps = torch.randn_like(hidden_state)
            z_noisy = hidden_state + gated_noise_map * eps

            # 5. Deep Latent Diffusion Denoising Step
            z_clean_est = self.latent_refiner(z_noisy, gated_noise_map)

            # 6. Gate the refinement: blend refined with original based on confidence
            # High confidence frames keep their original hidden state
            z_blended = gate * z_clean_est + (1.0 - gate) * hidden_state

            # 7. Confidence-weighted Diffusion Reconstruction Loss
            diff_sq = (z_blended - hidden_state) ** 2
            if getattr(self.config, "confidence_weighted_loss", True):
                # Weight loss by gate: focus on ambiguous frames, ignore confident ones
                weight_map = gated_noise_map + 1e-6
                weight_norm = (weight_map.sum() * D) + 1e-6
                diff_loss = (weight_map * diff_sq).sum() / weight_norm
            else:
                weight_norm = (noise_map.sum() * D) + 1e-6
                diff_loss = (noise_map * diff_sq).sum() / weight_norm

            # 8. Refined CTC Phoneme Loss on blended latents
            ref_s_logits = self.state_router(z_blended)
            ref_p_logits = self.phoneme_head(z_blended)
            ref_composite_log_probs = self.compute_composite_log_probs(ref_s_logits, ref_p_logits)
            ref_ctc_log_probs = ref_composite_log_probs.transpose(0, 1).float()
            ref_ctc_loss = self.ctc_loss_fn(ref_ctc_log_probs, targets, input_lengths, target_lengths)

            # 9. Combined Multi-Task Objective
            diff_w = getattr(self.config, "diffusion_loss_weight", 1.0)
            ref_w = getattr(self.config, "refined_ctc_loss_weight", 0.5)
            out["loss"] = out["loss"] + (diff_w * diff_loss) + (ref_w * ref_ctc_loss)
            out["diff_loss"] = float(diff_loss.item())
            out["refined_ctc_loss"] = float(ref_ctc_loss.item())
            out["refined_logits"] = ref_composite_log_probs
            out["gate_bypass_pct"] = float(self._gate_bypass_ema.item()) * 100.0
            out["gate_partial_pct"] = float(self._gate_partial_ema.item()) * 100.0
            out["gate_full_pct"] = float(self._gate_full_ema.item()) * 100.0
        else:
            out["diff_loss"] = 0.0
            out["refined_ctc_loss"] = 0.0
            out["refined_logits"] = out.get("logits")
            out["gate_bypass_pct"] = float(self._gate_bypass_ema.item()) * 100.0
            out["gate_partial_pct"] = float(self._gate_partial_ema.item()) * 100.0
            out["gate_full_pct"] = float(self._gate_full_ema.item()) * 100.0

        return out

    def decode_gated_diffusion(
        self,
        audio: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        window_width: Optional[int] = None,
        window_stride: Optional[int] = None,
    ) -> List[List[int]]:
        """Confidence-Gated Sliding Gaussian Window Diffusion Decoding.

        Like V6.3's sliding diffusion, but high-confidence frames bypass refinement
        entirely. Only ambiguous frames are iteratively denoised.
        """
        self.eval()
        with torch.no_grad():
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            B = audio.shape[0]

            out = PhonoV62SparseForPreTraining.forward(self, audio=audio, audio_lengths=lengths)
            hidden_states = out["hidden_state"]  # [B, T_frames, D]
            input_lengths = out["input_lengths"]

            w = float(window_width or getattr(self.config, "diffusion_window_width", 16))
            stride = int(window_stride or getattr(self.config, "diffusion_window_stride", 16))
            steps = int(num_steps or getattr(self.config, "diffusion_inference_steps", 3))

            batch_results = []
            for b in range(B):
                T_b = int(input_lengths[b].item())
                z_curr = hidden_states[b, :T_b].clone().unsqueeze(0)  # [1, T_b, D]
                device = z_curr.device

                # Compute per-frame confidence gate
                gate = self._compute_confidence_gate(
                    z_curr, torch.tensor([T_b], device=device)
                )  # [1, T_b, 1]

                # Sliding Gaussian window sweep with confidence gating
                for tau in range(0, T_b + int(w), stride):
                    t_start = max(0, int(tau - 2 * w))
                    t_end = min(T_b, int(tau + 2 * w))
                    if t_start >= t_end:
                        continue

                    active_z = z_curr[:, t_start:t_end]  # [1, T_active, D]
                    active_gate = gate[:, t_start:t_end]  # [1, T_active, 1]
                    active_T = active_z.shape[1]

                    # Skip this window if all frames are confident (gate ~ 0)
                    if active_gate.mean().item() < 0.05:
                        continue

                    time_indices = torch.arange(t_start, t_end, device=device, dtype=torch.float32)

                    # Iterative refinement within the active Gaussian window
                    for k in range(steps):
                        sigma_k = getattr(self.config, "diffusion_noise_max", 0.8) * ((steps - k) / steps)
                        diff = time_indices - tau
                        noise_map = sigma_k * torch.exp(-0.5 * (diff / (w + 1e-6)) ** 2)
                        noise_map = noise_map.view(1, active_T, 1)

                        # Gate the noise: only apply to ambiguous frames
                        gated_noise = noise_map * active_gate

                        refined_active = self.latent_refiner(active_z, gated_noise)

                        # Blend: confident frames keep original, ambiguous frames get refined
                        active_z = active_gate * (0.5 * active_z + 0.5 * refined_active) + \
                                   (1.0 - active_gate) * active_z

                    z_curr[:, t_start:t_end] = active_z

                # Decode from refined latents
                s_logits = self.state_router(z_curr)
                p_logits = self.phoneme_head(z_curr)
                log_probs = self.compute_composite_log_probs(s_logits, p_logits)
                decoded = self.decode_greedy(log_probs, lengths=torch.tensor([T_b], device=device))[0]
                batch_results.append(decoded)

            return batch_results

    def decode_sliding_diffusion(
        self,
        audio: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        window_width: Optional[int] = None,
        window_stride: Optional[int] = None,
    ) -> List[List[int]]:
        """Override V6.3's decode to use confidence-gated version."""
        return self.decode_gated_diffusion(
            audio, lengths=lengths, num_steps=num_steps,
            window_width=window_width, window_stride=window_stride,
        )

