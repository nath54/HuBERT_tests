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

        out_flat = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = topk_indices[:, k]
            weight = topk_probs[:, k].unsqueeze(-1)
            for e_id in range(self.num_experts):
                mask = (expert_idx == e_id)
                if mask.any():
                    expert_out = self.experts[e_id](x_flat[mask])
                    out_flat[mask] += weight[mask] * expert_out

        # Switch-Transformer auxiliary load-balancing loss
        density = (router_probs > (1.0 / self.num_experts)).float().mean(dim=0)
        aux_loss = self.num_experts * torch.sum(density * router_probs.mean(dim=0))

        return out_flat.view(B, T, C), aux_loss


class SparseLocalSelfAttention(nn.Module):
    """Local Syllabic Window Multi-Head Self-Attention."""

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
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # Symmetric local window mask (|i - j| <= window_size)
        indices = torch.arange(T, device=x.device)
        distance = (indices.unsqueeze(0) - indices.unsqueeze(1)).abs()
        window_mask = distance > self.window_size
        attn_scores = attn_scores.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

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
        **kwargs,
    ) -> Dict[str, Any]:
        pos = self.pos_conv(x)
        x = x + pos
        x = self.layer_norm(x)
        x = self.dropout(x)

        total_aux_loss = torch.tensor(0.0, device=x.device)
        for layer in self.layers:
            x, _, layer_aux = layer(x, key_padding_mask=key_padding_mask, return_attn_weights=output_attentions)
            total_aux_loss = total_aux_loss + layer_aux

        x = self.final_layer_norm(x)
        return {
            "last_hidden_state": x,
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
# VARIANT 6.2: MoE + Sparse Local Syllabic Attention
# ==============================================================================
@dataclass
class PhonoV62SparseConfig(PhonoV61MoEConfig):
    """Variant 6.2: Variant 6.1 + Sparse Local Syllabic Attention (Window +-320ms)."""
    use_sparse_attention: bool = True
    sparse_window_size: int = 16


class PhonoV62SparseForPreTraining(PhonoV61MoEForPreTraining):
    """Speech Transformer with MoE FFN and Sparse Local Syllabic Attention."""

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


# ==============================================================================
# VARIANT 6.3: MoE + Sparse Attention + Sliding Window Diffusion Decoding
# ==============================================================================
@dataclass
class PhonoV63DiffusionConfig(PhonoV62SparseConfig):
    """Variant 6.3: Variant 6.2 + Sliding Window Diffusion Denoising Decoding."""
    diffusion_steps: int = 4
    diffusion_window_size: int = 64
    diffusion_stride: int = 32
    diffusion_confidence_threshold: float = 0.35


class PhonoV63DiffusionForPreTraining(PhonoV62SparseForPreTraining):
    """Hierarchical MoE Sparse Transformer with Sliding Window Diffusion Decoding."""

    def decode_diffusion(
        self,
        logits_or_audio: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
    ) -> List[List[int]]:
        """Iterative Sliding Window Denoising Decoding over acoustic representations."""
        steps = num_steps or getattr(self.config, "diffusion_steps", 4)
        threshold = confidence_threshold or getattr(self.config, "diffusion_confidence_threshold", 0.35)

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

        B, T, V = log_probs.shape
        batch_results = []

        for b in range(B):
            curr_T = int(lengths[b]) if lengths is not None else T
            curr_probs = log_probs[b, :curr_T].clone()

            # Iterative diffusion denoising
            for step in range(steps):
                probs = F.softmax(curr_probs, dim=-1)
                top2 = torch.topk(probs, 2, dim=-1).values
                margin = top2[:, 0] - top2[:, 1]
                ambiguous = margin < threshold

                if not ambiguous.any():
                    break

                # Sliding window bidirectional diffusion smoothing over ambiguous frames
                kernel = torch.tensor([0.2, 0.6, 0.2], device=curr_probs.device, dtype=curr_probs.dtype).unsqueeze(0).unsqueeze(0)
                smoothed = F.conv1d(
                    curr_probs.transpose(0, 1).unsqueeze(0),
                    kernel.repeat(V, 1, 1),
                    padding=1,
                    groups=V,
                ).squeeze(0).transpose(0, 1)

                curr_probs[ambiguous] = 0.5 * curr_probs[ambiguous] + 0.5 * smoothed[ambiguous]

            # Decode via prefix beam search on refined logits
            refined_batch = curr_probs.unsqueeze(0)
            refined_len = torch.tensor([curr_T], device=curr_probs.device)
            decoded = self.decode_beam(refined_batch, lengths=refined_len)[0]
            batch_results.append(decoded)

        return batch_results

