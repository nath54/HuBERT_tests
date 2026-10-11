"""Phono-V8.0 Acoustic Conformer Backbone with DeepSeek MoE and Phoneme Engram Memory.

Integrates:
1. 7-Layer Pretrained Acoustic CNN Feature Extractor
2. 12-Layer Conformer Backbone with DeepSeek Mixture of Experts (1 Shared + 8 Routed Experts, Top-2)
3. Intermediate Draft Phoneme Emission Head (Layer 4)
4. Multi-Scale Phoneme Engram Conditional Memory (Layers 6 and 10)
5. Decoupled Word Boundary Gate & Hierarchical Recursive Phoneme Head (Layer 12)
"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.phono_variants import (
    PhonoV64GatedDiffusionConfig,
    PhonoV64GatedDiffusionForPreTraining,
)
from src.models.conformer_layers import ConformerEngramMoEEncoder
from src.models.phono_v7_3_speech_model import (
    DecoupledBoundaryGateHead,
    HierarchicalRecursivePhonemeHead,
    NUM_ARTICULATORY_GROUPS,
    PHONEME_VOCAB_SIZE,
)


class PhonoV8AcousticBackbone(PhonoV64GatedDiffusionForPreTraining):
    """Phono-V8.0 Acoustic Backbone with DeepSeek MoE & Engram Conditional Memory."""

    def __init__(
        self,
        config: Optional[PhonoV64GatedDiffusionConfig] = None,
        conformer_kernel_size: int = 31,
        num_routed_experts: int = 8,
        top_k: int = 2,
        num_shared_experts: int = 1,
        draft_layer_idx: int = 3,
        engram_layers: Tuple[int, ...] = (4, 7),
        engram_heads: int = 4,
        engram_memory_dim: int = 128,
    ):
        if config is None:
            config = PhonoV64GatedDiffusionConfig()
        super().__init__(config)

        self.vocab_size = PHONEME_VOCAB_SIZE  # 112
        self.phoneme_offset = 8
        self.num_phonemes = self.vocab_size - self.phoneme_offset
        self.config.vocab_size = self.vocab_size

        # 1. 12-Layer Conformer Encoder with DeepSeek MoE & Engram Memory
        self.encoder = ConformerEngramMoEEncoder(
            embed_dim=self.config.encoder_embed_dim,
            num_layers=self.config.encoder_layers,
            num_heads=self.config.encoder_heads,
            ffn_dim=self.config.encoder_ffn_dim,
            num_experts=num_routed_experts,
            top_k=top_k,
            use_deepseek_moe=True,
            num_shared_experts=num_shared_experts,
            kernel_size=conformer_kernel_size,
            dropout=self.config.dropout,
            attention_dropout=self.config.attention_dropout,
            pos_conv_kernel=self.config.pos_conv_kernel,
            pos_conv_groups=self.config.pos_conv_groups,
            use_multi_scale_conv=True,
            vocab_size=self.vocab_size,
            draft_layer_idx=draft_layer_idx,
            engram_layers=engram_layers,
            engram_heads=engram_heads,
            engram_memory_dim=engram_memory_dim,
        )

        # 2. Decoupled Word Boundary Gate Head
        self.boundary_gate_head = DecoupledBoundaryGateHead(self.config.encoder_embed_dim)

        # 3. Hierarchical Recursive Phoneme Head
        self.phoneme_head = HierarchicalRecursivePhonemeHead(
            embed_dim=self.config.encoder_embed_dim,
            vocab_size=self.num_phonemes,
            num_macro_classes=NUM_ARTICULATORY_GROUPS,
        )

        # 4. Auxiliary CTC Losses
        self.macro_ctc_loss_fn = nn.CTCLoss(
            blank=0,
            zero_infinity=True,
        )
        self.draft_ctc_loss_fn = nn.CTCLoss(
            blank=0,
            zero_infinity=True,
        )

    def forward(
        self,
        audio: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        boundary_targets: Optional[torch.Tensor] = None,
        macro_targets: Optional[torch.Tensor] = None,
        macro_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # 1. Pretrained CNN acoustic feature extraction & projection
        cnn_features = self.feature_extractor(audio)
        features = self.feature_projection(cnn_features)
        input_lengths = (
            self._get_feat_extract_output_lengths(audio_lengths)
            if audio_lengths is not None
            else None
        )

        # Dynamic SpecAugment in training mode
        if self.training:
            features, _ = self.apply_specaugment(features)

        # 2. Conformer Engram MoE Encoding
        enc_out = self.encoder(features, teacher_tokens=None)
        hidden_state = enc_out["last_hidden_state"]
        draft_logits = enc_out["draft_logits"]
        aux_moe_loss = enc_out["aux_loss"]

        # 3. Boundary Gate Emissions
        boundary_logits = self.boundary_gate_head(hidden_state)  # [B, T, 3]

        # 4. Hierarchical Recursive Phoneme Emissions
        refined_p_logits, base_p_logits, macro_logits = self.phoneme_head(hidden_state)
        state_logits = self.state_router(hidden_state)  # [B, T, 4]

        refined_logits = self.compute_composite_log_probs(state_logits, refined_p_logits)
        base_logits = self.compute_composite_log_probs(state_logits, base_p_logits)

        # 5. Loss Computations
        ctc_loss = torch.tensor(0.0, device=audio.device)
        draft_ctc_loss = torch.tensor(0.0, device=audio.device)
        macro_ctc_loss = torch.tensor(0.0, device=audio.device)
        boundary_loss = torch.tensor(0.0, device=audio.device)

        B_cur, T_cur, _ = refined_logits.shape
        i_lens = (
            input_lengths
            if input_lengths is not None
            else torch.full((B_cur,), T_cur, device=audio.device, dtype=torch.long)
        )

        if targets is not None:
            log_probs = refined_logits.transpose(0, 1).float()  # (T, B, 112)
            t_lens = (
                target_lengths
                if target_lengths is not None
                else torch.full((B_cur,), targets.shape[1], device=audio.device, dtype=torch.long)
            )
            ctc_loss = self.ctc_loss_fn(log_probs, targets, i_lens, t_lens)

            # Draft CTC loss
            if draft_logits is not None:
                draft_log_probs = F.log_softmax(draft_logits, dim=-1).transpose(0, 1).float()
                draft_ctc_loss = self.draft_ctc_loss_fn(draft_log_probs, targets, i_lens, t_lens)

        if macro_targets is not None:
            macro_log_probs = F.log_softmax(macro_logits, dim=-1).transpose(0, 1).float()
            m_lens = (
                macro_lengths
                if macro_lengths is not None
                else (
                    target_lengths
                    if target_lengths is not None
                    else torch.full((B_cur,), macro_targets.shape[1], device=audio.device, dtype=torch.long)
                )
            )
            macro_ctc_loss = self.macro_ctc_loss_fn(macro_log_probs, macro_targets, i_lens, m_lens)

        if boundary_targets is not None:
            T_bt = boundary_targets.shape[1]
            T_common = min(T_cur, T_bt)
            bg_sliced = boundary_logits[:, :T_common]
            bt_sliced = boundary_targets[:, :T_common]
            weights = torch.tensor([1.0, 10.0, 1.0], device=audio.device)
            boundary_loss = F.cross_entropy(
                bg_sliced.reshape(-1, 3),
                bt_sliced.reshape(-1),
                weight=weights,
                ignore_index=-100,
            )

        total_loss = (
            ctc_loss
            + 0.25 * draft_ctc_loss
            + 0.20 * macro_ctc_loss
            + 1.00 * boundary_loss
            + 0.05 * aux_moe_loss
        )

        return {
            "loss": total_loss,
            "ctc_loss": ctc_loss,
            "draft_ctc_loss": draft_ctc_loss,
            "macro_ctc_loss": macro_ctc_loss,
            "boundary_loss": boundary_loss,
            "aux_moe_loss": aux_moe_loss,
            "logits": refined_logits,
            "base_logits": base_logits,
            "draft_logits": draft_logits,
            "macro_logits": macro_logits,
            "boundary_logits": boundary_logits,
            "hidden_state": hidden_state,
            "input_lengths": input_lengths,
        }

    @torch.no_grad()
    def forward_sliding_chunks(
        self,
        audio: torch.Tensor,
        chunk_seconds: float = 8.0,
        overlap_seconds: float = 2.0,
        sample_rate: int = 16000,
    ) -> Dict[str, torch.Tensor]:
        """Runs multi-window overlapping forward passes with smooth cross-fade blending."""
        B, total_samples = audio.shape
        chunk_size = int(chunk_seconds * sample_rate)
        overlap_size = int(overlap_seconds * sample_rate)
        hop_size = max(1, chunk_size - overlap_size)

        if total_samples <= chunk_size:
            out = self.forward(audio)
            return {
                "logits": out["logits"],
                "boundary_logits": out["boundary_logits"],
            }

        total_frames = total_samples // 320
        device = audio.device

        blended_logits = torch.zeros(B, total_frames, self.vocab_size, device=device)
        blended_boundaries = torch.zeros(B, total_frames, 3, device=device)
        weight_sum = torch.zeros(B, total_frames, 1, device=device)

        start = 0
        while start < total_samples:
            end = min(start + chunk_size, total_samples)
            chunk_audio = audio[:, start:end]
            cur_len = end - start

            out = self.forward(chunk_audio)
            c_logits = out["logits"]
            c_boundaries = out["boundary_logits"]
            c_frames = c_logits.shape[1]

            f_start = start // 320
            f_end = min(f_start + c_frames, total_frames)
            valid_frames = f_end - f_start

            w = torch.ones(1, valid_frames, 1, device=device)
            ov_frames = int((overlap_seconds * sample_rate) // 320)
            if f_start > 0 and ov_frames > 0:
                ramp_len = min(ov_frames, valid_frames)
                w[:, :ramp_len, :] *= torch.linspace(0.0, 1.0, ramp_len, device=device).view(1, -1, 1)
            if end < total_samples and ov_frames > 0:
                ramp_len = min(ov_frames, valid_frames)
                w[:, -ramp_len:, :] *= torch.linspace(1.0, 0.0, ramp_len, device=device).view(1, -1, 1)

            blended_logits[:, f_start:f_end, :] += c_logits[:, :valid_frames, :] * w
            blended_boundaries[:, f_start:f_end, :] += c_boundaries[:, :valid_frames, :] * w
            weight_sum[:, f_start:f_end, :] += w

            start += hop_size

        safe_weight = weight_sum.clamp(min=1e-5)
        blended_logits = blended_logits / safe_weight
        blended_boundaries = blended_boundaries / safe_weight

        return {
            "logits": blended_logits,
            "boundary_logits": blended_boundaries,
        }


def warm_start_phono_v8_from_v75(
    v8_model: PhonoV8AcousticBackbone,
    v75_ckpt_path: str,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, int]:
    """Warm-starts PhonoV8AcousticBackbone from a V7.5 checkpoint.

    Transfers:
    - Pretrained CNN feature extractor & feature projection
    - Positional convolution & layer norms
    - Conformer multi-head self-attentions and depthwise convolutions
    - MoE shared & routed experts
    - Hierarchical recursive phoneme head & state router
    - Decoupled boundary gate head

    Returns:
        (transferred_count, fresh_count)
    """
    ckpt = torch.load(v75_ckpt_path, map_location=device, weights_only=False)
    if "acoustic_state_dict" in ckpt:
        state_dict = ckpt["acoustic_state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    v8_dict = v8_model.state_dict()
    transferred = 0
    fresh = 0

    for k, v in v8_dict.items():
        if k in state_dict and state_dict[k].shape == v.shape:
            v8_dict[k].copy_(state_dict[k])
            transferred += 1
        elif "moe_ffn" in k:
            # Handle DeepSeek MoE transfer from standard 4-expert MoE
            parts = k.split(".")
            # Example: encoder.layers.0.moe_ffn.shared_expert.0.weight
            layer_prefix = ".".join(parts[:3])  # encoder.layers.X
            if "shared_expert" in k:
                # Initialize shared expert from expert 0
                sub_k = k.replace("shared_expert", "experts.0")
                if sub_k in state_dict and state_dict[sub_k].shape == v.shape:
                    v8_dict[k].copy_(state_dict[sub_k])
                    transferred += 1
                else:
                    fresh += 1
            elif "routed_experts" in k:
                # Routed experts 0..3 from old experts 0..3; 4..7 from old 0..3 with subtle noise
                expert_idx = int(parts[5])
                old_expert_idx = expert_idx % 4
                sub_k = k.replace(f"routed_experts.{expert_idx}", f"experts.{old_expert_idx}")
                if sub_k in state_dict and state_dict[sub_k].shape == v.shape:
                    noise = (
                        torch.randn_like(v) * 0.005 if expert_idx >= 4 else 0.0
                    )
                    v8_dict[k].copy_(state_dict[sub_k] + noise)
                    transferred += 1
                else:
                    fresh += 1
            elif "router.weight" in k:
                # Router shape was [4, D], now [8, D]
                old_k = k
                if old_k in state_dict:
                    old_w = state_dict[old_k]
                    v8_dict[k][:4, :].copy_(old_w)
                    v8_dict[k][4:, :].copy_(old_w + torch.randn_like(old_w) * 0.005)
                    transferred += 1
                else:
                    fresh += 1
            elif "router.bias" in k:
                old_k = k
                if old_k in state_dict:
                    old_b = state_dict[old_k]
                    v8_dict[k][:4].copy_(old_b)
                    v8_dict[k][4:].fill_(-20.0)
                    transferred += 1
                else:
                    fresh += 1
            else:
                fresh += 1
        else:
            fresh += 1

    v8_model.load_state_dict(v8_dict)

    # Strictly guarantee that auxiliary residual additions start at exactly 0.0:
    # 1. Zero out the shared expert output projections so shared_out == 0 at Step 0
    for layer in v8_model.encoder.layers:
        if hasattr(layer, "moe_ffn") and hasattr(layer.moe_ffn, "shared_expert") and layer.moe_ffn.shared_expert is not None:
            nn.init.zeros_(layer.moe_ffn.shared_expert[3].weight)
            nn.init.zeros_(layer.moe_ffn.shared_expert[3].bias)

    # 2. Zero out the engram memory output projections so engram residual injection == 0 at Step 0
    for mod in v8_model.encoder.engram_modules.values():
        nn.init.zeros_(mod.memory_projection[-1].weight)
        nn.init.zeros_(mod.memory_projection[-1].bias)
        nn.init.constant_(mod.gate_net[-2].bias, -3.0)

    return transferred, fresh
