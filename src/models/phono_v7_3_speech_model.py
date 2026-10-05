"""Phono-V7.3 Real-Time Streaming Speech Model.

Major Innovations over V7.2:
1. Multi-Scale Dilated Conformer Convolutions (3 branches: 300ms, 1220ms, 2420ms receptive field)
   for 4x wider acoustic context without parameter increase.
2. Decoupled Word Boundary Gate Head (3-class: SPEECH, WORD_SPACE, SILENCE_BLANK) with weighted loss.
3. Recursive 2-Pass Phoneme Head with causal depthwise recurrent refinement for phonotactic memory.
4. Boundary-Gated Online Slicing with multi-frame hysteresis (silence run confirmation & min burst protection).
5. 100% Zero-Perturbation Warm-Start from V7.1 and V7.2 checkpoints.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.conformer_layers import ConformerMoETransformerEncoder
from src.models.phono_variants import PhonoV64GatedDiffusionConfig, PhonoV64GatedDiffusionForPreTraining
from src.models.phono_v7_1_alignment import extract_streaming_ctc_slices
from src.models.phono_v7_1_speech_model import PhonoV71SpeechConfig
from src.models.phono_v7_2_speech_model import PhonoV72Decoder, PhonoV72SpeechModel


class DecoupledBoundaryGateHead(nn.Module):
    """Dedicated 3-class boundary gate head for temporal speech segmentation.

    Classes:
    - 0: SPEECH (voiced phonemes inside words)
    - 1: WORD_SPACE (inter-word boundary delimiter)
    - 2: SILENCE_BLANK (ambient silence / non-emitting transition)
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, 3)
        # Zero-initialize so all classes start with equal probability (zero perturbation)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, T, D] acoustic representations
        Returns:
            logits: [B, T, 3] boundary class logits
        """
        return self.proj(h)


class RecursivePhonemeHead(nn.Module):
    """Recursive 2-Pass Phoneme Emission Head with Causal Phonotactic Refinement.

    Pass 1: Base linear projection h_t -> z_t^(0) (inherits V7.2 phoneme_head weights).
    Pass 2: Causal depthwise separable convolution taking [h_t; softmax(z_{t-1}^(0))]
            to enforce phonotactic grammar (no adjacent spaces, smoothed consonant closures).
    """

    def __init__(self, embed_dim: int, vocab_size: int, refiner_dim: int = 128, kernel_size: int = 5):
        super().__init__()
        self.vocab_size = vocab_size
        self.base_head = nn.Linear(embed_dim, vocab_size)

        # Pass 2: Causal Refiner
        self.kernel_size = kernel_size
        self.pad_len = kernel_size - 1  # Strictly causal padding

        in_channels = embed_dim + vocab_size
        self.refiner_in = nn.Conv1d(in_channels, refiner_dim, kernel_size=1)
        self.refiner_conv = nn.Conv1d(
            refiner_dim,
            refiner_dim,
            kernel_size=kernel_size,
            padding=self.pad_len,
            groups=refiner_dim,
            bias=False,
        )
        self.refiner_norm = nn.BatchNorm1d(refiner_dim)
        self.refiner_act = nn.SiLU()
        self.refiner_out = nn.Linear(refiner_dim, vocab_size)

        # Zero-initialize the refiner output so on step 0, refined_logits == base_logits
        nn.init.zeros_(self.refiner_out.weight)
        if self.refiner_out.bias is not None:
            nn.init.zeros_(self.refiner_out.bias)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [B, T, D] acoustic representations
        Returns:
            refined_logits: [B, T, V]
            base_logits: [B, T, V]
        """
        B, T, D = h.shape

        # Pass 1: Base CTC logits
        base_logits = self.base_head(h)  # [B, T, V]

        # Shifted past emission probabilities (causal context at t-1)
        base_probs = F.softmax(base_logits, dim=-1)
        zeros_t0 = torch.zeros((B, 1, self.vocab_size), device=h.device, dtype=h.dtype)
        shifted_probs = torch.cat([zeros_t0, base_probs[:, :-1, :]], dim=1)  # [B, T, V]

        # Pass 2: Causal Refinement
        u = torch.cat([h, shifted_probs], dim=-1).transpose(1, 2)  # [B, D+V, T]
        r = self.refiner_in(u)
        r = self.refiner_conv(r)
        if self.pad_len > 0:
            r = r[:, :, :T]  # Enforce causal length truncation
        r = self.refiner_norm(r)
        r = self.refiner_act(r)
        r = r.transpose(1, 2)  # [B, T, refiner_dim]

        residual = self.refiner_out(r)  # [B, T, V]
        refined_logits = base_logits + residual

        return refined_logits, base_logits


class PhonoV73AcousticBackbone(PhonoV64GatedDiffusionForPreTraining):
    """Conformer Acoustic Backbone for Phono-V7.3 with Multi-Scale Dilated Convolutions,
    Decoupled Boundary Gate, and Recursive Phoneme Emission Head."""

    def __init__(self, config: PhonoV64GatedDiffusionConfig, conformer_kernel_size: int = 31):
        super().__init__(config)

        # Multi-Scale Dilated Conformer Encoder
        self.encoder = ConformerMoETransformerEncoder(
            embed_dim=self.config.encoder_embed_dim,
            num_layers=self.config.encoder_layers,
            num_heads=self.config.encoder_heads,
            ffn_dim=self.config.encoder_ffn_dim,
            num_experts=getattr(self.config, "num_experts", 4),
            top_k=getattr(self.config, "moe_top_k", 2),
            kernel_size=conformer_kernel_size,
            dropout=self.config.dropout,
            attention_dropout=self.config.attention_dropout,
            pos_conv_kernel=self.config.pos_conv_kernel,
            pos_conv_groups=self.config.pos_conv_groups,
            use_multi_scale_conv=True,
        )

        # 1. Decoupled Word Boundary Gate
        self.boundary_gate_head = DecoupledBoundaryGateHead(self.config.encoder_embed_dim)

        # 2. Recursive 2-Pass Phoneme Head
        self.phoneme_head = RecursivePhonemeHead(
            embed_dim=self.config.encoder_embed_dim,
            vocab_size=getattr(self, "num_phonemes", getattr(self.config, "vocab_size", 56)),
        )

    def forward(
        self,
        audio: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        boundary_targets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # 1. CNN acoustic feature extraction & projection
        cnn_features = self.feature_extractor(audio)
        features = self.feature_projection(cnn_features)
        input_lengths = self._get_feat_extract_output_lengths(audio_lengths) if audio_lengths is not None else None

        # 2. Multi-Scale Conformer Encoding
        enc_out = self.encoder(features)
        hidden_state = enc_out.get("last_hidden_state", enc_out.get("hidden_state"))

        # 3. Decoupled Boundary Gate Emissions
        boundary_logits = self.boundary_gate_head(hidden_state)  # [B, T, 3]

        # 4. Recursive Phoneme Emissions
        refined_p_logits, base_p_logits = self.phoneme_head(hidden_state)  # [B, T, 56]
        state_logits = self.state_router(hidden_state)  # [B, T, 4]

        refined_logits = self.compute_composite_log_probs(state_logits, refined_p_logits)  # [B, T, 64]
        base_logits = self.compute_composite_log_probs(state_logits, base_p_logits)  # [B, T, 64]

        # 5. CTC Loss computation
        ctc_loss = torch.tensor(0.0, device=audio.device)
        boundary_loss = torch.tensor(0.0, device=audio.device)

        if targets is not None:
            B_cur, T_cur, _ = refined_logits.shape
            log_probs = refined_logits.transpose(0, 1).float()  # (T, B, 64)
            i_lens = input_lengths if input_lengths is not None else torch.full((B_cur,), T_cur, device=audio.device, dtype=torch.long)
            t_lens = target_lengths if target_lengths is not None else torch.full((B_cur,), targets.shape[1], device=audio.device, dtype=torch.long)

            ctc_loss = self.ctc_loss_fn(log_probs, targets, i_lens, t_lens)

        # 6. Boundary Gate Cross-Entropy Loss with weighted space penalty
        if boundary_targets is not None:
            T_cur = boundary_logits.shape[1]
            T_bt = boundary_targets.shape[1]
            T_common = min(T_cur, T_bt)
            bg_sliced = boundary_logits[:, :T_common]
            bt_sliced = boundary_targets[:, :T_common]
            weights = torch.tensor([1.0, 4.0, 1.0], device=audio.device)  # 4x penalty for missed/false spaces
            boundary_loss = F.cross_entropy(
                bg_sliced.reshape(-1, 3),
                bt_sliced.reshape(-1),
                weight=weights,
                ignore_index=-100,
            )

        total_enc_loss = ctc_loss + 0.3 * boundary_loss

        return {
            "loss": total_enc_loss,
            "ctc_loss": ctc_loss,
            "boundary_loss": boundary_loss,
            "logits": refined_logits,
            "base_logits": base_logits,
            "boundary_logits": boundary_logits,
            "hidden_state": hidden_state,
            "input_lengths": input_lengths,
        }


class PhonoV73SpeechModel(PhonoV72SpeechModel):
    """Phono-V7.3 Speech Model with Multi-Scale Dilated Conformer, Decoupled Boundary Gate,
    Recursive 2-Pass Phoneme Head, and Level-1/Level-2 Partitioned MoE Decoder."""

    def __init__(self, config: Optional[PhonoV71SpeechConfig] = None):
        cfg = config or PhonoV71SpeechConfig.medium()
        super().__init__(cfg)

        # Replace encoder with PhonoV73AcousticBackbone
        self.encoder = PhonoV73AcousticBackbone(
            cfg.encoder_config,
            conformer_kernel_size=cfg.conformer_kernel_size,
        )

        # Decoder is PhonoV72Decoder (Level-1 Gate + Level-2 Partitioned MoE + Horizon Capping)
        self.decoder = PhonoV72Decoder(
            config=cfg.decoder_config,
            levenshtein_loss_weight=cfg.levenshtein_loss_weight,
            levenshtein_gamma=cfg.levenshtein_gamma,
            band_window_words=cfg.band_window_words,
            macro_history_noise_std=cfg.macro_history_noise_std,
            length_mode=cfg.length_mode,
            asymmetric_beta_under=cfg.asymmetric_beta_under,
            asymmetric_beta_over=cfg.asymmetric_beta_over,
            asymmetric_delta=cfg.asymmetric_delta,
            length_loss_weight=cfg.length_loss_weight,
        )

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        phoneme_targets: Optional[torch.Tensor] = None,
        phoneme_lengths: Optional[torch.Tensor] = None,
        num_words: Optional[torch.Tensor] = None,
        input_byte_ids: Optional[torch.Tensor] = None,
        target_byte_ids: Optional[torch.Tensor] = None,
        path_targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        boundary_targets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # 1. Acoustic Forward Pass
        enc_out = self.encoder(
            audio=audio,
            audio_lengths=audio_lengths,
            targets=phoneme_targets,
            target_lengths=phoneme_lengths,
            boundary_targets=boundary_targets,
            **kwargs,
        )
        hidden_state = enc_out["hidden_state"]
        input_lengths = enc_out.get("input_lengths")
        refined_logits = enc_out["logits"]
        boundary_logits = enc_out["boundary_logits"]
        enc_loss = enc_out.get("loss", torch.tensor(0.0, device=audio.device))

        # 2. Forced Alignment (during training)
        word_centers = None
        align_loss = torch.tensor(0.0, device=audio.device)
        boundary_loss = enc_out.get("boundary_loss", torch.tensor(0.0, device=audio.device))
        if self.config.use_forced_alignment and phoneme_targets is not None and refined_logits is not None:
            try:
                B_cur, T_frames, _ = refined_logits.shape
                log_probs = refined_logits
                i_lens = input_lengths if input_lengths is not None else torch.full((B_cur,), T_frames, device=audio.device)
                p_lens = phoneme_lengths if phoneme_lengths is not None else torch.full((B_cur,), phoneme_targets.shape[1], device=audio.device)

                aligned_tokens, _ = self.aligner.align(
                    log_probs=log_probs,
                    targets=phoneme_targets,
                    input_lengths=i_lens,
                    target_lengths=p_lens,
                )
                if aligned_tokens is not None:
                    # Ground-truth boundary targets from alignment (class 1 for space, 0 for phoneme, 2 for blank)
                    if boundary_targets is None:
                        is_space_token = (aligned_tokens == 8)
                        is_blank_token = (aligned_tokens == 1)
                        auto_boundary = torch.zeros_like(aligned_tokens)
                        auto_boundary[is_space_token] = 1
                        auto_boundary[is_blank_token] = 2
                        weights = torch.tensor([1.0, 4.0, 1.0], device=audio.device)
                        b_loss = F.cross_entropy(
                            boundary_logits.reshape(-1, 3),
                            auto_boundary.reshape(-1),
                            weight=weights,
                            ignore_index=-100,
                        )
                        enc_loss = enc_loss + 0.3 * b_loss
                        boundary_loss = b_loss

                    centers = self.aligner.get_word_centers(aligned_tokens, space_id=8)
                    if centers is not None and num_words is not None:
                        L_req = int(num_words.max().item())
                        pad_w = L_req - centers.shape[1]
                        if pad_w > 0:
                            centers = F.pad(centers, (0, pad_w), value=0)
                        word_centers = centers[:, :L_req]
            except Exception:
                word_centers = None

        # 3. Micro Character Decoder Forward Pass with Boundary-Guided Slicing
        dec_out = self.decoder(
            acoustic_memory=hidden_state,
            memory_lengths=input_lengths,
            num_words=num_words,
            ctc_logits=refined_logits,
            forced_word_centers=word_centers,
            input_byte_ids=input_byte_ids,
            target_byte_ids=target_byte_ids,
            path_targets=path_targets,
            target_lengths=target_lengths,
        )

        dec_loss = dec_out["loss"]
        total_loss = dec_loss + (self.config.ctc_loss_weight * enc_loss) + align_loss

        return {
            "loss": total_loss,
            "total_loss": total_loss,
            "dec_loss": dec_loss,
            "enc_loss": enc_loss,
            "boundary_loss": boundary_loss,
            "char_loss": dec_out["char_loss"],
            "lev_loss": dec_out.get("lev_loss", torch.tensor(0.0)),
            "align_loss": align_loss,
            "path_loss": dec_out["path_loss"],
            "length_loss": dec_out.get("length_loss", torch.tensor(0.0, device=audio.device)),
            "length_headroom": dec_out.get("length_headroom", torch.tensor(0.0, device=audio.device)),
            "diff_loss": dec_out.get("diff_loss", torch.tensor(0.0, device=audio.device)),
            "char_acc": dec_out["char_acc"],
            "path_acc": dec_out["path_acc"],
            "near_acc": dec_out.get("near_acc", torch.tensor(0.0, device=audio.device)),
            "aux_loss": dec_out["aux_loss"],
            "logits": dec_out["logits"],
            "path_logits": dec_out["path_logits"],
            "k_hat": dec_out.get("k_hat"),
            "ctc_logits": refined_logits,
            "boundary_logits": boundary_logits,
        }

    def warm_start_from_v7_2(self, checkpoint_path: str) -> Dict[str, int]:
        """Loads weights from a V7.1 or V7.2 checkpoint with zero-perturbation preservation."""
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)

        model_dict = self.state_dict()
        transferred = 0
        skipped = 0

        for k, v in state_dict.items():
            # Direct match
            if k in model_dict and model_dict[k].shape == v.shape:
                model_dict[k].copy_(v)
                transferred += 1
            # Map legacy linear phoneme_head to RecursivePhonemeHead.base_head
            elif "encoder.phoneme_head.weight" in k and "encoder.phoneme_head.base_head.weight" in model_dict:
                model_dict["encoder.phoneme_head.base_head.weight"].copy_(v)
                transferred += 1
            elif "encoder.phoneme_head.bias" in k and "encoder.phoneme_head.base_head.bias" in model_dict:
                model_dict["encoder.phoneme_head.base_head.bias"].copy_(v)
                transferred += 1
            else:
                skipped += 1

        self.load_state_dict(model_dict)

        missing_keys = [
            k for k in model_dict.keys()
            if k not in state_dict
            and "base_head" not in k
            and "refiner" not in k
            and "boundary_gate" not in k
            and "conv_d" not in k
            and "dilated_proj" not in k
        ]
        return {
            "transferred": transferred,
            "skipped": skipped,
            "missing": len(missing_keys),
        }
