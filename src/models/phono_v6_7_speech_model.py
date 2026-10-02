"""Phono-V6.7 End-to-End Joint Continuous Speech Model with Double Loss.

Integrates:
1. First Model (Acoustic Backbone): PhonoV64GatedDiffusionForPreTraining
   - Ingests raw 16kHz audio waveforms
   - Evaluated / penalized with CTC Phoneme Loss and Confidence-Gated Latent Diffusion Loss
   - Produces continuous 50Hz speech representations H in R^{B x T x 512}
2. Second Model (Character Decoder): PhonoV67WindowedDecoder
   - Ingests H directly via cross-attention with sliding multi-word context (W=4 preceding words)
   - 4-Path length-adaptive routing (Special, Short, Medium, Long)
   - Micro recursive character head with 30% scheduled sampling
3. Double Loss Training Objective:
   L_total = L_decoder + lambda_ctc * L_phoneme_ctc
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.models.phono_variants import (
    PhonoV64GatedDiffusionConfig,
    PhonoV64GatedDiffusionForPreTraining,
)
from src.models.phono_v6_7_windowed_decoder import (
    WindowedAdaptivePathConfig,
    PhonoV67WindowedDecoder,
)


@dataclass
class PhonoV67SpeechConfig:
    """Configuration for composite Phono-V6.7 speech-to-text model."""

    encoder_config: PhonoV64GatedDiffusionConfig = field(
        default_factory=lambda: PhonoV64GatedDiffusionConfig.medium()
    )
    decoder_config: WindowedAdaptivePathConfig = field(
        default_factory=lambda: WindowedAdaptivePathConfig.medium()
    )

    # Double Loss weighting
    ctc_loss_weight: float = 0.5  # Weight of encoder's CTC phoneme loss in total loss
    encoder_learning_rate: float = 1e-5  # Fine-tuning LR for pretrained 5.87% PER encoder
    decoder_learning_rate: float = 3e-4  # Primary LR for decoder

    @classmethod
    def medium(cls, **kwargs) -> "PhonoV67SpeechConfig":
        enc_cfg = PhonoV64GatedDiffusionConfig(
            encoder_layers=8,
            encoder_heads=8,
            encoder_embed_dim=512,
            encoder_ffn_dim=2048,
            vocab_size=64,
            num_experts=4,
            moe_top_k=2,
            use_deep_refiner=True,
            learnable_gating=True,
        )
        dec_cfg = WindowedAdaptivePathConfig.medium()
        cfg = cls(encoder_config=enc_cfg, decoder_config=dec_cfg)
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


class PhonoV67SpeechModel(nn.Module):
    """End-to-End Joint Continuous Speech Model with Double Loss."""

    def __init__(self, config: PhonoV67SpeechConfig):
        super().__init__()
        self.config = config
        self.encoder = PhonoV64GatedDiffusionForPreTraining(config.encoder_config)
        self.decoder = PhonoV67WindowedDecoder(config.decoder_config)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        # Encoder phoneme supervision (Loss 1)
        phoneme_targets: Optional[torch.Tensor] = None,
        phoneme_lengths: Optional[torch.Tensor] = None,
        frame_targets: Optional[torch.Tensor] = None,
        frame_lengths: Optional[torch.Tensor] = None,
        # Decoder word & character supervision (Loss 2)
        num_words: Optional[torch.Tensor] = None,
        input_byte_ids: Optional[torch.Tensor] = None,
        target_byte_ids: Optional[torch.Tensor] = None,
        path_targets: Optional[torch.Tensor] = None,
        expected_total_len: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass with double loss: CTC phoneme loss + character decoder loss.

        Args:
            audio: [B, num_samples] 16kHz audio waveforms.
            audio_lengths: [B] valid audio sample counts.
            phoneme_targets: [B, S_ph] or 1D target phoneme token IDs for CTC loss.
            phoneme_lengths: [B] valid phoneme token lengths.
            frame_targets: [B, T_frames] frame-aligned phoneme targets.
            frame_lengths: [B] valid frame counts.
            num_words: [B] number of words per utterance.
            input_byte_ids: [B, L, K] teacher-forced input character IDs.
            target_byte_ids: [B, L, K] target character IDs (-100 for padding).
            path_targets: [B, L] 4-path length target IDs (0..3).
            expected_total_len: Max sequence length for banded attention.

        Returns:
            Dict containing total_loss, enc_loss, dec_loss, path_acc, logits, etc.
        """
        # 1. Forward through Model 1 (PhonoV6.4 Gated Diffusion Backbone)
        enc_out = self.encoder(
            audio=audio,
            targets=phoneme_targets,
            target_lengths=phoneme_lengths,
            audio_lengths=audio_lengths,
            frame_targets=frame_targets,
            frame_lengths=frame_lengths,
            **kwargs,
        )

        hidden_state = enc_out["hidden_state"]  # [B, T_frames, 512] continuous speech latents
        input_lengths = enc_out.get("input_lengths")
        enc_loss = enc_out.get("loss", torch.tensor(0.0, device=audio.device))

        # 2. Forward through Model 2 (PhonoV6.7 Windowed MoE Character Decoder)
        # Ingests continuous hidden_state directly with zero discrete phoneme conversion
        dec_out = self.decoder(
            acoustic_memory=hidden_state,
            memory_lengths=input_lengths,
            num_words=num_words,
            input_byte_ids=input_byte_ids,
            target_byte_ids=target_byte_ids,
            path_targets=path_targets,
            expected_total_len=expected_total_len,
        )

        dec_loss = dec_out["loss"]

        # 3. Double Loss Objective
        # Both the acoustic encoder and character decoder receive gradient updates
        if phoneme_targets is not None and self.training:
            total_loss = dec_loss + (self.config.ctc_loss_weight * enc_loss)
        else:
            total_loss = dec_loss

        return {
            "loss": total_loss,
            "total_loss": total_loss,
            "enc_loss": enc_loss,
            "dec_loss": dec_loss,
            "char_loss": dec_out.get("char_loss", torch.tensor(0.0)),
            "path_loss": dec_out.get("path_loss", torch.tensor(0.0)),
            "path_acc": dec_out.get("path_acc", torch.tensor(0.0)),
            "aux_loss": dec_out.get("aux_loss", torch.tensor(0.0)),
            "logits": dec_out.get("logits"),
            "path_logits": dec_out.get("path_logits"),
            "enc_logits": enc_out.get("logits"),
            "refined_logits": enc_out.get("refined_logits"),
            "gate_bypass_pct": enc_out.get("gate_bypass_pct", 0.0),
            "gate_full_pct": enc_out.get("gate_full_pct", 0.0),
        }

    def warm_start_from_checkpoints(
        self,
        encoder_checkpoint_path: Optional[str] = None,
        decoder_checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Warm start encoder and decoder from their respective best checkpoints.

        Args:
            encoder_checkpoint_path: Path to PhonoV6.4 Gated Diffusion checkpoint (5.87% PER).
            decoder_checkpoint_path: Path to PhonoV6.6 / V6.7 Windowed Decoder checkpoint.

        Returns:
            Dict reporting transferred parameters and matching status.
        """
        results = {"encoder": None, "decoder": None}

        # 1. Warm-start Encoder
        if encoder_checkpoint_path:
            ckpt = torch.load(encoder_checkpoint_path, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            if any(k.startswith("encoder.feature_extractor") for k in state_dict):
                clean_sd = {
                    (k[len("encoder."):] if k.startswith("encoder.") else k): v
                    for k, v in state_dict.items()
                }
            else:
                clean_sd = state_dict
            enc_missing, enc_unexpected = self.encoder.load_state_dict(clean_sd, strict=False)
            results["encoder"] = {
                "transferred": len(clean_sd) - len(enc_unexpected),
                "missing": len(enc_missing),
                "unexpected": len(enc_unexpected),
            }

        # 2. Warm-start Decoder
        if decoder_checkpoint_path:
            ckpt = torch.load(decoder_checkpoint_path, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            if any(k.startswith("decoder.macro_layers") for k in state_dict):
                clean_sd = {
                    (k[len("decoder."):] if k.startswith("decoder.") else k): v
                    for k, v in state_dict.items()
                }
            else:
                clean_sd = state_dict
            dec_missing, dec_unexpected = self.decoder.load_state_dict(clean_sd, strict=False)
            results["decoder"] = {
                "transferred": len(clean_sd) - len(dec_unexpected),
                "missing": len(dec_missing),
                "unexpected": len(dec_unexpected),
            }

        return results

    @torch.no_grad()
    def decode_greedy(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        max_words: Optional[int] = None,
        max_bytes_per_word: int = 24,
    ) -> List[List[List[int]]]:
        """Greedy inference directly from raw 16kHz audio to character sequences.

        Args:
            audio: [B, num_samples] audio waveforms.
            audio_lengths: [B] valid audio sample counts.
            max_words: Maximum words to decode (inferred from duration if None).
            max_bytes_per_word: Max characters per word.

        Returns:
            List of batch utterances, each being a list of decoded word token lists.
        """
        self.eval()
        enc_out = self.encoder(audio=audio, audio_lengths=audio_lengths)
        hidden_state = enc_out["hidden_state"]
        input_lengths = enc_out.get("input_lengths")

        return self.decoder.generate(
            acoustic_memory=hidden_state,
            memory_lengths=input_lengths,
            max_words=max_words,
        )
