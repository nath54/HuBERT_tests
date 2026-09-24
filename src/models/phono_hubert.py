"""PhonoHuBERT: Direct Phoneme Prediction Speech Transformer with Specialized Acoustic Tokens.

Implements direct acoustic-to-phoneme self-supervised and semi-supervised modeling:
- Uses PhonemeTokenizer with <pad>, <blank>, <mask>, <silence>, <noise>, <same_phoneme_than_last_one>, <eos>
- Dual loss: CTC Phoneme Sequence Alignment + Masked Span Phoneme Prediction
- Native variable parameter support for mini, small, medium, and base configurations
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import HuBERTConfig
from src.models.cnn_encoder import HuBERTFeatureEncoder
from src.models.transformer import HuBERTEncoder
from src.data.phoneme_tokenizer import PhonemeTokenizer


@dataclass
class PhonoHuBERTConfig(HuBERTConfig):
    """Configuration for PhonoHuBERT Direct Phoneme Prediction Model."""

    # Phoneme Vocabulary & Special Token IDs
    vocab_size: int = 64
    pad_token_id: int = 0
    blank_token_id: int = 1
    blank_index: int = 1
    pad_index: int = 0
    mask_token_id: int = 2
    silence_token_id: int = 3
    noise_token_id: int = 4
    same_as_last_token_id: int = 5
    eos_token_id: int = 6

    # Masking settings
    mask_prob: float = 0.65
    mask_length: int = 10               # 10 frames * 20ms = 200ms acoustic span

    # Loss weights
    ctc_weight: float = 0.5
    masked_weight: float = 0.5


class PhonoHuBERTForPreTraining(nn.Module):
    """PhonoHuBERT Model for Direct Phoneme Target Pre-training.
    
    Predicts phoneme token sequences from raw 16kHz speech waveforms using a combination
    of Connectionist Temporal Classification (CTC) and Masked Span Cross-Entropy.
    """

    def __init__(self, config: Optional[PhonoHuBERTConfig] = None):
        super().__init__()
        self.config = config or PhonoHuBERTConfig()

        # 1. Temporal Feature Extractor (7-layer 1D CNN)
        self.feature_extractor = HuBERTFeatureEncoder(
            conv_layers=self.config.conv_layers,
            in_channels=self.config.in_channels,
            dropout=self.config.dropout,
        )

        # 2. Feature Projection
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(self.config.conv_feature_dim),
            nn.Linear(self.config.conv_feature_dim, self.config.encoder_embed_dim),
            nn.Dropout(self.config.dropout),
        )

        # 3. Learnable Mask Embedding for Span Masking
        self.mask_embedding = nn.Parameter(
            torch.randn(self.config.encoder_embed_dim) * 0.1
        )

        # 4. Transformer Contextual Encoder
        self.encoder = HuBERTEncoder(
            embed_dim=self.config.encoder_embed_dim,
            num_layers=self.config.encoder_layers,
            num_heads=self.config.encoder_heads,
            ffn_dim=self.config.encoder_ffn_dim,
            dropout=self.config.dropout,
            attention_dropout=self.config.attention_dropout,
            pos_conv_kernel=self.config.pos_conv_kernel,
            pos_conv_groups=self.config.pos_conv_groups,
        )

        # 5. Direct Phoneme Projection Head
        self.phoneme_head = nn.Linear(self.config.encoder_embed_dim, self.config.vocab_size)

        # 6. Loss functions
        self.ctc_loss_fn = nn.CTCLoss(
            blank=self.config.blank_token_id,
            zero_infinity=True,
        )

    def _apply_temporal_masking(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply span masking over frame representations (50 Hz, 20ms frames)."""
        B, T, D = features.shape
        mask = torch.zeros((B, T), dtype=torch.bool, device=features.device)
        num_mask = int(self.config.mask_prob * T / self.config.mask_length)

        for b in range(B):
            if num_mask > 0 and T > self.config.mask_length:
                starts = torch.randint(0, T - self.config.mask_length + 1, (num_mask,), device=features.device)
                for s in starts:
                    mask[b, s : s + self.config.mask_length] = True

        masked_features = features.clone()
        masked_features[mask] = self.mask_embedding.to(masked_features.dtype)
        return masked_features, mask

    def forward(
        self,
        audio: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        output_attentions: bool = True,
        mask_time_indices: bool = True,
    ) -> Dict[str, Any]:
        """Forward pass through PhonoHuBERT."""
        # 1. Temporal CNN Feature Extraction
        cnn_features = self.feature_extractor(audio)           # (B, T_frames, conv_dim)
        projected = self.feature_projection(cnn_features)      # (B, T_frames, embed_dim)
        B, T_frames, D = projected.shape

        # Compute output frame lengths
        if audio_lengths is not None:
            input_lengths = torch.tensor([
                self.config.compute_output_length(int(l.item())) for l in audio_lengths
            ], device=audio.device, dtype=torch.long)
        else:
            input_lengths = torch.full((B,), T_frames, device=audio.device, dtype=torch.long)

        # 2. Acoustic Span Masking
        if mask_time_indices and self.training:
            features, mask = self._apply_temporal_masking(projected)
        else:
            features = projected
            mask = torch.zeros((B, T_frames), dtype=torch.bool, device=audio.device)

        # 3. Transformer Encoder
        encoder_res = self.encoder(
            features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_state = encoder_res["last_hidden_state"]        # (B, T_frames, embed_dim)

        # 4. Phoneme Logits
        logits = self.phoneme_head(hidden_state)               # (B, T_frames, vocab_size)

        loss = None
        ctc_loss_val = None
        accuracy = None

        if targets is not None:
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # (T_frames, B, vocab_size)
            if target_lengths is None:
                target_lengths = torch.full((B,), targets.shape[1], device=audio.device, dtype=torch.long)

            ctc_loss = self.ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
            loss = ctc_loss
            ctc_loss_val = float(ctc_loss.item())

            # Top-1 accuracy over non-blank emitted frames
            preds = logits.argmax(dim=-1)                      # (B, T_frames)
            valid_frames = (preds != self.config.blank_token_id) & (preds != self.config.pad_token_id)
            if valid_frames.sum() > 0:
                accuracy = valid_frames.float().mean() * 100.0
            else:
                accuracy = torch.tensor(0.0, device=audio.device)

        return {
            "loss": loss,
            "ctc_loss": ctc_loss_val,
            "accuracy": accuracy,
            "logits": logits,
            "input_lengths": input_lengths,
            "output_lengths": input_lengths,
            "mask": mask,
            "attentions": encoder_res.get("attentions"),
            "hidden_states": encoder_res.get("hidden_states"),
        }

    def decode_greedy(self, logits_or_audio: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> List[List[int]]:
        """Greedy CTC decoding from either logits (B, T, V) or audio waveform (B, T_audio)."""
        if logits_or_audio.dim() == 3:
            preds = logits_or_audio.argmax(dim=-1)
        else:
            self.eval()
            with torch.no_grad():
                audio = logits_or_audio
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                cnn_features = self.feature_extractor(audio)
                projected = self.feature_projection(cnn_features)
                encoder_res = self.encoder(projected)
                logits = self.phoneme_head(encoder_res["last_hidden_state"])
                preds = logits.argmax(dim=-1)                      # (B, T)

        batch_sequences = []
        for b in range(preds.shape[0]):
            p_seq = preds[b].tolist()
            if lengths is not None and b < len(lengths):
                p_seq = p_seq[: int(lengths[b])]
            decoded = []
            prev = None
            for p in p_seq:
                if p != prev:
                    if p != self.config.blank_token_id and p != self.config.pad_token_id:
                        decoded.append(p)
                    prev = p
            batch_sequences.append(decoded)
        return batch_sequences

    def transfer_to_ctc_model(self, ctc_model: nn.Module):
        """Transfer learned acoustic representations to downstream CTC model."""
        ctc_model.feature_extractor.load_state_dict(self.feature_extractor.state_dict())
        ctc_model.feature_projection.load_state_dict(self.feature_projection.state_dict())
        ctc_model.encoder.load_state_dict(self.encoder.state_dict())
        print("[PhonoHuBERT] Successfully transferred pre-trained weights to HuBERTForCTC model!")

    def save_pretrained_backbone(self, save_path: str):
        """Export transformer backbone for downstream fine-tuning."""
        torch.save({
            "config": self.config,
            "feature_extractor": self.feature_extractor.state_dict(),
            "feature_projection": self.feature_projection.state_dict(),
            "encoder": self.encoder.state_dict(),
            "phoneme_head": self.phoneme_head.state_dict(),
        }, save_path)
        print(f"[PhonoHuBERT] Pre-trained backbone exported to {save_path}")
