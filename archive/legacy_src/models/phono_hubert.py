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
    masking_mode: str = "none"          # 'none', 'span', 'specaugment', 'dual'
    mask_prob: float = 0.0              # Default 0.0 for clean CTC alignment; set >0 for span/specaugment
    mask_length: int = 10               # 10 frames * 20ms = 200ms acoustic span

    # Loss weights
    ctc_weight: float = 0.5
    masked_weight: float = 0.5

    # Anti-blank regularization & decoding calibration
    blank_penalty_weight: float = 2.0   # Weight for quadratic penalty on excessive blank emission
    blank_threshold: float = 0.25        # Upper target bound on mean blank probability
    blank_eval_penalty: float = 0.0     # Default logit deduction for blank during evaluation


class PhonoHuBERTForPreTraining(nn.Module):
    """PhonoHuBERT Model for Direct Phoneme Target Pre-training.
    
    Predicts phoneme token sequences from raw 16kHz speech waveforms using a combination
    of Connectionist Temporal Classification (CTC) and optional Masked Span Modeling.
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

    def _apply_masking(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply parameterized acoustic masking over frame representations."""
        B, T, D = features.shape
        mode = getattr(self.config, "masking_mode", "none")
        mask_prob = getattr(self.config, "mask_prob", 0.0)

        if mode == "none" or mask_prob <= 0.0:
            mask = torch.zeros((B, T), dtype=torch.bool, device=features.device)
            return features, mask

        if mode == "specaugment":
            # SpecAugment time & feature frequency masking without replacing with static embedding
            masked_features = features.clone()
            mask = torch.zeros((B, T), dtype=torch.bool, device=features.device)
            num_mask = int(mask_prob * T / max(1, self.config.mask_length))
            for b in range(B):
                if num_mask > 0 and T > self.config.mask_length:
                    starts = torch.randint(0, T - self.config.mask_length + 1, (num_mask,), device=features.device)
                    for s in starts:
                        masked_features[b, s : s + self.config.mask_length] = 0.0
                        mask[b, s : s + self.config.mask_length] = True
            # Mask random feature channels
            ch_count = max(1, D // 10)
            ch_idx = torch.randperm(D, device=features.device)[:ch_count]
            masked_features[:, :, ch_idx] = 0.0
            return masked_features, mask

        # Classic span masking (replaces with self.mask_embedding)
        mask = torch.zeros((B, T), dtype=torch.bool, device=features.device)
        num_mask = int(mask_prob * T / max(1, self.config.mask_length))
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
        frame_targets: Optional[torch.Tensor] = None,
        frame_lengths: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        output_attentions: bool = True,
        mask_time_indices: bool = True,
        **kwargs,
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

        # 2. Acoustic Masking (Parameterized via config.masking_mode)
        if mask_time_indices and self.training and self.config.mask_prob > 0.0:
            features, mask = self._apply_masking(projected)
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
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1).float()  # (T_frames, B, vocab_size)
            if target_lengths is None:
                target_lengths = torch.full((B,), targets.shape[1], device=audio.device, dtype=torch.long)

            ctc_loss = self.ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
            ctc_loss_val = float(ctc_loss.item())

            # Anti-blank margin regularization
            blank_penalty_weight = getattr(self.config, "blank_penalty_weight", 0.0)
            blank_threshold = getattr(self.config, "blank_threshold", 0.25)
            if blank_penalty_weight > 0.0 and self.training:
                probs = F.softmax(logits, dim=-1)
                blank_probs = probs[:, :, self.config.blank_token_id]
                mask_frames = torch.arange(T_frames, device=audio.device).unsqueeze(0) < input_lengths.unsqueeze(1)
                valid_blank_probs = blank_probs[mask_frames]
                mean_blank_prob = valid_blank_probs.mean() if valid_blank_probs.numel() > 0 else blank_probs.mean()
                blank_loss = blank_penalty_weight * torch.relu(mean_blank_prob - blank_threshold) ** 2
                loss = ctc_loss + blank_loss
            else:
                blank_loss = torch.tensor(0.0, device=audio.device)
                loss = ctc_loss

            # Real phoneme token accuracy based on greedy sequence Levenshtein distance
            with torch.no_grad():
                preds_list = self.decode_greedy(logits, lengths=input_lengths)
                matches = 0.0
                valid_count = 0
                for b in range(B):
                    ref_seq = targets[b, : int(target_lengths[b])].tolist()
                    pred_seq = preds_list[b]
                    if len(ref_seq) > 0:
                        try:
                            import editdistance
                            dist = editdistance.eval(pred_seq, ref_seq)
                            seq_acc = max(0.0, 1.0 - (dist / max(1, len(ref_seq))))
                        except Exception:
                            seq_acc = 1.0 if pred_seq == ref_seq else 0.0
                        matches += seq_acc
                        valid_count += 1
                accuracy = torch.tensor((matches / max(1, valid_count)) * 100.0, device=audio.device)

        return {
            "loss": loss,
            "ctc_loss": ctc_loss_val,
            "blank_loss": float(blank_loss.item()) if loss is not None else 0.0,
            "accuracy": accuracy,
            "logits": logits,
            "input_lengths": input_lengths,
            "output_lengths": input_lengths,
            "mask": mask,
            "attentions": encoder_res.get("attentions"),
            "hidden_states": encoder_res.get("hidden_states"),
        }

    def decode_greedy(
        self,
        logits_or_audio: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        blank_penalty: Optional[float] = None,
    ) -> List[List[int]]:
        """Greedy CTC decoding from either logits (B, T, V) or audio waveform (B, T_audio).
        
        Supports optional calibrated blank_penalty to prevent blank-dominance collapse.
        """
        penalty = blank_penalty if blank_penalty is not None else getattr(self.config, "blank_eval_penalty", 0.0)

        if logits_or_audio.dim() == 3:
            logits = logits_or_audio
            if penalty > 0.0:
                logits = logits.clone()
                logits[:, :, self.config.blank_token_id] -= penalty
            preds = logits.argmax(dim=-1)
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
                if penalty > 0.0:
                    logits = logits.clone()
                    logits[:, :, self.config.blank_token_id] -= penalty
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
