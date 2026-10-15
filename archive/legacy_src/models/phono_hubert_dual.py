"""PhonoHuBERTDual: Dual-Loss Masked Frame-Synchronous + Sequence CTC Speech Transformer.

Solves the CTC blank-collapse pathology by pairing:
1. Frame-Synchronous Masked Phoneme Cross-Entropy (cannot collapse to blank, directly trains acoustic features)
2. Auxiliary Sequence-Level CTC Alignment for clean boundary segmentation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import HuBERTConfig
from src.models.cnn_encoder import HuBERTFeatureEncoder
from src.models.transformer import HuBERTEncoder
from src.data.phoneme_tokenizer import PhonemeTokenizer


@dataclass
class PhonoHuBERTDualConfig(HuBERTConfig):
    """Configuration for PhonoHuBERTDual Model."""

    # Vocabulary & Special Tokens
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

    # Masking settings (HuBERT SSL span masking)
    masking_mode: str = "span"          # 'none', 'span', 'specaugment'
    mask_prob: float = 0.4              # 40% frames masked during SSL pre-training
    mask_length: int = 10               # 10 frames * 20ms = 200ms acoustic span

    # Dual Loss Weights
    frame_loss_weight: float = 0.2      # Weight for frame-level Cross-Entropy regularizer
    ctc_loss_weight: float = 1.0        # Weight for sequence CTC loss
    ignore_index: int = -100            # Padding index for Cross-Entropy loss

    # Anti-Blank Margin Regularization
    blank_penalty_weight: float = 2.0   # Penalty weight on blank over-dominance
    blank_threshold: float = 0.25       # Minimum expected non-blank probability margin
    blank_eval_penalty: float = 0.0     # Calibrated decoding penalty


class PhonoHuBERTDualForPreTraining(nn.Module):
    """Dual-Loss Speech Transformer with simultaneous frame-level Cross-Entropy and CTC heads."""

    def __init__(self, config: Optional[PhonoHuBERTDualConfig] = None):
        super().__init__()
        self.config = config or PhonoHuBERTDualConfig()

        # 1. Temporal Feature Extractor (7-layer 1D CNN downsampling factor 320)
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

        # 5. Dual Projection Heads
        # Frame-level phoneme head (masked frame classification)
        self.frame_head = nn.Linear(self.config.encoder_embed_dim, self.config.vocab_size)
        # Sequence-level CTC head
        self.ctc_head = nn.Linear(self.config.encoder_embed_dim, self.config.vocab_size)

        # 6. Loss Criteria
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
        self.ctc_loss_fn = nn.CTCLoss(
            blank=self.config.blank_token_id,
            zero_infinity=True,
        )

    def _apply_masking(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply acoustic span masking over 20ms frames."""
        B, T, D = features.shape
        mode = getattr(self.config, "masking_mode", "span")
        mask_prob = getattr(self.config, "mask_prob", 0.0)

        if mode == "none" or mask_prob <= 0.0:
            mask = torch.zeros((B, T), dtype=torch.bool, device=features.device)
            return features, mask

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
        """Forward pass through PhonoHuBERTDual."""
        # 1. Temporal CNN Feature Extraction
        cnn_features = self.feature_extractor(audio)           # (B, T_frames, conv_dim)
        projected = self.feature_projection(cnn_features)      # (B, T_frames, embed_dim)
        B, T_frames, D = projected.shape

        if audio_lengths is not None:
            input_lengths = torch.tensor([
                self.config.compute_output_length(int(l.item())) for l in audio_lengths
            ], device=audio.device, dtype=torch.long)
        else:
            input_lengths = torch.full((B,), T_frames, device=audio.device, dtype=torch.long)

        # 2. Acoustic Masking
        if mask_time_indices and self.training and self.config.mask_prob > 0.0:
            features, mask = self._apply_masking(projected)
        else:
            features = projected
            mask = torch.zeros((B, T_frames), dtype=torch.bool, device=audio.device)

        # 3. Contextual Transformer Encoder
        encoder_res = self.encoder(
            features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_state = encoder_res["last_hidden_state"]        # (B, T_frames, embed_dim)

        # 4. Dual Logits
        frame_logits = self.frame_head(hidden_state)           # (B, T_frames, vocab_size)
        ctc_logits = self.ctc_head(hidden_state)               # (B, T_frames, vocab_size)

        loss = None
        ce_loss_val = 0.0
        ctc_loss_val = 0.0
        frame_acc = None
        seq_acc = None

        # 5. Frame Cross-Entropy Loss
        if frame_targets is not None:
            # Align temporal dimension if length differs slightly due to CNN padding
            target_T = frame_targets.shape[1]
            if target_T != T_frames:
                min_T = min(target_T, T_frames)
                fl_slice = frame_logits[:, :min_T, :].contiguous()
                ft_slice = frame_targets[:, :min_T].contiguous()
            else:
                fl_slice = frame_logits
                ft_slice = frame_targets

            ce_loss = self.ce_loss_fn(
                fl_slice.view(-1, self.config.vocab_size),
                ft_slice.view(-1),
            )
            ce_loss_val = float(ce_loss.item())

            # Frame accuracy
            with torch.no_grad():
                valid_mask = ft_slice != self.config.ignore_index
                if valid_mask.any():
                    pred_frame_tokens = fl_slice.argmax(dim=-1)
                    correct = (pred_frame_tokens == ft_slice) & valid_mask
                    frame_acc = (correct.sum().float() / valid_mask.sum().float()) * 100.0

        # 6. Sequence CTC Loss
        if targets is not None:
            log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1).float()  # (T_frames, B, vocab_size)
            if target_lengths is None:
                target_lengths = torch.full((B,), targets.shape[1], device=audio.device, dtype=torch.long)

            ctc_loss = self.ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
            ctc_loss_val = float(ctc_loss.item())

            # Anti-blank margin regularization to prevent all-blank collapse
            blank_penalty_weight = getattr(self.config, "blank_penalty_weight", 0.0)
            blank_threshold = getattr(self.config, "blank_threshold", 0.25)
            if blank_penalty_weight > 0.0 and self.training:
                probs = F.softmax(ctc_logits, dim=-1)
                blank_probs = probs[:, :, self.config.blank_token_id]
                mask_frames = torch.arange(T_frames, device=audio.device).unsqueeze(0) < input_lengths.unsqueeze(1)
                valid_blank_probs = blank_probs[mask_frames]
                mean_blank_prob = valid_blank_probs.mean() if valid_blank_probs.numel() > 0 else blank_probs.mean()
                blank_loss = blank_penalty_weight * torch.relu(mean_blank_prob - blank_threshold) ** 2
                ctc_loss = ctc_loss + blank_loss

            with torch.no_grad():
                preds_list = self.decode_greedy(ctc_logits, lengths=input_lengths)
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
                seq_acc = torch.tensor((matches / max(1, valid_count)) * 100.0, device=audio.device)

        # 7. Total Combined Loss
        if frame_targets is not None and targets is not None:
            w_frame = self.config.frame_loss_weight
            w_ctc = self.config.ctc_loss_weight
            loss = w_frame * ce_loss + w_ctc * ctc_loss
        elif frame_targets is not None:
            loss = ce_loss
        elif targets is not None:
            loss = ctc_loss

        accuracy = seq_acc if seq_acc is not None else frame_acc

        return {
            "loss": loss,
            "ce_loss": ce_loss_val,
            "ctc_loss": ctc_loss_val,
            "accuracy": accuracy,
            "frame_accuracy": frame_acc,
            "seq_accuracy": seq_acc,
            "logits": ctc_logits,
            "frame_logits": frame_logits,
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
        use_frame_head: bool = False,
        blank_penalty: float = 0.0,
    ) -> List[List[int]]:
        """Greedy sequence decoding using either the CTC head or frame head run-length collapse."""
        if logits_or_audio.dim() == 3:
            logits = logits_or_audio
            if blank_penalty > 0.0:
                logits = logits.clone()
                logits[:, :, self.config.blank_token_id] -= blank_penalty
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
                head = self.frame_head if use_frame_head else self.ctc_head
                logits = head(encoder_res["last_hidden_state"])
                if blank_penalty > 0.0:
                    logits = logits.clone()
                    logits[:, :, self.config.blank_token_id] -= blank_penalty
                preds = logits.argmax(dim=-1)

        batch_sequences = []
        for b in range(preds.shape[0]):
            p_seq = preds[b].tolist()
            if lengths is not None and b < len(lengths):
                p_seq = p_seq[: int(lengths[b])]
            decoded = []
            prev = None
            for p in p_seq:
                if p != prev:
                    # Filter blank, pad, silence, noise
                    if p not in (self.config.blank_token_id, self.config.pad_token_id, self.config.silence_token_id, self.config.noise_token_id):
                        decoded.append(p)
                    prev = p
            batch_sequences.append(decoded)
        return batch_sequences
