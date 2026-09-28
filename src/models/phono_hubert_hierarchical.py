"""PhonoHuBERTHierarchical: 2-Stage Gated Hierarchical Speech Transformer.

Architectural Defense Against Blank-Dominance Collapse:
- Decomposes frame decoding into two distinct probabilistic stages:
  1. Acoustic State Router: 4-way classification [blank, silence, noise, speech]
  2. Linguistic Phoneme Head: classifies exclusively among legitimate phonetic tokens (no blank attractor!)
- Log-space composite distribution: log P(phoneme) = log P(speech) + log P(phoneme | speech).
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
class PhonoHuBERTHierarchicalConfig(HuBERTConfig):
    """Configuration for PhonoHuBERTHierarchical Model."""

    # Vocabulary & Special Tokens
    vocab_size: int = 64
    phoneme_offset: int = 8             # Index where phonetic tokens begin (after special tokens 0..7)
    pad_token_id: int = 0
    blank_token_id: int = 1
    blank_index: int = 1
    pad_index: int = 0
    mask_token_id: int = 2
    silence_token_id: int = 3
    noise_token_id: int = 4
    same_as_last_token_id: int = 5
    eos_token_id: int = 6
    unk_token_id: int = 7

    # Router & Head settings
    num_acoustic_states: int = 4        # 0: blank, 1: silence, 2: noise, 3: speech
    router_loss_weight: float = 0.5     # Weight for acoustic state router auxiliary supervision
    ctc_loss_weight: float = 1.0        # Weight for composite CTC sequence loss
    blank_penalty_weight: float = 2.0   # Penalty weight on blank over-dominance
    blank_threshold: float = 0.25       # Minimum expected non-blank probability margin
    blank_eval_penalty: float = 0.0     # Evaluation calibration penalty for blank


class PhonoHuBERTHierarchicalForPreTraining(nn.Module):
    """Hierarchical Gated Speech Transformer with Acoustic State Router and Phoneme Head."""

    def __init__(self, config: Optional[PhonoHuBERTHierarchicalConfig] = None):
        super().__init__()
        self.config = config or PhonoHuBERTHierarchicalConfig()
        self.phoneme_offset = self.config.phoneme_offset
        self.num_phonemes = max(1, self.config.vocab_size - self.phoneme_offset)

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

        # 3. Contextual Transformer Encoder
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

        # 4. Hierarchical Two-Stage Heads
        # Stage 1: Acoustic State Router (blank=0, silence=1, noise=2, speech=3)
        self.state_router = nn.Linear(self.config.encoder_embed_dim, self.config.num_acoustic_states)
        # Stage 2: Linguistic Phoneme Classifier (clean phonemes only)
        self.phoneme_head = nn.Linear(self.config.encoder_embed_dim, self.num_phonemes)

        # 5. Loss Criteria
        self.ctc_loss_fn = nn.CTCLoss(
            blank=self.config.blank_token_id,
            zero_infinity=True,
        )

    def compute_composite_log_probs(self, state_logits: torch.Tensor, phoneme_logits: torch.Tensor) -> torch.Tensor:
        """Form normalized composite full-vocabulary log-probabilities from hierarchical outputs."""
        B, T_frames, _ = state_logits.shape
        log_p_state = F.log_softmax(state_logits, dim=-1)
        log_p_phoneme = F.log_softmax(phoneme_logits, dim=-1)

        composite_log_probs = torch.full(
            (B, T_frames, self.config.vocab_size),
            -100.0,
            device=state_logits.device,
            dtype=state_logits.dtype,
        )
        # Non-speech acoustic state emissions
        composite_log_probs[:, :, self.config.blank_token_id] = log_p_state[:, :, 0]
        composite_log_probs[:, :, self.config.silence_token_id] = log_p_state[:, :, 1]
        composite_log_probs[:, :, self.config.noise_token_id] = log_p_state[:, :, 2]

        # Linguistic speech phoneme emissions: log P(phoneme) = log P(speech) + log P(phoneme | speech)
        speech_log_p = log_p_state[:, :, 3].unsqueeze(-1)
        composite_log_probs[:, :, self.phoneme_offset:] = speech_log_p + log_p_phoneme
        return composite_log_probs

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
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass through Hierarchical Gated Speech Transformer."""
        # 1. Feature Extraction & Projection
        cnn_features = self.feature_extractor(audio)           # (B, T_frames, conv_dim)
        projected = self.feature_projection(cnn_features)      # (B, T_frames, embed_dim)
        B, T_frames, D = projected.shape

        if audio_lengths is not None:
            input_lengths = torch.tensor([
                self.config.compute_output_length(int(l.item())) for l in audio_lengths
            ], device=audio.device, dtype=torch.long)
        else:
            input_lengths = torch.full((B,), T_frames, device=audio.device, dtype=torch.long)

        # 2. Transformer Contextual Encoder
        encoder_res = self.encoder(
            projected,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_state = encoder_res["last_hidden_state"]        # (B, T_frames, embed_dim)

        # 3. Two-Stage Hierarchical Projections
        state_logits = self.state_router(hidden_state)         # (B, T_frames, 4)
        phoneme_logits = self.phoneme_head(hidden_state)       # (B, T_frames, num_phonemes)

        # 4. Form Normalized Composite Full-Vocabulary Log-Probabilities
        composite_log_probs = self.compute_composite_log_probs(state_logits, phoneme_logits)

        loss = None
        ctc_loss_val = 0.0
        router_loss_val = 0.0
        accuracy = None

        # 5. Composite Sequence CTC Loss
        if targets is not None:
            # Transpose to (T, B, V) for CTCLoss
            ctc_log_probs = composite_log_probs.transpose(0, 1).float()
            if target_lengths is None:
                target_lengths = torch.full((B,), targets.shape[1], device=audio.device, dtype=torch.long)

            ctc_loss = self.ctc_loss_fn(ctc_log_probs, targets, input_lengths, target_lengths)
            ctc_loss_val = float(ctc_loss.item())

            # Anti-blank margin regularization on composite state probs
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

            # Auxiliary Router Supervision from frame targets if available
            if frame_targets is not None:
                target_T = frame_targets.shape[1]
                min_T = min(target_T, T_frames)
                sl_slice = state_logits[:, :min_T, :].contiguous()
                ft_slice = frame_targets[:, :min_T].contiguous()

                state_targets = torch.full_like(ft_slice, -100)
                valid_ft = (ft_slice != -100)
                state_targets[valid_ft & (ft_slice == self.config.blank_token_id)] = 0
                state_targets[valid_ft & (ft_slice == self.config.silence_token_id)] = 1
                state_targets[valid_ft & (ft_slice == self.config.noise_token_id)] = 2
                state_targets[
                    valid_ft
                    & (ft_slice != self.config.blank_token_id)
                    & (ft_slice != self.config.silence_token_id)
                    & (ft_slice != self.config.noise_token_id)
                ] = 3

                if (state_targets != -100).any():
                    router_loss = F.cross_entropy(
                        sl_slice.view(-1, self.config.num_acoustic_states),
                        state_targets.view(-1),
                        ignore_index=-100,
                    )
                    router_loss_val = float(router_loss.item())
                    loss = loss + self.config.router_loss_weight * router_loss

            # Phoneme Sequence Accuracy
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
            "router_loss": router_loss_val,
            "accuracy": accuracy,
            "logits": composite_log_probs,
            "state_logits": state_logits,
            "phoneme_logits": phoneme_logits,
            "input_lengths": input_lengths,
            "output_lengths": input_lengths,
            "attentions": encoder_res.get("attentions"),
            "hidden_states": encoder_res.get("hidden_states"),
        }

    def decode_greedy(
        self,
        logits_or_audio: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        blank_penalty: Optional[float] = None,
    ) -> List[List[int]]:
        """Greedy CTC sequence decoding over composite log-probabilities."""
        penalty = blank_penalty if blank_penalty is not None else getattr(self.config, "blank_eval_penalty", 0.0)

        if logits_or_audio.dim() == 3:
            log_probs = logits_or_audio
            if penalty > 0.0:
                log_probs = log_probs.clone()
                log_probs[:, :, self.config.blank_token_id] -= penalty
            preds = log_probs.argmax(dim=-1)
        else:
            self.eval()
            with torch.no_grad():
                audio = logits_or_audio
                if audio.dim() == 1:
                    audio = audio.unsqueeze(0)
                out = self.forward(audio=audio)
                log_probs = out["logits"]
                if penalty > 0.0:
                    log_probs = log_probs.clone()
                    log_probs[:, :, self.config.blank_token_id] -= penalty
                preds = log_probs.argmax(dim=-1)

        batch_sequences = []
        for b in range(preds.shape[0]):
            p_seq = preds[b].tolist()
            if lengths is not None and b < len(lengths):
                p_seq = p_seq[: int(lengths[b])]
            decoded = []
            prev = None
            for p in p_seq:
                if p != prev:
                    if p not in (self.config.blank_token_id, self.config.pad_token_id, self.config.silence_token_id, self.config.noise_token_id):
                        decoded.append(p)
                    prev = p
            batch_sequences.append(decoded)
        return batch_sequences
