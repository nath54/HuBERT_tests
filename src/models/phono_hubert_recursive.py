"""PhonoHuBERTRecursive: Recurrent Temporal Feedback Speech Transformer.

Breaks the CTC frame-independence assumption by equipping the acoustic Transformer
with an autoregressive recurrent feedback head:
- Each frame's prediction conditions on both the acoustic context h_t and the previous frame emission y_{t-1}.
- Sustained phonemes and transitions are modeled explicitly via recurrent hidden state memory.
- Eliminates blank-dominance attractors and sudden empty-sequence collapses.
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
class PhonoHuBERTRecursiveConfig(HuBERTConfig):
    """Configuration for PhonoHuBERTRecursive Model."""

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

    # Recurrent Feedback Head settings
    token_embed_dim: int = 64           # Dimension of token feedback embeddings
    teacher_forcing_ratio: float = 0.5  # Probability of feeding ground truth frame token during training
    frame_ce_weight: float = 0.2        # Weight for frame-level Cross-Entropy loss regularizer
    ctc_weight: float = 1.0             # Weight for sequence CTC loss
    blank_penalty_weight: float = 2.0   # Penalty weight on blank over-dominance
    blank_threshold: float = 0.25       # Minimum expected non-blank probability margin
    blank_eval_penalty: float = 0.0     # Evaluation calibration penalty for blank
    ignore_index: int = -100


class PhonoHuBERTRecursiveForPreTraining(nn.Module):
    """Speech Transformer with Autoregressive Recurrent Frame-Memory Feedback Head."""

    def __init__(self, config: Optional[PhonoHuBERTRecursiveConfig] = None):
        super().__init__()
        self.config = config or PhonoHuBERTRecursiveConfig()

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

        # 4. Recurrent Autoregressive Feedback Head
        embed_dim = self.config.encoder_embed_dim
        tok_dim = self.config.token_embed_dim
        head_dim = embed_dim

        self.token_embedding = nn.Embedding(self.config.vocab_size, tok_dim, padding_idx=self.config.pad_token_id)
        self.recurrent_cell = nn.GRUCell(embed_dim + tok_dim, head_dim)
        self.head_norm = nn.LayerNorm(head_dim)
        self.head_proj = nn.Linear(head_dim, self.config.vocab_size)

        # 5. Loss Functions
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
        self.ctc_loss_fn = nn.CTCLoss(
            blank=self.config.blank_token_id,
            zero_infinity=True,
        )

    def _unroll_head(
        self,
        hidden_states: torch.Tensor,
        frame_targets: Optional[torch.Tensor] = None,
        blank_penalty: float = 0.0,
    ) -> torch.Tensor:
        """Unroll the recurrent cell across time frames with scheduled sampling."""
        B, T, D = hidden_states.shape
        device = hidden_states.device
        tok_dim = self.config.token_embed_dim
        head_dim = self.config.encoder_embed_dim

        # Initial recurrent state: zeros
        h_t = torch.zeros(B, head_dim, device=device, dtype=hidden_states.dtype)
        # Initial previous token: pad token
        prev_tok = torch.full((B,), self.config.pad_token_id, device=device, dtype=torch.long)

        all_logits = []
        tf_ratio = self.config.teacher_forcing_ratio if (self.training and frame_targets is not None) else 0.0

        for t in range(T):
            # Acoustic feature at frame t
            x_t = hidden_states[:, t, :]  # (B, D)

            # Previous token feedback embedding
            if tf_ratio > 0.0 and frame_targets is not None and t > 0:
                use_tf = (torch.rand(1).item() < tf_ratio)
                if use_tf and t - 1 < frame_targets.shape[1]:
                    target_prev = frame_targets[:, t - 1]
                    # Fallback to pad if target is ignore_index
                    valid_mask = target_prev != self.config.ignore_index
                    effective_prev = torch.where(
                        valid_mask,
                        target_prev,
                        torch.full_like(target_prev, self.config.pad_token_id),
                    )
                    e_prev = self.token_embedding(effective_prev)
                else:
                    e_prev = self.token_embedding(prev_tok)
            else:
                e_prev = self.token_embedding(prev_tok)

            # Combined input: [acoustic; feedback]
            cell_in = torch.cat([x_t, e_prev], dim=-1)
            h_t = self.recurrent_cell(cell_in, h_t)
            logits_t = self.head_proj(self.head_norm(h_t))  # (B, V)

            if blank_penalty > 0.0:
                logits_t_adj = logits_t.clone()
                logits_t_adj[:, self.config.blank_token_id] -= blank_penalty
                prev_tok = logits_t_adj.argmax(dim=-1)
            else:
                prev_tok = logits_t.argmax(dim=-1)

            all_logits.append(logits_t.unsqueeze(1))

        return torch.cat(all_logits, dim=1)  # (B, T, V)

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
        """Forward pass through PhonoHuBERTRecursive."""
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

        # 3. Recurrent Autoregressive Head Unroll
        logits = self._unroll_head(
            hidden_states=hidden_state,
            frame_targets=frame_targets,
            blank_penalty=0.0,
        )  # (B, T_frames, vocab_size)

        loss = None
        ce_loss_val = 0.0
        ctc_loss_val = 0.0
        accuracy = None

        # 4. Frame Cross-Entropy Loss
        if frame_targets is not None:
            target_T = frame_targets.shape[1]
            min_T = min(target_T, T_frames)
            l_slice = logits[:, :min_T, :].contiguous()
            ft_slice = frame_targets[:, :min_T].contiguous()

            ce_loss = self.ce_loss_fn(
                l_slice.view(-1, self.config.vocab_size),
                ft_slice.view(-1),
            )
            ce_loss_val = float(ce_loss.item())
            loss = self.config.frame_ce_weight * ce_loss

        # 5. Sequence CTC Loss
        if targets is not None:
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1).float()  # (T, B, V)
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
                ctc_loss = ctc_loss + blank_loss

            if loss is not None:
                loss = loss + self.config.ctc_weight * ctc_loss
            else:
                loss = ctc_loss

            # Accuracy on phoneme sequence
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
                            s_acc = max(0.0, 1.0 - (dist / max(1, len(ref_seq))))
                        except Exception:
                            s_acc = 1.0 if pred_seq == ref_seq else 0.0
                        matches += s_acc
                        valid_count += 1
                accuracy = torch.tensor((matches / max(1, valid_count)) * 100.0, device=audio.device)

        return {
            "loss": loss,
            "ce_loss": ce_loss_val,
            "ctc_loss": ctc_loss_val,
            "accuracy": accuracy,
            "logits": logits,
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
        """Greedy sequence decoding with recurrent temporal memory unroll."""
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
                logits = self._unroll_head(
                    hidden_states=encoder_res["last_hidden_state"],
                    frame_targets=None,
                    blank_penalty=penalty,
                )
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
                    if p not in (self.config.blank_token_id, self.config.pad_token_id, self.config.silence_token_id, self.config.noise_token_id):
                        decoded.append(p)
                    prev = p
            batch_sequences.append(decoded)
        return batch_sequences
