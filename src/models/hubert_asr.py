"""Complete HuBERT architecture adapted for Automatic Speech Recognition (ASR) with CTC."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import HuBERTConfig
from src.models.cnn_encoder import HuBERTFeatureEncoder
from src.models.transformer import HuBERTEncoder


class HuBERTForCTC(nn.Module):
    """HuBERT Acoustic Model with a CTC projection head for ASR.
    
    Architecture:
      Audio Waveform (16kHz)
         │
      [HuBERTFeatureEncoder] (7-layer 1D CNN with GELU & LayerNorm, downsampling 320x)
         │
      [FeatureProjection] (Linear projection to embed_dim + LayerNorm)
         │
      [HuBERTEncoder] (Convolutional Pos-Embed + Stack of Transformer Layers)
         │
      [CTCHead] (Linear projection to character vocabulary logits)
         │
      CTC Loss / Greedy CTC Decoding
    """

    def __init__(self, config: Optional[HuBERTConfig] = None):
        super().__init__()
        self.config = config or HuBERTConfig()

        # 1. Temporal Feature Extractor
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

        # 3. Transformer Encoder
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

        # 4. CTC Head
        self.ctc_head = nn.Linear(self.config.encoder_embed_dim, self.config.vocab_size)

    def extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract CNN representations from raw audio.
        
        Args:
            audio: Tensor of shape (B, T_audio) or (B, 1, T_audio).
        Returns:
            Projected features of shape (B, T_frames, encoder_embed_dim).
        """
        cnn_feats = self.feature_extractor(audio)  # (B, T_frames, conv_feature_dim)
        projected = self.feature_projection(cnn_feats)  # (B, T_frames, encoder_embed_dim)
        return projected

    def compute_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Calculate downsampled sequence lengths through CNN layers."""
        lengths = input_lengths.clone()
        for _, kernel, stride in self.config.conv_layers:
            lengths = torch.div(lengths - kernel, stride, rounding_mode="floor") + 1
        return torch.clamp(lengths, min=0)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
    ) -> Dict[str, any]:
        """Forward pass for training and inference.
        
        Args:
            audio: Audio waveforms of shape (B, T_audio).
            audio_lengths: Actual lengths of unpadded audio (B,).
            targets: CTC target token IDs of shape (B, max_target_len).
            target_lengths: Actual target lengths (B,).
            output_hidden_states: Return all intermediate Transformer layer outputs.
            output_attentions: Return attention weights.
            
        Returns:
            Dictionary with:
              - 'logits': (B, T_frames, vocab_size)
              - 'log_probs': (T_frames, B, vocab_size) formatted for CTCLoss
              - 'loss': CTC Loss scalar (if targets provided)
              - 'output_lengths': Output sequence lengths (B,)
              - 'hidden_states': List of layer outputs
              - 'neuron_activations': List of FFN activations for each layer
        """
        # Feature extraction
        feats = self.extract_features(audio)  # (B, T_frames, embed_dim)
        B, T_frames, _ = feats.shape

        # Masking for padded frames if audio_lengths provided
        padding_mask = None
        if audio_lengths is not None:
            out_lengths = self.compute_output_lengths(audio_lengths)
            # Create boolean mask: True where padded
            idx = torch.arange(T_frames, device=audio.device).unsqueeze(0).expand(B, -1)
            padding_mask = idx >= out_lengths.unsqueeze(1)
        else:
            out_lengths = torch.full((B,), T_frames, dtype=torch.long, device=audio.device)

        # Transformer encoding
        encoder_outputs = self.encoder(
            feats,
            key_padding_mask=padding_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        last_hidden = encoder_outputs["last_hidden_state"]  # (B, T_frames, embed_dim)
        logits = self.ctc_head(last_hidden)  # (B, T_frames, vocab_size)

        # Log-softmax for CTC: shape (T_frames, B, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)

        loss = None
        if targets is not None and target_lengths is not None:
            loss = F.ctc_loss(
                log_probs,
                targets,
                out_lengths,
                target_lengths,
                blank=self.config.blank_index,
                reduction="mean",
                zero_infinity=True,
            )

        return {
            "logits": logits,
            "log_probs": log_probs,
            "loss": loss,
            "output_lengths": out_lengths,
            "hidden_states": encoder_outputs["hidden_states"],
            "attentions": encoder_outputs["attentions"],
            "neuron_activations": encoder_outputs["neuron_activations"],
        }

    def decode_greedy(
        self,
        logits: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> List[List[int]]:
        """Greedy CTC Decoding.
        
        Args:
            logits: (B, T_frames, vocab_size)
            lengths: (B,)
            
        Returns:
            List of decoded token ID sequences (blank and repeated tokens removed).
        """
        best_tokens = logits.argmax(dim=-1)  # (B, T_frames)
        decoded_batch = []

        for b in range(logits.size(0)):
            seq_len = lengths[b].item() if lengths is not None else logits.size(1)
            raw_seq = best_tokens[b, :seq_len].tolist()

            collapsed = []
            prev_token = None
            for token in raw_seq:
                if token != prev_token:
                    if token != self.config.blank_index:
                        collapsed.append(token)
                    prev_token = token

            decoded_batch.append(collapsed)

        return decoded_batch
