"""HuBERT Self-Supervised Pre-training Architecture: Masked Acoustic Unit Prediction."""

from typing import Dict, Optional, Tuple, Union
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import HuBERTConfig
from src.models.cnn_encoder import HuBERTFeatureEncoder
from src.models.transformer import HuBERTEncoder


class HuBERTForPreTraining(nn.Module):
    """HuBERT Acoustic Model for Self-Supervised Pre-training via Masked Acoustic Unit Prediction.
    
    Architecture:
      Audio Waveform (16kHz)
         │
      [HuBERTFeatureEncoder] (7-layer 1D CNN with downsampling 320x)
         │
      [FeatureProjection] (Linear projection to embed_dim + LayerNorm)
         │
      [Span Masking] (Random spans replaced with learnable mask embedding)
         │
      [HuBERTEncoder] (Convolutional Pos-Embed + Transformer Layers)
         │
      [Cluster Head] (Linear projection to K acoustic cluster units)
         │
      Cross-Entropy Loss on Masked Frames
    """

    def __init__(self, config: Optional[HuBERTConfig] = None, num_clusters: int = 100):
        super().__init__()
        self.config = config or HuBERTConfig()
        self.num_clusters = num_clusters

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

        # 5. Cluster Unit Prediction Head (k-means pseudo-label classifier)
        self.cluster_head = nn.Linear(self.config.encoder_embed_dim, self.num_clusters)

    def extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract continuous projected frame features from raw audio."""
        cnn_feats = self.feature_extractor(audio)
        return self.feature_projection(cnn_feats)

    def generate_span_mask(
        self,
        batch_size: int,
        seq_len: int,
        mask_prob: float = 0.08,
        mask_length: int = 10,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generate contiguous boolean span masks (True where masked)."""
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        for b in range(batch_size):
            starts = torch.rand(seq_len, device=device) < mask_prob
            nonzero_idx = torch.nonzero(starts).squeeze(-1)
            for idx in nonzero_idx:
                end = min(seq_len, int(idx.item()) + mask_length)
                mask[b, idx:end] = True
            # Fallback if no frames got masked
            if not mask[b].any():
                start = torch.randint(0, max(1, seq_len - mask_length), (1,)).item()
                mask[b, start:start + mask_length] = True
        return mask

    def forward(
        self,
        audio: torch.Tensor,
        target_clusters: Optional[torch.Tensor] = None,
        mask_prob: float = 0.08,
        mask_length: int = 10,
        custom_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, any]:
        """Forward pass for masked acoustic unit pre-training.
        
        Args:
            audio: Waveforms of shape (B, T_audio).
            target_clusters: Discrete cluster targets of shape (B, T_frames).
            mask_prob: Probability of span start.
            mask_length: Length of consecutive masked frames.
            custom_mask: Optional explicit boolean mask tensor (B, T_frames).
            
        Returns:
            Dictionary with:
              - 'loss': Cross-entropy loss over masked frames
              - 'accuracy': Cluster prediction accuracy on masked frames
              - 'logits': (B, T_frames, num_clusters)
              - 'mask': (B, T_frames) boolean mask
              - 'target_clusters': aligned target clusters
        """
        feats = self.extract_features(audio)  # (B, T_frames, embed_dim)
        B, T_frames, D = feats.shape

        # Align lengths if target_clusters provided
        if target_clusters is not None:
            min_len = min(T_frames, target_clusters.shape[1])
            feats = feats[:, :min_len, :]
            target_clusters = target_clusters[:, :min_len]
            T_frames = min_len

        # Generate or use span mask
        if custom_mask is not None:
            mask = custom_mask[:, :T_frames]
        else:
            mask = self.generate_span_mask(
                batch_size=B,
                seq_len=T_frames,
                mask_prob=mask_prob,
                mask_length=mask_length,
                device=audio.device,
            )

        # Replace masked positions with mask_embedding
        masked_feats = feats.clone()
        masked_feats[mask] = self.mask_embedding.to(masked_feats.dtype)

        # Transformer encoding
        encoder_out = self.encoder(masked_feats)["last_hidden_state"]

        # Predict cluster logits
        logits = self.cluster_head(encoder_out)  # (B, T_frames, num_clusters)

        loss = None
        acc = None
        if target_clusters is not None:
            # Mask out invalid padding targets (-100)
            valid_targets = target_clusters != -100
            active_mask = mask & valid_targets

            if active_mask.any():
                masked_logits = logits[active_mask]
                masked_targets = target_clusters[active_mask]

                loss = F.cross_entropy(masked_logits, masked_targets)
                preds = torch.argmax(masked_logits, dim=-1)
                acc = (preds == masked_targets).float().mean()
            else:
                loss = torch.tensor(0.0, device=audio.device, requires_grad=True)
                acc = torch.tensor(0.0, device=audio.device)

        return {
            "loss": loss,
            "accuracy": acc,
            "logits": logits,
            "mask": mask,
            "target_clusters": target_clusters,
            "encoder_states": encoder_out,
        }

    def save_pretrained_backbone(self, path: str):
        """Save pre-trained backbone weights for downstream ASR fine-tuning."""
        torch.save(
            {
                "config": self.config,
                "feature_extractor": self.feature_extractor.state_dict(),
                "feature_projection": self.feature_projection.state_dict(),
                "encoder": self.encoder.state_dict(),
                "num_clusters": self.num_clusters,
            },
            path,
        )
        print(f"[HuBERT Pretrain] Pre-trained backbone successfully saved to {path}")

    def load_pretrained_backbone(self, path_or_dict):
        """Load pre-trained backbone weights from path or checkpoint dictionary."""
        if isinstance(path_or_dict, (str, Path)):
            checkpoint = torch.load(path_or_dict, map_location="cpu")
        else:
            checkpoint = path_or_dict

        if "feature_extractor" in checkpoint:
            self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
        if "feature_projection" in checkpoint:
            self.feature_projection.load_state_dict(checkpoint["feature_projection"])
        if "encoder" in checkpoint:
            self.encoder.load_state_dict(checkpoint["encoder"])
        print("[HuBERT Pretrain] Pre-trained backbone successfully loaded!")

    def transfer_to_ctc_model(self, ctc_model: nn.Module):
        """Transfer pre-trained feature extractor and Transformer encoder to a HuBERTForCTC instance."""
        ctc_model.feature_extractor.load_state_dict(self.feature_extractor.state_dict())
        ctc_model.feature_projection.load_state_dict(self.feature_projection.state_dict())
        ctc_model.encoder.load_state_dict(self.encoder.state_dict())
        print("[HuBERT Pretrain] Successfully transferred pre-trained weights to HuBERTForCTC model!")
