"""
Modular Convolution Modules for Speech Acoustic Processing.
Includes:
- AudioCNNFeatureExtractor: 7-layer 1D CNN downsampling 16kHz audio by factor 320
- ConformerConvModule: Depthwise separable convolution block with GLU and BatchNorm
"""

# Import Modules
from typing import List, Optional, Tuple
import os
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .activations import GatedLinearUnit, SwishActivation


class AudioCNNFeatureExtractor(nn.Module):
    """
    7-layer 1D Temporal CNN Acoustic Feature Extractor (Wav2Vec2 / HuBERT standard).
    Downsamples continuous 16,000 Hz raw speech waveforms into 50 Hz acoustic frame features
    (20 ms frame rate) via a cumulative temporal stride of 320:
    Stride decomposition: 5 * 2 * 2 * 2 * 2 * 2 * 2 = 320.
    Receptive field: 400 audio samples (25 ms).
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: int = 512,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialize AudioCNNFeatureExtractor.

        Args:
            in_channels (int): Input audio channels (1 for mono waveform).
            conv_channels (int): Feature channel dimension across all 7 layers (512).
            dropout (float): Dropout probability between convolution blocks.
        """
        # Parent initialization
        super().__init__()

        # Conv architecture specifications: (out_channels, kernel_size, stride)
        # Total stride product = 5 * 2 * 2 * 2 * 2 * 2 * 2 = 320
        self.conv_spec: List[Tuple[int, int, int]] = [
            (conv_channels, 10, 5),  # Layer 0: Stride 5, receptive field 10 samples
            (conv_channels, 3, 2),   # Layer 1: Stride 2
            (conv_channels, 3, 2),   # Layer 2: Stride 2
            (conv_channels, 3, 2),   # Layer 3: Stride 2
            (conv_channels, 3, 2),   # Layer 4: Stride 2
            (conv_channels, 2, 2),   # Layer 5: Stride 2
            (conv_channels, 2, 2),   # Layer 6: Stride 2
        ]

        # Sequential stack of 7 convolution blocks
        self.layers: nn.ModuleList = nn.ModuleList()
        current_in_channels: int = in_channels

        for i, (out_ch, k_size, stride) in enumerate(self.conv_spec):
            # Conv1d layer: [B, C_in, T_audio] -> [B, C_out, T_out]
            conv_layer = nn.Conv1d(
                in_channels=current_in_channels,
                out_channels=out_ch,
                kernel_size=k_size,
                stride=stride,
                bias=False,
            )

            # LayerNorm over channels: normalized along channel dim
            norm_layer = nn.GroupNorm(num_groups=1, num_channels=out_ch)

            # GELU non-linearity
            act_layer = nn.GELU()

            # Dropout
            drop_layer = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

            # Pack into block
            block = nn.Sequential(conv_layer, norm_layer, act_layer, drop_layer)
            self.layers.append(block)
            current_in_channels = out_ch

    def forward(self, audio: Tensor) -> Tensor:
        """
        Extract acoustic feature frames from raw audio waveforms.

        Args:
            audio (Tensor): Raw 16kHz speech waveform.
                            Shape: [B, T_samples] or [B, 1, T_samples].
                            Type: torch.float32.
                            Role: Continuous physical speech audio stream.

        Returns:
            Tensor: Extracted acoustic frame features.
                    Shape: [B, T_frames, 512], where T_frames = (T_samples - 400) // 320 + 1.
                    Type: torch.float32.
                    Role: 50 Hz discrete frame-level acoustic representations.
        """
        # Ensure 3D shape [B, 1, T_samples]
        if audio.ndim == 2:
            # Dimension: [B, 1, T_samples], Type: torch.float32
            x: Tensor = audio.unsqueeze(1)
        else:
            x = audio

        # Pass through 7 temporal 1D convolution layers
        for layer in self.layers:
            # Dimension: [B, 512, T_downsampled], Type: torch.float32
            x = layer(x)

        # Transpose to sequence format: [B, 512, T_frames] -> [B, T_frames, 512]
        out: Tensor = x.transpose(1, 2)
        return out

    def get_output_lengths(self, input_lengths: Tensor) -> Tensor:
        """
        Computes exact temporal output lengths following 7 convolution strides.

        Args:
            input_lengths (Tensor): Raw audio sample lengths [B].

        Returns:
            Tensor: Output frame lengths [B].
        """
        lengths: Tensor = input_lengths
        for _, k_size, stride in self.conv_spec:
            lengths = torch.div(lengths - k_size, stride, rounding_mode="floor") + 1
        return lengths.clamp(min=1)

    def load_pretrained_weights(self, cnn_weights_path: str, device: torch.device) -> bool:
        """
        Loads pretrained 7-layer acoustic filter weights (e.g. from data/pretrained_cnn_ls960.pt).

        Args:
            cnn_weights_path (str): Path to checkpoint file.
            device (torch.device): Device to map tensors onto.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        if not os.path.isfile(cnn_weights_path):
            return False

        ckpt = torch.load(cnn_weights_path, map_location=device, weights_only=False)
        sd = ckpt.get("feature_extractor_state_dict", ckpt.get("state_dict", ckpt))
        self.load_state_dict(sd, strict=False)
        return True


class ConformerConvModule(nn.Module):
    """
    Conformer Depthwise Separable Convolution Module (Gulati et al., 2020).
    Captures fine-grained local co-articulation transitions and formant shifts.

    Architecture pipeline:
      x [B, T, D]
        │
        ▼ (Pre-LayerNorm)
      LayerNorm(D)
        │
        ▼ (Pointwise Conv 1)
      Conv1d(D -> 2*D, kernel=1)
        │
        ▼ (GLU Gating)
      GatedLinearUnit(dim=1) -> [B, D, T]
        │
        ▼ (Depthwise Conv)
      Conv1d(D -> D, kernel=kernel_size, groups=D, padding=symmetric)
        │
        ▼
      BatchNorm1d(D)
        │
        ▼
      SwishActivation()
        │
        ▼ (Pointwise Conv 2)
      Conv1d(D -> D, kernel=1)  (Zero-initialized for seamless warm-start)
        │
        ▼
      Dropout
        │
        ▼
      out [B, T, D]
    """

    def __init__(
        self,
        embed_dim: int = 512,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize ConformerConvModule.

        Args:
            embed_dim (int): Channel / hidden dimension D (e.g. 512).
            kernel_size (int): Receptive field kernel size (must be odd, default 31 = ~620ms).
            dropout (float): Dropout probability.
        """
        # Parent initialization
        super().__init__()
        assert kernel_size % 2 == 1, f"kernel_size must be odd, got {kernel_size}"

        self.embed_dim: int = embed_dim
        self.kernel_size: int = kernel_size
        padding: int = (kernel_size - 1) // 2

        # 1. Pre-LayerNorm: [B, T, D]
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(embed_dim)

        # 2. Pointwise Conv 1: projects channel dimension D -> 2*D
        # Shape: [B, D, T] -> [B, 2*D, T]
        self.pointwise_conv1: nn.Conv1d = nn.Conv1d(embed_dim, 2 * embed_dim, kernel_size=1)

        # 3. Gated Linear Unit: cuts channels back to D: [B, 2*D, T] -> [B, D, T]
        self.glu: GatedLinearUnit = GatedLinearUnit(dim=1)

        # 4. 1D Depthwise Convolution: per-channel spatial convolution
        # Shape: [B, D, T] -> [B, D, T]
        self.depthwise_conv: nn.Conv1d = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=embed_dim,
            bias=False,
        )

        # 5. Batch Normalization across temporal frames
        self.batch_norm: nn.BatchNorm1d = nn.BatchNorm1d(embed_dim)

        # 6. Swish Non-Linearity
        self.activation: SwishActivation = SwishActivation()

        # 7. Pointwise Conv 2: projects channel dimension D -> D
        self.pointwise_conv2: nn.Conv1d = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)

        # 8. Dropout
        self.dropout: nn.Dropout = nn.Dropout(dropout)

        # Zero-initialize the final pointwise projection for seamless warm-start
        nn.init.zeros_(self.pointwise_conv2.weight)
        if self.pointwise_conv2.bias is not None:
            nn.init.zeros_(self.pointwise_conv2.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through Conformer Depthwise Convolution block.

        Args:
            x (Tensor): Input hidden states.
                        Shape: [B, T, D].
                        Type: torch.float32.
                        Role: Frame representations entering convolutional mixing.

        Returns:
            Tensor: Convolved output residual update.
                    Shape: [B, T, D].
                    Type: torch.float32.
                    Role: Local articulatory features to add to residual stream.
        """
        # 1. Pre-LN: [B, T, D]
        h: Tensor = self.layer_norm(x)

        # 2. Transpose for Conv1d: [B, T, D] -> [B, D, T]
        h = h.transpose(1, 2)

        # 3. Pointwise Conv 1: [B, D, T] -> [B, 2*D, T]
        h = self.pointwise_conv1(h)

        # 4. GLU: [B, 2*D, T] -> [B, D, T]
        h = self.glu(h)

        # 5. Depthwise Conv: [B, D, T] -> [B, D, T]
        h = self.depthwise_conv(h)

        # 6. BatchNorm: [B, D, T] -> [B, D, T]
        h = self.batch_norm(h)

        # 7. Swish: [B, D, T] -> [B, D, T]
        h = self.activation(h)

        # 8. Pointwise Conv 2: [B, D, T] -> [B, D, T]
        h = self.pointwise_conv2(h)

        # 9. Dropout: [B, D, T] -> [B, D, T]
        h = self.dropout(h)

        # 10. Transpose back: [B, D, T] -> [B, T, D]
        out: Tensor = h.transpose(1, 2)
        return out
