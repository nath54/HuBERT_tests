"""
Modular Convolution Modules for Speech Acoustic Processing.
Includes:
- TransposeLayerNorm: Channel-wise LayerNorm for (B, C, T) tensors
- HuBERTConvLayer: Standard 1D convolution block with TransposeLayerNorm and GELU
- AudioCNNFeatureExtractor: 7-layer 1D CNN downsampling 16kHz audio by factor 320
- ConformerConvModule: Depthwise separable convolution block with GLU and BatchNorm
- MultiScaleDilatedConformerConvModule: Multi-scale dilated depthwise separable convolution
"""

from typing import List, Optional, Tuple
import os
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .activations import GatedLinearUnit, SwishActivation


class TransposeLayerNorm(nn.Module):
    """
    LayerNorm applied across channels for (B, C, T) tensors.
    Transposes to (B, T, C), applies LayerNorm, and transposes back to (B, C, T).
    """

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm: nn.LayerNorm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        x_t = x.transpose(1, 2)
        normed = self.norm(x_t)
        return normed.transpose(1, 2)


class HuBERTConvLayer(nn.Module):
    """
    Single 1D Convolution block with normalization and activation.
    State dict keys: 'conv.weight', 'norm.norm.weight', 'norm.norm.bias'.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv: nn.Conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.norm: TransposeLayerNorm = TransposeLayerNorm(out_channels)
        self.activation: nn.GELU = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.dropout(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


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
        super().__init__()

        self.conv_spec: List[Tuple[int, int, int]] = [
            (conv_channels, 10, 5),
            (conv_channels, 3, 2),
            (conv_channels, 3, 2),
            (conv_channels, 3, 2),
            (conv_channels, 3, 2),
            (conv_channels, 2, 2),
            (conv_channels, 2, 2),
        ]

        layers: List[HuBERTConvLayer] = []
        curr_in = in_channels
        for out_dim, kernel_size, stride in self.conv_spec:
            layers.append(
                HuBERTConvLayer(
                    in_channels=curr_in,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=False,
                    dropout=dropout,
                )
            )
            curr_in = out_dim

        self.layers: nn.ModuleList = nn.ModuleList(layers)

    def forward(self, audio: Tensor) -> Tensor:
        if audio.ndim == 2:
            x: Tensor = audio.unsqueeze(1)
        else:
            x = audio

        for layer in self.layers:
            x = layer(x)

        return x.transpose(1, 2)

    def get_output_lengths(self, input_lengths: Tensor) -> Tensor:
        lengths: Tensor = input_lengths
        for _, k_size, stride in self.conv_spec:
            lengths = torch.div(lengths - k_size, stride, rounding_mode="floor") + 1
        return lengths.clamp(min=1)

    def load_pretrained_weights(self, cnn_weights_path: str, device: torch.device) -> bool:
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
    """

    def __init__(
        self,
        embed_dim: int = 512,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, f"kernel_size must be odd, got {kernel_size}"

        self.embed_dim: int = embed_dim
        self.kernel_size: int = kernel_size
        padding: int = (kernel_size - 1) // 2

        self.layer_norm: nn.LayerNorm = nn.LayerNorm(embed_dim)
        self.pointwise_conv1: nn.Conv1d = nn.Conv1d(embed_dim, 2 * embed_dim, kernel_size=1)
        self.glu: GatedLinearUnit = GatedLinearUnit(dim=1)
        self.depthwise_conv: nn.Conv1d = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=embed_dim,
            bias=False,
        )
        self.batch_norm: nn.BatchNorm1d = nn.BatchNorm1d(embed_dim)
        self.activation: SwishActivation = SwishActivation()
        self.pointwise_conv2: nn.Conv1d = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.pointwise_conv2.weight)
        if self.pointwise_conv2.bias is not None:
            nn.init.zeros_(self.pointwise_conv2.bias)

    def forward(self, x: Tensor) -> Tensor:
        h: Tensor = self.layer_norm(x).transpose(1, 2)
        h = self.glu(self.pointwise_conv1(h))
        h = self.depthwise_conv(h)
        h = self.batch_norm(h)
        h = self.activation(h)
        h = self.pointwise_conv2(h)
        h = self.dropout(h)
        return h.transpose(1, 2)


class MultiScaleDilatedConformerConvModule(nn.Module):
    """
    Multi-Scale Dilated Depthwise Separable Convolution.
    Expands temporal receptive field from 620ms to 2,420ms:
    - Base Depthwise Conv: kernel=31, dilation=1 -> 620ms
    - Branch 2: kernel=31, dilation=2 -> 1,220ms
    - Branch 3: kernel=31, dilation=4 -> 2,420ms
    """

    def __init__(
        self,
        embed_dim: int = 512,
        kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2

        self.layer_norm: nn.LayerNorm = nn.LayerNorm(embed_dim)
        self.pointwise_conv1: nn.Conv1d = nn.Conv1d(embed_dim, 2 * embed_dim, kernel_size=1)
        self.glu: GatedLinearUnit = GatedLinearUnit(dim=1)

        self.depthwise_conv: nn.Conv1d = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=embed_dim,
            bias=False,
        )
        self.conv_d2: nn.Conv1d = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1),
            dilation=2,
            groups=embed_dim,
            bias=False,
        )
        self.conv_d4: nn.Conv1d = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) * 2,
            dilation=4,
            groups=embed_dim,
            bias=False,
        )
        self.dilated_proj: nn.Conv1d = nn.Conv1d(2 * embed_dim, embed_dim, kernel_size=1)
        nn.init.zeros_(self.dilated_proj.weight)
        if self.dilated_proj.bias is not None:
            nn.init.zeros_(self.dilated_proj.bias)

        self.batch_norm: nn.BatchNorm1d = nn.BatchNorm1d(embed_dim)
        self.activation: SwishActivation = SwishActivation()
        self.pointwise_conv2: nn.Conv1d = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.layer_norm(x).transpose(1, 2)
        h = self.glu(self.pointwise_conv1(h))

        h_d2 = self.conv_d2(h)
        h_d4 = self.conv_d4(h)
        h_dilated = self.dilated_proj(torch.cat([h_d2, h_d4], dim=1))
        h = self.depthwise_conv(h) + h_dilated

        h = self.batch_norm(h)
        h = self.activation(h)
        h = self.pointwise_conv2(h)
        h = self.dropout(h)
        return h.transpose(1, 2)
