"""Temporal Convolutional Feature Extractor for HuBERT."""

import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransposeLayerNorm(nn.Module):
    """LayerNorm applied across channels for (B, C, T) tensors."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, T, C) -> norm -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)


class HuBERTConvLayer(nn.Module):
    """Single 1D Convolution block with normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.norm = TransposeLayerNorm(out_channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        x = self.conv(x)
        x = self.dropout(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class HuBERTFeatureEncoder(nn.Module):
    """7-layer Temporal Convolutional Feature Extractor.
    
    Downsamples 16kHz raw audio waveforms (320x total downsampling factor)
    into 50Hz frame representations (each frame represents ~20ms of audio).
    """

    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        in_channels: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        curr_in = in_channels
        for i, (out_dim, kernel_size, stride) in enumerate(conv_layers):
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
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Audio tensor of shape (B, T_audio) or (B, 1, T_audio).
            
        Returns:
            Features tensor of shape (B, T_frames, C_feat).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, T)

        for layer in self.layers:
            x = layer(x)

        # Transpose to (B, T_frames, C_feat) for Transformer processing
        return x.transpose(1, 2)
