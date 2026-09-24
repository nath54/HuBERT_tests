"""HuBERT Model Configuration."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class HuBERTConfig:
    """Configuration class for HuBERT ASR model."""

    # Audio input properties
    sample_rate: int = 16000
    in_channels: int = 1

    # Feature Extractor (Temporal CNN)
    # (out_channels, kernel_size, stride)
    # Default 7 layers downsample by 5 * 2 * 2 * 2 * 2 * 2 * 2 = 320x (20ms frames at 16kHz)
    conv_layers: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ]
    )
    conv_bias: bool = False
    conv_feature_dim: int = 512

    # Transformer Encoder
    encoder_embed_dim: int = 512
    encoder_layers: int = 6           # Default 6 for agile training/testing (or 12 for base)
    encoder_heads: int = 8
    encoder_ffn_dim: int = 2048
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_fn: str = "gelu"
    layer_norm_first: bool = True     # Pre-LN architecture

    # Positional Embedding (Convolutional)
    pos_conv_kernel: int = 128
    pos_conv_groups: int = 16

    # ASR / CTC Head
    vocab_size: int = 32              # Character tokens (blank, space, a-z, etc.)
    blank_index: int = 0
    pad_index: int = 0

    def compute_output_length(self, input_length: int) -> int:
        """Compute the sequence length after CNN downsampling."""
        length = input_length
        for _, kernel, stride in self.conv_layers:
            length = (length - kernel) // stride + 1
        return max(0, length)
