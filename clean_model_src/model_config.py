"""
Unified Clean Model Configuration.
Provides modular dataclasses for all model parts, device mappings, and dynamic class resolution.
"""

# Import Modules
from dataclasses import dataclass, field
import importlib
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar
import torch

T = TypeVar("T")


def get_class(path: str) -> Type[T]:
    """
    Get a class from a module path.

    Args:
        path (str): Module path and class name, separated by dots (e.g. 'clean_model_src.model1_a2p.Model_A2P').

    Returns:
        Type[T]: The resolved class object.
    """
    # Split by last dot to separate module from class name
    parts: List[str] = path.split(".")
    if len(parts) == 1:
        # Local module within package
        module_path: str = f"clean_model_src.{parts[0]}"
        class_name: str = parts[0]
    else:
        module_path = ".".join(parts[:-1])
        class_name = parts[-1]

    # Resolve module
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError:
        # Fallback to local import if called from within package
        mod = importlib.import_module(f".{module_path.split('.')[-1]}", package="clean_model_src")

    cls: Type[T] = getattr(mod, class_name)
    return cls


@dataclass
class ModelA2PConfig:
    """
    Configuration for Model Part 1: Audio to Phonemes (A2P).
    12-layer Conformer Backbone with DeepSeek MoE, Phoneme Engram Memory, and 7-layer CNN frontend.
    """
    # Audio sampling rate in Hertz
    sample_rate: int = 16000

    # CNN Feature Extractor parameters (downsamples audio by 320x -> 50 Hz)
    in_channels: int = 1
    conv_channels: int = 512
    cnn_weights_path: str = "data/pretrained_cnn_ls960.pt"

    # Conformer Encoder parameters
    embed_dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    ffn_dim: int = 2048
    conv_kernel_size: int = 31
    dropout: float = 0.1
    attention_dropout: float = 0.1
    use_rope: bool = True

    # DeepSeek Mixture-of-Experts (MoE) parameters
    use_deepseek_moe: bool = True
    num_routed_experts: int = 8
    top_k: int = 2
    num_shared_experts: int = 1
    load_balance_weight: float = 0.01
    router_jitter_std: float = 0.02

    # Intermediate Draft Supervision parameters
    draft_layer_idx: int = 3

    # Multi-Scale Phoneme Engram Memory parameters
    engram_layers: Tuple[int, ...] = (5, 9)
    engram_heads: int = 4
    engram_memory_dim: int = 128
    engram_table_size: int = 4096
    engram_ngram_orders: Tuple[int, ...] = (2, 3, 4)

    # Gradient Checkpointing (essential for training Ultra/Large tiers in <=12GB VRAM)
    gradient_checkpointing: bool = False

    # Phoneme Vocabulary & Tokens
    vocab_size: int = 112
    blank_id: int = 1
    space_id: int = 8
    pad_id: int = 0


@dataclass
class ModelPEConfig:
    """
    Configuration for Model Part 2: Phoneme Encoder (PE).
    FastText-style phoneme word encoder mapping discrete phoneme slices to continuous word latents z_word.
    """
    d_macro: int = 512
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    phoneme_vocab_size: int = 128
    max_seq_len: int = 32


@dataclass
class ModelWDConfig:
    """
    Configuration for Model Part 3: Word Decoder (WD).
    Overlapping Length Experts MoE Router with Band-Causal Attention and Autoregressive Character Head.
    """
    macro_dim: int = 512
    micro_dim: int = 512
    num_macro_layers: int = 4
    num_heads: int = 8
    band_window_words: int = 8
    num_overlapping_experts: int = 7
    char_vocab_size: int = 128
    max_word_len: int = 24
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = 0


@dataclass
class ModelConfig:
    """
    Master Unified Configuration coordinating all 3 model parts,
    device assignments, and dynamic instantiation paths.
    """

    # --- Dynamic Class Loading Paths ---
    class_model_a2p: str = "clean_model_src.model1_a2p.Model_A2P"
    class_model_pe: str = "clean_model_src.model2_pe.Model_PE"
    class_model_wd: str = "clean_model_src.model3_wd.Model_WD"

    # Aliases for flexible access
    @property
    def model_a2p_class(self) -> str:
        return self.class_model_a2p

    @property
    def model_pe_class(self) -> str:
        return self.class_model_pe

    @property
    def model_wd_class(self) -> str:
        return self.class_model_wd

    # --- Device Assignments per Model Part ---
    device_model_a2p: str = "CUDA"
    device_model_pe: str = "CPU"
    device_model_wd: str = "CPU"

    @property
    def device_a2p(self) -> str:
        return self.device_model_a2p

    @device_a2p.setter
    def device_a2p(self, val: str) -> None:
        self.device_model_a2p = val

    @property
    def device_pe(self) -> str:
        return self.device_model_pe

    @device_pe.setter
    def device_pe(self, val: str) -> None:
        self.device_model_pe = val

    @property
    def device_wd(self) -> str:
        return self.device_model_wd

    @device_wd.setter
    def device_wd(self, val: str) -> None:
        self.device_model_wd = val

    # --- Sub-Component Configurations ---
    a2p: ModelA2PConfig = field(default_factory=ModelA2PConfig)
    pe: ModelPEConfig = field(default_factory=ModelPEConfig)
    wd: ModelWDConfig = field(default_factory=ModelWDConfig)

    def get_device(self, part: str) -> torch.device:
        """
        Get the torch.device object for the requested model part.

        Args:
            part (str): The model part ('A2P', 'PE', or 'WD').

        Returns:
            torch.device: The resolved device (e.g. cuda:0 or cpu).
        """
        part_upper: str = part.upper()
        if part_upper in ("A2P", "MODEL1", "MODEL_1", "1"):
            target_str = self.device_model_a2p
        elif part_upper in ("PE", "MODEL2", "MODEL_2", "2"):
            target_str = self.device_model_pe
        elif part_upper in ("WD", "MODEL3", "MODEL_3", "3"):
            target_str = self.device_model_wd
        else:
            raise ValueError(f"Unknown model part: `{part}`. Expected 'A2P', 'PE', or 'WD'.")

        # Resolve CUDA availability fallback
        if target_str.upper() == "CUDA" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(target_str.lower())

    # --- Official Model Size Tier Presets (NaVox v1.0) ---

    @classmethod
    def tiny(cls, **kwargs: Any) -> "ModelConfig":
        """NaVox-Tiny (~15M params): Ultra-lightweight for microcontrollers & embedded devices."""
        cfg = cls(
            a2p=ModelA2PConfig(
                embed_dim=256,
                num_layers=6,
                num_heads=4,
                ffn_dim=1024,
                use_deepseek_moe=False,
                engram_layers=(2, 4),
                draft_layer_idx=1,
            ),
            pe=ModelPEConfig(d_macro=512, d_model=256, num_layers=2, nhead=4, dim_feedforward=512),
            wd=ModelWDConfig(macro_dim=512, micro_dim=256, num_macro_layers=3, num_heads=4, num_overlapping_experts=5),
            **kwargs,
        )
        return cfg

    @classmethod
    def small(cls, **kwargs: Any) -> "ModelConfig":
        """NaVox-Small (~45M params): Real-time on-device edge / Google Pixel 10 NPU."""
        cfg = cls(
            a2p=ModelA2PConfig(
                embed_dim=384,
                num_layers=8,
                num_heads=6,
                ffn_dim=1536,
                use_deepseek_moe=True,
                num_routed_experts=4,
                top_k=2,
                num_shared_experts=1,
                engram_layers=(3, 6),
                draft_layer_idx=2,
            ),
            pe=ModelPEConfig(d_macro=512, d_model=384, num_layers=3, nhead=6, dim_feedforward=768),
            wd=ModelWDConfig(macro_dim=512, micro_dim=384, num_macro_layers=4, num_heads=6, num_overlapping_experts=7),
            **kwargs,
        )
        return cfg

    @classmethod
    def base(cls, **kwargs: Any) -> "ModelConfig":
        """NaVox-Base (~120M params / 45M active): Balanced desktop / high-accuracy streaming."""
        cfg = cls(
            a2p=ModelA2PConfig(
                embed_dim=512,
                num_layers=12,
                num_heads=8,
                ffn_dim=2048,
                use_deepseek_moe=True,
                num_routed_experts=8,
                top_k=2,
                num_shared_experts=1,
                engram_layers=(5, 9),
                draft_layer_idx=3,
            ),
            pe=ModelPEConfig(d_macro=512, d_model=512, num_layers=4, nhead=8, dim_feedforward=1024),
            wd=ModelWDConfig(macro_dim=512, micro_dim=512, num_macro_layers=4, num_heads=8, num_overlapping_experts=7),
            **kwargs,
        )
        return cfg

    @classmethod
    def large(cls, **kwargs: Any) -> "ModelConfig":
        """NaVox-Large (~510M params / 175M active): High-capacity multilingual standard."""
        cfg = cls(
            a2p=ModelA2PConfig(
                embed_dim=512,
                num_layers=16,
                num_heads=8,
                ffn_dim=2048,
                use_deepseek_moe=True,
                num_routed_experts=12,
                top_k=2,
                num_shared_experts=1,
                engram_layers=(7, 13),
                draft_layer_idx=4,
                gradient_checkpointing=True,
            ),
            pe=ModelPEConfig(d_macro=512, d_model=512, num_layers=6, nhead=8, dim_feedforward=2048),
            wd=ModelWDConfig(macro_dim=512, micro_dim=512, num_macro_layers=6, num_heads=8, num_overlapping_experts=7),
            **kwargs,
        )
        return cfg

    @classmethod
    def ultra(cls, **kwargs: Any) -> "ModelConfig":
        """NaVox-Ultra (~995M params / 256M active): Flagship SOTA capacity on 12GB VRAM."""
        cfg = cls(
            a2p=ModelA2PConfig(
                embed_dim=512,
                num_layers=24,
                num_heads=8,
                ffn_dim=2048,
                use_deepseek_moe=True,
                num_routed_experts=16,
                top_k=2,
                num_shared_experts=1,
                engram_layers=(10, 19),
                draft_layer_idx=5,
                gradient_checkpointing=True,
            ),
            pe=ModelPEConfig(d_macro=512, d_model=512, num_layers=8, nhead=8, dim_feedforward=2048),
            wd=ModelWDConfig(macro_dim=512, micro_dim=512, num_macro_layers=8, num_heads=8, num_overlapping_experts=7),
            **kwargs,
        )
        return cfg
