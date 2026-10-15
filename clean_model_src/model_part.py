"""
Abstract Base Class for Model Parts.
Provides device routing, quantization hooks, mobile export interfaces, and standard training/inference APIs.
"""

# Import Modules
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import torch
from torch import nn, Tensor

from .model_config import ModelConfig


class Model_Part(nn.Module, ABC):
    """
    Abstract Base Class for modular model parts (Model 1 A2P, Model 2 PE, Model 3 WD).
    Enforces clean OOP structure, per-part device routing, quantization, and mobile export.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        model_part: str,
    ) -> None:
        """
        Initialize base Model_Part.

        Args:
            model_config (ModelConfig): Unified configuration dataclass.
            model_part (str): Identifier of this model part ('A2P', 'PE', or 'WD').
        """
        # Parent initialization
        super().__init__()

        # Model config: ModelConfig, Role: shared hyperparameter tree
        self.model_config: ModelConfig = model_config

        # Model part identifier: str ('A2P', 'PE', 'WD')
        self.model_part: str = model_part.upper()

    def get_device(self) -> torch.device:
        """
        Get the assigned torch.device for this model part.

        Returns:
            torch.device: The designated device (e.g. cuda or cpu).
        """
        return self.model_config.get_device(part=self.model_part)

    def to_assigned_device(self) -> "Model_Part":
        """
        Moves all parameters of this model part to its configured device.

        Returns:
            Model_Part: self moved to target device.
        """
        target_dev: torch.device = self.get_device()
        return self.to(target_dev)

    @abstractmethod
    def forward_training(self, *args: Any, **kwargs: Any) -> Dict[str, Tensor]:
        """
        Clean training forward pass computing multi-task losses.

        Returns:
            Dict[str, Tensor]: Dictionary containing primary loss, auxiliary losses, and metrics.
        """
        pass

    @abstractmethod
    def forward_inference(self, *args: Any, **kwargs: Any) -> Any:
        """
        Clean, highly optimized forward pass for real-time inference.

        Returns:
            Any: Decoded predictions (phoneme tokens, word latents, or text characters).
        """
        pass

    def load_weights(
        self,
        checkpoint_path: str,
        state_dict_key: Optional[str] = None,
        strict: bool = False,
    ) -> bool:
        """
        Loads weights from a saved checkpoint into this model part.

        Args:
            checkpoint_path (str): File path to .pt checkpoint.
            state_dict_key (Optional[str]): Specific state_dict dictionary key inside checkpoint.
            strict (bool): Whether to enforce exact key matching.

        Returns:
            bool: True if loaded successfully, False if file not found.
        """
        if not os.path.isfile(checkpoint_path):
            return False

        device: torch.device = self.get_device()
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if state_dict_key and state_dict_key in ckpt:
            sd = ckpt[state_dict_key]
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt

        # Load weights into self
        missing, unexpected = self.load_state_dict(sd, strict=strict)
        return True

    def quantize(
        self,
        dtype: str = "qint8",
        layer_types: Optional[Tuple[type, ...]] = None,
    ) -> "Model_Part":
        """
        Granular per-part quantization hook.
        Quantizes this model part's linear / convolution layers to int8 or float16.

        Args:
            dtype (str): Target quantized dtype ('qint8', 'float16').
            layer_types (Optional[Tuple[type, ...]]): Specific layer classes to quantize (default: nn.Linear).

        Returns:
            Model_Part: Quantized module.
        """
        if dtype.lower() == "float16":
            return self.half()

        # Dynamic int8 quantization for CPU inference
        targets = layer_types or (nn.Linear,)
        try:
            quantized = torch.ao.quantization.quantize_dynamic(
                self,
                qconfig_spec=set(targets),
                dtype=torch.qint8,
            )
            return quantized
        except Exception:
            # Fallback if dynamic quantization not supported on current platform
            return self
