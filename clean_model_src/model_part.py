"""
Abstract class for model parts.
"""

# Import Modules
from torch import nn, Tensor

from .model_config import ModelConfig


class Model_Part(nn.Module):
    """
    Abstrat audio to phoneme class.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        model_part: str
    ) -> None:
        """
        Init function.

        Args:
            model_config (ModelConfig): Config for the model.
        """

        # Parent init
        super().__init__()

        # Model config
        self.model_config: ModelConfig = model_config

        # Model part
        self.model_part: str = model_part

    def get_device(self) -> str:
        """
        Get the device to load a module on.

        Return:
            str: the device to load a module on.
        """

        return self.model_config.get_device(part=self.model_part)

