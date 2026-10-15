"""
Model Audio to phonemes part.
"""

# Import Modules
from torch import nn, Tensor

from .model_config import ModelConfig
from .model_part import Model_Part


class Model_A2P(Model_Part):
    """
    Abstrat audio to phoneme class.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        """
        Init function.

        Args:
            model_config (ModelConfig): Config for the model.
        """

        # Init Parent class
        super().__init__(model_config=model_config, model_part="A2P")

