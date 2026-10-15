"""
Main model end to end class.
Modular and clean.
"""

# Import Modules
from torch import nn, Tensor

from .model_config import ModelConfig, get_class
from .model_a2p import Model_A2P



class Model(nn.Module):
    """
    TODO: clean and short docstring
    """

    def __init__(
        self,
        model_config: ModelConfig
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

        # Part 1 of the model: phoneme encoder
        self.model_a2p: Model_A2P = get_class(self.model_config.model_a2p_class)

        # Part 2 of the model: phoneme encoder
        self.model_pe: Model_A2P = get_class(self.model_config.model_a2p_class)

        # Part 3 of the model: phoneme encoder
        self.model_wd: Model_A2P = get_class(self.model_config.model_a2p_class)
