"""
Unified clean model config here.
"""

# Import Modules
from dataclasses import dataclass

import importlib


T = TypeVar("T")


def get_class(path: str) -> type[T]:
    """
    Get a class from a module

    Args:
        path (str) : Module path and class name, separated by the last dot.

    Returns:
        type[T] : The class from the module
    """

    cls: type[T] = getattr(
        importlib.import_module(".".join(path.split(".")[:-1])),
        path.split(".")[-1],
    )

    return cls


@dataclass
class ModelConfig:

    # --- Dynamic Class Loading ---

    # Class to load for the model audio to phonemes (Model part 1)
    class_model_a2p: str = "model1_a2p.Model_A2P"

    # Class to load for the model phoneme encoder (Model part 2)
    class_model_pe: str = "model2_pe.Model_PE"

    # Class to load for the model word decoder (Model part 3)
    class_model_wd: str = "model3_wd.Model_WD"

    # Device for the model audio to phonemes (Model part 1)
    device_model_a2p: str = "CUDA"

    # Device for the model phoneme encoder (Model part 2)
    device_model_pe: str = "CPU"

    # Device for the model word decoder (Model part 3)
    device_model_wd: str = "CPU"

    # TODO
    pass


    def get_device(self, part: str) -> str:
        """
        Get the device for the correct model part.

        Args:
            part (str): The model part that request the device

        Return:
            str: The device to load Parameters or Tensors on
        """

        if part == "A2P":
            return self.device_model_a2p
        elif part == "PE":
            return self.device_model_pe
        elif part == "WD":
            return self.device_model_wd
        else:
            raise ValueError(f"Unknown model part: `{part}`")
