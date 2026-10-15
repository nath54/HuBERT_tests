"""
Modular Activation Modules for Neural Network Layers.
Provides clean OOP implementations of standard and speech-specific activations.
"""

# Import Modules
from typing import Optional, Union
import torch
from torch import nn, Tensor


class SwishActivation(nn.Module):
    """
    Swish / SiLU Activation: f(x) = x * sigmoid(beta * x).
    Default beta = 1.0 corresponds to standard SiLU (Sigmoid Linear Unit).
    Commonly used in Conformer convolution modules for smooth non-linearities.
    """

    def __init__(self, beta: float = 1.0, learnable: bool = False) -> None:
        """
        Initialize Swish Activation.

        Args:
            beta (float): Initial slope multiplier. Default is 1.0.
            learnable (bool): Whether beta is a learnable parameter.
        """
        # Parent initialization
        super().__init__()

        # Learnable parameter or fixed float
        if learnable:
            # Dimension: [1], Type: torch.float32, Role: learnable temperature
            self.beta: Optional[nn.Parameter] = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
        else:
            self.beta = None
            self.fixed_beta: float = beta

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass applying Swish non-linearity elementwise.

        Args:
            x (Tensor): Input tensor of arbitrary shape [*dims].

        Returns:
            Tensor: Output tensor of identical shape [*dims] and dtype.
        """
        # Shape: [*dims], Type: identical to x, Role: scaling factor for gating
        beta_val: Union[float, Tensor] = self.beta if self.beta is not None else self.fixed_beta

        # Shape: [*dims], Type: identical to x, Role: activated output x * sigmoid(beta * x)
        out: Tensor = x * torch.sigmoid(beta_val * x)
        return out


class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU): Splits input tensor along specified dimension into two halves (A, B)
    and computes A * sigmoid(B).
    Commonly used in speech acoustic frontends and Conformer depthwise convolution blocks.
    """

    def __init__(self, dim: int = 1) -> None:
        """
        Initialize GatedLinearUnit.

        Args:
            dim (int): Dimension along which to split the tensor into halves.
        """
        # Parent initialization
        super().__init__()

        # Split dimension
        self.dim: int = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass applying GLU gating.

        Args:
            x (Tensor): Input tensor of shape [B, 2*C, T] or [B, T, 2*C].

        Returns:
            Tensor: Gated tensor with half the channels along self.dim.
        """
        # Shape of a and b: [B, C, T] (half-channels along self.dim)
        # Type: torch.float32 or torch.float16, Role: gated features
        a, b = x.chunk(2, dim=self.dim)

        # Shape: identical to a, Role: elementwise gated activation
        out: Tensor = a * torch.sigmoid(b)
        return out


class QuickGELU(nn.Module):
    """
    Fast approximation of Gaussian Error Linear Unit (GELU):
    f(x) = x * sigmoid(1.702 * x).
    Highly optimized for mobile NPUs and TFLite inference.
    """

    def __init__(self) -> None:
        """
        Initialize QuickGELU.
        """
        # Parent initialization
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass applying QuickGELU.

        Args:
            x (Tensor): Input tensor of arbitrary shape [*dims].

        Returns:
            Tensor: Activated tensor of identical shape [*dims].
        """
        # Shape: [*dims], Type: identical to x, Role: fast sigmoid approximation of GELU
        out: Tensor = x * torch.sigmoid(1.702 * x)
        return out
