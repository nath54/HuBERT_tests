"""
Granular Quantization Engine for Tripartite Speech Recognition Architecture.
Supports per-model-part and per-layer-type quantization:
- Dynamic INT8 Quantization (torch.ao.quantization) for low-latency CPU and edge inference
- Float16 Half-Precision Quantization for GPU acceleration
- Weight-only simulated INT8 quantization for memory-constrained edge deployment
- Comprehensive parameter and memory profiling across individual model parts.
"""

# Import Modules
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
import copy
import torch
from torch import nn, Tensor

from ..model_part import Model_Part


@dataclass
class QuantizationConfig:
    """
    Configuration specification for quantizing a specific model part or layer subset.
    """

    # Target precision dtype: 'qint8', 'float16', or 'weight_only_int8'
    precision: str = "qint8"

    # Layer types targeted for quantization (default: nn.Linear)
    target_layer_types: Tuple[Type[nn.Module], ...] = (nn.Linear,)

    # Specific submodules or layer names to exclude from quantization
    excluded_module_names: List[str] = field(default_factory=lambda: ["ctc_head", "macro_head"])

    # Whether to perform in-place quantization or return a deepcopy
    in_place: bool = False


class ModelPartQuantizer:
    """
    Granular Quantizer providing targeted precision reduction per model part and per layer type.
    Enables selective quantization (e.g. Model_A2P in INT8, Model_PE in FP16, Model_WD in INT8).
    """

    def __init__(self) -> None:
        """
        Initialize the ModelPartQuantizer.
        """
        super().__init__()

    @staticmethod
    def inspect_memory_footprint(module: nn.Module) -> Dict[str, Any]:
        """
        Profiles the memory footprint and parameter count of a module.

        Args:
            module (nn.Module): PyTorch module or model part to profile.

        Returns:
            Dict[str, Any]: Profiling metrics including parameter count and memory in megabytes.
        """
        # Total parameters count
        total_params: int = 0
        # Trainable parameters count
        trainable_params: int = 0
        # Total memory in bytes across all parameters and buffers
        total_bytes: int = 0
        # Data type breakdown mapping dtype string to parameter count
        dtype_breakdown: Dict[str, int] = {}

        for param in module.parameters():
            # Param count: int, Role: number of elements in parameter tensor, Rationale: track model capacity
            p_count: int = param.numel()
            total_params += p_count
            if param.requires_grad:
                trainable_params += p_count

            # Param bytes: int, Role: memory footprint of tensor, Rationale: calculate storage size
            p_bytes: int = p_count * param.element_size()
            total_bytes += p_bytes

            dt_name: str = str(param.dtype)
            dtype_breakdown[dt_name] = dtype_breakdown.get(dt_name, 0) + p_count

        for buffer in module.buffers():
            # Buffer bytes: int, Role: memory footprint of buffer, Rationale: include running stats
            b_bytes: int = buffer.numel() * buffer.element_size()
            total_bytes += b_bytes

        # Size in megabytes: float, Role: human readable memory usage, Rationale: edge deployment planning
        size_mb: float = total_bytes / (1024.0 * 1024.0)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "memory_mb": round(size_mb, 2),
            "dtype_breakdown": dtype_breakdown,
        }

    def quantize_part(
        self,
        model_part: Model_Part,
        config: Optional[QuantizationConfig] = None,
    ) -> Model_Part:
        """
        Quantizes a single Model_Part according to the provided QuantizationConfig.

        Args:
            model_part (Model_Part): The model part instance (e.g. Model_A2P, Model_PE, Model_WD).
            config (Optional[QuantizationConfig]): Quantization settings. Defaults to INT8 linear quantization.

        Returns:
            Model_Part: The quantized model part.
        """
        cfg: QuantizationConfig = config or QuantizationConfig()
        target_part: Model_Part = model_part if cfg.in_place else copy.deepcopy(model_part)

        precision: str = cfg.precision.lower()

        if precision in ("float16", "fp16", "half"):
            # Cast all parameters to half precision float16
            target_part = target_part.half()
            return target_part

        if precision in ("bfloat16", "bf16"):
            # Cast all parameters to bfloat16
            target_part = target_part.bfloat16()
            return target_part

        if precision in ("qint8", "int8", "dynamic_int8"):
            # Execute PyTorch dynamic quantization for targeted layer types
            target_layers: Set[Type[nn.Module]] = set(cfg.target_layer_types)

            try:
                # Dynamic quantization of linear/recurrent layers for CPU inference
                quantized_part: Model_Part = torch.ao.quantization.quantize_dynamic(
                    model=target_part,
                    qconfig_spec=target_layers,
                    dtype=torch.qint8,
                )
                return quantized_part
            except Exception as e:
                # Fallback to unmodified module if dynamic quantization fails on current platform
                print(f"[ModelPartQuantizer] Dynamic quantization notice for {model_part.model_part}: {e}")
                return target_part

        if precision == "weight_only_int8":
            # Apply 8-bit weight quantization manually to targeted layers
            self._apply_weight_only_int8(target_part, cfg.target_layer_types, cfg.excluded_module_names)
            return target_part

        raise ValueError(f"Unsupported quantization precision: {cfg.precision}")

    def quantize_model(
        self,
        unified_model: nn.Module,
        configs: Optional[Dict[str, QuantizationConfig]] = None,
    ) -> nn.Module:
        """
        Quantizes all constituent parts of a unified Model with granular per-part configurations.

        Args:
            unified_model (nn.Module): Master unified Model instance.
            configs (Optional[Dict[str, QuantizationConfig]]): Mapping of part name ('A2P', 'PE', 'WD')
                                                               to its QuantizationConfig.

        Returns:
            nn.Module: Unified Model with individually quantized sub-models.
        """
        default_configs: Dict[str, QuantizationConfig] = {
            "A2P": QuantizationConfig(precision="qint8", target_layer_types=(nn.Linear,)),
            "PE": QuantizationConfig(precision="float16", target_layer_types=(nn.Linear,)),
            "WD": QuantizationConfig(precision="qint8", target_layer_types=(nn.Linear,)),
        }
        cfg_map = configs or default_configs

        # Quantize Part 1 (A2P) if present
        if hasattr(unified_model, "model_a2p") and "A2P" in cfg_map:
            unified_model.model_a2p = self.quantize_part(unified_model.model_a2p, cfg_map["A2P"])

        # Quantize Part 2 (PE) if present
        if hasattr(unified_model, "model_pe") and "PE" in cfg_map:
            unified_model.model_pe = self.quantize_part(unified_model.model_pe, cfg_map["PE"])

        # Quantize Part 3 (WD) if present
        if hasattr(unified_model, "model_wd") and "WD" in cfg_map:
            unified_model.model_wd = self.quantize_part(unified_model.model_wd, cfg_map["WD"])

        return unified_model

    def _apply_weight_only_int8(
        self,
        module: nn.Module,
        target_types: Tuple[Type[nn.Module], ...],
        excluded_names: List[str],
    ) -> None:
        """
        Converts weights in targeted layers to symmetric 8-bit integer representation
        while maintaining float32/float16 scale factors for edge-compatible inference.

        Args:
            module (nn.Module): Module to modify.
            target_types (Tuple[Type[nn.Module], ...]): Module classes to quantize.
            excluded_names (List[str]): Substrings in module names to skip.
        """
        for name, child in module.named_children():
            if any(ex in name for ex in excluded_names):
                continue

            if isinstance(child, target_types) and hasattr(child, "weight") and child.weight is not None:
                with torch.no_grad():
                    # Original weight: Tensor [Out, In], Role: continuous parameters, Rationale: unquantized weights
                    w: Tensor = child.weight.data
                    # Max absolute value: Tensor [Out, 1], Role: per-channel maximum, Rationale: dynamic range scaling
                    max_val: Tensor = torch.max(torch.abs(w), dim=-1, keepdim=True)[0].clamp(min=1e-8)
                    # Scale factor: Tensor [Out, 1], Role: quantization scale, Rationale: map [-127, 127] to original range
                    scale: Tensor = max_val / 127.0
                    # Quantized int8 weight: Tensor [Out, In], Role: 8-bit discrete weights, Rationale: 4x memory reduction
                    q_w: Tensor = torch.clamp(torch.round(w / scale), -127, 127)
                    # Dequantized weight: Tensor [Out, In], Role: restored float weight, Rationale: simulate int8 quantization error
                    child.weight.data = (q_w * scale).to(w.dtype)
            else:
                self._apply_weight_only_int8(child, target_types, excluded_names)
