"""
Clean, Modular Tripartite Speech Recognition Package.
Architecture:
- Model 1 (Model_A2P): 16kHz Audio -> Acoustic Phonemes & Spaces (Conformer + MoE + Engram)
- Model 2 (Model_PE):  Intra-word Phonemes -> Continuous Word Latents (z_word in R^512)
- Model 3 (Model_WD):  Word Latents -> Orthographic Characters (Band-Causal Decoder + Length MoE)
- Model: Complete unified orchestrator with per-part device routing
- Quantization: Granular per-part and per-layer quantization engine (INT8 / FP16)
- Streaming: Dual-threaded asynchronous streaming runner (Thread 1: Audio->Phonemes, Thread 2: Phonemes->Text)
- Export: Google Pixel 10 (Tensor G5 NPU) static-shape ONNX & TFLite export
"""

# Import Modules
from .model_config import (
    ModelConfig,
    ModelA2PConfig,
    ModelPEConfig,
    ModelWDConfig,
    get_class,
)
from .model_part import Model_Part
from .model1_a2p import Model_A2P
from .model2_pe import Model_PE
from .model3_wd import Model_WD
from .model import Model
from .quantization import (
    ModelPartQuantizer,
    QuantizationConfig,
)
from .streaming import (
    AudioCircularBuffer,
    PhonemeWordQueue,
    DualThreadStreamingASR,
)
from .export import (
    TFLiteExporter,
    Pixel10MobileRunner,
)
from .weights_manager import (
    WeightsManager,
    WeightMetadata,
)

__all__ = [
    # Configurations
    "ModelConfig",
    "ModelA2PConfig",
    "ModelPEConfig",
    "ModelWDConfig",
    "get_class",
    # Base and Models
    "Model_Part",
    "Model_A2P",
    "Model_PE",
    "Model_WD",
    "Model",
    # Quantization
    "ModelPartQuantizer",
    "QuantizationConfig",
    # Streaming
    "AudioCircularBuffer",
    "PhonemeWordQueue",
    "DualThreadStreamingASR",
    # Mobile Export & Inference
    "TFLiteExporter",
    "Pixel10MobileRunner",
    # Checkpoints & Weights Manager
    "WeightsManager",
    "WeightMetadata",
]
