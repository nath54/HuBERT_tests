"""Central Model & Architecture Registry for AudioLearn.

Provides a modular factory pattern allowing dynamic registration, parameter variation,
and instant UI/API discovery of new speech models and target extractors.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type
import torch
import torch.nn as nn

from src.models.config import HuBERTConfig
from src.models.hubert_pretrain import HuBERTForPreTraining
from src.models.phono_hubert import PhonoHuBERTConfig, PhonoHuBERTForPreTraining
from src.models.phono_hubert_dual import PhonoHuBERTDualConfig, PhonoHuBERTDualForPreTraining
from src.models.phono_hubert_hierarchical import (
    PhonoHuBERTHierarchicalConfig,
    PhonoHuBERTHierarchicalForPreTraining,
)
from src.models.phono_hubert_recursive import (
    PhonoHuBERTRecursiveConfig,
    PhonoHuBERTRecursiveForPreTraining,
)
from src.models.phono_variants import (
    PhonoV1FrontendConfig,
    PhonoV1FrontendForPreTraining,
    PhonoV2SpecAugmentConfig,
    PhonoV2SpecAugmentForPreTraining,
    PhonoV3HybridConfig,
    PhonoV3HybridForPreTraining,
    PhonoV4ScaledConfig,
    PhonoV4ScaledForPreTraining,
    PhonoV5BeamConfig,
    PhonoV5BeamForPreTraining,
    PhonoV61MoEConfig,
    PhonoV61MoEForPreTraining,
    PhonoV62SparseConfig,
    PhonoV62SparseForPreTraining,
    PhonoV63DiffusionConfig,
    PhonoV63DiffusionForPreTraining,
    PhonoV64GatedDiffusionConfig,
    PhonoV64GatedDiffusionForPreTraining,
)
from src.models.phono_v6_5_hierarchical_decoder import (
    HierarchicalByteConfig,
    PhonoV65HierarchicalByteDecoder,
)
from src.models.phono_v6_6_adaptive_decoder import (
    AdaptivePathConfig,
    PhonoV66AdaptiveDecoder,
)
from src.models.phono_v6_7_windowed_decoder import (
    WindowedAdaptivePathConfig,
    PhonoV67WindowedDecoder,
)
from src.models.phono_v6_7_speech_model import (
    PhonoV67SpeechConfig,
    PhonoV67SpeechModel,
)
from src.data.target_extractors import (
    BaseTargetExtractor,
    KMeansUnitExtractor,
    PhonemeTargetExtractor,
)


# Standard Architecture Scaling Tiers
STANDARD_TIERS: Dict[str, Dict[str, int]] = {
    "mini": {
        "encoder_layers": 4,
        "encoder_heads": 4,
        "encoder_embed_dim": 256,
        "encoder_ffn_dim": 1024,
    },
    "small": {
        "encoder_layers": 6,
        "encoder_heads": 6,
        "encoder_embed_dim": 384,
        "encoder_ffn_dim": 1536,
    },
    "medium": {
        "encoder_layers": 8,
        "encoder_heads": 8,
        "encoder_embed_dim": 512,
        "encoder_ffn_dim": 2048,
    },
    "base": {
        "encoder_layers": 12,
        "encoder_heads": 12,
        "encoder_embed_dim": 768,
        "encoder_ffn_dim": 3072,
    },
}


class ModelRegistry:
    """Registry maintaining speech architectures, parameter factories, and metadata."""

    _models: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        model_id: str,
        display_name: str,
        description: str,
        model_cls: Type[nn.Module],
        config_cls: Type[Any],
        target_extractor_cls: Type[BaseTargetExtractor],
        target_type: str,
        supported_tiers: Optional[List[str]] = None,
        default_tier: str = "mini",
        custom_param_specs: Optional[Dict[str, Any]] = None,
    ):
        """Decorator or direct call to register a new model architecture."""
        def decorator(fn_or_cls):
            cls._models[model_id] = {
                "model_id": model_id,
                "display_name": display_name,
                "description": description,
                "model_cls": model_cls,
                "config_cls": config_cls,
                "target_extractor_cls": target_extractor_cls,
                "target_type": target_type,
                "supported_tiers": supported_tiers or list(STANDARD_TIERS.keys()),
                "default_tier": default_tier,
                "custom_param_specs": custom_param_specs or {},
            }
            return fn_or_cls

        return decorator

    @classmethod
    def list_models(cls) -> List[Dict[str, Any]]:
        """Return catalog of all registered models for UI dropdowns and API discovery."""
        catalog = []
        for mid, m in cls._models.items():
            tiers_info = {}
            for t in m["supported_tiers"]:
                cfg_params = STANDARD_TIERS.get(t, {})
                tiers_info[t] = cfg_params

            catalog.append({
                "id": mid,
                "model_id": mid,
                "name": m["display_name"],
                "display_name": m["display_name"],
                "description": m["description"],
                "target_type": m["target_type"],
                "supported_tiers": m["supported_tiers"],
                "default_tier": m["default_tier"],
                "tiers": tiers_info,
            })
        return catalog

    @classmethod
    def get_entry(cls, model_id: str) -> Dict[str, Any]:
        if model_id not in cls._models:
            raise KeyError(f"Model architecture '{model_id}' not found in registry. Registered: {list(cls._models.keys())}")
        return cls._models[model_id]

    @classmethod
    def build_config(
        cls,
        model_id: str,
        tier: str = "mini",
        **kwargs,
    ) -> Any:
        """Instantiate architecture configuration with variable parameter overrides."""
        entry = cls.get_entry(model_id)
        config_cls = entry["config_cls"]

        # Base tier parameters
        tier_params = dict(STANDARD_TIERS.get(tier, STANDARD_TIERS["mini"]))
        
        # Apply any variable parameter overrides passed by caller
        tier_params.update(kwargs)

        # Filter parameters to only fields accepted by config_cls
        import dataclasses
        import inspect
        if dataclasses.is_dataclass(config_cls):
            valid_fields = {f.name for f in dataclasses.fields(config_cls)}
            filtered_params = {k: v for k, v in tier_params.items() if k in valid_fields}
        else:
            sig = inspect.signature(config_cls.__init__)
            valid_fields = set(sig.parameters.keys()) - {"self"}
            filtered_params = {k: v for k, v in tier_params.items() if k in valid_fields}

        return config_cls(**filtered_params)

    @classmethod
    def build_model(
        cls,
        model_id: str,
        tier: str = "mini",
        config: Optional[Any] = None,
        **kwargs,
    ) -> nn.Module:
        """Instantiate speech model with specified architecture and tier parameters."""
        entry = cls.get_entry(model_id)
        model_cls = entry["model_cls"]

        if not isinstance(tier, str):
            config = tier
            tier = getattr(config, "tier", "mini")

        if config is None:
            config = cls.build_config(model_id, tier=tier, **kwargs)

        return model_cls(config)

    @classmethod
    def build_target_extractor(
        cls,
        model_id: str,
        **kwargs,
    ) -> BaseTargetExtractor:
        """Build the target extractor corresponding to the selected model architecture."""
        entry = cls.get_entry(model_id)
        extractor_cls = entry["target_extractor_cls"]
        return extractor_cls(**kwargs)

    @classmethod
    def get_checkpoint_path(cls, model_id: str, tier: str) -> Path:
        """Get standard checkpoint path for an architecture and tier, with backwards compatibility."""
        arch_dir = Path("checkpoints") / model_id / tier
        arch_file = arch_dir / "latest_checkpoint.pt"
        if arch_file.exists():
            return arch_file

        # Fallbacks for existing runs
        if model_id == "hubert_kmeans" and tier == "mini":
            for cand in [Path("checkpoints/pretrain_checkpoint_latest.pt"), Path("checkpoints/best_model.pt")]:
                if cand.exists():
                    return cand

        return arch_file

    @classmethod
    def get_history_path(cls, model_id: str, tier: str) -> Path:
        """Get scaling history JSON path for an architecture and tier."""
        tier_path = Path("logs") / f"{model_id}_{tier}_history.json"
        if tier_path.exists():
            return tier_path
        
        if model_id == "hubert_kmeans" and tier == "mini":
            return Path("logs/scaling_benchmark_history.json")

        return tier_path


# -------------------------------------------------------------
# Register Built-in Core Speech Models
# -------------------------------------------------------------

ModelRegistry.register(
    model_id="hubert_kmeans",
    display_name="HuBERT (Acoustic K-Means SSL)",
    description="Self-supervised pre-training using 39-dim MFCC acoustic cluster pseudo-labels (Hsu et al., 2021).",
    model_cls=HuBERTForPreTraining,
    config_cls=HuBERTConfig,
    target_extractor_cls=KMeansUnitExtractor,
    target_type="kmeans_cluster",
)(HuBERTForPreTraining)

ModelRegistry.register(
    model_id="phono_hubert",
    display_name="PhonoHuBERT (Anti-Blank Regularized CTC)",
    description="Direct acoustic-to-phoneme prediction with quadratic anti-blank margin regularization & calibrated decoding.",
    model_cls=PhonoHuBERTForPreTraining,
    config_cls=PhonoHuBERTConfig,
    target_extractor_cls=PhonemeTargetExtractor,
    target_type="phoneme_tokens",
)(PhonoHuBERTForPreTraining)

ModelRegistry.register(
    model_id="phono_hubert_dual",
    display_name="PhonoHuBERT-Dual (Masked Frame SSL + CTC)",
    description="Dual-loss speech Transformer pairing frame-synchronous masked phoneme Cross-Entropy with auxiliary sequence CTC.",
    model_cls=PhonoHuBERTDualForPreTraining,
    config_cls=PhonoHuBERTDualConfig,
    target_extractor_cls=PhonemeTargetExtractor,
    target_type="phoneme_tokens",
)(PhonoHuBERTDualForPreTraining)

ModelRegistry.register(
    model_id="phono_hubert_hierarchical",
    display_name="PhonoHuBERT-Hierarchical (2-Stage Gated Head)",
    description="Two-stage gated architecture decomposing decoding into an Acoustic State Router and a pure Phoneme Head.",
    model_cls=PhonoHuBERTHierarchicalForPreTraining,
    config_cls=PhonoHuBERTHierarchicalConfig,
    target_extractor_cls=PhonemeTargetExtractor,
    target_type="phoneme_tokens",
)(PhonoHuBERTHierarchicalForPreTraining)

ModelRegistry.register(
    model_id="phono_hubert_recursive",
    display_name="PhonoHuBERT-Recursive (Recurrent Temporal Feedback)",
    description="Autoregressive recurrent frame-memory feedback head eliminating the conditional independence assumption of CTC.",
    model_cls=PhonoHuBERTRecursiveForPreTraining,
    config_cls=PhonoHuBERTRecursiveConfig,
    target_extractor_cls=PhonemeTargetExtractor,
    target_type="phoneme_tokens",
)(PhonoHuBERTRecursiveForPreTraining)

# -------------------------------------------------------------
# Progressive Levers for SOTA Phoneme Recognition (< 10% PER)
# -------------------------------------------------------------

ModelRegistry.register(
    model_id="phono_v1_frontend",
    display_name="Phono-V1 (Pretrained Front-End)",
    description="Variant 1: Hierarchical router with Meta HuBERT 960h formant-tuned 7-layer CNN feature extractor.",
    model_cls=PhonoV1FrontendForPreTraining,
    config_cls=PhonoV1FrontendConfig,
    target_extractor_cls=PhonemeTargetExtractor,
    target_type="phoneme_tokens",
)(PhonoV1FrontendForPreTraining)

ModelRegistry.register(
    model_id="phono_v2_specaugment",
    display_name="Phono-V2 (SpecAugment)",
    description="Variant 2: Variant 1 + dynamic acoustic time-span and frequency-channel SpecAugment.",
    model_cls=PhonoV2SpecAugmentForPreTraining,
    config_cls=PhonoV2SpecAugmentConfig,
    target_extractor_cls=PhonemeTargetExtractor,
    target_type="phoneme_tokens",
)(PhonoV2SpecAugmentForPreTraining)

ModelRegistry.register(
    model_id="phono_v3_hybrid",
    display_name="Phono-V3 (Hybrid Real Data)",
    description="Variant 3: Variant 2 + hybrid streaming of real human speech (LibriSpeech) and synthetic TTS.",
    model_cls=PhonoV3HybridForPreTraining,
    config_cls=PhonoV3HybridConfig,
    target_extractor_cls=PhonemeTargetExtractor,
    target_type="phoneme_tokens",
)(PhonoV3HybridForPreTraining)

ModelRegistry.register(
    model_id="phono_v4_scaled",
    display_name="Phono-V4 (Deep Scaled)",
    description="Variant 4: Variant 3 + deep scaled transformer capacity and extended schedule.",
    model_cls=PhonoV4ScaledForPreTraining,
    config_cls=PhonoV4ScaledConfig,
    target_extractor_cls=PhonemeTargetExtractor,
    target_type="phoneme_tokens",
)(PhonoV4ScaledForPreTraining)

ModelRegistry.register(
    model_id="phono_v5_beam",
    display_name="Phono-V5 (Phonotactic Beam)",
    description="Variant 5: Variant 4 + CTC Prefix Beam Search decoding with phonotactic transition constraints.",
    model_cls=PhonoV5BeamForPreTraining,
    config_cls=PhonoV5BeamConfig,
    target_extractor_cls=PhonemeTargetExtractor,
    target_type="phoneme_tokens",
)(PhonoV5BeamForPreTraining)

ModelRegistry.register(
    model_id="phono_v6_1_moe",
    display_name="Phono-V6.1 (Mixture of Experts)",
    description="Variant 6.1: Variant 5 + Top-2 Gated Mixture of Experts FFN with Switch auxiliary load balancing.",
    model_cls=PhonoV61MoEForPreTraining,
    config_cls=PhonoV61MoEConfig,
    target_extractor_cls=PhonemeTargetExtractor,
    target_type="phoneme_tokens",
)(PhonoV61MoEForPreTraining)

ModelRegistry.register(
    model_id="phono_v6_2_sparse",
    display_name="Phono-V6.2 (Sparse Syllabic Attention)",
    description="Variant 6.2: Variant 6.1 + Sparse Local Syllabic Attention (+-320ms window) preventing attention diffusion.",
    model_cls=PhonoV62SparseForPreTraining,
    config_cls=PhonoV62SparseConfig,
    target_extractor_cls=PhonemeTargetExtractor,
    target_type="phoneme_tokens",
)(PhonoV62SparseForPreTraining)

ModelRegistry.register(
    model_id="phono_v6_3_diffusion",
    display_name="Phono-V6.3 (Diffusion Decoding)",
    description="Variant 6.3: Variant 6.2 + Iterative Sliding Window Denoising / Diffusion Decoding over acoustic representations.",
    model_cls=PhonoV63DiffusionForPreTraining,
    config_cls=PhonoV63DiffusionConfig,
    target_extractor_cls=PhonemeTargetExtractor,
    target_type="phoneme_tokens",
)(PhonoV63DiffusionForPreTraining)

ModelRegistry.register(
    model_id="phono_v6_4_gated_diffusion",
    display_name="Phono-V6.4 (Confidence-Gated Diffusion)",
    description="Variant 6.4: Variant 6.3 + Confidence-Gated Latent Diffusion. High-confidence CTC frames bypass diffusion; only ambiguous frames are refined via a deep 3-block denoiser.",
    model_cls=PhonoV64GatedDiffusionForPreTraining,
    config_cls=PhonoV64GatedDiffusionConfig,
    target_extractor_cls=PhonemeTargetExtractor,
    target_type="phoneme_tokens",
)(PhonoV64GatedDiffusionForPreTraining)

ModelRegistry.register(
    model_id="phono_v6_5_hierarchical_decoder",
    display_name="Phono-V6.5 (Hierarchical Byte Decoder)",
    description="Variant 6.5: Token-free universal multilingual hierarchical word-to-byte decoder. Features a 261-class UTF-8 recursive byte head (<150 KB RAM) with 0.0% OOV across all languages.",
    model_cls=PhonoV65HierarchicalByteDecoder,
    config_cls=HierarchicalByteConfig,
    target_extractor_cls=PhonemeTargetExtractor,
    target_type="phoneme_tokens",
)(PhonoV65HierarchicalByteDecoder)

ModelRegistry.register(
    model_id="phono_v6_6_adaptive_decoder",
    display_name="Phono-V6.6 (4-Path Length-Adaptive Decoder)",
    description="Variant 6.6: 4-Path Length-Adaptive Word Routing & Conditioned MoE Character Decoder. Dynamically bounds unrolling across Blank/Special (0 FLOPs), Short (K=5), Medium (K=9), and Long (K=24) words.",
    model_cls=PhonoV66AdaptiveDecoder,
    config_cls=AdaptivePathConfig,
    target_extractor_cls=PhonemeTargetExtractor,
    target_type="phoneme_tokens",
)(PhonoV66AdaptiveDecoder)

ModelRegistry.register(
    model_id="phono_v6_7_windowed_decoder",
    display_name="Phono-V6.7 (Windowed Multi-Word MoE Decoder)",
    description="Variant 6.7: Windowed Multi-Word Context Cross-Attention (W=4 preceding words) with relative positional bias & ramped 30% scheduled sampling. Eliminates homophone ambiguity and grammatical agreement bottlenecks.",
    model_cls=PhonoV67WindowedDecoder,
    config_cls=WindowedAdaptivePathConfig,
    target_extractor_cls=PhonemeTargetExtractor,
    target_type="phoneme_tokens",
)(PhonoV67WindowedDecoder)

ModelRegistry.register(
    model_id="phono_v6_7_speech_model",
    display_name="Phono-V6.7 (Continuous Speech Model with Double Loss)",
    description="Variant 6.7 End-to-End: Direct continuous speech wiring with double loss (CTC phoneme loss on frozen/fine-tuned PhonoV6.4 Gated Diffusion encoder + character cross-entropy on PhonoV6.7 Windowed MoE Decoder).",
    model_cls=PhonoV67SpeechModel,
    config_cls=PhonoV67SpeechConfig,
    target_extractor_cls=PhonemeTargetExtractor,
    target_type="phoneme_tokens",
)(PhonoV67SpeechModel)



