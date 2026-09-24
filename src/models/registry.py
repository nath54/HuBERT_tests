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

        return config_cls(**tier_params)

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
    display_name="PhonoHuBERT (Direct Phoneme Prediction)",
    description="Direct acoustic-to-phoneme prediction with specialized tokens (<same_as_last>, <silence>, <mask>, <blank>).",
    model_cls=PhonoHuBERTForPreTraining,
    config_cls=PhonoHuBERTConfig,
    target_extractor_cls=PhonemeTargetExtractor,
    target_type="phoneme_tokens",
)(PhonoHuBERTForPreTraining)
