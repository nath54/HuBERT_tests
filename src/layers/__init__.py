"""
Modular Neural Network Layers for Clean Model Architecture.
"""

from .activations import GatedLinearUnit, QuickGELU, SwishActivation
from .attention import AttentionPooling, BandCausalSelfAttention, MultiHeadSelfAttention
from .conformer_block import ConformerBlock
from .conformer_layers import (
    ConformerEngramMoEEncoder,
    ConformerMoETransformerLayer,
    DeepSeekMoEFeedForwardNetwork,
)
from .conv_modules import (
    AudioCNNFeatureExtractor,
    ConformerConvModule,
    MultiScaleDilatedConformerConvModule,
    HuBERTConvLayer,
    TransposeLayerNorm,
)
from .embeddings import ConvolutionalPositionalEmbedding, LearnablePositionalEmbedding, RotaryEmbedding
from .engram_memory import PhonemeEngramMemory
from .ffn import DeepSeekMoEFFN, FeedForwardNetwork
from .heads import (
    DecoupledBoundaryGateHead,
    HierarchicalRecursivePhonemeHead,
    WindowedMicroPartitionedRecursiveHead,
    decode_calibrated_phonemes,
    decode_top_k_calibrated_phonemes,
    LengthExpertRouter,
    WordLengthPredictor,
    DeepLatentDiffusionRefiner,
    GaussianNoiseScheduler,
    SoftLevenshteinLoss,
    build_band_causal_mask,
    LENGTH_INTERVALS,
    NUM_ARTICULATORY_GROUPS,
    NUM_OVERLAPPING_EXPERTS,
    PHONEME_VOCAB_SIZE,
    EXPERT_PARTITIONS,
)

__all__ = [
    "SwishActivation",
    "GatedLinearUnit",
    "QuickGELU",
    "ConvolutionalPositionalEmbedding",
    "RotaryEmbedding",
    "LearnablePositionalEmbedding",
    "AudioCNNFeatureExtractor",
    "ConformerConvModule",
    "MultiScaleDilatedConformerConvModule",
    "HuBERTConvLayer",
    "TransposeLayerNorm",
    "MultiHeadSelfAttention",
    "BandCausalSelfAttention",
    "AttentionPooling",
    "FeedForwardNetwork",
    "DeepSeekMoEFFN",
    "ConformerBlock",
    "ConformerMoETransformerLayer",
    "DeepSeekMoEFeedForwardNetwork",
    "ConformerEngramMoEEncoder",
    "PhonemeEngramMemory",
    "DecoupledBoundaryGateHead",
    "HierarchicalRecursivePhonemeHead",
    "WindowedMicroPartitionedRecursiveHead",
    "decode_calibrated_phonemes",
    "decode_top_k_calibrated_phonemes",
    "LengthExpertRouter",
    "WordLengthPredictor",
    "DeepLatentDiffusionRefiner",
    "GaussianNoiseScheduler",
    "SoftLevenshteinLoss",
    "build_band_causal_mask",
    "LENGTH_INTERVALS",
    "NUM_ARTICULATORY_GROUPS",
    "NUM_OVERLAPPING_EXPERTS",
    "PHONEME_VOCAB_SIZE",
    "EXPERT_PARTITIONS",
]
