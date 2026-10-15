"""
Modular Neural Network Layers for Clean Model Architecture.
"""

# Import Modules
from .activations import GatedLinearUnit, QuickGELU, SwishActivation
from .attention import AttentionPooling, BandCausalSelfAttention, MultiHeadSelfAttention
from .conformer_block import ConformerBlock
from .conv_modules import AudioCNNFeatureExtractor, ConformerConvModule
from .embeddings import ConvolutionalPositionalEmbedding, LearnablePositionalEmbedding, RotaryEmbedding
from .engram_memory import PhonemeEngramMemory
from .ffn import DeepSeekMoEFFN, FeedForwardNetwork
from .heads import (
    ArticulatoryMacroHead,
    BoundaryGateHead,
    CTCProjectionHead,
    LengthExpertRouter,
    LENGTH_INTERVALS,
    NUM_ARTICULATORY_GROUPS,
    NUM_OVERLAPPING_EXPERTS,
    PHONEME_VOCAB_SIZE,
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
    "MultiHeadSelfAttention",
    "BandCausalSelfAttention",
    "AttentionPooling",
    "FeedForwardNetwork",
    "DeepSeekMoEFFN",
    "ConformerBlock",
    "PhonemeEngramMemory",
    "CTCProjectionHead",
    "ArticulatoryMacroHead",
    "BoundaryGateHead",
    "LengthExpertRouter",
    "LENGTH_INTERVALS",
    "NUM_ARTICULATORY_GROUPS",
    "NUM_OVERLAPPING_EXPERTS",
    "PHONEME_VOCAB_SIZE",
]
