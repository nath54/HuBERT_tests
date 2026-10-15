"""
Modular Prediction Heads for Speech Recognition and Decoding.
Includes:
- CTCProjectionHead: Discrete phoneme vocabulary projection and CTC collapse decoding
- ArticulatoryMacroHead: Broad manner-of-articulation phonetic constraint projection (8 groups)
- BoundaryGateHead: Decoupled acoustic pause / boundary detection gate
- LengthExpertRouter: Word length expert classification across 7 overlapping intervals
"""

# Import Modules
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F

# Articulatory manner categories
NUM_ARTICULATORY_GROUPS: int = 8
# 112 standard IPA phonetic tokens
PHONEME_VOCAB_SIZE: int = 112
# Overlapping word length intervals: [min_len, max_len]
LENGTH_INTERVALS: List[Tuple[int, int]] = [
    (1, 5),    # 0: VERY_SHORT
    (3, 7),    # 1: SHORT
    (5, 10),   # 2: MEDIUM_SHORT
    (7, 12),   # 3: MEDIUM
    (9, 15),   # 4: MEDIUM_LARGE
    (12, 20),  # 5: LARGE
    (15, 30),  # 6: VERY_LARGE
]
NUM_OVERLAPPING_EXPERTS: int = len(LENGTH_INTERVALS)


class CTCProjectionHead(nn.Module):
    """
    Phonetic Connectionist Temporal Classification (CTC) Linear Projection Head.
    Maps frame hidden representations [B, T, D] to unnormalized log-probabilities [B, T, V]
    over the 112-token phoneme vocabulary (blank index: 1, space index: 8).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        vocab_size: int = PHONEME_VOCAB_SIZE,
        blank_id: int = 1,
        space_id: int = 8,
    ) -> None:
        """
        Initialize CTCProjectionHead.

        Args:
            embed_dim (int): Model hidden dimension D (512).
            vocab_size (int): Phoneme vocabulary size V (112).
            blank_id (int): CTC blank token ID (1).
            space_id (int): Acoustic space token ID (8).
        """
        # Parent initialization
        super().__init__()
        self.embed_dim: int = embed_dim
        self.vocab_size: int = vocab_size
        self.blank_id: int = blank_id
        self.space_id: int = space_id

        # Linear projection: [D, V]
        self.proj: nn.Linear = nn.Linear(embed_dim, vocab_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass computing raw CTC logits.

        Args:
            x (Tensor): Frame representations of shape [B, T, D].

        Returns:
            Tensor: CTC logits of shape [B, T, V].
        """
        # Dimension: [B, T, V], Type: torch.float32, Role: unnormalized phoneme logits
        logits: Tensor = self.proj(x)
        return logits

    @torch.no_grad()
    def decode_greedy_ids(self, logits: Tensor, pad_id: int = 0) -> List[List[int]]:
        """
        Standard greedy argmax CTC collapse decoding.

        Args:
            logits (Tensor): CTC logits of shape [B, T, V].
            pad_id (int): Padding token index to discard (0).

        Returns:
            List[List[int]]: Decoded token IDs per batch element (collapsed repeats and blanks).
        """
        # Argmax along vocabulary dimension: [B, T]
        argmax_ids: Tensor = logits.argmax(dim=-1)
        batch_results: List[List[int]] = []

        for b in range(argmax_ids.shape[0]):
            collapsed: List[int] = []
            prev_tok: Optional[int] = None
            seq: List[int] = argmax_ids[b].cpu().tolist()

            for tok in seq:
                # Discard consecutive duplicates
                if tok != prev_tok:
                    # Discard CTC blank and pad tokens
                    if tok != self.blank_id and tok != pad_id:
                        collapsed.append(tok)
                    prev_tok = tok

            batch_results.append(collapsed)

        return batch_results


class ArticulatoryMacroHead(nn.Module):
    """
    Articulatory Manner-of-Articulation Macro Head.
    Supervises broad linguistic categories (vowel, stop, fricative, nasal, liquid, affricate, etc.)
    serving as an articulatory inductive bias that stabilizes acoustic CTC convergence.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_classes: int = NUM_ARTICULATORY_GROUPS,
    ) -> None:
        """
        Initialize ArticulatoryMacroHead.

        Args:
            embed_dim (int): Model hidden dimension D (512).
            num_classes (int): Number of articulatory manner classes (8).
        """
        # Parent initialization
        super().__init__()
        self.embed_dim: int = embed_dim
        self.num_classes: int = num_classes

        # Linear projection: [D, num_classes]
        self.proj: nn.Linear = nn.Linear(embed_dim, num_classes, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass computing articulatory macro logits.

        Args:
            x (Tensor): Frame representations [B, T, D].

        Returns:
            Tensor: Macro articulatory logits [B, T, num_classes].
        """
        return self.proj(x)


class BoundaryGateHead(nn.Module):
    """
    Decoupled Acoustic Word Boundary & Pause Detection Gate.
    Predicts physical acoustic silence/pause boundaries between words directly from speech frames.
    """

    def __init__(self, embed_dim: int = 512) -> None:
        """
        Initialize BoundaryGateHead.

        Args:
            embed_dim (int): Hidden dimension D (512).
        """
        # Parent initialization
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass computing frame-level boundary probabilities.

        Args:
            x (Tensor): Frame representations [B, T, D].

        Returns:
            Tensor: Sigmoid boundary probabilities [B, T, 1].
        """
        return torch.sigmoid(self.net(x))


class LengthExpertRouter(nn.Module):
    """
    Overlapping Word-Length Experts MoE Router.
    Classifies continuous word latent vectors z_word into 7 overlapping length intervals:
    VERY_SHORT [1, 5], SHORT [3, 7], MEDIUM_SHORT [5, 10], MEDIUM [7, 12],
    MEDIUM_LARGE [9, 15], LARGE [12, 20], VERY_LARGE [15, 30].
    """

    def __init__(
        self,
        macro_dim: int = 512,
        num_experts: int = NUM_OVERLAPPING_EXPERTS,
    ) -> None:
        """
        Initialize LengthExpertRouter.

        Args:
            macro_dim (int): Word latent representation dimension (512).
            num_experts (int): Number of overlapping length experts (7).
        """
        # Parent initialization
        super().__init__()
        self.macro_dim: int = macro_dim
        self.num_experts: int = num_experts

        # Linear router: [macro_dim, num_experts]
        self.router: nn.Linear = nn.Linear(macro_dim, num_experts, bias=True)

    def forward(self, z_word: Tensor) -> Tensor:
        """
        Forward pass computing expert routing logits.

        Args:
            z_word (Tensor): Word latent representations of shape [B, L, macro_dim].

        Returns:
            Tensor: Expert routing logits of shape [B, L, num_experts].
        """
        return self.router(z_word)
