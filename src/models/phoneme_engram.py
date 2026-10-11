"""Phoneme Engram Memory Module (Phono-V8.0)

Implements conditional memory retrieval via multi-order N-gram lookups and
basis-level context-aware gating, inspired by DeepSeek V4.1 Flash and FactorEngram
(arXiv:2601.07372, arXiv:2609.35578).

In a multilingual unified phoneme space (112 tokens), this module offloads static
phonotactic patterns (transition rules, permissible clusters, assimilation priors)
into an O(1) constant-time parameter memory table, freeing the Conformer backbone
to focus on continuous acoustic discrimination.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# Prime multipliers for multi-head hashing
HASH_PRIMES_2GRAM: List[Tuple[int, int]] = [
    (10007, 10009),
    (10037, 10039),
    (10067, 10069),
    (10091, 10093),
    (10133, 10139),
    (10159, 10163),
    (10177, 10181),
    (10211, 10223),
]

HASH_PRIMES_3GRAM: List[Tuple[int, int, int]] = [
    (10007, 10009, 10037),
    (10039, 10067, 10069),
    (10091, 10093, 10133),
    (10139, 10159, 10163),
    (10177, 10181, 10211),
    (10223, 10243, 10247),
    (10253, 10259, 10267),
    (10271, 10273, 10289),
]

HASH_PRIMES_4GRAM: List[Tuple[int, int, int, int]] = [
    (10007, 10009, 10037, 10039),
    (10067, 10069, 10091, 10093),
    (10133, 10139, 10159, 10163),
    (10177, 10181, 10211, 10223),
    (10243, 10247, 10253, 10259),
    (10267, 10271, 10273, 10289),
    (10301, 10303, 10313, 10321),
    (10331, 10333, 10337, 10343),
]


class PhonemeEngramMemory(nn.Module):
    """DeepSeek-style Engram Memory for Phonetic N-Gram Retrieval.

    Args:
        embed_dim: Dimension of the Conformer hidden states (e.g., 384 or 512).
        vocab_size: Phoneme vocabulary size (default 112).
        table_size_2gram: Size of the 2-gram hash table (default 16384).
        table_size_3gram: Size of the 3-gram hash table (default 32768).
        table_size_4gram: Size of the 4-gram hash table (default 32768).
        num_heads: Number of independent hash lookup heads (default 4).
        memory_dim: Dimension of the retrieved memory vector (default 128).
        dropout: Dropout rate for memory injection.
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int = 112,
        table_size_2gram: int = 16384,
        table_size_3gram: int = 32768,
        table_size_4gram: int = 32768,
        num_heads: int = 4,
        memory_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.table_size_2gram = table_size_2gram
        self.table_size_3gram = table_size_3gram
        self.table_size_4gram = table_size_4gram
        self.num_heads = num_heads
        self.memory_dim = memory_dim

        # 1. Static parameter memory tables
        self.table_2gram = nn.Embedding(table_size_2gram, memory_dim)
        self.table_3gram = nn.Embedding(table_size_3gram, memory_dim)
        self.table_4gram = nn.Embedding(table_size_4gram, memory_dim)

        # Initialize tables with scaled normal
        nn.init.normal_(self.table_2gram.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.table_3gram.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.table_4gram.weight, mean=0.0, std=0.02)

        # Register prime multipliers as buffers for device portability
        p2 = torch.tensor(HASH_PRIMES_2GRAM[:num_heads], dtype=torch.long)
        p3 = torch.tensor(HASH_PRIMES_3GRAM[:num_heads], dtype=torch.long)
        p4 = torch.tensor(HASH_PRIMES_4GRAM[:num_heads], dtype=torch.long)
        self.register_buffer("primes_2gram", p2)
        self.register_buffer("primes_3gram", p3)
        self.register_buffer("primes_4gram", p4)

        # 2. Multi-head projection and combination: (3 orders * num_heads * memory_dim) -> embed_dim
        total_retrieved_dim = 3 * num_heads * memory_dim
        self.memory_projection = nn.Sequential(
            nn.Linear(total_retrieved_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Zero-initialize the final projection layer so Step 0 residual contribution is strictly 0.0
        nn.init.zeros_(self.memory_projection[-1].weight)
        nn.init.zeros_(self.memory_projection[-1].bias)

        # 3. FactorEngram Context-Aware Channel-Wise Gating
        # Computes per-channel gate g_t in [0, 1] based on context h_t and memory m_t
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        # Initialize gate bias negative so gate starts near 0 (~0.05)
        nn.init.constant_(self.gate_net[-2].bias, -3.0)

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _hash_2gram(self, tokens: torch.Tensor) -> torch.Tensor:
        """Hash 2-grams: tokens shape [B, T]. Returns [B, T, num_heads]."""
        t_prev = F.pad(tokens[:, :-1], (1, 0), value=0)
        t_curr = tokens
        pairs = torch.stack([t_prev, t_curr], dim=-1).unsqueeze(2)  # [B, T, 1, 2]
        primes = self.primes_2gram.unsqueeze(0).unsqueeze(0)  # [1, 1, num_heads, 2]
        hashes = (pairs * primes).sum(dim=-1) % self.table_size_2gram  # [B, T, num_heads]
        return hashes

    def _hash_3gram(self, tokens: torch.Tensor) -> torch.Tensor:
        """Hash 3-grams: tokens shape [B, T]. Returns [B, T, num_heads]."""
        t_prev2 = F.pad(tokens[:, :-2], (2, 0), value=0)
        t_prev1 = F.pad(tokens[:, :-1], (1, 0), value=0)
        t_curr = tokens
        triplets = torch.stack([t_prev2, t_prev1, t_curr], dim=-1).unsqueeze(2)  # [B, T, 1, 3]
        primes = self.primes_3gram.unsqueeze(0).unsqueeze(0)  # [1, 1, num_heads, 3]
        hashes = (triplets * primes).sum(dim=-1) % self.table_size_3gram  # [B, T, num_heads]
        return hashes

    def _hash_4gram(self, tokens: torch.Tensor) -> torch.Tensor:
        """Hash 4-grams: tokens shape [B, T]. Returns [B, T, num_heads]."""
        t_prev3 = F.pad(tokens[:, :-3], (3, 0), value=0)
        t_prev2 = F.pad(tokens[:, :-2], (2, 0), value=0)
        t_prev1 = F.pad(tokens[:, :-1], (1, 0), value=0)
        t_curr = tokens
        quads = torch.stack([t_prev3, t_prev2, t_prev1, t_curr], dim=-1).unsqueeze(2)  # [B, T, 1, 4]
        primes = self.primes_4gram.unsqueeze(0).unsqueeze(0)  # [1, 1, num_heads, 4]
        hashes = (quads * primes).sum(dim=-1) % self.table_size_4gram  # [B, T, num_heads]
        return hashes

    def forward(
        self,
        hidden_states: torch.Tensor,
        phoneme_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query Engram memory with phoneme tokens and context-gate into hidden states.

        Args:
            hidden_states: Conformer hidden states [batch, time, embed_dim]
            phoneme_tokens: Discrete phoneme token IDs [batch, time], clamped to [0, vocab_size-1]

        Returns:
            fused_states: Modulated hidden states [batch, time, embed_dim]
            gate_values: Gate activation [batch, time, embed_dim]
        """
        B, T, D = hidden_states.shape
        clamped_tokens = phoneme_tokens.clamp(0, self.vocab_size - 1)

        # 1. Multi-head hash computation for 2-gram, 3-gram, 4-gram
        h2 = self._hash_2gram(clamped_tokens)  # [B, T, H]
        h3 = self._hash_3gram(clamped_tokens)  # [B, T, H]
        h4 = self._hash_4gram(clamped_tokens)  # [B, T, H]

        # 2. O(1) table lookups across heads
        emb2 = self.table_2gram(h2).view(B, T, -1)  # [B, T, H * memory_dim]
        emb3 = self.table_3gram(h3).view(B, T, -1)  # [B, T, H * memory_dim]
        emb4 = self.table_4gram(h4).view(B, T, -1)  # [B, T, H * memory_dim]

        # Concatenate all orders: [B, T, 3 * H * memory_dim]
        raw_memory = torch.cat([emb2, emb3, emb4], dim=-1)

        # 3. Project retrieved memory to embed_dim
        projected_memory = self.memory_projection(raw_memory)  # [B, T, D]

        # 4. FactorEngram Context-Aware Channel-Wise Gating
        gate_input = torch.cat([hidden_states, projected_memory], dim=-1)
        gate = self.gate_net(gate_input)  # [B, T, D] in [0, 1]

        # 5. Modulate and residual inject
        modulated_memory = gate * projected_memory
        out = hidden_states + self.dropout(modulated_memory)

        return out, gate
