"""
Modular Multi-Order Phoneme Engram Memory Module.
Injects learned n-gram phonotactic memory representations into intermediate Conformer layers
conditioned on draft phoneme token sequences (2-gram, 3-gram, 4-gram).
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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
    """DeepSeek-style Engram Memory for Phonetic N-Gram Retrieval."""

    def __init__(
        self,
        embed_dim: int = 512,
        vocab_size: int = 112,
        table_size_2gram: int = 16384,
        table_size_3gram: int = 32768,
        table_size_4gram: int = 32768,
        num_heads: int = 4,
        memory_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim: int = embed_dim
        self.vocab_size: int = vocab_size
        self.table_size_2gram: int = table_size_2gram
        self.table_size_3gram: int = table_size_3gram
        self.table_size_4gram: int = table_size_4gram
        self.num_heads: int = num_heads
        self.memory_dim: int = memory_dim

        # 1. Static parameter memory tables
        self.table_2gram: nn.Embedding = nn.Embedding(table_size_2gram, memory_dim)
        self.table_3gram: nn.Embedding = nn.Embedding(table_size_3gram, memory_dim)
        self.table_4gram: nn.Embedding = nn.Embedding(table_size_4gram, memory_dim)

        nn.init.normal_(self.table_2gram.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.table_3gram.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.table_4gram.weight, mean=0.0, std=0.02)

        p2 = torch.tensor(HASH_PRIMES_2GRAM[:num_heads], dtype=torch.long)
        p3 = torch.tensor(HASH_PRIMES_3GRAM[:num_heads], dtype=torch.long)
        p4 = torch.tensor(HASH_PRIMES_4GRAM[:num_heads], dtype=torch.long)
        self.register_buffer("primes_2gram", p2)
        self.register_buffer("primes_3gram", p3)
        self.register_buffer("primes_4gram", p4)

        # 2. Multi-head projection
        total_retrieved_dim: int = 3 * num_heads * memory_dim
        self.memory_projection: nn.Sequential = nn.Sequential(
            nn.Linear(total_retrieved_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        nn.init.zeros_(self.memory_projection[-1].weight)
        nn.init.zeros_(self.memory_projection[-1].bias)

        # 3. FactorEngram Context-Aware Gating
        self.gate_net: nn.Sequential = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate_net[-2].bias, -3.0)

        self.norm: nn.LayerNorm = nn.LayerNorm(embed_dim)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def _hash_2gram(self, tokens: Tensor) -> Tensor:
        t_prev = F.pad(tokens[:, :-1], (1, 0), value=0)
        t_curr = tokens
        pairs = torch.stack([t_prev, t_curr], dim=-1).unsqueeze(2)
        primes = self.primes_2gram.unsqueeze(0).unsqueeze(0)
        hashes = (pairs * primes).sum(dim=-1) % self.table_size_2gram
        return hashes

    def _hash_3gram(self, tokens: Tensor) -> Tensor:
        t_p2 = F.pad(tokens[:, :-2], (2, 0), value=0)
        t_p1 = F.pad(tokens[:, :-1], (1, 0), value=0)
        t_curr = tokens
        triplets = torch.stack([t_p2, t_p1, t_curr], dim=-1).unsqueeze(2)
        primes = self.primes_3gram.unsqueeze(0).unsqueeze(0)
        hashes = (triplets * primes).sum(dim=-1) % self.table_size_3gram
        return hashes

    def _hash_4gram(self, tokens: Tensor) -> Tensor:
        t_p3 = F.pad(tokens[:, :-3], (3, 0), value=0)
        t_p2 = F.pad(tokens[:, :-2], (2, 0), value=0)
        t_p1 = F.pad(tokens[:, :-1], (1, 0), value=0)
        t_curr = tokens
        quads = torch.stack([t_p3, t_p2, t_p1, t_curr], dim=-1).unsqueeze(2)
        primes = self.primes_4gram.unsqueeze(0).unsqueeze(0)
        hashes = (quads * primes).sum(dim=-1) % self.table_size_4gram
        return hashes

    def forward(self, x: Tensor, draft_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        B, T, D = x.shape
        h2 = self._hash_2gram(draft_tokens)
        h3 = self._hash_3gram(draft_tokens)
        h4 = self._hash_4gram(draft_tokens)

        m2 = self.table_2gram(h2).view(B, T, self.num_heads * self.memory_dim)
        m3 = self.table_3gram(h3).view(B, T, self.num_heads * self.memory_dim)
        m4 = self.table_4gram(h4).view(B, T, self.num_heads * self.memory_dim)

        m_concat = torch.cat([m2, m3, m4], dim=-1)
        m_proj = self.memory_projection(m_concat)

        gate_in = torch.cat([x, m_proj], dim=-1)
        g = self.gate_net(gate_in)

        out = self.norm(x + self.dropout(g * m_proj))
        return out, g
