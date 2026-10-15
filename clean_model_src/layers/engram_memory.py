"""
Modular Multi-Order Phoneme Engram Memory Module.
Injects learned n-gram phonotactic memory representations into intermediate Conformer layers
conditioned on draft phoneme token sequences (2-gram, 3-gram, 4-gram).
"""

# Import Modules
from typing import List, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class PhonemeEngramMemory(nn.Module):
    """
    Multi-Order Phoneme Engram Conditional Memory Module.
    Inspired by human auditory cortex engram retrieval and n-gram memory hashing (DeepSeek Engram, 2024).
    Computes polynomial rolling hashes of draft phoneme sequences over multiple context spans
    (bigram, trigram, 4-gram), retrieves continuous memory patterns across H heads, and gates them
    into Conformer representations using a zero-initialized sigmoid gating projection.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        vocab_size: int = 112,
        num_heads: int = 4,
        memory_dim: int = 128,
        table_size: int = 4096,
        ngram_orders: Tuple[int, ...] = (2, 3, 4),
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize PhonemeEngramMemory.

        Args:
            embed_dim (int): Conformer model channel dimension D (512).
            vocab_size (int): Discrete phoneme vocabulary size V (112).
            num_heads (int): Number of parallel hash heads H (4).
            memory_dim (int): Memory channel dimension per head (128).
            table_size (int): Hash table slot size per order (4096).
            ngram_orders (Tuple[int, ...]): Context orders to hash (2, 3, 4).
            dropout (float): Dropout probability.
        """
        # Parent initialization
        super().__init__()
        self.embed_dim: int = embed_dim
        self.vocab_size: int = vocab_size
        self.num_heads: int = num_heads
        self.memory_dim: int = memory_dim
        self.table_size: int = table_size
        self.ngram_orders: Tuple[int, ...] = ngram_orders

        # Distinct polynomial hash multipliers per head
        # Register as constant buffer: [num_heads]
        hash_primes = torch.tensor([131, 137, 149, 151][:num_heads], dtype=torch.long)
        self.register_buffer("hash_primes", hash_primes, persistent=False)

        # Embedding tables for each n-gram order: ModuleDict of Embedding(table_size, memory_dim)
        self.memory_tables: nn.ModuleDict = nn.ModuleDict({
            f"order_{n}": nn.Embedding(table_size, memory_dim)
            for n in ngram_orders
        })

        # Projection from retrieved memory (len(ngram_orders) * memory_dim) -> embed_dim
        total_mem_dim: int = len(ngram_orders) * memory_dim
        self.memory_proj: nn.Sequential = nn.Sequential(
            nn.Linear(total_mem_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Gating projection: learns a dynamic sigmoid blend factor [B, T, 1]
        self.gate_proj: nn.Linear = nn.Linear(embed_dim, 1)

        # Zero-initialize the gate projection so initial engram contribution is exactly 0
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -4.0)  # sigmoid(-4.0) ~ 0.018

        self.layer_norm: nn.LayerNorm = nn.LayerNorm(embed_dim)

    def _compute_ngram_hashes(self, tokens: Tensor, n: int) -> Tensor:
        """
        Computes rolling polynomial n-gram hash values modulo table_size.

        Args:
            tokens (Tensor): Discrete phoneme IDs [B, T].
            n (int): N-gram order (e.g. 2, 3, or 4).

        Returns:
            Tensor: Hash indices [B, T] bounded in [0..table_size - 1].
        """
        b_sz, seq_len = tokens.shape
        prime: int = int(self.hash_primes[0].item())

        # Pad left by (n - 1) tokens with blank token (1)
        # Shape: [B, T + n - 1]
        padded: Tensor = F.pad(tokens, (n - 1, 0), mode="constant", value=1)

        # Accumulate polynomial hash: sum_{i=0}^{n-1} token_{t-i} * prime^i
        # Shape: [B, T], Type: torch.long
        hash_val: Tensor = torch.zeros((b_sz, seq_len), dtype=torch.long, device=tokens.device)
        mult = 1
        for i in range(n):
            # Window slice: [B, T]
            window_slice = padded[:, (n - 1 - i) : (n - 1 - i + seq_len)].long()
            hash_val = (hash_val + window_slice * mult) % self.table_size
            mult = (mult * prime) % self.table_size

        return hash_val

    def forward(self, x: Tensor, draft_tokens: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass retrieving and injecting engram phonotactic memory.

        Args:
            x (Tensor): Conformer hidden states at current layer.
                        Shape: [B, T, D].
                        Type: torch.float32.
                        Role: Current acoustic-contextual representations.
            draft_tokens (Tensor): Intermediate draft phoneme predictions.
                        Shape: [B, T].
                        Type: torch.long.
                        Role: Early phonetic hypothesis from Layer 4 draft head.

        Returns:
            Tuple[Tensor, Tensor]:
                - out: Memory-conditioned representations [B, T, D].
                - gate: Dynamic gating values [B, T, 1].
        """
        # Retrieve memory vectors for each n-gram order
        retrieved_memories: List[Tensor] = []

        for n in self.ngram_orders:
            # 1. Compute rolling hash indices: [B, T]
            hashes: Tensor = self._compute_ngram_hashes(draft_tokens, n)

            # 2. Look up continuous memory patterns: [B, T, memory_dim]
            mem_table: nn.Embedding = self.memory_tables[f"order_{n}"]
            mem_vecs: Tensor = mem_table(hashes)
            retrieved_memories.append(mem_vecs)

        # Concatenate multi-order memories: [B, T, len(orders) * memory_dim]
        concat_mem: Tensor = torch.cat(retrieved_memories, dim=-1)

        # Project memory into model dimension D: [B, T, D]
        projected_mem: Tensor = self.memory_proj(concat_mem)

        # Compute dynamic gate score: [B, T, 1] in range (0..1)
        gate: Tensor = torch.sigmoid(self.gate_proj(x + projected_mem))

        # Gated residual addition: [B, T, D]
        gated_update: Tensor = gate * projected_mem
        out: Tensor = self.layer_norm(x + gated_update)

        return out, gate
