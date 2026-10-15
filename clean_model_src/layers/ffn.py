"""
Modular Feed-Forward and Mixture-of-Experts (MoE) Layers.
Includes:
- FeedForwardNetwork: Standard 2-layer Transformer FFN
- DeepSeekMoEFFN: 1 Shared Expert + N Routed Experts with Top-K Softmax Gating & Load Balancing
"""

# Import Modules
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):
    """
    Standard 2-Layer Feed-Forward Network with GeLU/SiLU activation.
    Expands channel dimension D -> ffn_dim -> D.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = "silu",
    ) -> None:
        """
        Initialize FeedForwardNetwork.

        Args:
            embed_dim (int): Input / output channel dimension D (512).
            ffn_dim (int): Inner hidden expansion dimension (2048).
            dropout (float): Dropout probability.
            activation (str): Activation function ('silu', 'gelu', 'relu').
        """
        # Parent initialization
        super().__init__()
        self.embed_dim: int = embed_dim
        self.ffn_dim: int = ffn_dim

        # Up-projection: [B, T, D] -> [B, T, ffn_dim]
        self.fc1: nn.Linear = nn.Linear(embed_dim, ffn_dim, bias=True)

        # Activation
        if activation.lower() == "silu":
            self.act: nn.Module = nn.SiLU()
        elif activation.lower() == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()

        self.dropout: nn.Dropout = nn.Dropout(dropout)

        # Down-projection: [B, T, ffn_dim] -> [B, T, D]
        self.fc2: nn.Linear = nn.Linear(ffn_dim, embed_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through FFN.

        Args:
            x (Tensor): Input tensor of shape [B, T, D].
                        Type: torch.float32.
                        Role: Representations entering feed-forward transformation.

        Returns:
            Tensor: Output tensor of shape [B, T, D].
        """
        # Linear 1: [B, T, D] -> [B, T, ffn_dim]
        h: Tensor = self.fc1(x)

        # Non-linear activation: [B, T, ffn_dim]
        h = self.act(h)

        # Dropout: [B, T, ffn_dim]
        h = self.dropout(h)

        # Linear 2: [B, T, ffn_dim] -> [B, T, D]
        out: Tensor = self.fc2(h)
        return out


class DeepSeekMoEFFN(nn.Module):
    """
    DeepSeek-style Mixture-of-Experts Feed-Forward Network (DeepSeek-AI, 2024).
    Features:
    1. Shared Expert (num_shared=1): Dedicated expert executed for 100% of tokens,
       capturing language-invariant acoustic features and common phonetic structure.
    2. Routed Experts (num_routed=8): Specialized experts conditionally activated
       by a learned gating router with Top-K selection (k=2).
    3. Anti-Collapse Load Balancing Loss: Strictly prevents router collapse where
       only 1 or 2 dominant experts receive all gradient flow.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        ffn_dim: int = 2048,
        num_routed_experts: int = 8,
        top_k: int = 2,
        num_shared_experts: int = 1,
        dropout: float = 0.1,
        load_balance_weight: float = 0.01,
        router_jitter_std: float = 0.02,
    ) -> None:
        """
        Initialize DeepSeekMoEFFN.

        Args:
            embed_dim (int): Model hidden dimension D (512).
            ffn_dim (int): Inner hidden dimension per expert (2048).
            num_routed_experts (int): Number of routed experts E (8).
            top_k (int): Number of routed experts selected per token (2).
            num_shared_experts (int): Number of shared experts S (1).
            dropout (float): Dropout probability.
            load_balance_weight (float): Auxiliary loss weight for router balance.
            router_jitter_std (float): Standard deviation of Gaussian jitter during training.
        """
        # Parent initialization
        super().__init__()
        self.embed_dim: int = embed_dim
        self.ffn_dim: int = ffn_dim
        self.num_routed_experts: int = num_routed_experts
        self.top_k: int = top_k
        self.num_shared_experts: int = num_shared_experts
        self.load_balance_weight: float = load_balance_weight
        self.router_jitter_std: float = router_jitter_std

        # 1. Shared Expert (always active)
        if num_shared_experts > 0:
            self.shared_expert: Optional[FeedForwardNetwork] = FeedForwardNetwork(
                embed_dim=embed_dim,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
        else:
            self.shared_expert = None

        # 2. Routed Experts (ModuleList of N independent FFNs)
        self.routed_experts: nn.ModuleList = nn.ModuleList([
            FeedForwardNetwork(
                embed_dim=embed_dim,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_routed_experts)
        ])

        # 3. Router Gating Head: projects D -> E expert logits
        # Shape: [D, E], Type: torch.float32, Role: expert dispatch scores
        self.router: nn.Linear = nn.Linear(embed_dim, num_routed_experts, bias=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass executing DeepSeek Mixture of Experts.

        Args:
            x (Tensor): Input tensor.
                        Shape: [B, T, D].
                        Type: torch.float32.
                        Role: Hidden representations to be routed to specialized experts.

        Returns:
            Tuple[Tensor, Tensor]:
                - out: Combined output tensor [B, T, D].
                - aux_loss: Scalar auxiliary load balancing loss [1].
        """
        # Dimensions
        b_sz, seq_len, dim = x.shape
        # Flatten batch and time: [B * T, D]
        flat_x: Tensor = x.view(-1, dim)
        n_tokens: int = flat_x.shape[0]

        # 1. Compute router logits: [B * T, E]
        router_logits: Tensor = self.router(flat_x)

        # 2. Add exploration jitter noise during training
        if self.training and self.router_jitter_std > 0.0:
            noise: Tensor = torch.randn_like(router_logits) * self.router_jitter_std
            routed_logits: Tensor = router_logits + noise
        else:
            routed_logits = router_logits

        # 3. Softmax probabilities over all E routed experts: [B * T, E]
        routing_probs: Tensor = F.softmax(routed_logits, dim=-1)

        # 4. Top-K Expert Selection: indices [B * T, top_k], weights [B * T, top_k]
        topk_weights, topk_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        # Normalize top-k weights to sum to 1.0 across selected experts
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Store routing decisions for model analysis and verbose logging
        self.last_routing: Dict[str, Tensor] = {
            "indices": topk_indices.detach(),
            "weights": topk_weights.detach(),
            "probs": routing_probs.detach(),
        }

        # 5. Compute Anti-Collapse Load Balancing Loss
        # f_i: fraction of tokens routed to expert i
        # P_i: average routing probability for expert i
        if self.training:
            # Token dispatch frequency: [E]
            tokens_per_expert = torch.zeros(self.num_routed_experts, device=x.device)
            flat_indices = topk_indices.view(-1)
            tokens_per_expert.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float32))
            f_i = tokens_per_expert / (n_tokens * self.top_k)

            # Average router probability: [E]
            p_i = routing_probs.mean(dim=0)

            # Switch-transformer load balance formula: N * sum(f_i * P_i)
            aux_loss = self.num_routed_experts * (f_i * p_i).sum() * self.load_balance_weight
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        # 6. Execute Routed Experts
        # Output buffer: [B * T, D], Type: torch.float32
        routed_out: Tensor = torch.zeros_like(flat_x)

        if torch.jit.is_tracing() or not self.training:
            # Static trace-safe dispatch avoiding dynamic tensor slicing and boolean guards
            for exp_idx, expert_module in enumerate(self.routed_experts):
                token_mask: Tensor = (topk_indices == exp_idx).float()
                expert_weight: Tensor = (topk_weights * token_mask).sum(dim=-1, keepdim=True)
                routed_out = routed_out + expert_module(flat_x) * expert_weight
        else:
            # Sparse training dispatch
            for exp_idx, expert_module in enumerate(self.routed_experts):
                token_mask_bool: Tensor = (topk_indices == exp_idx)
                token_assigned: Tensor = token_mask_bool.any(dim=-1)

                if token_assigned.any():
                    exp_in: Tensor = flat_x[token_assigned]
                    exp_out: Tensor = expert_module(exp_in)
                    weight_val: Tensor = (topk_weights * token_mask_bool.float()).sum(dim=-1)[token_assigned].unsqueeze(-1)
                    routed_out[token_assigned] += exp_out * weight_val

        # 7. Execute Shared Expert (if configured)
        if self.shared_expert is not None:
            # Dimension: [B * T, D], Type: torch.float32
            shared_out: Tensor = self.shared_expert(flat_x)
            # Combine shared + routed experts
            combined_flat: Tensor = shared_out + routed_out
        else:
            combined_flat = routed_out

        # Reshape back to [B, T, D]
        out: Tensor = combined_flat.view(b_sz, seq_len, dim)
        return out, aux_loss
