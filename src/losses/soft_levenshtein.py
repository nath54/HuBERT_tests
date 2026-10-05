"""Differentiable Soft-Levenshtein Loss for Character Decoding.

Computes a smooth, differentiable approximation of the Levenshtein (edit) distance
between predicted character probability distributions and target character sequences.

Uses softmin in the dynamic programming recursion:
    softmin_gamma(u_1, u_2, u_3) = -gamma * log(exp(-u_1/gamma) + exp(-u_2/gamma) + exp(-u_3/gamma))
implemented numerically stably via torch.logsumexp.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def softmin_gamma(x: torch.Tensor, gamma: float = 0.2, dim: int = 0) -> torch.Tensor:
    """Numerically stable softmin using logsumexp.
    
    softmin_gamma(x) = -gamma * logsumexp(-x / gamma)
    As gamma -> 0, converges to hard min.
    """
    return -gamma * torch.logsumexp(-x / gamma, dim=dim)


class SoftLevenshteinLoss(nn.Module):
    """Differentiable Soft-Levenshtein Edit Distance Loss.
    
    Args:
        gamma: Smoothing temperature for softmin (lower = closer to hard min, default: 0.2).
        cost_ins: Cost of an insertion operation (default: 1.0).
        cost_del: Cost of a deletion operation (default: 1.0).
        cost_mode: 'prob' for 2*(1 - P(target)), 'nll' for min(-log P(target), 2.0).
        normalize_by_len: Whether to divide total edit distance by target length (yields smooth CER).
        ignore_index: Index to ignore in target (padding, e.g. -100).
    """

    def __init__(
        self,
        gamma: float = 0.2,
        cost_ins: float = 1.0,
        cost_del: float = 1.0,
        cost_mode: str = "prob",
        normalize_by_len: bool = True,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = gamma
        self.cost_ins = cost_ins
        self.cost_del = cost_del
        self.cost_mode = cost_mode
        self.normalize_by_len = normalize_by_len
        self.ignore_index = ignore_index

    def forward(
        self,
        pred_logits: torch.Tensor,
        target_ids: torch.Tensor,
        pred_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Soft-Levenshtein loss.
        
        Args:
            pred_logits: [N, K, V] or [B, L, K, V] Logits over vocabulary for each character slot.
            target_ids: [N, K] or [B, L, K] Ground-truth character token IDs.
            pred_lengths: Optional [N] lengths of predicted sequences (defaults to target length if None).
            target_lengths: Optional [N] lengths of target sequences (computed from ignore_index if None).
            
        Returns:
            Scalar tensor containing mean Soft-Levenshtein loss across valid words.
        """
        if pred_logits.dim() == 4:
            B, L, K, V = pred_logits.shape
            pred_logits = pred_logits.reshape(B * L, K, V)
        if target_ids.dim() == 3:
            B, L, K = target_ids.shape
            target_ids = target_ids.reshape(B * L, K)

        N, K, V = pred_logits.shape
        device = pred_logits.device

        if N == 0 or K == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 1. Determine lengths
        if target_lengths is None:
            valid_targets = (target_ids != self.ignore_index) & (target_ids >= 0)
            t_lens = valid_targets.sum(dim=1)  # [N]
        else:
            t_lens = target_lengths

        if pred_lengths is None:
            # Default to target length clamped to max prediction slots K
            p_lens = torch.clamp(t_lens, max=K)
        else:
            p_lens = pred_lengths

        # Filter out empty words
        active_mask = (t_lens > 0) & (p_lens > 0)
        if not active_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        active_indices = torch.nonzero(active_mask, as_tuple=True)[0]
        sub_logits = pred_logits[active_indices]       # [N_act, K, V]
        sub_targets = target_ids[active_indices]       # [N_act, K]
        sub_t_lens = t_lens[active_indices]            # [N_act]
        sub_p_lens = p_lens[active_indices]            # [N_act]
        N_act = sub_logits.shape[0]

        max_p = int(sub_p_lens.max().item())
        max_t = int(sub_t_lens.max().item())

        if max_p == 0 or max_t == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Truncate to maximum active lengths
        sub_logits = sub_logits[:, :max_p, :]
        sub_targets = sub_targets[:, :max_t].clamp(min=0, max=V - 1)

        # 2. Compute substitution/emission cost matrix C [N_act, max_p, max_t]
        probs = F.softmax(sub_logits, dim=-1)

        tgt_expanded = sub_targets.unsqueeze(1).expand(-1, max_p, -1)
        tgt_probs = torch.gather(probs, dim=2, index=tgt_expanded)  # [N_act, max_p, max_t]

        if self.cost_mode == "prob":
            # Cost is 2.0 * (1 - P(target)) so cost in [0, 2]
            # When P=1, cost=0; when P=0, cost=2 (matching delete + insert)
            C = 2.0 * (1.0 - tgt_probs)
        else:
            eps = 1e-6
            C = torch.clamp(-torch.log(tgt_probs + eps), min=0.0, max=2.0)

        # 3. Dynamic Programming Table D: [N_act, max_p + 1, max_t + 1]
        D = torch.full(
            (N_act, max_p + 1, max_t + 1),
            float("inf"),
            dtype=pred_logits.dtype,
            device=device,
        )

        # Base conditions:
        D[:, 0, 0] = 0.0
        for i in range(1, max_p + 1):
            D[:, i, 0] = i * self.cost_del

        for j in range(1, max_t + 1):
            D[:, 0, j] = j * self.cost_ins

        # DP recurrence with softmin
        for i in range(1, max_p + 1):
            for j in range(1, max_t + 1):
                c_del = D[:, i - 1, j] + self.cost_del
                c_ins = D[:, i, j - 1] + self.cost_ins
                c_sub = D[:, i - 1, j - 1] + C[:, i - 1, j - 1]

                stacked = torch.stack([c_del, c_ins, c_sub], dim=0)  # [3, N_act]
                D[:, i, j] = softmin_gamma(stacked, gamma=self.gamma, dim=0)

        # 4. Extract final cost D[b, p_len[b], t_len[b]]
        b_idx = torch.arange(N_act, device=device)
        final_costs = D[b_idx, sub_p_lens, sub_t_lens]  # [N_act]

        # Clamp tiny negative values caused by softmin logsumexp
        final_costs = torch.clamp(final_costs, min=0.0)

        if self.normalize_by_len:
            denom = torch.clamp(sub_t_lens.to(final_costs.dtype), min=1.0)
            word_loss = final_costs / denom
        else:
            word_loss = final_costs

        return word_loss.mean()
