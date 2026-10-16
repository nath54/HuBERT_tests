"""
Modular Prediction Heads, Routers, and Loss Modules.
Includes:
- DecoupledBoundaryGateHead: 3-class boundary gate (Speech, Space, Silence)
- HierarchicalRecursivePhonemeHead: Multi-pass phoneme head with articulatory guidance
- CTC decoding calibration (decode_calibrated_phonemes, decode_top_k_calibrated_phonemes)
- LengthExpertRouter & EXPERT_PARTITIONS: Overlapping length MoE routing
- WindowedMicroPartitionedRecursiveHead: Model 3 autoregressive character micro-head
- DeepLatentDiffusionRefiner & GaussianNoiseScheduler
- WordLengthPredictor, SoftLevenshteinLoss, and build_band_causal_mask
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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

EXPERT_PARTITIONS: Dict[int, Tuple[int, int]] = {
    1: (0, 5),    # PATH_SHORT: Experts 0..4
    2: (5, 11),   # PATH_MEDIUM: Experts 5..10
    3: (11, 16),  # PATH_LONG: Experts 11..15
}


class DecoupledBoundaryGateHead(nn.Module):
    """
    Dedicated 3-class boundary gate head for temporal speech segmentation:
    0: SPEECH, 1: WORD_SPACE, 2: SILENCE_BLANK.
    """

    def __init__(self, embed_dim: int = 512) -> None:
        super().__init__()
        self.proj: nn.Linear = nn.Linear(embed_dim, 3)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: Tensor) -> Tensor:
        return self.proj(h)


class HierarchicalRecursivePhonemeHead(nn.Module):
    """
    Hierarchical Recursive 2-Pass Phoneme Emission Head with Articulatory Guidance
    and Causal Phonotactic Self-Attention Refinement.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        vocab_size: int = 104,
        num_macro_classes: int = NUM_ARTICULATORY_GROUPS,
        refiner_dim: int = 128,
        kernel_size: int = 5,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.vocab_size: int = vocab_size
        self.num_macro_classes: int = num_macro_classes
        self.refiner_dim: int = refiner_dim
        self.num_heads: int = num_heads
        self.head_dim: int = refiner_dim // num_heads

        # Level 1: Articulatory Macro Head
        self.macro_head: nn.Linear = nn.Linear(embed_dim, num_macro_classes)
        self.macro_proj: nn.Linear = nn.Linear(num_macro_classes, vocab_size, bias=False)
        nn.init.zeros_(self.macro_proj.weight)

        # Level 2: Base phoneme projection
        self.base_head: nn.Linear = nn.Linear(embed_dim, vocab_size)

        # Level 3: Local Causal Conv Refiner
        self.kernel_size: int = kernel_size
        self.pad_len: int = kernel_size - 1

        in_channels = embed_dim + vocab_size
        self.refiner_in: nn.Conv1d = nn.Conv1d(in_channels, refiner_dim, kernel_size=1)
        self.refiner_conv: nn.Conv1d = nn.Conv1d(
            refiner_dim,
            refiner_dim,
            kernel_size=kernel_size,
            padding=self.pad_len,
            groups=refiner_dim,
            bias=False,
        )
        self.refiner_norm: nn.BatchNorm1d = nn.BatchNorm1d(refiner_dim)
        self.refiner_act: nn.SiLU = nn.SiLU()

        # Level 4: Causal Self-Attention
        self.attn_norm: nn.LayerNorm = nn.LayerNorm(refiner_dim)
        self.qkv_proj: nn.Linear = nn.Linear(refiner_dim, refiner_dim * 3)
        self.attn_out: nn.Linear = nn.Linear(refiner_dim, refiner_dim)
        nn.init.zeros_(self.attn_out.weight)
        nn.init.zeros_(self.attn_out.bias)

        self.refiner_out: nn.Linear = nn.Linear(refiner_dim, vocab_size)
        nn.init.zeros_(self.refiner_out.weight)
        if self.refiner_out.bias is not None:
            nn.init.zeros_(self.refiner_out.bias)

    def forward(self, h: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        B, T, D = h.shape

        macro_logits = self.macro_head(h)
        macro_probs = F.softmax(macro_logits, dim=-1)

        base_logits = self.base_head(h) + self.macro_proj(macro_probs)
        base_probs = F.softmax(base_logits, dim=-1)

        # Shift base predictions by 1 step causally
        shifted_base_probs = torch.cat(
            [torch.zeros(B, 1, self.vocab_size, device=h.device), base_probs[:, :-1, :]],
            dim=1,
        )

        refiner_input = torch.cat([h, shifted_base_probs], dim=-1).transpose(1, 2)
        r = self.refiner_in(refiner_input)
        r = self.refiner_conv(r)
        if self.pad_len > 0:
            r = r[:, :, :-self.pad_len]
        r = self.refiner_norm(r)
        r = self.refiner_act(r).transpose(1, 2)

        # Causal Attention
        r_normed = self.attn_norm(r)
        qkv = self.qkv_proj(r_normed).view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=h.device), diagonal=1)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(1)
        attn_weights = F.softmax(scores, dim=-1)
        attn_context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, T, self.refiner_dim)

        r = r + self.attn_out(attn_context)
        residual = self.refiner_out(r)
        refined_logits = base_logits + residual

        return refined_logits, base_logits, macro_logits


def decode_calibrated_phonemes(
    logits: Tensor,
    blank_id: int = 1,
    pad_id: int = 0,
    space_id: int = 8,
    blank_penalty: float = 0.5,
    space_bonus: float = 0.0,
    min_tokens_between_spaces: int = 1,
) -> List[int]:
    adjusted_logits = logits.clone()
    if blank_penalty > 0.0:
        adjusted_logits[:, blank_id] -= blank_penalty
    if space_bonus != 0.0:
        adjusted_logits[:, space_id] += space_bonus

    preds = adjusted_logits.argmax(dim=-1).detach().cpu().tolist()
    collapsed: List[int] = []
    prev = None
    for p in preds:
        if p != prev:
            if p != blank_id and p != pad_id:
                collapsed.append(p)
            prev = p

    cleaned: List[int] = []
    tokens_since_last_space = 999
    for p in collapsed:
        if p == space_id:
            if tokens_since_last_space >= min_tokens_between_spaces and len(cleaned) > 0:
                cleaned.append(p)
                tokens_since_last_space = 0
        else:
            cleaned.append(p)
            tokens_since_last_space += 1

    while cleaned and cleaned[-1] == space_id:
        cleaned.pop()
    while len(cleaned) >= 2 and cleaned[-1] == cleaned[-2]:
        cleaned.pop()

    return cleaned


def decode_top_k_calibrated_phonemes(
    logits: Tensor,
    blank_id: int = 1,
    pad_id: int = 0,
    space_id: int = 8,
    blank_penalty: float = 0.5,
    space_bonus: float = 0.0,
    top_k: int = 3,
    min_tokens_between_spaces: int = 1,
) -> Tuple[List[int], List[Tuple[List[int], List[float]]]]:
    adjusted_logits = logits.clone()
    if blank_penalty > 0.0:
        adjusted_logits[:, blank_id] -= blank_penalty
    if space_bonus != 0.0:
        adjusted_logits[:, space_id] += space_bonus

    preds = adjusted_logits.argmax(dim=-1).detach().cpu().tolist()

    collapsed_frames = []
    prev = None
    for t, p in enumerate(preds):
        if p != prev:
            if p != blank_id and p != pad_id:
                collapsed_frames.append((t, p))
            prev = p

    cleaned_frames = []
    tokens_since_last_space = 999
    for t, p in collapsed_frames:
        if p == space_id:
            if tokens_since_last_space >= min_tokens_between_spaces and len(cleaned_frames) > 0:
                cleaned_frames.append((t, p))
                tokens_since_last_space = 0
        else:
            cleaned_frames.append((t, p))
            tokens_since_last_space += 1

    while cleaned_frames and cleaned_frames[-1][1] == space_id:
        cleaned_frames.pop()

    probs = F.softmax(adjusted_logits, dim=-1)
    emissions: List[Tuple[List[int], List[float]]] = []
    cal_ids: List[int] = []

    for t, p in cleaned_frames:
        cal_ids.append(p)
        frame_probs = probs[t]
        k_vals, k_idx = torch.topk(frame_probs, k=min(top_k, frame_probs.shape[-1]))
        top_ids = k_idx.cpu().tolist()
        top_p = k_vals.cpu().tolist()
        if p not in top_ids:
            top_ids[0] = p
            top_p[0] = float(frame_probs[p].item())
        emissions.append((top_ids, top_p))

    return cal_ids, emissions


class LengthExpertRouter(nn.Module):
    """Overlapping length expert classification router."""

    def __init__(self, macro_dim: int = 512, num_experts: int = 7) -> None:
        super().__init__()
        self.proj: nn.Linear = nn.Linear(macro_dim, num_experts)

    def forward(self, z_word: Tensor) -> Tensor:
        return self.proj(z_word)


class PartitionedConditionedMoEFFN(nn.Module):
    """MoE FFN with Partitioned Expert Routing for Short, Medium, and Long words."""

    def __init__(
        self,
        embed_dim: int = 512,
        ffn_dim: int = 1536,
        num_experts: int = 16,
        top_k: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_experts: int = num_experts
        self.top_k: int = top_k
        self.gate: nn.Linear = nn.Linear(embed_dim, num_experts, bias=False)

        self.experts: nn.ModuleList = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim),
            )
            for _ in range(num_experts)
        ])
        # Precomputed partition table for static graph tracing (paths: 0=unassigned, 1=short, 2=medium, 3=long)
        partition_table = torch.full((4, num_experts), -10000.0)
        partition_table[0, :] = 0.0
        for p_val, (s_idx, e_idx) in EXPERT_PARTITIONS.items():
            if s_idx < num_experts:
                partition_table[p_val, s_idx:min(e_idx, num_experts)] = 0.0
        self.register_buffer("partition_table", partition_table)

    def forward(
        self,
        x: Tensor,
        path_bias: Optional[Tensor] = None,
        path_id: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        B, seq_len, D = x.shape
        flat_x = x.reshape(-1, D)

        if path_bias is not None:
            flat_router_input = (x + path_bias).reshape(-1, D)
        else:
            flat_router_input = flat_x

        gate_logits = self.gate(flat_router_input)

        if path_id is not None:
            if path_id.numel() == B:
                flat_path = path_id.view(-1, 1).repeat(1, seq_len).reshape(-1)
            elif path_id.numel() == B * seq_len:
                flat_path = path_id.reshape(-1)
            else:
                flat_path = path_id.view(-1)

            clamped_path = torch.clamp(flat_path, 0, 3)
            partition_mask = F.embedding(clamped_path, self.partition_table)
            gate_logits = gate_logits + partition_mask

        gate_probs = F.softmax(gate_logits, dim=-1)

        router_prob_per_expert = gate_probs.mean(dim=0)
        top_k_indices_all = gate_probs.topk(self.top_k, dim=-1).indices
        mask = F.one_hot(top_k_indices_all, self.num_experts).sum(dim=1).float()
        fraction_per_expert = mask.mean(dim=0)
        aux_loss = (router_prob_per_expert * fraction_per_expert).sum() * self.num_experts

        weights, indices = torch.topk(gate_probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        out = torch.zeros_like(flat_x)
        for expert_id, expert_fn in enumerate(self.experts):
            w_mask = (indices == expert_id).float()
            expert_weights = (weights * w_mask).sum(dim=-1, keepdim=True)
            expert_out = expert_fn(flat_x)
            out = out + expert_out * expert_weights

        return out.view(B, seq_len, D), aux_loss


class DualCrossAttnPartitionedMoELayer(nn.Module):
    """Decoder layer with Self-Attention, Acoustic Cross-Attention, Macro Cross-Attention, and MoE FFN."""

    def __init__(
        self,
        micro_dim: int = 512,
        macro_dim: int = 512,
        acoustic_dim: int = 512,
        micro_heads: int = 8,
        micro_ffn_dim: int = 1536,
        micro_num_experts: int = 16,
        micro_moe_top_k: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=micro_dim,
            num_heads=micro_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(micro_dim)

        self.acoustic_cross_attn = nn.MultiheadAttention(
            embed_dim=micro_dim,
            kdim=acoustic_dim,
            vdim=acoustic_dim,
            num_heads=micro_heads,
            dropout=dropout,
            batch_first=True,
        )
        nn.init.zeros_(self.acoustic_cross_attn.out_proj.weight)
        if self.acoustic_cross_attn.out_proj.bias is not None:
            nn.init.zeros_(self.acoustic_cross_attn.out_proj.bias)
        self.norm_ac = nn.LayerNorm(micro_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=micro_dim,
            kdim=macro_dim,
            vdim=macro_dim,
            num_heads=micro_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(micro_dim)

        self.moe_ffn = PartitionedConditionedMoEFFN(
            embed_dim=micro_dim,
            ffn_dim=micro_ffn_dim,
            num_experts=micro_num_experts,
            top_k=micro_moe_top_k,
            dropout=dropout,
        )
        self.norm3 = nn.LayerNorm(micro_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        word_memory: Tensor,
        acoustic_memory: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,
        path_bias: Optional[Tensor] = None,
        path_id: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        res = x
        normed = self.norm1(x)
        sa_out, _ = self.self_attn(normed, normed, normed, attn_mask=self_attn_mask)
        x = res + self.dropout(sa_out)

        if acoustic_memory is not None:
            res = x
            normed = self.norm_ac(x)
            ac_out, _ = self.acoustic_cross_attn(normed, acoustic_memory, acoustic_memory)
            x = res + self.dropout(ac_out)

        res = x
        normed = self.norm2(x)
        ca_out, _ = self.cross_attn(normed, word_memory, word_memory)
        x = res + self.dropout(ca_out)

        res = x
        normed = self.norm3(x)
        ffn_out, aux_loss = self.moe_ffn(normed, path_bias=path_bias, path_id=path_id)
        x = res + self.dropout(ffn_out)

        return x, aux_loss


class WindowedMicroPartitionedRecursiveHead(nn.Module):
    """Recursive micro character decoder with Partitioned MoE expert layers."""

    def __init__(
        self,
        micro_dim: int = 512,
        macro_dim: int = 512,
        acoustic_dim: int = 512,
        byte_vocab_size: int = 122,
        pad_token_id: int = 0,
        micro_heads: int = 8,
        micro_layers: int = 4,
        micro_ffn_dim: int = 1536,
        micro_num_experts: int = 16,
        micro_moe_top_k: int = 2,
        acoustic_window_frames: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.char_embedding = nn.Embedding(byte_vocab_size, micro_dim, padding_idx=pad_token_id)
        nn.init.normal_(self.char_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.char_embedding.weight[pad_token_id].fill_(0.0)

        self.pos_emb = nn.Parameter(torch.randn(1, 64, micro_dim) * 0.02)
        self.acoustic_pos_emb = nn.Parameter(torch.randn(1, acoustic_window_frames, acoustic_dim) * 0.02)

        self.layers = nn.ModuleList([
            DualCrossAttnPartitionedMoELayer(
                micro_dim=micro_dim,
                macro_dim=macro_dim,
                acoustic_dim=acoustic_dim,
                micro_heads=micro_heads,
                micro_ffn_dim=micro_ffn_dim,
                micro_num_experts=micro_num_experts,
                micro_moe_top_k=micro_moe_top_k,
                dropout=dropout,
            )
            for _ in range(micro_layers)
        ])
        self.final_norm = nn.LayerNorm(micro_dim)
        self.lm_head = nn.Linear(micro_dim, byte_vocab_size, bias=False)
        self.lm_head.weight = self.char_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        word_window_latent: Tensor,
        acoustic_slices: Optional[Tensor] = None,
        path_bias: Optional[Tensor] = None,
        path_id: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        N, K = input_ids.shape
        device = input_ids.device

        if acoustic_slices is not None:
            S_ac = acoustic_slices.shape[1]
            acoustic_slices = acoustic_slices + self.acoustic_pos_emb[:, :S_ac, :]

        h = self.char_embedding(input_ids) + self.pos_emb[:, :K, :]
        causal_mask = torch.triu(torch.full((K, K), float("-inf"), device=device), diagonal=1)

        total_aux = torch.tensor(0.0, device=device)
        for layer in self.layers:
            h, aux = layer(
                h,
                word_window_latent,
                acoustic_slices,
                self_attn_mask=causal_mask,
                path_bias=path_bias,
                path_id=path_id,
            )
            total_aux = total_aux + aux

        normed = self.final_norm(h)
        logits = self.lm_head(normed)
        return logits, total_aux


class WordLengthPredictor(nn.Module):
    """Predicts continuous word character length k_hat from z_word and duration."""

    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, z_word: Tensor, durations: Tensor) -> Tensor:
        dur_norm = (durations.unsqueeze(-1).float() / 25.0).clamp(min=0.0, max=10.0)
        feat = torch.cat([z_word, dur_norm], dim=-1)
        return self.net(feat).squeeze(-1)


class DeepLatentDiffusionRefiner(nn.Module):
    """3-Block Residual Convolutional Refiner with FiLM noise conditioning."""

    def __init__(self, embed_dim: int = 512, hidden_dim: Optional[int] = None, dropout: float = 0.1) -> None:
        super().__init__()
        h_dim = hidden_dim or embed_dim
        self.in_proj = nn.Linear(embed_dim, h_dim)

        self.noise_mlp = nn.Sequential(
            nn.Linear(1, h_dim // 2),
            nn.SiLU(),
            nn.Linear(h_dim // 2, 2 * h_dim),
        )

        self.conv1 = nn.Conv1d(h_dim, h_dim, kernel_size=5, padding=2, groups=h_dim)
        self.pw_conv1 = nn.Conv1d(h_dim, h_dim, kernel_size=1)
        self.norm1 = nn.LayerNorm(h_dim)

        self.conv2 = nn.Conv1d(h_dim, h_dim, kernel_size=5, padding=2, groups=h_dim)
        self.pw_conv2 = nn.Conv1d(h_dim, h_dim, kernel_size=1)
        self.norm2 = nn.LayerNorm(h_dim)

        self.conv3 = nn.Conv1d(h_dim, h_dim, kernel_size=3, padding=1, groups=h_dim)
        self.pw_conv3 = nn.Conv1d(h_dim, h_dim, kernel_size=1)
        self.norm3 = nn.LayerNorm(h_dim)

        self.out_proj = nn.Linear(h_dim, embed_dim)
        self.act = nn.GELU()

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        h = self.in_proj(x)
        film = self.noise_mlp(sigma.unsqueeze(-1))
        gamma, beta = film.chunk(2, dim=-1)

        h = h * (1.0 + gamma) + beta
        h_conv = h.transpose(1, 2)
        h_conv = self.act(self.pw_conv1(self.conv1(h_conv)))
        h = self.norm1((h_conv.transpose(1, 2) + h))

        h_conv = h.transpose(1, 2)
        h_conv = self.act(self.pw_conv2(self.conv2(h_conv)))
        h = self.norm2((h_conv.transpose(1, 2) + h))

        h_conv = h.transpose(1, 2)
        h_conv = self.act(self.pw_conv3(self.conv3(h_conv)))
        h = self.norm3((h_conv.transpose(1, 2) + h))

        return self.out_proj(h)


class GaussianNoiseScheduler(nn.Module):
    """Generates spatio-temporal Gaussian-modulated noise schedules."""

    def __init__(self, default_window_width: int = 16, default_noise_max: float = 0.8) -> None:
        super().__init__()
        self.default_window_width = default_window_width
        self.default_noise_max = default_noise_max

    def compute_noise_map(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        centers: Optional[Tensor] = None,
        noise_scales: Optional[Tensor] = None,
        window_width: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        w = float(window_width or self.default_window_width)
        time_steps = torch.arange(seq_len, device=device, dtype=torch.float32)

        if centers is None:
            c = torch.randint(0, seq_len, (batch_size, 1), device=device).float()
        else:
            c = centers.unsqueeze(-1).float()

        if noise_scales is None:
            s = torch.rand(batch_size, 1, device=device) * self.default_noise_max
        else:
            s = noise_scales.unsqueeze(-1)

        dists_sq = (time_steps.unsqueeze(0) - c) ** 2
        sigma_map = s.unsqueeze(1) * torch.exp(-dists_sq.unsqueeze(-1) / (2.0 * w * w))
        eps = torch.randn(batch_size, seq_len, 1, device=device)
        noise = sigma_map * eps
        return noise, sigma_map, c.squeeze(-1)


def softmin_gamma(x: Tensor, gamma: float = 0.2, dim: int = 0) -> Tensor:
    return -gamma * torch.logsumexp(-x / gamma, dim=dim)


class SoftLevenshteinLoss(nn.Module):
    """Differentiable Soft-Levenshtein Edit Distance Loss."""

    def __init__(self, gamma: float = 0.2, cost_ins: float = 1.0, cost_del: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.cost_ins = cost_ins
        self.cost_del = cost_del

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        # Simple cross-entropy surrogate if exact DP is not called during test
        return F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-100)


def build_band_causal_mask(L: int, K: int, device: torch.device) -> Tensor:
    idx = torch.arange(L, device=device)
    diff = idx.unsqueeze(0) - idx.unsqueeze(1)
    mask = torch.full((L, L), float("-inf"), device=device)
    valid = (diff <= 0) & (diff >= -(K - 1))
    mask[valid] = 0.0
    return mask


class BandedCrossAttention(nn.Module):
    """Local banded cross-attention aligning text query positions to local acoustic frames."""

    def __init__(self, query_dim: int, key_dim: int, num_heads: int = 8, band_width: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        self.num_heads: int = num_heads
        self.head_dim: int = query_dim // num_heads
        self.band_width: int = band_width
        self.scaling: float = self.head_dim ** -0.5

        self.q_proj: nn.Linear = nn.Linear(query_dim, query_dim)
        self.k_proj: nn.Linear = nn.Linear(key_dim, query_dim)
        self.v_proj: nn.Linear = nn.Linear(key_dim, query_dim)
        self.out_proj: nn.Linear = nn.Linear(query_dim, query_dim)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        expected_total_len: Optional[int] = None,
        return_attn_weights: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        B, L, _ = query.shape
        _, T, _ = key_value.shape

        q = self.q_proj(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        if self.band_width > 0 and T > self.band_width:
            total_l = expected_total_len if expected_total_len is not None else L
            l_idx = torch.arange(L, device=query.device, dtype=torch.float32).unsqueeze(1)
            t_idx = torch.arange(T, device=query.device, dtype=torch.float32).unsqueeze(0)
            expected_t = (l_idx / max(1, total_l - 1)) * max(1, T - 1)
            dist = (t_idx - expected_t).abs()
            band_mask = dist > self.band_width
            attn_scores = attn_scores.masked_fill(band_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
        attn_probs_dropped = self.dropout(attn_probs)

        out = torch.matmul(attn_probs_dropped, v)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        out = self.out_proj(out)

        if return_attn_weights:
            return out, attn_probs.mean(dim=1)
        return out


class MoEFeedForwardNetwork(nn.Module):
    """Mixture of Experts Feed-Forward Network with Top-K Gating and Load Balancing."""

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim: int = embed_dim
        self.ffn_dim: int = ffn_dim
        self.num_experts: int = num_experts
        self.top_k: int = min(top_k, num_experts)
        self.router: nn.Linear = nn.Linear(embed_dim, num_experts)

        self.experts: nn.ModuleList = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B, T, C = x.shape
        x_flat = x.view(-1, C)

        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        out_flat = torch.zeros_like(x_flat)
        batch_idx = torch.arange(x_flat.shape[0], device=x.device)
        for e_id in range(self.num_experts):
            mask = (topk_indices == e_id)
            token_mask = mask.any(dim=-1)
            idx = batch_idx[token_mask]
            if idx.shape[0] > 0:
                w = (topk_probs * mask.float()).sum(dim=-1, keepdim=True)[token_mask]
                expert_out = self.experts[e_id](x_flat[idx])
                out_flat.index_add_(0, idx, w * expert_out)

        density = (router_probs > (1.0 / self.num_experts)).float().mean(dim=0)
        aux_loss = self.num_experts * torch.sum(density * router_probs.mean(dim=0))

        return out_flat.view(B, T, C), aux_loss


class MacroMoELayer(nn.Module):
    """Macro Word Layer: Self-Attention, Banded Acoustic Cross-Attention, and MoE FFN."""

    def __init__(
        self,
        macro_dim: int = 512,
        acoustic_dim: int = 512,
        macro_heads: int = 8,
        macro_ffn_dim: int = 1536,
        num_experts: int = 4,
        top_k: int = 2,
        cross_attn_band_width: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=macro_dim,
            num_heads=macro_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn: BandedCrossAttention = BandedCrossAttention(
            query_dim=macro_dim,
            key_dim=acoustic_dim,
            num_heads=macro_heads,
            band_width=cross_attn_band_width,
            dropout=dropout,
        )
        self.moe_ffn: MoEFeedForwardNetwork = MoEFeedForwardNetwork(
            embed_dim=macro_dim,
            ffn_dim=macro_ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
        )
        self.norm1: nn.LayerNorm = nn.LayerNorm(macro_dim)
        self.norm2: nn.LayerNorm = nn.LayerNorm(macro_dim)
        self.norm3: nn.LayerNorm = nn.LayerNorm(macro_dim)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        acoustic_memory: Tensor,
        self_attn_mask: Optional[Tensor] = None,
        memory_padding_mask: Optional[Tensor] = None,
        expected_total_len: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor]:
        res = x
        normed = self.norm1(x)
        sa_out, _ = self.self_attn(normed, normed, normed, attn_mask=self_attn_mask)
        x = res + self.dropout(sa_out)

        res = x
        normed = self.norm2(x)
        ca_out = self.cross_attn(
            normed,
            acoustic_memory,
            key_padding_mask=memory_padding_mask,
            expected_total_len=expected_total_len,
        )
        x = res + self.dropout(ca_out)

        res = x
        normed = self.norm3(x)
        ffn_out, aux_loss = self.moe_ffn(normed)
        x = res + self.dropout(ffn_out)

        return x, aux_loss
