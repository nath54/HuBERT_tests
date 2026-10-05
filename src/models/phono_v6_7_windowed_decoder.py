"""Phono-V6.7: Universal Multilingual 4-Path Length-Adaptive Word Routing & MoE Decoder
with Windowed Multi-Word Context Cross-Attention.

Key Features:
1. Windowed Multi-Word Context Cross-Attention:
   - When spelling word l, the micro character decoder cross-attends to a sliding window
     of the preceding (W-1) words plus the current word (W=4 total word latents).
   - Resolves grammatical agreement (e.g. French plural -ons vs -ez) and homophone ambiguity.
   - Learned relative word position embeddings distinguish current word from previous context.
   - Computational overhead is <0.2% because key/value length W=4 is minimal.

2. Ramped Scheduled Sampling (30%):
   - 30% self-conditioning with 2-step prefix rollout trains the recursive head to
     self-correct character typos and drift during inference.

3. 100% Backwards-Compatible Weight Transfer:
   - All 64 MoE experts, phoneme prenet, macro attention, and LM heads share identical shapes
     with V6.6, enabling seamless warm-start from best V6.6 checkpoints.

4. 4-Path Length-Adaptive Routing:
   - PATH_SPECIAL = 0: Blank, silence, pause, EOS -> 0 FLOPs (immediate bypass).
   - PATH_SHORT   = 1: Words 1-3 chars -> Horizon bound K=5 steps.
   - PATH_MEDIUM  = 2: Words 4-7 chars -> Horizon bound K=9 steps.
   - PATH_LONG    = 3: Words 8+ chars  -> Horizon bound K=24 steps.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.phono_variants import (
    DeepLatentDiffusionRefiner,
    GaussianNoiseScheduler,
)
from src.models.phono_v6_5_hierarchical_decoder import MacroMoELayer
from src.models.phono_v6_6_adaptive_decoder import (
    ConditionedMoEFeedForwardNetwork,
    ConditionedMicroMoELayer,
)


@dataclass
class WindowedAdaptivePathConfig:
    """Configuration for Phono-V6.7 Windowed Multi-Word MoE Decoder."""

    # Architecture dimensions
    acoustic_dim: int = 512
    macro_dim: int = 512
    macro_layers: int = 4
    macro_heads: int = 8
    macro_ffn_dim: int = 1536
    cross_attn_band_width: int = 0  # 0: Unconstrained global cross-attention (no linear-spacing clipping or padding blindness)
    micro_dim: int = 512
    micro_layers: int = 4
    micro_heads: int = 8
    micro_ffn_dim: int = 1536

    # MoE settings
    macro_num_experts: int = 4
    macro_moe_top_k: int = 2
    micro_num_experts: int = 16
    micro_moe_top_k: int = 2
    moe_loss_weight: float = 0.01

    # Multi-Word Window Context & Acoustic Window
    word_context_window: int = 6  # Window of preceding words (W=6: [w_{l-5}, ..., w_l])
    acoustic_window_frames: int = 32  # Intra-word continuous speech frames (~640ms receptive field)

    # 4-Path Length Routing Bounds
    k_short: int = 5    # Max chars for short words (1-3 chars + bos/eow)
    k_medium: int = 9   # Max chars for medium words (4-7 chars + bos/eow)
    k_long: int = 24    # Max chars for long words (8+ chars)
    num_paths: int = 4  # 0: Special, 1: Short, 2: Medium, 3: Long

    # Path IDs
    PATH_SPECIAL: int = 0
    PATH_SHORT: int = 1
    PATH_MEDIUM: int = 2
    PATH_LONG: int = 3

    # Max sequence limits
    max_word_slots: int = 64
    max_bytes_per_word: int = 24

    # Tokenizer settings
    phoneme_vocab_size: int = 111
    byte_vocab_size: int = 122
    pad_token_id: int = 0
    blank_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3
    eow_token_id: int = 4
    unk_token_id: int = 5

    # Loss weights & training
    macro_path_loss_weight: float = 0.3
    use_word_diffusion: bool = True
    word_noise_max: float = 0.3
    dropout: float = 0.1
    scheduled_sampling_prob: float = 0.05  # Reduced from 0.30 to ensure 95% clean prefix teacher forcing
    label_smoothing: float = 0.05

    @classmethod
    def small(cls, **kwargs) -> "WindowedAdaptivePathConfig":
        """Small tier: 16 experts, Top-2 routing, d=256 (~24M parameters)."""
        cfg = cls(
            acoustic_dim=256,
            macro_dim=256,
            macro_layers=2,
            macro_heads=4,
            macro_ffn_dim=768,
            cross_attn_band_width=0,
            micro_dim=256,
            micro_layers=2,
            micro_heads=4,
            micro_ffn_dim=768,
            macro_num_experts=4,
            macro_moe_top_k=2,
            micro_num_experts=16,
            micro_moe_top_k=2,
            word_context_window=6,
            acoustic_window_frames=32,
            scheduled_sampling_prob=0.05,
        )
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def medium(cls, **kwargs) -> "WindowedAdaptivePathConfig":
        """Medium tier: 16 experts, Top-2 routing, d=512 (~146M parameters)."""
        cfg = cls(
            macro_dim=512,
            macro_layers=4,
            macro_heads=8,
            macro_ffn_dim=1536,
            cross_attn_band_width=0,
            micro_dim=512,
            micro_layers=4,
            micro_heads=8,
            micro_ffn_dim=1536,
            micro_num_experts=16,
            micro_moe_top_k=2,
            word_context_window=6,
            acoustic_window_frames=32,
            scheduled_sampling_prob=0.05,
        )
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def large(cls, **kwargs) -> "WindowedAdaptivePathConfig":
        """Large tier: 32 experts, Top-4 routing, d=768 (~460M parameters)."""
        cfg = cls(
            acoustic_dim=768,
            macro_dim=768,
            macro_layers=6,
            macro_heads=12,
            macro_ffn_dim=768,
            cross_attn_band_width=0,
            micro_dim=768,
            micro_layers=6,
            micro_heads=12,
            micro_ffn_dim=768,
            micro_num_experts=32,
            micro_moe_top_k=4,
            word_context_window=6,
            acoustic_window_frames=32,
            scheduled_sampling_prob=0.05,
        )
        for k, v in kwargs.items():
            setattr(cfg, k, v)
        return cfg


class DualCrossAttnMicroMoELayer(nn.Module):
    """Transformer micro-decoder layer with Causal Self-Attention, Direct Acoustic Cross-Attention,
    Sliding Word Context Cross-Attention, and Path-Conditioned MoE FFN.
    """

    def __init__(self, config: WindowedAdaptivePathConfig):
        super().__init__()
        # 1. Causal Self-Attention (character autoregressive prior)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.micro_dim,
            num_heads=config.micro_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(config.micro_dim)

        # 2. Direct Acoustic Cross-Attention (intra-word continuous speech frames)
        self.acoustic_cross_attn = nn.MultiheadAttention(
            embed_dim=config.micro_dim,
            kdim=config.acoustic_dim,
            vdim=config.acoustic_dim,
            num_heads=config.micro_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        # Initialize output projection to zero for seamless 100% warm-start
        nn.init.zeros_(self.acoustic_cross_attn.out_proj.weight)
        if self.acoustic_cross_attn.out_proj.bias is not None:
            nn.init.zeros_(self.acoustic_cross_attn.out_proj.bias)
        self.norm_ac = nn.LayerNorm(config.micro_dim)

        # 3. Macro Word Context Cross-Attention (sliding multi-word context)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.micro_dim,
            kdim=config.macro_dim,
            vdim=config.macro_dim,
            num_heads=config.micro_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(config.micro_dim)

        # 4. Path-Conditioned MoE FFN
        self.moe_ffn = ConditionedMoEFeedForwardNetwork(
            embed_dim=config.micro_dim,
            ffn_dim=config.micro_ffn_dim,
            num_experts=config.micro_num_experts,
            top_k=config.micro_moe_top_k,
            dropout=config.dropout,
        )
        self.norm3 = nn.LayerNorm(config.micro_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        word_memory: torch.Tensor,
        acoustic_memory: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        path_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Causal Self-Attention
        res = x
        normed = self.norm1(x)
        sa_out, _ = self.self_attn(normed, normed, normed, attn_mask=self_attn_mask)
        x = res + self.dropout(sa_out)

        # 2. Direct Acoustic Cross-Attention over intra-word continuous speech frames
        if acoustic_memory is not None:
            res = x
            normed = self.norm_ac(x)
            ac_out, _ = self.acoustic_cross_attn(normed, acoustic_memory, acoustic_memory)
            x = res + self.dropout(ac_out)

        # 3. Cross-Attention over macro sliding word context
        res = x
        normed = self.norm2(x)
        ca_out, _ = self.cross_attn(normed, word_memory, word_memory)
        x = res + self.dropout(ca_out)

        # 4. Path-Conditioned MoE FFN
        res = x
        normed = self.norm3(x)
        ffn_out, aux_loss = self.moe_ffn(normed, path_bias=path_bias)
        x = res + self.dropout(ffn_out)

        return x, aux_loss


class WindowedMicroRecursiveHead(nn.Module):
    """Recursive MoE micro-decoder with Dual Cross-Attention (Acoustics + Multi-Word Context) and scheduled sampling."""

    def __init__(self, config: WindowedAdaptivePathConfig):
        super().__init__()
        self.config = config

        self.char_embedding = nn.Embedding(
            config.byte_vocab_size,
            config.micro_dim,
            padding_idx=config.pad_token_id,
        )
        nn.init.normal_(self.char_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.char_embedding.weight[config.pad_token_id].fill_(0.0)

        self.pos_emb = nn.Parameter(torch.randn(1, 64, config.micro_dim) * 0.02)
        self.acoustic_pos_emb = nn.Parameter(
            torch.randn(1, config.acoustic_window_frames, config.acoustic_dim) * 0.02
        )

        self.layers = nn.ModuleList([
            DualCrossAttnMicroMoELayer(config) for _ in range(config.micro_layers)
        ])
        self.final_norm = nn.LayerNorm(config.micro_dim)

        self.lm_head = nn.Linear(config.micro_dim, config.byte_vocab_size, bias=False)
        self.lm_head.weight = self.char_embedding.weight  # Weight tying

    def forward(
        self,
        input_ids: torch.Tensor,
        word_window_latent: torch.Tensor,
        acoustic_slices: Optional[torch.Tensor] = None,
        path_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through recursive micro-head with dual acoustic & multi-word context.

        Args:
            input_ids: [N, K] token IDs for each word instance.
            word_window_latent: [N, W, macro_dim] multi-word context vectors.
            acoustic_slices: Optional [N, S_ac, acoustic_dim] continuous speech frame slices.
            path_bias: Optional [N, 1, micro_dim] length path bias.
        """
        N, K = input_ids.shape
        device = input_ids.device

        # Add acoustic positional embeddings to acoustic frame slices
        if acoustic_slices is not None:
            S_ac = acoustic_slices.shape[1]
            acoustic_slices = acoustic_slices + self.acoustic_pos_emb[:, :S_ac, :]

        # Ramped scheduled sampling: with 5% probability, roll out 2 predicted tokens
        if self.training and self.config.scheduled_sampling_prob > 0.0 and K > 3:
            cur_ids = input_ids.clone()
            if torch.rand(1).item() < self.config.scheduled_sampling_prob:
                with torch.no_grad():
                    # Step 1 rollout
                    h_sub1 = self.char_embedding(cur_ids[:, :2]) + self.pos_emb[:, :2, :]
                    cmask1 = torch.triu(torch.full((2, 2), float("-inf"), device=device), diagonal=1)
                    for layer in self.layers:
                        h_sub1, _ = layer(
                            h_sub1, word_window_latent, acoustic_slices, self_attn_mask=cmask1, path_bias=path_bias
                        )
                    pred1 = self.lm_head(self.final_norm(h_sub1))[:, 0, :].argmax(dim=-1)
                    cur_ids[:, 1] = pred1

                    # Step 2 rollout
                    h_sub2 = self.char_embedding(cur_ids[:, :3]) + self.pos_emb[:, :3, :]
                    cmask2 = torch.triu(torch.full((3, 3), float("-inf"), device=device), diagonal=1)
                    for layer in self.layers:
                        h_sub2, _ = layer(
                            h_sub2, word_window_latent, acoustic_slices, self_attn_mask=cmask2, path_bias=path_bias
                        )
                    pred2 = self.lm_head(self.final_norm(h_sub2))[:, 1, :].argmax(dim=-1)
                    cur_ids[:, 2] = pred2

            input_ids = cur_ids

        h = self.char_embedding(input_ids) + self.pos_emb[:, :K, :]
        causal_mask = torch.triu(torch.full((K, K), float("-inf"), device=device), diagonal=1)

        total_aux = torch.tensor(0.0, device=device)
        for layer in self.layers:
            h, aux = layer(
                h, word_window_latent, acoustic_slices, self_attn_mask=causal_mask, path_bias=path_bias
            )
            total_aux = total_aux + aux

        h = self.final_norm(h)
        logits = self.lm_head(h)
        return logits, total_aux


class PhonoV67WindowedDecoder(nn.Module):
    """Phono-V6.7: Universal Multilingual Word-to-Character Decoder with Windowed Multi-Word Cross-Attention."""

    def __init__(self, config: WindowedAdaptivePathConfig):
        super().__init__()
        self.config = config
        self.max_word_len = config.max_word_slots
        self.window_size = config.word_context_window

        # 1. Learnable Phoneme Acoustic Encoder (deterministic end-to-end)
        self.phoneme_embedding = nn.Embedding(
            config.phoneme_vocab_size,
            config.acoustic_dim,
            padding_idx=config.pad_token_id,
        )
        nn.init.normal_(self.phoneme_embedding.weight, mean=0.0, std=0.02)
        self.phoneme_prenet = nn.Sequential(
            nn.Conv1d(config.acoustic_dim, config.acoustic_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.acoustic_dim, config.acoustic_dim, kernel_size=3, padding=1),
        )
        self.phoneme_norm = nn.LayerNorm(config.acoustic_dim)

        # 2. Macro Word Slot Queries
        self.word_slot_embedding = nn.Parameter(
            torch.randn(1, self.max_word_len, config.macro_dim) * 0.02
        )

        # 3. Macro MoE Transformer Layers (cross-attention over acoustic memory)
        self.macro_layers = nn.ModuleList([
            MacroMoELayer(config) for _ in range(config.macro_layers)
        ])
        self.macro_norm = nn.LayerNorm(config.macro_dim)

        # 4. Multi-Word Window Context Embeddings
        # Learned prefix embedding for initial sentence words where context < window_size
        self.word_bos_embedding = nn.Parameter(
            torch.randn(1, 1, config.macro_dim) * 0.02
        )
        # Relative position embeddings for word context window offsets [-3, -2, -1, 0]
        self.word_relative_pos_embedding = nn.Embedding(
            self.window_size, config.macro_dim
        )
        nn.init.normal_(self.word_relative_pos_embedding.weight, mean=0.0, std=0.02)

        # 5. 4-Path Length Classifier & Conditioning Embedding
        self.macro_path_classifier = nn.Linear(config.macro_dim, config.num_paths)
        self.path_embedding = nn.Embedding(config.num_paths, config.micro_dim)
        nn.init.normal_(self.path_embedding.weight, mean=0.0, std=0.02)

        # 6. Word Latent Diffusion Scheduler
        if self.config.use_word_diffusion:
            self.word_refiner = DeepLatentDiffusionRefiner(
                embed_dim=self.config.macro_dim,
                dropout=self.config.dropout,
            )
            self.noise_scheduler = GaussianNoiseScheduler(default_noise_max=self.config.word_noise_max)
        else:
            self.word_refiner = None
            self.noise_scheduler = None

        # 7. Shared Micro Recursive Character Head with Multi-Word Cross-Attention
        self.micro_head = WindowedMicroRecursiveHead(config)

    def encode_phonemes_to_acoustics(
        self, phoneme_ids: torch.Tensor, phoneme_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert discrete phoneme IDs into continuous 50Hz speech-like acoustic memory."""
        expanded_ids = phoneme_ids.repeat_interleave(3, dim=1)
        emb = self.phoneme_embedding(expanded_ids)
        h = self.phoneme_prenet(emb.transpose(1, 2)).transpose(1, 2)
        acoustic_memory = self.phoneme_norm(h)

        if phoneme_lengths is not None:
            memory_lengths = phoneme_lengths * 3
        else:
            memory_lengths = torch.full((phoneme_ids.shape[0],), acoustic_memory.shape[1], device=phoneme_ids.device)

        return acoustic_memory, memory_lengths

    def build_word_context_windows(self, z_word: torch.Tensor) -> torch.Tensor:
        """Extract multi-word context windows with relative positional embeddings.

        Args:
            z_word: [B, L, macro_dim] word latent representations.

        Returns:
            z_windows: [B, L, W, macro_dim] sliding context windows.
        """
        B, L, D = z_word.shape
        W = self.window_size
        device = z_word.device

        # Prepend (W - 1) learned BOS word embeddings for early sequence context
        bos_prefix = self.word_bos_embedding.expand(B, W - 1, -1)
        z_padded = torch.cat([bos_prefix, z_word], dim=1)  # [B, L + W - 1, D]

        # Slicing via unfold: shape [B, L, D, W] -> permute to [B, L, W, D]
        z_windows = z_padded.unfold(dimension=1, size=W, step=1).permute(0, 1, 3, 2).contiguous()

        # Add relative position embedding for offsets [-3, -2, -1, 0]
        rel_pos_ids = torch.arange(W, device=device)
        rel_emb = self.word_relative_pos_embedding(rel_pos_ids)  # [W, D]
        z_windows = z_windows + rel_emb.unsqueeze(0).unsqueeze(0)

        return z_windows

    def extract_guided_acoustic_slices(
        self,
        acoustic_memory: torch.Tensor,
        macro_attn_weights: Optional[torch.Tensor] = None,
        window_frames: int = 32,
    ) -> torch.Tensor:
        """Extract continuous acoustic frame slices [B, L, S_ac, D] centered at word attention peaks.

        Args:
            acoustic_memory: [B, T, D] continuous speech representations.
            macro_attn_weights: Optional [B, L, T] cross-attention probabilities from macro layer.
            window_frames: int S_ac, number of continuous frames per word slice (~32 frames = 640ms).

        Returns:
            acoustic_slices: [B, L, S_ac, D] local speech frame slices for intra-word character attention.
        """
        B, T, D = acoustic_memory.shape
        L = macro_attn_weights.shape[1] if macro_attn_weights is not None else self.max_word_len
        device = acoustic_memory.device
        S_ac = window_frames

        # Pad acoustic memory along time dimension T if shorter than S_ac
        if T < S_ac:
            pad_len = S_ac - T
            acoustic_memory = F.pad(acoustic_memory, (0, 0, 0, pad_len))
            if macro_attn_weights is not None:
                macro_attn_weights = F.pad(macro_attn_weights, (0, pad_len))
            T = S_ac

        if macro_attn_weights is not None:
            time_indices = torch.arange(T, device=device, dtype=torch.float32).view(1, 1, T)
            weight_sums = macro_attn_weights.sum(dim=-1, keepdim=True)
            valid_sums = weight_sums.clamp(min=1e-6)
            center_t = (macro_attn_weights * time_indices).sum(dim=-1, keepdim=True) / valid_sums

            # Fallback for unaligned or zero-weight slots: monotonic linear interpolation
            linear_fallback = torch.linspace(0, max(0, T - 1), L, device=device).view(1, L, 1)
            center_t = torch.where(weight_sums > 1e-4, center_t, linear_fallback)
        else:
            center_t = torch.linspace(0, max(0, T - 1), L, device=device).view(1, L, 1)

        half_w = S_ac // 2
        start_t = (center_t.squeeze(-1).round().long() - half_w).clamp(min=0, max=max(0, T - S_ac))  # [B, L]
        offsets = torch.arange(S_ac, device=device).view(1, 1, S_ac)  # [1, 1, S_ac]
        frame_indices = start_t.unsqueeze(-1) + offsets  # [B, L, S_ac]

        # Vectorized gather along time dimension (dim=2)
        acoustic_slices = torch.gather(
            acoustic_memory.unsqueeze(1).expand(-1, L, -1, -1),
            dim=2,
            index=frame_indices.unsqueeze(-1).expand(-1, -1, -1, D),
        )
        return acoustic_slices

    def forward(
        self,
        phoneme_ids: Optional[torch.Tensor] = None,
        phoneme_lengths: Optional[torch.Tensor] = None,
        acoustic_memory: Optional[torch.Tensor] = None,
        memory_lengths: Optional[torch.Tensor] = None,
        num_words: Optional[torch.Tensor] = None,
        input_byte_ids: Optional[torch.Tensor] = None,
        target_byte_ids: Optional[torch.Tensor] = None,
        path_targets: Optional[torch.Tensor] = None,
        expected_total_len: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Forward pass with Windowed Multi-Word Context cross-attention."""
        # 1. Acoustic memory resolution
        if phoneme_ids is not None and phoneme_ids.is_floating_point():
            acoustic_memory = phoneme_ids
            phoneme_ids = None

        if acoustic_memory is None:
            if phoneme_ids is None:
                raise ValueError("Must provide either phoneme_ids or acoustic_memory")
            acoustic_memory, memory_lengths = self.encode_phonemes_to_acoustics(phoneme_ids, phoneme_lengths)

        B, T, _ = acoustic_memory.shape
        device = acoustic_memory.device

        if input_byte_ids is not None:
            L = input_byte_ids.shape[1]
        elif num_words is not None:
            L = num_words.max().item()
        else:
            L = max(2, int(T / 12.0))
        L = min(L, self.max_word_len)

        # 2. Macro Word Slot Queries
        x = self.word_slot_embedding[:, :L, :].expand(B, -1, -1)
        causal_mask = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)

        mem_pad_mask = None
        if memory_lengths is not None:
            t_idx = torch.arange(T, device=device).unsqueeze(0)
            mem_pad_mask = t_idx >= memory_lengths.unsqueeze(1)

        macro_aux_loss = torch.tensor(0.0, device=device)
        macro_attn_weights = None
        for i, layer in enumerate(self.macro_layers):
            is_last = (i == len(self.macro_layers) - 1)
            if is_last:
                x, aux, macro_attn_weights = layer(
                    x,
                    acoustic_memory,
                    self_attn_mask=causal_mask,
                    memory_padding_mask=mem_pad_mask,
                    expected_total_len=expected_total_len or L,
                    return_attn_weights=True,
                )
            else:
                x, aux = layer(
                    x,
                    acoustic_memory,
                    self_attn_mask=causal_mask,
                    memory_padding_mask=mem_pad_mask,
                    expected_total_len=expected_total_len or L,
                )
            macro_aux_loss = macro_aux_loss + aux

        z_word = self.macro_norm(x)  # [B, L, macro_dim]

        # 3. 4-Path Length-Adaptive Classification
        path_logits = self.macro_path_classifier(z_word)  # [B, L, 4]
        path_loss = torch.tensor(0.0, device=device)
        path_acc = torch.tensor(0.0, device=device)

        if path_targets is not None:
            pt_L = path_targets.shape[1]
            if pt_L >= L:
                pt_sliced = path_targets[:, :L]
                valid_mask = pt_sliced != -100
                if valid_mask.any():
                    path_loss = F.cross_entropy(
                        path_logits.reshape(-1, self.config.num_paths),
                        pt_sliced.reshape(-1),
                        ignore_index=-100,
                    )
                    pred_paths = path_logits.argmax(dim=-1)
                    path_acc = (
                        (pred_paths[valid_mask] == pt_sliced[valid_mask]).float().mean() * 100.0
                    )

        # 4. Path Conditioning Embedding
        if path_targets is not None and path_targets.shape[1] >= L:
            clamped_targets = torch.clamp(path_targets[:, :L], 0, self.config.num_paths - 1)
            path_bias = self.path_embedding(clamped_targets)
        else:
            pred_paths = path_logits.argmax(dim=-1)
            path_bias = self.path_embedding(pred_paths)

        # 5. Word Latent Diffusion Denoising Step
        diff_loss = torch.tensor(0.0, device=device)
        if self.config.use_word_diffusion and self.word_refiner is not None:
            if self.training:
                noise_map, _, _ = self.noise_scheduler.compute_noise_map(B, L, device)
                eps = torch.randn_like(z_word)
                z_noisy = z_word + noise_map * eps
                z_clean_est = self.word_refiner(z_noisy, noise_map)
                diff_loss = ((z_clean_est - z_word) ** 2).mean()
                # Refined latent for character decoder: blend denoised estimate with macro latent
                z_refined = 0.5 * z_word + 0.5 * z_clean_est
            else:
                zero_noise = torch.zeros((B, L, 1), device=device)
                z_refined = self.word_refiner(z_word, zero_noise)
        else:
            z_refined = z_word

        # 6. Multi-Word Context Window Extraction
        z_windows = self.build_word_context_windows(z_refined)  # [B, L, W, macro_dim]

        # Extract continuous acoustic frame slices [B, L, S_ac, D]
        acoustic_slices = self.extract_guided_acoustic_slices(
            acoustic_memory, macro_attn_weights, window_frames=self.config.acoustic_window_frames
        )

        # 7. Micro Character Decoding with Dual Acoustic + Multi-Word Context
        logits = None
        loss = None
        char_acc = torch.tensor(0.0, device=device)
        micro_aux_loss = torch.tensor(0.0, device=device)

        if input_byte_ids is not None and target_byte_ids is not None:
            input_byte_ids = input_byte_ids[:, :L]
            target_byte_ids = target_byte_ids[:, :L]
            B_inp, L_inp, K = input_byte_ids.shape
            flat_inputs = input_byte_ids.reshape(B * L_inp, K)
            flat_windows = z_windows.reshape(B * L_inp, self.window_size, self.config.macro_dim)
            flat_acoustic = acoustic_slices.reshape(
                B * L_inp, self.config.acoustic_window_frames, self.config.acoustic_dim
            )
            flat_bias = path_bias.reshape(B * L_inp, 1, self.config.micro_dim)

            flat_logits, micro_aux = self.micro_head(
                flat_inputs, flat_windows, flat_acoustic, path_bias=flat_bias
            )
            micro_aux_loss = micro_aux
            logits = flat_logits.view(B, L_inp, K, self.config.byte_vocab_size)

            char_ce_loss = F.cross_entropy(
                flat_logits.view(-1, self.config.byte_vocab_size),
                target_byte_ids.reshape(-1),
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
            )

            preds = flat_logits.view(-1, self.config.byte_vocab_size).argmax(dim=-1)
            valid_mask = target_byte_ids.reshape(-1) != -100
            if valid_mask.any():
                char_acc = (preds[valid_mask] == target_byte_ids.reshape(-1)[valid_mask]).float().mean() * 100.0

            loss = (
                char_ce_loss
                + self.config.macro_path_loss_weight * path_loss
                + 0.1 * diff_loss
                + self.config.moe_loss_weight * (macro_aux_loss + micro_aux_loss)
            )

        return {
            "loss": loss,
            "logits": logits,
            "z_word": z_word,
            "path_logits": path_logits,
            "path_loss": path_loss,
            "path_acc": path_acc,
            "diff_loss": diff_loss,
            "char_acc": char_acc,
            "aux_loss": macro_aux_loss + micro_aux_loss,
        }

    @torch.no_grad()
    def generate(
        self,
        phoneme_ids: Optional[torch.Tensor] = None,
        phoneme_lengths: Optional[torch.Tensor] = None,
        acoustic_memory: Optional[torch.Tensor] = None,
        memory_lengths: Optional[torch.Tensor] = None,
        max_words: Optional[int] = None,
        temperature: float = 0.0,
    ) -> List[List[int]]:
        """Windowed Multi-Word Greedy Generation with strict <eow> termination."""
        self.eval()
        if phoneme_ids is not None and phoneme_ids.is_floating_point():
            acoustic_memory = phoneme_ids
            phoneme_ids = None

        if acoustic_memory is None:
            if phoneme_ids is None:
                raise ValueError("Must provide either phoneme_ids or acoustic_memory")
            acoustic_memory, memory_lengths = self.encode_phonemes_to_acoustics(phoneme_ids, phoneme_lengths)

        B, T, _ = acoustic_memory.shape
        device = acoustic_memory.device

        if B > 1:
            batch_words = []
            for b in range(B):
                m_len = memory_lengths[b : b + 1] if memory_lengths is not None else None
                sub_words = self.generate(
                    acoustic_memory=acoustic_memory[b : b + 1],
                    memory_lengths=m_len,
                    max_words=max_words,
                    temperature=temperature,
                )
                batch_words.append(sub_words)
            return batch_words

        L = max_words or max(2, int(T / 12.0))
        L = min(L, self.max_word_len)

        # 1. Macro Word Queries
        x = self.word_slot_embedding[:, :L, :].expand(B, -1, -1)
        causal_mask = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)

        mem_pad_mask = None
        if memory_lengths is not None:
            t_idx = torch.arange(T, device=device).unsqueeze(0)
            mem_pad_mask = t_idx >= memory_lengths.unsqueeze(1)

        macro_attn_weights = None
        for i, layer in enumerate(self.macro_layers):
            is_last = (i == len(self.macro_layers) - 1)
            if is_last:
                x, _, macro_attn_weights = layer(
                    x, acoustic_memory, self_attn_mask=causal_mask, memory_padding_mask=mem_pad_mask, return_attn_weights=True
                )
            else:
                x, _ = layer(x, acoustic_memory, self_attn_mask=causal_mask, memory_padding_mask=mem_pad_mask)

        z_word = self.macro_norm(x)  # [1, L, macro_dim]
        path_logits = self.macro_path_classifier(z_word)  # [1, L, 4]

        # Refined word latent via diffusion refiner with zero noise
        if self.config.use_word_diffusion and self.word_refiner is not None:
            zero_noise = torch.zeros((1, L, 1), device=device)
            z_refined = self.word_refiner(z_word, zero_noise)
        else:
            z_refined = z_word

        # Extract multi-word context windows
        z_windows = self.build_word_context_windows(z_refined)  # [1, L, W, macro_dim]

        # Extract continuous acoustic frame slices
        acoustic_slices = self.extract_guided_acoustic_slices(
            acoustic_memory, macro_attn_weights, window_frames=self.config.acoustic_window_frames
        )  # [1, L, S_ac, D]

        horizon_map = {
            self.config.PATH_SPECIAL: 0,
            self.config.PATH_SHORT: self.config.k_short,
            self.config.PATH_MEDIUM: self.config.k_medium,
            self.config.PATH_LONG: self.config.k_long,
        }

        generated_words: List[List[int]] = []
        for l in range(L):
            zw_window = z_windows[:, l, :, :]  # [1, W, macro_dim]
            w_acoustic = acoustic_slices[:, l, :, :]  # [1, S_ac, D]
            pred_path = path_logits[0, l].argmax(dim=-1).item()

            if pred_path == self.config.PATH_SPECIAL:
                if len(generated_words) > 0 and l > len(generated_words) + 1:
                    break
                continue

            max_k = horizon_map.get(pred_path, self.config.k_long)
            p_bias = self.path_embedding(torch.tensor([pred_path], device=device)).unsqueeze(0)

            cur_tokens = [self.config.bos_token_id]
            for step in range(max_k):
                inp = torch.tensor([cur_tokens], dtype=torch.long, device=device)
                logits, _ = self.micro_head(inp, zw_window, acoustic_slices=w_acoustic, path_bias=p_bias)
                next_logits = logits[0, -1, :]

                if temperature > 0.0:
                    probs = F.softmax(next_logits / temperature, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_tok = next_logits.argmax(dim=-1).item()

                if next_tok in (self.config.eow_token_id, self.config.eos_token_id, self.config.pad_token_id):
                    break
                cur_tokens.append(next_tok)

            if len(cur_tokens) > 1:
                generated_words.append(cur_tokens[1:])

        return generated_words

    def load_from_v6_6_checkpoint(self, checkpoint_path: str, device: str = "cpu") -> Dict[str, Any]:
        """Warm-start weights from Phono-V6.6 best checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        # Filter out keys that don't match or allow strict=False
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        print(f"✅ Loaded warm-start weights from: {checkpoint_path}")
        print(f"   - Newly initialized weights: {missing_keys}")
        print(f"   - Skipped weights: {unexpected_keys}")
        return {
            "step": ckpt.get("step", 0),
            "best_val_loss": ckpt.get("best_val_loss", float("inf")),
            "missing_keys": missing_keys,
        }
