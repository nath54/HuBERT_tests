"""Phono-V7.2 Real-Time Streaming Speech Model.

Architectural Improvements over V7.1:
1. Two-Level Hierarchical Router:
   - Level 1: Binary Gate Router (SPECIAL / PAUSE / BREATH vs SPEECH WORD).
     If SPECIAL -> 0 FLOPs immediate bypass.
   - Level 2: Partitioned MoE Expert Routing:
     - SHORT  (Experts 0..4, nominal max K=5): function words, articles, prepositions
     - MEDIUM (Experts 5..10, nominal max K=9): standard stems, regular nouns and verbs
     - LONG   (Experts 11..15, nominal max K=24): compound words, conjugations, polysyllabic
2. Strict Continuous Length Horizon Capping:
   - Rollout steps are bounded by:
     max_k = min(head_bound, ceil(k_hat) + 1)
   - Eliminates trailing character hallucinations and saves compute on short stems.
3. 100% Parameter Compatible with V7.1:
   - All 16 experts in the micro-decoder map directly to the 3 partitions without altering tensor shapes.
"""

from typing import Dict, List, Optional, Tuple, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.phono_v6_7_windowed_decoder import WindowedAdaptivePathConfig
from src.models.phono_v7_1_alignment import (
    extract_streaming_ctc_slices,
    detect_online_word_peaks,
)
from src.models.phono_v7_1_speech_model import (
    PhonoV71SpeechConfig,
    PhonoV71Decoder,
    PhonoV71SpeechModel,
    build_band_causal_mask,
    asymmetric_length_loss,
)
from src.losses.soft_levenshtein import SoftLevenshteinLoss


# Expert Partition Definitions
EXPERT_PARTITIONS = {
    1: (0, 5),    # PATH_SHORT:  Experts 0..4 (5 experts)
    2: (5, 11),   # PATH_MEDIUM: Experts 5..10 (6 experts)
    3: (11, 16),  # PATH_LONG:   Experts 11..15 (5 experts)
}


class PartitionedConditionedMoEFFN(nn.Module):
    """MoE FFN with Partitioned Expert Routing for Short, Medium, and Long words.
    
    Experts are partitioned into:
      - Short words:  Experts 0..4  (Function words, high frequency)
      - Medium words: Experts 5..10 (Regular stems, nouns, verbs)
      - Long words:   Experts 11..15 (Polysyllabic words, compounds)
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_experts: int = 16,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(embed_dim, num_experts, bias=False)

        # 16 Expert FFNs: FC1 -> GELU -> FC2 (100% warm-start compatible with V7/V7.1)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, embed_dim),
            )
            for _ in range(num_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        path_bias: Optional[torch.Tensor] = None,
        path_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with partitioned expert routing.
        
        Args:
            x: [B, seq_len, D] input character states.
            path_bias: Optional [B, seq_len, D] or [B, 1, D] path embedding bias.
            path_id: Optional [B] or [B, 1] tensor containing path IDs (1=Short, 2=Med, 3=Long).
        """
        B, seq_len, D = x.shape
        flat_x = x.reshape(-1, D)

        if path_bias is not None:
            if path_bias.shape[1] == 1 and seq_len > 1:
                flat_router_input = (x + path_bias.expand(-1, seq_len, -1)).reshape(-1, D)
            else:
                flat_router_input = (x + path_bias).reshape(-1, D)
        else:
            flat_router_input = flat_x

        gate_logits = self.gate(flat_router_input)  # [N, num_experts]

        # Apply Partition Masking if path_id is supplied
        if path_id is not None:
            # Flatten path_id to match [N] where N = B * seq_len
            if path_id.numel() == B:
                flat_path = path_id.view(B, 1).expand(B, seq_len).reshape(-1)
            elif path_id.numel() == B * seq_len:
                flat_path = path_id.reshape(-1)
            else:
                flat_path = path_id.view(-1)

            partition_mask = torch.full_like(gate_logits, float("-inf"))
            for p_val, (start_idx, end_idx) in EXPERT_PARTITIONS.items():
                p_mask = (flat_path == p_val)
                if p_mask.any():
                    partition_mask[p_mask, start_idx:end_idx] = 0.0

            # If path <= 0 or path > 3 (special or unassigned), allow all experts
            unassigned = (flat_path <= 0) | (flat_path > 3)
            if unassigned.any():
                partition_mask[unassigned, :] = 0.0

            gate_logits = gate_logits + partition_mask

        gate_probs = F.softmax(gate_logits, dim=-1)

        # Load balancing auxiliary loss
        router_prob_per_expert = gate_probs.mean(dim=0)
        top_k_indices_all = gate_probs.topk(self.top_k, dim=-1).indices
        mask = F.one_hot(top_k_indices_all, self.num_experts).sum(dim=1).float()
        fraction_per_expert = mask.mean(dim=0)
        aux_loss = (router_prob_per_expert * fraction_per_expert).sum() * self.num_experts

        # Select Top-k within partitioned probability distribution
        weights, indices = torch.topk(gate_probs, self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        out = torch.zeros_like(flat_x)
        for expert_id, expert_fn in enumerate(self.experts):
            is_chosen = (indices == expert_id).any(dim=-1)
            if not is_chosen.any():
                continue

            expert_in = flat_x[is_chosen]
            expert_out = expert_fn(expert_in)

            weight_mask = (indices[is_chosen] == expert_id).float()
            expert_weights = (weights[is_chosen] * weight_mask).sum(dim=-1, keepdim=True)
            out[is_chosen] += expert_out * expert_weights

        return out.view(B, seq_len, D), aux_loss


class DualCrossAttnPartitionedMoELayer(nn.Module):
    """Transformer micro-decoder layer with Causal Self-Attention, Direct Acoustic Cross-Attention,
    Sliding Word Context Cross-Attention, and Partitioned MoE FFN.
    """

    def __init__(self, config: WindowedAdaptivePathConfig):
        super().__init__()
        # 1. Causal Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.micro_dim,
            num_heads=config.micro_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(config.micro_dim)

        # 2. Direct Acoustic Cross-Attention
        self.acoustic_cross_attn = nn.MultiheadAttention(
            embed_dim=config.micro_dim,
            kdim=config.acoustic_dim,
            vdim=config.acoustic_dim,
            num_heads=config.micro_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        nn.init.zeros_(self.acoustic_cross_attn.out_proj.weight)
        if self.acoustic_cross_attn.out_proj.bias is not None:
            nn.init.zeros_(self.acoustic_cross_attn.out_proj.bias)
        self.norm_ac = nn.LayerNorm(config.micro_dim)

        # 3. Macro Word Context Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.micro_dim,
            kdim=config.macro_dim,
            vdim=config.macro_dim,
            num_heads=config.micro_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(config.micro_dim)

        # 4. Partitioned MoE FFN
        self.moe_ffn = PartitionedConditionedMoEFFN(
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
        path_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Causal Self-Attention
        res = x
        normed = self.norm1(x)
        sa_out, _ = self.self_attn(normed, normed, normed, attn_mask=self_attn_mask)
        x = res + self.dropout(sa_out)

        # 2. Direct Acoustic Cross-Attention
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

        # 4. Partitioned MoE FFN
        res = x
        normed = self.norm3(x)
        ffn_out, aux_loss = self.moe_ffn(normed, path_bias=path_bias, path_id=path_id)
        x = res + self.dropout(ffn_out)

        return x, aux_loss


class WindowedMicroPartitionedRecursiveHead(nn.Module):
    """Recursive micro character decoder with Partitioned MoE expert layers."""

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
            DualCrossAttnPartitionedMoELayer(config) for _ in range(config.micro_layers)
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
        path_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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


class PhonoV72Decoder(PhonoV71Decoder):
    """Phono-V7.2 Streaming Decoder with Level-1 Binary Gate, Level-2 Partitioned MoE, and Horizon Capping."""

    def __init__(
        self,
        config: WindowedAdaptivePathConfig,
        levenshtein_loss_weight: float = 0.2,
        levenshtein_gamma: float = 0.2,
        band_window_words: int = 8,
        macro_history_noise_std: float = 0.05,
        length_mode: str = "regression",
        asymmetric_beta_under: float = 4.0,
        asymmetric_beta_over: float = 0.5,
        asymmetric_delta: float = 1.0,
        length_loss_weight: float = 0.3,
    ):
        super().__init__(
            config=config,
            levenshtein_loss_weight=levenshtein_loss_weight,
            levenshtein_gamma=levenshtein_gamma,
            band_window_words=band_window_words,
            macro_history_noise_std=macro_history_noise_std,
            length_mode=length_mode,
            asymmetric_beta_under=asymmetric_beta_under,
            asymmetric_beta_over=asymmetric_beta_over,
            asymmetric_delta=asymmetric_delta,
            length_loss_weight=length_loss_weight,
        )
        # Replace micro_head with Partitioned MoE Head
        self.micro_head = WindowedMicroPartitionedRecursiveHead(config)

    def forward(
        self,
        acoustic_memory: torch.Tensor,
        memory_lengths: Optional[torch.Tensor] = None,
        num_words: Optional[torch.Tensor] = None,
        ctc_logits: Optional[torch.Tensor] = None,
        forced_word_centers: Optional[torch.Tensor] = None,
        input_byte_ids: Optional[torch.Tensor] = None,
        target_byte_ids: Optional[torch.Tensor] = None,
        path_targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        expected_total_len: Optional[int] = None,
    ) -> Dict[str, Any]:
        B, T, _ = acoustic_memory.shape
        device = acoustic_memory.device

        if input_byte_ids is not None:
            L = input_byte_ids.shape[1]
        elif num_words is not None:
            L = int(num_words.max().item())
        else:
            L = self.max_word_len

        # 1. Shift-Invariant Word Query Initialization (Infinite Stream Capable)
        if hasattr(self, "word_slot_embedding") and self.word_slot_embedding.shape[1] >= L:
            x = self.word_slot_embedding[:, :L, :].expand(B, -1, -1) + self.word_query_base.expand(B, L, -1) * 0.1
        else:
            x = self.word_query_base.expand(B, L, -1)

        # 2. Band-Causal Local Attention Mask (K=8 words)
        band_mask = build_band_causal_mask(L, self.band_window_words, device=device)

        mem_pad_mask = None
        if memory_lengths is not None:
            t_idx = torch.arange(T, device=device).unsqueeze(0)
            mem_pad_mask = t_idx >= memory_lengths.unsqueeze(1)

        # 3. Macro History Noise Injection during training
        macro_aux_loss = torch.tensor(0.0, device=device)
        for layer in self.macro_layers:
            if self.training and self.macro_history_noise_std > 0.0:
                jitter = torch.randn_like(x) * self.macro_history_noise_std
                x_in = x + jitter
            else:
                x_in = x

            x, aux = layer(
                x_in,
                acoustic_memory,
                self_attn_mask=band_mask,
                memory_padding_mask=mem_pad_mask,
                expected_total_len=expected_total_len or L,
            )
            macro_aux_loss = macro_aux_loss + aux

        z_word = self.macro_norm(x)  # [B, L, macro_dim]

        # 4. Length-Adaptive Classification
        path_logits = self.macro_path_classifier(z_word)
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
                    path_acc = (pred_paths[valid_mask] == pt_sliced[valid_mask]).float().mean()

        # 5. Extract continuous speech slices with online event-driven centers
        acoustic_slices, durations = extract_streaming_ctc_slices(
            acoustic_memory=acoustic_memory,
            memory_lengths=memory_lengths,
            num_words=num_words,
            ctc_logits=ctc_logits,
            max_word_slots=L,
            window_frames=self.config.acoustic_window_frames,
            return_durations=True,
        )

        # 6. Word Length Prediction
        k_hat = self.length_predictor(z_word, durations)
        length_bias = self.length_to_micro_bias((k_hat / 10.0).unsqueeze(-1))

        length_loss = torch.tensor(0.0, device=device)
        length_headroom = torch.tensor(0.0, device=device)
        if target_lengths is not None and target_lengths.shape[1] >= L:
            tl_sliced = target_lengths[:, :L]
            length_loss, length_headroom = asymmetric_length_loss(
                k_hat=k_hat,
                k_true=tl_sliced,
                beta_under=self.asymmetric_beta_under,
                beta_over=self.asymmetric_beta_over,
                delta=self.asymmetric_delta,
            )

        # 7. Sliding Multi-Word Context Windows
        z_windows = self.build_word_context_windows(z_word)

        # 8. Micro Character Decoding with Partitioned MoE
        char_loss = torch.tensor(0.0, device=device)
        levenshtein_loss = torch.tensor(0.0, device=device)
        char_acc = torch.tensor(0.0, device=device)
        micro_aux_loss = torch.tensor(0.0, device=device)

        if input_byte_ids is not None and target_byte_ids is not None:
            L_inp = input_byte_ids.shape[1]
            K_inp = input_byte_ids.shape[2]

            if path_targets is not None and path_targets.shape[1] >= L_inp:
                clamped_targets = torch.clamp(path_targets[:, :L_inp], 0, self.config.num_paths - 1)
                path_bias = self.path_embedding(clamped_targets)
                active_paths = clamped_targets
            else:
                pred_paths = path_logits[:, :L_inp].argmax(dim=-1)
                path_bias = self.path_embedding(pred_paths)
                active_paths = pred_paths

            path_bias = path_bias + length_bias[:, :L_inp, :]

            flat_inputs = input_byte_ids.reshape(B * L_inp, K_inp)
            flat_windows = z_windows[:, :L_inp].reshape(B * L_inp, self.window_size, self.config.macro_dim)
            flat_acoustic = acoustic_slices[:, :L_inp].reshape(B * L_inp, self.config.acoustic_window_frames, self.config.acoustic_dim)
            flat_bias = path_bias.reshape(B * L_inp, 1, self.config.micro_dim)
            flat_path_id = active_paths.reshape(B * L_inp)

            flat_logits, micro_aux_loss = self.micro_head(
                flat_inputs,
                flat_windows,
                acoustic_slices=flat_acoustic,
                path_bias=flat_bias,
                path_id=flat_path_id,
            )
            char_logits = flat_logits.view(B, L_inp, K_inp, -1)

            targets = target_byte_ids[:, :L_inp, :K_inp]
            flat_targets = targets.reshape(-1)
            flat_preds = flat_logits.reshape(-1, self.config.byte_vocab_size)

            char_loss = F.cross_entropy(
                flat_preds,
                flat_targets,
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
            )

            if self.levenshtein_loss_weight > 0.0:
                flat_prob_logits = flat_logits.view(B * L_inp, K_inp, -1)
                flat_char_targets = targets.reshape(B * L_inp, K_inp)
                levenshtein_loss = self.levenshtein_loss_fn(flat_prob_logits, flat_char_targets)

            valid_chars = flat_targets != -100
            if valid_chars.any():
                correct = (flat_preds.argmax(dim=-1)[valid_chars] == flat_targets[valid_chars]).float()
                char_acc = correct.mean()

        total_loss = (
            char_loss
            + self.config.macro_path_loss_weight * path_loss
            + self.length_loss_weight * length_loss
            + self.levenshtein_loss_weight * levenshtein_loss
            + self.config.moe_loss_weight * (macro_aux_loss + micro_aux_loss)
        )

        return {
            "loss": total_loss,
            "char_loss": char_loss,
            "path_loss": path_loss,
            "length_loss": length_loss,
            "levenshtein_loss": levenshtein_loss,
            "path_logits": path_logits,
            "path_acc": path_acc,
            "char_acc": char_acc,
            "length_headroom": length_headroom,
            "k_hat": k_hat,
            "z_word": z_word,
            "aux_loss": macro_aux_loss + micro_aux_loss,
        }

    def decode_greedy(
        self,
        acoustic_memory: torch.Tensor,
        memory_lengths: Optional[torch.Tensor] = None,
        max_words: Optional[int] = None,
        ctc_logits: Optional[torch.Tensor] = None,
        temperature: float = 0.0,
    ) -> List[List[int]]:
        """Greedy autoregressive decoding with:
        1. Level 1 Binary Gate (Skip SPECIAL/PAUSE)
        2. Level 2 Partitioned Expert Routing (Short 0..4, Med 5..10, Long 11..15)
        3. Strict Horizon Capping: max_k = min(head_bound, ceil(k_hat) + 1)
        """
        B, T, D = acoustic_memory.shape
        device = acoustic_memory.device
        L = max_words if max_words is not None else self.max_word_len

        if hasattr(self, "word_slot_embedding") and self.word_slot_embedding.shape[1] >= L:
            x = self.word_slot_embedding[:, :L, :].expand(B, -1, -1) + self.word_query_base.expand(B, L, -1) * 0.1
        else:
            x = self.word_query_base.expand(B, L, -1)

        band_mask = build_band_causal_mask(L, self.band_window_words, device=device)
        mem_pad_mask = None
        if memory_lengths is not None:
            t_idx = torch.arange(T, device=device).unsqueeze(0)
            mem_pad_mask = t_idx >= memory_lengths.unsqueeze(1)

        for layer in self.macro_layers:
            x, _ = layer(x, acoustic_memory, self_attn_mask=band_mask, memory_padding_mask=mem_pad_mask, expected_total_len=L)

        z_word = self.macro_norm(x)
        path_logits = self.macro_path_classifier(z_word)
        z_windows = self.build_word_context_windows(z_word)

        n_w = torch.full((B,), L, device=device, dtype=torch.long)
        acoustic_slices, durations = extract_streaming_ctc_slices(
            acoustic_memory=acoustic_memory,
            memory_lengths=memory_lengths,
            num_words=n_w,
            ctc_logits=ctc_logits,
            max_word_slots=L,
            window_frames=self.config.acoustic_window_frames,
            return_durations=True,
        )

        k_hat = self.length_predictor(z_word, durations)
        length_bias = self.length_to_micro_bias((k_hat / 10.0).unsqueeze(-1))

        horizon_map = {
            self.config.PATH_SPECIAL: 0,
            self.config.PATH_SHORT: self.config.k_short,      # nominal max 5
            self.config.PATH_MEDIUM: self.config.k_medium,    # nominal max 9
            self.config.PATH_LONG: self.config.k_long,        # nominal max 24
        }

        generated_words: List[List[int]] = []
        for l in range(L):
            zw_window = z_windows[:, l, :, :]
            w_acoustic = acoustic_slices[:, l, :, :]
            pred_path = path_logits[0, l].argmax(dim=-1).item()

            # Level 1 Gate: Skip SPECIAL / PAUSE / SILENCE
            if pred_path == self.config.PATH_SPECIAL:
                if len(generated_words) > 0 and l > len(generated_words) + 1:
                    break
                continue

            # Level 2 Horizon Capping: min(head_bound, ceil(k_hat) + 1)
            nominal_head_bound = horizon_map.get(pred_path, self.config.k_long)
            pred_k = int(torch.ceil(k_hat[0, l]).item()) + 1
            max_k = min(nominal_head_bound, min(self.max_word_len, max(3, pred_k)))

            p_bias = self.path_embedding(torch.tensor([pred_path], device=device)).unsqueeze(0) + length_bias[:, l : l + 1, :]
            path_id_tensor = torch.tensor([pred_path], device=device)

            cur_tokens = [self.config.bos_token_id]
            for step in range(max_k):
                inp = torch.tensor([cur_tokens], dtype=torch.long, device=device)
                logits, _ = self.micro_head(
                    inp,
                    zw_window,
                    acoustic_slices=w_acoustic,
                    path_bias=p_bias,
                    path_id=path_id_tensor,
                )
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
                word_chars = cur_tokens[1:]
                word_chars.append(self.config.eow_token_id)
                generated_words.append(word_chars)

        return generated_words


class PhonoV72SpeechModel(PhonoV71SpeechModel):
    """Phono-V7.2 Speech Model with Conformer Encoder and Level-1/Level-2 Partitioned MoE Decoder."""

    def __init__(self, config: Optional[PhonoV71SpeechConfig] = None):
        cfg = config or PhonoV71SpeechConfig.medium()
        super().__init__(cfg)

        # Replace decoder with PhonoV72Decoder
        self.decoder = PhonoV72Decoder(
            config=cfg.decoder_config,
            levenshtein_loss_weight=cfg.levenshtein_loss_weight,
            levenshtein_gamma=cfg.levenshtein_gamma,
            band_window_words=cfg.band_window_words,
            macro_history_noise_std=cfg.macro_history_noise_std,
            length_mode=cfg.length_mode,
            asymmetric_beta_under=cfg.asymmetric_beta_under,
            asymmetric_beta_over=cfg.asymmetric_beta_over,
            asymmetric_delta=cfg.asymmetric_delta,
            length_loss_weight=cfg.length_loss_weight,
        )

    def warm_start_from_v7_1(self, checkpoint_path: str) -> Dict[str, int]:
        """Loads weights from a V7 or V7.1 checkpoint with 100% parameter compatibility."""
        return self.warm_start_from_v7(checkpoint_path)
