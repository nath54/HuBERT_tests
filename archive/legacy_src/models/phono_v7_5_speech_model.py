"""Phono-V7.5 Speech Model with Overlapping Word-Length Experts MoE Decoder.

Key Innovations over V7.3:
1. Overlapping Length Experts (7 intervals):
   - VERY_SHORT:   [1, 5]
   - SHORT:        [3, 7]
   - MEDIUM_SHORT: [5, 10]
   - MEDIUM:       [7, 12]
   - MEDIUM_LARGE: [9, 15]
   - LARGE:        [12, 20]
   - VERY_LARGE:   [15, 30]
2. Multi-Positive Interval Validity Supervision:
   - Any expert covering the target word length is considered valid (no hard bin cliff).
   - Interval Path Accuracy measures if chosen expert covers the word (>90% expected).
3. Anti-Collapse Load Balancing Loss (L_balance):
   - Strictly prevents router collapse to a single central expert.
4. Seamless 100% Zero-Perturbation Warm-Start from Phono-V7.3 checkpoint.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.phono_v7_1_alignment import extract_streaming_ctc_slices
from src.models.phono_v7_1_speech_model import (
    PhonoV71SpeechConfig,
    build_band_causal_mask,
    asymmetric_length_loss,
)
from src.models.phono_v7_2_speech_model import (
    PhonoV72Decoder,
    WindowedMicroPartitionedRecursiveHead,
)
from src.models.phono_v7_3_speech_model import (
    PhonoV73SpeechModel,
    PhonoV73AcousticBackbone,
)


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


def compute_interval_validity_mask(
    target_lengths: torch.Tensor,
    intervals: List[Tuple[int, int]] = LENGTH_INTERVALS,
) -> torch.Tensor:
    """Computes a multi-positive binary validity mask across overlapping experts.

    Args:
        target_lengths: [B, L] or [N] tensor of word character counts
        intervals: list of (min_len, max_len) tuples
    Returns:
        mask: [..., E] float tensor where mask[..., e] = 1.0 if target_lengths in interval
    """
    E = len(intervals)
    orig_shape = target_lengths.shape
    flat_lens = target_lengths.reshape(-1)  # [N]
    mask = torch.zeros((flat_lens.shape[0], E), device=target_lengths.device, dtype=torch.float32)

    for e, (min_l, max_l) in enumerate(intervals):
        in_interval = (flat_lens >= min_l) & (flat_lens <= max_l)
        mask[:, e] = in_interval.float()

    return mask.reshape(*orig_shape, E)


def compute_load_balancing_loss(
    expert_probs: torch.Tensor,
    top1_indices: torch.Tensor,
    num_experts: int = NUM_OVERLAPPING_EXPERTS,
) -> torch.Tensor:
    """Computes auxiliary load balancing loss to prevent expert collapse.

    L_balance = E * sum_e (f_e * P_e)
    where f_e is the fraction of tokens routed to expert e,
    and P_e is the mean probability assigned to expert e.
    """
    N = expert_probs.shape[0]
    if N == 0:
        return torch.tensor(0.0, device=expert_probs.device)

    # Frequency of expert selection
    f_e = torch.zeros(num_experts, device=expert_probs.device)
    for e in range(num_experts):
        f_e[e] = (top1_indices == e).float().mean()

    # Mean routing probability
    P_e = expert_probs.mean(dim=0)

    # Balanced loss
    return num_experts * (f_e * P_e).sum()


class OverlappingLengthMoEDecoder(PhonoV72Decoder):
    """Phono-V7.5 Streaming Decoder with 7 Overlapping Length Experts and Anti-Collapse Balancing."""

    def __init__(
        self,
        config: Any,
        levenshtein_loss_weight: float = 0.2,
        levenshtein_gamma: float = 0.2,
        band_window_words: int = 8,
        macro_history_noise_std: float = 0.05,
        length_mode: str = "regression",
        asymmetric_beta_under: float = 4.0,
        asymmetric_beta_over: float = 0.5,
        asymmetric_delta: float = 1.0,
        length_loss_weight: float = 0.3,
        num_experts: int = NUM_OVERLAPPING_EXPERTS,
        load_balance_weight: float = 0.1,
        routing_jitter_std: float = 0.05,
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
        self.num_experts = num_experts
        self.load_balance_weight = load_balance_weight
        self.routing_jitter_std = routing_jitter_std

        # Overlapping Length Experts Router: z_word -> 7 expert logits
        self.macro_expert_router = nn.Linear(config.macro_dim, num_experts)
        # Expert Conditioning Embeddings
        self.expert_embeddings = nn.Embedding(num_experts, config.micro_dim)

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

        # 1. Shift-Invariant Word Query Initialization
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

        # 3. Macro History Noise Injection
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

        # 4. Overlapping Length Experts Classification & Anti-Collapse MoE Routing
        expert_logits = self.macro_expert_router(z_word)  # [B, L, 7]
        if self.training and self.routing_jitter_std > 0.0:
            router_noise = torch.randn_like(expert_logits) * self.routing_jitter_std
            routed_logits = expert_logits + router_noise
        else:
            routed_logits = expert_logits

        expert_probs = F.softmax(routed_logits, dim=-1)  # [B, L, 7]
        top1_experts = routed_logits.argmax(dim=-1)      # [B, L]

        path_loss = torch.tensor(0.0, device=device)
        balance_loss = torch.tensor(0.0, device=device)
        path_acc = torch.tensor(0.0, device=device)

        if target_lengths is not None and target_lengths.shape[1] >= L:
            tl_sliced = target_lengths[:, :L]
            valid_len_mask = tl_sliced > 0  # Ignore padding words

            if valid_len_mask.any():
                valid_mask_all = compute_interval_validity_mask(tl_sliced)  # [B, L, 7]
                flat_valid_lens = valid_len_mask.reshape(-1)
                flat_logits = expert_logits.reshape(-1, self.num_experts)[flat_valid_lens]
                flat_targets = valid_mask_all.reshape(-1, self.num_experts)[flat_valid_lens]
                flat_probs = expert_probs.reshape(-1, self.num_experts)[flat_valid_lens]
                flat_top1 = top1_experts.reshape(-1)[flat_valid_lens]

                # Multi-Positive Binary Cross-Entropy Loss
                bce_loss = F.binary_cross_entropy_with_logits(flat_logits, flat_targets)

                # Anti-Collapse Load Balancing Loss
                balance_loss = compute_load_balancing_loss(flat_probs, flat_top1, self.num_experts)

                path_loss = bce_loss + self.load_balance_weight * balance_loss

                # Interval Path Accuracy: True if the selected expert is within ANY valid interval!
                is_valid = flat_targets.gather(-1, flat_top1.unsqueeze(-1)).squeeze(-1) > 0.5
                path_acc = is_valid.float().mean() * 100.0

        # Expert Conditioning Vector
        expert_cond = self.expert_embeddings(top1_experts)  # [B, L, micro_dim]
        z_conditioned = z_word

        # 5. Extract continuous speech slices with forced word centers
        acoustic_slices, durations = extract_streaming_ctc_slices(
            acoustic_memory=acoustic_memory,
            memory_lengths=memory_lengths,
            num_words=num_words,
            ctc_logits=ctc_logits,
            max_word_slots=L,
            window_frames=self.config.acoustic_window_frames,
            return_durations=True,
            forced_word_centers=forced_word_centers,
        )

        # 6. Word Length Prediction
        k_hat = self.length_predictor(z_conditioned, durations)

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
        z_windows = self.build_word_context_windows(z_conditioned)

        # 8. Micro Character Decoding with Partitioned MoE Head
        char_loss = torch.tensor(0.0, device=device)
        levenshtein_loss = torch.tensor(0.0, device=device)
        char_acc = torch.tensor(0.0, device=device)
        micro_aux_loss = torch.tensor(0.0, device=device)
        char_logits = None

        if input_byte_ids is not None and target_byte_ids is not None:
            L_inp = input_byte_ids.shape[1]
            K_inp = input_byte_ids.shape[2]

            path_bias = expert_cond[:, :L_inp]

            flat_slices = acoustic_slices[:, :L_inp].reshape(B * L_inp, self.config.acoustic_window_frames, -1)
            flat_windows = z_windows[:, :L_inp].reshape(B * L_inp, self.window_size, -1)
            flat_bytes = input_byte_ids.reshape(B * L_inp, K_inp)
            flat_path_bias = path_bias.reshape(B * L_inp, 1, self.config.micro_dim)
            flat_path_id = top1_experts[:, :L_inp].reshape(B * L_inp).clamp(0, self.config.num_paths - 1)

            flat_logits, micro_aux = self.micro_head(
                flat_bytes,
                flat_windows,
                acoustic_slices=flat_slices,
                path_bias=flat_path_bias,
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
                char_acc = correct.mean() * 100.0

        total_loss = (
            char_loss
            + self.config.macro_path_loss_weight * path_loss
            + self.length_loss_weight * length_loss
            + self.levenshtein_loss_weight * levenshtein_loss
            + self.config.moe_loss_weight * (macro_aux_loss + micro_aux_loss)
        )

        res = {
            "loss": total_loss,
            "char_loss": char_loss,
            "lev_loss": levenshtein_loss,
            "levenshtein_loss": levenshtein_loss,
            "path_loss": path_loss,
            "balance_loss": balance_loss,
            "length_loss": length_loss,
            "length_headroom": length_headroom,
            "diff_loss": torch.tensor(0.0, device=device),
            "char_acc": char_acc,
            "path_acc": path_acc,
            "near_acc": torch.tensor(0.0, device=device),
            "aux_loss": macro_aux_loss + micro_aux_loss,
            "logits": char_logits,
            "path_logits": expert_logits,
            "k_hat": k_hat,
            "z_word": z_word,
            "acoustic_slices": acoustic_slices,
        }
        self.last_dec_out = res
        return res

    def forward_text(
        self,
        z_word: torch.Tensor,
        input_byte_ids: torch.Tensor,
        target_byte_ids: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Dict[str, Any]:
        """Forward pass for pure Text Middle-Training without acoustic encoder."""
        B, L, _ = z_word.shape
        device = z_word.device

        # 1. Overlapping Length Experts Classification & Anti-Collapse MoE Routing
        expert_logits = self.macro_expert_router(z_word)  # [B, L, 7]
        if self.training and self.routing_jitter_std > 0.0:
            router_noise = torch.randn_like(expert_logits) * self.routing_jitter_std
            routed_logits = expert_logits + router_noise
        else:
            routed_logits = expert_logits

        expert_probs = F.softmax(routed_logits, dim=-1)
        top1_experts = routed_logits.argmax(dim=-1)

        valid_len_mask = target_lengths > 0
        valid_mask_all = compute_interval_validity_mask(target_lengths)

        flat_valid_lens = valid_len_mask.reshape(-1)
        flat_logits = expert_logits.reshape(-1, self.num_experts)[flat_valid_lens]
        flat_targets = valid_mask_all.reshape(-1, self.num_experts)[flat_valid_lens]
        flat_probs = expert_probs.reshape(-1, self.num_experts)[flat_valid_lens]
        flat_top1 = top1_experts.reshape(-1)[flat_valid_lens]

        bce_loss = F.binary_cross_entropy_with_logits(flat_logits, flat_targets)
        balance_loss = compute_load_balancing_loss(flat_probs, flat_top1, self.num_experts)
        path_loss = bce_loss + self.load_balance_weight * balance_loss

        is_valid = flat_targets.gather(-1, flat_top1.unsqueeze(-1)).squeeze(-1) > 0.5
        path_acc = is_valid.float().mean() * 100.0

        # Expert Conditioning Vector
        expert_cond = self.expert_embeddings(top1_experts)
        z_conditioned = z_word

        # Continuous Length Guidance
        durations = torch.zeros((B, L), device=device)
        k_hat = self.length_predictor(z_conditioned, durations)
        length_loss, length_headroom = asymmetric_length_loss(
            k_hat=k_hat,
            k_true=target_lengths,
            beta_under=self.asymmetric_beta_under,
            beta_over=self.asymmetric_beta_over,
            delta=self.asymmetric_delta,
        )

        # Sliding Multi-Word Context Windows
        z_windows = self.build_word_context_windows(z_conditioned)

        # Micro Character Decoding
        K_inp = input_byte_ids.shape[2]
        flat_windows = z_windows.reshape(B * L, self.window_size, self.config.macro_dim)
        flat_bytes = input_byte_ids.reshape(B * L, K_inp)
        flat_path_bias = expert_cond.reshape(B * L, 1, self.config.micro_dim)
        flat_path_id = top1_experts.reshape(B * L).clamp(0, self.config.num_paths - 1)

        flat_logits, micro_aux = self.micro_head(
            flat_bytes,
            flat_windows,
            acoustic_slices=None,
            path_bias=flat_path_bias,
            path_id=flat_path_id,
        )
        char_logits = flat_logits.view(B, L, K_inp, -1)

        targets = target_byte_ids[:, :L, :K_inp]
        flat_targets = targets.reshape(-1)
        flat_preds = flat_logits.reshape(-1, self.config.byte_vocab_size)

        char_loss = F.cross_entropy(
            flat_preds,
            flat_targets,
            ignore_index=-100,
            label_smoothing=self.config.label_smoothing,
        )

        levenshtein_loss = torch.tensor(0.0, device=device)
        if self.levenshtein_loss_weight > 0.0:
            flat_prob_logits = flat_logits.view(B * L, K_inp, -1)
            flat_char_targets = targets.reshape(B * L, K_inp)
            levenshtein_loss = self.levenshtein_loss_fn(flat_prob_logits, flat_char_targets)

        valid_chars = flat_targets != -100
        char_acc = torch.tensor(0.0, device=device)
        if valid_chars.any():
            correct = (flat_preds.argmax(dim=-1)[valid_chars] == flat_targets[valid_chars]).float()
            char_acc = correct.mean() * 100.0

        aux_moe = torch.nan_to_num(micro_aux, nan=0.0, posinf=0.0, neginf=0.0)
        total_loss = (
            char_loss
            + self.config.macro_path_loss_weight * path_loss
            + self.length_loss_weight * length_loss
            + self.levenshtein_loss_weight * levenshtein_loss
            + self.config.moe_loss_weight * aux_moe
        )

        return {
            "loss": total_loss,
            "char_loss": char_loss,
            "path_loss": path_loss,
            "balance_loss": balance_loss,
            "length_loss": length_loss,
            "length_headroom": length_headroom,
            "lev_loss": levenshtein_loss,
            "char_acc": char_acc,
            "path_acc": path_acc,
            "logits": char_logits,
            "k_hat": k_hat,
            "z_word": z_word,
        }


@dataclass
class PhonoV75SpeechConfig(PhonoV71SpeechConfig):
    """Configuration for Phono-V7.5 Speech Model."""
    num_overlapping_experts: int = NUM_OVERLAPPING_EXPERTS
    load_balance_weight: float = 0.1
    routing_jitter_std: float = 0.05


class PhonoV75SpeechModel(PhonoV73SpeechModel):
    """Phono-V7.5 Speech Model with Overlapping Length Experts MoE Decoder."""

    def __init__(self, config: Optional[PhonoV75SpeechConfig] = None):
        cfg = config or PhonoV75SpeechConfig.medium()
        super().__init__(cfg)

        # Replace decoder with OverlappingLengthMoEDecoder
        self.decoder = OverlappingLengthMoEDecoder(
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
            num_experts=cfg.num_overlapping_experts,
            load_balance_weight=cfg.load_balance_weight,
            routing_jitter_std=cfg.routing_jitter_std,
        )

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        out = super().forward(*args, **kwargs)
        if hasattr(self.decoder, 'last_dec_out') and self.decoder.last_dec_out is not None:
            out['balance_loss'] = self.decoder.last_dec_out.get('balance_loss', torch.tensor(0.0))
            out['z_word'] = self.decoder.last_dec_out.get('z_word')
        return out

    def forward_text(
        self,
        z_word: torch.Tensor,
        input_byte_ids: torch.Tensor,
        target_byte_ids: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Dict[str, Any]:
        return self.decoder.forward_text(
            z_word=z_word,
            input_byte_ids=input_byte_ids,
            target_byte_ids=target_byte_ids,
            target_lengths=target_lengths,
        )

    def warm_start_from_v7_3(self, ckpt_path: Union[str, Path]) -> Dict[str, int]:
        """Loads weights from Phono-V7.3 checkpoint with zero perturbation."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        # Filter out macro_path_classifier (4 classes) since V7.5 uses macro_expert_router (7 classes)
        filtered_sd = {}
        transferred = 0
        skipped = 0
        for k, v in state_dict.items():
            if "macro_path_classifier" in k:
                skipped += 1
                continue
            if k in self.state_dict():
                if self.state_dict()[k].shape == v.shape:
                    filtered_sd[k] = v
                    transferred += 1
                else:
                    skipped += 1
            else:
                skipped += 1

        missing, unexpected = self.load_state_dict(filtered_sd, strict=False)
        return {
            "transferred": transferred,
            "skipped": skipped,
            "missing": len(missing),
        }
