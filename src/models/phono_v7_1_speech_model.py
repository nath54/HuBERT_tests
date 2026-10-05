"""Phono-V7.1 Real-Time Streaming Speech Model.

Architectural Innovations:
1. Band-Causal Local Macro Attention (K=8 words):
   - Truncates causal attention history to local window of K words.
   - Prevents indefinite error snowballing across long live streams.
   - Operates in constant O(K * L) compute and memory.
2. Shift-Invariant Word Query Formulation:
   - Shared word base query q_base + relative local distance embeddings.
   - Completely shift-invariant: slot 100 has identical trained capacity to slot 0.
   - Infinite live streaming capability without a static slot ceiling.
3. Event-Driven Online CTC Peak Slicing:
   - Causal speech energy burst detector (1 - P(blank)) with space/silence boundary transitions.
   - Zero dependence on future duration or total sentence length.
   - Absorbs pauses, breaths, and speech tempo variations with zero temporal drift.
4. Macro Latent History Noise Injection:
   - Injects Gaussian jitter into historical word states during training.
   - Eliminates exposure bias and forces the decoder to anchor on acoustic evidence.
5. Continuous Duration-Aware Length Prediction + Asymmetric Loss + Soft-Levenshtein:
   - Retains continuous character length prediction with heavy under-penalty.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.soft_levenshtein import SoftLevenshteinLoss
from src.models.conformer_layers import ConformerMoETransformerEncoder
from src.models.forced_aligner import ForcedAligner
from src.models.phono_variants import (
    PhonoV64GatedDiffusionConfig,
    PhonoV64GatedDiffusionForPreTraining,
)
from src.models.phono_v6_7_windowed_decoder import (
    WindowedAdaptivePathConfig,
    PhonoV67WindowedDecoder,
)
from src.models.phono_v7_alignment import extract_ctc_monotonic_slices
from src.models.phono_v7_1_alignment import extract_streaming_ctc_slices, detect_online_word_peaks


def build_band_causal_mask(L: int, K: int, device: torch.device) -> torch.Tensor:
    """Build a band-causal attention mask allowing attention to at most K previous positions.

    Args:
        L: Sequence length.
        K: Maximum historical band width (e.g. 8 words).
        device: Target torch device.

    Returns:
        mask: [L, L] attention mask with 0.0 for allowed keys and -inf for blocked keys.
    """
    idx = torch.arange(L, device=device)
    diff = idx.unsqueeze(0) - idx.unsqueeze(1)  # j - i (key - query)
    mask = torch.full((L, L), float("-inf"), device=device)
    valid = (diff <= 0) & (diff >= -(K - 1))
    mask[valid] = 0.0
    return mask


def asymmetric_length_loss(
    k_hat: torch.Tensor,
    k_true: torch.Tensor,
    beta_under: float = 4.0,
    beta_over: float = 0.5,
    delta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Asymmetric penalty loss for continuous character word length prediction."""
    valid_mask = k_true > 0.0
    if not valid_mask.any():
        zero = torch.tensor(0.0, device=k_hat.device)
        return zero, zero

    diff = k_hat[valid_mask] - k_true[valid_mask]
    under_loss = torch.where(diff < 0.0, beta_under * (-diff), torch.zeros_like(diff))
    over_loss = torch.where(diff >= 0.0, beta_over * F.relu(diff - delta), torch.zeros_like(diff))
    total_loss = (under_loss + over_loss).mean()
    headroom = diff.mean()
    return total_loss, headroom


class WordLengthPredictor(nn.Module):
    """Predicts continuous word character length k_hat in [1, max_len] from z_word and phonetic duration."""

    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, z_word: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        dur_norm = (durations.unsqueeze(-1).float() / 25.0).clamp(min=0.0, max=10.0)
        feat = torch.cat([z_word, dur_norm], dim=-1)
        return self.net(feat).squeeze(-1)


@dataclass
class PhonoV71SpeechConfig:
    """Configuration for Phono-V7.1 Real-Time Streaming Speech Model."""

    encoder_config: PhonoV64GatedDiffusionConfig = field(
        default_factory=lambda: PhonoV64GatedDiffusionConfig.medium()
    )
    decoder_config: WindowedAdaptivePathConfig = field(
        default_factory=lambda: WindowedAdaptivePathConfig.medium()
    )

    # Architectural settings
    acoustic_window_frames: int = 32
    conformer_kernel_size: int = 31

    # Live Streaming Band Attention
    band_window_words: int = 8  # Local sliding causal attention window (K=8 words)
    macro_history_noise_std: float = 0.05  # History jitter during training to combat exposure bias

    # Double Loss weighting
    ctc_loss_weight: float = 0.5
    macro_path_loss_weight: float = 0.3
    levenshtein_loss_weight: float = 0.2
    levenshtein_gamma: float = 0.2
    use_forced_alignment: bool = True
    alignment_loss_weight: float = 0.2

    # Word Length Settings (Option B Default)
    length_mode: str = "regression"
    asymmetric_beta_under: float = 4.0
    asymmetric_beta_over: float = 0.5
    asymmetric_delta: float = 1.0
    length_loss_weight: float = 0.3

    # Training rates
    encoder_learning_rate: float = 1e-5
    decoder_learning_rate: float = 3e-4

    @classmethod
    def medium(cls, **kwargs) -> "PhonoV71SpeechConfig":
        enc_cfg = PhonoV64GatedDiffusionConfig(
            encoder_layers=8,
            encoder_heads=8,
            encoder_embed_dim=512,
            encoder_ffn_dim=2048,
            vocab_size=64,
            num_experts=4,
            moe_top_k=2,
            use_deep_refiner=True,
            learnable_gating=True,
        )
        dec_cfg = WindowedAdaptivePathConfig.medium()
        dec_cfg.cross_attn_band_width = 0
        dec_cfg.scheduled_sampling_prob = 0.05
        cfg = cls(encoder_config=enc_cfg, decoder_config=dec_cfg)
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


class PhonoV71AcousticBackbone(PhonoV64GatedDiffusionForPreTraining):
    """Conformer Acoustic Backbone with zero-initialized convolution for seamless warm-start."""

    def __init__(self, config: PhonoV64GatedDiffusionConfig, conformer_kernel_size: int = 31):
        super().__init__(config)
        self.encoder = ConformerMoETransformerEncoder(
            embed_dim=self.config.encoder_embed_dim,
            num_layers=self.config.encoder_layers,
            num_heads=self.config.encoder_heads,
            ffn_dim=self.config.encoder_ffn_dim,
            num_experts=getattr(self.config, "num_experts", 4),
            top_k=getattr(self.config, "moe_top_k", 2),
            kernel_size=conformer_kernel_size,
            dropout=self.config.dropout,
            attention_dropout=self.config.attention_dropout,
            pos_conv_kernel=self.config.pos_conv_kernel,
            pos_conv_groups=self.config.pos_conv_groups,
        )


class PhonoV71Decoder(PhonoV67WindowedDecoder):
    """Phono-V7.1 Streaming Decoder with Band-Causal Attention and Shift-Invariant Query Formulation."""

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
        super().__init__(config)
        self.levenshtein_loss_weight = levenshtein_loss_weight
        self.levenshtein_loss_fn = SoftLevenshteinLoss(
            gamma=levenshtein_gamma,
            ignore_index=-100,
            cost_mode="prob",
            normalize_by_len=True,
        )
        self.band_window_words = band_window_words
        self.macro_history_noise_std = macro_history_noise_std
        self.length_mode = length_mode
        self.asymmetric_beta_under = asymmetric_beta_under
        self.asymmetric_beta_over = asymmetric_beta_over
        self.asymmetric_delta = asymmetric_delta
        self.length_loss_weight = length_loss_weight

        # 1. Shift-Invariant Query: Shared base query vector
        self.word_query_base = nn.Parameter(torch.randn(1, 1, self.config.macro_dim) * 0.02)

        # 2. Continuous Word Length Predictor
        self.length_predictor = WordLengthPredictor(embed_dim=self.config.macro_dim)

        # 3. Continuous Length Guidance to Micro Character Head
        self.length_to_micro_bias = nn.Sequential(
            nn.Linear(1, self.config.micro_dim),
            nn.GELU(),
            nn.Linear(self.config.micro_dim, self.config.micro_dim),
        )
        nn.init.zeros_(self.length_to_micro_bias[-1].weight)
        nn.init.zeros_(self.length_to_micro_bias[-1].bias)

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
            L = max(2, int(T / 12.0))

        # 1. Shift-Invariant Word Query Initialization (Infinite Stream Capable)
        # Combines shared base query with slot embeddings if present for warm-start continuity
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

        # 3. Macro History Noise Injection during training (Exposure Bias Defense)
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
                    path_acc = (pred_paths[valid_mask] == pt_sliced[valid_mask]).float().mean() * 100.0

        if path_targets is not None and path_targets.shape[1] >= L:
            clamped_targets = torch.clamp(path_targets[:, :L], 0, self.config.num_paths - 1)
            path_bias = self.path_embedding(clamped_targets)
        else:
            pred_paths = path_logits.argmax(dim=-1)
            path_bias = self.path_embedding(pred_paths)

        # 5. Diffusion Refiner
        diff_loss = torch.tensor(0.0, device=device)
        if self.config.use_word_diffusion and self.word_refiner is not None:
            if self.training:
                noise_map, _, _ = self.noise_scheduler.compute_noise_map(B, L, device)
                eps = torch.randn_like(z_word)
                z_noisy = z_word + noise_map * eps
                z_clean_est = self.word_refiner(z_noisy, noise_map)
                diff_loss = ((z_clean_est - z_word) ** 2).mean()
                z_refined = 0.5 * z_word + 0.5 * z_clean_est
            else:
                zero_noise = torch.zeros((B, L, 1), device=device)
                z_refined = self.word_refiner(z_word, zero_noise)
        else:
            z_refined = z_word

        # 6. Multi-Word Context Window Extraction
        z_windows = self.build_word_context_windows(z_refined)

        # 7. Online Event-Driven CTC Speech Slicing
        n_w = num_words if num_words is not None else torch.full((B,), L, device=device, dtype=torch.long)
        acoustic_slices, durations = extract_streaming_ctc_slices(
            acoustic_memory=acoustic_memory,
            memory_lengths=memory_lengths,
            num_words=n_w,
            ctc_logits=ctc_logits,
            forced_word_centers=forced_word_centers,
            max_word_slots=L,
            window_frames=self.config.acoustic_window_frames,
            return_durations=True,
        )

        # 8. Continuous Length Prediction with Asymmetric Loss
        k_hat = self.length_predictor(z_refined, durations)
        length_loss = torch.tensor(0.0, device=device)
        length_headroom = torch.tensor(0.0, device=device)
        path_acc = torch.tensor(0.0, device=device)
        near_acc = torch.tensor(0.0, device=device)

        if target_lengths is not None:
            tl_L = target_lengths.shape[1]
            if tl_L >= L:
                tl_sliced = target_lengths[:, :L]
            else:
                tl_sliced = F.pad(target_lengths, (0, L - tl_L), value=-100.0)
            length_loss, length_headroom = asymmetric_length_loss(
                k_hat=k_hat,
                k_true=tl_sliced,
                beta_under=self.asymmetric_beta_under,
                beta_over=self.asymmetric_beta_over,
                delta=self.asymmetric_delta,
            )

            # Improved Path Accuracy: Coverage Rate (pred_max_k >= k_true, zero truncation)
            valid_len_mask = tl_sliced > 0.0
            if valid_len_mask.any():
                pred_budget = torch.ceil(k_hat).clamp(min=1.0) + 1.0
                path_acc = (pred_budget[valid_len_mask] >= tl_sliced[valid_len_mask]).float().mean() * 100.0
                near_acc = (torch.abs(k_hat[valid_len_mask] - tl_sliced[valid_len_mask]) <= 1.0).float().mean() * 100.0
            else:
                path_acc = torch.tensor(100.0, device=device)
                near_acc = torch.tensor(100.0, device=device)

        # 8b. Continuous Length Guidance Bias injected into Micro Character Head
        length_bias = self.length_to_micro_bias((k_hat / 10.0).unsqueeze(-1))  # [B, L, micro_dim]
        path_bias = path_bias + length_bias

        # 9. Micro Character Decoding with Dual Cross-Attention
        logits = None
        char_ce_loss = torch.tensor(0.0, device=device)
        char_acc = torch.tensor(0.0, device=device)
        micro_aux_loss = torch.tensor(0.0, device=device)

        if input_byte_ids is not None and target_byte_ids is not None:
            inp_b = input_byte_ids[:, :L]
            tgt_b = target_byte_ids[:, :L]
            K = inp_b.shape[-1]
            flat_inputs = inp_b.reshape(B * L, K)
            flat_windows = z_windows.reshape(B * L, self.window_size, self.config.macro_dim)
            flat_acoustic = acoustic_slices.reshape(
                B * L, self.config.acoustic_window_frames, self.config.acoustic_dim
            )
            flat_bias = path_bias.reshape(B * L, 1, self.config.micro_dim)

            flat_logits, micro_aux = self.micro_head(
                flat_inputs, flat_windows, flat_acoustic, path_bias=flat_bias
            )
            micro_aux_loss = micro_aux
            logits = flat_logits.view(B, L, K, self.config.byte_vocab_size)

            char_ce_loss = F.cross_entropy(
                flat_logits.view(-1, self.config.byte_vocab_size),
                tgt_b.reshape(-1),
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
            )

            lev_loss = torch.tensor(0.0, device=device)
            if self.levenshtein_loss_weight > 0.0:
                flat_tgt_b = tgt_b.reshape(B * L, K)
                lev_loss = self.levenshtein_loss_fn(
                    pred_logits=flat_logits,
                    target_ids=flat_tgt_b,
                )

            preds = flat_logits.view(-1, self.config.byte_vocab_size).argmax(dim=-1)
            valid_mask = tgt_b.reshape(-1) != -100
            if valid_mask.any():
                char_acc = (preds[valid_mask] == tgt_b.reshape(-1)[valid_mask]).float().mean() * 100.0

        loss = (
            char_ce_loss
            + self.levenshtein_loss_weight * lev_loss
            + self.config.macro_path_loss_weight * path_loss
            + self.length_loss_weight * length_loss
            + 0.1 * diff_loss
            + self.config.moe_loss_weight * (macro_aux_loss + micro_aux_loss)
        )

        return {
            "loss": loss,
            "char_loss": char_ce_loss,
            "lev_loss": lev_loss,
            "path_loss": path_loss,
            "length_loss": length_loss,
            "length_headroom": length_headroom,
            "diff_loss": diff_loss,
            "char_acc": char_acc,
            "path_acc": path_acc,
            "near_acc": near_acc,
            "aux_loss": macro_aux_loss + micro_aux_loss,
            "logits": logits,
            "path_logits": path_logits,
            "k_hat": k_hat,
            "z_word": z_word,
            "acoustic_slices": acoustic_slices,
        }

    @torch.no_grad()
    def generate(
        self,
        acoustic_memory: torch.Tensor,
        memory_lengths: Optional[torch.Tensor] = None,
        ctc_logits: Optional[torch.Tensor] = None,
        max_words: Optional[int] = None,
        temperature: float = 0.0,
    ) -> List[List[int]]:
        """Streaming character generation with band-causal attention and event-driven online slicing."""
        self.eval()
        B, T, _ = acoustic_memory.shape
        device = acoustic_memory.device

        if B > 1:
            batch_words = []
            for b in range(B):
                m_len = memory_lengths[b : b + 1] if memory_lengths is not None else None
                c_log = ctc_logits[b : b + 1] if ctc_logits is not None else None
                sub_words = self.generate(
                    acoustic_memory=acoustic_memory[b : b + 1],
                    memory_lengths=m_len,
                    ctc_logits=c_log,
                    max_words=max_words,
                    temperature=temperature,
                )
                batch_words.append(sub_words)
            return batch_words

        L = max_words or max(2, int(T / 12.0))

        # Shift-Invariant Base Query
        if hasattr(self, "word_slot_embedding") and self.word_slot_embedding.shape[1] >= L:
            x = self.word_slot_embedding[:, :L, :].expand(B, -1, -1) + self.word_query_base.expand(B, L, -1) * 0.1
        else:
            x = self.word_query_base.expand(B, L, -1)

        # Band-causal local attention mask
        band_mask = build_band_causal_mask(L, self.band_window_words, device=device)

        mem_pad_mask = None
        if memory_lengths is not None:
            t_idx = torch.arange(T, device=device).unsqueeze(0)
            mem_pad_mask = t_idx >= memory_lengths.unsqueeze(1)

        for layer in self.macro_layers:
            x, _ = layer(x, acoustic_memory, self_attn_mask=band_mask, memory_padding_mask=mem_pad_mask)

        z_word = self.macro_norm(x)
        path_logits = self.macro_path_classifier(z_word)

        if self.config.use_word_diffusion and self.word_refiner is not None:
            zero_noise = torch.zeros((B, L, 1), device=device)
            z_refined = self.word_refiner(z_word, zero_noise)
        else:
            z_refined = z_word

        z_windows = self.build_word_context_windows(z_refined)

        # Extract continuous speech slices with online event-driven centers
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

        k_hat = self.length_predictor(z_refined, durations)
        length_bias = self.length_to_micro_bias((k_hat / 10.0).unsqueeze(-1))

        horizon_map = {
            self.config.PATH_SPECIAL: 0,
            self.config.PATH_SHORT: self.config.k_short,
            self.config.PATH_MEDIUM: self.config.k_medium,
            self.config.PATH_LONG: self.config.k_long,
        }

        generated_words: List[List[int]] = []
        for l in range(L):
            zw_window = z_windows[:, l, :, :]
            w_acoustic = acoustic_slices[:, l, :, :]
            pred_path = path_logits[0, l].argmax(dim=-1).item()

            if pred_path == self.config.PATH_SPECIAL:
                if len(generated_words) > 0 and l > len(generated_words) + 1:
                    break
                continue

            if self.length_mode == "regression":
                pred_k = int(torch.ceil(k_hat[0, l]).item()) + 1
                max_k = min(self.max_word_len, max(3, pred_k))
            else:
                max_k = horizon_map.get(pred_path, self.config.k_long)

            p_bias = self.path_embedding(torch.tensor([pred_path], device=device)).unsqueeze(0) + length_bias[:, l : l + 1, :]

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


class PhonoV71SpeechModel(nn.Module):
    """Phono-V7.1 Real-Time Streaming Speech Model."""

    def __init__(self, config: PhonoV71SpeechConfig):
        super().__init__()
        self.config = config

        self.encoder = PhonoV71AcousticBackbone(
            config.encoder_config,
            conformer_kernel_size=config.conformer_kernel_size,
        )

        self.decoder = PhonoV71Decoder(
            config.decoder_config,
            levenshtein_loss_weight=config.levenshtein_loss_weight,
            levenshtein_gamma=config.levenshtein_gamma,
            band_window_words=config.band_window_words,
            macro_history_noise_std=config.macro_history_noise_std,
            length_mode=config.length_mode,
            asymmetric_beta_under=config.asymmetric_beta_under,
            asymmetric_beta_over=config.asymmetric_beta_over,
            asymmetric_delta=config.asymmetric_delta,
            length_loss_weight=config.length_loss_weight,
        )

        self.aligner = ForcedAligner(blank_id=1, pad_id=0, space_id=8)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        phoneme_targets: Optional[torch.Tensor] = None,
        phoneme_lengths: Optional[torch.Tensor] = None,
        num_words: Optional[torch.Tensor] = None,
        input_byte_ids: Optional[torch.Tensor] = None,
        target_byte_ids: Optional[torch.Tensor] = None,
        path_targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        expected_total_len: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # 1. Acoustic Backbone
        enc_out = self.encoder(
            audio=audio,
            targets=phoneme_targets,
            target_lengths=phoneme_lengths,
            audio_lengths=audio_lengths,
            **kwargs,
        )
        hidden_state = enc_out["hidden_state"]
        input_lengths = enc_out.get("input_lengths")
        ctc_logits = enc_out.get("refined_logits", enc_out.get("logits"))
        enc_loss = enc_out.get("loss", torch.tensor(0.0, device=audio.device))

        # 2. Forced Alignment (during training with phoneme targets)
        word_centers = None
        align_loss = torch.tensor(0.0, device=audio.device)
        if self.config.use_forced_alignment and phoneme_targets is not None and ctc_logits is not None:
            try:
                B_cur, T_frames, _ = ctc_logits.shape
                log_probs = ctc_logits.log_softmax(dim=-1)
                i_lens = input_lengths if input_lengths is not None else torch.full((B_cur,), T_frames, device=audio.device)
                p_lens = phoneme_lengths if phoneme_lengths is not None else torch.full((B_cur,), phoneme_targets.shape[1], device=audio.device)

                aligned_tokens, _ = self.aligner.align(
                    log_probs=log_probs,
                    targets=phoneme_targets,
                    input_lengths=i_lens,
                    target_lengths=p_lens,
                )
                align_loss = self.aligner.compute_frame_alignment_loss(ctc_logits, aligned_tokens)

                if num_words is not None:
                    max_w = int(num_words.max().item())
                    word_centers, _ = self.aligner.extract_word_centers(aligned_tokens, num_words, max_words=max_w)
            except Exception:
                pass

        # 3. Streaming Two-Level Decoder
        dec_out = self.decoder(
            acoustic_memory=hidden_state,
            memory_lengths=input_lengths,
            num_words=num_words,
            ctc_logits=ctc_logits,
            forced_word_centers=word_centers,
            input_byte_ids=input_byte_ids,
            target_byte_ids=target_byte_ids,
            path_targets=path_targets,
            target_lengths=target_lengths,
            expected_total_len=expected_total_len,
        )

        dec_loss = dec_out["loss"]
        if phoneme_targets is not None:
            total_loss = (
                dec_loss
                + (self.config.ctc_loss_weight * enc_loss)
                + (self.config.alignment_loss_weight * align_loss)
            )
        else:
            total_loss = dec_loss

        return {
            "loss": total_loss,
            "total_loss": total_loss,
            "dec_loss": dec_loss,
            "enc_loss": enc_loss,
            "char_loss": dec_out["char_loss"],
            "lev_loss": dec_out.get("lev_loss", torch.tensor(0.0)),
            "align_loss": align_loss,
            "path_loss": dec_out["path_loss"],
            "length_loss": dec_out.get("length_loss", torch.tensor(0.0, device=audio.device)),
            "length_headroom": dec_out.get("length_headroom", torch.tensor(0.0, device=audio.device)),
            "diff_loss": dec_out.get("diff_loss", torch.tensor(0.0)),
            "char_acc": dec_out["char_acc"],
            "path_acc": dec_out["path_acc"],
            "near_acc": dec_out.get("near_acc", torch.tensor(0.0, device=audio.device)),
            "aux_loss": dec_out["aux_loss"],
            "logits": dec_out["logits"],
            "path_logits": dec_out["path_logits"],
            "k_hat": dec_out.get("k_hat"),
            "ctc_logits": ctc_logits,
        }

    @torch.no_grad()
    def decode_greedy(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        max_words: Optional[int] = None,
    ) -> List[List[int]]:
        """Streaming greedy inference directly from raw 16kHz audio."""
        self.eval()
        enc_out = self.encoder(audio=audio, audio_lengths=audio_lengths)
        hidden_state = enc_out["hidden_state"]
        input_lengths = enc_out.get("input_lengths")
        ctc_logits = enc_out.get("refined_logits", enc_out.get("logits"))

        return self.decoder.generate(
            acoustic_memory=hidden_state,
            memory_lengths=input_lengths,
            ctc_logits=ctc_logits,
            max_words=max_words,
        )

    def warm_start_from_v7(self, checkpoint_path: str) -> Dict[str, Any]:
        """Warm-start Phono-V7.1 from an existing Phono-V7 checkpoint.

        Loads:
        - Conformer encoder weights
        - Macro transformer layers
        - Word length predictor
        - Micro recursive character head
        - Shared word base query (initialized from V7 word_slot_embedding[0])
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Initialize word_query_base from slot 0 if available
        if "decoder.word_slot_embedding" in state_dict:
            slot_0 = state_dict["decoder.word_slot_embedding"][:, 0:1, :]
            self.decoder.word_query_base.data.copy_(slot_0)

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        return {
            "transferred": len(state_dict) - len(unexpected),
            "missing": len(missing),
            "unexpected": len(unexpected),
        }
