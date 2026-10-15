"""Phono-V7 End-to-End Joint Multilingual Conformer Speech Model with Two-Level Character Decoding.

Key Architectural Components:
1. Model 1 (Conformer Acoustic Backbone):
   - Conformer-augmented acoustic encoder (depthwise separable conv with kernel=31, Pre-LN, GLU, Swish).
   - Zero-initialized final projection guarantees 100% identity preservation when warm-starting from 5.87% PER checkpoint.
   - Confidence-gated latent diffusion refiner + CTC loss.

2. Guaranteed Monotonic Temporal Alignment (Axes 1 & 2):
   - Proportional monotonic time slicing on actual valid duration T_valid (Axe 1).
   - CTC phoneme peak activation refinement (1 - P(blank)) (Axe 2).
   - Continuous 32-frame speech slices extracted via extract_ctc_monotonic_slices for each word slot.

3. Model 2 & 3 (Two-Level Hierarchical Character Decoder):
   - Macro Word Layer with causal self-attention and cross-attention over speech.
   - 4-Path Length-Adaptive Routing (Special, Short, Medium, Long) + Word Diffusion Refiner.
   - Micro Recursive Character Head with Dual Cross-Attention (32 continuous frames + sliding multi-word window).
   - Byte-level vocabulary (122 characters), completely immune to the Zipfian whole-word collapse.

4. Double Loss Objective:
   L_total = L_decoder + lambda_ctc * L_phoneme_ctc
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.phono_variants import (
    PhonoV64GatedDiffusionConfig,
    PhonoV64GatedDiffusionForPreTraining,
)
from src.models.conformer_layers import (
    ConformerConvModule,
    ConformerMoETransformerEncoder,
)
from src.models.phono_v6_7_windowed_decoder import (
    WindowedAdaptivePathConfig,
    PhonoV67WindowedDecoder,
    WindowedMicroRecursiveHead,
)
from src.models.phono_v7_alignment import extract_ctc_monotonic_slices
from src.losses.soft_levenshtein import SoftLevenshteinLoss
from src.models.forced_aligner import ForcedAligner


@dataclass
class PhonoV7SpeechConfig:
    """Configuration for Phono-V7 Conformer Speech Model."""

    encoder_config: PhonoV64GatedDiffusionConfig = field(
        default_factory=lambda: PhonoV64GatedDiffusionConfig.medium()
    )
    decoder_config: WindowedAdaptivePathConfig = field(
        default_factory=lambda: WindowedAdaptivePathConfig.medium()
    )

    # Architectural settings
    acoustic_window_frames: int = 32
    conformer_kernel_size: int = 31

    # Double Loss weighting
    ctc_loss_weight: float = 0.5
    macro_path_loss_weight: float = 0.3
    levenshtein_loss_weight: float = 0.2
    levenshtein_gamma: float = 0.2
    use_forced_alignment: bool = True
    alignment_loss_weight: float = 0.2

    # Word Horizon & Length settings (Option B default, Option A fallback toggle)
    length_mode: str = "regression"  # "regression" (Option B) or "categorical" (Option A)
    asymmetric_beta_under: float = 4.0
    asymmetric_beta_over: float = 0.5
    asymmetric_delta: float = 1.0
    length_loss_weight: float = 0.3

    # Training rates
    encoder_learning_rate: float = 1e-5
    decoder_learning_rate: float = 3e-4

    @classmethod
    def medium(cls, **kwargs) -> "PhonoV7SpeechConfig":
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


class PhonoV7AcousticBackbone(PhonoV64GatedDiffusionForPreTraining):
    """Phono-V7 Conformer-Augmented Acoustic Backbone.

    Replaces standard Transformer encoder with ConformerMoETransformerEncoder.
    Zero-initialized convolution ensures seamless, non-destructive warm-start.
    """

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


def asymmetric_length_loss(
    k_hat: torch.Tensor,
    k_true: torch.Tensor,
    beta_under: float = 4.0,
    beta_over: float = 0.5,
    delta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Asymmetric penalty loss for continuous character word length prediction.

    Under-prediction (k_hat < k_true) is penalized heavily by beta_under (default 4.0)
    to prevent catastrophic word truncation during inference.
    Over-prediction (k_hat >= k_true) is lightly penalized by beta_over (default 0.5)
    with a delta margin buffer (default 1.0) so +1 character of safety headroom is free.

    Returns:
        loss: scalar tensor loss
        headroom: scalar tensor average margin (k_hat - k_true) for valid words
    """
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
        """
        Args:
            z_word: [B, L, embed_dim] macro word embeddings
            durations: [B, L] acoustic frame durations between word centers
        Returns:
            k_hat: [B, L] continuous predicted character length (>= 0)
        """
        dur_norm = (durations.unsqueeze(-1).float() / 25.0).clamp(min=0.0, max=10.0)
        feat = torch.cat([z_word, dur_norm], dim=-1)
        return self.net(feat).squeeze(-1)


class PhonoV7Decoder(PhonoV67WindowedDecoder):
    """Phono-V7 Decoder with Axes 1 & 2 Monotonic Speech Slicing and Two-Level Character Decoding."""

    def __init__(
        self,
        config: WindowedAdaptivePathConfig,
        levenshtein_loss_weight: float = 0.2,
        levenshtein_gamma: float = 0.2,
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
        self.length_mode = length_mode
        self.asymmetric_beta_under = asymmetric_beta_under
        self.asymmetric_beta_over = asymmetric_beta_over
        self.asymmetric_delta = asymmetric_delta
        self.length_loss_weight = length_loss_weight
        self.length_predictor = WordLengthPredictor(embed_dim=self.config.macro_dim)

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
        """Forward pass with guaranteed Axes 1 & 2 Monotonic Slicing and Dual Cross-Attention."""
        B, T, _ = acoustic_memory.shape
        device = acoustic_memory.device

        if input_byte_ids is not None:
            L = min(input_byte_ids.shape[1], self.max_word_len)
        elif num_words is not None:
            L = min(int(num_words.max().item()), self.max_word_len)
        else:
            L = min(max(2, int(T / 12.0)), self.max_word_len)

        # 1. Macro Word Queries
        x = self.word_slot_embedding[:, :L, :].expand(B, -1, -1)
        causal_mask = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)

        mem_pad_mask = None
        if memory_lengths is not None:
            t_idx = torch.arange(T, device=device).unsqueeze(0)
            mem_pad_mask = t_idx >= memory_lengths.unsqueeze(1)

        macro_aux_loss = torch.tensor(0.0, device=device)
        for layer in self.macro_layers:
            x, aux = layer(
                x,
                acoustic_memory,
                self_attn_mask=causal_mask,
                memory_padding_mask=mem_pad_mask,
                expected_total_len=expected_total_len or L,
            )
            macro_aux_loss = macro_aux_loss + aux

        z_word = self.macro_norm(x)  # [B, L, macro_dim]

        # 2. 4-Path Length-Adaptive Classification
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
                    path_acc = (pred_paths[valid_mask] == pt_sliced[valid_mask]).float().mean() * 100.0

        if path_targets is not None and path_targets.shape[1] >= L:
            clamped_targets = torch.clamp(path_targets[:, :L], 0, self.config.num_paths - 1)
            path_bias = self.path_embedding(clamped_targets)
        else:
            pred_paths = path_logits.argmax(dim=-1)
            path_bias = self.path_embedding(pred_paths)

        # 3. Word Diffusion Refiner
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

        # 4. Multi-Word Context Window Extraction (W=6)
        z_windows = self.build_word_context_windows(z_refined)  # [B, L, W, macro_dim]

        # 5. Guaranteed Axes 1 & 2 Monotonic Continuous Speech Slicing!
        n_w = num_words if num_words is not None else torch.full((B,), L, device=device, dtype=torch.long)
        acoustic_slices, durations = extract_ctc_monotonic_slices(
            acoustic_memory=acoustic_memory,
            memory_lengths=memory_lengths,
            num_words=n_w,
            ctc_logits=ctc_logits,
            forced_word_centers=forced_word_centers,
            max_word_slots=L,
            window_frames=self.config.acoustic_window_frames,
            return_durations=True,
        )

        # 5b. Continuous Word Length Prediction with Asymmetric Loss
        k_hat = self.length_predictor(z_refined, durations)
        length_loss = torch.tensor(0.0, device=device)
        length_headroom = torch.tensor(0.0, device=device)
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

        # 6. Micro Character Decoding with Dual Cross-Attention
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
        """Greedy character generation with Axes 1 & 2 monotonic acoustic slicing."""
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
        L = min(L, self.max_word_len)

        # 1. Macro Word Queries
        x = self.word_slot_embedding[:, :L, :].expand(B, -1, -1)
        causal_mask = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)

        mem_pad_mask = None
        if memory_lengths is not None:
            t_idx = torch.arange(T, device=device).unsqueeze(0)
            mem_pad_mask = t_idx >= memory_lengths.unsqueeze(1)

        for layer in self.macro_layers:
            x, _ = layer(x, acoustic_memory, self_attn_mask=causal_mask, memory_padding_mask=mem_pad_mask)

        z_word = self.macro_norm(x)
        path_logits = self.macro_path_classifier(z_word)

        if self.config.use_word_diffusion and self.word_refiner is not None:
            zero_noise = torch.zeros((B, L, 1), device=device)
            z_refined = self.word_refiner(z_word, zero_noise)
        else:
            z_refined = z_word

        z_windows = self.build_word_context_windows(z_refined)

        # Extract continuous acoustic frame slices via Axes 1 & 2
        n_w = torch.full((B,), L, device=device, dtype=torch.long)
        acoustic_slices, durations = extract_ctc_monotonic_slices(
            acoustic_memory=acoustic_memory,
            memory_lengths=memory_lengths,
            num_words=n_w,
            ctc_logits=ctc_logits,
            max_word_slots=L,
            window_frames=self.config.acoustic_window_frames,
            return_durations=True,
        )

        k_hat = self.length_predictor(z_refined, durations)

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


class PhonoV7SpeechModel(nn.Module):
    """Phono-V7 End-to-End Joint Conformer Multilingual Speech Model."""

    def __init__(self, config: PhonoV7SpeechConfig):
        super().__init__()
        self.config = config

        # Model 1: Conformer-Augmented Acoustic Backbone
        self.encoder = PhonoV7AcousticBackbone(
            config.encoder_config,
            conformer_kernel_size=config.conformer_kernel_size,
        )

        # Model 2 & 3: Two-Level Decoder (Macro Word Routing + Micro Recursive Dual Cross-Attention)
        self.decoder = PhonoV7Decoder(
            config.decoder_config,
            levenshtein_loss_weight=config.levenshtein_loss_weight,
            levenshtein_gamma=config.levenshtein_gamma,
            length_mode=config.length_mode,
            asymmetric_beta_under=config.asymmetric_beta_under,
            asymmetric_beta_over=config.asymmetric_beta_over,
            asymmetric_delta=config.asymmetric_delta,
            length_loss_weight=config.length_loss_weight,
        )

        # Native Forced Aligner for exact physical frame supervision and word timestamp extraction
        self.aligner = ForcedAligner(blank_id=1, pad_id=0, space_id=8)

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        # Phoneme supervision (Loss 1)
        phoneme_targets: Optional[torch.Tensor] = None,
        phoneme_lengths: Optional[torch.Tensor] = None,
        # Decoder word & character supervision (Loss 2)
        num_words: Optional[torch.Tensor] = None,
        input_byte_ids: Optional[torch.Tensor] = None,
        target_byte_ids: Optional[torch.Tensor] = None,
        path_targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        expected_total_len: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass with Double Loss: CTC phoneme loss + Hierarchical Character Decoder loss."""
        # 1. Forward Model 1: Conformer Acoustic Backbone
        enc_out = self.encoder(
            audio=audio,
            targets=phoneme_targets,
            target_lengths=phoneme_lengths,
            audio_lengths=audio_lengths,
            **kwargs,
        )
        hidden_state = enc_out["hidden_state"]  # [B, T, 512] continuous speech representations
        input_lengths = enc_out.get("input_lengths")
        ctc_logits = enc_out.get("refined_logits")
        if ctc_logits is None:
            ctc_logits = enc_out.get("logits")
        enc_loss = enc_out.get("loss")
        if enc_loss is None:
            enc_loss = torch.tensor(0.0, device=audio.device)

        # 2. Forced Alignment (if enabled and phoneme targets are provided)
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

        # 3. Forward Model 2 & 3: Two-Level Windowed Character Decoder
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
            "aux_loss": dec_out["aux_loss"],
            "logits": dec_out["logits"],
            "path_logits": dec_out["path_logits"],
            "k_hat": dec_out.get("k_hat"),
            "ctc_logits": ctc_logits,
            "gate_bypass_pct": enc_out.get("gate_bypass_pct", 0.0),
            "gate_full_pct": enc_out.get("gate_full_pct", 0.0),
        }

    def warm_start_from_checkpoints(
        self,
        encoder_checkpoint_path: Optional[str] = None,
        decoder_checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Warm start encoder and decoder from best checkpoints.

        Encoder loads from V6.4 (5.87% PER record) with Conformer convs zero-initialized.
        Decoder loads from best V6.7 checkpoint.
        """
        results = {"encoder": None, "decoder": None}

        # 1. Warm-start Encoder
        if encoder_checkpoint_path:
            ckpt = torch.load(encoder_checkpoint_path, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            if any(k.startswith("encoder.feature_extractor") for k in state_dict):
                clean_sd = {
                    k[len("encoder."):]: v
                    for k, v in state_dict.items()
                    if k.startswith("encoder.")
                }
            else:
                clean_sd = {k: v for k, v in state_dict.items() if not k.startswith("decoder.")}
            enc_missing, enc_unexpected = self.encoder.load_state_dict(clean_sd, strict=False)
            results["encoder"] = {
                "transferred": len(clean_sd) - len(enc_unexpected),
                "missing": len(enc_missing),
                "unexpected": len(enc_unexpected),
            }

        # 2. Warm-start Decoder
        if decoder_checkpoint_path:
            ckpt = torch.load(decoder_checkpoint_path, map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt)
            if any(k.startswith("decoder.macro_layers") for k in state_dict):
                clean_sd = {
                    k[len("decoder."):]: v
                    for k, v in state_dict.items()
                    if k.startswith("decoder.")
                }
            else:
                clean_sd = {k: v for k, v in state_dict.items() if not k.startswith("encoder.")}
            dec_missing, dec_unexpected = self.decoder.load_state_dict(clean_sd, strict=False)
            results["decoder"] = {
                "transferred": len(clean_sd) - len(dec_unexpected),
                "missing": len(dec_missing),
                "unexpected": len(dec_unexpected),
            }

        return results

    @torch.no_grad()
    def decode_greedy(
        self,
        audio: torch.Tensor,
        audio_lengths: Optional[torch.Tensor] = None,
        max_words: Optional[int] = None,
    ) -> List[List[int]]:
        """Greedy inference directly from raw 16kHz audio to character sequences."""
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
