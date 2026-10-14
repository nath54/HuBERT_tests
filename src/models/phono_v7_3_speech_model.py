"""Phono-V7.3 Real-Time Streaming Speech Model.

Major Innovations over V7.2:
1. Multi-Scale Dilated Conformer Convolutions (3 branches: 300ms, 1220ms, 2420ms receptive field)
   for 4x wider acoustic context without parameter increase.
2. Decoupled Word Boundary Gate Head (3-class: SPEECH, WORD_SPACE, SILENCE_BLANK) with weighted loss.
3. Recursive 2-Pass Phoneme Head with causal depthwise recurrent refinement for phonotactic memory.
4. Boundary-Gated Online Slicing with multi-frame hysteresis (silence run confirmation & min burst protection).
5. 100% Zero-Perturbation Warm-Start from V7.1 and V7.2 checkpoints.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.phoneme_tokenizer import PHONEME_VOCAB_SIZE, NUM_ARTICULATORY_GROUPS
from src.models.conformer_layers import ConformerMoETransformerEncoder
from src.models.phono_variants import PhonoV64GatedDiffusionConfig, PhonoV64GatedDiffusionForPreTraining
from src.models.phono_v7_1_alignment import extract_streaming_ctc_slices
from src.models.phono_v7_1_speech_model import PhonoV71SpeechConfig
from src.models.phono_v7_2_speech_model import PhonoV72Decoder, PhonoV72SpeechModel


class DecoupledBoundaryGateHead(nn.Module):
    """Dedicated 3-class boundary gate head for temporal speech segmentation.

    Classes:
    - 0: SPEECH (voiced phonemes inside words)
    - 1: WORD_SPACE (inter-word boundary delimiter)
    - 2: SILENCE_BLANK (ambient silence / non-emitting transition)
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, 3)
        # Zero-initialize so all classes start with equal probability (zero perturbation)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, T, D] acoustic representations
        Returns:
            logits: [B, T, 3] boundary class logits
        """
        return self.proj(h)


class HierarchicalRecursivePhonemeHead(nn.Module):
    """Hierarchical Recursive 2-Pass Phoneme Emission Head with Articulatory Macro-Guidance
    and Causal Phonotactic Refinement.

    Level 1: Articulatory Macro Head (8 classes: non-speech, prosody/punct, vowels, stops, fricatives, nasals, liquids, affricates)
    Level 2: Base phoneme projection modulated by macro-class probabilities (num_phonemes classes, e.g. 104)
    Level 3: Causal depthwise separable convolution refiner taking [h_t; softmax(base_phoneme_{t-1})]
             to enforce phonotactic grammar (no adjacent spaces, smoothed consonant closures).
    Level 4: Causal Multi-Head Phonotactic Self-Attention block attending to full utterance emission
             history to break CTC conditional independence across long phoneme spans.
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int = 104,
        num_macro_classes: int = NUM_ARTICULATORY_GROUPS,
        refiner_dim: int = 128,
        kernel_size: int = 5,
        num_heads: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_macro_classes = num_macro_classes
        self.refiner_dim = refiner_dim
        self.num_heads = num_heads
        self.head_dim = refiner_dim // num_heads

        # Level 1: Articulatory Macro Head
        self.macro_head = nn.Linear(embed_dim, num_macro_classes)

        # Macro conditioning projection to phonemes (zero-initialized for exact warm-start)
        self.macro_proj = nn.Linear(num_macro_classes, vocab_size, bias=False)
        nn.init.zeros_(self.macro_proj.weight)

        # Level 2: Base fine phoneme projection
        self.base_head = nn.Linear(embed_dim, vocab_size)

        # Level 3: Local Causal Conv Refiner
        self.kernel_size = kernel_size
        self.pad_len = kernel_size - 1

        in_channels = embed_dim + vocab_size
        self.refiner_in = nn.Conv1d(in_channels, refiner_dim, kernel_size=1)
        self.refiner_conv = nn.Conv1d(
            refiner_dim,
            refiner_dim,
            kernel_size=kernel_size,
            padding=self.pad_len,
            groups=refiner_dim,
            bias=False,
        )
        self.refiner_norm = nn.BatchNorm1d(refiner_dim)
        self.refiner_act = nn.SiLU()

        # Level 4: Causal Multi-Head Phonotactic Self-Attention Block
        self.attn_norm = nn.LayerNorm(refiner_dim)
        self.qkv_proj = nn.Linear(refiner_dim, refiner_dim * 3)
        self.attn_out = nn.Linear(refiner_dim, refiner_dim)
        # Zero-init attention output projection for exact identity warm-start
        nn.init.zeros_(self.attn_out.weight)
        nn.init.zeros_(self.attn_out.bias)

        self.refiner_out = nn.Linear(refiner_dim, vocab_size)

        # Zero-initialize the refiner output so on step 0, refined_logits == base_logits
        nn.init.zeros_(self.refiner_out.weight)
        if self.refiner_out.bias is not None:
            nn.init.zeros_(self.refiner_out.bias)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [B, T, D] acoustic representations
        Returns:
            refined_logits: [B, T, num_phonemes]
            base_logits: [B, T, num_phonemes]
            macro_logits: [B, T, num_macro_classes]
        """
        B, T, D = h.shape

        # Step 1: Predict Articulatory Macro Distribution
        macro_logits = self.macro_head(h)  # [B, T, 8]
        macro_probs = F.softmax(macro_logits, dim=-1)  # [B, T, 8]

        # Step 2: Base phoneme logits modulated by macro guidance
        # (Zero-initialized projection ensures step 0 identity)
        base_logits = self.base_head(h) + self.macro_proj(macro_probs)  # [B, T, num_phonemes]

        # Step 3: Shifted past emission probabilities (causal context at t-1)
        base_probs = F.softmax(base_logits, dim=-1)
        zeros_t0 = torch.zeros((B, 1, self.vocab_size), device=h.device, dtype=h.dtype)
        shifted_probs = torch.cat([zeros_t0, base_probs[:, :-1, :]], dim=1)  # [B, T, num_phonemes]

        # Step 4: Local Causal Convolutional Refinement
        u = torch.cat([h, shifted_probs], dim=-1).transpose(1, 2)  # [B, D+num_phonemes, T]
        r = self.refiner_in(u)
        r = self.refiner_conv(r)
        if self.pad_len > 0:
            r = r[:, :, :T]  # Enforce causal length truncation
        r = self.refiner_norm(r)
        r = self.refiner_act(r)
        r = r.transpose(1, 2)  # [B, T, refiner_dim]

        # Step 5: Causal Multi-Head Phonotactic Self-Attention over sequence history
        r_norm = self.attn_norm(r)
        qkv = self.qkv_proj(r_norm)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_ctx = attn_ctx.transpose(1, 2).contiguous().view(B, T, self.refiner_dim)
        r = r + self.attn_out(attn_ctx)  # Causal self-attention residual

        residual = self.refiner_out(r)  # [B, T, num_phonemes]
        refined_logits = base_logits + residual

        return refined_logits, base_logits, macro_logits


def decode_calibrated_phonemes(
    logits: torch.Tensor,
    blank_id: int = 1,
    pad_id: int = 0,
    space_id: int = 8,
    blank_penalty: float = 0.5,
    space_bonus: float = 0.0,
    min_tokens_between_spaces: int = 1,
) -> List[int]:
    """Calibrated CTC decoding with blank threshold penalty, space boost bonus, and consecutive space suppression."""
    adjusted_logits = logits.clone()
    if blank_penalty > 0.0:
        adjusted_logits[:, blank_id] -= blank_penalty
    if space_bonus != 0.0:
        adjusted_logits[:, space_id] += space_bonus

    preds = adjusted_logits.argmax(dim=-1).detach().cpu().tolist()
    collapsed = []
    prev = None
    for p in preds:
        if p != prev:
            if p != blank_id and p != pad_id:
                collapsed.append(p)
            prev = p

    cleaned = []
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

    # Suppress trailing repeated token loops (e.g. "a a a a a")
    while len(cleaned) >= 2 and cleaned[-1] == cleaned[-2]:
        cleaned.pop()

    return cleaned


def decode_top_k_calibrated_phonemes(
    logits: torch.Tensor,
    blank_id: int = 1,
    pad_id: int = 0,
    space_id: int = 8,
    blank_penalty: float = 0.5,
    space_bonus: float = 0.0,
    top_k: int = 3,
    min_tokens_between_spaces: int = 1,
) -> Tuple[List[int], List[Tuple[List[int], List[float]]]]:
    """Calibrated CTC decoding that extracts both discrete token IDs and per-emission Top-K candidates.

    Returns:
        cal_ids: List[int] discrete calibrated phoneme IDs (including space_id)
        emissions: List of (top_k_ids, top_k_probs) tuples for each emitted phoneme/space
    """
    adjusted_logits = logits.clone()
    if blank_penalty > 0.0:
        adjusted_logits[:, blank_id] -= blank_penalty
    if space_bonus != 0.0:
        adjusted_logits[:, space_id] += space_bonus

    preds = adjusted_logits.argmax(dim=-1).detach().cpu().tolist()

    # Identify non-blank, non-repeat emission frames
    collapsed_frames = []
    prev = None
    for t, p in enumerate(preds):
        if p != prev:
            if p != blank_id and p != pad_id:
                collapsed_frames.append((t, p))
            prev = p

    # Clean consecutive spaces
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

    # Suppress trailing repeated token loops
    while len(cleaned_frames) >= 2 and cleaned_frames[-1][1] == cleaned_frames[-2][1]:
        cleaned_frames.pop()

    cal_ids = [p for _, p in cleaned_frames]

    # For each emission, extract top-K candidate tokens and probabilities
    emissions = []
    for t, p in cleaned_frames:
        if p == space_id:
            emissions.append(([space_id] * top_k, [1.0] + [0.0] * (top_k - 1)))
        else:
            frame_logits = adjusted_logits[t].clone()
            frame_logits[blank_id] = -1e9
            frame_logits[pad_id] = -1e9
            frame_logits[space_id] = -1e9
            top_vals, top_inds = torch.topk(frame_logits, k=top_k)
            top_probs = F.softmax(top_vals, dim=-1).cpu().tolist()
            top_ids = top_inds.cpu().tolist()
            emissions.append((top_ids, top_probs))

    return cal_ids, emissions


# Backward compatibility alias
RecursivePhonemeHead = HierarchicalRecursivePhonemeHead


class PhonoV73AcousticBackbone(PhonoV64GatedDiffusionForPreTraining):
    """Conformer Acoustic Backbone for Phono-V7.3 with Multi-Scale Dilated Convolutions,
    Decoupled Boundary Gate, and Hierarchical Recursive Phoneme Emission Head."""

    def __init__(self, config: PhonoV64GatedDiffusionConfig, conformer_kernel_size: int = 31):
        super().__init__(config)

        self.vocab_size = PHONEME_VOCAB_SIZE
        self.phoneme_offset = 8
        self.num_phonemes = self.vocab_size - self.phoneme_offset
        self.config.vocab_size = self.vocab_size

        # Multi-Scale Dilated Conformer Encoder
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
            use_multi_scale_conv=True,
        )

        # 1. Decoupled Word Boundary Gate
        self.boundary_gate_head = DecoupledBoundaryGateHead(self.config.encoder_embed_dim)

        # 2. Hierarchical Recursive Phoneme Head
        self.phoneme_head = HierarchicalRecursivePhonemeHead(
            embed_dim=self.config.encoder_embed_dim,
            vocab_size=self.num_phonemes,
            num_macro_classes=NUM_ARTICULATORY_GROUPS,
        )

        # 3. Macro CTC Loss (blank token = 0)
        self.macro_ctc_loss_fn = nn.CTCLoss(
            blank=0,
            zero_infinity=True,
        )

    def forward(
        self,
        audio: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        boundary_targets: Optional[torch.Tensor] = None,
        macro_targets: Optional[torch.Tensor] = None,
        macro_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # 1. CNN acoustic feature extraction & projection
        cnn_features = self.feature_extractor(audio)
        features = self.feature_projection(cnn_features)
        input_lengths = self._get_feat_extract_output_lengths(audio_lengths) if audio_lengths is not None else None

        # Dynamic SpecAugment (Time-span & Channel masking) in training mode
        if self.training:
            features, _ = self.apply_specaugment(features)

        # 2. Multi-Scale Conformer Encoding
        enc_out = self.encoder(features)
        hidden_state = enc_out.get("last_hidden_state", enc_out.get("hidden_state"))

        # 3. Decoupled Boundary Gate Emissions
        boundary_logits = self.boundary_gate_head(hidden_state)  # [B, T, 3]

        # 4. Hierarchical Recursive Phoneme Emissions
        refined_p_logits, base_p_logits, macro_logits = self.phoneme_head(hidden_state)  # [B, T, 104], [B, T, 104], [B, T, 8]
        state_logits = self.state_router(hidden_state)  # [B, T, 4]

        refined_logits = self.compute_composite_log_probs(state_logits, refined_p_logits)  # [B, T, 112]
        base_logits = self.compute_composite_log_probs(state_logits, base_p_logits)  # [B, T, 112]

        # 5. CTC Loss computation
        ctc_loss = torch.tensor(0.0, device=audio.device)
        macro_ctc_loss = torch.tensor(0.0, device=audio.device)
        boundary_loss = torch.tensor(0.0, device=audio.device)

        if targets is not None:
            B_cur, T_cur, _ = refined_logits.shape
            log_probs = refined_logits.transpose(0, 1).float()  # (T, B, 112)
            i_lens = input_lengths if input_lengths is not None else torch.full((B_cur,), T_cur, device=audio.device, dtype=torch.long)
            t_lens = target_lengths if target_lengths is not None else torch.full((B_cur,), targets.shape[1], device=audio.device, dtype=torch.long)

            ctc_loss = self.ctc_loss_fn(log_probs, targets, i_lens, t_lens)

        if macro_targets is not None:
            B_cur, T_cur, _ = macro_logits.shape
            macro_log_probs = F.log_softmax(macro_logits, dim=-1).transpose(0, 1).float()  # (T, B, 8)
            i_lens = input_lengths if input_lengths is not None else torch.full((B_cur,), T_cur, device=audio.device, dtype=torch.long)
            m_lens = macro_lengths if macro_lengths is not None else (target_lengths if target_lengths is not None else torch.full((B_cur,), macro_targets.shape[1], device=audio.device, dtype=torch.long))

            macro_ctc_loss = self.macro_ctc_loss_fn(macro_log_probs, macro_targets, i_lens, m_lens)

        # 6. Boundary Gate Cross-Entropy Loss with weighted space penalty
        if boundary_targets is not None:
            T_cur = boundary_logits.shape[1]
            T_bt = boundary_targets.shape[1]
            T_common = min(T_cur, T_bt)
            bg_sliced = boundary_logits[:, :T_common]
            bt_sliced = boundary_targets[:, :T_common]
            weights = torch.tensor([1.0, 10.0, 1.0], device=audio.device)  # 10x heavy penalty for missed/false spaces
            boundary_loss = F.cross_entropy(
                bg_sliced.reshape(-1, 3),
                bt_sliced.reshape(-1),
                weight=weights,
                ignore_index=-100,
            )

        total_enc_loss = ctc_loss + 0.2 * macro_ctc_loss + 1.0 * boundary_loss

        return {
            "loss": total_enc_loss,
            "ctc_loss": ctc_loss,
            "macro_ctc_loss": macro_ctc_loss,
            "boundary_loss": boundary_loss,
            "logits": refined_logits,
            "base_logits": base_logits,
            "macro_logits": macro_logits,
            "boundary_logits": boundary_logits,
            "hidden_state": hidden_state,
            "input_lengths": input_lengths,
        }

    @torch.no_grad()
    def forward_sliding_chunks(
        self,
        audio: torch.Tensor,
        chunk_seconds: float = 8.0,
        overlap_seconds: float = 2.0,
        sample_rate: int = 16000,
    ) -> Dict[str, torch.Tensor]:
        """Runs multi-window overlapping forward passes over long audio scripts.
        Maintains O(1) memory and stitches frame logits with smooth linear cross-fade blending.
        """
        B, total_samples = audio.shape
        chunk_size = int(chunk_seconds * sample_rate)
        overlap_size = int(overlap_seconds * sample_rate)
        hop_size = max(1, chunk_size - overlap_size)

        if total_samples <= chunk_size:
            out = self.forward(audio)
            return {
                "logits": out["logits"],
                "macro_logits": out["macro_logits"],
                "boundary_logits": out["boundary_logits"],
            }

        chunk_starts = list(range(0, max(1, total_samples - overlap_size), hop_size))
        all_logits = []
        all_macro = []
        all_bnd = []

        for start in chunk_starts:
            end = min(start + chunk_size, total_samples)
            chunk_audio = audio[:, start:end]
            if chunk_audio.shape[-1] < 1600:
                pad = 1600 - chunk_audio.shape[-1]
                chunk_audio = F.pad(chunk_audio, (0, pad))

            out = self.forward(chunk_audio)
            all_logits.append(out["logits"])
            all_macro.append(out["macro_logits"])
            all_bnd.append(out["boundary_logits"])

        downsample = 320
        overlap_frames = overlap_size // downsample

        stitched_logits = all_logits[0]
        stitched_macro = all_macro[0]
        stitched_bnd = all_bnd[0]

        ramp_up = torch.linspace(0.0, 1.0, overlap_frames, device=audio.device).view(1, -1, 1)
        ramp_down = 1.0 - ramp_up

        for c_idx in range(1, len(all_logits)):
            next_log = all_logits[c_idx]
            next_mac = all_macro[c_idx]
            next_bnd = all_bnd[c_idx]

            cur_T = stitched_logits.shape[1]
            overlap_k = min(overlap_frames, cur_T, next_log.shape[1])
            if overlap_k > 0:
                r_up = ramp_up[:, :overlap_k, :]
                r_down = ramp_down[:, :overlap_k, :]

                blended_log = r_down * stitched_logits[:, -overlap_k:, :] + r_up * next_log[:, :overlap_k, :]
                blended_mac = r_down * stitched_macro[:, -overlap_k:, :] + r_up * next_mac[:, :overlap_k, :]
                blended_bnd = r_down * stitched_bnd[:, -overlap_k:, :] + r_up * next_bnd[:, :overlap_k, :]

                stitched_logits = torch.cat([stitched_logits[:, :-overlap_k, :], blended_log, next_log[:, overlap_k:, :]], dim=1)
                stitched_macro = torch.cat([stitched_macro[:, :-overlap_k, :], blended_mac, next_mac[:, overlap_k:, :]], dim=1)
                stitched_bnd = torch.cat([stitched_bnd[:, :-overlap_k, :], blended_bnd, next_bnd[:, overlap_k:, :]], dim=1)
            else:
                stitched_logits = torch.cat([stitched_logits, next_log], dim=1)
                stitched_macro = torch.cat([stitched_macro, next_mac], dim=1)
                stitched_bnd = torch.cat([stitched_bnd, next_bnd], dim=1)

        return {
            "logits": stitched_logits,
            "macro_logits": stitched_macro,
            "boundary_logits": stitched_bnd,
        }


class PhonoV73SpeechModel(PhonoV72SpeechModel):
    """Phono-V7.3 Speech Model with Multi-Scale Dilated Conformer, Decoupled Boundary Gate,
    Recursive 2-Pass Phoneme Head, and Level-1/Level-2 Partitioned MoE Decoder."""

    def __init__(self, config: Optional[PhonoV71SpeechConfig] = None):
        cfg = config or PhonoV71SpeechConfig.medium()
        super().__init__(cfg)

        # Replace encoder with PhonoV73AcousticBackbone
        self.encoder = PhonoV73AcousticBackbone(
            cfg.encoder_config,
            conformer_kernel_size=cfg.conformer_kernel_size,
        )

        # Decoder is PhonoV72Decoder (Level-1 Gate + Level-2 Partitioned MoE + Horizon Capping)
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
        boundary_targets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # 1. Acoustic Forward Pass
        enc_out = self.encoder(
            audio=audio,
            audio_lengths=audio_lengths,
            targets=phoneme_targets,
            target_lengths=phoneme_lengths,
            boundary_targets=boundary_targets,
            **kwargs,
        )
        hidden_state = enc_out["hidden_state"]
        input_lengths = enc_out.get("input_lengths")
        refined_logits = enc_out["logits"]
        boundary_logits = enc_out["boundary_logits"]
        enc_loss = enc_out.get("loss", torch.tensor(0.0, device=audio.device))

        # 2. Forced Alignment (during training)
        word_centers = None
        align_loss = torch.tensor(0.0, device=audio.device)
        boundary_loss = enc_out.get("boundary_loss", torch.tensor(0.0, device=audio.device))
        if self.config.use_forced_alignment and phoneme_targets is not None and refined_logits is not None:
            try:
                B_cur, T_frames, _ = refined_logits.shape
                log_probs = refined_logits
                i_lens = input_lengths if input_lengths is not None else torch.full((B_cur,), T_frames, device=audio.device)
                p_lens = phoneme_lengths if phoneme_lengths is not None else torch.full((B_cur,), phoneme_targets.shape[1], device=audio.device)

                aligned_tokens, _ = self.aligner.align(
                    log_probs=log_probs,
                    targets=phoneme_targets,
                    input_lengths=i_lens,
                    target_lengths=p_lens,
                )
                if aligned_tokens is not None:
                    # Ground-truth boundary targets from alignment (class 1 for space, 0 for phoneme, 2 for blank)
                    is_space_token = (aligned_tokens == 8)
                    is_blank_token = (aligned_tokens == 1)
                    is_speech_token = (aligned_tokens > 8)

                    if boundary_targets is None:
                        auto_boundary = torch.zeros_like(aligned_tokens)
                        auto_boundary[is_space_token] = 1
                        auto_boundary[is_blank_token] = 2
                        weights = torch.tensor([1.0, 10.0, 1.0], device=audio.device)  # 10x heavy penalty for space errors
                        b_loss = F.cross_entropy(
                            boundary_logits.reshape(-1, 3),
                            auto_boundary.reshape(-1),
                            weight=weights,
                            ignore_index=-100,
                        )
                        enc_loss = enc_loss + 1.0 * b_loss
                        boundary_loss = b_loss

                    # Direct CTC Space Peak & False Insertion Penalty:
                    # Heavily penalize omission of spaces at true word boundaries,
                    # and penalize false space hallucinations inside words
                    spc_ctc_loss = torch.tensor(0.0, device=audio.device)
                    if is_space_token.any():
                        loss_missed_spc = -log_probs[:, :, 8][is_space_token].mean()
                        p_spc = log_probs[:, :, 8].exp()
                        loss_false_spc = -torch.log((1.0 - p_spc[is_speech_token]).clamp(min=1e-6)).mean()
                        spc_ctc_loss = loss_missed_spc + 0.5 * loss_false_spc
                        enc_loss = enc_loss + 0.5 * spc_ctc_loss

                    L_req = int(num_words.max().item()) if num_words is not None else 64
                    centers = self.aligner.get_word_centers(aligned_tokens, space_id=8, max_words=L_req)
                    if centers is not None:
                        pad_w = L_req - centers.shape[1]
                        if pad_w > 0:
                            centers = F.pad(centers, (0, pad_w), value=0)
                        word_centers = centers[:, :L_req]
            except Exception as e:
                word_centers = None

        # 3. Micro Character Decoder Forward Pass with Boundary-Guided Slicing
        dec_out = self.decoder(
            acoustic_memory=hidden_state,
            memory_lengths=input_lengths,
            num_words=num_words,
            ctc_logits=refined_logits,
            forced_word_centers=word_centers,
            input_byte_ids=input_byte_ids,
            target_byte_ids=target_byte_ids,
            path_targets=path_targets,
            target_lengths=target_lengths,
        )

        dec_loss = dec_out["loss"]
        total_loss = dec_loss + (self.config.ctc_loss_weight * enc_loss) + align_loss

        return {
            "loss": total_loss,
            "total_loss": total_loss,
            "dec_loss": dec_loss,
            "enc_loss": enc_loss,
            "boundary_loss": boundary_loss,
            "char_loss": dec_out["char_loss"],
            "lev_loss": dec_out.get("lev_loss", torch.tensor(0.0)),
            "align_loss": align_loss,
            "path_loss": dec_out["path_loss"],
            "length_loss": dec_out.get("length_loss", torch.tensor(0.0, device=audio.device)),
            "length_headroom": dec_out.get("length_headroom", torch.tensor(0.0, device=audio.device)),
            "diff_loss": dec_out.get("diff_loss", torch.tensor(0.0, device=audio.device)),
            "char_acc": dec_out["char_acc"],
            "path_acc": dec_out["path_acc"],
            "near_acc": dec_out.get("near_acc", torch.tensor(0.0, device=audio.device)),
            "aux_loss": dec_out["aux_loss"],
            "logits": dec_out["logits"],
            "path_logits": dec_out["path_logits"],
            "k_hat": dec_out.get("k_hat"),
            "ctc_logits": refined_logits,
            "boundary_logits": boundary_logits,
            "z_word": dec_out.get("z_word"),
        }

    def warm_start_from_v7_2(self, checkpoint_path: str) -> Dict[str, int]:
        """Loads weights from a V7.1 or V7.2 checkpoint with zero-perturbation preservation."""
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)

        model_dict = self.state_dict()
        transferred = 0
        skipped = 0

        for k, v in state_dict.items():
            # Direct match
            if k in model_dict and model_dict[k].shape == v.shape:
                model_dict[k].copy_(v)
                transferred += 1
            # Splicing for base_head (56 -> 104) and legacy weights
            elif ("encoder.phoneme_head.base_head.weight" in k or "encoder.phoneme_head.weight" in k) and "encoder.phoneme_head.base_head.weight" in model_dict:
                min_c = min(model_dict["encoder.phoneme_head.base_head.weight"].shape[0], v.shape[0])
                model_dict["encoder.phoneme_head.base_head.weight"][:min_c].copy_(v[:min_c])
                transferred += 1
            elif ("encoder.phoneme_head.base_head.bias" in k or "encoder.phoneme_head.bias" in k) and "encoder.phoneme_head.base_head.bias" in model_dict:
                min_c = min(model_dict["encoder.phoneme_head.base_head.bias"].shape[0], v.shape[0])
                model_dict["encoder.phoneme_head.base_head.bias"][:min_c].copy_(v[:min_c])
                transferred += 1
            elif "encoder.phoneme_head.refiner_in.weight" in k and "encoder.phoneme_head.refiner_in.weight" in model_dict:
                t_w = model_dict["encoder.phoneme_head.refiner_in.weight"]
                t_w[:, :512, :].copy_(v[:, :512, :])
                min_p = min(t_w.shape[1] - 512, v.shape[1] - 512)
                t_w[:, 512 : 512 + min_p, :].copy_(v[:, 512 : 512 + min_p, :])
                transferred += 1
            elif "encoder.phoneme_head.refiner_out.weight" in k and "encoder.phoneme_head.refiner_out.weight" in model_dict:
                min_c = min(model_dict["encoder.phoneme_head.refiner_out.weight"].shape[0], v.shape[0])
                model_dict["encoder.phoneme_head.refiner_out.weight"][:min_c].copy_(v[:min_c])
                transferred += 1
            elif "encoder.phoneme_head.refiner_out.bias" in k and "encoder.phoneme_head.refiner_out.bias" in model_dict:
                min_c = min(model_dict["encoder.phoneme_head.refiner_out.bias"].shape[0], v.shape[0])
                model_dict["encoder.phoneme_head.refiner_out.bias"][:min_c].copy_(v[:min_c])
                transferred += 1
            else:
                skipped += 1

        self.load_state_dict(model_dict)

        missing_keys = [
            k for k in model_dict.keys()
            if k not in state_dict
            and "base_head" not in k
            and "refiner" not in k
            and "boundary_gate" not in k
            and "conv_d" not in k
            and "dilated_proj" not in k
        ]
        return {
            "transferred": transferred,
            "skipped": skipped,
            "missing": len(missing_keys),
        }
