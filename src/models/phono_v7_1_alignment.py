"""Phono-V7.1 Online Event-Driven CTC Peak Slicing and Acoustic Alignment.

Provides causal, streamable acoustic slicing that tracks phonetic speech energy
peaks and inter-word silence boundaries online, completely absorbing tempo variations,
pauses, and breaths without cumulative temporal drift.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def detect_online_word_peaks(
    ctc_logits: torch.Tensor,
    blank_id: int = 1,
    space_id: int = 8,
    min_frames_per_word: int = 3,
    energy_threshold: float = 0.35,
    silence_threshold: float = 0.15,
    max_words: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Detect word acoustic centers incrementally from CTC emissions in temporal order.

    Operates causally frame-by-frame:
    - Identifies non-blank speech bursts: E(t) = 1.0 - P_t(blank).
    - Detects word transitions via CTC space token peaks or silence valleys.
    - Computes exact center of mass for each speech burst.

    Args:
        ctc_logits: [B, T, V] CTC emission logits.
        blank_id: Token ID for CTC blank.
        space_id: Token ID for word delimiter space.
        min_frames_per_word: Minimum frame span for a valid word burst.
        energy_threshold: Non-blank threshold to trigger word onset.
        silence_threshold: Threshold to trigger word boundary / silence.
        max_words: Maximum word slots to return.

    Returns:
        word_centers: [B, max_words] frame indices for each detected word center.
        word_mask: [B, max_words] boolean mask of detected words.
    """
    B, T, V = ctc_logits.shape
    device = ctc_logits.device
    probs = F.softmax(ctc_logits, dim=-1)

    blank_probs = probs[:, :, blank_id] if V > blank_id else torch.zeros((B, T), device=device)
    speech_energy = (1.0 - blank_probs).clamp(min=0.0, max=1.0)  # [B, T]
    space_probs = probs[:, :, space_id] if V > space_id else torch.zeros((B, T), device=device)

    word_centers = torch.zeros((B, max_words), dtype=torch.long, device=device)
    word_mask = torch.zeros((B, max_words), dtype=torch.bool, device=device)

    for b in range(B):
        energy = speech_energy[b].tolist()
        sp_probs = space_probs[b].tolist()

        detected_centers: List[int] = []
        in_word = False
        current_burst_frames: List[int] = []
        current_burst_weights: List[float] = []

        for t in range(T):
            e_t = energy[t]
            is_space = sp_probs[t] > 0.4

            if not in_word:
                if e_t >= energy_threshold and not is_space:
                    in_word = True
                    current_burst_frames = [t]
                    current_burst_weights = [e_t]
            else:
                # Word continuation or transition
                if is_space or e_t < silence_threshold:
                    # Word boundary reached
                    if len(current_burst_frames) >= min_frames_per_word:
                        w_sum = sum(current_burst_weights)
                        if w_sum > 1e-4:
                            center = int(sum(f * w for f, w in zip(current_burst_frames, current_burst_weights)) / w_sum)
                        else:
                            center = current_burst_frames[len(current_burst_frames) // 2]
                        detected_centers.append(min(center, T - 1))
                        if len(detected_centers) >= max_words:
                            break
                    in_word = False
                    current_burst_frames = []
                    current_burst_weights = []
                else:
                    current_burst_frames.append(t)
                    current_burst_weights.append(e_t)

        # Catch trailing word at end of audio
        if in_word and len(current_burst_frames) >= min_frames_per_word and len(detected_centers) < max_words:
            w_sum = sum(current_burst_weights)
            if w_sum > 1e-4:
                center = int(sum(f * w for f, w in zip(current_burst_frames, current_burst_weights)) / w_sum)
            else:
                center = current_burst_frames[len(current_burst_frames) // 2]
            detected_centers.append(min(center, T - 1))

        # Fill output tensor
        num_found = min(len(detected_centers), max_words)
        for i in range(num_found):
            word_centers[b, i] = detected_centers[i]
            word_mask[b, i] = True

    return word_centers, word_mask


def extract_streaming_ctc_slices(
    acoustic_memory: torch.Tensor,
    memory_lengths: Optional[torch.Tensor] = None,
    num_words: Optional[torch.Tensor] = None,
    ctc_logits: Optional[torch.Tensor] = None,
    forced_word_centers: Optional[torch.Tensor] = None,
    max_word_slots: int = 64,
    window_frames: int = 32,
    blank_id: int = 1,
    space_id: int = 8,
    return_durations: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Extract continuous acoustic slices with event-driven online temporal centers.

    Zero dependency on future duration or total sentence length.
    If forced_word_centers is provided (during training), uses exact alignment centers.
    Otherwise, tracks online CTC speech energy bursts in real time.

    Args:
        acoustic_memory: [B, T, D] speech representations.
        memory_lengths: [B] valid frames per batch item.
        num_words: [B] number of word slots.
        ctc_logits: Optional [B, T, V] CTC emission logits.
        forced_word_centers: Optional [B, L] ground-truth word centers.
        max_word_slots: Maximum word slots L.
        window_frames: Continuous window size (32 frames = 640ms).
        blank_id: CTC blank token ID.
        space_id: CTC space delimiter ID.
        return_durations: Whether to return physical inter-word durations.

    Returns:
        acoustic_slices: [B, L, S_ac, D]
        durations: Optional [B, L] physical duration in frames between consecutive word centers.
    """
    B, T, D = acoustic_memory.shape
    device = acoustic_memory.device
    S_ac = window_frames

    # Pad acoustic memory if shorter than window
    if T < S_ac:
        pad_len = S_ac - T
        acoustic_memory = F.pad(acoustic_memory, (0, 0, 0, pad_len))
        if ctc_logits is not None:
            ctc_logits = F.pad(ctc_logits, (0, 0, 0, pad_len))
        T = S_ac

    # Determine L
    if num_words is not None:
        L = min(max_word_slots, max(2, int(num_words.max().item())))
    else:
        L = min(max_word_slots, max(2, int(T / 12.0)))

    # Determine valid duration
    if memory_lengths is not None:
        t_valid = memory_lengths.clamp(min=1, max=T).float()
    else:
        t_valid = torch.full((B,), T, device=device, dtype=torch.float32)

    # 1. Obtain Word Centers
    if forced_word_centers is not None:
        t_center = forced_word_centers[:, :L].clamp(min=0, max=T - 1).float()
    elif ctc_logits is not None:
        # Online event-driven word peak detection
        online_centers, online_mask = detect_online_word_peaks(
            ctc_logits=ctc_logits,
            blank_id=blank_id,
            space_id=space_id,
            max_words=L,
        )
        t_center = online_centers.float()

        # For any word slot that had no detected peak, fallback gracefully
        unmatched = ~online_mask
        if unmatched.any():
            l_idx = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
            l_val = num_words.clamp(min=1, max=L).float().unsqueeze(1) if num_words is not None else float(L)
            t_fallback = ((l_idx + 0.5) / l_val) * t_valid.unsqueeze(1)
            t_center = torch.where(online_mask, t_center, t_fallback.clamp(min=0, max=T - 1))
    else:
        l_idx = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
        l_val = num_words.clamp(min=1, max=L).float().unsqueeze(1) if num_words is not None else float(L)
        t_center = (((l_idx + 0.5) / l_val) * t_valid.unsqueeze(1)).clamp(min=0, max=T - 1)

    # 2. Window bounds calculation
    half_w = S_ac // 2
    start_t = (t_center.round().long() - half_w).clamp(min=0, max=max(0, T - S_ac))

    # 3. Gather continuous speech slices: [B, L, S_ac, D]
    window_offsets = torch.arange(S_ac, device=device).view(1, 1, S_ac)
    frame_indices = start_t.unsqueeze(-1) + window_offsets  # [B, L, S_ac]

    acoustic_slices = torch.gather(
        acoustic_memory.unsqueeze(1).expand(-1, L, -1, -1),
        dim=2,
        index=frame_indices.unsqueeze(-1).expand(-1, -1, -1, D),
    )

    if return_durations:
        durations = torch.zeros((B, L), device=device, dtype=torch.float32)
        if L > 1:
            durations[:, :-1] = (t_center[:, 1:] - t_center[:, :-1]).clamp(min=1.0, max=60.0)
            durations[:, -1] = (t_valid.unsqueeze(1) - t_center[:, -1:]).squeeze(-1).clamp(min=1.0, max=60.0)
        else:
            durations[:, 0] = t_valid.clamp(min=1.0, max=60.0)
        return acoustic_slices, durations

    return acoustic_slices
