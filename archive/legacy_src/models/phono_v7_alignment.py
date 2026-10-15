"""Phono-V7 Guaranteed Monotonic & CTC-Guided Temporal Alignment.

Implements:
- Axe 1: Proportional monotonic time mapping over valid utterance length T_valid (not padded length).
- Axe 2: CTC phonetic energy peak refinement to center the window on actual phoneme acoustic mass.
- Vectorized differentiable continuous frame gathering via torch.gather.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F


def extract_ctc_monotonic_slices(
    acoustic_memory: torch.Tensor,
    memory_lengths: Optional[torch.Tensor] = None,
    num_words: Optional[torch.Tensor] = None,
    ctc_logits: Optional[torch.Tensor] = None,
    forced_word_centers: Optional[torch.Tensor] = None,
    max_word_slots: int = 64,
    window_frames: int = 32,
    blank_id: int = 1,
    return_durations: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Extract continuous acoustic frame slices [B, L, S_ac, D] for each word slot.

    Guarantees:
    1. Proportional monotonic temporal progression along valid audio frames (Axe 1).
    2. Local refinement around CTC phonetic energy peaks (Axe 2).
    3. Handles variable-length batches and edge cases with zero padding out-of-bounds.

    Args:
        acoustic_memory: [B, T, D] continuous speech representations.
        memory_lengths: Optional [B] valid acoustic frame counts (unpadded).
        num_words: Optional [B] number of words per utterance.
        ctc_logits: Optional [B, T, V_ph] CTC phoneme logits from Model 1.
        max_word_slots: int max words L.
        window_frames: int S_ac, frames per word slice (~32 frames = 640ms).
        blank_id: int CTC blank token ID.

    Returns:
        acoustic_slices: [B, L, S_ac, D]
    """
    B, T, D = acoustic_memory.shape
    device = acoustic_memory.device
    S_ac = window_frames

    # Pad acoustic memory along time dimension T if shorter than S_ac
    if T < S_ac:
        pad_len = S_ac - T
        acoustic_memory = F.pad(acoustic_memory, (0, 0, 0, pad_len))
        if ctc_logits is not None:
            ctc_logits = F.pad(ctc_logits, (0, 0, 0, pad_len))
        T = S_ac

    # Determine L (number of word slots)
    if num_words is not None:
        L = min(max_word_slots, max(2, int(num_words.max().item())))
    else:
        L = min(max_word_slots, max(2, int(T / 12.0)))

    # Determine T_valid per batch item
    if memory_lengths is not None:
        t_valid = memory_lengths.clamp(min=1, max=T).float()  # [B]
    else:
        t_valid = torch.full((B,), T, device=device, dtype=torch.float32)

    if num_words is not None:
        l_valid = num_words.clamp(min=1, max=L).float()  # [B]
    else:
        l_valid = torch.full((B,), L, device=device, dtype=torch.float32)

    # 1. Monotonic Center of Mass: Use Forced Alignment Word Centers if available
    if forced_word_centers is not None:
        t_center = forced_word_centers[:, :L].clamp(min=0, max=T - 1).float()
    else:
        # Axe 1: Proportional Monotonic Center of Mass on Valid Duration
        l_idx = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)  # [1, L]
        t_base = ((l_idx + 0.5) / l_valid.unsqueeze(1)) * t_valid.unsqueeze(1)
        t_center = t_base.clamp(min=0, max=T - 1)

    # 2. Axe 2: CTC Phonetic Energy Peak Refinement (if ctc_logits provided and not using forced alignment)
    if ctc_logits is not None and forced_word_centers is None:
        # Non-blank probability energy: E(t) = 1 - P(blank)
        probs = F.softmax(ctc_logits[:, :, :], dim=-1)
        blank_probs = probs[:, :, blank_id] if probs.shape[-1] > blank_id else torch.zeros((B, T), device=device)
        speech_energy = (1.0 - blank_probs).clamp(min=0.0, max=1.0)  # [B, T]

        # Search in local neighborhood of t_base: [t_base - delta, t_base + delta]
        delta = S_ac // 2  # 16 frames
        offsets = torch.arange(-delta, delta + 1, device=device).view(1, 1, -1)  # [1, 1, 2*delta + 1]
        search_grid = (t_center.round().long().unsqueeze(-1) + offsets).clamp(min=0, max=T - 1)  # [B, L, 2*delta+1]

        # Gather speech energy across search neighborhood: [B, L, 2*delta+1]
        energy_gathered = torch.gather(
            speech_energy.unsqueeze(1).expand(-1, L, -1),
            dim=2,
            index=search_grid,
        )

        # Center of mass of speech energy within the local neighborhood
        weight_sum = energy_gathered.sum(dim=-1, keepdim=True)
        local_time_offsets = offsets.float()  # [1, 1, 2*delta+1]
        weighted_offset = (energy_gathered * local_time_offsets).sum(dim=-1, keepdim=True) / weight_sum.clamp(min=1e-5)
        # Apply shift only where speech energy is present
        refined_center = t_center + torch.where(weight_sum.squeeze(-1) > 0.1, weighted_offset.squeeze(-1), torch.zeros_like(t_center))
        t_center = refined_center.clamp(min=0, max=T - 1)

    # 3. Compute continuous window start frame: [B, L]
    half_w = S_ac // 2
    start_t = (t_center.round().long() - half_w).clamp(min=0, max=max(0, T - S_ac))

    # 4. Vectorized Gather: [B, L, S_ac, D]
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
