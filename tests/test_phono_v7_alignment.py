"""Unit tests for Phono-V7 Monotonic & CTC-Guided Temporal Alignment."""

import pytest
import torch
from src.models.phono_v7_alignment import extract_ctc_monotonic_slices


def test_monotonic_proportional_slicing_variable_lengths():
    B, T, D = 2, 300, 64
    acoustic_memory = torch.randn(B, T, D)
    memory_lengths = torch.tensor([100, 300], dtype=torch.long)
    num_words = torch.tensor([5, 10], dtype=torch.long)

    slices = extract_ctc_monotonic_slices(
        acoustic_memory=acoustic_memory,
        memory_lengths=memory_lengths,
        num_words=num_words,
        max_word_slots=10,
        window_frames=32,
    )
    assert slices.shape == (B, 10, 32, D)

    # Utterance 0 (length 100, 5 words): word 0 slice must NOT be at frame 150!
    # Instead, word 0 is at ~(0.5/5)*100 = 10 (window ~0..31)
    # word 4 is at ~(4.5/5)*100 = 90 (window ~74..105, clamped to 68..99)
    # Utterance 1 (length 300, 10 words): word 0 is at ~(0.5/10)*300 = 15, word 9 is at ~(9.5/10)*300 = 285
    # Distinct word slices guaranteed:
    diff = (slices[1, 9] - slices[1, 0]).abs().sum()
    assert diff > 0, "Word 0 and Word 9 must have distinct slices!"


def test_ctc_peak_refinement():
    B, T, D = 1, 100, 32
    acoustic_memory = torch.randn(B, T, D)
    memory_lengths = torch.tensor([100], dtype=torch.long)
    num_words = torch.tensor([2], dtype=torch.long)

    # Synthesize CTC logits with a strong speech burst at frame 35 (and blanks elsewhere)
    ctc_logits = torch.zeros(B, T, 10)
    ctc_logits[:, :, 1] = 5.0  # blank is high
    ctc_logits[:, 35, 2] = 20.0  # speech burst at t=35

    slices = extract_ctc_monotonic_slices(
        acoustic_memory=acoustic_memory,
        memory_lengths=memory_lengths,
        num_words=num_words,
        ctc_logits=ctc_logits,
        max_word_slots=2,
        window_frames=32,
    )
    assert slices.shape == (1, 2, 32, D)


def test_extract_ctc_monotonic_slices_with_durations():
    B, T, D = 2, 200, 32
    acoustic_memory = torch.randn(B, T, D)
    memory_lengths = torch.tensor([150, 200], dtype=torch.long)
    num_words = torch.tensor([4, 6], dtype=torch.long)

    slices, durations = extract_ctc_monotonic_slices(
        acoustic_memory=acoustic_memory,
        memory_lengths=memory_lengths,
        num_words=num_words,
        max_word_slots=6,
        window_frames=32,
        return_durations=True,
    )
    assert slices.shape == (B, 6, 32, D)
    assert durations.shape == (B, 6)
    # Durations should be strictly positive
    assert (durations > 0).all()

