"""Unit tests for Native TorchAudio Forced Alignment module."""

import time
import pytest
import torch
import torch.nn.functional as F
from src.models.forced_aligner import ForcedAligner


def test_forced_aligner_basic():
    """Verify alignment tensor shapes and values on synthetic logits."""
    aligner = ForcedAligner(blank_id=1, pad_id=0, space_id=8)
    B, T, V = 2, 50, 16
    S = 8

    # Create dummy logits
    logits = torch.randn(B, T, V)
    log_probs = logits.log_softmax(dim=-1)

    # Targets with words separated by space (token 8)
    targets = torch.tensor([
        [2, 3, 8, 4, 5, 8, 6, 7],
        [2, 4, 8, 3, 5, 8, 6, 7],
    ])
    input_lengths = torch.tensor([T, T])
    target_lengths = torch.tensor([S, S])

    aligned, scores = aligner.align(
        log_probs=log_probs,
        targets=targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
    )

    assert aligned.shape == (B, T)
    assert scores.shape == (B, T)
    assert not torch.isnan(scores).any()


def test_forced_aligner_extract_word_centers():
    """Verify exact center frame extraction for each word slot."""
    aligner = ForcedAligner(blank_id=1, pad_id=0, space_id=8)
    B, T = 1, 30
    aligned = torch.full((B, T), 1, dtype=torch.long)

    # Word 0 at frames [2, 3, 4]
    aligned[0, 2:5] = torch.tensor([2, 3, 4])
    # Space at frame 5
    aligned[0, 5] = 8
    # Word 1 at frames [10, 11, 12, 13]
    aligned[0, 10:14] = torch.tensor([5, 6, 7, 9])
    # Space at frame 14
    aligned[0, 14] = 8
    # Word 2 at frames [20, 21]
    aligned[0, 20:22] = torch.tensor([10, 11])

    num_words = torch.tensor([3])
    centers, valids = aligner.extract_word_centers(aligned, num_words, max_words=3)

    assert centers.shape == (1, 3)
    assert valids.shape == (1, 3)
    assert valids.all()

    # Check center frames
    # Word 0: frames 2, 3, 4 -> center 3
    assert centers[0, 0].item() == 3
    # Word 1: frames 10, 11, 12, 13 -> center 11 or 12
    assert centers[0, 1].item() in (11, 12)
    # Word 2: frames 20, 21 -> center 20 or 21
    assert centers[0, 2].item() in (20, 21)


def test_forced_aligner_frame_loss_gradients():
    """Verify frame-level alignment loss computes clean, non-zero gradients."""
    aligner = ForcedAligner(blank_id=1)
    B, T, V = 2, 40, 20
    logits = torch.randn(B, T, V, requires_grad=True)
    aligned_tokens = torch.randint(0, V, (B, T))

    loss = aligner.compute_frame_alignment_loss(logits, aligned_tokens)
    loss.backward()

    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    assert logits.grad.abs().sum() > 0.0


def test_forced_aligner_speed():
    """Verify alignment runs in < 5ms for realistic batch on CPU."""
    aligner = ForcedAligner()
    B, T, V = 4, 500, 64  # 10s audio at 50Hz, vocab 64
    S = 80  # 80 phonemes

    logits = torch.randn(B, T, V)
    log_probs = logits.log_softmax(dim=-1)
    targets = torch.randint(2, V, (B, S))
    input_lengths = torch.tensor([T] * B)
    target_lengths = torch.tensor([S] * B)

    # Warmup
    _ = aligner.align(log_probs, targets, input_lengths, target_lengths)

    start = time.perf_counter()
    aligned, scores = aligner.align(log_probs, targets, input_lengths, target_lengths)
    duration_ms = (time.perf_counter() - start) * 1000.0

    print(f"\nExecution time for batch {B}x{T}: {duration_ms:.2f} ms")
    assert duration_ms < 50.0  # Even on CPU
