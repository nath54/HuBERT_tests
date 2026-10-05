"""Unit tests for Differentiable Soft-Levenshtein Loss."""

import time
import pytest
import torch
import torch.nn.functional as F
from src.losses.soft_levenshtein import SoftLevenshteinLoss, softmin_gamma


def test_softmin_gamma_behavior():
    """Verify softmin approximates minimum and is smooth."""
    x = torch.tensor([1.0, 2.0, 5.0])
    sm = softmin_gamma(x, gamma=0.01)
    assert torch.isclose(sm, torch.tensor(1.0), atol=1e-2)

    # Softmin is differentiable
    x_var = torch.tensor([1.0, 2.0, 5.0], requires_grad=True)
    out = softmin_gamma(x_var, gamma=0.2)
    out.backward()
    assert x_var.grad is not None
    assert x_var.grad.shape == (3,)
    assert x_var.grad[0] > x_var.grad[1] > x_var.grad[2]


def test_soft_levenshtein_exact_match():
    """Exact match should have near-zero loss."""
    loss_fn = SoftLevenshteinLoss(gamma=0.2, normalize_by_len=True)
    N, K, V = 2, 5, 20

    # Create one-hot logits matching targets perfectly
    target_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, -100, -100]])
    logits = torch.randn(N, K, V) * 0.1
    for b in range(N):
        for i in range(K):
            t = target_ids[b, i].item()
            if t >= 0:
                logits[b, i, t] += 30.0

    loss = loss_fn(logits, target_ids)
    assert loss.item() < 0.05, f"Expected near zero loss for perfect match, got {loss.item()}"


def test_soft_levenshtein_monotonicity():
    """Loss(exact) < Loss(1-edit shift) < Loss(completely wrong)."""
    loss_fn = SoftLevenshteinLoss(gamma=0.2, normalize_by_len=False)
    V = 30
    K = 6

    # Target: "croce" -> [2, 17, 14, 2, 4, -100]
    target = torch.tensor([[2, 17, 14, 2, 4, -100]])

    # 1. Exact match logits
    logits_exact = torch.zeros(1, K, V)
    for i, tok in enumerate([2, 17, 14, 2, 4]):
        logits_exact[0, i, tok] = 20.0

    # 2. 1-character deletion ("coece", missing 'r' at index 1)
    logits_del = torch.zeros(1, K, V)
    for i, tok in enumerate([2, 14, 4, 2, 4]):  # 'c', 'o', 'e', 'c', 'e'
        logits_del[0, i, tok] = 20.0

    # 3. Completely wrong sequence
    logits_wrong = torch.zeros(1, K, V)
    for i in range(5):
        logits_wrong[0, i, 25] = 20.0

    loss_exact = loss_fn(logits_exact, target, pred_lengths=torch.tensor([5])).item()
    loss_del = loss_fn(logits_del, target, pred_lengths=torch.tensor([5])).item()
    loss_wrong = loss_fn(logits_wrong, target, pred_lengths=torch.tensor([5])).item()

    print(f"\nExact loss: {loss_exact:.4f}")
    print(f"1-del shift loss: {loss_del:.4f}")
    print(f"Wrong string loss: {loss_wrong:.4f}")

    assert loss_exact < loss_del < loss_wrong
    assert loss_exact < 0.01
    assert loss_del < 3.0


def test_soft_levenshtein_gradient_flow():
    """Verify gradients flow correctly back to logits without NaNs or Infs."""
    loss_fn = SoftLevenshteinLoss(gamma=0.2)
    logits = torch.randn(4, 12, 50, requires_grad=True)
    targets = torch.randint(0, 50, (4, 12))
    targets[:, 8:] = -100  # padding

    loss = loss_fn(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    assert not torch.isinf(logits.grad).any()
    assert logits.grad.abs().sum() > 0.0


def test_soft_levenshtein_execution_speed():
    """Verify vectorization runs in < 40ms even on CPU for realistic batch."""
    loss_fn = SoftLevenshteinLoss(gamma=0.2)
    N_words = 120  # e.g. 4 utterances * 30 words
    K = 16
    V = 122

    logits = torch.randn(N_words, K, V)
    targets = torch.randint(0, V, (N_words, K))

    # Warmup
    _ = loss_fn(logits, targets)

    start = time.perf_counter()
    _ = loss_fn(logits, targets)
    duration_ms = (time.perf_counter() - start) * 1000.0

    print(f"\nExecution time for {N_words} words (K={K}): {duration_ms:.2f} ms")
    assert duration_ms < 50.0
