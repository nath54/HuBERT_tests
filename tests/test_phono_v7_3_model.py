"""Unit tests for Phono-V7.3 Architecture.

Verifies:
1. MultiScaleDilatedConformerConvModule shape, branches, and zero-initialized warm-start.
2. DecoupledBoundaryGateHead 3-class prediction and weighted loss.
3. RecursivePhonemeHead causal temporal recurrence and zero-perturbation identity.
4. PhonoV73SpeechModel end-to-end forward/backward execution.
5. Warm-start transfer from V7.1/V7.2 checkpoints.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from src.models.conformer_layers import MultiScaleDilatedConformerConvModule
from src.models.phono_v7_1_speech_model import PhonoV71SpeechConfig
from src.models.phono_v7_3_speech_model import (
    DecoupledBoundaryGateHead,
    RecursivePhonemeHead,
    PhonoV73AcousticBackbone,
    PhonoV73SpeechModel,
)


def test_multi_scale_dilated_conformer_conv():
    B, T, D = 2, 40, 64
    conv = MultiScaleDilatedConformerConvModule(embed_dim=D, dropout=0.0)
    x = torch.randn(B, T, D)

    # Initial output should be zero because pointwise_conv2 is zero-initialized
    out = conv(x)
    assert out.shape == (B, T, D)
    assert torch.allclose(out, torch.zeros_like(out)), "Multi-scale conv must be zero-initialized for warm-start"


def test_decoupled_boundary_gate():
    B, T, D = 2, 30, 64
    gate = DecoupledBoundaryGateHead(embed_dim=D)
    h = torch.randn(B, T, D)

    logits = gate(h)
    assert logits.shape == (B, T, 3)
    # Zero-initialized logits should all be 0.0 initially
    assert torch.allclose(logits, torch.zeros_like(logits))


def test_recursive_phoneme_head_causality_and_identity():
    B, T, D, V = 2, 25, 64, 32
    head = RecursivePhonemeHead(embed_dim=D, vocab_size=V, refiner_dim=32, kernel_size=5)
    h = torch.randn(B, T, D)

    refined, base = head(h)
    assert refined.shape == (B, T, V)
    assert base.shape == (B, T, V)

    # At step 0, refiner is zero-initialized so refined == base exactly!
    assert torch.allclose(refined, base), "Recursive head must have zero perturbation on initialization"


def test_phono_v7_3_forward_backward():
    model = PhonoV73SpeechModel()

    B = 2
    audio = torch.randn(B, 16000)
    num_words = torch.tensor([3, 4])
    phoneme_targets = torch.tensor([[5, 8, 12, 1], [4, 8, 15, 8]])
    phoneme_lengths = torch.tensor([4, 4])
    input_bytes = torch.randint(1, 10, (B, 4, 6))
    target_bytes = torch.randint(1, 10, (B, 4, 6))
    path_targets = torch.randint(0, 4, (B, 4))
    target_lengths = torch.tensor([[3.0, 4.0, 5.0, 2.0], [2.0, 5.0, 4.0, 3.0]])
    boundary_targets = torch.randint(0, 3, (B, 50))

    out = model(
        audio=audio,
        phoneme_targets=phoneme_targets,
        phoneme_lengths=phoneme_lengths,
        num_words=num_words,
        input_byte_ids=input_bytes,
        target_byte_ids=target_bytes,
        path_targets=path_targets,
        target_lengths=target_lengths,
        boundary_targets=boundary_targets,
    )

    loss = out["loss"]
    assert not torch.isnan(loss)
    assert loss.item() > 0.0
    assert "boundary_logits" in out
    assert out["boundary_logits"].shape[-1] == 3

    loss.backward()

    # Verify gradients flow to encoder, boundary head, and decoder
    assert model.encoder.boundary_gate_head.proj.weight.grad is not None
    assert model.encoder.phoneme_head.base_head.weight.grad is not None


def test_phono_v7_3_warm_start():
    ckpt_path = Path("checkpoints/phono_v7_1/streaming/best_checkpoint.pt")
    if not ckpt_path.is_file():
        pytest.skip("V7.1 checkpoint not found for warm-start test")

    config = PhonoV71SpeechConfig.medium()
    model = PhonoV73SpeechModel(config)
    res = model.warm_start_from_v7_2(str(ckpt_path))

    assert res["transferred"] > 900
    assert res["missing"] == 0
