"""Comprehensive Unit Tests for Novel Speech Transformer Architectures.

Verifies:
1. ModelRegistry discovery and parameter parity for all 4 novel architectures.
2. PhonoHuBERT anti-blank margin regularization and calibrated decoding.
3. PhonoHuBERTDual frame-synchronous Cross-Entropy + sequence CTC dual-loss.
4. PhonoHuBERTHierarchical two-stage acoustic state router and phoneme head.
5. PhonoHuBERTRecursive autoregressive recurrent frame feedback head.
6. Data pipeline frame_targets generation and batch collation.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import torch.nn as nn

from src.models.registry import ModelRegistry, STANDARD_TIERS
from src.data.phoneme_tokenizer import PhonemeTokenizer
from src.data.target_extractors import PhonemeTargetExtractor
from src.data.threaded_dataset import collate_modular_batch


def test_registry_contains_all_architectures():
    """Verify all 4 novel architectures are registered in ModelRegistry."""
    catalog = ModelRegistry.list_models()
    model_ids = {m["id"] for m in catalog}
    expected = {
        "hubert_kmeans",
        "phono_hubert",
        "phono_hubert_dual",
        "phono_hubert_hierarchical",
        "phono_hubert_recursive",
    }
    assert expected.issubset(model_ids), f"Missing models in registry: {expected - model_ids}"


@pytest.mark.parametrize("arch", [
    "phono_hubert",
    "phono_hubert_dual",
    "phono_hubert_hierarchical",
    "phono_hubert_recursive",
])
def test_architecture_instantiation_and_parameter_parity(arch):
    """Verify each architecture builds with ~8.0-8.5M parameters in mini tier."""
    model = ModelRegistry.build_model(arch, tier="mini")
    num_params = sum(p.numel() for p in model.parameters())
    # Parity check: 7.5M to 9.0M parameters
    assert 7_500_000 <= num_params <= 9_000_000, f"{arch} has unexpected param count: {num_params:,}"


def test_phoneme_target_extractor_frame_targets():
    """Verify PhonemeTargetExtractor outputs valid frame_targets and frame_lengths."""
    extractor = PhonemeTargetExtractor()
    dummy_wav = torch.randn(16000 * 2)  # 2 seconds = 100 frames (hop 320)
    res = extractor.extract_targets(dummy_wav, text="hello world")

    assert "targets" in res
    assert "target_lengths" in res
    assert "frame_targets" in res
    assert "frame_lengths" in res

    num_frames = res["frame_lengths"].item()
    assert num_frames == 100
    assert res["frame_targets"].shape[0] == 100
    assert (res["frame_targets"] >= 0).all()


def test_batch_collation_with_frame_targets():
    """Verify collate_modular_batch correctly batches both sequence and frame targets."""
    item1 = {
        "audio": torch.randn(16000),
        "targets": torch.tensor([8, 9, 10, 6], dtype=torch.long),
        "target_length": torch.tensor(4, dtype=torch.long),
        "frame_targets": torch.tensor([3, 8, 8, 9, 10, 3], dtype=torch.long),
        "frame_length": torch.tensor(6, dtype=torch.long),
        "duration": 1.0,
        "text": "hi",
        "voice_name": "lessac",
    }
    item2 = {
        "audio": torch.randn(32000),
        "targets": torch.tensor([8, 12, 14, 15, 16, 6], dtype=torch.long),
        "target_length": torch.tensor(6, dtype=torch.long),
        "frame_targets": torch.tensor([3, 8, 12, 12, 14, 15, 16, 3], dtype=torch.long),
        "frame_length": torch.tensor(8, dtype=torch.long),
        "duration": 2.0,
        "text": "test",
        "voice_name": "lessac",
    }

    batch = collate_modular_batch([item1, item2])
    assert "frame_targets" in batch
    assert "frame_lengths" in batch
    assert batch["frame_targets"].shape == (2, 8)
    assert batch["frame_lengths"].tolist() == [6, 8]
    # Check padding is -100 for unpadded slots
    assert batch["frame_targets"][0, 6].item() == -100
    assert batch["frame_targets"][0, 7].item() == -100


def test_phono_hubert_anti_blank_regularization():
    """Verify PhonoHuBERT forward, anti-blank penalty loss, and decode_greedy."""
    config = ModelRegistry.build_config("phono_hubert", tier="mini", blank_penalty_weight=2.0)
    model = ModelRegistry.build_model("phono_hubert", tier="mini", config=config)
    model.train()

    dummy_audio = torch.randn(2, 16000)
    targets = torch.tensor([[8, 9, 10, 6], [11, 12, 13, 6]], dtype=torch.long)
    target_lengths = torch.tensor([4, 4], dtype=torch.long)

    out = model(audio=dummy_audio, targets=targets, target_lengths=target_lengths)
    assert "loss" in out
    assert out["loss"] is not None
    assert "ctc_loss" in out
    assert "blank_loss" in out

    out["loss"].backward()

    # Test decode_greedy with blank penalty
    model.eval()
    with torch.no_grad():
        preds_normal = model.decode_greedy(dummy_audio, blank_penalty=0.0)
        preds_penalized = model.decode_greedy(dummy_audio, blank_penalty=3.0)
        assert isinstance(preds_normal, list)
        assert isinstance(preds_penalized, list)


def test_phono_hubert_dual_forward_and_backward():
    """Verify PhonoHuBERTDual dual frame CE + sequence CTC losses and gradients."""
    config = ModelRegistry.build_config("phono_hubert_dual", tier="mini", frame_loss_weight=1.0, ctc_loss_weight=0.5)
    model = ModelRegistry.build_model("phono_hubert_dual", tier="mini", config=config)
    model.train()

    dummy_audio = torch.randn(2, 16000)  # ~50 frames
    targets = torch.tensor([[8, 9, 10, 6], [11, 12, 13, 6]], dtype=torch.long)
    target_lengths = torch.tensor([4, 4], dtype=torch.long)
    frame_targets = torch.full((2, 50), 8, dtype=torch.long)
    frame_lengths = torch.tensor([50, 50], dtype=torch.long)

    out = model(
        audio=dummy_audio,
        targets=targets,
        target_lengths=target_lengths,
        frame_targets=frame_targets,
        frame_lengths=frame_lengths,
    )

    assert "loss" in out
    assert "ce_loss" in out
    assert "ctc_loss" in out
    assert out["ce_loss"] > 0.0
    assert out["ctc_loss"] > 0.0

    out["loss"].backward()

    # Check gradients exist
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad

    # Greedy decode check
    model.eval()
    with torch.no_grad():
        preds_ctc = model.decode_greedy(dummy_audio, use_frame_head=False)
        preds_frame = model.decode_greedy(dummy_audio, use_frame_head=True)
        assert isinstance(preds_ctc, list)
        assert isinstance(preds_frame, list)


def test_phono_hubert_hierarchical_forward_and_backward():
    """Verify PhonoHuBERTHierarchical two-stage acoustic state router and phoneme head."""
    config = ModelRegistry.build_config("phono_hubert_hierarchical", tier="mini")
    model = ModelRegistry.build_model("phono_hubert_hierarchical", tier="mini", config=config)
    model.train()

    dummy_audio = torch.randn(2, 16000)
    targets = torch.tensor([[8, 9, 10, 6], [11, 12, 13, 6]], dtype=torch.long)
    target_lengths = torch.tensor([4, 4], dtype=torch.long)
    frame_targets = torch.full((2, 50), 8, dtype=torch.long)

    out = model(
        audio=dummy_audio,
        targets=targets,
        target_lengths=target_lengths,
        frame_targets=frame_targets,
    )

    assert "loss" in out
    assert "ctc_loss" in out
    assert "state_logits" in out
    assert "phoneme_logits" in out
    assert out["state_logits"].shape[-1] == 4

    out["loss"].backward()

    model.eval()
    with torch.no_grad():
        preds = model.decode_greedy(dummy_audio, blank_penalty=1.0)
        assert isinstance(preds, list)


def test_phono_hubert_recursive_forward_and_backward():
    """Verify PhonoHuBERTRecursive recurrent autoregressive feedback head unrolling."""
    config = ModelRegistry.build_config("phono_hubert_recursive", tier="mini", teacher_forcing_ratio=0.5)
    model = ModelRegistry.build_model("phono_hubert_recursive", tier="mini", config=config)
    model.train()

    dummy_audio = torch.randn(2, 16000)
    targets = torch.tensor([[8, 9, 10, 6], [11, 12, 13, 6]], dtype=torch.long)
    target_lengths = torch.tensor([4, 4], dtype=torch.long)
    frame_targets = torch.full((2, 50), 8, dtype=torch.long)

    out = model(
        audio=dummy_audio,
        targets=targets,
        target_lengths=target_lengths,
        frame_targets=frame_targets,
    )

    assert "loss" in out
    assert "ce_loss" in out
    assert "ctc_loss" in out
    assert out["logits"].shape[0] == 2
    assert out["logits"].shape[2] == config.vocab_size

    out["loss"].backward()

    # Autoregressive inference unroll
    model.eval()
    with torch.no_grad():
        preds = model.decode_greedy(dummy_audio, blank_penalty=0.5)
        assert isinstance(preds, list)
        assert len(preds) == 2
