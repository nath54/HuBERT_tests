import pytest
import torch
from src.models.phono_v7_5_speech_model import (
    compute_interval_validity_mask,
    compute_load_balancing_loss,
    LENGTH_INTERVALS,
    NUM_OVERLAPPING_EXPERTS,
    PhonoV75SpeechConfig,
    PhonoV75SpeechModel,
)


def test_compute_interval_validity_mask():
    # Length 4: should be valid for VERY_SHORT (1-5) and SHORT (3-7)
    lengths = torch.tensor([[4, 6, 9, 14, 25]])
    mask = compute_interval_validity_mask(lengths)
    assert mask.shape == (1, 5, NUM_OVERLAPPING_EXPERTS)

    # Word 0 (len=4): VERY_SHORT(0) and SHORT(1) are 1.0
    assert mask[0, 0, 0].item() == 1.0
    assert mask[0, 0, 1].item() == 1.0
    assert mask[0, 0, 2].item() == 0.0  # MEDIUM_SHORT (5-10) is 0

    # Word 1 (len=6): SHORT(1) and MEDIUM_SHORT(2) are 1.0
    assert mask[0, 1, 1].item() == 1.0
    assert mask[0, 1, 2].item() == 1.0

    # Word 2 (len=9): MEDIUM_SHORT(2), MEDIUM(3), MEDIUM_LARGE(4) are 1.0
    assert mask[0, 2, 2].item() == 1.0
    assert mask[0, 2, 3].item() == 1.0
    assert mask[0, 2, 4].item() == 1.0


def test_compute_load_balancing_loss():
    probs = torch.ones((10, NUM_OVERLAPPING_EXPERTS)) / NUM_OVERLAPPING_EXPERTS
    top1 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 1, 2])
    loss = compute_load_balancing_loss(probs, top1, NUM_OVERLAPPING_EXPERTS)
    assert loss >= 0.0
    assert torch.isfinite(loss)


def test_v7_5_model_forward():
    config = PhonoV75SpeechConfig.small()
    model = PhonoV75SpeechModel(config)
    model.eval()

    B = 2
    T = 16000  # 1 sec audio at 16kHz
    audio = torch.randn(B, T)
    audio_lengths = torch.tensor([T, T])
    num_words = torch.tensor([3, 4])
    input_bytes = torch.randint(1, 50, (B, 4, 8))
    target_bytes = torch.randint(1, 50, (B, 4, 8))
    target_lengths = torch.tensor([[3.0, 5.0, 8.0, 0.0], [4.0, 7.0, 12.0, 15.0]])

    with torch.no_grad():
        out = model(
            audio=audio,
            audio_lengths=audio_lengths,
            num_words=num_words,
            input_byte_ids=input_bytes,
            target_byte_ids=target_bytes,
            target_lengths=target_lengths,
        )

    assert "loss" in out
    assert "path_acc" in out
    assert "balance_loss" in out
    assert "z_word" in out
    assert out["path_acc"] >= 0.0
    print(f"Path accuracy: {out['path_acc'].item():.1f}%")
