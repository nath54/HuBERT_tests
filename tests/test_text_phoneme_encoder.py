import pytest
import torch
from src.models.text_phoneme_encoder import FastTextPhonemeWordEncoder
from src.models.latent_distillation import LatentSpaceDistillationLoss


def test_fast_text_phoneme_word_encoder():
    encoder = FastTextPhonemeWordEncoder(d_macro=512, d_model=128, nhead=4, num_layers=2)
    encoder.eval()

    N_words = 16
    K = 12  # max characters
    byte_ids = torch.randint(1, 100, (N_words, K))

    with torch.no_grad():
        z_pred_bytes = encoder(byte_ids=byte_ids)
    assert z_pred_bytes.shape == (N_words, 512)

    P = 8  # max phonemes
    phoneme_ids = torch.randint(1, 60, (N_words, P))
    with torch.no_grad():
        z_pred_ph = encoder(phoneme_ids=phoneme_ids)
    assert z_pred_ph.shape == (N_words, 512)


def test_latent_space_distillation_loss():
    loss_fn = LatentSpaceDistillationLoss(mse_weight=1.0, cos_weight=0.5, nce_weight=0.1)

    N = 8
    D = 512
    z_student = torch.randn(N, D, requires_grad=True)
    z_teacher = torch.randn(N, D)

    loss, metrics = loss_fn(z_student, z_teacher)
    assert loss > 0.0
    assert "mse" in metrics
    assert "cos_sim" in metrics
    assert "nce" in metrics

    loss.backward()
    assert z_student.grad is not None
