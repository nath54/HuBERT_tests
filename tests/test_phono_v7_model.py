"""Unit tests for Phono-V7 Composite Conformer Speech Model with Two-Level Character Decoding."""

from pathlib import Path
import pytest
import torch

from src.models.phono_v7_speech_model import (
    PhonoV7SpeechConfig,
    PhonoV7SpeechModel,
)


@pytest.fixture
def mini_model():
    cfg = PhonoV7SpeechConfig.medium()
    # Shrink layers for fast test execution
    cfg.encoder_config.encoder_layers = 2
    cfg.encoder_config.encoder_heads = 4
    cfg.encoder_config.encoder_embed_dim = 64
    cfg.encoder_config.encoder_ffn_dim = 128
    cfg.encoder_config.num_experts = 2
    cfg.encoder_config.moe_top_k = 1
    cfg.encoder_config.use_deep_refiner = False
    cfg.encoder_config.learnable_gating = False

    cfg.decoder_config.acoustic_dim = 64
    cfg.decoder_config.macro_dim = 64
    cfg.decoder_config.macro_layers = 2
    cfg.decoder_config.macro_heads = 4
    cfg.decoder_config.macro_ffn_dim = 128
    cfg.decoder_config.macro_num_experts = 2
    cfg.decoder_config.macro_moe_top_k = 1
    cfg.decoder_config.micro_dim = 64
    cfg.decoder_config.micro_layers = 2
    cfg.decoder_config.micro_heads = 4
    cfg.decoder_config.micro_ffn_dim = 128
    cfg.decoder_config.micro_num_experts = 2
    cfg.decoder_config.micro_moe_top_k = 1
    cfg.decoder_config.use_word_diffusion = False

    cfg.acoustic_window_frames = 16
    cfg.conformer_kernel_size = 5

    model = PhonoV7SpeechModel(cfg)
    return model


def test_phono_v7_forward_backward(mini_model):
    B, L, K = 2, 3, 8
    num_samples = 3200  # 200ms of 16kHz audio
    audio = torch.randn(B, num_samples)
    audio_lengths = torch.tensor([3200, 2400], dtype=torch.long)

    # Fake phoneme targets for CTC
    phoneme_targets = torch.tensor([[5, 8, 12, 0], [7, 9, 0, 0]], dtype=torch.long)
    phoneme_lengths = torch.tensor([3, 2], dtype=torch.long)

    input_byte_ids = torch.randint(2, 60, (B, L, K), dtype=torch.long)
    target_byte_ids = torch.randint(2, 60, (B, L, K), dtype=torch.long)
    path_targets = torch.tensor([[1, 2, 3], [1, 2, 0]], dtype=torch.long)
    target_lengths = torch.tensor([[4.0, 6.0, 3.0], [2.0, 5.0, -100.0]], dtype=torch.float32)
    num_words = torch.tensor([3, 2], dtype=torch.long)

    out = mini_model(
        audio=audio,
        audio_lengths=audio_lengths,
        phoneme_targets=phoneme_targets,
        phoneme_lengths=phoneme_lengths,
        num_words=num_words,
        input_byte_ids=input_byte_ids,
        target_byte_ids=target_byte_ids,
        path_targets=path_targets,
        target_lengths=target_lengths,
    )

    assert "loss" in out
    assert "char_loss" in out
    assert "enc_loss" in out
    assert "path_loss" in out
    assert "length_loss" in out
    assert "length_headroom" in out
    assert "char_acc" in out
    assert "logits" in out
    assert "k_hat" in out
    assert out["logits"].shape[:3] == (B, L, K)

    # Verify backward pass computes valid gradients across all modules
    loss = out["loss"]
    loss.backward()

    # Check gradient exists in Conformer depthwise conv
    conv_layer = mini_model.encoder.encoder.layers[0].conv_module.depthwise_conv
    assert conv_layer.weight.grad is not None
    assert not torch.isnan(conv_layer.weight.grad).any()

    # Check gradient exists in micro recursive head
    assert mini_model.decoder.micro_head.char_embedding.weight.grad is not None

    # Check gradient exists in word length predictor
    assert mini_model.decoder.length_predictor.net[0].weight.grad is not None


def test_asymmetric_length_loss():
    from src.models.phono_v7_speech_model import asymmetric_length_loss

    # Under-prediction: k_hat = 3.0, k_true = 5.0 -> diff = -2.0 -> loss = 4.0 * 2.0 = 8.0
    k_hat_under = torch.tensor([3.0])
    k_true = torch.tensor([5.0])
    loss_under, headroom_under = asymmetric_length_loss(k_hat_under, k_true, beta_under=4.0, beta_over=0.5, delta=1.0)
    assert torch.isclose(loss_under, torch.tensor(8.0))
    assert torch.isclose(headroom_under, torch.tensor(-2.0))

    # Safe over-prediction within delta=1.0 buffer: k_hat = 5.8, k_true = 5.0 -> diff = 0.8 <= 1.0 -> loss = 0.0
    k_hat_safe = torch.tensor([5.8])
    loss_safe, headroom_safe = asymmetric_length_loss(k_hat_safe, k_true, beta_under=4.0, beta_over=0.5, delta=1.0)
    assert torch.isclose(loss_safe, torch.tensor(0.0))
    assert torch.isclose(headroom_safe, torch.tensor(0.8))

    # Large over-prediction beyond delta=1.0: k_hat = 8.0, k_true = 5.0 -> diff = 3.0 -> loss = 0.5 * (3.0 - 1.0) = 1.0
    k_hat_large = torch.tensor([8.0])
    loss_large, headroom_large = asymmetric_length_loss(k_hat_large, k_true, beta_under=4.0, beta_over=0.5, delta=1.0)
    assert torch.isclose(loss_large, torch.tensor(1.0))
    assert torch.isclose(headroom_large, torch.tensor(3.0))


def test_phono_v7_decode_greedy(mini_model):
    audio = torch.randn(2, 3200)
    # Test Option B (Regression, default)
    mini_model.decoder.length_mode = "regression"
    res = mini_model.decode_greedy(audio=audio, max_words=3)
    assert len(res) == 2
    assert isinstance(res[0], list)

    # Test Option A (Categorical toggle)
    mini_model.decoder.length_mode = "categorical"
    res_cat = mini_model.decode_greedy(audio=audio, max_words=3)
    assert len(res_cat) == 2
    assert isinstance(res_cat[0], list)


def test_warm_start_real_checkpoints():
    enc_path = "checkpoints/phono_v6_4_gated_diffusion/medium/v6_4_960h/best_checkpoint.pt"
    dec_path = "checkpoints/phono_v6_7_speech/medium/best_checkpoint.pt"

    if not Path(enc_path).exists() or not Path(dec_path).exists():
        pytest.skip("Checkpoints not available on disk")

    cfg = PhonoV7SpeechConfig.medium()
    model = PhonoV7SpeechModel(cfg)

    ws_res = model.warm_start_from_checkpoints(
        encoder_checkpoint_path=enc_path,
        decoder_checkpoint_path=dec_path,
    )

    assert ws_res["encoder"] is not None
    assert ws_res["encoder"]["transferred"] > 100
    assert ws_res["decoder"] is not None
    assert ws_res["decoder"]["transferred"] > 100
