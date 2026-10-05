"""Unit tests for Phono-V7.1 Real-Time Streaming Speech Model."""

from pathlib import Path
import pytest
import torch

from src.models.phono_v7_1_alignment import detect_online_word_peaks, extract_streaming_ctc_slices
from src.models.phono_v7_1_speech_model import (
    PhonoV71SpeechConfig,
    PhonoV71SpeechModel,
    build_band_causal_mask,
)


def test_build_band_causal_mask():
    L, K = 10, 3
    device = torch.device("cpu")
    mask = build_band_causal_mask(L=L, K=K, device=device)

    assert mask.shape == (L, L)
    # Diagonal is allowed (self-attention)
    for i in range(L):
        assert mask[i, i] == 0.0

    # Future positions are masked (-inf)
    assert mask[0, 1] == float("-inf")
    assert mask[3, 5] == float("-inf")

    # History within K=3 is allowed (0.0)
    # Query at index 5 can attend to 5, 4, 3
    assert mask[5, 5] == 0.0
    assert mask[5, 4] == 0.0
    assert mask[5, 3] == 0.0

    # History older than K=3 is blocked (-inf)
    assert mask[5, 2] == float("-inf")
    assert mask[5, 1] == float("-inf")
    assert mask[5, 0] == float("-inf")


def test_detect_online_word_peaks():
    B, T, V = 1, 60, 64
    ctc_logits = torch.full((B, T, V), -10.0)
    # Set blank (index 1) to high prob everywhere initially
    ctc_logits[:, :, 1] = 5.0

    # Inject word 1 energy burst: frames 10..15 non-blank (e.g. token 5)
    ctc_logits[:, 10:16, 1] = -5.0
    ctc_logits[:, 10:16, 5] = 5.0

    # Inject space delimiter: frame 20 space (token 8)
    ctc_logits[:, 20, 1] = -5.0
    ctc_logits[:, 20, 8] = 5.0

    # Inject word 2 energy burst: frames 25..32 non-blank (token 12)
    ctc_logits[:, 25:33, 1] = -5.0
    ctc_logits[:, 25:33, 12] = 5.0

    word_centers, word_mask = detect_online_word_peaks(
        ctc_logits=ctc_logits,
        blank_id=1,
        space_id=8,
        min_frames_per_word=3,
        max_words=10,
    )

    assert word_mask[0, 0].item() is True
    # Center of 10..15 is ~12
    assert 10 <= word_centers[0, 0].item() <= 15

    assert word_mask[0, 1].item() is True
    # Center of 25..32 is ~28
    assert 25 <= word_centers[0, 1].item() <= 32


@pytest.fixture
def mini_v71_model():
    cfg = PhonoV71SpeechConfig.medium()
    # Fast test configuration
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
    cfg.band_window_words = 4

    return PhonoV71SpeechModel(cfg)


def test_phono_v7_1_forward_backward(mini_v71_model):
    B, L, K = 2, 5, 8
    num_samples = 3200
    audio = torch.randn(B, num_samples)
    audio_lengths = torch.tensor([3200, 2400], dtype=torch.long)

    phoneme_targets = torch.tensor([[5, 8, 12, 8, 4], [7, 8, 9, 0, 0]], dtype=torch.long)
    phoneme_lengths = torch.tensor([5, 3], dtype=torch.long)

    input_byte_ids = torch.randint(2, 60, (B, L, K), dtype=torch.long)
    target_byte_ids = torch.randint(2, 60, (B, L, K), dtype=torch.long)
    path_targets = torch.tensor([[1, 2, 3, 1, 2], [1, 2, 0, -100, -100]], dtype=torch.long)
    target_lengths = torch.tensor([[4.0, 6.0, 3.0, 5.0, 2.0], [2.0, 5.0, 1.0, -100.0, -100.0]], dtype=torch.float32)
    num_words = torch.tensor([5, 3], dtype=torch.long)

    out = mini_v71_model(
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
    assert "length_loss" in out
    assert "length_headroom" in out
    assert "char_acc" in out
    assert "logits" in out
    assert out["logits"].shape[:3] == (B, L, K)

    loss = out["loss"]
    loss.backward()

    # Verify gradients flow into Conformer depthwise conv
    conv_layer = mini_v71_model.encoder.encoder.layers[0].conv_module.depthwise_conv
    assert conv_layer.weight.grad is not None

    # Verify gradients flow into shared word query
    assert mini_v71_model.decoder.word_query_base.grad is not None

    # Verify gradients flow into length predictor
    assert mini_v71_model.decoder.length_predictor.net[0].weight.grad is not None


def test_phono_v7_1_decode_greedy(mini_v71_model):
    audio = torch.randn(2, 3200)
    res = mini_v71_model.decode_greedy(audio=audio, max_words=4)
    assert len(res) == 2
    assert isinstance(res[0], list)


def test_phono_v7_1_warm_start():
    v7_ckpt_path = "checkpoints/phono_v7/char/best_checkpoint.pt"
    if not Path(v7_ckpt_path).exists():
        pytest.skip("V7 checkpoint not found on disk")

    cfg = PhonoV71SpeechConfig.medium()
    model = PhonoV71SpeechModel(cfg)

    ws_res = model.warm_start_from_v7(v7_ckpt_path)
    assert ws_res["transferred"] > 800
    assert ws_res["unexpected"] == 0
