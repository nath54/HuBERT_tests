"""Unit tests for the 5 Progressive Speech Transformer Variants (< 10% PER)."""

import pytest
import torch

from src.models.registry import ModelRegistry
from src.models.phono_variants import (
    PhonoV1FrontendForPreTraining,
    PhonoV2SpecAugmentForPreTraining,
    PhonoV3HybridForPreTraining,
    PhonoV4ScaledForPreTraining,
    PhonoV5BeamForPreTraining,
)


@pytest.mark.parametrize("arch", [
    "phono_v1_frontend",
    "phono_v2_specaugment",
    "phono_v3_hybrid",
    "phono_v4_scaled",
    "phono_v5_beam",
    "phono_v6_1_moe",
    "phono_v6_2_sparse",
    "phono_v6_3_diffusion",
])
def test_phono_variants_registry_and_instantiation(arch):
    """Verify all 5 progressive variants can be discovered and instantiated via ModelRegistry."""
    catalog = ModelRegistry.list_models()
    model_ids = {m["id"] for m in catalog}
    assert arch in model_ids, f"{arch} missing from ModelRegistry catalog"

    model = ModelRegistry.build_model(arch, tier="mini")
    assert isinstance(model, torch.nn.Module)
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 1_000_000, f"Unexpectedly small parameter count: {num_params}"


def test_v2_specaugment_masking():
    """Verify SpecAugment applies masking only during training and respects lengths."""
    config = ModelRegistry.build_config("phono_v2_specaugment", tier="mini")
    model = ModelRegistry.build_model("phono_v2_specaugment", tier="mini", config=config)

    # 1. Eval mode: no masking
    model.eval()
    features = torch.ones(2, 50, config.encoder_embed_dim)
    masked, mask = model.apply_specaugment(features)
    assert torch.all(masked == 1.0)
    assert not torch.any(mask)

    # 2. Train mode: masking applied
    model.train()
    masked_train, mask_train = model.apply_specaugment(features)
    assert torch.any(masked_train == 0.0)
    assert torch.any(mask_train)


def test_v2_specaugment_forward_backward_with_audio_lengths():
    """Verify forward and backward pass with audio_lengths parameter."""
    config = ModelRegistry.build_config("phono_v2_specaugment", tier="mini")
    model = ModelRegistry.build_model("phono_v2_specaugment", tier="mini", config=config)
    model.train()

    dummy_audio = torch.randn(2, 16000)
    audio_lengths = torch.tensor([16000, 12000], dtype=torch.long)
    targets = torch.tensor([[8, 9, 10, 6], [11, 12, 13, 6]], dtype=torch.long)
    target_lengths = torch.tensor([4, 4], dtype=torch.long)

    out = model(
        audio=dummy_audio,
        audio_lengths=audio_lengths,
        targets=targets,
        target_lengths=target_lengths,
    )

    assert "loss" in out
    assert out["loss"] is not None
    assert out["loss"].item() > 0
    assert "mask" in out

    out["loss"].backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad


def test_v5_beam_decoding():
    """Verify beam search decoding interface on PhonoV5Beam."""
    config = ModelRegistry.build_config("phono_v5_beam", tier="mini")
    model = ModelRegistry.build_model("phono_v5_beam", tier="mini", config=config)
    model.eval()

    dummy_audio = torch.randn(2, 16000)
    with torch.no_grad():
        preds_beam = model.decode_beam(dummy_audio)
        preds_greedy = model.decode_greedy(dummy_audio)
        assert len(preds_beam) == 2
        assert len(preds_greedy) == 2


def test_v3_hybrid_batch_generator():
    """Verify BufferedSpeechBatchGenerator with hybrid real speech manifest."""
    from src.data.streaming_piper import PiperVoiceManager, ProceduralTextSampler
    from src.data.target_extractors import PhonemeTargetExtractor
    from src.data.threaded_dataset import BufferedSpeechBatchGenerator

    vm = PiperVoiceManager()
    ts = ProceduralTextSampler()
    extractor = PhonemeTargetExtractor()

    manifest = "data/librispeech/librispeech_train.json"
    gen = BufferedSpeechBatchGenerator(
        voice_manager=vm,
        text_sampler=ts,
        target_extractor=extractor,
        batch_size=2,
        max_buffer_size=5,
        low_watermark=2,
        num_workers=2,
        use_rolling_pool=False,
        real_speech_manifest=manifest,
        real_ratio=1.0,  # Force real speech
    )
    try:
        batch = gen.get_batch(timeout=20.0)
        assert "audio" in batch
        assert "targets" in batch
        assert batch["audio"].shape[0] == 2
        assert any("human_" in v for v in batch["voices"])
    finally:
        gen.stop()


def test_v6_1_moe_forward_backward():
    """Verify MoE Top-2 routing, auxiliary load-balancing loss, and backward pass."""
    config = ModelRegistry.build_config("phono_v6_1_moe", tier="mini")
    model = ModelRegistry.build_model("phono_v6_1_moe", tier="mini", config=config)
    model.train()

    dummy_audio = torch.randn(2, 16000)
    audio_lengths = torch.tensor([16000, 14000], dtype=torch.long)
    targets = torch.tensor([[8, 9, 10, 6], [11, 12, 13, 6]], dtype=torch.long)
    target_lengths = torch.tensor([4, 4], dtype=torch.long)

    out = model(
        audio=dummy_audio,
        audio_lengths=audio_lengths,
        targets=targets,
        target_lengths=target_lengths,
    )

    assert "loss" in out
    assert out["loss"] is not None
    assert out["loss"].item() > 0
    assert "aux_loss" in out
    assert out["aux_loss"] > 0, "MoE auxiliary loss should be strictly positive"

    out["loss"].backward()
    # Check that expert router and experts received gradients
    has_router_grad = any(
        "router" in name and p.grad is not None and p.grad.abs().sum() > 0
        for name, p in model.named_parameters()
    )
    assert has_router_grad, "MoE router should receive gradients from aux_loss and CTC"


def test_v6_2_sparse_attention_window():
    """Verify SparseLocalSelfAttention applies local window masking."""
    from src.models.phono_variants import SparseLocalSelfAttention

    embed_dim = 64
    num_heads = 4
    window_size = 8
    attn = SparseLocalSelfAttention(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size)

    x = torch.randn(2, 50, embed_dim)
    out, weights = attn(x, return_attn_weights=True)

    assert out.shape == (2, 50, embed_dim)
    assert weights is not None
    # For any frame i and j where |i - j| > window_size, attention weight should be 0.0
    for i in range(50):
        for j in range(50):
            if abs(i - j) > window_size:
                assert weights[:, :, i, j].max().item() == 0.0, f"Attention leaked outside window at ({i}, {j})"


def test_v6_2_sparse_forward_backward_with_intermediate_ctc():
    """Verify PhonoV62Sparse forward, backward, and intermediate CTC multi-task loss."""
    config = ModelRegistry.build_config("phono_v6_2_sparse", tier="mini")
    # Mini tier has 4 layers, set inter_ctc_layers=(2,) for mini
    config.inter_ctc_layers = (2,)
    model = ModelRegistry.build_model("phono_v6_2_sparse", tier="mini", config=config)
    model.train()

    dummy_audio = torch.randn(2, 16000)
    audio_lengths = torch.tensor([16000, 14000], dtype=torch.long)
    targets = torch.tensor([[8, 9, 10, 6], [11, 12, 13, 6]], dtype=torch.long)
    target_lengths = torch.tensor([4, 4], dtype=torch.long)

    out = model(
        audio=dummy_audio,
        audio_lengths=audio_lengths,
        targets=targets,
        target_lengths=target_lengths,
    )

    assert "loss" in out
    assert out["loss"] is not None
    assert out["loss"].item() > 0
    assert "inter_ctc_loss" in out
    assert out["inter_ctc_loss"] > 0, "Intermediate CTC loss should be strictly positive"

    out["loss"].backward()
    # Check that layer 1 received gradients
    has_layer1_grad = any(
        "layers.0" in name and p.grad is not None and p.grad.abs().sum() > 0
        for name, p in model.named_parameters()
    )
    assert has_layer1_grad, "Layer 1 should receive backward gradients"


def test_v6_2_sparse_early_exit():
    """Verify adaptive layer-skipping inference on PhonoV62Sparse."""
    config = ModelRegistry.build_config("phono_v6_2_sparse", tier="mini")
    config.inter_ctc_layers = (2,)
    model = ModelRegistry.build_model("phono_v6_2_sparse", tier="mini", config=config)
    model.eval()

    dummy_audio = torch.randn(2, 16000)
    with torch.no_grad():
        exit_res = model.decode_early_exit(dummy_audio, confidence_threshold=0.50)
        assert "tokens" in exit_res
        assert len(exit_res["tokens"]) == 2
        assert "avg_exit_layer" in exit_res
        assert "compute_saved_pct" in exit_res
        assert 0.0 <= exit_res["compute_saved_pct"] <= 100.0


def test_v6_3_diffusion_components():
    """Verify GaussianNoiseScheduler and LatentDiffusionRefiner modules."""
    from src.models.phono_variants import GaussianNoiseScheduler, LatentDiffusionRefiner

    scheduler = GaussianNoiseScheduler(default_window_width=16, default_noise_max=0.8)
    B, T, D = 2, 64, 256
    device = torch.device("cpu")

    centers = torch.tensor([16.0, 32.0])
    noise_scales = torch.tensor([0.5, 0.8])
    noise_map, c_out, s_out = scheduler.compute_noise_map(
        batch_size=B, seq_len=T, device=device, centers=centers, noise_scales=noise_scales, window_width=16
    )

    assert noise_map.shape == (B, T, 1)
    # Peak should occur at the center frame
    assert torch.isclose(noise_map[0, 16, 0], torch.tensor(0.5), atol=1e-3)
    assert torch.isclose(noise_map[1, 32, 0], torch.tensor(0.8), atol=1e-3)
    # Beyond 2 sigma (32 frames), value should decay to < 14% of peak
    assert noise_map[0, 16 + 32, 0] < 0.15

    refiner = LatentDiffusionRefiner(embed_dim=D, dropout=0.0)
    refiner.eval()
    z = torch.randn(B, T, D)
    # At zero initialization, refiner out_proj is zero, so z_hat == z
    z_hat = refiner(z, noise_map)
    assert z_hat.shape == (B, T, D)
    assert torch.allclose(z_hat, z, atol=1e-5)


def test_v6_3_diffusion_training_forward():
    """Verify V6.3 training forward pass with multi-task diffusion and refined CTC loss."""
    config = ModelRegistry.build_config("phono_v6_3_diffusion", tier="mini")
    model = ModelRegistry.build_model("phono_v6_3_diffusion", tier="mini", config=config)
    model.train()

    dummy_audio = torch.randn(2, 16000)
    dummy_targets = torch.randint(1, 40, (2, 10))
    target_lengths = torch.tensor([10, 8])

    out = model(audio=dummy_audio, targets=dummy_targets, target_lengths=target_lengths)
    assert "loss" in out
    assert "diff_loss" in out
    assert "refined_ctc_loss" in out
    assert "refined_logits" in out
    assert out["loss"].requires_grad

    # Test backward pass to confirm gradient flow through both backbone and refiner
    out["loss"].backward()
    assert model.latent_refiner.out_proj.weight.grad is not None
    assert model.state_router.weight.grad is not None


def test_v6_3_diffusion_decoding():
    """Verify sliding window diffusion decoding and backward compatibility on PhonoV63Diffusion."""
    config = ModelRegistry.build_config("phono_v6_3_diffusion", tier="mini")
    model = ModelRegistry.build_model("phono_v6_3_diffusion", tier="mini", config=config)
    model.eval()

    dummy_audio = torch.randn(2, 16000)
    with torch.no_grad():
        preds_sliding = model.decode_sliding_diffusion(dummy_audio, num_steps=2, window_width=8, window_stride=8)
        preds_diffusion = model.decode_diffusion(dummy_audio, num_steps=2)
        preds_beam = model.decode_beam(dummy_audio)

        assert len(preds_sliding) == 2
        assert len(preds_diffusion) == 2
        assert len(preds_beam) == 2
        assert isinstance(preds_sliding[0], list)
        assert isinstance(preds_diffusion[0], list)


def test_v6_4_gated_diffusion():
    """Verify PhonoV64GatedDiffusion forward pass, confidence gating, and deep refiner."""
    from src.models.phono_variants import DeepLatentDiffusionRefiner

    config = ModelRegistry.build_config("phono_v6_4_gated_diffusion", tier="mini")
    assert config.gate_confidence_high == 0.8
    assert config.gate_confidence_low == 0.3
    assert config.use_deep_refiner is True

    model = ModelRegistry.build_model("phono_v6_4_gated_diffusion", tier="mini", config=config)
    assert isinstance(model.latent_refiner, DeepLatentDiffusionRefiner)

    model.train()
    dummy_audio = torch.randn(2, 16000)
    dummy_targets = torch.randint(1, 40, (2, 10))
    target_lengths = torch.tensor([10, 8])

    out = model(audio=dummy_audio, targets=dummy_targets, target_lengths=target_lengths)
    assert "loss" in out
    assert "diff_loss" in out
    assert "gate_bypass_pct" in out
    assert "gate_partial_pct" in out
    assert "gate_full_pct" in out
    assert out["loss"].requires_grad

    out["loss"].backward()
    assert model.latent_refiner.out_proj.weight.grad is not None
    if getattr(model, "learnable_gating", False) and hasattr(model, "gate_mlp"):
        assert model.gate_mlp[0].weight.grad is not None

    model.eval()
    with torch.no_grad():
        preds_gated = model.decode_gated_diffusion(dummy_audio, num_steps=2, window_width=8, window_stride=8)
        assert len(preds_gated) == 2
        assert isinstance(preds_gated[0], list)


def test_word_denoising_decoder():
    """Verify Phoneme-to-Word Denoising Decoder with Banded Cross-Attention & MoE Text Modeling."""
    from src.models.word_denoising_decoder import WordDecoderConfig, WordDenoisingDecoder

    cfg = WordDecoderConfig(vocab_size=500, word_embed_dim=128, acoustic_embed_dim=256, decoder_layers=2)
    decoder = WordDenoisingDecoder(cfg)
    decoder.train()

    B, L, T = 2, 8, 32
    dummy_words = torch.randint(1, 400, (B, L))
    dummy_targets = torch.randint(1, 400, (B, L))
    dummy_acoustic = torch.randn(B, T, 256)

    out = decoder(word_ids=dummy_words, acoustic_memory=dummy_acoustic, target_word_ids=dummy_targets)
    assert "loss" in out
    assert out["loss"] is not None
    assert out["loss"].item() > 0
    assert "diff_loss" in out
    assert "contrastive_loss" in out

    out["loss"].backward()
    assert decoder.word_embedding.weight.grad is not None



