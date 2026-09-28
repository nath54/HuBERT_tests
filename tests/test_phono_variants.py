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


def test_v6_3_diffusion_decoding():
    """Verify sliding window diffusion decoding on PhonoV63Diffusion."""
    config = ModelRegistry.build_config("phono_v6_3_diffusion", tier="mini")
    model = ModelRegistry.build_model("phono_v6_3_diffusion", tier="mini", config=config)
    model.eval()

    dummy_audio = torch.randn(2, 16000)
    with torch.no_grad():
        preds_diffusion = model.decode_diffusion(dummy_audio, num_steps=2)
        preds_beam = model.decode_beam(dummy_audio)
        assert len(preds_diffusion) == 2
        assert len(preds_beam) == 2
        assert isinstance(preds_diffusion[0], list)

