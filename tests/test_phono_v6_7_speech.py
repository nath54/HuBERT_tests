"""Unit tests for Phono-V6.7 Joint Continuous Speech Model with Double Loss."""

import pytest
import torch
import torch.nn as nn

from src.models.phono_v6_7_speech_model import (
    PhonoV67SpeechConfig,
    PhonoV67SpeechModel,
)
from src.data.roman_tokenizer import RomanCharTokenizer
from src.data.multilingual_audio_dataset import (
    AudioUtterance,
    MultilingualAudioDataset,
    MultilingualBalancedBatchSampler,
    MultilingualSpeechCollator,
)


@pytest.fixture
def tiny_speech_model():
    """Create a lightweight speech model for fast testing."""
    config = PhonoV67SpeechConfig.medium()
    # Shrink layers for fast unit testing
    config.encoder_config.encoder_layers = 2
    config.encoder_config.encoder_heads = 4
    config.encoder_config.encoder_embed_dim = 256
    config.encoder_config.encoder_ffn_dim = 512
    config.encoder_config.num_experts = 4
    config.encoder_config.moe_top_k = 2

    config.decoder_config.acoustic_dim = 256
    config.decoder_config.macro_dim = 256
    config.decoder_config.macro_layers = 2
    config.decoder_config.macro_heads = 4
    config.decoder_config.macro_ffn_dim = 512
    config.decoder_config.micro_dim = 256
    config.decoder_config.micro_layers = 2
    config.decoder_config.micro_heads = 4
    config.decoder_config.micro_ffn_dim = 512
    config.decoder_config.micro_num_experts = 4
    config.decoder_config.micro_moe_top_k = 2

    return PhonoV67SpeechModel(config)


def test_speech_model_forward_double_loss(tiny_speech_model):
    """Verify forward pass computes double loss and gradients backprop to both encoder & decoder."""
    model = tiny_speech_model
    model.train()

    B = 2
    num_samples = 16000  # 1 second of 16kHz audio
    audio = torch.randn(B, num_samples)
    audio_lengths = torch.tensor([16000, 12000], dtype=torch.long)

    # Phoneme targets for Model 1 (CTC loss)
    phoneme_targets = torch.randint(1, 40, (B, 15), dtype=torch.long)
    phoneme_lengths = torch.tensor([15, 10], dtype=torch.long)

    # Character targets for Model 2 (Decoder)
    num_words = torch.tensor([3, 2], dtype=torch.long)
    input_byte_ids = torch.randint(0, 50, (B, 3, 10), dtype=torch.long)
    target_byte_ids = torch.randint(0, 50, (B, 3, 10), dtype=torch.long)
    path_targets = torch.tensor([[1, 2, 1], [2, 1, 0]], dtype=torch.long)

    out = model(
        audio=audio,
        audio_lengths=audio_lengths,
        phoneme_targets=phoneme_targets,
        phoneme_lengths=phoneme_lengths,
        num_words=num_words,
        input_byte_ids=input_byte_ids,
        target_byte_ids=target_byte_ids,
        path_targets=path_targets,
    )

    assert "loss" in out
    assert "enc_loss" in out
    assert "dec_loss" in out
    assert "path_acc" in out
    assert not torch.isnan(out["loss"])
    assert out["loss"].item() > 0.0

    # Backpropagation test
    out["loss"].backward()

    # Check encoder received gradients
    enc_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert enc_has_grad, "Encoder did not receive gradients from double loss!"

    # Check decoder received gradients
    dec_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.decoder.parameters())
    assert dec_has_grad, "Decoder did not receive gradients!"


def test_speech_model_greedy_decode(tiny_speech_model):
    """Verify greedy character decoding directly from audio."""
    model = tiny_speech_model
    model.eval()

    audio = torch.randn(2, 16000)
    audio_lengths = torch.tensor([16000, 16000], dtype=torch.long)

    with torch.no_grad():
        decoded_batch = model.decode_greedy(
            audio=audio,
            audio_lengths=audio_lengths,
            max_words=3,
            max_bytes_per_word=8,
        )

    assert len(decoded_batch) == 2
    for utt in decoded_batch:
        assert isinstance(utt, list)
        for word in utt:
            assert isinstance(word, list)


def test_balanced_multilingual_sampler():
    """Verify balanced batch sampler distributes languages equally."""
    utterances = [
        AudioUtterance("en_1", "dummy.wav", "hello world", "en"),
        AudioUtterance("en_2", "dummy.wav", "how are you", "en"),
        AudioUtterance("it_1", "dummy.wav", "ciao mondo", "it"),
        AudioUtterance("it_2", "dummy.wav", "buongiorno a tutti", "it"),
        AudioUtterance("es_1", "dummy.wav", "hola mundo", "es"),
        AudioUtterance("es_2", "dummy.wav", "buenas tardes", "es"),
        AudioUtterance("fr_1", "dummy.wav", "bonjour le monde", "fr"),
        AudioUtterance("fr_2", "dummy.wav", "comment allez vous", "fr"),
    ]

    sampler = MultilingualBalancedBatchSampler(
        utterances=utterances,
        batch_size=4,
        languages=["en", "it", "es", "fr"],
        seed=123,
    )

    batches = list(sampler)
    assert len(batches) > 0
    # In each batch of size 4 with 4 languages, each language should appear exactly once
    first_batch = batches[0]
    batch_langs = [utterances[idx].lang for idx in first_batch]
    assert set(batch_langs) == {"en", "it", "es", "fr"}
