"""Unit test suite for PhonemeTokenizer, ModelRegistry, PhonoHuBERT architecture, and Server API endpoints."""

import pytest
import torch
from fastapi.testclient import TestClient

from src.data.phoneme_tokenizer import PhonemeTokenizer
from src.data.target_extractors import PhonemeTargetExtractor, KMeansUnitExtractor
from src.models.registry import ModelRegistry, STANDARD_TIERS
from src.models.phono_hubert import PhonoHuBERTConfig, PhonoHuBERTForPreTraining
from src.models.hubert_asr import HuBERTForCTC
from src.server.app import app, state


def test_phoneme_tokenizer_special_tokens():
    tok = PhonemeTokenizer()
    assert tok.pad_token_id == 0
    assert tok.blank_token_id == 1
    assert tok.mask_token_id == 2
    assert tok.silence_token_id == 3
    assert tok.noise_token_id == 4
    assert tok.same_as_last_token_id == 5
    assert tok.eos_token_id == 6

    # Test encoding and decoding
    text = "hello world"
    ids_no_eos = tok.encode(text)
    assert len(ids_no_eos) > 0
    ids_with_eos = tok.encode(text, add_eos=True)
    assert ids_with_eos[-1] == tok.eos_token_id

    # Test repeat collapsing
    sample_seq = [10, 10, 10, 20, 20]
    collapsed = tok.collapse_repeats_to_special(sample_seq)
    assert collapsed == [10, 5, 5, 20, 5]
    expanded = tok.expand_repeats(collapsed)
    assert expanded == sample_seq


def test_model_registry_catalog_and_building():
    models = ModelRegistry.list_models()
    ids = [m["id"] for m in models]
    assert "hubert_kmeans" in ids
    assert "phono_hubert" in ids

    # Test building config with variable parameter overrides
    cfg = ModelRegistry.build_config("phono_hubert", tier="mini", encoder_layers=3, mask_prob=0.75)
    assert cfg.encoder_layers == 3
    assert cfg.mask_prob == 0.75

    # Test building model
    model = ModelRegistry.build_model("phono_hubert", cfg)
    assert isinstance(model, PhonoHuBERTForPreTraining)
    assert len(model.encoder.layers) == 3


def test_phono_hubert_forward_and_loss():
    config = ModelRegistry.build_config("phono_hubert", tier="mini", encoder_layers=2)
    model = PhonoHuBERTForPreTraining(config)
    model.eval()

    # 1 second of audio at 16kHz
    audio = torch.randn(2, 16000)
    out = model(audio)

    assert "logits" in out
    assert "output_lengths" in out
    assert "attentions" in out
    assert "hidden_states" in out
    assert out["logits"].shape[-1] == config.vocab_size

    # Test forward with target phoneme tokens
    model.train()
    targets = torch.tensor([[10, 15, 20, 6], [12, 14, 18, 6]], dtype=torch.long)
    target_lengths = torch.tensor([4, 4], dtype=torch.long)
    out_loss = model(audio, targets=targets, target_lengths=target_lengths)

    assert out_loss["loss"] is not None
    assert out_loss["loss"].item() > 0


def test_target_extractors():
    phono_ext = PhonemeTargetExtractor()
    dummy_audio = torch.randn(16000)
    res = phono_ext.extract_targets(dummy_audio, transcript="bonjour le monde")
    targets = res["targets"]
    assert targets.ndim == 1
    assert len(targets) > 0
    assert targets[-1].item() == phono_ext.tokenizer.eos_token_id


def test_server_catalog_and_selection_api():
    client = TestClient(app)

    # 1. Test /api/models/catalog
    resp = client.get("/api/models/catalog")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    assert "current" in data
    assert any(m["id"] == "phono_hubert" for m in data["models"])

    # 2. Test switching to phono_hubert
    resp_select = client.post("/api/models/select", json={"arch": "phono_hubert", "tier": "mini"})
    assert resp_select.status_code == 200
    sel_data = resp_select.json()
    assert sel_data["status"] == "ok"
    assert sel_data["arch"] == "phono_hubert"

    # 3. Test /api/status reflects new model
    st = client.get("/api/status").json()
    assert st["current_arch"] == "phono_hubert"

    # 4. Test /api/pretrain/preview for phono_hubert
    prev = client.get("/api/pretrain/preview?arch=phono_hubert").json()
    assert prev["target_type"] == "phonemes"
    assert "tokens" in prev
    assert len(prev["tokens"]) > 0

    # 5. Switch back to hubert_kmeans
    resp_back = client.post("/api/models/select", json={"arch": "hubert_kmeans", "tier": "mini"})
    assert resp_back.status_code == 200
