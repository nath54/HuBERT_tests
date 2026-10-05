"""Unit tests for MultilingualLexicon."""

import tempfile
from pathlib import Path
import pytest
import torch
from src.data.multilingual_lexicon import MultilingualLexicon


def test_lexicon_vocab_and_encoding():
    words = ["hello", "monde", "ciao", "mundo", "bonjour", "hello"]
    lex = MultilingualLexicon(words=words, embed_dim=768)

    # Check reserved tokens + unique words (5 special + 5 unique)
    assert lex.vocab_size == 10
    assert lex.PAD_ID == 0
    assert lex.BLANK_ID == 1

    encoded = lex.encode_words(["hello", "ciao", "nonexistent"])
    assert encoded[0] == lex.word_to_id["hello"]
    assert encoded[1] == lex.word_to_id["ciao"]
    assert encoded[2] == lex.UNK_ID

    decoded = lex.decode_ids(encoded)
    assert decoded == ["hello", "ciao", "<unk>"]


def test_lexicon_embeddings_and_loss():
    words = ["the", "cat", "sat", "on", "the", "mat"]
    lex = MultilingualLexicon(words=words, embed_dim=768)

    B, L, D = 2, 4, 768
    target_ids = torch.tensor([[lex.word_to_id["the"], lex.word_to_id["cat"], lex.word_to_id["sat"], lex.PAD_ID],
                               [lex.word_to_id["on"], lex.word_to_id["the"], lex.word_to_id["mat"], lex.PAD_ID]], dtype=torch.long)

    # 1. Perfect predictions: z_pred matching target embeddings
    target_embs = lex.lookup_target_embeddings(target_ids)
    loss, acc = lex.compute_lexical_loss(target_embs, target_ids)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    # For perfect normalized match, cosine loss = 0, InfoNCE is low
    assert loss.item() >= 0.0
    assert acc.item() == 100.0

    # 2. Gradient flow
    z_pred = torch.randn(B, L, D, requires_grad=True)
    loss_rand, _ = lex.compute_lexical_loss(z_pred, target_ids)
    loss_rand.backward()
    assert z_pred.grad is not None
    assert not torch.isnan(z_pred.grad).any()


def test_lexicon_save_and_load():
    words = ["arbre", "casa", "sun", "sole"]
    lex = MultilingualLexicon(words=words, embed_dim=768)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "lexicon.pt"
        lex.save(save_path)

        loaded_lex = MultilingualLexicon.load(save_path)
        assert loaded_lex.vocab_size == lex.vocab_size
        assert loaded_lex.embed_dim == lex.embed_dim
        assert loaded_lex.word_to_id == lex.word_to_id

        # Verify weights match
        assert torch.allclose(lex.embeddings.weight, loaded_lex.embeddings.weight)


def test_lexicon_predict_words():
    words = ["chien", "chat", "oiseau"]
    lex = MultilingualLexicon(words=words, embed_dim=768)

    target_ids = torch.tensor([[lex.word_to_id["chien"], lex.word_to_id["chat"]]], dtype=torch.long)
    target_embs = lex.lookup_target_embeddings(target_ids)

    preds = lex.predict_words(target_embs)
    assert preds == [["chien", "chat"]]
