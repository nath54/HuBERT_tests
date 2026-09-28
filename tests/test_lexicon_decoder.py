"""Unit tests for Lexicon-Constrained CTC Decoder."""

import pytest
import torch
from src.decoder.lexicon_decoder import LexiconDecoder
from src.data.phoneme_tokenizer import PhonemeTokenizer


def test_lexicon_decoder_initialization():
    """Verify LexiconDecoder initializes and builds Trie."""
    decoder = LexiconDecoder(lexicon_path="data/librispeech-lexicon.txt", max_words=1000)
    assert len(decoder.root.children) > 0
    assert len(decoder.word_to_phones) > 0


def test_lexicon_decoder_utterance_decoding():
    """Verify LexiconDecoder produces valid words and phonemes from log_probs."""
    decoder = LexiconDecoder(lexicon_path="data/librispeech-lexicon.txt", max_words=1000)
    
    # 20 frames, vocab size 65
    log_probs = torch.randn(20, 65).log_softmax(dim=-1)
    res = decoder.decode_utterance(log_probs, beam_width=4)
    
    assert "words" in res
    assert "text" in res
    assert "phoneme_ids" in res
    assert "score" in res
    assert isinstance(res["words"], list)
    assert isinstance(res["text"], str)
    assert isinstance(res["phoneme_ids"], list)


def test_lexicon_decoder_batch():
    """Verify batch decoding on variable lengths."""
    decoder = LexiconDecoder(lexicon_path="data/librispeech-lexicon.txt", max_words=500)
    batch_lp = torch.randn(2, 25, 65).log_softmax(dim=-1)
    lengths = torch.tensor([25, 18], dtype=torch.long)
    
    batch_res = decoder.decode_batch(batch_lp, lengths=lengths, beam_width=4)
    assert len(batch_res) == 2
    assert isinstance(batch_res[0]["text"], str)
    assert isinstance(batch_res[1]["text"], str)
