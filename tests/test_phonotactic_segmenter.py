"""Tests for PhonotacticLexiconSegmenter."""

import pytest
from src.data.phonotactic_segmenter import PhonotacticLexiconSegmenter


def test_segmenter_initialization():
    seg = PhonotacticLexiconSegmenter()
    assert len(seg.lexicon) > 100
    assert len(seg.known_words) > 100


def test_fused_clitic_splitting():
    seg = PhonotacticLexiconSegmenter()
    
    # English "there would" fused: [ð, ɛ, ɹ, w, ʊ, d]
    there_would = ["ð", "ɛ", "ɹ", "w", "ʊ", "d"]
    splits = seg.segment_phoneme_tokens(there_would)
    assert len(splits) == 2, f"Expected 2 segments for 'there would', got {splits}"
    assert splits[0] == ["ð", "ɛ", "ɹ"]
    assert splits[1] == ["w", "ʊ", "d"]


def test_single_word_preservation():
    seg = PhonotacticLexiconSegmenter()
    
    # "carrots": [k, æ, ɹ, ɪ, t, s] should NOT be split
    carrots = ["k", "æ", "ɹ", "ɪ", "t", "s"]
    splits = seg.segment_phoneme_tokens(carrots)
    assert len(splits) == 1, f"Expected 1 segment for 'carrots', got {splits}"
    assert splits[0] == carrots


def test_emission_tuple_segmentation():
    seg = PhonotacticLexiconSegmenter()
    
    # Create fake emissions for [ð, ɛ, ɹ, w, ʊ, d]
    tokens = ["ð", "ɛ", "ɹ", "w", "ʊ", "d"]
    emissions = []
    for t in tokens:
        tid = seg.ph_tok.token_to_id.get(t, 10)
        c_ids = [tid, tid + 1, tid + 2]
        c_probs = [0.8, 0.15, 0.05]
        emissions.append((c_ids, c_probs))
        
    result = seg.segment_word_emissions(emissions, seg.ph_tok.id_to_token)
    assert len(result) == 2
    assert len(result[0]) == 3
    assert len(result[1]) == 3
