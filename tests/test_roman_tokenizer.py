"""Unit tests for RomanCharTokenizer."""

import pytest
from src.data.roman_tokenizer import RomanCharTokenizer


def test_roman_char_tokenizer_basic():
    tok = RomanCharTokenizer()
    assert tok.vocab_size == 122

    # Test specials
    assert tok.special_tokens[tok.pad_id] == "<pad>"
    assert tok.special_tokens[tok.blank_id] == "<blank>"
    assert tok.special_tokens[tok.bos_id] == "<bos>"
    assert tok.special_tokens[tok.eos_id] == "<eos>"
    assert tok.special_tokens[tok.eow_id] == "<eow>"
    assert tok.special_tokens[tok.unk_id] == "<unk>"


def test_roman_char_tokenizer_multilingual():
    tok = RomanCharTokenizer()
    
    samples = [
        ("English", "He hoped there would be stew for dinner 123!"),
        ("French", "Le renard brun rapide saute par-dessus le chien endormi, où sont les élèves?"),
        ("Spanish", "El rápido zorro marrón salta sobre el perro perezoso. ¡Hola! ¿Cómo estás hoy?"),
        ("German", "Der schnelle braune Fuchs springt über den faulen Hund in der Straße 42."),
        ("Italian", "La pasta al pomodoro è deliziosa, così com'è a città di Roma!"),
        ("Chinese Pinyin", "Ni3 hao3 shi4 jie4, zhe4 shi4 yi2 ge4 ce4 shi4."),
        ("Japanese Romaji", "Konnichiha sekai , koreha tesuto desu ."),
        ("Korean Romaja", "Annyeonghaseyo segye, igeoseun teseuteuipnida."),
        ("Arabic Arabizi", "mr7ba b-al3alam hadha ikhtibar 100%"),
    ]

    for lang, text in samples:
        enc = tok.encode(text, add_bos=True, add_eos=True)
        assert enc[0] == tok.bos_id
        assert enc[-1] == tok.eos_id
        assert tok.unk_id not in enc, f"Unexpected UNK in {lang}: '{text}'"

        dec = tok.decode(enc, skip_special=True)
        expected = tok.normalize(text)
        assert dec == expected, f"Roundtrip failed for {lang}: '{expected}' != '{dec}'"


def test_roman_char_tokenizer_word_encoding():
    tok = RomanCharTokenizer()
    text = "Hello World Café 100%"
    words = tok.encode_words(text)
    assert len(words) == 4
    for w_seq in words:
        assert w_seq[-1] == tok.eow_id

    dec_text = tok.decode_words(words)
    assert dec_text == "hello world café 100%"


def test_roman_char_tokenizer_typography_normalization():
    tok = RomanCharTokenizer()
    raw = "“AudioLearn’s model” — version 6.5…"
    enc = tok.encode(raw)
    assert tok.unk_id not in enc
    dec = tok.decode(enc)
    assert dec == '"audiolearn\'s model" - version 6.5...'
