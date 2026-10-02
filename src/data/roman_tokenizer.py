"""Unified Lowercase Roman Character Tokenizer for Multilingual Edge Speech Recognition.

Covers (all normalized to lowercase):
- Control tokens (<pad>, <blank>, <bos>, <eos>, <eow>, <unk>)
- Digits: 0-9
- Lowercase Latin: a-z
- Standard ASCII punctuation and math/typography symbols (excluding space, which is strictly <eow>)
- Spanish punctuation: ¡, ¿
- Accented Latin vowels and consonants (French, Spanish, German, Italian):
  à, á, â, ã, ä, å, æ, ç, è, é, ê, ë, ì, í, î, ï, ð, ñ, ò, ó, ô, õ, ö, ø, ù, ú, û, ü, ý, þ, ÿ, ß, œ
- Pinyin vowels with tone marks: ā, á, ǎ, à, ē, é, ě, è, ī, í, ǐ, ì, ō, ó, ǒ, ò, ū, ú, ǔ, ù, ǖ, ǘ, ǚ, ǜ.

Vocab size: 122 tokens (~125 KB embedding table in FP16 at d=512).
Spaces are strictly represented by the <eow> word-boundary token, completely eliminating
whitespace duplication, intra-word spacing, and trailing whitespace artifacts.
"""

from typing import Dict, List, Optional, Set, Union
import unicodedata


class RomanCharTokenizer:
    """Compact Lowercase Roman Character Tokenizer with dedicated control tokens."""

    PAD_ID = 0
    BLANK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    EOW_ID = 4  # End of Word / Word separator
    UNK_ID = 5

    # Typographic normalization mapping for web/corpus text robustness
    NORM_MAP = {
        "’": "'",
        "‘": "'",
        "`": "'",
        "“": '"',
        "”": '"',
        "«": '"',
        "»": '"',
        "–": "-",
        "—": "-",
        "―": "-",
        "…": "...",
        "\xa0": " ",   # Non-breaking space
        "\u200b": "",  # Zero-width space
    }

    def __init__(self, normalize_quotes_and_dashes: bool = True):
        self.pad_id = self.PAD_ID
        self.blank_id = self.BLANK_ID
        self.bos_id = self.BOS_ID
        self.eos_id = self.EOS_ID
        self.eow_id = self.EOW_ID
        self.unk_id = self.UNK_ID
        self.normalize_quotes_and_dashes = normalize_quotes_and_dashes

        self.special_tokens: Dict[int, str] = {
            self.pad_id: "<pad>",
            self.blank_id: "<blank>",
            self.bos_id: "<bos>",
            self.eos_id: "<eos>",
            self.eow_id: "<eow>",
            self.unk_id: "<unk>",
        }

        # Build ordered unique character list (lowercase only, NO whitespace character)
        chars = []

        # 1. Digits
        chars.extend([str(i) for i in range(10)])

        # 2. Lowercase ASCII
        chars.extend([chr(c) for c in range(ord("a"), ord("z") + 1)])

        # 3. Standard ASCII punctuation & math symbols (excluding space)
        ascii_punct = [
            "'", '"', ",", ".", "-", "?", "!", ":", ";",
            "(", ")", "[", "]", "{", "}", "/", "\\", "@", "#",
            "$", "%", "^", "&", "*", "_", "+", "=", "<", ">",
            "~", "|",
        ]
        chars.extend(ascii_punct)

        # 4. Spanish punctuation
        chars.extend(["¡", "¿"])

        # 5. Latin Accented Lowercase (French, Spanish, German, Italian)
        accented_lower = [
            "à", "á", "â", "ã", "ä", "å", "æ", "ç",
            "è", "é", "ê", "ë", "ì", "í", "î", "ï",
            "ð", "ñ", "ò", "ó", "ô", "õ", "ö", "ø",
            "ù", "ú", "û", "ü", "ý", "þ", "ÿ", "ß", "œ",
        ]
        chars.extend(accented_lower)

        # 6. Pinyin tones with diacritics
        pinyin_diacritics = [
            "ā", "á", "ǎ", "à",
            "ē", "é", "ě", "è",
            "ī", "í", "ǐ", "ì",
            "ō", "ó", "ǒ", "ò",
            "ū", "ú", "ǔ", "ù",
            "ǖ", "ǘ", "ǚ", "ǜ",
        ]
        chars.extend(pinyin_diacritics)

        # Deduplicate while preserving order
        seen: Set[str] = set()
        ordered_chars = []
        for ch in chars:
            if ch not in seen:
                seen.add(ch)
                ordered_chars.append(ch)

        # Build char2id and id2char starting after special tokens
        self.char2id: Dict[str, int] = {}
        self.id2char: Dict[int, str] = {}

        # Add specials
        for tid, sym in self.special_tokens.items():
            self.id2char[tid] = sym

        current_id = len(self.special_tokens)
        for ch in ordered_chars:
            self.char2id[ch] = current_id
            self.id2char[current_id] = ch
            current_id += 1

        self._vocab_size = current_id

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def normalize(self, text: str) -> str:
        """Normalize unicode forms, typographic variants, and enforce lowercase."""
        if self.normalize_quotes_and_dashes:
            for k, v in self.NORM_MAP.items():
                if k in text:
                    text = text.replace(k, v)
        text = unicodedata.normalize("NFC", text)
        return text.lower()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode arbitrary Romanized text string to lowercase token IDs."""
        text = self.normalize(text)
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)

        for ch in text:
            if ch == " ":
                tokens.append(self.eow_id)
            elif ch in self.char2id:
                tokens.append(self.char2id[ch])
            else:
                tokens.append(self.unk_id)

        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def encode_words(self, text: str) -> List[List[int]]:
        """Encode text into a list of word token sequences, each terminating with <eow>."""
        text = self.normalize(text)
        words = text.strip().split()
        word_sequences = []
        for w in words:
            seq = []
            for ch in w.strip():
                if ch in self.char2id:
                    seq.append(self.char2id[ch])
                else:
                    seq.append(self.unk_id)
            seq.append(self.eow_id)
            word_sequences.append(seq)
        return word_sequences

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to a lowercase string, truncating at <eos> or <eow>."""
        chars = []
        for tid in token_ids:
            if tid == self.eow_id:
                chars.append(" ")
            elif tid == self.eos_id:
                break
            elif tid in self.id2char and tid not in self.special_tokens:
                chars.append(self.id2char[tid])
            elif not skip_special:
                chars.append(self.special_tokens.get(tid, ""))

        return "".join(chars).strip()

    def decode_words(self, word_token_lists: List[List[int]]) -> str:
        """Decode word token sequences into cleanly spaced text, stripping whitespace per word."""
        decoded_words = []
        for seq in word_token_lists:
            clean_tokens = []
            for tid in seq:
                # Early stop at word delimiter
                if tid in (self.eow_id, self.eos_id):
                    break
                if tid not in (self.pad_id, self.bos_id, self.blank_id):
                    clean_tokens.append(tid)

            w = "".join(self.id2char.get(t, "") for t in clean_tokens if t in self.id2char).strip()
            if w:
                decoded_words.append(w)

        return " ".join(decoded_words).strip()


if __name__ == "__main__":
    tok = RomanCharTokenizer()
    print(f"RomanCharTokenizer Vocab Size: {tok.vocab_size} tokens")

    samples = [
        "  He hoped there would be stew for dinner 123!  ",
        "Le renard brun rapide saute par-dessus le chien endormi, où sont les élèves?",
        "El rápido zorro marrón salta sobre el perro perezoso. ¡Hola! ¿Cómo estás hoy?",
        "Der schnelle braune Fuchs springt über den faulen Hund in der Straße 42.",
        "La pasta al pomodoro è deliziosa, così com'è a città di Roma!",
        "Ni3 Hao3 Shi4 Jie4, zhe4 shi4 yi2 ge4 ce4 shi4.",
        "Konnichiha sekai , koreha tesuto desu .",
        "Annyeonghaseyo segye, igeoseun teseuteuipnida.",
        "mr7ba b-al3alam hadha ikhtibar 100%",
        "“Edge Computing” — AudioLearn 2026: cost $0.00!",
    ]
    for s in samples:
        words = tok.encode_words(s)
        dec = tok.decode_words(words)
        norm_s = tok.normalize(s).strip()
        print(f"'{s.strip()[:35]}' -> '{dec[:35]}'")
        assert dec == norm_s or dec == " ".join(norm_s.split())
    print("✅ All samples cleanly stripped without whitespace issues!")
