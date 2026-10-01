"""Universal Token-Free UTF-8 Byte Tokenizer for Multilingual Edge Speech Recognition."""

from typing import Dict, List, Optional, Union


class ByteTokenizer:
    """UTF-8 Byte Tokenizer with dedicated control tokens.
    
    Total vocabulary size: 261 (5 control tokens + 256 raw bytes).
    Guarantees 0.0% OOV across all languages and Unicode scripts.
    """

    PAD_ID = 0
    BLANK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    EOW_ID = 4  # End of Word / Word separator
    BYTE_OFFSET = 5

    def __init__(self):
        self.pad_id = self.PAD_ID
        self.blank_id = self.BLANK_ID
        self.bos_id = self.BOS_ID
        self.eos_id = self.EOS_ID
        self.eow_id = self.EOW_ID
        self.byte_offset = self.BYTE_OFFSET
        self._vocab_size = 261

        self.special_tokens: Dict[int, str] = {
            self.pad_id: "<pad>",
            self.blank_id: "<blank>",
            self.bos_id: "<bos>",
            self.eos_id: "<eos>",
            self.eow_id: "<eow>",
        }

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode arbitrary string to UTF-8 byte token IDs."""
        raw_bytes = text.encode("utf-8")
        tokens = [b + self.byte_offset for b in raw_bytes]
        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def encode_words(self, text: str) -> List[List[int]]:
        """Encode text into a list of word byte sequences, each terminating with <eow>."""
        words = text.strip().split()
        word_sequences = []
        for w in words:
            w_bytes = w.encode("utf-8")
            seq = [b + self.byte_offset for b in w_bytes] + [self.eow_id]
            word_sequences.append(seq)
        return word_sequences

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back to a UTF-8 string."""
        byte_vals = bytearray()
        for tid in token_ids:
            if tid == self.eow_id:
                byte_vals.append(ord(" "))
            elif tid >= self.byte_offset and tid < self.vocab_size:
                byte_vals.append(tid - self.byte_offset)
            elif not skip_special:
                # If keeping special tokens, decode as string tag
                name = self.special_tokens.get(tid, "")
                byte_vals.extend(name.encode("utf-8"))

        return byte_vals.decode("utf-8", errors="replace").strip()

    def decode_words(self, word_token_lists: List[List[int]]) -> str:
        """Decode a list of word byte token sequences into space-separated text."""
        decoded_words = []
        for seq in word_token_lists:
            w = self.decode(seq, skip_special=True)
            if w:
                decoded_words.append(w)
        return " ".join(decoded_words)


if __name__ == "__main__":
    tok = ByteTokenizer()
    print(f"ByteTokenizer vocab size: {tok.vocab_size}")
    
    samples = [
        "he hoped there would be stew for dinner",
        "le renard brun rapide saute par-dessus le chien endormi",
        "el rápido zorro marrón salta sobre el perro perezoso",
        "der schnelle braune Fuchs springt über den faulen Hund",
        "你好世界，这是一个测试",
        "مرحبا بالعالم",
        "안녕하세요 세계",
    ]
    for s in samples:
        enc = tok.encode(s)
        dec = tok.decode(enc)
        assert dec == s, f"Mismatch: '{s}' != '{dec}'"
        print(f"✅ [{len(s)} chars -> {len(enc)} bytes] '{s}' -> Decoded: '{dec}'")
