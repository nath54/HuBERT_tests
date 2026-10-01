"""Word-level Tokenizer for Word Denoising Decoder."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union


class WordTokenizer:
    """Word-level vocabulary tokenizer with special tokens."""

    DEFAULT_SPECIAL = ["<pad>", "<blank>", "<unk>", "<bos>", "<eos>"]

    def __init__(self, vocab_path: Optional[Union[str, Path]] = None, vocab: Optional[List[str]] = None):
        if vocab is not None:
            self.vocab = vocab
        elif vocab_path is not None and Path(vocab_path).exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)
        else:
            self.vocab = list(self.DEFAULT_SPECIAL)

        self.word_to_id: Dict[str, int] = {w: idx for idx, w in enumerate(self.vocab)}
        self.id_to_word: Dict[int, str] = {idx: w for idx, w in enumerate(self.vocab)}

        self.pad_id = self.word_to_id.get("<pad>", 0)
        self.blank_id = self.word_to_id.get("<blank>", 1)
        self.unk_id = self.word_to_id.get("<unk>", 2)
        self.bos_id = self.word_to_id.get("<bos>", 3)
        self.eos_id = self.word_to_id.get("<eos>", 4)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode words in text to token IDs."""
        words = text.lower().strip().split()
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)
        for w in words:
            tokens.append(self.word_to_id.get(w, self.unk_id))
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to space-separated words."""
        words = []
        special_ids = {self.pad_id, self.blank_id, self.unk_id, self.bos_id, self.eos_id}
        for tid in token_ids:
            if skip_special and tid in special_ids:
                continue
            word = self.id_to_word.get(tid, "<unk>")
            words.append(word)
        return " ".join(words)

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2)


if __name__ == "__main__":
    tok = WordTokenizer("data/word_vocab_10k.json")
    print(f"Loaded {tok.vocab_size} tokens.")
    ids = tok.encode("he hoped there would be stew for dinner", add_bos=True, add_eos=True)
    print("Encoded IDs:", ids)
    print("Decoded text:", tok.decode(ids))
