"""Character-level CTC Tokenizer for ASR."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union


class CharacterTokenizer:
    """Character tokenizer suited for CTC speech recognition models."""

    DEFAULT_VOCAB = [
        "<blank>",  # 0: CTC blank symbol
        "<pad>",    # 1: Padding symbol
        " ",        # 2: Word separator
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "'",        # 29: Apostrophe
        "<unk>",    # 30: Unknown symbol
    ]

    def __init__(self, vocab: Optional[List[str]] = None):
        self.vocab = vocab or self.DEFAULT_VOCAB
        self.char_to_id: Dict[str, int] = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char: Dict[int, str] = {idx: char for idx, char in enumerate(self.vocab)}

        self.blank_id = self.char_to_id.get("<blank>", 0)
        self.pad_id = self.char_to_id.get("<pad>", 1)
        self.unk_id = self.char_to_id.get("<unk>", len(self.vocab) - 1)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> List[int]:
        """Convert a text string to a list of token IDs.
        
        Args:
            text: Input transcript string.
        Returns:
            List of integer token IDs.
        """
        text = text.lower().strip()
        tokens = []
        for char in text:
            tokens.append(self.char_to_id.get(char, self.unk_id))
        return tokens

    def decode(
        self,
        token_ids: List[int],
        collapse_repeats: bool = True,
        remove_blank: bool = True,
        remove_pad: bool = True,
    ) -> str:
        """Decode a list of token IDs into text.
        
        Args:
            token_ids: Sequence of integer token IDs.
            collapse_repeats: Collapse repeated identical consecutive tokens (CTC rule).
            remove_blank: Remove blank tokens.
            remove_pad: Remove padding tokens.
        Returns:
            Decoded string.
        """
        output_chars = []
        prev_token = None

        for token in token_ids:
            if collapse_repeats and token == prev_token:
                continue
            prev_token = token

            if remove_blank and token == self.blank_id:
                continue
            if remove_pad and token == self.pad_id:
                continue

            char = self.id_to_char.get(token, "")
            if char in ("<blank>", "<pad>", "<unk>"):
                continue
            output_chars.append(char)

        return "".join(output_chars).strip()

    def save(self, path: Union[str, Path]) -> None:
        """Save vocabulary to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CharacterTokenizer":
        """Load vocabulary from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)
