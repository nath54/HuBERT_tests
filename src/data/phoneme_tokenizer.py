"""Modular Bilingual Phoneme Tokenizer for Direct Phoneme Prediction Speech Models.

Equipped with specialized acoustic and linguistic tokens:
- <pad>: Batch padding
- <blank>: CTC non-emitting / transition blank
- <mask>: Masked acoustic span placeholder
- <silence>: Explicit unvoiced / pause interval
- <noise>: Acoustic perturbation / ambient noise frame
- <same_phoneme_than_last_one>: Phoneme continuation across consecutive 20ms frames
- <eos>: End of utterance
Plus full French and English International Phonetic Alphabet (IPA) phonemes.
"""

from typing import Dict, List, Optional, Set, Union


class PhonemeTokenizer:
    """Bilingual (FR + EN) Phoneme Tokenizer with dedicated special tokens."""

    SPECIAL_TOKENS = [
        "<pad>",
        "<blank>",
        "<mask>",
        "<silence>",
        "<noise>",
        "<same_phoneme_than_last_one>",
        "<eos>",
    ]

    # Standard English and French IPA phoneme alphabet
    IPA_INVENTORY = [
        # Word boundary / separator
        " ",
        "-",
        # Stress & Length Markers
        "ˈ", "ˌ", "ː", "ˑ",
        # English & French Vowels (Oral)
        "a", "e", "i", "o", "u", "y",
        "ɑ", "ɒ", "ɔ", "ə", "ɛ", "ɜ", "ɪ", "ʊ", "ʌ", "æ",
        "ø", "œ",
        # Combining Nasal Diacritic (Crucial for French: an, en, in, on, un)
        "\u0303",  # combining tilde ̃
        # French Semi-vowels & Approximants
        "ɥ", "j", "w",
        # English & French Consonants
        "b", "d", "f", "ɡ", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z",
        "θ", "ð", "ʃ", "ʒ", "ŋ", "ɲ", "ʁ", "ɹ", "ʔ",
        # Affricates / Digraphs
        "tʃ", "dʒ", "ts", "dz",
    ]

    def __init__(self, extra_tokens: Optional[List[str]] = None):
        self.special_tokens = list(self.SPECIAL_TOKENS)
        self.pad_token = "<pad>"
        self.blank_token = "<blank>"
        self.mask_token = "<mask>"
        self.silence_token = "<silence>"
        self.noise_token = "<noise>"
        self.same_as_last_token = "<same_phoneme_than_last_one>"
        self.eos_token = "<eos>"

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # 1. Register Special Tokens (Fixed IDs 0..6)
        for idx, tok in enumerate(self.special_tokens):
            self.token_to_id[tok] = idx
            self.id_to_token[idx] = tok

        # 2. Register IPA Inventory
        all_ipa = list(self.IPA_INVENTORY)
        if extra_tokens:
            for et in extra_tokens:
                if et not in all_ipa and et not in self.special_tokens:
                    all_ipa.append(et)

        current_id = len(self.special_tokens)
        for ph in all_ipa:
            if ph not in self.token_to_id:
                self.token_to_id[ph] = current_id
                self.id_to_token[current_id] = ph
                current_id += 1

        self.pad_id = self.token_to_id[self.pad_token]
        self.blank_id = self.token_to_id[self.blank_token]
        self.mask_id = self.token_to_id[self.mask_token]
        self.silence_id = self.token_to_id[self.silence_token]
        self.noise_id = self.token_to_id[self.noise_token]
        self.same_as_last_id = self.token_to_id[self.same_as_last_token]
        self.eos_id = self.token_to_id[self.eos_token]

        # Standard attribute aliases
        self.pad_token_id = self.pad_id
        self.blank_token_id = self.blank_id
        self.mask_token_id = self.mask_id
        self.silence_token_id = self.silence_id
        self.noise_token_id = self.noise_id
        self.same_as_last_token_id = self.same_as_last_id
        self.eos_token_id = self.eos_id
        self.id_to_phoneme = self.id_to_token
        self.phoneme_to_id = self.token_to_id
        self.vocab = self.token_to_id
        self.unk_id = self.pad_id  # Fallback for unknown tokens

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, phonemes: Union[str, List[str]], add_eos: bool = False) -> List[int]:
        """Convert a phoneme sequence into token IDs."""
        ids: List[int] = []
        if isinstance(phonemes, str):
            # Decompose unicode string into characters
            chars = list(phonemes)
        else:
            chars = phonemes

        for ch in chars:
            token_id = self.token_to_id.get(ch, self.unk_id)
            ids.append(token_id)

        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, token_ids: List[int], skip_special: bool = False) -> str:
        """Convert token IDs back to a readable phoneme string."""
        chars: List[str] = []
        for tid in token_ids:
            if tid in self.id_to_token:
                tok = self.id_to_token[tid]
                if skip_special and tok in self.special_tokens:
                    continue
                chars.append(tok)
        return "".join(chars)

    def collapse_repeats_to_special(self, token_ids: List[int]) -> List[int]:
        """Transform consecutive duplicate frame tokens into <same_phoneme_than_last_one>."""
        collapsed: List[int] = []
        last_id = None
        for tid in token_ids:
            if tid == last_id and tid != self.blank_id and tid != self.silence_id:
                collapsed.append(self.same_as_last_id)
            else:
                collapsed.append(tid)
                last_id = tid
        return collapsed

    def expand_repeats(self, token_ids: List[int]) -> List[int]:
        """Expand <same_phoneme_than_last_one> back to the preceding phoneme."""
        expanded: List[int] = []
        last_id = self.silence_id
        for tid in token_ids:
            if tid == self.same_as_last_id:
                expanded.append(last_id)
            else:
                expanded.append(tid)
                last_id = tid
        return expanded
