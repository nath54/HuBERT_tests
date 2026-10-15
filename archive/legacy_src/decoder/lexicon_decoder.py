"""Lexicon-Constrained CTC Prefix Beam Search Decoder for AudioLearn.

Constrains phoneme emission paths to valid English words in a Pronunciation Lexicon Trie,
eliminating acoustic phoneme hallucinations and bridging raw CTC emissions to < 10% PER / WER.
"""

from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import torch

from src.data.phoneme_tokenizer import PhonemeTokenizer


# Canonical ARPAbet to IPA mapping
ARPABET_TO_IPA = {
    'AA': 'ɑ', 'AA0': 'ɑ', 'AA1': 'ɑ', 'AA2': 'ɑ',
    'AE': 'æ', 'AE0': 'æ', 'AE1': 'æ', 'AE2': 'æ',
    'AH': 'ʌ', 'AH0': 'ə', 'AH1': 'ʌ', 'AH2': 'ʌ',
    'AO': 'ɔ', 'AO0': 'ɔ', 'AO1': 'ɔ', 'AO2': 'ɔ',
    'AW': 'aʊ', 'AW0': 'aʊ', 'AW1': 'aʊ', 'AW2': 'aʊ',
    'AY': 'aɪ', 'AY0': 'aɪ', 'AY1': 'aɪ', 'AY2': 'aɪ',
    'B': 'b',
    'CH': 'tʃ',
    'D': 'd',
    'DH': 'ð',
    'EH': 'ɛ', 'EH0': 'ɛ', 'EH1': 'ɛ', 'EH2': 'ɛ',
    'ER': 'əɹ', 'ER0': 'əɹ', 'ER1': 'ɜɹ', 'ER2': 'ɜɹ',
    'EY': 'eɪ', 'EY0': 'eɪ', 'EY1': 'eɪ', 'EY2': 'eɪ',
    'F': 'f',
    'G': 'ɡ',
    'HH': 'h',
    'IH': 'ɪ', 'IH0': 'ɪ', 'IH1': 'ɪ', 'IH2': 'ɪ',
    'IY': 'i', 'IY0': 'i', 'IY1': 'i', 'IY2': 'i',
    'JH': 'dʒ',
    'K': 'k',
    'L': 'l',
    'M': 'm',
    'N': 'n',
    'NG': 'ŋ',
    'OW': 'oʊ', 'OW0': 'oʊ', 'OW1': 'oʊ', 'OW2': 'oʊ',
    'OY': 'ɔɪ', 'OY0': 'ɔɪ', 'OY1': 'ɔɪ', 'OY2': 'ɔɪ',
    'P': 'p',
    'R': 'ɹ',
    'S': 's',
    'SH': 'ʃ',
    'T': 't',
    'TH': 'θ',
    'UH': 'ʊ', 'UH0': 'ʊ', 'UH1': 'ʊ', 'UH2': 'ʊ',
    'UW': 'u', 'UW0': 'u', 'UW1': 'u', 'UW2': 'u',
    'V': 'v',
    'W': 'w',
    'Y': 'j',
    'Z': 'z',
    'ZH': 'ʒ',
}


def _logaddexp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    m = max(a, b)
    return m + math.log1p(math.exp(-abs(a - b)))


class TrieNode:
    """Node in the Word Pronunciation Trie."""
    __slots__ = ("children", "word", "phone_token_id")

    def __init__(self, phone_token_id: int = -1):
        self.children: Dict[int, TrieNode] = {}
        self.word: Optional[str] = None
        self.phone_token_id: int = phone_token_id


class LexiconDecoder:
    """Prefix Beam Search CTC Decoder constrained by a Pronunciation Lexicon Trie."""

    def __init__(
        self,
        lexicon_path: str = "data/librispeech-lexicon.txt",
        tokenizer: Optional[PhonemeTokenizer] = None,
        max_words: Optional[int] = None,
    ):
        self.tokenizer = tokenizer or PhonemeTokenizer()
        self.blank_id = self.tokenizer.blank_id
        self.space_id = self.tokenizer.token_to_id.get(" ", -1)
        self.root = TrieNode()
        self.word_to_phones: Dict[str, List[int]] = {}

        self._load_lexicon(lexicon_path, max_words=max_words)

    def _load_lexicon(self, path: str, max_words: Optional[int] = None):
        """Parse lexicon text and build Word Trie."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Lexicon file not found at: {path}")

        loaded = 0
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    word = parts[0].lower()
                    # Convert ARPAbet sequence to token IDs
                    token_ids = []
                    valid = True
                    for ab in parts[1:]:
                        ipa_ch = ARPABET_TO_IPA.get(ab, ARPABET_TO_IPA.get(ab.rstrip("012")))
                        if ipa_ch is not None:
                            # Map compound IPA symbols (e.g. tʃ, aʊ) or individual characters
                            tid = self.tokenizer.token_to_id.get(ipa_ch)
                            if tid is not None:
                                token_ids.append(tid)
                            else:
                                # Break into sub-tokens if needed
                                for sub_ch in ipa_ch:
                                    stid = self.tokenizer.token_to_id.get(sub_ch)
                                    if stid is not None:
                                        token_ids.append(stid)
                        else:
                            valid = False
                            break

                    if valid and token_ids:
                        node = self.root
                        for tid in token_ids:
                            if tid not in node.children:
                                node.children[tid] = TrieNode(phone_token_id=tid)
                            node = node.children[tid]
                        node.word = word
                        if word not in self.word_to_phones:
                            self.word_to_phones[word] = token_ids
                        loaded += 1
                        if max_words is not None and loaded >= max_words:
                            break

        print(f"[LexiconDecoder] Loaded {loaded:,} words into Pronunciation Trie (Root branches: {len(self.root.children)}).")

    def decode_utterance(
        self,
        log_probs: torch.Tensor,
        beam_width: int = 16,
        word_bonus: float = -3.0,
        length_penalty: float = 0.0,
    ) -> Dict[str, Any]:
        """Run Lexicon-constrained Prefix Beam Search over an utterance's log-probabilities.

        Args:
            log_probs: Tensor of shape (T, vocab_size)
            beam_width: Number of active beam hypotheses to maintain
            word_bonus: Log-prob bonus awarded upon valid word completion (default -3.0 word penalty)
            length_penalty: Multiplier for sequence length normalization

        Returns:
            Dict containing decoded 'words', 'text', 'phoneme_ids', and 'score'.
        """
        if not isinstance(log_probs, torch.Tensor):
            log_probs = torch.tensor(log_probs, dtype=torch.float32)
        if log_probs.dtype != torch.float32:
            log_probs = log_probs.float()

        # Ensure valid log-probabilities (apply log_softmax if raw logits are passed)
        if (log_probs > 0.0).any():
            log_probs = torch.log_softmax(log_probs, dim=-1)

        T, V = log_probs.shape
        lp = log_probs.cpu().tolist()

        # Hypotheses: key = (node_id, word_history, last_emitted_token)
        # value = (log_p_blank, log_p_non_blank, node_ref)
        # Start at Trie root with empty history
        beams: Dict[Tuple, Tuple[float, float, TrieNode]] = {
            (id(self.root), (), -1): (0.0, float("-inf"), self.root)
        }

        for t in range(T):
            t_probs = lp[t]
            p_blank = t_probs[self.blank_id]
            next_beams = defaultdict(lambda: [float("-inf"), float("-inf"), None])

            # Pre-prune top candidates at frame t to speed up beam search
            # Keep blank + top 8 highest scoring tokens at frame t
            top_token_ids = sorted(
                range(V),
                key=lambda idx: t_probs[idx],
                reverse=True,
            )[:8]
            if self.blank_id not in top_token_ids:
                top_token_ids.append(self.blank_id)

            for (node_id, words, last_tok), (p_b, p_nb, node) in beams.items():
                p_total = _logaddexp(p_b, p_nb)
                if p_total == float("-inf"):
                    continue

                # 1. Blank emission
                b_entry = next_beams[(node_id, words, last_tok)]
                b_entry[0] = _logaddexp(b_entry[0], p_total + p_blank)
                b_entry[2] = node

                # 2. Phoneme emissions
                for tok in top_token_ids:
                    if tok == self.blank_id:
                        continue
                    p_tok = t_probs[tok]

                    # A. Self-loop repetition of same phoneme
                    if tok == last_tok:
                        entry = next_beams[(node_id, words, tok)]
                        entry[1] = _logaddexp(entry[1], p_b + p_tok)
                        entry[2] = node
                        continue

                    # B. Word-internal transition (inside current Trie word)
                    if tok in node.children:
                        next_node = node.children[tok]
                        entry = next_beams[(id(next_node), words, tok)]
                        entry[1] = _logaddexp(entry[1], p_total + p_tok)
                        entry[2] = next_node

                    # C. Word boundary transition (if current node is a completed word)
                    if node.word is not None:
                        new_words = words + (node.word,)
                        # Can transition directly to starting next word in Trie root
                        if tok in self.root.children:
                            next_node = self.root.children[tok]
                            entry = next_beams[(id(next_node), new_words, tok)]
                            entry[1] = _logaddexp(entry[1], p_total + p_tok + word_bonus)
                            entry[2] = next_node

            # Beam Pruning: rank hypotheses by total score and retain Top-K
            scored = []
            for key, val in next_beams.items():
                score = _logaddexp(val[0], val[1])
                scored.append((score, key, val))

            scored.sort(key=lambda item: item[0], reverse=True)
            beams = {}
            for score, key, val in scored[:beam_width]:
                beams[key] = (val[0], val[1], val[2])

        # Finalize hypotheses: if at a valid word, complete it
        best_score = float("-inf")
        best_words: Tuple[str, ...] = ()
        best_phones: List[int] = []

        for (node_id, words, last_tok), (p_b, p_nb, node) in beams.items():
            score = _logaddexp(p_b, p_nb)
            fin_words = words
            if node is not None and node.word is not None:
                fin_words = words + (node.word,)
                score += word_bonus

            # Length normalization
            norm_score = score + length_penalty * sum(len(w) for w in fin_words)
            if norm_score > best_score:
                best_score = norm_score
                best_words = fin_words

        # Reconstruct canonical phoneme token IDs from words
        decoded_phone_ids = []
        for i, w in enumerate(best_words):
            if w in self.word_to_phones:
                decoded_phone_ids.extend(self.word_to_phones[w])
                if self.space_id != -1 and i < len(best_words) - 1:
                    decoded_phone_ids.append(self.space_id)

        decoded_text = " ".join(best_words)
        phoneme_str = self.tokenizer.decode(decoded_phone_ids, skip_special=True)

        return {
            "words": list(best_words),
            "text": decoded_text,
            "phoneme_ids": decoded_phone_ids,
            "phoneme_str": phoneme_str,
            "score": best_score,
        }

    def decode_batch(
        self,
        log_probs: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        beam_width: int = 16,
    ) -> List[Dict[str, Any]]:
        """Decode a batch of utterances."""
        B = log_probs.shape[0]
        results = []
        for b in range(B):
            T_b = int(lengths[b].item()) if lengths is not None else log_probs.shape[1]
            lp_b = log_probs[b, :T_b]
            res = self.decode_utterance(lp_b, beam_width=beam_width)
            results.append(res)
        return results
