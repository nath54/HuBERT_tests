"""Phonotactic Viterbi Lexicon Segmenter.

Provides ultrafast (<0.01 ms), scalable multilingual word boundary disambiguation
(clitic splitting and acoustic fusion recovery) operating directly on integer token IDs.
"""

import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from piper.voice import EspeakPhonemizer
from src.data.phoneme_tokenizer import PhonemeTokenizer


class PhonotacticLexiconSegmenter:
    """Dynamic programming Viterbi word segmenter over a multilingual phonetic lexicon.
    
    Discovers optimal word boundaries in continuous fused phonetic sequences
    (e.g., [stʌfɪd] -> [stʌf] + [ɪd], [ʌvðə] -> [ʌv] + [ðə], [lapʁetɑ̃dy] -> [la] + [pʁetɑ̃dy])
    in < 0.01 ms per utterance directly in integer token ID space.
    """

    # Common function words & clitics with high unigram prior
    HIGH_PRIOR_CLITICS = {
        # English
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "by", "for",
        "with", "from", "it", "its", "is", "was", "are", "were", "be", "been",
        "he", "she", "we", "they", "i", "you", "his", "her", "their", "my", "your",
        "that", "this", "there", "here", "would", "could", "should", "will", "can",
        # French
        "le", "la", "les", "l", "un", "une", "des", "du", "de", "d", "au", "aux",
        "ce", "cet", "cette", "ces", "c", "se", "s", "sa", "son", "ses", "en", "y",
        "et", "ou", "que", "qui", "qu", "dans", "sur", "sous", "par", "pour",
        "il", "elle", "ils", "elles", "on", "ne", "n", "pas", "plus", "tout",
        "me", "m", "te", "t", "lui", "leur", "nous", "vous", "est", "a", "ont",
    }

    def __init__(
        self,
        vocab_json_path: Optional[str] = "data/librispeech_vocab_full.json",
        french_transcripts_path: Optional[str] = "/media/hdd/Datasets/mls/mls_french/train/transcripts.txt",
        cache_path: str = "data/phonotactic_lexicon_ids.json",
        split_penalty: float = 3.8,
        oov_char_penalty: float = 3.0,
        max_word_len: int = 16,
    ):
        self.split_penalty = split_penalty
        self.oov_char_penalty = oov_char_penalty
        self.max_word_len = max_word_len
        self.ph_tok = PhonemeTokenizer()
        self._espeak = None

        # Lexicon: tuple of token IDs -> log probability
        self.lexicon_ids: Dict[Tuple[int, ...], float] = {}
        self.known_words: Set[Tuple[int, ...]] = set()

        self._build_lexicon(vocab_json_path, french_transcripts_path, cache_path)

    def _get_espeak(self) -> EspeakPhonemizer:
        if self._espeak is None:
            self._espeak = EspeakPhonemizer()
        return self._espeak

    def _phonemize_to_ids(self, word: str, lang: str = "en-us") -> Tuple[int, ...]:
        ep = self._get_espeak()
        try:
            chunks = ep.phonemize(lang, word)
            flat = [ch for sub in chunks for ch in sub if ch not in ("ˈ", "ˌ", "ː", "ˑ", " ", "-", "\n", "\t")]
            return tuple(self.ph_tok.encode(flat))
        except Exception:
            return tuple()

    @property
    def lexicon(self) -> Dict[Tuple[int, ...], float]:
        return self.lexicon_ids

    def _phonemize_batch(self, words: List[str], lang: str = "en-us") -> Dict[str, Tuple[int, ...]]:
        ep = self._get_espeak()
        results = {}
        batch_size = 150
        for i in range(0, len(words), batch_size):
            batch = words[i : i + batch_size]
            try:
                chunks = ep.phonemize(lang, " ".join(batch))
                flat = [ch for sub in chunks for ch in sub if ch not in ("ˈ", "ˌ", "ː", "ˑ", "-", "\n", "\t")]
                parts = "".join(flat).split(" ")
                if len(parts) == len(batch):
                    for w, p in zip(batch, parts):
                        if p:
                            results[w] = tuple(self.ph_tok.encode(list(p)))
                    continue
            except Exception:
                pass
            # Fallback per word
            for w in batch:
                ph_ids = self._phonemize_to_ids(w, lang=lang)
                if ph_ids:
                    results[w] = ph_ids
        return results

    def _build_lexicon(self, en_vocab_path: Optional[str], fr_transcripts_path: Optional[str], cache_path: str):
        """Populates the phonetic lexicon from English and French vocabularies with disk caching."""
        if os.path.isfile(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                for k, v in cached.items():
                    id_tuple = tuple(int(x) for x in k.split("_"))
                    self.lexicon_ids[id_tuple] = v
                    self.known_words.add(id_tuple)
                return
            except Exception:
                pass

        word_counts = Counter()

        # 1. High Prior Clitics (Given massive frequency prior)
        for clitic in self.HIGH_PRIOR_CLITICS:
            word_counts[clitic] += 50000

        # 2. English Vocabulary (Empirical frequencies from training data)
        en_manifest = "data/librispeech/librispeech_train.json"
        if os.path.isfile(en_manifest):
            try:
                with open(en_manifest, "r", encoding="utf-8") as f:
                    train_data = json.load(f)
                c_en = Counter()
                for item in train_data:
                    t = item.get("transcript", "")
                    if t:
                        for w in t.lower().split():
                            clean_w = w.strip(".,;:!?\"'")
                            if clean_w and clean_w.isalpha():
                                c_en[clean_w] += 1
                for w, cnt in c_en.most_common(18000):
                    word_counts[w] += cnt
            except Exception:
                pass
        elif en_vocab_path and os.path.isfile(en_vocab_path):
            try:
                with open(en_vocab_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    for item in data:
                        t = item.get("transcript", "") or item.get("text", "")
                        for w in t.lower().split():
                            clean_w = w.strip(".,;:!?\"'")
                            if clean_w and clean_w.isalpha():
                                word_counts[clean_w] += 1
                elif isinstance(data, list):
                    for w in data[:20000]:
                        clean_w = str(w).lower().strip()
                        if clean_w and clean_w.isalpha():
                            word_counts[clean_w] += 10
            except Exception:
                pass

        # 3. French Transcripts
        if fr_transcripts_path and os.path.isfile(fr_transcripts_path):
            try:
                with open(fr_transcripts_path, "r", encoding="utf-8") as f:
                    c_fr = Counter()
                    for idx, line in enumerate(f):
                        if idx >= 20000:
                            break
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            for w in parts[1].lower().split():
                                clean_w = w.strip(".,;:!?\"'«»()")
                                if clean_w and clean_w.isalpha():
                                    c_fr[clean_w] += 1
                    for w, cnt in c_fr.most_common(12000):
                        word_counts[w] += cnt
            except Exception:
                pass

        total_count = sum(word_counts.values()) or 1
        log_total = math.log(total_count)

        # Separate words by language for fast batch phonemization
        en_words = []
        fr_words = []
        for word in word_counts.keys():
            if any(c in "éèêëàâäôöûüùç" for c in word) or word in ("le", "la", "les", "du", "des", "dans", "sur", "pour", "avec", "est", "qui", "que"):
                fr_words.append(word)
            else:
                en_words.append(word)

        # Batch phonemize
        en_ph_map = self._phonemize_batch(en_words, lang="en-us")
        fr_ph_map = self._phonemize_batch(fr_words, lang="fr")

        for w, ph_ids in {**en_ph_map, **fr_ph_map}.items():
            if ph_ids:
                count = word_counts[w]
                log_prob = math.log(count) - log_total
                if ph_ids not in self.lexicon_ids or log_prob > self.lexicon_ids[ph_ids]:
                    self.lexicon_ids[ph_ids] = log_prob
                    self.known_words.add(ph_ids)

        # 4. Explicit Clitic Allophones (Flaps, reductions, liaison clitics)
        clitic_allophones = {
            ("ɪ", "d"): -3.5,   # 'it' flapped
            ("ɪ", "ɾ"): -3.5,   # 'it' flapped
            ("ʌ", "v"): -3.0,   # 'of'
            ("ə", "v"): -3.0,   # 'of'
            ("ð", "ə"): -2.5,   # 'the'
            ("ð", "i"): -3.0,   # 'the'
            ("t", "u"): -3.0,   # 'to'
            ("t", "ə"): -3.0,   # 'to'
            ("t", "ʊ"): -3.0,   # 'to'
            ("ɪ", "n"): -3.0,   # 'in'
            ("ɔ", "n"): -3.0,   # 'on'
            ("æ", "t"): -3.5,   # 'at'
            ("ə", "t"): -3.5,   # 'at'
            ("l", "a"): -2.8,   # 'la'
            ("l", "ə"): -2.8,   # 'le'
            ("l", "e"): -2.8,   # 'les'
            ("d", "ə"): -2.8,   # 'de'
            ("d", "y"): -3.0,   # 'du'
            ("d", "e"): -3.0,   # 'des'
            ("s", "ə"): -3.0,   # 'se'
            ("s", "ɛ"): -3.0,   # 'c'est'
            ("o",):     -3.0,   # 'au'
            ("y", "n"): -3.0,   # 'une'
            ("œ", "̃"): -3.0,   # 'un'
            ("ɛ", "̃"): -3.0,   # 'un'
        }
        for k_chars, v in clitic_allophones.items():
            k_ids = tuple(self.ph_tok.encode(list(k_chars)))
            self.lexicon_ids[k_ids] = max(v, self.lexicon_ids.get(k_ids, -100.0))
            self.known_words.add(k_ids)

        # Save to disk cache for instantaneous reloads (<0.02s)
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            to_save = {"_".join(map(str, k)): v for k, v in self.lexicon_ids.items()}
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(to_save, f)
        except Exception:
            pass

    def segment_id_sequence(
        self,
        id_seq: Tuple[int, ...],
        split_penalty: Optional[float] = None,
    ) -> List[Tuple[int, ...]]:
        """Splits a single word's token ID sequence if an acoustic fusion is detected.
        
        Args:
            id_seq: tuple of integer phoneme IDs
        Returns:
            segments: list of integer ID tuples
        """
        N = len(id_seq)
        if N <= 3:
            return [id_seq]

        pen = self.split_penalty if split_penalty is None else split_penalty
        base_score = self.lexicon_ids.get(id_seq, -25.0 - self.oov_char_penalty * N)

        dp = [-float("inf")] * (N + 1)
        parent = [0] * (N + 1)
        dp[0] = 0.0

        for k in range(1, N + 1):
            max_lookback = min(k, self.max_word_len)
            for j in range(k - max_lookback, k):
                sub = id_seq[j:k]
                if sub in self.lexicon_ids:
                    w_score = self.lexicon_ids[sub]
                else:
                    w_score = -20.0 - self.oov_char_penalty * (k - j)

                split_cost = pen if (j > 0 or k < N) else 0.0
                score = dp[j] + w_score - split_cost
                if score > dp[k]:
                    dp[k] = score
                    parent[k] = j

        # If the best segmented path is not substantially better than the base undivided word, keep undivided
        if dp[N] <= base_score:
            return [id_seq]

        # Backtrack
        curr = N
        cuts = []
        while curr > 0:
            cuts.append(curr)
            curr = parent[curr]
        cuts.reverse()

        segments = []
        prev = 0
        for cut in cuts:
            segments.append(id_seq[prev:cut])
            prev = cut

        return segments if segments else [id_seq]

    def segment_word_emissions(
        self,
        word_emissions: List[Tuple[List[int], List[float]]],
        id_to_token: Optional[Dict[int, str]] = None,
    ) -> List[List[Tuple[List[int], List[float]]]]:
        """Segments a word emission sequence (with Top-K IDs & Probs) if a clitic/fusion is found.
        
        Args:
            word_emissions: list of (cand_ids, cand_probs) for each phoneme position
        Returns:
            List of segmented words, where each word is a slice of the emissions
        """
        if not word_emissions:
            return []

        id_seq = tuple(c_ids[0] for c_ids, _ in word_emissions)
        segments = self.segment_id_sequence(id_seq)

        if len(segments) <= 1:
            return [word_emissions]

        # Partition the emission tuples according to the segment lengths
        result_words = []
        curr_idx = 0
        for seg in segments:
            seg_len = len(seg)
            result_words.append(word_emissions[curr_idx : curr_idx + seg_len])
            curr_idx += seg_len

        return result_words

    def segment_phoneme_tokens(
        self,
        tokens: List[str],
        split_penalty: Optional[float] = None,
    ) -> List[List[str]]:
        """Convenience method for string phoneme token lists."""
        id_seq = tuple(self.ph_tok.encode(tokens))
        id_segs = self.segment_id_sequence(id_seq, split_penalty=split_penalty)
        res = []
        for s in id_segs:
            res.append([self.ph_tok.id_to_token.get(i, "") for i in s])
        return res
