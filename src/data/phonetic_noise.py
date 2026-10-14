"""Empirical Phonetic Noise & Acoustic Jitter Simulator.

Simulates realistic acoustic front-end errors (phonetic allophone confusions,
space drop/insertions, unstressed vowel elisions) to train downstream decoders
(Module 3) to be robust against imperfect acoustic inputs.
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple, Union
import torch


class PhoneticNoiseSimulator:
    """Simulates realistic acoustic recognition errors on clean phonetic sequences."""

    def __init__(
        self,
        confusion_matrix_path: Optional[str] = "data/phoneme_confusion_matrix.json",
        sub_rate: float = 0.08,
        del_rate: float = 0.03,
        ins_rate: float = 0.02,
        space_drop_rate: float = 0.08,
        space_insert_rate: float = 0.03,
        space_token_id: int = 8,
        seed: Optional[int] = None,
    ):
        self.sub_rate = sub_rate
        self.del_rate = del_rate
        self.ins_rate = ins_rate
        self.space_drop_rate = space_drop_rate
        self.space_insert_rate = space_insert_rate
        self.space_token_id = space_token_id
        self.rng = random.Random(seed)

        # Load empirical confusion probabilities if available
        self.confusion_probs: Dict[str, Dict[str, float]] = {}
        if confusion_matrix_path and os.path.isfile(confusion_matrix_path):
            try:
                with open(confusion_matrix_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.confusion_probs = data.get("substitution_probabilities", {})
            except Exception:
                self.confusion_probs = {}

        # Default fallback allophone confusion clusters
        self.fallback_clusters = {
            # Vowel height & reduction
            "i": ["ɪ", "e", "ij"],
            "ɪ": ["i", "ə", "e"],
            "e": ["ɛ", "eɪ", "i"],
            "ɛ": ["e", "æ", "ə"],
            "æ": ["ɛ", "ɑ", "a"],
            "a": ["ɑ", "æ", "o"],
            "ɑ": ["a", "ɔ", "ʌ"],
            "ɔ": ["o", "ɑ", "oʊ"],
            "o": ["ɔ", "u", "oʊ"],
            "u": ["ʊ", "o", "uw"],
            "ʊ": ["u", "ə", "o"],
            "ə": ["ʌ", "ɪ", "ɛ"],
            "ʌ": ["ə", "ɑ", "ʊ"],
            # Spanish/Italian allophones & spirantization
            "b": ["β", "p", "v"],
            "β": ["b", "v", "w"],
            "d": ["ð", "t", "ɾ"],
            "ð": ["d", "z", "θ"],
            "ɡ": ["ɣ", "k"],
            "ɣ": ["ɡ", "x", "w"],
            "r": ["ɾ", "rr", "ʁ"],
            "ɾ": ["r", "d", "t"],
            # Nasals & Fricatives
            "m": ["n", "ŋ"],
            "n": ["m", "ŋ", "ɲ"],
            "s": ["z", "ʃ", "θ"],
            "z": ["s", "ʒ", "ð"],
            "ʃ": ["s", "tʃ", "ʒ"],
        }

    def _sample_substitution_str(self, phone: str) -> str:
        """Sample a realistic replacement for a phoneme string."""
        if phone in self.confusion_probs:
            candidates = list(self.confusion_probs[phone].keys())
            weights = list(self.confusion_probs[phone].values())
            if candidates and sum(weights) > 0:
                return self.rng.choices(candidates, weights=weights, k=1)[0]
        if phone in self.fallback_clusters:
            return self.rng.choice(self.fallback_clusters[phone])
        return phone

    def perturb_phonemes(self, phonemes: Union[str, List[str]]) -> Union[str, List[str]]:
        """Apply noise to a string of space-separated phonemes or token list."""
        is_str = isinstance(phonemes, str)
        tokens = phonemes.split() if is_str else list(phonemes)

        perturbed = []
        for i, token in enumerate(tokens):
            # Space handling
            if token == "<space>" or token == " ":
                if self.rng.random() < self.space_drop_rate:
                    continue  # Missed space
                perturbed.append(token)
                continue

            # Random space insertion before token
            if self.rng.random() < self.space_insert_rate:
                perturbed.append("<space>" if is_str else " ")

            # Phoneme deletion
            if self.rng.random() < self.del_rate:
                continue

            # Phoneme substitution
            if self.rng.random() < self.sub_rate:
                token = self._sample_substitution_str(token)

            perturbed.append(token)

            # Phoneme insertion (reduplication or glide)
            if self.rng.random() < self.ins_rate:
                perturbed.append(token)

        if is_str:
            return " ".join(perturbed)
        return perturbed

    def generate_top_k_mixture_for_tokens(
        self,
        tokens: List[str],
        top_k: int = 3,
        mix_rate: float = 0.5,
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """Generate top-k phoneme candidates with probabilities for each token position.

        Args:
            tokens: clean phoneme tokens of a word
            top_k: number of candidates per position (default 3)
            mix_rate: probability that a phoneme position has ambiguity / mixture noise
        Returns:
            candidates: [P, top_k] list of tokens
            probs: [P, top_k] list of probabilities summing to 1.0
        """
        all_cands = []
        all_probs = []
        for token in tokens:
            if token in ("<space>", " ") or self.rng.random() > mix_rate:
                cands = [token] * top_k
                probs = [1.0] + [0.0] * (top_k - 1)
            else:
                conf_dict = self.confusion_probs.get(token, {})
                fallbacks = self.fallback_clusters.get(token, [])
                alt_cands = [c for c in list(conf_dict.keys()) if c != token] if conf_dict else [c for c in fallbacks if c != token]

                # Check if substitution applies to top-1
                if self.rng.random() < self.sub_rate and alt_cands:
                    c0 = self.rng.choice(alt_cands)
                    c1 = token
                else:
                    c0 = token
                    c1 = alt_cands[0] if alt_cands else token

                remaining = [c for c in alt_cands if c not in (c0, c1)]
                c2 = remaining[0] if remaining else (fallbacks[0] if fallbacks and fallbacks[0] not in (c0, c1) else c1)

                cands = [c0, c1, c2][:top_k]
                while len(cands) < top_k:
                    cands.append(cands[-1])

                # Sample probability mass
                p0 = self.rng.uniform(0.60, 0.88)
                p1 = self.rng.uniform(0.08, 1.0 - p0)
                p2 = max(0.0, 1.0 - p0 - p1)
                probs = [p0, p1, p2][:top_k]
                total = sum(probs)
                probs = [p / total for p in probs]
            all_cands.append(cands)
            all_probs.append(probs)
        return all_cands, all_probs

    def perturb_token_ids(self, token_ids: List[int]) -> List[int]:
        """Apply noise directly to token integer IDs."""
        perturbed = []
        for tid in token_ids:
            if tid == self.space_token_id:
                if self.rng.random() < self.space_drop_rate:
                    continue
                perturbed.append(tid)
                continue

            if self.rng.random() < self.space_insert_rate:
                perturbed.append(self.space_token_id)

            if self.rng.random() < self.del_rate:
                continue

            # Random jitter
            if self.rng.random() < self.sub_rate:
                # Slight token id shift within typical phoneme vocabulary (8..111)
                jitter = self.rng.choice([-2, -1, 1, 2])
                tid = max(8, min(111, tid + jitter))

            perturbed.append(tid)

            if self.rng.random() < self.ins_rate:
                perturbed.append(tid)

        return perturbed
