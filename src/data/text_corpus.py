"""
Streaming Text Corpus Module.
Streams clean, natural sentences in English and French from Wikipedia,
classic literature (Opus Books / Project Gutenberg), and curated dialogue banks.
Designed for 0-disk streaming text-to-speech synthesis.
"""

import re
import random
from typing import Iterator, Optional, Generator


class StreamingTextCorpus:
    """
    Streams clean, naturally punctuated sentences from high-quality online sources
    (Wikipedia, Opus Books) and offline fallbacks with 0 disk storage footprint.
    """

    def __init__(self, languages: tuple[str, ...] = ("en", "fr")):
        self.languages = list(languages)
        self._book_stream = None
        self._wiki_stream = None
        self._init_streams()

    def _init_streams(self):
        """Initialize streaming generators from HuggingFace datasets."""
        from datasets import load_dataset

        # 1. Literary Books (Opus Books en-fr)
        try:
            self._book_stream = iter(
                load_dataset(
                    "Helsinki-NLP/opus_books",
                    "en-fr",
                    split="train",
                    streaming=True,
                )
            )
            print("[TextCorpus] Initialized streaming Opus Books (EN-FR literature).")
        except Exception as e:
            print(f"[TextCorpus] Note: Opus Books stream unavailable ({e}).")
            self._book_stream = None

        # 2. Wikipedia (Salesforce/wikitext for fast encyclopedic text)
        try:
            self._wiki_stream = iter(
                load_dataset(
                    "Salesforce/wikitext",
                    "wikitext-103-v1",
                    split="train",
                    streaming=True,
                )
            )
            print("[TextCorpus] Initialized streaming Wikipedia (Wikitext-103).")
        except Exception as e:
            print(f"[TextCorpus] Note: Wikipedia stream unavailable ({e}).")
            self._wiki_stream = None

    def _clean_sentence(self, text: str) -> Optional[str]:
        """Normalize and filter text into a clean TTS sentence."""
        if not text:
            return None

        # Remove Wiki headers (= = Title = =) and HTML tags
        text = re.sub(r"=+.*?=+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"\(.*?\)", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Split into individual sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        valid = []
        for s in sentences:
            s = s.strip().strip("\"'«»“”")
            words = s.split()
            # Retain sentences with 6 to 25 words (optimal for TTS and HuBERT 2-8s frames)
            if 6 <= len(words) <= 25 and re.search(r"[a-zA-Zà-üÀ-Ü]", s):
                if not s.endswith((".", "!", "?")):
                    s += "."
                valid.append(s)

        return random.choice(valid) if valid else None

    def sample_sentence(self, lang: Optional[str] = None) -> tuple[str, str]:
        """
        Sample a clean sentence.
        Returns: (sentence_text, language_code)
        """
        chosen_lang = lang if lang in self.languages else random.choice(self.languages)

        # Try books (supports EN and FR)
        if self._book_stream is not None:
            try:
                for _ in range(12):
                    item = next(self._book_stream)
                    trans = item.get("translation", {})
                    candidate = trans.get(chosen_lang, "")
                    cleaned = self._clean_sentence(candidate)
                    if cleaned:
                        return cleaned, chosen_lang
            except StopIteration:
                self._init_streams()
            except Exception:
                pass

        # Try Wikipedia (primarily EN)
        if chosen_lang == "en" and self._wiki_stream is not None:
            try:
                for _ in range(12):
                    item = next(self._wiki_stream)
                    raw = item.get("text", "")
                    cleaned = self._clean_sentence(raw)
                    if cleaned:
                        return cleaned, "en"
            except StopIteration:
                self._init_streams()
            except Exception:
                pass

        # High-quality fallback dialogue bank
        if chosen_lang == "fr":
            fr_samples = [
                "Il arriva chez nous un dimanche matin au début de l'hiver.",
                "Les conditions météorologiques se sont nettement améliorées ces dernières heures.",
                "L'intelligence artificielle permet d'explorer des représentations acoustiques profondes.",
                "Nous avons parcouru la forêt pendant plusieurs heures sans rencontrer personne.",
                "Cette nouvelle méthode d'apprentissage auto-supervisé offre d'excellentes perspectives.",
                "Le train à grande vitesse arrivera en gare avec quelques minutes d'avance.",
            ]
            return random.choice(fr_samples), "fr"

        en_samples = [
            "The unexpected discovery led to significant advances in acoustic modeling.",
            "Scientists have observed unusual acoustic patterns under these experimental conditions.",
            "Modern speech recognition systems require substantial amounts of diverse speech.",
            "The library contains an extensive collection of rare historical manuscripts.",
            "Every morning the professor took a long solitary walk through the gardens.",
            "Careful calibration of the instruments is essential before conducting the test.",
        ]
        return random.choice(en_samples), "en"
