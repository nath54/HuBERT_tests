"""Automated Spoken Number & Roman Numeral Disambiguation & Patching.

Leverages the Acoustic Model's frame log-likelihoods (CTC forced alignment)
to automatically select the exact spoken form (cardinal, ordinal, gendered, or raw)
that matches the ground-truth audio.
"""

import itertools
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import soundfile as sf
import torch
import torch.nn as nn
from num2words import num2words

from src.data.phoneme_tokenizer import PhonemeTokenizer
from src.data.target_extractors import PhonemeTargetExtractor


# Unambiguous multi-character Roman numerals that are NEVER natural words in EN/FR/ES/IT
UNAMBIGUOUS_ROMAN: Dict[str, int] = {
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "vii": 7,
    "viii": 8,
    "ix": 9,
    "xi": 11,
    "xii": 12,
    "xiii": 13,
    "xiv": 14,
    "xv": 15,
    "xvi": 16,
    "xvii": 17,
    "xviii": 18,
    "xix": 19,
    "xx": 20,
    "xxi": 21,
    "xxii": 22,
    "xxiii": 23,
    "xxiv": 24,
    "xxv": 25,
    "xxvi": 26,
    "xxvii": 27,
    "xxviii": 28,
    "xxix": 29,
    "xxx": 30,
    "xl": 40,
    "lx": 60,
    "lxx": 70,
    "lxxx": 80,
    "xc": 90,
}

# Ambiguous Roman numerals that can be natural words or clitics
# (e.g. 'vi' is pronoun/verb in IT/ES, 'x'/'v'/'l'/'c'/'d'/'m' are letters or abbreviations)
CONTEXT_ROMAN: Dict[str, int] = {
    "v": 5,
    "vi": 6,
    "x": 10,
    "l": 50,
    "c": 100,
}

# Literary work context words (chapters, cantos, acts, books)
WORK_CONTEXT_WORDS = {
    "canto", "capitolo", "capitulo", "chapitre", "chapter", "tome", "livre", "libro", "book",
    "part", "parte", "partie", "act", "atto", "acto", "acte", "scena", "scene", "favola",
    "canzone", "sonetto", "section", "sezione", "seccion", "canon", "canone", "numero",
    "numéro", "number", "regola", "art", "article", "articolo", "articulo", "secolo",
    "siècle", "siglo", "century",
}

# Monarch and Pope names (regnal numerals max out around 25)
REGNAL_NAMES = {
    "louis", "henri", "charles", "carlo", "carlos", "alfonso", "george", "edward",
    "felipe", "pietro", "papa", "pape", "pope", "re", "roi", "king", "duc", "duca", "duque",
}

ALL_NUMERAL_CONTEXTS = WORK_CONTEXT_WORDS.union(REGNAL_NAMES)

# Head nouns with known grammatical gender across Romance languages
MASCULINE_HEAD_NOUNS = {
    "canto", "capitolo", "capitulo", "chapitre", "chapter", "livre", "libro", "book",
    "tome", "sonetto", "atto", "act", "acto", "acte", "re", "roi", "king", "duc", "duca",
    "duque", "pape", "papa", "pope", "secolo", "siècle", "siglo", "century",
}

FEMININE_HEAD_NOUNS = {
    "favola", "canzone", "parte", "partie", "part", "scena", "scene", "regola", "regla",
    "règle", "sezione", "sección", "section", "reine", "regina", "queen",
}

# Common Italian verbs that follow clitic pronoun 'vi' (e.g. 'vi ha preso', 'vi sta')
ITALIAN_VERB_FOLLOWING_CLITIC = {
    "ha", "ho", "hanno", "hai", "è", "era", "erano", "fu", "furono", "sta", "stanno",
    "stava", "sono", "pare", "parve", "sembra", "disse", "dico", "offre", "offrono",
    "prego", "fece", "fanno", "possono", "può", "lascio", "spiego", "feci",
}


def get_number_expansions(val: int, lang: str, head_noun_gender: Optional[str] = None) -> List[Tuple[str, str]]:
    """Generate potential spoken word candidates for a given integer value."""
    expansions = []
    seen = set()

    def add(label: str, txt: str):
        cleaned = txt.strip().lower()
        cleaned = cleaned.replace(",", " ").replace("-", " ")
        cleaned = " ".join(cleaned.split())
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            expansions.append((label, cleaned))

    # 1. Cardinal
    try:
        card = num2words(val, lang=lang)
        add("cardinal", card)
    except Exception:
        pass

    # 2. Ordinal Masculine (add if gender is not strictly feminine)
    if head_noun_gender != "f":
        try:
            ord_m = num2words(val, lang=lang, to="ordinal")
            add("ordinal_m", ord_m)
        except Exception:
            pass

    # 3. Ordinal Feminine (add if gender is not strictly masculine)
    if head_noun_gender != "m":
        if lang in ("it", "es"):
            try:
                ord_base = num2words(val, lang=lang, to="ordinal")
                words = ord_base.split()
                fem_words = [w[:-1] + "a" if w.endswith("o") else w for w in words]
                add("ordinal_f", " ".join(fem_words))
            except Exception:
                pass
        elif lang == "fr":
            try:
                ord_base = num2words(val, lang=lang, to="ordinal")
                if ord_base == "premier":
                    add("ordinal_f", "première")
                else:
                    add("ordinal_f", ord_base)
            except Exception:
                pass

    # 4. Years (e.g. 1806, 1914)
    if 1000 <= val <= 2099:
        try:
            yr = num2words(val, lang=lang, to="year")
            add("year", yr)
        except Exception:
            pass

        # French 1100-1999: "dix-neuf cent quatorze"
        if lang == "fr" and 1100 <= val <= 1999:
            century = val // 100
            remainder = val % 100
            try:
                c_str = num2words(century, lang="fr")
                r_str = num2words(remainder, lang="fr") if remainder > 0 else ""
                fr_alt = f"{c_str} cent {r_str}".strip()
                add("year_hundreds", fr_alt)
            except Exception:
                pass

    return expansions


def find_number_spans(text: str, lang: str) -> List[Tuple[int, int, str, int, str, Optional[str]]]:
    """Locate all number and Roman numeral spans in text.

    Returns:
        List of tuples: (start_char, end_char, original_token, integer_value, type, head_noun_gender)
    """
    spans = []
    tokens_matches = list(re.finditer(r"\b[a-zA-Z0-9\'-]+\b", text))

    for idx, match in enumerate(tokens_matches):
        raw_token = match.group().lower()
        start, end = match.start(), match.end()

        # Ignore French demonstrative clitics like "-ci" (e.g. "celle-ci")
        if raw_token.endswith("-ci") or raw_token.startswith("-ci"):
            continue

        # Strip leading elisions (e.g. "l'18", "d'14")
        token = raw_token
        if token.startswith("l'") or token.startswith("d'") or token.startswith("c'"):
            start += 2
            token = token[2:]

        prev_token = tokens_matches[idx - 1].group().lower() if idx > 0 else ""
        clean_prev = re.sub(r"^[ldc]\'", "", prev_token)
        next_token = tokens_matches[idx + 1].group().lower() if idx + 1 < len(tokens_matches) else ""
        clean_next = re.sub(r"^[ldc]\'", "", next_token)

        gender = None
        if clean_prev in MASCULINE_HEAD_NOUNS:
            gender = "m"
        elif clean_prev in FEMININE_HEAD_NOUNS:
            gender = "f"

        # Case 1: Arabic Digits
        if token.isdigit():
            val = int(token)
            spans.append((start, end, text[start:end], val, "digit", gender))
            continue

        # Filter out Italian clitic pronoun 'vi' followed by verb
        if lang == "it" and token == "vi" and clean_next in ITALIAN_VERB_FOLLOWING_CLITIC:
            continue

        # Filter out Spanish verb 'vi'
        if lang == "es" and token == "vi" and clean_prev not in ALL_NUMERAL_CONTEXTS:
            continue

        # Case 2: Unambiguous Roman Numerals
        if token in UNAMBIGUOUS_ROMAN:
            val = UNAMBIGUOUS_ROMAN[token]
            if clean_prev in REGNAL_NAMES and val > 25:
                continue
            spans.append((start, end, text[start:end], val, "unambiguous_roman", gender))
            continue

        # Case 3: Contextual Roman Numerals
        if token in CONTEXT_ROMAN:
            if clean_prev in ALL_NUMERAL_CONTEXTS:
                val = CONTEXT_ROMAN[token]
                if clean_prev in REGNAL_NAMES and val > 25:
                    continue
                # Do not treat as numeral if immediately followed by verb
                if lang == "it" and token == "vi" and clean_next in ITALIAN_VERB_FOLLOWING_CLITIC:
                    continue
                spans.append((start, end, text[start:end], val, f"contextual_roman_{clean_prev}", gender))

    return spans


def generate_candidate_transcripts(text: str, lang: str) -> List[Tuple[str, str]]:
    """Generate all combinatorial candidate transcripts with expanded numbers.

    Returns:
        List of tuples: (candidate_label, candidate_text)
    """
    spans = find_number_spans(text, lang)
    if not spans:
        return [("raw", text)]

    span_candidates: List[List[Tuple[str, str, int, int]]] = []
    for start, end, orig_txt, val, span_type, gender in spans:
        replacements = [("raw", orig_txt, start, end)]
        num_exps = get_number_expansions(val, lang, head_noun_gender=gender)
        for lbl, exp in num_exps:
            replacements.append((f"{orig_txt}_{lbl}", exp, start, end))
        span_candidates.append(replacements)

    all_combos = list(itertools.product(*span_candidates))
    candidates = []
    seen_texts = set()

    for combo in all_combos:
        new_text = text
        labels = []
        for lbl, replacement, start, end in sorted(combo, key=lambda x: x[2], reverse=True):
            new_text = new_text[:start] + replacement + new_text[end:]
            labels.append(lbl)

        new_text = " ".join(new_text.split())
        if new_text not in seen_texts:
            seen_texts.add(new_text)
            candidates.append(("-".join(labels), new_text))

    return candidates


class SpokenNumberPatcher:
    """Acoustic Model Forced Alignment Patcher for Spoken Numbers."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[PhonemeTokenizer] = None,
        extractor: Optional[PhonemeTargetExtractor] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer or PhonemeTokenizer()
        self.extractor = extractor or PhonemeTargetExtractor(self.tokenizer)
        self.ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    @torch.no_grad()
    def evaluate_candidates(
        self,
        audio_tensor: torch.Tensor,
        candidates: List[Tuple[str, str]],
        lang: str,
    ) -> List[Dict[str, Any]]:
        """Score candidate transcripts against audio using CTC loss."""
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)

        out = self.model(audio_tensor)
        logits = out["logits"]
        log_probs = logits.transpose(0, 1).float()
        t_len = torch.tensor([logits.shape[1]], device=self.device)

        results = []
        for label, cand_text in candidates:
            td = self.extractor.extract_targets(waveform=audio_tensor[0], text=cand_text, lang=lang)
            targets = td["targets"].to(self.device).unsqueeze(0)
            target_len = torch.tensor([targets.shape[1]], device=self.device)

            loss = self.ctc_loss_fn(log_probs, targets, t_len, target_len).item()
            results.append({
                "label": label,
                "text": cand_text,
                "ctc_loss": loss,
                "target_len": targets.shape[1],
                "phoneme_str": td.get("phoneme_str", ""),
            })

        results.sort(key=lambda x: x["ctc_loss"])
        return results

    def patch_utterance(
        self,
        audio_path: str,
        transcript: str,
        lang: str,
        gain_threshold: float = 0.05,
    ) -> Dict[str, Any]:
        """Examine an utterance, generate candidate expansions, and patch if winner is better."""
        candidates = generate_candidate_transcripts(transcript, lang)

        if len(candidates) <= 1:
            return {
                "patched": False,
                "original_text": transcript,
                "patched_text": transcript,
                "best_label": "raw",
                "loss_gain": 0.0,
                "candidates": [],
            }

        if not os.path.isfile(audio_path):
            return {
                "patched": False,
                "original_text": transcript,
                "patched_text": transcript,
                "best_label": "audio_missing",
                "loss_gain": 0.0,
                "candidates": [],
            }

        wav, _ = sf.read(audio_path)
        x = torch.from_numpy(wav).float()

        scored = self.evaluate_candidates(x, candidates, lang)
        raw_result = next((r for r in scored if "raw" in r["label"].lower()), scored[-1])
        winner = scored[0]

        gain = raw_result["ctc_loss"] - winner["ctc_loss"]
        is_patched = (winner["text"] != transcript) and (gain >= gain_threshold)

        return {
            "patched": is_patched,
            "original_text": transcript,
            "patched_text": winner["text"] if is_patched else transcript,
            "best_label": winner["label"],
            "raw_loss": raw_result["ctc_loss"],
            "best_loss": winner["ctc_loss"],
            "loss_gain": gain,
            "candidates": scored,
        }
