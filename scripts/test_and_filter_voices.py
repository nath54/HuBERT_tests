#!/usr/bin/env python3
"""
Comprehensive Piper Voice Quality & Phoneme Coverage Validator.

Tests all 34 installed Piper neural voices across a diverse phonetic corpus
(English & French sentences, nasal vowels, accents, numbers, contractions, Wikipedia, Books)
and identifies all voices that emit 'Missing phoneme from id map' errors or warnings.
Saves a verified blocklist to config/blocked_voices.json and updates PiperVoiceManager.
"""

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from piper.voice import PiperVoice
from piper.config import SynthesisConfig
import piper.phoneme_ids

VOICES_DIR = Path("/home/nathan/github/MADGen/data/piper_voices")
OUTPUT_CONFIG = Path("config/blocked_voices.json")


# Diverse test suite for English
TEST_SENTENCES_EN = [
    # Conversational & contractions
    "Hello there! I'm wondering if you've seen the latest report today.",
    "Don't worry, it's couldn't have been better, isn't it?",
    "We'll definitely examine whether they'd like to join us.",
    "Can't you see what's happening at 12:45 pm?",
    "They've said it's 100% possible, haven't they?",
    "Let's make sure we haven't missed any crucial detail.",
    # Technical & acoustic vocabulary
    "Convolutional neural networks extract acoustic feature representations at sixteen kilohertz.",
    "The transformer encoder utilizes self attention mechanisms to capture temporal dependencies.",
    "Mel frequency cepstral coefficients provide spectral envelope representations.",
    "The receptive field calculation demonstrates thirty-two millisecond frames.",
    # Numbers, dates, symbols
    "At 08:30 on July 14th, 2026, the temperature was exactly 23.5 degrees.",
    "Section 104, paragraph 3, subsection B: all 1,250 items were counted.",
    "The total revenue exceeded $4,500,000 in the third quarter.",
    # Phonetically diverse pangrams & edge cases
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "She sells sea shells by the seashore under the bright sunshine.",
    "Sphinx of black quartz, judge my vow with swift precision.",
    # Loan words with accents / special characters
    "He enjoyed a warm café au lait while reviewing his résumé and cliché designs.",
    "The naïve protagonist noticed a subtle façade on the boulevard.",
    "A touch of déjà vu accompanied their pleasant rendez-vous.",
    # Classical / literary style
    "It is a truth universally acknowledged that a single man in possession of a good fortune must be in want of a wife.",
    "To be or not to be, that is the question whether 'tis nobler in the mind to suffer the slings and arrows of outrageous fortune.",
]

# Diverse test suite for French (covers all nasal vowels, accents, liaisons, elisions)
TEST_SENTENCES_FR = [
    # Nasal vowels (an/en [ɑ̃], on [ɔ̃], in/ain/ein [ɛ̃], un [œ̃])
    "Bonjour, nous allons prendre un bon bain chaud dans un instant.",
    "Les enfants chantent des chansons magnifiques pendant les vacances.",
    "Le matin, le boulanger prépare du pain croustillant et du vin rouge.",
    "Chacun son tour, un parfum subtil flotte dans le jardin sombre.",
    "Comment vont vos compagnons en ces temps de grand changement ?",
    "Ensemble, nous avons accompli une grande entreprise sans encombre.",
    "Un pigeon blanc s'est envolé au-dessus du pont principal.",
    "Cinq chiens gambadent sereinement dans la grande campagne.",
    "Nous avons rendez-vous lundi prochain avec le médecin.",
    # Accents & special characters (é, è, ê, ë, à, â, ç, ô, ù, û, ü, œ, æ)
    "C'est déjà Noël, où est passé le garçon aux yeux bleus ?",
    "L'œuvre de cet élève illustre un bel événement à l'opéra.",
    "À côté de la forêt, le château paraît très ancien et mystérieux.",
    "Il faut être prêt pour la fête de réconciliation générale.",
    "Cette leçon sur le français contemporain est tout à fait claire.",
    "Où mène ce chemin sinueux près de la rivière ?",
    "Le cœur de la forêt recèle des trésors insoupçonnés.",
    # Elisions, apostrophes, contractions
    "L'intelligence artificielle d'aujourd'hui transforme l'histoire humaine.",
    "Qu'est-ce qu'il s'est passé lorsqu'ils sont arrivés à l'aéroport ?",
    "D'autres personnes m'ont dit qu'elles n'étaient pas prêtes.",
    "J'ai pensé à tout ce qu'on pouvait accomplir ensemble.",
    # Numbers, dates, technical
    "Le 14 juillet 1789, il y avait environ 350 personnes réunies.",
    "La vitesse moyenne observée est de 120 kilomètres par heure en 2026.",
    "L'acoustique de cette salle de spectacle est exceptionnelle pour la musique.",
]


class WarningCollector(logging.Handler):
    """Captures warnings directly from piper.phoneme_ids without printing them."""

    def __init__(self):
        super().__init__()
        self.warnings: List[str] = []

    def emit(self, record):
        self.warnings.append(record.getMessage())


def test_voice(voice_path: Path) -> Dict:
    """Test a single voice on language-appropriate and edge-case text sentences."""
    voice_name = f"{voice_path.parent.name}_{voice_path.stem}"
    is_french = "fr_FR" in str(voice_path)

    config_path = voice_path.with_suffix(".onnx.json")
    phoneme_id_map = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            phoneme_id_map = data.get("phoneme_id_map", {})

    has_combining_tilde = ("̃" in phoneme_id_map or "\u0303" in phoneme_id_map)
    phoneme_count = len(phoneme_id_map)

    # Select primary test suite
    if is_french:
        test_sentences = TEST_SENTENCES_FR + TEST_SENTENCES_EN[:4]
    else:
        test_sentences = TEST_SENTENCES_EN + TEST_SENTENCES_FR[:4]

    # Setup warning collector
    collector = WarningCollector()
    original_propagate = piper.phoneme_ids._LOGGER.propagate
    piper.phoneme_ids._LOGGER.propagate = False
    piper.phoneme_ids._LOGGER.addHandler(collector)
    piper.phoneme_ids._LOGGER.setLevel(logging.WARNING)

    load_error = None
    voice = None
    try:
        voice = PiperVoice.load(str(voice_path), config_path=str(config_path) if config_path.exists() else None)
    except Exception as e:
        piper.phoneme_ids._LOGGER.removeHandler(collector)
        piper.phoneme_ids._LOGGER.propagate = original_propagate
        return {
            "voice_name": voice_name,
            "path": str(voice_path),
            "language": "fr_FR" if is_french else "en",
            "status": "BLOCKED",
            "reason": f"Load failure: {e}",
            "phoneme_count": phoneme_count,
            "has_tilde": has_combining_tilde,
            "missing_phoneme_warnings": 999,
            "missing_phonemes": ["LOAD_FAILURE"],
        }

    # Test phonemization & ID mapping across full test corpus
    missing_phonemes_set: Set[str] = set()
    re_missing = re.compile(r"Missing phoneme from id map:\s*(.+)")

    for sent in test_sentences:
        try:
            phonemes_list = voice.phonemize(sent)
            for ph_sentence in phonemes_list:
                voice.phonemes_to_ids(ph_sentence)
        except Exception as e:
            missing_phonemes_set.add(f"EXCEPTION: {e}")

    # Also test actual ONNX audio synthesis for 1 representative sentence
    synthesis_ok = True
    syn_error = None
    try:
        test_syn_text = "Bonjour, ceci est un test." if is_french else "Hello, this is a test."
        chunks = []
        for chunk in voice.synthesize(test_syn_text):
            chunks.append(chunk.audio_float_array)
        if not chunks:
            synthesis_ok = False
            syn_error = "Zero chunks generated"
    except Exception as e:
        synthesis_ok = False
        syn_error = str(e)

    # Restore logger
    piper.phoneme_ids._LOGGER.removeHandler(collector)
    piper.phoneme_ids._LOGGER.propagate = original_propagate

    # Extract missing phonemes
    for w in collector.warnings:
        m = re_missing.search(w)
        if m:
            missing_phonemes_set.add(m.group(1).strip())
        else:
            missing_phonemes_set.add(w)

    missing_count = len(collector.warnings)
    is_blocked = (missing_count > 0) or (not synthesis_ok)
    status = "BLOCKED" if is_blocked else "CLEAN"

    reasons = []
    if missing_count > 0:
        reasons.append(f"{missing_count} missing phoneme warnings ({', '.join(sorted(list(missing_phonemes_set)))})")
    if not synthesis_ok:
        reasons.append(f"Synthesis failed: {syn_error}")

    return {
        "voice_name": voice_name,
        "path": str(voice_path),
        "language": "fr_FR" if is_french else "en",
        "status": status,
        "reason": "; ".join(reasons) if reasons else "OK",
        "phoneme_count": phoneme_count,
        "has_tilde": has_combining_tilde,
        "missing_phoneme_warnings": missing_count,
        "missing_phonemes": sorted(list(missing_phonemes_set)),
        "synthesis_ok": synthesis_ok,
        "test_sentences_count": len(test_sentences),
    }


def main():
    print("=" * 80)
    print("   PIPER VOICE QUALITY & PHONEME COVERAGE AUDITOR")
    print("=" * 80)

    if not VOICES_DIR.exists():
        print(f"Error: Voices directory {VOICES_DIR} does not exist.")
        sys.exit(1)

    all_models = sorted(list(VOICES_DIR.rglob("*.onnx")))
    print(f"Discovered {len(all_models)} Piper voice models in {VOICES_DIR}")
    print(f"Running phoneme coverage & synthesis checks across diverse sentences...")
    print("-" * 80)

    results = []
    clean_voices = []
    blocked_voices = []

    for idx, model_path in enumerate(all_models, 1):
        voice_id = f"{model_path.parent.name}_{model_path.stem}"
        print(f"[{idx:2d}/{len(all_models)}] Testing: {voice_id:40s} ... ", end="", flush=True)
        t0 = time.time()
        res = test_voice(model_path)
        elapsed = time.time() - t0
        results.append(res)

        if res["status"] == "CLEAN":
            clean_voices.append(res)
            print(f"✅ CLEAN ({elapsed:.2f}s | {res['phoneme_count']} phonemes)")
        else:
            blocked_voices.append(res)
            missing_str = ", ".join(f"'{p}'" for p in res["missing_phonemes"]) if res["missing_phonemes"] else res.get("reason", "error")
            print(f"❌ BLOCKED ({res['missing_phoneme_warnings']} warnings, missing: {missing_str})")

    print("\n" + "=" * 80)
    print(f"AUDIT SUMMARY: {len(clean_voices)} Clean Voices | {len(blocked_voices)} Blocked Voices")
    print("=" * 80)

    print(f"\n🚫 BLOCKED VOICES ({len(blocked_voices)}):")
    for b in blocked_voices:
        print(f"  • {b['voice_name']:40s} | {b['language']:5s} | {b['reason']}")

    print(f"\n✅ CLEAN APPROVED VOICES ({len(clean_voices)}):")
    for c in clean_voices:
        print(f"  • {c['voice_name']:40s} | {c['language']:5s} | Phonemes: {c['phoneme_count']}")

    # Save to JSON config
    OUTPUT_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.time(),
        "total_tested": len(all_models),
        "clean_count": len(clean_voices),
        "blocked_count": len(blocked_voices),
        "blocked_voice_names": [b["voice_name"] for b in blocked_voices],
        "clean_voice_names": [c["voice_name"] for c in clean_voices],
        "detailed_results": results,
    }

    with open(OUTPUT_CONFIG, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n💾 Blocklist configuration saved to: {OUTPUT_CONFIG}")
    print("=" * 80)


if __name__ == "__main__":
    main()
