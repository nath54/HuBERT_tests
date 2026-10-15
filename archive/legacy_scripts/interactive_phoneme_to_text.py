#!/usr/bin/env python3
"""Interactive CLI for Phono-V7.5 True Phoneme-to-Text (P2G).

Allows live interactive input:
  1. Text input -> IPA Phonemization (with automatic language detection & code-switching support).
  2. Display of generated IPA phonemes.
  3. Encoding via FastTextPhonemeWordEncoder -> Latent z_word (512D).
  4. Autoregressive decoding via OverlappingLengthMoEDecoder.
  5. Display of reconstructed orthographic text.

Exit words: 'q', 'exit', 'quit', 'leave', 'stop' or Ctrl+C.
Device: selectable via --device (cpu or cuda).
"""

import argparse
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from piper.voice import EspeakPhonemizer
from src.data.phoneme_tokenizer import PhonemeTokenizer
from src.data.roman_tokenizer import RomanCharTokenizer
from src.models.phono_v7_5_speech_model import PhonoV75SpeechConfig, PhonoV75SpeechModel
from src.models.text_phoneme_encoder import FastTextPhonemeWordEncoder


# --- Vocabulary Sets for Robust Zero-Dependency Language Identification ---
FR_WORDS = {
    'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'd', 'et', 'est', 'en', 'que', 'qui',
    'dans', 'pour', 'pas', 'sur', 'ce', 'cette', 'ces', 'il', 'elle', 'ils', 'elles', 'on',
    'nous', 'vous', 'avec', 'son', 'sa', 'ses', 'au', 'aux', 'ne', 'par', 'mais', 'ou',
    'si', 'tout', 'tous', 'faire', 'plus', 'comme', 'je', 'tu', 'moi', 'toi', 'mon', 'ma',
    'mes', 'ton', 'ta', 'tes', 'notre', 'votre', 'leur', 'leurs', 'y', 'été', 'très', 'bien',
    'ici', 'aussi', 'encore', 'même', 'autre', 'autres', 'va', 'vais', 'sont', 'bonjour',
    'salut', 'merci', 'oui', 'non', 'pourquoi', 'comment', 'quand', 'quel', 'quelle', 'quels'
}

ES_WORDS = {
    'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'de', 'del', 'en', 'y', 'e',
    'que', 'es', 'por', 'para', 'con', 'no', 'se', 'su', 'sus', 'al', 'lo', 'como', 'más',
    'pero', 'yo', 'tú', 'él', 'ella', 'ellos', 'ellas', 'nosotros', 'usted', 'ustedes',
    'muy', 'sin', 'sobre', 'ser', 'estar', 'hacer', 'tener', 'hola', 'gracias', 'amigo',
    'amiga', 'buenos', 'buenas', 'días', 'tardes', 'noches', 'si', 'sí', 'cómo', 'dónde'
}

IT_WORDS = {
    'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una', 'di', 'del', 'della', 'dello',
    'degli', 'delle', 'a', 'al', 'allo', 'alla', 'in', 'nel', 'nello', 'nella', 'nei', 'negli',
    'nelle', 'con', 'su', 'sul', 'sullo', 'sulla', 'per', 'tra', 'fra', 'e', 'è', 'che', 'chi',
    'non', 'si', 'sono', 'ha', 'hanno', 'questo', 'questa', 'questi', 'queste', 'più', 'anche',
    'come', 'molto', 'ciao', 'stai', 'sta', 'amico', 'amica', 'mio', 'mia', 'miei', 'bene',
    'grazie', 'prego', 'perché', 'quando', 'dove', 'cosa', 'quanto', 'qui', 'buongiorno'
}

EN_WORDS = {
    'the', 'is', 'and', 'to', 'in', 'of', 'a', 'that', 'it', 'was', 'for', 'on', 'are', 'as',
    'with', 'his', 'they', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by',
    'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said',
    'there', 'use', 'an', 'each', 'which', 'she', 'do', 'how', 'their', 'if', 'will', 'up',
    'other', 'about', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would',
    'make', 'like', 'him', 'into', 'time', 'has', 'look', 'two', 'more', 'write', 'go',
    'see', 'no', 'way', 'could', 'people', 'my', 'than', 'first', 'water', 'been', 'call',
    'who', 'oil', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made',
    'may', 'part', 'hello', 'hi', 'thanks', 'cool', 'good', 'morning', 'night', 'yes'
}

COMMON_EN_LOANWORDS = {
    'computer', 'software', 'hardware', 'code', 'coding', 'meeting', 'weekend', 'email', 'mail',
    'feedback', 'developer', 'dev', 'bug', 'debug', 'start', 'stop', 'challenge', 'manager',
    'boss', 'job', 'design', 'designer', 'cool', 'stream', 'streaming', 'gamer', 'game',
    'gaming', 'post', 'podcast', 'online', 'offline', 'fast', 'slow', 'workflow', 'team',
    'feature', 'release', 'update', 'link', 'click', 'business', 'marketing', 'smart', 'smartphone'
}

LANG_VOICES = {
    'en': 'en-us',
    'fr': 'fr',
    'es': 'es',
    'it': 'it',
}

LANG_FLAGS = {
    'en': '🇬🇧 English',
    'fr': '🇫🇷 French',
    'es': '🇪🇸 Spanish',
    'it': '🇮🇹 Italian',
}


def detect_sentence_language(text: str) -> str:
    """Detects the primary language of the sentence among en, fr, es, it."""
    tag_match = re.match(r'^\s*\[(en|fr|es|it)\]\s*', text, re.IGNORECASE)
    if tag_match:
        return tag_match.group(1).lower()

    tokens = [t.lower() for t in re.findall(r'\b[a-zA-Záàâäéèêëíìîïóòôöúùûüñçœæ]+\b', text)]
    if not tokens:
        return 'en'

    scores = {'fr': 0, 'en': 0, 'es': 0, 'it': 0}
    for tok in tokens:
        if tok in FR_WORDS:
            scores['fr'] += 3
        if tok in EN_WORDS:
            scores['en'] += 3
        if tok in ES_WORDS:
            scores['es'] += 3
        if tok in IT_WORDS:
            scores['it'] += 3

        # Character features
        if any(c in tok for c in 'çœæâêîôûëï'):
            scores['fr'] += 4
        if any(c in tok for c in 'ñáíóú'):
            scores['es'] += 4
        if any(c in tok for c in 'àèéìòù') and not any(c in tok for c in 'çœ'):
            scores['it'] += 2
        if tok.endswith('zione') or tok.endswith('issimo') or tok.endswith('mente'):
            scores['it'] += 3
        if tok.endswith('ción') or tok.endswith('idad'):
            scores['es'] += 3
        if tok.endswith('ing') and tok not in FR_WORDS:
            scores['en'] += 2

    best_lang = max(scores, key=scores.get)
    if scores[best_lang] > 0:
        return best_lang
    return 'en'


def get_word_language(word: str, default_lang: str) -> str:
    """Determines the specific language for a single word, handling code-switching."""
    clean_w = word.lower().strip(".,!?:;\"'()[]{}«»")
    if not clean_w:
        return default_lang

    # Check for English technical/loanwords in non-English sentences
    if default_lang != 'en':
        if clean_w in COMMON_EN_LOANWORDS:
            return 'en'
        if clean_w.endswith('ing') and len(clean_w) > 4:
            return 'en'
        if clean_w in EN_WORDS and clean_w not in (FR_WORDS | ES_WORDS | IT_WORDS):
            return 'en'

    return default_lang


@lru_cache(maxsize=50000)
def phonemize_word(phonemizer: EspeakPhonemizer, word: str, voice: str) -> List[str]:
    """Phonemizes a single word into a flat list of IPA phonemes."""
    try:
        chunks = phonemizer.phonemize(voice, word)
        return [ch for sub in chunks for ch in sub]
    except Exception:
        return []


import time


def safe_torch_load(path: Path, map_location: torch.device, max_retries: int = 5) -> Dict[str, Any]:
    """Safely loads torch checkpoint with retries in case the training process is writing to it."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(0.5)
    raise RuntimeError(f"Failed to safely load checkpoint from {path} after {max_retries} retries: {last_err}")


def clean_input_sentence(text: str) -> Tuple[str, Optional[str]]:
    """Strips manual language tags if present and returns clean text and forced lang."""
    tag_match = re.match(r'^\s*\[(en|fr|es|it)\]\s*(.*)$', text, re.IGNORECASE)
    if tag_match:
        return tag_match.group(2).strip(), tag_match.group(1).lower()
    return text.strip(), None


def process_sentence(
    raw_input: str,
    encoder: nn.Module,
    model: nn.Module,
    cfg: Any,
    phonemizer: EspeakPhonemizer,
    ph_tok: PhonemeTokenizer,
    rom_tok: RomanCharTokenizer,
    device: torch.device,
    args: Any,
) -> str:
    """Processes a single sentence, prints IPA breakdown, and returns decoded text."""
    clean_text, forced_lang = clean_input_sentence(raw_input)
    if not clean_text:
        return ""

    # Language Detection
    sentence_lang = args.lang or forced_lang or detect_sentence_language(clean_text)
    flag_info = LANG_FLAGS.get(sentence_lang, sentence_lang.upper())

    # Extract genuine words, stripping isolated punctuation marks (?, !, ., etc.)
    raw_tokens = [w.strip() for w in clean_text.split() if w.strip()]
    words: List[str] = []
    for w in raw_tokens:
        w_clean = re.sub(r'^[^\w]+|[^\w]+$', '', w)
        if w_clean and any(c.isalnum() for c in w_clean):
            words.append(w_clean)

    if not words:
        return ""

    # Convert Words -> IPA Phonemes (with word-level code-switching)
    word_phoneme_chunks: List[List[str]] = []
    word_phoneme_ids: List[List[int]] = []
    detected_word_langs: List[str] = []

    for w_clean in words:
        # Detect language for word (allows English loanwords in French/Spanish/Italian)
        w_lang = args.lang or forced_lang or get_word_language(w_clean, sentence_lang)
        detected_word_langs.append(w_lang)

        voice = LANG_VOICES.get(w_lang, "en-us")
        ipa_chars = phonemize_word(phonemizer, w_clean, voice)
        word_phoneme_chunks.append(ipa_chars)

        # Tokenize to IDs
        p_ids = ph_tok.encode(ipa_chars)[:24]
        word_phoneme_ids.append(p_ids)

    # Build full sentence IPA representation
    full_ipa = " ".join("".join(chunk) if chunk else "?" for chunk in word_phoneme_chunks)

    # Notice code-switching in display if any
    mixed_notes = []
    for w, w_l in zip(words, detected_word_langs):
        if w_l != sentence_lang:
            mixed_notes.append(f"'{w}' ({w_l})")
    mixed_str = f" [Code-Switching: {', '.join(mixed_notes)}]" if mixed_notes else ""

    print(f"  🌐 \033[1mDetected Language\033[0m : {flag_info}{mixed_str}")
    print(f"  🗣️  \033[1mIPA Phonemes\033[0m      : \033[1;35m{full_ipa}\033[0m")

    # Prepare Batch Tensor for Model
    L = len(words)
    max_k_ph = max(1, max(len(p) for p in word_phoneme_ids))
    input_ph = torch.zeros(1, L, max_k_ph, dtype=torch.long, device=device)
    for i, p in enumerate(word_phoneme_ids):
        if p:
            input_ph[0, i, :len(p)] = torch.tensor(p, dtype=torch.long, device=device)

    # Forward Pass: IPA -> z_word (512D) -> Autoregressive MoE Decode
    with torch.no_grad():
        flat_ph = input_ph.reshape(L, max_k_ph)
        z_words = encoder(phoneme_ids=flat_ph).view(1, L, -1)

        expert_logits = model.decoder.macro_expert_router(z_words)
        top1_experts = expert_logits.argmax(dim=-1)
        expert_cond = model.decoder.expert_embeddings(top1_experts)
        z_windows = model.decoder.build_word_context_windows(z_words)

        pred_word_chars: List[List[int]] = []
        for l in range(L):
            zw_window = z_windows[:, l, :, :]
            p_bias = expert_cond[:, l : l + 1, :]
            p_id = top1_experts[:, l].clamp(0, cfg.decoder_config.num_paths - 1)

            cur_tokens = [rom_tok.bos_id]
            for _ in range(24):
                inp = torch.tensor([cur_tokens], dtype=torch.long, device=device)
                flat_logits, _ = model.decoder.micro_head(
                    inp,
                    zw_window,
                    acoustic_slices=None,
                    path_bias=p_bias,
                    path_id=p_id,
                )
                next_logits = flat_logits[0, -1, :]
                if args.temperature > 0.0:
                    probs = F.softmax(next_logits / args.temperature, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_tok = next_logits.argmax(dim=-1).item()

                if next_tok in (rom_tok.eos_id, rom_tok.pad_id):
                    break
                cur_tokens.append(next_tok)

            pred_word_chars.append(cur_tokens[1:])

        output_text = rom_tok.decode_words(pred_word_chars)

    print(f"  📝 \033[1;32mDecoded Text\033[0m      : \033[1m{output_text}\033[0m")
    return output_text


def main():
    parser = argparse.ArgumentParser(description="Interactive Phono-V7.5 True Phoneme-to-Text CLI")
    parser.add_argument(
        "-p",
        "--prompt",
        "--text",
        type=str,
        default=None,
        help="Direct text to test. Runs inference once and exits without entering interactive loop.",
    )
    parser.add_argument(
        "--checkpoint",
        "--weights",
        type=str,
        default="checkpoints/phono_v7_5_phoneme_to_text/best_p2g.pt",
        help="Path to trained Phoneme-to-Text model weights (default: best_p2g.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda", "auto"],
        help="Target device to run inference on ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for deterministic greedy decoding)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        choices=["en", "fr", "es", "it"],
        help="Force a specific language across all inputs (disables auto-detection)",
    )
    args = parser.parse_args()

    # Determine device
    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        # Fallback to latest_p2g.pt if best_p2g.pt not found
        alt_path = ckpt_path.parent / "latest_p2g.pt"
        if alt_path.is_file():
            ckpt_path = alt_path
        else:
            print(f"\n❌ Error: Checkpoint file not found at: {args.checkpoint}", file=sys.stderr)
            sys.exit(1)

    print("=" * 80)
    print("🎧 Phono-V7.5 | Interactive True Phoneme-to-Text Terminal")
    print("=" * 80)
    print(f"📦 Loading weights from: \033[1;36m{ckpt_path}\033[0m")
    print(f"⚡ Device               : \033[1;32m{device.type.upper()}\033[0m")

    # 1. Initialize Tokenizers & Phonemizer
    phonemizer = EspeakPhonemizer()
    ph_tok = PhonemeTokenizer()
    rom_tok = RomanCharTokenizer()

    # 2. Build Model & Safely Load Checkpoint (with retry protection against concurrent training writes)
    cfg = PhonoV75SpeechConfig.medium()
    encoder = FastTextPhonemeWordEncoder(
        d_macro=cfg.decoder_config.macro_dim,
        d_model=256,
        nhead=4,
        num_layers=2,
        phoneme_vocab_size=128,
    ).to(device)

    model = PhonoV75SpeechModel(cfg).to(device)

    ckpt = safe_torch_load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt["phoneme_encoder_state_dict"])
    model.decoder.load_state_dict(ckpt["decoder_state_dict"])
    encoder.eval()
    model.eval()

    step_info = ckpt.get("step", "N/A")
    val_wer = ckpt.get("val_wer", None)
    wer_str = f"{val_wer:.2f}%" if val_wer is not None else "N/A"
    print(f"✅ Model successfully loaded (Step {step_info} | Best Val WER: {wer_str})")
    print("-" * 80)

    # If -p / --prompt was passed, execute directly and exit
    if args.prompt:
        print(f"\033[1;33mInput Text\033[0m           : {args.prompt}")
        process_sentence(args.prompt, encoder, model, cfg, phonemizer, ph_tok, rom_tok, device, args)
        print("=" * 80)
        return

    print("💡 Instructions:")
    print("  • Type any sentence in English, French, Spanish, or Italian.")
    print("  • Language & English loanwords in mixed sentences are automatically detected!")
    print("  • Prefix with [en], [fr], [es], or [it] to force a specific language.")
    print("  • Type 'q', 'exit', 'quit', 'leave', 'stop' or press Ctrl+C to quit.")
    print("-" * 80)

    EXIT_COMMANDS = {"q", "exit", "quit", "leave", "stop"}

    while True:
        try:
            raw_input = input("\n\033[1;33mPhono-V7.5 > \033[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Exiting. Goodbye!")
            break

        if not raw_input:
            continue

        clean_text, _ = clean_input_sentence(raw_input)
        if clean_text.lower() in EXIT_COMMANDS:
            print("\n👋 Exiting. Goodbye!")
            break

        process_sentence(raw_input, encoder, model, cfg, phonemizer, ph_tok, rom_tok, device, args)


if __name__ == "__main__":
    main()
