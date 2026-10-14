#!/usr/bin/env python3
"""Phono-V8.0 Comprehensive End-to-End Multilingual Evaluation Suite.

Evaluates the complete tripartite pipeline:
  Audio (16 kHz)
    │
    ▼ (Model 1: PhonoV8AcousticBackbone)
  Acoustic Phonemes + Spaces
    │
    ▼ (Model 2: FastTextPhonemeWordEncoder)
  Word Latents (z_word)
    │
    ▼ (Model 3: OverlappingLengthMoEDecoder)
  Orthographic Text Characters

Metrics computed:
  • Acoustic Phoneme Error Rate (PER % - Greedy & Calibrated)
  • Space / Boundary Delimiter Accuracy (%)
  • End-to-End Word Error Rate (WER %)
  • End-to-End Character Error Rate (CER %)
  • Per-Language Breakdown (EN, FR, ES, IT)
"""

import argparse
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import editdistance
import soundfile as sf
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.phoneme_tokenizer import PhonemeTokenizer
from src.data.roman_tokenizer import RomanCharTokenizer
from src.models.phono_v8_acoustic import PhonoV8AcousticBackbone
from src.models.phono_v7_5_speech_model import PhonoV75SpeechConfig, PhonoV75SpeechModel
from src.models.text_phoneme_encoder import FastTextPhonemeWordEncoder
from src.models.phono_v7_3_speech_model import decode_calibrated_phonemes, decode_top_k_calibrated_phonemes
from src.inference.character_beam_search import WordTrie, beam_search_character_rollout
from src.data.phonotactic_segmenter import PhonotacticLexiconSegmenter
from src.training.metrics import compute_cer, compute_wer


def normalize_text(text: str) -> str:
    """Normalizes text by lowercasing and standardizing whitespace and punctuation."""
    t = text.lower().strip()
    t = re.sub(r"[^\w\s\'\-àâäéèêëîïôöùûüçñ]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def decode_ctc_phoneme_ids(logits: torch.Tensor, blank_id: int = 1, pad_id: int = 0) -> List[int]:
    """Greedy CTC collapse."""
    preds = logits.argmax(dim=-1).detach().cpu().tolist()
    collapsed = []
    prev = None
    for p in preds:
        if p != prev:
            if p != blank_id and p != pad_id:
                collapsed.append(p)
            prev = p
    return collapsed


def compute_spacing_errors(gt_tokens: List[int], pred_tokens: List[int], space_token: int = 8) -> Dict[str, Any]:
    matcher = difflib.SequenceMatcher(None, gt_tokens, pred_tokens)
    missed_spaces = 0
    extra_spaces = 0
    correct_spaces = 0
    total_gt_spaces = sum(1 for t in gt_tokens if t == space_token)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for t in gt_tokens[i1:i2]:
                if t == space_token:
                    correct_spaces += 1
        elif tag == "delete":
            for t in gt_tokens[i1:i2]:
                if t == space_token:
                    missed_spaces += 1
        elif tag == "insert":
            for t in pred_tokens[j1:j2]:
                if t == space_token:
                    extra_spaces += 1

    accuracy = (correct_spaces / max(1, total_gt_spaces)) * 100.0 if total_gt_spaces > 0 else 100.0
    return {
        "correct": correct_spaces,
        "missed": missed_spaces,
        "extra": extra_spaces,
        "total_gt": total_gt_spaces,
        "accuracy": accuracy,
    }


class EndToEndPhonoV8Pipeline:
    def __init__(
        self,
        acoustic_ckpt_path: Path,
        p2g_ckpt_path: Path,
        device: torch.device,
        trie: Optional[WordTrie] = None,
        segmenter: Optional[PhonotacticLexiconSegmenter] = None,
        beam_width: int = 1,
        space_bonus: float = 0.0,
        use_top_k_phonemes: bool = False,
        top_k: int = 3,
    ):
        self.device = device
        self.ph_tok = PhonemeTokenizer()
        self.rom_tok = RomanCharTokenizer()
        self.trie = trie
        self.segmenter = segmenter
        self.beam_width = beam_width
        self.space_bonus = space_bonus
        self.use_top_k_phonemes = use_top_k_phonemes
        self.top_k = top_k

        # 1. Acoustic Model (Phono-V8.0)
        print(f"🔄 Loading Phono-V8.0 Acoustic Model from: {acoustic_ckpt_path}...", flush=True)
        self.acoustic_model = PhonoV8AcousticBackbone().to(device)
        ac_ckpt = torch.load(acoustic_ckpt_path, map_location=device, weights_only=False)
        sd_ac = ac_ckpt.get("acoustic_state_dict", ac_ckpt.get("model_state_dict", ac_ckpt))
        self.acoustic_model.load_state_dict(sd_ac, strict=False)
        self.acoustic_model.eval()
        self.acoustic_step = ac_ckpt.get("step", "N/A")
        print(f"  • Acoustic Model loaded successfully (Step: {self.acoustic_step})", flush=True)

        # 2. P2G Decoders (Model 2: FastTextPhonemeWordEncoder, Model 3: Overlapping MoE Decoder)
        print(f"🔄 Loading Decoders 2 & 3 from: {p2g_ckpt_path}...", flush=True)
        self.cfg = PhonoV75SpeechConfig.medium()
        self.encoder_m2 = FastTextPhonemeWordEncoder(
            d_macro=self.cfg.decoder_config.macro_dim,
            d_model=256,
            nhead=4,
            num_layers=2,
            phoneme_vocab_size=128,
        ).to(device)
        self.model_m3 = PhonoV75SpeechModel(self.cfg).to(device)

        p2g_ckpt = torch.load(p2g_ckpt_path, map_location=device, weights_only=False)
        self.encoder_m2.load_state_dict(p2g_ckpt["phoneme_encoder_state_dict"])
        self.model_m3.decoder.load_state_dict(p2g_ckpt["decoder_state_dict"])
        self.encoder_m2.eval()
        self.model_m3.eval()
        self.p2g_step = p2g_ckpt.get("step", "N/A")
        print(f"  • Decoders 2 & 3 loaded successfully (Step: {self.p2g_step})", flush=True)

    @torch.no_grad()
    def transcribe(
        self,
        audio_path: str,
        blank_penalty: float = 0.4,
    ) -> Dict[str, Any]:
        """Runs end-to-end transcription from audio file to text."""
        audio, sr = sf.read(audio_path)
        if sr != 16000:
            import torchaudio.transforms as T
            resampler = T.Resample(sr, 16000)
            audio_t = resampler(torch.tensor(audio, dtype=torch.float32).unsqueeze(0)).to(self.device)
        else:
            audio_t = torch.tensor(audio, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Stage 1: Model 1 Acoustic Forward
        if audio_t.shape[1] > 160000:  # > 10s audio: chunk forward
            chunk_out = self.acoustic_model.forward_sliding_chunks(audio_t, chunk_seconds=8.0, overlap_seconds=2.0)
            logits = chunk_out["logits"]
        else:
            out = self.acoustic_model(audio_t)
            logits = out["logits"]

        greedy_ids = decode_ctc_phoneme_ids(logits[0])
        if self.use_top_k_phonemes:
            cal_ids, emissions = decode_top_k_calibrated_phonemes(
                logits[0],
                blank_penalty=blank_penalty,
                space_bonus=self.space_bonus,
                top_k=self.top_k,
                min_tokens_between_spaces=1,
            )
        else:
            cal_ids = decode_calibrated_phonemes(
                logits[0],
                blank_penalty=blank_penalty,
                space_bonus=self.space_bonus,
                min_tokens_between_spaces=1,
            )
            emissions = None

        pred_ph_str = "".join([self.ph_tok.id_to_token.get(i, "") for i in cal_ids])

        # Stage 2 & 3: Word Slicing and Model 2 Word Encoding
        if self.use_top_k_phonemes and emissions is not None:
            word_emissions = []
            cur_w = []
            for ids, probs in emissions:
                if ids[0] == 8:
                    if cur_w:
                        word_emissions.append(cur_w)
                        cur_w = []
                else:
                    cur_w.append((ids, probs))
            if cur_w:
                word_emissions.append(cur_w)

            # Apply Phonotactic Viterbi Segmenter to split fused clitics & words
            if self.segmenter is not None:
                split_emissions = []
                for w in word_emissions:
                    w_splits = self.segmenter.segment_word_emissions(w, self.ph_tok.id_to_token)
                    split_emissions.extend(w_splits)
                word_emissions = split_emissions

            L = len(word_emissions)
            if L == 0:
                return {
                    "pred_text": "",
                    "pred_phones_str": pred_ph_str,
                    "greedy_ids": greedy_ids,
                    "cal_ids": cal_ids,
                }

            max_k = min(24, max(1, max(len(w) for w in word_emissions)))
            top_ids = torch.zeros(L, max_k, self.top_k, dtype=torch.long, device=self.device)
            top_probs = torch.zeros(L, max_k, self.top_k, dtype=torch.float32, device=self.device)
            for l, w in enumerate(word_emissions):
                for k, (c_ids, c_probs) in enumerate(w[:max_k]):
                    top_ids[l, k] = torch.tensor(c_ids, dtype=torch.long, device=self.device)
                    top_probs[l, k] = torch.tensor(c_probs, dtype=torch.float32, device=self.device)
            z_words = self.encoder_m2(top_k_ids=top_ids, top_k_probs=top_probs).view(1, L, -1)
        else:
            words_ph = [w.strip() for w in pred_ph_str.split(" ") if w.strip()]
            word_chunks = [list(w) for w in words_ph]
            if self.segmenter is not None:
                split_chunks = []
                for chs in word_chunks:
                    c_splits = self.segmenter.segment_phoneme_tokens(chs)
                    split_chunks.extend(c_splits)
                word_chunks = split_chunks
            L = len(word_chunks)
            if L == 0:
                return {
                    "pred_text": "",
                    "pred_phones_str": pred_ph_str,
                    "greedy_ids": greedy_ids,
                    "cal_ids": cal_ids,
                }
            word_ids = [self.ph_tok.encode(ch)[:24] for ch in word_chunks]
            max_k = max(1, max(len(w) for w in word_ids))
            input_ph = torch.zeros(1, L, max_k, dtype=torch.long, device=self.device)
            for i, ids in enumerate(word_ids):
                if ids:
                    input_ph[0, i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
            flat_ph = input_ph.reshape(L, max_k)
            z_words = self.encoder_m2(phoneme_ids=flat_ph).view(1, L, -1)

        # Stage 4: Model 3 (Overlapping MoE Character Decoder Rollout)
        pred_words = beam_search_character_rollout(
            model_m3=self.model_m3,
            z_words=z_words,
            rom_tok=self.rom_tok,
            trie=self.trie,
            beam_width=self.beam_width,
            device=self.device,
        )

        pred_text = self.rom_tok.decode_words(pred_words)
        return {
            "pred_text": pred_text,
            "pred_phones_str": pred_ph_str,
            "greedy_ids": greedy_ids,
            "cal_ids": cal_ids,
        }


def main():
    parser = argparse.ArgumentParser(description="Phono-V8.0 Comprehensive End-to-End Multilingual Evaluation")
    parser.add_argument("--val_manifest", type=str, default="data/clean_multilingual_val.json")
    parser.add_argument("--acoustic_ckpt", type=str, default="checkpoints/phono_v8_acoustic/best_v8_acoustic.pt")
    parser.add_argument("--p2g_ckpt", type=str, default="checkpoints/phono_v8_robust_p2g/best_robust_p2g.pt")
    parser.add_argument("--output_json", type=str, default="results/phono_v8_e2e_beam.json")
    parser.add_argument("--max_samples", type=int, default=0, help="0 for full validation set")
    parser.add_argument("--blank_penalty", type=float, default=0.4)
    parser.add_argument("--space_bonus", type=float, default=0.35, help="Space calibration bonus for Model 1")
    parser.add_argument("--beam_width", type=int, default=4, help="Model 3 character beam search width (1 for greedy)")
    parser.add_argument("--vocab_file", type=str, default="data/librispeech_vocab_full.json", help="Path to vocabulary JSON for Trie")
    parser.add_argument("--no_trie", action="store_true", default=False, help="Disable Trie constraint in beam search")
    parser.add_argument("--use_top_k_phonemes", action="store_true", default=False, help="Use Top-K soft phoneme mixture input to Model 2")
    parser.add_argument("--phoneme_top_k", type=int, default=3, help="Number of phoneme candidates for soft mixture")
    parser.add_argument("--use_clitic_segmenter", action="store_true", default=False, help="Use Phonotactic Viterbi Segmenter to split fused clitics")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("=" * 80)
    print("🏆 PHONO-V8.0 COMPREHENSIVE END-TO-END MULTILINGUAL EVALUATION")
    print("=" * 80)
    print(f"🎙️  Acoustic Checkpoint : {args.acoustic_ckpt}")
    print(f"📦 Decoders 2 & 3      : {args.p2g_ckpt}")
    print(f"📁 Validation Manifest : {args.val_manifest}")
    print(f"⚡ Device              : {device.type.upper()}")
    print(f"🎯 Beam Width          : {args.beam_width} ({'Greedy' if args.beam_width == 1 else 'Trie Beam Search'})")
    print(f"🌌 Space Bonus         : {args.space_bonus}")
    print(f"🔬 Top-K Phoneme Input : {args.use_top_k_phonemes} (Top-{args.phoneme_top_k} soft mixture)")
    print(f"✂️  Clitic Segmenter   : {args.use_clitic_segmenter}")

    trie = None
    if not args.no_trie and os.path.isfile(args.vocab_file):
        print(f"📖 Loading Vocabulary Trie from {args.vocab_file}...", flush=True)
        with open(args.vocab_file, "r", encoding="utf-8") as f:
            vocab_words = json.load(f)
        rom_tok = RomanCharTokenizer()
        trie = WordTrie.from_word_list(vocab_words, rom_tok)
        print(f"  • Vocabulary Trie loaded: {trie.word_count} words", flush=True)
    elif not args.no_trie:
        print(f"⚠️ Vocab file {args.vocab_file} not found; running unconstrained beam search", flush=True)

    segmenter = None
    if args.use_clitic_segmenter:
        print("⚡ Initializing Phonotactic Viterbi Lexicon Segmenter...", flush=True)
        segmenter = PhonotacticLexiconSegmenter()
        print(f"  • Phonotactic Segmenter active ({len(segmenter.lexicon)} phonetic words loaded)", flush=True)

    pipeline = EndToEndPhonoV8Pipeline(
        acoustic_ckpt_path=Path(args.acoustic_ckpt),
        p2g_ckpt_path=Path(args.p2g_ckpt),
        device=device,
        trie=trie,
        segmenter=segmenter,
        beam_width=args.beam_width,
        space_bonus=args.space_bonus,
        use_top_k_phonemes=args.use_top_k_phonemes,
        top_k=args.phoneme_top_k,
    )

    with open(args.val_manifest, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    if args.max_samples > 0:
        val_data = val_data[: args.max_samples]

    print(f"\n🚀 Running evaluation on {len(val_data)} validation utterances...\n", flush=True)

    from piper.voice import EspeakPhonemizer
    ep = EspeakPhonemizer()
    ph_tok = PhonemeTokenizer()
    diacritics = {"ˈ", "ˌ", "ː", "ˑ"}
    diacritic_ids = {ph_tok.token_to_id[ch] for ch in diacritics if ch in ph_tok.token_to_id}

    def get_gt_phoneme_ids(text: str, lang: str) -> List[int]:
        lang_map = {"en": "en-us", "fr": "fr", "it": "it", "es": "es"}
        target_voice = lang_map.get(lang, "en-us")
        try:
            chunks = ep.phonemize(target_voice, text)
            flat = []
            for c in chunks:
                flat.extend(c)
                flat.append(" ")
            if flat and flat[-1] == " ":
                flat.pop()
            token_ids = ph_tok.encode(flat)
            return [t for t in token_ids if t not in diacritic_ids]
        except Exception:
            return []

    # Accumulators
    total_samples = 0
    total_audio_sec = 0.0
    total_proc_sec = 0.0

    all_pred_texts: List[str] = []
    all_ref_texts: List[str] = []

    total_ed_greedy = 0
    total_ed_calib = 0
    total_ref_ph_len = 0

    total_spc_correct = 0
    total_spc_gt = 0

    per_lang_stats: Dict[str, Dict[str, Any]] = {}
    samples_by_lang: Dict[str, List[Dict[str, Any]]] = {}

    start_time = time.time()

    for idx, item in enumerate(val_data):
        audio_path = item["audio_path"]
        ref_raw = item["transcript"]
        lang = item.get("lang", "en")
        ref_norm = normalize_text(ref_raw)

        if not os.path.isfile(audio_path):
            continue

        gt_ph_ids = get_gt_phoneme_ids(ref_norm, lang)
        if not gt_ph_ids:
            continue

        t0 = time.time()
        try:
            res = pipeline.transcribe(audio_path, blank_penalty=args.blank_penalty)
        except Exception as e:
            print(f"⚠️ Error on {audio_path}: {e}", file=sys.stderr)
            continue
        proc_time = time.time() - t0

        audio_info = sf.info(audio_path)
        audio_dur = audio_info.duration

        total_audio_sec += audio_dur
        total_proc_sec += proc_time
        total_samples += 1

        pred_text = normalize_text(res["pred_text"])
        all_pred_texts.append(pred_text)
        all_ref_texts.append(ref_norm)

        # Phoneme metrics
        greedy_ids = res["greedy_ids"]
        cal_ids = res["cal_ids"]
        ed_g = editdistance.eval(greedy_ids, gt_ph_ids)
        ed_c = editdistance.eval(cal_ids, gt_ph_ids)
        r_len = max(1, len(gt_ph_ids))

        spc_res = compute_spacing_errors(gt_ph_ids, cal_ids, space_token=8)

        total_ed_greedy += ed_g
        total_ed_calib += ed_c
        total_ref_ph_len += r_len
        total_spc_correct += spc_res["correct"]
        total_spc_gt += spc_res["total_gt"]

        # Per language accumulation
        if lang not in per_lang_stats:
            per_lang_stats[lang] = {
                "samples": 0,
                "audio_sec": 0.0,
                "proc_sec": 0.0,
                "pred_texts": [],
                "ref_texts": [],
                "ed_greedy": 0,
                "ed_calib": 0,
                "ref_ph_len": 0,
                "spc_correct": 0,
                "spc_gt": 0,
            }
        p_stat = per_lang_stats[lang]
        p_stat["samples"] += 1
        p_stat["audio_sec"] += audio_dur
        p_stat["proc_sec"] += proc_time
        p_stat["pred_texts"].append(pred_text)
        p_stat["ref_texts"].append(ref_norm)
        p_stat["ed_greedy"] += ed_g
        p_stat["ed_calib"] += ed_c
        p_stat["ref_ph_len"] += r_len
        p_stat["spc_correct"] += spc_res["correct"]
        p_stat["spc_gt"] += spc_res["total_gt"]

        if lang not in samples_by_lang:
            samples_by_lang[lang] = []
        if len(samples_by_lang[lang]) < 3:
            samples_by_lang[lang].append({
                "ref_text": ref_raw,
                "pred_text": pred_text,
                "gt_phones": ph_tok.decode(gt_ph_ids, skip_special=True),
                "pred_phones": res["pred_phones_str"],
                "wer": compute_wer([pred_text], [ref_norm]) * 100.0,
                "cer": compute_cer([pred_text], [ref_norm]) * 100.0,
                "per": (ed_c / r_len) * 100.0,
            })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(val_data):
            cur_wer = compute_wer(all_pred_texts, all_ref_texts) * 100.0
            cur_cer = compute_cer(all_pred_texts, all_ref_texts) * 100.0
            cur_per = (total_ed_calib / max(1, total_ref_ph_len)) * 100.0
            cur_spc = (total_spc_correct / max(1, total_spc_gt)) * 100.0
            elapsed = time.time() - start_time
            rate = (idx + 1) / max(1e-3, elapsed)
            print(
                f"[{idx + 1:4d}/{len(val_data)}] | "
                f"Cal PER: {cur_per:5.2f}% | Spc Acc: {cur_spc:5.2f}% | "
                f"WER: {cur_wer:5.2f}% | CER: {cur_cer:5.2f}% | "
                f"Speed: {rate:4.1f} utt/s",
                flush=True,
            )

    # Compute Global Metrics
    global_wer = compute_wer(all_pred_texts, all_ref_texts) * 100.0
    global_cer = compute_cer(all_pred_texts, all_ref_texts) * 100.0
    global_per_greedy = (total_ed_greedy / max(1, total_ref_ph_len)) * 100.0
    global_per_calib = (total_ed_calib / max(1, total_ref_ph_len)) * 100.0
    global_spc_acc = (total_spc_correct / max(1, total_spc_gt)) * 100.0 if total_spc_gt > 0 else 100.0
    global_rtf = total_proc_sec / max(1e-4, total_audio_sec)

    flag_map = {"en": "🇬🇧 EN", "fr": "🇫🇷 FR", "es": "🇪🇸 ES", "it": "🇮🇹 IT"}

    print("\n" + "=" * 80)
    print("           🌟 FINAL PHONO-V8.0 END-TO-END BENCHMARK RESULTS")
    print("=" * 80)
    print(f"📊 Global Overview ({total_samples} utterances, {total_audio_sec/3600:.2f} hours of audio):")
    print(f"   • Calibrated Phoneme Error Rate (PER) : \033[1;32m{global_per_calib:5.2f}%\033[0m")
    print(f"   • Greedy Phoneme Error Rate (PER)     : {global_per_greedy:5.2f}%")
    print(f"   • Word Boundary <space> Accuracy      : \033[1;36m{global_spc_acc:5.2f}%\033[0m")
    print(f"   • End-to-End Word Error Rate (WER)    : \033[1;31m{global_wer:5.2f}%\033[0m")
    print(f"   • End-to-End Character Error Rate(CER): {global_cer:5.2f}%")
    print(f"   • Real-Time Factor (RTF)              : {global_rtf:5.3f}x ({total_audio_sec/total_proc_sec:.1f}x real-time speed)")
    print("─" * 80)

    print("🌍 Per-Language Breakdown:")
    lang_summary = {}
    for lang, s in sorted(per_lang_stats.items()):
        l_flag = flag_map.get(lang, lang.upper())
        l_wer = compute_wer(s["pred_texts"], s["ref_texts"]) * 100.0
        l_cer = compute_cer(s["pred_texts"], s["ref_texts"]) * 100.0
        l_per = (s["ed_calib"] / max(1, s["ref_ph_len"])) * 100.0
        l_spc = (s["spc_correct"] / max(1, s["spc_gt"])) * 100.0 if s["spc_gt"] > 0 else 100.0
        l_rtf = s["proc_sec"] / max(1e-4, s["audio_sec"])

        lang_summary[lang] = {
            "samples": s["samples"],
            "wer": l_wer,
            "cer": l_cer,
            "per": l_per,
            "space_acc": l_spc,
            "rtf": l_rtf,
        }
        print(
            f"   {l_flag}: PER: {l_per:5.2f}% | Spc Acc: {l_spc:5.2f}% | "
            f"WER: {l_wer:5.2f}% | CER: {l_cer:5.2f}% | "
            f"Samples: {s['samples']:3d}"
        )

    print("─" * 80)
    print("🔍 Representative End-to-End Transcriptions:")
    for lang, samples in sorted(samples_by_lang.items()):
        l_flag = flag_map.get(lang, lang.upper())
        for idx, s in enumerate(samples[:2]):
            print(f"   [{l_flag} Sample #{idx+1}] PER: {s['per']:4.1f}% | WER: {s['wer']:4.1f}% | CER: {s['cer']:4.1f}%")
            print(f"     • GT Text  : {s['ref_text']}")
            print(f"     • Pred Text: {s['pred_text']}")
            print(f"     • Pred Ph  : {s['pred_phones'][:120]}")
    print("=" * 80 + "\n")

    # Save to JSON
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "acoustic_checkpoint": str(args.acoustic_ckpt),
            "p2g_checkpoint": str(args.p2g_ckpt),
            "total_samples": total_samples,
            "total_audio_hours": total_audio_sec / 3600.0,
            "global_per_calib": global_per_calib,
            "global_per_greedy": global_per_greedy,
            "global_spc_acc": global_spc_acc,
            "global_wer": global_wer,
            "global_cer": global_cer,
            "global_rtf": global_rtf,
            "languages": lang_summary,
            "samples": samples_by_lang,
        }, f, indent=2)
    print(f"💾 Full evaluation results saved to: {out_path}")


if __name__ == "__main__":
    main()
