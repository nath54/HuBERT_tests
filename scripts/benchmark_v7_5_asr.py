#!/usr/bin/env python3
"""Phono-V7.5 Comprehensive Monolingual & Multilingual ASR Benchmark Suite.

Evaluates Phono-V7.5 across standard Automatic Speech Recognition (ASR) benchmarks
WITHOUT speech-to-text translation (audio in language X -> text in language X).

Included Benchmarks:
1. LibriSpeech test-clean (English standard clean read speech)
2. LibriSpeech test-other (English challenging noisy / accented speech)
3. Google FLEURS en_us (English conversational & read speech)
4. Google FLEURS fr_fr (French speech)
5. Google FLEURS es_419 (Spanish speech)
6. Google FLEURS it_it (Italian speech)
7. Multilingual LibriSpeech (MLS) French test split
8. Multilingual LibriSpeech (MLS) Spanish test split
9. Multilingual LibriSpeech (MLS) Italian test split

Metrics computed:
  • Word Error Rate (WER %)
  • Character Error Rate (CER %)
  • Real-Time Factor (RTF)
"""

import argparse
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import soundfile as sf
import torch
import torch.nn.functional as F
from piper.voice import EspeakPhonemizer

from src.data.phoneme_tokenizer import PhonemeTokenizer
from src.data.roman_tokenizer import RomanCharTokenizer
from src.models.phono_v7_3_speech_model import PhonoV73SpeechModel
from src.models.phono_v7_5_speech_model import PhonoV75SpeechConfig, PhonoV75SpeechModel
from src.models.text_phoneme_encoder import FastTextPhonemeWordEncoder
from src.training.metrics import compute_cer, compute_per, compute_wer




def normalize_text(text: str) -> str:
    """Normalizes text by lowercasing and removing extraneous punctuation."""
    t = text.lower().strip()
    t = re.sub(r"[^\w\s\'\-àâäéèêëîïôöùûüçñ]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def safe_torch_load(path: Path, map_location: torch.device, max_retries: int = 5) -> Dict[str, Any]:
    last_err = None
    for attempt in range(max_retries):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(0.5)
    raise RuntimeError(f"Failed to load checkpoint from {path}: {last_err}")


def format_interval(seconds: float) -> str:
    """Formats a duration in seconds into human-readable HHhMMmSSs or MMmSSs."""
    seconds = max(0, int(seconds))
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    elif m > 0:
        return f"{m:02d}m{s:02d}s"
    else:
        return f"{s:02d}s"


# --- Dataset Iterators & Total Count Providers ---

def get_librispeech_data(manifest_path: str, max_samples: Optional[int] = None) -> Tuple[Iterator[Dict[str, Any]], Optional[int]]:
    p = Path(manifest_path)
    if not p.is_file():
        print(f"⚠️ LibriSpeech manifest not found at {manifest_path}", file=sys.stderr)
        return iter([]), 0
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    total = len(data)
    if max_samples and max_samples < total:
        total = max_samples

    def _generator():
        count = 0
        for item in data:
            audio_path = item.get("audio_path", item.get("wav", ""))
            ref_text = item.get("text", item.get("transcript", ""))
            if not audio_path or not ref_text or not Path(audio_path).is_file():
                continue
            try:
                audio, sr = sf.read(audio_path)
                yield {
                    "audio": audio,
                    "sr": sr,
                    "ref_text": normalize_text(ref_text),
                    "lang": "en",
                }
                count += 1
                if max_samples and count >= max_samples:
                    break
            except Exception:
                continue

    return _generator(), total


def get_mls_data(transcripts_path: str, audio_root: str, lang: str, max_samples: Optional[int] = None) -> Tuple[Iterator[Dict[str, Any]], Optional[int]]:
    t_file = Path(transcripts_path)
    a_root = Path(audio_root)
    if not t_file.is_file():
        print(f"⚠️ MLS transcripts not found at {transcripts_path}", file=sys.stderr)
        return iter([]), 0

    with open(t_file, "r", encoding="utf-8") as f:
        lines = [line.strip().split("\t", 1) for line in f if len(line.strip().split("\t", 1)) == 2]

    total = len(lines)
    if max_samples and max_samples < total:
        total = max_samples

    def _generator():
        count = 0
        for audio_id, ref_text in lines:
            id_parts = audio_id.split("_")
            if len(id_parts) >= 2:
                spk, book = id_parts[0], id_parts[1]
                wav_path = a_root / spk / book / f"{audio_id}.flac"
            else:
                wav_path = a_root / f"{audio_id}.flac"

            if not wav_path.is_file():
                continue
            try:
                audio, sr = sf.read(str(wav_path))
                yield {
                    "audio": audio,
                    "sr": sr,
                    "ref_text": normalize_text(ref_text),
                    "lang": lang,
                }
                count += 1
                if max_samples and count >= max_samples:
                    break
            except Exception:
                continue

    return _generator(), total


FLEURS_TEST_COUNTS = {
    "en_us": 647,
    "fr_fr": 676,
    "es_419": 908,
    "it_it": 865,
}

def get_fleurs_data(lang_code: str, target_lang: str, max_samples: Optional[int] = None) -> Tuple[Iterator[Dict[str, Any]], Optional[int]]:
    total = FLEURS_TEST_COUNTS.get(lang_code, None)
    if total is not None and max_samples and max_samples < total:
        total = max_samples

    def _generator():
        try:
            from datasets import load_dataset, Audio
            ds = load_dataset("google/fleurs", lang_code, split="test", streaming=True)
            ds = ds.cast_column("audio", Audio(decode=False))
            count = 0
            for item in ds:
                ref_text = item.get("transcription", "")
                raw_bytes = item["audio"]["bytes"]
                audio, sr = sf.read(io.BytesIO(raw_bytes))
                yield {
                    "audio": audio,
                    "sr": sr,
                    "ref_text": normalize_text(ref_text),
                    "lang": target_lang,
                }
                count += 1
                if max_samples and count >= max_samples:
                    break
        except Exception as e:
            print(f"⚠️ Error streaming FLEURS {lang_code}: {e}", file=sys.stderr)

    return _generator(), total


# --- Phoneme-to-Text Pipeline Decoder ---

# --- Complete 3-Stage End-to-End & P2G-Only Pipeline ---

class FullPhonoV75Pipeline:
    def __init__(
        self,
        acoustic_ckpt_path: Path,
        p2g_ckpt_path: Path,
        device: torch.device,
        mode: str = "end_to_end",
    ):
        self.device = device
        self.mode = mode
        self.phonemizer = EspeakPhonemizer()
        self.ph_tok = PhonemeTokenizer()
        self.rom_tok = RomanCharTokenizer()

        # 1. Acoustic Model (Conformer V7.3)
        self.acoustic_model = None
        self.acoustic_step = "N/A"
        if mode == "end_to_end":
            self.acoustic_model = PhonoV73SpeechModel().to(device)
            ac_ckpt = safe_torch_load(acoustic_ckpt_path, map_location=device)
            sd = ac_ckpt.get("model_state_dict", ac_ckpt)
            self.acoustic_model.load_state_dict(sd, strict=False)
            self.acoustic_model.eval()
            self.acoustic_step = ac_ckpt.get("step", "N/A")

        # 2. P2G Decoder (V7.5)
        self.cfg = PhonoV75SpeechConfig.medium()
        self.encoder = FastTextPhonemeWordEncoder(
            d_macro=self.cfg.decoder_config.macro_dim,
            d_model=256,
            nhead=4,
            num_layers=2,
            phoneme_vocab_size=128,
        ).to(device)
        self.p2g_model = PhonoV75SpeechModel(self.cfg).to(device)

        p2g_ckpt = safe_torch_load(p2g_ckpt_path, map_location=device)
        self.encoder.load_state_dict(p2g_ckpt["phoneme_encoder_state_dict"])
        self.p2g_model.decoder.load_state_dict(p2g_ckpt["decoder_state_dict"])
        self.encoder.eval()
        self.p2g_model.eval()

        self.p2g_step = p2g_ckpt.get("step", "N/A")
        self.best_p2g_wer = p2g_ckpt.get("val_wer", None)

    @torch.no_grad()
    def transcribe(
        self,
        audio: Any,
        sr: int,
        ref_text: str,
        lang_voice: str,
    ) -> Tuple[str, List[str], List[str]]:
        """Transcribes an audio sample either end-to-end or via P2G only.
        
        Returns:
            (pred_text, pred_phonemes_tokens, ref_phonemes_tokens)
        """
        # Ground truth phonemes with space boundary
        ref_words = ref_text.split()
        ref_chunks = [list(cached_phonemize_word(lang_voice, w)) for w in ref_words]
        ref_ph_tokens = []
        for c in ref_chunks:
            ref_ph_tokens.extend(c)
            ref_ph_tokens.append(" ")
        if ref_ph_tokens and ref_ph_tokens[-1] == " ":
            ref_ph_tokens.pop()

        if self.mode == "p2g_only":
            pred_text = self.decode_phoneme_chunks(ref_chunks)
            pred_words = pred_text.split()
            pred_chunks = [list(cached_phonemize_word(lang_voice, w)) for w in pred_words]
            pred_ph_tokens = []
            for c in pred_chunks:
                pred_ph_tokens.extend(c)
                pred_ph_tokens.append(" ")
            if pred_ph_tokens and pred_ph_tokens[-1] == " ":
                pred_ph_tokens.pop()
            return pred_text, pred_ph_tokens, ref_ph_tokens

        # Mode: end_to_end (Audio -> Conformer -> Phonemes -> P2G -> Text)
        if sr != 16000:
            import torchaudio.transforms as T
            resampler = T.Resample(sr, 16000)
            audio_t = resampler(torch.tensor(audio, dtype=torch.float32).unsqueeze(0)).to(self.device)
        else:
            audio_t = torch.tensor(audio, dtype=torch.float32, device=self.device).unsqueeze(0)

        enc_out = self.acoustic_model.encoder(audio=audio_t)
        logits = enc_out.get("logits")  # [1, T, 64]
        preds = logits.argmax(dim=-1)[0].tolist()

        # CTC collapse
        decoded_ph_ids = []
        prev = None
        for p in preds:
            if p != prev:
                if p not in (0, 1):
                    decoded_ph_ids.append(p)
                prev = p

        pred_ph_tokens = [self.ph_tok.id_to_token.get(i, "") for i in decoded_ph_ids]
        pred_ph_str = "".join(pred_ph_tokens)

        # Split into word chunks by space
        words_ph = [w.strip() for w in pred_ph_str.split(" ") if w.strip()]
        word_chunks = [list(w) for w in words_ph]

        pred_text = self.decode_phoneme_chunks(word_chunks)
        return pred_text, pred_ph_tokens, ref_ph_tokens

    def decode_phoneme_chunks(self, phoneme_chunks: List[List[str]]) -> str:
        """Decodes word-level IPA chunks into orthographic characters."""
        L = len(phoneme_chunks)
        if L == 0:
            return ""

        word_ids = [self.ph_tok.encode(ch)[:24] for ch in phoneme_chunks]
        max_k = max(1, max(len(w) for w in word_ids))

        input_ph = torch.zeros(1, L, max_k, dtype=torch.long, device=self.device)
        for i, ids in enumerate(word_ids):
            if ids:
                input_ph[0, i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)

        with torch.no_grad():
            flat_ph = input_ph.reshape(L, max_k)
            z_words = self.encoder(phoneme_ids=flat_ph).view(1, L, -1)

            expert_logits = self.p2g_model.decoder.macro_expert_router(z_words)
            top1_experts = expert_logits.argmax(dim=-1)
            expert_cond = self.p2g_model.decoder.expert_embeddings(top1_experts)
            z_windows = self.p2g_model.decoder.build_word_context_windows(z_words)

            pred_words = []
            for l in range(L):
                zw_window = z_windows[:, l, :, :]
                p_bias = expert_cond[:, l : l + 1, :]
                p_id = top1_experts[:, l].clamp(0, self.cfg.decoder_config.num_paths - 1)

                cur_tokens = [self.rom_tok.bos_id]
                for _ in range(24):
                    inp = torch.tensor([cur_tokens], dtype=torch.long, device=self.device)
                    logits, _ = self.p2g_model.decoder.micro_head(
                        inp,
                        zw_window,
                        acoustic_slices=None,
                        path_bias=p_bias,
                        path_id=p_id,
                    )
                    next_tok = logits[0, -1, :].argmax(dim=-1).item()
                    if next_tok in (self.rom_tok.eos_id, self.rom_tok.pad_id):
                        break
                    cur_tokens.append(next_tok)

                pred_words.append(cur_tokens[1:])

            return self.rom_tok.decode_words(pred_words)


# --- Benchmark Evaluation Runner ---

from functools import lru_cache

_global_phonemizer = None

def get_global_phonemizer():
    global _global_phonemizer
    if _global_phonemizer is None:
        _global_phonemizer = EspeakPhonemizer()
    return _global_phonemizer

@lru_cache(maxsize=100000)
def cached_phonemize_word(voice: str, word: str) -> Tuple[str, ...]:
    ep = get_global_phonemizer()
    try:
        ch = ep.phonemize(voice, word)
        return tuple(c for sub in ch for c in sub)
    except Exception:
        return tuple()


def run_benchmark(
    name: str,
    lang: str,
    iterator: Iterator[Dict[str, Any]],
    pipeline: FullPhonoV75Pipeline,
    total_samples: Optional[int] = None,
    verbose_samples: int = 3,
    log_interval_samples: int = 25,
    log_interval_sec: float = 12.0,
) -> Dict[str, Any]:
    print(f"\n{'='*80}")
    total_hdr = f"({total_samples} samples)" if total_samples else ""
    mode_hdr = "End-to-End (Audio -> Phonemes -> Text)" if pipeline.mode == "end_to_end" else "Phoneme-to-Text Only"
    print(f"📊 Running Benchmark: \033[1;36m{name}\033[0m ({lang.upper()}) {total_hdr}")
    print(f"   Mode: \033[1;35m{mode_hdr}\033[0m")
    print(f"{'='*80}", flush=True)

    all_preds = []
    all_refs = []
    all_pred_phs = []
    all_ref_phs = []
    total_audio_sec = 0.0
    total_proc_sec = 0.0
    samples_shown = 0

    bench_start_time = time.time()
    last_log_time = bench_start_time

    lang_voice = {"en": "en-us", "fr": "fr", "es": "es", "it": "it"}.get(lang, "en-us")

    for idx, sample in enumerate(iterator, 1):
        audio = sample["audio"]
        sr = sample["sr"]
        ref_text = sample["ref_text"]
        if not ref_text.strip():
            continue

        audio_len_sec = len(audio) / max(1, sr)
        total_audio_sec += audio_len_sec

        t0 = time.time()
        pred_text, pred_ph_tokens, ref_ph_tokens = pipeline.transcribe(
            audio=audio,
            sr=sr,
            ref_text=ref_text,
            lang_voice=lang_voice,
        )
        proc_time = time.time() - t0
        total_proc_sec += proc_time

        all_preds.append(pred_text)
        all_refs.append(ref_text)
        all_pred_phs.append(pred_ph_tokens)
        all_ref_phs.append(ref_ph_tokens)

        if samples_shown < verbose_samples:
            sample_wer = compute_wer([pred_text], [ref_text]) * 100.0
            sample_cer = compute_cer([pred_text], [ref_text]) * 100.0
            sample_per = compute_per([pred_ph_tokens], [ref_ph_tokens]) * 100.0
            tot_label = f"/{total_samples}" if total_samples else ""
            print(f"  🔍 [Sample {samples_shown + 1}{tot_label}] | WER: {sample_wer:4.1f}% | CER: {sample_cer:4.1f}% | PER: {sample_per:4.1f}%")
            print(f"    • Ref Text : {ref_text}")
            print(f"    • Pred Text: {pred_text}")
            print(f"    • Ref Phon : {''.join(ref_ph_tokens)[:80]}")
            print(f"    • Pred Phon: {''.join(pred_ph_tokens)[:80]}", flush=True)
            samples_shown += 1

        now = time.time()
        elapsed_sec = now - bench_start_time
        time_since_last_log = now - last_log_time

        # Frequent updates: at sample 10, or every 25 samples, or every 12 seconds
        should_log = (
            (idx == 10) or
            (idx % log_interval_samples == 0) or
            (time_since_last_log >= log_interval_sec and idx > samples_shown)
        )

        if should_log:
            interim_wer = compute_wer(all_preds, all_refs) * 100.0
            interim_cer = compute_cer(all_preds, all_refs) * 100.0
            interim_per = compute_per(all_pred_phs, all_ref_phs) * 100.0
            interim_rtf = total_proc_sec / max(1e-4, total_audio_sec)
            rate = idx / max(1e-4, elapsed_sec)

            elapsed_str = format_interval(elapsed_sec)
            if total_samples and total_samples > 0:
                pct = (idx / total_samples) * 100.0
                rem_samples = max(0, total_samples - idx)
                eta_sec = rem_samples / max(1e-4, rate)
                eta_str = format_interval(eta_sec)
                progress_tag = f"[{idx:>4d}/{total_samples:>4d}] ({pct:5.1f}%)"
            else:
                progress_tag = f"[{idx:>4d} samples]"
                eta_str = "--"

            print(
                f"  ⏳ {progress_tag} | Running WER: \033[1;33m{interim_wer:5.2f}%\033[0m | "
                f"CER: \033[1;35m{interim_cer:5.2f}%\033[0m | PER: \033[1;36m{interim_per:5.2f}%\033[0m | "
                f"Rate: {rate:4.1f} s/s | RTF: {interim_rtf:5.3f}x | "
                f"Elapsed: {elapsed_str} | ETA: \033[1;32m{eta_str}\033[0m",
                flush=True,
            )
            last_log_time = now

    if not all_refs:
        print(f"  ❌ No valid samples found for {name}!", flush=True)
        return {"name": name, "lang": lang, "samples": 0, "wer": None, "cer": None, "per": None, "rtf": None}

    wer = compute_wer(all_preds, all_refs) * 100.0
    cer = compute_cer(all_preds, all_refs) * 100.0
    per = compute_per(all_pred_phs, all_ref_phs) * 100.0
    rtf = total_proc_sec / max(1e-4, total_audio_sec)
    total_time_str = format_interval(time.time() - bench_start_time)

    print(f"  ────────────────────────────────────────────────────────────────────────────")
    print(
        f"  ✅ \033[1mResults for {name}\033[0m: "
        f"\033[1;32mWER: {wer:5.2f}%\033[0m | \033[1;34mCER: {cer:5.2f}%\033[0m | \033[1;36mPER: {per:5.2f}%\033[0m | "
        f"RTF: {rtf:5.3f}x | Samples: {len(all_refs)} | Took: {total_time_str}",
        flush=True,
    )

    return {
        "name": name,
        "lang": lang,
        "samples": len(all_refs),
        "total_audio_hours": total_audio_sec / 3600.0,
        "wer": wer,
        "cer": cer,
        "per": per,
        "rtf": rtf,
    }


def save_intermediate_results(output_json: Path, p2g_ckpt: Path, ac_step: Any, p2g_step: Any, mode: str, results: List[Dict[str, Any]]):
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({
            "mode": mode,
            "p2g_checkpoint": str(p2g_ckpt),
            "acoustic_step": ac_step,
            "p2g_step": p2g_step,
            "results": results,
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Phono-V7.5 Comprehensive Monolingual/Multilingual ASR Benchmarks")
    parser.add_argument(
        "--mode",
        type=str,
        default="end_to_end",
        choices=["end_to_end", "p2g_only"],
        help="Evaluation mode: 'end_to_end' (Audio -> Conformer -> Phonemes -> P2G -> Text) or 'p2g_only' (Text -> Phonemes -> P2G -> Text)",
    )
    parser.add_argument(
        "--acoustic_checkpoint",
        type=str,
        default="checkpoints/phono_v7_3/streaming/best_checkpoint.pt",
        help="Path to trained Conformer Acoustic checkpoint (used for end_to_end mode)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/phono_v7_5_phoneme_to_text/best_p2g.pt",
        help="Path to trained Phoneme-to-Text checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to run benchmarks on ('cpu' or 'cuda')",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["librispeech_clean", "fleurs_fr", "fleurs_es"],
        choices=[
            "all",
            "librispeech_clean",
            "librispeech_other",
            "fleurs_en",
            "fleurs_fr",
            "fleurs_es",
            "fleurs_it",
            "mls_fr",
            "mls_es",
            "mls_it",
        ],
        help="List of benchmarks to run (default: 'librispeech_clean' 'fleurs_fr' 'fleurs_es')",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Max samples per benchmark (e.g. 100 for rapid audit, 0/None for full set)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/benchmarks_v7_5.json",
        help="Output path for benchmark results JSON",
    )
    args = parser.parse_args()

    max_samples = None if args.max_samples <= 0 else args.max_samples
    device = torch.device(args.device)
    out_p = Path(args.output_json)

    print("=" * 80)
    print("🏆 Phono-V7.5 Official Speech Recognition (ASR) Benchmark Suite")
    print("=" * 80)
    print(f"🎯 Evaluation Mode: \033[1;33m{args.mode.upper()}\033[0m")
    if args.mode == "end_to_end":
        print(f"🎙️  Acoustic Checkpoint : {args.acoustic_checkpoint}")
    print(f"📦 P2G Checkpoint      : {args.checkpoint}")
    print(f"⚡ Device              : {device.type.upper()}")
    print(f"🎯 Max Samples         : {max_samples or 'FULL DATASET (100% of test splits)'}")

    p2g_path = Path(args.checkpoint)
    if not p2g_path.is_file():
        alt_path = p2g_path.parent / "latest_p2g.pt"
        if alt_path.is_file():
            p2g_path = alt_path
        else:
            print(f"❌ P2G Checkpoint not found at: {args.checkpoint}", file=sys.stderr)
            sys.exit(1)

    ac_path = Path(args.acoustic_checkpoint)
    pipeline = FullPhonoV75Pipeline(
        acoustic_ckpt_path=ac_path,
        p2g_ckpt_path=p2g_path,
        device=device,
        mode=args.mode,
    )
    if args.mode == "end_to_end":
        print(f"✅ Full Pipeline loaded (Acoustic Step: {pipeline.acoustic_step} | P2G Step: {pipeline.p2g_step})")
    else:
        print(f"✅ P2G Pipeline loaded (P2G Step: {pipeline.p2g_step} | Best Val WER: {pipeline.best_p2g_wer:.2f}%)")

    selected = args.benchmarks
    if "all" in selected:
        selected = [
            "librispeech_clean",
            "librispeech_other",
            "fleurs_en",
            "fleurs_fr",
            "fleurs_es",
            "fleurs_it",
            "mls_fr",
            "mls_es",
            "mls_it",
        ]

    benchmark_tasks = []
    for b_id in selected:
        if b_id == "librispeech_clean":
            benchmark_tasks.append((
                "LibriSpeech (test-clean)",
                "en",
                lambda: get_librispeech_data("data/librispeech/librispeech_test_clean.json", max_samples),
            ))
        elif b_id == "librispeech_other":
            benchmark_tasks.append((
                "LibriSpeech (test-other)",
                "en",
                lambda: get_librispeech_data("data/librispeech/librispeech_test_other.json", max_samples),
            ))
        elif b_id == "fleurs_en":
            benchmark_tasks.append((
                "Google FLEURS (en_us)",
                "en",
                lambda: get_fleurs_data("en_us", "en", max_samples),
            ))
        elif b_id == "fleurs_fr":
            benchmark_tasks.append((
                "Google FLEURS (fr_fr)",
                "fr",
                lambda: get_fleurs_data("fr_fr", "fr", max_samples),
            ))
        elif b_id == "fleurs_es":
            benchmark_tasks.append((
                "Google FLEURS (es_419)",
                "es",
                lambda: get_fleurs_data("es_419", "es", max_samples),
            ))
        elif b_id == "fleurs_it":
            benchmark_tasks.append((
                "Google FLEURS (it_it)",
                "it",
                lambda: get_fleurs_data("it_it", "it", max_samples),
            ))
        elif b_id == "mls_fr":
            benchmark_tasks.append((
                "MLS French (test)",
                "fr",
                lambda: get_mls_data("/media/hdd/Datasets/mls/mls_french/test/transcripts.txt", "/media/hdd/Datasets/mls/mls_french/test/audio", "fr", max_samples),
            ))
        elif b_id == "mls_es":
            benchmark_tasks.append((
                "MLS Spanish (test)",
                "es",
                lambda: get_mls_data("/media/hdd/Datasets/mls/mls_spanish/test/transcripts.txt", "/media/hdd/Datasets/mls/mls_spanish/test/audio", "es", max_samples),
            ))
        elif b_id == "mls_it":
            benchmark_tasks.append((
                "MLS Italian (test)",
                "it",
                lambda: get_mls_data("/media/hdd/Datasets/mls/mls_italian/test/transcripts.txt", "/media/hdd/Datasets/mls/mls_italian/test/audio", "it", max_samples),
            ))

    total_tasks = len(benchmark_tasks)
    print(f"📋 Queued {total_tasks} benchmark datasets to evaluate.\n")

    results = []
    overall_start_time = time.time()

    for idx_task, (b_name, b_lang, data_fn) in enumerate(benchmark_tasks, 1):
        print(f"\n▶ [{idx_task}/{total_tasks}] Preparing dataset: {b_name}...")
        iterator, total_s = data_fn()
        res = run_benchmark(
            name=b_name,
            lang=b_lang,
            iterator=iterator,
            pipeline=pipeline,
            total_samples=total_s,
            verbose_samples=3,
            log_interval_samples=25,
            log_interval_sec=12.0,
        )
        results.append(res)
        save_intermediate_results(out_p, p2g_path, pipeline.acoustic_step, pipeline.p2g_step, args.mode, results)
        print(f"  💾 Results checkpointed to {out_p}")

    total_duration_str = format_interval(time.time() - overall_start_time)

    # Final Summary Table
    print("\n" + "=" * 92)
    print(f"📈 FINAL BENCHMARK SUMMARY (Phono-V7.5 | Mode: {args.mode.upper()} | Total Time: {total_duration_str})")
    print("=" * 92)
    print(f"| {'Benchmark Dataset':<28} | {'Lang':<5} | {'Samples':<8} | {'WER (%)':<9} | {'CER (%)':<9} | {'PER (%)':<9} | {'RTF':<7} |")
    print(f"|{'-'*30}|{'-'*7}|{'-'*10}|{'-'*11}|{'-'*11}|{'-'*11}|{'-'*9}|")
    for r in results:
        if r.get("wer") is not None:
            per_val = f"{r['per']:<9.2f}" if r.get("per") is not None else f"{'N/A':<9}"
            print(f"| {r['name']:<28} | {r['lang']:<5} | {r['samples']:<8} | {r['wer']:<9.2f} | {r['cer']:<9.2f} | {per_val} | {r['rtf']:<7.3f} |")
        else:
            print(f"| {r['name']:<28} | {r['lang']:<5} | {'N/A':<8} | {'N/A':<9} | {'N/A':<9} | {'N/A':<9} | {'N/A':<7} |")
    print("=" * 92)

    # Final Save
    save_intermediate_results(out_p, p2g_path, pipeline.acoustic_step, pipeline.p2g_step, args.mode, results)
    print(f"\n💾 Full final results saved to: {out_p}")


if __name__ == "__main__":
    main()


