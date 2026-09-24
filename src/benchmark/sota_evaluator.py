"""SOTA Benchmark Evaluator: Compare Scratch HuBERT vs Meta HuBERT & OpenAI Whisper."""

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import soundfile as sf
import torch
import jiwer
from transformers import HubertForCTC, Wav2Vec2Processor, WhisperForConditionalGeneration, WhisperProcessor

from src.models.config import HuBERTConfig
from src.models.hubert_asr import HuBERTForCTC


def normalize_text(text: str) -> str:
    """Standard ASR text normalization: lowercasing, stripping punctuation, collapsing spaces."""
    if not text:
        return ""
    text = text.lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


class SOTABenchmarkRunner:
    """Manages loading and running inference for Scratch HuBERT, Meta HuBERT, and OpenAI Whisper."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scratch_model = None
        self.scratch_vocab = ['<blank>', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'", '<unk>']
        self.meta_processor = None
        self.meta_model = None
        self.whisper_processor = None
        self.whisper_model = None
        self.models_loaded = False
        self.phonemizer_voice = None

    def get_phonemizer(self):
        """Lazy load Piper voice for phonetic transcription and PER evaluation."""
        if self.phonemizer_voice is None:
            try:
                from piper.voice import PiperVoice
                p = Path("/home/nathan/github/MADGen/data/piper_voices/en/en_US/lessac/en_US-lessac-medium.onnx")
                if p.exists():
                    self.phonemizer_voice = PiperVoice.load(str(p))
            except Exception:
                pass
        return self.phonemizer_voice

    def phonemize_text(self, text: str) -> str:
        """Convert transcript to canonical IPA phonemes for Phoneme Error Rate (PER)."""
        if not text:
            return ""
        v = self.get_phonemizer()
        if v is not None:
            try:
                chunks = v.phonemize(text)
                return " ".join("".join(c) for c in chunks)
            except Exception:
                pass
        return ""

    def load_models(self, include_meta: bool = True, include_whisper: bool = True):
        """Lazy-load the models to GPU/CPU memory."""
        if self.models_loaded:
            return

        print(f"[Benchmark] Loading benchmark models on {self.device}...")

        # 1. Scratch HuBERT
        ckpt_path = Path("checkpoints/best_model.pt")
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.scratch_model = HuBERTForCTC(ckpt["config"]).to(self.device)
            self.scratch_model.load_state_dict(ckpt["model_state_dict"])
            self.scratch_model.eval()
            print("[Benchmark] Loaded Scratch HuBERT (8.03M params).")
        else:
            print("[Benchmark] Checkpoint not found; creating scratch architecture.")
            cfg = HuBERTConfig(vocab_size=30, encoder_layers=4, encoder_heads=4, encoder_embed_dim=256)
            self.scratch_model = HuBERTForCTC(cfg).to(self.device).eval()

        # 2. Meta HuBERT Large
        if include_meta:
            print("[Benchmark] Loading Meta HuBERT-Large (facebook/hubert-large-ls960-ft)...")
            self.meta_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
            self.meta_model = HubertForCTC.from_pretrained(
                "facebook/hubert-large-ls960-ft", use_safetensors=True
            ).to(self.device).eval()
            print("[Benchmark] Loaded Meta HuBERT-Large (315.5M params).")

        # 3. OpenAI Whisper Tiny
        if include_whisper:
            print("[Benchmark] Loading OpenAI Whisper-Tiny (openai/whisper-tiny)...")
            self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                "openai/whisper-tiny", use_safetensors=True
            ).to(self.device).eval()
            print("[Benchmark] Loaded OpenAI Whisper-Tiny (37.8M params).")

        self.models_loaded = True

    def transcribe_scratch(self, speech_tensor: torch.Tensor, blank_penalty: float = 0.0) -> Dict:
        """Run Scratch HuBERT inference."""
        audio = speech_tensor.to(self.device)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            outputs = self.scratch_model(audio)
            logits = outputs["logits"]
            if blank_penalty > 0:
                logits[:, :, 0] -= blank_penalty

            top_indices = torch.argmax(logits, dim=-1)[0].cpu().tolist()
            res = []
            prev = None
            for idx in top_indices:
                if idx != prev:
                    if idx != 0 and idx < len(self.scratch_vocab):
                        res.append(self.scratch_vocab[idx])
                    prev = idx
            text = "".join(res)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat_ms = (time.perf_counter() - t0) * 1000.0

        return {"text": text, "latency_ms": round(lat_ms, 2)}

    def transcribe_meta_hubert(self, speech_np) -> Dict:
        """Run Meta HuBERT-Large inference."""
        if self.meta_model is None or self.meta_processor is None:
            self.load_models()

        inputs = self.meta_processor(speech_np, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            logits = self.meta_model(inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.meta_processor.batch_decode(predicted_ids)[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat_ms = (time.perf_counter() - t0) * 1000.0

        return {"text": transcription.lower(), "latency_ms": round(lat_ms, 2)}

    def transcribe_whisper(self, speech_np) -> Dict:
        """Run OpenAI Whisper-Tiny inference."""
        if self.whisper_model is None or self.whisper_processor is None:
            self.load_models()

        inputs = self.whisper_processor(speech_np, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            predicted_ids = self.whisper_model.generate(
                inputs,
                language="en",
                task="transcribe",
            )
            transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat_ms = (time.perf_counter() - t0) * 1000.0

        return {"text": transcription.strip().lower(), "latency_ms": round(lat_ms, 2)}

    def compare_audio(
        self,
        audio_path_or_array: Union[str, torch.Tensor],
        sample_rate: int = 16000,
        ground_truth: Optional[str] = None,
        blank_penalty: float = 0.0,
    ) -> Dict:
        """Run head-to-head transcription across all 3 models on a single audio recording."""
        self.load_models()

        if isinstance(audio_path_or_array, (str, Path)):
            speech_np, sr = sf.read(str(audio_path_or_array))
            if sr != 16000:
                import torchaudio.transforms as T
                t_audio = torch.tensor(speech_np, dtype=torch.float32)
                resampler = T.Resample(sr, 16000)
                speech_np = resampler(t_audio).numpy()
            speech_tensor = torch.tensor(speech_np, dtype=torch.float32)
        elif isinstance(audio_path_or_array, torch.Tensor):
            speech_tensor = audio_path_or_array.squeeze().cpu()
            speech_np = speech_tensor.numpy()
        else:
            speech_np = audio_path_or_array
            speech_tensor = torch.tensor(speech_np, dtype=torch.float32)

        duration = len(speech_np) / 16000.0

        # Transcribe
        out_meta = self.transcribe_meta_hubert(speech_np)
        out_whisper = self.transcribe_whisper(speech_np)
        out_scratch = self.transcribe_scratch(speech_tensor, blank_penalty=blank_penalty)

        # Normalize texts and phonemes for evaluation
        ref_norm = normalize_text(ground_truth) if ground_truth else ""
        ref_phonemes = self.phonemize_text(ground_truth) if ground_truth else ""

        def eval_pred(raw_text):
            pred_norm = normalize_text(raw_text)
            pred_ph = self.phonemize_text(raw_text)
            if not ref_norm:
                return {"pred_norm": pred_norm, "pred_phonemes": pred_ph, "wer": None, "cer": None, "per": None}
            w = round(float(jiwer.wer(ref_norm, pred_norm)), 4) if ref_norm else 0.0
            c = round(float(jiwer.cer(ref_norm, pred_norm)), 4) if ref_norm else 0.0
            p = round(float(jiwer.wer(ref_phonemes, pred_ph)), 4) if ref_phonemes and pred_ph else (1.0 if ref_phonemes else 0.0)
            return {"pred_norm": pred_norm, "pred_phonemes": pred_ph, "wer": w, "cer": c, "per": p}

        meta_eval = eval_pred(out_meta["text"])
        whisper_eval = eval_pred(out_whisper["text"])
        scratch_eval = eval_pred(out_scratch["text"])

        return {
            "audio_duration_sec": round(duration, 2),
            "ground_truth": ground_truth or "",
            "ground_truth_normalized": ref_norm,
            "ground_truth_phonemes": ref_phonemes,
            "models": {
                "scratch_hubert": {
                    "name": "HuBERT (Ours - Scratch)",
                    "parameters": "8.03M",
                    "pretraining_hours": "0h (Scratch)",
                    "supervised_hours": "5.39h",
                    "raw_text": out_scratch["text"],
                    "normalized_text": scratch_eval["pred_norm"],
                    "phonemes": scratch_eval["pred_phonemes"],
                    "wer": scratch_eval["wer"],
                    "cer": scratch_eval["cer"],
                    "per": scratch_eval["per"],
                    "latency_ms": out_scratch["latency_ms"],
                    "rtf": round((out_scratch["latency_ms"] / 1000.0) / duration, 4),
                    "throughput_x": round(duration / (out_scratch["latency_ms"] / 1000.0 + 1e-6), 1),
                },
                "meta_hubert_large": {
                    "name": "Meta HuBERT-Large",
                    "parameters": "315.5M",
                    "pretraining_hours": "60,000h (Libri-Light)",
                    "supervised_hours": "960h (LibriSpeech)",
                    "raw_text": out_meta["text"],
                    "normalized_text": meta_eval["pred_norm"],
                    "phonemes": meta_eval["pred_phonemes"],
                    "wer": meta_eval["wer"],
                    "cer": meta_eval["cer"],
                    "per": meta_eval["per"],
                    "latency_ms": out_meta["latency_ms"],
                    "rtf": round((out_meta["latency_ms"] / 1000.0) / duration, 4),
                    "throughput_x": round(duration / (out_meta["latency_ms"] / 1000.0 + 1e-6), 1),
                },
                "whisper_tiny": {
                    "name": "OpenAI Whisper-Tiny",
                    "parameters": "37.8M",
                    "pretraining_hours": "680,000h (Multilingual)",
                    "supervised_hours": "680,000h (Weakly Supervised)",
                    "raw_text": out_whisper["text"],
                    "normalized_text": whisper_eval["pred_norm"],
                    "phonemes": whisper_eval["pred_phonemes"],
                    "wer": whisper_eval["wer"],
                    "cer": whisper_eval["cer"],
                    "per": whisper_eval["per"],
                    "latency_ms": out_whisper["latency_ms"],
                    "rtf": round((out_whisper["latency_ms"] / 1000.0) / duration, 4),
                    "throughput_x": round(duration / (out_whisper["latency_ms"] / 1000.0 + 1e-6), 1),
                },
            },
        }

    def evaluate_benchmark(
        self,
        manifest_path: str = "data/librispeech/librispeech_test_clean.json",
        max_samples: int = 50,
        blank_penalty: float = 0.0,
    ) -> Dict:
        """Run batch evaluation over a standard benchmark test split."""
        self.load_models()

        p = Path(manifest_path)
        if not p.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(p, "r", encoding="utf-8") as f:
            samples = json.load(f)

        if max_samples > 0:
            samples = samples[:max_samples]

        print(f"[Benchmark] Evaluating {len(samples)} samples from {p.name} across 3 models...")

        total_audio_sec = 0.0
        results_by_model = {
            "scratch_hubert": {"wers": [], "cers": [], "pers": [], "latencies": []},
            "meta_hubert_large": {"wers": [], "cers": [], "pers": [], "latencies": []},
            "whisper_tiny": {"wers": [], "cers": [], "pers": [], "latencies": []},
        }

        detailed_samples = []

        for i, sample in enumerate(samples):
            audio_path = sample["audio_path"]
            ground_truth = sample["transcript"]

            comp = self.compare_audio(
                audio_path_or_array=audio_path,
                ground_truth=ground_truth,
                blank_penalty=blank_penalty,
            )
            dur = comp["audio_duration_sec"]
            total_audio_sec += dur

            for m_key in ["scratch_hubert", "meta_hubert_large", "whisper_tiny"]:
                m_info = comp["models"][m_key]
                if m_info["wer"] is not None:
                    results_by_model[m_key]["wers"].append(m_info["wer"])
                    results_by_model[m_key]["cers"].append(m_info["cer"])
                    if m_info.get("per") is not None:
                        results_by_model[m_key]["pers"].append(m_info["per"])
                results_by_model[m_key]["latencies"].append(m_info["latency_ms"])

            detailed_samples.append({
                "id": sample["id"],
                "duration": dur,
                "ground_truth": ground_truth,
                "ground_truth_phonemes": comp.get("ground_truth_phonemes", ""),
                "scratch_pred": comp["models"]["scratch_hubert"]["raw_text"],
                "scratch_wer": comp["models"]["scratch_hubert"]["wer"],
                "scratch_cer": comp["models"]["scratch_hubert"]["cer"],
                "scratch_per": comp["models"]["scratch_hubert"].get("per"),
                "meta_pred": comp["models"]["meta_hubert_large"]["raw_text"],
                "meta_wer": comp["models"]["meta_hubert_large"]["wer"],
                "meta_cer": comp["models"]["meta_hubert_large"]["cer"],
                "meta_per": comp["models"]["meta_hubert_large"].get("per"),
                "whisper_pred": comp["models"]["whisper_tiny"]["raw_text"],
                "whisper_wer": comp["models"]["whisper_tiny"]["wer"],
                "whisper_cer": comp["models"]["whisper_tiny"]["cer"],
                "whisper_per": comp["models"]["whisper_tiny"].get("per"),
            })

            if (i + 1) % 10 == 0 or (i + 1) == len(samples):
                print(f"  Processed [{i+1}/{len(samples)}] samples...")

        # Aggregate metrics
        summary = {}
        for m_key, m_name, params, train_hrs in [
            ("scratch_hubert", "HuBERT (Ours - Scratch)", "8.03M", "5.39h"),
            ("meta_hubert_large", "Meta HuBERT-Large", "315.5M", "60,000h (pre-train) + 960h (fine-tune)"),
            ("whisper_tiny", "OpenAI Whisper-Tiny", "37.8M", "680,000h (weakly supervised)"),
        ]:
            wers = results_by_model[m_key]["wers"]
            cers = results_by_model[m_key]["cers"]
            pers = results_by_model[m_key]["pers"]
            lats = results_by_model[m_key]["latencies"]

            avg_wer = sum(wers) / len(wers) if wers else 0.0
            avg_cer = sum(cers) / len(cers) if cers else 0.0
            avg_per = sum(pers) / len(pers) if pers else 0.0
            avg_lat = sum(lats) / len(lats) if lats else 0.0
            total_lat_sec = sum(lats) / 1000.0
            rtf = total_lat_sec / total_audio_sec if total_audio_sec > 0 else 0.0

            summary[m_key] = {
                "name": m_name,
                "parameters": params,
                "training_data": train_hrs,
                "avg_wer": round(avg_wer * 100.0, 2),  # as %
                "avg_cer": round(avg_cer * 100.0, 2),  # as %
                "avg_per": round(avg_per * 100.0, 2),  # as %
                "avg_latency_ms": round(avg_lat, 2),
                "rtf": round(rtf, 4),
                "throughput_x": round(total_audio_sec / (total_lat_sec + 1e-6), 1),
            }

        benchmark_report = {
            "benchmark_dataset": p.stem,
            "num_samples_evaluated": len(samples),
            "total_audio_seconds": round(total_audio_sec, 2),
            "total_audio_hours": round(total_audio_sec / 3600.0, 3),
            "models_summary": summary,
            "detailed_samples": detailed_samples[:20],  # keep top 20 for preview
        }

        # Save to logs
        os.makedirs("logs", exist_ok=True)
        report_file = Path("logs/sota_benchmark_results.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(benchmark_report, f, indent=2)

        print(f"[Benchmark] Benchmark report saved to: {report_file}")
        return benchmark_report
