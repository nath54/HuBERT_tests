"""
CLI Streaming & Diagnostic Application for Tripartite Speech Recognition.
Features:
- Direct audio file input mode (one-pass offline or chunked streaming simulation)
- Live microphone streaming with hardware audio device enumeration and selection
- Deep Diagnostic Verbose Mode:
  * Frame-by-frame Top-3 Phoneme Candidates with IPA symbols and softmax probabilities
  * Conformer DeepSeek MoE routing decisions per layer per frame (top-2 experts & weights)
  * Layer 4 Draft Head vs Layer 12 Final Head comparison
  * Articulatory manner-of-articulation category predictions
  * Acoustic word boundary gate probabilities
  * Model 2 Word Latent statistics (z_word in R^512)
  * Model 3 Overlapping Length Expert MoE routing distribution
  * Autoregressive character rollout step-by-step probabilities
  * Detailed Markdown/Text diagnostic report generation for in-depth model improvement
- Granular per-part quantization (INT8, FP16, weight-only) and hardware device routing.
"""

# Import Modules
import argparse
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import Tensor

# Add project root to Python search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from clean_model_src import (
    ModelConfig,
    Model,
    ModelPartQuantizer,
    QuantizationConfig,
    DualThreadStreamingASR,
)
from clean_model_src.layers.heads import LENGTH_INTERVALS

# Phoneme Vocabulary Inventory (112 tokens matching Phono-V8 vocabulary)
PHONEME_VOCAB: List[str] = [
    "<pad>", "<blank>", "<mask>", "<silence>", "<noise>", "<same>", "<eos>", "<unk>",
    " ", "-", "ˈ", "ˌ", "ː", "ˑ",
    "a", "e", "i", "o", "u", "y", "ɑ", "ɒ", "ɔ", "ə", "ɛ", "ɜ", "ɪ", "ʊ", "ʌ", "æ", "ø", "œ",
    "\u0303", "ɥ", "j", "w",
    "b", "d", "f", "ɡ", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z",
    "θ", "ð", "ʃ", "ʒ", "ŋ", "ɲ", "ʁ", "ɹ", "ʔ",
    "tʃ", "dʒ", "ts", "dz",
    "x", "ɣ", "β", "ɾ", "ħ", "ʕ", "χ", "ɕ", "ʑ", "ç", "ɴ", "q", "ɽ",
    "ɐ", "ɯ", "ɤ", "ɨ", "1", "2", "3", "4", "5", ".",
    "\u031e", "\u0308", "\u032a", "\u0361", "\u1d5d",
    "c", "ɫ", "ɸ", "ʎ", "ʏ", "ʐ", "ʝ", "ɚ", "ᵻ",
    "\u02b2", "\u02e4", "\u0327", "\u0329",
    "!", ",", ":", ";", "?", "<reserved_111>"
]

ARTICULATORY_CATEGORIES: List[str] = [
    "NON_SPEECH", "PROSODY_PUNCT", "VOWEL", "STOP_PLOSIVE",
    "FRICATIVE", "NASAL", "LIQUID_APPROXIMANT", "AFFRICATE_TRILL"
]


class InferenceDiagnosticAnalyzer:
    """
    Diagnostic Profiler extracting deep internal representations across all 3 model parts.
    Logs Top-3 phonemes, MoE router expert selections per layer, boundary gates, and character rollout steps.
    """

    def __init__(self, vocab: List[str] = PHONEME_VOCAB) -> None:
        self.vocab: List[str] = vocab

    def get_token_str(self, token_id: int) -> str:
        """Translates integer token ID into human-readable IPA string."""
        if 0 <= token_id < len(self.vocab):
            return self.vocab[token_id]
        return f"<{token_id}>"

    def analyze_frame_predictions(
        self,
        phoneme_logits: Tensor,
        macro_logits: Optional[Tensor] = None,
        boundary_probs: Optional[Tensor] = None,
        draft_logits: Optional[Tensor] = None,
        frame_time_step_ms: float = 20.0,
    ) -> List[Dict[str, Any]]:
        """
        Extracts frame-by-frame Top-3 candidates, articulatory manner, and boundary activations.

        Args:
            phoneme_logits (Tensor): Main CTC logits [1, T, 112].
            macro_logits (Optional[Tensor]): Manner logits [1, T, 8].
            boundary_probs (Optional[Tensor]): Boundary gate activations [1, T, 1].
            draft_logits (Optional[Tensor]): Layer 4 draft phoneme logits [1, T, 112].
            frame_time_step_ms (float): Frame duration in ms (50Hz = 20ms).

        Returns:
            List[Dict[str, Any]]: Per-frame diagnostic records.
        """
        # Softmax probabilities: Tensor [T, 112]
        probs = F.softmax(phoneme_logits[0], dim=-1)
        t_frames: int = probs.shape[0]

        macro_probs = F.softmax(macro_logits[0], dim=-1) if macro_logits is not None else None
        b_probs = boundary_probs[0].squeeze(-1) if boundary_probs is not None else None
        draft_p = F.softmax(draft_logits[0], dim=-1) if draft_logits is not None else None

        records: List[Dict[str, Any]] = []

        for t in range(t_frames):
            # Top-3 phoneme candidates: Tensor [3], Role: most likely speech sounds
            top3_vals, top3_idx = torch.topk(probs[t], k=3)
            top3_info = [
                {
                    "id": int(top3_idx[k].item()),
                    "symbol": self.get_token_str(int(top3_idx[k].item())),
                    "prob": float(top3_vals[k].item()),
                }
                for k in range(3)
            ]

            frame_data: Dict[str, Any] = {
                "frame": t,
                "time_ms": round(t * frame_time_step_ms, 1),
                "top3": top3_info,
            }

            if macro_probs is not None:
                macro_val, macro_id = torch.max(macro_probs[t], dim=-1)
                m_idx = int(macro_id.item())
                cat_str = ARTICULATORY_CATEGORIES[m_idx] if m_idx < len(ARTICULATORY_CATEGORIES) else f"CAT_{m_idx}"
                frame_data["manner"] = {"category": cat_str, "prob": float(macro_val.item())}

            if b_probs is not None:
                frame_data["boundary_prob"] = float(b_probs[t].item())

            if draft_p is not None:
                d_val, d_id = torch.max(draft_p[t], dim=-1)
                frame_data["draft_head"] = {
                    "id": int(d_id.item()),
                    "symbol": self.get_token_str(int(d_id.item())),
                    "prob": float(d_val.item()),
                }

            records.append(frame_data)

        return records

    def analyze_moe_layers(
        self,
        moe_routings: List[Dict[str, Tensor]],
    ) -> List[Dict[str, Any]]:
        """
        Analyzes DeepSeek MoE routing decisions across all Conformer layers.

        Args:
            moe_routings (List[Dict[str, Tensor]]): List of routing dictionaries per layer.

        Returns:
            List[Dict[str, Any]]: Layer-by-layer MoE load distribution and router entropy.
        """
        layer_summaries: List[Dict[str, Any]] = []

        for l_idx, r_dict in enumerate(moe_routings):
            # Indices: Tensor [T, top_k], Weights: Tensor [T, top_k], Probs: Tensor [T, num_experts]
            probs = r_dict["probs"]
            indices = r_dict["indices"]
            t_tokens, num_exp = probs.shape

            # Compute load per expert (average routing probability)
            avg_probs = probs.mean(dim=0).cpu().numpy()

            # Dispatch counts per expert
            dispatch_counts = np.zeros(num_exp, dtype=int)
            flat_indices = indices.view(-1).cpu().numpy()
            for idx in flat_indices:
                if 0 <= idx < num_exp:
                    dispatch_counts[idx] += 1

            # Compute Shannon entropy of routing distribution: H = -sum(p * log(p))
            norm_dispatch = dispatch_counts / max(1, dispatch_counts.sum())
            entropy = -float(np.sum([p * math.log(p + 1e-12) for p in norm_dispatch if p > 0]))
            max_entropy = math.log(num_exp)
            load_balance_score = round((entropy / max_entropy) * 100.0, 1) if max_entropy > 0 else 100.0

            layer_summaries.append({
                "layer_index": l_idx,
                "num_experts": num_exp,
                "tokens_evaluated": t_tokens,
                "expert_load_percentage": [round(float(p) * 100.0, 2) for p in avg_probs],
                "expert_dispatch_counts": dispatch_counts.tolist(),
                "load_balance_score_pct": load_balance_score,
            })

        return layer_summaries

    def generate_markdown_report(
        self,
        audio_path: str,
        audio_duration_sec: float,
        final_transcript: str,
        word_slices: List[List[int]],
        decoded_words: List[str],
        frame_diagnostics: List[Dict[str, Any]],
        moe_summaries: List[Dict[str, Any]],
        wd_details: Optional[Dict[str, Any]] = None,
        perf_metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generates a comprehensive Markdown diagnostic report detailing all inference internals.

        Returns:
            str: Markdown formatted diagnostic report.
        """
        md = []
        md.append("# Model Inference Diagnostic & Deep Analysis Report")
        md.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        md.append(f"**Audio Source**: `{audio_path}` ({audio_duration_sec:.2f} s)")
        md.append("")

        # 1. Executive Summary
        md.append("## 1. Executive Summary")
        md.append(f"- **Final Orthographic Transcript**: `{final_transcript}`")
        md.append(f"- **Total Words Segmented by Model 1**: {len(word_slices)}")
        md.append(f"- **Total Words Decoded by Models 2+3**: {len(decoded_words)}")
        if perf_metrics:
            md.append(f"- **Total Latency**: {perf_metrics.get('total_latency_ms', 0.0)} ms | "
                      f"**Model 1 (Acoustic)**: {perf_metrics.get('avg_acoustic_latency_ms', 0.0)} ms | "
                      f"**Models 2+3 (Decoder)**: {perf_metrics.get('avg_decoder_latency_ms', 0.0)} ms")
        md.append("")

        # 2. Word-by-Word Deep Dive
        md.append("## 2. Word-by-Word Decoding Analysis")
        md.append("| Word # | Emitted Text | Phoneme IDs | IPA Representation | Length Expert Interval |")
        md.append("| :---: | :--- | :--- | :--- | :--- |")

        top1_experts = wd_details.get("top1_experts", torch.zeros(len(decoded_words), dtype=torch.long)).cpu().tolist() if wd_details else []

        for w_idx in range(len(decoded_words)):
            ph_ids = word_slices[w_idx] if w_idx < len(word_slices) else []
            ipa_str = " ".join([self.get_token_str(t) for t in ph_ids])
            w_text = decoded_words[w_idx] if w_idx < len(decoded_words) else ""
            exp_idx = top1_experts[w_idx] if w_idx < len(top1_experts) else 0
            interval_str = str(LENGTH_INTERVALS[exp_idx]) if exp_idx < len(LENGTH_INTERVALS) else "Unknown"

            md.append(f"| {w_idx+1} | **{w_text}** | `{ph_ids}` | `/{ipa_str}/` | `{interval_str}` |")
        md.append("")

        # 3. MoE Layer Load Balancing & Router Utilization
        md.append("## 3. Conformer DeepSeek MoE Routing Distribution per Layer")
        md.append("Analysis of expert specialization across all Conformer blocks (8 routed experts, top-2 gating):")
        md.append("")
        md.append("| Layer # | Balance Score | Exp 0 | Exp 1 | Exp 2 | Exp 3 | Exp 4 | Exp 5 | Exp 6 | Exp 7 |")
        md.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for layer in moe_summaries:
            loads = layer["expert_load_percentage"]
            load_cells = " | ".join([f"{l:.1f}%" for l in loads[:8]])
            md.append(f"| Layer {layer['layer_index']} | **{layer['load_balance_score_pct']}%** | {load_cells} |")
        md.append("")

        # 4. Frame-by-Frame Top-3 Phoneme Candidates (Sampled every 5th frame for brevity)
        md.append("## 4. Acoustic Frame-by-Frame Diagnostics (Sampled Snippet)")
        md.append("| Time (ms) | Top-1 (Prob) | Top-2 (Prob) | Top-3 (Prob) | Articulatory Manner | Boundary Gate | Draft Head (L4) |")
        md.append("| :---: | :--- | :--- | :--- | :--- | :---: | :--- |")

        for f_data in frame_diagnostics[::3]:  # sample every 3rd frame (60ms intervals)
            t_ms = f_data["time_ms"]
            t1 = f"{f_data['top3'][0]['symbol']} ({f_data['top3'][0]['prob']:.2f})"
            t2 = f"{f_data['top3'][1]['symbol']} ({f_data['top3'][1]['prob']:.2f})"
            t3 = f"{f_data['top3'][2]['symbol']} ({f_data['top3'][2]['prob']:.2f})"
            manner = f"{f_data.get('manner', {}).get('category', '-')} ({f_data.get('manner', {}).get('prob', 0.0):.2f})"
            bg = f"{f_data.get('boundary_prob', 0.0):.2f}"
            dh = f"{f_data.get('draft_head', {}).get('symbol', '-')} ({f_data.get('draft_head', {}).get('prob', 0.0):.2f})"
            md.append(f"| {t_ms} | **{t1}** | {t2} | {t3} | {manner} | {bg} | {dh} |")

        md.append("")

        # 5. Model Improvement Recommendations
        md.append("## 5. Architectural Optimization & Improvement Insights")
        # Identify underutilized experts
        collapsed_layers = [l["layer_index"] for l in moe_summaries if l["load_balance_score_pct"] < 70.0]
        if collapsed_layers:
            md.append(f"- ⚠️ **MoE Imbalance Warning**: Layers `{collapsed_layers}` exhibit low load balance (< 70%). Increase router exploration jitter (`router_jitter_std`) or increase auxiliary loss coefficient.")
        else:
            md.append("- ✅ **Healthy MoE Routing**: Experts across all 12 layers are uniformly engaged without token collapse.")

        # Identify low confidence phoneme intervals
        low_conf_frames = [f for f in frame_diagnostics if f["top3"][0]["prob"] < 0.40 and f["top3"][0]["symbol"] not in ("<blank>", "<silence>")]
        if low_conf_frames:
            md.append(f"- ℹ️ **Phonetic Ambiguity**: Detected {len(low_conf_frames)} frames with confidence < 40%. These regions benefit from the multi-task Articulatory Macro Head loss.")
        else:
            md.append("- ✅ **High Phonetic Confidence**: Most emitted speech phones exceed 60% argmax confidence.")

        return "\n".join(md)


class StreamingCLIApp:
    """
    Object-Oriented CLI Controller for Dual-Threaded Streaming Speech Recognition.
    Manages audio capture (microphone, streaming file, or offline file), engine lifecycle, and deep diagnostics.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args: argparse.Namespace = args
        self.sample_rate: int = args.sample_rate
        self.chunk_duration_sec: float = args.chunk_duration_sec
        self.chunk_size_samples: int = int(self.chunk_duration_sec * self.sample_rate)

        self.unified_model: Optional[Model] = None
        self.streamer: Optional[DualThreadStreamingASR] = None
        self.analyzer: InferenceDiagnosticAnalyzer = InferenceDiagnosticAnalyzer()

        self._words_printed: int = 0
        self._start_time: float = 0.0

    @staticmethod
    def list_audio_devices() -> None:
        """Enumerates and prints all audio input devices."""
        try:
            import sounddevice as sd
        except ImportError:
            print("[Error] sounddevice is not installed. Run: pip install sounddevice")
            return

        devices = sd.query_devices()
        default_in, _ = sd.default.device
        print("=" * 70)
        print("                 DETECTED AUDIO INPUT DEVICES                  ")
        print("=" * 70)
        input_count = 0
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                is_def = " (DEFAULT)" if idx == default_in else ""
                print(
                    f"  [{idx:2d}] {dev['name']}{is_def}\n"
                    f"       Channels: {dev['max_input_channels']} | Default SR: {int(dev['default_samplerate'])} Hz"
                )
                input_count += 1
        print("=" * 70)
        if input_count == 0:
            print("  [Warning] No audio input devices detected.")

    def build_model(self) -> Model:
        """Builds, routes, loads checkpoints, and quantizes the unified Model."""
        print("\n[Engine] Initializing Model Architecture & Routing...")
        cfg = ModelConfig()
        cfg.device_model_a2p = self.args.device_a2p
        cfg.device_model_pe = self.args.device_pe
        cfg.device_model_wd = self.args.device_wd

        model = Model(model_config=cfg)
        print(f"  - Model 1 (A2P) device: {model.model_a2p.get_device()}")
        print(f"  - Model 2 (PE)  device: {model.model_pe.get_device()}")
        print(f"  - Model 3 (WD)  device: {model.model_wd.get_device()}")

        # Checkpoints
        a2p_ckpt: str = self.args.a2p_ckpt
        p2g_ckpt: str = self.args.p2g_ckpt

        print("[Engine] Loading Checkpoints...")
        if os.path.isfile(a2p_ckpt):
            loaded_a2p = model.model_a2p.load_pretrained_checkpoint(a2p_ckpt)
            print(f"  - Model 1 weights from {a2p_ckpt}: {'SUCCESS' if loaded_a2p else 'FAILED'}")
        if os.path.isfile(p2g_ckpt):
            loaded_pe = model.model_pe.load_pretrained_checkpoint(p2g_ckpt)
            loaded_wd = model.model_wd.load_pretrained_checkpoint(p2g_ckpt)
            print(f"  - Models 2 & 3 weights from {p2g_ckpt}: PE={loaded_pe}, WD={loaded_wd}")

        # Quantization
        quantizer = ModelPartQuantizer()
        quant_map: Dict[str, QuantizationConfig] = {}
        if self.args.quantize_a2p != "none":
            quant_map["A2P"] = QuantizationConfig(precision=self.args.quantize_a2p)
        if self.args.quantize_pe != "none":
            quant_map["PE"] = QuantizationConfig(precision=self.args.quantize_pe)
        if self.args.quantize_wd != "none":
            quant_map["WD"] = QuantizationConfig(precision=self.args.quantize_wd)

        if quant_map:
            model = quantizer.quantize_model(model, quant_map)
            print("  - Granular quantization applied.")

        self.unified_model = model
        return model

    def on_word_emitted(self, word: str) -> None:
        """Callback for real-time word streaming to terminal."""
        clean_word = word.strip()
        if not clean_word:
            return
        if self._words_printed == 0:
            sys.stdout.write(f"\r[Transcribing] » {clean_word}")
        else:
            sys.stdout.write(f" {clean_word}")
        sys.stdout.flush()
        self._words_printed += 1

    def build_streamer(self) -> DualThreadStreamingASR:
        """Configures and returns the DualThreadStreamingASR runner."""
        if self.unified_model is None:
            self.build_model()

        phono_range = (
            (self.args.min_phoneme_id, self.args.max_phoneme_id)
            if self.args.min_phoneme_id > 0 and self.args.max_phoneme_id > 0
            else None
        )

        streamer = DualThreadStreamingASR(
            model_a2p=self.unified_model.model_a2p,
            model_pe=self.unified_model.model_pe,
            model_wd=self.unified_model.model_wd,
            chunk_duration_sec=self.args.chunk_duration_sec,
            sample_rate=self.sample_rate,
            word_callback=self.on_word_emitted,
            min_phonemes_per_word=self.args.min_phonemes_per_word,
            max_phonemes_per_word=self.args.max_phonemes_per_word,
            space_id=self.args.space_id,
            blank_id=self.args.blank_id,
            allowed_phoneme_range=phono_range,
            max_queue_size=self.args.queue_size,
        )
        self.streamer = streamer
        return streamer

    def run_offline_file(self, file_path: str) -> None:
        """
        Executes one-pass full offline inference with comprehensive diagnostic logging.

        Args:
            file_path (str): File path to audio.
        """
        if not os.path.isfile(file_path):
            print(f"[Error] Audio file not found: {file_path}")
            sys.exit(1)

        print(f"\n[Offline Mode] Loading audio from: {file_path}")
        wav, sr = sf.read(file_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        audio_t = torch.from_numpy(wav).float()
        if sr != self.sample_rate:
            import torchaudio.transforms as T
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            audio_t = resampler(audio_t.unsqueeze(0)).squeeze(0)

        total_sec = audio_t.shape[0] / float(self.sample_rate)
        print(f"[Offline Mode] Audio Duration: {total_sec:.2f} s ({audio_t.shape[0]} samples)")

        if self.unified_model is None:
            self.build_model()

        audio_tensor = audio_t.unsqueeze(0).to(self.unified_model.model_a2p.get_device())

        # One-pass forward inference with diagnostics
        t0 = time.perf_counter()
        with torch.no_grad():
            a2p_out = self.unified_model.model_a2p.forward_inference(audio_tensor)
            word_slices = a2p_out["word_slices"][0] if a2p_out["word_slices"] else []

            t_a2p = time.perf_counter() - t0

            # Forward through Model 2 & 3
            t_dec_start = time.perf_counter()
            if word_slices:
                z_words = self.unified_model.model_pe.forward_inference(word_slices)
                z_words_wd = z_words.to(self.unified_model.model_wd.get_device())
                decoded_words = self.unified_model.model_wd.forward_inference(z_words_wd)
            else:
                decoded_words = []
            t_dec = time.perf_counter() - t_dec_start

        total_time = time.perf_counter() - t0
        final_text = " ".join(decoded_words).strip()

        print("\n" + "=" * 70)
        print("                       FINAL TRANSCRIPT                        ")
        print("=" * 70)
        print(final_text if final_text else "[No speech recognized]")
        print("=" * 70)

        perf = {
            "total_latency_ms": round(total_time * 1000.0, 2),
            "avg_acoustic_latency_ms": round(t_a2p * 1000.0, 2),
            "avg_decoder_latency_ms": round(t_dec * 1000.0, 2),
            "rtf": round(total_time / total_sec, 4) if total_sec > 0 else 0.0,
        }

        print(f"\n[Performance Summary] Total Time: {perf['total_latency_ms']} ms | RTF: {perf['rtf']}")

        # Diagnostic Analysis & Report Generation
        if self.args.verbose or self.args.verbose_log:
            print("\n[Diagnostics] Compiling Deep Diagnostic Analysis Report...")
            frame_diag = self.analyzer.analyze_frame_predictions(
                phoneme_logits=a2p_out["logits"],
                macro_logits=a2p_out.get("macro_logits"),
                boundary_probs=a2p_out.get("boundary_probs"),
                draft_logits=a2p_out.get("draft_logits"),
            )
            moe_diag = self.analyzer.analyze_moe_layers(a2p_out.get("moe_routing", []))
            wd_details = getattr(self.unified_model.model_wd, "last_decoding_details", {})

            report_md = self.analyzer.generate_markdown_report(
                audio_path=file_path,
                audio_duration_sec=total_sec,
                final_transcript=final_text,
                word_slices=word_slices,
                decoded_words=decoded_words,
                frame_diagnostics=frame_diag,
                moe_summaries=moe_diag,
                wd_details=wd_details,
                perf_metrics=perf,
            )

            log_file = self.args.verbose_log or "inference_diagnostic_log.md"
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(report_md)
            print(f"[Diagnostics] Deep Analysis Report saved to: {log_file}")

        if self.args.output_file:
            with open(self.args.output_file, "w", encoding="utf-8") as f:
                f.write(final_text + "\n")

    def run_file_stream(self, file_path: str) -> None:
        """Simulates live streaming from audio file."""
        if not os.path.isfile(file_path):
            print(f"[Error] Audio file not found: {file_path}")
            sys.exit(1)

        print(f"\n[Streaming File] Loading audio from: {file_path}")
        wav, sr = sf.read(file_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        audio_tensor = torch.from_numpy(wav).float()
        if sr != self.sample_rate:
            import torchaudio.transforms as T
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            audio_tensor = resampler(audio_tensor.unsqueeze(0)).squeeze(0)

        total_samples = audio_tensor.shape[0]
        total_sec = total_samples / float(self.sample_rate)

        streamer = self.build_streamer()
        streamer.start()

        print("\n" + "=" * 70)
        print("  STREAMING AUDIO CHUNKS... Press Ctrl+C to abort.")
        print("=" * 70)

        self._start_time = time.perf_counter()
        pos = 0
        try:
            while pos < total_samples and streamer.is_running():
                end_pos = min(pos + self.chunk_size_samples, total_samples)
                chunk = audio_tensor[pos:end_pos]
                streamer.push_audio(chunk)
                pos = end_pos
                if self.args.simulate_realtime:
                    time.sleep(chunk.shape[0] / float(self.sample_rate) * 0.95)
        except KeyboardInterrupt:
            print("\n[Stream] Interrupted by user.")
        finally:
            self._finalize_stream()

    def run_microphone_stream(self) -> None:
        """Captures audio live from microphone."""
        try:
            import sounddevice as sd
        except ImportError:
            print("[Error] sounddevice required for microphone. Run: pip install sounddevice")
            sys.exit(1)

        dev_idx = None
        if self.args.device_index is not None:
            dev_idx = self.args.device_index
        elif self.args.device_name:
            devices = sd.query_devices()
            query = self.args.device_name.lower()
            for idx, d in enumerate(devices):
                if d["max_input_channels"] > 0 and query in d["name"].lower():
                    dev_idx = idx
                    break

        dev_info = sd.query_devices(dev_idx, kind="input")
        print(f"\n[Microphone] Device [{dev_idx if dev_idx is not None else 'DEFAULT'}]: {dev_info['name']}")

        streamer = self.build_streamer()
        streamer.start()

        native_sr = int(dev_info["default_samplerate"])
        needs_resample = native_sr != self.sample_rate
        resampler = None
        if needs_resample:
            import torchaudio.transforms as T
            resampler = T.Resample(orig_freq=native_sr, new_freq=self.sample_rate)

        def audio_callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            samples_t = torch.from_numpy(indata[:, 0].copy()).float()
            if needs_resample and resampler is not None:
                samples_16k = resampler(samples_t.unsqueeze(0)).squeeze(0)
            else:
                samples_16k = samples_t
            streamer.push_audio(samples_16k)

        blocksize = int(native_sr * 0.05)
        print("\n" + "=" * 70)
        print("  LISTENING LIVE... Speak into microphone. Press Ctrl+C to stop.")
        print("=" * 70)

        self._start_time = time.perf_counter()
        try:
            with sd.InputStream(
                device=dev_idx,
                channels=1,
                samplerate=native_sr,
                blocksize=blocksize,
                callback=audio_callback,
            ):
                while streamer.is_running():
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[Microphone] Stopping stream...")
        finally:
            self._finalize_stream()

    def _finalize_stream(self) -> None:
        """Finalizes streaming inference."""
        if self.streamer is None:
            return

        print("\n[Stream] Flushing remaining words...")
        final_text = self.streamer.finish_stream(timeout=5.0)
        elapsed = time.perf_counter() - self._start_time

        print("\n" + "=" * 70)
        print("                       FINAL TRANSCRIPT                        ")
        print("=" * 70)
        print(final_text if final_text else "[No speech recognized]")
        print("=" * 70)

        metrics = self.streamer.get_performance_metrics()
        print("\n[Performance Breakdown]")
        print(f"  - Elapsed Time:             {elapsed:.2f} s")
        print(f"  - Audio Chunks Processed:   {metrics['chunks_processed']}")
        print(f"  - Words Decoded:            {metrics['words_decoded']}")
        print(f"  - Model 1 Acoustic Latency: {metrics['avg_acoustic_latency_ms']} ms / chunk")
        print(f"  - Models 2+3 Dec Latency:   {metrics['avg_decoder_latency_ms']} ms / word")
        print(f"  - Total Latency:            {metrics['total_latency_ms']} ms")

        if self.args.output_file:
            with open(self.args.output_file, "w", encoding="utf-8") as f:
                f.write(final_text + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Clean Tripartite Speech Recognition CLI & Diagnostic Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional optional audio file
    parser.add_argument(
        "audio_file",
        nargs="?",
        default=None,
        help="Direct path to an audio file (wav/flac/mp3) to transcribe",
    )

    # 1. Mode and Audio Input Settings
    inp_group = parser.add_argument_group("Audio Input & Mode Settings")
    inp_group.add_argument(
        "--mode",
        type=str,
        choices=["auto", "offline", "streaming", "mic"],
        default="auto",
        help="Execution mode: 'offline' for fast full-file inference, 'streaming' for chunked simulation, 'mic' for live microphone",
    )
    inp_group.add_argument(
        "--input_audio",
        type=str,
        default=None,
        help="Path to an audio file (alternative to positional argument)",
    )
    inp_group.add_argument(
        "--list_devices",
        action="store_true",
        help="List available audio input devices and exit",
    )
    inp_group.add_argument(
        "--device_index",
        type=int,
        default=None,
        help="Integer index of the audio input device",
    )
    inp_group.add_argument(
        "--device_name",
        type=str,
        default=None,
        help="Substring name of the audio input device to match",
    )
    inp_group.add_argument(
        "--simulate_realtime",
        action="store_true",
        default=True,
        help="Simulate real-time chunk delay when streaming audio files",
    )
    inp_group.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Audio sampling rate in Hertz",
    )

    # 2. Verbose Diagnostic Mode
    diag_group = parser.add_argument_group("Diagnostic & Verbose Settings")
    diag_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable deep diagnostic analysis logging",
    )
    diag_group.add_argument(
        "--verbose_log",
        type=str,
        default=None,
        help="File path to write detailed diagnostic markdown analysis report (e.g. diagnostic_report.md)",
    )

    # 3. Inference Engine Parametrization
    inf_group = parser.add_argument_group("Inference Engine Parametrization")
    inf_group.add_argument(
        "--chunk_duration_sec",
        type=float,
        default=0.3,
        help="Duration of streaming chunks in seconds (latency control)",
    )
    inf_group.add_argument(
        "--min_phonemes_per_word",
        type=int,
        default=1,
        help="Minimum phonemes required to form a word",
    )
    inf_group.add_argument(
        "--max_phonemes_per_word",
        type=int,
        default=32,
        help="Maximum phoneme sequence length passed to Model 2",
    )
    inf_group.add_argument(
        "--min_phoneme_id",
        type=int,
        default=1,
        help="Minimum valid phoneme ID filter",
    )
    inf_group.add_argument(
        "--max_phoneme_id",
        type=int,
        default=111,
        help="Maximum valid phoneme ID filter",
    )
    inf_group.add_argument(
        "--space_id",
        type=int,
        default=8,
        help="Token ID representing word-boundary spaces",
    )
    inf_group.add_argument(
        "--blank_id",
        type=int,
        default=0,
        help="CTC blank token ID to ignore",
    )
    inf_group.add_argument(
        "--queue_size",
        type=int,
        default=500,
        help="Capacity of thread-safe word queue",
    )

    # 4. Hardware Device & Quantization Routing
    hw_group = parser.add_argument_group("Hardware & Quantization Routing")
    hw_group.add_argument(
        "--device_a2p",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for Model 1 ('cuda', 'cpu')",
    )
    hw_group.add_argument(
        "--device_pe",
        type=str,
        default="cpu",
        help="Device for Model 2 ('cpu', 'cuda')",
    )
    hw_group.add_argument(
        "--device_wd",
        type=str,
        default="cpu",
        help="Device for Model 3 ('cpu', 'cuda')",
    )
    hw_group.add_argument(
        "--quantize_a2p",
        type=str,
        choices=["none", "int8", "fp16"],
        default="none",
        help="Quantization for Model 1",
    )
    hw_group.add_argument(
        "--quantize_pe",
        type=str,
        choices=["none", "int8", "fp16"],
        default="none",
        help="Quantization for Model 2",
    )
    hw_group.add_argument(
        "--quantize_wd",
        type=str,
        choices=["none", "int8", "fp16", "weight_only_int8"],
        default="none",
        help="Quantization for Model 3",
    )

    # 5. Checkpoints & Outputs
    ckpt_group = parser.add_argument_group("Checkpoints & Output")
    ckpt_group.add_argument(
        "--a2p_ckpt",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints/phono_v8_acoustic_multilingual_sota/latest_v8_acoustic.pt"),
        help="Path to Model 1 weights",
    )
    ckpt_group.add_argument(
        "--p2g_ckpt",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints/phono_v8_robust_p2g/best_robust_p2g.pt"),
        help="Path to Models 2 & 3 weights",
    )
    ckpt_group.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save the final recognized transcript",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_devices:
        StreamingCLIApp.list_audio_devices()
        sys.exit(0)

    # Determine input audio file (positional or flag)
    target_audio = args.audio_file or args.input_audio

    app = StreamingCLIApp(args=args)

    mode = args.mode.lower()
    if mode == "auto":
        mode = "offline" if target_audio else "mic"

    if mode == "offline":
        if not target_audio:
            print("[Error] Offline mode requires an audio file argument. Example:")
            print("  python clean_model_src/scripts/run_streaming_cli.py path/to/speech.wav --verbose")
            sys.exit(1)
        app.run_offline_file(target_audio)
    elif mode == "streaming":
        if not target_audio:
            print("[Error] Streaming file mode requires an audio file.")
            sys.exit(1)
        app.run_file_stream(target_audio)
    elif mode == "mic":
        app.run_microphone_stream()
    else:
        print(f"[Error] Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
