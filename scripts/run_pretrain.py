#!/usr/bin/env python3
"""Unified Modular Training Launcher for AudioLearn Speech Architectures.

Supports all registered model architectures (HuBERT K-Means, PhonoHuBERT Direct Phonemes, etc.)
and all parameter tiers (mini, small, medium, base) with variable hyperparameter overrides.
"""

import argparse
import collections
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import math

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.config import HuBERTConfig
from src.models.hubert_asr import HuBERTForCTC
from src.models.registry import ModelRegistry, STANDARD_TIERS
from src.data.tokenizer import CharacterTokenizer
from src.data.phoneme_tokenizer import PhonemeTokenizer
from src.data.streaming_piper import (
    PiperVoiceManager,
    ProceduralTextSampler,
    VoiceDepletionError,
)
from src.data.target_extractors import BaseTargetExtractor
from data.sample_dataset import load_manifest
from src.data.dataset import AudioASRDataset, AudioCollateFn
from src.benchmark.sota_evaluator import SOTABenchmarkRunner
from src.data.threaded_dataset import StepProfiler, BufferedSpeechBatchGenerator
from src.training.run_manager import RunManager


class ModularStreamingDataset(torch.utils.data.IterableDataset):
    """Modular IterableDataset pairing procedural audio with dynamic target extractors."""

    def __init__(
        self,
        voice_manager: PiperVoiceManager,
        text_sampler: ProceduralTextSampler,
        target_extractor: BaseTargetExtractor,
        min_duration_sec: float = 1.0,
        max_duration_sec: float = 12.0,
    ):
        super().__init__()
        self.voice_manager = voice_manager
        self.text_sampler = text_sampler
        self.target_extractor = target_extractor
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec

    def __iter__(self):
        while True:
            res = self.text_sampler.sample_sentence()
            if isinstance(res, tuple):
                text, lang = res
            else:
                text, lang = res, "en"

            try:
                waveform, voice_name, dur = self.voice_manager.synthesize_to_tensor_16k(text, lang=lang)
            except VoiceDepletionError:
                raise
            except Exception:
                continue

            if dur < self.min_duration_sec or dur > self.max_duration_sec:
                continue

            # Retrieve active voice model instance for phonemization if needed
            active_voice_inst = self.voice_manager.voice_cache.get(voice_name.split("#")[0])

            target_dict = self.target_extractor.extract_targets(
                waveform=waveform,
                text=text,
                voice=active_voice_inst,
                lang=lang,
            )

            yield {
                "audio": waveform,
                "targets": target_dict["targets"],
                "target_length": target_dict["target_lengths"],
                "duration": dur,
                "text": text,
                "voice_name": voice_name,
            }


def collate_modular_batch(batch: List[Dict]) -> Dict[str, any]:
    """Collate variable-length audio and target sequences into padded batches."""
    audio_list = [item["audio"] for item in batch]
    target_list = [item["targets"] for item in batch]
    durations = [item["duration"] for item in batch]
    texts = [item["text"] for item in batch]
    voices = [item["voice_name"] for item in batch]

    audio_lengths = torch.tensor([len(a) for a in audio_list], dtype=torch.long)
    max_audio_len = int(audio_lengths.max().item())
    padded_audio = torch.zeros((len(batch), max_audio_len), dtype=torch.float32)
    for i, a in enumerate(audio_list):
        padded_audio[i, : len(a)] = a

    target_lengths = torch.tensor([len(t) for t in target_list], dtype=torch.long)
    max_target_len = int(target_lengths.max().item())
    padded_targets = torch.zeros((len(batch), max_target_len), dtype=torch.long)
    for i, t in enumerate(target_list):
        padded_targets[i, : len(t)] = t

    return {
        "audio": padded_audio,
        "audio_lengths": audio_lengths,
        "targets": padded_targets,
        "target_lengths": target_lengths,
        "durations": durations,
        "texts": texts,
        "voices": voices,
    }


def parse_overrides(override_str: str) -> Dict[str, Any]:
    """Parse comma-separated variable parameter overrides, e.g. 'mask_prob=0.7,encoder_layers=6'."""
    overrides = {}
    if not override_str:
        return overrides
    for pair in override_str.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            k = k.strip()
            v = v.strip()
            if v.isdigit():
                overrides[k] = int(v)
            else:
                try:
                    overrides[k] = float(v)
                except ValueError:
                    if v.lower() in ("true", "yes"):
                        overrides[k] = True
                    elif v.lower() in ("false", "no"):
                        overrides[k] = False
                    else:
                        overrides[k] = v
    return overrides


def run_quick_ctc_calibration(
    pretrain_model: nn.Module,
    config: Any,
    tokenizer: CharacterTokenizer,
    device: torch.device,
    probe_steps: int = 25,
) -> HuBERTForCTC:
    """Calibrate CTC head on LibriSpeech to measure downstream recognition."""
    ctc_model = HuBERTForCTC(config=config).to(device)
    if hasattr(pretrain_model, "transfer_to_ctc_model"):
        pretrain_model.transfer_to_ctc_model(ctc_model)

    if probe_steps <= 0:
        return ctc_model

    train_manifest = Path("data/librispeech/librispeech_train.json")
    if not train_manifest.exists():
        return ctc_model

    samples = load_manifest(str(train_manifest))[:120]
    ds = AudioASRDataset(samples, tokenizer, target_sample_rate=16000, max_duration_s=6.0)
    loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=AudioCollateFn(pad_token_id=tokenizer.pad_id))

    optimizer = torch.optim.AdamW(ctc_model.parameters(), lr=0.0003, weight_decay=1e-2)
    ctc_model.train()

    step_count = 0
    for batch in loader:
        if step_count >= probe_steps:
            break
        optimizer.zero_grad()
        out = ctc_model(
            audio=batch["audio"].to(device),
            audio_lengths=batch["audio_lengths"].to(device),
            targets=batch["targets"].to(device),
            target_lengths=batch["target_lengths"].to(device),
        )
        loss = out["loss"]
        if loss is not None and not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ctc_model.parameters(), max_norm=1.0)
            optimizer.step()
        step_count += 1

    return ctc_model


def evaluate_on_benchmark(ctc_model: HuBERTForCTC, tokenizer: CharacterTokenizer, device: torch.device, num_samples: int = 20) -> Dict:
    """Evaluate calibrated CTC model on standard LibriSpeech test-clean benchmark."""
    test_manifest = Path("data/librispeech/librispeech_test_clean.json")
    if not test_manifest.exists():
        return {"wer": 100.0, "cer": 100.0, "per": 100.0, "sample_pred": ""}

    with open(test_manifest, "r", encoding="utf-8") as f:
        samples = json.load(f)[:num_samples]

    runner = SOTABenchmarkRunner(device=str(device))
    wers, cers, pers = [], [], []
    sample_pred = ""
    ctc_model.eval()

    import soundfile as sf
    import torchaudio.transforms as T

    resamplers = {}
    for idx, s in enumerate(samples):
        speech_np, sr = sf.read(s["audio_path"])
        if sr != 16000:
            if sr not in resamplers:
                resamplers[sr] = T.Resample(sr, 16000)
            t_audio = torch.tensor(speech_np, dtype=torch.float32).unsqueeze(0)
            speech_tensor = resamplers[sr](t_audio).squeeze(0).to(device)
        else:
            speech_tensor = torch.tensor(speech_np, dtype=torch.float32).to(device)

        ref_text = s.get("transcript") or s.get("text", "")
        if speech_tensor.ndim == 1:
            speech_tensor = speech_tensor.unsqueeze(0)

        with torch.no_grad():
            out = ctc_model(speech_tensor)
            logits = out["logits"]
            top_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()

            # CTC greedy decoding
            pred_text = tokenizer.decode(top_ids)
            if idx == 0:
                sample_pred = pred_text

        import jiwer
        from src.benchmark.sota_evaluator import normalize_text

        ref_norm = normalize_text(ref_text)
        pred_norm = normalize_text(pred_text)
        ref_phonemes = runner.phonemize_text(ref_text)
        pred_phonemes = runner.phonemize_text(pred_text)

        w = round(float(jiwer.wer(ref_norm, pred_norm)), 4) if ref_norm else 1.0
        c = round(float(jiwer.cer(ref_norm, pred_norm)), 4) if ref_norm else 1.0
        p = round(float(jiwer.wer(ref_phonemes, pred_phonemes)), 4) if ref_phonemes and pred_phonemes else (1.0 if ref_phonemes else 0.0)

        wers.append(w)
        cers.append(c)
        pers.append(p)

    avg_wer = round(float(sum(wers) / max(1, len(wers))) * 100.0, 2)
    avg_cer = round(float(sum(cers) / max(1, len(cers))) * 100.0, 2)
    avg_per = round(float(sum(pers) / max(1, len(pers))) * 100.0, 2)

    return {"wer": avg_wer, "cer": avg_cer, "per": avg_per, "sample_pred": sample_pred}


# Global cache for LexiconDecoder to avoid rebuilding 206k trie every evaluation step
_GLOBAL_LEX_DECODER = None

def evaluate_direct_phonemes(
    model: nn.Module,
    phoneme_tokenizer: PhonemeTokenizer,
    device: torch.device,
    manifest_path: Union[str, Path] = "data/librispeech/librispeech_test_clean.json",
    num_samples: int = 20,
    blank_penalty: float = 0.0,
    compute_lexicon: bool = False,
) -> Dict[str, Any]:
    """Evaluate direct phoneme prediction model on a given JSON manifest."""
    global _GLOBAL_LEX_DECODER
    manifest = Path(manifest_path)
    if not manifest.exists():
        return {"wer": 100.0, "cer": 100.0, "per": 100.0, "sample_pred": ""}

    with open(manifest, "r", encoding="utf-8") as f:
        samples = json.load(f)[:num_samples]

    runner = SOTABenchmarkRunner(device=str(device))
    pers, cers = [], []
    sample_pred = ""
    model.eval()

    import soundfile as sf
    import torchaudio.transforms as T
    import jiwer
    import inspect

    # Optional Lexicon Decoder support
    lex_decoder = None
    lex_pers = []
    lex_wers = []
    sample_lex_pred = ""
    try:
        if _GLOBAL_LEX_DECODER is None:
            from src.decoder.lexicon_decoder import LexiconDecoder
            lex_file = Path("data/librispeech-lexicon.txt")
            if lex_file.exists():
                _GLOBAL_LEX_DECODER = LexiconDecoder(lexicon_path=str(lex_file), tokenizer=phoneme_tokenizer)
        lex_decoder = _GLOBAL_LEX_DECODER
    except Exception:
        lex_decoder = None

    sig = inspect.signature(model.decode_greedy) if hasattr(model, "decode_greedy") else None
    has_penalty_param = sig is not None and "blank_penalty" in sig.parameters

    resamplers = {}
    for idx, s in enumerate(samples):
        speech_np, sr = sf.read(s["audio_path"])
        if sr != 16000:
            if sr not in resamplers:
                resamplers[sr] = T.Resample(sr, 16000)
            t_audio = torch.tensor(speech_np, dtype=torch.float32).unsqueeze(0)
            speech_tensor = resamplers[sr](t_audio).squeeze(0).to(device)
        else:
            speech_tensor = torch.tensor(speech_np, dtype=torch.float32).to(device)

        if speech_tensor.ndim == 1:
            speech_tensor = speech_tensor.unsqueeze(0)

        ref_text = s.get("transcript") or s.get("text", "")
        # Ground truth phonemes via eSpeak (standard LibriSpeech phonemization)
        ref_phonemes = runner.phonemize_text(ref_text)

        with torch.no_grad():
            if hasattr(model, "decode_greedy"):
                if has_penalty_param:
                    decoded_ids = model.decode_greedy(speech_tensor, blank_penalty=blank_penalty)[0]
                else:
                    decoded_ids = model.decode_greedy(speech_tensor)[0]
            elif hasattr(model, "decode_beam") and getattr(getattr(model, "config", None), "beam_width", 0) > 0:
                decoded_ids = model.decode_beam(speech_tensor)[0]
            else:
                out = model(audio=speech_tensor)
                decoded_ids = out["logits"].argmax(dim=-1)[0].tolist()
            pred_phonemes = phoneme_tokenizer.decode(decoded_ids, skip_special=True)
            if idx == 0:
                sample_pred = pred_phonemes

        # Acoustic phoneme evaluation: ignore non-acoustic diacritics, blanks, and whitespace
        filter_ids = {
            phoneme_tokenizer.pad_id,
            phoneme_tokenizer.blank_id,
            phoneme_tokenizer.silence_id,
            phoneme_tokenizer.noise_id,
            phoneme_tokenizer.eos_id,
            phoneme_tokenizer.unk_id,
        }
        for ch in ("ˈ", "ˌ", "ː", "ˑ", " ", "-", "\n", "\t"):
            if ch in phoneme_tokenizer.token_to_id:
                filter_ids.add(phoneme_tokenizer.token_to_id[ch])

        clean_ref_ids = [tid for tid in phoneme_tokenizer.encode(ref_phonemes) if tid not in filter_ids]
        clean_pred_ids = [tid for tid in decoded_ids if tid not in filter_ids]

        import editdistance
        if len(clean_ref_ids) > 0:
            dist = editdistance.eval(clean_ref_ids, clean_pred_ids)
            p = round(float(dist / len(clean_ref_ids)), 4)
        else:
            p = 1.0 if len(clean_pred_ids) > 0 else 0.0

        ref_str_clean = phoneme_tokenizer.decode(clean_ref_ids)
        pred_str_clean = phoneme_tokenizer.decode(clean_pred_ids)
        c = round(float(jiwer.cer(ref_str_clean, pred_str_clean)), 4) if ref_str_clean else (1.0 if pred_str_clean else 0.0)
        pers.append(p)
        cers.append(c)

        # Lexicon-constrained evaluation on first 10 samples (when enabled)
        if compute_lexicon and lex_decoder is not None and idx < 10:
            try:
                with torch.no_grad():
                    out_lex = model(audio=speech_tensor)
                    lp_single = out_lex["logits"][0]
                    lex_res = lex_decoder.decode_utterance(lp_single, beam_width=16, word_bonus=-3.0)
                    clean_lex_ids = [tid for tid in lex_res["phoneme_ids"] if tid not in filter_ids]
                    if len(clean_ref_ids) > 0:
                        dist_lex = editdistance.eval(clean_ref_ids, clean_lex_ids)
                        lex_pers.append(round(float(dist_lex / len(clean_ref_ids)), 4))
                    w_err = jiwer.wer(ref_text.lower(), lex_res["text"].lower())
                    lex_wers.append(w_err)
                    if idx == 0:
                        sample_lex_pred = lex_res["text"]
            except Exception:
                pass

    avg_per = round(float(sum(pers) / max(1, len(pers))) * 100.0, 2)
    avg_cer = round(float(sum(cers) / max(1, len(cers))) * 100.0, 2)
    avg_lex_per = round(float(sum(lex_pers) / max(1, len(lex_pers))) * 100.0, 2) if lex_pers else None
    avg_lex_wer = round(float(sum(lex_wers) / max(1, len(lex_wers))) * 100.0, 2) if lex_wers else None

    return {
        "wer": avg_per,
        "cer": avg_cer,
        "per": avg_per,
        "sample_pred": sample_pred,
        "lexicon_per": avg_lex_per,
        "lexicon_wer": avg_lex_wer,
        "sample_lex_pred": sample_lex_pred,
    }

def set_feature_extractor_grad(model: nn.Module, enabled: bool):
    """Enable or disable gradients for the temporal 1D CNN feature encoder."""
    if hasattr(model, "feature_extractor"):
        for p in model.feature_extractor.parameters():
            p.requires_grad = enabled


def get_scheduled_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int, min_lr: float = 1e-5) -> float:
    """Compute learning rate with linear warmup and cosine decay bounded by min_lr."""
    if warmup_steps > 0 and step <= warmup_steps:
        return max(min_lr, base_lr * (step / float(warmup_steps)))
    if warmup_steps > 0:
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr


def main():
    parser = argparse.ArgumentParser(description="Unified Modular Training for Speech Models.")
    parser.add_argument("--arch", type=str, default="phono_hubert", help="Model architecture ID")
    parser.add_argument("--tier", type=str, default="mini", choices=["mini", "small", "medium", "base"], help="Model size tier")
    parser.add_argument("--override", type=str, default="", help="Custom parameter overrides (e.g. 'mask_prob=0.7,encoder_layers=6')")
    parser.add_argument("--steps", type=int, default=625, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (utterances per step)")
    parser.add_argument("--num_workers", type=int, default=4, help="Parallel speech synthesis worker threads")
    parser.add_argument("--buffer_size", type=int, default=20, help="Max batch buffer queue size")
    parser.add_argument("--watermark", type=int, default=10, help="Low watermark batch threshold to resume synthesis")
    parser.add_argument("--use_rolling_pool", action="store_true", default=True, help="Enable dynamic in-RAM utterance pool")
    parser.add_argument("--no_rolling_pool", dest="use_rolling_pool", action="store_false", help="Disable rolling pool (direct queue mode)")
    parser.add_argument("--pool_size", type=int, default=250, help="In-RAM utterance pool capacity")
    parser.add_argument("--eval_interval", type=int, default=125, help="Benchmark evaluation interval")
    parser.add_argument("--probe_steps", type=int, default=25, help="Downstream CTC probe steps")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate floor for cosine schedule")
    parser.add_argument("--blank_penalty", type=float, default=0.0, help="Evaluation blank logit penalty subtraction")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Linear learning rate warmup steps")
    parser.add_argument("--freeze_cnn_steps", type=int, default=200, help="Steps to freeze temporal CNN feature extractor")
    parser.add_argument("--masking_mode", type=str, default=None, choices=["none", "span", "specaugment", "dual"], help="Acoustic masking strategy (none, span, specaugment, dual)")
    parser.add_argument("--mask_prob", type=float, default=None, help="Probability of acoustic masking")
    parser.add_argument("--mask_length", type=int, default=None, help="Span mask length in 20ms frames")
    parser.add_argument("--save_interval", type=int, default=4000, help="Checkpoint interval (default 4000 to minimize SSD writes)")
    parser.add_argument("--val_manifest", type=str, default="data/librispeech/benchmark_val.json", help="Path to held-out validation JSON manifest")
    parser.add_argument("--test_manifest", type=str, default="data/librispeech/librispeech_test_clean.json", help="Path to LibriSpeech test-clean benchmark JSON manifest")
    parser.add_argument("--val_samples", type=int, default=50, help="Number of validation samples evaluated every eval_interval")
    parser.add_argument("--test_samples", type=int, default=100, help="Number of test-clean samples evaluated at the end of training")
    parser.add_argument("--only_save_best", action="store_true", default=False, help="Only save the single best checkpoint (protects SSD wear and eliminates intermediate checkpoints)")
    parser.add_argument("--real_speech_manifest", type=str, default=None, help="Path to real speech JSON manifest for pre-training")
    parser.add_argument("--real_ratio", type=float, default=1.0, help="Ratio of real human speech in streaming (0.0=all synthetic, 1.0=all real LibriSpeech)")
    parser.add_argument("--min_duration_sec", type=float, default=1.0, help="Minimum utterance duration in seconds")
    parser.add_argument("--max_duration_sec", type=float, default=30.0, help="Maximum utterance duration in seconds")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run_name", type=str, default=None, help="Name of this training run (default: auto-incremented run_1, run_2, ...)")
    parser.add_argument("--resume", nargs="?", const="auto", default=None, help="Resume training")
    parser.add_argument("--warm_start", type=str, default=None, help="Path to checkpoint from which to initialize model weights (starts from step 1)")
    args = parser.parse_args()

    # V6.3 Diffusion Refiner Specific Tuning:
    # Use lr=5e-5 for latent diffusion stability and auto warm-start from V6.2 Sparse backbone if available
    if args.arch == "phono_v6_3_diffusion":
        if args.lr == 0.0003:
            args.lr = 5e-5
            print(f"🎯 [V6.3 Hyperparameter Tuning] Calibrated learning rate to 5e-5 for latent diffusion stability.")
        if args.warm_start is None:
            v6_2_candidates = [
                Path(f"checkpoints/phono_v6_2_sparse/{args.tier}/bench_phono_v6_2_sparse_{args.tier}/best_checkpoint.pt"),
                Path(f"checkpoints/phono_v6_2_sparse/{args.tier}/v6_2_sparse_100h_run1/checkpoint_step_4000.pt"),
                Path(f"checkpoints/phono_v6_2_sparse/{args.tier}/best_checkpoint.pt"),
            ]
            for cand in v6_2_candidates:
                if cand.exists():
                    args.warm_start = str(cand)
                    print(f"🔥 [V6.3 Auto Warm-Start] Warm-starting acoustic backbone from V6.2 checkpoint: {cand}")
                    break

    device = torch.device(args.device)
    overrides = parse_overrides(args.override)

    if args.masking_mode is not None:
        overrides["masking_mode"] = args.masking_mode
    if args.mask_prob is not None:
        overrides["mask_prob"] = args.mask_prob
    if args.mask_length is not None:
        overrides["mask_length"] = args.mask_length

    # 1. Build Config & Model via Registry
    config = ModelRegistry.build_config(args.arch, tier=args.tier, **overrides)
    target_extractor = ModelRegistry.build_target_extractor(args.arch)
    model = ModelRegistry.build_model(args.arch, tier=args.tier, config=config).to(device)

    if args.warm_start is not None:
        p = Path(args.warm_start)
        if p.exists():
            payload = torch.load(p, map_location=device, weights_only=False)
            model.load_state_dict(payload["model_state_dict"], strict=False)
            print(f"🔥 [Warm-Start] Loaded model weights from: {p} (starting fresh from Step 1)")
        else:
            print(f"⚠️ [Warm-Start] Checkpoint not found at: {p}")

    num_params = sum(p.numel() for p in model.parameters())

    print("=" * 75)
    print(f"🚀 AUDIOLEARN MODULAR PRE-TRAINING: {args.arch.upper()} [{args.tier.upper()}]")
    print("=" * 75)
    print(f"Architecture: {args.arch} | Tier: {args.tier} | Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"Device: {device} | Total Steps: {args.steps} | Batch Size: {args.batch_size}")
    print(f"Workers: {args.num_workers} | Buffer Capacity: {args.buffer_size} (Watermark: {args.watermark})")
    print(f"Target Type: {target_extractor.target_type} | Vocab Size: {target_extractor.vocab_size}")
    if overrides:
        print(f"Variable Parameter Overrides: {overrides}")
    print("-" * 75)

    # 2. Setup RunManager, Isolated Run Dirs, and Save Full Training Configuration
    run_mgr = RunManager(
        arch=args.arch,
        tier=args.tier,
        run_name=args.run_name,
        is_resume=(args.resume is not None),
    )

    full_config = RunManager.assemble_full_config(
        arch=args.arch,
        tier=args.tier,
        run_name=run_mgr.run_name,
        model=model,
        model_config=config,
        target_extractor=target_extractor,
        args=args,
    )
    saved_config_path = run_mgr.init_run(full_config)

    print(f"🏷️  Run Identifier: {run_mgr.run_name}")
    print(f"📄 Full Config Saved: {saved_config_path}")
    print(f"💾 Checkpoints Dir: {run_mgr.run_ckpt_dir}")
    print(f"📊 Telemetry Dir: {run_mgr.run_log_dir}")
    print("-" * 75)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # 3. Setup Threaded Data Pipeline & Microsecond Profiler
    profiler = StepProfiler(window_size=20)
    voice_manager = PiperVoiceManager()
    text_sampler = ProceduralTextSampler()

    # Resolve hybrid real speech dataset if enabled or requested
    real_manifest = args.real_speech_manifest
    if not real_manifest:
        if Path("data/librispeech/benchmark_train.json").exists():
            real_manifest = "data/librispeech/benchmark_train.json"
        elif (
            getattr(config, "hybrid_training", False)
            or args.arch in ("phono_v3_hybrid", "phono_v4_scaled", "phono_v5_beam")
            or args.arch.startswith("phono_v6")
        ):
            real_manifest = getattr(config, "librispeech_manifest", "data/librispeech/librispeech_train.json")

    # For v6.2+ variants, default to 100% human speech if real_ratio wasn't explicitly changed
    if args.arch in ("phono_v6_2_sparse", "phono_v6_3_diffusion") and args.real_ratio == 0.5:
        args.real_ratio = 1.0

    # For 100% human speech datasets, disable rolling pool replacement to eliminate worker CPU/GIL churn
    use_pool = args.use_rolling_pool if args.real_ratio < 1.0 else False

    batch_generator = BufferedSpeechBatchGenerator(
        voice_manager=voice_manager,
        text_sampler=text_sampler,
        target_extractor=target_extractor,
        batch_size=args.batch_size,
        max_buffer_size=args.buffer_size,
        low_watermark=args.watermark,
        num_workers=args.num_workers,
        use_rolling_pool=use_pool,
        pool_capacity=args.pool_size,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
        real_speech_manifest=real_manifest,
        real_ratio=args.real_ratio,
        profiler=profiler,
    )

    tokenizer = CharacterTokenizer()
    phoneme_tokenizer = PhonemeTokenizer()
    history = []
    start_step = 1
    total_audio_sec = 0.0

    # 4. Checkpoint Resumption (Run-Aware Auto-detect or Fresh Start)
    if args.resume is not None:
        ckpt_to_load = run_mgr.find_resume_checkpoint(args.resume)

        if ckpt_to_load and ckpt_to_load.exists():
            print(f"🔄 [Resume] Loading existing checkpoint from: {ckpt_to_load}")
            payload = torch.load(ckpt_to_load, map_location=device, weights_only=False)
            model.load_state_dict(payload["model_state_dict"])
            if "optimizer_state_dict" in payload:
                try:
                    optimizer.load_state_dict(payload["optimizer_state_dict"])
                except Exception as e:
                    print(f"⚠️ [Resume] Optimizer state could not be restored: {e}")
            if "scaler_state_dict" in payload and scaler.is_enabled():
                try:
                    scaler.load_state_dict(payload["scaler_state_dict"])
                except Exception as e:
                    print(f"⚠️ [Resume] Scaler state could not be restored: {e}")
            start_step = payload.get("step", 0) + 1
            total_audio_sec = payload.get("total_audio_sec", 0.0)
            history = payload.get("history", [])
            print(f"✅ [Resume] Resumed successfully! Starting at Step {start_step} (Cumulative Audio: {total_audio_sec/3600.0:.3f}h)")
        else:
            print(f"ℹ️ [Resume] No previous checkpoint found for run '{run_mgr.run_name}'. Starting fresh from Step 1.")

    if not history:
        run_history_file = run_mgr.run_log_dir / "history.json"
        if run_history_file.exists():
            try:
                with open(run_history_file, "r", encoding="utf-8") as f:
                    history = json.load(f).get("history", [])
            except Exception:
                pass
        elif run_mgr.tier_history_file.exists():
            try:
                with open(run_mgr.tier_history_file, "r", encoding="utf-8") as f:
                    history = json.load(f).get("history", [])
            except Exception:
                pass

    step_history = []
    run_step_history_file = run_mgr.run_log_dir / "step_history.json"
    if run_step_history_file.exists():
        try:
            with open(run_step_history_file, "r", encoding="utf-8") as f:
                step_history = json.load(f)
        except Exception:
            pass
    elif run_mgr.tier_step_history_file.exists():
        try:
            with open(run_mgr.tier_step_history_file, "r", encoding="utf-8") as f:
                step_history = json.load(f)
        except Exception:
            pass

    if args.freeze_cnn_steps > 0:
        if start_step <= args.freeze_cnn_steps:
            set_feature_extractor_grad(model, False)
            print(f"❄️  [CNN Frozen] Temporal 1D feature encoder frozen for steps {start_step}..{args.freeze_cnn_steps}")
        else:
            set_feature_extractor_grad(model, True)

    recent_step_durations = collections.deque(maxlen=20)
    best_val_per = float("inf")
    best_val_step = 0
    best_model_state_dict = None
    print(f"\n[Ready] Starting decoupled threaded streaming loop (Step {start_step} -> {args.steps})...\n")

    try:
        for step in range(start_step, args.steps + 1):
            t_step_start = time.perf_counter()

            # Dynamic LR schedule: linear warmup + cosine decay
            lr_current = get_scheduled_lr(step, args.steps, args.lr, args.warmup_steps, min_lr=args.min_lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_current

            # Unfreeze CNN feature encoder when freeze threshold is reached
            if args.freeze_cnn_steps > 0 and step == args.freeze_cnn_steps + 1:
                set_feature_extractor_grad(model, True)
                print(f"\n🔥 [CNN Unfrozen] Temporal 1D feature encoder unfrozen at step {step}!\n")

            batch = batch_generator.get_batch(timeout=60.0)

            t_train_start = time.perf_counter()

            with profiler.time_block("time_device_transfer"):
                audio = batch["audio"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
                target_lengths = batch["target_lengths"].to(device, non_blocking=True)
                audio_lengths = batch["audio_lengths"].to(device, non_blocking=True)
                frame_targets = batch["frame_targets"].to(device, non_blocking=True) if "frame_targets" in batch and batch["frame_targets"] is not None else None
                frame_lengths = batch["frame_lengths"].to(device, non_blocking=True) if "frame_lengths" in batch and batch["frame_lengths"] is not None else None

            batch_dur = sum(batch["durations"])
            total_audio_sec += batch_dur

            optimizer.zero_grad()
            with profiler.time_block("time_forward"):
                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
                    if args.arch.startswith("phono_"):
                        out = model(
                            audio=audio,
                            targets=targets,
                            target_lengths=target_lengths,
                            audio_lengths=audio_lengths,
                            frame_targets=frame_targets,
                            frame_lengths=frame_lengths,
                        )
                    else:
                        out = model(audio=audio, target_clusters=targets)

            with profiler.time_block("time_loss"):
                loss = out["loss"]
                acc = out["accuracy"]

            with profiler.time_block("time_backward"):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            with profiler.time_block("time_optimizer_step"):
                scaler.step(optimizer)
                scaler.update()

            t_train_total = time.perf_counter() - t_train_start
            profiler.record("total_train_step_sec", t_train_total)

            step_elapsed = time.perf_counter() - t_step_start
            profiler.record("total_step_sec", step_elapsed)
            recent_step_durations.append(step_elapsed)

            loss_val = round(float(loss.item()), 4)
            acc_val = round(float(acc.item()), 2) if acc is not None else 0.0
            inter_loss_val = round(float(out.get("inter_ctc_loss", 0.0)), 4)
            diff_loss_val = round(float(out.get("diff_loss", 0.0)), 4)
            ref_ctc_loss_val = round(float(out.get("refined_ctc_loss", 0.0)), 4)

            avg_step_sec = sum(recent_step_durations) / len(recent_step_durations)
            rem_steps = max(0, args.steps - step)
            rem_evals = max(0, rem_steps // args.eval_interval)
            eta_seconds = (rem_steps * avg_step_sec) + (rem_evals * 35.0)
            hours = total_audio_sec / 3600.0

            # Export Live Status for Website & UI
            live_status = {
                "is_running": True,
                "arch": args.arch,
                "tier": args.tier,
                "step": step,
                "total_steps": args.steps,
                "progress_pct": round((step / args.steps) * 100.0, 1),
                "learning_rate": lr_current,
                "loss": loss_val,
                "inter_ctc_loss": inter_loss_val,
                "diff_loss": diff_loss_val,
                "refined_ctc_loss": ref_ctc_loss_val,
                "masked_accuracy_pct": acc_val,
                "cumulative_audio_sec": round(total_audio_sec, 1),
                "cumulative_audio_hours": round(hours, 4),
                "avg_step_sec": round(avg_step_sec, 3),
                "eta_seconds": int(eta_seconds),
                "eta_formatted": f"{int(eta_seconds//60)}m {int(eta_seconds%60):02d}s",
                "active_voices": batch["voices"][:2],
                "active_voice": ", ".join(batch["voices"][:2]),
                "buffer_occupancy": batch_generator.buffer_occupancy,
                "buffer_capacity": args.buffer_size,
                "buffer_watermark": args.watermark,
                "pool_size": batch_generator.pool_size,
                "profiler": profiler.get_summary(),
                "disk_bytes_used": 0,
                "last_heartbeat": time.time(),
            }
            run_mgr.update_status(live_status)

            step_record = {
                "step": step,
                "learning_rate": lr_current,
                "loss": loss_val,
                "inter_ctc_loss": inter_loss_val,
                "diff_loss": diff_loss_val,
                "refined_ctc_loss": ref_ctc_loss_val,
                "accuracy": acc_val,
                "masked_accuracy_pct": acc_val,
                "cumulative_audio_sec": round(total_audio_sec, 1),
                "cumulative_audio_hours": round(hours, 4),
                "disk_bytes_used": 0,
                "active_voices": batch["voices"],
                "active_voice": ", ".join(batch["voices"][:2]),
            }
            step_history.append(step_record)
            if step % 50 == 0 or step == args.steps:
                run_mgr.update_step_history(step_history)

            if step % 5 == 0 or step == start_step:
                inter_str = f" | InterCTC: {inter_loss_val:.3f}" if inter_loss_val > 0 else ""
                diff_str = f" | Diff: {diff_loss_val:.3f}" if diff_loss_val > 0 else ""
                ref_str = f" | RefCTC: {ref_ctc_loss_val:.3f}" if ref_ctc_loss_val > 0 else ""
                print(f"[{args.arch.upper()}] Step {step:4d}/{args.steps} | LR: {lr_current:.2e} | Loss: {loss_val:.4f}{inter_str}{diff_str}{ref_str} | Acc: {acc_val:5.1f}% | "
                      f"Step: {step_elapsed:.3f}s (GPU: {t_train_total:.3f}s) | Buffer: {batch_generator.buffer_occupancy}/{args.buffer_size} | "
                      f"Audio: {total_audio_sec:6.1f}s ({hours:.3f}h) | ETA: {live_status['eta_formatted']}")

            if step % 50 == 0 or step == 5:
                print("\n" + profiler.format_console_breakdown(batch_generator.buffer_occupancy, args.buffer_size) + "\n")

            # Milestone Validation Evaluation (Every eval_interval or at final step)
            if step % args.eval_interval == 0 or step == args.steps:
                val_manifest_path = Path(args.val_manifest)
                manifest_to_eval = val_manifest_path if val_manifest_path.exists() else Path("data/librispeech/librispeech_test_clean.json")
                is_eval_val = (manifest_to_eval == val_manifest_path)
                eval_tag = "Validation Set" if is_eval_val else "Test-Clean (Fallback)"

                print(f"\n--- [Step {step}] Evaluating on {eval_tag} ({manifest_to_eval}) ---")
                if target_extractor.target_type == "phoneme_tokens":
                    bench_res = evaluate_direct_phonemes(
                        model,
                        phoneme_tokenizer,
                        device,
                        manifest_path=manifest_to_eval,
                        num_samples=args.val_samples,
                        blank_penalty=args.blank_penalty,
                        compute_lexicon=False,
                    )
                else:
                    calibrated_ctc = run_quick_ctc_calibration(model, config, tokenizer, device, probe_steps=args.probe_steps)
                    bench_res = evaluate_on_benchmark(calibrated_ctc, tokenizer, device, num_samples=args.val_samples)

                val_per = bench_res["per"]
                print(f"📊 Step {step} {eval_tag} -> PER: {val_per:.2f}% | CER: {bench_res['cer']:.2f}%")

                is_new_best = val_per < best_val_per
                if is_new_best:
                    best_val_per = val_per
                    best_val_step = step
                    best_model_state_dict = copy.deepcopy(model.state_dict())
                    best_ckpt_path = run_mgr.run_ckpt_dir / "best_checkpoint.pt"
                    best_save_payload = {
                        "step": step,
                        "arch": args.arch,
                        "tier": args.tier,
                        "run_name": run_mgr.run_name,
                        "config": config,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "val_per": best_val_per,
                        "total_audio_sec": total_audio_sec,
                        "history": history,
                        "saved_at": time.time(),
                    }
                    try:
                        torch.save(best_save_payload, best_ckpt_path)
                        print(f"🌟 [New Best Model Saved] Step {step} achieved lowest Val PER: {best_val_per:.2f}% -> {best_ckpt_path.name}")
                    except Exception as e:
                        print(f"⚠️ Failed to save best checkpoint: {e}")

                entry = {
                    "step": step,
                    "cumulative_audio_hours": round(hours, 4),
                    "pretrain_loss": loss_val,
                    "masked_acc_pct": acc_val,
                    "val_per": val_per,
                    "val_cer": bench_res["cer"],
                    "best_val_per": best_val_per,
                    "best_val_step": best_val_step,
                    "is_best_val": is_new_best,
                    "sample_prediction": bench_res["sample_pred"],
                }
                history.append(entry)
                run_mgr.update_history(history)

            # Save final checkpoint with optimizer state for future resumption (periodic checkpoints skipped if only_save_best)
            if (not args.only_save_best and step % args.save_interval == 0) or (step == args.steps):
                ckpt_path, latest_path = run_mgr.get_checkpoint_paths(step)
                save_payload = {
                    "step": step,
                    "arch": args.arch,
                    "tier": args.tier,
                    "run_name": run_mgr.run_name,
                    "config": config,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "total_audio_sec": total_audio_sec,
                    "history": history,
                    "saved_at": time.time(),
                }
                try:
                    torch.save(save_payload, latest_path)
                    if step == args.steps or step % 4000 == 0:
                        torch.save(save_payload, ckpt_path)
                    if step == args.steps:
                        print(f"💾 [Final Step Checkpoint Saved] Saved Step {step} weights, optimizer & scaler states -> {ckpt_path.name}")
                    run_mgr.on_checkpoint_saved(
                        step=step,
                        loss=loss_val,
                        acc=acc_val,
                        per=history[-1].get("val_per") if history else None,
                        audio_hours=round(hours, 4),
                    )
                except Exception as e:
                    print(f"[Warning] Failed to save pretrain checkpoint: {e}")

    except KeyboardInterrupt:
        print(f"\n⚠️  [Interrupt] Run '{run_mgr.run_name}' interrupted by user (Ctrl+C).")
        if "step" in locals() and "model" in locals() and step > 0:
            try:
                ckpt_path, latest_path = run_mgr.get_checkpoint_paths(step)
                save_payload = {
                    "step": step,
                    "arch": args.arch,
                    "tier": args.tier,
                    "run_name": run_mgr.run_name,
                    "config": config,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "total_audio_sec": total_audio_sec,
                    "history": history,
                    "saved_at": time.time(),
                }
                print(f"💾 [Ctrl+C Save] Saving emergency checkpoint at step {step} to: {latest_path} ...")
                torch.save(save_payload, latest_path)
                print(f"✅ [Ctrl+C Save] Emergency resume checkpoint saved successfully ({latest_path})!")
                run_mgr.on_checkpoint_saved(
                    step=step,
                    loss=loss_val if "loss_val" in locals() else None,
                    acc=acc_val if "acc_val" in locals() else None,
                    per=history[-1].get("librispeech_per") if history else None,
                    audio_hours=round(total_audio_sec / 3600.0, 4),
                )
            except Exception as e:
                print(f"⚠️ [Interrupt Save Error] Failed to save checkpoint on interrupt: {e}")
        run_mgr.finish_run(
            status="interrupted",
            final_metrics={
                "step": step if "step" in locals() else 0,
                "loss": loss_val if "loss_val" in locals() else None,
                "accuracy": acc_val if "acc_val" in locals() else None,
                "cumulative_audio_hours": round(total_audio_sec / 3600.0, 4),
            },
        )
        raise
    except Exception as e:
        is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
        err_type = "CUDA OOM" if is_oom else type(e).__name__
        print(f"\n💥 [Run Failed: {err_type}] Architecture {args.arch} [{args.tier}] failed at Step {step if 'step' in locals() else 0}: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        summary_path = run_mgr.run_log_dir / "benchmark_summary.json"
        summary_data = {
            "arch": args.arch,
            "tier": args.tier,
            "total_steps": args.steps,
            "status": "failed",
            "error_type": err_type,
            "error_message": str(e),
            "failed_at_step": step if "step" in locals() else 0,
            "best_val_step": best_val_step if "best_val_step" in locals() else 0,
            "best_val_per": best_val_per if "best_val_per" in locals() and best_val_per != float("inf") else None,
            "test_clean_per": None,
            "test_clean_cer": None,
            "test_clean_lexicon_per": None,
            "test_clean_lexicon_wer": None,
            "cumulative_audio_hours": round(total_audio_sec / 3600.0, 4) if "total_audio_sec" in locals() else 0.0,
            "completed_at": time.time(),
        }
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2)
        except Exception:
            pass

        run_mgr.finish_run(
            status="failed",
            final_metrics={
                "step": step if "step" in locals() else 0,
                "error": str(e),
                "error_type": err_type,
            },
        )
        sys.exit(2 if is_oom else 1)
    finally:
        batch_generator.stop()

    # Final Unbiased Evaluation on LibriSpeech Test-Clean using the BEST validation model
    test_manifest_path = Path(args.test_manifest)
    final_test_res = {}
    if test_manifest_path.exists():
        print(f"\n=======================================================")
        print(f"🏆 FINAL UNBIASED EVALUATION ON LIBRISPEECH TEST-CLEAN")
        print(f"   Architecture: {args.arch} [{args.tier}]")
        print(f"   Restoring best validation model from Step {best_val_step} (Lowest Val PER: {best_val_per:.2f}%)")
        print(f"=======================================================")
        if best_model_state_dict is not None:
            model.load_state_dict(best_model_state_dict)
        elif (run_mgr.run_ckpt_dir / "best_checkpoint.pt").exists():
            payload = torch.load(run_mgr.run_ckpt_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
            model.load_state_dict(payload["model_state_dict"])

        if target_extractor.target_type == "phoneme_tokens":
            final_test_res = evaluate_direct_phonemes(
                model,
                phoneme_tokenizer,
                device,
                manifest_path=test_manifest_path,
                num_samples=args.test_samples,
                blank_penalty=args.blank_penalty,
                compute_lexicon=True,
            )
        else:
            calibrated_ctc = run_quick_ctc_calibration(model, config, tokenizer, device, probe_steps=args.probe_steps)
            final_test_res = evaluate_on_benchmark(calibrated_ctc, tokenizer, device, num_samples=args.test_samples)

        print(f"🎯 Final Test-Clean Scores (Best Model from Step {best_val_step}):")
        print(f"   • Test PER (Greedy CTC)   : {final_test_res['per']:.2f}%")
        print(f"   • Test CER                : {final_test_res['cer']:.2f}%")
        if final_test_res.get("lexicon_per") is not None:
            print(f"   • Test Lexicon PER        : {final_test_res['lexicon_per']:.2f}%")
            print(f"   • Test Lexicon WER        : {final_test_res.get('lexicon_wer', 0.0):.2f}%")
            print(f"   • Sample Lexicon Decode   : {final_test_res.get('sample_lex_pred', '')[:65]}...")
        print(f"=======================================================\n")

        summary_path = run_mgr.run_log_dir / "benchmark_summary.json"
        summary_data = {
            "arch": args.arch,
            "tier": args.tier,
            "total_steps": args.steps,
            "best_val_step": best_val_step,
            "best_val_per": best_val_per,
            "test_clean_per": final_test_res["per"],
            "test_clean_cer": final_test_res["cer"],
            "test_clean_lexicon_per": final_test_res.get("lexicon_per"),
            "test_clean_lexicon_wer": final_test_res.get("lexicon_wer"),
            "sample_prediction": final_test_res.get("sample_pred"),
            "sample_lex_prediction": final_test_res.get("sample_lex_pred"),
            "cumulative_audio_hours": round(total_audio_sec / 3600.0, 4),
            "completed_at": time.time(),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

    # Mark run as finished
    live_status["is_running"] = False
    live_status["status"] = "completed"
    run_mgr.update_status(live_status)
    run_mgr.finish_run(
        status="completed",
        final_metrics={
            "step": args.steps,
            "loss": loss_val,
            "accuracy": acc_val,
            "cumulative_audio_hours": round(hours, 4),
            "best_val_per": best_val_per,
            "best_val_step": best_val_step,
            "test_clean_per": final_test_res.get("per"),
            "test_clean_lexicon_per": final_test_res.get("lexicon_per"),
            "test_clean_lexicon_wer": final_test_res.get("lexicon_wer"),
        },
    )
    print(f"\n[Completed] Pre-training run '{run_mgr.run_name}' finished for {args.arch} [{args.tier}] at step {args.steps}!")


if __name__ == "__main__":
    main()
