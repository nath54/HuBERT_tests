#!/usr/bin/env python3
"""Unified Modular Training Launcher for AudioLearn Speech Architectures.

Supports all registered model architectures (HuBERT K-Means, PhonoHuBERT Direct Phonemes, etc.)
and all parameter tiers (mini, small, medium, base) with variable hyperparameter overrides.
"""

import argparse
import collections
from datetime import datetime
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.config import HuBERTConfig
from src.models.hubert_asr import HuBERTForCTC
from src.models.registry import ModelRegistry, STANDARD_TIERS
from src.data.tokenizer import CharacterTokenizer
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
    parser.add_argument("--save_interval", type=int, default=25, help="Checkpoint interval")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", nargs="?", const="auto", default=None, help="Resume training")
    args = parser.parse_args()

    device = torch.device(args.device)
    overrides = parse_overrides(args.override)

    # 1. Build Config & Model via Registry
    config = ModelRegistry.build_config(args.arch, tier=args.tier, **overrides)
    target_extractor = ModelRegistry.build_target_extractor(args.arch)
    model = ModelRegistry.build_model(args.arch, tier=args.tier, config=config).to(device)

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

    # Output paths
    ckpt_dir = Path("checkpoints") / args.arch / args.tier
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_file = ckpt_dir / "latest_checkpoint.pt"
    status_file = Path("logs") / f"{args.arch}_status_live.json"
    tier_status_file = Path("logs") / f"{args.arch}_{args.tier}_status_live.json"
    history_file = Path("logs") / f"{args.arch}_{args.tier}_history.json"
    step_history_file = Path("logs") / f"{args.arch}_{args.tier}_step_history.json"

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # 2. Setup Threaded Data Pipeline & Microsecond Profiler
    profiler = StepProfiler(window_size=20)
    voice_manager = PiperVoiceManager()
    text_sampler = ProceduralTextSampler()

    batch_generator = BufferedSpeechBatchGenerator(
        voice_manager=voice_manager,
        text_sampler=text_sampler,
        target_extractor=target_extractor,
        batch_size=args.batch_size,
        max_buffer_size=args.buffer_size,
        low_watermark=args.watermark,
        num_workers=args.num_workers,
        use_rolling_pool=args.use_rolling_pool,
        pool_capacity=args.pool_size,
        profiler=profiler,
    )

    tokenizer = CharacterTokenizer()
    history = []
    start_step = 1
    total_audio_sec = 0.0

    # 3. Checkpoint Resumption (Auto-detect or Fresh Start)
    if args.resume is not None:
        ckpt_to_load = None
        if args.resume in ("auto", "latest", "", True):
            candidates = [
                ckpt_dir / "checkpoint_latest.pt",
                ckpt_dir / "latest_checkpoint.pt",
            ]
            for cand in candidates:
                if cand.exists():
                    ckpt_to_load = cand
                    break
            if not ckpt_to_load:
                step_ckpts = sorted(list(ckpt_dir.glob("checkpoint_step_*.pt")), key=os.path.getmtime)
                if step_ckpts:
                    ckpt_to_load = step_ckpts[-1]
        else:
            custom_path = Path(args.resume)
            if custom_path.exists():
                ckpt_to_load = custom_path

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
            print(f"ℹ️ [Resume] No previous checkpoint found in '{ckpt_dir}'. Starting fresh from Step 1.")

    if not history and history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f).get("history", [])
        except Exception:
            pass

    step_history = []
    if step_history_file.exists():
        try:
            with open(step_history_file, "r", encoding="utf-8") as f:
                step_history = json.load(f)
        except Exception:
            pass

    recent_step_durations = collections.deque(maxlen=20)
    print(f"\n[Ready] Starting decoupled threaded streaming loop (Step {start_step} -> {args.steps})...\n")

    try:
        for step in range(start_step, args.steps + 1):
            t_step_start = time.perf_counter()
            batch = batch_generator.get_batch(timeout=60.0)

            t_train_start = time.perf_counter()

            with profiler.time_block("time_device_transfer"):
                audio = batch["audio"].to(device)
                targets = batch["targets"].to(device)
                target_lengths = batch["target_lengths"].to(device)
                audio_lengths = batch["audio_lengths"].to(device)

            batch_dur = sum(batch["durations"])
            total_audio_sec += batch_dur

            optimizer.zero_grad()
            with profiler.time_block("time_forward"):
                with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
                    if args.arch == "phono_hubert":
                        out = model(audio=audio, targets=targets, target_lengths=target_lengths, audio_lengths=audio_lengths)
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
                "loss": loss_val,
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
            try:
                with open(status_file, "w", encoding="utf-8") as f:
                    json.dump(live_status, f, indent=2)
                with open(tier_status_file, "w", encoding="utf-8") as f:
                    json.dump(live_status, f, indent=2)
            except Exception:
                pass

            step_record = {
                "step": step,
                "loss": loss_val,
                "accuracy": acc_val,
                "masked_accuracy_pct": acc_val,
                "cumulative_audio_sec": round(total_audio_sec, 1),
                "cumulative_audio_hours": round(hours, 4),
                "disk_bytes_used": 0,
                "active_voices": batch["voices"],
                "active_voice": ", ".join(batch["voices"][:2]),
            }
            step_history.append(step_record)
            if step % 5 == 0 or step == args.steps:
                try:
                    with open(step_history_file, "w", encoding="utf-8") as f:
                        json.dump(step_history, f)
                except Exception:
                    pass

            if step % 5 == 0 or step == start_step:
                print(f"[{args.arch.upper()}] Step {step:4d}/{args.steps} | Loss: {loss_val:.4f} | Acc: {acc_val:5.1f}% | "
                      f"Step: {step_elapsed:.3f}s (GPU: {t_train_total:.3f}s) | Buffer: {batch_generator.buffer_occupancy}/{args.buffer_size} | "
                      f"Audio: {total_audio_sec:6.1f}s ({hours:.3f}h) | ETA: {live_status['eta_formatted']}")

            if step % 25 == 0 or step == 5:
                print("\n" + profiler.format_console_breakdown(batch_generator.buffer_occupancy, args.buffer_size) + "\n")

            # Milestone Benchmark Evaluation (Inside Loop)
            if step % args.eval_interval == 0 or step == args.steps:
                print(f"\n--- [Milestone Step {step}] Downstream LibriSpeech Evaluation ---")
                calibrated_ctc = run_quick_ctc_calibration(model, config, tokenizer, device, probe_steps=args.probe_steps)
                bench_res = evaluate_on_benchmark(calibrated_ctc, tokenizer, device, num_samples=20)
                print(f"🏆 Milestone Step {step} -> WER: {bench_res['wer']}% | CER: {bench_res['cer']}% | PER: {bench_res['per']}%")

                entry = {
                    "step": step,
                    "cumulative_audio_hours": round(hours, 4),
                    "pretrain_loss": loss_val,
                    "masked_acc_pct": acc_val,
                    "librispeech_wer": bench_res["wer"],
                    "librispeech_cer": bench_res["cer"],
                    "librispeech_per": bench_res["per"],
                    "sample_prediction": bench_res["sample_pred"],
                }
                history.append(entry)
                try:
                    with open(history_file, "w", encoding="utf-8") as f:
                        json.dump({"arch": args.arch, "tier": args.tier, "history": history}, f, indent=2)
                except Exception:
                    pass

            # Save model checkpoint (Inside Loop)
            if step % args.save_interval == 0 or step == args.steps:
                ckpt_path = ckpt_dir / f"checkpoint_step_{step}.pt"
                latest_path = ckpt_dir / "checkpoint_latest.pt"
                save_payload = {
                    "step": step,
                    "arch": args.arch,
                    "tier": args.tier,
                    "config": config,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "total_audio_sec": total_audio_sec,
                    "history": history,
                    "saved_at": time.time(),
                }
                try:
                    torch.save(save_payload, ckpt_path)
                    torch.save(save_payload, latest_path)
                except Exception as e:
                    print(f"[Warning] Failed to save pretrain checkpoint: {e}")

    finally:
        batch_generator.stop()

    # Mark run as finished
    live_status["is_running"] = False
    live_status["status"] = "completed"
    try:
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(live_status, f, indent=2)
    except Exception:
        pass
    print(f"\n[Completed] Pre-training finished for {args.arch} [{args.tier}] at step {args.steps}!")


if __name__ == "__main__":
    main()
