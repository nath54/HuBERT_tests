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
    print(f"Target Type: {target_extractor.target_type} | Vocab Size: {target_extractor.vocab_size}")
    if overrides:
        print(f"Variable Parameter Overrides: {overrides}")
    print("-" * 75)

    # Output paths
    ckpt_dir = Path("checkpoints") / args.arch / args.tier
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_file = ckpt_dir / "latest_checkpoint.pt"
    status_file = Path("logs") / f"{args.arch}_status_live.json"
    history_file = Path("logs") / f"{args.arch}_{args.tier}_history.json"

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # 2. Setup Data Pipeline
    voice_manager = PiperVoiceManager()
    text_sampler = ProceduralTextSampler()
    dataset = ModularStreamingDataset(
        voice_manager=voice_manager,
        text_sampler=text_sampler,
        target_extractor=target_extractor,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_modular_batch)
    data_iter = iter(dataloader)

    tokenizer = CharacterTokenizer()
    history = []
    start_step = 1
    total_audio_sec = 0.0

    recent_step_durations = collections.deque(maxlen=20)
    print(f"\n[Ready] Starting streaming pre-training loop (Step {start_step} -> {args.steps})...\n")

    for step in range(start_step, args.steps + 1):
        t0 = time.time()
        batch = next(data_iter)
        audio = batch["audio"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        audio_lengths = batch["audio_lengths"].to(device)

        batch_dur = sum(batch["durations"])
        total_audio_sec += batch_dur

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
            if args.arch == "phono_hubert":
                out = model(audio=audio, targets=targets, target_lengths=target_lengths, audio_lengths=audio_lengths)
            else:
                out = model(audio=audio, target_clusters=targets)

            loss = out["loss"]
            acc = out["accuracy"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        step_elapsed = time.time() - t0
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
            "avg_step_sec": round(avg_step_sec, 2),
            "eta_seconds": int(eta_seconds),
            "eta_formatted": f"{int(eta_seconds//60)}m {int(eta_seconds%60):02d}s",
            "active_voices": batch["voices"][:2],
            "active_voice": ", ".join(batch["voices"][:2]),
            "disk_bytes_used": 0,
            "last_heartbeat": time.time(),
        }
        try:
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump(live_status, f, indent=2)
        except Exception:
            pass

        if step % 10 == 0 or step == start_step:
            print(f"[{args.arch.upper()}] Step {step:4d}/{args.steps} | Loss: {loss_val:.4f} | Acc: {acc_val:5.1f}% | Audio: {total_audio_sec:6.1f}s ({hours:.3f}h) | ETA: {live_status['eta_formatted']}")

        # Milestone Benchmark Evaluation
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

        # Save model checkpoint
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
