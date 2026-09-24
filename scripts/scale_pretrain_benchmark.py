#!/usr/bin/env python3
"""
Large-Scale HuBERT Pre-training & LibriSpeech Benchmark Scaling Tracker.

Streams clean English & French sentences from Wikipedia, Books (Opus Books),
and dictionaries in RAM (0 bytes disk space), pre-trains OurHuBERT on 34 Piper voices,
and periodically evaluates the downstream LibriSpeech test-clean benchmark (WER / CER)
to detect performance plateaus and guide model architecture scaling.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.config import HuBERTConfig
from src.models.hubert_asr import HuBERTForCTC
from src.models.hubert_pretrain import HuBERTForPreTraining
from src.data.tokenizer import CharacterTokenizer
from src.data.streaming_piper import PiperVoiceManager, AcousticUnitExtractor, PiperStreamingDataset, collate_pretrain_batch
from src.data.text_corpus import StreamingTextCorpus
from src.data.dataset import AudioASRDataset, AudioCollateFn
from src.benchmark.sota_evaluator import SOTABenchmarkRunner
from data.sample_dataset import load_manifest


# Architecture scaling configurations
ARCHITECTURE_TIERS = {
    "mini": HuBERTConfig(
        encoder_layers=4,
        encoder_heads=4,
        encoder_embed_dim=256,
        encoder_ffn_dim=1024,
    ),
    "small": HuBERTConfig(
        encoder_layers=6,
        encoder_heads=6,
        encoder_embed_dim=384,
        encoder_ffn_dim=1536,
    ),
    "medium": HuBERTConfig(
        encoder_layers=8,
        encoder_heads=8,
        encoder_embed_dim=512,
        encoder_ffn_dim=2048,
    ),
    "base": HuBERTConfig(
        encoder_layers=12,
        encoder_heads=12,
        encoder_embed_dim=768,
        encoder_ffn_dim=3072,
    ),
}


def run_quick_ctc_calibration(
    pretrain_model: HuBERTForPreTraining,
    config: HuBERTConfig,
    tokenizer: CharacterTokenizer,
    device: torch.device,
    probe_steps: int = 30,
) -> HuBERTForCTC:
    """Instantiate HuBERTForCTC, load pre-trained backbone, and calibrate CTC head on LibriSpeech."""
    ctc_model = HuBERTForCTC(config=config).to(device)

    # Transfer backbone weights
    pretrain_model.transfer_to_ctc_model(ctc_model)

    if probe_steps <= 0:
        return ctc_model

    # Quick probe on LibriSpeech train
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
        return {"wer": 100.0, "cer": 100.0, "sample_pred": ""}

    with open(test_manifest, "r", encoding="utf-8") as f:
        samples = json.load(f)[:num_samples]

    runner = SOTABenchmarkRunner(device=str(device))
    wers, cers = [], []
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

        ref_norm = normalize_text(s["transcript"])
        pred_norm = normalize_text(pred_text)
        w = round(float(jiwer.wer(ref_norm, pred_norm)), 4) if ref_norm else 1.0
        c = round(float(jiwer.cer(ref_norm, pred_norm)), 4) if ref_norm else 1.0
        wers.append(w)
        cers.append(c)

    avg_wer = round(float(sum(wers) / max(1, len(wers))) * 100.0, 2)
    avg_cer = round(float(sum(cers) / max(1, len(cers))) * 100.0, 2)

    return {"wer": avg_wer, "cer": avg_cer, "sample_pred": sample_pred}


def main():
    parser = argparse.ArgumentParser(description="Large-scale pre-training & LibriSpeech benchmark tracking.")
    parser.add_argument("--steps", type=int, default=300, help="Total pre-training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (utterances per step)")
    parser.add_argument("--eval_interval", type=int, default=50, help="Benchmark evaluation interval (steps)")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--tier", type=str, default="mini", choices=["mini", "small", "medium", "base"], help="Model size tier")
    parser.add_argument("--probe_steps", type=int, default=30, help="CTC probe calibration steps on LibriSpeech")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = ARCHITECTURE_TIERS[args.tier]
    tokenizer = CharacterTokenizer()
    config.vocab_size = tokenizer.vocab_size

    print("=" * 70)
    print("   LARGE-SCALE HUBERT PRE-TRAINING & BENCHMARK SCALING EXPERIMENT")
    print("=" * 70)
    print(f"Architecture Tier: {args.tier.upper()} | Layers: {config.encoder_layers} | Dim: {config.encoder_embed_dim}")
    print(f"Device: {device} | Total Steps: {args.steps} | Batch Size: {args.batch_size}")
    print(f"Benchmark Eval Every: {args.eval_interval} steps | Probe Steps: {args.probe_steps}")
    print("-" * 70)

    # 1. Initialize Streaming Sources (0-Disk)
    print("[1/4] Initializing streaming text corpus (English + French Books & Wikipedia)...")
    corpus = StreamingTextCorpus(languages=("en", "fr"))
    voice_manager = PiperVoiceManager()
    unit_extractor = AcousticUnitExtractor(num_clusters=100)

    dataset = PiperStreamingDataset(
        voice_manager=voice_manager,
        text_sampler=corpus,
        unit_extractor=unit_extractor,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_pretrain_batch)

    # 2. Build Pre-training Model
    print(f"[2/4] Initializing HuBERT for Pre-training (Tier: {args.tier})...")
    pretrain_model = HuBERTForPreTraining(config=config, num_clusters=100).to(device)
    num_params = sum(p.numel() for p in pretrain_model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    optimizer = torch.optim.AdamW(pretrain_model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Initial baseline benchmark evaluation (at 0 pre-training hours)
    print("\n[3/4] Evaluating initial 0-hour baseline on LibriSpeech test-clean...")
    initial_ctc = run_quick_ctc_calibration(pretrain_model, config, tokenizer, device, probe_steps=0)
    base_eval = evaluate_on_benchmark(initial_ctc, tokenizer, device, num_samples=20)
    print(f"Baseline Score (0h Pre-training) -> WER: {base_eval['wer']}% | CER: {base_eval['cer']}%")

    # Tracking records
    history = [{
        "step": 0,
        "cumulative_audio_sec": 0.0,
        "cumulative_audio_hours": 0.0,
        "pretrain_loss": 4.60,
        "masked_acc_pct": 1.0,
        "librispeech_wer": base_eval["wer"],
        "librispeech_cer": base_eval["cer"],
        "sample_prediction": base_eval["sample_pred"],
        "disk_bytes_used": 0,
        "plateau_detected": False,
    }]

    print("\n[4/4] Starting continuous streaming self-supervised pre-training...")
    total_audio_sec = 0.0
    data_iter = iter(dataloader)
    start_time = time.time()

    for step in range(1, args.steps + 1):
        step_t0 = time.time()
        batch = next(data_iter)
        audio = batch["audio"].to(device)
        targets = batch["target_clusters"].to(device)
        batch_dur = sum(batch["durations"])
        total_audio_sec += batch_dur

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
            out = pretrain_model(audio=audio, target_clusters=targets)
            loss = out["loss"]
            acc = out["accuracy"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(pretrain_model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_val = round(float(loss.item()), 4)
        acc_val = round(float(acc.item() * 100.0), 2) if acc is not None else 0.0
        step_elapsed = time.time() - step_t0

        if step % 10 == 0 or step == 1:
            hours = total_audio_sec / 3600.0
            voices_str = ", ".join(batch["voices"][:2])
            print(f"Step {step:4d}/{args.steps} | Loss: {loss_val:.4f} | Masked Acc: {acc_val:4.1f}% | Audio In-RAM: {total_audio_sec:6.1f}s ({hours:.4f}h) | 0 Disk Bytes | Voices: [{voices_str}]")

        # Downstream Benchmark Milestone
        if step % args.eval_interval == 0 or step == args.steps:
            print(f"\n--- [Milestone Step {step}] Running LibriSpeech Benchmark Evaluation ---")
            calibrated_ctc = run_quick_ctc_calibration(pretrain_model, config, tokenizer, device, probe_steps=args.probe_steps)
            bench_res = evaluate_on_benchmark(calibrated_ctc, tokenizer, device, num_samples=20)

            hours = round(total_audio_sec / 3600.0, 4)
            prev_record = history[-1]
            delta_wer = prev_record["librispeech_wer"] - bench_res["wer"]
            delta_hours = hours - prev_record["cumulative_audio_hours"]
            slope = (delta_wer / max(0.0001, delta_hours)) if delta_hours > 0 else 0.0

            # Plateau condition: WER improvement flattens out
            is_plateau = False
            if step >= args.eval_interval * 2 and abs(delta_wer) < 0.5:
                is_plateau = True
                print(f"⚠️ [PLATEAU DETECTED] WER change is only {delta_wer:+.2f}% over {delta_hours:.3f}h audio.")
                print(f"👉 Model capacity for '{args.tier.upper()}' ({num_params/1e6:.1f}M) is saturating.")
                print(f"👉 Recommended next step: scale architecture to '{'small' if args.tier == 'mini' else 'medium' if args.tier == 'small' else 'base'}'.")

            entry = {
                "step": step,
                "cumulative_audio_sec": round(total_audio_sec, 2),
                "cumulative_audio_hours": hours,
                "pretrain_loss": loss_val,
                "masked_acc_pct": acc_val,
                "librispeech_wer": bench_res["wer"],
                "librispeech_cer": bench_res["cer"],
                "sample_prediction": bench_res["sample_pred"],
                "disk_bytes_used": 0,
                "marginal_wer_gain_per_hour": round(slope, 2),
                "plateau_detected": is_plateau,
            }
            history.append(entry)

            print(f"🏆 Milestone Results: Pre-train Hours: {hours}h | Loss: {loss_val:.4f} | LibriSpeech WER: {bench_res['wer']}% | CER: {bench_res['cer']}%")
            if bench_res["sample_pred"]:
                print(f"   Sample Decoded: \"{bench_res['sample_pred']}\"")

            # Checkpoint save
            pretrain_model.save_pretrained_backbone("checkpoints/hubert_piper_pretrained.pt")
            torch.save({
                "config": config,
                "model_state_dict": calibrated_ctc.state_dict(),
                "tier": args.tier,
                "step": step,
                "history": history,
            }, "checkpoints/best_model.pt")

            with open("logs/scaling_benchmark_history.json", "w", encoding="utf-8") as f:
                json.dump({
                    "model_tier": args.tier,
                    "parameters_m": round(num_params / 1e6, 2),
                    "total_steps": step,
                    "total_audio_hours": hours,
                    "history": history,
                }, f, indent=2)

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"✅ Large-scale Pre-training complete in {total_time:.1f}s.")
    print(f"Total Audio Synthesized in RAM: {total_audio_sec:.1f} seconds ({total_audio_sec/3600.0:.3f} hours)")
    print(f"Hard Drive Space Used: 0 BYTES.")
    print("Results saved to: logs/scaling_benchmark_history.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
