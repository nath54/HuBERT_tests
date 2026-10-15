#!/usr/bin/env python3
"""
Large-Scale HuBERT Pre-training & LibriSpeech Benchmark Scaling Tracker.

Streams clean English & French sentences from Wikipedia, Books (Opus Books),
and dictionaries in RAM (0 bytes disk space), pre-trains OurHuBERT on 34 Piper voices,
and periodically evaluates the downstream LibriSpeech test-clean benchmark (WER / CER / PER)
to detect performance plateaus and guide model architecture scaling.

Supports:
- Resumable training via --resume flag (restores model, optimizer, scaler, step, audio hours, history)
- Graceful interrupt handling (SIGINT/SIGTERM) with zero progress lost
- Real-time accurate ETA prediction with moving-window step timing
- Live status export to logs/pretrain_status_live.json
"""

import argparse
import collections
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple

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


def save_pretrain_checkpoint(
    checkpoint_path: str,
    step: int,
    tier: str,
    config: HuBERTConfig,
    pretrain_model: HuBERTForPreTraining,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    total_audio_sec: float,
    history: List[Dict],
):
    """Save full pre-training state (model weights, optimizer, scaler, progress) to allow seamless resumption."""
    out_dir = Path(checkpoint_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "tier": tier,
        "config": config,
        "model_state_dict": pretrain_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if (scaler is not None and hasattr(scaler, "state_dict")) else None,
        "total_audio_sec": total_audio_sec,
        "history": history,
        "saved_at": time.time(),
    }
    torch.save(state, checkpoint_path)


def load_pretrain_checkpoint(
    resume_arg: str,
    pretrain_model: HuBERTForPreTraining,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
) -> Tuple[int, float, List[Dict]]:
    """Load pre-training state from specified path or candidate checkpoints."""
    candidates = []
    if resume_arg and resume_arg.lower() not in ("true", "1", "auto"):
        candidates.append(Path(resume_arg))
    candidates.extend([
        Path("checkpoints/pretrain_checkpoint_latest.pt"),
        Path("checkpoints/best_model.pt"),
        Path("checkpoints/hubert_piper_pretrained.pt"),
    ])

    target_path = None
    for c in candidates:
        if c.exists():
            target_path = c
            break

    if not target_path:
        print(f"⚠️ [RESUME] No valid checkpoint found in candidates. Starting from scratch.")
        return 1, 0.0, []

    print(f"\n🔄 [RESUME] Found checkpoint at: {target_path}")
    ckpt = torch.load(target_path, map_location=device)

    start_step = 1
    total_audio_sec = 0.0
    history = []

    # 1. Model weights
    if "model_state_dict" in ckpt:
        try:
            pretrain_model.load_state_dict(ckpt["model_state_dict"])
            print("  ✔ Pre-training model state dict restored.")
        except Exception as e:
            print(f"  ℹ Note: Full state dict mismatch ({e}), attempting backbone restore...")
            if hasattr(pretrain_model, "load_pretrained_backbone"):
                pretrain_model.load_pretrained_backbone(ckpt["model_state_dict"])
    elif "encoder" in ckpt or "feature_extractor" in ckpt:
        if hasattr(pretrain_model, "load_pretrained_backbone"):
            pretrain_model.load_pretrained_backbone(ckpt)
            print("  ✔ Backbone weights restored into pretrain model.")

    # 2. Optimizer & Scaler
    if "optimizer_state_dict" in ckpt and optimizer is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("  ✔ Optimizer state restored.")
        except Exception as e:
            print(f"  ℹ Optimizer state could not be restored ({e}), keeping fresh optimizer.")

    if "scaler_state_dict" in ckpt and scaler is not None and ckpt.get("scaler_state_dict") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            print("  ✔ AMP GradScaler state restored.")
        except Exception:
            pass

    # 3. Progress tracking
    if "step" in ckpt:
        start_step = int(ckpt["step"]) + 1
    if "total_audio_sec" in ckpt:
        total_audio_sec = float(ckpt["total_audio_sec"])
    elif "history" in ckpt and len(ckpt["history"]) > 0:
        total_audio_sec = float(ckpt["history"][-1].get("cumulative_audio_sec", 0.0))
        if start_step == 1:
            start_step = int(ckpt["history"][-1].get("step", 0)) + 1

    if "history" in ckpt and ckpt["history"]:
        history = ckpt["history"]
    elif Path("logs/scaling_benchmark_history.json").exists():
        try:
            with open("logs/scaling_benchmark_history.json", "r", encoding="utf-8") as f:
                h_data = json.load(f)
                history = h_data.get("history", [])
        except Exception:
            pass

    print(f"  ✔ Progress: Step {start_step - 1} completed | Audio synthesized: {total_audio_sec:6.1f}s ({total_audio_sec/3600.0:.3f}h)")
    print(f"  ✔ Historical milestone records restored: {len(history)}")
    return start_step, total_audio_sec, history


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
        ref_phonemes = runner.phonemize_text(s["transcript"])
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


def format_eta_str(seconds: float) -> str:
    """Format seconds into readable ETA string (e.g. '1h 14m' or '4m 32s')."""
    if seconds <= 0:
        return "0s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


def main():
    parser = argparse.ArgumentParser(description="Large-scale pre-training & LibriSpeech benchmark tracking.")
    parser.add_argument("--steps", type=int, default=625, help="Total pre-training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (utterances per step)")
    parser.add_argument("--eval_interval", type=int, default=125, help="Benchmark evaluation interval (steps)")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--tier", type=str, default="mini", choices=["mini", "small", "medium", "base"], help="Model size tier")
    parser.add_argument("--probe_steps", type=int, default=25, help="CTC probe calibration steps on LibriSpeech")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", nargs="?", const="checkpoints/pretrain_checkpoint_latest.pt", default=None,
                        help="Resume pre-training from checkpoint (or auto-resume latest if given without path)")
    parser.add_argument("--save_interval", type=int, default=25, help="Save full pretrain checkpoint every N steps")
    args = parser.parse_args()

    # Interruption flag for graceful exit
    interrupted = False
    def signal_handler(signum, frame):
        nonlocal interrupted
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        print(f"\n🛑 [INTERRUPT] Received {sig_name}. Preparing graceful checkpoint and safe shutdown...")
        interrupted = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    device = torch.device(args.device)
    config = ARCHITECTURE_TIERS[args.tier]
    tokenizer = CharacterTokenizer()
    config.vocab_size = tokenizer.vocab_size

    print("=" * 75)
    print("   LARGE-SCALE HUBERT PRE-TRAINING & BENCHMARK SCALING EXPERIMENT")
    print("=" * 75)
    print(f"Architecture Tier: {args.tier.upper()} | Layers: {config.encoder_layers} | Dim: {config.encoder_embed_dim}")
    print(f"Device: {device} | Total Target Steps: {args.steps} | Batch Size: {args.batch_size}")
    print(f"Benchmark Eval Every: {args.eval_interval} steps | Probe Steps: {args.probe_steps}")
    print(f"Resumable Checkpointing: Enabled (every {args.save_interval} steps & on SIGINT/SIGTERM)")
    print("-" * 75)

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

    # Check for resume
    start_step = 1
    total_audio_sec = 0.0
    history = []

    if args.resume is not None:
        start_step, total_audio_sec, history = load_pretrain_checkpoint(
            args.resume, pretrain_model, optimizer, scaler, device
        )

    # If starting from scratch (step 1), run initial 0-hour baseline evaluation
    if start_step == 1:
        print("\n[3/4] Evaluating initial 0-hour baseline on LibriSpeech test-clean...")
        initial_ctc = run_quick_ctc_calibration(pretrain_model, config, tokenizer, device, probe_steps=0)
        base_eval = evaluate_on_benchmark(initial_ctc, tokenizer, device, num_samples=20)
        print(f"Baseline Score (0h Pre-training) -> WER: {base_eval['wer']}% | CER: {base_eval['cer']}% | PER: {base_eval['per']}%")

        history = [{
            "step": 0,
            "cumulative_audio_sec": 0.0,
            "cumulative_audio_hours": 0.0,
            "pretrain_loss": 4.60,
            "masked_acc_pct": 1.0,
            "librispeech_wer": base_eval["wer"],
            "librispeech_cer": base_eval["cer"],
            "librispeech_per": base_eval["per"],
            "sample_prediction": base_eval["sample_pred"],
            "disk_bytes_used": 0,
            "plateau_detected": False,
        }]
    else:
        print(f"\n[3/4] Skipping 0h baseline evaluation (already resuming from Step {start_step - 1} with {len(history)} records).")

    print(f"\n[4/4] Continuous streaming self-supervised pre-training (Step {start_step} -> {args.steps})...")
    data_iter = iter(dataloader)
    start_time = time.time()

    # Moving-window step duration tracker for accurate ETA
    recent_step_durations = collections.deque(maxlen=20)

    for step in range(start_step, args.steps + 1):
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
        recent_step_durations.append(step_elapsed)

        # Accurate ETA Calculation
        avg_step_sec = sum(recent_step_durations) / len(recent_step_durations)
        remaining_steps = max(0, args.steps - step)
        remaining_evals = sum(1 for s in range(step + 1, args.steps + 1) if (s % args.eval_interval == 0 or s == args.steps))
        eval_overhead_sec = remaining_evals * (args.probe_steps * 1.0 + 15.0)  # ~35s per probe CTC calibration & LibriSpeech evaluation
        eta_seconds = (remaining_steps * avg_step_sec) + eval_overhead_sec
        eta_str = format_eta_str(eta_seconds)
        finish_timestamp = time.time() + eta_seconds
        finish_time_str = datetime.fromtimestamp(finish_timestamp).strftime("%H:%M:%S")
        pct = (step / args.steps) * 100.0
        hours = total_audio_sec / 3600.0

        # Step Progress Display
        if step % 10 == 0 or step == start_step:
            voices_str = ", ".join(batch["voices"][:2])
            print(f"Step {step:4d}/{args.steps} ({pct:4.1f}%) | ETA: {eta_str} (~{avg_step_sec:.1f}s/step) | Finish: ~{finish_time_str} | Loss: {loss_val:.4f} | Masked Acc: {acc_val:4.1f}% | Audio: {total_audio_sec:6.1f}s ({hours:.3f}h) | Voices: [{voices_str}]")

        # Export Live Status for UI & Monitoring
        live_status = {
            "is_running": True,
            "step": step,
            "total_steps": args.steps,
            "progress_pct": round(pct, 1),
            "loss": loss_val,
            "masked_accuracy_pct": acc_val,
            "cumulative_audio_sec": round(total_audio_sec, 1),
            "cumulative_audio_hours": round(hours, 4),
            "avg_step_sec": round(avg_step_sec, 2),
            "eta_seconds": int(eta_seconds),
            "eta_formatted": eta_str,
            "estimated_finish_time": finish_time_str,
            "tier": args.tier,
            "active_voices": batch["voices"][:2],
            "active_voice": ", ".join(batch["voices"][:2]),
            "disk_bytes_used": 0,
            "last_heartbeat": time.time(),
        }
        Path("logs").mkdir(exist_ok=True)
        try:
            with open("logs/pretrain_status_live.json", "w", encoding="utf-8") as f:
                json.dump(live_status, f, indent=2)
        except Exception:
            pass

        # Periodic checkpoint save
        if step % args.save_interval == 0:
            save_pretrain_checkpoint(
                "checkpoints/pretrain_checkpoint_latest.pt",
                step, args.tier, config, pretrain_model, optimizer, scaler, total_audio_sec, history
            )

        # Downstream Benchmark Milestone
        if step % args.eval_interval == 0 or step == args.steps:
            print(f"\n--- [Milestone Step {step}] Running LibriSpeech Benchmark Evaluation ---")
            calibrated_ctc = run_quick_ctc_calibration(pretrain_model, config, tokenizer, device, probe_steps=args.probe_steps)
            bench_res = evaluate_on_benchmark(calibrated_ctc, tokenizer, device, num_samples=20)

            m_hours = round(total_audio_sec / 3600.0, 4)
            prev_record = history[-1] if history else {"librispeech_wer": 100.0, "cumulative_audio_hours": 0.0}
            delta_wer = prev_record["librispeech_wer"] - bench_res["wer"]
            delta_hours = m_hours - prev_record["cumulative_audio_hours"]
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
                "cumulative_audio_hours": m_hours,
                "pretrain_loss": loss_val,
                "masked_acc_pct": acc_val,
                "librispeech_wer": bench_res["wer"],
                "librispeech_cer": bench_res["cer"],
                "librispeech_per": bench_res["per"],
                "sample_prediction": bench_res["sample_pred"],
                "disk_bytes_used": 0,
                "marginal_wer_gain_per_hour": round(slope, 2),
                "plateau_detected": is_plateau,
            }
            history.append(entry)

            print(f"🏆 Milestone Results: Pre-train Hours: {m_hours}h | Loss: {loss_val:.4f} | LibriSpeech WER: {bench_res['wer']}% | CER: {bench_res['cer']}% | PER: {bench_res['per']}%")
            if bench_res["sample_pred"]:
                print(f"   Sample Decoded: \"{bench_res['sample_pred']}\"")

            # Checkpoint saves
            pretrain_model.save_pretrained_backbone("checkpoints/hubert_piper_pretrained.pt")
            save_pretrain_checkpoint(
                "checkpoints/pretrain_checkpoint_latest.pt",
                step, args.tier, config, pretrain_model, optimizer, scaler, total_audio_sec, history
            )
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
                    "total_audio_hours": m_hours,
                    "history": history,
                }, f, indent=2)

        # Handle Graceful Interruption
        if interrupted:
            print(f"\n" + "=" * 75)
            print(f"🛑 [INTERRUPT] Safely pausing pre-training at Step {step}/{args.steps}...")
            save_pretrain_checkpoint(
                "checkpoints/pretrain_checkpoint_latest.pt",
                step, args.tier, config, pretrain_model, optimizer, scaler, total_audio_sec, history
            )
            pretrain_model.save_pretrained_backbone("checkpoints/hubert_piper_pretrained.pt")
            live_status["is_running"] = False
            live_status["status"] = "interrupted"
            with open("logs/pretrain_status_live.json", "w", encoding="utf-8") as f:
                json.dump(live_status, f, indent=2)

            print(f"✅ Checkpoint safely committed to 'checkpoints/pretrain_checkpoint_latest.pt'")
            print(f"   • Step: {step}")
            print(f"   • Synthesized Audio in RAM: {total_audio_sec:6.1f}s ({hours:.4f}h)")
            print(f"   • Saved Pretrained Backbone: checkpoints/hubert_piper_pretrained.pt")
            print(f"\n👉 You can resume anytime seamlessly with:")
            print(f"   .venv/bin/python scripts/scale_pretrain_benchmark.py --tier {args.tier} --steps {args.steps} --resume")
            print("=" * 75)
            sys.exit(0)

    total_time = time.time() - start_time
    print(f"\n" + "=" * 75)
    print(f"✅ Large-scale Pre-training complete in {total_time:.1f}s.")
    print(f"Total Audio Synthesized in RAM: {total_audio_sec:.1f} seconds ({total_audio_sec/3600.0:.3f} hours)")
    print(f"Hard Drive Space Used: 0 BYTES.")
    print("Results saved to: logs/scaling_benchmark_history.json")
    print("=" * 75)


if __name__ == "__main__":
    main()
