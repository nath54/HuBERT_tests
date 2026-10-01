#!/usr/bin/env python3
"""Evaluate acoustic backbone on LibriSpeech test-other vs test-clean."""

import argparse
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.registry import ModelRegistry
from src.data.phoneme_tokenizer import PhonemeTokenizer
from scripts.run_pretrain import evaluate_direct_phonemes


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on LibriSpeech test-other")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/phono_v6_4_gated_diffusion/medium/v6_4_960h/best_checkpoint.pt",
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/librispeech/librispeech_test_other.json",
        help="Path to manifest (default: test-other)",
    )
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--blank_penalty", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"📦 Loading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    arch = ckpt.get("arch", "phono_v6_4_gated_diffusion")
    tier = ckpt.get("tier", "medium")
    config = ckpt["config"]
    step = ckpt.get("step", 70000)

    model_cls = ModelRegistry.get_entry(arch)["model_cls"]
    model = model_cls(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    phoneme_tokenizer = PhonemeTokenizer()

    print(f"=======================================================")
    print(f"🔍 EVALUATION ON: {args.manifest}")
    print(f"   Architecture: {arch} [{tier}] | Step: {step}")
    print(f"   Samples: {args.num_samples} | Device: {device}")
    print(f"=======================================================")

    results = evaluate_direct_phonemes(
        model=model,
        phoneme_tokenizer=phoneme_tokenizer,
        device=device,
        manifest_path=args.manifest,
        num_samples=args.num_samples,
        blank_penalty=args.blank_penalty,
        compute_lexicon=True,
    )

    print("\n" + "=" * 55)
    print(f"🎯 RESULTS ON {Path(args.manifest).name.upper()}")
    print("=" * 55)
    print(f" • Test PER (Greedy CTC)   : {results['per']:.2f}%")
    print(f" • Test CER                : {results['cer']:.2f}%")
    if results.get("lexicon_per") is not None:
        print(f" • Test Lexicon PER        : {results['lexicon_per']:.2f}%")
        print(f" • Test Lexicon WER        : {results.get('lexicon_wer', 0.0):.2f}%")
        print(f" • Sample Lexicon Decode   : {results.get('sample_lex_pred', '')[:65]}...")
    print(f" • Sample Phoneme Decode   : {results.get('sample_pred', '')[:65]}...")
    print("=" * 55 + "\n")

    out_file = Path("logs/test_other_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "manifest": args.manifest,
                "num_samples": args.num_samples,
                "step": step,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"💾 Results saved to {out_file}")


if __name__ == "__main__":
    main()
