#!/usr/bin/env python3
"""Master pipeline demonstration: Dataset Generation -> HuBERT ASR Training -> Evaluation -> Complete XAI Suite."""

import subprocess
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_command(cmd, desc):
    print("\n" + "=" * 65)
    print(f"  STEP: {desc}")
    print("=" * 65)
    result = subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"[Error] Command failed with return code {result.returncode}: {cmd}")
        sys.exit(result.returncode)


def main():
    python_bin = sys.executable

    print("\n" + "#" * 65)
    print("  AUDIOLEARN: COMPLETE HUBERT ASR + XAI DEMONSTRATION")
    print("#" * 65)

    # 1. Dataset Generation
    run_command(
        f"{python_bin} scripts/download_data.py --output_dir data/raw/synthetic --num_train 80 --num_val 20",
        "1. Generating Synthetic Speech Dataset with Formant Synthesizer",
    )

    # 2. HuBERT Training
    run_command(
        f"{python_bin} scripts/train.py --epochs 6 --batch_size 16 --lr 0.001",
        "2. Training HuBERT ASR Model with CTC Loss and AMP",
    )

    # 3. Model Evaluation
    run_command(
        f"{python_bin} scripts/evaluate.py --checkpoint checkpoints/best_model.pt",
        "3. Evaluating Trained HuBERT Checkpoint (WER & CER)",
    )

    # 4. XAI Suite
    run_command(
        f"{python_bin} scripts/run_xai.py --checkpoint checkpoints/best_model.pt --output_dir outputs",
        "4. Running Explainable AI Suite (Captum Gradients, NAPS, Trained Decoders)",
    )

    print("\n" + "#" * 65)
    print("  PIPELINE EXECUTION COMPLETE!")
    print("  - Trained Model Checkpoints: checkpoints/")
    print("  - Training History & TensorBoard Logs: logs/")
    print("  - Explainable AI Figures & Reports: outputs/")
    print("#" * 65)


if __name__ == "__main__":
    main()
