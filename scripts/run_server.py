#!/usr/bin/env python3
"""Launcher for the Live HuBERT ASR & XAI Studio Web Server."""

import argparse
import sys
from pathlib import Path
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Start the Interactive HuBERT ASR & XAI Studio Server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to run web server on")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload for development")
    args = parser.parse_args()

    print("=" * 65)
    print(f"  Starting HuBERT Live Studio Server on http://{args.host}:{args.port}")
    print("=" * 65)
    print("Features available in the browser UI:")
    print("  • Real-time speech synthesis & custom audio upload")
    print("  • Live PyTorch forward pass on CUDA GPU")
    print("  • Interactive Layer-by-Layer tensor feature maps & attention")
    print("  • Real-time Captum Integrated Gradients on any predicted character")
    print("  • Live causal layer ablation (NAPS intervention)")
    print("  • Live background training console with real-time CER/WER gauges")
    print("=" * 65)

    uvicorn.run("src.server.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
