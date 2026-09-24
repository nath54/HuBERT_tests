"""Benchmarking suite for comparing Scratch HuBERT against SOTA ASR models (Meta HuBERT, OpenAI Whisper)."""

from src.benchmark.sota_evaluator import SOTABenchmarkRunner, normalize_text

__all__ = ["SOTABenchmarkRunner", "normalize_text"]
