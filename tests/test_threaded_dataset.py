"""Unit tests for StepProfiler and BufferedSpeechBatchGenerator."""

import time
import pytest
import torch

from src.data.threaded_dataset import StepProfiler, BufferedSpeechBatchGenerator, collate_modular_batch
from src.data.target_extractors import PhonemeTargetExtractor
from src.data.streaming_piper import PiperVoiceManager, ProceduralTextSampler


def test_step_profiler_timing_and_summary():
    profiler = StepProfiler(window_size=5)

    # Test record
    profiler.record("time_forward", 0.05)
    profiler.record("time_forward", 0.07)
    mean_val = profiler.get_metric_mean("time_forward")
    assert abs(mean_val - 0.06) < 1e-4

    # Test time_block context manager
    with profiler.time_block("time_loss"):
        time.sleep(0.01)

    summary = profiler.get_summary()
    assert "time_forward" in summary
    assert "time_loss" in summary
    assert summary["time_forward"]["ms"] > 50.0
    assert summary["time_loss"]["ms"] >= 9.0

    # Test breakdown table formatting
    table = profiler.format_console_breakdown(buffer_occupancy=15, max_buffer_size=20)
    assert "PRODUCER PIPELINE" in table
    assert "CONSUMER PIPELINE" in table
    assert "15/20" in table


def test_collate_modular_batch():
    batch = [
        {
            "audio": torch.randn(1600),
            "targets": torch.tensor([1, 2, 3]),
            "duration": 0.1,
            "text": "test one",
            "voice_name": "voice_1",
        },
        {
            "audio": torch.randn(3200),
            "targets": torch.tensor([4, 5]),
            "duration": 0.2,
            "text": "test two",
            "voice_name": "voice_2",
        },
    ]

    collated = collate_modular_batch(batch)
    assert collated["audio"].shape == (2, 3200)
    assert collated["targets"].shape == (2, 3)
    assert collated["audio_lengths"].tolist() == [1600, 3200]
    assert collated["target_lengths"].tolist() == [3, 2]
    assert collated["texts"] == ["test one", "test two"]
    assert collated["voices"] == ["voice_1", "voice_2"]


def test_buffered_generator_lifecycle():
    voice_manager = PiperVoiceManager()
    text_sampler = ProceduralTextSampler()
    target_extractor = PhonemeTargetExtractor()

    profiler = StepProfiler()

    gen = BufferedSpeechBatchGenerator(
        voice_manager=voice_manager,
        text_sampler=text_sampler,
        target_extractor=target_extractor,
        batch_size=2,
        max_buffer_size=5,
        low_watermark=2,
        num_workers=2,
        use_rolling_pool=True,
        pool_capacity=10,
        profiler=profiler,
    )

    try:
        # Fetch one batch (producer will generate and collate)
        batch = gen.get_batch(timeout=25.0)
        assert "audio" in batch
        assert "targets" in batch
        assert batch["audio"].shape[0] == 2
        assert len(batch["durations"]) == 2

        # Check profiler has recorded metrics
        summary = profiler.get_summary()
        assert "time_queue_get_wait" in summary
        assert "time_sample_text" in summary
        assert "time_piper_synth" in summary

    finally:
        gen.stop()
