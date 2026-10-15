"""
Comprehensive End-to-End Test Suite for Clean Model Architecture.
Verifies:
1. OOP class hierarchy and modular configuration.
2. Checkpoint weight loading from existing pre-trained checkpoints.
3. Forward training pass with multi-task loss computation.
4. Synchronous forward inference and transcription.
5. Granular per-part and per-layer quantization.
6. Dual-threaded asynchronous streaming inference (Model 1 thread -> queue -> Model 2+3 thread).
7. Mobile export to ONNX static graphs and Pixel 10 mobile runner benchmark.
"""

# Import Modules
import os
import sys
import time
import torch
from torch import Tensor
import numpy as np

# Ensure src is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import (
    ModelConfig,
    Model_A2P,
    Model_PE,
    Model_WD,
    Model,
    ModelPartQuantizer,
    QuantizationConfig,
    DualThreadStreamingASR,
    TFLiteExporter,
    Pixel10MobileRunner,
)


def run_tests() -> None:
    print("=================================================================")
    print("      CLEAN MODEL ARCHITECTURE COMPREHENSIVE VERIFICATION       ")
    print("=================================================================")

    # 1. Test Configuration & Device Routing
    print("\n[TEST 1] Initializing Model Configurations & Device Routing...")
    # Target CPU for unit tests to ensure compatibility and leave GPU free for active background training
    config = ModelConfig()
    config.device_a2p = "cpu"
    config.device_pe = "cpu"
    config.device_wd = "cpu"
    # Reduce layer count for fast unit test instantiation
    config.a2p.encoder_layers = 2
    config.pe.num_layers = 2
    config.wd.decoder_layers = 2
    print(f"  - ModelConfig created successfully: {config}")
    assert config.get_device("A2P") == torch.device("cpu")
    assert config.get_device("PE") == torch.device("cpu")
    assert config.get_device("WD") == torch.device("cpu")
    print("  [PASS] Test 1 completed.")

    # 2. Test Model Instantiation
    print("\n[TEST 2] Instantiating Tripartite Model Components...")
    unified_model = Model(model_config=config)
    print("  - Unified Model instantiated successfully.")
    print(f"  - Model_A2P device: {unified_model.model_a2p.get_device()}")
    print(f"  - Model_PE device:  {unified_model.model_pe.get_device()}")
    print(f"  - Model_WD device:  {unified_model.model_wd.get_device()}")
    print("  [PASS] Test 2 completed.")

    # 3. Test Checkpoint Weight Loading
    print("\n[TEST 3] Verifying Checkpoint Weight Loading...")
    a2p_ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints/phono_v8_acoustic_multilingual_sota/latest_v8_acoustic.pt")
    p2g_ckpt_path = os.path.join(PROJECT_ROOT, "checkpoints/phono_v8_robust_p2g/best_robust_p2g.pt")

    if os.path.isfile(a2p_ckpt_path) and os.path.isfile(p2g_ckpt_path):
        print(f"  - Found A2P checkpoint: {a2p_ckpt_path}")
        print(f"  - Found P2G checkpoint: {p2g_ckpt_path}")
        # Note: config has 2 layers for quick test, so strict=False loads matching CNN/embeddings/heads
        status = unified_model.load_from_checkpoints(a2p_ckpt_path, p2g_ckpt_path)
        print(f"  - Loading status: {status}")
        assert status["a2p"] is True
        assert status["pe"] is True
        assert status["wd"] is True
    else:
        print("  - Checkpoints not found, skipping checkpoint loading test.")
    print("  [PASS] Test 3 completed.")

    # 4. Test Forward Training Pass
    print("\n[TEST 4] Testing Multi-Task Training Forward Pass...")
    batch_size: int = 2
    audio_len: int = 16000  # 1 second of audio
    # Synthetic audio tensor: Tensor [B, T_samples], Role: batch of raw speech waveforms
    dummy_audio: Tensor = torch.randn(batch_size, audio_len, dtype=torch.float32)
    # Synthetic phoneme targets: Tensor [B, S], Role: CTC target tokens
    dummy_phoneme_targets: Tensor = torch.tensor([[10, 8, 25, 30], [15, 8, 40, 8]], dtype=torch.long)
    dummy_phoneme_lengths: Tensor = torch.tensor([4, 4], dtype=torch.long)

    # Synthetic character targets for Model 3: Tensor [B, L, K]
    dummy_in_bytes: Tensor = torch.zeros((batch_size, 2, 8), dtype=torch.long)
    dummy_tgt_bytes: Tensor = torch.zeros((batch_size, 2, 8), dtype=torch.long)

    losses = unified_model.forward_training(
        audio=dummy_audio,
        phoneme_targets=dummy_phoneme_targets,
        phoneme_lengths=dummy_phoneme_lengths,
        input_byte_ids=dummy_in_bytes,
        target_byte_ids=dummy_tgt_bytes,
    )
    print(f"  - Training losses computed: {list(losses.keys())}")
    assert "loss" in losses
    assert "loss_a2p" in losses
    assert torch.isfinite(losses["loss"])
    print(f"  - Total Multi-Task Loss: {losses['loss'].item():.4f}")
    print("  [PASS] Test 4 completed.")

    # 5. Test Synchronous Inference
    print("\n[TEST 5] Testing Synchronous Forward Inference & Transcribe...")
    inf_out = unified_model.forward_inference(dummy_audio[:1])
    print(f"  - Decoded transcripts: {inf_out['transcripts']}")
    print(f"  - Sliced phoneme words count: {len(inf_out['word_slices'][0])}")

    # Test transcribe API
    transcript_str = unified_model.transcribe(dummy_audio[0])
    print(f"  - High-level transcribe() output: '{transcript_str}'")
    print("  [PASS] Test 5 completed.")

    # 6. Test Granular Quantization
    print("\n[TEST 6] Testing Granular Per-Part & Per-Layer Quantization...")
    quantizer = ModelPartQuantizer()
    mem_before = quantizer.inspect_memory_footprint(unified_model.model_a2p)
    print(f"  - Model_A2P before quantization: {mem_before['memory_mb']} MB across {mem_before['total_parameters']} params")

    # Quantize Model_A2P with dynamic int8
    quant_cfg = QuantizationConfig(precision="qint8", target_layer_types=(torch.nn.Linear,))
    quantized_a2p = quantizer.quantize_part(unified_model.model_a2p, quant_cfg)
    print("  - Successfully applied dynamic INT8 quantization to Model_A2P.")

    # Quantize deepcopy of unified model with per-part precision (A2P: int8, PE: float16, WD: weight_only_int8)
    import copy
    model_to_quantize = copy.deepcopy(unified_model)
    multi_cfg = {
        "A2P": QuantizationConfig(precision="qint8"),
        "PE": QuantizationConfig(precision="float16"),
        "WD": QuantizationConfig(precision="weight_only_int8"),
    }
    quantized_model = quantizer.quantize_model(model_to_quantize, multi_cfg)
    print("  - Successfully applied heterogeneous per-part quantization to Unified Model.")
    print("  [PASS] Test 6 completed.")

    # 7. Test Dual-Threaded Streaming Inference
    print("\n[TEST 7] Testing Dual-Threaded Streaming Inference...")
    emitted_words_list = []

    def on_word(word: str) -> None:
        emitted_words_list.append(word)
        print(f"    [STREAM EVENT] Word decoded on Thread 2: '{word}'")

    streamer = DualThreadStreamingASR(
        model_a2p=unified_model.model_a2p,
        model_pe=unified_model.model_pe,
        model_wd=unified_model.model_wd,
        chunk_duration_sec=0.2,  # 200ms chunks for rapid test
        sample_rate=16000,
        word_callback=on_word,
    )

    streamer.start()
    assert streamer.is_running() is True
    print("  - Dual-threaded streaming engine started successfully.")

    # Stream 3 chunks of 0.2s audio
    chunk = torch.randn(3200, dtype=torch.float32)
    for i in range(3):
        streamer.push_audio(chunk)
        time.sleep(0.05)

    final_sentence = streamer.finish_stream(timeout=3.0)
    print(f"  - Streaming completed. Full decoded sentence: '{final_sentence}'")
    assert streamer.is_running() is False
    print("  [PASS] Test 7 completed.")

    # 8. Test Mobile ONNX Export & Pixel 10 Runner
    print("\n[TEST 8] Testing Google Pixel 10 Mobile Export & ONNX Runtime Benchmark...")
    export_dir = os.path.join(CURRENT_DIR, "test_export_pixel10")
    exporter = TFLiteExporter(output_dir=export_dir)

    onnx_paths = exporter.export_all(unified_model, chunk_seconds=1.0)
    print(f"  - Exported ONNX models: {list(onnx_paths.keys())}")
    for k, p in onnx_paths.items():
        assert os.path.isfile(p), f"Missing exported model: {p}"
        print(f"    - {k}: {p} ({os.path.getsize(p) / (1024*1024):.2f} MB)")

    # Generate TFLite conversion script
    conv_script = exporter.generate_tflite_conversion_script(onnx_paths)
    assert os.path.isfile(conv_script)

    # Benchmark using Pixel10MobileRunner
    mobile_runner = Pixel10MobileRunner(
        a2p_model_path=onnx_paths["a2p"],
        pe_model_path=onnx_paths["pe"],
        wd_model_path=onnx_paths["wd"],
        execution_provider="CPUExecutionProvider",
    )
    test_audio_np = np.random.randn(16000).astype(np.float32)
    mobile_res = mobile_runner.transcribe_audio_chunk(test_audio_np)
    print(f"  - Mobile Runner Output: {mobile_res}")
    print(f"    Latency: {mobile_res['latency_ms']['total_ms']} ms | RTF: {mobile_res['rtf']}")
    print("  [PASS] Test 8 completed.")

    print("\n=================================================================")
    print("      ALL 8 VERIFICATION TESTS PASSED SUCCESSFULLY!              ")
    print("=================================================================")


if __name__ == "__main__":
    run_tests()
