# Tripartite End-to-End Speech Recognition Architecture (`clean_model_src`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![ONNX 1.17+](https://img.shields.io/badge/ONNX-opset_17-005ced.svg)](https://onnx.ai/)
[![TFLite Ready](https://img.shields.io/badge/TFLite-Pixel_10_NPU-orange.svg)](https://www.tensorflow.org/lite)

Production-grade, modular, Object-Oriented Programming (OOP) implementation of our latest tripartite multilingual speech-to-text architecture. Designed for high-accuracy multilingual recognition, dual-threaded real-time streaming, granular per-part quantization, and efficient on-device deployment (Google Pixel 10 Tensor G5 NPU).

---

## 1. Tripartite Architecture Overview

The pipeline decouples heavy acoustic modeling from linguistic word and character synthesis across three specialized sub-models:

```
Raw Speech Audio (16 kHz, Mono)
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│ MODEL 1: Model_A2P (Acoustic to Phonemes)                       │
│  - 7-layer 1D-CNN Frontend (320x downsampling -> 50 Hz frames)  │
│  - 12 Conformer Blocks with Rotary Positional Embeddings (RoPE) │
│  - DeepSeek MoE FFN (1 shared expert + 8 routed experts, top-2) │
│  - Phoneme Engram Memory (Layers 6 & 10, polynomial hash table) │
│  - Multi-Task Heads: CTC Phonemes (112), Manner Macro (8), Gate │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼ Collapsed Phoneme Stream with Acoustic Word Boundaries (Token 8 = Space)
┌─────────────────────────────────────────────────────────────────┐
│ MODEL 2: Model_PE (Phoneme Word Encoder)                        │
│  - Intra-word FastText-style Phoneme Slot Embeddings            │
│  - Lightweight 2-layer Transformer with Attention Pooling       │
│  - Maps discrete phoneme slices to continuous word latents:     │
│       z_word in R^512                                           │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼ Continuous Word Latents (z_word in R^512)
┌─────────────────────────────────────────────────────────────────┐
│ MODEL 3: Model_WD (Linguistic Word Decoder)                     │
│  - Overlapping Word-Length Experts MoE Router (7 intervals)     │
│  - Band-Causal Word Context Windows (past K=8 words)            │
│  - Autoregressive Character Micro-Head (emits bytes c_1..c_K)   │
│  - Vectorized Greedy Rollout (< 1-2 ms per word)                │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
Final Orthographic Text ("hello world ...")
```

---

## 2. Directory Structure

```
clean_model_src/
├── __init__.py                  # Public exports (Model, ModelConfig, Quantizer, Streamer, Exporter)
├── model_config.py              # Modular dataclasses and per-part device routing (CUDA/CPU)
├── model_part.py                # Abstract Base Class Model_Part(nn.Module, ABC)
├── model1_a2p.py                # Model Part 1: Model_A2P (Audio -> Phonemes + Spaces)
├── model2_pe.py                 # Model Part 2: Model_PE (Phonemes -> Word Latents z_word)
├── model3_wd.py                 # Model Part 3: Model_WD (Word Latents -> Orthographic Characters)
├── model.py                     # Unified Master Orchestrator coordinating Parts 1, 2, and 3
├── weights_manager.py           # Checkpoint versioning, model sizes, and auto-download engine
│
├── layers/                      # Fully factorized, reusable sub-modules
│   ├── activations.py           # Swish, GLU, QuickGELU
│   ├── embeddings.py            # ConvPositionalEmbedding, RotaryEmbedding (RoPE)
│   ├── conv_modules.py          # 7-layer AudioCNNFeatureExtractor, ConformerConvModule
│   ├── attention.py             # MultiHeadSelfAttention, BandCausalSelfAttention, AttentionPooling
│   ├── ffn.py                   # FeedForwardNetwork, DeepSeekMoEFFN (trace-safe top-2 gating)
│   ├── conformer_block.py       # ConformerBlock (Macaron FFN1 -> MHSA -> Conv -> FFN2/MoE -> LN)
│   ├── engram_memory.py         # PhonemeEngramMemory (multi-scale polynomial hashing, table lookup)
│   └── heads.py                 # CTCProjectionHead, ArticulatoryMacroHead, BoundaryGateHead, LengthExpertRouter
│
├── quantization/                # Granular quantization engine
│   └── quantizer.py             # ModelPartQuantizer (Dynamic INT8, FP16, weight-only int8, profiling)
│
├── streaming/                   # Dual-threaded asynchronous streaming runner
│   ├── stream_buffer.py         # Thread-safe AudioCircularBuffer and PhonemeWordQueue
│   └── dual_thread_runner.py    # DualThreadStreamingASR (Thread 1: Acoustic; Thread 2: Decoder)
│
├── export/                      # Google Pixel 10 NPU / TFLite static export
│   ├── tflite_exporter.py       # TFLiteExporter (Static ONNX graphs, operator conversion script)
│   └── mobile_runner.py         # Pixel10MobileRunner (ONNX Runtime / NNAPI delegate simulation)
│
├── scripts/                     # Production CLI Applications
│   ├── run_streaming_cli.py     # Live streaming & offline diagnostic CLI runner
│   └── __init__.py
│
├── test_clean_model.py          # End-to-end verification test suite (Tests 1 through 8)
└── README.md                    # This documentation file
```

---

## 3. Pretrained Weights & GitHub Release Assets

The model relies on three modular checkpoint assets that can be uploaded to a GitHub Release:

| Asset Filename | Model Part | Size Tier | Approx Size | Description |
| :--- | :---: | :---: | :---: | :--- |
| `pretrained_cnn_ls960.pt` | CNN | `base` | 17 MB | 7-layer 1D CNN feature extractor (LibriSpeech 960h pretrained). |
| `phono_v8_acoustic_multilingual_base.pt` | Model 1 (A2P) | `base` | 778 MB | 12-layer Conformer MoE + Engram + 112-phoneme CTC head. |
| `phono_v8_robust_p2g_base.pt` | Models 2 & 3 (P2G) | `base` | 582 MB | FastText Phoneme Encoder + 7-expert Word Decoder. |

### Model Size Tiers
- **`tiny`**: 4-layer Conformer (~25M params, ~85 MB) for micro-controllers / low-power IoT.
- **`small`**: 6-layer Conformer (~50M params, ~195 MB) optimized for Google Pixel 10 smartphone NPU.
- **`base`** *(current standard)*: 12-layer Conformer MoE (~280M total, ~70M active params) SOTA multilingual model.
- **`large`**: 24-layer Conformer MoE (~600M params) for cloud cluster inference.

### Automatic Checkpoint Downloading
The [`clean_model_src/weights_manager.py`](file:///home/nathan/github/audiolearn/clean_model_src/weights_manager.py) engine automatically discovers local files or downloads missing weights directly from GitHub Release URLs:

```python
from clean_model_src import WeightsManager

wm = WeightsManager()
# Check local status or download from registered URL:
a2p_path = wm.ensure_weights(part="a2p", size="base", version="v8.0")
p2g_path = wm.ensure_weights(part="p2g", size="base", version="v8.0")
```

You can configure release URLs via environment variables:
```bash
export PHONO_URL_CNN="https://github.com/USER/REPO/releases/download/v8.0/pretrained_cnn_ls960.pt"
export PHONO_URL_A2P_BASE="https://github.com/USER/REPO/releases/download/v8.0/phono_v8_acoustic_multilingual_base.pt"
export PHONO_URL_P2G_BASE="https://github.com/USER/REPO/releases/download/v8.0/phono_v8_robust_p2g_base.pt"
```

---

## 4. CLI Application Usage

The command-line streaming app [`clean_model_src/scripts/run_streaming_cli.py`](file:///home/nathan/github/audiolearn/clean_model_src/scripts/run_streaming_cli.py) provides live microphone streaming, audio file streaming simulation, and offline deep diagnostic profiling.

### 4.1. Fast Offline Audio File Transcription
```bash
# Basic transcription
python clean_model_src/scripts/run_streaming_cli.py path/to/audio.flac

# Offline transcription with deep diagnostic Markdown report
python clean_model_src/scripts/run_streaming_cli.py path/to/audio.flac \
  --verbose \
  --verbose_log diagnostic_report.md \
  --output_file transcript.txt
```

### 4.2. Dual-Threaded Streaming Simulation
Simulates real-time chunk-by-chunk streaming over pre-recorded files:
```bash
python clean_model_src/scripts/run_streaming_cli.py \
  --mode streaming \
  --input_audio path/to/audio.wav \
  --chunk_duration_sec 0.25 \
  --device_a2p cuda \
  --device_pe cpu \
  --device_wd cpu
```

### 4.3. Live Microphone Streaming
```bash
# 1. List detected hardware audio devices
python clean_model_src/scripts/run_streaming_cli.py --list_devices

# 2. Stream live from selected microphone
python clean_model_src/scripts/run_streaming_cli.py \
  --mode mic \
  --device_name pulse \
  --chunk_duration_sec 0.3
```

---

## 5. Diagnostic Profiler & Model Analysis

When invoked with `--verbose` or `--verbose_log <report.md>`, the system logs detailed internal representations across all layers:
1. **Frame-by-Frame Top-3 Phoneme Candidates**:
   - Time in milliseconds ($t \times 20$ ms)
   - Top-1, Top-2, and Top-3 phonemes with **IPA symbols** and **softmax probabilities**
   - Articulatory manner classification (`VOWEL`, `STOP_PLOSIVE`, `FRICATIVE`, `NASAL`, `LIQUID`)
   - Acoustic word boundary pause probability $p_{\text{boundary}} \in [0, 1]$
   - Layer 4 Draft head intermediate predictions
2. **Conformer DeepSeek MoE Routing Distribution per Layer (Layers 0..11)**:
   - Expert utilization percentages across all 8 routed experts
   - **Shannon Entropy Load Balance Score (%)** per layer to detect router collapse
3. **Model 2 Word Latent Statistics**:
   - Phoneme sequence per word, phoneme length, and latent vector stats ($\|\mathbf{z}\|_2$, mean, std)
4. **Model 3 Overlapping Length Expert MoE Routing**:
   - Chosen length interval among the 7 overlapping intervals `[1,5]` to `[22,30]`
   - Step-by-step autoregressive character rollout probabilities
5. **Architectural Improvement Insights**:
   - Automated flagging of MoE layers with imbalance (< 70% balance score)
   - Identification of phonetic ambiguity intervals (< 40% confidence)

---

## 6. Granular Quantization

The [`ModelPartQuantizer`](file:///home/nathan/github/audiolearn/clean_model_src/quantization/quantizer.py) enables precision reduction per model part or per layer type:

```python
from clean_model_src import Model, ModelPartQuantizer, QuantizationConfig

model = Model()
quantizer = ModelPartQuantizer()

# Profile memory usage before quantization
stats = quantizer.inspect_memory_footprint(model.model_a2p)
print(f"Model 1 Size: {stats['memory_mb']} MB across {stats['total_parameters']} params")

# Heterogeneous per-part quantization:
# Model 1 in INT8 for CPU/edge, Model 2 in FP16, Model 3 with 8-bit weights
config_map = {
    "A2P": QuantizationConfig(precision="qint8"),
    "PE": QuantizationConfig(precision="float16"),
    "WD": QuantizationConfig(precision="weight_only_int8"),
}
quantized_model = quantizer.quantize_model(model, config_map)
```

---

## 7. Google Pixel 10 (Tensor G5 NPU) & TFLite Export

For on-device deployment on the Google Pixel 10 smartphone:
- Converts Conformer MoE and Word Decoder models to static-shape ONNX graphs.
- Generates automated TFLite conversion scripts with INT8 Post-Training Quantization (PTQ) flags.

```python
from clean_model_src import Model, TFLiteExporter, Pixel10MobileRunner

model = Model()
exporter = TFLiteExporter(output_dir="exported_pixel10")

# Export all parts with fixed static shapes for NPU compilation
onnx_paths = exporter.export_all(model, chunk_seconds=1.0)
exporter.generate_tflite_conversion_script(onnx_paths)

# Benchmark on-device latency via ONNX Runtime / NNAPI delegate
runner = Pixel10MobileRunner(
    a2p_model_path=onnx_paths["a2p"],
    pe_model_path=onnx_paths["pe"],
    wd_model_path=onnx_paths["wd"],
)
res = runner.transcribe_audio_chunk(audio_np_16k)
print(f"Latency: {res['latency_ms']['total_ms']} ms | RTF: {res['rtf']}")
```

---

## 8. Verification Test Suite

Run the comprehensive 8-stage verification test suite:
```bash
python clean_model_src/test_clean_model.py
```
Validates:
- [x] **TEST 1**: Configuration dataclasses & dynamic device routing
- [x] **TEST 2**: Tripartite model instantiation & parameter registration
- [x] **TEST 3**: Pretrained checkpoint weight loading
- [x] **TEST 4**: Multi-task joint training forward pass ($\mathcal{L}_{\text{total}}$ computation)
- [x] **TEST 5**: Synchronous forward inference & transcription
- [x] **TEST 6**: Granular per-part and per-layer quantization
- [x] **TEST 7**: Dual-threaded asynchronous streaming runner (Thread 1 $\to$ Queue $\to$ Thread 2)
- [x] **TEST 8**: Static ONNX graph export & Pixel 10 mobile runner benchmark
