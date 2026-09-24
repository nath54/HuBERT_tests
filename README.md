# AudioLearn: HuBERT Speech Self-Supervision, Modular Architectures & XAI Studio

A modular PyTorch speech framework implementing **HuBERT** (Hidden-Unit BERT) and **PhonoHuBERT** (Direct Phoneme Prediction) from scratch. Features **0-disk streaming pre-training** in RAM via multi-speaker neural TTS, an **asynchronous multi-threaded generator with microsecond profiling**, a **Voice Quality Guardian**, scaling tiers from **8M to 95M parameters**, a **SOTA Whisper shootout benchmark**, and a full **Explainable AI (XAI)** suite with a 6-tab interactive web studio.

---

## Key Highlights & Innovations

1. **Modular Architecture Registry**:
   - **HuBERT (Acoustic K-Means SSL)**: Self-supervised pre-training via 39-dim MFCC acoustic cluster pseudo-labels (100 clusters) as in Hsu et al. (2021).
   - **PhonoHuBERT (Direct Phonemes)**: Direct acoustic-to-phoneme prediction with 64 IPA tokens and special tokens (`<same_phoneme_than_last_one>`, `<silence>`, `<noise>`, `<mask>`, `<blank>`, `<eos>`, `<unk>`).
   - **Parameter Scaling Tiers**: **Mini** (8.0M), **Small** (24.2M), **Medium** (31.8M / 48.5M), and **Base** (94.7M) with on-the-fly parameter variation and checkpoint hot-swapping.

2. **0-Disk In-RAM Streaming Pre-Training (Adapted from MADGen)**:
   - Infinite procedural speech synthesized directly in RAM across 39 verified clean neural voices (English & French).
   - **Zero Hard Drive Footprint**: Audio waveforms are synthesized, mapped to acoustic or phoneme targets, trained through PyTorch on GPU, and immediately deallocated.
   - Saves gigabytes to terabytes of disk storage over multi-hour training runs.

3. **High-Throughput Asynchronous Multi-Threaded Generator**:
   - Solves CPU synthesis vs. GPU training starvation using an asynchronous producer-consumer architecture.
   - Dedicated multi-worker synthesis pool (`BufferedSpeechBatchGenerator`) feeding a thread-safe bounded queue (`Queue(maxsize=50)`) with low-watermark thresholding (`watermark=25`).
   - Dynamic in-RAM utterance pool (250–500 items) with continuous sliding replacement and online acoustic perturbations.
   - **40× to 50× End-to-End Speedup**: Achieves 100% GPU training utilization with $< 1\text{ms}$ queue starvation latency.

4. **Dual-Thread Microsecond-Precision Profiler (`StepProfiler`)**:
   - Tracks producer timings (`time_sample_text`, `time_piper_synth`, `time_retry_error`, `time_target_extract`, `time_batch_collate`, `time_queue_put_wait`).
   - Tracks consumer timings (`time_queue_get_wait`, `time_device_transfer`, `time_forward`, `time_loss`, `time_backward`, `time_optimizer_step`).
   - Formats ASCII latency breakdown tables to the console and streams real-time telemetry to the Web UI.

5. **Voice Quality Guardian (`VoiceQualityGuardian`)**:
   - Intercepts text before synthesis and validates it against each Piper ONNX model's phoneme map.
   - Automatically tracks warning counts per voice and dynamically quarantines defective models into a persistent blocklist.

6. **Interactive 6-Tab Web Studio (FastAPI + Tailwind CSS + HTML5 Canvas)**:
   - Live PyTorch inference, interactive spectrogram & layer-by-layer feature maps (CNN to L8).
   - Real-time Captum Integrated Gradients & Saliency on any predicted token.
   - Causal activation patching (NAPS) with instant layer ablation.
   - Real LibriSpeech dataset explorer (2,703 utterances) & test-clean benchmark (2,620 utterances).
   - SOTA comparative shootout vs. OpenAI Whisper (Base, Small, Medium).
   - Live pre-training dashboard with cumulative audio gauges, loss/accuracy curves, queue buffer bar, and profiler timings.

---

## System Architecture

```
                       ┌─────────────────────────────────────────────────────────────┐
                       │           0-Disk In-RAM Streaming Speech Generator          │
                       │  Procedural Text Sampler (100k words) -> Piper Neural TTS   │
                       │   39 Clean Voices (EN / FR) | Voice Quality Guardian Filter │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      │ Padded Waveform (16 kHz, Mono)
                                                      ▼
                       ┌─────────────────────────────────────────────────────────────┐
                       │  HuBERTFeatureEncoder (Temporal 1D Convolution)             │
                       │  7-layer 1D CNN with LayerNorm & GELU (Strides: 5,2,2,2...) │
                       │  Downsampling Factor: 320x (20ms frames / 50Hz)             │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      │ CNN Representations (B, T, C)
                                                      ▼
                       ┌─────────────────────────────────────────────────────────────┐
                       │  Feature Projection & LayerNorm (Linear: C -> embed_dim)    │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       │                                                             │
                       ▼                                                             ▼
     [HuBERT: K-Means SSL Branch]                                   [PhonoHuBERT: Direct Phoneme Branch]
     - Span Masking (65% of frames)                                 - Dual Loss / Unmasked CTC Alignment
     - 4 to 12 Transformer Layers                                   - 4 to 12 Transformer Layers
     - 100 Acoustic Cluster Units (MFCC)                            - 64 IPA Phoneme Tokens (<blank>, <mask...>
     - Cross-Entropy Masked Loss                                    - Direct Acoustic-to-Phoneme Head
                       │                                                             │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      │
                                                      ▼
                       ┌─────────────────────────────────────────────────────────────┐
                       │  Downstream CTC ASR / Phoneme Decoder & XAI Attribution     │
                       │  Captum Integrated Gradients | NAPS Ablation | SOTA Bench   │
                       └─────────────────────────────────────────────────────────────┘
```

---

## Parameter Scaling Tiers

| Tier | Transformer Layers | Attention Heads | Embedding Dim ($D$) | FFN Dim ($4D$) | Parameters (HuBERT) | Parameters (PhonoHuBERT) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mini** | 4 | 4 | 256 | 1024 | **8.04 M** | **8.06 M** |
| **Small** | 6 | 6 | 384 | 1536 | **24.21 M** | **24.23 M** |
| **Medium** | 8 | 8 | 512 | 2048 | **48.52 M** | **31.82 M** |
| **Base** | 12 | 12 | 768 | 3072 | **94.68 M** | **94.73 M** |

---

## Directory Structure

```
audiolearn/
├── .venv/                      # Python 3.12 Virtualenv (CUDA 12 + PyTorch)
├── configs/                    # YAML configuration files
│   ├── hubert_base.yaml        # HuBERT architecture hyperparameters
│   ├── train_asr.yaml          # CTC training hyperparameters
│   └── xai_config.yaml         # XAI parameters
├── data/                       # Datasets & manifests
│   ├── librispeech/            # LibriSpeech train & test-clean manifests (2,703 + 2,620 utterances)
│   ├── raw/synthetic/          # Synthetic speech samples & manifests
│   └── sample_dataset.py       # Audio dataset synthesizer & loader
├── src/                        # Core codebase
│   ├── models/                 # Model implementations & registry
│   │   ├── config.py           # HuBERTConfig dataclass
│   │   ├── cnn_encoder.py      # 7-layer Temporal 1D CNN Feature Extractor
│   │   ├── transformer.py      # Pos-conv embeddings + MHSA + Pre-LN FFN
│   │   ├── hubert_asr.py       # Full HuBERTForCTC model + greedy CTC decoder
│   │   ├── hubert_pretrain.py  # HuBERT masked acoustic cluster SSL model
│   │   ├── phono_hubert.py     # PhonoHuBERT direct phoneme model with special tokens
│   │   └── registry.py         # ModelRegistry factory for dynamic discovery & scaling
│   ├── data/                   # Data pipelines & tokenization
│   │   ├── tokenizer.py        # Character CTC Tokenizer (31 tokens)
│   │   ├── phoneme_tokenizer.py# Bilingual (EN/FR) IPA Tokenizer (64 tokens + 8 special tokens)
│   │   ├── target_extractors.py# KMeansUnitExtractor & PhonemeTargetExtractor
│   │   ├── streaming_piper.py  # 0-disk Piper voice manager & VoiceQualityGuardian
│   │   ├── threaded_dataset.py # Asynchronous BufferedSpeechBatchGenerator & StepProfiler
│   │   ├── dataset.py          # AudioASRDataset + dynamic padding collate fn
│   │   └── augmentations.py    # Waveform noise, gain, and time-masking
│   ├── benchmark/              # Comparative benchmarking suite
│   │   └── sota_evaluator.py   # SOTABenchmarkRunner (HuBERT vs. OpenAI Whisper Shootout)
│   ├── server/                 # Full-stack interactive web application
│   │   ├── app.py              # FastAPI backend (Inference, XAI, Streaming, Pretrain, SOTA)
│   │   └── static/             # Frontend single-page app
│   │       └── index.html      # 6-Tab Web Studio (Tailwind CSS, Canvas charts, gauges)
│   ├── training/               # Fine-tuning & trainer utilities
│   │   ├── trainer.py          # HuBERT ASR Trainer (AMP, clipping, checkpointing)
│   │   └── metrics.py          # CER, WER, and MetricTracker
│   ├── xai/                    # Explainable AI suite
│   │   ├── captum_gradients.py # Captum Integrated Gradients & Saliency
│   │   ├── naps.py             # Neuron Activation Profiles, Patching & Saliency
│   │   ├── probes.py           # Layer-wise Diagnostic Probes & Inversion Decoders
│   │   └── visualizer.py       # Matplotlib visualization suite
│   └── utils/                  # Audio I/O & logging utilities
├── checkpoints/                # Saved weights (HuBERT, PhonoHuBERT, best_model.pt)
│   ├── phono_hubert/medium/    # Checkpoints for PhonoHuBERT Medium
│   └── hubert_kmeans/mini/     # Checkpoints for HuBERT K-Means Mini
├── logs/                       # Real-time status JSONs & TensorBoard telemetry
├── scripts/                    # Command-line entry points
│   ├── run_server.py           # Start the FastAPI interactive studio server
│   ├── run_pretrain.py         # Unified modular 0-disk streaming pre-training CLI
│   ├── train.py                # Supervised CTC fine-tuning on LibriSpeech
│   ├── evaluate.py             # Checkpoint evaluator
│   ├── run_xai.py              # Generate static XAI visualization plots
│   └── demo_pipeline.py        # Automated end-to-end master pipeline
├── tests/                      # Pytest automated test suite (20 tests)
│   ├── test_model.py
│   ├── test_data.py
│   ├── test_phono_architecture.py
│   ├── test_threaded_dataset.py
│   ├── test_voice_guardian.py
│   └── test_xai.py
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Environment Setup

The project uses **Python 3.12** and **PyTorch with CUDA**:

```bash
# Clone the repository
git clone https://github.com/nathan/audiolearn.git
cd audiolearn

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install -r requirements.txt
```

### 2. Launch the Interactive Web Studio

Start the FastAPI application and open your browser:

```bash
.venv/bin/python scripts/run_server.py --port 8000
```
Navigate to **http://localhost:8000** to access the 6 tabs:
1. **Studio**: Real-time PyTorch synthesis, layer feature maps, attention inspector, Captum Integrated Gradients, and causal NAPS layer ablation.
2. **Dataset**: Real LibriSpeech browser (2,703 utterances) with audio player and transcript inspect.
3. **Training**: Downstream ASR CTC fine-tuning console with real-time CER/WER convergence curves.
4. **Weights & Scaling Laws**: SVD rank analysis, parameter distributions, and scaling tier comparison.
5. **SOTA Benchmark**: Side-by-side shootout comparing AudioLearn models against OpenAI Whisper (Base, Small, Medium).
6. **Piper SSL Pre-train (0-Disk)**: Live streaming pre-training console with cumulative in-RAM audio gauges, buffer occupancy bar, and microsecond profiler metrics.

---

## Pre-Training Speech Models (0-Disk Streaming)

Run self-supervised pre-training using multi-speaker neural speech synthesized entirely in RAM:

### Pre-Train PhonoHuBERT (Direct Phonemes)
```bash
# PhonoHuBERT Medium (31.8M params) with 6 worker threads and bounded buffer
.venv/bin/python scripts/run_pretrain.py \
    --arch phono_hubert \
    --tier medium \
    --batch_size 8 \
    --num_workers 6 \
    --buffer_size 50 \
    --watermark 25 \
    --steps 625 \
    --lr 0.0001 \
    --resume
```

### Pre-Train HuBERT (Acoustic K-Means SSL)
```bash
# HuBERT Mini (8.0M params) with 100 acoustic clusters
.venv/bin/python scripts/run_pretrain.py \
    --arch hubert_kmeans \
    --tier mini \
    --batch_size 8 \
    --num_workers 4 \
    --buffer_size 20 \
    --watermark 10 \
    --steps 500
```

### Key Pre-Training CLI Options
- `--arch`: `phono_hubert` (direct phonemes) or `hubert_kmeans` (acoustic cluster units).
- `--tier`: `mini`, `small`, `medium`, or `base`.
- `--override`: Custom parameter overrides (e.g. `--override "mask_prob=0.5,encoder_layers=10"`).
- `--num_workers`: Number of parallel Piper synthesis threads on CPU.
- `--buffer_size`: Max batches in the bounded queue (default: `20` or `50`).
- `--watermark`: Low watermark batch count to resume worker synthesis (default: `10` or `25`).
- `--use_rolling_pool` / `--no_rolling_pool`: Toggle the dynamic in-RAM utterance pool.
- `--pool_size`: Number of synthesized utterances maintained in RAM (default: `250`).
- `--resume`: Auto-resume from `checkpoint_latest.pt`.

---

## Explainable AI (XAI) Suite

AudioLearn implements three interpretability paradigms:

### 1. Gradient-Based Attribution (via Captum)
- **Integrated Gradients (`captum.attr.IntegratedGradients`)**: Path integral of gradients from a silence baseline to the input speech waveform.
- **Saliency (`captum.attr.Saliency`)**: First-order input gradient magnitude.
- **Layer Integrated Gradients**: Attributes token predictions back to specific intermediate Transformer layers.

### 2. NAPS (Neuron Activation Profiles & Causal Patching)
- **Neuron Activation Profiles (NAP)**: Profiles FFN expansion activations across phonetic conditions.
- **Neuron Selectivity Index**: Locates specialized acoustic neurons (vowels, fricatives, silence).
- **Causal Activation Patching / Ablation**: Dynamically zeroes out layers or skip connections in memory to quantify causal degradation on logits.

### 3. Diagnostic Probing & Acoustic Inversion
- **Linear Diagnostic Probes**: Measures where acoustic energy vs. phonetic identity is encoded across layers.
- **Acoustic Inversion**: Reconstructs filterbanks from frozen representations, demonstrating feature abstraction from acoustics to symbolic tokens.

---

## Automated Testing

AudioLearn includes a 20-test suite covering data synthesis, model architectures, the Voice Quality Guardian, the threaded batch generator, and XAI attribution:

```bash
PYTHONPATH=. .venv/bin/pytest tests/ -v
```

All 20 tests pass in $< 20\text{s}$ on CPU/GPU.

---

## License

This project is licensed under the Apache 2.0 License. Model weights and synthetic audio pipelines are provided for educational and research purposes.
