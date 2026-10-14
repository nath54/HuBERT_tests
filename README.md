# AudioLearn: HuBERT Speech Self-Supervision, Modular Architectures & XAI Studio

A modular PyTorch speech framework implementing **HuBERT** (Hidden-Unit BERT) and **PhonoHuBERT** (Direct Phoneme Prediction) from scratch. Features **dual data streaming pipelines** (0-disk in-RAM procedural neural TTS synthesis for initial exploration, and 100% genuine LibriSpeech human speech streaming for SOTA acoustic scaling), an **asynchronous multi-threaded generator with microsecond profiling**, a **Voice Quality Guardian**, scaling tiers from **8M to 95M parameters**, a **SOTA Whisper shootout benchmark**, and a full **Explainable AI (XAI)** suite with a 6-tab interactive web studio.

---

## Key Highlights & Innovations

1. **Evolutionary Modular Architecture Family**:
   - **HuBERT SSL**: Acoustic $k$-means self-supervised pre-training via 39-dim MFCC cluster pseudo-labels (100 clusters).
   - **PhonoHuBERT (V1–V2)**: Direct acoustic-to-phoneme prediction with 64 IPA tokens, articulatory macro-classes, and SpecAugment.
   - **Phono-V3–V5**: Scaled 8-layer Transformer backbones, hybrid human/synthetic training, and phonotactic prefix beam search with 3-gram language model priors.
   - **Phono-V6 Progressive Suite**: Top-2 sparse Mixture-of-Experts (V6.1), 30s sliding sparse attention + InterCTC (V6.2), and continuous latent phoneme diffusion (V6.3–V6.4).
   - **Phono-V7 Streaming & Decoupled Decoders**: Strictly causal streaming (V7.1), binary silence gating with 16 partitioned micro experts (V7.2), multi-scale dilated Conformer convolutions with decoupled boundary gates (V7.3), and the **Phono-V7.5 Tripartite Modular Decoupled Model** (Acoustic front-end + Distilled latent bridge + 7-expert overlapping length MoE decoder).
   - **Phono-V8.0 Flagship**: Scaled Conformer with DeepSeek MoE (1 shared + 8 routed experts), multi-order FactorEngram associative parameter memory, and an autonomous verbatim corpus sanitization pipeline.

2. **📚 Architecture Documentation Hub (`docs/`)**:
   Every architecture in AudioLearn is accompanied by a dedicated, **self-contained and non-dependent** technical document in [`docs/architectures/`](docs/architectures/).

| Architecture | Paradigm & Core Innovation | Self-Contained Documentation |
| :--- | :--- | :--- |
| **HuBERT** | Acoustic K-Means SSL with 39-dim MFCC cluster pseudo-labels | [📖 `docs/architectures/01_hubert_ssl.md`](docs/architectures/01_hubert_ssl.md) |
| **PhonoHuBERT (V1–V2)** | Direct acoustic-to-phoneme prediction with 64 IPA tokens and articulatory hierarchies | [📖 `docs/architectures/02_phono_hubert_direct_ctc.md`](docs/architectures/02_phono_hubert_direct_ctc.md) |
| **Phono-V3–V5** | Scaled 8-layer backbone, hybrid training, and CTC prefix beam search with phonotactic LM | [📖 `docs/architectures/03_phono_scaled_and_beam.md`](docs/architectures/03_phono_scaled_and_beam.md) |
| **Phono-V6.1–V6.4** | Sparse MoE Transformer, 30s sliding attention + InterCTC, and latent phoneme continuous diffusion | [📖 `docs/architectures/04_phono_v6_moe_diffusion.md`](docs/architectures/04_phono_v6_moe_diffusion.md) |
| **Phono-V6.7** | Continuous speech double-loss model (CTC phoneme backbone + character decoder) | [📖 `docs/architectures/05_phono_v6_7_continuous_speech.md`](docs/architectures/05_phono_v6_7_continuous_speech.md) |
| **Phono-V7.0** | Conformer backbone + two-level character decoder with DP forced alignment & Soft-Levenshtein | [📖 `docs/architectures/06_phono_v7_0_hierarchical_conformer.md`](docs/architectures/06_phono_v7_0_hierarchical_conformer.md) |
| **Phono-V7.1** | Real-time causal streaming model with band-causal macro attention and online peak slicing | [📖 `docs/architectures/07_phono_v7_1_causal_streaming.md`](docs/architectures/07_phono_v7_1_causal_streaming.md) |
| **Phono-V7.2** | Binary non-speech gating and 16-expert partitioned length MoE with dynamic horizon capping | [📖 `docs/architectures/08_phono_v7_2_partitioned_moe.md`](docs/architectures/08_phono_v7_2_partitioned_moe.md) |
| **Phono-V7.3** | Multi-scale dilated Conformer convolutions ($4\times$ receptive field) and decoupled boundary gate | [📖 `docs/architectures/09_phono_v7_3_dilated_boundary.md`](docs/architectures/09_phono_v7_3_dilated_boundary.md) |
| **Phono-V7.5** | Tripartite modular decoupling: acoustic front-end, FastText latent bridge, and 7-expert MoE decoder | [📖 `docs/architectures/10_phono_v7_5_tripartite_modular.md`](docs/architectures/10_phono_v7_5_tripartite_modular.md) |
| **Phono-V8.0** | Acoustic Conformer with DeepSeek MoE (1 shared + 8 routed) and FactorEngram associative memory | [📖 `docs/architectures/11_phono_v8_0_deepseek_moe_engram.md`](docs/architectures/11_phono_v8_0_deepseek_moe_engram.md) |

---

## Parameter Scaling Tiers

| Tier | Transformer Layers | Attention Heads | Embedding Dim ($D$) | FFN Dim ($4D$) | Parameters (HuBERT / Phono) | Parameters (Phono-V6 MoE / Diffusion) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mini** | 4 | 4 | 256 | 1024 | **8.06 M** | **18.42 M** |
| **Small** | 6 | 6 | 384 | 1536 | **24.23 M** | **45.18 M** |
| **Medium** | 8 | 8 | 512 | 2048 | **31.82 M** | **83.55 M** (Active SOTA) |
| **Base** | 12 | 12 | 768 | 3072 | **94.73 M** | **172.40 M** |

---

## Current Flagship Architectures: Phono-V7.5 & Phono-V8.0

AudioLearn's production tier features two complementary flagship architectures dividing speech processing into specialized, highly optimized stages:

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 Phono-V8.0 ACOUSTIC                    │
                  │   DeepSeek MoE (1 Shared + 8 Routed Experts, Top-2)    │
                  │     + Multi-Order FactorEngram Memory Retrieval        │
                  │        (2-Gram, 3-Gram, 4-Gram Prime Hashing)          │
                  └───────────────────────────┬────────────────────────────┘
                                              │ Pristine Phonemes / Acoustic Latents
                                              ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                   Phono-V7.5 ASR SYSTEM                                  │
│                                                                                          │
│  [Module 1: Acoustic Front-End] ──► [Acoustic Latents z_word]                            │
│                                              ▲                                           │
│  [Module 2: Distilled Latent Bridge] ────────┤ (98.7% Cosine Similarity @ 50x speed)     │
│  (FastTextPhonemeWordEncoder, 1.5M params)   │                                           │
│                                              ▼                                           │
│  [Module 3: Dual-Level MoE Text Decoder] (51.7M params)                                  │
│  • Level 1: Macro Word Decoder with 7 Overlapping Length MoE Experts (Routing: 98.6%)     │
│  • Level 2: Micro Character Decoder with 16 Partitioned MoE Experts + Horizon Capping    │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 1. Phono-V7.5: Complete Tripartite Modular Decoupled Speech-to-Text Model
Phono-V7.5 completely supersedes monolithic speech recognition by establishing a **fully decoupled, 3-module architecture**:
1. **Module 1 (Acoustic Phoneme Front-End, 31.8M params)**: 7-layer 1D CNN + 8-layer Conformer backbone with 4 MoE experts, multi-scale dilated convolutions ($4\times$ temporal receptive field, 2,420ms), decoupled word boundary gate head (`SPEECH`, `WORD_SPACE`, `SILENCE_BLANK`), and recursive 2-pass phoneme head.
2. **Module 2 (Cross-Modal Distilled Latent Bridge, 1.51M params)**: Ultra-fast 2-layer Transformer (`FastTextPhonemeWordEncoder`) projecting text bytes and phonemes to the canonical acoustic latent space with **98.7% Cosine Similarity**, operating **$50\times$ faster** than audio Conformer processing.
3. **Module 3 (Dual-Level MoE Recursive Character Decoder, 51.7M params)**: Level-1 Macro Word Decoder with 7 overlapping length experts and asymmetric truncation penalty, plus Level-2 Micro Character Decoder with 16 partitioned experts and dynamic horizon capping.
* **Text Middle-Training Breakthrough**: Pre-trained on >208,000 multilingual sentences across English, Italian, Spanish, and French in pure FP32, reaching **95.5% character accuracy**, **98.0% routing accuracy**, and a record validation loss of **`0.8496`**.

👉 **Full Technical Specification & Equations**: See [`docs/architectures/10_phono_v7_5_tripartite_modular.md`](docs/architectures/10_phono_v7_5_tripartite_modular.md).

---

### 2. Phono-V8.0: Scaled Acoustic Conformer with DeepSeek MoE & FactorEngram Memory
Phono-V8.0 is engineered to reach the **$< 5\%$ multilingual Phoneme Error Rate (PER)** milestone across 4 European languages (EN, FR, ES, IT) without language ID conditioning vectors:
1. **DeepSeek MoE Backbone (1 Shared + 8 Routed Experts)**: 1 always-active shared expert handles language-invariant acoustic phonetics, while 8 fine-grained routed experts (Top-2 gating) specialize in manner of articulation. Total model capacity: **203.69M parameters** (~103M active compute parameters per frame).
2. **FactorEngram Multi-Order Phoneme Memory (`PhonemeEngramMemory`)**: Stores static phonotactic transition priors in $O(1)$ constant-time parameter tables via 4-head polynomial prime hashing across 2-grams (16k), 3-grams (32k), and 4-grams (32k). FactorEngram channel-wise gates ($g_t \in [0, 1]^{512}$) inject phonotactic memory residually at Layers 4 and 7 based on Layer 3 draft candidate emissions.
3. **Top-3 Phoneme Candidate Emissions with Exact Probabilities (`decode_topk_phonemes`)**: For every emitted acoustic segment, Model 1 extracts the top 3 candidate phonemes along with their softmax confidence probabilities (e.g. `t -> [t: 0.61 | n: 0.13 | z: 0.13]`), revealing phonetic ambiguities and providing rich soft distributions for downstream lexical disambiguation.
4. **Space-Free Continuous Phoneme-to-Word Architecture (`ContinuousPhonemeWordSegmenter`)**: In conversational speech, acoustic silences do not physically exist at every word boundary. Rather than forcing Model 1 to predict fragile `<space>` tokens at 50Hz, Model 1 emits continuous phonemes; Model 2 uses bidirectional Transformer contextual representations and boundary gating ($b_t > \tau$) to slice word spans dynamically and pool them into discrete word latents $\mathbf{z}_{\text{word}}$.
5. **Phonotactic Beam Search & Noise Robustness**: `PhonotacticBeamSearchDecoder` integrates prefix trie constraints and syllable onset/nucleus/coda transition priors. Paired with `PhoneticNoiseInjector`, models are trained with realistic articulatory confusions for robust acoustic-to-grapheme decoding.
6. **Bit-for-Bit Zero-Initialization Warm-Start**: Strict zero-initialization of shared expert and memory projections guarantees exact mathematical equivalence at Step 0 ($\max |\Delta| \le 7.63 \times 10^{-6}$ vs `best_acoustic.pt`), eliminating warm-start shocks.
7. **Autonomous Verbatim Corpus Sanitization**: High-throughput acoustic filtering pipeline ([`scripts/clean_and_filter_corpus.py`](scripts/clean_and_filter_corpus.py)) retaining **82,601 pristine training utterances** and **828 clean validation utterances**.

👉 **Full Technical Specification & Equations**: See [`docs/architectures/11_phono_v8_0_deepseek_moe_engram.md`](docs/architectures/11_phono_v8_0_deepseek_moe_engram.md).

---

## Directory Structure

```
audiolearn/
├── .venv/                      # Python 3.12 Virtualenv (CUDA 12 + PyTorch)
├── configs/                    # YAML configuration files
│   ├── hubert_base.yaml        # HuBERT architecture hyperparameters
│   ├── train_asr.yaml          # CTC training hyperparameters
│   └── xai_config.yaml         # XAI parameters
├── docs/                       # Self-contained Architecture Documentation Hub
│   ├── README.md               # Architecture navigation map & design principles
│   └── architectures/          # Dedicated standalone docs for models 01 to 11
├── data/                       # Datasets & manifests
│   ├── librispeech/            # LibriSpeech train & test-clean manifests (2,703 + 2,620 utterances)
│   ├── raw/synthetic/          # Synthetic speech samples & manifests
│   └── sample_dataset.py       # Audio dataset synthesizer & loader
├── src/                        # Core codebase
│   ├── models/                 # Model implementations & registry
│   │   ├── config.py           # HuBERTConfig dataclass
│   │   ├── cnn_encoder.py      # 7-layer Temporal 1D CNN Feature Extractor
│   │   ├── conformer_layers.py # Conformer depthwise convolution & feed-forward blocks
│   │   ├── transformer.py      # Pos-conv embeddings + MHSA + Pre-LN FFN
│   │   ├── hubert_asr.py       # Full HuBERTForCTC model + greedy CTC decoder
│   │   ├── hubert_pretrain.py  # HuBERT masked acoustic cluster SSL model
│   │   ├── phono_hubert.py     # PhonoHuBERT direct phoneme model with special tokens
│   │   ├── phono_variants.py   # V6.1 MoE, V6.2 Sparse, V6.3 Diffusion & V6.4 Gated Diffusion
│   │   ├── word_denoising_decoder.py # MoE Banded Cross-Attention Word Diffusion Decoder
│   │   ├── forced_aligner.py   # CTC dynamic programming forced aligner
│   │   ├── phono_v7_alignment.py # Offline CTC peak & acoustic word slicing
│   │   ├── phono_v7_speech_model.py # Phono-V7 Conformer + Two-Level Windowed Character Decoder
│   │   ├── phono_v7_1_alignment.py # Online event-driven streaming peak tracker
│   │   ├── phono_v7_1_speech_model.py # Phono-V7.1 Real-Time Streaming Conformer Speech Model
│   │   ├── phono_v7_2_speech_model.py # Phono-V7.2 Partitioned MoE Decoder with Horizon Capping
│   │   ├── phono_v8_acoustic.py# Phono-V8.0 DeepSeek MoE Conformer Backbone + Top-3 candidate emission
│   │   ├── text_phoneme_encoder.py # FastTextPhonemeWordEncoder & ContinuousPhonemeWordSegmenter
│   │   ├── phonotactic_beam_search.py # Phonotactic beam search decoder with trie constraints
│   │   └── registry.py         # ModelRegistry factory for dynamic discovery & scaling
│   ├── losses/                 # Differentiable loss implementations
│   │   └── soft_levenshtein.py # Differentiable Soft-Levenshtein DP string alignment loss
│   ├── data/                   # Data pipelines & tokenization
│   │   ├── tokenizer.py        # Character CTC Tokenizer (31 tokens)
│   │   ├── phoneme_tokenizer.py# Bilingual (EN/FR) IPA Tokenizer (64 tokens + 8 special tokens)
│   │   ├── multilingual_lexicon.py # Offline pronunciation lexicon for EN, IT, ES, FR
│   │   ├── multilingual_audio_dataset.py # 4-way balanced multilingual dataset loader & collator
│   │   ├── target_extractors.py# KMeansUnitExtractor & PhonemeTargetExtractor
│   │   ├── phonetic_noise.py   # Articulatory confusion matrix & acoustic noise injection
│   │   ├── phonotactic_segmenter.py # Linguistic syllable onset/nucleus/coda segmenter
│   │   ├── streaming_piper.py  # 0-disk Piper voice manager & VoiceQualityGuardian
│   │   ├── threaded_dataset.py # Asynchronous BufferedSpeechBatchGenerator & StepProfiler
│   │   ├── dataset.py          # AudioASRDataset + dynamic padding collate fn
│   │   └── augmentations.py    # Waveform noise, gain, and time-masking
│   ├── inference/              # Decoding & search engines
│   │   └── character_beam_search.py # Lexical character beam search with trie prefix guidance
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
├── checkpoints/                # Saved weights (HuBERT, PhonoHuBERT, V7, V7.5, V8.0)
│   ├── phono_v7/char/          # Phono-V7 Two-Level Character Decoder checkpoints
│   ├── phono_v7_1/streaming/   # Phono-V7.1 Real-Time Streaming checkpoints
│   ├── phono_v7_2/streaming/   # Phono-V7.2 Partitioned MoE Streaming checkpoints
│   ├── phono_v7_5_acoustic/    # Phono-V7.5 Acoustic Conformer checkpoints
│   ├── phono_v8_acoustic/      # Phono-V8.0 Baseline checkpoints
│   └── phono_v8_acoustic_multilingual_80k/ # Phono-V8.0 Pure Model 1 Multilingual 80k checkpoints
├── logs/                       # Real-time status JSONs & TensorBoard telemetry
├── scripts/                    # Command-line entry points
│   ├── run_server.py           # Start the FastAPI interactive studio server
│   ├── run_pretrain.py         # Unified modular 0-disk streaming pre-training CLI
│   ├── train_phono_v8_acoustic.py # Pure Model 1 Multilingual 80,000-Step training with Top-3 logs
│   ├── train_phono_v8_continuous_p2g.py # Continuous space-free phoneme-to-grapheme training
│   ├── train_phono_v8_robust_p2g.py # Noise-robust phoneme-to-grapheme training
│   ├── evaluate_phono_v8_e2e.py# End-to-end multi-module evaluation suite
│   ├── audit_phonetic_confusions.py # Acoustic phoneme confusion matrix auditor
│   ├── train_phono_v7_stage1.py# Phono-V7 Stage 1 Conformer & alignment pre-training
│   ├── train_phono_v7_char.py  # Phono-V7 Stage 2 Two-Level Character Decoder training
│   ├── train_phono_v7_1_streaming.py # Phono-V7.1 Real-Time Streaming training
│   ├── train_phono_v7_2_streaming.py # Phono-V7.2 Partitioned MoE Streaming training
│   ├── train.py                # Supervised CTC fine-tuning on LibriSpeech
│   ├── evaluate.py             # Checkpoint evaluator
│   ├── run_xai.py              # Generate static XAI visualization plots
│   └── demo_pipeline.py        # Automated end-to-end master pipeline
├── tests/                      # Pytest automated test suite (100+ tests)
│   ├── test_phonetic_noise.py  # Articulatory noise injection tests
│   ├── test_phonotactic_beam_search.py # Phonotactic beam search decoder tests
│   ├── test_phonotactic_segmenter.py # Syllable segmentation tests
│   ├── test_text_phoneme_encoder.py # FastText and continuous word encoder tests
│   ├── test_conformer_conv.py  # Conformer depthwise convolution tests
│   ├── test_forced_aligner.py  # CTC dynamic programming aligner tests
│   ├── test_multilingual_lexicon.py # Pronunciation lexicon tests
│   ├── test_soft_levenshtein.py # Differentiable Soft-Levenshtein loss tests
│   ├── test_phono_v7_model.py  # Phono-V7 two-level decoder architecture tests
│   ├── test_phono_v7_alignment.py # V7 acoustic slicing & peak detection tests
│   ├── test_phono_v7_1_streaming.py # V7.1 real-time streaming & band-causal tests
│   ├── test_phono_v7_2_model.py # Phono-V7.2 Partitioned MoE & horizon capping tests
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

## Pre-Training Speech Models (Universal Standardized Protocol)

To ensure rigorous, fair, and scientific comparisons across all architectures in AudioLearn, **every architecture in the registry is now pre-trained under the exact same standardized benchmark setup**:
- **Acoustic Speech Source**: 100% genuine human speech from LibriSpeech 100h clean (`--real_ratio 1.0`, default).
- **Long-Range Context Window**: Up to **30.0 seconds** per sample (`--max_duration_sec 30.0`).
- **High-Throughput Async Engine**: GIL-free multi-threaded batch generator (`futex_wait`) sustaining 94%–100% GPU compute saturation with 0.0 ms starvation latency.
- **SSD Wear Protection**: Zero intermediate 1GB checkpoint writes during training (`--save_interval 4000`), persisting weights strictly at milestones and upon graceful `Ctrl+C` interrupt.
- *(Note: Any model can alternatively run in 0-disk procedural neural TTS mode simply by passing `--real_ratio 0.0`).*

### 1. PhonoHuBERT (Dense Standard Baseline)
Direct acoustic-to-phoneme prediction with standard dense self-attention under the 30s LibriSpeech setup:
```bash
.venv/bin/python scripts/run_pretrain.py \
    --arch phono_hubert \
    --tier medium \
    --batch_size 4 \
    --num_workers 4 \
    --steps 4000 \
    --max_duration_sec 30.0 \
    --run_name phono_hubert_std_run1
```

### 2. PhonoHuBERT-Hierarchical (Acoustic Router + Phoneme Head)
Two-stage gated architecture decomposing decoding into an Acoustic State Router (`blank`, `silence`, `noise`, `speech`) and a pure Linguistic Phoneme Head:
```bash
.venv/bin/python scripts/run_pretrain.py \
    --arch phono_hubert_hierarchical \
    --tier medium \
    --batch_size 4 \
    --num_workers 4 \
    --steps 4000 \
    --max_duration_sec 30.0 \
    --run_name hierarchical_std_run1
```

### 3. PhonoHuBERT-Dual (Masked Frame SSL + Sequence CTC)
Dual-loss speech Transformer pairing frame-synchronous masked phoneme Cross-Entropy with auxiliary CTC sequence alignment:
```bash
.venv/bin/python scripts/run_pretrain.py \
    --arch phono_hubert_dual \
    --tier medium \
    --batch_size 4 \
    --num_workers 4 \
    --steps 4000 \
    --max_duration_sec 30.0 \
    --masking_mode span \
    --mask_prob 0.4 \
    --run_name dual_loss_std_run1
```

### 4. PhonoHuBERT-Recursive (Recurrent Temporal Frame Feedback)
Autoregressive recurrent frame-memory feedback head breaking CTC conditional independence and explicitly modeling sustained-phoneme durations:
```bash
.venv/bin/python scripts/run_pretrain.py \
    --arch phono_hubert_recursive \
    --tier medium \
    --batch_size 4 \
    --num_workers 4 \
    --steps 4000 \
    --max_duration_sec 30.0 \
    --run_name recursive_std_run1
```

### 5. Phono-V6.1 (Hierarchical Mixture-of-Experts)
Sparse MoE speech Transformer featuring 4 FFN experts per layer, Top-2 load-balanced gating, and hierarchical factored heads:
```bash
.venv/bin/python scripts/run_pretrain.py \
    --arch phono_v6_1_moe \
    --tier medium \
    --batch_size 4 \
    --num_workers 4 \
    --steps 4000 \
    --max_duration_sec 30.0 \
    --run_name v6_1_moe_std_run1
```

### 6. Phono-V6.2 (Sparse Attention + 30s Context + Intermediate CTC)
Breakthrough architecture combining sparse sliding attention ($\pm 320$ms), Intermediate CTC multi-task supervision across layers 4 & 8, and up to 30-second context audio on genuine LibriSpeech:
```bash
.venv/bin/python scripts/run_pretrain.py \
    --arch phono_v6_2_sparse \
    --tier medium \
    --batch_size 4 \
    --num_workers 4 \
    --steps 4000 \
    --lr 0.0003 \
    --min_duration_sec 1.0 \
    --max_duration_sec 30.0 \
    --real_ratio 1.0 \
    --save_interval 4000 \
    --run_name v6_2_sparse_100h_run1
```

### 7. Phono-V6.3 (Sliding Gaussian Latent Diffusion Refiner)
Latent diffusion refiner applying a spatio-temporal Gaussian noise envelope in continuous frame space with FiLM conditioning and a trailing-window streaming phoneme decoder:
```bash
.venv/bin/python scripts/run_pretrain.py \
    --arch phono_v6_3_diffusion \
    --tier medium \
    --warm_start checkpoints/phono_v6_2_sparse/medium/v6_2_sparse_100h_run1/checkpoint_step_4000.pt \
    --batch_size 4 \
    --num_workers 4 \
    --steps 4000 \
    --lr 0.00005 \
    --min_duration_sec 1.0 \
    --max_duration_sec 30.0 \
    --real_ratio 1.0 \
    --save_interval 4000 \
    --run_name v6_3_diffusion_run1
```

### 8. Phono-V6.4 (Confidence-Gated Diffusion & 960h Scaling Run)
Pre-train or resume on the full 960-hour deterministically interleaved LibriSpeech corpus with learnable gate thresholds and SSD wear protection:
```bash
# Launch Stage 1 (Initial 30,000 steps on 960h dataset)
.venv/bin/python scripts/run_pretrain.py \
    --arch phono_v6_4_gated_diffusion \
    --tier medium \
    --run_name v6_4_960h \
    --steps 30000 \
    --eval_interval 500 \
    --val_samples 50 \
    --test_samples 100 \
    --batch_size 4 \
    --lr 5e-5 \
    --min_lr 1e-5 \
    --real_ratio 1.0 \
    --real_speech_manifest data/librispeech/benchmark_train_960h.json \
    --val_manifest data/librispeech/benchmark_val.json \
    --warm_start checkpoints/phono_v6_4_gated_diffusion/medium/bench_phono_v6_4_gated_diffusion_medium/best_checkpoint.pt \
    --only_save_best

# Modular Continuation (Easily resume for any additional X steps):
.venv/bin/python scripts/run_pretrain.py \
    --arch phono_v6_4_gated_diffusion \
    --tier medium \
    --run_name v6_4_960h \
    --resume auto \
    --additional_steps 30000 \
    --batch_size 4 \
    --real_ratio 1.0 \
    --real_speech_manifest data/librispeech/benchmark_train_960h.json \
    --val_manifest data/librispeech/benchmark_val.json \
    --only_save_best
```

### 9. Phono-V7 (Conformer Backbone + Two-Level Windowed Character Decoder)
Two-level hierarchical speech architecture combining an interleaved depthwise convolution Conformer acoustic backbone with a macro word decoder and an autoregressive micro character decoder supervised by differentiable Soft-Levenshtein loss:
```bash
# Stage 1: Train Conformer acoustic encoder and CTC alignment on 4-way balanced multilingual audio
PYTHONPATH=. .venv/bin/python scripts/train_phono_v7_stage1.py \
    --batch_size 4 \
    --grad_accum 4 \
    --max_steps 15000 \
    --lr 3e-4 \
    --max_duration_seconds 20.0

# Stage 2: Train Two-Level Windowed Character Decoder with dynamic length predictor and Soft-Levenshtein loss
PYTHONPATH=. .venv/bin/python scripts/train_phono_v7_char.py \
    --warm_start_encoder checkpoints/phono_v7/stage1/best_checkpoint.pt \
    --batch_size 2 \
    --grad_accum 8 \
    --max_steps 15000 \
    --encoder_lr 1e-5 \
    --decoder_lr 3e-4 \
    --max_duration_seconds 20.0
```

### 10. Phono-V7.1 (Real-Time Causal Streaming Speech Model)
Strictly causal real-time streaming speech model featuring $K=8$ word band-causal macro attention, shift-invariant base queries ($\mathbf{q}_{\text{base}}$), online event-driven CTC peak detection, continuous length guidance to the micro head (`length_to_micro_bias`), and asymmetric truncation protection:
```bash
# Launch strictly causal streaming training with 4-way balanced multilingual speech (EN, IT, ES, FR)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. .venv/bin/python scripts/train_phono_v7_1_streaming.py \
    --warm_start_v7 checkpoints/phono_v7/char/best_checkpoint.pt \
    --band_window 8 \
    --batch_size 2 \
    --grad_accum 8 \
    --max_steps 15000 \
    --eval_every 200 \
    --log_every 10 \
    --encoder_lr 1e-5 \
    --decoder_lr 3e-4 \
    --max_duration_seconds 20.0
```

### 11. Phono-V7.2 (Level-1 Binary Gate & Partitioned MoE Decoder with Horizon Capping)
Two-level hierarchical streaming speech model combining Level-1 binary gate routing (`SPECIAL/PAUSE` bypass) with Level-2 partitioned MoE expert clusters (Short, Medium, Long) and dynamic length horizon capping ($\min(\text{HeadBound},\, \lceil \hat{k} \rceil + 1)$):
```bash
# Launch Phono-V7.2 streaming training with partitioned MoE warm-started from V7.1 record
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. .venv/bin/python scripts/train_phono_v7_2_streaming.py \
    --warm_start_v7_1 checkpoints/phono_v7_1/streaming/best_checkpoint.pt \
    --band_window 8 \
    --batch_size 2 \
    --grad_accum 8 \
    --max_steps 15000 \
    --eval_every 200 \
    --log_every 10 \
    --encoder_lr 1e-5 \
    --decoder_lr 3e-4 \
    --checkpoint_dir checkpoints/phono_v7_2/streaming
```

### 12. Phono-V8.0 Pure Acoustic Model 1 (80,000-Step Multilingual Training with Top-3 Emissions)
Isolated Conformer CTC training with DeepSeek MoE (1 shared + 8 routed experts, Top-2), Layer-4 draft supervision, FactorEngram associative memory, and real-time Top-3 phoneme candidate probability extraction:
```bash
# Launch Pure Model 1 Multilingual Conformer Training (80,000 steps)
.venv/bin/python scripts/train_phono_v8_acoustic.py \
    --clean_train_manifest data/clean_multilingual_train.json \
    --clean_val_manifest data/clean_multilingual_val.json \
    --warm_start_ckpt checkpoints/phono_v8_acoustic/best_v8_acoustic.pt \
    --output_dir checkpoints/phono_v8_acoustic_multilingual_80k \
    --max_steps 80000 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 4.0e-5 \
    --min_lr 2.0e-6 \
    --warmup_steps 1000 \
    --log_every 1 \
    --sample_every 20 \
    --eval_every 2000
```
- **Real-Time Step Logging**: Live 1-line update every step displaying PER, Acoustic Sound PER (spaces excluded), Phonetic WER (P-WER), Space Accuracy, loss, steps/sec, and ETA.
- **Top-3 Probability Trace**: Rich inspection every 20 steps showing exact softmax candidate distributions (e.g. `[t: 0.61 | n: 0.13 | z: 0.13]`).

### 13. Space-Free Continuous Phoneme-to-Word & Robust P2G Training
Trains `ContinuousPhonemeWordSegmenter` on continuous phoneme streams without fragile acoustic `<space>` tokens, using boundary attention gating and word span pooling:
```bash
# Launch Continuous Space-Free P2G Training
.venv/bin/python scripts/train_phono_v8_continuous_p2g.py

# Launch Noise-Robust P2G Training with Articulatory Confusion Simulation
.venv/bin/python scripts/train_phono_v8_robust_p2g.py

# Evaluate End-to-End ASR with Phonotactic Beam Search
.venv/bin/python scripts/evaluate_phono_v8_e2e.py
```

---

## Longitudinal PER Benchmark Progression

Evaluated on genuine downstream LibriSpeech test utterances:

| Architecture Generation | Context Window | Training Data Mix | Training Steps | Key Innovation | Phoneme Error Rate (PER) | Lexicon / Char Acc |
| :--- | :---: | :--- | :---: | :--- | :---: | :---: |
| **Phono-V5 (Dense Baseline)** | 7.0s | 100% Synthetic Procedural TTS | 1,000 steps (~2.5h) | Dense Attention + Standard CTC | **75.0%** | ~80% |
| **Phono-V6.0 (Procedural)** | 7.0s | 100% Synthetic Procedural TTS | 1,000 steps (~2.5h) | Procedural Clean Speech | **58.0%** | ~65% |
| **Phono-V6.1 (MoE 4-Experts)** | 7.0s | 50% Synthetic / 50% LibriSpeech | 2,000 steps (~15h) | Hierarchical Mixture-of-Experts | **45.92%** | 50.1% |
| **Phono-V6.2 (Sparse Attention)** | 30.0s | 100% Genuine LibriSpeech Clean | 4,000 steps (52.1h) | Sparse Local Attention + InterCTC | **15.10%** | **22.18%** |
| **Phono-V6.3 (Latent Diffusion)** | 30.0s | 100% Genuine LibriSpeech Clean | 4,000 steps (56.2h) | Latent Diffusion Multi-Task Regularization | **13.70%** | **20.74%** |
| **Phono-V6.4 (Gated Diffusion 100h)** | 30.0s | 100% Genuine LibriSpeech Clean | 4,000 steps (56.2h) | Confidence-Gated Diffusion + Deep Refiner | **14.63%** | **16.78%** |
| **Phono-V6.4 (Gated Diff 960h - Stage 1)** | 30.0s | 100% Full 960h LibriSpeech Mixed | 30,000 steps (410.3h) | Learnable Gate MLP | **7.50%** | **12.43%** |
| **Phono-V6.4 (Gated Diff 960h - Full Epoch)** | 25.0s | 100% Full 960h LibriSpeech Mixed | 70,000 steps (956.8h) | 1 Full Epoch 960h  | **5.87%** *(Project Record)* | **11.92%** *(Lexicon PER)* |
| **Phono-V7 (Conformer + Two-Level Decoder)** | 20.0s | 4-Way Balanced (EN, IT, ES, FR) | 15,000 steps | Conformer + Macro/Micro Windowed Decoder + Soft Levenshtein | **36.07%** *(Multilingual)* | **64.8%** *(Char Acc)* |
| **Phono-V7.1 (Streaming Band-Causal Conformer)** | 20.0s | 4-Way Balanced (EN, IT, ES, FR) | 2,000 steps | $K=8$ Band-Causal Macro Attn + Online CTC Slicing + Length Guidance | **35.16%** *(Project Record)* | **61.0%** *(Char Acc)* |
| **Phono-V7.2 (Partitioned MoE + Horizon Capping)** | 20.0s | 4-Way Balanced (EN, IT, ES, FR) | 15,000 steps (Active) | Level-1 Gate + Level-2 Partitioned Experts (Short/Med/Long) + Horizon Cap | **35.16%** *(Warm-Started)* | **97.2%** *(Coverage Rate)* |

```
PER Progression Across Model Generations:
  Phono-V5 (Dense Baseline):     ██████████████████████████████ 75.0%
  Phono-V6.0 (Procedural):       ███████████████████████ 58.0%
  Phono-V6.1 (MoE 4-Experts):    ██████████████████ 45.92%
  Phono-V6.2 (Sparse + 30s):     ██████ 15.10%
  Phono-V6.3 (Diffusion):        █████ 13.70%
  Phono-V6.4 (Gated Diff 100h):  █████ 14.63%
  Phono-V6.4 (Gated Diff 960h):  ██ 5.87% (Project Record: 5.87% PER / 11.92% Lexicon PER)
  Phono-V7.1 (Streaming Online): █▎ 35.16% (Real-Time Causal Streaming across 4 languages)
  Phono-V7.2 (Partitioned MoE):  █▎ 35.16% (Active Training: Level-1 Gate + Level-2 Partitioned MoE + Horizon Capping)
```

### Standardized Cross-Architecture Benchmark Suite (Medium Tier, 4,000 Steps Each)

Cheat-free evaluation protocol: model selection performed strictly on held-out validation split (`benchmark_val.json`, 5.09h), followed by unbiased evaluation on standard `librispeech_test_clean.json`:

| Architecture | Tier | Best Val PER | Best Step | Test PER (Greedy) | Test Lexicon PER | Test CER | Audio Hours |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **phono_v6_4_gated_diffusion** | medium | 18.27% | 2800 | **14.63%** | **16.78%** | 14.63% | 56.2h |
| **phono_v6_3_diffusion** | medium | 20.58% | 3600 | **16.05%** | 19.46% | 16.05% | 56.3h |
| **phono_v6_2_sparse** | medium | 24.82% | 4000 | **19.13%** | 20.27% | 19.13% | 56.4h |
| **phono_v4_scaled** | medium | 32.73% | 3400 | **27.49%** | 36.96% | 27.49% | 56.4h |
| **phono_v6_1_moe** | medium | 35.49% | 3800 | **28.60%** | 33.83% | 28.60% | 56.4h |
| **phono_v5_beam** | medium | 35.50% | 3600 | **29.77%** | 32.64% | 29.77% | 56.6h |
| **phono_v2_specaugment** | medium | 37.08% | 4000 | **30.38%** | 34.75% | 30.38% | 56.3h |
| **phono_v3_hybrid** | medium | 39.34% | 4000 | **34.83%** | 40.89% | 34.83% | 56.4h |
| **phono_v1_frontend** | medium | 43.51% | 3800 | **37.98%** | 30.37% | 37.98% | 56.5h |

### Key Pre-Training CLI Options
- `--arch`: Registered architecture to train (`phono_hubert`, `phono_hubert_hierarchical`, `phono_hubert_dual`, `phono_hubert_recursive`, `phono_v6_1_moe`, `phono_v6_2_sparse`, `phono_v6_3_diffusion`, `phono_v6_4_gated_diffusion`, or `hubert_kmeans`).
- `--tier`: Architecture scale tier (`mini`, `small`, `medium`, or `base`).
- `--real_ratio`: Ratio of real human speech in streaming (default: `1.0` = 100% genuine LibriSpeech clean audio; set `0.0` for 0-disk procedural neural TTS).
- `--max_duration_sec`: Maximum utterance duration in seconds (default: `30.0` seconds, unlocking long-range context).
- `--save_interval`: Checkpoint persistence interval (default: `4000`, eliminating intermediate SSD wear; emergency checkpoint always saved on `Ctrl+C`).
- `--warm_start`: Path to existing checkpoint to warm-start weights from (e.g. initializing V6.3 diffusion refiners on top of a V6.2 acoustic backbone).
- `--min_lr`: Minimum learning rate floor for cosine decay (default: `1e-5`, ensures optimizer never stalls at 0).
- `--blank_penalty`: Calibrated blank logit deduction for greedy decoding (e.g. `1.5 - 2.5`).
- `--masking_mode`: Configurable masking scheme (`none`, `specaugment`, `span`, `dual`).
- `--mask_prob` / `--mask_length`: Masking probability and span length in 20ms frames (default: `0.65`, `10`).
- `--warmup_steps`: Linear learning rate warmup steps before cosine decay (e.g. `100`).
- `--freeze_cnn_steps`: Freeze the 7-layer temporal 1D CNN feature encoder for initial steps to stabilize the Transformer backbone (default: `200`).
- `--override`: Custom parameter overrides (e.g. `--override "encoder_layers=10,hidden_dropout=0.1"`).
- `--num_workers`: Number of parallel data loader threads on CPU (default: `4`).
- `--buffer_size`: Max batches in the bounded queue (default: `20`).
- `--watermark`: Low watermark batch count to resume worker loading (default: `10`).
- `--run_name`: Unique name for the training run (e.g. `--run_name v6_3_diffusion_run1`). If omitted, auto-assigns next sequential run (`run_1`, `run_2`).
- `--resume`: Auto-resume from the latest checkpoint for the active run (or specify a custom checkpoint path).

---

## Multi-Run Management & Full Configuration Persistence

Every training run is completely isolated with its own frozen configuration, checkpoints, and telemetry:
- **Automatic / Custom Run Naming**: Multiple runs of the same model and tier (e.g., `phono_hubert` / `medium`) are saved into isolated subdirectories: `checkpoints/<arch>/<tier>/<run_name>/` and `logs/<arch>/<tier>/<run_name>/`.
- **Full Configuration Saved at Start**: Upon launch, `train_config.json` is generated capturing:
  - Model architecture parameters (`layers`, `heads`, `embed_dim`, `ffn_dim`, `conv_layers`, `vocab_size`, trainable parameter count).
  - Training hyperparameters (`lr`, `warmup_steps`, `freeze_cnn_steps`, `batch_size`, `steps`, `weight_decay`, `clip_grad_norm`, AMP).
  - Masking mode and parameters (`none`, `specaugment`, `span`, `dual`).
  - Streaming data generator settings (`num_workers`, `buffer_size`, `watermark`, `pool_size`).
  - Environment metadata (PyTorch version, CUDA version, GPU model, Git commit, CLI command).

### Managing Runs via CLI (`manage_runs.py`)

```bash
# 1. List all training runs across models and tiers
python scripts/manage_runs.py list

# 2. Show complete frozen configuration, checkpoints, and milestones of a run
python scripts/manage_runs.py show run_1 --arch phono_hubert --tier medium

# 3. Compare multiple runs side-by-side
python scripts/manage_runs.py compare run_1 run_2 --arch phono_hubert --tier medium

# 4. Safely delete a run from disk and the registry
python scripts/manage_runs.py delete run_1 --arch phono_hubert --tier medium -y
```

### REST API Endpoints
- `GET /api/training/runs`: Returns a catalog of all runs with configurations, steps, loss, PER, and status.
- `GET /api/training/runs/{arch}/{tier}/{run_name}`: Returns the complete frozen configuration and checkpoint history for a specific run.

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

AudioLearn includes a comprehensive 100+ test suite covering data synthesis, model architectures (HuBERT, PhonoHuBERT, V6.1 MoE, V6.2 Sparse, V6.3 Diffusion, V6.4 Gated Diffusion, Word Denoising Decoder, Phono-V7/V7.1/V7.2 Conformer MoE), articulatory confusion matrix simulation (`PhoneticNoiseInjector`), phonotactic syllable segmentation (`PhonotacticSegmenter`), prefix beam search decoders (`PhonotacticBeamSearchDecoder`), the Voice Quality Guardian, the threaded batch generator, RunManager isolation, XAI attribution, Conformer convolutions, CTC forced alignment, multilingual lexicon tokenization, differentiable Soft-Levenshtein loss, and the continuous phoneme segmenter:

```bash
PYTHONPATH=. .venv/bin/pytest tests/ -v
```

All 100+ tests pass across CPU/GPU environments.

---

## License

This project is licensed under the Apache 2.0 License. Model weights and synthetic audio pipelines are provided for educational and research purposes.
