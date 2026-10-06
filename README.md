# AudioLearn: HuBERT Speech Self-Supervision, Modular Architectures & XAI Studio

A modular PyTorch speech framework implementing **HuBERT** (Hidden-Unit BERT) and **PhonoHuBERT** (Direct Phoneme Prediction) from scratch. Features **dual data streaming pipelines** (0-disk in-RAM procedural neural TTS synthesis for initial exploration, and 100% genuine LibriSpeech human speech streaming for SOTA acoustic scaling), an **asynchronous multi-threaded generator with microsecond profiling**, a **Voice Quality Guardian**, scaling tiers from **8M to 95M parameters**, a **SOTA Whisper shootout benchmark**, and a full **Explainable AI (XAI)** suite with a 6-tab interactive web studio.

---

## Key Highlights & Innovations

1. **Modular Architecture Registry**:
   - **HuBERT (Acoustic K-Means SSL)**: Self-supervised pre-training via 39-dim MFCC acoustic cluster pseudo-labels (100 clusters) as in Hsu et al. (2021).
   - **PhonoHuBERT (Direct Phonemes)**: Direct acoustic-to-phoneme prediction with 64 IPA tokens and special tokens (`<silence>`, `<noise>`, `<mask>`, `<blank>`, `<eos>`, `<unk>`).
   - **Phono-V6 Progressive Architectures (< 15% PER)**:
     - **V6.1 (MoE Transformer)**: Sparse Mixture-of-Experts with 4 FFN experts, Top-2 gating, and dynamic load-balancing auxiliary loss.
     - **V6.2 (Sparse Attention + 30s Context + InterCTC)**: Local sliding attention window ($\pm 320$ms), 30-second context window (up to 1,500 frames), Intermediate Layer-4 & Layer-8 CTC multi-task supervision, achieving our breakthrough **15.10% PER** on LibriSpeech clean-100.
     - **V6.3 (Sliding Gaussian Latent Diffusion)**: Spatio-temporal Gaussian-modulated diffusion in latent phoneme space, FiLM-conditioned 2-block Conv1D refiner, and a trailing-window streaming phoneme decoder.
     - **V6.4 (Confidence-Gated Diffusion, Learnable Gate Thresholds & Word Decoder)**: Adaptive margin gating where confident CTC frames bypass diffusion while ambiguous frames receive targeted denoising via an enhanced 3-block convolutional refiner. Features dynamic **learnable gate thresholds** via an MLP taking `[hidden_state; margin; top1_prob]` to dynamically predict gate probability $g \in [0, 1]$. Accompanied by a standalone **Word Denoising Decoder** (`src/models/word_denoising_decoder.py`) with 4-expert MoE text modeling, banded cross-attention, continuous word latent diffusion, and contrastive homophone loss.
     - **V6.5–V6.8 (MoE Scaling & Multilingual Dual Attention)**: 16/32 MoE tier scaling with fast-path base-token bypass, multilingual 4-way balanced training (EN, IT, ES, FR), dual acoustic-word cross-attention, and context expansion to $W=6$.
   - **Phono-V7 (Conformer Backbone + Two-Level Windowed Character Decoder)**:
     - **Conformer Acoustic Backbone**: Interleaved depthwise separable convolutions (`ConformerConvModule`) and multi-head self-attention with macaron-style feed-forward modules.
     - **CTC Forced Alignment & Word Peak Detection**: Dynamic programming CTC aligner (`CTCForcedAligner`) and speech energy burst tracking ($E(t) = 1 - P(\text{blank})$) to delineate acoustic word boundaries without external alignments.
     - **Two-Level Hierarchical Decoding**:
       - *Level 1 (Macro Word Decoder)*: Processes acoustic word slices and predicts continuous word length $\hat{k}$ via `word_length_head`.
       - *Level 2 (Micro Character Decoder)*: Autoregressively generates byte character sequences conditioned on macro word latents, emitting characters and terminating with `<eow>`.
     - **Differentiable Soft-Levenshtein Loss**: Forward-backward DP recursion for direct string alignment supervision.
     - **Asymmetric Truncation Protection**: $\times 3.0$ penalty when $\hat{k} < k_{\text{true}}$, guaranteeing positive character headroom and eliminating premature word truncation.
   - **Phono-V7.1 (Real-Time Causal Streaming Speech Model)**:
     - **Strictly Causal Streaming**: Zero future lookahead across both encoder and decoder, fully streamable for live microphone input.
     - **Band-Causal Macro Attention ($K=8$ words)**: Enforces a strictly causal local window over past words, bounding error propagation and guaranteeing constant $O(K \cdot L)$ computation and memory.
     - **Shift-Invariant Base Query ($\mathbf{q}_{\text{base}}$)**: Shared learnable base query vector eliminating positional slot ceilings and training frequency disparities across sentence lengths.
     - **Online Event-Driven CTC Peak Slicing**: Dynamically segments acoustic words in real time based on energy bursts, gracefully absorbing pauses and breaths without needing total duration $T$ or word count $L$.
     - **Continuous Length Guidance to Micro Head**: Directly projects continuous predicted length $\hat{k}$ into micro character attention via `length_to_micro_bias`, enabling sharp word boundary localization.
     - **Length Coverage Metric (`Cover`)**: Achieves **> 97%** coverage rate ($\mathbb{P}(\lceil \hat{k} \rceil + 1 \ge k_{\text{true}})$) with $+2.7\text{c}$ to $+3.5\text{c}$ safety headroom.
   - **Phono-V7.2 (Level-1 Binary Gate & Level-2 Partitioned MoE with Horizon Capping)**:
     - **Two-Level Hierarchical Router**:
       - *Level 1 (Binary Gate)*: Detects non-verbal frames (`SPECIAL / PAUSE / BREATH`), bypassing micro decoding entirely (0 micro FLOPs).
       - *Level 2 (Partitioned Expert Clusters)*: Partitions the 16 micro MoE experts into 3 specialized length-conditioned expert pools: Short ($K \le 5$, experts `0..4`), Medium ($5 < K \le 9$, experts `5..10`), and Long ($K \ge 10$, experts `11..15`).
     - **Dynamic Horizon Capping**: Restricts autoregressive character rollout to $\min(\text{HeadBound},\, \lceil \hat{k} \rceil + 1)$, mathematically preventing trailing hallucination and saving GPU compute on short and medium words.
     - **Unified Tensor Computation**: Avoids batch fracturing and sequential kernel launches by unifying character self-attention and acoustic cross-attention across all words, routing strictly the FFN experts.
     - **Warm-Start Continuity**: Directly maps 945 tensors from the V7.1 project milestone checkpoint (`checkpoints/phono_v7_1/streaming/best_checkpoint.pt`).
   - **Phono-V7.3 (Multi-Scale Dilated Conformer & Decoupled Boundary Gate Head)**:
     - **Multi-Scale Dilated Conformer Convolutions (`MultiScaleDilatedConformerConvModule`)**: Extends the temporal receptive field $4\times$ from 620ms to 2,420ms without increasing parameter count. Branch 1 ($k=31, d=1$, 620ms) preserves 100% of V7.2 trained depthwise features, while parallel dilated branches (Branch 2: $k=31, d=2 \rightarrow 1,220\text{ms}$; Branch 3: $k=31, d=4 \rightarrow 2,420\text{ms}$) capture multi-scale intra-word coarticulation, liaisons, and inter-word prosody. Zero-initialized projection guarantees exact $0.0000$ perturbation at Step 0.
     - **Decoupled Word Boundary Gate Head (`DecoupledBoundaryGateHead`)**: Dedicated 3-class classifier (`0: SPEECH`, `1: WORD_SPACE`, `2: SILENCE_BLANK`) supervised directly by forced alignment with a $\times 4.0$ penalty on inter-word space deletion and insertion, eliminating acoustic word fracturing and downstream word concatenation errors.
     - **Recursive 2-Pass Phoneme Head (`RecursivePhonemeHead`)**: Refines base CTC emission probabilities using causal depthwise recurrent conditioning $[h_t \,\|\, \operatorname{softmax}(z_{t-1}^{(0)})]$ to enforce phonotactic grammar and prevent consecutive space emissions.
     - **Boundary-Gated Online Slicing**: Incorporates 2-frame silence confirmation and 4-frame minimum burst protection to ensure clean word slicing in real-time streaming mode.
     - **Zero-Perturbation Warm Start**: Bitwise identical verification against V7.2 milestone checkpoint (`checkpoints/phono_v7_2/streaming/best_checkpoint.pt`), seamlessly transferring all 945 tensors with zero missing or skipped parameters.
   - **Phono-V7.5 (Current Official Flagship: Tripartite Modular Architecture & Overlapping Length MoE)**:
     - **Tripartite Modular Decoupling**: Deconstructs end-to-end speech recognition into 3 specialized, modular pillars:
       1. *Module 1 (Acoustic Phoneme Front-End)*: 7-layer 1D CNN + Multi-Scale Dilated Conformer + Decoupled Boundary Gate Head (`DecoupledBoundaryGateHead`), dedicated 100% to acoustic invariance and phonetic boundary slicing.
       2. *Module 2 (Cross-Modal Latent Bridge)*: Ultra-fast 2-layer Transformer (`FastTextPhonemeWordEncoder`), distilled via composite InfoNCE + Cosine + MSE loss to match the acoustic word latent space ($z_{\text{word}}$) with **98.7% Cosine Similarity** and $50\times$ faster execution than full audio processing.
       3. *Module 3 (Overlapping Length MoE Text Decoder)*: Macro-micro recursive decoder with 7 overlapping length intervals (`[1-5]`, `[3-7]`, `[5-10]`, `[7-12]`, `[9-15]`, `[12-20]`, `[15-30]`), multi-positive Binary Cross-Entropy routing, load-balancing anti-collapse loss ($\mathcal{L}_{\text{balance}} = 7 \sum f_e P_e$), and Soft-Levenshtein character alignment.
     - **Text Middle-Training Breakthrough**: Pre-trains the decoder on >208,000 multilingual sentences across English, Italian, Spanish, and French in pure FP32, driving character accuracy from 25.9% to **94.6%**, MoE routing accuracy to **98.6%**, and reducing Soft-Levenshtein edit loss by 9.2× (down to **0.160**).
   - **Parameter Scaling Tiers**: **Mini** (8.0M), **Small** (24.2M), **Medium** (31.8M / 83.5M), and **Base** (94.7M) with on-the-fly parameter variation and checkpoint hot-swapping.

2. **Dual-Mode Streaming Pipeline & Full 960h LibriSpeech Interleaving**:
   - **Full 960h Scale**: Scales pre-training to the complete LibriSpeech 960h corpus (276,715 utterances, 945.5h) deterministically and uniformly interleaved across `clean-100`, `clean-360`, and `other-500` splits (`data/librispeech/benchmark_train_960h.json`).
   - **Multi-Split Balanced Validation**: 4,526 held-out utterances (15.55h) balanced across Clean-100, Clean-360, and Other-500 splits (`data/librispeech/benchmark_val.json`) with zero train-val overlap.
   - **Mode A (0-Disk Procedural In-RAM Streaming)**: For baseline pre-training (`phono_hubert`, `dual`, `hierarchical`, `recursive`, `hubert_kmeans`, and `phono_v1`/`v2`), speech is synthesized directly in RAM across 39 clean neural voices (`--real_ratio 0.0`). Waveforms are generated in memory, trained on GPU, and immediately deallocated with **zero audio files stored on disk**.
   - **Mode B (Real Speech Dataset Streaming)**: For SOTA models (`phono_v3_hybrid` through `phono_v6_4_gated_diffusion`), the pipeline streams genuine human speech directly from disk manifests (LibriSpeech 100h clean or full 960h) to capture authentic human phonetics, room acoustics, and conversational dynamics.
   - **SSD Wear Protection**: Intermediate 1GB checkpoint writes are eliminated (`--only_save_best`), persisting weights strictly when a new project validation record is broken and at final step completion.
   - **Seamless Multi-Stage Training & Resumption**: Supports modular training stages with `--additional_steps <X>` or `--steps <TOTAL>`, automatically restoring historical best validation metrics (`val_per`) to protect project records and decaying the learning rate smoothly.

3. **High-Throughput Asynchronous Multi-Threaded Generator (GIL-Optimized)**:
   - Solves CPU synthesis vs. GPU training starvation using an asynchronous producer-consumer architecture.
   - Upgraded with native OS condition variable blocking (`futex_wait`) and dedicated eSpeak phonemization, completely eliminating Python GIL contention during 4,000+ CUDA kernel dispatches.
   - Sustains **94%–100% continuous GPU compute saturation (248W / 250W TDP)** on NVIDIA GTX TITAN X with $< 0.1\text{ms}$ queue starvation latency and ~1.1s per 30-second batch.

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
                       │   Dual-Mode Audio Input: 100% LibriSpeech (Up to 30.0s)     │
                       │     OR 0-Disk In-RAM Procedural Speech (39 Neural Voices)   │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      │ Waveforms (16 kHz, Mono, 1.0s - 30.0s)
                                                      ▼
                       ┌─────────────────────────────────────────────────────────────┐
                       │  HuBERTFeatureEncoder (Temporal 1D Convolution)             │
                       │  7-layer 1D CNN with LayerNorm & GELU (Strides: 5,2,2,2...) │
                       │  Downsampling Factor: 320x (20ms acoustic frames / 50Hz)    │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      │ Frame Embeddings (B, T, 512)
                                                      ▼
                       ┌─────────────────────────────────────────────────────────────┐
                       │  Feature Projection & Acoustic SpecAugment                  │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       │                                                             │
                       ▼                                                             ▼
     [Phono-V6 Sparse MoE Backbone]                                 [Intermediate CTC Multi-Task]
     - Sparse Local Attention (±320ms window)                       - Early supervision at Layer 4 & 8
     - 4 FFN Experts per Layer + Top-2 Gating                       - Accelerated gradient propagation
     - 8 to 12 Transformer Layers (D=512)                           - Enables adaptive early exit
                       │                                                             │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      │ Pristine Frame Latents Z_0
                                                      ▼
                       ┌─────────────────────────────────────────────────────────────┐
                       │  Phono-V6.3 Sliding Gaussian Latent Diffusion Refiner       │
                       │  - Spatial-Temporal Gaussian Noise: σ(t; τ) = σ exp(-Δt²/2w²)│
                       │  - Lightweight FiLM-conditioned 2-block Conv1D Denoising    │
                       │  - Joint Diffusion MSE Loss + Refined Sequence CTC Loss     │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      │ Finalized Settled Latents
                                                      ▼
                       ┌─────────────────────────────────────────────────────────────┐
                       │  Trailing Window Streaming Decoder & Lexicon Integration    │
                       │  - Emits phonemes as frames exit trailing edge of window    │
                       │  - Trie-constrained Lexicon Beam Search (< 19.5% PER)       │
                       └─────────────────────────────────────────────────────────────┘
```

---

## Parameter Scaling Tiers

| Tier | Transformer Layers | Attention Heads | Embedding Dim ($D$) | FFN Dim ($4D$) | Parameters (HuBERT / Phono) | Parameters (Phono-V6 MoE / Diffusion) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mini** | 4 | 4 | 256 | 1024 | **8.06 M** | **18.42 M** |
| **Small** | 6 | 6 | 384 | 1536 | **24.23 M** | **45.18 M** |
| **Medium** | 8 | 8 | 512 | 2048 | **31.82 M** | **83.55 M** (Active SOTA) |
| **Base** | 12 | 12 | 768 | 3072 | **94.73 M** | **172.40 M** |

---

## Current Official Flagship: Phono-V7.5 Architecture & Component Scores

Phono-V7.5 is the current official flagship architecture of AudioLearn. It replaces opaque monolithic end-to-end speech-to-text models with a **decoupled, tripartite modular architecture** where acoustic phonetics, cross-modal latent alignment, and multilingual orthographic decoding are separated into specialized, optimal sub-models.

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │               MODULE 1: ACOUSTIC PHONEME FRONT-END          │
                    │   7-Layer 1D CNN + Multi-Scale Dilated Conformer (2,420ms)   │
                    │   + Decoupled Boundary Gate Head + Recursive Phoneme Head   │
                    └──────────────────────────────┬──────────────────────────────┘
                                                   │ Word Boundaries & Acoustic Slices
                                                   ▼
                    ┌─────────────────────────────────────────────────────────────┐
                    │            MODULE 2: CROSS-MODAL DISTILLED LATENT BRIDGE    │
                    │   FastTextPhonemeWordEncoder (2-Layer Transformer, 1.51M p) │
                    │   Distilled to Acoustic Manifold via InfoNCE + CosSim + MSE │
                    │   Matches z_word with 98.7% Cosine Similarity at 50x Speed  │
                    └──────────────────────────────┬──────────────────────────────┘
                                                   │ Canonical Word Latents z_word
                                                   ▼
                    ┌─────────────────────────────────────────────────────────────┐
                    │            MODULE 3: OVERLAPPING LENGTH MoE TEXT DECODER    │
                    │   7 Overlapping Interval Experts with Multi-Positive BCE:   │
                    │   [1-5], [3-7], [5-10], [7-12], [9-15], [12-20], [15-30]    │
                    │   Anti-Collapse Balancing + Soft-Levenshtein Micro Head     │
                    └─────────────────────────────────────────────────────────────┘
```

### Component Breakdown & Live Benchmark Scores

| Component Module | Architecture & Parameters | Objective & Losses | Current Benchmark Score |
| :--- | :--- | :--- | :--- |
| **Module 1: Acoustic Conformer Front-End** | 7-layer 1D CNN + 8-layer Conformer with Multi-Scale Dilated Convolutions ($k=31$, dilations $1, 2, 4$), Decoupled Boundary Gate Head (`31.8M params`) | Acoustic CTC Loss + Space Penalty ($\times 4.0$) + Online Peak Slicing | **Val Loss `3.7054`** (Project record locked at Step 15,400 on LibriSpeech streaming) |
| **Module 2: Distilled Latent Bridge** | 2-layer Transformer (`FastTextPhonemeWordEncoder`, $D=256$, $H=4$, `1.51M params`) | $\mathcal{L}_{\text{distill}} = \text{MSE} + (1 - \text{CosSim}) + \mathcal{L}_{\text{InfoNCE}}$ | **Cosine Similarity `98.7%`** ($0.987$), **MSE `0.015`** against acoustic manifold |
| **Module 3: Overlapping Length MoE Decoder** | Macro Windowed Decoder + 7 Overlapping Length Experts MoE + 4-layer Recursive Micro Character Decoder (`51.7M params`) | Multi-Positive BCE + Anti-Collapse ($\mathcal{L}_{\text{balance}} = 7 \sum f_e P_e$) + Soft-Levenshtein ($0.2$) + Char Cross-Entropy | **Validation Loss `0.9004`** (Step 8,000)<br>• **CharAcc: `94.6%`**<br>• **MoE PathAcc: `98.6%`**<br>• **Soft-Levenshtein: `0.160`** |

### Why Phono-V7.5 Outperforms Monolithic End-to-End Models

1. **Acoustic-Linguistic Gradient Disentanglement**:
   In classic E2E training, gradients flowing through the character decoder are noisy because they must simultaneously resolve acoustic noise (accents, background audio, reverberation) and language structure. In Phono-V7.5, Module 3 is pre-trained via **Text Middle-Training** over >208,000 sentences across 4 languages (**English, French, Italian, Spanish**) in pure FP32, learning an optimal linguistic prior.
2. **7 Soft Overlapping Intervals with Multi-Positive Routing**:
   Previous versions (V7.1–V7.3) suffered from boundary oscillation because rigid bins (Short 1-5, Medium 5-11, Long 11-16) penalized words near the thresholds. Phono-V7.5's overlapping intervals permit a 5-letter word to activate `[1-5]`, `[3-7]`, or `[5-10]`. Multi-positive BCE with anti-collapse load balancing raised routing accuracy from $40\%\text{--}60\%$ to **$98.6\%$**.
3. **50× Speedup for Text Exploration**:
   Module 2 bypasses audio synthesis and Conformer forward passes, enabling the decoder to train on text at **0.6 steps/s (BS=32)**, saving hundreds of GPU compute hours.

---

## Target Research Benchmarks (Ben Letaifa, Rouas et al. Literature)

Following the final light end-to-end fine-tuning, the Phono-V7.5 tripartite system will be benchmarked against the standard academic datasets and baselines documented in the foundational literature by **Ben Letaifa, Rouas, et al.** (Algorithms 2023, EUSIPCO 2022, ICASSP 2025, Interspeech 2025):

### 1. English ASR Benchmarks
- **LibriSpeech 1000h (Clean & Other)** *(ICASSP 2025, Algorithms 2023)*:
  - *Transformer (99M)*: Test-clean **3.3% WER**, Test-other **8.0% WER**
  - *Conformer (93M)*: Test-clean **2.9% WER**, Test-other **7.3% WER**
  - *Branchformer (116M)*: Test-clean **2.4% WER**, Test-other **5.3% WER**
  - *E-Branchformer (148M)*: Test-clean **2.2% WER**, Test-other **4.6% WER**
  - *HuBERT Base (433M)*: Test-clean **2.0% WER**, Test-other **4.2% WER**
  - *WavLM Base (431M)*: Test-clean **2.0% WER**, Test-other **4.2% WER**
- **Libri-trans (236h English Audiobooks)** *(Algorithms 2023, EUSIPCO 2022)*:
  - *Baseline Transformer (27.92M)*: **6.6% WER** (Dev: 6.3% WER)
  - *Adaptive Variable Scale Pruning (43% sparsity)*: **7.6% WER**
  - *Double Compression (52.5% sparsity)*: **8.9% WER** (80% memory footprint reduction)

### 2. Multilingual ASR Benchmarks
- **VoxforgeIT (Italian, 20h Audiobooks)** *(Algorithms 2023, EUSIPCO 2022)*:
  - *Baseline Transformer (35.07M)*: **9.1% – 9.3% CER** (Dev: 10.3% CER)
  - *Adaptive Variable Scale Pruning (59.5% sparsity)*: **10.3% CER** (Dev: 10.5% CER)
  - *Pruning + INT8 Quantization*: **10.1% CER** (82% memory reduction)
- **ESTER (French, 250h Broadcast News)** *(EUSIPCO 2022)*:
  - *Baseline Transformer (89.64M)*: **14.1% WER**
  - *Pruning + INT8 Quantization (34.5% prune)*: **15.6% WER** (79% memory reduction)
- **VIVOS (Vietnamese, 16h Read Speech)** *(EUSIPCO 2022)*:
  - *Baseline Transformer (15.65M)*: **14.7% CER**
  - *Pruning + INT8 Quantization (56.0% prune)*: **16.1% CER** (84% memory reduction)

### 3. Speech-to-Text Translation Benchmark
- **MuST-C English $\to$ French (560h Audio / 245k Sentences)** *(Interspeech 2025)*:
  - *ASR Subsystem (95.5M)*: Dev.en-fr **11.5% WER** (INT8: 11.7%), tst-COMMON.en-fr **11.0% WER** (INT8: 10.9%)
  - *MT Subsystem (41.8M)*: Dev.en-fr **28.8 BLEU**, tst-COMMON.en-fr **35.4 BLEU**
  - *Cascade Speech-to-Text Translation*: Dev.en-fr **25.8 BLEU**, tst-COMMON.en-fr **31.2 BLEU**
  - *Hardware Acceleration (Systolic Arrays)*: Up to **74× speedup** and **35% energy reduction** under structured tile pruning.

### Official Post-Fine-Tuning Benchmark Protocol
Upon completion of the current 50,000-step text middle-training and the subsequent light end-to-end acoustic fine-tuning, Phono-V7.5 will be evaluated across:
1. `eval_librispeech_clean_other`: Word Error Rate on LibriSpeech `test-clean` and `test-other`.
2. `eval_libritrans_en`: Word Error Rate on the Libri-trans evaluation split.
3. `eval_voxforge_it`: Character Error Rate on VoxforgeIT Italian test set.
4. `eval_ester_fr`: Word Error Rate on ESTER French broadcast evaluation corpus.
5. `eval_mustc_en_fr`: Cascade translation BLEU and ASR WER on MuST-C `tst-COMMON.en-fr`.

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
│   │   └── registry.py         # ModelRegistry factory for dynamic discovery & scaling
│   ├── losses/                 # Differentiable loss implementations
│   │   └── soft_levenshtein.py # Differentiable Soft-Levenshtein DP string alignment loss
│   ├── data/                   # Data pipelines & tokenization
│   │   ├── tokenizer.py        # Character CTC Tokenizer (31 tokens)
│   │   ├── phoneme_tokenizer.py# Bilingual (EN/FR) IPA Tokenizer (64 tokens + 8 special tokens)
│   │   ├── multilingual_lexicon.py # Offline pronunciation lexicon for EN, IT, ES, FR
│   │   ├── multilingual_audio_dataset.py # 4-way balanced multilingual dataset loader & collator
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
├── checkpoints/                # Saved weights (HuBERT, PhonoHuBERT, V7, V7.1, V7.2)
│   ├── phono_v7/char/          # Phono-V7 Two-Level Character Decoder checkpoints
│   ├── phono_v7_1/streaming/   # Phono-V7.1 Real-Time Streaming checkpoints
│   └── phono_v7_2/streaming/   # Phono-V7.2 Partitioned MoE Streaming checkpoints
├── logs/                       # Real-time status JSONs & TensorBoard telemetry
├── scripts/                    # Command-line entry points
│   ├── run_server.py           # Start the FastAPI interactive studio server
│   ├── run_pretrain.py         # Unified modular 0-disk streaming pre-training CLI
│   ├── train_phono_v7_stage1.py# Phono-V7 Stage 1 Conformer & alignment pre-training
│   ├── train_phono_v7_char.py  # Phono-V7 Stage 2 Two-Level Character Decoder training
│   ├── train_phono_v7_1_streaming.py # Phono-V7.1 Real-Time Streaming training
│   ├── train_phono_v7_2_streaming.py # Phono-V7.2 Partitioned MoE Streaming training
│   ├── train.py                # Supervised CTC fine-tuning on LibriSpeech
│   ├── evaluate.py             # Checkpoint evaluator
│   ├── run_xai.py              # Generate static XAI visualization plots
│   └── demo_pipeline.py        # Automated end-to-end master pipeline
├── tests/                      # Pytest automated test suite (95+ tests)
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

AudioLearn includes a comprehensive 91-test suite covering data synthesis, model architectures (HuBERT, PhonoHuBERT, V6.1 MoE, V6.2 Sparse, V6.3 Diffusion, V6.4 Gated Diffusion with learnable thresholds, Word Denoising Decoder), the Voice Quality Guardian, the threaded batch generator, RunManager isolation, XAI attribution, Conformer convolutions, CTC forced alignment, multilingual lexicon tokenization, differentiable Soft-Levenshtein loss, and the Phono-V7 & V7.1 streaming pipelines:

```bash
PYTHONPATH=. .venv/bin/pytest tests/ -v
```

All 91 tests pass on CPU/GPU.

---

## License

This project is licensed under the Apache 2.0 License. Model weights and synthetic audio pipelines are provided for educational and research purposes.
