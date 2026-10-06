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
    - **Phono-V7.5 (Current Official Flagship: Complete Tripartite Modular Architecture & Dual-Level MoE)**:
      - **Tripartite Modular Decoupling**: Deconstructs monolithic speech recognition into 3 decoupled, independently optimizable sub-models:
        1. *Module 1 (Acoustic Phoneme Front-End, 31.8M params)*: 7-layer 1D CNN feature extractor (strides [5,2,2,2,2,2,2], downsampling factor $320\times$, 20ms frames) + 8-layer Conformer backbone ($D=512$, $H=8$, FFN=2048) with 4 MoE FFN experts per layer (Top-2 gating), **Multi-Scale Dilated Conformer Convolutions** ($k=31$, dilations $d=1, 2, 4$, extending receptive field $4\times$ from 620ms to 2,420ms), **Decoupled Boundary Gate Head** (`DecoupledBoundaryGateHead`: 3 classes `SPEECH`, `WORD_SPACE`, `SILENCE_BLANK` with $\times 4.0$ boundary penalty), **Recursive 2-Pass Phoneme Head** (`RecursivePhonemeHead`: causal depthwise recurrent conditioning on $[h_t \,\|\, \operatorname{softmax}(z_{t-1}^{(0)})]$ enforcing phonotactic grammar), and **Online Event-Driven CTC Peak Slicing** ($1 - P(\text{blank})$) dynamically delineating acoustic word bounds in real time.
        2. *Module 2 (Cross-Modal Distilled Latent Bridge, 1.51M params)*: Ultra-fast 2-layer Transformer (`FastTextPhonemeWordEncoder`, $D=256$, $H=4$, FFN=512) taking orthographic text characters (byte IDs $0..127$) and phonemes ($0..63$), pooling via `AttentionPooling` and projecting to the canonical acoustic word latent space ($z_{\text{word}}$, $D=512$) with **98.7% Cosine Similarity** and MSE of $0.015$, operating **$50\times$ faster** than full audio Conformer processing.
        3. *Module 3 (Dual-Level MoE Recursive Character Decoder, 51.7M params)*:
           - *Level-1 Macro Word Decoder (4 layers, $D=512$)*: Shift-invariant base query $\mathbf{q}_{\text{base}}$, Band-Causal local attention ($K=8$ words, constant $O(K \cdot L)$ compute), macro history noise injection ($\sigma=0.05$), and sliding multi-word context windows ($W=6$ words).
           - *7 Overlapping Word-Length Experts MoE Router*: 7 soft intervals (`[1-5]`, `[3-7]`, `[5-10]`, `[7-12]`, `[9-15]`, `[12-20]`, `[15-30]`), multi-positive Binary Cross-Entropy routing, and anti-collapse load-balancing loss ($\mathcal{L}_{\text{balance}} = 7 \sum f_e P_e$) achieving **98.0%–98.6% routing accuracy**.
           - *Continuous Word Length Predictor & Dynamic Horizon Capping*: Predicts character length $\hat{k}$ via MLP with Softplus under asymmetric truncation loss ($\times 4.0$ penalty if $\hat{k} < k_{\text{true}}$), capping autoregressive rollout to $\min(\text{HeadBound},\, \lceil \hat{k} \rceil + 1)$ to eliminate trailing hallucination.
           - *Level-2 Micro Character Recursive Decoder (`WindowedMicroPartitionedRecursiveHead`, 4 layers, $D=512$)*: Tri-Modal Cross-Attention per layer (causal char self-attention, direct acoustic slice cross-attention, macro sliding word context cross-attention), **16 Partitioned Micro MoE Experts** with Top-2 routing (Short `0..4`, Medium `5..10`, Long `11..15`), weight-tied byte vocabulary ($V=128$), and **Differentiable Soft-Levenshtein Loss** forward-backward DP edit-distance alignment.
      - **Text Middle-Training Breakthrough**: Pre-trains Module 3 on >208,000 multilingual sentences across English, Italian, Spanish, and French in pure FP32 on TITAN X, driving character accuracy from 25.1% to **95.5%**, MoE routing accuracy from 41.0% to **98.0%**, reducing Soft-Levenshtein edit loss down to **0.153**, and locking a new project record validation loss of **`0.8496`**.
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

## Current Official Flagship: Phono-V7.5 Architecture & Complete Specification

Phono-V7.5 is the current official flagship architecture of AudioLearn. It completely supersedes monolithic end-to-end speech models by establishing a **fully decoupled, tripartite modular architecture** where acoustic phonetics, cross-modal latent alignment, and multilingual orthographic decoding are separated into dedicated, specialized sub-models.

A user or researcher can read this section independently to grasp **100% of the architecture, its dual-level Mixture-of-Experts (MoE), recursive character decoders, and loss functions** without consulting previous version notes.

---

### 1. High-Level System Architecture & Dataflow

```
═════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                                   MODULE 1: ACOUSTIC PHONEME FRONT-END (31.8M params)
═════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 [Raw Audio (16 kHz)]
        │
        ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │  HuBERT 7-Layer 1D Convolutional Feature Extractor (Temporal Strides: [5, 2, 2, 2, 2, 2, 2], Factor: 320x) │
 │  Emits 20ms acoustic frames (50 Hz) with LayerNorm & GELU -> [B, T_audio, 512]                              │
 └──────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │  8-Layer Multi-Scale Dilated Conformer Backbone (D=512, H=8, FFN=2048)                                      │
 │  • Macaron-style Feed-Forward blocks with 4 MoE FFN Experts per layer (Top-2 gating)                        │
 │  • Multi-Scale Dilated Depthwise Convolutions (k=31): Branch 1 (d=1, 620ms), Branch 2 (d=2, 1220ms),       │
 │    Branch 3 (d=4, 2420ms) -> 4x temporal receptive field capturing intra-word coarticulation & prosody      │
 └──────────────────────┬───────────────────────────────┬──────────────────────────────────────────────────────┘
                        │                               │
                        ▼                               ▼
 ┌────────────────────────────────────────┐   ┌────────────────────────────────────────────────────────────────┐
 │ Decoupled Word Boundary Gate Head      │   │ Recursive 2-Pass Phoneme Emission Head (RecursivePhonemeHead)  │
 │ 3-class classifier:                    │   │ • Pass 1: Base CTC projection h_t -> z_t^(0)                   │
 │ [0: SPEECH, 1: WORD_SPACE, 2: SILENCE] │   │ • Pass 2: Causal depthwise recurrent conv on [h_t; P(z_{t-1})] │
 │ Supervised with x4.0 space penalty     │   │   enforcing strict phonotactic grammar (no adjacent spaces)    │
 └──────────────────────┬─────────────────┘   └────────────────────────────────┬───────────────────────────────┘
                        │                                                      │
                        └───────────────────────┬──────────────────────────────┘
                                                │
                                                ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │  Online Event-Driven CTC Peak & Boundary Slicing (extract_streaming_ctc_slices)                             │
 │  Detects energy bursts E(t) = 1 - P(blank), confirms silences via 2-frame hysteresis, and slices            │
 │  continuous acoustic speech into discrete word chunks [B, L, 32_frames, 512] in real time (zero lookahead) │
 └──────────────────────────────────────────────┬──────────────────────────────────────────────────────────────┘
                                                │
                        ┌───────────────────────┴───────────────────────┐
                        │ Acoustic Word Latents & Slices                │ Fast Text/Phoneme Words
                        ▼                                               ▼
════════════════════════════════════════════════    ═════════════════════════════════════════════════════════════
  [Acoustic Audio Path]                              MODULE 2: CROSS-MODAL DISTILLED LATENT BRIDGE (1.51M params)
                                                    ═════════════════════════════════════════════════════════════
                                                     ┌─────────────────────────────────────────────────────────┐
                                                     │ FastTextPhonemeWordEncoder (2-Layer Transformer)        │
                                                     │ • Takes byte characters (0..127) or phonemes (0..63)    │
                                                     │ • AttentionPooling across token sequence                │
                                                     │ • 2-layer MLP projection [256 -> 512 -> 512]            │
                                                     │ • Distilled via InfoNCE + CosSim (98.7%) + MSE (0.015)  │
                                                     │ • Bypasses audio encoder: 50x faster for text training  │
                                                     └────────────────────────┬────────────────────────────────┘
                                                                              │
                                                ┌─────────────────────────────┘
                                                │ Canonical Word Latents z_word [B, L, 512]
                                                ▼
═════════════════════════════════════════════════════════════════════════════════════════════════════════════════
                         MODULE 3: DUAL-LEVEL MoE RECURSIVE TEXT DECODER (51.7M params)
═════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │  LEVEL 1: MACRO WORD DECODER (4 Layers, D=512, H=8, FFN=1536)                                               │
 │  • Shift-Invariant Base Query (q_base): shared learnable vector eliminating positional slot ceilings        │
 │  • Band-Causal Macro Attention (K=8 words): local historical attention window, constant O(K*L) complexity   │
 │  • Macro History Noise Injection (sigma=0.05): eliminates exposure bias during teacher forcing              │
 │  • Sliding Multi-Word Context Windows (W=6 words): collects preceding 5 words + current word for micro head │
 └──────────────────────┬──────────────────────────────────────────────────────────────────────────────────────┘
                        │
       ┌────────────────┴──────────────────────────────┐
       ▼                                               ▼
 ┌────────────────────────────────────────┐   ┌────────────────────────────────────────────────────────────────┐
 │ 7 Overlapping Length Experts MoE Router│   │ Continuous Word Length Predictor & Dynamic Horizon Capping     │
 │ 7 soft intervals:                      │   │ • Predicts continuous char length k_hat from [z_word; dur]     │
 │ [1-5], [3-7], [5-10], [7-12],          │   │ • Asymmetric Truncation Loss (x4.0 penalty when k_hat < k_true)│
 │ [9-15], [12-20], [15-30]               │   │ • Guarantees +2.7c to +3.5c safety headroom                    │
 │ • Multi-Positive BCE supervision       │   │ • Horizon Capping: min(HeadBound, ceil(k_hat) + 1)             │
 │ • Anti-Collapse Loss (L_balance)       │   │   eliminates trailing hallucination and saves micro FLOPs      │
 │ • Yields 98.0% - 98.6% routing accuracy│   └────────────────────────────────┬───────────────────────────────┘
 └──────────────────────┬─────────────────┘                                    │
                        │                                                      │
                        │ Expert Conditioning Embeddings & Length Bounds       │
                        ▼                                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │  LEVEL 2: MICRO CHARACTER RECURSIVE DECODER (WindowedMicroPartitionedRecursiveHead, 4 Layers, D=512)        │
 │                                                                                                             │
 │  Tri-Modal Attention per Layer:                                                                             │
 │  1. Causal Character Self-Attention over generated byte tokens (V=128)                                      │
 │  2. Direct Acoustic Cross-Attention to word speech slice (32 frames / ~640ms receptive field)               │
 │  3. Sliding Macro Context Cross-Attention to preceding W=6 word representations                             │
 │                                                                                                             │
 │  16 Partitioned Micro MoE Experts (Top-2 Gating):                                                           │
 │  • Short Words pool: Experts 0..4 (lengths 1 to 5)                                                          │
 │  • Medium Words pool: Experts 5..10 (lengths 5 to 9)                                                        │
 │  • Long Words pool: Experts 11..15 (lengths 10+)                                                            │
 │  • Dynamic softmax mask restricts routing strictly to the active length cluster                             │
 │                                                                                                             │
 │  Supervision & Losses:                                                                                      │
 │  • Autoregressive byte Cross-Entropy with label smoothing (0.05)                                            │
 │  • Differentiable Soft-Levenshtein Loss (DP edit-distance alignment with forward-backward gamma recursion)  │
 └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 2. Comprehensive Sub-Module Specifications

#### Module 1: Acoustic Phoneme Front-End (`PhonoV73AcousticBackbone`)
- **1D CNN Feature Extractor**:
  - 7 convolutional blocks with strides `[5, 2, 2, 2, 2, 2, 2]` and kernel sizes `[10, 3, 3, 3, 3, 2, 2]`.
  - Temporal downsampling factor of **$320\times$**: transforms raw 16 kHz waveform audio into 20ms acoustic frames (50 Hz).
  - Normalization: LayerNorm on channel dimension after each convolution + GELU activations.
- **8-Layer Conformer Backbone with MoE FFNs**:
  - Hidden dimension $D = 512$, 8 attention heads, Feed-Forward dimension $4D = 2048$.
  - Interleaved Macaron-style structure: half-step FFN $\to$ Multi-Head Self-Attention $\to$ Depthwise Convolution Module $\to$ half-step FFN $\to$ LayerNorm.
  - **4 FFN Experts per Conformer Layer** with Top-2 router gating and auxiliary load-balancing loss, specializing on varying phoneme classes (vowels, plosives, fricatives, nasals).
- **Multi-Scale Dilated Conformer Convolutions (`MultiScaleDilatedConformerConvModule`)**:
  - Extends the temporal receptive field from 620ms to **2,420ms** without parameter growth:
    - *Branch 1 ($k=31, d=1$)*: 620ms receptive field (preserves localized acoustic features).
    - *Branch 2 ($k=31, d=2$)*: 1,220ms receptive field (captures intra-word phonetic transitions).
    - *Branch 3 ($k=31, d=4$)*: 2,420ms receptive field (captures long-range prosody, rhythm, and inter-word liaisons).
  - Zero-initialized linear projection guarantees exact bitwise zero perturbation at Step 0.
- **Decoupled Word Boundary Gate Head (`DecoupledBoundaryGateHead`)**:
  - Independent 3-class linear classifier emitting boundary logits for each acoustic frame:
    - `0: SPEECH` (active voiced phoneme frames)
    - `1: WORD_SPACE` (inter-word boundary delimiter)
    - `2: SILENCE_BLANK` (ambient silence / non-emitting transition)
  - Supervised directly via forced alignment targets with a **$\times 4.0$ penalty** on false space deletions and insertions, eliminating word agglutination.
- **Recursive 2-Pass Phoneme Emission Head (`RecursivePhonemeHead`)**:
  - *Pass 1*: Base linear projection $h_t \to z_t^{(0)}$ predicting raw frame CTC probabilities.
  - *Pass 2 (Causal Depthwise Refiner)*: Causal 1D convolution ($k=5$) conditioned on concatenated $[h_t \,\|\, \operatorname{softmax}(z_{t-1}^{(0)})]$. Past emission feedback enforces phonotactic language grammar (e.g. preventing consecutive spaces or unvoiced transition anomalies) without future lookahead.
- **Online Event-Driven CTC Peak Slicing (`extract_streaming_ctc_slices`)**:
  - Tracks causal speech energy bursts $E(t) = 1 - P(\text{blank})$ in real time.
  - Incorporates 2-frame silence confirmation and 4-frame minimum burst protection.
  - Slices continuous audio into localized acoustic word tensors $[B, L, 32, 512]$ without knowing total utterance duration.

---

#### Module 2: Cross-Modal Distilled Latent Bridge (`FastTextPhonemeWordEncoder`)
- **Architecture**:
  - Ultra-compact 2-layer Transformer ($D_{\text{model}} = 256$, 4 attention heads, $\text{FFN} = 512$, $D_{\text{macro}} = 512$, **1.51M parameters**).
- **Dual-Modality Input Flexibility**:
  - *Byte Modality*: Character byte token sequences ($0..127$) for written orthography.
  - *Phoneme Modality*: IPA phoneme token sequences ($0..63$) for acoustic phonetics.
- **Attention Pooling & Projection**:
  - `AttentionPooling`: Softmax-weighted learnable attention pooling collapses token sequences into a fixed-size word vector while respecting padding masks.
  - 2-layer MLP projection with LayerNorm and GELU projects from $256 \to 512$, aligning with canonical acoustic word latents $z_{\text{word}}$.
- **Tripartite Distillation Loss**:
  $$\mathcal{L}_{\text{distill}} = \text{MSE}(z_{\text{pred}}, z_{\text{word}}) + \big(1 - \operatorname{CosSim}(z_{\text{pred}}, z_{\text{word}})\big) + \lambda \mathcal{L}_{\text{InfoNCE}}$$
- **Empirical Fidelity & Compute Advantage**:
  - Achieves **98.7% Cosine Similarity** ($0.987$) and $\text{MSE} \le 0.015$ with true Conformer acoustic word embeddings.
  - Runs **$50\times$ faster** than full audio processing, enabling the character decoder to be pre-trained on millions of text tokens at **0.6 steps/s (BS=32)** on a single TITAN X.

---

#### Module 3: Dual-Level MoE Recursive Character Decoder (`OverlappingLengthMoEDecoder`)

##### Level 1: Macro Word Decoder
- **4 Transformer Layers** ($D = 512$, 8 attention heads, $\text{FFN} = 1536$).
- **Shift-Invariant Base Query ($\mathbf{q}_{\text{base}}$)**: Shared learnable base query vector eliminates hard positional slot limits. Word slot 100 has the same trained capacity as word slot 0.
- **Band-Causal Local Macro Attention ($K=8$ words)**: Enforces a strictly causal local attention window over the past 8 words, bounding error snowballing and providing constant $O(K \cdot L)$ computation.
- **Macro History Noise Injection ($\sigma = 0.05$)**: Injects Gaussian jitter into historical word representations during training, eliminating exposure bias and forcing the decoder to attend to grounding evidence.
- **Sliding Multi-Word Context Windows ($W=6$ words)**: For every word $l$, packages representations from $[w_{l-5}, \dots, w_l]$ into context keys and values for the micro character decoder.

##### 7 Overlapping Word-Length Experts MoE Router
- **7 Soft Overlapping Length Intervals**:
  - Expert 0: `[1, 5]` (Very Short: particles, articles, acronyms)
  - Expert 1: `[3, 7]` (Short: common nouns, auxiliary verbs)
  - Expert 2: `[5, 10]` (Medium-Short: regular vocabulary)
  - Expert 3: `[7, 12]` (Medium: standard vocabulary)
  - Expert 4: `[9, 15]` (Medium-Large: compound words)
  - Expert 5: `[12, 20]` (Large: complex terminology, conjugated forms)
  - Expert 6: `[15, 30]` (Very Large: technical jargon, agglutinated expressions)
- **Multi-Positive Binary Cross-Entropy Loss**: Unlike rigid bins that create boundary instability, any expert covering the word length receives positive gradient reinforcement.
- **Anti-Collapse Load Balancing Loss**:
  $$\mathcal{L}_{\text{balance}} = 7 \sum_{e=0}^{6} f_e P_e$$
  Prevents expert collapse and guarantees uniform specialization, reaching **98.0%–98.6% routing accuracy**.
- **Expert Conditioning Embeddings**: Selected top expert generates a 512-dim embedding injected into the micro character decoder.

##### Continuous Length Guidance & Dynamic Horizon Capping
- **Word Length Predictor (`WordLengthPredictor`)**: Continuous character length prediction $\hat{k} = \operatorname{Softplus}(\operatorname{MLP}([z_{\text{word}}; \text{duration}]))$.
- **Asymmetric Truncation Loss**:
  $$\mathcal{L}_{\text{len}} = \begin{cases} \beta_{\text{under}} \cdot (k_{\text{true}} - \hat{k}) & \text{if } \hat{k} < k_{\text{true}} \ (\beta_{\text{under}} = 4.0) \\ \beta_{\text{over}} \cdot \operatorname{ReLU}(\hat{k} - k_{\text{true}} - \delta) & \text{if } \hat{k} \ge k_{\text{true}} \ (\beta_{\text{over}} = 0.5) \end{cases}$$
  Penalizes under-prediction $8\times$ more heavily than over-prediction, maintaining a consistent $+2.7\text{c}$ to $+3.5\text{c}$ safety headroom.
- **Dynamic Horizon Capping**: Autoregressive rollout is strictly bounded to $\min(\text{HeadBound},\, \lceil \hat{k} \rceil + 1)$, cutting trailing character hallucination and saving micro decoder FLOPs.

##### Level 2: Micro Character Recursive Decoder (`WindowedMicroPartitionedRecursiveHead`)
- **4 Recursive Transformer Layers** ($D = 512$, 8 attention heads, $\text{FFN} = 1536$).
- **Tri-Modal Cross-Attention per Layer**:
  1. *Causal Character Self-Attention*: Models orthographic sequences and byte dependencies.
  2. *Direct Acoustic Cross-Attention*: Attends directly to the 32 acoustic frames (~640ms) corresponding to the current acoustic word slice.
  3. *Sliding Macro Context Cross-Attention*: Cross-attends across the preceding $W=6$ macro word latents, resolving grammatical agreements and homophone ambiguities.
- **16 Partitioned Micro MoE Experts with Top-2 Routing**:
  - Expert pool partitioned by word length:
    - *Short Pool (Experts 0..4)*: active when length $\le 5$.
    - *Medium Pool (Experts 5..10)*: active when length $5 < k \le 9$.
    - *Long Pool (Experts 11..15)*: active when length $\ge 10$.
  - Softmax router dynamically applies $-\infty$ masks outside the selected pool, focusing compute where needed.
- **Differentiable Soft-Levenshtein DP Loss**:
  Supervises sequence generation through forward-backward DP edit-distance recursion ($\gamma = 0.2$, weight $0.2$), directly penalizing insertions, deletions, and substitutions.
- **Byte Vocabulary ($V=128$)**:
  Weight-tied character embedding and output LM head with label smoothing ($0.05$).

---

### 3. Unified Parameter Breakdown & Loss Formulation

| Component Module | Layer Details | Parameters |
| :--- | :--- | :---: |
| **Module 1: Acoustic Front-End** | 7-layer 1D CNN + 8-layer Conformer (4 MoE experts/layer) + Decoupled Boundary Gate + Recursive Phoneme Head | **31.82 M** |
| **Module 2: Distilled Latent Bridge** | 2-layer Transformer + AttentionPooling + 2-layer MLP projection ($256 \to 512 \to 512$) | **1.51 M** |
| **Module 3: Level-1 Macro Decoder** | 4-layer Band-Causal Macro Transformer ($K=8$, $W=6$) + 7 Overlapping Length Experts MoE Router | **17.45 M** |
| **Module 3: Level-2 Micro Decoder** | 4-layer Tri-Modal Recursive Decoder + 16 Partitioned MoE Experts (Top-2) + Byte LM Head | **34.22 M** |
| **Total Phono-V7.5 Flagship System** | **Complete Tripartite Architecture (Module 1 + Module 2 + Module 3)** | **85.00 M** |

#### Unified Training Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{char}} + \lambda_{\text{path}} \mathcal{L}_{\text{path}} + \lambda_{\text{balance}} \mathcal{L}_{\text{balance}} + \lambda_{\text{len}} \mathcal{L}_{\text{len}} + \lambda_{\text{lev}} \mathcal{L}_{\text{lev}} + \lambda_{\text{moe}} \mathcal{L}_{\text{moe}}$$
where:
- $\mathcal{L}_{\text{char}}$: Cross-entropy loss on byte character tokens with label smoothing ($0.05$).
- $\mathcal{L}_{\text{path}}$: Multi-positive Binary Cross-Entropy loss on the 7 overlapping length intervals.
- $\mathcal{L}_{\text{balance}}$: Anti-collapse load-balancing loss on macro MoE experts ($7 \sum f_e P_e$).
- $\mathcal{L}_{\text{len}}$: Asymmetric duration-guided word length loss ($\beta_{\text{under}}=4.0, \beta_{\text{over}}=0.5$).
- $\mathcal{L}_{\text{lev}}$: Differentiable Soft-Levenshtein edit-distance alignment loss ($\gamma=0.2$, weight $0.2$).
- $\mathcal{L}_{\text{moe}}$: Auxiliary load-balancing loss on the 16 micro MoE experts.

---

### 4. Text Middle-Training Empirical Performance Milestones

Module 3 is pre-trained via **Text Middle-Training** across >208,000 sentences in 4 languages (**English, French, Italian, Spanish**) in pure FP32 on NVIDIA TITAN X:

| Milestone / Training Step | Total Validation Loss | Byte Character Accuracy | MoE Path Accuracy (7 Experts) | Soft-Levenshtein Edit Loss |
| :--- | :---: | :---: | :---: | :---: |
| **Step 0 (Random Init)** | `4.7798` | 25.1% | 41.0% | 1.482 |
| **Step 1,000** | `1.8429` | 73.0% | 89.8% | 0.630 |
| **Step 2,000** | `1.4146` | 81.7% | 92.8% | 0.412 |
| **Step 3,000** | `1.2183` | 86.5% | 94.7% | 0.315 |
| **Step 5,000** | `1.0305` | 91.0% | 97.0% | 0.220 |
| **Step 8,000** | `0.9004` | 94.6% | 98.6% | 0.160 |
| **Step 10,000 (Active Milestone)** | **`0.8496`** | **`95.5%`** | **`98.0%`** | **`0.153`** |

Poids enregistrés et validés : [`checkpoints/phono_v7_5_text_middle/best_decoder.pt`](file:///home/nathan/github/audiolearn/checkpoints/phono_v7_5_text_middle/best_decoder.pt).

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
