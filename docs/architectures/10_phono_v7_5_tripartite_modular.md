# Phono-V7.5: Complete Tripartite Modular Architecture & Dual-Level MoE

## 1. Overview & Motivation
**Phono-V7.5** deconstructs monolithic speech recognition into **3 decoupled, independently optimizable sub-models**. 

Monolithic ASR models couple acoustic feature extraction with language modeling. When speech data is scarce or noisy, the text decoder fails to learn vocabulary and grammar. In Phono-V7.5:
* The acoustic model operates strictly in the phonetic domain (where acoustic labels are clean).
* An ultra-fast distilled bridge projects text into the acoustic latent space.
* The character decoder is pre-trained on massive text corpora independently of audio, achieving near-perfect language generation prior to acoustic coupling.

---

## 2. Tripartite Architectural Blueprint

```
Audio Waveform (16 kHz)
       │
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│  MODULE 1: Acoustic Phoneme Front-End (31.8M params)                   │
│  • 7-Layer 1D CNN Feature Extractor (Stride 320x, 20ms frames)         │
│  • 8-Layer Conformer Backbone (D=512, 4 MoE Experts per layer)         │
│  • Multi-Scale Dilations (d=1, 2, 4 -> 2,420ms Receptive Field)        │
│  • Decoupled Boundary Gate Head (Speech / Space / Blank)               │
│  • Recursive 2-Pass Phoneme Head (Causal grammar refinement)           │
│  • Online Event-Driven CTC Peak Slicing                                │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Acoustic Word Latents z_word [B, 512]
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│  MODULE 3: Dual-Level MoE Character Decoder (51.7M params)             │
│                                                                        │
│  Level-1 Macro Word Decoder (4 Layers, D=512):                         │
│  • Band-Causal Macro Attention (K=8 past words)                        │
│  • 7 Overlapping Word-Length MoE Router ([1-5], [3-7], [5-10]...)      │
│  • Continuous Length Head k_hat (Asymmetric Truncation Penalty x4.0)   │
│                                                                        │
│  Level-2 Micro Character Recursive Decoder (4 Layers, D=512):          │
│  • Tri-Modal Cross-Attention (Chars, Acoustic Slices, Macro Context)   │
│  • 16 Partitioned Micro MoE Experts (Short 0..4, Med 5..10, Long 11..15)│
│  • Horizon Capping: Rollout strictly capped at min(HeadBound, ceil(k)+1)│
│  • Differentiable Soft-Levenshtein Loss DP Alignment                   │
└────────────────────────────────────────────────────────────────────────┘
                                   ▲
                                   │ Latent Alignment (98.7% Cosine Sim)
┌──────────────────────────────────┴─────────────────────────────────────┐
│  MODULE 2: Distilled Latent Bridge (FastTextPhonemeWordEncoder, 1.51M) │
│  Maps raw text bytes + phonemes directly to z_word at 50x speed.       │
│  Enables massive text middle-training without running audio forward.   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Sub-Model Specifications

### Module 1: Acoustic Phoneme Front-End (`PhonoV73AcousticBackbone`)
* **Vocabulary**: 112 unified multilingual phoneme tokens.
* **Architecture**: 7-layer CNN + 8-layer Conformer with 4 routed MoE experts per layer (Top-2 gating).
* **Innovations**: Multi-scale dilated convolutions ($4\times$ temporal receptive field: 2,420ms) and decoupled 3-class boundary gate head (`SPEECH`, `WORD_SPACE`, `SILENCE_BLANK`).

### Module 2: Distilled Latent Bridge (`FastTextPhonemeWordEncoder`)
* **Purpose**: Allows Module 3 to be trained on hundreds of thousands of written sentences without synthesizing or streaming audio.
* **Architecture**: 2-layer Transformer ($D=256$, $H=4$, FFN=512) with multi-head attention pooling, projecting text to the canonical 512-dim acoustic word latent space ($z_{\text{word}}$).
* **Fidelity**: Achieves **98.7% Cosine Similarity** and MSE of $0.015$ against full Conformer audio representations while running **$50\times$ faster**.

### Module 3: Dual-Level MoE Character Decoder (`PhonoV75OverlappingMoEDecoder`)
* **7 Overlapping Word-Length MoE Experts**:
  Overcomes rigid integer boundaries by using soft overlapping interval targets:
  `[1-5]`, `[3-7]`, `[5-10]`, `[7-12]`, `[9-15]`, `[12-20]`, `[15-30]`.
  Trained with multi-positive Binary Cross-Entropy routing and an anti-collapse load-balancing loss, reaching **98.0%–98.6% routing accuracy**.
* **16 Partitioned Micro MoE Experts**:
  Routes character tokens into length-specialized expert clusters (Short, Medium, Long) with dynamic horizon capping.

---

## 4. Text Middle-Training Breakthrough

By decoupling Module 3 via the Module 2 bridge, Phono-V7.5 introduced **Text Middle-Training**:
* Trained on **> 208,000 multilingual sentences** across English, Italian, Spanish, and French in pure FP32 on TITAN X.
* Character accuracy surged from $25.1\% \rightarrow \mathbf{95.5\%}$.
* MoE routing accuracy reached **$98.0\%$**.
* Soft-Levenshtein edit-distance loss dropped to **$0.153$**, achieving a milestone validation loss of **`0.8496`**.

---

## 5. Source Code Pointers

* Full Tripartite Model & Decoders: [`src/models/phono_v7_5_speech_model.py`](file:///home/nathan/github/audiolearn/src/models/phono_v7_5_speech_model.py)
* Distilled Latent Bridge: [`src/models/text_phoneme_encoder.py`](file:///home/nathan/github/audiolearn/src/models/text_phoneme_encoder.py) (`FastTextPhonemeWordEncoder`)
* Acoustic Front-End Training: `scripts/train_phono_v7_5_acoustic.py`
* Decoder Text Middle-Training: `scripts/train_phoneme_to_text.py`
