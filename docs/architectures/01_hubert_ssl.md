# HuBERT: Acoustic K-Means Self-Supervised Learning

## 1. Overview & Motivation
**HuBERT** (Hidden-Unit BERT, Hsu et al., 2021) is a self-supervised speech representation architecture implemented in AudioLearn. Unlike end-to-end ASR models that require paired text transcripts, HuBERT learns foundational acoustic representations directly from unlabelled continuous speech waveforms.

The core intuition is that speech contains both continuous acoustic details (speaker identity, pitch, noise) and discrete phonetic units. HuBERT discovers these discrete phonetic units autonomously by clustering 39-dimensional acoustic MFCC vectors using $k$-means into $K=100$ discrete cluster indices, then training a Transformer encoder via masked prediction over time.

---

## 2. Input Representation & Feature Pipeline

* **Input Waveform**: 16 kHz mono raw audio $x \in \mathbb{R}^T$ (e.g., 1.0s to 30.0s).
* **Feature Extraction**: 7-layer temporal 1D convolution block with downsampling strides $[5, 2, 2, 2, 2, 2, 2]$, total downsampling factor $320\times$.
  * Each audio frame represents **20 ms** of speech ($16000 / 320 = 50\text{ Hz}$).
* **Cluster Targets ($K=100$)**: 39-dimensional MFCCs (13 static + 13 $\Delta$ + 13 $\Delta\Delta$) are extracted offline or in RAM and quantized into cluster IDs $c_t \in \{0, \dots, 99\}$.

---

## 3. Layer-by-Layer Architectural Pipeline

```
Raw Audio x [Batch, Samples] (16 kHz)
   │
   ▼
┌────────────────────────────────────────────────────────┐
│  HuBERTFeatureEncoder (7-Layer 1D Convolutions)        │
│  Channels: (1, 512, 512, 512, 512, 512, 512, 512)      │
│  Strides:  (5, 2, 2, 2, 2, 2, 2) -> Total 320x         │
└──────────────────────────┬─────────────────────────────┘
                           │ Features [Batch, Time, 512]
                           ▼
┌────────────────────────────────────────────────────────┐
│  Time-Masking Module (Span Masking)                   │
│  Replaces ~8% of spans (length 10) with learnable mask  │
└──────────────────────────┬─────────────────────────────┘
                           │ Masked Features
                           ▼
┌────────────────────────────────────────────────────────┐
│  Transformer Contextual Encoder                        │
│  - Convolutional Positional Embedding (kernel 128)    │
│  - L Transformer Layers (Self-Attention + FFN)         │
└──────────────────────────┬─────────────────────────────┘
                           │ Hidden States h_t [Batch, Time, D]
                           ▼
┌────────────────────────────────────────────────────────┐
│  K-Means Cluster Projection Head                       │
│  Linear(D, 100) -> Cross-Entropy on Masked Frames      │
└────────────────────────────────────────────────────────┘
```

### Component Details
1. **CNN Feature Extractor (`src/models/cnn_encoder.py`)**:
   Projects raw 1D pressure waves into continuous acoustic embeddings using Gelu activations and layer/group normalization.
2. **Convolutional Positional Embedding (`ConvolutionalPositionalEmbedding`)**:
   A 1D depthwise convolution with kernel size 128 and 16 groups followed by GELU activation, encoding relative positional timing without rigid absolute sinusoid encodings.
3. **Transformer Contextual Encoder (`src/models/transformer.py`)**:
   Stack of multi-head self-attention blocks with pre-LayerNorm, projecting acoustic features across long temporal horizons.
4. **Cluster Classifier Head**:
   A single linear layer projecting hidden states $h_t \in \mathbb{R}^D$ into 100 logits representing the $k$-means pseudo-labels.

---

## 4. Mathematical Formulation

Let $z_t = \text{CNN}(x)_t$ be the continuous feature vector at frame $t$. A binary mask $M \in \{0, 1\}^T$ selects time frames to obscure with a learnable vector $e_{\text{mask}}$:

$$\tilde{z}_t = \begin{cases} e_{\text{mask}}, & \text{if } M_t = 1 \\ z_t, & \text{if } M_t = 0 \end{cases}$$

The Transformer contextualizes the masked sequence:
$$H = \text{Transformer}(\tilde{Z})$$

The prediction loss is evaluated **strictly over masked frames**:
$$\mathcal{L}_{\text{HuBERT}} = - \frac{1}{\sum_t M_t} \sum_{t: M_t = 1} \log P(c_t \mid h_t) = - \frac{1}{\sum_t M_t} \sum_{t: M_t = 1} \log \frac{\exp(w_{c_t}^\top h_t + b_{c_t})}{\sum_{k=1}^K \exp(w_k^\top h_t + b_k)}$$

---

## 5. Source Code Pointers

* Model Implementation: [`src/models/hubert_pretrain.py`](file:///home/nathan/github/audiolearn/src/models/hubert_pretrain.py) (`HuBERTForPreTraining`)
* Feature Extractor: [`src/models/cnn_encoder.py`](file:///home/nathan/github/audiolearn/src/models/cnn_encoder.py) (`HuBERTFeatureEncoder`)
* Transformer Layers: [`src/models/transformer.py`](file:///home/nathan/github/audiolearn/src/models/transformer.py) (`TransformerEncoder`)
* Configuration: [`src/models/config.py`](file:///home/nathan/github/audiolearn/src/models/config.py) (`HuBERTConfig`)
