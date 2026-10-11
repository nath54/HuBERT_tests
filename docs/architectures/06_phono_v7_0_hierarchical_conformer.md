# Phono-V7.0: Conformer Backbone, Two-Level Character Decoding & Soft-Levenshtein Loss

## 1. Overview & Motivation
**Phono-V7.0** redesigned the acoustic backbone and text generation architecture:
1. Replaced standard Transformer self-attention with **Conformer layers** (interleaving depthwise separable convolutions with multi-head self-attention).
2. Introduced **Two-Level Hierarchical Decoding**: separating word-level acoustics (Macro Word Decoder) from byte-level character generation (Micro Character Decoder).
3. Developed **Differentiable Soft-Levenshtein Loss** and **Asymmetric Truncation Loss** to train word length estimation and character alignment directly.

---

## 2. Conformer Acoustic Backbone

Each Conformer block (`src/models/conformer_layers.py`) combines the global context of multi-head self-attention with the inductive bias of local depthwise convolutions:

```
Input x
  │
  ├────────────────────────────────────────┐ (Residual)
  ▼                                        │
Pre-LN Multi-Head Self-Attention ──► (+) ◄─┘
  │
  ├────────────────────────────────────────┐ (Residual)
  ▼                                        │
Conformer Convolution Module     ──► (+) ◄─┘
  [Pre-LN -> Pointwise Conv1 -> GLU -> Depthwise Conv -> BatchNorm -> SiLU -> Pointwise Conv2]
  │
  ├────────────────────────────────────────┐ (Residual)
  ▼                                        │
Pre-LN Feed-Forward Network      ──► (+) ◄─┘
  │
  ▼
Output
```

---

## 3. Two-Level Hierarchical Decoding

Instead of decoding all characters in an utterance in one massive sequence, Phono-V7.0 breaks the problem hierarchically:

```
Audio Utterance -> Conformer -> Acoustic Word Boundaries (via CTC Forced Aligner)
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
             [Acoustic Word 1]                     [Acoustic Word 2]
                    │                                     │
                    ▼                                     ▼
        Level-1: Macro Word Decoder           Level-1: Macro Word Decoder
     Predicts continuous word length k_hat    Predicts continuous word length k_hat
                    │                                     │
                    ▼                                     ▼
        Level-2: Micro Character Decoder      Level-2: Micro Character Decoder
        Autoregressive rollout of characters  Autoregressive rollout of characters
        e.g. ['c', 'a', 't', '<eow>']         e.g. ['r', 'u', 'n', '<eow>']
```

### Word Boundary Slicing via CTC Forced Aligner
Phono-V7.0 computes speech energy bursts $E(t) = 1 - P(\text{blank})$ and runs a dynamic programming forced aligner (`src/models/forced_aligner.py`) to segment the continuous acoustic stream into discrete acoustic word slices without requiring external alignment tools.

---

## 4. Loss Formulations

### Asymmetric Truncation Loss ($\mathcal{L}_{\text{length}}$)
The Macro Word Decoder predicts continuous character length $\hat{k} = \operatorname{Softplus}(W_k h_{\text{macro}})$. Because predicting too few characters truncates words catastrophically, an asymmetric penalty is applied:

$$\mathcal{L}_{\text{length}} = \begin{cases} 3.0 \cdot (k_{\text{true}} - \hat{k})^2, & \text{if } \hat{k} < k_{\text{true}} \\ 1.0 \cdot (\hat{k} - k_{\text{true}})^2, & \text{if } \hat{k} \ge k_{\text{true}} \end{cases}$$

### Differentiable Soft-Levenshtein Loss ($\mathcal{L}_{\text{lev}}$)
To supervise character sequences without requiring rigid step-by-step alignments, Phono-V7.0 implements a differentiable dynamic programming matrix using forward-backward recursion over insertion, deletion, and substitution costs:

$$D(i, j) = \operatorname{softmin}\left( D(i-1, j) + c_{\text{del}},\; D(i, j-1) + c_{\text{ins}},\; D(i-1, j-1) + c_{\text{sub}}(y_i, \hat{y}_j) \right)$$

---

## 5. Source Code Pointers

* Full Speech Model: [`src/models/phono_v7_speech_model.py`](file:///home/nathan/github/audiolearn/src/models/phono_v7_speech_model.py) (`PhonoV7SpeechModel`)
* Forced Aligner: [`src/models/forced_aligner.py`](file:///home/nathan/github/audiolearn/src/models/forced_aligner.py) (`CTCForcedAligner`)
* Conformer Layers: [`src/models/conformer_layers.py`](file:///home/nathan/github/audiolearn/src/models/conformer_layers.py)
