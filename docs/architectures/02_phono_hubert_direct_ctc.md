# PhonoHuBERT: Direct Acoustic-to-Phoneme CTC & Articulatory Hierarchies (V1–V2)

## 1. Overview & Motivation
**PhonoHuBERT** replaces cluster pseudo-labels with **direct phoneme prediction** using Connectionist Temporal Classification (CTC). In speech recognition, subword/BPE models suffer when mapping ambiguous acoustics directly to irregular spelling (especially in English). Phonemes serve as the optimal bottleneck: acoustic signals map directly and deterministically to phonetic pronunciations.

Phono-V1 and V2 introduce:
1. **Unified IPA Phoneme Vocabulary**: 64 fine-grained IPA tokens covering vowels, consonants, diphthongs, and special acoustic tokens.
2. **Articulatory Macro Guidance**: An intermediate 8-class articulatory hierarchy grouping phonemes by phonetic manner of production.
3. **Dynamic SpecAugment**: Data augmentation directly on latent convolutional feature maps before the Transformer.

---

## 2. Input & Vocabulary Specification

* **Audio Input**: 16 kHz raw waveforms.
* **Frame Rate**: Downsampled $320\times$ by a 7-layer 1D CNN to 50 Hz (1 frame per 20ms).
* **Phoneme Vocabulary ($V = 64$)**:
  * Special tokens: `<pad>`, `<blank>`, `<silence>`, `<noise>`, `<mask>`, `<eos>`, `<unk>`
  * IPA Vowels: `i, ɪ, e, ɛ, æ, ɑ, ɔ, o, ʊ, u, ʌ, ə, aɪ, aʊ, ɔɪ, eɪ, oʊ...`
  * IPA Consonants: `p, b, t, d, k, g, f, v, θ, ð, s, z, ʃ, ʒ, h, tʃ, dʒ, m, n, ŋ, l, ɹ, j, w...`
* **Articulatory Macro Groups ($C = 8$)**:
  `{Non-Speech, Prosody/Punctuation, Vowels, Stops, Fricatives, Nasals, Liquids, Affricates}`.

---

## 3. Architecture Pipeline

```
Raw Audio x [Batch, Samples]
   │
   ▼
┌────────────────────────────────────────────────────────┐
│  7-Layer CNN Feature Extractor (Stride 320x)           │
└──────────────────────────┬─────────────────────────────┘
                           │ Continuous Features [Batch, Time, 512]
                           ▼
┌────────────────────────────────────────────────────────┐
│  Dynamic SpecAugment Module                            │
│  - Time Masking: Blocks of frames zeroed out           │
│  - Channel Masking: Blocks of feature dims zeroed out   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Transformer Contextual Encoder (L Layers, D=512)      │
└──────────────────────────┬─────────────────────────────┘
                           │ Hidden States h_t
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│ Level 1: Macro Head      │  │ Level 2: Phoneme Head    │
│ Linear(D, 8)             │  │ Linear(D, 64)            │
│ Articulatory Classes     │  │ Fine IPA Classes         │
└──────────────────────────┘  └──────────────────────────┘
```

### Articulatory Hierarchical Conditioning
In the hierarchical variant (`src/models/phono_hubert_hierarchical.py`), the fine phoneme head is directly conditioned on articulatory macro logits:
$$z_t^{\text{macro}} = W_{\text{macro}} h_t + b_{\text{macro}}$$
$$z_t^{\text{fine}} = W_{\text{fine}} h_t + W_{\text{proj}} \operatorname{softmax}(z_t^{\text{macro}}) + b_{\text{fine}}$$

---

## 4. Loss Formulation: Multi-Task Hierarchical CTC

The model is trained end-to-end with Connectionist Temporal Classification (CTC), allowing alignment-free supervision:

$$\mathcal{L}_{\text{CTC}}(z, y) = - \log \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_{t=1}^T P(\pi_t \mid z_t)$$

The composite objective trains both the fine IPA phoneme sequence $y_{\text{phoneme}}$ and the articulatory macro sequence $y_{\text{macro}}$:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CTC}}(z^{\text{fine}}, y_{\text{phoneme}}) + \lambda_{\text{macro}} \mathcal{L}_{\text{CTC}}(z^{\text{macro}}, y_{\text{macro}})$$
where $\lambda_{\text{macro}} = 0.30$.

---

## 5. Source Code Pointers

* Base Phoneme Model: [`src/models/phono_hubert.py`](file:///home/nathan/github/audiolearn/src/models/phono_hubert.py) (`PhonoHuBERTForPreTraining`)
* Dual/Articulatory Model: [`src/models/phono_hubert_dual.py`](file:///home/nathan/github/audiolearn/src/models/phono_hubert_dual.py)
* Hierarchical Architecture: [`src/models/phono_hubert_hierarchical.py`](file:///home/nathan/github/audiolearn/src/models/phono_hubert_hierarchical.py)
* Recursive Model: [`src/models/phono_hubert_recursive.py`](file:///home/nathan/github/audiolearn/src/models/phono_hubert_recursive.py)
