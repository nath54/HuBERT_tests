# Phono-V6.7: Continuous Speech Double-Loss Architecture

## 1. Overview & Motivation
**Phono-V6.7** is the first end-to-end continuous speech recognition model in AudioLearn that unifies phoneme acoustics and written character generation into a single end-to-end differentiable network.

Prior systems either output phonemes without generating readable text or attempted to map raw audio directly to characters (skipping the phonetic bottleneck). Phono-V6.7 introduces the **Double-Loss Framework**:
* **Loss 1 (Acoustic Phoneme CTC)**: Forces the acoustic encoder to learn precise phonetic boundaries and pronunciations.
* **Loss 2 (Autoregressive Character Cross-Entropy)**: Forces the decoder to learn language spelling, orthography, and word syntax conditioned on the acoustic representations.

---

## 2. Architectural Pipeline

```
Audio Input x [Batch, Samples]
   │
   ▼
┌────────────────────────────────────────────────────────┐
│  7-Layer CNN Feature Extractor (Stride 320x)           │
└──────────────────────────┬─────────────────────────────┘
                           │ 50 Hz Acoustic Embeddings
                           ▼
┌────────────────────────────────────────────────────────┐
│  8-Layer MoE Conformer Encoder (D=512, 4 Experts)       │
└──────────────────────────┬─────────────────────────────┘
                           │ Acoustic Representations h_t
             ┌─────────────┴─────────────┐
             ▼                           ▼
┌──────────────────────────┐   ┌─────────────────────────────────────────┐
│ Acoustic Phoneme Head    │   │ Autoregressive Micro Character Decoder  │
│ Linear(D, 64)            │   │ Cross-Attends to h_t                    │
│ Loss 1: CTC Phoneme Loss │   │ Loss 2: Character Cross-Entropy Loss    │
└──────────────────────────┘   └─────────────────────────────────────────┘
```

---

## 3. Double-Loss Formulation

During training, an audio utterance with paired phoneme targets $y_{\text{ph}}$ and character transcript $y_{\text{char}}$ produces two concurrent losses:

### Loss 1: Acoustic Phoneme CTC
$$\mathcal{L}_{\text{CTC}} = - \log \sum_{\pi \in \mathcal{B}^{-1}(y_{\text{ph}})} \prod_{t=1}^T P(\pi_t \mid h_t)$$

### Loss 2: Character Autoregressive Cross-Entropy
$$\mathcal{L}_{\text{char}} = - \frac{1}{U} \sum_{u=1}^U \log P(y_{\text{char}, u} \mid y_{\text{char}, <u}, h)$$

### Joint Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CTC}} + \lambda_{\text{char}} \mathcal{L}_{\text{char}} + \lambda_{\text{balance}} \mathcal{L}_{\text{balance}}$$
where $\lambda_{\text{char}} = 1.0$ and $\lambda_{\text{balance}} = 0.05$.

This formulation prevents the character decoder from degrading the acoustic representations while ensuring that phonetic representations are immediately grounded into readable text.

---

## 4. Source Code Pointers

* Full Speech Model: [`src/models/phono_v6_7_speech_model.py`](file:///home/nathan/github/audiolearn/src/models/phono_v6_7_speech_model.py) (`PhonoV67SpeechModel`)
* Windowed Decoder: [`src/models/phono_v6_7_windowed_decoder.py`](file:///home/nathan/github/audiolearn/src/models/phono_v6_7_windowed_decoder.py)
* Training Script: `scripts/train_phono_v6_7_speech.py`
