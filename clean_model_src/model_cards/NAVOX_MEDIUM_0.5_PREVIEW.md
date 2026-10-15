# Model Card: NaVox-Medium-0.5-Preview

[![Architecture](https://img.shields.io/badge/Architecture-Tripartite_MoE-blue.svg)](https://github.com/nath54/HuBERT_tests)
[![Status](https://img.shields.io/badge/Release_Status-Preview_v0.5-yellow.svg)](#overview)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Evaluation](https://img.shields.io/badge/Validation-Empirically_Measured-brightgreen.svg)](#official-evaluation-results)

**NaVox-Medium-0.5-Preview** is the official intermediate checkpoint release of the NaVox architecture family, trained on the **400,000 utterance multilingual dataset** (English 🇬🇧, French 🇫🇷, Spanish 🇪🇸, Italian 🇮🇹).

It serves as the empirical milestone model preceding the full 3-stage NaVox v1.0 suite.

---

## 1. Architecture Specification

* **Total Parameters:** ~285 Million (Model 1 Base) + ~35 Million (P2G Decoders)
* **Active Parameters per Audio Frame:** ~133 Million (sparse execution via DeepSeek-style MoE)
* **Model 1 (Acoustic Conformer MoE):**
  * 7-layer 1D CNN Frontend (320x downsampling $\to$ 50 Hz frame rate)
  * 12 Conformer Blocks with Rotary Positional Embeddings (RoPE)
  * DeepSeek MoE FFN: 1 Shared Expert + 8 Routed Experts (Top-2 routing)
  * Phoneme Engram Memory: Layers 6 & 10 (polynomial hash tables)
  * Intermediate Draft Supervision: Layer 4 Draft Head
* **Model 2 & 3 (P2G Decoders):**
  * Model 2: FastText-style Phoneme Word Encoder $\to \mathbf{z}_{\text{word}} \in \mathbb{R}^{512}$
  * Model 3: Overlapping Length Experts MoE Router (7 intervals) with Band-Causal Attention

---

## 2. Official Evaluation Results (Measured on 4,000-Utterance Multilingual Benchmark)

*The following scores are **100% empirical measurements** evaluated directly on the balanced 4,000-utterance test split (`data/full_multilingual_val_4k.json` with 1,000 utterances per language) at Step 14,000.*

### Overall Multilingual Metrics

| Metric | Score | Notes |
| :--- | :---: | :--- |
| **Acoustic Phoneme Error Rate (PER)** | **10.10%** | Pure acoustic phoneme sequence distance |
| **Overall Phoneme Error Rate (PER)** | **10.45%** | Including space delimiter tokens |
| **Acoustic Space Boundary Accuracy** | **93.15%** | Precision/Recall on acoustic word boundary token `8` |
| **Estimated Phoneme Word Error Rate (P-WER)** | **34.58%** | Word-level error before linguistic orthography correction |
| **Full End-to-End Pipeline Text WER** | **43.45%** | Raw unconstrained greedy character rollout (no trie/LM) |
| **Full End-to-End Pipeline Text CER** | **18.53%** | Character-level edit distance across all 4 languages |

---

### Per-Language Breakdown

| Language | Utterances | Acoustic PER (%) | P-WER (%) | End-to-End Text WER (%) | Text CER (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🇬🇧 **English (EN)** | 1,000 | **8.4%** | **28.2%** | **45.9%** | **21.7%** |
| 🇪🇸 **Spanish (ES)** | 1,000 | **8.4%** | **29.3%** | **34.3%** | **14.5%** |
| 🇮🇹 **Italian (IT)** | 1,000 | **11.1%** | **40.7%** | **43.4%** | **15.2%** |
| 🇫🇷 **French (FR)** | 1,000 | **14.1%** | **43.0%** | **46.9%** | **22.2%** |

---

## 3. Real-Time Streaming Performance (Measured on NVIDIA TITAN X & CPU)

| Mode | Hardware | Audio Length | Processing Time | Real-Time Factor (RTF) |
| :--- | :--- | :---: | :---: | :---: |
| **Offline One-Pass** | GPU (TITAN X) | 16.17 s | 490 ms | **0.0303 RTF** (~33x faster than real-time) |
| **Dual-Threaded Stream** | GPU + CPU | Live Mic | 48 ms / chunk | **~0.05 RTF** |
| **Offline One-Pass** | CPU (Intel/AMD) | 16.17 s | 5.01 s | **0.3100 RTF** (~3.2x faster than real-time) |

---

## 4. Qualitative Examples (Empirical Model Output)

### Example 1: French (🇫🇷)
* **Ground Truth Text:** `d'autres traces en un sens différent qui paraissaient toujours avoir rasé la surface du sable`
* **Predicted Text:** `d'autre trace un ancense différents qui paraissait toujours avoir rager la surface du sable`
* **Ground Truth Phonemes:** `dotʁ tʁasz ɑ̃n œ̃ sɑ̃s difeʁɑ̃ ki paʁɛsɛ tuʒuʁz avwaʁ ʁaze la- syʁfas dy- sabl`
* **Predicted Phonemes:** `dotʁ tʁas ɑ̃n asɑ̃s difeʁɑ̃ ki paʁɛsɛ tuʒuʁz avwaʁ ʁaʒe la- syʁfas dy- sabl`

### Example 2: English (🇬🇧)
* **Ground Truth Text:** `chapter five advice from the caterpillar the caterpillar and alice looked at each other for some time in silence`
* **Predicted Text:** `chapter five advice from the captar pillar the catter peter and is can't it other for some time in silence`
* **Ground Truth Phonemes:** `tʃæptəɹ faɪv ədvaɪs fɹʌm ə kætəɹ pɪləɹ ðə kætəɹ pɪləɹ ænd ælɪs lʊkt æt iːtʃ ʌðəɹ fɔːɹ sʌm taɪm ɪn saɪləns`

---

## 5. Checkpoints

* **Model 1 (A2P Acoustic):** `checkpoints/phono_v8_acoustic_multilingual_sota/best_v8_acoustic.pt` (Step 14,000)
* **Model 2 & 3 (P2G Decoders):** `checkpoints/phono_v8_robust_p2g/best_robust_p2g.pt` (Step 25,000)
