# Phono-V3–V5: Scaled Backbones, Hybrid Training & Phonotactic Prefix Beam Search

## 1. Overview & Evolution
Building upon the PhonoHuBERT direct phoneme framework, **Phono-V3**, **Phono-V4**, and **Phono-V5** address capacity scaling, training data diversity, and inference decoding:

* **Phono-V3 (Hybrid Pre-Training)**: Introduces interleaved real human speech (LibriSpeech 100h) and zero-disk in-RAM procedural neural TTS speech across 39 distinct voices to build invariant phonetic representations.
* **Phono-V4 (Deep Scaled Capacity)**: Expands encoder capacity to 8 layers, 8 attention heads, 512 embedding dimension, and 2,048 FFN dimension (31.8M parameters).
* **Phono-V5 (Phonotactic Prefix Beam Search)**: Integrates CTC prefix beam search with a 3-gram phonotactic language model prior to resolve acoustic ambiguities and illegal phoneme sequences during decoding.

---

## 2. Model Specifications

| Parameter | Phono-V1/V2 | Phono-V3 | Phono-V4 | Phono-V5 |
| :--- | :--- | :--- | :--- | :--- |
| **Encoder Layers** | 4 | 4 | **8** | **8** |
| **Embedding Dimension ($D$)**| 384 | 384 | **512** | **512** |
| **Attention Heads ($H$)** | 6 | 6 | **8** | **8** |
| **FFN Dimension** | 1,536 | 1,536 | **2,048** | **2,048** |
| **Data Pipeline** | 100% Synthetic | **Hybrid Human+Synthetic** | Hybrid | Hybrid |
| **Inference Decoder** | Greedy Argmax | Greedy Argmax | Greedy Argmax | **Prefix Beam Search + 3-Gram LM** |

---

## 3. Phono-V5: CTC Prefix Beam Search Decoding

Pure greedy CTC decoding ($\hat{\pi}_t = \arg\max P(\pi_t \mid x_t)$) suffers because CTC frames are conditionally independent. When an acoustic frame is ambiguous between two phonemes (e.g. `/d/` vs `/t/`), greedy decoding cannot incorporate transition history.

### The Prefix Beam Search Algorithm
Phono-V5 maintains a set of candidate prefix sequences $\ell \in \text{Beam}$, tracking probabilities ending in blank ($P_b(\ell, t)$) and non-blank ($P_{nb}(\ell, t)$):

1. **Extension by Blank**:
   $$P_b(\ell, t) = (P_b(\ell, t-1) + P_{nb}(\ell, t-1)) \cdot P(\text{blank} \mid x_t)$$
2. **Extension by Same Phoneme ($\ell$ ends with $c$)**:
   $$P_{nb}(\ell, t) = P_{nb}(\ell, t-1) \cdot P(c \mid x_t)$$
   $$P_{nb}(\ell \circ c, t) = P_b(\ell, t-1) \cdot P(c \mid x_t)$$
3. **Extension by Different Phoneme ($c \ne \text{last}(\ell)$)**:
   $$P_{nb}(\ell \circ c, t) = (P_b(\ell, t-1) + P_{nb}(\ell, t-1)) \cdot P(c \mid x_t)$$

### Phonotactic Language Model Integration
When extending prefix $\ell$ with character $c$, Phono-V5 injects a 3-gram phonotactic prior:

$$S(\ell \circ c) = \log(P_{\text{CTC}}(\ell \circ c)) + \beta \log P_{\text{LM}}(c \mid \ell_{[-2:]}) + \alpha |\ell \circ c|$$
* $\beta$: LM weight (default $0.20$), penalizing phonotactically illegal combinations (such as invalid consonant triples).
* $\alpha$: Length reward bonus (default $0.05$), compensating for CTC's preference for shorter sequences.

---

## 4. Source Code Pointers

* Model Definitions: [`src/models/phono_variants.py`](file:///home/nathan/github/audiolearn/src/models/phono_variants.py)
  * `PhonoV3HybridConfig`, `PhonoV3HybridForPreTraining`
  * `PhonoV4ScaledConfig`, `PhonoV4ScaledForPreTraining`
  * `PhonoV5BeamConfig`, `PhonoV5BeamForPreTraining`
* LM Prior Data: `data/phonotactic_3gram.json`
