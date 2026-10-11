# Phono-V7.3: Multi-Scale Dilated Conformer & Decoupled Boundary Gate Head

## 1. Overview & Motivation
**Phono-V7.3** resolves two fundamental acoustic bottlenecks identified in real-time streaming recognition:
1. **The Receptive Field Ceiling**: Standard Conformer depthwise convolutions with kernel size $k=31$ and stride $320\times$ span only 620ms of temporal context. While sufficient for isolated phones, 620ms cannot capture multi-word prosody, co-articulation across word boundaries, or French liaisons.
2. **Word Boundary Fracturing & Concatenation**: Standard CTC tends to place `<space>` tokens arbitrarily within vowel elongations or delete spaces between rapid words, causing downstream decoders to hallucinate or fuse distinct words together.

Phono-V7.3 introduces **Multi-Scale Dilated Conformer Convolutions** and the **Decoupled Word Boundary Gate Head**.

---

## 2. Multi-Scale Dilated Conformer Convolutions

The convolution module in each Conformer block is upgraded into 3 parallel temporal dilation paths:

```
Pre-LN Acoustic Hidden Representation h [Batch, 512, Time]
   │
   ├──────────────────────────────┬──────────────────────────────┐
   ▼                              ▼                              ▼
Branch 1: Base (d=1)           Branch 2: Dilated (d=2)        Branch 3: Dilated (d=4)
Kernel: 31, Dilation: 1        Kernel: 31, Dilation: 2        Kernel: 31, Dilation: 4
Receptive Field: 620ms         Receptive Field: 1,220ms       Receptive Field: 2,420ms
   │                              │                              │
   │                              └──────────────┬───────────────┘
   │                                             ▼
   │                                    Concatenate Branches 2 & 3
   │                                             ▼
   │                                    Dilated Projection (1x1 Conv)
   │                                             ▼
   └──────────────────────────────(+)────────────┘
                                   │
                                   ▼
                           BatchNorm + SiLU + Pointwise Conv2
```

### Zero-Initialization for Warm-Start Preservation
To allow bit-for-bit weight transfer from V7.2 checkpoints, the dilated linear projection `dilated_proj` is initialized with zeros:
$$\operatorname{weight}(\text{dilated\_proj}) = \mathbf{0}, \quad \operatorname{bias}(\text{dilated\_proj}) = \mathbf{0}$$
At Step 0, Branch 2 and Branch 3 output 0, perfectly preserving existing trained acoustic features while allowing the model to smoothly learn long-range temporal dependencies.

---

## 3. Decoupled Word Boundary Gate Head

Rather than relying on the general phoneme CTC head to implicitly identify spaces, Phono-V7.3 introduces a dedicated **3-class boundary discriminator**:
* Class 0: `SPEECH` (Active speech phoneme interior)
* Class 1: `WORD_SPACE` (Boundary separating two distinct words)
* Class 2: `SILENCE_BLANK` (Inter-sentence pause or non-verbal breath)

### Asymmetric Boundary Loss
To eradicate space deletions and false insertions, the boundary loss enforces an explicit $\times 4.0$ penalty on inter-word boundary transitions:

$$\mathcal{L}_{\text{boundary}} = - \frac{1}{T} \sum_{t=1}^T w_{y_t} \log P(y_t \mid h_t), \quad w = [1.0,\; 4.0,\; 1.0]$$

---

## 4. Hierarchical Recursive 2-Pass Phoneme Head

To prevent consecutive spaces or illegal consonant closures, the phoneme head refines base logits using causal recurrent feedback:
1. **Pass 1 (Base Emission)**: Emits initial logits $z_t^{(0)} = W_{\text{base}} h_t$.
2. **Pass 2 (Causal Depthwise Refiner)**: Evaluates the concatenation of acoustic representations and previous phoneme distributions:
   $$r_t = \text{CausalConv1D}\left( [h_t;\; \operatorname{softmax}(z_{t-1}^{(0)})] \right)$$
   $$z_t = z_t^{(0)} + W_{\text{refine}} r_t$$

---

## 5. Source Code Pointers

* Backbone & Heads: [`src/models/phono_v7_3_speech_model.py`](file:///home/nathan/github/audiolearn/src/models/phono_v7_3_speech_model.py) (`PhonoV73AcousticBackbone`, `DecoupledBoundaryGateHead`, `HierarchicalRecursivePhonemeHead`)
* Multi-Scale Conv Module: [`src/models/conformer_layers.py`](file:///home/nathan/github/audiolearn/src/models/conformer_layers.py) (`MultiScaleDilatedConformerConvModule`)
* Training Script: `scripts/train_phono_v7_3_streaming.py`
