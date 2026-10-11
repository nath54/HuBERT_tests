# Phono-V8.0: Acoustic Conformer with DeepSeek MoE & FactorEngram Memory

## 1. Overview & Motivation
**Phono-V8.0** is the flagship acoustic model in AudioLearn engineered to break through the $< 5\%$ Phoneme Error Rate (PER) milestone across four major European languages (English, French, Spanish, Italian) without language conditioning vectors or language ID prompts.

Prior acoustic architectures hit two ceilings:
1. **The Ground Truth Noise Floor (~8–10% PER)**: Standard audiobook corpora contain reader ad-libs, omitted sentences, and G2P mismatches that penalize the acoustic model even when speech is faithful.
2. **The CTC Conditional Independence Ceiling**: Pure CTC cannot model phoneme transitions or phonotactics without an external language model.

Phono-V8.0 addresses these challenges through:
1. **Autonomous Corpus Sanitization**: Discards noisy reader departures to train and validate exclusively on gold-standard verbatim speech.
2. **DeepSeek-Style MoE (1 Shared + 8 Routed Experts)**: Preserves continuous shared acoustic representations while enabling fine phonetic manner specialization.
3. **DeepSeek V4.1 / Qwen-Style FactorEngram Memory**: An $O(1)$ constant-time associative parameter memory that stores static phonetic transition priors, breaking CTC conditional independence.

---

## 2. Architectural Blueprint

```
Audio Input (16 kHz, Mono)
       │
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│  7-Layer 1D CNN Feature Extractor (Stride 320x, 20ms frames)           │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Acoustic Embeddings [Batch, Time, 512]
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│  8-Layer Conformer Backbone with DeepSeek MoE (D=512, H=8, FFN=2048)   │
│  • 1 Shared Expert (always active) + 8 Fine Routed Experts (Top-2)     │
│  • Multi-Scale Dilated Convolutions (k=31, dilations d=1, 2, 4)        │
│                                                                        │
│  Layer 3: Intermediate Draft Phoneme Head                              │
│           Emits draft phoneme tokens p_t in [0, 111]                   │
│                                                                        │
│  Layers 4 & 7: FactorEngram Conditional Memory Injections              │
│           • Queries 2-gram, 3-gram, 4-gram tables via 4 prime hashes   │
│           • FactorEngram context-aware channel gate: g_t in [0, 1]^512 │
│           • Injects static phonotactic memory residually: h + g * m     │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ Hidden States h_t [Batch, Time, 512]
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
┌──────────────────────────────────┐  ┌──────────────────────────────────┐
│ Decoupled Boundary Gate Head     │  │ Hierarchical Recursive Phoneme   │
│ 3 Classes: Speech / Space / Blank│  │ Head: Macro Guidance + Causal    │
│ Penalty: x4.0 on space errors    │  │ 2-Pass Phonotactic Refinement    │
└──────────────────────────────────┘  └──────────────────────────────────┘
```

---

## 3. DeepSeek-Style Mixture of Experts

Standard MoE models distribute all capacity into routed experts. However, speech features have a universal component (formants, energy, basic vowels) that every token shares.

Phono-V8.0 adopts the DeepSeek MoE configuration:
$$\text{MoE}(x) = \text{SharedExpert}(x) + \sum_{e \in \text{Top-2}} G(x)_e \cdot \text{RoutedExpert}_e(x)$$

* **1 Shared Expert**: Dedicated to language-agnostic acoustic dynamics.
* **8 Routed Experts**: Fine-grained experts specializing in articulatory manners (nasals, plosives, sibilants, liquids).
* **Top-2 Routing**: Only 2 routed experts fire per token.
* **Capacity & Compute**:
  * Total Model Parameters: **203.69M parameters**
  * Active Parameters per Frame: **~103.02M parameters**

---

## 4. Multi-Order FactorEngram Memory

### Intuition
Rather than forcing Conformer attention layers to rediscover that certain phoneme sequences (like `/s-t-ɹ/`) are permissible while illegal clusters are forbidden, Phono-V8.0 offloads static phonotactic transitions into an $O(1)$ parameter table.

### Multi-Head Polynomial Prime Hashing
At Layer 3, draft phoneme tokens $\hat{p}_t$ are emitted. Context windows of 2-grams, 3-grams, and 4-grams are hashed into embedding tables using $K=4$ independent prime hash heads:

$$\text{Hash}_k(p_1, \dots, p_N) = \left( \sum_{i=1}^N p_i \cdot P_{k, i} \right) \pmod{M_N}$$

* 2-Gram table: $M_2 = 16,384$ entries
* 3-Gram table: $M_3 = 32,768$ entries
* 4-Gram table: $M_4 = 32,768$ entries

### Retrieved Representation & FactorEngram Gating
Each hash lookup retrieves a dense vector of dimension $\text{memory\_dim} = 128$:
$$\text{raw\_memory}_t \in \mathbb{R}^{3 \times 4 \times 128} = \mathbb{R}^{1536}$$

A two-layer projection projects this to the Conformer embedding dimension $D=512$:
$$m_t = \text{MemoryProjection}(\text{raw\_memory}_t) \in \mathbb{R}^{512}$$

A channel-wise gate evaluates acoustic context $h_t$ against memory $m_t$:
$$g_t = \sigma(\text{GateNet}([h_t;\; m_t])) \in [0, 1]^{512}$$
$$h_t^{\text{out}} = h_t + \text{Dropout}(g_t \odot m_t)$$

---

## 5. Zero-Initialization Warm-Start Guarantee

To prevent catastrophic performance collapse when warm-starting from V7.5 checkpoints, Phono-V8.0 strictly zero-initializes the auxiliary paths:
1. `shared_expert[-2].weight = 0`, `shared_expert[-2].bias = 0` (shared expert starts at 0 contribution).
2. `router.bias[4:] = -20.0` (new routed experts are initially suppressed, routing 100% to the 4 trained V7.5 experts).
3. `memory_projection[-1].weight = 0`, `memory_projection[-1].bias = 0` (engram memory starts at 0 contribution).

This mathematically ensures that at Step 0:
$$\max |\text{Logits}_{\text{V8.0}} - \text{Logits}_{\text{V7.5}}| = 7.63 \times 10^{-6}$$
The model begins with identical baseline validation accuracy (12.97% calibrated PER) and learns smoothly without initial shock.

---

## 6. Source Code Pointers

* Backbone Model: [`src/models/phono_v8_acoustic.py`](file:///home/nathan/github/audiolearn/src/models/phono_v8_acoustic.py) (`PhonoV8AcousticBackbone`)
* FactorEngram Module: [`src/models/phoneme_engram.py`](file:///home/nathan/github/audiolearn/src/models/phoneme_engram.py) (`PhonemeEngramMemory`)
* Conformer Layers & MoE: [`src/models/conformer_layers.py`](file:///home/nathan/github/audiolearn/src/models/conformer_layers.py) (`DeepSeekMoEFeedForwardNetwork`, `ConformerEngramMoEEncoder`)
* Corpus Sanitizer: [`scripts/clean_and_filter_corpus.py`](file:///home/nathan/github/audiolearn/scripts/clean_and_filter_corpus.py)
* Training Script: [`scripts/train_phono_v8_acoustic.py`](file:///home/nathan/github/audiolearn/scripts/train_phono_v8_acoustic.py)
