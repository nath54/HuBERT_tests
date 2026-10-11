# Phono-V7.1: Real-Time Strictly Causal Streaming Speech Model

## 1. Overview & Motivation
**Phono-V7.1** transforms the offline Conformer architecture into a **strictly causal, live streaming speech model**. Offline ASR models attend bidirectionally across the entire sentence, preventing real-time microphone deployment and causing high latency.

Phono-V7.1 enforces zero future lookahead across both the acoustic encoder and the hierarchical decoder:
1. **Strictly Causal Conformer**: Convolutions and self-attention masks never attend to future time frames ($t' > t$).
2. **Band-Causal Macro Attention ($K=8$)**: Words attend strictly to a causal sliding window of the past 8 words, bounding latency and guaranteeing constant $O(K)$ computation per word.
3. **Online Event-Driven CTC Slicing**: Words are detected and sliced in real time from speech energy bursts without needing total audio duration $T$.
4. **Shift-Invariant Base Query ($\mathbf{q}_{\text{base}}$)**: Replaces fixed absolute positional query vectors with a shared, learnable base query that generalizes to unbounded streaming streams.

---

## 2. Streaming Architectural Mechanics

```
Live Audio Buffer (Continuous Stream)
   │
   ▼
┌────────────────────────────────────────────────────────┐
│  Causal 1D CNN + Causal Conformer Backbone             │
│  (Strictly lower-triangular causal attention mask)     │
└──────────────────────────┬─────────────────────────────┘
                           │ Causal Acoustic Representations
                           ▼
┌────────────────────────────────────────────────────────┐
│  Online Event-Driven Slicer (Peak Detection)           │
│  Tracks E(t) = 1 - P(blank). Detects word boundaries   │
│  when silence exceeds 2 frames. Emits word slice.      │
└──────────────────────────┬─────────────────────────────┘
                           │ Acoustic Word Slice
                           ▼
┌────────────────────────────────────────────────────────┐
│  Band-Causal Macro Word Decoder (K=8 Past Words)       │
│  Query: Shift-Invariant q_base                         │
│  Predicts word length k_hat and latent vector z_word   │
└──────────────────────────┬─────────────────────────────┘
                           │ Continuous Length Guidance Bias
                           ▼
┌────────────────────────────────────────────────────────┐
│  Causal Micro Character Decoder                        │
│  Autoregressively rolls out characters for current word│
└────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Innovations

### Band-Causal Macro Attention
For acoustic word $i$, attention weights over prior words $j$ are restricted to:

$$\text{AttentionMask}_{i, j} = \begin{cases} 0, & \text{if } 0 \le i - j < K \\ -\infty, & \text{otherwise} \end{cases}$$
where $K=8$. This bounds memory consumption to $O(K \cdot L)$ rather than $O(L^2)$ and eliminates error propagation from long-past context.

### Continuous Length Guidance (`length_to_micro_bias`)
The predicted continuous length $\hat{k}_i$ directly conditions micro character decoding:
$$b_k = \operatorname{Linear}(\hat{k}_i) \in \mathbb{R}^{D}$$
This bias is added to the character cross-attention queries, informing the decoder of the remaining character budget and preventing trailing hallucinations.

### Length Coverage Metric
The model maintains a safety margin ensuring characters are never prematurely truncated:
$$\text{Cover} = \mathbb{P}(\lceil \hat{k} \rceil + 1 \ge k_{\text{true}}) > 97\%$$

---

## 4. Source Code Pointers

* Streaming Model: [`src/models/phono_v7_1_speech_model.py`](file:///home/nathan/github/audiolearn/src/models/phono_v7_1_speech_model.py) (`PhonoV71StreamingSpeechModel`)
* Online Event Slicer: [`src/models/phono_v7_1_alignment.py`](file:///home/nathan/github/audiolearn/src/models/phono_v7_1_alignment.py) (`OnlineEventSlicer`)
* Training Script: `scripts/train_phono_v7_1_streaming.py`
