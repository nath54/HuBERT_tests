# Phono-V6.1–V6.4: MoE Transformer, Sliding Sparse Attention & Latent Phoneme Diffusion

## 1. Overview & Motivation
The **Phono-V6 series** introduced architectural breakthroughs that drove Phoneme Error Rates (PER) below 15% on LibriSpeech clean-100:
1. **Phono-V6.1 (MoE Transformer)**: Expands parameter capacity without proportional compute cost via Sparse Mixture-of-Experts.
2. **Phono-V6.2 (Sparse Windowed Attention + InterCTC)**: Scales acoustic context up to 30.0 seconds (1,500 frames) using banded sliding-window attention and intermediate layer supervision.
3. **Phono-V6.3 (Continuous Latent Diffusion)**: Treats phoneme generation as continuous denoising diffusion in latent space, breaking CTC conditional independence.
4. **Phono-V6.4 (Confidence-Gated Diffusion & Word Denoising Decoder)**: Dynamically gates diffusion so confident acoustic frames bypass denoising, and adds a decoupled word-level decoder.

---

## 2. Phono-V6.1: Sparse Mixture-of-Experts (MoE)

In standard Transformers, the Feed-Forward Network (FFN) consumes ~66% of parameters. Phono-V6.1 replaces the monolithic FFN with $E=4$ specialized experts:

$$y_t = \sum_{e \in \text{Top-2}} G(h_t)_e \cdot \text{Expert}_e(h_t)$$

### Router & Load-Balancing
The gating network computes router probabilities:
$$P(e \mid h_t) = \operatorname{softmax}(W_g h_t + b_g)_e$$

To prevent all tokens from routing to a single expert (expert collapse), an auxiliary load-balancing loss is enforced:
$$\mathcal{L}_{\text{balance}} = E \sum_{e=1}^E f_e \cdot P_e$$
where $f_e$ is the fraction of tokens dispatched to expert $e$, and $P_e$ is the average gating probability assigned to expert $e$.

---

## 3. Phono-V6.2: 30-Second Sliding Attention & InterCTC

Long utterances (> 15 seconds) cause $O(T^2)$ self-attention compute to explode. Phono-V6.2 solves this with:

### Banded Sliding-Window Attention ($\pm 320\text{ms}$)
Each frame $t$ attends only to neighboring frames within window $W = 16$ (representing $\pm 320\text{ms}$):
$$\text{Mask}_{i, j} = \begin{cases} 0, & \text{if } |i - j| \le W \\ -\infty, & \text{otherwise} \end{cases}$$
This reduces memory and compute complexity from $O(T^2)$ to $O(T \cdot W)$, allowing effortless scaling to 30-second inputs.

### Intermediate CTC (InterCTC)
To prevent gradient vanishing across deep layers, intermediate CTC losses are applied at Layer 4 and Layer 8:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CTC}}^{(8)} + 0.3 \mathcal{L}_{\text{CTC}}^{(4)} + \mathcal{L}_{\text{balance}}$$

---

## 4. Phono-V6.3 & V6.4: Latent Phoneme Continuous Diffusion

Standard CTC assumes frame independence ($P(y \mid x) = \prod_t P(y_t \mid x_t)$). Phono-V6.3 and V6.4 break this assumption by formulating phoneme refinement as a continuous **denoising diffusion process**:

```
Conformer Output h_t ──► Initial CTC Logits ──► Continuous Phoneme Latents z_0
                                                         │
                                                         ▼
                                                Add Noise (Timestep s): z_s
                                                         │
                                                         ▼
                                       ┌──────────────────────────────────┐
                                       │ Conv1D U-Net / FiLM Refiner      │
                                       │ Conditioned on h_t and step s    │
                                       └─────────────────┬────────────────┘
                                                         │
                                                         ▼
                                                Predicted Noise eps_theta
                                                         │
                                                         ▼
                                                Denoised Phoneme Latents z_hat_0
```

### Confidence-Gated Diffusion (V6.4)
Running diffusion across all frames is computationally wasteful. Phono-V6.4 adds an adaptive **confidence margin gate**:
$$M_t = \operatorname{Top1}(P_t) - \operatorname{Top2}(P_t)$$
$$g_t = \sigma(W_m [h_t; M_t; \operatorname{Top1}(P_t)])$$
* Frames with high margin ($M_t > 0.85$) bypass the diffusion refiner completely ($g_t \approx 0$).
* Ambiguous acoustic frames ($M_t \le 0.85$) receive iterative convolutional diffusion denoising ($g_t \approx 1$).

---

## 5. Source Code Pointers

* MoE & Diffusion Models: [`src/models/phono_variants.py`](file:///home/nathan/github/audiolearn/src/models/phono_variants.py)
  * `PhonoV61MoEForPreTraining`
  * `PhonoV62SparseForPreTraining`
  * `PhonoV63DiffusionForPreTraining`
  * `PhonoV64GatedDiffusionForPreTraining`
* Word Denoising Decoder: [`src/models/word_denoising_decoder.py`](file:///home/nathan/github/audiolearn/src/models/word_denoising_decoder.py)
