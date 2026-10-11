# AudioLearn Architecture Documentation Hub

This directory provides self-contained technical references for every model architecture implemented in the AudioLearn framework. Each document is completely independent and describes the motivation, input representation, layer-by-layer neural pipeline, mathematical formulations, loss functions, source code locations, and inference procedures for that architecture.

---

## Architecture Navigation Index

| Document | Architecture | Core Concept & Innovation | Key Source Files |
| :--- | :--- | :--- | :--- |
| [01. HuBERT Acoustic K-Means SSL](architectures/01_hubert_ssl.md) | **HuBERT** | Self-supervised speech representation learning with 39-dim MFCC cluster pseudo-labels (100 clusters) | `src/models/hubert_pretrain.py`, `src/models/cnn_encoder.py` |
| [02. PhonoHuBERT Direct Phoneme CTC](architectures/02_phono_hubert_direct_ctc.md) | **PhonoHuBERT (V1–V2)** | Direct acoustic-to-phoneme prediction with 64 IPA tokens, articulatory macro-classes, and SpecAugment | `src/models/phono_hubert.py`, `src/models/phono_hubert_hierarchical.py` |
| [03. Scaled Backbones & Prefix Beam Search](architectures/03_phono_scaled_and_beam.md) | **Phono-V3–V5** | Hybrid human/synthetic pretraining (V3), 8-layer deep scaled backbone (V4), and CTC prefix beam search with phonotactic LM (V5) | `src/models/phono_variants.py`, `src/models/transformer.py` |
| [04. MoE Transformer & Latent Diffusion](architectures/04_phono_v6_moe_diffusion.md) | **Phono-V6.1–V6.4** | Top-2 sparse Mixture-of-Experts (V6.1), 30s sliding sparse attention + InterCTC (V6.2), and continuous latent phoneme diffusion (V6.3–V6.4) | `src/models/phono_variants.py`, `src/models/word_denoising_decoder.py` |
| [05. Continuous Speech Double-Loss Model](architectures/05_phono_v6_7_continuous_speech.md) | **Phono-V6.7** | Dual-loss architecture pairing an acoustic CTC phoneme encoder with an autoregressive micro character decoder | `src/models/phono_v6_7_speech_model.py` |
| [06. Two-Level Hierarchical Conformer](architectures/06_phono_v7_0_hierarchical_conformer.md) | **Phono-V7.0** | Conformer backbone + two-level hierarchical character decoder with DP forced alignment and differentiable Soft-Levenshtein loss | `src/models/phono_v7_speech_model.py`, `src/models/forced_aligner.py` |
| [07. Real-Time Causal Streaming Model](architectures/07_phono_v7_1_causal_streaming.md) | **Phono-V7.1** | Strictly causal speech recognition with band-causal macro attention ($K=8$), shift-invariant base queries, and online CTC peak slicing | `src/models/phono_v7_1_speech_model.py`, `src/models/phono_v7_1_alignment.py` |
| [08. Two-Level Partitioned MoE Decoder](architectures/08_phono_v7_2_partitioned_moe.md) | **Phono-V7.2** | Binary non-speech gating (Level 1) + 16 partitioned length-conditioned micro experts (Level 2) with dynamic horizon capping | `src/models/phono_v7_2_speech_model.py` |
| [09. Multi-Scale Dilated Conformer & Boundary Gate](architectures/09_phono_v7_3_dilated_boundary.md) | **Phono-V7.3** | Multi-scale dilated depthwise convolutions ($4\times$ temporal receptive field, 2,420ms) + decoupled 3-class word boundary gate | `src/models/phono_v7_3_speech_model.py`, `src/models/conformer_layers.py` |
| [10. Complete Tripartite Modular Decoupled Model](architectures/10_phono_v7_5_tripartite_modular.md) | **Phono-V7.5** | 3 decoupled modules: Dilated Conformer acoustic front-end, FastText distilled latent bridge, and 7-expert overlapping length MoE decoder | `src/models/phono_v7_5_speech_model.py`, `src/models/text_phoneme_encoder.py` |
| [11. DeepSeek MoE & FactorEngram Memory](architectures/11_phono_v8_0_deepseek_moe_engram.md) | **Phono-V8.0** | Scaled Conformer with DeepSeek-style MoE (1 shared + 8 routed experts) and multi-order polynomial prime FactorEngram memory | `src/models/phono_v8_acoustic.py`, `src/models/phoneme_engram.py` |

---

## Design Principles Across AudioLearn Architectures

1. **Self-Sufficiency**: Every architecture document is written to be understandable without reading earlier versions.
2. **Modularity**: Sub-modules (feature encoders, Conformer layers, MoE routers, decoders) are isolated and reusable across models.
3. **Hardware-Aware Design**: Optimized specifically for real-time throughput and tight VRAM bounds (targeting $< 10.5$ GB on consumer 12 GB GPUs like NVIDIA GTX TITAN X).
4. **Smooth Evolutionary Warm-Starts**: Progressive weight inheritance guarantees that newer architectural iterations warm-start cleanly from earlier checkpoints without catastrophic parameter shocks.
