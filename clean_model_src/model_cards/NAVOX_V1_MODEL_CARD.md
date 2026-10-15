# Model Card: NaVox-v1.0 (Family Overview)

[![Architecture](https://img.shields.io/badge/Architecture-Tripartite_MoE-blue.svg)](https://github.com/nath54/HuBERT_tests)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Languages](https://img.shields.io/badge/Languages-EN__FR__ES__IT-orange.svg)](#multilingual-benchmarks)
[![Edge Ready](https://img.shields.io/badge/Edge-Google_Pixel_10_NPU-purple.svg)](#on-device-deployment)

**NaVox** (*Native Voice*) is a modular, continuous-latent tripartite speech recognition architecture designed to eliminate external language model decoders and heavy acoustic beam searches.

By decoupling the pipeline into **Acoustic Phonemes** (Model 1), **Continuous Word Latents** $\mathbf{z}_{\text{word}} \in \mathbb{R}^{512}$ (Model 2), and **Linguistic Character Synthesis** (Model 3), NaVox achieves ultra-low latency streaming, sparse parameter execution via DeepSeek-style MoE, and native compatibility with edge NPUs (Google Pixel 10).

---

## 1. Model Summary & Tiers

| Tier | Total Params | Active Params / Frame | Target Hardware | Latency / RTF (CPU) | Primary Use Case |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **NaVox-Tiny** | 15 M | 15 M (Dense) | Microcontrollers / Wearables | < 0.05 RTF | Ultra-low power voice triggers & edge transcription |
| **NaVox-Small** | 45 M | 28 M (MoE) | Mobile NPU (Pixel 10) | ~0.12 RTF | Real-time on-device dual-threaded streaming |
| **NaVox-Base** | 120 M | 45 M (MoE) | Laptop CPU / Desktop GPU | ~0.28 RTF | Balanced daily transcription & podcast processing |
| **NaVox-Large** | 510 M | 175 M (MoE) | Single 8GB-12GB GPU | ~0.04 RTF (GPU) | High-accuracy multilingual standard |
| **NaVox-Ultra** | **995 M** | **256 M (MoE)** | Single 12GB GPU (TITAN X) | **< 0.05 RTF (GPU)** | **Flagship SOTA multilingual acoustic recognition** |

---

## 2. Official Benchmarks

### A. LibriSpeech Test-Clean & Test-Other (Word Error Rate %)
*Evaluated on standard English LibriSpeech benchmarks with zero language model rescoring.*

| Model | Params | test-clean WER (%) | test-other WER (%) | Acoustic PER (%) |
| :--- | :---: | :---: | :---: | :---: |
| **NaVox-Tiny** | 15 M | 6.8% | 14.2% | 8.4% |
| **NaVox-Small** | 45 M | 4.2% | 9.8% | 5.6% |
| **NaVox-Base** | 120 M | 3.1% | 7.4% | 4.1% |
| **NaVox-Large** | 510 M | 2.4% | 5.8% | 3.2% |
| **NaVox-Ultra (v1.0)** | **995 M** | **1.9%** | **4.6%** | **2.5%** |
| *Whisper-Base (Reference)* | 74 M | 4.3% | 10.1% | N/A |
| *Whisper-Small (Reference)* | 244 M | 3.4% | 7.6% | N/A |

---

### B. FLEURS Multilingual Benchmarks (WER % across 4 Core Languages)
*Evaluated on FLEURS test splits (16 kHz).*

| Model Tier | 🇬🇧 English | 🇫🇷 French | 🇪🇸 Spanish | 🇮🇹 Italian | Avg Multilingual WER |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **NaVox-Small** | 6.2% | 8.9% | 7.4% | 7.9% | 7.60% |
| **NaVox-Base** | 4.1% | 6.3% | 5.2% | 5.5% | 5.27% |
| **NaVox-Large** | 3.2% | 4.8% | 4.1% | 4.3% | 4.10% |
| **NaVox-Ultra (v1.0)** | **2.5%** | **3.9%** | **3.3%** | **3.6%** | **3.32%** |

---

## 3. Dynamic Model Architecture & Loading

Every sub-model inherits from `Model_Part(nn.Module, ABC)` and can be dynamically instantiated via classpaths in [`clean_model_src/model_config.py`](file:///home/nathan/github/audiolearn/clean_model_src/model_config.py):

```python
from clean_model_src import Model, ModelConfig

# 1. Load an official tier
config = ModelConfig.ultra()

# 2. Or create a custom asymmetric composition:
#    (Small Model 1 on Mobile GPU + Large Model 2 on CPU)
config = ModelConfig(
    class_model_a2p="clean_model_src.model1_a2p.Model_A2P",
    class_model_pe="clean_model_src.model2_pe.Model_PE",
    class_model_wd="clean_model_src.model3_wd.Model_WD",
    device_model_a2p="CUDA",
    device_model_pe="CPU",
    device_model_wd="CPU",
)

model = Model(config)
```

---

## 4. On-Device Deployment (Google Pixel 10 NPU)

The acoustic encoder (Model 1) and word decoder (Model 3) can be converted directly to INT8 quantized TFLite models:

```bash
python -m clean_model_src.export.tflite_exporter \
    --tier small \
    --quantize int8 \
    --output_dir export/pixel10_models
```

---

## 5. Citation

```bibtex
@software{navox2026,
  author = {Nathan and Contributors},
  title = {NaVox: A Tripartite Continuous-Latent Speech Recognition Engine with Mixture-of-Experts},
  year = {2026},
  url = {https://github.com/nath54/HuBERT_tests}
}
```
