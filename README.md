# AudioLearn: HuBERT ASR & Explainable AI (XAI) Pipeline

A complete PyTorch framework implementing the **HuBERT** (Hidden-Unit BERT) architecture from scratch, trained on speech for **Automatic Speech Recognition (ASR)** via CTC Loss, equipped with a comprehensive **Explainable AI (XAI)** suite using **PyTorch** and **Captum** (Gradients, NAPS, and Trained Diagnostic Decoders).

---

## Architecture Overview

```
Raw Audio Waveform (16 kHz, Mono)
              │
              ▼
   ┌─────────────────────────────────────────────────────────┐
   │  HuBERTFeatureEncoder (Temporal 1D Convolution)         │
   │  7-layer 1D CNN with LayerNorm & GELU (Strides 5,2,2...)│
   │  Downsampling Factor: 320x (20ms frames / 50Hz)         │
   └──────────────────────────┬──────────────────────────────┘
                              │ Frame features (B, T, C)
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │  Feature Projection & LayerNorm                         │
   │  Linear(C, embed_dim) + Dropout                         │
   └──────────────────────────┬──────────────────────────────┘
                              │
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │  HuBERTEncoder (Transformer Stack)                      │
   │  - Depthwise Convolutional Positional Embeddings        │
   │  - Multi-Head Self-Attention (MHSA)                     │
   │  - Feed-Forward Networks (FFN: embed_dim -> ffn_dim)    │
   │  - Pre-LayerNorm residual blocks                        │
   └──────────────────────────┬──────────────────────────────┘
                              │ Hidden states (L layers)
                              ▼
   ┌─────────────────────────────────────────────────────────┐
   │  CTC ASR Head & Decoding                                │
   │  Linear(embed_dim, vocab_size) -> CTC Loss / Greedy Arg │
   └─────────────────────────────────────────────────────────┘
```

---

## Explainable AI (XAI) Suite

This framework implements three complementary interpretability paradigms:

### 1. Gradient-Based Attribution (via Captum)
- **Integrated Gradients (`captum.attr.IntegratedGradients`)**:
  Computes the path integral of gradients along the straight line from a baseline (silence) to the input audio waveform, attributing scalar token predictions or sequence likelihood to specific millisecond audio segments.
- **Saliency (`captum.attr.Saliency`)**:
  First-order gradient magnitude $\left|\frac{\partial y}{\partial x}\right|$ pinpointing high-sensitivity waveform samples.
- **Layer Integrated Gradients (`captum.attr.LayerIntegratedGradients`)**:
  Attributes model outputs to intermediate Transformer layer representations.

### 2. NAPS (Neuron Activation Profiles, Saliency & Patching)
- **Neuron Activation Profiles (NAP)**:
  Extracts intermediate activations of all FFN neurons ($H \to 4H$ expansion) across different phonetic and acoustic conditions (speech vs. silence, vowels vs. consonants).
- **Neuron Selectivity Index**:
  Computes a selectivity metric $SI_n = \frac{\mu_{target} - \mu_{base}}{|\mu_{target}| + |\mu_{base}| + \epsilon}$ to locate specialized neurons (e.g. vowel detectors, silence detectors).
- **Causal Activation Patching / Ablation**:
  Intervenes on individual layers (zero-ablation or skip-connection bypass) to trace the causal impact of each layer on output logits.
- **Neuron Saliency**:
  Computes activation-gradient products ($h_n \odot \frac{\partial \mathcal{L}}{\partial h_n}$) to rank the most critical individual neurons for a given utterance.

### 3. Trained Decoders & Diagnostic Probing
- **Diagnostic Probes (Linear Probes)**:
  Freezes the HuBERT backbone and trains linear decoders on top of each layer $l \in [0, \dots, L]$ to predict acoustic energy and phonetic categories. Generates layer-by-layer probing curves demonstrating feature abstraction.
- **Acoustic Inversion Decoders**:
  Trains light decoders to reconstruct input spectral/filterbank features from frozen layer representations, proving that early layers preserve raw physical acoustic properties while deeper layers discard surface acoustics in favor of symbolic linguistic tokens.

---

## Directory Structure

```
audiolearn/
├── .venv/                      # Python 3.12 Virtualenv (CUDA 12 + PyTorch)
├── configs/                    # YAML configuration files
│   ├── hubert_base.yaml        # Model architecture hyperparameters
│   ├── train_asr.yaml          # Training hyperparameters
│   └── xai_config.yaml         # XAI parameters
├── data/                       # Datasets & manifests
│   ├── raw/synthetic/          # Generated synthetic speech audio & manifests
│   └── sample_dataset.py       # Audio dataset synthesizer & loader
├── src/                        # Core codebase
│   ├── models/                 # HuBERT architecture from scratch
│   │   ├── config.py           # HuBERTConfig dataclass
│   │   ├── cnn_encoder.py      # 7-layer Temporal 1D CNN Feature Extractor
│   │   ├── transformer.py      # Pos-conv embeddings + MHSA + FFN
│   │   └── hubert_asr.py       # Full HuBERTForCTC model + greedy CTC decoder
│   ├── data/                   # Tokenization and collation
│   │   ├── tokenizer.py        # Character CTC Tokenizer
│   │   ├── dataset.py          # AudioASRDataset + dynamic padding collate fn
│   │   └── augmentations.py    # Waveform noise, gain, and time-masking
│   ├── training/               # Training pipeline
│   │   ├── trainer.py          # HuBERTASTTrainer (AMP, clipping, checkpointing)
│   │   └── metrics.py          # CER, WER, and MetricTracker
│   ├── xai/                    # Explainable AI suite
│   │   ├── captum_gradients.py # Captum Integrated Gradients & Saliency
│   │   ├── naps.py             # Neuron Activation Profiles, Patching & Saliency
│   │   ├── probes.py           # Layer-wise Diagnostic Probes & Inversion Decoders
│   │   └── visualizer.py       # Matplotlib visualization suite
│   └── utils/                  # Audio I/O & logging
├── checkpoints/                # Model checkpoints (best_model.pt, latest_model.pt)
├── logs/                       # TensorBoard events & training history JSON
├── outputs/                    # Generated XAI plots
│   ├── gradients/              # Integrated Gradients & Saliency plots
│   ├── naps/                   # Neuron Selectivity Heatmaps & Ablation
│   └── probes/                 # Layer Probing Curves & Inversion Loss
├── scripts/                    # Command-line entry points
│   ├── download_data.py        # Dataset generation / preparation
│   ├── train.py                # Train HuBERT ASR
│   ├── evaluate.py             # Evaluate checkpoint on test set
│   ├── run_xai.py              # Run complete XAI suite
│   └── demo_pipeline.py        # Automated end-to-end master pipeline
├── tests/                      # Unit test suite
│   ├── test_model.py
│   ├── test_data.py
│   └── test_xai.py
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Virtual Environment & Dependencies

The project uses **Python 3.12** and **PyTorch with CUDA 12**:

```bash
# Create venv with Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Captum and dependencies
pip install captum numpy scipy soundfile matplotlib pandas tqdm pyyaml editdistance tensorboard
```

### 2. Run All-in-One Automated Demo Pipeline

Runs dataset generation, HuBERT training, evaluation, and all XAI methods in one command:

```bash
.venv/bin/python scripts/demo_pipeline.py
```

### 3. Step-by-Step CLI Execution

#### Step A: Generate / Prepare Data
```bash
.venv/bin/python scripts/download_data.py --output_dir data/raw/synthetic --num_train 120 --num_val 30
```

#### Step B: Train HuBERT ASR
```bash
.venv/bin/python scripts/train.py --epochs 10 --batch_size 16 --lr 0.0005
```

#### Step C: Evaluate Checkpoint
```bash
.venv/bin/python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

#### Step D: Run Explainable AI Suite
```bash
.venv/bin/python scripts/run_xai.py --checkpoint checkpoints/best_model.pt --output_dir outputs
```

### 4. Running Unit Tests

```bash
.venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
```

---

## Generated XAI Output Visualizations

- `outputs/gradients/integrated_gradients_waveform.png`: Raw audio waveform with time-aligned Integrated Gradients score overlay.
- `outputs/gradients/integrated_gradients_spectrogram.png`: Spectrogram aligned with temporal gradient attributions.
- `outputs/gradients/saliency_waveform.png`: First-order gradient saliency map.
- `outputs/naps/neuron_selectivity_heatmap.png`: Heatmap of the top selective neurons across all Transformer layers.
- `outputs/probes/diagnostic_probing_curve.png`: Classification accuracy curve across layers from CNN out to final Transformer layer.

---

## Interactive Visualizations & Beginner Guide

A full suite of dynamic visualizations is provided to explore what happens across every layer:

- **Launch Interactive Browser Dashboard**:
  Open [`outputs/visualizations/interactive_dashboard.html`](file:///home/nathan/github/audiolearn/outputs/visualizations/interactive_dashboard.html) in any modern web browser or IDE preview.
  - Interactive scrubbers through raw waveforms and frequency spectrograms.
  - Step-by-step layer representation inspector (CNN $\to$ L1 $\to$ L2 $\to$ L3 $\to$ L4).
  - 320x CNN downsampling and receptive field calculator.
  - HuBERT discrete audio unit cluster simulator (k-means quantization).
  - Interactive CTC collapse animator (frame argmax $\to$ duplicate collapse $\to$ blank removal $\to$ final transcript).

- **Regenerate Visualization Artifacts**:
  ```bash
  .venv/bin/python scripts/visualize_interactive.py --checkpoint checkpoints/best_model.pt
  ```
  Generates:
  - `outputs/visualizations/hubert_pipeline_flow.png`: 4-panel complete journey from raw waveform to CTC posteriorgram.
  - `outputs/visualizations/receptive_field_flow.png`: Log-scale receptive field growth across 7 CNN downsampling strides.
  - `outputs/visualizations/layer_inspection_data.json`: Full numeric layer dumps for custom downstream analysis.

---

## Live Interactive Web Studio (Connected to PyTorch & Captum)

A **live web application** backed by a real **FastAPI + PyTorch** backend server is running on **http://localhost:8000**:

- **Real-Time PyTorch Inference**: Type any word or phrase (or upload a `.wav` file) $\to$ the backend synthesizes/loads the audio, runs the PyTorch forward pass on your GPU, and extracts all intermediate layer tensors.
- **Dynamic Layer-by-Layer Inspection**: Switch between CNN Out and Transformer Layers 1–4 to view live 2D feature matrices and Multi-Head Attention weights computed directly from the current model.
- **On-the-Fly Captum Attribution**: Click any character token in the decoded transcript $\to$ Python calls Captum's `IntegratedGradients` on the GPU and returns the exact millisecond attribution curve overlaid on the input waveform.
- **Live Causal Ablation (NAPS)**: Click "Ablate L1", "Ablate L2", etc., to zero out that Transformer layer in memory and observe real-time logit degradation.
- **Live Training Console**: Click "Live Training Console" in the top bar to trigger background PyTorch training runs with real-time loss tracking and CER/WER gauges.

### Starting / Managing the Live Server:
```bash
# Start server manually (already running as background daemon on port 8000)
.venv/bin/python scripts/run_server.py --port 8000
```
