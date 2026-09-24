"""FastAPI Backend Server for Interactive HuBERT ASR & XAI Visualizations."""

import io
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.models.config import HuBERTConfig
from src.models.hubert_asr import HuBERTForCTC
from src.data.tokenizer import CharacterTokenizer
from src.data.dataset import AudioASRDataset, AudioCollateFn
from src.data.augmentations import WaveformAugmenter
from src.utils.audio import synthesize_spoken_word
from src.xai.captum_gradients import AudioGradientExplainer
from src.xai.naps import ActivationPatcher
from data.sample_dataset import load_manifest, generate_synthetic_asr_dataset


app = FastAPI(title="HuBERT Live ASR & XAI Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Container
class StudioState:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CharacterTokenizer()
        self.checkpoint_path = Path("checkpoints/best_model.pt")
        self.model: Optional[HuBERTForCTC] = None
        self.explainer: Optional[AudioGradientExplainer] = None
        self.patcher: Optional[ActivationPatcher] = None

        # Training state
        self.is_training = False
        self.should_stop_training = False
        self.training_thread: Optional[threading.Thread] = None
        self.training_status = {
            "is_training": False,
            "epoch": 0,
            "total_epochs": 0,
            "step": 0,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "val_cer": 0.0,
            "val_wer": 0.0,
            "history": [],
            "sample_prediction": "",
            "sample_reference": "",
        }

        self.last_audio_tensor: Optional[torch.Tensor] = None
        self.last_sample_rate = 16000
        self.init_model()

    def init_model(self):
        print(f"[Studio] Initializing model on device: {self.device}")
        if self.checkpoint_path.exists():
            print(f"[Studio] Loading checkpoint: {self.checkpoint_path}")
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self.model = HuBERTForCTC(ckpt["config"]).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            print("[Studio] No checkpoint found, creating base model.")
            config = HuBERTConfig(
                vocab_size=self.tokenizer.vocab_size,
                encoder_layers=4,
                encoder_heads=4,
                encoder_embed_dim=256,
                encoder_ffn_dim=1024,
            )
            self.model = HuBERTForCTC(config).to(self.device)

        self.model.eval()
        self.explainer = AudioGradientExplainer(self.model, self.device)
        self.patcher = ActivationPatcher(self.model, self.device)


state = StudioState()


# Pydantic Request Models
class SynthesizeRequest(BaseModel):
    word: str = "hello"
    duration: float = 0.8
    f0: float = 140.0


class AttributeRequest(BaseModel):
    frame_idx: int
    token_idx: int
    method: str = "integrated_gradients"
    steps: int = 20


class AblationRequest(BaseModel):
    layer_idx: int
    ablation_type: str = "zero"  # 'zero' or 'skip'


class TrainStartRequest(BaseModel):
    epochs: int = 5
    lr: float = 0.0005
    batch_size: int = 16


@app.get("/api/status")
def get_status():
    return {
        "device": str(state.device),
        "cuda_available": torch.cuda.is_available(),
        "checkpoint_exists": state.checkpoint_path.exists(),
        "is_training": state.is_training,
        "vocab_size": state.tokenizer.vocab_size,
        "encoder_layers": state.model.config.encoder_layers if state.model else 0,
        "embed_dim": state.model.config.encoder_embed_dim if state.model else 0,
    }


def process_audio_tensor(audio: torch.Tensor, transcript: str = ""):
    """Helper to run model and prepare complete layer inspection payload."""
    state.last_audio_tensor = audio.clone()
    audio = audio.to(state.device)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    # 1. Forward pass
    with torch.no_grad():
        outputs = state.model(audio, output_hidden_states=True, output_attentions=True)

    logits = outputs["logits"]  # (1, T_frames, V)
    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (T_frames, V)
    t_frames = probs.shape[0]

    # Greedy decode
    out_lengths = outputs["output_lengths"]
    decoded_tokens = state.model.decode_greedy(logits, lengths=out_lengths)[0]
    decoded_text = state.tokenizer.decode(decoded_tokens)

    # Raw audio downsampled for snappy web transfer
    raw_audio = audio.squeeze().cpu().numpy()
    num_samples = len(raw_audio)
    target_plot_points = 500
    step = max(1, num_samples // target_plot_points)
    plot_waveform = raw_audio[::step].tolist()

    # Time-frequency spectrogram approximation for display
    # Downsample time to match frames
    frame_step = max(1, num_samples // t_frames)
    spec_energy = []
    for f in range(t_frames):
        chunk = raw_audio[f * frame_step : (f + 1) * frame_step]
        if len(chunk) > 0:
            fft_mag = np.abs(np.fft.rfft(chunk, n=64))[:16]
            spec_energy.append(fft_mag.tolist())
        else:
            spec_energy.append([0.0] * 16)

    # Frame emissions
    argmax_ids = logits.squeeze(0).argmax(dim=-1).cpu().tolist()
    frame_emissions = []
    for t in range(t_frames):
        tok_id = argmax_ids[t]
        char = state.tokenizer.id_to_char.get(tok_id, "")
        is_blank = (tok_id == state.model.config.blank_index)
        frame_emissions.append({
            "frame": t,
            "time_sec": round(t * 0.02, 3),
            "token_id": tok_id,
            "char": "_" if is_blank else char,
            "confidence": round(float(probs[t, tok_id]), 4),
            "is_blank": is_blank,
            "top_probs": [
                {"char": state.tokenizer.id_to_char.get(i, ""), "prob": round(float(probs[t, i]), 3)}
                for i in np.argsort(probs[t])[-4:][::-1]
            ]
        })

    # Layer hidden representations (first 32 dimensions across all frames)
    layer_representations = []
    for l_idx, h in enumerate(outputs["hidden_states"]):
        h_np = h.squeeze(0).cpu().numpy()  # (T_frames, embed_dim)
        layer_representations.append({
            "layer_idx": l_idx,
            "name": "CNN Out" if l_idx == 0 else f"Layer {l_idx}",
            "features_2d": h_np[:, :32].T.tolist(),  # (32 dims, T_frames)
            "mean_energy": float(np.mean(h_np ** 2)),
        })

    # Multi-head attention (average across heads for each layer)
    attentions = []
    if outputs["attentions"] is not None:
        for l_idx, attn in enumerate(outputs["attentions"]):
            attn_np = attn.squeeze(0).cpu().numpy()  # (H, T, T)
            avg_attn = np.mean(attn_np, axis=0).tolist()
            attentions.append({
                "layer_idx": l_idx,
                "attention_matrix": avg_attn,
            })

    return {
        "num_audio_samples": num_samples,
        "duration_sec": round(num_samples / 16000, 3),
        "num_frames": t_frames,
        "transcript": transcript,
        "decoded_text": decoded_text,
        "waveform": plot_waveform,
        "spectrogram": spec_energy,
        "frame_emissions": frame_emissions,
        "layer_representations": layer_representations,
        "attentions": attentions,
        "vocab": state.tokenizer.vocab,
    }


@app.post("/api/synthesize_and_infer")
def synthesize_and_infer(req: SynthesizeRequest):
    """Synthesize speech word on the fly with formant synthesizer and run live PyTorch inference."""
    audio = synthesize_spoken_word(req.word, duration_s=req.duration, sample_rate=16000, f0=req.f0)
    return process_audio_tensor(audio, transcript=req.word)


@app.post("/api/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload custom WAV/audio file and run live PyTorch inference."""
    contents = await file.read()
    wav_np, sr = sf.read(io.BytesIO(contents), dtype="float32")
    audio = torch.from_numpy(wav_np)
    if audio.ndim > 1:
        audio = audio.mean(dim=-1)

    if sr != 16000:
        import torchaudio
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)

    return process_audio_tensor(audio, transcript=file.filename)


@app.post("/api/xai/attribute")
def run_attribution(req: AttributeRequest):
    """Run real-time Captum Integrated Gradients or Saliency on the current audio tensor."""
    if state.last_audio_tensor is None:
        return JSONResponse(status_code=400, content={"error": "No active audio. Run inference first."})

    audio = state.last_audio_tensor.clone()
    result = state.explainer.explain_token(
        audio=audio,
        frame_idx=req.frame_idx,
        token_idx=req.token_idx,
        method=req.method,
        n_steps=req.steps,
    )

    attr = result["attributions"].cpu().numpy()
    # Downsample to 500 points for plot
    step = max(1, len(attr) // 500)
    plot_attr = attr[::step].tolist()

    return {
        "frame_idx": req.frame_idx,
        "token_idx": req.token_idx,
        "method": req.method,
        "attribution_curve": plot_attr,
        "max_attr": float(np.max(np.abs(attr))),
    }


@app.post("/api/xai/ablate")
def run_ablation(req: AblationRequest):
    """Live causal intervention: zero-ablate or bypass a Transformer layer and measure logit shift."""
    if state.last_audio_tensor is None:
        return JSONResponse(status_code=400, content={"error": "No active audio. Run inference first."})

    res = state.patcher.ablate_layer(
        audio=state.last_audio_tensor,
        layer_idx=req.layer_idx,
        ablation_type=req.ablation_type,
    )
    orig_text = state.tokenizer.decode(res["orig_prediction"])
    ablated_text = state.tokenizer.decode(res["ablated_prediction"])

    return {
        "layer_idx": req.layer_idx,
        "ablation_type": req.ablation_type,
        "logit_mean_diff": round(res["logit_mean_diff"], 4),
        "orig_text": orig_text,
        "ablated_text": ablated_text,
    }


# Background Training Loop
def background_train_task(epochs: int, lr: float, batch_size: int):
    state.is_training = True
    state.should_stop_training = False
    state.training_status["is_training"] = True
    state.training_status["total_epochs"] = epochs
    state.training_status["history"] = []

    print(f"[Live Training] Starting background training for {epochs} epochs (lr={lr}, batch={batch_size})...")

    data_dir = Path("data/raw/synthetic")
    train_manifest = data_dir / "train_manifest.json"
    val_manifest = data_dir / "val_manifest.json"

    if not train_manifest.exists():
        generate_synthetic_asr_dataset(str(data_dir), num_train=60, num_val=15)

    train_samples = load_manifest(str(train_manifest))
    val_samples = load_manifest(str(val_manifest))

    train_dataset = AudioASRDataset(train_samples, state.tokenizer, target_sample_rate=16000, augmenter=WaveformAugmenter())
    val_dataset = AudioASRDataset(val_samples, state.tokenizer, target_sample_rate=16000)
    collate_fn = AudioCollateFn(pad_token_id=state.tokenizer.pad_id)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(state.model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        if state.should_stop_training:
            print("[Live Training] User requested stop.")
            break

        state.model.train()
        epoch_loss = 0.0
        steps = 0

        for batch in train_loader:
            if state.should_stop_training:
                break
            audio = batch["audio"].to(state.device)
            audio_lengths = batch["audio_lengths"].to(state.device)
            targets = batch["targets"].to(state.device)
            target_lengths = batch["target_lengths"].to(state.device)

            optimizer.zero_grad()
            outputs = state.model(audio, audio_lengths=audio_lengths, targets=targets, target_lengths=target_lengths)
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(state.model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

            state.training_status["step"] += 1
            state.training_status["train_loss"] = round(loss.item(), 4)

        # Quick val step
        state.model.eval()
        val_loss = 0.0
        val_preds, val_refs = [], []
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(state.device)
                outputs = state.model(audio, targets=batch["targets"].to(state.device), target_lengths=batch["target_lengths"].to(state.device), audio_lengths=batch["audio_lengths"].to(state.device))
                val_loss += outputs["loss"].item()
                decoded = state.model.decode_greedy(outputs["logits"], lengths=outputs["output_lengths"])
                val_preds.extend([state.tokenizer.decode(t) for t in decoded])
                val_refs.extend(batch["texts"])

        from src.training.metrics import compute_cer, compute_wer
        cer = compute_cer(val_preds, val_refs)
        wer = compute_wer(val_preds, val_refs)

        state.training_status["epoch"] = epoch
        state.training_status["val_loss"] = round(val_loss / max(1, len(val_loader)), 4)
        state.training_status["val_cer"] = round(cer * 100, 2)
        state.training_status["val_wer"] = round(wer * 100, 2)
        if len(val_preds) > 0:
            state.training_status["sample_prediction"] = val_preds[0]
            state.training_status["sample_reference"] = val_refs[0]

        state.training_status["history"].append({
            "epoch": epoch,
            "train_loss": round(epoch_loss / max(1, steps), 4),
            "val_loss": state.training_status["val_loss"],
            "val_cer": state.training_status["val_cer"],
        })

        # Save latest model
        torch.save({
            "epoch": epoch,
            "model_state_dict": state.model.state_dict(),
            "config": state.model.config,
        }, "checkpoints/best_model.pt")

    state.is_training = False
    state.training_status["is_training"] = False
    print("[Live Training] Background training finished.")


@app.post("/api/train/start")
def start_training(req: TrainStartRequest, background_tasks: BackgroundTasks):
    if state.is_training:
        return {"status": "already_running"}
    state.training_thread = threading.Thread(target=background_train_task, args=(req.epochs, req.lr, req.batch_size))
    state.training_thread.start()
    return {"status": "started", "epochs": req.epochs}


@app.post("/api/train/stop")
def stop_training():
    if not state.is_training:
        return {"status": "not_running"}
    state.should_stop_training = True
    return {"status": "stopping"}


@app.get("/api/train/status")
def get_training_status():
    return state.training_status


# Serve HTML Dashboard
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<h1>HuBERT Studio Backend Running. index.html not found.</h1>"
