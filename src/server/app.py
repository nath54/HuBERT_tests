"""FastAPI Backend Server for Interactive HuBERT ASR, Real Dataset & XAI Studio."""

import base64
import io
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, UploadFile, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.models.config import HuBERTConfig
from src.models.hubert_asr import HuBERTForCTC
from src.models.phono_hubert import PhonoHuBERTConfig, PhonoHuBERTForPreTraining
from src.models.registry import ModelRegistry, STANDARD_TIERS
from src.data.tokenizer import CharacterTokenizer
from src.data.phoneme_tokenizer import PhonemeTokenizer
from src.data.target_extractors import PhonemeTargetExtractor, KMeansUnitExtractor
from src.data.dataset import AudioASRDataset, AudioCollateFn
from src.data.augmentations import WaveformAugmenter
from src.utils.audio import synthesize_spoken_word
from src.utils.benchmarks import (
    get_model_parameters_breakdown,
    benchmark_inference_speed,
    compute_scaling_analysis,
)
from src.xai.captum_gradients import AudioGradientExplainer
from src.xai.naps import ActivationPatcher
from data.sample_dataset import load_manifest


app = FastAPI(title="HuBERT Live ASR & XAI Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StudioState:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.char_tokenizer = CharacterTokenizer()
        self.phono_tokenizer = PhonemeTokenizer()
        self.tokenizer = self.char_tokenizer
        self.current_arch = "hubert_kmeans"
        self.current_tier = "mini"
        self.checkpoint_path = Path("checkpoints/best_model.pt")
        self.model: Optional[HuBERTForCTC] = None
        self.explainer: Optional[AudioGradientExplainer] = None
        self.patcher: Optional[ActivationPatcher] = None

        # Real LibriSpeech samples cache
        self.librispeech_samples: List[Dict] = []
        self.librispeech_sample_map: Dict[str, Dict] = {}
        self.load_dataset_manifests()

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
        self.test_clean_samples: List[Dict] = []
        self.sota_runner = None
        self.pretrain_thread: Optional[threading.Thread] = None
        self.is_pretraining = False
        self.should_stop_pretraining = False
        self.streaming_components = None
        self.init_pretrain_status()
        self.init_model()

    def init_pretrain_status(self):
        pretrain_log = Path("logs/pretrain_history.json")
        if pretrain_log.exists():
            try:
                with open(pretrain_log, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    h = data.get("history", [])
                    last_entry = h[-1] if h else {}
                    self.pretrain_status = {
                        "is_running": False,
                        "step": data.get("total_steps", 0),
                        "total_steps": max(50, data.get("total_steps", 50)),
                        "loss": last_entry.get("loss", 4.8),
                        "accuracy": last_entry.get("masked_accuracy_pct", 1.0),
                        "cumulative_audio_sec": last_entry.get("cumulative_audio_sec", 0.0),
                        "cumulative_audio_hours": data.get("cumulative_audio_hours", 0.0),
                        "disk_space_saved_mb": round((data.get("cumulative_audio_hours", 0.0) * 3600.0 * 32000) / 1024 / 1024, 2),
                        "active_voice": last_entry.get("active_voices", ["lessac"])[0],
                        "sample_prompt": last_entry.get("sample_prompt", ""),
                        "history": h,
                    }
                    return
            except Exception:
                pass

        self.pretrain_status = {
            "is_running": False,
            "step": 0,
            "total_steps": 50,
            "loss": 4.85,
            "accuracy": 1.0,
            "cumulative_audio_sec": 0.0,
            "cumulative_audio_hours": 0.0,
            "disk_space_saved_mb": 0.0,
            "active_voice": "lessac",
            "sample_prompt": "",
            "history": [],
        }

    def get_streaming_components(self):
        if self.streaming_components is None:
            from src.data.streaming_piper import PiperVoiceManager, ProceduralTextSampler, AcousticUnitExtractor
            self.streaming_components = {
                "voice_mgr": PiperVoiceManager(),
                "text_sampler": ProceduralTextSampler(),
                "unit_extractor": AcousticUnitExtractor(num_clusters=100),
            }
        return self.streaming_components

    def load_dataset_manifests(self):
        manifest_path = Path("data/librispeech/librispeech_all.json")
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.librispeech_samples = json.load(f)
                self.librispeech_sample_map = {s["id"]: s for s in self.librispeech_samples}
            print(f"[Studio] Loaded {len(self.librispeech_samples):,} real LibriSpeech utterances.")

        test_clean_path = Path("data/librispeech/librispeech_test_clean.json")
        if test_clean_path.exists():
            with open(test_clean_path, "r", encoding="utf-8") as f:
                self.test_clean_samples = json.load(f)
                for s in self.test_clean_samples:
                    self.librispeech_sample_map[s["id"]] = s
            print(f"[Studio] Loaded {len(self.test_clean_samples):,} test-clean benchmark utterances.")

    def get_sota_runner(self):
        if self.sota_runner is None:
            from src.benchmark.sota_evaluator import SOTABenchmarkRunner
            self.sota_runner = SOTABenchmarkRunner(device=self.device)
        return self.sota_runner

    def init_model(self):
        print(f"[Studio] Initializing model on device: {self.device}")
        if self.checkpoint_path.exists():
            print(f"[Studio] Loading checkpoint: {self.checkpoint_path}")
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self.model = HuBERTForCTC(ckpt["config"]).to(self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            print("[Studio] Creating base model.")
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

    def select_model(self, arch: str, tier: str = "mini", checkpoint_path: Optional[str] = None, overrides: Optional[Dict] = None):
        overrides = overrides or {}
        print(f"[Studio] Switching to model architecture: {arch} | tier: {tier} | overrides: {overrides}")

        if arch == "phono_hubert":
            self.tokenizer = self.phono_tokenizer
        else:
            self.tokenizer = self.char_tokenizer

        ckpt_to_load = None
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt_to_load = Path(checkpoint_path)
        else:
            if arch == "phono_hubert":
                cand1 = Path(f"checkpoints/phono_hubert/{tier}/checkpoint_latest.pt")
                cand2 = Path(f"checkpoints/phono_hubert/{tier}/ctc_downstream_latest.pt")
                if cand1.exists():
                    ckpt_to_load = cand1
                elif cand2.exists():
                    ckpt_to_load = cand2
            else:
                if tier == "mini":
                    cand1 = Path("checkpoints/best_model.pt")
                    cand2 = Path("checkpoints/pretrain_checkpoint_latest.pt")
                    if cand1.exists():
                        ckpt_to_load = cand1
                    elif cand2.exists():
                        ckpt_to_load = cand2
                else:
                    cand = Path(f"checkpoints/hubert_kmeans/{tier}/checkpoint_latest.pt")
                    if cand.exists():
                        ckpt_to_load = cand

        if ckpt_to_load and ckpt_to_load.exists():
            print(f"[Studio] Loading checkpoint for {arch} [{tier}]: {ckpt_to_load}")
            ckpt = torch.load(ckpt_to_load, map_location=self.device, weights_only=False)
            cfg = ckpt.get("config")
            if cfg is None:
                cfg = ModelRegistry.build_config(arch, tier=tier, **overrides)

            if arch == "phono_hubert":
                self.model = PhonoHuBERTForPreTraining(cfg).to(self.device)
            else:
                self.model = HuBERTForCTC(cfg).to(self.device)

            try:
                self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            except Exception as e:
                print(f"[Studio] Note on state dict load: {e}")
            self.checkpoint_path = ckpt_to_load
        else:
            print(f"[Studio] Creating fresh model for {arch} [{tier}].")
            self.model = ModelRegistry.build_model(arch, tier=tier, **overrides).to(self.device)
            self.checkpoint_path = Path(f"checkpoints/{arch}_{tier}_init.pt")

        self.model.eval()
        self.current_arch = arch
        self.current_tier = tier
        self.explainer = AudioGradientExplainer(self.model, self.device)
        self.patcher = ActivationPatcher(self.model, self.device)

        num_params = sum(p.numel() for p in self.model.parameters())
        return {
            "arch": self.current_arch,
            "tier": self.current_tier,
            "checkpoint_path": str(self.checkpoint_path),
            "parameters": num_params,
            "parameters_m": round(num_params / 1e6, 2),
            "vocab_size": self.tokenizer.vocab_size,
        }


state = StudioState()


# Request Models
class SynthesizeRequest(BaseModel):
    word: str = "hello"
    duration: float = 0.8
    f0: float = 140.0
    blank_penalty: float = 0.0


class AttributeRequest(BaseModel):
    frame_idx: int = 15
    token_idx: Optional[int] = None
    char: Optional[str] = None
    method: str = "integrated_gradients"
    steps: int = 20


class AblationRequest(BaseModel):
    layer_idx: int
    ablation_type: str = "zero"


class TrainStartRequest(BaseModel):
    epochs: int = 5
    lr: float = 0.0003
    batch_size: int = 8
    dataset_type: str = "librispeech"  # 'librispeech' or 'synthetic'


class InferSampleRequest(BaseModel):
    sample_id: str
    blank_penalty: float = 0.0


class ModelSelectRequest(BaseModel):
    arch: str = "hubert_kmeans"
    tier: str = "mini"
    checkpoint_path: Optional[str] = None
    overrides: Optional[Dict] = None


@app.get("/api/models/catalog")
def get_models_catalog():
    """Returns all available architectures, sizes/tiers, active selection, and available checkpoints."""
    catalog = ModelRegistry.list_models()

    for m in catalog:
        m["available_checkpoints"] = {}
        for tier in m["supported_tiers"]:
            ckpts = []
            if m["id"] == "hubert_kmeans":
                if tier == "mini":
                    if Path("checkpoints/best_model.pt").exists():
                        ckpts.append("checkpoints/best_model.pt")
                    if Path("checkpoints/pretrain_checkpoint_latest.pt").exists():
                        ckpts.append("checkpoints/pretrain_checkpoint_latest.pt")
            elif m["id"] == "phono_hubert":
                tier_dir = Path(f"checkpoints/phono_hubert/{tier}")
                if tier_dir.exists():
                    for f in tier_dir.glob("*.pt"):
                        ckpts.append(str(f))
            m["available_checkpoints"][tier] = ckpts

    num_params = sum(p.numel() for p in state.model.parameters()) if state.model else 0
    return {
        "models": catalog,
        "tiers": STANDARD_TIERS,
        "current": {
            "arch": state.current_arch,
            "tier": state.current_tier,
            "checkpoint_path": str(state.checkpoint_path) if state.checkpoint_path else None,
            "parameters": num_params,
            "parameters_m": round(num_params / 1e6, 2),
            "target_type": "phoneme_tokens" if state.current_arch == "phono_hubert" else "acoustic_clusters",
            "vocab_size": state.tokenizer.vocab_size,
        }
    }


@app.post("/api/models/select")
def select_model_endpoint(req: ModelSelectRequest):
    """Switch active model architecture and size tier dynamically."""
    try:
        res = state.select_model(
            arch=req.arch,
            tier=req.tier,
            checkpoint_path=req.checkpoint_path,
            overrides=req.overrides,
        )
        return {"status": "ok", **res}
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})


@app.get("/api/status")
def get_status():
    num_params = sum(p.numel() for p in state.model.parameters()) if state.model else 0
    return {
        "device": str(state.device),
        "cuda_available": torch.cuda.is_available(),
        "checkpoint_exists": state.checkpoint_path.exists() if state.checkpoint_path else False,
        "is_training": state.is_training,
        "vocab_size": state.tokenizer.vocab_size,
        "vocab": getattr(state.tokenizer, "vocab", getattr(state.tokenizer, "id_to_phoneme", {})),
        "encoder_layers": state.model.config.encoder_layers if state.model else 0,
        "embed_dim": state.model.config.encoder_embed_dim if state.model else 0,
        "real_dataset_samples": len(state.librispeech_samples),
        "current_arch": state.current_arch,
        "current_tier": state.current_tier,
        "parameters": num_params,
        "parameters_m": round(num_params / 1e6, 2),
    }


# -------------------------------------------------------------
# BENCHMARKS & SCALING ENDPOINT
# -------------------------------------------------------------
@app.get("/api/benchmark")
def get_benchmarks():
    """Returns detailed parameter breakdown, latency, RTF, and scaling laws."""
    params_breakdown = get_model_parameters_breakdown(state.model)
    speed_results = benchmark_inference_speed(state.model, state.device, durations=[0.5, 1.0, 3.0, 5.0])
    scaling_analysis = compute_scaling_analysis(state.model.config, durations=[1.0, 5.0, 10.0, 30.0, 60.0])

    return {
        "parameters": params_breakdown,
        "speed_benchmarks": speed_results,
        "scaling_analysis": scaling_analysis,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


# -------------------------------------------------------------
# REAL DATASET ENDPOINTS
# -------------------------------------------------------------
@app.get("/api/dataset/stats")
def get_dataset_stats():
    stats_file = Path("data/librispeech/dataset_stats.json")
    if stats_file.exists():
        with open(stats_file, "r") as f:
            return json.load(f)
    return {"error": "Stats not yet generated"}


@app.get("/api/dataset/samples")
def get_dataset_samples(page: int = Query(1, ge=1), page_size: int = Query(15, ge=1, le=100), search: str = ""):
    """Returns paginated real LibriSpeech recordings."""
    samples = state.librispeech_samples
    if search:
        s_lower = search.lower()
        samples = [s for s in samples if s_lower in s["transcript"] or s_lower in s["id"] or s_lower in s["speaker_id"]]

    total = len(samples)
    start = (page - 1) * page_size
    end = start + page_size
    paged = samples[start:end]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
        "samples": paged,
    }


@app.get("/api/dataset/audio/{sample_id}")
def get_dataset_audio(sample_id: str):
    """Serve real audio file for browser playback."""
    sample = state.librispeech_sample_map.get(sample_id)
    if not sample or not Path(sample["audio_path"]).exists():
        return JSONResponse(status_code=404, content={"error": "Audio file not found"})
    return FileResponse(sample["audio_path"], media_type="audio/flac")


@app.post("/api/dataset/infer_sample")
def infer_dataset_sample(req: InferSampleRequest):
    """Run full HuBERT inference and XAI inspection on a real LibriSpeech sample."""
    sample = state.librispeech_sample_map.get(req.sample_id)
    if not sample or not Path(sample["audio_path"]).exists():
        return JSONResponse(status_code=404, content={"error": "Sample not found"})

    wav_np, sr = sf.read(sample["audio_path"], dtype="float32")
    audio = torch.from_numpy(wav_np)
    if audio.ndim > 1:
        audio = audio.mean(dim=-1)

    return process_audio_tensor(audio, transcript=sample["transcript"], blank_penalty=req.blank_penalty)


# -------------------------------------------------------------
# CORE INFERENCE & XAI PIPELINE
# -------------------------------------------------------------
def process_audio_tensor(audio: torch.Tensor, transcript: str = "", blank_penalty: float = 0.0):
    state.last_audio_tensor = audio.clone()
    audio = audio.to(state.device)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    with torch.no_grad():
        outputs = state.model(audio, output_hidden_states=True, output_attentions=True)

    logits = outputs["logits"].clone()
    t_frames = logits.shape[1]

    penalized_logits = logits.clone()
    if blank_penalty > 0.0:
        penalized_logits[:, :, state.model.config.blank_index] -= blank_penalty

    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    penalized_probs = F.softmax(penalized_logits, dim=-1).squeeze(0).cpu().numpy()

    out_lengths = outputs["output_lengths"]
    decoded_tokens = state.model.decode_greedy(penalized_logits, lengths=out_lengths)[0]
    decoded_text = state.tokenizer.decode(decoded_tokens)

    def get_tok_char(idx: int) -> str:
        if hasattr(state.tokenizer, "id_to_char"):
            return state.tokenizer.id_to_char.get(idx, "")
        elif hasattr(state.tokenizer, "id_to_phoneme"):
            return state.tokenizer.id_to_phoneme.get(idx, "")
        return str(idx)

    blank_id = getattr(state.model.config, "blank_index", 0)
    if blank_id == 0:
        non_blank_logits = logits.squeeze(0)[:, 1:]
        top_nb_ids = (non_blank_logits.argmax(dim=-1) + 1).cpu().tolist()
    else:
        nb_logits = logits.squeeze(0).clone()
        nb_logits[:, blank_id] = -1e9
        top_nb_ids = nb_logits.argmax(dim=-1).cpu().tolist()

    if hasattr(state.tokenizer, "id_to_char"):
        top_non_blank_text = "".join([get_tok_char(i) for i in top_nb_ids])
    else:
        top_non_blank_text = " ".join([get_tok_char(i) for i in top_nb_ids if i not in (0, blank_id)])

    raw_audio = audio.squeeze().cpu().numpy()
    num_samples = len(raw_audio)
    step = max(1, num_samples // 500)
    plot_waveform = raw_audio[::step].tolist()

    frame_step = max(1, num_samples // t_frames)
    spec_energy = []
    for f in range(t_frames):
        chunk = raw_audio[f * frame_step : (f + 1) * frame_step]
        if len(chunk) > 0:
            fft_mag = np.abs(np.fft.rfft(chunk, n=64))[:16]
            spec_energy.append(fft_mag.tolist())
        else:
            spec_energy.append([0.0] * 16)

    argmax_ids = penalized_logits.squeeze(0).argmax(dim=-1).cpu().tolist()
    frame_emissions = []
    candidate_tokens = []

    for t in range(t_frames):
        tok_id = argmax_ids[t]
        char = get_tok_char(tok_id)
        is_blank = (tok_id == blank_id)

        nb_tok_id = top_nb_ids[t]
        nb_char = get_tok_char(nb_tok_id)
        nb_prob = float(probs[t, nb_tok_id])

        top_indices = np.argsort(probs[t])[-5:][::-1]
        top_list = [
            {
                "char": get_tok_char(int(i)),
                "token_id": int(i),
                "prob": round(float(probs[t, i]), 4),
                "is_blank": int(i) == blank_id,
            }
            for i in top_indices
        ]

        frame_data = {
            "frame": t,
            "time_sec": round(t * 0.02, 3),
            "token_id": tok_id,
            "char": "_" if is_blank else char,
            "is_blank": is_blank,
            "confidence": round(float(penalized_probs[t, tok_id]), 4),
            "top_non_blank_char": nb_char,
            "top_non_blank_id": nb_tok_id,
            "top_non_blank_prob": round(nb_prob, 4),
            "top_probs": top_list,
        }
        frame_emissions.append(frame_data)

        if not is_blank or nb_prob > 0.005:
            candidate_tokens.append({
                "frame": t,
                "token_id": nb_tok_id,
                "char": nb_char,
                "prob": round(nb_prob, 4),
                "time_sec": round(t * 0.02, 3),
            })

    layer_representations = []
    for l_idx, h in enumerate(outputs["hidden_states"]):
        h_np = h.squeeze(0).cpu().numpy()
        layer_representations.append({
            "layer_idx": l_idx,
            "name": "CNN Out" if l_idx == 0 else f"Layer {l_idx}",
            "features_2d": h_np[:, :32].T.tolist(),
            "mean_energy": float(np.mean(h_np ** 2)),
        })

    attentions = []
    if outputs["attentions"] is not None:
        for l_idx, attn in enumerate(outputs["attentions"]):
            attn_np = attn.squeeze(0).cpu().numpy()
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
        "top_non_blank_text": top_non_blank_text,
        "blank_penalty": blank_penalty,
        "waveform": plot_waveform,
        "spectrogram": spec_energy,
        "frame_emissions": frame_emissions,
        "candidate_tokens": candidate_tokens,
        "layer_representations": layer_representations,
        "attentions": attentions,
        "vocab": getattr(state.tokenizer, "vocab", getattr(state.tokenizer, "id_to_phoneme", {})),
    }


@app.post("/api/synthesize_and_infer")
def synthesize_and_infer(req: SynthesizeRequest):
    audio = synthesize_spoken_word(req.word, duration_s=req.duration, sample_rate=16000, f0=req.f0)
    return process_audio_tensor(audio, transcript=req.word, blank_penalty=req.blank_penalty)


@app.post("/api/upload_audio")
async def upload_audio(file: UploadFile = File(...), blank_penalty: float = Form(0.0)):
    contents = await file.read()
    wav_np, sr = sf.read(io.BytesIO(contents), dtype="float32")
    audio = torch.from_numpy(wav_np)
    if audio.ndim > 1:
        audio = audio.mean(dim=-1)

    if sr != 16000:
        import torchaudio
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)

    return process_audio_tensor(audio, transcript=file.filename, blank_penalty=blank_penalty)


@app.post("/api/xai/attribute")
def run_attribution(req: AttributeRequest):
    if state.last_audio_tensor is None:
        return JSONResponse(status_code=400, content={"error": "No active audio. Run inference first."})

    token_idx = req.token_idx
    if token_idx is None and req.char is not None:
        token_idx = state.tokenizer.char_to_id.get(req.char, 0)
    elif token_idx is None:
        token_idx = 0

    audio = state.last_audio_tensor.clone()
    result = state.explainer.explain_token(
        audio=audio,
        frame_idx=req.frame_idx,
        token_idx=token_idx,
        method=req.method,
        n_steps=req.steps,
    )

    attr = result["attributions"].cpu().numpy()
    step = max(1, len(attr) // 500)
    plot_attr = attr[::step].tolist()
    char_name = state.tokenizer.id_to_char.get(token_idx, f"id_{token_idx}")

    return {
        "frame_idx": req.frame_idx,
        "token_idx": token_idx,
        "char": char_name,
        "method": req.method,
        "attribution_curve": plot_attr,
        "max_attr": float(np.max(np.abs(attr))),
    }


@app.post("/api/xai/ablate")
def run_ablation(req: AblationRequest):
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


def get_live_scaling_status(arch: Optional[str] = None, tier: Optional[str] = None) -> Optional[Dict]:
    """Dynamically discover and parse active or recent large-scale pre-training status."""
    import subprocess
    import glob
    import re
    import time
    from datetime import datetime, timedelta

    arch = arch or state.current_arch
    tier = tier or state.current_tier

    if arch == "phono_hubert":
        p = subprocess.run(["pgrep", "-f", "run_pretrain.py.*phono_hubert"], capture_output=True, text=True)
        is_running = (p.returncode == 0 and len(p.stdout.strip()) > 0)

        live_status_path = Path("logs/phono_hubert_status_live.json")
        live_data = {}
        if live_status_path.exists():
            try:
                with open(live_status_path, "r", encoding="utf-8") as f:
                    live_data = json.load(f)
            except Exception:
                pass

        heartbeat_fresh = (time.time() - live_data.get("last_heartbeat", 0)) < 90
        if heartbeat_fresh and live_data.get("is_running"):
            is_running = True

        history_file = Path(f"logs/phono_hubert_{tier}_history.json")
        milestones = []
        history = []
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    ph_data = json.load(f)
                    for h in ph_data.get("history", []):
                        if "librispeech_wer" in h:
                            milestones.append({
                                "step": h.get("step", 0),
                                "hours": h.get("cumulative_audio_hours", 0.0),
                                "loss": h.get("pretrain_loss", 0.0),
                                "wer": h.get("librispeech_wer", 100.0),
                                "cer": h.get("librispeech_cer", 100.0),
                                "per": h.get("librispeech_per", 100.0),
                                "sample_pred": h.get("sample_prediction", ""),
                            })
                        history.append({
                            "step": h.get("step", 0),
                            "loss": h.get("pretrain_loss", 0.0),
                            "accuracy": h.get("masked_acc_pct", 1.0),
                            "masked_accuracy_pct": h.get("masked_acc_pct", 1.0),
                            "cumulative_audio_sec": h.get("cumulative_audio_hours", 0.0) * 3600.0,
                            "cumulative_audio_hours": h.get("cumulative_audio_hours", 0.0),
                            "disk_bytes_used": 0,
                            "active_voices": ["piper_neural"],
                            "active_voice": "piper_neural",
                        })
            except Exception:
                pass

        step = live_data.get("step", history[-1]["step"] if history else 0)
        total_steps = live_data.get("total_steps", 500)
        loss = live_data.get("loss", history[-1]["loss"] if history else 8.5)
        acc = live_data.get("masked_accuracy_pct", history[-1]["accuracy"] if history else 90.0)
        audio_sec = live_data.get("cumulative_audio_sec", history[-1]["cumulative_audio_sec"] if history else 0.0)
        audio_hours = live_data.get("cumulative_audio_hours", history[-1]["cumulative_audio_hours"] if history else 0.0)
        active_voice = live_data.get("active_voice", "piper")
        active_voices = live_data.get("active_voices", ["piper"])
        avg_step_sec = live_data.get("avg_step_sec", 3.0)
        eta_formatted = live_data.get("eta_formatted", "--")
        estimated_finish_time = live_data.get("estimated_finish_time", "--")

        clean_recent_logs = []
        if history:
            for h in history[-25:]:
                clean_recent_logs.append(
                    f"[PHONO_HUBERT] Step {h['step']:4d}/{total_steps} | Loss: {h['loss']:.4f} | Masked Acc: {h.get('masked_accuracy_pct', 90.0):.1f}% | Audio: {h['cumulative_audio_hours']:.3f}h"
                )

        return {
            "is_running": is_running,
            "source": "phono_hubert",
            "arch": "phono_hubert",
            "tier": tier,
            "parameters_m": 8.04,
            "step": step,
            "total_steps": total_steps,
            "loss": loss,
            "accuracy": acc,
            "masked_accuracy_pct": acc,
            "cumulative_audio_sec": audio_sec,
            "cumulative_audio_hours": audio_hours,
            "disk_space_saved_mb": round((audio_sec * 32000) / 1024 / 1024, 2),
            "disk_bytes_used": 0,
            "active_voice": active_voice,
            "active_voices": active_voices,
            "eta_formatted": eta_formatted or "--",
            "estimated_finish_time": estimated_finish_time or "--",
            "avg_step_sec": round(avg_step_sec, 2),
            "history": history,
            "milestones": milestones,
            "recent_logs": clean_recent_logs[-35:],
        }

    # Check if process is running
    p = subprocess.run(["pgrep", "-f", "scale_pretrain_benchmark"], capture_output=True, text=True)
    is_running = (p.returncode == 0 and len(p.stdout.strip()) > 0)

    # Check live status JSON
    live_status_path = Path("logs/pretrain_status_live.json")
    live_data = {}
    if live_status_path.exists():
        try:
            with open(live_status_path, "r", encoding="utf-8") as f:
                live_data = json.load(f)
        except Exception:
            pass

    # If process is running or heartbeat is fresh (< 90s), mark running
    heartbeat_fresh = (time.time() - live_data.get("last_heartbeat", 0)) < 90
    if heartbeat_fresh:
        is_running = True

    # If not running and no live data exists at all, bail out to state.pretrain_status
    if not is_running and not live_data:
        return None

    # Check scaling milestones file
    scaling_file = Path("logs/scaling_benchmark_history.json")
    milestones = []
    tier = live_data.get("tier", "mini")
    params_m = 8.05
    if scaling_file.exists():
        try:
            with open(scaling_file, "r", encoding="utf-8") as f:
                sc_data = json.load(f)
                tier = sc_data.get("model_tier", tier)
                params_m = sc_data.get("parameters_m", params_m)
                for h in sc_data.get("history", []):
                    if "librispeech_wer" in h:
                        milestones.append({
                            "step": h.get("step", 0),
                            "hours": h.get("cumulative_audio_hours", 0.0),
                            "loss": h.get("pretrain_loss", 0.0),
                            "wer": h.get("librispeech_wer", 100.0),
                            "cer": h.get("librispeech_cer", 100.0),
                            "per": h.get("librispeech_per", 100.0),
                            "sample_pred": h.get("sample_prediction", ""),
                        })
        except Exception:
            pass

    # Persistent step history tracking
    step_history_file = Path("logs/scaling_benchmark_step_history.json")
    history = []
    if step_history_file.exists():
        try:
            with open(step_history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    # If history is empty, populate from milestone anchors
    if not history and milestones:
        for m in milestones:
            if m["step"] > 0:
                history.append({
                    "step": m["step"],
                    "loss": m["loss"],
                    "accuracy": 1.0,
                    "masked_accuracy_pct": 1.0,
                    "cumulative_audio_sec": m["hours"] * 3600.0,
                    "cumulative_audio_hours": m["hours"],
                    "disk_bytes_used": 0,
                    "active_voices": ["piper"],
                    "active_voice": "piper",
                })

    # Search for any task logs across all sessions for detailed historical lines
    logs = glob.glob(os.path.expanduser("~/.gemini/antigravity/brain/*/.system_generated/tasks/*.log"))
    clean_recent_logs = []
    step_pattern = re.compile(
        r"Step\s+(\d+)/(\d+)\s+\|\s+Loss:\s+([\d\.]+)\s+\|\s+Masked Acc:\s+([\d\.]+)%\s+\|\s+Audio In-RAM:\s+([\d\.]+)s\s+\(([\d\.]+)h\)\s+\|\s+(\d+)\s+Disk Bytes\s+\|\s+Voices:\s+\[(.*?)\]"
    )

    for l in sorted(logs, key=os.path.getmtime, reverse=True)[:25]:
        try:
            with open(l, "r", errors="ignore") as f:
                head = f.read(2000)
                if "scale_pretrain_benchmark" in head or "LARGE-SCALE HUBERT PRE-TRAINING" in head:
                    f.seek(0)
                    for line in f:
                        if "Missing phoneme from id map" in line:
                            continue
                        line_clean = line.strip()
                        if line_clean:
                            clean_recent_logs.append(line_clean)
                        m = step_pattern.search(line)
                        if m:
                            st = int(m.group(1))
                            if not any(h["step"] == st for h in history):
                                history.append({
                                    "step": st,
                                    "total_steps": int(m.group(2)),
                                    "loss": float(m.group(3)),
                                    "accuracy": float(m.group(4)),
                                    "masked_accuracy_pct": float(m.group(4)),
                                    "cumulative_audio_sec": float(m.group(5)),
                                    "cumulative_audio_hours": float(m.group(6)),
                                    "disk_bytes_used": int(m.group(7)),
                                    "active_voices": [v.strip() for v in m.group(8).split(",")],
                                    "active_voice": m.group(8),
                                })
        except Exception:
            pass

    # Append current live step to history if live_data is present
    current_step = live_data.get("step")
    if current_step and not any(h["step"] == current_step for h in history):
        new_entry = {
            "step": current_step,
            "total_steps": live_data.get("total_steps", 625),
            "loss": live_data.get("loss", 4.3),
            "accuracy": live_data.get("masked_accuracy_pct", 1.0),
            "masked_accuracy_pct": live_data.get("masked_accuracy_pct", 1.0),
            "cumulative_audio_sec": live_data.get("cumulative_audio_sec", 0.0),
            "cumulative_audio_hours": live_data.get("cumulative_audio_hours", 0.0),
            "disk_bytes_used": 0,
            "active_voices": live_data.get("active_voices", []),
            "active_voice": live_data.get("active_voice", ""),
        }
        history.append(new_entry)
        history.sort(key=lambda x: x["step"])
        try:
            with open(step_history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        except Exception:
            pass

    history.sort(key=lambda x: x["step"])

    # Extract current stats
    step = live_data.get("step", history[-1]["step"] if history else 0)
    total_steps = live_data.get("total_steps", 625)
    loss = live_data.get("loss", history[-1]["loss"] if history else 4.5)
    acc = live_data.get("masked_accuracy_pct", history[-1]["accuracy"] if history else 1.0)
    audio_sec = live_data.get("cumulative_audio_sec", history[-1]["cumulative_audio_sec"] if history else 0.0)
    audio_hours = live_data.get("cumulative_audio_hours", history[-1]["cumulative_audio_hours"] if history else 0.0)
    active_voice = live_data.get("active_voice", history[-1].get("active_voice", "piper") if history else "piper")
    active_voices = live_data.get("active_voices", history[-1].get("active_voices", []) if history else [])
    avg_step_sec = live_data.get("avg_step_sec", 12.5)

    eta_formatted = live_data.get("eta_formatted")
    estimated_finish_time = live_data.get("estimated_finish_time")
    if not eta_formatted and step > 0:
        rem_steps = max(0, total_steps - step)
        rem_evals = max(0, rem_steps // 125)
        eta_seconds = (rem_steps * avg_step_sec) + (rem_evals * 35.0)
        h = int(eta_seconds // 3600)
        m = int((eta_seconds % 3600) // 60)
        eta_formatted = f"{h}h {m:02d}m" if h > 0 else f"{m}m"
        estimated_finish_time = (datetime.now() + timedelta(seconds=eta_seconds)).strftime("%H:%M:%S")

    # If clean_recent_logs is sparse, synthesise from recent history entries
    if len(clean_recent_logs) < 10 and history:
        for h in history[-20:]:
            clean_recent_logs.append(
                f"Step {h['step']:4d}/{total_steps} | Loss: {h['loss']:.4f} | Masked Acc: {h.get('masked_accuracy_pct', 1.0):.1f}% | Audio: {h['cumulative_audio_hours']:.3f}h | Voices: [{h.get('active_voice', '')}]"
            )

    return {
        "is_running": is_running,
        "source": "scaling_benchmark",
        "tier": tier,
        "parameters_m": params_m,
        "step": step,
        "total_steps": total_steps,
        "loss": loss,
        "accuracy": acc,
        "masked_accuracy_pct": acc,
        "cumulative_audio_sec": audio_sec,
        "cumulative_audio_hours": audio_hours,
        "disk_space_saved_mb": round((audio_sec * 32000) / 1024 / 1024, 2),
        "disk_bytes_used": 0,
        "active_voice": active_voice,
        "active_voices": active_voices,
        "eta_formatted": eta_formatted or "--",
        "estimated_finish_time": estimated_finish_time or "--",
        "avg_step_sec": round(avg_step_sec, 2),
        "history": history,
        "milestones": milestones,
        "recent_logs": clean_recent_logs[-35:],
    }


# -------------------------------------------------------------
# TRAINING CONTROLLER & LOGS
# -------------------------------------------------------------
@app.get("/api/train/logs")
def get_training_logs():
    history_file = Path("logs/training_history.json")
    saved_history = {}
    if history_file.exists():
        try:
            with open(history_file, "r") as f:
                saved_history = json.load(f)
        except Exception:
            pass

    live_pretrain = get_live_scaling_status()
    return {
        "live_status": state.training_status,
        "saved_history": saved_history,
        "active_pretrain": live_pretrain if live_pretrain else state.pretrain_status,
    }


def background_train_task(epochs: int, lr: float, batch_size: int, dataset_type: str = "librispeech"):
    state.is_training = True
    state.should_stop_training = False
    state.training_status["is_training"] = True
    state.training_status["total_epochs"] = epochs
    state.training_status["history"] = []

    print(f"[Training] Starting training on '{dataset_type}' for {epochs} epochs...")

    if dataset_type == "librispeech" and len(state.librispeech_samples) > 0:
        samples = [s for s in state.librispeech_samples[:300] if s["duration"] <= 6.0]
        val_samples = [s for s in state.librispeech_samples[300:350] if s["duration"] <= 6.0]
    else:
        from data.sample_dataset import generate_synthetic_asr_dataset
        samples, val_samples = generate_synthetic_asr_dataset("data/raw/synthetic", num_train=100, num_val=20)

    train_ds = AudioASRDataset(samples, state.tokenizer, target_sample_rate=16000, max_duration_s=6.0, augmenter=WaveformAugmenter())
    val_ds = AudioASRDataset(val_samples, state.tokenizer, target_sample_rate=16000, max_duration_s=6.0)
    collate_fn = AudioCollateFn(pad_token_id=state.tokenizer.pad_id)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(state.model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        if state.should_stop_training:
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

        # Val step
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
            "val_wer": state.training_status["val_wer"],
        })

        torch.save({
            "epoch": epoch,
            "model_state_dict": state.model.state_dict(),
            "config": state.model.config,
        }, "checkpoints/best_model.pt")

    state.is_training = False
    state.training_status["is_training"] = False


@app.post("/api/train/start")
def start_training(req: TrainStartRequest, background_tasks: BackgroundTasks):
    if state.is_training:
        return {"status": "already_running"}
    state.training_thread = threading.Thread(
        target=background_train_task,
        args=(req.epochs, req.lr, req.batch_size, req.dataset_type),
    )
    state.training_thread.start()
    return {"status": "started", "epochs": req.epochs, "dataset": req.dataset_type}


@app.post("/api/train/stop")
def stop_training():
    if not state.is_training:
        return {"status": "not_running"}
    state.should_stop_training = True
    return {"status": "stopping"}


@app.get("/api/train/status")
def get_training_status():
    return state.training_status


# -------------------------------------------------------------
# SOTA Benchmark Endpoints
# -------------------------------------------------------------
class SOTACompareRequest(BaseModel):
    sample_id: Optional[str] = None
    blank_penalty: float = 0.0


@app.get("/api/benchmark/sota/report")
def get_sota_benchmark_report():
    report_file = Path("logs/sota_benchmark_results.json")
    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "no_report_available"}


@app.get("/api/benchmark/sota/samples")
def get_sota_samples(limit: int = 30):
    samples = getattr(state, "test_clean_samples", [])
    if not samples and state.librispeech_samples:
        samples = state.librispeech_samples[:limit]
    return samples[:limit]


@app.post("/api/benchmark/sota/compare")
def compare_sota_sample(req: SOTACompareRequest):
    runner = state.get_sota_runner()
    audio_path = None
    ground_truth = None

    if req.sample_id and req.sample_id in state.librispeech_sample_map:
        sample = state.librispeech_sample_map[req.sample_id]
        audio_path = sample["audio_path"]
        ground_truth = sample.get("transcript", "")
    elif state.last_audio_tensor is not None:
        return runner.compare_audio(state.last_audio_tensor, ground_truth=ground_truth, blank_penalty=req.blank_penalty)
    else:
        samples = getattr(state, "test_clean_samples", [])
        if samples:
            sample = samples[0]
            audio_path = sample["audio_path"]
            ground_truth = sample.get("transcript", "")
        else:
            return {"error": "No audio sample available for comparison."}

    return runner.compare_audio(audio_path, ground_truth=ground_truth, blank_penalty=req.blank_penalty)


# -------------------------------------------------------------
# Piper SSL Pre-training Endpoints (0-Disk Streaming)
# -------------------------------------------------------------
class PretrainStartRequest(BaseModel):
    steps: int = 50
    batch_size: int = 4
    lr: float = 3e-4
    num_clusters: int = 100


def background_pretrain_worker(steps: int = 50, batch_size: int = 4, lr: float = 3e-4, num_clusters: int = 100):
    from src.data.streaming_piper import PiperStreamingDataset, collate_pretrain_batch
    from src.models.hubert_pretrain import HuBERTForPreTraining
    from torch.utils.data import DataLoader

    state.is_pretraining = True
    state.should_stop_pretraining = False
    state.pretrain_status["is_running"] = True
    state.pretrain_status["total_steps"] = steps

    comps = state.get_streaming_components()
    voice_mgr = comps["voice_mgr"]
    text_sampler = comps["text_sampler"]
    unit_extractor = comps["unit_extractor"]

    dataset = PiperStreamingDataset(
        voice_manager=voice_mgr,
        text_sampler=text_sampler,
        unit_extractor=unit_extractor,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_pretrain_batch,
    )

    config = HuBERTConfig(
        encoder_layers=4,
        encoder_heads=4,
        encoder_embed_dim=256,
        encoder_ffn_dim=1024,
    )
    pretrain_model = HuBERTForPreTraining(config=config, num_clusters=num_clusters).to(state.device)
    optimizer = torch.optim.AdamW(pretrain_model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(state.device == "cuda"))

    data_iter = iter(dataloader)
    total_audio_sec = state.pretrain_status.get("cumulative_audio_sec", 0.0)
    current_step = state.pretrain_status.get("step", 0)

    for i in range(steps):
        if state.should_stop_pretraining:
            break
        current_step += 1
        batch = next(data_iter)
        audio = batch["audio"].to(state.device)
        targets = batch["target_clusters"].to(state.device)
        batch_dur = sum(batch["durations"])
        total_audio_sec += batch_dur

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda" if state.device == "cuda" else "cpu", enabled=(state.device == "cuda")):
            outputs = pretrain_model(audio=audio, target_clusters=targets)
            loss = outputs["loss"]
            acc = outputs["accuracy"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(pretrain_model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_val = round(float(loss.item()), 4)
        acc_val = round(float(acc.item() * 100.0), 2) if acc is not None else 0.0
        saved_mb = round((total_audio_sec * 32000) / 1024 / 1024, 2)

        entry = {
            "step": current_step,
            "loss": loss_val,
            "masked_accuracy_pct": acc_val,
            "lr": lr,
            "cumulative_audio_sec": round(total_audio_sec, 2),
            "cumulative_audio_hours": round(total_audio_sec / 3600.0, 4),
            "disk_bytes_used": 0,
            "active_voices": batch["voices"],
            "sample_prompt": batch["texts"][0],
        }
        state.pretrain_status["step"] = current_step
        state.pretrain_status["loss"] = loss_val
        state.pretrain_status["accuracy"] = acc_val
        state.pretrain_status["cumulative_audio_sec"] = round(total_audio_sec, 2)
        state.pretrain_status["cumulative_audio_hours"] = round(total_audio_sec / 3600.0, 4)
        state.pretrain_status["disk_space_saved_mb"] = saved_mb
        state.pretrain_status["active_voice"] = batch["voices"][0]
        state.pretrain_status["sample_prompt"] = batch["texts"][0]
        state.pretrain_status["history"].append(entry)

        if current_step % 10 == 0 or i == steps - 1 or state.should_stop_pretraining:
            pretrain_model.save_pretrained_backbone("checkpoints/hubert_piper_pretrained.pt")
            with open("logs/pretrain_history.json", "w", encoding="utf-8") as f:
                json.dump({
                    "total_steps": current_step,
                    "cumulative_audio_hours": round(total_audio_sec / 3600.0, 4),
                    "disk_space_used_mb": 0.0,
                    "history": state.pretrain_status["history"],
                }, f, indent=2)

    state.is_pretraining = False
    state.pretrain_status["is_running"] = False


@app.get("/api/pretrain/status")
def get_pretrain_status(arch: Optional[str] = None, tier: Optional[str] = None):
    arch = arch or state.current_arch
    tier = tier or state.current_tier
    live = get_live_scaling_status(arch=arch, tier=tier)
    if live is not None:
        return live
    return state.pretrain_status


@app.post("/api/pretrain/start")
def start_pretrain(req: PretrainStartRequest):
    if state.is_pretraining:
        return {"status": "already_running"}
    state.pretrain_thread = threading.Thread(
        target=background_pretrain_worker,
        args=(req.steps, req.batch_size, req.lr, req.num_clusters),
        daemon=True,
    )
    state.pretrain_thread.start()
    return {"status": "started", "steps": req.steps}


@app.post("/api/pretrain/stop")
def stop_pretrain():
    if not state.is_pretraining:
        return {"status": "not_running"}
    state.should_stop_pretraining = True
    return {"status": "stopping"}


@app.get("/api/pretrain/preview")
def preview_streaming_sample(arch: Optional[str] = None):
    arch = arch or state.current_arch
    comps = state.get_streaming_components()
    voice_mgr = comps["voice_mgr"]
    text_sampler = comps["text_sampler"]

    text = text_sampler.sample_sentence()
    waveform, voice_name, dur = voice_mgr.synthesize_to_tensor_16k(text)

    # Encode audio into base64
    buf = io.BytesIO()
    sf.write(buf, waveform.numpy(), 16000, format="WAV")
    buf.seek(0)
    audio_b64 = base64.b64encode(buf.read()).decode("utf-8")

    if arch == "phono_hubert":
        if "phono_extractor" not in comps:
            comps["phono_extractor"] = PhonemeTargetExtractor()
        phono_ext = comps["phono_extractor"]
        target_dict = phono_ext.extract_targets(waveform=waveform, transcript=text)
        tokens = target_dict["targets"]
        token_strings = [phono_ext.tokenizer.id_to_phoneme.get(t, f"id_{t}") for t in tokens.tolist()]
        seq_len = len(token_strings)
        masked_indices = []
        if seq_len > 10:
            s1 = random.randint(1, max(1, seq_len // 2 - 2))
            masked_indices.extend(list(range(s1, min(seq_len, s1 + 4))))
        return {
            "arch": arch,
            "text": text,
            "voice": voice_name,
            "duration": round(dur, 2),
            "num_frames": seq_len,
            "tokens": token_strings[:75],
            "token_ids": tokens.tolist()[:75],
            "clusters": token_strings[:75],  # Fallback for frontend compatibility
            "masked_indices": masked_indices,
            "audio_base64": f"data:audio/wav;base64,{audio_b64}",
            "target_type": "phonemes",
        }
    else:
        unit_extractor = comps["unit_extractor"]
        _, cluster_labels = unit_extractor.get_cluster_labels(waveform)
        seq_len = len(cluster_labels)
        masked_indices = []
        if seq_len > 16:
            start1 = random.randint(3, seq_len // 2 - 4)
            masked_indices.extend(list(range(start1, min(seq_len, start1 + 8))))
            if seq_len > 35:
                start2 = random.randint(seq_len // 2 + 2, seq_len - 10)
                masked_indices.extend(list(range(start2, min(seq_len, start2 + 8))))
        return {
            "arch": arch,
            "text": text,
            "voice": voice_name,
            "duration": round(dur, 2),
            "num_frames": seq_len,
            "clusters": cluster_labels.tolist()[:75],
            "masked_indices": masked_indices,
            "audio_base64": f"data:audio/wav;base64,{audio_b64}",
            "target_type": "clusters",
        }


# Serve HTML Dashboard
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<h1>HuBERT Studio Backend Running.</h1>"
