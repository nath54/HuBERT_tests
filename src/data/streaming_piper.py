"""On-the-fly streaming speech generator and acoustic unit extractor adapting MADGen and Piper-TTS.

Generates infinite diverse synthetic multi-speaker speech entirely in RAM (0 bytes disk space used)
and assigns k-means acoustic unit pseudo-labels for HuBERT self-supervised pre-training.
"""

import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from piper.config import SynthesisConfig
from piper.voice import PiperVoice
from sklearn.cluster import MiniBatchKMeans


# Path constants pointing to MADGen resources
MADGEN_ROOT = Path("/home/nathan/github/MADGen")
PIPER_VOICES_DIR = MADGEN_ROOT / "data" / "piper_voices"
DICTIONARY_PATH = MADGEN_ROOT / "data" / "dictionaries" / "words_100k.txt"


class ProceduralTextSampler:
    """Procedural text generator combining MADGen 100k dictionary and conversational patterns."""

    def __init__(self, dictionary_path: Path = DICTIONARY_PATH):
        self.words: List[str] = []
        self._load_dictionary(dictionary_path)

        # Diverse conversational sentence frames
        self.templates = [
            "We should investigate the relationship between {w1} and {w2} during the next experiment.",
            "The unexpected {w1} caused a significant shift in the {w2} measurement yesterday.",
            "Can you verify whether the {w1} system is properly connected to the {w2}?",
            "I was reviewing the latest findings regarding {w1} and found a remarkable {w2}.",
            "Before we finalize the {w1}, let us carefully examine the {w2} data.",
            "The technician reported that the {w1} demonstrated unusual stability near the {w2}.",
            "After hours of testing, the team discovered that {w1} directly influences {w2}.",
            "Please ensure that the {w1} is thoroughly calibrated before processing the {w2}.",
            "In modern acoustic engineering, {w1} plays an essential role alongside {w2}.",
            "The sudden change in {w1} led to an intriguing discussion about {w2}.",
            "It is evident that {w1} provides substantial advantages when paired with {w2}.",
            "We noticed that {w1} and {w2} exhibit strong structural correlation under these conditions.",
            "Have you considered how {w1} might affect the performance of the {w2}?",
            "The primary objective is to enhance {w1} while maintaining strict control over {w2}."
        ]

        # Natural conversational bank from MADGen
        self.dialogue_bank = [
            "I was thinking about the acoustic project we discussed yesterday.",
            "The weather has been unusually pleasant for this time of year.",
            "Have you seen the latest updates on the self supervised speech model?",
            "We definitely need to schedule another brainstorming session soon.",
            "That sounds like a very promising direction to explore in deep learning.",
            "I read an interesting paper about acoustic unit clustering earlier.",
            "Let us review the experimental training data before making a decision.",
            "Wait, hold on a moment, let me verify the transformer layers first.",
            "Actually, before you continue, can I clarify one important point?",
            "Notice how the convolutional filterbanks extract localized representations."
        ]

    def _load_dictionary(self, path: Path):
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                self.words = [line.strip().lower() for line in f if len(line.strip()) >= 4]
            print(f"[TextSampler] Loaded {len(self.words):,} words from {path.name}")
        else:
            self.words = [
                "acoustic", "waveform", "frequency", "resonance", "spectrum", "convolution",
                "representation", "attention", "transformer", "phoneme", "utterance", "gradient",
                "projection", "dimension", "sequence", "encoder", "decoder", "latency", "entropy"
            ]
            print("[TextSampler] Warning: Dictionary file not found; using fallback words.")

    def sample_sentence(self) -> str:
        """Sample a grammatically varied, phonetically rich sentence."""
        mode = random.random()
        if mode < 0.25 and self.dialogue_bank:
            return random.choice(self.dialogue_bank)
        
        template = random.choice(self.templates)
        w1 = random.choice(self.words)
        w2 = random.choice(self.words)
        return template.format(w1=w1, w2=w2)


class PiperVoiceManager:
    """Manages loaded Piper neural voice models with in-memory caching."""

    def __init__(self, voices_dir: Path = PIPER_VOICES_DIR, max_cached_voices: int = 8):
        self.voices_dir = voices_dir
        self.max_cached = max_cached_voices
        self.voice_models: List[Path] = []
        self.voice_cache: Dict[str, PiperVoice] = {}
        self.resamplers: Dict[int, T.Resample] = {}

        self._discover_voices()

    def _discover_voices(self):
        if self.voices_dir.exists():
            import json
            import piper.phoneme_ids
            # Suppress missing phoneme warning spam
            piper.phoneme_ids._LOGGER.setLevel(logging.ERROR)

            all_models = sorted(list(self.voices_dir.rglob("*.onnx")))

            # Load blocked voices configuration
            blocked_names = set()
            blocklist_path = Path("config/blocked_voices.json")
            if blocklist_path.exists():
                try:
                    with open(blocklist_path, "r", encoding="utf-8") as f:
                        b_data = json.load(f)
                        blocked_names = set(b_data.get("blocked_voice_names", []))
                except Exception:
                    pass

            # Filter out blocked voices with missing phonemes
            clean_models = [m for m in all_models if f"{m.parent.name}_{m.stem}" not in blocked_names]

            self.en_voices = [m for m in clean_models if "en_US" in str(m) or "en_GB" in str(m)]
            self.fr_voices = [m for m in clean_models if "fr_FR" in str(m)]
            self.voice_models = self.en_voices + self.fr_voices
            if not self.voice_models:
                self.voice_models = clean_models

            blocked_count = len(all_models) - len(clean_models)
            print(f"[PiperManager] Loaded {len(self.voice_models)} verified clean voices ({len(self.en_voices)} EN, {len(self.fr_voices)} FR). Blocked {blocked_count} defective models.")
        else:
            print(f"[PiperManager] Warning: Voices directory {self.voices_dir} not found.")

    def get_random_voice(self, lang: Optional[str] = None) -> Tuple[PiperVoice, str]:
        """Retrieve a cached or newly loaded PiperVoice instance matching language."""
        if not self.voice_models:
            raise RuntimeError(f"No Piper models found in {self.voices_dir}")

        if lang == "fr" and self.fr_voices:
            pool = self.fr_voices
        elif lang == "en" and self.en_voices:
            pool = self.en_voices
        else:
            pool = self.voice_models

        chosen_model_path = random.choice(pool)
        voice_name = f"{chosen_model_path.parent.name}_{chosen_model_path.stem}"

        if voice_name in self.voice_cache:
            return self.voice_cache[voice_name], voice_name

        # Manage cache capacity
        if len(self.voice_cache) >= self.max_cached:
            evict_key = next(iter(self.voice_cache))
            del self.voice_cache[evict_key]

        config_path = chosen_model_path.with_suffix(".onnx.json")
        if not config_path.exists():
            config_path = Path(f"{chosen_model_path}.json")

        loaded_voice = PiperVoice.load(
            str(chosen_model_path),
            config_path=str(config_path) if config_path.exists() else None,
        )
        self.voice_cache[voice_name] = loaded_voice
        return loaded_voice, voice_name

    def synthesize_to_tensor_16k(self, text: str, lang: Optional[str] = None) -> Tuple[torch.Tensor, str, float]:
        """Synthesize text entirely in RAM and return a 16 kHz float32 PyTorch tensor."""
        voice, voice_name = self.get_random_voice(lang=lang)

        # Randomize speaking rate and prosody
        length_scale = random.uniform(0.90, 1.12)
        noise_scale = random.uniform(0.55, 0.75)
        syn_config = SynthesisConfig(
            length_scale=length_scale,
            noise_scale=noise_scale,
            volume=1.0,
        )

        chunks = []
        source_sr = 22050
        for chunk in voice.synthesize(text, syn_config=syn_config):
            source_sr = chunk.sample_rate
            chunks.append(chunk.audio_float_array)

        if not chunks:
            # Fallback silence
            return torch.zeros(16000), voice_name, 1.0

        audio_np = np.concatenate(chunks).astype(np.float32)
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)  # (1, T_raw)

        # Resample to 16,000 Hz if needed
        if source_sr != 16000:
            if source_sr not in self.resamplers:
                self.resamplers[source_sr] = T.Resample(orig_freq=source_sr, new_freq=16000)
            audio_tensor = self.resamplers[source_sr](audio_tensor)

        waveform = audio_tensor.squeeze(0)  # (T_16k,)
        duration = len(waveform) / 16000.0

        # Normalize amplitude safely
        max_val = torch.max(torch.abs(waveform))
        if max_val > 1e-4:
            waveform = waveform / max_val * 0.95

        return waveform, voice_name, duration


class AcousticUnitExtractor:
    """Extracts 39-dim MFCCs + deltas and computes k-means acoustic unit pseudo-labels."""

    def __init__(self, num_clusters: int = 100, hop_length: int = 320):
        self.num_clusters = num_clusters
        self.hop_length = hop_length

        self.mfcc_transform = T.MFCC(
            sample_rate=16000,
            n_mfcc=13,
            melkwargs={"n_fft": 400, "hop_length": hop_length, "n_mels": 40},
        )
        self.kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            random_state=42,
            batch_size=512,
            n_init="auto",
        )
        self.is_fitted = False
        self.collected_features: List[np.ndarray] = []

    def compute_mfcc_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute 39-dimensional acoustic features [MFCC, Delta, Delta-Delta].
        
        Args:
            waveform: (T_samples,) or (1, T_samples).
        Returns:
            Features of shape (T_frames, 39).
        """
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # mfcc: (1, 13, T_frames)
        mfcc = self.mfcc_transform(waveform)
        delta1 = torchaudio.functional.compute_deltas(mfcc)
        delta2 = torchaudio.functional.compute_deltas(delta1)

        feats = torch.cat([mfcc, delta1, delta2], dim=1)  # (1, 39, T_frames)
        return feats.squeeze(0).transpose(0, 1)  # (T_frames, 39)

    def warm_start_clusters(self, initial_features_list: List[torch.Tensor]):
        """Fit initial k-means cluster centers using representative audio samples."""
        all_feats = torch.cat(initial_features_list, dim=0).cpu().numpy()
        print(f"[UnitExtractor] Fitting MiniBatchKMeans on {all_feats.shape[0]:,} acoustic frames...")
        self.kmeans.fit(all_feats)
        self.is_fitted = True
        print(f"[UnitExtractor] K-Means codebook initialized with {self.num_clusters} acoustic clusters.")

    def get_cluster_labels(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cluster pseudo-labels for an audio waveform.
        
        Returns:
            feats: (T_frames, 39)
            labels: (T_frames,) containing cluster indices in [0, num_clusters - 1]
        """
        feats = self.compute_mfcc_features(waveform)  # (T_frames, 39)
        feats_np = feats.cpu().numpy()

        if not self.is_fitted:
            # Online incremental fit
            self.kmeans.partial_fit(feats_np)
            if self.kmeans.n_steps_ > 10:
                self.is_fitted = True

        labels_np = self.kmeans.predict(feats_np)
        labels_tensor = torch.from_numpy(labels_np).long()
        return feats, labels_tensor


class PiperStreamingDataset(torch.utils.data.IterableDataset):
    """PyTorch IterableDataset synthesizing audio and acoustic units on-the-fly in RAM.
    
    Zero disk space used: samples are synthesized in-memory and yielded directly to DataLoader.
    """

    def __init__(
        self,
        voice_manager: Optional[PiperVoiceManager] = None,
        text_sampler: Optional[ProceduralTextSampler] = None,
        unit_extractor: Optional[AcousticUnitExtractor] = None,
        max_duration_sec: float = 12.0,
        min_duration_sec: float = 1.0,
    ):
        super().__init__()
        self.voice_manager = voice_manager or PiperVoiceManager()
        self.text_sampler = text_sampler or ProceduralTextSampler()
        self.unit_extractor = unit_extractor or AcousticUnitExtractor()
        self.max_duration_sec = max_duration_sec
        self.min_duration_sec = min_duration_sec

    def __iter__(self) -> Iterator[Dict[str, any]]:
        while True:
            res = self.text_sampler.sample_sentence()
            if isinstance(res, tuple):
                text, lang = res
            else:
                text, lang = res, "en"

            try:
                waveform, voice_name, dur = self.voice_manager.synthesize_to_tensor_16k(text, lang=lang)
            except Exception as e:
                continue

            if dur < self.min_duration_sec or dur > self.max_duration_sec:
                continue

            _, cluster_labels = self.unit_extractor.get_cluster_labels(waveform)

            yield {
                "audio": waveform,
                "cluster_labels": cluster_labels,
                "duration": dur,
                "text": text,
                "voice_name": voice_name,
            }


def collate_pretrain_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate variable-length audio waveforms and cluster targets into padded batch tensors."""
    audio_list = [item["audio"] for item in batch]
    cluster_list = [item["cluster_labels"] for item in batch]
    durations = [item["duration"] for item in batch]
    texts = [item["text"] for item in batch]
    voices = [item["voice_name"] for item in batch]

    # Pad audio to max length
    audio_lengths = torch.tensor([len(a) for a in audio_list], dtype=torch.long)
    max_audio_len = int(audio_lengths.max().item())
    batch_size = len(audio_list)

    padded_audio = torch.zeros(batch_size, max_audio_len, dtype=torch.float32)
    for i, a in enumerate(audio_list):
        padded_audio[i, :len(a)] = a

    # Pad cluster labels to max frame length
    target_lengths = torch.tensor([len(c) for c in cluster_list], dtype=torch.long)
    max_target_len = int(target_lengths.max().item())

    # Use -100 as standard ignore_index for CrossEntropy
    padded_targets = torch.full((batch_size, max_target_len), -100, dtype=torch.long)
    for i, c in enumerate(cluster_list):
        padded_targets[i, :len(c)] = c

    return {
        "audio": padded_audio,
        "audio_lengths": audio_lengths,
        "target_clusters": padded_targets,
        "target_lengths": target_lengths,
        "durations": durations,
        "texts": texts,
        "voices": voices,
    }
