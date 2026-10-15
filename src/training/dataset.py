"""
Training Dataset and Collator for NaVox Speech Models.
Handles loading json manifests, tokenizing phonemes/characters, and balanced multilingual batch collation.
"""

# Import Modules
from dataclasses import dataclass
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.utils.data import Dataset, Sampler
import torchaudio


@dataclass
class AudioUtterance:
    """Dataclass holding metadata for a single audio sample."""
    audio_filepath: str
    duration: float
    text: str
    phonemes: str
    language: str
    dataset_name: str = "generic"


class NaVoxAudioDataset(Dataset):
    """
    Multilingual speech dataset reading paired audio files, phoneme strings, and text transcripts.
    """

    def __init__(
        self,
        manifest_path: str,
        sample_rate: int = 16000,
        max_duration: float = 14.0,
        min_duration: float = 0.5,
    ) -> None:
        """
        Initialize NaVoxAudioDataset.

        Args:
            manifest_path (str): Path to JSON manifest file.
            sample_rate (int): Target audio sampling rate (16000 Hz).
            max_duration (float): Maximum audio duration in seconds.
            min_duration (float): Minimum audio duration in seconds.
        """
        super().__init__()
        self.sample_rate: int = sample_rate
        self.max_duration: float = max_duration
        self.min_duration: float = min_duration
        self.samples: List[AudioUtterance] = []

        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Manifest not found: `{manifest_path}`")

        # Load JSON manifest
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            dur = float(item.get("duration", 0.0))
            if dur < min_duration or dur > max_duration:
                continue
            
            # Extract utterance fields
            utt = AudioUtterance(
                audio_filepath=item.get("audio_filepath", item.get("audio_path", "")),
                duration=dur,
                text=item.get("text", item.get("transcript", "")),
                phonemes=item.get("phonemes", item.get("phoneme_tokens", "")),
                language=item.get("language", item.get("lang", "en")).upper(),
                dataset_name=item.get("dataset", "generic"),
            )
            if os.path.isfile(utt.audio_filepath):
                self.samples.append(utt)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        utt = self.samples[idx]
        waveform, sr = torchaudio.load(utt.audio_filepath)
        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        wav = waveform.squeeze(0)

        # Ambient Noise & Silence Augmentation for Live Mic Robustness:
        # Prepend 0.1s to 0.4s of low-level ambient noise/silence to teach the model not to hallucinate on pauses
        if random.random() < 0.4:
            lead_silence_len = int(self.sample_rate * random.uniform(0.1, 0.4))
            trail_silence_len = int(self.sample_rate * random.uniform(0.1, 0.3))
            noise_scale = random.uniform(1e-5, 3e-3)
            lead_noise = torch.randn(lead_silence_len) * noise_scale
            trail_noise = torch.randn(trail_silence_len) * noise_scale
            wav = torch.cat([lead_noise, wav, trail_noise], dim=0)

        return {
            "waveform": wav,  # [T_samples]
            "text": utt.text,
            "phonemes": utt.phonemes,
            "language": utt.language,
            "duration": utt.duration,
        }


class NaVoxCollator:
    """
    Collator that dynamically pads audio waveforms and tokenizes phonemes for CTC training.
    """

    def __init__(
        self,
        phoneme_tokenizer: Any,
        pad_id: int = 0,
        blank_id: int = 1,
        space_id: int = 8,
    ) -> None:
        self.tokenizer: Any = phoneme_tokenizer
        self.pad_id: int = pad_id
        self.blank_id: int = blank_id
        self.space_id: int = space_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_size = len(batch)
        waveforms = [item["waveform"] for item in batch]
        audio_lengths = torch.tensor([len(w) for w in waveforms], dtype=torch.long)
        max_audio_len = audio_lengths.max().item()

        # Pad audio: [B, max_audio_len]
        padded_audio = torch.zeros(batch_size, max_audio_len, dtype=torch.float32)
        for i, w in enumerate(waveforms):
            padded_audio[i, :len(w)] = w

        # Encode phonemes
        tokenized_ph: List[List[int]] = []
        for item in batch:
            ph_str = item["phonemes"]
            if hasattr(self.tokenizer, "encode"):
                t_ids = self.tokenizer.encode(ph_str)
            else:
                # Fallback char/word level split
                t_ids = [ord(c) % 110 + 2 for c in ph_str]
            tokenized_ph.append(t_ids)

        target_lengths = torch.tensor([len(t) for t in tokenized_ph], dtype=torch.long)
        max_target_len = max(1, target_lengths.max().item())
        padded_targets = torch.full((batch_size, max_target_len), self.pad_id, dtype=torch.long)
        for i, t in enumerate(tokenized_ph):
            padded_targets[i, :len(t)] = torch.tensor(t, dtype=torch.long)

        return {
            "audio": padded_audio,
            "audio_lengths": audio_lengths,
            "targets": padded_targets,
            "target_lengths": target_lengths,
            "texts": [item["text"] for item in batch],
            "phoneme_strings": [item["phonemes"] for item in batch],
            "languages": [item["language"] for item in batch],
        }
