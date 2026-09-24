"""Dataset loaders and collation for Audio ASR."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from src.data.tokenizer import CharacterTokenizer
from src.data.augmentations import WaveformAugmenter


class AudioASRDataset(Dataset):
    """Audio ASR Dataset loading WAV/FLAC files and text transcripts."""

    def __init__(
        self,
        samples: List[Dict[str, any]],
        tokenizer: CharacterTokenizer,
        target_sample_rate: int = 16000,
        augmenter: Optional[WaveformAugmenter] = None,
        normalize_audio: bool = True,
        max_duration_s: Optional[float] = None,
    ):
        """
        Args:
            samples: List of dicts, each with 'audio_path' (or 'waveform'), 'transcript', and optional 'id'.
            tokenizer: Tokenizer instance.
            target_sample_rate: Expected sample rate (default 16000).
            augmenter: Optional waveform augmenter.
            normalize_audio: If True, normalizes waveform to zero mean and unit variance.
            max_duration_s: Filter or clip samples longer than this duration.
        """
        self.tokenizer = tokenizer
        self.target_sample_rate = target_sample_rate
        self.augmenter = augmenter
        self.normalize_audio = normalize_audio
        self.max_duration_s = max_duration_s

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_audio(self, item: Dict[str, any]) -> torch.Tensor:
        if "waveform" in item and item["waveform"] is not None:
            waveform = item["waveform"]
            if isinstance(waveform, torch.Tensor):
                wav = waveform.clone()
            else:
                wav = torch.tensor(waveform, dtype=torch.float32)
            sr = item.get("sample_rate", self.target_sample_rate)
        else:
            path = item["audio_path"]
            wav_np, sr = sf.read(path, dtype="float32")
            wav = torch.from_numpy(wav_np)

        # Convert to mono if multi-channel
        if wav.ndim > 1:
            if wav.shape[0] > wav.shape[1]:  # (T, C)
                wav = wav.mean(dim=-1)
            else:  # (C, T)
                wav = wav.mean(dim=0)
        else:
            wav = wav.squeeze()

        # Resample if needed
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            wav = resampler(wav)

        # Max duration limit
        if self.max_duration_s is not None:
            max_len = int(self.max_duration_s * self.target_sample_rate)
            if wav.shape[-1] > max_len:
                wav = wav[:max_len]

        # Augmentation
        if self.augmenter is not None:
            wav = self.augmenter(wav)

        # Normalization
        if self.normalize_audio:
            mean = wav.mean()
            std = wav.std()
            if std > 1e-6:
                wav = (wav - mean) / std

        return wav

    def __getitem__(self, idx: int) -> Dict[str, any]:
        item = self.samples[idx]
        wav = self._load_audio(item)
        transcript = item["transcript"]
        token_ids = self.tokenizer.encode(transcript)

        return {
            "id": item.get("id", f"sample_{idx}"),
            "audio": wav,
            "audio_length": wav.shape[-1],
            "target": torch.tensor(token_ids, dtype=torch.long),
            "target_length": len(token_ids),
            "text": transcript,
        }


class AudioCollateFn:
    """Collate function for dynamic padding in batches."""

    def __init__(self, pad_token_id: int = 1):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, any]]) -> Dict[str, any]:
        batch_size = len(batch)
        audio_lengths = torch.tensor([item["audio_length"] for item in batch], dtype=torch.long)
        target_lengths = torch.tensor([item["target_length"] for item in batch], dtype=torch.long)

        max_audio_len = audio_lengths.max().item()
        max_target_len = target_lengths.max().item()

        # Padded audio tensor: (B, max_audio_len)
        padded_audio = torch.zeros(batch_size, max_audio_len, dtype=torch.float32)
        # Padded target tensor: (B, max_target_len)
        padded_targets = torch.full(
            (batch_size, max_target_len),
            self.pad_token_id,
            dtype=torch.long,
        )

        for i, item in enumerate(batch):
            audio = item["audio"]
            padded_audio[i, : audio.shape[-1]] = audio
            target = item["target"]
            padded_targets[i, : target.shape[-1]] = target

        return {
            "ids": [item["id"] for item in batch],
            "audio": padded_audio,
            "audio_lengths": audio_lengths,
            "targets": padded_targets,
            "target_lengths": target_lengths,
            "texts": [item["text"] for item in batch],
        }
