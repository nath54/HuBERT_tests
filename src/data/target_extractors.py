"""Modular Target Extractor interfaces for self-supervised and semi-supervised speech pre-training.

Allows plug-and-play switching between:
1. KMeansUnitExtractor: Acoustic MFCC cluster centroids (HuBERT baseline)
2. PhonemeTargetExtractor: Direct linguistic phoneme tokens with special continuation & silence tokens
"""

import abc
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from sklearn.cluster import MiniBatchKMeans

from src.data.phoneme_tokenizer import PhonemeTokenizer


class BaseTargetExtractor(abc.ABC):
    """Abstract Base Class for all speech pre-training target extractors."""

    @property
    @abc.abstractmethod
    def target_type(self) -> str:
        """Name of the target type (e.g. 'kmeans_cluster', 'phoneme_tokens')."""
        pass

    @property
    @abc.abstractmethod
    def vocab_size(self) -> int:
        """Target label space dimension (number of classes)."""
        pass

    @abc.abstractmethod
    def extract_targets(
        self,
        waveform: torch.Tensor,
        text: str = "",
        voice: any = None,
        lang: str = "en",
        transcript: str = "",
    ) -> Dict[str, torch.Tensor]:
        """Extract training targets for a single synthesized or recorded utterance.
        
        Returns dictionary containing:
            'targets': 1D Tensor of target token IDs (or frame-aligned IDs)
            'target_lengths': Tensor scalar of valid length
        """
        pass


class KMeansUnitExtractor(BaseTargetExtractor):
    """Acoustic MFCC feature extractor with MiniBatchKMeans clustering."""

    def __init__(self, num_clusters: int = 100, sample_rate: int = 16000):
        self.num_clusters = num_clusters
        self.sample_rate = sample_rate
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=13,
            melkwargs={"n_fft": 400, "hop_length": 320, "n_mels": 40, "center": False},
        )
        self.kmeans = MiniBatchKMeans(
            n_clusters=num_clusters,
            batch_size=1024,
            random_state=42,
            n_init="auto",
        )
        self.is_fitted = False

    @property
    def target_type(self) -> str:
        return "kmeans_cluster"

    @property
    def vocab_size(self) -> int:
        return self.num_clusters

    def compute_mfcc_features(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        mfcc = self.mfcc_transform(waveform)
        delta1 = torchaudio.functional.compute_deltas(mfcc)
        delta2 = torchaudio.functional.compute_deltas(delta1)
        feats = torch.cat([mfcc, delta1, delta2], dim=1)  # (1, 39, T_frames)
        return feats.squeeze(0).transpose(0, 1)  # (T_frames, 39)

    def extract_targets(
        self,
        waveform: torch.Tensor,
        text: str = "",
        voice: any = None,
        lang: str = "en",
    ) -> Dict[str, torch.Tensor]:
        feats = self.compute_mfcc_features(waveform)
        feats_np = feats.cpu().numpy()
        if not self.is_fitted:
            self.kmeans.partial_fit(feats_np)
            if self.kmeans.n_steps_ > 10:
                self.is_fitted = True
        labels_np = self.kmeans.predict(feats_np)
        labels = torch.from_numpy(labels_np).long()
        t_len = torch.tensor(len(labels), dtype=torch.long)
        return {
            "targets": labels,
            "target_lengths": t_len,
            "frame_targets": labels,
            "frame_lengths": t_len,
        }


class PhonemeTargetExtractor(BaseTargetExtractor):
    """Direct Phoneme Target Extractor using specialized linguistic tokens.
    
    Generates exact phoneme sequences with support for:
    - <silence> for leading/trailing silence
    - <same_phoneme_than_last_one> for sustained frame modeling
    - <eos> for sequence termination
    - Frame-level interpolated targets for frame-synchronous SSL models
    """

    def __init__(self, tokenizer: Optional[PhonemeTokenizer] = None):
        self.tokenizer = tokenizer or PhonemeTokenizer()

    @property
    def target_type(self) -> str:
        return "phoneme_tokens"

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def extract_targets(
        self,
        waveform: torch.Tensor,
        text: str = "",
        voice: any = None,
        lang: str = "en",
        transcript: str = "",
    ) -> Dict[str, torch.Tensor]:
        if not text and transcript:
            text = transcript
        # 1. Phonemize text
        if voice is not None and hasattr(voice, "phonemize"):
            try:
                phoneme_sentences = voice.phonemize(text)
                # Flatten nested phoneme list
                flat_phonemes = []
                for s in phoneme_sentences:
                    flat_phonemes.extend(s)
            except Exception:
                flat_phonemes = list(text)
        else:
            flat_phonemes = list(text)

        # 2. Encode to token IDs
        token_ids = self.tokenizer.encode(flat_phonemes, add_eos=True)

        # 3. Build frame-synchronous phoneme targets (16kHz / 320 hop = 50Hz frames)
        # Filter non-acoustic diacritics from CTC sequence targets (preserve eos token at end)
        diacritic_ids = {self.tokenizer.token_to_id.get(ch) for ch in ("ˈ", "ˌ", "ː", "ˑ") if ch in self.tokenizer.token_to_id}
        ctc_token_ids = [t for t in token_ids if t not in diacritic_ids]
        if not ctc_token_ids:
            ctc_token_ids = [self.tokenizer.silence_token_id, self.tokenizer.eos_token_id]

        # 3. Build frame-synchronous phoneme targets (16kHz / 320 hop = 50Hz frames)
        num_samples = waveform.shape[-1] if waveform.ndim > 0 else 1
        num_frames = max(1, num_samples // 320)

        # For frame targets: also filter spaces/whitespace (spaces are orthographic, not acoustic temporal frames)
        non_frame_tokens = {"ˈ", "ˌ", "ː", "ˑ", " ", "-", "\n", "\t"}
        non_frame_ids = {self.tokenizer.token_to_id.get(ch) for ch in non_frame_tokens if ch in self.tokenizer.token_to_id}
        speech_token_ids = [t for t in token_ids if t != self.tokenizer.eos_token_id and t not in non_frame_ids]
        if not speech_token_ids:
            speech_token_ids = [self.tokenizer.silence_token_id]

        silence_pad = min(2, max(0, num_frames // 10))
        speech_frames = max(1, num_frames - 2 * silence_pad)

        frame_targets = torch.full((num_frames,), self.tokenizer.silence_token_id, dtype=torch.long)
        L = len(speech_token_ids)
        for j in range(speech_frames):
            idx = min(L - 1, (j * L) // speech_frames)
            frame_targets[silence_pad + j] = speech_token_ids[idx]

        if num_frames > 2 and silence_pad > 0:
            frame_targets[-1] = self.tokenizer.silence_token_id

        return {
            "targets": torch.tensor(ctc_token_ids, dtype=torch.long),
            "target_lengths": torch.tensor(len(ctc_token_ids), dtype=torch.long),
            "frame_targets": frame_targets,
            "frame_lengths": torch.tensor(num_frames, dtype=torch.long),
            "phoneme_str": self.tokenizer.decode(ctc_token_ids, skip_special=True),
        }
