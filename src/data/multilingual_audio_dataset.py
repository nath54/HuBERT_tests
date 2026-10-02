"""Balanced Multilingual Speech Dataset & Batch Sampler for Phono-V6.7.

Indexes and serves balanced 4-way speech batches across:
- 🇬🇧 English: LibriSpeech 100h / 960h
- 🇮🇹 Italian: OpenSLR MLS Italian
- 🇪🇸 Spanish: OpenSLR MLS Spanish
- 🇫🇷 French: OpenSLR MLS French (+ Mozilla Common Voice)

Prevents MoE router collapse and catastrophic forgetting by strictly interleaving
equal proportions of all 4 languages in every training batch.
"""

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, Sampler

from src.data.phoneme_tokenizer import PhonemeTokenizer
from src.data.roman_tokenizer import RomanCharTokenizer
from src.data.target_extractors import PhonemeTargetExtractor


@dataclass
class AudioUtterance:
    """Represents a single speech audio recording with transcription."""
    id: str
    audio_path: str
    transcript: str
    lang: str
    duration: float = 0.0


def load_librispeech_manifest(json_path: str, max_samples: Optional[int] = None) -> List[AudioUtterance]:
    """Load LibriSpeech English JSON manifest."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for item in data:
        p = item.get("audio_path", "")
        t = item.get("transcript", "").strip()
        if p and t and os.path.exists(p):
            samples.append(
                AudioUtterance(
                    id=item.get("id", os.path.basename(p)),
                    audio_path=p,
                    transcript=t,
                    lang="en",
                    duration=item.get("duration", 0.0),
                )
            )
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def load_mls_manifest(mls_dir: str, lang: str, split: str = "train", max_samples: Optional[int] = None) -> List[AudioUtterance]:
    """Load OpenSLR MLS manifest (transcripts.txt + audio/<speaker>/<chapter>/<id>.flac)."""
    base = Path(mls_dir)
    transcripts_path = base / split / "transcripts.txt"
    audio_root = base / split / "audio"

    if not transcripts_path.exists():
        return []

    samples = []
    with open(transcripts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            uid, text = parts[0].strip(), parts[1].strip()
            if not text:
                continue

            id_parts = uid.split("_")
            if len(id_parts) >= 2:
                spk, chap = id_parts[0], id_parts[1]
                flac_path = audio_root / spk / chap / f"{uid}.flac"
                if flac_path.exists():
                    samples.append(
                        AudioUtterance(
                            id=uid,
                            audio_path=str(flac_path),
                            transcript=text,
                            lang=lang,
                        )
                    )
                    if max_samples and len(samples) >= max_samples:
                        break
    return samples


def load_common_voice_manifest(cv_dir: str, lang: str = "fr", split: str = "train", max_samples: Optional[int] = None) -> List[AudioUtterance]:
    """Load Mozilla Common Voice TSV manifest."""
    base = Path(cv_dir)
    tsv_path = base / f"{split}.tsv"
    clips_root = base / "clips"

    if not tsv_path.exists():
        return []

    samples = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        try:
            path_idx = header.index("path")
            sent_idx = header.index("sentence")
        except ValueError:
            return []

        for line in f:
            cols = line.strip().split("\t")
            if len(cols) <= max(path_idx, sent_idx):
                continue
            clip_name = cols[path_idx].strip()
            text = cols[sent_idx].strip()
            if not text:
                continue

            clip_path = clips_root / clip_name
            if clip_path.exists():
                samples.append(
                    AudioUtterance(
                        id=clip_name.replace(".mp3", ""),
                        audio_path=str(clip_path),
                        transcript=text,
                        lang=lang,
                    )
                )
                if max_samples and len(samples) >= max_samples:
                    break
    return samples


class MultilingualAudioDataset(Dataset):
    """Dataset providing joint audio waveforms, phoneme CTC targets, and Roman character words."""

    def __init__(
        self,
        utterances: List[AudioUtterance],
        roman_tokenizer: Optional[RomanCharTokenizer] = None,
        phoneme_extractor: Optional[PhonemeTargetExtractor] = None,
        max_duration_seconds: float = 12.0,
        sample_rate: int = 16000,
    ):
        self.utterances = utterances
        self.roman_tok = roman_tokenizer or RomanCharTokenizer()
        self.ph_extractor = phoneme_extractor or PhonemeTargetExtractor()
        self.max_duration = max_duration_seconds
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_seconds * sample_rate)

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.utterances[idx]
        text = self.roman_tok.normalize(item.transcript)

        # 1. Load Audio Waveform
        try:
            wav, sr = sf.read(item.audio_path, dtype="float32")
            if wav.ndim > 1:
                wav = np.mean(wav, axis=1)  # downmix to mono
            if sr != self.sample_rate:
                # Basic linear down/upsample if sample rate differs
                indices = np.round(np.arange(0, len(wav), sr / self.sample_rate)).astype(int)
                indices = indices[indices < len(wav)]
                wav = wav[indices]
            # Truncate to max samples
            if len(wav) > self.max_samples:
                wav = wav[: self.max_samples]
            audio_tensor = torch.from_numpy(wav).float()
        except Exception:
            # Fallback zero audio if file read fails
            audio_tensor = torch.zeros(self.sample_rate * 2, dtype=torch.float32)

        # 2. Extract Discrete Phoneme CTC Targets for Model 1
        targets_dict = self.ph_extractor.extract_targets(
            waveform=audio_tensor,
            text=text,
            lang=item.lang,
            transcript=text,
        )
        phoneme_targets = targets_dict["targets"]

        # 3. Extract Word Token Sequences for Model 2
        words = self.roman_tok.encode_words(text)
        if not words:
            words = [[self.roman_tok.unk_id, self.roman_tok.eow_id]]

        return {
            "id": item.id,
            "lang": item.lang,
            "audio": audio_tensor,
            "audio_length": len(audio_tensor),
            "transcript": text,
            "phoneme_targets": phoneme_targets,
            "phoneme_length": len(phoneme_targets),
            "words": words,
        }


class MultilingualBalancedBatchSampler(Sampler[List[int]]):
    """Yields balanced batches containing an equal number of samples per language.

    Example: Batch size 16 -> 4 English, 4 Italian, 4 Spanish, 4 French per step.
    """

    def __init__(
        self,
        utterances: List[AudioUtterance],
        batch_size: int = 16,
        languages: Optional[List[str]] = None,
        drop_last: bool = True,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.rng = random.Random(seed)

        self.languages = languages or ["en", "it", "es", "fr"]
        self.per_lang = max(1, batch_size // len(self.languages))

        # Index dataset by language
        self.lang_indices: Dict[str, List[int]] = {lang: [] for lang in self.languages}
        for idx, u in enumerate(utterances):
            if u.lang in self.lang_indices:
                self.lang_indices[u.lang].append(idx)

        # Determine number of batches: capped by largest pool with repetition, or average
        min_pool = min((len(idxs) for idxs in self.lang_indices.values()), default=1)
        max_pool = max((len(idxs) for idxs in self.lang_indices.values()), default=1)
        # We cycle smaller pools so training covers the full dataset
        self.num_batches = max_pool // self.per_lang if max_pool >= self.per_lang else 1

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle each language pool
        shuffled_pools = {}
        pool_iters = {}
        for lang in self.languages:
            pool = list(self.lang_indices[lang])
            if not pool:
                continue
            self.rng.shuffle(pool)
            shuffled_pools[lang] = pool
            pool_iters[lang] = 0

        for _ in range(self.num_batches):
            batch = []
            for lang in self.languages:
                pool = shuffled_pools.get(lang, [])
                if not pool:
                    continue
                k = self.per_lang
                start = pool_iters[lang]
                if start + k > len(pool):
                    self.rng.shuffle(pool)
                    start = 0
                    pool_iters[lang] = 0

                batch.extend(pool[start : start + k])
                pool_iters[lang] = start + k

            if len(batch) >= len(self.languages):
                self.rng.shuffle(batch)
                yield batch


class MultilingualSpeechCollator:
    """Collates variable-length audio and character word sequences into padded batches."""

    def __init__(
        self,
        roman_tokenizer: RomanCharTokenizer,
        max_bytes_per_word: int = 24,
    ):
        self.rom_tok = roman_tokenizer
        self.max_bytes_per_word = max_bytes_per_word

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(batch)
        K = self.max_bytes_per_word

        # 1. Collate Audio
        audio_lengths = torch.tensor([s["audio_length"] for s in batch], dtype=torch.long)
        max_audio_len = int(audio_lengths.max().item())
        padded_audio = torch.zeros(B, max_audio_len, dtype=torch.float32)
        for i, s in enumerate(batch):
            padded_audio[i, : s["audio_length"]] = s["audio"]

        # 2. Collate Phoneme Targets (for Model 1 CTC loss)
        phoneme_lengths = torch.tensor([s["phoneme_length"] for s in batch], dtype=torch.long)
        max_ph_len = max(1, int(phoneme_lengths.max().item()))
        padded_phonemes = torch.zeros(B, max_ph_len, dtype=torch.long)
        for i, s in enumerate(batch):
            ph = s["phoneme_targets"]
            padded_phonemes[i, : len(ph)] = ph if isinstance(ph, torch.Tensor) else torch.tensor(ph, dtype=torch.long)

        # 3. Collate Character Word Sequences (for Model 2 Decoder)
        num_words_list = [len(s["words"]) for s in batch]
        max_words = max(num_words_list)
        max_word_slots = max(max_words, 2)

        input_bytes = torch.zeros(B, max_word_slots, K, dtype=torch.long)
        target_bytes = torch.full((B, max_word_slots, K), -100, dtype=torch.long)
        path_targets = torch.zeros(B, max_word_slots, dtype=torch.long)  # 0: Special

        for b_idx, sample in enumerate(batch):
            words = sample["words"]
            for w_idx, w_tokens in enumerate(words):
                char_len = len(w_tokens) - 1 if (w_tokens and w_tokens[-1] == self.rom_tok.eow_id) else len(w_tokens)
                if char_len <= 3:
                    path_targets[b_idx, w_idx] = 1  # SHORT
                elif char_len <= 7:
                    path_targets[b_idx, w_idx] = 2  # MEDIUM
                else:
                    path_targets[b_idx, w_idx] = 3  # LONG

                seq_len = min(len(w_tokens), K - 1)
                input_bytes[b_idx, w_idx, 0] = self.rom_tok.bos_id
                input_bytes[b_idx, w_idx, 1 : seq_len + 1] = torch.tensor(w_tokens[:seq_len], dtype=torch.long)
                target_bytes[b_idx, w_idx, :seq_len] = torch.tensor(w_tokens[:seq_len], dtype=torch.long)

        return {
            "audio": padded_audio,
            "audio_lengths": audio_lengths,
            "phoneme_targets": padded_phonemes,
            "phoneme_lengths": phoneme_lengths,
            "num_words": torch.tensor(num_words_list, dtype=torch.long),
            "input_byte_ids": input_bytes,
            "target_byte_ids": target_bytes,
            "path_targets": path_targets,
            "transcripts": [s["transcript"] for s in batch],
            "languages": [s["lang"] for s in batch],
        }
