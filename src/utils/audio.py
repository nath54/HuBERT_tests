"""Audio utilities for I/O, processing, and sequential acoustic speech synthesis."""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
import soundfile as sf
import torch


def load_audio(path: Union[str, Path], target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """Load an audio file as a mono float tensor."""
    wav_np, sr = sf.read(str(path), dtype="float32")
    wav = torch.from_numpy(wav_np)
    if wav.ndim > 1:
        wav = wav.mean(dim=-1)
    return wav, sr


def save_audio(waveform: torch.Tensor, path: Union[str, Path], sample_rate: int = 16000):
    """Save a waveform tensor as a WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav_np = waveform.detach().cpu().squeeze().numpy()
    sf.write(str(path), wav_np, sample_rate)


def synthesize_phoneme_segment(
    char: str,
    duration_s: float = 0.10,
    sample_rate: int = 16000,
    f0: float = 130.0,
) -> np.ndarray:
    """Generate distinct acoustic waveform for a single phoneme/character."""
    num_samples = max(int(duration_s * sample_rate), 64)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)

    # Voiced harmonic resonator
    def get_voiced(f1: float, f2: float, f3: float = 2500.0) -> np.ndarray:
        pitch = f0 * (1.0 - 0.1 * (t / max(duration_s, 1e-4)))
        phase = 2 * np.pi * np.cumsum(pitch) / sample_rate
        glottal = (
            np.sin(phase)
            + 0.5 * np.sin(2 * phase)
            + 0.25 * np.sin(3 * phase)
            + 0.12 * np.sin(4 * phase)
        )
        # Resonant formant filter simulation
        res1 = np.sin(2 * np.pi * f1 * t)
        res2 = np.sin(2 * np.pi * f2 * t)
        res3 = np.sin(2 * np.pi * f3 * t)
        audio = glottal * (0.55 * res1 + 0.30 * res2 + 0.15 * res3)
        return audio.astype(np.float32)

    # Fricative noise generator
    def get_noise(center_freq: float, bandwidth: float = 1000.0) -> np.ndarray:
        white = np.random.normal(0, 0.4, num_samples).astype(np.float32)
        mod = np.sin(2 * np.pi * center_freq * t)
        return (white * mod).astype(np.float32)

    c = char.lower()

    # Vowels with distinct Formants (F1, F2)
    vowel_formants = {
        "a": (800.0, 1250.0),
        "e": (500.0, 1850.0),
        "i": (300.0, 2300.0),
        "o": (500.0, 950.0),
        "u": (350.0, 800.0),
        "y": (320.0, 2100.0),
    }

    if c in vowel_formants:
        f1, f2 = vowel_formants[c]
        sig = get_voiced(f1, f2)
    elif c == "s":
        sig = get_noise(5800.0, 1500.0) * 0.9  # High frequency hiss
    elif c == "f":
        sig = get_noise(3200.0, 1200.0) * 0.6  # Broadband soft noise
    elif c == "h":
        sig = get_noise(1800.0, 900.0) * 0.65  # Aspiration breath
    elif c in "ptk":
        # Unvoiced plosive: brief silence followed by sharp burst
        sig = np.zeros(num_samples, dtype=np.float32)
        burst_len = int(0.35 * num_samples)
        freq = 4800.0 if c == "t" else (2200.0 if c == "k" else 750.0)
        sig[-burst_len:] = get_noise(freq)[:burst_len] * 1.3
    elif c in "bdg":
        # Voiced plosive: low-frequency voice bar + burst
        sig = np.sin(2 * np.pi * (f0 * 0.8) * t).astype(np.float32) * 0.25
        burst_len = int(0.35 * num_samples)
        freq = 3200.0 if c == "d" else (1600.0 if c == "g" else 550.0)
        sig[-burst_len:] += get_noise(freq)[:burst_len] * 0.8
    elif c in "mn":
        # Nasals: low F1 resonance + sharp high-frequency attenuation
        sig = get_voiced(280.0, 1600.0 if c == "n" else 1050.0) * 0.75
    elif c in "lrw":
        # Liquids and glides
        sig = get_voiced(350.0, 1300.0 if c == "r" else (700.0 if c == "w" else 1150.0))
    elif c == " ":
        # Word boundary pause
        sig = np.random.normal(0, 0.005, num_samples).astype(np.float32)
    else:
        sig = get_voiced(450.0, 1400.0) * 0.5

    # Smooth attack and decay envelope to prevent clicks
    fade = min(int(0.012 * sample_rate), num_samples // 4)
    if fade > 0:
        env = np.ones(num_samples, dtype=np.float32)
        env[:fade] = np.linspace(0.0, 1.0, fade)
        env[-fade:] = np.linspace(1.0, 0.0, fade)
        sig = sig * env

    return sig


def synthesize_spoken_word(
    word: str,
    duration_s: float = None,
    sample_rate: int = 16000,
    f0: float = 135.0,
) -> torch.Tensor:
    """Generate a realistic sequential acoustic speech waveform for words or sentences.
    
    Each character is synthesized sequentially in time with its own distinct acoustic
    formants and frequency bursts, allowing the speech recognition model to resolve
    and explain individual phonemes across the audio timeline.
    """
    clean_text = "".join([c for c in word.lower() if c.isalpha() or c in " '"]).strip()
    if not clean_text:
        clean_text = "speech"

    # Dynamic duration based on sentence/word length
    # Short plosives: ~60ms, Vowels/Fricatives: ~90-110ms
    chunks = []
    # Lead-in silence
    chunks.append(np.zeros(int(0.04 * sample_rate), dtype=np.float32))

    for char in clean_text:
        if char in "ptkbdg'":
            char_dur = 0.065
        elif char == " ":
            char_dur = 0.080
        elif char in "aeiouy":
            char_dur = 0.110
        else:
            char_dur = 0.085
        chunks.append(synthesize_phoneme_segment(char, duration_s=char_dur, sample_rate=sample_rate, f0=f0))

    # Trailing silence
    chunks.append(np.zeros(int(0.04 * sample_rate), dtype=np.float32))

    audio = np.concatenate(chunks)

    # Normalize amplitude
    peak = np.max(np.abs(audio)) + 1e-6
    audio = (audio / peak) * 0.90

    return torch.from_numpy(audio.astype(np.float32))
