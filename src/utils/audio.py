"""Audio utilities for I/O and processing."""

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


def synthesize_spoken_word(
    word: str,
    duration_s: float = 0.8,
    sample_rate: int = 16000,
    f0: float = 130.0,
) -> torch.Tensor:
    """Generate a synthetic acoustic waveform for a word using harmonic-formant modeling.
    
    This provides realistic vowel formants and consonant bursts so acoustic
    models and XAI attributions have genuine acoustic structure to learn and explain.
    """
    num_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)

    # Fundamental frequency pitch contour with slight drop
    pitch_contour = f0 * (1.0 - 0.15 * (t / duration_s))
    phase = 2 * np.pi * np.cumsum(pitch_contour) / sample_rate

    # Harmonic glottal pulse excitation
    excitation = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase) + 0.1 * np.sin(4 * phase)

    # Phoneme formant table (F1, F2, F3 frequencies in Hz)
    vowel_formants = {
        "a": (800, 1200, 2500),
        "e": (500, 1800, 2600),
        "i": (300, 2300, 3000),
        "o": (500, 1000, 2500),
        "u": (350, 800, 2400),
    }

    # Extract dominant vowels or fall back
    vowels = [c for c in word.lower() if c in vowel_formants]
    f1, f2, f3 = vowel_formants[vowels[0]] if vowels else (500, 1500, 2500)

    # Resonant filtering (Formant resonators)
    def formant_filter(signal, center_freq, bandwidth=90):
        # 2nd-order resonator
        r = np.exp(-np.pi * bandwidth / sample_rate)
        theta = 2 * np.pi * center_freq / sample_rate
        a1 = -2 * r * np.cos(theta)
        a2 = r * r
        # Simple recursive filtering
        out = np.zeros_like(signal)
        for i in range(2, len(signal)):
            out[i] = signal[i] - a1 * out[i - 1] - a2 * out[i - 2]
        return out

    formant_signal = (
        0.5 * formant_filter(excitation, f1, 80)
        + 0.3 * formant_filter(excitation, f2, 100)
        + 0.15 * formant_filter(excitation, f3, 120)
    )

    # Consonant noise burst (for stops/fricatives like s, t, k, p)
    noise = np.random.normal(0, 0.05, num_samples)
    if any(c in "stkpfch" for c in word.lower()):
        consonant_onset = int(0.1 * num_samples)
        formant_signal[:consonant_onset] += noise[:consonant_onset] * 0.4

    # Smooth Tukey envelope (attack and decay)
    fade_len = int(0.08 * num_samples)
    envelope = np.ones(num_samples)
    envelope[:fade_len] = 0.5 * (1 - np.cos(np.pi * np.arange(fade_len) / fade_len))
    envelope[-fade_len:] = 0.5 * (1 - np.cos(np.pi * np.arange(fade_len)[::-1] / fade_len))

    audio = formant_signal * envelope
    # Normalize peak to 0.9
    peak = np.max(np.abs(audio)) + 1e-7
    audio = (audio / peak) * 0.9

    return torch.from_numpy(audio.astype(np.float32))
