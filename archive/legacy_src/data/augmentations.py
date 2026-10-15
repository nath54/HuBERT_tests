"""Audio augmentations for robust ASR training."""

import random
import torch
import torch.nn as nn


class WaveformAugmenter:
    """Waveform-level data augmentations."""

    def __init__(
        self,
        noise_prob: float = 0.3,
        noise_snr_db: float = 20.0,
        gain_prob: float = 0.3,
        min_gain: float = 0.8,
        max_gain: float = 1.2,
        time_mask_prob: float = 0.2,
        max_mask_duration_s: float = 0.2,
        sample_rate: int = 16000,
    ):
        self.noise_prob = noise_prob
        self.noise_snr_db = noise_snr_db
        self.gain_prob = gain_prob
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.time_mask_prob = time_mask_prob
        self.max_mask_samples = int(max_mask_duration_s * sample_rate)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply augmentations with given probabilities.
        
        Args:
            waveform: Tensor of shape (T,) or (1, T).
        """
        wav = waveform.clone()

        # Gain perturbation
        if random.random() < self.gain_prob:
            gain = random.uniform(self.min_gain, self.max_gain)
            wav = wav * gain

        # Additive white Gaussian noise
        if random.random() < self.noise_prob:
            signal_power = torch.mean(wav ** 2) + 1e-8
            noise_power = signal_power / (10 ** (self.noise_snr_db / 10))
            noise = torch.randn_like(wav) * torch.sqrt(noise_power)
            wav = wav + noise

        # Time masking (zeroing out random chunks of audio)
        if random.random() < self.time_mask_prob and wav.shape[-1] > self.max_mask_samples:
            mask_len = random.randint(100, self.max_mask_samples)
            start_idx = random.randint(0, wav.shape[-1] - mask_len)
            wav[..., start_idx : start_idx + mask_len] = 0.0

        return wav
