"""Unit tests for XAI components (Captum, NAPS, Probing)."""

import unittest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.config import HuBERTConfig
from src.models.hubert_asr import HuBERTForCTC
from src.data.tokenizer import CharacterTokenizer
from src.data.dataset import AudioASRDataset
from src.xai.captum_gradients import AudioGradientExplainer
from src.xai.naps import NeuronActivationProfiler, ActivationPatcher
from src.xai.probes import LayerwiseProbeTrainer


class TestXAIComponents(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.config = HuBERTConfig(
            encoder_embed_dim=64,
            encoder_layers=2,
            encoder_heads=2,
            encoder_ffn_dim=128,
            pos_conv_kernel=16,
            vocab_size=32,
        )
        self.model = HuBERTForCTC(self.config).to(self.device)
        self.tokenizer = CharacterTokenizer()

        self.samples = [
            {"id": f"s_{i}", "waveform": torch.randn(8000), "transcript": "test", "sample_rate": 16000}
            for i in range(4)
        ]
        self.dataset = AudioASRDataset(self.samples, self.tokenizer)

    def test_captum_integrated_gradients(self):
        explainer = AudioGradientExplainer(self.model, self.device)
        audio = torch.randn(8000)
        res = explainer.explain_token(
            audio=audio,
            frame_idx=5,
            token_idx=3,
            method="integrated_gradients",
            n_steps=5,
        )
        self.assertIn("attributions", res)
        self.assertEqual(res["attributions"].shape, (8000,))

    def test_naps_profiling_and_patching(self):
        profiler = NeuronActivationProfiler(self.model, self.device)
        profiles = profiler.collect_profiles(self.dataset, max_samples=2)
        self.assertIn(0, profiles)

        patcher = ActivationPatcher(self.model, self.device)
        res = patcher.ablate_layer(torch.randn(8000), layer_idx=0, ablation_type="zero")
        self.assertIn("logit_mean_diff", res)

    def test_layerwise_probing(self):
        probe_trainer = LayerwiseProbeTrainer(self.model, self.device)
        results = probe_trainer.train_diagnostic_probes(self.dataset, epochs=1, num_classes=4)
        self.assertIn("accuracies", results)
        self.assertEqual(len(results["accuracies"]), self.config.encoder_layers + 1)


if __name__ == "__main__":
    unittest.main()
