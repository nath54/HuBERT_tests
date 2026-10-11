import unittest
import torch
from src.models.phono_v8_acoustic import PhonoV8AcousticBackbone, warm_start_phono_v8_from_v75
from src.models.phono_variants import PhonoV64GatedDiffusionConfig


class TestPhonoV8Acoustic(unittest.TestCase):
    def setUp(self):
        # Use CPU for standalone unit tests to run cleanly alongside GPU training processes
        self.device = torch.device("cpu")
        self.config = PhonoV64GatedDiffusionConfig()
        self.model = PhonoV8AcousticBackbone(
            config=self.config,
            num_routed_experts=8,
            top_k=2,
            num_shared_experts=1,
            draft_layer_idx=3,
            engram_layers=(4, 7),
        ).to(self.device)

    def test_forward_pass_and_loss(self):
        B = 2
        audio = torch.randn(B, 16000 * 2, device=self.device)
        targets = torch.randint(8, 112, (B, 15), device=self.device)
        target_lengths = torch.tensor([15, 12], device=self.device)
        macro_targets = torch.randint(0, 8, (B, 15), device=self.device)

        out = self.model(
            audio=audio,
            targets=targets,
            target_lengths=target_lengths,
            macro_targets=macro_targets,
            macro_lengths=target_lengths,
        )

        self.assertIn("loss", out)
        self.assertIn("logits", out)
        self.assertIn("draft_logits", out)
        self.assertIn("boundary_logits", out)
        self.assertTrue(torch.isfinite(out["loss"]))

        out["loss"].backward()
        self.assertIsNotNone(self.model.encoder.layers[0].self_attn.q_proj.weight.grad)

    def test_warm_start_preservation(self):
        t, f = warm_start_phono_v8_from_v75(
            self.model,
            "checkpoints/phono_v7_5_acoustic/best_acoustic.pt",
            device=self.device,
        )
        self.assertGreater(t, 500)

    def test_forward_sliding_chunks_awkward_lengths(self):
        # Test short, long, and irregular length audios that previously risked stub chunk underflows
        lengths = [
            16000 * 2,           # 2s (shorter than 8s chunk)
            16000 * 8 + 1,       # 8s + 1 sample
            16000 * 12 + 10,     # 12s + 10 samples
            16000 * 14,          # 14s
        ]
        for L in lengths:
            audio = torch.randn(1, L, device=self.device)
            out = self.model.forward_sliding_chunks(audio, chunk_seconds=8.0, overlap_seconds=2.0)
            self.assertIn("logits", out)
            self.assertIn("boundary_logits", out)
            expected_frames = self.model.config.compute_output_length(L)
            self.assertEqual(out["logits"].shape[1], expected_frames)


if __name__ == "__main__":
    unittest.main()
