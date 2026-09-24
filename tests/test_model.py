"""Unit tests for HuBERT model architecture."""

import unittest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.config import HuBERTConfig
from src.models.cnn_encoder import HuBERTFeatureEncoder
from src.models.transformer import HuBERTEncoder
from src.models.hubert_asr import HuBERTForCTC


class TestHuBERTArchitecture(unittest.TestCase):

    def setUp(self):
        self.config = HuBERTConfig(
            encoder_embed_dim=128,
            encoder_layers=2,
            encoder_heads=2,
            encoder_ffn_dim=256,
            pos_conv_kernel=32,
            vocab_size=32,
        )

    def test_feature_encoder_downsampling(self):
        """Test that CNN downsamples audio as expected."""
        feat_enc = HuBERTFeatureEncoder(self.config.conv_layers)
        batch_size = 2
        # 16,000 samples = 1 second of audio
        audio = torch.randn(batch_size, 16000)
        feats = feat_enc(audio)

        expected_frames = self.config.compute_output_length(16000)
        self.assertEqual(feats.shape[0], batch_size)
        self.assertEqual(feats.shape[1], expected_frames)
        self.assertEqual(feats.shape[2], self.config.conv_feature_dim)

    def test_transformer_encoder(self):
        """Test Transformer forward pass and intermediate representations."""
        encoder = HuBERTEncoder(
            embed_dim=self.config.encoder_embed_dim,
            num_layers=self.config.encoder_layers,
            num_heads=self.config.encoder_heads,
            ffn_dim=self.config.encoder_ffn_dim,
        )
        x = torch.randn(2, 50, self.config.encoder_embed_dim)
        out = encoder(x, output_hidden_states=True, output_attentions=True)

        self.assertEqual(out["last_hidden_state"].shape, (2, 50, self.config.encoder_embed_dim))
        # 1 input embedding + 2 layers = 3 hidden states
        self.assertEqual(len(out["hidden_states"]), self.config.encoder_layers + 1)
        self.assertEqual(len(out["attentions"]), self.config.encoder_layers)
        self.assertEqual(len(out["neuron_activations"]), self.config.encoder_layers)

    def test_hubert_asr_ctc(self):
        """Test end-to-end HuBERT forward pass with CTC loss and greedy decoding."""
        model = HuBERTForCTC(self.config)
        audio = torch.randn(2, 16000)
        targets = torch.tensor([[3, 4, 5, 1], [6, 7, 0, 0]], dtype=torch.long)
        target_lengths = torch.tensor([3, 2], dtype=torch.long)
        audio_lengths = torch.tensor([16000, 12000], dtype=torch.long)

        out = model(
            audio=audio,
            audio_lengths=audio_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

        self.assertIn("logits", out)
        self.assertIn("loss", out)
        self.assertFalse(torch.isnan(out["loss"]))

        decoded = model.decode_greedy(out["logits"], lengths=out["output_lengths"])
        self.assertEqual(len(decoded), 2)
        self.assertIsInstance(decoded[0], list)


if __name__ == "__main__":
    unittest.main()
