"""Unit tests for dataset and tokenizer."""

import unittest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.tokenizer import CharacterTokenizer
from src.data.dataset import AudioASRDataset, AudioCollateFn
from src.utils.audio import synthesize_spoken_word


class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_tokenizer_encoding_decoding(self):
        text = "hello world"
        encoded = self.tokenizer.encode(text)
        self.assertIsInstance(encoded, list)
        self.assertTrue(all(isinstance(x, int) for x in encoded))

        # Test CTC greedy collapse decoding
        # Suppose model outputs repeated characters with blanks
        ctc_stream = [
            self.tokenizer.blank_id,
            self.tokenizer.char_to_id["h"],
            self.tokenizer.char_to_id["h"],  # repeat
            self.tokenizer.blank_id,
            self.tokenizer.char_to_id["i"],
        ]
        decoded = self.tokenizer.decode(ctc_stream)
        self.assertEqual(decoded, "hi")

    def test_audio_collate_fn(self):
        samples = [
            {
                "id": "s1",
                "waveform": torch.randn(16000),
                "transcript": "one",
                "sample_rate": 16000,
            },
            {
                "id": "s2",
                "waveform": torch.randn(24000),
                "transcript": "two words",
                "sample_rate": 16000,
            },
        ]
        dataset = AudioASRDataset(samples, self.tokenizer)
        collate_fn = AudioCollateFn(pad_token_id=self.tokenizer.pad_id)

        batch = collate_fn([dataset[0], dataset[1]])
        self.assertEqual(batch["audio"].shape, (2, 24000))
        self.assertEqual(batch["audio_lengths"].tolist(), [16000, 24000])
        self.assertEqual(len(batch["texts"]), 2)


if __name__ == "__main__":
    unittest.main()
