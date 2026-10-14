import unittest
import torch
from src.models.phonotactic_beam_search import PhonotacticPrefixBeamSearch


class TestPhonotacticPrefixBeamSearch(unittest.TestCase):
    def setUp(self):
        self.decoder = PhonotacticPrefixBeamSearch(beam_width=8, blank_id=1, space_id=8)

    def test_basic_decode(self):
        T, V = 20, 112
        # Default all frames to blank
        logits = torch.full((T, V), -10.0)
        logits[:, 1] = 5.0
        # Peak token 15 at t=2..5, token 20 at t=11..15
        logits[2:6, 15] = 15.0
        logits[11:16, 20] = 15.0

        res = self.decoder.decode(logits)
        self.assertEqual(res, [15, 20])

    def test_no_double_spaces(self):
        T, V = 15, 112
        # Default all frames to blank, but space dominates
        logits = torch.full((T, V), -10.0)
        logits[:, 1] = 0.0
        logits[:, 8] = 10.0

        res = self.decoder.decode(logits)
        for i in range(len(res) - 1):
            self.assertFalse(res[i] == 8 and res[i + 1] == 8)


if __name__ == "__main__":
    unittest.main()
