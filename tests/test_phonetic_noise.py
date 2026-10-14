import unittest
from src.data.phonetic_noise import PhoneticNoiseSimulator


class TestPhoneticNoiseSimulator(unittest.TestCase):
    def setUp(self):
        self.sim = PhoneticNoiseSimulator(
            confusion_matrix_path=None,
            sub_rate=0.10,
            del_rate=0.05,
            ins_rate=0.03,
            space_drop_rate=0.10,
            space_insert_rate=0.05,
            seed=42,
        )

    def test_perturb_phonemes(self):
        clean = "h ɛ l oʊ <space> w ɜ l d"
        perturbed = self.sim.perturb_phonemes(clean)
        self.assertIsInstance(perturbed, str)
        self.assertGreater(len(perturbed), 0)

    def test_perturb_token_ids(self):
        clean_ids = [15, 22, 35, 8, 44, 55, 60]  # with space token 8
        perturbed_ids = self.sim.perturb_token_ids(clean_ids)
        self.assertIsInstance(perturbed_ids, list)
        self.assertGreater(len(perturbed_ids), 0)

    def test_reproducibility(self):
        sim1 = PhoneticNoiseSimulator(seed=123)
        sim2 = PhoneticNoiseSimulator(seed=123)
        clean = "b ɑɹ b eɪ k <space> w ʌ z <space> f j ʊ ɹ i ə s"
        self.assertEqual(sim1.perturb_phonemes(clean), sim2.perturb_phonemes(clean))


if __name__ == "__main__":
    unittest.main()
