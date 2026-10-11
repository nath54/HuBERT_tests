import unittest
import torch
from src.models.phoneme_engram import PhonemeEngramMemory


class TestPhonemeEngramMemory(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = 384
        self.vocab_size = 112
        self.engram = PhonemeEngramMemory(
            embed_dim=self.embed_dim,
            vocab_size=self.vocab_size,
            num_heads=4,
            memory_dim=128,
        ).to(self.device)

    def test_forward_shape_and_range(self):
        B, T = 2, 50
        hidden_states = torch.randn(B, T, self.embed_dim, device=self.device, requires_grad=True)
        tokens = torch.randint(0, self.vocab_size, (B, T), device=self.device)

        out, gate = self.engram(hidden_states, tokens)

        self.assertEqual(out.shape, (B, T, self.embed_dim))
        self.assertEqual(gate.shape, (B, T, self.embed_dim))
        self.assertTrue((gate >= 0.0).all() and (gate <= 1.0).all())

    def test_backward_gradient_flow(self):
        B, T = 2, 20
        hidden_states = torch.randn(B, T, self.embed_dim, device=self.device, requires_grad=True)
        tokens = torch.randint(0, self.vocab_size, (B, T), device=self.device)

        out, _ = self.engram(hidden_states, tokens)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(hidden_states.grad)
        self.assertIsNotNone(self.engram.table_2gram.weight.grad)
        self.assertIsNotNone(self.engram.table_3gram.weight.grad)
        self.assertIsNotNone(self.engram.table_4gram.weight.grad)
        self.assertIsNotNone(self.engram.memory_projection[0].weight.grad)
        self.assertIsNotNone(self.engram.gate_net[0].weight.grad)

    def test_deterministic_hashing(self):
        B, T = 1, 10
        tokens = torch.randint(0, self.vocab_size, (B, T), device=self.device)
        h2_a = self.engram._hash_2gram(tokens)
        h2_b = self.engram._hash_2gram(tokens)
        self.assertTrue(torch.equal(h2_a, h2_b))

        h3_a = self.engram._hash_3gram(tokens)
        h3_b = self.engram._hash_3gram(tokens)
        self.assertTrue(torch.equal(h3_a, h3_b))

        h4_a = self.engram._hash_4gram(tokens)
        h4_b = self.engram._hash_4gram(tokens)
        self.assertTrue(torch.equal(h4_a, h4_b))


if __name__ == "__main__":
    unittest.main()
