"""Phono-V7 Multilingual 768-Dimensional Lexical Word Embedding Space.

Manages the vocabulary of words across English, Italian, Spanish, and French.
Provides:
1. Bidirectional mapping: word string <-> word ID.
2. Dense 768-dimensional normalized word embedding table.
3. Contrastive InfoNCE + Cosine Similarity loss over word latents.
4. Top-1 and Top-5 word accuracy metrics.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultilingualLexicon(nn.Module):
    """Multilingual 768D Lexical Word Embedding Space."""

    PAD_ID: int = 0
    BLANK_ID: int = 1
    BOS_ID: int = 2
    EOS_ID: int = 3
    UNK_ID: int = 4

    def __init__(
        self,
        words: Optional[List[str]] = None,
        embed_dim: int = 768,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature

        self.id_to_word: Dict[int, str] = {
            self.PAD_ID: "<pad>",
            self.BLANK_ID: "<blank>",
            self.BOS_ID: "<bos>",
            self.EOS_ID: "<eos>",
            self.UNK_ID: "<unk>",
        }
        self.word_to_id: Dict[str, int] = {v: k for k, v in self.id_to_word.items()}

        if words is not None:
            for w in sorted(set(words)):
                w_clean = w.strip().lower()
                if w_clean and w_clean not in self.word_to_id:
                    idx = len(self.id_to_word)
                    self.id_to_word[idx] = w_clean
                    self.word_to_id[w_clean] = idx

        self.vocab_size = len(self.id_to_word)

        # 768-dim Word Embedding Table
        self.embeddings = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.PAD_ID)
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embeddings.weight[self.PAD_ID].fill_(0.0)

    def encode_words(self, words: List[str]) -> List[int]:
        """Convert a list of word strings to word IDs."""
        return [self.word_to_id.get(w.strip().lower(), self.UNK_ID) for w in words]

    def decode_ids(self, ids: List[int]) -> List[str]:
        """Convert a list of word IDs to word strings."""
        return [self.id_to_word.get(i, "<unk>") for i in ids if i not in (self.PAD_ID, self.BLANK_ID, self.BOS_ID, self.EOS_ID)]

    def get_normalized_embeddings(self) -> torch.Tensor:
        """Return L2-normalized embedding matrix [V, 768]."""
        return F.normalize(self.embeddings.weight, p=2, dim=-1)

    def lookup_target_embeddings(self, word_ids: torch.Tensor) -> torch.Tensor:
        """Lookup L2-normalized embeddings for given word IDs [B, L, 768]."""
        raw_emb = self.embeddings(word_ids)  # [B, L, 768]
        return F.normalize(raw_emb, p=2, dim=-1)

    def compute_lexical_loss(
        self,
        z_pred: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Cosine Similarity + Contrastive Word Loss.

        Args:
            z_pred: [B, L, 768] predicted word latent vectors from macro decoder.
            target_ids: [B, L] target word token IDs.

        Returns:
            loss: scalar loss.
            acc: Top-1 accuracy percentage on non-pad word tokens.
        """
        B, L, D = z_pred.shape
        device = z_pred.device

        valid_mask = (target_ids != self.PAD_ID) & (target_ids != -100)
        if not valid_mask.any():
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        z_flat = z_pred[valid_mask]  # [N, 768]
        targets_flat = target_ids[valid_mask]  # [N]

        # 1. Cosine Loss against ground-truth target embedding
        z_norm = F.normalize(z_flat, p=2, dim=-1)  # [N, 768]
        target_emb = self.lookup_target_embeddings(targets_flat).to(z_norm.dtype)  # [N, 768]
        cos_sim = (z_norm * target_emb).sum(dim=-1)  # [N]
        cosine_loss = (1.0 - cos_sim).mean()

        # 2. Multilingual Lexicon Vocabulary Cross-Entropy / Contrastive Loss
        vocab_embs = self.get_normalized_embeddings().to(z_norm.dtype)  # [V, 768]
        logits = torch.matmul(z_norm, vocab_embs.t()) / self.temperature  # [N, V]
        ce_loss = F.cross_entropy(logits, targets_flat)

        total_loss = cosine_loss + ce_loss

        # Top-1 accuracy over the full multilingual vocabulary
        preds = logits.argmax(dim=-1)
        acc = (preds == targets_flat).float().mean() * 100.0

        return total_loss, acc

    def predict_words(self, z_pred: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[List[str]]:
        """Decode predicted 768D word latents to word strings via nearest neighbor."""
        B, L, D = z_pred.shape
        z_norm = F.normalize(z_pred, p=2, dim=-1)  # [B, L, 768]
        vocab_embs = self.get_normalized_embeddings().to(z_norm.dtype)  # [V, 768]
        sims = torch.matmul(z_norm, vocab_embs.t())  # [B, L, V]
        pred_ids = sims.argmax(dim=-1).cpu().tolist()  # [B, L]

        results = []
        for b in range(B):
            b_words = []
            for l in range(L):
                if mask is not None and not mask[b, l]:
                    continue
                w_id = pred_ids[b][l]
                if w_id not in (self.PAD_ID, self.BLANK_ID, self.BOS_ID, self.EOS_ID):
                    b_words.append(self.id_to_word.get(w_id, "<unk>"))
            results.append(b_words)
        return results

    def save(self, path: Union[str, Path]):
        """Save lexicon vocabulary and embedding weights to file."""
        torch.save(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "id_to_word": self.id_to_word,
                "word_to_id": self.word_to_id,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MultilingualLexicon":
        """Load lexicon vocabulary and weights from file."""
        data = torch.load(path, map_location="cpu")
        lex = cls(embed_dim=data["embed_dim"])
        lex.vocab_size = data["vocab_size"]
        lex.id_to_word = data["id_to_word"]
        lex.word_to_id = data["word_to_id"]
        lex.embeddings = nn.Embedding(lex.vocab_size, lex.embed_dim, padding_idx=cls.PAD_ID)
        lex.load_state_dict(data["state_dict"])
        return lex
