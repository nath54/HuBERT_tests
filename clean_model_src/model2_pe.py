"""
Model Part 2: Phoneme Encoder (Model_PE).
Encodes discrete phoneme token sequences into continuous acoustic word latents z_word in R^512.
Architecture:
- FastText-style phoneme token and positional embeddings
- 2-layer Lightweight Transformer Encoder (d_model=256, nhead=4)
- Attention-Weighted Sequence-to-Vector Pooling
- Multi-layer Latent Projection Head (256 -> 512 -> 512)
"""

# Import Modules
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .model_config import ModelConfig, ModelPEConfig
from .model_part import Model_Part
from .layers.attention import AttentionPooling


class Model_PE(Model_Part):
    """
    Clean Modular Phoneme Encoder (PE) Model Architecture.
    Maps intra-word phoneme sequences into the continuous word latent space (z_word) of Model 3.
    Operates at sub-millisecond latencies on CPU and mobile NPUs.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        """
        Initialize Model_PE.

        Args:
            model_config (ModelConfig): Unified model configuration dataclass.
        """
        # Parent initialization with tag 'PE'
        super().__init__(model_config=model_config, model_part="PE")

        cfg: ModelPEConfig = model_config.pe
        self.cfg: ModelPEConfig = cfg

        self.d_macro: int = cfg.d_macro  # 512
        self.d_model: int = cfg.d_model  # 256
        self.max_seq_len: int = cfg.max_seq_len  # 32

        # 1. Modality Token Embeddings
        # Maps discrete phoneme IDs [0..111] to continuous vectors [d_model=256]
        self.phoneme_embedding: nn.Embedding = nn.Embedding(
            cfg.phoneme_vocab_size,
            cfg.d_model,
            padding_idx=0,
        )

        # Optional byte embedding for cross-modal text representation
        self.byte_embedding: nn.Embedding = nn.Embedding(
            128,
            cfg.d_model,
            padding_idx=0,
        )

        # 2. Learned Positional Embeddings for intra-word token slots [0..max_seq_len-1]
        self.pos_embedding: nn.Embedding = nn.Embedding(
            cfg.max_seq_len,
            cfg.d_model,
        )

        # 3. Modality Bias: distinguishes text bytes (0) from acoustic phonemes (1)
        self.modality_bias: nn.Embedding = nn.Embedding(2, cfg.d_model)

        # 4. Lightweight 2-layer Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_layers,
            enable_nested_tensor=False,
        )
        self.norm: nn.LayerNorm = nn.LayerNorm(cfg.d_model)

        # 5. Attention Pooling across intra-word phonemes
        self.pooling: AttentionPooling = AttentionPooling(embed_dim=cfg.d_model)

        # 6. Latent Space Projection: [d_model=256] -> [d_macro=512]
        self.proj: nn.Sequential = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_macro),
            nn.GELU(),
            nn.LayerNorm(cfg.d_macro),
            nn.Linear(cfg.d_macro, cfg.d_macro),
        )

    def _encode_tokens(
        self,
        token_ids: Tensor,
        is_phoneme: bool = True,
    ) -> Tensor:
        """
        Embeds, contextualizes, and pools token sequences into continuous latent vectors.

        Args:
            token_ids (Tensor): Input token IDs of shape [N_words, K_tokens].
                                Type: torch.long.
                                Role: Intra-word discrete character or phoneme IDs.
            is_phoneme (bool): True if inputs are phonemes, False if orthographic bytes.

        Returns:
            Tensor: Continuous word latent representations z_word of shape [N_words, 512].
        """
        n_words, k_tokens = token_ids.shape
        device: torch.device = token_ids.device

        # 1. Base embedding lookup: [N, K, 256]
        if is_phoneme:
            # Shape: [N, K, 256], Type: torch.float32
            tok_emb: Tensor = self.phoneme_embedding(token_ids.clamp(0, self.cfg.phoneme_vocab_size - 1))
            mod_idx: int = 1
        else:
            tok_emb = self.byte_embedding(token_ids.clamp(0, 127))
            mod_idx = 0

        # 2. Add position embedding: [1, K, 256]
        pos_idx: Tensor = torch.arange(k_tokens, device=device).unsqueeze(0).clamp(0, self.max_seq_len - 1)
        pos_emb: Tensor = self.pos_embedding(pos_idx)

        # 3. Add modality bias: [1, 1, 256]
        mod_emb: Tensor = self.modality_bias(torch.tensor([mod_idx], device=device)).unsqueeze(1)

        # Combined initial token representations: [N, K, 256]
        h: Tensor = tok_emb + pos_emb + mod_emb

        # 4. Padding mask for attention (0 is pad token): [N, K]
        pad_mask: Tensor = token_ids == 0

        # 5. Transformer Contextual Mixing: [N, K, 256]
        h = self.transformer(h, src_key_padding_mask=pad_mask)
        h = self.norm(h)

        # 6. Attention Pooling: [N, K, 256] -> [N, 256]
        pooled: Tensor = self.pooling(h, pad_mask=pad_mask)

        # 7. Project into Word Latent Space (z_word): [N, 256] -> [N, 512]
        z_word: Tensor = self.proj(pooled)
        return z_word

    def forward_training(
        self,
        phoneme_ids: Tensor,
        target_latents: Optional[Tensor] = None,
        byte_ids: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Clean Training Forward Pass.
        Computes latent regression loss against target acoustic latents or cross-modal byte embeddings.

        Args:
            phoneme_ids (Tensor): Input phoneme IDs of shape [B, L, K] or [N, K].
            target_latents (Optional[Tensor]): Ground truth teacher word latents [B, L, 512].
            byte_ids (Optional[Tensor]): Orthographic byte IDs for cross-modal contrastive alignment.

        Returns:
            Dict[str, Tensor]: Dictionary with latent regression loss and predicted latents.
        """
        original_ndim: int = phoneme_ids.ndim
        if original_ndim == 3:
            b_sz, l_words, k_tokens = phoneme_ids.shape
            flat_tokens: Tensor = phoneme_ids.view(-1, k_tokens)
        else:
            flat_tokens = phoneme_ids

        # Compute predicted word latents: [N, 512]
        pred_z: Tensor = self._encode_tokens(flat_tokens, is_phoneme=True)

        if original_ndim == 3:
            # Reshape back to sequence: [B, L, 512]
            pred_z_seq: Tensor = pred_z.view(b_sz, l_words, self.d_macro)
        else:
            pred_z_seq = pred_z

        losses: Dict[str, Tensor] = {"z_word": pred_z_seq}

        # Compute latent alignment loss if teacher targets provided
        if target_latents is not None:
            # MSE Latent Loss + Cosine Similarity Loss
            mse_loss: Tensor = F.mse_loss(pred_z_seq, target_latents)
            cos_loss: Tensor = 1.0 - F.cosine_similarity(pred_z_seq, target_latents, dim=-1).mean()
            total_loss: Tensor = mse_loss + 0.5 * cos_loss
            losses["loss"] = total_loss
            losses["mse_loss"] = mse_loss
            losses["cos_loss"] = cos_loss
        else:
            losses["loss"] = torch.tensor(0.0, device=phoneme_ids.device)

        return losses

    @torch.no_grad()
    def forward_inference(
        self,
        word_slices: List[List[int]],
        max_tokens_per_word: int = 24,
    ) -> Tensor:
        """
        Clean, Vectorized Inference Forward Pass.
        Encodes a list of variable-length phoneme word slices into a batched word latent tensor.

        Args:
            word_slices (List[List[int]]): Sequence of L words, each word containing a list of phoneme token IDs.
            max_tokens_per_word (int): Maximum phoneme truncation limit per word (default: 24).

        Returns:
            Tensor: Word latent representations z_word of shape [1, L, 512].
                    Type: torch.float32.
                    Role: Continuous word-level representations ready for Model 3.
        """
        self.eval()
        device: torch.device = self.get_device()
        l_words: int = len(word_slices)

        if l_words == 0:
            # Return empty latent: [1, 0, 512]
            return torch.zeros((1, 0, self.d_macro), dtype=torch.float32, device=device)

        # 1. Determine maximum word length in current batch: K <= 24
        max_k: int = min(max_tokens_per_word, max(1, max(len(w) for w in word_slices)))

        # 2. Pad into tensor: [L, max_k]
        padded_tokens: Tensor = torch.zeros((l_words, max_k), dtype=torch.long, device=device)
        for w_idx, phoneme_list in enumerate(word_slices):
            valid_len: int = min(len(phoneme_list), max_k)
            if valid_len > 0:
                padded_tokens[w_idx, :valid_len] = torch.tensor(
                    phoneme_list[:valid_len],
                    dtype=torch.long,
                    device=device,
                )

        # 3. Compute continuous word latents: [L, 512]
        z_words_flat: Tensor = self._encode_tokens(padded_tokens, is_phoneme=True)

        # 4. Add batch dimension: [1, L, 512]
        z_words: Tensor = z_words_flat.unsqueeze(0)
        return z_words

    def load_pretrained_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Loads weights from existing robust P2G checkpoints (e.g. best_robust_p2g.pt).

        Args:
            checkpoint_path (str): Path to saved checkpoint file.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        device: torch.device = self.get_device()
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = ckpt.get("phoneme_encoder_state_dict", ckpt.get("model_state_dict", ckpt))

        own_state = self.state_dict()
        filtered_sd = {}
        for k, v in sd.items():
            if k in own_state:
                if own_state[k].shape == v.shape:
                    filtered_sd[k] = v
                elif own_state[k].ndim == v.ndim:
                    # Partial slice copy for embedding or layer resizing
                    min_slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(own_state[k].shape, v.shape))
                    own_state[k][min_slices] = v[min_slices]
                    filtered_sd[k] = own_state[k]

        self.load_state_dict(filtered_sd, strict=False)
        return True
