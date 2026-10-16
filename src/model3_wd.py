"""
Model Part 3: Word Decoder (Model_WD).
Decodes continuous word latents z_word in R^512 into orthographic text character sequences.
Architecture:
- Overlapping Word-Length Experts MoE Router (7 intervals: [1,5] to [15,30])
- Context Window Builder: Local context windows (band-causal past words)
- WindowedMicroPartitionedRecursiveHead: 4-layer partitioned MoE character micro-decoder
- Trie-Constrained Character Beam Search and Vectorized Greedy Rollout
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .model_config import ModelConfig, ModelWDConfig
from .model_part import Model_Part
from .layers.heads import (
    LengthExpertRouter,
    WindowedMicroPartitionedRecursiveHead,
    DeepLatentDiffusionRefiner,
    GaussianNoiseScheduler,
    WordLengthPredictor,
    SoftLevenshteinLoss,
    MacroMoELayer,
    build_band_causal_mask,
    LENGTH_INTERVALS,
    NUM_OVERLAPPING_EXPERTS,
)
from .tokenizers.roman_tokenizer import RomanCharTokenizer
from .tokenizers.trie import WordTrie, beam_search_character_rollout


class Model_WD(Model_Part):
    """
    Clean Modular Word Decoder (WD) Model Architecture.
    Translates continuous acoustic word latents z_word into clean orthographic text characters.
    Combines overlapping word-length MoE routing with an autoregressive micro-decoder.
    """

    def __init__(self, model_config: Optional[ModelConfig] = None) -> None:
        if model_config is None:
            model_config = ModelConfig()
        super().__init__(model_config=model_config, model_part="WD")

        cfg: ModelWDConfig = model_config.wd
        self.cfg: ModelWDConfig = cfg

        self.macro_dim: int = cfg.macro_dim      # 512
        self.micro_dim: int = cfg.micro_dim      # 512
        self.acoustic_dim: int = 512
        self.byte_vocab_size: int = 122
        self.max_word_len: int = cfg.max_word_len  # 24
        self.num_experts: int = NUM_OVERLAPPING_EXPERTS  # 7
        self.band_window_words: int = 8

        # 1. Phoneme Pre-net & Embeddings
        self.phoneme_embedding: nn.Embedding = nn.Embedding(111, self.macro_dim)
        self.phoneme_prenet: nn.Sequential = nn.Sequential(
            nn.Conv1d(self.macro_dim, self.macro_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.macro_dim, self.macro_dim, kernel_size=3, padding=1),
        )
        self.phoneme_norm: nn.LayerNorm = nn.LayerNorm(self.macro_dim)

        # 2. Shift-Invariant Word Query Embeddings
        self.word_slot_embedding: nn.Parameter = nn.Parameter(torch.randn(1, 64, self.macro_dim) * 0.02)
        self.word_bos_embedding: nn.Parameter = nn.Parameter(torch.randn(1, 1, self.macro_dim) * 0.02)
        self.word_query_base: nn.Parameter = nn.Parameter(torch.randn(1, 1, self.macro_dim) * 0.02)
        self.word_relative_pos_embedding: nn.Embedding = nn.Embedding(6, self.macro_dim)

        # 3. Macro Cross-Attention MoE Decoder Layers
        self.macro_layers: nn.ModuleList = nn.ModuleList([
            MacroMoELayer(
                macro_dim=self.macro_dim,
                acoustic_dim=self.acoustic_dim,
                macro_heads=8,
                macro_ffn_dim=1536,
                num_experts=4,
                top_k=2,
                cross_attn_band_width=0,
                dropout=0.1,
            )
            for _ in range(4)
        ])
        self.macro_norm: nn.LayerNorm = nn.LayerNorm(self.macro_dim)

        # 4. Path Classifier & Embeddings
        self.macro_path_classifier: nn.Linear = nn.Linear(self.macro_dim, 4)
        self.path_embedding: nn.Embedding = nn.Embedding(4, self.macro_dim)

        # 5. Word Latent Diffusion Refiner & Noise Scheduler
        self.word_refiner: DeepLatentDiffusionRefiner = DeepLatentDiffusionRefiner(self.macro_dim)
        self.noise_scheduler: GaussianNoiseScheduler = GaussianNoiseScheduler()

        # 6. Autoregressive Micro Character Head (Partitioned MoE)
        self.micro_head: WindowedMicroPartitionedRecursiveHead = WindowedMicroPartitionedRecursiveHead(
            micro_dim=self.micro_dim,
            macro_dim=self.macro_dim,
            acoustic_dim=self.acoustic_dim,
            byte_vocab_size=self.byte_vocab_size,
            pad_token_id=0,
            micro_heads=8,
            micro_layers=4,
            micro_ffn_dim=1536,
            micro_num_experts=16,
            micro_moe_top_k=2,
            acoustic_window_frames=32,
            dropout=0.1,
        )

        # 7. Word Length Predictor & Length Conditioning Bias
        self.length_predictor: WordLengthPredictor = WordLengthPredictor(self.macro_dim, 256)
        self.length_to_micro_bias: nn.Sequential = nn.Sequential(
            nn.Linear(1, 512),
            nn.GELU(),
            nn.Linear(512, self.micro_dim),
        )

        # 8. Overlapping Length Experts Router & Conditioning Embeddings
        self.macro_expert_router: nn.Linear = nn.Linear(self.macro_dim, self.num_experts)
        self.expert_embeddings: nn.Embedding = nn.Embedding(self.num_experts, self.micro_dim)

        # 9. Loss module
        self.levenshtein_loss_fn: SoftLevenshteinLoss = SoftLevenshteinLoss()

        self.last_decoding_details: Dict[str, Any] = {}

    @property
    def decoder(self) -> "Model_WD":
        """Property alias for compatibility with legacy script calls."""
        return self

    def build_word_context_windows(self, z_words: Tensor, window_size: int = 6) -> Tensor:
        """
        Builds sliding causal context windows of surrounding word latents for each word slot.

        Args:
            z_words (Tensor): Sequence of word latents [B, L, macro_dim].
            window_size (int): Context window length W (default: 6 words).

        Returns:
            Tensor: Context windows [B, L, W, macro_dim].
        """
        b_sz, l_words, dim = z_words.shape
        w = window_size
        device = z_words.device

        bos_prefix = self.word_bos_embedding.expand(b_sz, w - 1, -1)
        z_padded = torch.cat([bos_prefix, z_words], dim=1)

        # Slicing is fully supported by ONNX exporter unlike tensor.unfold
        slices = [z_padded[:, i : i + l_words, :] for i in range(w)]
        windows = torch.stack(slices, dim=2).contiguous()

        rel_pos_ids = torch.arange(w, device=device)
        rel_emb = self.word_relative_pos_embedding(rel_pos_ids)
        windows = windows + rel_emb.unsqueeze(0).unsqueeze(0)

        return windows

    def forward(
        self,
        z_words: Tensor,
        input_byte_ids: Optional[Tensor] = None,
        target_byte_ids: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Training forward pass."""
        b_sz, l_words, _ = z_words.shape
        expert_logits = self.macro_expert_router(z_words)
        top1_experts = expert_logits.argmax(dim=-1)
        expert_cond = self.expert_embeddings(top1_experts).unsqueeze(2)

        z_windows = self.build_word_context_windows(z_words)

        if input_byte_ids is not None:
            k_chars = input_byte_ids.shape[-1]
            flat_inputs = input_byte_ids.view(b_sz * l_words, k_chars)
            flat_windows = z_windows.view(b_sz * l_words, z_windows.shape[2], self.macro_dim)
            flat_cond = expert_cond.view(b_sz * l_words, 1, self.micro_dim)
            flat_top1 = top1_experts.view(b_sz * l_words)

            char_logits, aux_loss = self.micro_head(
                flat_inputs,
                flat_windows,
                acoustic_slices=None,
                path_bias=flat_cond,
                path_id=flat_top1,
            )
            out: Dict[str, Any] = {
                "char_logits": char_logits.view(b_sz, l_words, k_chars, -1),
                "expert_logits": expert_logits,
                "aux_loss": aux_loss,
            }
            if target_byte_ids is not None:
                char_loss = F.cross_entropy(
                    char_logits.view(-1, self.byte_vocab_size),
                    target_byte_ids.view(-1),
                    ignore_index=-100,
                )
                out["loss"] = char_loss + 0.1 * aux_loss
            return out

        return {"expert_logits": expert_logits, "z_windows": z_windows}

    def forward_single_step(self, z_word: Tensor, byte_ids: Tensor) -> Tensor:
        """Predict character logits for a single step or batch of prefix tokens."""
        out = self.forward(z_words=z_word, input_byte_ids=byte_ids)
        return out["char_logits"]

    def forward_training(
        self,
        z_words: Tensor,
        input_byte_ids: Tensor,
        target_byte_ids: Tensor,
        target_lengths: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        return self.forward(
            z_words=z_words,
            input_byte_ids=input_byte_ids,
            target_byte_ids=target_byte_ids,
        )

    @torch.no_grad()
    def forward_inference(
        self,
        z_words: Tensor,
        trie: Optional[WordTrie] = None,
        beam_width: int = 1,
        max_len: int = 24,
    ) -> List[str]:
        """
        Vectorized Inference Forward Pass using Trie-constrained Beam Search or Greedy Rollout.

        Args:
            z_words (Tensor): Continuous word latents of shape [1, L, 512] or [L, 512].
            trie (Optional[WordTrie]): Vocabulary Trie prefix tree for dictionary constraint.
            beam_width (int): Beam search width (1 = fast greedy rollout).
            max_len (int): Maximum character length per word (24).

        Returns:
            List[str]: Decoded orthographic text words.
        """
        self.eval()
        device: torch.device = self.get_device()

        if z_words.ndim == 2:
            z_words = z_words.unsqueeze(0)

        b_sz, l_words, dim = z_words.shape
        if l_words == 0:
            return []

        z_words = z_words.to(device)
        rom_tok = RomanCharTokenizer()

        pred_word_tokens = beam_search_character_rollout(
            model_m3=self,
            z_words=z_words,
            rom_tok=rom_tok,
            trie=trie,
            beam_width=beam_width,
            max_len=max_len,
            device=device,
        )

        decoded_words = [rom_tok.decode_word(w) for w in pred_word_tokens]
        self.last_decoding_details = {
            "l_words": l_words,
            "decoded_words": decoded_words,
        }
        return decoded_words

    def load_pretrained_checkpoint(self, checkpoint_path: str) -> bool:
        if not os.path.isfile(checkpoint_path):
            return False

        device: torch.device = self.get_device()
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = ckpt.get("decoder_state_dict", ckpt.get("model_state_dict", ckpt))

        own_state = self.state_dict()
        filtered_sd = {}
        for k, v in sd.items():
            if k in own_state and own_state[k].shape == v.shape:
                filtered_sd[k] = v

        res = self.load_state_dict(filtered_sd, strict=False)
        return len(filtered_sd) > 0
