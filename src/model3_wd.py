"""
Model Part 3: Word Decoder (Model_WD).
Decodes continuous word latents z_word in R^512 into orthographic text character sequences.
Architecture:
- Overlapping Word-Length Experts MoE Router (7 intervals: [1,5] to [15,30])
- Context Window Builder: Local context windows (band-causal past words)
- Autoregressive Character Micro-Head: Emits character byte tokens c_1..c_K per word
- Vectorized Greedy Rollout for Real-Time Inference (< 1-2 ms per word)
"""

# Import Modules
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .model_config import ModelConfig, ModelWDConfig
from .model_part import Model_Part
from .layers.heads import LengthExpertRouter, LENGTH_INTERVALS, NUM_OVERLAPPING_EXPERTS


class WindowedCharacterHead(nn.Module):
    """
    Autoregressive Character Micro-Head for Intra-Word Decoding.
    Given a word latent representation z_word and expert path conditioning,
    autoregressively predicts character tokens [c_1, c_2, ..., c_K, <eos>].
    """

    def __init__(
        self,
        micro_dim: int = 512,
        char_vocab_size: int = 128,
        max_word_len: int = 24,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize WindowedCharacterHead.

        Args:
            micro_dim (int): Hidden dimension D (512).
            char_vocab_size (int): Character vocabulary size (128 ASCII/Latin).
            max_word_len (int): Maximum character length per word (24).
            dropout (float): Dropout probability.
        """
        # Parent initialization
        super().__init__()
        self.micro_dim: int = micro_dim
        self.char_vocab_size: int = char_vocab_size
        self.max_word_len: int = max_word_len

        # 1. Character token embedding: [char_vocab, micro_dim]
        self.char_embedding: nn.Embedding = nn.Embedding(
            char_vocab_size,
            micro_dim,
            padding_idx=0,
        )

        # 2. Position embedding within word: [max_word_len, micro_dim]
        self.pos_embedding: nn.Embedding = nn.Embedding(
            max_word_len,
            micro_dim,
        )

        # 3. Micro-transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=micro_dim,
            nhead=4,
            dim_feedforward=1024,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder: nn.TransformerDecoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=2,
        )

        # 4. Final character logit projection: [micro_dim, char_vocab_size]
        self.out_proj: nn.Linear = nn.Linear(micro_dim, char_vocab_size, bias=True)

    def forward(
        self,
        char_inputs: Tensor,
        z_context: Tensor,
        expert_cond: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass predicting next-character logits.

        Args:
            char_inputs (Tensor): Input prefix character IDs of shape [N_words, S_chars].
            z_context (Tensor): Conditioning word latents of shape [N_words, 1, micro_dim]
                                or context window [N_words, W, micro_dim].
            expert_cond (Optional[Tensor]): Learned expert path vector [N_words, 1, micro_dim].

        Returns:
            Tensor: Character logits of shape [N_words, S_chars, char_vocab_size].
        """
        n_words, s_chars = char_inputs.shape
        device: torch.device = char_inputs.device

        # 1. Embed prefix characters: [N, S, micro_dim]
        char_emb: Tensor = self.char_embedding(char_inputs)

        # 2. Add position embedding: [1, S, micro_dim]
        pos_idx: Tensor = torch.arange(s_chars, device=device).unsqueeze(0).clamp(0, self.max_word_len - 1)
        pos_emb: Tensor = self.pos_embedding(pos_idx)
        h: Tensor = char_emb + pos_emb

        # 3. Add expert conditioning bias if provided: [N, 1, micro_dim]
        if expert_cond is not None:
            h = h + expert_cond

        # 4. Construct causal mask: prevents attending to future characters within the word
        causal_mask: Tensor = nn.Transformer.generate_square_subsequent_mask(s_chars, device=device)

        # 5. Decoder cross-attention against word latent memory: [N, S, micro_dim]
        decoded: Tensor = self.decoder(
            tgt=h,
            memory=z_context,
            tgt_mask=causal_mask,
        )

        # 6. Project to vocabulary: [N, S, char_vocab_size]
        logits: Tensor = self.out_proj(decoded)
        return logits


class Model_WD(Model_Part):
    """
    Clean Modular Word Decoder (WD) Model Architecture.
    Translates continuous acoustic word latents z_word into clean orthographic text characters.
    Combines overlapping word-length MoE routing with an autoregressive character head.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        """
        Initialize Model_WD.

        Args:
            model_config (ModelConfig): Unified model configuration dataclass.
        """
        # Parent initialization with tag 'WD'
        super().__init__(model_config=model_config, model_part="WD")

        cfg: ModelWDConfig = model_config.wd
        self.cfg: ModelWDConfig = cfg

        self.macro_dim: int = cfg.macro_dim  # 512
        self.micro_dim: int = cfg.micro_dim  # 512
        self.num_experts: int = cfg.num_overlapping_experts  # 7
        self.max_word_len: int = cfg.max_word_len  # 24

        # 1. Overlapping Length Experts Router: [macro_dim -> 7 expert logits]
        self.macro_expert_router: LengthExpertRouter = LengthExpertRouter(
            macro_dim=cfg.macro_dim,
            num_experts=cfg.num_overlapping_experts,
        )

        # 2. Expert Conditioning Embeddings: [7, micro_dim]
        self.expert_embeddings: nn.Embedding = nn.Embedding(
            cfg.num_overlapping_experts,
            cfg.micro_dim,
        )

        # 3. Autoregressive Character Micro-Head
        self.micro_head: WindowedCharacterHead = WindowedCharacterHead(
            micro_dim=cfg.micro_dim,
            char_vocab_size=cfg.char_vocab_size,
            max_word_len=cfg.max_word_len,
        )

        # Submodules registered
    
    @property
    def decoder(self) -> "Model_WD":
        """Property alias for backward compatibility without creating circular module hierarchy."""
        return self

    def build_word_context_windows(self, z_words: Tensor, window_size: int = 4) -> Tensor:
        """
        Builds local causal context windows of surrounding word latents for each word slot.

        Args:
            z_words (Tensor): Sequence of word latents [B, L, macro_dim].
            window_size (int): Context window length W (default: 4 words).

        Returns:
            Tensor: Padded context windows [B, L, W, macro_dim].
        """
        b_sz, l_words, dim = z_words.shape
        device: torch.device = z_words.device

        # Left-pad by (window_size - 1) words
        padded = F.pad(z_words, (0, 0, window_size - 1, 0), mode="constant", value=0.0)

        # Unfold into sliding windows: [B, L, W, dim]
        windows = padded.unfold(dimension=1, size=window_size, step=1)
        # Permute to [B, L, W, dim]
        windows = windows.permute(0, 1, 3, 2).contiguous()
        return windows

    def forward_training(
        self,
        z_words: Tensor,
        input_byte_ids: Tensor,
        target_byte_ids: Tensor,
        target_lengths: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Clean Training Forward Pass.
        Computes character cross-entropy loss and interval-based length routing loss.

        Args:
            z_words (Tensor): Input word latents of shape [B, L, 512].
            input_byte_ids (Tensor): Teacher-forced input character prefixes [B, L, K].
            target_byte_ids (Tensor): Target character IDs [B, L, K] (-100 on pad).
            target_lengths (Optional[Tensor]): True word lengths [B, L] in characters.

        Returns:
            Dict[str, Tensor]: Multi-task loss dict with character cross-entropy and routing loss.
        """
        b_sz, l_words, k_chars = input_byte_ids.shape
        n_total_words: int = b_sz * l_words

        # Flatten batch and words: [N_words, K_chars]
        flat_inputs: Tensor = input_byte_ids.view(n_total_words, k_chars)
        flat_targets: Tensor = target_byte_ids.view(n_total_words, k_chars)
        flat_z: Tensor = z_words.view(n_total_words, 1, self.macro_dim)

        # 1. Overlapping Length Experts MoE Routing: [B, L, 7]
        expert_logits: Tensor = self.macro_expert_router(z_words)
        flat_exp_logits: Tensor = expert_logits.view(n_total_words, self.num_experts)

        # Best-matching expert index: [N_words]
        top1_experts: Tensor = flat_exp_logits.argmax(dim=-1)
        # Retrieve expert conditioning vectors: [N_words, 1, micro_dim]
        expert_cond: Tensor = self.expert_embeddings(top1_experts).unsqueeze(1)

        # 2. Forward through Autoregressive Character Head: [N_words, K_chars, char_vocab]
        char_logits: Tensor = self.micro_head(
            char_inputs=flat_inputs,
            z_context=flat_z,
            expert_cond=expert_cond,
        )

        # 3. Character Cross-Entropy Loss: ignores -100 padding slots
        char_loss: Tensor = F.cross_entropy(
            char_logits.view(-1, self.cfg.char_vocab_size),
            flat_targets.view(-1),
            ignore_index=-100,
        )

        # 4. Length Expert Supervision Loss (if target lengths provided)
        if target_lengths is not None:
            flat_lengths: Tensor = target_lengths.view(n_total_words)
            valid_mask: Tensor = flat_lengths > 0

            # Construct binary multi-positive targets: 1.0 if interval covers the length
            multi_pos_targets = torch.zeros_like(flat_exp_logits)
            for exp_idx, (min_l, max_l) in enumerate(LENGTH_INTERVALS):
                covers = (flat_lengths >= min_l) & (flat_lengths <= max_l)
                multi_pos_targets[:, exp_idx] = covers.float()

            if valid_mask.any():
                router_loss: Tensor = F.binary_cross_entropy_with_logits(
                    flat_exp_logits[valid_mask],
                    multi_pos_targets[valid_mask],
                )
            else:
                router_loss = torch.tensor(0.0, device=z_words.device)
        else:
            router_loss = torch.tensor(0.0, device=z_words.device)

        total_loss: Tensor = char_loss + 0.1 * router_loss
        return {
            "loss": total_loss,
            "char_loss": char_loss,
            "router_loss": router_loss,
            "char_logits": char_logits,
        }

    def forward_single_step(self, z_word: Tensor, byte_ids: Tensor) -> Tensor:
        """
        Executes single-step character logit prediction for mobile static graph export.

        Args:
            z_word (Tensor): Word latent tensor [B, L, 512].
            byte_ids (Tensor): Prefix character tokens [B, L, S_chars].

        Returns:
            Tensor: Character logits [B, L, S_chars, char_vocab_size].
        """
        b_sz, l_words, dim = z_word.shape
        flat_z: Tensor = z_word.view(-1, 1, dim)
        s_chars = byte_ids.shape[-1]
        flat_bytes: Tensor = byte_ids.view(-1, s_chars)

        expert_logits: Tensor = self.macro_expert_router(z_word)
        top1_experts: Tensor = expert_logits.view(-1, self.num_experts).argmax(dim=-1)
        expert_cond: Tensor = self.expert_embeddings(top1_experts).unsqueeze(1)

        logits: Tensor = self.micro_head(
            char_inputs=flat_bytes,
            z_context=flat_z,
            expert_cond=expert_cond,
        )
        return logits.view(b_sz, l_words, s_chars, -1)

    @torch.no_grad()
    def forward_inference(
        self,
        z_words: Tensor,
        max_len: int = 24,
    ) -> List[str]:
        """
        Clean, Vectorized Inference Forward Pass.
        Executes parallel greedy character rollout across all L words simultaneously in tensor space.

        Args:
            z_words (Tensor): Word latent tensor of shape [1, L, 512] or [B, L, 512].
            max_len (int): Maximum rollout character steps per word (default: 24).

        Returns:
            List[str]: Reconstructed decoded text words.
        """
        self.eval()
        device: torch.device = self.get_device()

        if z_words.ndim == 2:
            z_words = z_words.unsqueeze(0)

        b_sz, l_words, dim = z_words.shape
        if l_words == 0:
            return []

        # Flatten batch and word slots: [L, 1, 512]
        flat_z: Tensor = z_words.view(l_words, 1, dim)

        # 1. Expert Router: [L, 7] -> argmax expert index [L]
        expert_logits: Tensor = self.macro_expert_router(z_words)
        top1_experts: Tensor = expert_logits.view(l_words, self.num_experts).argmax(dim=-1)

        # 2. Expert Conditioning: [L, 1, micro_dim]
        expert_cond: Tensor = self.expert_embeddings(top1_experts).unsqueeze(1)

        # 3. Vectorized Greedy Character Rollout
        # Initialize prefix buffer with BOS token (1): [L, max_len]
        active_words: Tensor = torch.full((l_words, max_len), self.cfg.pad_id, dtype=torch.long, device=device)
        active_words[:, 0] = self.cfg.bos_id
        finished: Tensor = torch.zeros(l_words, dtype=torch.bool, device=device)

        # Step through characters 1..max_len-1
        for step in range(1, max_len):
            cur_prefix: Tensor = active_words[:, :step]
            # Character logits: [L, step, char_vocab_size]
            logits: Tensor = self.micro_head(
                char_inputs=cur_prefix,
                z_context=flat_z,
                expert_cond=expert_cond,
            )
            # Greedy prediction at current step: [L]
            next_tokens: Tensor = logits[:, -1, :].argmax(dim=-1)

            for l in range(l_words):
                if not finished[l]:
                    token_val: int = int(next_tokens[l].item())
                    # Check for EOS (2) or PAD (0)
                    if token_val in (self.cfg.eos_id, self.cfg.pad_id):
                        finished[l] = True
                    else:
                        active_words[l, step] = token_val

            # Break early if all L words finished
            if finished.all():
                break

        # 4. Decode orthographic character strings via RomanCharTokenizer vocabulary
        from src.tokenizers.roman_tokenizer import RomanCharTokenizer
        _tokenizer = RomanCharTokenizer()

        decoded_words: List[str] = []
        for l in range(l_words):
            char_list: List[str] = []
            for tok_val in active_words[l, 1:].cpu().tolist():
                if tok_val in (self.cfg.bos_id, self.cfg.eos_id, self.cfg.pad_id, 4):  # ignore control / eow
                    continue
                char_sym = _tokenizer.id2char.get(tok_val, "")
                if char_sym and not char_sym.startswith("<"):
                    char_list.append(char_sym)
            word_str: str = "".join(char_list).strip()
            decoded_words.append(word_str)

        # Store decoding diagnostic details
        self.last_decoding_details: Dict[str, Any] = {
            "expert_logits": expert_logits.detach(),
            "top1_experts": top1_experts.detach(),
            "active_words": active_words.detach(),
            "decoded_words": decoded_words,
        }

        return decoded_words

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
        sd = ckpt.get("decoder_state_dict", ckpt.get("model_state_dict", ckpt))

        own_state = self.state_dict()
        filtered_sd = {}
        for k, v in sd.items():
            if k in own_state:
                if own_state[k].shape == v.shape:
                    filtered_sd[k] = v
                elif own_state[k].ndim == v.ndim:
                    min_slices = tuple(slice(0, min(s1, s2)) for s1, s2 in zip(own_state[k].shape, v.shape))
                    own_state[k][min_slices] = v[min_slices]
                    filtered_sd[k] = own_state[k]

        self.load_state_dict(filtered_sd, strict=False)
        return True
