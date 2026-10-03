"""Phono-V6.5: Hierarchical Word-to-Byte Recursive Denoising Decoder with 8-Expert MoE Byte Head.

Edge-optimized, token-free multilingual speech recognition architecture:
1. Macro Word Decoder: Emits continuous word latent representations z_w via Banded Cross-Attention & 4-Expert MoE.
2. Latent Diffusion Refiner: Denoises word latent space against acoustic perturbations.
3. Micro Recursive Byte Head: Token-free 261-class UTF-8 byte generator with 8-Expert MoE (Top-2 routing) and tied weights (<300 KB RAM).
4. Universal Multilingual Coverage: 0.0% OOV across all human languages, scripts, and Unicode symbols.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.phono_variants import (
    DeepLatentDiffusionRefiner,
    GaussianNoiseScheduler,
    MoEFeedForwardNetwork,
)
from src.models.word_denoising_decoder import BandedCrossAttention


# Macro Base-Token Fast-Path Constants
SLOT_SILENCE: int = 0   # Acoustic silence / non-speech pause (bypasses byte head with 0 FLOPs)
SLOT_WORD: int = 1      # Standard lexical word (dispatches to Micro Recursive Byte Head)
SLOT_EOS: int = 2       # End-of-utterance (instant early exit)
SLOT_SPACE: int = 3     # Whitespace delimiter (direct emit without byte loop)
NUM_SLOT_CLASSES: int = 4


@dataclass
class HierarchicalByteConfig:
    """Configuration for Phono-V6.5 Hierarchical Word-to-Byte Decoder with MoE Byte Head."""
    # Macro Word Latent Decoder
    macro_dim: int = 512                # Dimension of word latent space
    acoustic_dim: int = 512             # Dimension of frozen V6.4 acoustic memory
    macro_layers: int = 4               # MoE transformer layers for macro decoder
    macro_heads: int = 8                # Macro attention heads
    macro_ffn_dim: int = 1536           # FFN intermediate expansion
    macro_num_experts: int = 4          # 4-expert MoE FFN for Macro Word Decoder
    macro_moe_top_k: int = 2            # Top-2 expert dispatch for Macro Word Decoder
    num_experts: int = 4                # Backward compatibility alias
    moe_top_k: int = 2                  # Backward compatibility alias
    cross_attn_band_width: int = 32     # Banded acoustic receptive window (~640ms)
    dropout: float = 0.1

    # Word Latent Diffusion Refiner
    use_word_diffusion: bool = True
    word_noise_max: float = 0.5
    diffusion_loss_weight: float = 0.5

    # Contrastive Homophone Separation
    contrastive_loss_weight: float = 0.2
    contrastive_temperature: float = 0.07

    # Micro Recursive Byte Head (16-Expert MoE by default)
    micro_dim: int = 512                # Full semantic resolution (no lossy downprojection)
    micro_layers: int = 4               # 4 transformer layers in micro byte head
    micro_heads: int = 8                # 8 attention heads in micro head
    micro_ffn_dim: int = 1536           # FFN expansion in micro head
    micro_num_experts: int = 16         # 16 specialized orthographic experts in micro head (Medium Tier)
    micro_moe_top_k: int = 2            # Top-2 sparse routing
    byte_vocab_size: int = 261          # 5 control tokens + 256 UTF-8 bytes
    pad_token_id: int = 0
    blank_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3
    eow_token_id: int = 4
    max_bytes_per_word: int = 24

    # Macro Base-Token Fast-Path Routing (Instant EOS early exit & Silence bypass)
    enable_macro_fastpath: bool = True
    num_slot_classes: int = NUM_SLOT_CLASSES  # 0: SILENCE, 1: WORD, 2: EOS, 3: SPACE
    macro_slot_loss_weight: float = 0.5       # Auxiliary loss weight for macro base-token detection

    @classmethod
    def medium(cls, **kwargs) -> "HierarchicalByteConfig":
        """Medium Tier: 16 experts in micro byte head with Top-2 routing (~144.6M params)."""
        defaults = {
            "macro_dim": 512,
            "acoustic_dim": 512,
            "macro_layers": 4,
            "macro_heads": 8,
            "macro_ffn_dim": 1536,
            "macro_num_experts": 4,
            "macro_moe_top_k": 2,
            "micro_dim": 512,
            "micro_layers": 4,
            "micro_heads": 8,
            "micro_ffn_dim": 1536,
            "micro_num_experts": 16,
            "micro_moe_top_k": 2,
            "enable_macro_fastpath": True,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def large(cls, **kwargs) -> "HierarchicalByteConfig":
        """Large Tier: 32 fine-grained experts in micro byte head with Top-4 routing (~310M params)."""
        defaults = {
            "macro_dim": 768,
            "acoustic_dim": 768,
            "macro_layers": 6,
            "macro_heads": 12,
            "macro_ffn_dim": 2304,
            "macro_num_experts": 8,
            "macro_moe_top_k": 2,
            "micro_dim": 768,
            "micro_layers": 6,
            "micro_heads": 12,
            "micro_ffn_dim": 768,       # Fine-grained FFN (1x dimension for high modularity)
            "micro_num_experts": 32,    # 32 fine-grained orthographic experts
            "micro_moe_top_k": 4,       # Top-4 sparse routing (DeepSeek style)
            "enable_macro_fastpath": True,
        }
        defaults.update(kwargs)
        return cls(**defaults)


class MacroMoELayer(nn.Module):
    """Macro Word Layer: Self-Attention, Banded Acoustic Cross-Attention, and MoE FFN."""

    def __init__(self, config: HierarchicalByteConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.macro_dim,
            num_heads=config.macro_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_attn = BandedCrossAttention(
            query_dim=config.macro_dim,
            key_dim=config.acoustic_dim,
            num_heads=config.macro_heads,
            band_width=config.cross_attn_band_width,
            dropout=config.dropout,
        )
        n_exp = getattr(config, "macro_num_experts", getattr(config, "num_experts", 4))
        top_k = getattr(config, "macro_moe_top_k", getattr(config, "moe_top_k", 2))
        self.moe_ffn = MoEFeedForwardNetwork(
            embed_dim=config.macro_dim,
            ffn_dim=config.macro_ffn_dim,
            num_experts=n_exp,
            top_k=top_k,
            dropout=config.dropout,
        )
        self.norm1 = nn.LayerNorm(config.macro_dim)
        self.norm2 = nn.LayerNorm(config.macro_dim)
        self.norm3 = nn.LayerNorm(config.macro_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        acoustic_memory: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
        expected_total_len: Optional[int] = None,
        return_attn_weights: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # 1. Masked Causal Self-Attention
        res = x
        normed = self.norm1(x)
        sa_out, _ = self.self_attn(normed, normed, normed, attn_mask=self_attn_mask)
        x = res + self.dropout(sa_out)

        # 2. Banded Acoustic Cross-Attention
        res = x
        normed = self.norm2(x)
        attn_weights = None
        if return_attn_weights:
            ca_out, attn_weights = self.cross_attn(
                normed,
                acoustic_memory,
                key_padding_mask=memory_padding_mask,
                expected_total_len=expected_total_len,
                return_attn_weights=True,
            )
        else:
            ca_out = self.cross_attn(
                normed,
                acoustic_memory,
                key_padding_mask=memory_padding_mask,
                expected_total_len=expected_total_len,
            )
        x = res + self.dropout(ca_out)

        # 3. MoE FFN
        res = x
        normed = self.norm3(x)
        ffn_out, aux_loss = self.moe_ffn(normed)
        x = res + self.dropout(ffn_out)

        if return_attn_weights:
            return x, aux_loss, attn_weights
        return x, aux_loss


class MicroMoEByteLayer(nn.Module):
    """Micro Byte Layer: Causal Self-Attention, Word Latent Cross-Attention, and 8-Expert MoE FFN."""

    def __init__(self, config: HierarchicalByteConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.micro_dim,
            num_heads=config.micro_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.micro_dim,
            num_heads=config.micro_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.moe_ffn = MoEFeedForwardNetwork(
            embed_dim=config.micro_dim,
            ffn_dim=config.micro_ffn_dim,
            num_experts=config.micro_num_experts,
            top_k=config.micro_moe_top_k,
            dropout=config.dropout,
        )
        self.norm1 = nn.LayerNorm(config.micro_dim)
        self.norm2 = nn.LayerNorm(config.micro_dim)
        self.norm3 = nn.LayerNorm(config.micro_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        word_memory: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Masked Causal Self-Attention (intra-word bytes)
        res = x
        normed = self.norm1(x)
        sa_out, _ = self.self_attn(normed, normed, normed, attn_mask=self_attn_mask)
        x = res + self.dropout(sa_out)

        # 2. Cross-Attention over macro word latent
        res = x
        normed = self.norm2(x)
        ca_out, _ = self.cross_attn(normed, word_memory, word_memory)
        x = res + self.dropout(ca_out)

        # 3. 8-Expert MoE FFN with Top-2 routing
        res = x
        normed = self.norm3(x)
        ffn_out, aux_loss = self.moe_ffn(normed)
        x = res + self.dropout(ffn_out)

        return x, aux_loss


class MicroRecursiveByteHead(nn.Module):
    """Shared recursive 8-expert MoE micro-decoder that emits UTF-8 bytes conditioned on word latent z_w."""

    def __init__(self, config: HierarchicalByteConfig):
        super().__init__()
        self.config = config

        # Micro Byte Embedding (Tied with LM head)
        self.byte_embedding = nn.Embedding(
            config.byte_vocab_size,
            config.micro_dim,
            padding_idx=config.pad_token_id,
        )
        nn.init.normal_(self.byte_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.byte_embedding.weight[config.pad_token_id].fill_(0.0)

        # Macro Latent conditioning projection (identity if dimensions match)
        if config.macro_dim != config.micro_dim:
            self.word_cond_proj = nn.Linear(config.macro_dim, config.micro_dim)
        else:
            self.word_cond_proj = nn.Identity()

        # Positional embedding for intra-word byte positions (up to 64 bytes)
        self.pos_emb = nn.Parameter(torch.randn(1, 64, config.micro_dim) * 0.02)

        # 4 MoE Transformer layers with 8 experts each
        self.layers = nn.ModuleList([
            MicroMoEByteLayer(config) for _ in range(config.micro_layers)
        ])
        self.final_norm = nn.LayerNorm(config.micro_dim)

        # Output projection head (tied with byte_embedding)
        self.lm_head = nn.Linear(config.micro_dim, config.byte_vocab_size, bias=False)
        self.lm_head.weight = self.byte_embedding.weight

    def forward(
        self,
        byte_ids: torch.Tensor,
        word_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            byte_ids: [N, K] byte token IDs for N words, length K bytes
            word_latent: [N, 1, macro_dim] latent representations for the N words
        Returns:
            logits: [N, K, 261] byte prediction logits
            aux_loss: scalar MoE load balancing auxiliary loss
        """
        N, K = byte_ids.shape
        x = self.byte_embedding(byte_ids) + self.pos_emb[:, :K, :]

        mem = self.word_cond_proj(word_latent)  # [N, 1, micro_dim]

        # Causal self-attention mask for intra-word bytes
        causal_mask = torch.triu(torch.full((K, K), float("-inf"), device=byte_ids.device), diagonal=1)

        total_aux_loss = torch.tensor(0.0, device=byte_ids.device)
        for layer in self.layers:
            x, aux = layer(x, mem, self_attn_mask=causal_mask)
            total_aux_loss = total_aux_loss + aux

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, total_aux_loss


class PhonoV65HierarchicalByteDecoder(nn.Module):
    """Phono-V6.5: Hierarchical Word-to-Byte Recursive Denoising Decoder with 8-Expert MoE."""

    def __init__(self, config: Optional[HierarchicalByteConfig] = None):
        super().__init__()
        self.config = config or HierarchicalByteConfig()

        # Learned Macro Word Queries
        self.max_word_len = 128
        self.word_slot_embedding = nn.Parameter(torch.randn(1, self.max_word_len, self.config.macro_dim) * 0.02)

        # Macro MoE Transformer layers (4 layers, 4 experts)
        self.macro_layers = nn.ModuleList([
            MacroMoELayer(self.config) for _ in range(self.config.macro_layers)
        ])
        self.macro_norm = nn.LayerNorm(self.config.macro_dim)

        # Word Latent Denoising Refiner
        if self.config.use_word_diffusion:
            self.word_refiner = DeepLatentDiffusionRefiner(
                embed_dim=self.config.macro_dim,
                dropout=self.config.dropout,
            )
            self.noise_scheduler = GaussianNoiseScheduler(default_noise_max=self.config.word_noise_max)

        # Micro Recursive Byte Head (MoE)
        self.micro_byte_head = MicroRecursiveByteHead(self.config)

        # Macro Base-Token Fast-Path Classifier (0: SILENCE, 1: WORD, 2: EOS, 3: SPACE)
        if self.config.enable_macro_fastpath:
            self.macro_slot_classifier = nn.Linear(self.config.macro_dim, self.config.num_slot_classes)
        else:
            self.macro_slot_classifier = None

    def forward(
        self,
        acoustic_memory: torch.Tensor,
        memory_lengths: Optional[torch.Tensor] = None,
        num_words: Optional[torch.Tensor] = None,
        input_byte_ids: Optional[torch.Tensor] = None,
        target_byte_ids: Optional[torch.Tensor] = None,
        slot_targets: Optional[torch.Tensor] = None,
        expected_total_len: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            acoustic_memory: [B, T, 512] from frozen V6.4
            memory_lengths: [B] valid frame counts
            num_words: [B] number of words per utterance (determines L)
            input_byte_ids: [B, L, K] teacher-forced input byte IDs
            target_byte_ids: [B, L, K] target byte IDs for cross-entropy
            slot_targets: [B, L] discrete slot type labels (0: SILENCE, 1: WORD, 2: EOS, 3: SPACE)
        """
        B, T, _ = acoustic_memory.shape
        device = acoustic_memory.device

        if input_byte_ids is not None:
            L = input_byte_ids.shape[1]
        elif num_words is not None:
            L = num_words.max().item()
        else:
            L = max(2, int(T / 12.0))

        L = min(L, self.max_word_len)

        # 1. Macro Word Slot Queries
        x = self.word_slot_embedding[:, :L, :].expand(B, -1, -1)
        causal_mask = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)

        mem_pad_mask = None
        if memory_lengths is not None:
            t_idx = torch.arange(T, device=device).unsqueeze(0)
            mem_pad_mask = t_idx >= memory_lengths.unsqueeze(1)

        macro_aux_loss = torch.tensor(0.0, device=device)
        for layer in self.macro_layers:
            x, aux = layer(
                x,
                acoustic_memory,
                self_attn_mask=causal_mask,
                memory_padding_mask=mem_pad_mask,
                expected_total_len=expected_total_len or L,
            )
            macro_aux_loss = macro_aux_loss + aux

        z_word = self.macro_norm(x)  # [B, L, 512]

        loss = None
        diff_loss = torch.tensor(0.0, device=device)
        contrastive_loss = torch.tensor(0.0, device=device)
        slot_loss = torch.tensor(0.0, device=device)
        slot_logits = None
        logits = None
        byte_acc = torch.tensor(0.0, device=device)
        micro_aux_loss = torch.tensor(0.0, device=device)

        # 2. Macro Fast-Path Base-Token Detection
        if self.macro_slot_classifier is not None:
            slot_logits = self.macro_slot_classifier(z_word)  # [B, L, num_slot_classes]
            if slot_targets is not None:
                # Align lengths if needed
                st_L = slot_targets.shape[1]
                if st_L >= L:
                    st_slice = slot_targets[:, :L]
                    sl_slice = slot_logits
                else:
                    st_slice = slot_targets
                    sl_slice = slot_logits[:, :st_L]

                slot_loss = F.cross_entropy(
                    sl_slice.reshape(-1, self.config.num_slot_classes),
                    st_slice.reshape(-1),
                    ignore_index=-100,
                )

        # 3. Word Latent Diffusion Denoising Step
        if self.config.use_word_diffusion and self.training:
            noise_map, _, _ = self.noise_scheduler.compute_noise_map(B, L, device)
            eps = torch.randn_like(z_word)
            z_noisy = z_word + noise_map * eps
            z_clean_est = self.word_refiner(z_noisy, noise_map)
            diff_sq = (z_clean_est - z_word) ** 2
            diff_loss = diff_sq.mean()

        # 4. Contrastive Homophone Separation
        if self.config.contrastive_loss_weight > 0.0 and self.training and L > 1:
            normed_reps = F.normalize(z_word, dim=-1)
            sim_matrix = torch.matmul(normed_reps, normed_reps.transpose(-2, -1)) / self.config.contrastive_temperature
            contrastive_loss = -torch.diagonal(sim_matrix, dim1=-2, dim2=-1).mean() + torch.logsumexp(sim_matrix, dim=-1).mean()

        # 5. Micro Recursive Byte Emission with MoE Byte Head
        if input_byte_ids is not None and target_byte_ids is not None:
            _, _, K = input_byte_ids.shape
            flat_bytes = input_byte_ids.view(B * L, K)
            flat_z = z_word.view(B * L, 1, self.config.macro_dim)

            flat_logits, micro_aux_loss = self.micro_byte_head(flat_bytes, flat_z)  # [B * L, K, 261]
            logits = flat_logits.view(B, L, K, self.config.byte_vocab_size)

            # Compute Byte Cross-Entropy Loss
            flat_targets = target_byte_ids.view(-1)
            ce_loss = F.cross_entropy(
                flat_logits.view(-1, self.config.byte_vocab_size),
                flat_targets,
                ignore_index=self.config.pad_token_id,
            )

            # Byte accuracy metric
            valid_mask = flat_targets != self.config.pad_token_id
            if valid_mask.sum() > 0:
                preds = flat_logits.view(-1, self.config.byte_vocab_size).argmax(dim=-1)
                correct = (preds[valid_mask] == flat_targets[valid_mask]).float()
                byte_acc = correct.mean() * 100.0

            total_aux_loss = macro_aux_loss + micro_aux_loss
            loss = (
                ce_loss
                + 0.01 * total_aux_loss
                + self.config.diffusion_loss_weight * diff_loss
                + self.config.contrastive_loss_weight * contrastive_loss
            )
            if self.macro_slot_classifier is not None and slot_targets is not None:
                loss = loss + self.config.macro_slot_loss_weight * slot_loss

        return {
            "loss": loss,
            "logits": logits,
            "word_latents": z_word,
            "slot_logits": slot_logits,
            "slot_loss": slot_loss,
            "byte_acc": byte_acc,
            "aux_loss": macro_aux_loss + micro_aux_loss,
            "diff_loss": diff_loss,
            "contrastive_loss": contrastive_loss,
        }

    @torch.no_grad()
    def generate(
        self,
        acoustic_memory: torch.Tensor,
        memory_lengths: Optional[torch.Tensor] = None,
        max_words: Optional[int] = None,
        max_bytes_per_word: int = 16,
    ) -> List[List[List[int]]]:
        """Hierarchical Autoregressive Generation with Macro Fast-Path Early-Exit & Silence Bypass."""
        B, T, _ = acoustic_memory.shape
        device = acoustic_memory.device

        if max_words is None:
            max_words = max(2, min(self.max_word_len, int(T / 12.0)))

        # 1. Macro Word Latents
        x = self.word_slot_embedding[:, :max_words, :].expand(B, -1, -1)
        causal_mask = torch.triu(torch.full((max_words, max_words), float("-inf"), device=device), diagonal=1)

        mem_pad_mask = None
        if memory_lengths is not None:
            t_idx = torch.arange(T, device=device).unsqueeze(0)
            mem_pad_mask = t_idx >= memory_lengths.unsqueeze(1)

        for layer in self.macro_layers:
            x, _ = layer(
                x,
                acoustic_memory,
                self_attn_mask=causal_mask,
                memory_padding_mask=mem_pad_mask,
                expected_total_len=max_words,
            )
        z_words = self.macro_norm(x)  # [B, max_words, macro_dim]

        batch_results = []
        for b in range(B):
            words_for_utt = []
            for w in range(max_words):
                zw = z_words[b:b+1, w:w+1, :]  # [1, 1, macro_dim]

                # Macro Fast-Path Shortcut Routing
                if self.macro_slot_classifier is not None:
                    slot_logits_w = self.macro_slot_classifier(zw)  # [1, 1, num_classes]
                    slot_pred = slot_logits_w.argmax(dim=-1).item()
                    if slot_pred == SLOT_EOS:
                        # Instant Early Exit: Utterance finished, stop decoding
                        break
                    elif slot_pred == SLOT_SILENCE:
                        # Acoustic Silence Bypass: Skip Tier-2 micro byte head (0 byte FLOPs)
                        continue
                    elif slot_pred == SLOT_SPACE:
                        # Direct space emit without byte-level recurrence
                        words_for_utt.append([0x20])
                        continue

                # Unroll micro byte head for standard lexical words
                cur_bytes = torch.full((1, 1), self.config.bos_token_id, device=device, dtype=torch.long)
                emitted_word = []
                for _ in range(max_bytes_per_word):
                    logits, _ = self.micro_byte_head(cur_bytes, zw)  # [1, cur_k, 261]
                    next_byte = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
                    b_val = next_byte.item()

                    if b_val in (self.config.eow_token_id, self.config.eos_token_id, self.config.pad_token_id):
                        break
                    emitted_word.append(b_val)
                    cur_bytes = torch.cat([cur_bytes, next_byte], dim=1)

                if emitted_word:
                    words_for_utt.append(emitted_word)

            batch_results.append(words_for_utt)

        return batch_results
