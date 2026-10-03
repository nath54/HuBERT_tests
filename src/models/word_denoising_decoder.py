"""Phoneme-to-Word Denoising Decoder with Banded Cross-Attention & MoE Text Modeling.

Takes frozen frame latents or phoneme representations from an acoustic backbone (e.g. V6.4)
and performs non-autoregressive or semi-autoregressive word token decoding using:
1. Frozen Acoustic Backbone Interface: No gradient computation required on audio encoder.
2. Banded Cross-Attention: Local temporal receptive alignment between phoneme frames and word tokens.
3. 4-Expert MoE Transformer Decoder: Dispatches language modeling to specialized experts.
4. Word Latent Denoising Refiner: Differentiable FiLM-conditioned diffusion denoiser over word embeddings.
5. Contrastive Hard-Negative Objective: Separates phonetically identical / homophonic words (e.g. "their" vs "there").
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.phono_variants import MoEFeedForwardNetwork, DeepLatentDiffusionRefiner, GaussianNoiseScheduler


@dataclass
class WordDecoderConfig:
    """Configuration for Phoneme-to-Word Denoising Decoder."""
    # Vocabulary & Embedding
    vocab_size: int = 10000            # Word / subword vocabulary size
    word_embed_dim: int = 512          # Dimension of word embeddings
    acoustic_embed_dim: int = 512      # Dimension of incoming acoustic latents (from V6.4)
    pad_token_id: int = 0
    blank_token_id: int = 1
    unk_token_id: int = 2
    bos_token_id: int = 3
    eos_token_id: int = 4

    # Transformer Decoder layers
    decoder_layers: int = 4            # Number of MoE text decoder layers
    decoder_heads: int = 8             # Number of attention heads
    decoder_ffn_dim: int = 1536        # FFN intermediate expansion
    num_experts: int = 4               # 4-expert MoE FFN
    moe_top_k: int = 2                 # Top-2 expert dispatch
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Banded Cross-Attention
    cross_attn_band_width: int = 32    # Frame receptive window for acoustic alignment (~640ms)

    # Word Diffusion Refiner
    use_word_diffusion: bool = True
    word_diffusion_steps: int = 3
    word_noise_max: float = 0.5
    diffusion_loss_weight: float = 0.5

    # Contrastive Hard-Negative Loss
    contrastive_temperature: float = 0.07
    contrastive_loss_weight: float = 0.2


class BandedCrossAttention(nn.Module):
    """Local banded cross-attention aligning text query positions to local acoustic frames."""

    def __init__(self, query_dim: int, key_dim: int, num_heads: int = 8, band_width: int = 32, dropout: float = 0.1):
        super().__init__()
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.band_width = band_width
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        expected_total_len: Optional[int] = None,
        return_attn_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            query: [B, L_words, D] text token queries
            key_value: [B, T_audio, D_audio] acoustic representations
            key_padding_mask: [B, T_audio] boolean mask where True indicates padding
            expected_total_len: Optional estimated full word length (for autoregressive generation)
            return_attn_weights: If True, return (output, attn_weights [B, L, T])
        """
        B, L, _ = query.shape
        _, T, _ = key_value.shape

        q = self.q_proj(query).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling  # [B, H, L, T]

        # Monotonic linear center alignment: token position l maps proportionally to time position t
        if self.band_width > 0 and T > self.band_width:
            total_l = expected_total_len if expected_total_len is not None else L
            l_idx = torch.arange(L, device=query.device, dtype=torch.float32).unsqueeze(1)  # [L, 1]
            t_idx = torch.arange(T, device=query.device, dtype=torch.float32).unsqueeze(0)  # [1, T]
            expected_t = (l_idx / max(1, total_l - 1)) * max(1, T - 1)
            dist = (t_idx - expected_t).abs()
            band_mask = dist > self.band_width
            attn_scores = attn_scores.masked_fill(band_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        # Handle all-masked edge cases
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
        attn_dropped = self.dropout(attn_probs)

        out = torch.matmul(attn_dropped, v).transpose(1, 2).contiguous().view(B, L, -1)
        proj_out = self.out_proj(out)
        if return_attn_weights:
            return proj_out, attn_probs.mean(dim=1)
        return proj_out


class MoEWordDecoderLayer(nn.Module):
    """Word Decoder Layer with Self-Attention, Banded Cross-Attention, and MoE FFN."""

    def __init__(self, config: WordDecoderConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.word_embed_dim,
            num_heads=config.decoder_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.cross_attn = BandedCrossAttention(
            query_dim=config.word_embed_dim,
            key_dim=config.acoustic_embed_dim,
            num_heads=config.decoder_heads,
            band_width=config.cross_attn_band_width,
            dropout=config.attention_dropout,
        )
        self.moe_ffn = MoEFeedForwardNetwork(
            embed_dim=config.word_embed_dim,
            ffn_dim=config.decoder_ffn_dim,
            num_experts=config.num_experts,
            top_k=config.moe_top_k,
            dropout=config.dropout,
        )
        self.norm1 = nn.LayerNorm(config.word_embed_dim)
        self.norm2 = nn.LayerNorm(config.word_embed_dim)
        self.norm3 = nn.LayerNorm(config.word_embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        acoustic_memory: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
        expected_total_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Masked Self-Attention
        res = x
        normed = self.norm1(x)
        sa_out, _ = self.self_attn(normed, normed, normed, attn_mask=self_attn_mask)
        x = res + self.dropout(sa_out)

        # 2. Banded Acoustic Cross-Attention
        res = x
        normed = self.norm2(x)
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

        return x, aux_loss


class WordDenoisingDecoder(nn.Module):
    """Full Downstream Phoneme-to-Word Denoising Decoder with MoE text backbone."""

    def __init__(self, config: Optional[WordDecoderConfig] = None):
        super().__init__()
        self.config = config or WordDecoderConfig()

        self.word_embedding = nn.Embedding(self.config.vocab_size, self.config.word_embed_dim, padding_idx=self.config.pad_token_id)
        nn.init.normal_(self.word_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.word_embedding.weight[self.config.pad_token_id].fill_(0.0)

        self.pos_embedding = nn.Parameter(torch.randn(1, 512, self.config.word_embed_dim) * 0.02)

        self.layers = nn.ModuleList([
            MoEWordDecoderLayer(self.config) for _ in range(self.config.decoder_layers)
        ])
        self.final_norm = nn.LayerNorm(self.config.word_embed_dim)
        self.lm_head = nn.Linear(self.config.word_embed_dim, self.config.vocab_size, bias=False)

        # Tie weights between embedding and output projection
        self.lm_head.weight = self.word_embedding.weight

        # Word Latent Denoising Refiner
        if self.config.use_word_diffusion:
            self.word_refiner = DeepLatentDiffusionRefiner(
                embed_dim=self.config.word_embed_dim,
                dropout=self.config.dropout,
            )
            self.noise_scheduler = GaussianNoiseScheduler(default_noise_max=self.config.word_noise_max)

    def forward(
        self,
        word_ids: torch.Tensor,
        acoustic_memory: torch.Tensor,
        memory_lengths: Optional[torch.Tensor] = None,
        target_word_ids: Optional[torch.Tensor] = None,
        expected_total_len: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            word_ids: [B, L] input text token IDs
            acoustic_memory: [B, T, D] acoustic representations from frozen V6.4
        """
        B, L = word_ids.shape
        _, T, _ = acoustic_memory.shape

        # Text embedding + positional encoding
        x = self.word_embedding(word_ids) + self.pos_embedding[:, :L, :]

        # Causal auto-regressive / upper-triangular self-attention mask
        causal_mask = torch.triu(torch.full((L, L), float("-inf"), device=word_ids.device), diagonal=1)

        mem_pad_mask = None
        if memory_lengths is not None:
            t_idx = torch.arange(T, device=word_ids.device).unsqueeze(0)
            mem_pad_mask = t_idx >= memory_lengths.unsqueeze(1)

        total_aux_loss = torch.tensor(0.0, device=word_ids.device)
        for layer in self.layers:
            x, aux = layer(
                x,
                acoustic_memory,
                self_attn_mask=causal_mask,
                memory_padding_mask=mem_pad_mask,
                expected_total_len=expected_total_len,
            )
            total_aux_loss = total_aux_loss + aux

        x = self.final_norm(x)

        # Baseline text logits
        logits = self.lm_head(x)

        loss = None
        diff_loss = torch.tensor(0.0, device=word_ids.device)
        contrastive_loss = torch.tensor(0.0, device=word_ids.device)

        if target_word_ids is not None:
            ce_loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                target_word_ids.view(-1),
                ignore_index=self.config.pad_token_id,
            )

            # Word latent diffusion denoising
            if self.config.use_word_diffusion and self.training:
                noise_map, _, _ = self.noise_scheduler.compute_noise_map(B, L, word_ids.device)
                eps = torch.randn_like(x)
                x_noisy = x + noise_map * eps
                x_clean_est = self.word_refiner(x_noisy, noise_map)
                diff_sq = (x_clean_est - x) ** 2
                diff_loss = diff_sq.mean()

            # Contrastive Hard-Negative Loss (homophone separation)
            if self.config.contrastive_loss_weight > 0.0 and self.training and L > 1:
                # Contrast consecutive token representations against other positions
                normed_reps = F.normalize(x, dim=-1)  # [B, L, D]
                sim_matrix = torch.matmul(normed_reps, normed_reps.transpose(-2, -1)) / self.config.contrastive_temperature  # [B, L, L]
                # Target is the identity self-similarity, scaled by temperature
                contrastive_loss = -torch.diagonal(sim_matrix, dim1=-2, dim2=-1).mean() + torch.logsumexp(sim_matrix, dim=-1).mean()

            loss = (
                ce_loss
                + 0.01 * total_aux_loss
                + self.config.diffusion_loss_weight * diff_loss
                + self.config.contrastive_loss_weight * contrastive_loss
            )

        return {
            "loss": loss,
            "logits": logits,
            "latent_state": x,
            "aux_loss": total_aux_loss,
            "diff_loss": diff_loss,
            "contrastive_loss": contrastive_loss,
        }

    @torch.no_grad()
    def generate(
        self,
        acoustic_memory: torch.Tensor,
        memory_lengths: Optional[torch.Tensor] = None,
        max_len: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive greedy generation given acoustic representations."""
        B, T, _ = acoustic_memory.shape
        device = acoustic_memory.device
        
        # Estimate expected target word length based on acoustic frame count (~12-15 frames per word)
        expected_len = max(5, int(T / 12.0))
        
        cur_tokens = torch.full((B, 1), self.config.bos_token_id, device=device, dtype=torch.long)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            out = self.forward(
                word_ids=cur_tokens,
                acoustic_memory=acoustic_memory,
                memory_lengths=memory_lengths,
                expected_total_len=expected_len,
            )
            next_logits = out["logits"][:, -1, :]
            if temperature > 0.0 and temperature != 1.0:
                next_logits = next_logits / temperature
            next_tokens = next_logits.argmax(dim=-1, keepdim=True)
            
            # Mask out already finished sequences
            next_tokens = torch.where(finished.unsqueeze(1), torch.full_like(next_tokens, self.config.pad_token_id), next_tokens)
            cur_tokens = torch.cat([cur_tokens, next_tokens], dim=1)
            
            finished = finished | (next_tokens.squeeze(1) == self.config.eos_token_id)
            if finished.all():
                break

        return cur_tokens
