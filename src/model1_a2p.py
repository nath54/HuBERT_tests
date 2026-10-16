"""
Model Part 1: Audio to Phonemes (Model_A2P).
Translates raw 16kHz speech waveforms into discrete phoneme sequences with acoustic pause delimiters (token 8).
Architecture:
- 7-layer 1D CNN Acoustic Feature Extractor (320x downsampling -> 50 Hz frame rate)
- Linear Feature Projection + LayerNorm
- Conformer Backbone with DeepSeek Mixture-of-Experts (1 Shared + 8 Routed, Top-2)
- Layer 4 Draft Phoneme Auxiliary Supervision
- Multi-Scale Phoneme Engram Memory Injection
- Hierarchical Recursive Phoneme Head (104 classes + 8 articulatory groups)
- Decoupled Word Boundary Gate Head (3 classes: Speech, Space, Silence)
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .model_config import ModelConfig, ModelA2PConfig
from .model_part import Model_Part
from .layers.conv_modules import AudioCNNFeatureExtractor
from .layers.conformer_layers import ConformerEngramMoEEncoder, ConformerMoETransformerLayer
from .layers.engram_memory import PhonemeEngramMemory
from .layers.heads import (
    DecoupledBoundaryGateHead,
    HierarchicalRecursivePhonemeHead,
    DeepLatentDiffusionRefiner,
    GaussianNoiseScheduler,
    decode_calibrated_phonemes,
    decode_top_k_calibrated_phonemes,
    PHONEME_VOCAB_SIZE,
    NUM_ARTICULATORY_GROUPS,
)


class Model_A2P(Model_Part):
    """
    Modular Audio to Phonemes (A2P) Model Architecture.
    Combines 1D CNN acoustic feature extraction with a Conformer MoE Engram backbone.
    """

    def __init__(self, model_config: Optional[ModelConfig] = None) -> None:
        if model_config is None:
            model_config = ModelConfig()
        super().__init__(model_config=model_config, model_part="A2P")

        cfg: ModelA2PConfig = model_config.a2p
        self.cfg: ModelA2PConfig = cfg

        self.vocab_size: int = PHONEME_VOCAB_SIZE  # 112
        self.phoneme_offset: int = 8
        self.num_phonemes: int = self.vocab_size - self.phoneme_offset  # 104
        self.blank_id: int = 1
        self.space_id: int = 8
        self.pad_id: int = 0

        # 1. 7-layer CNN Feature Extractor (320x temporal downsampling)
        self.feature_extractor: AudioCNNFeatureExtractor = AudioCNNFeatureExtractor(
            in_channels=cfg.in_channels,
            conv_channels=cfg.conv_channels,
            dropout=cfg.dropout,
        )

        # 2. Linear Feature Projection
        self.feature_projection: nn.Sequential = nn.Sequential(
            nn.LayerNorm(cfg.conv_channels),
            nn.Linear(cfg.conv_channels, cfg.embed_dim),
            nn.Dropout(cfg.dropout),
        )

        # 3. Conformer Backbone with DeepSeek MoE & Engram Memory
        self.encoder: ConformerEngramMoEEncoder = ConformerEngramMoEEncoder(
            embed_dim=cfg.embed_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            ffn_dim=cfg.ffn_dim,
            num_experts=cfg.num_routed_experts,
            top_k=cfg.top_k,
            use_deepseek_moe=cfg.use_deepseek_moe,
            num_shared_experts=cfg.num_shared_experts,
            kernel_size=cfg.conv_kernel_size,
            dropout=cfg.dropout,
            attention_dropout=cfg.dropout,
            pos_conv_kernel=128,
            pos_conv_groups=16,
            use_multi_scale_conv=True,
            vocab_size=self.vocab_size,
            draft_layer_idx=cfg.draft_layer_idx,
            engram_layers=cfg.engram_layers,
            engram_heads=cfg.engram_heads,
            engram_memory_dim=cfg.engram_memory_dim,
            gradient_checkpointing=cfg.gradient_checkpointing,
        )

        # 4. Decoupled Word Boundary Gate Head
        self.boundary_gate_head: DecoupledBoundaryGateHead = DecoupledBoundaryGateHead(cfg.embed_dim)

        # 5. Hierarchical Recursive Phoneme Head
        self.phoneme_head: HierarchicalRecursivePhonemeHead = HierarchicalRecursivePhonemeHead(
            embed_dim=cfg.embed_dim,
            vocab_size=self.num_phonemes,
            num_macro_classes=NUM_ARTICULATORY_GROUPS,
        )

        # 6. Diffusion Gating & Latent Refinement Modules
        self.state_router: nn.Linear = nn.Linear(cfg.embed_dim, 4)
        self.latent_refiner: DeepLatentDiffusionRefiner = DeepLatentDiffusionRefiner(cfg.embed_dim)
        self.gate_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(cfg.embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 3),
        )

        # 7. CTC Loss Functions
        self.ctc_loss_fn: nn.CTCLoss = nn.CTCLoss(blank=self.blank_id, zero_infinity=True)
        self.macro_ctc_loss_fn: nn.CTCLoss = nn.CTCLoss(blank=0, zero_infinity=True)
        self.draft_ctc_loss_fn: nn.CTCLoss = nn.CTCLoss(blank=self.blank_id, zero_infinity=True)
        self.noise_scheduler: GaussianNoiseScheduler = GaussianNoiseScheduler()

        # EMA routing buffers
        self.register_buffer("_gate_bypass_ema", torch.tensor(0.5))
        self.register_buffer("_gate_partial_ema", torch.tensor(0.3))
        self.register_buffer("_gate_full_ema", torch.tensor(0.2))

    def adapt_architecture_to_checkpoint(self, sd: Dict[str, Tensor]) -> None:
        """Adapts layer count and engram modules to match pretrained checkpoint."""
        ckpt_layers: Set[int] = set()
        ckpt_engram_layers: Set[int] = set()

        for k in sd.keys():
            if k.startswith("encoder.layers."):
                ckpt_layers.add(int(k.split(".")[2]))
            elif "encoder.engram_modules.layer_" in k:
                seg = k.split("encoder.engram_modules.layer_")[1].split(".")[0]
                ckpt_engram_layers.add(int(seg))

        if ckpt_layers:
            num_l = max(ckpt_layers) + 1
            if len(self.encoder.layers) != num_l:
                cfg = self.cfg
                self.encoder.layers = nn.ModuleList([
                    ConformerMoETransformerLayer(
                        embed_dim=cfg.embed_dim,
                        num_heads=cfg.num_heads,
                        ffn_dim=cfg.ffn_dim,
                        num_experts=cfg.num_routed_experts,
                        top_k=cfg.top_k,
                        kernel_size=cfg.conv_kernel_size,
                        dropout=cfg.dropout,
                        attention_dropout=cfg.dropout,
                        use_multi_scale_conv=True,
                        use_deepseek_moe=cfg.use_deepseek_moe,
                        num_shared_experts=cfg.num_shared_experts,
                    )
                    for _ in range(num_l)
                ])

        if ckpt_engram_layers:
            cfg = self.cfg
            self.encoder.engram_layers = ckpt_engram_layers
            self.encoder.engram_modules = nn.ModuleDict({
                f"layer_{l_idx}": PhonemeEngramMemory(
                    embed_dim=cfg.embed_dim,
                    vocab_size=self.vocab_size,
                    num_heads=cfg.engram_heads,
                    memory_dim=cfg.engram_memory_dim,
                    dropout=cfg.dropout,
                )
                for l_idx in sorted(ckpt_engram_layers)
            })

    def compute_composite_log_probs(self, state_logits: Tensor, phoneme_logits: Tensor) -> Tensor:
        """Form normalized composite full-vocabulary log-probabilities from hierarchical outputs."""
        B, T_frames, _ = state_logits.shape
        log_p_state = F.log_softmax(state_logits, dim=-1)
        log_p_phoneme = F.log_softmax(phoneme_logits, dim=-1)

        composite_log_probs = torch.full(
            (B, T_frames, self.vocab_size),
            -100.0,
            device=state_logits.device,
            dtype=state_logits.dtype,
        )
        # Non-speech acoustic state emissions
        composite_log_probs[:, :, self.blank_id] = log_p_state[:, :, 0]
        composite_log_probs[:, :, 3] = log_p_state[:, :, 1]  # silence
        composite_log_probs[:, :, 4] = log_p_state[:, :, 2]  # noise

        # Linguistic speech phoneme emissions: log P(phoneme) = log P(speech) + log P(phoneme | speech)
        speech_log_p = log_p_state[:, :, 3].unsqueeze(-1)
        composite_log_probs[:, :, self.phoneme_offset:] = speech_log_p + log_p_phoneme
        return composite_log_probs

    def apply_specaugment(
        self,
        features: Tensor,
        time_mask_prob: float = 0.15,
        time_mask_len: int = 5,
        channel_mask_prob: float = 0.08,
    ) -> Tensor:
        """Dynamic SpecAugment (Time-span & Channel masking) applied to projected acoustic features."""
        if not self.training:
            return features

        B, T, D = features.shape
        masked = features.clone()

        # 1. Time Span Masking (5 frames = 100ms at 50Hz)
        num_mask = int(time_mask_prob * T / max(1, time_mask_len))
        if num_mask > 0 and T > time_mask_len:
            for b in range(B):
                starts = torch.randint(0, T - time_mask_len + 1, (num_mask,), device=features.device)
                for s in starts:
                    masked[b, s : s + time_mask_len] = 0.0

        # 2. Channel Feature Masking
        ch_count = max(1, int(D * channel_mask_prob))
        for b in range(B):
            ch_idx = torch.randperm(D, device=features.device)[:ch_count]
            masked[b, :, ch_idx] = 0.0

        return masked

    def forward(
        self,
        audio: Tensor,
        audio_lengths: Optional[Tensor] = None,
        targets: Optional[Tensor] = None,
        target_lengths: Optional[Tensor] = None,
        macro_targets: Optional[Tensor] = None,
        macro_lengths: Optional[Tensor] = None,
        boundary_targets: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Unified forward pass executing acoustic feature extraction and conformer backbone."""
        cnn_features = self.feature_extractor(audio)
        x = self.feature_projection(cnn_features)

        if self.training:
            x = self.apply_specaugment(x)

        encoder_out = self.encoder(
            x,
            teacher_tokens=targets,
            output_hidden_states=False,
            output_attentions=False,
        )
        h = encoder_out["last_hidden_state"]

        # Word boundary gate
        boundary_logits = self.boundary_gate_head(h)

        # Hierarchical Phoneme Head
        refined_phonemes, base_phonemes, macro_logits = self.phoneme_head(h)
        state_logits = self.state_router(h)

        full_logits = self.compute_composite_log_probs(state_logits, refined_phonemes)
        boundary_probs = F.softmax(boundary_logits, dim=-1)

        out: Dict[str, Any] = {
            "logits": full_logits,
            "refined_phonemes": refined_phonemes,
            "base_phonemes": base_phonemes,
            "macro_logits": macro_logits,
            "boundary_logits": boundary_logits,
            "boundary_probs": boundary_probs,
            "draft_logits": encoder_out.get("draft_logits"),
            "aux_loss": encoder_out.get("aux_loss", torch.tensor(0.0, device=h.device)),
            "last_hidden_state": h,
        }

        # Compute CTC loss during training
        if targets is not None and target_lengths is not None:
            B, T = full_logits.shape[:2]
            log_probs = F.log_softmax(full_logits, dim=-1).transpose(0, 1)  # [T, B, V]
            if audio_lengths is not None:
                # Downsample 320x to get frame lengths
                input_lengths = torch.clamp(audio_lengths // 320, min=1, max=T)
            else:
                input_lengths = torch.full((B,), T, dtype=torch.long, device=h.device)
            ctc_loss = self.ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
            out["loss"] = ctc_loss + 0.1 * out["aux_loss"]
            out["ctc_loss"] = ctc_loss

        return out

    def forward_training(
        self,
        audio: Tensor,
        audio_lengths: Optional[Tensor] = None,
        targets: Optional[Tensor] = None,
        target_lengths: Optional[Tensor] = None,
        macro_targets: Optional[Tensor] = None,
        macro_lengths: Optional[Tensor] = None,
        boundary_targets: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        return self.forward(
            audio=audio,
            audio_lengths=audio_lengths,
            targets=targets,
            target_lengths=target_lengths,
            macro_targets=macro_targets,
            macro_lengths=macro_lengths,
            boundary_targets=boundary_targets,
            **kwargs,
        )

    @torch.no_grad()
    def forward_inference(
        self,
        audio: Tensor,
        blank_penalty: float = 2.6,
        space_bonus: float = 0.35,
        min_tokens_between_spaces: int = 1,
    ) -> Dict[str, Any]:
        self.eval()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        blank_penalty = 2.6 if blank_penalty is None else float(blank_penalty)
        space_bonus = 0.35 if space_bonus is None else float(space_bonus)

        out = self.forward(audio)
        logits = out["logits"]
        boundary_probs = out.get("boundary_probs")

        pred_tokens_batch: List[List[int]] = []
        word_slices_batch: List[List[List[int]]] = []

        for b in range(logits.shape[0]):
            cal_ids = decode_calibrated_phonemes(
                logits[b],
                blank_id=self.blank_id,
                pad_id=self.pad_id,
                space_id=self.space_id,
                blank_penalty=blank_penalty,
                space_bonus=space_bonus,
                min_tokens_between_spaces=min_tokens_between_spaces,
            )
            pred_tokens_batch.append(cal_ids)

            word_slices: List[List[int]] = []
            curr_word: List[int] = []
            for t_id in cal_ids:
                if t_id == self.space_id:
                    if curr_word:
                        word_slices.append(curr_word)
                        curr_word = []
                elif t_id != self.blank_id and t_id != self.pad_id:
                    curr_word.append(t_id)
            if curr_word:
                word_slices.append(curr_word)

            word_slices_batch.append(word_slices)

        return {
            "logits": logits,
            "pred_tokens": pred_tokens_batch,
            "word_slices": word_slices_batch,
            "boundary_probs": boundary_probs,
            "draft_logits": out.get("draft_logits"),
            "macro_logits": out.get("macro_logits"),
        }

    def load_pretrained_checkpoint(self, checkpoint_path: str) -> bool:
        if not os.path.isfile(checkpoint_path):
            return False

        device: torch.device = self.get_device()
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = ckpt.get("acoustic_state_dict", ckpt.get("model_state_dict", ckpt))

        # Adapt architecture to match checkpoint layers if needed
        self.adapt_architecture_to_checkpoint(sd)
        self.to(device)

        own_state = self.state_dict()
        filtered_sd = {}
        for k, v in sd.items():
            if k in own_state and own_state[k].shape == v.shape:
                filtered_sd[k] = v

        res = self.load_state_dict(filtered_sd, strict=False)
        return len(filtered_sd) > 0
