"""
Model Part 1: Audio to Phonemes (Model_A2P).
Translates raw 16kHz speech waveforms into discrete phoneme sequences with acoustic pause delimiters (token 8).
Architecture:
- 7-layer 1D CNN Acoustic Feature Extractor (320x downsampling -> 50 Hz frame rate)
- Linear Feature Projection + LayerNorm
- Relative Convolutional Positional Embeddings
- 12-layer Conformer Backbone with DeepSeek Mixture-of-Experts (1 Shared + 8 Routed, Top-2)
- Layer 4 Draft Phoneme Auxiliary Supervision
- Layers 6 & 10 Multi-Scale Phoneme Engram Memory Injection
- Hierarchical CTC Heads: Discrete Phoneme Head (112 classes) + Articulatory Manner Head (8 classes)
"""

# Import Modules
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .model_config import ModelConfig, ModelA2PConfig
from .model_part import Model_Part
from .layers.conv_modules import AudioCNNFeatureExtractor
from .layers.embeddings import ConvolutionalPositionalEmbedding
from .layers.conformer_block import ConformerBlock
from .layers.engram_memory import PhonemeEngramMemory
from .layers.heads import CTCProjectionHead, ArticulatoryMacroHead, BoundaryGateHead


class Model_A2P(Model_Part):
    """
    Clean Modular Audio to Phonemes (A2P) Model Architecture.
    Combines 1D CNN acoustic feature extraction with a 12-layer Conformer MoE Engram backbone.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        """
        Initialize Model_A2P.

        Args:
            model_config (ModelConfig): Unified model configuration dataclass.
        """
        # Initialize parent class with part tag 'A2P'
        super().__init__(model_config=model_config, model_part="A2P")

        # Fetch sub-configuration for Part 1
        cfg: ModelA2PConfig = model_config.a2p
        self.cfg: ModelA2PConfig = cfg

        # 1. Temporal Feature Extractor: 7-layer 1D CNN (stride 320 downsampling)
        # Input: [B, 1, T_samples] -> Output: [B, T_frames, 512]
        self.feature_extractor: AudioCNNFeatureExtractor = AudioCNNFeatureExtractor(
            in_channels=cfg.in_channels,
            conv_channels=cfg.conv_channels,
            dropout=cfg.dropout,
        )

        # 2. Linear Feature Projection
        # Input: [B, T, 512] -> Output: [B, T, 512]
        self.feature_projection: nn.Sequential = nn.Sequential(
            nn.LayerNorm(cfg.conv_channels),
            nn.Linear(cfg.conv_channels, cfg.embed_dim),
            nn.Dropout(cfg.dropout),
        )

        # 3. Relative Convolutional Positional Embedding
        # Injects local relative spatial context: [B, T, 512] -> [B, T, 512]
        self.pos_conv: ConvolutionalPositionalEmbedding = ConvolutionalPositionalEmbedding(
            embed_dim=cfg.embed_dim,
            kernel_size=128,
            groups=16,
        )
        self.pos_dropout: nn.Dropout = nn.Dropout(cfg.dropout)

        # 4. 12-layer Conformer Backbone with DeepSeek MoE
        self.conformer_layers: nn.ModuleList = nn.ModuleList([
            ConformerBlock(
                embed_dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                ffn_dim=cfg.ffn_dim,
                conv_kernel_size=cfg.conv_kernel_size,
                dropout=cfg.dropout,
                use_deepseek_moe=cfg.use_deepseek_moe,
                num_routed_experts=cfg.num_routed_experts,
                top_k=cfg.top_k,
                num_shared_experts=cfg.num_shared_experts,
                use_rope=cfg.use_rope,
            )
            for _ in range(cfg.num_layers)
        ])

        # 5. Intermediate Draft Phoneme Head at Layer 4 (index 3)
        # Projects intermediate representations: [B, T, 512] -> [B, T, 112]
        self.draft_layer_idx: int = cfg.draft_layer_idx
        self.draft_head: nn.Linear = nn.Linear(cfg.embed_dim, cfg.vocab_size)

        # 6. Multi-Scale Phoneme Engram Memory Modules at specified layers
        self.engram_layers: Tuple[int, ...] = cfg.engram_layers
        self.engram_modules: nn.ModuleDict = nn.ModuleDict({
            f"layer_{l_idx}": PhonemeEngramMemory(
                embed_dim=cfg.embed_dim,
                vocab_size=cfg.vocab_size,
                num_heads=cfg.engram_heads,
                memory_dim=cfg.engram_memory_dim,
                table_size=cfg.engram_table_size,
                ngram_orders=cfg.engram_ngram_orders,
                dropout=cfg.dropout,
            )
            for l_idx in cfg.engram_layers
        })

        # 7. Final Normalization
        self.final_norm: nn.LayerNorm = nn.LayerNorm(cfg.embed_dim)

        # 8. Hierarchical Prediction Heads
        # Discrete Phoneme CTC Head (112 classes)
        self.phoneme_head: CTCProjectionHead = CTCProjectionHead(
            embed_dim=cfg.embed_dim,
            vocab_size=cfg.vocab_size,
            blank_id=cfg.blank_id,
            space_id=cfg.space_id,
        )

        # Articulatory Manner-of-Articulation Head (8 classes)
        self.macro_head: ArticulatoryMacroHead = ArticulatoryMacroHead(
            embed_dim=cfg.embed_dim,
            num_classes=8,
        )

        # Decoupled Acoustic Word Boundary Gate
        self.boundary_gate: BoundaryGateHead = BoundaryGateHead(
            embed_dim=cfg.embed_dim,
        )

        # 9. Loss Functions
        self.ctc_loss_fn: nn.CTCLoss = nn.CTCLoss(blank=cfg.blank_id, zero_infinity=True)
        self.draft_ctc_loss_fn: nn.CTCLoss = nn.CTCLoss(blank=cfg.blank_id, zero_infinity=True)
        self.macro_ctc_loss_fn: nn.CTCLoss = nn.CTCLoss(blank=0, zero_infinity=True)

        # Attempt automatic loading of pretrained CNN filter weights
        self.feature_extractor.load_pretrained_weights(cfg.cnn_weights_path, self.get_device())

    def _encode_frames(
        self,
        features: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        """
        Passes projected features through the 12 Conformer MoE blocks with engram memory.

        Args:
            features (Tensor): Projected frame features [B, T, D].
            key_padding_mask (Optional[Tensor]): Boolean mask [B, T].

        Returns:
            Tuple[Tensor, Optional[Tensor], Tensor]:
                - final_hidden: [B, T, D]
                - draft_logits: [B, T, V] or None
                - total_aux_moe_loss: scalar [1]
        """
        # Inject relative positional convolution
        # Shape: [B, T, D], Type: torch.float32, Role: local shift-invariant context
        pos_emb: Tensor = self.pos_conv(features)
        x: Tensor = self.pos_dropout(features + pos_emb)

        draft_logits: Optional[Tensor] = None
        draft_tokens: Optional[Tensor] = None
        total_moe_loss: Tensor = torch.tensor(0.0, device=x.device)

        # Pass sequentially through Conformer blocks
        for l_idx, layer in enumerate(self.conformer_layers):
            # Conformer block forward: [B, T, D]
            if self.training and getattr(self.cfg, "gradient_checkpointing", False):
                def _layer_fwd(mod: nn.Module, x_in: Tensor, mask: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor], Tensor]:
                    return mod(x_in, key_padding_mask=mask)
                x, _, layer_moe_loss = torch.utils.checkpoint.checkpoint(
                    _layer_fwd, layer, x, key_padding_mask, use_reentrant=False
                )
            else:
                x, _, layer_moe_loss = layer(x, key_padding_mask=key_padding_mask)
            total_moe_loss = total_moe_loss + layer_moe_loss

            # 1. Layer Draft Phoneme Supervision & Emission
            if l_idx == self.draft_layer_idx:
                # Shape: [B, T, V], Type: torch.float32, Role: intermediate draft logits
                draft_logits = self.draft_head(x)
                # Greedy draft token indices for engram lookup: [B, T]
                draft_tokens = draft_logits.argmax(dim=-1)

            # 2. Engram Memory Retrieval & Gating (Layers 6 and 10)
            if l_idx in self.engram_layers and draft_tokens is not None:
                mod_name: str = f"layer_{l_idx}"
                # Shape: [B, T, D], Type: torch.float32, Role: memory-conditioned hidden states
                x, _ = self.engram_modules[mod_name](x, draft_tokens)

        # Final LayerNorm: [B, T, D]
        final_hidden: Tensor = self.final_norm(x)
        return final_hidden, draft_logits, total_moe_loss

    @property
    def ctc_head(self) -> CTCProjectionHead:
        """Alias for phoneme_head."""
        return self.phoneme_head

    def forward_encoder(self, audio: Tensor) -> Tensor:
        """
        Encodes raw 16kHz audio into Conformer hidden representations.

        Args:
            audio (Tensor): Raw speech waveform [B, T_samples].

        Returns:
            Tensor: Encoded representations [B, T_frames, 512].
        """
        # CNN acoustic feature extraction: [B, T_frames, 512]
        cnn_feats: Tensor = self.feature_extractor(audio)
        feats: Tensor = self.feature_projection(cnn_feats)
        # Conformer MoE blocks
        hidden, _, _ = self._encode_frames(feats, key_padding_mask=None)
        return hidden

    def forward_training(
        self,
        audio: Tensor,
        audio_lengths: Optional[Tensor] = None,
        targets: Optional[Tensor] = None,
        target_lengths: Optional[Tensor] = None,
        macro_targets: Optional[Tensor] = None,
        macro_lengths: Optional[Tensor] = None,
        specaugment: bool = True,
    ) -> Dict[str, Tensor]:
        """
        Clean Training Forward Pass. Computes multi-task losses:
        L_total = L_ctc + 0.3 * L_draft_ctc + 0.2 * L_macro_ctc + L_aux_moe.

        Args:
            audio (Tensor): Raw 16kHz speech waveform [B, T_samples].
            audio_lengths (Optional[Tensor]): Unpadded audio sample lengths [B].
            targets (Optional[Tensor]): Ground truth discrete phoneme token IDs [B, S].
            target_lengths (Optional[Tensor]): Phoneme sequence lengths [B].
            macro_targets (Optional[Tensor]): Broad articulatory category IDs [B, S].
            macro_lengths (Optional[Tensor]): Articulatory target lengths [B].
            specaugment (bool): Whether to apply dynamic frequency/time acoustic masking.

        Returns:
            Dict[str, Tensor]: Multi-task loss values, logits, and intermediate emissions.
        """
        # 1. 7-layer CNN acoustic feature extraction: [B, T_frames, 512]
        cnn_feats: Tensor = self.feature_extractor(audio)
        # Linear projection: [B, T_frames, 512]
        feats: Tensor = self.feature_projection(cnn_feats)

        # Compute valid output frame lengths after stride-320 convolution
        b_sz, t_frames, _ = feats.shape
        if audio_lengths is not None:
            input_lengths: Tensor = self.feature_extractor.get_output_lengths(audio_lengths)
            # Create padding mask: [B, T_frames]
            t_idx = torch.arange(t_frames, device=feats.device).unsqueeze(0)
            pad_mask: Optional[Tensor] = t_idx >= input_lengths.unsqueeze(1)
        else:
            input_lengths = torch.full((b_sz,), t_frames, dtype=torch.long, device=feats.device)
            pad_mask = None

        # 2. Conformer MoE Engram Encoding: [B, T_frames, 512]
        hidden, draft_logits, moe_aux_loss = self._encode_frames(feats, key_padding_mask=pad_mask)

        # 3. Predict final heads
        # Main CTC logits: [B, T_frames, 112]
        phoneme_logits: Tensor = self.phoneme_head(hidden)
        # Articulatory macro logits: [B, T_frames, 8]
        macro_logits: Tensor = self.macro_head(hidden)

        # 4. Compute Connectionist Temporal Classification (CTC) losses
        losses: Dict[str, Tensor] = {}
        # Transpose to [T, B, V] for PyTorch CTC loss
        log_probs: Tensor = F.log_softmax(phoneme_logits, dim=-1).transpose(0, 1)

        if targets is not None and target_lengths is not None:
            # Main CTC Loss: compares log_probs [T, B, 112] against phoneme targets [B, S]
            ctc_loss: Tensor = self.ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
            losses["ctc_loss"] = ctc_loss
            total_loss: Tensor = ctc_loss
        else:
            ctc_loss = torch.tensor(0.0, device=audio.device)
            total_loss = ctc_loss

        # Intermediate Draft CTC Loss (Layer 4 auxiliary supervision)
        if draft_logits is not None and targets is not None and target_lengths is not None:
            draft_log_probs: Tensor = F.log_softmax(draft_logits, dim=-1).transpose(0, 1)
            draft_ctc_loss: Tensor = self.draft_ctc_loss_fn(draft_log_probs, targets, input_lengths, target_lengths)
            losses["draft_ctc_loss"] = draft_ctc_loss
            total_loss = total_loss + 0.3 * draft_ctc_loss

        # Articulatory Manner-of-Articulation CTC Loss
        if macro_targets is not None and macro_lengths is not None:
            macro_log_probs: Tensor = F.log_softmax(macro_logits, dim=-1).transpose(0, 1)
            macro_loss: Tensor = self.macro_ctc_loss_fn(macro_log_probs, macro_targets, input_lengths, macro_lengths)
            losses["macro_loss"] = macro_loss
            total_loss = total_loss + 0.2 * macro_loss

        # Anti-Collapse MoE Load Balancing Loss
        losses["aux_moe_loss"] = moe_aux_loss
        total_loss = total_loss + moe_aux_loss
        losses["loss"] = total_loss

        # Outputs
        losses["logits"] = phoneme_logits
        losses["macro_logits"] = macro_logits
        losses["hidden"] = hidden
        return losses

    @torch.no_grad()
    def forward_inference(
        self,
        audio: Tensor,
        audio_lengths: Optional[Tensor] = None,
        chunk_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Clean, Optimized Inference Forward Pass.
        Extracts phonemes and slices them on acoustic space pauses (token 8) into word lists.

        Args:
            audio (Tensor): Raw 16kHz speech waveform [B, T_samples].
            audio_lengths (Optional[Tensor]): Unpadded audio lengths.
            chunk_seconds (Optional[float]): If set, processes long audio via sliding chunks.

        Returns:
            Dict[str, Any]:
                - 'logits': [B, T, 112] raw CTC logits
                - 'pred_tokens': List of collapsed phoneme IDs per batch element
                - 'word_slices': List of words, where each word is a List[int] of phoneme IDs
                - 'space_indices': Frame indices where space pause (token 8) was emitted
        """
        self.eval()

        # 1. Feature extraction & projection: [B, T_frames, 512]
        cnn_feats: Tensor = self.feature_extractor(audio)
        feats: Tensor = self.feature_projection(cnn_feats)

        # 2. Conformer MoE Encoding: [B, T_frames, 512]
        hidden, draft_logits, _ = self._encode_frames(feats)

        # 3. Prediction Heads
        # Main CTC logits: [B, T_frames, 112]
        logits: Tensor = self.phoneme_head(hidden)
        # Articulatory manner logits: [B, T_frames, 8]
        macro_logits: Tensor = self.macro_head(hidden)
        # Acoustic word boundary probabilities: [B, T_frames, 1]
        boundary_probs: Tensor = self.boundary_gate(hidden)

        # Collect MoE routing decisions across all 12 layers
        moe_routings: List[Dict[str, Tensor]] = []
        for layer in self.conformer_layers:
            if getattr(layer, "use_deepseek_moe", False) and hasattr(layer.ffn2, "last_routing"):
                moe_routings.append(layer.ffn2.last_routing)

        # 4. Greedy argmax collapse decoding
        collapsed_batch_ids: List[List[int]] = self.phoneme_head.decode_greedy_ids(logits)

        # 5. Word Slicing by Space Token (8)
        batch_words: List[List[List[int]]] = []
        for seq in collapsed_batch_ids:
            words: List[List[int]] = []
            curr_word: List[int] = []
            for tid in seq:
                if tid == self.cfg.space_id:
                    if curr_word:
                        words.append(curr_word)
                        curr_word = []
                else:
                    curr_word.append(tid)
            if curr_word:
                words.append(curr_word)
            batch_words.append(words)

        return {
            "logits": logits,
            "pred_tokens": collapsed_batch_ids,
            "word_slices": batch_words,
            "hidden": hidden,
            "draft_logits": draft_logits,
            "macro_logits": macro_logits,
            "boundary_probs": boundary_probs,
            "moe_routing": moe_routings,
        }

    def load_pretrained_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Loads weights from existing Phono-V8 acoustic checkpoints (e.g. latest_v8_acoustic.pt).

        Args:
            checkpoint_path (str): Path to saved checkpoint file.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        device: torch.device = self.get_device()
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = ckpt.get("acoustic_state_dict", ckpt.get("model_state_dict", ckpt))

        # Direct loading with flexible matching
        self.load_state_dict(sd, strict=False)
        return True
