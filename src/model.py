"""
Main Unified Model Architecture.
Orchestrates the complete tripartite end-to-end speech-to-text pipeline:
  Audio (16 kHz)
    │
    ▼ (Model Part 1: Model_A2P)
  Acoustic Phonemes + Spaces (token 8)
    │
    ▼ (Model Part 2: Model_PE)
  Continuous Word Latents (z_word in R^512)
    │
    ▼ (Model Part 3: Model_WD)
  Orthographic Character Text
"""

# Import Modules
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import torch
from torch import nn, Tensor

from .model_config import ModelConfig, get_class
from .model1_a2p import Model_A2P
from .model2_pe import Model_PE
from .model3_wd import Model_WD


class Model(nn.Module):
    """
    Unified Tripartite Speech Recognition Model.
    Coordinates Model_A2P (Part 1), Model_PE (Part 2), and Model_WD (Part 3)
    with clean OOP modularity, per-part device routing, and seamless checkpoint loading.
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        """
        Initialize the Unified Model.

        Args:
            model_config (Optional[ModelConfig]): Master model configuration dataclass.
                                                  If None, default ModelConfig() is used.
        """
        # Parent initialization
        super().__init__()

        # Master model configuration
        self.model_config: ModelConfig = model_config or ModelConfig()

        # Part 1 of the model: Audio to Phonemes (A2P)
        cls_a2p = get_class(self.model_config.model_a2p_class)
        self.model_a2p: Model_A2P = cls_a2p(model_config=self.model_config)

        # Part 2 of the model: Phoneme Encoder (PE)
        cls_pe = get_class(self.model_config.model_pe_class)
        self.model_pe: Model_PE = cls_pe(model_config=self.model_config)

        # Part 3 of the model: Word Decoder (WD)
        cls_wd = get_class(self.model_config.model_wd_class)
        self.model_wd: Model_WD = cls_wd(model_config=self.model_config)

        # Route each sub-model to its assigned hardware device
        self.model_a2p.to_assigned_device()
        self.model_pe.to_assigned_device()
        self.model_wd.to_assigned_device()

    def forward_training(
        self,
        audio: Tensor,
        audio_lengths: Optional[Tensor] = None,
        phoneme_targets: Optional[Tensor] = None,
        phoneme_lengths: Optional[Tensor] = None,
        macro_targets: Optional[Tensor] = None,
        macro_lengths: Optional[Tensor] = None,
        input_byte_ids: Optional[Tensor] = None,
        target_byte_ids: Optional[Tensor] = None,
        target_word_lengths: Optional[Tensor] = None,
        phoneme_word_targets: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Clean, End-to-End Multi-Task Training Forward Pass across all 3 model parts.

        Args:
            audio (Tensor): Raw 16kHz audio [B, T_samples].
            audio_lengths (Optional[Tensor]): Audio sample lengths [B].
            phoneme_targets (Optional[Tensor]): Phoneme targets for Model 1 CTC [B, S].
            phoneme_lengths (Optional[Tensor]): Phoneme target lengths [B].
            macro_targets (Optional[Tensor]): Articulatory manner targets [B, S].
            macro_lengths (Optional[Tensor]): Articulatory target lengths [B].
            input_byte_ids (Optional[Tensor]): Input character prefixes for Model 3 [B, L, K].
            target_byte_ids (Optional[Tensor]): Target characters for Model 3 [B, L, K].
            target_word_lengths (Optional[Tensor]): Word lengths in characters [B, L].
            phoneme_word_targets (Optional[Tensor]): Intra-word phoneme slices for Model 2 [B, L, K_phono].

        Returns:
            Dict[str, Tensor]: Total aggregated loss and breakdown per model part.
        """
        # --- Stage 1: Audio to Phonemes (Model 1) ---
        a2p_losses = self.model_a2p.forward_training(
            audio=audio.to(self.model_a2p.get_device()),
            audio_lengths=audio_lengths.to(self.model_a2p.get_device()) if audio_lengths is not None else None,
            targets=phoneme_targets.to(self.model_a2p.get_device()) if phoneme_targets is not None else None,
            target_lengths=phoneme_lengths.to(self.model_a2p.get_device()) if phoneme_lengths is not None else None,
            macro_targets=macro_targets.to(self.model_a2p.get_device()) if macro_targets is not None else None,
            macro_lengths=macro_lengths.to(self.model_a2p.get_device()) if macro_lengths is not None else None,
        )

        total_loss: Tensor = a2p_losses["loss"]
        loss_dict: Dict[str, Tensor] = {
            "loss_a2p": a2p_losses["loss"],
            "ctc_loss": a2p_losses.get("ctc_loss", torch.tensor(0.0)),
        }

        # --- Stage 2 & 3: Word Latents & Character Decoding (Models 2 & 3) ---
        if input_byte_ids is not None and target_byte_ids is not None:
            # When training with character targets, decode word latents through Model 3
            dev_wd: torch.device = self.model_wd.get_device()
            b_sz, l_words = input_byte_ids.shape[0], input_byte_ids.shape[1]

            # Pass word phonemes through Model 2 to obtain word latents
            if phoneme_word_targets is not None:
                z_words_out = self.model_pe.forward_training(
                    phoneme_ids=phoneme_word_targets.to(self.model_pe.get_device()),
                )
                z_words: Tensor = z_words_out["z_word"].to(dev_wd)
            elif phoneme_targets is not None and phoneme_targets.ndim == 3:
                z_words_out = self.model_pe.forward_training(
                    phoneme_ids=phoneme_targets.to(self.model_pe.get_device()),
                )
                z_words = z_words_out["z_word"].to(dev_wd)
            else:
                z_words = torch.zeros((b_sz, l_words, 512), device=dev_wd)

            wd_losses = self.model_wd.forward_training(
                z_words=z_words,
                input_byte_ids=input_byte_ids.to(dev_wd),
                target_byte_ids=target_byte_ids.to(dev_wd),
                target_lengths=target_word_lengths.to(dev_wd) if target_word_lengths is not None else None,
            )

            total_loss = total_loss + wd_losses["loss"].to(total_loss.device)
            loss_dict["loss_wd"] = wd_losses["loss"]
            loss_dict["char_loss"] = wd_losses.get("char_loss", torch.tensor(0.0))

        loss_dict["loss"] = total_loss
        return loss_dict

    @torch.no_grad()
    def forward_inference(
        self,
        audio: Tensor,
        audio_lengths: Optional[Tensor] = None,
        trie: Optional[Any] = None,
        beam_width: int = 1,
        blank_penalty: float = 2.6,
        space_bonus: float = 0.35,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Clean, Synchronous End-to-End Inference Forward Pass.
        Pipeline: Audio -> Model 1 -> Word Slices -> Model 2 -> Model 3 -> Emitted Text.

        Args:
            audio (Tensor): Raw 16kHz speech waveform [B, T_samples] or [1, T_samples].
            audio_lengths (Optional[Tensor]): Unpadded audio sample lengths.
            trie (Optional[Any]): Vocabulary Trie prefix tree for dictionary-constrained decoding.
            beam_width (int): Beam search width for Model 3 character decoder (1 = greedy).
            blank_penalty (float): Model 1 CTC blank penalty (default: 2.6).
            space_bonus (float): Model 1 CTC space token bonus (default: 0.35).

        Returns:
            Dict[str, Any]:
                - 'transcripts': List[str] of final decoded text strings per utterance
                - 'phoneme_tokens': List[List[int]] of collapsed phoneme IDs from Model 1
                - 'word_slices': Sliced intra-word phoneme sequences
        """
        self.eval()

        # Route audio to Model 1's device
        dev_a2p: torch.device = self.model_a2p.get_device()
        audio_dev: Tensor = audio.to(dev_a2p)

        # 1. Model 1 (Audio -> Phonemes with spaces)
        a2p_out = self.model_a2p.forward_inference(
            audio_dev,
            blank_penalty=blank_penalty,
            space_bonus=space_bonus,
        )
        batch_word_slices = a2p_out["word_slices"]

        # 2 & 3. Models 2 & 3 per utterance
        transcripts: List[str] = []
        for word_slices in batch_word_slices:
            if not word_slices:
                transcripts.append("")
                continue

            # Model 2: Word Slices -> z_words [1, L, 512] on Model 2's device
            z_words = self.model_pe.forward_inference(word_slices)

            # Route z_words to Model 3's device
            dev_wd: torch.device = self.model_wd.get_device()
            z_words_wd = z_words.to(dev_wd)

            # Model 3: z_words -> Reconstructed text words
            decoded_words: List[str] = self.model_wd.forward_inference(
                z_words_wd,
                trie=trie,
                beam_width=beam_width,
            )
            full_text: str = " ".join(decoded_words).strip()
            transcripts.append(full_text)

        return {
            "transcripts": transcripts,
            "phoneme_tokens": a2p_out["pred_tokens"],
            "word_slices": batch_word_slices,
        }

    def transcribe(
        self,
        audio_input: Union[str, Tensor],
        trie: Optional[Any] = None,
        beam_width: int = 1,
        blank_penalty: float = 2.6,
        space_bonus: float = 0.35,
        **kwargs,
    ) -> str:
        """
        Convenient, high-level speech-to-text transcription function.

        Args:
            audio_input (Union[str, Tensor]): File path to audio (wav/flac) or 1D/2D audio tensor.
            trie (Optional[Any]): Vocabulary Trie prefix tree for dictionary constraint.
            beam_width (int): Character beam search width (1 = greedy).
            blank_penalty (float): Model 1 CTC blank penalty.
            space_bonus (float): Model 1 CTC space token bonus.

        Returns:
            str: Decoded orthographic transcription text.
        """
        if isinstance(audio_input, str):
            import soundfile as sf
            wav, sr = sf.read(audio_input, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != self.model_config.a2p.sample_rate:
                import torchaudio.transforms as T
                resampler = T.Resample(sr, self.model_config.a2p.sample_rate)
                audio_t = resampler(torch.from_numpy(wav).float().unsqueeze(0)).squeeze(0)
            else:
                audio_t = torch.from_numpy(wav).float()
            audio_tensor = audio_t.unsqueeze(0)
        else:
            if audio_input.ndim == 1:
                audio_tensor = audio_input.unsqueeze(0)
            else:
                audio_tensor = audio_input

        res = self.forward_inference(
            audio_tensor,
            trie=trie,
            beam_width=beam_width,
            blank_penalty=blank_penalty,
            space_bonus=space_bonus,
            **kwargs,
        )
        return res["transcripts"][0] if res["transcripts"] else ""

    def load_from_checkpoints(
        self,
        a2p_checkpoint: Optional[str] = None,
        p2g_checkpoint: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Loads pretrained weights into all model parts.

        Args:
            a2p_checkpoint (Optional[str]): Path to Model 1 weights (e.g. latest_v8_acoustic.pt).
            p2g_checkpoint (Optional[str]): Path to Models 2 & 3 weights (e.g. best_robust_p2g.pt).

        Returns:
            Dict[str, bool]: Loading status for each model part.
        """
        status: Dict[str, bool] = {"a2p": False, "pe": False, "wd": False}

        if a2p_checkpoint and os.path.isfile(a2p_checkpoint):
            status["a2p"] = self.model_a2p.load_pretrained_checkpoint(a2p_checkpoint)

        if p2g_checkpoint and os.path.isfile(p2g_checkpoint):
            status["pe"] = self.model_pe.load_pretrained_checkpoint(p2g_checkpoint)
            status["wd"] = self.model_wd.load_pretrained_checkpoint(p2g_checkpoint)

        return status
