"""Gradient-based XAI using Captum (Integrated Gradients, Saliency, Layer Attributions)."""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from captum.attr import (
    IntegratedGradients,
    Saliency,
    InputXGradient,
    LayerIntegratedGradients,
)

from src.models.hubert_asr import HuBERTForCTC


class TokenLogitModelWrapper(nn.Module):
    """Wrapper that outputs a scalar logit for a target frame and target token.
    
    This conforms to Captum's requirement for a function mapping input -> scalar score.
    """

    def __init__(self, model: HuBERTForCTC):
        super().__init__()
        self.model = model
        self.target_frame = 0
        self.target_token = 0

    def set_target(self, frame_idx: int, token_idx: int):
        self.target_frame = frame_idx
        self.target_token = token_idx

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: (B, T_audio) or (B, 1, T_audio)
        Returns:
            Scalar tensor or (B,) tensor containing the logit of the target token at target frame.
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        outputs = self.model(audio)
        logits = outputs["logits"]  # (B, T_frames, vocab_size)

        frame = min(self.target_frame, logits.shape[1] - 1)
        token_logit = logits[:, frame, self.target_token]
        return token_logit


class CTCSequenceLossWrapper(nn.Module):
    """Wrapper that outputs the negative CTC loss for a target token sequence.
    
    Higher score indicates higher likelihood of the target sequence.
    """

    def __init__(self, model: HuBERTForCTC, target_tokens: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("target_tokens", target_tokens)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        outputs = self.model(audio)
        log_probs = outputs["log_probs"]  # (T_frames, B, vocab_size)
        out_lengths = outputs["output_lengths"]

        B = audio.size(0)
        target_len = torch.full((B,), self.target_tokens.size(1), dtype=torch.long, device=audio.device)
        targets = self.target_tokens.expand(B, -1)

        loss = torch.nn.functional.ctc_loss(
            log_probs,
            targets,
            out_lengths,
            target_len,
            blank=self.model.config.blank_index,
            reduction="none",
            zero_infinity=True,
        )
        # Negative loss as objective to attribute
        return -loss


class AudioGradientExplainer:
    """Gradient and Attribution Explainer for HuBERT ASR using Captum."""

    def __init__(self, model: HuBERTForCTC, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        self.token_wrapper = TokenLogitModelWrapper(self.model)
        self.ig = IntegratedGradients(self.token_wrapper)
        self.saliency = Saliency(self.token_wrapper)
        self.input_x_grad = InputXGradient(self.token_wrapper)

    def explain_token(
        self,
        audio: torch.Tensor,
        frame_idx: int,
        token_idx: int,
        method: str = "integrated_gradients",
        n_steps: int = 30,
        baseline: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Explain the prediction of a single token at a given time frame.
        
        Args:
            audio: Waveform tensor of shape (1, T_audio).
            frame_idx: Time frame index in the model's output.
            token_idx: Token ID to attribute.
            method: 'integrated_gradients', 'saliency', or 'input_x_gradient'.
            n_steps: Number of approximation steps for Integrated Gradients.
            baseline: Baseline reference waveform (defaults to silence / zeros).
            
        Returns:
            Dict containing 'attributions' tensor with shape (1, T_audio) and metadata.
        """
        self.model.eval()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device).requires_grad_(True)

        self.token_wrapper.set_target(frame_idx, token_idx)

        if baseline is None:
            baseline = torch.zeros_like(audio)
        else:
            baseline = baseline.to(self.device)

        if method == "integrated_gradients":
            attr = self.ig.attribute(
                audio,
                baselines=baseline,
                n_steps=n_steps,
                return_convergence_delta=False,
            )
        elif method == "saliency":
            attr = self.saliency.attribute(audio, abs=False)
        elif method == "input_x_gradient":
            attr = self.input_x_grad.attribute(audio)
        else:
            raise ValueError(f"Unknown attribution method: {method}")

        return {
            "attributions": attr.detach().cpu().squeeze(),
            "frame_idx": frame_idx,
            "token_idx": token_idx,
            "method": method,
        }

    def explain_layer_attributions(
        self,
        audio: torch.Tensor,
        frame_idx: int,
        token_idx: int,
        layer_idx: int = 0,
        n_steps: int = 20,
    ) -> torch.Tensor:
        """Attribute prediction to representations in intermediate Transformer layers.
        
        Args:
            audio: (1, T_audio)
            frame_idx: Output frame index
            token_idx: Target token index
            layer_idx: Transformer layer index (0 to num_layers - 1)
        """
        self.model.eval()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)

        self.token_wrapper.set_target(frame_idx, token_idx)
        target_layer = self.model.encoder.layers[layer_idx]

        layer_ig = LayerIntegratedGradients(self.token_wrapper, target_layer)
        layer_attr = layer_ig.attribute(
            audio,
            n_steps=n_steps,
            return_convergence_delta=False,
        )
        return layer_attr.detach().cpu().squeeze(0)  # (T_frames, embed_dim)
