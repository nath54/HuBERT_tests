"""Neuron Activation Profiles (NAP), Saliency, and Activation Patching (NAPS)."""

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from src.models.hubert_asr import HuBERTForCTC


class NeuronActivationProfiler:
    """Computes Neuron Activation Profiles (NAP) and Selectivity across phonetic/acoustic categories."""

    def __init__(self, model: HuBERTForCTC, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.num_layers = model.config.encoder_layers
        self.ffn_dim = model.config.encoder_ffn_dim

    @torch.no_grad()
    def collect_profiles(
        self,
        dataset,
        category_classifier_fn: Optional[Callable[[Dict], List[str]]] = None,
        max_samples: int = 50,
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Collect neuron activations grouped by condition/category.
        
        Args:
            dataset: AudioASRDataset instance.
            category_classifier_fn: Function mapping (sample) -> list of frame categories
                                   (e.g., ['silence', 'vowel', 'consonant', ...]).
                                   If None, uses simple energy/silence & character alignment.
            max_samples: Maximum number of dataset samples to profile.
            
        Returns:
            Dictionary mapping layer_idx -> {category_name: mean_neuron_activations (ffn_dim,)}
        """
        self.model.eval()

        # Storage: [layer_idx][category] -> list of activation vectors
        category_activations = {
            layer_idx: defaultdict(list) for layer_idx in range(self.num_layers)
        }

        num_samples = min(len(dataset), max_samples)
        for i in range(num_samples):
            sample = dataset[i]
            audio = sample["audio"].unsqueeze(0).to(self.device)

            outputs = self.model(audio, output_hidden_states=True)
            # neuron_activations is a list of [ (1, T_frames, ffn_dim) ] for each layer
            neuron_acts = outputs["neuron_activations"]
            t_frames = neuron_acts[0].shape[1]

            # Categorize frames
            if category_classifier_fn is not None:
                categories = category_classifier_fn(sample)
            else:
                # Default heuristic: frame energy & silence vs active speech
                feats = outputs["hidden_states"][0].squeeze(0)  # (T_frames, embed_dim)
                frame_energy = feats.pow(2).mean(dim=-1).cpu().numpy()
                thresh = np.percentile(frame_energy, 30)
                categories = ["silence" if e < thresh else "speech" for e in frame_energy]

            # Assign activations
            for l_idx, acts_tensor in enumerate(neuron_acts):
                acts = acts_tensor.squeeze(0).cpu().numpy()  # (T_frames, ffn_dim)
                for t in range(min(t_frames, len(categories))):
                    cat = categories[t]
                    category_activations[l_idx][cat].append(acts[t])

        # Compute mean profiles and selectivity
        profiles = {}
        for l_idx in range(self.num_layers):
            profiles[l_idx] = {}
            for cat, act_list in category_activations[l_idx].items():
                if len(act_list) > 0:
                    profiles[l_idx][cat] = np.mean(np.array(act_list), axis=0)

        return profiles

    def compute_selectivity(
        self,
        profiles: Dict[int, Dict[str, np.ndarray]],
        target_category: str,
        baseline_category: str = "silence",
    ) -> Dict[int, np.ndarray]:
        """Compute Selectivity Index for target category across layers.
        
        SI = (Mean_target - Mean_baseline) / (|Mean_target| + |Mean_baseline| + eps)
        """
        selectivity_per_layer = {}
        for l_idx, cat_dict in profiles.items():
            if target_category in cat_dict and baseline_category in cat_dict:
                target_mean = cat_dict[target_category]
                base_mean = cat_dict[baseline_category]
                diff = target_mean - base_mean
                denom = np.abs(target_mean) + np.abs(base_mean) + 1e-6
                selectivity_per_layer[l_idx] = diff / denom
        return selectivity_per_layer


class NeuronSaliencyAnalyzer:
    """Computes Neuron Saliency via Activation * Gradient."""

    def __init__(self, model: HuBERTForCTC, device: torch.device):
        self.model = model.to(device)
        self.device = device

    def compute_neuron_saliency(
        self,
        audio: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Compute saliency (Activation x Gradient) for each neuron in each Transformer layer.
        
        Returns:
            List of tensors [ (ffn_dim,) ], one per layer, indicating importance scores.
        """
        self.model.eval()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)
        target_tokens = target_tokens.to(self.device)
        if target_tokens.dim() == 1:
            target_tokens = target_tokens.unsqueeze(0)

        # Hook into FFN intermediate representations
        activations = {}
        gradients = {}
        hooks = []

        def get_hook(layer_i):
            def fwd_hook(module, inp, out):
                # out is (out, intermediate)
                intermediate = out[1]
                activations[layer_i] = intermediate
                intermediate.retain_grad()
            return fwd_hook

        for i, layer in enumerate(self.model.encoder.layers):
            hooks.append(layer.ffn.register_forward_hook(get_hook(i)))

        outputs = self.model(
            audio,
            targets=target_tokens,
            target_lengths=torch.tensor([target_tokens.size(1)], device=self.device),
        )
        loss = outputs["loss"]
        loss.backward()

        layer_saliency = []
        for i in range(len(self.model.encoder.layers)):
            act = activations[i]  # (1, T_frames, ffn_dim)
            grad = act.grad      # (1, T_frames, ffn_dim)
            if grad is not None:
                saliency = (act * grad).abs().mean(dim=(0, 1))  # (ffn_dim,)
            else:
                saliency = act.abs().mean(dim=(0, 1))
            layer_saliency.append(saliency.detach().cpu())

        # Clean up hooks
        for h in hooks:
            h.remove()

        return layer_saliency


class ActivationPatcher:
    """Activation Patching and Causal Ablation Interventions."""

    def __init__(self, model: HuBERTForCTC, device: torch.device):
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def ablate_layer(
        self,
        audio: torch.Tensor,
        layer_idx: int,
        ablation_type: str = "zero",
    ) -> Dict[str, any]:
        """Ablate a specific Transformer layer and measure output degradation.
        
        Args:
            audio: Input waveform.
            layer_idx: Layer to intervene on.
            ablation_type: 'zero' or 'skip' (identity bypass).
        """
        self.model.eval()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)

        # Baseline output
        orig_outputs = self.model(audio)
        orig_logits = orig_outputs["logits"]

        target_layer = self.model.encoder.layers[layer_idx]

        def ablate_hook(module, inp, out):
            x, attn, inter = out
            if ablation_type == "zero":
                return torch.zeros_like(x), attn, inter
            elif ablation_type == "skip":
                return inp[0], attn, inter  # Pass input through directly (skip layer)
            return out

        handle = target_layer.register_forward_hook(ablate_hook)
        ablated_outputs = self.model(audio)
        handle.remove()

        ablated_logits = ablated_outputs["logits"]
        logit_diff = (orig_logits - ablated_logits).abs().mean().item()

        return {
            "layer_idx": layer_idx,
            "ablation_type": ablation_type,
            "logit_mean_diff": logit_diff,
            "orig_prediction": self.model.decode_greedy(orig_logits)[0],
            "ablated_prediction": self.model.decode_greedy(ablated_logits)[0],
        }

    @torch.no_grad()
    def layer_causal_importance_scan(self, audio: torch.Tensor) -> List[float]:
        """Scan causal impact across all layers sequentially."""
        num_layers = len(self.model.encoder.layers)
        impacts = []
        for l in range(num_layers):
            res = self.ablate_layer(audio, l, ablation_type="zero")
            impacts.append(res["logit_mean_diff"])
        return impacts
