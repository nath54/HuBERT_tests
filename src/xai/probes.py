"""Trained Decoders and Diagnostic Probing for Layer-wise Interpretability."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.hubert_asr import HuBERTForCTC


class LinearProbe(nn.Module):
    """Linear diagnostic probe mapping hidden representation to class labels."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class AcousticInversionDecoder(nn.Module):
    """Trained Decoder reconstructing acoustic features (e.g. Mel-spectrum) from layer representations.
    
    This decoder is used to analyze how much low-level acoustic/spectral information
    is preserved vs discarded across the layers of HuBERT.
    """

    def __init__(self, hidden_dim: int, n_mels: int = 80):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_mels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T_frames, hidden_dim)
        Returns:
            Reconstructed Mel-features: (B, T_frames, n_mels)
        """
        return self.decoder(x)


class LayerwiseProbeTrainer:
    """Extracts representations across all layers and trains diagnostic probes."""

    def __init__(self, model: HuBERTForCTC, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.embed_dim = model.config.encoder_embed_dim
        self.num_layers = model.config.encoder_layers

    @torch.no_grad()
    def extract_all_layer_representations(
        self,
        dataset,
        max_samples: int = 100,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Extract hidden states for each layer across samples.
        
        Returns:
            layer_feats: List of tensors [ (N_total_frames, embed_dim) ], one per layer (0 to L)
            pseudo_labels: Frame-level labels (e.g. energy or pseudo-phoneme classes)
        """
        self.model.eval()
        num_samples = min(len(dataset), max_samples)

        all_layer_reps = [[] for _ in range(self.num_layers + 1)]
        all_labels = []

        for i in range(num_samples):
            sample = dataset[i]
            audio = sample["audio"].unsqueeze(0).to(self.device)
            outputs = self.model(audio, output_hidden_states=True)

            hidden_states = outputs["hidden_states"]  # List of L+1 tensors, each (1, T, embed_dim)
            t_frames = hidden_states[0].shape[1]

            for l_idx, h in enumerate(hidden_states):
                all_layer_reps[l_idx].append(h.squeeze(0).cpu())

            # Generate frame acoustic target: e.g. quantized energy level (4 classes: silence, low, med, high)
            frame_energy = hidden_states[0].squeeze(0).pow(2).mean(dim=-1).cpu()
            quantiles = torch.quantile(frame_energy, torch.tensor([0.25, 0.50, 0.75]))
            labels = torch.bucketize(frame_energy, quantiles)  # (T_frames,)
            all_labels.append(labels)

        concat_layer_feats = [torch.cat(reps, dim=0) for reps in all_layer_reps]
        concat_labels = torch.cat(all_labels, dim=0)

        return concat_layer_feats, concat_labels

    def train_diagnostic_probes(
        self,
        dataset,
        num_classes: int = 4,
        epochs: int = 5,
        batch_size: int = 128,
        lr: float = 1e-3,
    ) -> Dict[str, List[float]]:
        """Train and evaluate linear diagnostic probes for every layer.
        
        Returns:
            Dictionary with 'layer_indices' and 'probe_accuracies'.
        """
        print("[Diagnostic Probing] Extracting representations across all HuBERT layers...")
        layer_feats, labels = self.extract_all_layer_representations(dataset)

        accuracies = []
        num_layers_total = len(layer_feats)

        # Train/test split (80/20)
        n_total = labels.shape[0]
        perm = torch.randperm(n_total)
        n_train = int(0.8 * n_total)
        train_idx, val_idx = perm[:n_train], perm[n_train:]

        y_train = labels[train_idx].to(self.device)
        y_val = labels[val_idx].to(self.device)

        print(f"[Diagnostic Probing] Training probes across {num_layers_total} layers ({n_total} frames total)...")

        for l in range(num_layers_total):
            X_layer = layer_feats[l]
            X_train = X_layer[train_idx].to(self.device)
            X_val = X_layer[val_idx].to(self.device)

            train_ds = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            probe = LinearProbe(self.embed_dim, num_classes).to(self.device)
            optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

            probe.train()
            for ep in range(epochs):
                for bx, by in train_loader:
                    optimizer.zero_grad()
                    preds = probe(bx)
                    loss = F.cross_entropy(preds, by)
                    loss.backward()
                    optimizer.step()

            # Evaluation
            probe.eval()
            with torch.no_grad():
                val_preds = probe(X_val).argmax(dim=-1)
                acc = (val_preds == y_val).float().mean().item()
                accuracies.append(acc)

            layer_name = "CNN Out" if l == 0 else f"Layer {l}"
            print(f"  -> Probe [{layer_name}] Accuracy: {acc * 100:.2f}%")

        return {
            "layer_indices": list(range(num_layers_total)),
            "layer_names": ["CNN"] + [f"L{i}" for i in range(1, num_layers_total)],
            "accuracies": accuracies,
        }

    def train_reconstruction_decoders(
        self,
        dataset,
        n_mels: int = 40,
        epochs: int = 5,
        batch_size: int = 64,
        lr: float = 1e-3,
    ) -> Dict[str, List[float]]:
        """Train acoustic inversion decoders to quantify acoustic information retention per layer."""
        print("[Acoustic Inversion] Training reconstruction decoders per layer...")
        self.model.eval()

        layer_losses = []
        num_layers_total = self.num_layers + 1

        # Collect features and pseudo-mels (using 40 filterbank features from CNN features)
        layer_feats, _ = self.extract_all_layer_representations(dataset, max_samples=40)
        # Target: downsampled spectral target derived from early representation
        target_acoustic = layer_feats[0][:, :n_mels].to(self.device)

        n_total = target_acoustic.size(0)
        split = int(0.8 * n_total)

        for l in range(num_layers_total):
            X_layer = layer_feats[l].to(self.device)
            decoder = AcousticInversionDecoder(self.embed_dim, n_mels=n_mels).to(self.device)
            optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

            for ep in range(epochs):
                for i in range(0, split, batch_size):
                    bx = X_layer[i : i + batch_size]
                    by = target_acoustic[i : i + batch_size]
                    optimizer.zero_grad()
                    pred = decoder(bx)
                    loss = F.mse_loss(pred, by)
                    loss.backward()
                    optimizer.step()

            with torch.no_grad():
                val_x = X_layer[split:]
                val_y = target_acoustic[split:]
                val_loss = F.mse_loss(decoder(val_x), val_y).item()
                layer_losses.append(val_loss)

        return {
            "layer_names": ["CNN"] + [f"L{i}" for i in range(1, num_layers_total)],
            "reconstruction_mse": layer_losses,
        }
