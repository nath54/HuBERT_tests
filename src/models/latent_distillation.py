r"""Latent Space Distillation Loss and Objectives.

Aligns fast text/phoneme word representations (\hat{z}) to the frozen/detached
acoustic word representations (z_acoustic) produced by Phono Conformer + Aligner.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentSpaceDistillationLoss(nn.Module):
    """Multi-Objective Cross-Modal Distillation Loss.

    L_distill = mse_weight * MSE(z_student, z_teacher)
              + cos_weight * (1 - CosSim(z_student, z_teacher))
              + nce_weight * InfoNCE(z_student, z_teacher)
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        cos_weight: float = 0.5,
        nce_weight: float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.cos_weight = cos_weight
        self.nce_weight = nce_weight
        self.temperature = temperature

    def forward(
        self,
        z_student: torch.Tensor,
        z_teacher: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            z_student: [N, D] student latent vectors (from text/phoneme encoder)
            z_teacher: [N, D] teacher latent vectors (from acoustic encoder, detached)
            mask: Optional [N] boolean mask indicating valid non-padded words
        Returns:
            loss: scalar loss
            metrics: dict of individual loss values
        """
        if mask is not None:
            z_student = z_student[mask]
            z_teacher = z_teacher[mask]

        if z_student.numel() == 0:
            zero = torch.tensor(0.0, device=z_student.device)
            return zero, {"distill_loss": 0.0, "mse": 0.0, "cos_sim": 1.0, "nce": 0.0}

        z_teacher = z_teacher.detach()

        # 1. Mean Squared Error
        mse_loss = F.mse_loss(z_student, z_teacher)

        # 2. Cosine Similarity & Distance
        cos_sim = F.cosine_similarity(z_student, z_teacher, dim=-1)
        cos_loss = (1.0 - cos_sim).mean()

        # 3. InfoNCE Contrastive Loss (prevents representation collapse across batch)
        nce_loss = torch.tensor(0.0, device=z_student.device)
        if self.nce_weight > 0.0 and z_student.shape[0] > 1:
            z_s_norm = F.normalize(z_student, dim=-1)
            z_t_norm = F.normalize(z_teacher, dim=-1)
            # Similarity matrix: [N, N]
            logits = torch.matmul(z_s_norm, z_t_norm.T) / self.temperature
            labels = torch.arange(z_student.shape[0], device=z_student.device)
            nce_loss = F.cross_entropy(logits, labels)

        total_loss = (
            self.mse_weight * mse_loss
            + self.cos_weight * cos_loss
            + self.nce_weight * nce_loss
        )

        metrics = {
            "distill_loss": total_loss.item(),
            "mse": mse_loss.item(),
            "cos_sim": cos_sim.mean().item(),
            "nce": nce_loss.item() if isinstance(nce_loss, torch.Tensor) else 0.0,
        }
        return total_loss, metrics
