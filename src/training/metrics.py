"""Evaluation metrics for Automatic Speech Recognition."""

from typing import List, Tuple
import editdistance


def compute_cer(predictions: List[str], references: List[str]) -> float:
    """Compute Character Error Rate (CER).
    
    CER = (Substitutions + Deletions + Insertions) / Total Reference Characters
    """
    total_distance = 0
    total_chars = 0

    for pred, ref in zip(predictions, references):
        pred_clean = pred.strip()
        ref_clean = ref.strip()
        total_distance += editdistance.eval(pred_clean, ref_clean)
        total_chars += max(len(ref_clean), 1)

    return total_distance / max(total_chars, 1)


def compute_wer(predictions: List[str], references: List[str]) -> float:
    """Compute Word Error Rate (WER).
    
    WER = (Substitutions + Deletions + Insertions) / Total Reference Words
    """
    total_distance = 0
    total_words = 0

    for pred, ref in zip(predictions, references):
        pred_words = pred.strip().split()
        ref_words = ref.strip().split()
        total_distance += editdistance.eval(pred_words, ref_words)
        total_words += max(len(ref_words), 1)

    return total_distance / max(total_words, 1)


class MetricTracker:
    """Accumulates losses and calculates running averages."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.count = 0
        self.all_predictions: List[str] = []
        self.all_references: List[str] = []

    def update(
        self,
        loss: float,
        batch_size: int = 1,
        predictions: List[str] = None,
        references: List[str] = None,
    ):
        self.total_loss += loss * batch_size
        self.count += batch_size
        if predictions and references:
            self.all_predictions.extend(predictions)
            self.all_references.extend(references)

    @property
    def avg_loss(self) -> float:
        return self.total_loss / max(self.count, 1)

    @property
    def cer(self) -> float:
        if not self.all_references:
            return 1.0
        return compute_cer(self.all_predictions, self.all_references)

    @property
    def wer(self) -> float:
        if not self.all_references:
            return 1.0
        return compute_wer(self.all_predictions, self.all_references)
