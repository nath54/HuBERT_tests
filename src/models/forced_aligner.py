"""Native TorchAudio Forced Alignment Module for Phono-V7.

Uses torchaudio.functional.forced_align to:
1. Extract frame-level phoneme alignments from CTC posteriors.
2. Extract exact word start/end/center timestamps for continuous speech slicing.
3. Compute frame-level supervised alignment loss to anchor phoneme emissions to audio frames.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF


class ForcedAligner:
    """CUDA/C++ Native Forced Alignment using TorchAudio.
    
    Args:
        blank_id: Index of CTC blank token (default: 1).
        pad_id: Index of padding token (default: 0).
        space_id: Index of phoneme space delimiter token (default: 8).
    """

    def __init__(
        self,
        blank_id: int = 1,
        pad_id: int = 0,
        space_id: int = 8,
    ):
        self.blank_id = blank_id
        self.pad_id = pad_id
        self.space_id = space_id

    def align(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align phoneme sequence against CTC log probabilities.
        
        Args:
            log_probs: [B, T, V] Log-softmax CTC probabilities.
            targets: [B, S] Target phoneme token IDs.
            input_lengths: [B] Audio frame lengths T_b.
            target_lengths: [B] Phoneme sequence lengths S_b.
            
        Returns:
            aligned_tokens: [B, T] Exact token assigned to each time frame.
            scores: [B] Alignment path log-likelihood scores.
        """
        B, T, V = log_probs.shape
        device = log_probs.device

        # Ensure lengths are CPU int32 or int64 as required by C++ kernel
        i_lens = input_lengths.to(dtype=torch.long, device="cpu")
        t_lens = target_lengths.to(dtype=torch.long, device="cpu")

        # Clamp target_lengths to not exceed input_lengths (CTC requirement: S <= T)
        t_lens = torch.clamp(t_lens, max=i_lens)

        # Ensure targets are valid
        targets_clamped = torch.clamp(targets, min=0, max=V - 1)

        aligned_list = []
        scores_list = []

        for b in range(B):
            cur_i_len = int(i_lens[b].item())
            cur_t_len = int(t_lens[b].item())

            if cur_t_len <= 0 or cur_i_len < cur_t_len:
                aligned_full = torch.full((T,), self.blank_id, dtype=torch.long, device=device)
                score_full = torch.zeros((T,), dtype=log_probs.dtype, device=device)
                aligned_list.append(aligned_full)
                scores_list.append(score_full)
                continue

            sub_log_probs = log_probs[b : b + 1, :cur_i_len]  # [1, cur_i_len, V]
            sub_targets = targets_clamped[b : b + 1, :cur_t_len]  # [1, cur_t_len]
            sub_i_tensor = torch.tensor([cur_i_len], dtype=torch.long, device=device)
            sub_t_tensor = torch.tensor([cur_t_len], dtype=torch.long, device=device)

            aligned_b, score_b = AF.forced_align(
                log_probs=sub_log_probs,
                targets=sub_targets,
                input_lengths=sub_i_tensor,
                target_lengths=sub_t_tensor,
                blank=self.blank_id,
            )
            aligned_full = torch.full((T,), self.blank_id, dtype=torch.long, device=device)
            aligned_full[:cur_i_len] = aligned_b[0]
            aligned_list.append(aligned_full)

            score_full = torch.zeros((T,), dtype=score_b.dtype, device=device)
            score_full[:cur_i_len] = score_b[0]
            scores_list.append(score_full)

        aligned_tokens = torch.stack(aligned_list, dim=0)  # [B, T]
        scores = torch.stack(scores_list, dim=0)          # [B, T]
        return aligned_tokens, scores

    def extract_word_centers(
        self,
        aligned_tokens: torch.Tensor,
        num_words: torch.Tensor,
        max_words: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract exact acoustic center frame for each word from aligned token sequence.
        
        Args:
            aligned_tokens: [B, T] Aligned phoneme token sequence.
            num_words: [B] Number of words per utterance.
            max_words: Maximum number of word slots L.
            
        Returns:
            word_centers: [B, max_words] Center frame index for each word slot.
            word_valid_mask: [B, max_words] Boolean mask indicating valid words in audio.
        """
        B, T = aligned_tokens.shape
        device = aligned_tokens.device

        word_centers = torch.zeros((B, max_words), dtype=torch.long, device=device)
        word_valid = torch.zeros((B, max_words), dtype=torch.bool, device=device)

        aligned_cpu = aligned_tokens.cpu()
        nw_cpu = num_words.cpu()

        for b in range(B):
            n_w = min(int(nw_cpu[b].item()), max_words)
            tokens = aligned_cpu[b].tolist()

            # Group non-blank frames by word using space delimiter
            current_w = 0
            w_frames: Dict[int, List[int]] = {0: []}

            for t, tok in enumerate(tokens):
                if tok == self.blank_id or tok == self.pad_id:
                    continue
                if tok == self.space_id:
                    current_w += 1
                    w_frames[current_w] = []
                else:
                    if current_w not in w_frames:
                        w_frames[current_w] = []
                    w_frames[current_w].append(t)

            for w in range(n_w):
                frames = w_frames.get(w, [])
                if frames:
                    center = int(sum(frames) / len(frames))
                    word_centers[b, w] = center
                    word_valid[b, w] = True
                else:
                    # Fallback to proportional Axe 1 if word had no non-blank frames
                    center = int(((w + 0.5) / max(1, n_w)) * T)
                    word_centers[b, w] = min(center, T - 1)
                    word_valid[b, w] = False

        return word_centers, word_valid

    def compute_frame_alignment_loss(
        self,
        logits: torch.Tensor,
        aligned_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute frame-level Cross-Entropy on aligned non-blank phoneme frames.
        
        Args:
            logits: [B, T, V] Raw emission logits from acoustic encoder.
            aligned_tokens: [B, T] Aligned phoneme token sequence from forced_align.
            
        Returns:
            Scalar cross-entropy loss on non-blank frames.
        """
        B, T, V = logits.shape
        flat_logits = logits.reshape(B * T, V)
        flat_targets = aligned_tokens.reshape(B * T)

        # Ignore blanks and pads so we only supervise actual phonetic sound events
        # We can also supervise blanks with lower weight if desired
        loss = F.cross_entropy(
            flat_logits,
            flat_targets,
            ignore_index=self.blank_id,
        )
        return loss
