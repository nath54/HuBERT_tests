"""Phonotactic Prefix Beam Search Decoder for CTC Acoustic Models.

Implements prefix beam search with phonetic phonotactic constraints:
1. Space calibration & deduplication (prevents double spaces and leading/trailing spaces).
2. Blank penalty calibration.
3. Length normalization.
4. Optional phonotactic transition prior.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import torch


class PhonotacticPrefixBeamSearch:
    """Prefix beam search decoder for phoneme CTC logits."""

    def __init__(
        self,
        beam_width: int = 16,
        blank_id: int = 1,
        space_id: int = 8,
        space_bonus: float = 0.2,
        blank_penalty: float = 0.3,
        length_penalty: float = 0.5,
    ):
        self.beam_width = beam_width
        self.blank_id = blank_id
        self.space_id = space_id
        self.space_bonus = space_bonus
        self.blank_penalty = blank_penalty
        self.length_penalty = length_penalty

    def decode(self, log_probs: torch.Tensor) -> List[int]:
        """Decode a 2D log-probability tensor [T, Vocab] into the best phoneme sequence.
        
        Args:
            log_probs: Tensor of shape (T, Vocab) containing log-probabilities or logits.
        """
        if log_probs.dim() == 3:
            log_probs = log_probs[0]  # Take first batch item

        # Ensure log_softmax
        log_probs = torch.log_softmax(log_probs, dim=-1).detach().cpu()
        T, V = log_probs.shape

        # Initialize beam: mapping prefix (tuple of token ids) to (p_blank, p_non_blank)
        # Empty prefix () starts with p_blank = 1.0 (log 0.0), p_non_blank = 0.0 (log -inf)
        beam = {(): (0.0, -float("inf"))}

        for t in range(T):
            frame = log_probs[t]
            # Apply blank penalty
            if self.blank_penalty != 0.0:
                frame[self.blank_id] -= self.blank_penalty
            # Apply space bonus
            if self.space_bonus != 0.0:
                frame[self.space_id] += self.space_bonus

            next_beam = defaultdict(lambda: (-float("inf"), -float("inf")))

            # Consider top candidates at this frame
            top_vals, top_indices = torch.topk(frame, min(self.beam_width * 2, V))

            for prefix, (p_b, p_nb) in beam.items():
                p_tot = torch.logaddexp(torch.tensor(p_b), torch.tensor(p_nb)).item()

                for log_p, c in zip(top_vals.tolist(), top_indices.tolist()):
                    if c == self.blank_id:
                        # Blank transition maintains current prefix
                        cur_b, cur_nb = next_beam[prefix]
                        new_b = torch.logaddexp(torch.tensor(cur_b), torch.tensor(p_tot + log_p)).item()
                        next_beam[prefix] = (new_b, cur_nb)
                    else:
                        # Phonotactic constraints on spaces:
                        # 1. No leading space
                        if not prefix and c == self.space_id:
                            continue
                        # 2. No consecutive spaces
                        if prefix and prefix[-1] == self.space_id and c == self.space_id:
                            continue

                        # Check if c repeats last non-blank token
                        if prefix and c == prefix[-1]:
                            # Repeating token: from blank state forms new token; from non-blank state collapses
                            # a) Collapse with previous non-blank:
                            cur_b, cur_nb = next_beam[prefix]
                            new_nb = torch.logaddexp(torch.tensor(cur_nb), torch.tensor(p_nb + log_p)).item()
                            next_beam[prefix] = (cur_b, new_nb)

                            # b) Add new token from blank state:
                            new_prefix = prefix + (c,)
                            cur_b, cur_nb = next_beam[new_prefix]
                            new_nb2 = torch.logaddexp(torch.tensor(cur_nb), torch.tensor(p_b + log_p)).item()
                            next_beam[new_prefix] = (cur_b, new_nb2)
                        else:
                            # Distinct token: extends prefix
                            new_prefix = prefix + (c,)
                            cur_b, cur_nb = next_beam[new_prefix]
                            new_nb = torch.logaddexp(torch.tensor(cur_nb), torch.tensor(p_tot + log_p)).item()
                            next_beam[new_prefix] = (cur_b, new_nb)

            # Prune beam to beam_width
            def score(item):
                pref, (pb, pnb) = item
                tot = torch.logaddexp(torch.tensor(pb), torch.tensor(pnb)).item()
                # Length normalization
                lp = max(1, len(pref)) ** self.length_penalty
                return tot / lp

            sorted_beam = sorted(next_beam.items(), key=score, reverse=True)
            beam = dict(sorted_beam[: self.beam_width])

        # Return highest scoring prefix
        best_prefix = max(beam.keys(), key=lambda p: torch.logaddexp(torch.tensor(beam[p][0]), torch.tensor(beam[p][1])).item())
        # Clean trailing spaces
        res = list(best_prefix)
        while res and res[-1] == self.space_id:
            res.pop()
        return res
