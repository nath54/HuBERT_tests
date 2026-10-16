"""
Trie-Constrained Vectorized Character Beam Search Rollout for Word Decoding.

Empowers Model 3's autoregressive character micro-head to decode word candidates
guided by a prefix tree (Trie) of vocabulary words, completely eliminating
non-word typos (e.g. 'gossiple' -> 'gospel') using Model 3's own neural logits.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor

from .roman_tokenizer import RomanCharTokenizer


class TrieNode:
    __slots__ = ("children", "is_word", "word_str")

    def __init__(self) -> None:
        self.children: Dict[int, "TrieNode"] = {}
        self.is_word: bool = False
        self.word_str: Optional[str] = None


class WordTrie:
    """Prefix tree over RomanCharTokenizer token IDs."""

    def __init__(self, rom_tok: RomanCharTokenizer) -> None:
        self.root: TrieNode = TrieNode()
        self.rom_tok: RomanCharTokenizer = rom_tok
        self.word_count: int = 0

    def insert(self, word: str) -> None:
        w_clean = word.lower().strip()
        if not w_clean:
            return
        token_ids = self.rom_tok.encode(w_clean, add_bos=False, add_eos=False)
        if not token_ids:
            return

        node = self.root
        for tok_id in token_ids:
            if tok_id not in node.children:
                node.children[tok_id] = TrieNode()
            node = node.children[tok_id]
        node.is_word = True
        node.word_str = w_clean
        self.word_count += 1

    @classmethod
    def from_word_list(cls, words: List[str], rom_tok: RomanCharTokenizer) -> "WordTrie":
        trie = cls(rom_tok)
        for w in words:
            trie.insert(w)
        return trie

    @classmethod
    def from_manifests(cls, manifest_paths: List[str], rom_tok: RomanCharTokenizer) -> "WordTrie":
        import json
        import os
        import re

        trie = cls(rom_tok)
        words: Set[str] = set()
        for p in manifest_paths:
            if not os.path.isfile(p):
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            words.add(item.lower().strip())
                        elif isinstance(item, dict):
                            text = item.get("transcript", item.get("text", ""))
                            clean = re.sub(r"[^\w\s\'\-àâäéèêëîïôöùûüçñ]", " ", text.lower())
                            for w in clean.split():
                                if len(w) > 0:
                                    words.add(w)
                elif isinstance(data, dict):
                    for k in data.keys():
                        words.add(k.lower())
            except Exception:
                pass
        for w in words:
            trie.insert(w)
        return trie

    @classmethod
    def from_vocab_file(cls, path: str, rom_tok: Optional[RomanCharTokenizer] = None) -> "WordTrie":
        if rom_tok is None:
            rom_tok = RomanCharTokenizer()
        return cls.from_manifests([path], rom_tok)


class BeamHypothesis:
    __slots__ = ("score", "tokens", "trie_node", "is_finished")

    def __init__(
        self,
        score: float,
        tokens: List[int],
        trie_node: Optional[TrieNode],
        is_finished: bool = False,
    ) -> None:
        self.score: float = score
        self.tokens: List[int] = tokens
        self.trie_node: Optional[TrieNode] = trie_node
        self.is_finished: bool = is_finished


def beam_search_character_rollout(
    model_m3: Any,
    z_words: Tensor,
    rom_tok: RomanCharTokenizer,
    trie: Optional[WordTrie] = None,
    beam_width: int = 1,
    max_len: int = 24,
    oov_penalty: float = 3.5,
    device: Optional[torch.device] = None,
) -> List[List[int]]:
    """Runs Trie-constrained character beam search across L words.

    Args:
        model_m3: Word decoder model (Model_WD or PhonoV75SpeechModel)
        z_words: [1, L, 512] word latent vectors
        rom_tok: RomanCharTokenizer
        trie: Optional WordTrie. If None, unconstrained beam search is run.
        beam_width: number of beams per word (1 = greedy argmax)
        max_len: maximum character rollout steps
        oov_penalty: penalty added to open-vocabulary character transitions when trie is active
        device: torch device

    Returns:
        List of L word token ID lists (excluding BOS, EOS, PAD).
    """
    if device is None:
        device = z_words.device

    decoder = getattr(model_m3, "decoder", model_m3)

    L = z_words.shape[1]
    if L == 0:
        return []

    expert_logits = decoder.macro_expert_router(z_words)
    top1_experts = expert_logits.argmax(dim=-1)[0]  # [L]
    expert_cond = decoder.expert_embeddings(top1_experts).unsqueeze(1)  # [L, 1, 512]
    z_windows = decoder.build_word_context_windows(z_words)[0]  # [L, W, 512]

    # If beam_width == 1 and trie is None, use fast vectorized greedy argmax
    if beam_width == 1 and trie is None:
        active_words = torch.full((L, max_len), rom_tok.pad_id, dtype=torch.long, device=device)
        active_words[:, 0] = rom_tok.bos_id
        finished = torch.zeros(L, dtype=torch.bool, device=device)
        for step in range(1, max_len):
            cur_inp = active_words[:, :step]
            logits_c, _ = decoder.micro_head(
                cur_inp, z_windows, acoustic_slices=None, path_bias=expert_cond, path_id=top1_experts
            )
            next_tokens = logits_c[:, -1, :].argmax(dim=-1)
            for l in range(L):
                if not finished[l]:
                    if next_tokens[l].item() in (rom_tok.eos_id, rom_tok.pad_id):
                        finished[l] = True
                    else:
                        active_words[l, step] = next_tokens[l]
            if finished.all():
                break
        pred_words = []
        for l in range(L):
            w_toks = [t.item() for t in active_words[l, 1:] if t.item() not in (rom_tok.bos_id, rom_tok.eos_id, rom_tok.pad_id)]
            pred_words.append(w_toks)
        return pred_words

    # Initialize beams for each of the L words
    root_node = trie.root if trie is not None else None
    word_beams: List[List[BeamHypothesis]] = [
        [BeamHypothesis(score=0.0, tokens=[rom_tok.bos_id], trie_node=root_node, is_finished=False)]
        for _ in range(L)
    ]

    for step in range(1, max_len):
        # Check if all beams across all words are finished
        all_done = all(all(b.is_finished for b in beams) for beams in word_beams)
        if all_done:
            break

        # Flatten active hypotheses to evaluate in a single batched micro_head call
        flat_inputs = []
        flat_z_win = []
        flat_exp_cond = []
        flat_exp_id = []
        beam_mapping = []  # (word_idx, beam_idx)

        for l in range(L):
            for b_idx, hyp in enumerate(word_beams[l]):
                # Pad hyp.tokens to current step length
                inp = hyp.tokens + [rom_tok.pad_id] * (step - len(hyp.tokens))
                flat_inputs.append(inp)
                flat_z_win.append(z_windows[l])
                flat_exp_cond.append(expert_cond[l])
                flat_exp_id.append(top1_experts[l])
                beam_mapping.append((l, b_idx))

        cur_inp_tensor = torch.tensor(flat_inputs, dtype=torch.long, device=device)
        cur_z_win = torch.stack(flat_z_win, dim=0)
        cur_exp_c = torch.stack(flat_exp_cond, dim=0)
        cur_exp_id = torch.stack(flat_exp_id, dim=0)

        logits_c, _ = decoder.micro_head(
            cur_inp_tensor, cur_z_win, acoustic_slices=None, path_bias=cur_exp_c, path_id=cur_exp_id
        )
        log_probs = F.log_softmax(logits_c[:, -1, :], dim=-1)  # [N, vocab_size]

        # Group candidate expansions per word
        word_candidates: List[List[BeamHypothesis]] = [[] for _ in range(L)]

        for n, (l, b_idx) in enumerate(beam_mapping):
            hyp = word_beams[l][b_idx]
            if hyp.is_finished:
                word_candidates[l].append(hyp)
                continue

            node = hyp.trie_node
            lp_row = log_probs[n]

            # 1. Option to emit <eos> (End of word)
            can_eos = False
            eos_score_adj = 0.0
            if node is None:
                can_eos = True
            elif node.is_word:
                can_eos = True
            elif len(hyp.tokens) > 3:
                # OOV fallback for words that reached reasonable length
                can_eos = True
                eos_score_adj = -oov_penalty

            if can_eos:
                eos_score = hyp.score + lp_row[rom_tok.eos_id].item() + eos_score_adj
                word_candidates[l].append(
                    BeamHypothesis(
                        score=eos_score,
                        tokens=hyp.tokens + [rom_tok.eos_id],
                        trie_node=node,
                        is_finished=True,
                    )
                )

            # 2. Option to emit valid characters from Trie
            if node is not None and node.children:
                for char_tok, next_node in node.children.items():
                    char_score = hyp.score + lp_row[char_tok].item()
                    word_candidates[l].append(
                        BeamHypothesis(
                            score=char_score,
                            tokens=hyp.tokens + [char_tok],
                            trie_node=next_node,
                            is_finished=False,
                        )
                    )

            # 3. Open-vocabulary fallback (top-3 tokens with penalty) if trie dead-ends
            if node is None or not node.children or len(word_candidates[l]) == 0:
                top_k_toks = torch.topk(lp_row, k=min(3, lp_row.shape[-1]))
                for val, idx_t in zip(top_k_toks.values, top_k_toks.indices):
                    tok_val = idx_t.item()
                    if tok_val not in (rom_tok.pad_id, rom_tok.bos_id):
                        is_fin = tok_val in (rom_tok.eos_id, rom_tok.eow_id)
                        pen = -oov_penalty if node is not None else 0.0
                        word_candidates[l].append(
                            BeamHypothesis(
                                score=hyp.score + val.item() + pen,
                                tokens=hyp.tokens + [tok_val],
                                trie_node=None,
                                is_finished=is_fin,
                            )
                        )

        # Select top-B hypotheses for each word
        for l in range(L):
            candidates = word_candidates[l]
            # Deduplicate by token tuple
            seen = set()
            unique_cands = []
            for c in candidates:
                key = tuple(c.tokens)
                if key not in seen:
                    seen.add(key)
                    unique_cands.append(c)
            # Sort by score descending
            unique_cands.sort(key=lambda h: h.score, reverse=True)
            word_beams[l] = unique_cands[:beam_width]

    # Extract best candidate tokens for each word
    pred_words = []
    for l in range(L):
        best_hyp = word_beams[l][0]
        w_toks = [
            t for t in best_hyp.tokens[1:]
            if t not in (rom_tok.bos_id, rom_tok.eos_id, rom_tok.pad_id, rom_tok.eow_id)
        ]
        pred_words.append(w_toks)

    return pred_words
