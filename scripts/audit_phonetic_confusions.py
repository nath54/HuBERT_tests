#!/usr/bin/env python3
"""Deep Phonetic Confusion Matrix & Bottleneck Audit on Clean Multilingual Validation Set.

Runs on CPU to analyze exact error distributions without interrupting GPU training.
Outputs a structured confusion analysis and saves empirical confusion probabilities
for downstream noise simulation.
"""

import difflib
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import torch
import torchaudio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.phono_v8_acoustic import PhonoV8AcousticBackbone
from src.data.phoneme_tokenizer import PhonemeTokenizer
from src.data.roman_tokenizer import RomanCharTokenizer
from src.data.target_extractors import PhonemeTargetExtractor
from src.data.multilingual_audio_dataset import MultilingualAudioDataset, AudioUtterance
from src.models.phono_v7_3_speech_model import decode_calibrated_phonemes


def decode_ctc_phoneme_ids(logits: torch.Tensor, blank_id: int = 1, pad_id: int = 0) -> List[int]:
    preds = logits.argmax(dim=-1).detach().cpu().tolist()
    collapsed = []
    prev = None
    for p in preds:
        if p != prev:
            if p != blank_id and p != pad_id:
                collapsed.append(p)
            prev = p
    return collapsed


def align_and_count_errors(
    ref_tokens: List[int],
    hyp_tokens: List[int],
    ph_tok: PhonemeTokenizer,
) -> Tuple[Counter, Counter, Counter, Counter]:
    """Align ref and hyp sequences and collect exact substitution, deletion, and insertion counts."""
    matcher = difflib.SequenceMatcher(None, ref_tokens, hyp_tokens)
    sub_counts = Counter()
    del_counts = Counter()
    ins_counts = Counter()
    match_counts = Counter()

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for r in ref_tokens[i1:i2]:
                match_counts[r] += 1
        elif tag == "delete":
            for r in ref_tokens[i1:i2]:
                del_counts[r] += 1
        elif tag == "insert":
            for h in hyp_tokens[j1:j2]:
                ins_counts[h] += 1
        elif tag == "replace":
            r_slice = ref_tokens[i1:i2]
            h_slice = hyp_tokens[j1:j2]
            min_len = min(len(r_slice), len(h_slice))
            for k in range(min_len):
                sub_counts[(r_slice[k], h_slice[k])] += 1
            if len(r_slice) > min_len:
                for r in r_slice[min_len:]:
                    del_counts[r] += 1
            elif len(h_slice) > min_len:
                for h in h_slice[min_len:]:
                    ins_counts[h] += 1

    return sub_counts, del_counts, ins_counts, match_counts


def main():
    print("=" * 80)
    print("🔬 PHONETIC CONFUSION & BOTTLENECK AUDIT (Phono-V8.0 on CPU)")
    print("=" * 80)

    val_manifest = "data/clean_multilingual_val.json"
    with open(val_manifest) as f:
        val_raw = json.load(f)

    val_utts = [
        AudioUtterance(id=x["id"], audio_path=x["audio_path"], transcript=x["transcript"], lang=x["lang"])
        for x in val_raw
    ]

    ph_tokenizer = PhonemeTokenizer()
    ph_extractor = PhonemeTargetExtractor(ph_tokenizer)
    roman_tok = RomanCharTokenizer()
    val_dataset = MultilingualAudioDataset(val_utts, roman_tokenizer=roman_tok, phoneme_extractor=ph_extractor)

    # Evaluate 15 balanced samples per language = 60 utterances
    subset_items = []
    lang_counts = {}
    for i in range(len(val_dataset)):
        try:
            sample = val_dataset[i]
            lang = sample["lang"]
            if lang_counts.get(lang, 0) < 15:
                subset_items.append(sample)
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            if len(subset_items) >= 60:
                break
        except Exception:
            continue

    print(f"\n📦 Loading Model Checkpoint on CPU...")
    ckpt_path = "checkpoints/phono_v8_acoustic/latest_v8_acoustic.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = PhonoV8AcousticBackbone().to("cpu")
    model.load_state_dict(ckpt["acoustic_state_dict"], strict=False)
    model.eval()

    print(f"   • Auditing {len(subset_items)} balanced utterances ({lang_counts}) on CPU...\n")

    overall_sub = Counter()
    overall_del = Counter()
    overall_ins = Counter()
    overall_match = Counter()
    lang_subs = defaultdict(Counter)

    total_ref_tokens = 0
    total_errors = 0
    space_token = 8  # '<space>'

    for idx, item in enumerate(subset_items):
        lang = item["lang"]
        gt_tokens = item["phoneme_targets"].tolist()
        wav = item["audio"].unsqueeze(0)

        with torch.no_grad():
            out = model(wav)
            logits = out["logits"][0]

        pred_tokens = decode_ctc_phoneme_ids(logits, blank_id=1, pad_id=0)

        subs, dels, inss, matches = align_and_count_errors(gt_tokens, pred_tokens, ph_tokenizer)
        overall_sub.update(subs)
        overall_del.update(dels)
        overall_ins.update(inss)
        overall_match.update(matches)

        for pair, c in subs.items():
            lang_subs[lang][pair] += c

        total_ref_tokens += len(gt_tokens)
        total_errors += (sum(subs.values()) + sum(dels.values()) + sum(inss.values()))

    print(f"📊 Overall Statistics across {total_ref_tokens} reference phonemes:")
    print(f"   • Total Errors: {total_errors} (Raw Error Rate: {(total_errors / total_ref_tokens) * 100:.2f}%)")
    print(f"   • Substitutions: {sum(overall_sub.values())} ({(sum(overall_sub.values()) / total_errors) * 100:.1f}% of errors)")
    print(f"   • Deletions    : {sum(overall_del.values())} ({(sum(overall_del.values()) / total_errors) * 100:.1f}% of errors)")
    print(f"   • Insertions   : {sum(overall_ins.values())} ({(sum(overall_ins.values()) / total_errors) * 100:.1f}% of errors)")

    # Space Error Analysis
    space_del = overall_del[space_token]
    space_ins = overall_ins[space_token]
    space_sub_out = sum(c for (r, h), c in overall_sub.items() if r == space_token)
    space_sub_in = sum(c for (r, h), c in overall_sub.items() if h == space_token)
    total_space_errors = space_del + space_ins + space_sub_out + space_sub_in
    print(f"\n🏷️  Word Boundary (<space>) Error Analysis:")
    print(f"   • Total Space Errors: {total_space_errors} ({(total_space_errors / total_errors) * 100:.1f}% of ALL errors!)")
    print(f"   • Missed Spaces (Deletions): {space_del}")
    print(f"   • Extra Spaces  (Insertions): {space_ins}")
    print(f"   • Space substituted with phone: {space_sub_out}")
    print(f"   • Phone substituted with space: {space_sub_in}")

    # Top 20 Most Confused Phoneme Substitution Pairs
    print(f"\n🔍 Top 20 Most Frequent Phoneme Substitution Confusions:")
    print(f"   {'Ref Phone':15s} {'Pred Phone':15s} {'Count':8s} {'Phonetic Category'}")
    print(f"   {'-'*15} {'-'*15} {'-'*8} {'-'*30}")
    top_subs = overall_sub.most_common(20)
    for (r_id, h_id), cnt in top_subs:
        r_str = ph_tokenizer.id_to_token.get(r_id, f"ID_{r_id}")
        h_str = ph_tokenizer.id_to_token.get(h_id, f"ID_{h_id}")
        cat = "Space misalignment" if r_id == space_token or h_id == space_token else "Acoustic Allophone / Vowel"
        print(f"   {r_str:15s} {h_str:15s} {cnt:<8d} {cat}")

    # Top Deletions
    print(f"\n🗑️  Top 10 Most Frequently Deleted Phonemes:")
    for p_id, cnt in overall_del.most_common(10):
        p_str = ph_tokenizer.id_to_token.get(p_id, f"ID_{p_id}")
        print(f"   • {p_str:15s}: {cnt:4d} deletions")

    # Save structured confusion matrix to JSON for PhoneticNoiseSimulator
    output_matrix = {
        "metadata": {
            "model": "phono_v8_acoustic",
            "step": ckpt.get("step", 26000),
            "total_ref_tokens": total_ref_tokens,
            "total_errors": total_errors,
        },
        "error_rates": {
            "substitutions": sum(overall_sub.values()) / max(1, total_ref_tokens),
            "deletions": sum(overall_del.values()) / max(1, total_ref_tokens),
            "insertions": sum(overall_ins.values()) / max(1, total_ref_tokens),
            "space_error_share": total_space_errors / max(1, total_errors),
        },
        "substitution_probabilities": {},
    }

    # Group substitutions by reference phoneme to compute P(hyp | ref, error)
    ref_grouped = defaultdict(dict)
    ref_totals = defaultdict(int)
    for (r, h), cnt in overall_sub.items():
        r_str = ph_tokenizer.id_to_token.get(r, str(r))
        h_str = ph_tokenizer.id_to_token.get(h, str(h))
        ref_grouped[r_str][h_str] = cnt
        ref_totals[r_str] += cnt

    for r_str, hyp_map in ref_grouped.items():
        tot = ref_totals[r_str]
        output_matrix["substitution_probabilities"][r_str] = {
            h_str: round(cnt / tot, 4) for h_str, cnt in sorted(hyp_map.items(), key=lambda x: -x[1])
        }

    os.makedirs("data", exist_ok=True)
    out_file = "data/phoneme_confusion_matrix.json"
    with open(out_file, "w") as f:
        json.dump(output_matrix, f, indent=2)
    print(f"\n💾 Saved Empirical Phoneme Confusion Matrix to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
