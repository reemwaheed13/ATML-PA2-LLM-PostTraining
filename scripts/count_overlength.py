"""Read-only diagnostic for DPO dataset lengths.

Reports, per dataset: how many examples have an over-length PROMPT (>= max_length
tokens -- the condition that raises in common/data.py), prompt-length distribution,
over-counts at larger candidate caps, and a per-stratum breakdown. For the TRAIN
sets it additionally reports how many responses get RIGHT-TRUNCATED at caps 768 vs
1152 (prompt kept intact, response clipped to the remaining budget).

Uses the same stratum/length helpers as the real filter (common.data), so the
numbers here match what the filter will do.

Run:  python -m scripts.count_overlength --config configs/dpo.yaml
Writes: results/task1_dpo/overlength_report.json
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from common.data import (
    detect_stratum_key,
    load_yaml,
    preference_responses,
    prompt_token_length,
    read_jsonl,
    repo_path,
)
from common.logging_utils import save_json
from common.models import load_tokenizer

CANDIDATE_CAPS = [768, 896, 1024, 1152, 1280, 1536]
RESPONSE_TRUNC_CAPS = [768, 1152]


def response_truncation(tokenizer, rows, caps):
    """At each cap, among examples whose prompt fits (prompt < cap), how many have a
    right-truncated chosen / rejected / either response. Mirrors encode_prompt_response:
    content budget = max(0, (cap - prompt_len) - 1), reserving one slot for EOS."""
    triples = []
    for row in rows:
        pn = prompt_token_length(tokenizer, row)
        yc, yr = preference_responses(row)
        cl = len(tokenizer(yc, add_special_tokens=False)["input_ids"])
        rl = len(tokenizer(yr, add_special_tokens=False)["input_ids"])
        triples.append((pn, cl, rl))
    out = {}
    for cap in caps:
        considered = c_tr = r_tr = either = 0
        for pn, cl, rl in triples:
            if pn >= cap:
                continue  # prompt-dropped example, not a response-truncation case
            considered += 1
            budget = max(0, (cap - pn) - 1)
            ct, rt = cl > budget, rl > budget
            c_tr += ct; r_tr += rt; either += (ct or rt)
        out[str(cap)] = {
            "considered": considered,
            "chosen_truncated": c_tr,
            "rejected_truncated": r_tr,
            "either_truncated": either,
            "either_fraction": (either / considered) if considered else 0.0,
        }
    return out


def analyze(name, path, tokenizer, max_length, with_response):
    rows = read_jsonl(path)
    lengths = np.array([prompt_token_length(tokenizer, row) for row in rows])
    over_idx = [i for i, L in enumerate(lengths) if L >= max_length]

    rec = {
        "dataset": name, "path": str(path), "total": len(rows),
        "over_count": len(over_idx),
        "over_fraction": (len(over_idx) / len(rows)) if rows else 0.0,
        "prompt_len_median": float(np.median(lengths)),
        "prompt_len_p95": float(np.percentile(lengths, 95)),
        "prompt_len_p99": float(np.percentile(lengths, 99)),
        "prompt_len_max": int(lengths.max()),
        "over_at_cap": {str(c): int(np.sum(lengths >= c)) for c in CANDIDATE_CAPS},
        "first_row_keys": sorted(rows[0].keys()),
    }

    # Per-stratum ONLY when the dataset has a real stratum field; never derived.
    skey = detect_stratum_key(rows)
    rec["stratum_key"] = skey
    if skey:
        strat = [str(row[skey]) for row in rows]
        total_by = Counter(strat)
        over_by = Counter(strat[i] for i in over_idx)
        rec["per_stratum_total"] = dict(total_by)
        rec["per_stratum_over"] = {s: over_by.get(s, 0) for s in total_by}
        rec["per_stratum_retained"] = {s: total_by[s] - over_by.get(s, 0) for s in total_by}
        rec["strata_retained_uneven"] = len(set(rec["per_stratum_retained"].values())) > 1

    if with_response:
        rec["response_truncation"] = response_truncation(tokenizer, rows, RESPONSE_TRUNC_CAPS)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    max_length = int(cfg["max_sequence_length"])
    tokenizer = load_tokenizer(cfg["base_model"])

    specs = [
        # name, path, with_response
        ("dpo_standard_train", cfg["paths"]["dpo_standard_train"], True),
        ("dpo_standard_eval", cfg["paths"]["dpo_standard_eval"], False),
        ("dpo_length_balanced_train", cfg["paths"]["dpo_length_train"], True),
        ("dpo_length_stratified_eval", cfg["paths"]["dpo_length_eval"], False),
    ]
    report = {"max_length": max_length, "config": args.config, "datasets": []}
    for name, path, with_response in specs:
        rec = analyze(name, repo_path(path), tokenizer, max_length, with_response)
        report["datasets"].append(rec)
        print(f"\n=== {name} ===")
        print(f"  total={rec['total']}  over(>= {max_length})={rec['over_count']}  ({100*rec['over_fraction']:.2f}%)")
        print(f"  prompt_len: median={rec['prompt_len_median']:.0f} p95={rec['prompt_len_p95']:.0f} "
              f"p99={rec['prompt_len_p99']:.0f} max={rec['prompt_len_max']}")
        print(f"  over at caps: {rec['over_at_cap']}")
        if rec.get("stratum_key"):
            print(f"  stratum_key: {rec['stratum_key']}")
            print(f"  per-stratum retained: {rec['per_stratum_retained']}  (uneven={rec['strata_retained_uneven']})")
        if with_response:
            for cap, r in rec["response_truncation"].items():
                print(f"  response truncated @cap {cap}: either={r['either_truncated']}/{r['considered']} "
                      f"({100*r['either_fraction']:.1f}%)  chosen={r['chosen_truncated']} rejected={r['rejected_truncated']}")

    out = repo_path(cfg["results_dir"]) / "overlength_report.json"
    save_json(out, report)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
