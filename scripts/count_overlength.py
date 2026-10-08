"""Read-only diagnostic: how many DPO examples have an over-length PROMPT.

Reports, per dataset, the number/fraction of examples whose rendered prompt
(chat template, add_generation_prompt=True) has >= max_length tokens -- the exact
condition that raises in common/data.py:122 -- plus prompt-length distribution
and how many would be saved at larger candidate caps. For the length-controlled
sets it also breaks the over-length count down per stratum, so we can see whether
a prompt-length filter would un-balance the length-confounding experiment.

Run:  python -m scripts.count_overlength --config configs/dpo.yaml
Writes: results/task1_dpo/overlength_report.json  (diagnostic record)
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np

from common.data import (
    load_yaml,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.logging_utils import save_json
from common.metrics import word_count
from common.models import load_tokenizer

STRATUM_KEYS = ["stratum", "length_stratum", "length_bucket", "bucket", "group", "category", "length_group"]
CANDIDATE_CAPS = [768, 896, 1024, 1152, 1280, 1536]


def prompt_length(tokenizer, row) -> int:
    msgs = prompt_messages_from_preference(row)
    ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    return len(ids)


def detect_stratum_key(rows) -> str | None:
    keys = set(rows[0].keys())
    for k in STRATUM_KEYS:
        if k in keys:
            return k
    return None


def derived_stratum(row) -> str:
    yc, yr = preference_responses(row)
    c, r = word_count(yc), word_count(yr)
    if c > r:
        return "preferred_longer"
    if c < r:
        return "rejected_longer"
    return "matched"


def analyze(name, path, tokenizer, max_length, with_strata):
    rows = read_jsonl(path)
    lengths = [prompt_length(tokenizer, row) for row in rows]
    arr = np.array(lengths)
    over_idx = [i for i, L in enumerate(lengths) if L >= max_length]

    rec = {
        "dataset": name,
        "path": str(path),
        "total": len(rows),
        "over_count": len(over_idx),
        "over_fraction": (len(over_idx) / len(rows)) if rows else 0.0,
        "prompt_len_min": int(arr.min()),
        "prompt_len_median": float(np.median(arr)),
        "prompt_len_mean": float(arr.mean()),
        "prompt_len_p95": float(np.percentile(arr, 95)),
        "prompt_len_p99": float(np.percentile(arr, 99)),
        "prompt_len_max": int(arr.max()),
        "over_at_cap": {str(c): int(np.sum(arr >= c)) for c in CANDIDATE_CAPS},
        "first_row_keys": sorted(rows[0].keys()),
    }

    if with_strata:
        skey = detect_stratum_key(rows)
        rec["stratum_key"] = skey or "(none found -> derived from response word counts)"
        strat = [row[skey] if skey else derived_stratum(row) for row in rows]
        total_by = Counter(str(s) for s in strat)
        over_by = Counter(str(strat[i]) for i in over_idx)
        rec["per_stratum_total"] = dict(total_by)
        rec["per_stratum_over"] = {s: over_by.get(s, 0) for s in total_by}
        retained_by = {s: total_by[s] - over_by.get(s, 0) for s in total_by}
        rec["per_stratum_retained"] = retained_by
        uneven = max(over_by.values(), default=0) - min((over_by.get(s, 0) for s in total_by), default=0)
        rec["strata_retained_uneven"] = len(set(retained_by.values())) > 1
        if rec["strata_retained_uneven"]:
            rec["WARNING"] = "Prompt-length filter would leave UNEVEN strata; see per_stratum_retained."
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    max_length = int(cfg["max_sequence_length"])
    tokenizer = load_tokenizer(cfg["base_model"])

    specs = [
        ("dpo_standard_train", cfg["paths"]["dpo_standard_train"], False),
        ("dpo_standard_eval", cfg["paths"]["dpo_standard_eval"], False),
        ("dpo_length_balanced_train", cfg["paths"]["dpo_length_train"], True),
        ("dpo_length_stratified_eval", cfg["paths"]["dpo_length_eval"], True),
    ]
    report = {"max_length": max_length, "config": args.config, "datasets": []}
    for name, path, with_strata in specs:
        rec = analyze(name, repo_path(path), tokenizer, max_length, with_strata)
        report["datasets"].append(rec)
        print(f"\n=== {name} ===")
        print(f"  total={rec['total']}  over(>= {max_length})={rec['over_count']}  ({100*rec['over_fraction']:.2f}%)")
        print(f"  prompt_len: median={rec['prompt_len_median']:.0f} p95={rec['prompt_len_p95']:.0f} "
              f"p99={rec['prompt_len_p99']:.0f} max={rec['prompt_len_max']}")
        print(f"  over at caps: {rec['over_at_cap']}")
        if with_strata:
            print(f"  stratum_key: {rec['stratum_key']}")
            print(f"  per-stratum total:    {rec['per_stratum_total']}")
            print(f"  per-stratum over:     {rec['per_stratum_over']}")
            print(f"  per-stratum retained: {rec['per_stratum_retained']}")
            if rec.get("WARNING"):
                print(f"  ** {rec['WARNING']} **")

    out = repo_path(cfg["results_dir"]) / "overlength_report.json"
    save_json(out, report)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
