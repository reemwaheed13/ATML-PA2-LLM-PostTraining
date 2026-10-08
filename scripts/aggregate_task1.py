"""Aggregate Task 1 results into machine-readable tables. Pure read; no model.

Reads the per-run JSONs written by evaluate.py and analyze_length.py and emits:
  results/task1_dpo/task1_beta_table.csv        (standard + 3 beta forks)
  results/task1_dpo/task1_length_stratum.csv    (per-stratum pref acc: std vs length-balanced)
  results/task1_dpo/task1_wordlimit.csv         (word-limit compliance + gen length)
  results/task1_dpo/task1_summary.json          (everything, combined)

Run:  python -m scripts.aggregate_task1 --config configs/dpo.yaml
"""

from __future__ import annotations

import argparse
import csv

from common.data import load_yaml, repo_path
from common.logging_utils import load_json, save_json

BETA_RUNS = ["standard", "beta_0.03", "beta_0.10", "beta_0.30"]
BETA_COLS = ["run", "beta", "example_budget", "budget_note", "held_out_dpo_loss",
             "preference_accuracy", "kl_sampled", "reward_mean", "length_mean", "length_std", "length_iqr"]
LENGTH_RUNS = ["standard", "length_balanced"]


def _write_csv(path, header, rows):
    with repo_path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(c, "") for c in header])


def beta_table(results_dir):
    rows = []
    for run in BETA_RUNS:
        ev = results_dir / f"{run}_eval.json"
        if not ev.exists():
            print(f"  (skip {run}: no {run}_eval.json)")
            continue
        d = load_json(ev)
        meta_p = results_dir / f"{run}_meta.json"
        budget = load_json(meta_p).get("num_examples") if meta_p.exists() else None
        rows.append({
            "run": run,
            "beta": d.get("beta"),
            "example_budget": budget,
            "budget_note": "full 1-epoch" if run == "standard" else "600-ex short fork",
            "held_out_dpo_loss": d.get("held_out_dpo_loss"),
            "preference_accuracy": d.get("preference_accuracy"),
            "kl_sampled": d.get("kl_sampled"),
            "reward_mean": d.get("reward_mean"),
            "length_mean": d.get("length_mean"),
            "length_std": d.get("length_std"),
            "length_iqr": d.get("length_iqr"),
        })
    return rows


def length_stratum_table(results_dir):
    data = {}
    for run in LENGTH_RUNS:
        p = results_dir / f"{run}_length_stratum.json"
        if p.exists():
            data[run] = load_json(p)
        else:
            print(f"  (skip {run}: no {run}_length_stratum.json)")
    strata = set()
    for run in data:
        strata.update(data[run]["per_stratum"].keys())
    rows = []
    for s in sorted(strata):
        row = {"stratum": s}
        for run in LENGTH_RUNS:
            ps = data.get(run, {}).get("per_stratum", {}).get(s, {})
            row[f"{run}_pref_acc"] = ps.get("preference_accuracy")
            row[f"{run}_n"] = ps.get("n")
        rows.append(row)
    # overall row
    overall = {"stratum": "OVERALL"}
    for run in LENGTH_RUNS:
        overall[f"{run}_pref_acc"] = data.get(run, {}).get("overall", {}).get("preference_accuracy")
        overall[f"{run}_n"] = data.get(run, {}).get("overall", {}).get("n")
    rows.append(overall)
    header = ["stratum"] + [f"{r}_pref_acc" for r in LENGTH_RUNS] + [f"{r}_n" for r in LENGTH_RUNS]
    return header, rows, data


def wordlimit_table(results_dir):
    rows = []
    for run in LENGTH_RUNS:
        p = results_dir / f"{run}_wordlimit.json"
        if not p.exists():
            print(f"  (skip {run}: no {run}_wordlimit.json)")
            continue
        d = load_json(p)
        rows.append({
            "run": run,
            "compliance_rate": d.get("compliance_rate"),
            "n_with_limit": d.get("n_with_limit"),
            "length_mean": d.get("length_mean"),
            "length_std": d.get("length_std"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])

    print("beta table:")
    brows = beta_table(results_dir)
    _write_csv(results_dir / "task1_beta_table.csv", BETA_COLS, brows)

    print("length-stratum table:")
    lheader, lrows, ldata = length_stratum_table(results_dir)
    _write_csv(results_dir / "task1_length_stratum.csv", lheader, lrows)

    print("word-limit table:")
    wrows = wordlimit_table(results_dir)
    _write_csv(results_dir / "task1_wordlimit.csv",
               ["run", "compliance_rate", "n_with_limit", "length_mean", "length_std"], wrows)

    save_json(results_dir / "task1_summary.json", {
        "beta_table": brows,
        "length_stratum_table": lrows,
        "length_stratum_raw": ldata,
        "wordlimit_table": wrows,
    })
    print(f"wrote task1_beta_table.csv, task1_length_stratum.csv, task1_wordlimit.csv, task1_summary.json "
          f"to {results_dir}")


if __name__ == "__main__":
    main()
