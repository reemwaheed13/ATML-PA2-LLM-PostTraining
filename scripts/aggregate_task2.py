"""Aggregate Task 2 (PPO) results into machine-readable tables. Pure read; no model.

Reads the per-run JSONs/JSONLs written by continue_train.py, evaluate.py, and
analyze_clipping.py and emits to results/task2_ppo/:
  task2_continuation.csv   standard 20-update per-update trajectory
  task2_clip_table.csv     cached clip/affected fractions + eps-fork held-out metrics
  task2_kl_table.csv       kl-fork held-out metrics
  task2_summary.json       everything, combined

Run:  python -m scripts.aggregate_task2 --config configs/ppo.yaml
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from task2_ppo.continue_train import KL_CONVENTION, fork_name

# Stability statistic (defined once): std of the per-update policy loss over a run.
STABILITY = "policy_loss_std_over_updates"

CONT_COLS = ["update", "reward_mean", "terminal_reward_mean", "kl_sampled", "response_length_mean",
             "policy_loss", "value_loss", "entropy", "clip_fraction",
             "policy_grad_norm", "value_grad_norm", "any_step_skipped"]


def _write_csv(path, header, rows):
    with repo_path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(c, "") for c in header])


def per_update(train_path):
    """Collapse the per-(update,epoch) train log to one row per update: rollout metrics
    are constant within an update (take first); optimization metrics are averaged over the
    ppo epochs; a skipped step anywhere in the update sets any_step_skipped."""
    rows = read_jsonl(train_path)
    by_update = {}
    for r in rows:
        by_update.setdefault(r["update"], []).append(r)
    out = []
    for u in sorted(by_update):
        recs = by_update[u]
        out.append({
            "update": u,
            "reward_mean": recs[0]["reward_mean"],
            "terminal_reward_mean": recs[0].get("terminal_reward_mean"),
            "kl_sampled": recs[0]["kl_sampled"],
            "response_length_mean": recs[0]["response_length_mean"],
            "policy_loss": float(np.mean([x["policy_loss"] for x in recs])),
            "value_loss": float(np.mean([x["value_loss"] for x in recs])),
            "entropy": float(np.mean([x["entropy"] for x in recs])),
            "clip_fraction": float(np.mean([x["clip_fraction"] for x in recs])),
            "policy_grad_norm": float(np.mean([x["policy_grad_norm"] for x in recs])),
            "value_grad_norm": float(np.mean([x["value_grad_norm"] for x in recs])),
            "any_step_skipped": any(x.get("policy_step_skipped") or x.get("value_step_skipped") for x in recs),
        })
    return out


def _stability(train_path):
    if not train_path.exists():
        return None
    pu = per_update(train_path)
    if not pu:
        return None
    return float(np.std([r["policy_loss"] for r in pu]))


def _eval_metrics(results_dir, name):
    p = results_dir / f"{name}_eval.json"
    if not p.exists():
        return None
    d = load_json(p)
    return {k: d.get(k) for k in ["reward_mean", "reward_std", "kl_sampled", "length_mean",
                                   "length_std", "length_iqr", "entropy_mean"]}


def clip_table(results_dir, cfg):
    cached = load_json(results_dir / "clip_cached.json") if (results_dir / "clip_cached.json").exists() else {"per_eps": {}}
    kl = float(cfg["kl_beta"])
    rows = []
    for eps in [float(e) for e in cfg["clip_values"]]:
        name = fork_name(eps, kl)
        ce = cached.get("per_eps", {}).get(f"{eps:.2f}", {})
        row = {
            "eps": eps, "kl_beta": kl, "fork": name,
            "cached_clip_fraction": ce.get("clip_fraction"),
            "cached_affected_fraction": ce.get("affected_token_fraction"),
            "cached_surrogate_mean": ce.get("clipped_surrogate_mean"),
            STABILITY: _stability(results_dir / f"{name}_train.jsonl"),
        }
        ev = _eval_metrics(results_dir, name)
        if ev:
            row.update({f"heldout_{k}": v for k, v in ev.items()})
        rows.append(row)
    return rows


def kl_table(results_dir, cfg):
    eps = float(cfg["clip_epsilon"])
    rows = []
    for kl in [float(b) for b in cfg["kl_values"]]:
        name = fork_name(eps, kl)
        row = {"eps": eps, "kl_beta": kl, "fork": name,
               STABILITY: _stability(results_dir / f"{name}_train.jsonl")}
        ev = _eval_metrics(results_dir, name)
        if ev:
            row.update({f"heldout_{k}": v for k, v in ev.items()})
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])

    cont = per_update(results_dir / f"{args.name}_train.jsonl") if (results_dir / f"{args.name}_train.jsonl").exists() else []
    _write_csv(results_dir / "task2_continuation.csv", CONT_COLS, cont)

    crows = clip_table(results_dir, cfg)
    chead = ["eps", "kl_beta", "fork", "cached_clip_fraction", "cached_affected_fraction",
             "cached_surrogate_mean", STABILITY, "heldout_reward_mean", "heldout_kl_sampled",
             "heldout_length_mean", "heldout_length_std", "heldout_entropy_mean"]
    _write_csv(results_dir / "task2_clip_table.csv", chead, crows)

    krows = kl_table(results_dir, cfg)
    khead = ["eps", "kl_beta", "fork", STABILITY, "heldout_reward_mean", "heldout_kl_sampled",
             "heldout_entropy_mean", "heldout_length_mean", "heldout_length_std"]
    _write_csv(results_dir / "task2_kl_table.csv", khead, krows)

    summary_p = results_dir / f"{args.name}_summary.json"
    save_json(results_dir / "task2_summary.json", {
        "standard_summary": load_json(summary_p) if summary_p.exists() else None,
        "stability_statistic": STABILITY,
        "kl_convention": KL_CONVENTION,
        "continuation_trajectory": cont,
        "clip_table": crows,
        "kl_table": krows,
    })
    print(f"wrote task2_continuation.csv, task2_clip_table.csv, task2_kl_table.csv, task2_summary.json to {results_dir}")


if __name__ == "__main__":
    main()
