"""Aggregate Task 2 (PPO) results into report-ready tables (and, by default, figures).
Pure read; no model, no new experiments.

Reads the per-run JSONs/JSONLs written by continue_train.py, evaluate.py, and
analyze_clipping.py and emits to results/task2_ppo/:
  task2_standard_summary.csv   one-row standard-continuation summary (Table 1)
  task2_continuation.csv       standard 20-update per-update trajectory
  task2_clip_table.csv         cached clip/affected fractions + eps-fork held-out (Table 2)
  task2_kl_table.csv           kl-fork held-out metrics (Table 3)
  task2_summary.json           everything, combined
Unless --tables-only is passed, it then calls scripts.plot_task2 to (re)generate the
report figures from the same files.

Run:  python -m scripts.aggregate_task2 --config configs/ppo.yaml
      python -m scripts.aggregate_task2 --config configs/ppo.yaml --tables-only
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

# Table 1: one-row standard-continuation summary.
STD_SUMMARY_COLS = ["updates", "wall_seconds", "peak_vram_gb",
                    "reward_first", "reward_last", "kl_first", "kl_last",
                    "entropy_first", "entropy_last", "length_first", "length_last",
                    "mean_clip_fraction", "n_updates_step_skipped"]

# Table 2: fixed column order (clipping study). cached_affected_note is appended so the
# absence of the Required-Evidence affected-token fraction is always visible, never silent.
CLIP_COLS = ["eps", "cached_clip_frac", "cached_affected_frac", "clipped_surrogate",
             "heldout_reward_mean", "heldout_kl_sampled", "heldout_length_mean",
             "heldout_length_std", STABILITY, "cached_affected_note"]

# Table 3: kl study, with the shared-fork flag.
KL_COLS = ["eps", "kl_beta", "fork", "shared_with_clip_study", STABILITY,
           "heldout_reward_mean", "heldout_kl_sampled", "heldout_entropy_mean",
           "heldout_length_mean", "heldout_length_std"]


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


def standard_summary_row(results_dir, name, summary):
    """Table 1: one row from the per-update trajectory + the run summary."""
    train_path = results_dir / f"{name}_train.jsonl"
    if not train_path.exists():
        return None
    pu = per_update(train_path)
    if not pu:
        return None
    first, last = pu[0], pu[-1]
    return {
        "updates": (summary or {}).get("updates", len(pu)),
        "wall_seconds": (summary or {}).get("wall_seconds"),
        "peak_vram_gb": (summary or {}).get("peak_vram_gb"),
        "reward_first": first["reward_mean"], "reward_last": last["reward_mean"],
        "kl_first": first["kl_sampled"], "kl_last": last["kl_sampled"],
        "entropy_first": first["entropy"], "entropy_last": last["entropy"],
        "length_first": first["response_length_mean"], "length_last": last["response_length_mean"],
        "mean_clip_fraction": float(np.mean([r["clip_fraction"] for r in pu])),
        "n_updates_step_skipped": int(sum(1 for r in pu if r["any_step_skipped"])),
    }


def clip_table(results_dir, cfg):
    cpath = results_dir / "clip_cached.json"
    cached = load_json(cpath) if cpath.exists() else {}
    per = cached.get("per_eps", {})
    adv_avail = cached.get("affected_token_available")
    adv_src = cached.get("advantage_source")
    kl = float(cfg["kl_beta"])
    rows = []
    for eps in [float(e) for e in cfg["clip_values"]]:
        name = fork_name(eps, kl)
        ce = per.get(f"{eps:.2f}", {})
        affected = ce.get("affected_token_fraction")
        surrogate = ce.get("clipped_surrogate_mean")
        note = ""
        if affected is None:
            # Required Evidence absent -> explicit NA + a note naming why (never dropped silently).
            affected = "NA"
            surrogate = "NA" if surrogate is None else surrogate
            note = (f"affected-token fraction (Required Evidence) absent: "
                    f"affected_token_available={adv_avail}, advantage_source={adv_src}; "
                    f"cache lacked per-token advantages")
        row = {
            "eps": eps,
            "cached_clip_frac": ce.get("clip_fraction"),
            "cached_affected_frac": affected,
            "clipped_surrogate": surrogate,
            STABILITY: _stability(results_dir / f"{name}_train.jsonl"),
            "cached_affected_note": note,
        }
        ev = _eval_metrics(results_dir, name)
        if ev:
            row.update({f"heldout_{k}": v for k, v in ev.items()})
        rows.append(row)
    return rows


def kl_table(results_dir, cfg):
    eps = float(cfg["clip_epsilon"])
    kl_std = float(cfg["kl_beta"])
    rows = []
    for kl in [float(b) for b in cfg["kl_values"]]:
        name = fork_name(eps, kl)
        row = {"eps": eps, "kl_beta": kl, "fork": name,
               # The (eps=0.20, beta=0.10) fork is the one trained by the clipping study.
               "shared_with_clip_study": bool(abs(kl - kl_std) < 1e-9),
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
    ap.add_argument("--tables-only", action="store_true", help="skip figure (re)generation")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])

    train_path = results_dir / f"{args.name}_train.jsonl"
    cont = per_update(train_path) if train_path.exists() else []
    _write_csv(results_dir / "task2_continuation.csv", CONT_COLS, cont)

    summary_p = results_dir / f"{args.name}_summary.json"
    summary = load_json(summary_p) if summary_p.exists() else None
    std_row = standard_summary_row(results_dir, args.name, summary)
    _write_csv(results_dir / "task2_standard_summary.csv", STD_SUMMARY_COLS, [std_row] if std_row else [])

    crows = clip_table(results_dir, cfg)
    _write_csv(results_dir / "task2_clip_table.csv", CLIP_COLS, crows)

    krows = kl_table(results_dir, cfg)
    _write_csv(results_dir / "task2_kl_table.csv", KL_COLS, krows)

    save_json(results_dir / "task2_summary.json", {
        "standard_summary": summary,
        "standard_summary_row": std_row,
        "stability_statistic": STABILITY,
        "kl_convention": KL_CONVENTION,
        "continuation_trajectory": cont,
        "clip_table": crows,
        "kl_table": krows,
    })
    print(f"wrote task2_standard_summary.csv, task2_continuation.csv, task2_clip_table.csv, "
          f"task2_kl_table.csv, task2_summary.json to {results_dir}")

    if not args.tables_only:
        # Lazy import so --tables-only never pulls in matplotlib.
        from scripts.plot_task2 import make_diagnostics, make_reward_vs_kl, make_trajectory
        make_trajectory(results_dir, args.name)
        make_diagnostics(results_dir, args.name)
        make_reward_vs_kl(results_dir, cfg)


if __name__ == "__main__":
    main()
