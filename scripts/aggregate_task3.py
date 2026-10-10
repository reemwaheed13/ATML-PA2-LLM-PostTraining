"""Aggregate Task 3 (GRPO) results into report-ready tables (and, by default, figures).
Pure read; no model, no new experiments.

Reads the per-run JSONs/JSONLs written by continue_train.py, evaluate.py,
analyze_group_size.py, and compare_normalization.py and emits to results/task3_grpo/:
  task3_standard_summary.csv   one-row standard-continuation summary (Table 1)
  task3_continuation.csv       standard 20-update per-update trajectory
  task3_group_size.csv         equal-generation group-size study (Table 2; also by difficulty)
  task3_group_size_by_difficulty.csv
  task3_normalization.csv      canonical vs Dr. GRPO: held-out metrics + length-conditioned stat
  task3_summary.json           everything, combined
Unless --tables-only is passed, it then calls scripts.plot_task3 to (re)generate the figures.

Run:  python -m scripts.aggregate_task3 --config configs/grpo.yaml
      python -m scripts.aggregate_task3 --config configs/grpo.yaml --tables-only
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json
from task3_grpo.continue_train import KL_CONVENTION, fork_name

# Stability statistic (defined once): std of the per-update policy term over a run. The policy
# term is the clipped GRPO objective without the KL penalty (grpo_policy_loss "policy_term").
STABILITY = "policy_term_std_over_updates"

CONT_COLS = ["update", "reward_mean", "within_group_reward_std", "uninformative_group_fraction",
             "kl_sampled", "kl_penalty_k3", "response_length_mean", "policy_term", "entropy",
             "clip_fraction", "ratio_mean", "grad_norm", "any_step_skipped"]

# Table 1: one-row standard-continuation summary.
STD_SUMMARY_COLS = ["updates", "num_generations", "wall_seconds", "peak_vram_gb",
                    "reward_first", "reward_last", "kl_first", "kl_last",
                    "entropy_first", "entropy_last", "length_first", "length_last",
                    "within_group_std_mean", "uninformative_group_fraction_mean",
                    "mean_clip_fraction", "n_updates_step_skipped"]

# Table 2: group-size study (overall, one row per K).
GROUP_COLS = ["k", "n_groups", "total_generations", "informative_group_fraction",
              "mean_within_group_reward_std", "group_relative_signal_variance"]
GROUP_DIFF_COLS = ["k", "difficulty", "n_groups", "informative_group_fraction",
                   "mean_within_group_reward_std", "group_relative_signal_variance"]

# Table 3: normalization study. held-out metrics from the forks + the length-conditioned statistic
# from the analytic diagnostic (clearly flagged as an illustration, not fork measurement).
NORM_COLS = ["loss_type", "fork", STABILITY,
             "heldout_reward_mean", "heldout_kl_sampled", "heldout_length_mean",
             "heldout_length_std", "heldout_entropy_mean",
             "diag_corr_length_vs_per_sequence_grad", "diag_per_sequence_grad_long_over_short",
             "diag_per_token_grad_long_over_short", "diag_is_analytic_illustration"]


def _write_csv(path, header, rows):
    with repo_path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(c, "") for c in header])


def per_update(train_path):
    """Collapse the per-(update,epoch) train log to one row per update: rollout metrics are
    constant within an update (take first); optimization metrics are averaged over the policy
    epochs; a skipped step anywhere in the update sets any_step_skipped."""
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
            "within_group_reward_std": recs[0].get("within_group_reward_std"),
            "uninformative_group_fraction": recs[0].get("uninformative_group_fraction"),
            "kl_sampled": recs[0]["kl_sampled"],
            "kl_penalty_k3": float(np.mean([x.get("kl_penalty_k3", 0.0) for x in recs])),
            "response_length_mean": recs[0]["response_length_mean"],
            "policy_term": float(np.mean([x["policy_term"] for x in recs])),
            "entropy": float(np.mean([x["entropy"] for x in recs])),
            "clip_fraction": float(np.mean([x["clip_fraction"] for x in recs])),
            "ratio_mean": float(np.mean([x.get("ratio_mean", 1.0) for x in recs])),
            "grad_norm": float(np.mean([x["grad_norm"] for x in recs])),
            "any_step_skipped": any(x.get("step_skipped") for x in recs),
        })
    return out


def _stability(train_path):
    if not train_path.exists():
        return None
    pu = per_update(train_path)
    if not pu:
        return None
    return float(np.std([r["policy_term"] for r in pu]))


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
        "num_generations": (summary or {}).get("num_generations"),
        "wall_seconds": (summary or {}).get("wall_seconds"),
        "peak_vram_gb": (summary or {}).get("peak_vram_gb"),
        "reward_first": first["reward_mean"], "reward_last": last["reward_mean"],
        "kl_first": first["kl_sampled"], "kl_last": last["kl_sampled"],
        "entropy_first": first["entropy"], "entropy_last": last["entropy"],
        "length_first": first["response_length_mean"], "length_last": last["response_length_mean"],
        "within_group_std_mean": float(np.mean([r["within_group_reward_std"] for r in pu])),
        "uninformative_group_fraction_mean": float(np.mean([r["uninformative_group_fraction"] for r in pu])),
        "mean_clip_fraction": float(np.mean([r["clip_fraction"] for r in pu])),
        "n_updates_step_skipped": int(sum(1 for r in pu if r["any_step_skipped"])),
    }


def group_size_tables(results_dir):
    """Read group_size_study.json (written by analyze_group_size) -> overall + by-difficulty rows."""
    p = results_dir / "group_size_study.json"
    if not p.exists():
        return [], [], None
    study = load_json(p)
    overall_rows = []
    for k in study["group_sizes"]:
        m = study["overall"][str(k)]
        overall_rows.append({"k": k, **{c: m[c] for c in GROUP_COLS if c != "k"}})
    diff_rows = []
    for k in study["group_sizes"]:
        for b in ("hard", "medium", "easy"):
            m = study["by_difficulty"][str(k)][b]
            diff_rows.append({"k": k, "difficulty": b, **{c: m[c] for c in GROUP_DIFF_COLS if c not in ("k", "difficulty")}})
    return overall_rows, diff_rows, study


def normalization_table(results_dir):
    diag_p = results_dir / "norm_diagnostic.json"
    diag = load_json(diag_p) if diag_p.exists() else {}
    per_norm = diag.get("per_normalization", {})
    is_illustration = diag.get("analytic_illustration")
    rows = []
    for loss_type in ("grpo", "dr_grpo"):
        name = fork_name(loss_type)
        row = {"loss_type": loss_type, "fork": name,
               STABILITY: _stability(results_dir / f"{name}_train.jsonl")}
        ev = _eval_metrics(results_dir, name)
        if ev:
            row.update({f"heldout_{k}": v for k, v in ev.items()})
        d = per_norm.get(loss_type, {})
        row["diag_corr_length_vs_per_sequence_grad"] = d.get("corr_length_vs_per_sequence_grad")
        row["diag_per_sequence_grad_long_over_short"] = d.get("per_sequence_grad_long_over_short")
        row["diag_per_token_grad_long_over_short"] = d.get("per_token_grad_long_over_short")
        row["diag_is_analytic_illustration"] = is_illustration
        rows.append(row)
    return rows, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--tables-only", action="store_true", help="skip figure (re)generation")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])

    train_path = results_dir / f"{args.name}_train.jsonl"
    cont = per_update(train_path) if train_path.exists() else []
    _write_csv(results_dir / "task3_continuation.csv", CONT_COLS, cont)

    summary_p = results_dir / f"{args.name}_summary.json"
    summary = load_json(summary_p) if summary_p.exists() else None
    std_row = standard_summary_row(results_dir, args.name, summary)
    _write_csv(results_dir / "task3_standard_summary.csv", STD_SUMMARY_COLS, [std_row] if std_row else [])

    g_overall, g_diff, study = group_size_tables(results_dir)
    _write_csv(results_dir / "task3_group_size.csv", GROUP_COLS, g_overall)
    _write_csv(results_dir / "task3_group_size_by_difficulty.csv", GROUP_DIFF_COLS, g_diff)

    nrows, diag = normalization_table(results_dir)
    _write_csv(results_dir / "task3_normalization.csv", NORM_COLS, nrows)

    save_json(results_dir / "task3_summary.json", {
        "standard_summary": summary,
        "standard_summary_row": std_row,
        "stability_statistic": STABILITY,
        "kl_convention": KL_CONVENTION,
        "continuation_trajectory": cont,
        "group_size_study": study,
        "normalization_table": nrows,
        "normalization_diagnostic": diag,
    })
    print(f"wrote task3_standard_summary.csv, task3_continuation.csv, task3_group_size.csv, "
          f"task3_group_size_by_difficulty.csv, task3_normalization.csv, task3_summary.json to {results_dir}")

    if not args.tables_only:
        # Lazy import so --tables-only never pulls in matplotlib.
        from scripts.plot_task3 import make_diagnostics, make_group_size, make_normalization, make_trajectory
        make_trajectory(results_dir, args.name)
        make_diagnostics(results_dir, args.name)
        make_group_size(results_dir)
        make_normalization(results_dir)


if __name__ == "__main__":
    main()
