"""Task 3 (GRPO) report figures. Pure read; no model, no new experiments.

Four figures to report/figures/ (300 dpi, axis labels with units, legend, no title --
the report caption carries the title):
  task3_standard_trajectory.png   2x2: reward, KL, entropy, clip fraction vs update
  task3_standard_diagnostics.png  2x2: policy term, grad norm, response length, and the
                                  group-informativeness panel (within-group reward std +
                                  uninformative-group fraction) vs update
  task3_group_size.png            1x2: informative-group fraction and group-relative-signal
                                  variance vs K, overall and per difficulty bin
  task3_normalization.png         1x2: held-out response length for the two forks (measured)
                                  and the length-conditioned gradient ratios (analytic illustration)

Run:  python -m scripts.plot_task3 --config configs/grpo.yaml --name standard
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.data import load_yaml, repo_path
from common.logging_utils import load_json
from scripts.aggregate_task3 import per_update
from task3_grpo.continue_train import fork_name

DPI = 300
_BIN_COLORS = {"overall": "tab:gray", "hard": "tab:red", "medium": "tab:orange", "easy": "tab:green"}


def _figdir():
    d = repo_path("report/figures")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _trajectory_rows(results_dir, name):
    train_path = results_dir / f"{name}_train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"missing {train_path}; run task3_grpo.continue_train first")
    rows = per_update(train_path)
    x = [r["update"] for r in rows]
    skipped = [r["update"] for r in rows if r["any_step_skipped"]]
    return rows, x, skipped


def _mark_skips(ax, skipped):
    for s in skipped:
        ax.axvline(s, color="red", linestyle=":", linewidth=0.8, alpha=0.6, label="_skip")


def make_trajectory(results_dir, name):
    rows, x, skipped = _trajectory_rows(results_dir, name)
    panels = [
        ("reward_mean", "mean reward (RM score)", "tab:blue"),
        ("kl_sampled", "KL(π∥π_ref) (nats/token)", "tab:red"),
        ("entropy", "entropy (nats/token)", "tab:green"),
        ("clip_fraction", "clip fraction (of response tokens)", "tab:purple"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (key, label, color) in zip(axes.flat, panels):
        ax.plot(x, [r[key] for r in rows], color=color, marker=".", label=label)
        _mark_skips(ax, skipped)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("update")
    fig.tight_layout()
    out = _figdir() / "task3_standard_trajectory.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def make_diagnostics(results_dir, name):
    rows, x, skipped = _trajectory_rows(results_dir, name)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    (ax_pt, ax_gn), (ax_len, ax_info) = axes

    ax_pt.plot(x, [r["policy_term"] for r in rows], color="tab:blue", marker=".", label="policy term")
    ax_pt.set_ylabel("policy term (clipped obj.)")

    ax_gn.plot(x, [r["grad_norm"] for r in rows], color="tab:orange", marker=".", label="grad norm")
    ax_gn.set_ylabel("gradient L2 norm")

    ax_len.plot(x, [r["response_length_mean"] for r in rows], color="tab:green", marker=".", label="response length")
    ax_len.set_ylabel("response length (tokens)")
    ax_len.set_xlabel("update")

    # Group-informativeness panel: within-group reward std (left) + uninformative fraction (right).
    ax_info.plot(x, [r["within_group_reward_std"] for r in rows], color="tab:blue", marker=".",
                 label="within-group reward std")
    ax_info.set_ylabel("within-group reward std")
    ax_info.set_xlabel("update")
    ax_r = ax_info.twinx()
    ax_r.plot(x, [r["uninformative_group_fraction"] for r in rows], color="tab:red", marker="x",
              linestyle="--", label="uninformative-group fraction")
    ax_r.set_ylabel("uninformative-group fraction")
    lines = ax_info.get_lines() + ax_r.get_lines()
    ax_info.legend(lines, [ln.get_label() for ln in lines], loc="best", fontsize=8)

    for ax in (ax_pt, ax_gn, ax_len, ax_info):
        _mark_skips(ax, skipped)
        ax.grid(True, alpha=0.3)
    for ax in (ax_pt, ax_gn):
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = _figdir() / "task3_standard_diagnostics.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def make_group_size(results_dir):
    p = results_dir / "group_size_study.json"
    if not p.exists():
        print(f"skip group-size figure: missing {p}")
        return
    study = load_json(p)
    ks = study["group_sizes"]
    fig, (ax_f, ax_v) = plt.subplots(1, 2, figsize=(11, 4.5))

    def series(metric, bin_name):
        if bin_name == "overall":
            return [study["overall"][str(k)][metric] for k in ks]
        return [study["by_difficulty"][str(k)][bin_name][metric] for k in ks]

    for b in ("overall", "hard", "medium", "easy"):
        ax_f.plot(ks, series("informative_group_fraction", b), marker="o", color=_BIN_COLORS[b], label=b)
        ax_v.plot(ks, series("group_relative_signal_variance", b), marker="o", color=_BIN_COLORS[b], label=b)
    ax_f.set_xlabel("group size K")
    ax_f.set_ylabel("informative-group fraction")
    ax_v.set_xlabel("group size K")
    ax_v.set_ylabel("group-relative signal variance")
    for ax in (ax_f, ax_v):
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = _figdir() / "task3_group_size.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def make_normalization(results_dir):
    fig, (ax_len, ax_grad) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: MEASURED held-out response length for the two forks (from evaluate.py).
    labels, means, stds = [], [], []
    for loss_type in ("grpo", "dr_grpo"):
        p = results_dir / f"{fork_name(loss_type)}_eval.json"
        if p.exists():
            d = load_json(p)
            labels.append(loss_type)
            means.append(d.get("length_mean", np.nan))
            stds.append(d.get("length_std", 0.0))
    if labels:
        ax_len.bar(labels, means, yerr=stds, color=["tab:blue", "tab:orange"], capsize=4)
    ax_len.set_ylabel("held-out response length (tokens)")
    ax_len.set_xlabel("normalization (measured on forks)")
    ax_len.grid(True, alpha=0.3, axis="y")

    # Right: ANALYTIC length-conditioned gradient ratios (fixed K-cache, not fork data).
    p = results_dir / "norm_diagnostic.json"
    if p.exists():
        diag = load_json(p)
        per = diag["per_normalization"]
        groups = ("grpo", "dr_grpo")
        per_seq = [per[g]["per_sequence_grad_long_over_short"] for g in groups]
        per_tok = [per[g]["per_token_grad_long_over_short"] for g in groups]
        xpos = np.arange(len(groups))
        ax_grad.bar(xpos - 0.2, per_seq, width=0.4, color="tab:purple", label="per-sequence grad long/short")
        ax_grad.bar(xpos + 0.2, per_tok, width=0.4, color="tab:cyan", label="per-token grad long/short")
        ax_grad.axhline(1.0, color="gray", linestyle=":", linewidth=1)
        ax_grad.set_xticks(xpos)
        ax_grad.set_xticklabels(groups)
        ax_grad.set_ylabel("long/short tercile gradient ratio")
        ax_grad.set_xlabel("normalization (analytic illustration)")
        ax_grad.legend(loc="best", fontsize=8)
        ax_grad.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = _figdir() / "task3_normalization.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])
    make_trajectory(results_dir, args.name)
    make_diagnostics(results_dir, args.name)
    make_group_size(results_dir)
    make_normalization(results_dir)


if __name__ == "__main__":
    main()
