"""Task 2 (PPO) report figures. Pure read; no model, no new experiments.

Three figures to report/figures/ (300 dpi, axis labels with units, legend, no title
-- the report caption carries the title):
  task2_standard_trajectory.png   2x2: reward, KL, entropy, clip fraction vs update
  task2_standard_diagnostics.png  2x2: policy loss, value loss, grad norm, length vs update
  task2_reward_vs_kl.png          held-out reward vs KL; 3 eps + 3 beta forks, by marker

The first two read <name>_train.jsonl; the third reads the fork <name>_eval.json files.

Run:  python -m scripts.plot_task2 --config configs/ppo.yaml --name standard
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.data import load_yaml, repo_path
from common.logging_utils import load_json
from scripts.aggregate_task2 import per_update
from task2_ppo.continue_train import fork_name

DPI = 300


def _figdir():
    d = repo_path("report/figures")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _trajectory_rows(results_dir, name):
    train_path = results_dir / f"{name}_train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"missing {train_path}; run task2_ppo.continue_train first")
    rows = per_update(train_path)
    x = [r["update"] for r in rows]
    skipped = [r["update"] for r in rows if r["any_step_skipped"]]
    return rows, x, skipped


def _mark_skips(ax, skipped):
    for s in skipped:
        ax.axvline(s, color="red", linestyle=":", linewidth=0.8, alpha=0.6,
                   label="_skip")  # underscore -> kept out of the legend


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
    out = _figdir() / "task2_standard_trajectory.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def make_diagnostics(results_dir, name):
    rows, x, skipped = _trajectory_rows(results_dir, name)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    (ax_pl, ax_vl), (ax_gn, ax_len) = axes

    ax_pl.plot(x, [r["policy_loss"] for r in rows], color="tab:blue", marker=".", label="policy loss")
    ax_pl.set_ylabel("policy loss")

    ax_vl.plot(x, [r["value_loss"] for r in rows], color="tab:orange", marker=".", label="value loss (MSE)")
    ax_vl.set_ylabel("value loss (MSE)")

    ax_gn.plot(x, [r["policy_grad_norm"] for r in rows], color="tab:blue", marker=".", label="policy")
    ax_gn.plot(x, [r["value_grad_norm"] for r in rows], color="tab:orange", marker=".", label="value")
    ax_gn.set_ylabel("gradient L2 norm")

    ax_len.plot(x, [r["response_length_mean"] for r in rows], color="tab:green", marker=".", label="response length")
    ax_len.set_ylabel("response length (tokens)")

    for ax in axes.flat:
        _mark_skips(ax, skipped)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("update")
    fig.tight_layout()
    out = _figdir() / "task2_standard_diagnostics.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def _fork_point(results_dir, name):
    p = results_dir / f"{name}_eval.json"
    if not p.exists():
        return None
    d = load_json(p)
    kl, reward = d.get("kl_sampled"), d.get("reward_mean")
    if kl is None or reward is None:
        return None
    return float(kl), float(reward)


def make_reward_vs_kl(results_dir, cfg):
    kl_std = float(cfg["kl_beta"])
    eps_std = float(cfg["clip_epsilon"])
    eps_pts = [(float(e), _fork_point(results_dir, fork_name(float(e), kl_std))) for e in cfg["clip_values"]]
    beta_pts = [(float(b), _fork_point(results_dir, fork_name(eps_std, float(b)))) for b in cfg["kl_values"]]
    eps_pts = [(e, p) for e, p in eps_pts if p is not None]
    beta_pts = [(b, p) for b, p in beta_pts if p is not None]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    if eps_pts:
        ax.scatter([p[0] for _, p in eps_pts], [p[1] for _, p in eps_pts],
                   marker="o", color="tab:blue", s=60, label=f"ε sweep (β={kl_std:g})")
        for e, (kl, r) in eps_pts:
            ax.annotate(f"ε={e:g}", (kl, r), textcoords="offset points", xytext=(6, 4), fontsize=8)
    if beta_pts:
        ax.scatter([p[0] for _, p in beta_pts], [p[1] for _, p in beta_pts],
                   marker="^", color="tab:orange", s=60, label=f"β sweep (ε={eps_std:g})")
        for b, (kl, r) in beta_pts:
            ax.annotate(f"β={b:g}", (kl, r), textcoords="offset points", xytext=(6, -10), fontsize=8)

    ax.set_xlabel("held-out KL(π∥π_ref) (nats/token)")
    ax.set_ylabel("held-out mean reward (RM score)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out = _figdir() / "task2_reward_vs_kl.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])
    make_trajectory(results_dir, args.name)
    make_diagnostics(results_dir, args.name)
    make_reward_vs_kl(results_dir, cfg)


if __name__ == "__main__":
    main()
