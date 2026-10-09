"""Task 2 standard PPO continuation trajectory figure.

Plots the Required-Evidence trajectories over the 20-update continuation: reward and
KL (primary row), then policy/value loss, entropy, clip fraction, gradient norms, and
response length. One point per update (optimization metrics averaged over the ppo
epochs; see scripts.aggregate_task2.per_update). Updates containing a GradScaler-skipped
step are marked.

Run:  python -m scripts.plot_task2 --config configs/ppo.yaml --name standard
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.data import load_yaml, repo_path
from scripts.aggregate_task2 import per_update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])
    train_path = results_dir / f"{args.name}_train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"missing {train_path}; run task2_ppo.continue_train first")

    rows = per_update(train_path)
    x = [r["update"] for r in rows]
    skipped = [r["update"] for r in rows if r["any_step_skipped"]]

    panels = [
        ("reward_mean", "mean learned reward"),
        ("kl_sampled", "sampled KL from ref"),
        ("policy_loss", "policy loss"),
        ("value_loss", "value loss"),
        ("entropy", "entropy"),
        ("clip_fraction", "clip fraction"),
        ("policy_grad_norm", "grad norm"),
        ("response_length_mean", "response length"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True)
    for ax, (key, label) in zip(axes.flat, panels):
        ax.plot(x, [r[key] for r in rows], color="tab:blue", marker=".")
        if key == "policy_grad_norm":
            ax.plot(x, [r["value_grad_norm"] for r in rows], color="tab:orange", marker=".", label="value")
            ax.plot([], [], color="tab:blue", label="policy")
            ax.legend(loc="best", fontsize=8)
        for s in skipped:
            ax.axvline(s, color="red", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("update")

    note = f"{args.name} PPO continuation ({len(rows)} updates)"
    if skipped:
        note += f"; red = GradScaler-skipped update ({len(skipped)})"
    fig.suptitle(note)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    figdir = repo_path("report/figures")
    figdir.mkdir(parents=True, exist_ok=True)
    out = figdir / f"task2_{args.name}_trajectory.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}  ({len(rows)} updates plotted)")


if __name__ == "__main__":
    main()
