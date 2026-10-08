"""Task 1 standard-run trajectory figure.

Plots loss and preference accuracy against seen_examples (not opt_step). Excludes
GradScaler-discarded updates (records carrying an "event" field) and the trailing
partial accumulation window (window_examples != the full effective batch). Reference
lines at 0.693 (log 2) and 0.5. Saves to report/figures/.

Run:  python -m scripts.plot_task1 --config configs/dpo.yaml --name standard
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json

LOG2 = 0.6931471805599453


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])

    rows = read_jsonl(results_dir / f"{args.name}_train.jsonl")
    meta_p = results_dir / f"{args.name}_meta.json"
    full_window = int(load_json(meta_p).get("effective_batch")) if meta_p.exists() else None

    # Clean optimizer steps only: drop event rows (e.g. nonfinite_grad) and the
    # trailing partial window (window_examples below the full effective batch).
    clean = [r for r in rows if "event" not in r]
    if full_window is None:
        full_window = max((r.get("window_examples", 0) for r in clean), default=0)
    clean = [r for r in clean if r.get("window_examples") == full_window]

    x = [r["seen_examples"] for r in clean]
    loss = [r["loss"] for r in clean]
    pref = [r["preference_accuracy"] for r in clean]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
    ax1.plot(x, loss, color="tab:blue")
    ax1.axhline(LOG2, color="gray", linestyle="--", linewidth=1, label="log 2 = 0.693")
    ax1.set_ylabel("held-in DPO loss")
    ax1.legend(loc="best")
    ax1.set_title(f"Task 1 DPO training trajectory ({args.name}, excl. {len(rows) - len(clean)} non-step rows)")

    ax2.plot(x, pref, color="tab:green")
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance = 0.5")
    ax2.set_ylabel("preference accuracy")
    ax2.set_xlabel("seen_examples")
    ax2.legend(loc="best")

    fig.tight_layout()
    figdir = repo_path("report/figures")
    figdir.mkdir(parents=True, exist_ok=True)
    out = figdir / f"task1_{args.name}_trajectory.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}  ({len(clean)} plotted points, full_window={full_window})")


if __name__ == "__main__":
    main()
