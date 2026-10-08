from __future__ import annotations

import argparse

from common.data import load_yaml
from task1_dpo.train import run_training


def beta_run_name(beta: float) -> str:
    return f"beta_{beta:.2f}"


def run_beta_forks(config_path: str, resume: bool = False):
    """Train the three matched beta forks from the same fresh init and seed, each on
    the first `short_ablation_examples` CLEAN examples (drawn after the filter). Every
    other setting is identical across forks, so beta is the only variable.

    Equivalent to running task1_dpo.train once per beta with --beta/--max-examples;
    kept as one command for convenience. For crash-safety across a dying Colab
    session, prefer the explicit per-fork commands (see README) so each is resumable
    on its own.
    """
    cfg = load_yaml(config_path)
    n = int(cfg["short_ablation_examples"])
    summaries = {}
    for beta in cfg["betas"]:
        name = beta_run_name(float(beta))
        print(f"=== DPO beta fork {name} (beta={beta}, max_examples={n}) ===")
        summaries[name] = run_training(
            config_path, run_name=name, beta=float(beta), max_examples=n, resume=resume,
        )
    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run_beta_forks(args.config, args.resume)


if __name__ == "__main__":
    main()
