from __future__ import annotations

import argparse

from common.data import load_yaml
from task1_dpo.train import run_training


def beta_run_name(beta: float) -> str:
    return f"beta_{beta:.2f}"


def run_beta_forks(config_path: str, resume: bool = False):
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
