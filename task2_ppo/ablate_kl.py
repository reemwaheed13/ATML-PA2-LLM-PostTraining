from __future__ import annotations

import argparse

from common.data import load_yaml
from task2_ppo.continue_train import fork_name, run_fork


def run_kl_forks(config_path: str, resume: bool = False, force: bool = False):
    """Matched short continuations from the identical midpoint: fixed clip_epsilon, sweep
    kl_beta. The (eps=0.20, kl=0.10) fork is shared with the clipping study and runs once
    (run_fork skips if its summary already exists)."""
    cfg = load_yaml(config_path)
    eps = float(cfg["clip_epsilon"])
    summaries = {}
    for kl in [float(b) for b in cfg["kl_values"]]:
        name = fork_name(eps, kl)
        print(f"=== KL fork {name} (eps={eps}, kl={kl}, {cfg['fork_updates']} updates) ===")
        summaries[name] = run_fork(config_path, eps, kl, resume=resume, force=force)
    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true", help="retrain forks even if already done")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fixed clip epsilon:", cfg["clip_epsilon"])
    print("Fork update budget:", cfg["fork_updates"])
    run_kl_forks(args.config, args.resume, args.force)


if __name__ == "__main__":
    main()
