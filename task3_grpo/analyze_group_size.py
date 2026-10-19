from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json


CACHE_GROUP_SIZE = 8
REWARD_KEY = "reward"
COMPLETION_TOKENS_KEY = "completion_tokens"
REQUIRED_KEYS = ("source_index", "prompt_id", "generation_index",
                 COMPLETION_TOKENS_KEY, REWARD_KEY, "clipped_at_max")

DIFFICULTY_RULE = ("per-prompt mean cached reward over its 8 completions, split into terciles "
                   "(hard/medium/easy) by the 33.3/66.7 percentiles across all cached prompts")


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < CACHE_GROUP_SIZE}
    if bad:
        raise ValueError(f"Expected at least K={CACHE_GROUP_SIZE} cached completions per prompt; short groups: {bad}")
    sample = next(iter(by_prompt.values()))[0]
    missing = [k for k in REQUIRED_KEYS if k not in sample]
    if missing:
        raise KeyError(f"K-cache row is missing required fields {missing}; available: {sorted(sample.keys())}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def _rewards_for_prompt(rows) -> list[float]:
    return [float(r[REWARD_KEY]) for r in rows[:CACHE_GROUP_SIZE]]


def _tokens_for_prompt(rows) -> list[float]:
    return [float(r[COMPLETION_TOKENS_KEY]) for r in rows[:CACHE_GROUP_SIZE]]


def _relative_advantages(rewards: torch.Tensor, eps: float) -> torch.Tensor:
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    return (rewards - mean) / (std + eps)


def regroup_equal_generation_budget(by_prompt, k: int):
    if CACHE_GROUP_SIZE % k != 0:
        raise ValueError(
            f"k={k} must divide the cache group size {CACHE_GROUP_SIZE} for an equal-generation "
            "partition (supported k: 2, 4, 8)."
        )
    groups = []
    for pid, rows in by_prompt.items():
        rewards = _rewards_for_prompt(rows)
        for start in range(0, CACHE_GROUP_SIZE, k):
            groups.append({"prompt_id": pid, "rewards": rewards[start : start + k]})
    return groups


def _analyze_groups(groups, tol: float, eps: float) -> dict:
    stds, informative, all_adv = [], [], []
    for g in groups:
        r = torch.tensor(g["rewards"], dtype=torch.float32)
        s = float(r.std(unbiased=False))
        stds.append(s)
        informative.append(s > tol)
        all_adv.extend(_relative_advantages(r, eps).tolist())
    n = len(groups)
    return {
        "n_groups": n,
        "total_generations": int(sum(len(g["rewards"]) for g in groups)),
        "informative_group_fraction": float(np.mean(informative)) if n else 0.0,
        "mean_within_group_reward_std": float(np.mean(stds)) if n else 0.0,
        "group_relative_signal_variance": float(np.var(all_adv)) if all_adv else 0.0,
    }


def _difficulty_bins(by_prompt):
    means = {pid: float(np.mean(_rewards_for_prompt(rows))) for pid, rows in by_prompt.items()}
    vals = np.array(list(means.values()), dtype=float)
    q_low, q_high = np.percentile(vals, [100.0 / 3.0, 200.0 / 3.0])
    bin_of = {}
    for pid, m in means.items():
        bin_of[pid] = "hard" if m <= q_low else ("medium" if m <= q_high else "easy")
    edges = {"q_low": float(q_low), "q_high": float(q_high)}
    counts = {b: sum(1 for v in bin_of.values() if v == b) for b in ("hard", "medium", "easy")}
    return bin_of, edges, counts


def run_group_size_study(config_path: str):
    cfg = load_yaml(config_path)
    tol = float(cfg.get("group_std_tolerance", cfg.get("advantage_eps", 1e-6)))
    eps = float(cfg.get("advantage_eps", 1e-6))
    ks = [int(k) for k in cfg["group_sizes"]]

    by_prompt = load_k8_cache(cfg["group_cache"])
    bin_of, edges, bin_counts = _difficulty_bins(by_prompt)

    overall, by_difficulty = {}, {}
    for k in ks:
        groups = regroup_equal_generation_budget(by_prompt, k)
        overall[str(k)] = _analyze_groups(groups, tol, eps)
        per_bin = {}
        for b in ("hard", "medium", "easy"):
            sub = [g for g in groups if bin_of[g["prompt_id"]] == b]
            per_bin[b] = _analyze_groups(sub, tol, eps)
        by_difficulty[str(k)] = per_bin

    result = {
        "config": config_path,
        "n_prompts": len(by_prompt),
        "cache_group_size": CACHE_GROUP_SIZE,
        "reward_field": REWARD_KEY,
        "advantage_source": "local reconstruction (r-mu)/(sigma+eps), additive eps, NOT grpo.py",
        "advantage_eps": eps,
        "group_std_tolerance": tol,
        "group_sizes": ks,
        "difficulty_rule": DIFFICULTY_RULE,
        "difficulty_edges": edges,
        "difficulty_bin_counts": bin_counts,
        "overall": overall,
        "by_difficulty": by_difficulty,
        "equal_generation_note": (
            "every k uses all 8 cached completions per prompt; total_generations is identical "
            "across k (= n_prompts * 8), only the group size and group count change."
        ),
    }

    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(results_dir / "group_size_study.json", result)
    _write_tables(results_dir, result)
    return result


def _write_tables(results_dir, result):
    with (results_dir / "task3_group_size.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k", "n_groups", "total_generations", "informative_group_fraction",
                    "mean_within_group_reward_std", "group_relative_signal_variance"])
        for k in result["group_sizes"]:
            m = result["overall"][str(k)]
            w.writerow([k, m["n_groups"], m["total_generations"],
                        f"{m['informative_group_fraction']:.6f}",
                        f"{m['mean_within_group_reward_std']:.6f}",
                        f"{m['group_relative_signal_variance']:.6f}"])
    with (results_dir / "task3_group_size_by_difficulty.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k", "difficulty", "n_groups", "informative_group_fraction",
                    "mean_within_group_reward_std", "group_relative_signal_variance"])
        for k in result["group_sizes"]:
            for b in ("hard", "medium", "easy"):
                m = result["by_difficulty"][str(k)][b]
                w.writerow([k, b, m["n_groups"],
                            f"{m['informative_group_fraction']:.6f}",
                            f"{m['mean_within_group_reward_std']:.6f}",
                            f"{m['group_relative_signal_variance']:.6f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    result = run_group_size_study(args.config)
    print(f"cached prompts: {result['n_prompts']}  reward_field: {result['reward_field']}")
    print(f"difficulty bin counts: {result['difficulty_bin_counts']}  edges: {result['difficulty_edges']}")
    print("K   n_groups  total_gens  informative_frac  within_std  signal_var")
    for k in result["group_sizes"]:
        m = result["overall"][str(k)]
        print(f"{k:<3} {m['n_groups']:<9} {m['total_generations']:<11} "
              f"{m['informative_group_fraction']:<17.4f} {m['mean_within_group_reward_std']:<11.4f} "
              f"{m['group_relative_signal_variance']:.4f}")
    print("wrote group_size_study.json, task3_group_size.csv, task3_group_size_by_difficulty.csv")


if __name__ == "__main__":
    main()
