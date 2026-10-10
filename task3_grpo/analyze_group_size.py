from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json

# ---------------------------------------------------------------------------
# Equal-generation group-size study (cache-based; no training, no model).
#
# The supplied cache holds K0=8 completions per prompt (confirmed schema: source_index:int,
# prompt_id:str, generation_index:int, completion:str, completion_tokens:int,
# terminated_with_eos:bool, clipped_at_max:bool, reward:float; 192 rows = 24 prompts x 8). To
# compare K in {2,4,8} at EQUAL TOTAL GENERATIONS (the manual's requirement -- compare at equal
# total generations, not equal numbers of prompts), each prompt's 8 completions are partitioned
# into 8/K groups of size K: K=8 -> 1 group/prompt, K=4 -> 2, K=2 -> 4. Every condition consumes
# the same 8 generations per prompt; only the group size (hence group count) changes. Groups never
# mix prompts, so each is a valid within-prompt comparison.
#
# Advantage source: the group-relative signal here is computed LOCALLY (_relative_advantages) as a
# reconstruction of the manual's definition A_k = (r_k - mu_r)/(sigma_r + eps), population sigma_r,
# additive eps = advantage_eps (1e-6) from configs/grpo.yaml. It is deliberately NOT imported from
# task3_grpo.grpo.group_relative_advantages, whose clamp-based normalization (grpo.py:22,
# clamp_min(eps) = max(sigma,eps) rather than sigma+eps) is the suspected planted-defect site and
# has not been independently validated against the manual formula. Keeping the analysis
# self-contained makes these diagnostics correct regardless of the training helper's status.
# ---------------------------------------------------------------------------

CACHE_GROUP_SIZE = 8          # completions per prompt in the supplied K-cache (confirmed)
REWARD_KEY = "reward"         # scalar RM reward per completion (confirmed cache field)
COMPLETION_TOKENS_KEY = "completion_tokens"  # response length in tokens (confirmed cache field)
REQUIRED_KEYS = ("source_index", "prompt_id", "generation_index",
                 COMPLETION_TOKENS_KEY, REWARD_KEY)

# Difficulty binning rule, defined ONCE (manual: "Define the binning rule once"): a prompt's
# difficulty proxy is the mean cached reward over its 8 completions; prompts are split into
# terciles by the 33.3/66.7 percentiles across all cached prompts -> hard/medium/easy.
DIFFICULTY_RULE = ("per-prompt mean cached reward over its 8 completions, split into terciles "
                   "(hard/medium/easy) by the 33.3/66.7 percentiles across all cached prompts")


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
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
    """Reconstruction of the manual GRPO advantage A_k = (r_k - mu_r)/(sigma_r + eps) for ONE
    group: population std, ADDITIVE eps. Deliberately not task3_grpo.grpo.group_relative_advantages
    (see module docstring). Degenerate group (sigma=0): numerator is 0 -> advantage 0."""
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    return (rewards - mean) / (std + eps)


def regroup_equal_generation_budget(by_prompt, k: int):
    """Partition each prompt's cached completions into groups of size k at equal total
    generations. Returns a list of groups; each group is {prompt_id, rewards: [k floats]}."""
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
    """Informative-group fraction, mean within-group reward std, and variance of the
    group-relative signal (local reconstruction) over a set of groups."""
    stds, informative, all_adv = [], [], []
    for g in groups:
        r = torch.tensor(g["rewards"], dtype=torch.float32)
        s = float(r.std(unbiased=False))
        stds.append(s)
        informative.append(s > tol)  # uninformative <=> std <= tol
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
    # Overall table: one row per K.
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
    # Per-difficulty table: one row per (K, difficulty bin).
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
