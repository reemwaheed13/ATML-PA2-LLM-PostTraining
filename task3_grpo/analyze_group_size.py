from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task3_grpo.grpo import group_relative_advantages

# ---------------------------------------------------------------------------
# Equal-generation group-size study (cache-based; no training, no model).
#
# The supplied cache holds K0=8 completions per prompt. To compare K in {2,4,8} at EQUAL TOTAL
# GENERATIONS (the manual's requirement -- compare at equal total generations, not equal numbers
# of prompts), each prompt's 8 cached completions are partitioned into 8/K groups of size K:
#   K=8 -> 1 group/prompt, K=4 -> 2 groups/prompt, K=2 -> 4 groups/prompt.
# Every condition therefore consumes the same 8 generations per prompt; only the group size (and
# hence the number of groups) changes. Groups never mix prompts, so each is a valid within-prompt
# comparison. group_relative_advantages (the real training helper) is reused to standardize each
# group, so "informative" here means exactly what it means in training.
# ---------------------------------------------------------------------------

CACHE_GROUP_SIZE = 8  # completions per prompt in the supplied K-cache

# Difficulty binning rule, defined ONCE (manual: "Define the binning rule once"): a prompt's
# difficulty proxy is the mean cached reward over its 8 completions; prompts are split into
# terciles by the 33rd/67th percentiles across all cached prompts -> hard/medium/easy.
DIFFICULTY_RULE = ("per-prompt mean cached reward over its 8 completions, split into terciles "
                   "(hard/medium/easy) by the 33.3/66.7 percentiles across all cached prompts")

# Candidate scalar-reward field names in the cache row (one row == one completion). Probed in
# order; the first present wins and is recorded in the output so the report states which was used.
_REWARD_KEYS = ("reward", "rm_score", "reward_score", "reward_model_score", "score",
                "effective_terminal_reward", "terminal_reward", "raw_terminal_reward")


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < CACHE_GROUP_SIZE}
    if bad:
        raise ValueError(f"Expected at least K={CACHE_GROUP_SIZE} cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def _reward_field(sample_row) -> str:
    for k in _REWARD_KEYS:
        if k in sample_row and sample_row[k] is not None:
            return k
    raise KeyError(
        "Could not find a scalar reward field in the K-cache rows. Tried "
        f"{_REWARD_KEYS}; available keys: {sorted(sample_row.keys())}. "
        "Add the correct field name to _REWARD_KEYS."
    )


def _rewards_for_prompt(rows, reward_key) -> list[float]:
    return [float(r[reward_key]) for r in rows[:CACHE_GROUP_SIZE]]


def regroup_equal_generation_budget(by_prompt, k: int, reward_key: str):
    """Partition each prompt's cached completions into groups of size k at equal total
    generations. Returns a list of groups; each group is {prompt_id, rewards: [k floats]}."""
    if CACHE_GROUP_SIZE % k != 0:
        raise ValueError(
            f"k={k} must divide the cache group size {CACHE_GROUP_SIZE} for an equal-generation "
            "partition (supported k: 2, 4, 8)."
        )
    groups = []
    for pid, rows in by_prompt.items():
        rewards = _rewards_for_prompt(rows, reward_key)
        for start in range(0, CACHE_GROUP_SIZE, k):
            groups.append({"prompt_id": pid, "rewards": rewards[start : start + k]})
    return groups


def _analyze_groups(groups, tol: float) -> dict:
    """Informative-group fraction, mean within-group reward std, and variance of the
    group-relative signal (advantages from the real helper) over a set of groups."""
    stds, informative, all_adv = [], [], []
    for g in groups:
        r = torch.tensor(g["rewards"], dtype=torch.float32)
        s = float(r.std(unbiased=False))
        stds.append(s)
        informative.append(s > tol)  # uninformative <=> std <= tol (the clamp floor in the helper)
        adv = group_relative_advantages(r, torch.zeros(len(r), dtype=torch.long), eps=tol)
        all_adv.extend(adv.tolist())
    n = len(groups)
    return {
        "n_groups": n,
        "total_generations": int(sum(len(g["rewards"]) for g in groups)),
        "informative_group_fraction": float(np.mean(informative)) if n else 0.0,
        "mean_within_group_reward_std": float(np.mean(stds)) if n else 0.0,
        "group_relative_signal_variance": float(np.var(all_adv)) if all_adv else 0.0,
    }


def _difficulty_bins(by_prompt, reward_key):
    means = {pid: float(np.mean(_rewards_for_prompt(rows, reward_key))) for pid, rows in by_prompt.items()}
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
    ks = [int(k) for k in cfg["group_sizes"]]

    by_prompt = load_k8_cache(cfg["group_cache"])
    reward_key = _reward_field(next(iter(by_prompt.values()))[0])
    bin_of, edges, bin_counts = _difficulty_bins(by_prompt, reward_key)

    overall, by_difficulty = {}, {}
    for k in ks:
        groups = regroup_equal_generation_budget(by_prompt, k, reward_key)
        overall[str(k)] = _analyze_groups(groups, tol)
        per_bin = {}
        for b in ("hard", "medium", "easy"):
            sub = [g for g in groups if bin_of[g["prompt_id"]] == b]
            per_bin[b] = _analyze_groups(sub, tol)
        by_difficulty[str(k)] = per_bin

    result = {
        "config": config_path,
        "n_prompts": len(by_prompt),
        "cache_group_size": CACHE_GROUP_SIZE,
        "reward_field": reward_key,
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
