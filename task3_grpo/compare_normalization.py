from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from common.metrics import safe_corr
from task3_grpo.analyze_group_size import (
    CACHE_GROUP_SIZE,
    _reward_field,
    _rewards_for_prompt,
    load_k8_cache,
)
from task3_grpo.continue_train import fork_name, run_fork
from task3_grpo.grpo import group_relative_advantages

# ---------------------------------------------------------------------------
# Length-normalization study (Step 3).
#
# Two parts:
#  (A) Matched short continuations from the IDENTICAL midpoint, switching only the sequence
#      normalization: canonical GRPO (divide each response's token sum by its realized length
#      T_k) vs Dr. GRPO (divide by the constant max_completion_length). Everything else -- reward,
#      beta, epsilon, prompts, generation settings, token budget, seed -- is held fixed. Held-out
#      reward/KL/response length come from running task3_grpo.evaluate on each fork adapter; the
#      join lives in scripts.aggregate_task3.
#  (B) A CONTROLLED length-conditioned gradient diagnostic on ONE fixed set of completions (the
#      supplied K-cache), scored under BOTH normalizations. Because the completions are identical,
#      any difference in gradient allocation is attributable purely to the normalization.
#
# Gradient model (exact at ratio rho=1, which our 1-epoch updates use up to LoRA dropout):
#   L = -(1/N) sum_k (1/denom_k) sum_{t in k} min(rho*A_k, clip)  ->  at rho=1, within the clip
#   region, d L / d logpi_t  =  -(1/N) (1/denom_k) A_k  for every response token t of response k.
# So the gradient GRPO places is, for response k:
#   * per TOKEN      : |A_k| / denom_k
#   * per SEQUENCE   : (sum over its T_k tokens) = |A_k| * T_k / denom_k
# with denom_k = T_k (canonical)  or  denom_k = L_max (Dr. GRPO). Hence:
#   canonical  per-token = |A_k|/T_k (long responses down-weighted per token),
#              per-seq   = |A_k|      (length-independent);
#   dr_grpo    per-token = |A_k|/L_max (length-independent per token),
#              per-seq   = |A_k|*T_k/L_max (grows linearly with length -> favors long responses).
# ---------------------------------------------------------------------------

NORM_TYPES = ("grpo", "dr_grpo")

# Candidate response-length fields in the cache row (one row == one completion). The PPO cache
# stored an integer token COUNT under "response_tokens"; the GRPO cache may do the same, which is
# exactly the length we want. Probed in order; the first present wins and is recorded.
_LENGTH_KEYS = ("response_length", "completion_length", "length", "num_response_tokens",
                "response_tokens", "n_response_tokens", "gen_length")


def _length_field(sample_row) -> str:
    for k in _LENGTH_KEYS:
        v = sample_row.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return k
    raise KeyError(
        "Could not find a scalar response-length field in the K-cache rows. Tried "
        f"{_LENGTH_KEYS}; available keys: {sorted(sample_row.keys())}. "
        "Add the correct field name to _LENGTH_KEYS."
    )


def _lengths_for_prompt(rows, length_key) -> list[float]:
    return [float(r[length_key]) for r in rows[:CACHE_GROUP_SIZE]]


# ----------------------------- part A: forks --------------------------------
def run_normalization_forks(config_path: str, resume: bool = False, force: bool = False):
    """Matched short continuations from the identical midpoint: canonical GRPO vs Dr. GRPO.
    Each runs once (run_fork skips if its summary already exists unless --force)."""
    summaries = {}
    for loss_type in NORM_TYPES:
        name = fork_name(loss_type)
        cfg = load_yaml(config_path)
        print(f"=== normalization fork {name} (loss_type={loss_type}, {cfg['fork_updates']} updates) ===")
        summaries[name] = run_fork(config_path, loss_type, resume=resume, force=force)
    return summaries


# ----------------------- part B: controlled diagnostic ----------------------
def _build_completion_table(by_prompt, reward_key, length_key, k, tol):
    """For a fixed completion set, group each prompt's cache into size-k groups, compute
    within-group advantages (the real helper), and pair each completion with (|advantage|,
    length). Returns parallel numpy arrays (abs_adv, length)."""
    abs_adv, lengths = [], []
    for _, rows in by_prompt.items():
        rewards = _rewards_for_prompt(rows, reward_key)
        lens = _lengths_for_prompt(rows, length_key)
        for start in range(0, CACHE_GROUP_SIZE, k):
            r = torch.tensor(rewards[start : start + k], dtype=torch.float32)
            adv = group_relative_advantages(r, torch.zeros(k, dtype=torch.long), eps=tol)
            for i in range(k):
                abs_adv.append(abs(float(adv[i])))
                lengths.append(lens[start + i])
    return np.array(abs_adv, dtype=float), np.array(lengths, dtype=float)


def _tercile_ratio(values, lengths):
    """mean(values | longest third) / mean(values | shortest third)."""
    if len(values) < 3:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(lengths, [100.0 / 3.0, 200.0 / 3.0])
    short = values[lengths <= lo]
    long = values[lengths > hi]
    short_mean = float(np.mean(short)) if len(short) else float("nan")
    long_mean = float(np.mean(long)) if len(long) else float("nan")
    ratio = (long_mean / short_mean) if short_mean not in (0.0, float("nan")) else float("nan")
    return short_mean, long_mean, ratio


def _grad_stats(abs_adv, lengths, denom):
    """Per-token and per-sequence gradient magnitudes under denom_k, plus their
    length-conditioning (correlation with length and long/short tercile ratio)."""
    safe_len = np.clip(lengths, 1.0, None)
    per_token = abs_adv / denom          # |A_k| / denom_k, per response token
    per_seq = abs_adv * safe_len / denom  # summed over the response's T_k tokens
    pt_short, pt_long, pt_ratio = _tercile_ratio(per_token, lengths)
    ps_short, ps_long, ps_ratio = _tercile_ratio(per_seq, lengths)
    return {
        "mean_per_token_grad": float(np.mean(per_token)),
        "mean_per_sequence_grad": float(np.mean(per_seq)),
        "corr_length_vs_per_token_grad": safe_corr(lengths, per_token),
        "corr_length_vs_per_sequence_grad": safe_corr(lengths, per_seq),
        "per_token_grad_short_tercile_mean": pt_short,
        "per_token_grad_long_tercile_mean": pt_long,
        "per_token_grad_long_over_short": pt_ratio,
        "per_sequence_grad_short_tercile_mean": ps_short,
        "per_sequence_grad_long_tercile_mean": ps_long,
        "per_sequence_grad_long_over_short": ps_ratio,
    }


def length_conditioned_diagnostic(config_path: str):
    cfg = load_yaml(config_path)
    tol = float(cfg.get("group_std_tolerance", cfg.get("advantage_eps", 1e-6)))
    k = int(cfg["num_generations"])  # the training group size (baseline K=4)
    l_max = float(cfg["max_completion_length"])
    if CACHE_GROUP_SIZE % k != 0:
        raise ValueError(f"num_generations={k} must divide cache group size {CACHE_GROUP_SIZE}")

    by_prompt = load_k8_cache(cfg["group_cache"])
    first_row = next(iter(by_prompt.values()))[0]
    reward_key = _reward_field(first_row)
    length_key = _length_field(first_row)

    abs_adv, lengths = _build_completion_table(by_prompt, reward_key, length_key, k, tol)

    per_norm = {
        "grpo": _grad_stats(abs_adv, lengths, denom=np.clip(lengths, 1.0, None)),  # canonical: denom=T_k
        "dr_grpo": _grad_stats(abs_adv, lengths, denom=l_max),                      # Dr. GRPO: denom=L_max
    }

    result = {
        "config": config_path,
        "fixed_completion_source": cfg["group_cache"],
        "n_completions": int(len(abs_adv)),
        "group_size_k": k,
        "dr_grpo_normalizer_L_max": l_max,
        "reward_field": reward_key,
        "length_field": length_key,
        "group_std_tolerance": tol,
        "length_stats": {
            "mean": float(np.mean(lengths)), "std": float(np.std(lengths)),
            "min": float(np.min(lengths)), "max": float(np.max(lengths)),
            "p33": float(np.percentile(lengths, 100.0 / 3.0)),
            "p67": float(np.percentile(lengths, 200.0 / 3.0)),
        },
        "gradient_model": (
            "first-order (ratio rho=1) GRPO gradient magnitude placed on response k: per token "
            "|A_k|/denom_k, per sequence |A_k|*T_k/denom_k; denom_k=T_k (canonical) or L_max "
            "(Dr. GRPO). Exact at the epoch-0 ratio used by our 1-epoch updates."
        ),
        "per_normalization": per_norm,
        "interpretation_keys": {
            "per_token_grad_long_over_short": "canonical << 1 (long down-weighted); dr_grpo ~ 1",
            "per_sequence_grad_long_over_short": "canonical ~ 1 (length-independent); dr_grpo > 1 (favors long)",
            "corr_length_vs_per_sequence_grad": "canonical ~ 0; dr_grpo > 0",
        },
    }

    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(results_dir / "norm_diagnostic.json", result)
    _write_diagnostic_csv(results_dir, result)
    return result


def _write_diagnostic_csv(results_dir, result):
    cols = ["mean_per_token_grad", "mean_per_sequence_grad",
            "corr_length_vs_per_token_grad", "corr_length_vs_per_sequence_grad",
            "per_token_grad_long_over_short", "per_sequence_grad_long_over_short"]
    with (results_dir / "task3_norm_diagnostic.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["normalization"] + cols)
        for norm in NORM_TYPES:
            m = result["per_normalization"][norm]
            w.writerow([norm] + [f"{m[c]:.6f}" if isinstance(m[c], float) else m[c] for c in cols])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--forks-only", action="store_true", help="train the two forks, skip the diagnostic")
    ap.add_argument("--diagnostic-only", action="store_true", help="run the controlled diagnostic only (no training)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true", help="retrain forks even if already done")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    print(f"Normalization conditions: {list(NORM_TYPES)}  (loss_type grpo vs dr_grpo)")
    print(f"Fork update budget: {cfg['fork_updates']}  from the identical supplied midpoint")

    if not args.diagnostic_only:
        run_normalization_forks(args.config, resume=args.resume, force=args.force)
    if not args.forks_only:
        result = length_conditioned_diagnostic(args.config)
        print(f"\nControlled length-conditioned diagnostic ({result['n_completions']} completions, "
              f"length field '{result['length_field']}', reward field '{result['reward_field']}'):")
        for norm in NORM_TYPES:
            m = result["per_normalization"][norm]
            print(f"  [{norm}] per-token grad long/short={m['per_token_grad_long_over_short']:.3f}  "
                  f"per-seq grad long/short={m['per_sequence_grad_long_over_short']:.3f}  "
                  f"corr(len,per-seq grad)={m['corr_length_vs_per_sequence_grad']:.3f}")
        print("wrote norm_diagnostic.json, task3_norm_diagnostic.csv")


if __name__ == "__main__":
    main()
