from __future__ import annotations

import argparse

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import sample_entropy
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from common.precision import token_logprobs

# Identical convention string to Task 1 (task1_dpo/evaluate.py:233) and the PPO trainer,
# so KL is comparable across tasks.
KL_CONVENTION = "sampled_per_token_mean(sum_tokens/sum_response_tokens)"


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield i, seq[i : i + n]


def _resolve_decoding(cfg):
    # Held-out eval decoding is a shared Task 2/3 convention the author sets deliberately;
    # there is NO default. Fail loudly if it is missing so the two tasks cannot silently
    # diverge on the KL/length convention used for the cross-task synthesis.
    if "eval_do_sample" not in cfg:
        raise KeyError(
            "configs/ppo.yaml must set 'eval_do_sample' (bool) for held-out evaluation. "
            "There is no default: Task 2 and Task 3 share this decoding convention."
        )
    gen_block = cfg.get("generation", {})
    return {
        "do_sample": bool(cfg["eval_do_sample"]),
        "temperature": float(gen_block.get("temperature", 0.7)),
        "top_p": float(gen_block.get("top_p", 0.9)),
        "max_new_tokens": int(cfg["eval_max_response_length"]),
        "max_prompt_length": int(cfg["max_prompt_length"]),
        "samples_per_prompt": 1,
    }


def _generation_pass(policy, tokenizer, rm_model, rm_tok, rows, cfg, dec, gen_path, resume):
    """One response per prompt -> reward, length, sampled KL, entropy, eos/truncation.

    Per-prompt KL numerator (sum over valid tokens of logp_policy - logp_ref) and token
    count are stored so the aggregate KL = sum(kl_sum)/sum(n_tokens) is resume-correct
    from the file, exactly like task1_dpo/evaluate.py.
    """
    device = next(policy.parameters()).device
    gen_bs = int(cfg.get("gen_batch_size", 8))
    reward_max_length = int(cfg["reward_max_length"])

    done = set()
    if resume and gen_path.exists():
        done = {r["idx"] for r in read_jsonl(gen_path)}

    for start, chunk in _chunks(rows, gen_bs):
        idxs = [start + j for j in range(len(chunk))]
        keep = [k for k, i in enumerate(idxs) if i not in done]
        if not keep:
            continue
        sub = [chunk[k] for k in keep]
        sub_idxs = [idxs[k] for k in keep]
        prompts = [prompt_messages(r) for r in sub]

        gen = batch_generate(
            policy, tokenizer, prompts, dec["max_prompt_length"], dec["max_new_tokens"],
            temperature=dec["temperature"], top_p=dec["top_p"], do_sample=dec["do_sample"],
        )
        responses = gen["responses"]
        rewards = score_reward_pairs(rm_model, rm_tok, prompts, responses, reward_max_length).tolist()

        with torch.no_grad():
            p_tok = token_logprobs(policy, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"], cfg)[0]
            with reference_mode(policy):
                r_tok = token_logprobs(policy, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"], cfg)[0]
        mask = gen["response_mask"].to(p_tok.device)
        per_prompt_kl = ((p_tok - r_tok) * mask).sum(-1)
        per_prompt_tok = mask.sum(-1)
        per_prompt_entropy = [float(sample_entropy(p_tok[k : k + 1], mask[k : k + 1])) for k in range(len(sub_idxs))]

        for k, i in enumerate(sub_idxs):
            append_jsonl(gen_path, {
                "idx": i,
                "source_index": sub[k].get("source_index", i),
                "prompt": prompts[k],
                "response": responses[k],
                "reward": float(rewards[k]),
                "length": int(gen["response_lengths"][k]),
                "truncated": bool(gen["truncated"][k]),
                "terminated": bool(gen["terminated_with_eos"][k]),
                "entropy": per_prompt_entropy[k],
                "kl_sum": float(per_prompt_kl[k]),
                "n_tokens": int(per_prompt_tok[k]),
            })


def _aggregate(gen_path):
    recs = read_jsonl(gen_path)
    rewards = np.array([r["reward"] for r in recs], dtype=float)
    lengths = np.array([r["length"] for r in recs], dtype=float)
    entropy = np.array([r["entropy"] for r in recs], dtype=float)
    kl_num = float(sum(r["kl_sum"] for r in recs))
    kl_den = float(sum(r["n_tokens"] for r in recs)) or 1.0
    return recs, {
        "num_generations": len(recs),
        "reward_mean": float(rewards.mean()), "reward_std": float(rewards.std()),
        "length_mean": float(lengths.mean()), "length_std": float(lengths.std()),
        "length_iqr": float(np.percentile(lengths, 75) - np.percentile(lengths, 25)),
        "entropy_mean": float(entropy.mean()),
        "truncated_fraction": float(np.mean([r["truncated"] for r in recs])),
        "eos_fraction": float(np.mean([r["terminated"] for r in recs])),
        "kl_sampled": kl_num / kl_den,
    }


def _qualitative_candidates(recs, k=5):
    by_reward = sorted(recs, key=lambda r: r["reward"], reverse=True)
    med = float(np.median([r["reward"] for r in recs])) if recs else 0.0
    above = [r for r in recs if r["reward"] >= med]
    long_low_reward = sorted([r for r in recs if r["reward"] < med], key=lambda r: r["length"], reverse=True)[:k]
    short_high_reward = sorted(above, key=lambda r: r["length"])[:k]
    return {
        "high_reward_idx": [r["idx"] for r in by_reward[:k]],
        "low_reward_idx": [r["idx"] for r in by_reward[-k:]],
        "short_but_rewarded_idx": [r["idx"] for r in short_high_reward],
        "long_but_low_reward_idx": [r["idx"] for r in long_low_reward],
    }


def run_evaluation(config_path, adapter, name="standard", max_examples=None, resume=True, fresh=False):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    dec = _resolve_decoding(cfg)

    tokenizer = load_tokenizer(cfg["base_model"])
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    rm_model, rm_tok = load_reward_model(cfg)

    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    eval_path = results_dir / f"{name}_eval.json"
    gen_path = results_dir / f"{name}_generations.jsonl"
    if fresh and gen_path.exists():
        gen_path.unlink()

    timer = wall_timer()
    _generation_pass(policy, tokenizer, rm_model, rm_tok, rows, cfg, dec, gen_path, resume and not fresh)
    recs, metrics = _aggregate(gen_path)
    wall_seconds = timer()

    summary = {
        "name": name, "adapter": adapter, "seed": int(cfg["seed"]),
        **metrics,
        "decoding": dec,
        "precision": "base_fp16+lora_fp32_peft_native+autocast_fp16+gradscaler",
        "kl_convention": KL_CONVENTION,
        "wall_seconds": wall_seconds,
        "sec_per_generation": (wall_seconds / metrics["num_generations"]) if metrics["num_generations"] else None,
        "qualitative_candidates": _qualitative_candidates(recs),
    }
    save_json(eval_path, summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--fresh", action="store_true", help="ignore/overwrite existing generations")
    args = ap.parse_args()
    summary = run_evaluation(args.config, args.adapter, args.name, args.max_examples, fresh=args.fresh)
    print(f"wrote results/task2_ppo/{args.name}_eval.json")
    for key in ["num_generations", "reward_mean", "kl_sampled", "length_mean", "length_std", "entropy_mean"]:
        print(f"  {key}: {summary[key]}")


if __name__ == "__main__":
    main()
