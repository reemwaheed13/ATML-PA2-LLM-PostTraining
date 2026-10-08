from __future__ import annotations

import argparse

import numpy as np
import torch

from common.data import (
    load_yaml,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.filtering import load_filtered_rows
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from common.precision import sequence_logprobs, token_logprobs
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import make_collate


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield i, seq[i : i + n]


def _teacher_forced_pass(policy, tokenizer, rows, cfg, beta, pairs_path):
    """Pass A: held-out DPO loss + preference accuracy on the dataset's chosen/rejected.

    preference margin m_theta = (pc-rc) - (pr-rr), the reference-corrected margin
    from the handout (NOT common.metrics.preference_accuracy, which omits the
    reference and would be off-spec).
    """
    collate = make_collate(tokenizer, int(cfg["max_sequence_length"]))
    device = next(policy.parameters()).device
    eval_bs = int(cfg.get("eval_batch_size", cfg["batch_size"]))

    pc_all, pr_all, rc_all, rr_all = [], [], [], []
    pair_records = []

    for start, chunk in _chunks(rows, eval_bs):
        cb, rb = collate(chunk)
        cb = {k: v.to(device) for k, v in cb.items()}
        rb = {k: v.to(device) for k, v in rb.items()}
        with torch.no_grad():
            pc = sequence_logprobs(policy, cb, cfg)[0]
            pr = sequence_logprobs(policy, rb, cfg)[0]
            with reference_mode(policy):
                rc = sequence_logprobs(policy, cb, cfg)[0]
                rr = sequence_logprobs(policy, rb, cfg)[0]
        m = (pc - rc) - (pr - rr)
        pc_all.append(pc.cpu()); pr_all.append(pr.cpu())
        rc_all.append(rc.cpu()); rr_all.append(rr.cpu())
        for j, row in enumerate(chunk):
            yc, yr = preference_responses(row)
            pair_records.append({
                "idx": start + j,
                "source_index": row.get("source_index", start + j),
                "m_theta": float(m[j]),
                "policy_chosen_logp": float(pc[j]), "policy_rejected_logp": float(pr[j]),
                "ref_chosen_logp": float(rc[j]), "ref_rejected_logp": float(rr[j]),
                "chosen_text": yc, "rejected_text": yr,
            })

    pc_all = torch.cat(pc_all); pr_all = torch.cat(pr_all)
    rc_all = torch.cat(rc_all); rr_all = torch.cat(rr_all)
    m_all = (pc_all - rc_all) - (pr_all - rr_all)
    loss, _ = dpo_loss(pc_all, pr_all, rc_all, rr_all, beta)

    write_jsonl(pairs_path, pair_records)
    return {
        "num_pairs": int(m_all.numel()),
        "held_out_dpo_loss": float(loss),
        "preference_accuracy": float((m_all > 0).float().mean()),
        "mean_preference_margin": float(m_all.mean()),
    }, pair_records


def _generation_pass(policy, tokenizer, rm_model, rm_tok, rows, cfg, gen_path, resume):
    """Pass B: one seeded sample per prompt -> reward, length, sampled KL.

    Each record stores reward/length/eos/truncation plus the per-prompt KL
    numerator (sum of policy-ref over valid tokens) and token count, so the
    aggregate KL = sum(kl_sum)/sum(n_tokens) is resume-correct from the file.
    """
    device = next(policy.parameters()).device
    gen_block = cfg.get("generation", {})
    temperature = float(gen_block.get("temperature", 0.7))
    top_p = float(gen_block.get("top_p", 0.9))
    do_sample = bool(gen_block.get("do_sample", True))
    max_new_tokens = int(cfg["max_generation_tokens"])
    max_prompt_length = int(cfg.get("max_prompt_length", int(cfg["max_sequence_length"]) - max_new_tokens))
    gen_bs = int(cfg.get("gen_batch_size", 8))
    reward_max_length = int(cfg.get("reward_max_length", 1024))

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
        prompts = [prompt_messages_from_preference(r) for r in sub]

        gen = batch_generate(
            policy, tokenizer, prompts, max_prompt_length, max_new_tokens,
            temperature=temperature, top_p=top_p, do_sample=do_sample,
        )
        responses = gen["responses"]
        rewards = score_reward_pairs(rm_model, rm_tok, prompts, responses, reward_max_length).tolist()

        with torch.no_grad():
            p_tok = token_logprobs(policy, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"], cfg)[0]
            with reference_mode(policy):
                r_tok = token_logprobs(policy, gen["sequences"], gen["attention_mask"], gen["prompt_width"], gen["response_ids"], cfg)[0]
        mask = gen["response_mask"]
        per_prompt_kl = ((p_tok - r_tok) * mask).sum(-1)      # sum over tokens, per prompt
        per_prompt_tok = mask.sum(-1)

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
                "kl_sum": float(per_prompt_kl[k]),
                "n_tokens": int(per_prompt_tok[k]),
            })


def _aggregate_generations(gen_path):
    recs = read_jsonl(gen_path)
    rewards = np.array([r["reward"] for r in recs], dtype=float)
    lengths = np.array([r["length"] for r in recs], dtype=float)
    kl_num = float(sum(r["kl_sum"] for r in recs))
    kl_den = float(sum(r["n_tokens"] for r in recs)) or 1.0
    return recs, {
        "num_generations": len(recs),
        "reward_mean": float(rewards.mean()), "reward_std": float(rewards.std()),
        "length_mean": float(lengths.mean()), "length_std": float(lengths.std()),
        "length_iqr": float(np.percentile(lengths, 75) - np.percentile(lengths, 25)),
        "truncated_fraction": float(np.mean([r["truncated"] for r in recs])),
        "eos_fraction": float(np.mean([r["terminated"] for r in recs])),
        "kl_sampled": kl_num / kl_den,
    }


def _qualitative_candidates(gen_recs, pair_recs, k=5):
    by_reward = sorted(gen_recs, key=lambda r: r["reward"], reverse=True)
    med_reward = float(np.median([r["reward"] for r in gen_recs])) if gen_recs else 0.0
    above = [r for r in gen_recs if r["reward"] >= med_reward]
    short_high_reward = sorted(above, key=lambda r: r["length"])[:k]
    high_margin = sorted(pair_recs, key=lambda r: r["m_theta"], reverse=True)[:k]
    return {
        "high_reward_idx": [r["idx"] for r in by_reward[:k]],
        "short_but_rewarded_idx": [r["idx"] for r in short_high_reward],
        "high_pref_margin_idx": [r["idx"] for r in high_margin],
    }


def run_evaluation(config_path, adapter, name="standard", beta=None, max_examples=None, resume=True, fresh=False):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    beta = float(cfg["beta"] if beta is None else beta)

    tokenizer = load_tokenizer(cfg["base_model"])
    # Same prompt-length filter as training, recorded to <name>_filter.json, so the
    # held-out numbers are comparable across conditions. Subset after filtering.
    rows, _ = load_filtered_rows(
        cfg["paths"]["dpo_standard_eval"], tokenizer, int(cfg["max_sequence_length"]),
        results_dir=cfg["results_dir"], run_name=name,
    )
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    rm_model, rm_tok = load_reward_model(cfg)

    results_dir = repo_path(cfg["results_dir"])
    eval_path = results_dir / f"{name}_eval.json"
    gen_path = results_dir / f"{name}_generations.jsonl"
    pairs_path = results_dir / f"{name}_pairs.jsonl"
    if fresh and gen_path.exists():
        gen_path.unlink()

    timer = wall_timer()
    tf_metrics, pair_recs = _teacher_forced_pass(policy, tokenizer, rows, cfg, beta, pairs_path)
    _generation_pass(policy, tokenizer, rm_model, rm_tok, rows, cfg, gen_path, resume and not fresh)
    gen_recs, gen_metrics = _aggregate_generations(gen_path)
    wall_seconds = timer()

    summary = {
        "name": name, "adapter": adapter, "beta": beta, "seed": int(cfg["seed"]),
        **tf_metrics, **gen_metrics,
        "decoding": {
            "temperature": float(cfg.get("generation", {}).get("temperature", 0.7)),
            "top_p": float(cfg.get("generation", {}).get("top_p", 0.9)),
            "do_sample": bool(cfg.get("generation", {}).get("do_sample", True)),
            "max_new_tokens": int(cfg["max_generation_tokens"]),
            "max_prompt_length": int(cfg.get("max_prompt_length", int(cfg["max_sequence_length"]) - int(cfg["max_generation_tokens"]))),
            "samples_per_prompt": 1,
        },
        "precision": "base_fp16+lora_fp32_peft_native+autocast_fp16+gradscaler",
        "wall_seconds": wall_seconds,
        "sec_per_generation": (wall_seconds / gen_metrics["num_generations"]) if gen_metrics["num_generations"] else None,
        "qualitative_candidates": _qualitative_candidates(gen_recs, pair_recs),
    }
    save_json(eval_path, summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--fresh", action="store_true", help="ignore/overwrite existing generations")
    args = ap.parse_args()
    summary = run_evaluation(args.config, args.adapter, args.name, args.beta, args.max_examples, fresh=args.fresh)
    print(f"wrote results/task1_dpo/{args.name}_eval.json")
    for k in ["num_pairs", "held_out_dpo_loss", "preference_accuracy", "kl_sampled", "reward_mean", "length_mean", "length_std"]:
        print(f"  {k}: {summary[k]}")


if __name__ == "__main__":
    main()
