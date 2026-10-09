from __future__ import annotations

import argparse

import numpy as np
import torch

from common.data import (
    detect_stratum_key,
    load_yaml,
    prompt_messages,
    read_jsonl,
    repo_path,
    write_jsonl,
)
from common.filtering import load_filtered_rows
from common.generation import batch_generate
from common.logging_utils import save_json, set_seed
from common.metrics import parse_word_limit, word_count
from common.models import load_policy, load_tokenizer, reference_mode
from common.precision import sequence_logprobs
from task1_dpo.train import make_collate


def stratified_preference(policy, tokenizer, rows, cfg):
    """Per-stratum held-out preference accuracy on the length-stratified eval set.

    Uses the reference-corrected margin m_theta = (pc-rc) - (pr-rr); accuracy is the
    fraction with m_theta > 0. Split by the real length_stratum field."""
    collate = make_collate(tokenizer, int(cfg["max_sequence_length"]))
    device = next(policy.parameters()).device
    eval_bs = int(cfg.get("eval_batch_size", cfg["batch_size"]))
    skey = detect_stratum_key(rows)

    recs = []
    for start in range(0, len(rows), eval_bs):
        chunk = rows[start : start + eval_bs]
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
        for j, row in enumerate(chunk):
            recs.append({
                "idx": start + j,
                # Stable data index so standard/balanced pairs join on identity, not
                # row position, and qualitative picks trace back to the source row.
                "source_index": row.get("source_index", start + j),
                "stratum": str(row[skey]) if skey else None,
                "m_theta": float(m[j]),
            })

    by = {}
    for r in recs:
        by.setdefault(r["stratum"], []).append(r["m_theta"])
    # Accuracy is strictly m_theta > 0: exact-zero margins are failures, not dropped.
    # n_zero_margin surfaces any tie per stratum (and overall) instead of hiding it.
    per_stratum = {
        s: {
            "n": len(v),
            "preference_accuracy": float(np.mean([x > 0 for x in v])),
            "mean_m_theta": float(np.mean(v)),
            "n_zero_margin": int(sum(1 for x in v if x == 0.0)),
        }
        for s, v in sorted(by.items(), key=lambda kv: str(kv[0]))
    }
    overall = {
        "n": len(recs),
        "preference_accuracy": float(np.mean([r["m_theta"] > 0 for r in recs])),
        "mean_m_theta": float(np.mean([r["m_theta"] for r in recs])),
        "n_zero_margin": int(sum(1 for r in recs if r["m_theta"] == 0.0)),
    }
    return {"stratum_key": skey, "overall": overall, "per_stratum": per_stratum}, recs


def word_limit_eval(policy, tokenizer, cfg):
    """Generated length + explicit word-limit compliance on the tracked word-limit prompts."""
    rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
    gen_block = cfg.get("generation", {})
    max_new = int(cfg["max_generation_tokens"])
    max_prompt = int(cfg.get("max_prompt_length", int(cfg["max_sequence_length"]) - max_new))
    prompts = [prompt_messages(r) for r in rows]
    gen = batch_generate(
        policy, tokenizer, prompts, max_prompt, max_new,
        temperature=float(gen_block.get("temperature", 0.7)),
        top_p=float(gen_block.get("top_p", 0.9)),
        do_sample=bool(gen_block.get("do_sample", True)),
    )
    recs = []
    for i, row in enumerate(rows):
        text = gen["responses"][i]
        user_text = row["messages"][-1]["content"]
        limit = parse_word_limit(user_text)
        wc = word_count(text)
        recs.append({
            "prompt_id": row.get("prompt_id", i),
            "limit": limit,
            "word_count": wc,
            "length_tokens": int(gen["response_lengths"][i]),
            "compliant": (wc <= limit) if limit is not None else None,
            "response": text,
        })
    comp = [r["compliant"] for r in recs if r["compliant"] is not None]
    lengths = [r["length_tokens"] for r in recs]
    agg = {
        "n": len(recs),
        "n_with_limit": len(comp),
        "compliance_rate": float(np.mean(comp)) if comp else None,
        "length_mean": float(np.mean(lengths)),
        "length_std": float(np.std(lengths)),
    }
    return agg, recs


def run_analysis(config_path, adapter, name):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    results_dir = repo_path(cfg["results_dir"])

    # Same filter as training/eval, recorded, so stratified numbers are comparable.
    rows, _ = load_filtered_rows(
        cfg["paths"]["dpo_length_eval"], tokenizer, int(cfg["max_sequence_length"]),
        results_dir=cfg["results_dir"], run_name=name,
    )
    strat_metrics, strat_recs = stratified_preference(policy, tokenizer, rows, cfg)
    wl_agg, wl_recs = word_limit_eval(policy, tokenizer, cfg)

    save_json(results_dir / f"{name}_length_stratum.json", strat_metrics)
    write_jsonl(results_dir / f"{name}_length_pairs.jsonl", strat_recs)
    save_json(results_dir / f"{name}_wordlimit.json", wl_agg)
    write_jsonl(results_dir / f"{name}_wordlimit_gen.jsonl", wl_recs)

    print(f"[{name}] stratified preference accuracy:")
    for s, v in strat_metrics["per_stratum"].items():
        print(f"  {s}: acc={v['preference_accuracy']:.4f}  n={v['n']}")
    print(f"  overall: acc={strat_metrics['overall']['preference_accuracy']:.4f}")
    print(f"[{name}] word-limit compliance={wl_agg['compliance_rate']}  "
          f"gen_len mean={wl_agg['length_mean']:.1f} std={wl_agg['length_std']:.1f}")
    return {"stratified": strat_metrics, "word_limit": wl_agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    run_analysis(args.config, args.adapter, args.name)


if __name__ == "__main__":
    main()
