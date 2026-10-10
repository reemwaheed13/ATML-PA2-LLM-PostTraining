from __future__ import annotations

import argparse
import json

import pandas as pd

from common.data import load_yaml, repo_path
from common.generation import batch_generate
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import load_policy, load_tokenizer

# The four FIXED Task 4 policies, in fixed order. No beta-sweep / clip-KL fork / normalization
# fork is reachable: sft has no adapter, and the other three resolve ONLY to the standard
# Step-1 adapters in configs/feedback.yaml (task1/2/3 .../standard). Nothing else is accepted.
POLICY_ORDER = ("sft", "dpo", "ppo", "grpo")


def policy_adapter(cfg, policy_name: str):
    if policy_name == "sft":
        return None  # untouched Qwen2.5-1.5B-Instruct, NO adapter
    if policy_name not in ("dpo", "ppo", "grpo"):
        raise KeyError(
            f"Unknown Task 4 policy {policy_name!r}; the four fixed policies are {POLICY_ORDER} "
            "(no beta/clip/KL/normalization forks)."
        )
    adapter = cfg["policies"][policy_name]
    # Guard against a fork path sneaking into the config: the three standard adapters must end in
    # '/standard'. A beta/clip/kl/norm fork directory would not, so this fails loudly.
    if not str(adapter).rstrip("/").endswith("standard"):
        raise ValueError(
            f"Task 4 policy {policy_name!r} must be the Step-1 standard adapter, got {adapter!r}. "
            "Do not substitute a beta-sweep, clipping/KL fork, or normalization fork."
        )
    return adapter


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def _resolve_decoding(cfg):
    # Shared Task 2/3/4 deterministic-greedy convention; NO default. Fail loudly if unset so the
    # three tasks cannot silently diverge on decoding (mirrors task3_grpo/evaluate._resolve_decoding).
    if "eval_do_sample" not in cfg:
        raise KeyError(
            "configs/feedback.yaml must set 'eval_do_sample' (bool) for Task 4 generation. "
            "There is no default: Tasks 2/3/4 share this greedy decoding convention."
        )
    return {
        "do_sample": bool(cfg["eval_do_sample"]),
        "temperature": 0.0,  # ignored when do_sample is False; recorded for provenance
        "top_p": 1.0,
        "max_new_tokens": int(cfg["safety_max_new_tokens"]),
        "max_prompt_length": int(cfg.get("safety_max_prompt_length", cfg.get("max_prompt_length", 256))),
    }


def _done_ids(path):
    """xstest_ids already written. Crash-tolerant: skips a truncated final line left by a
    disconnect mid-write (that prompt is simply regenerated)."""
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["xstest_id"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue  # partial/garbled trailing line -> regenerate that prompt
    return done


def generate_for_policy(cfg, policy_name, dec, out_path, batch_size=8, resume=True):
    """One deterministic response per XSTest prompt for one policy. Resumable: appends one JSONL
    record per prompt and skips any xstest_id already present, so a disconnect mid-policy only
    recomputes the current batch (set batch_size=1 for strict per-prompt durability). Prompts are
    processed in fixed CSV order, so the file stays in that order across restarts."""
    if not resume and out_path.exists():
        out_path.unlink()
    done = _done_ids(out_path)

    adapter = policy_adapter(cfg, policy_name)
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)

    df = load_xstest(cfg)
    pending = [row for _, row in df.iterrows() if int(row["xstest_id"]) not in done]
    print(f"[{policy_name}] adapter={adapter}  total={len(df)}  done={len(done)}  pending={len(pending)}")

    timer = wall_timer()
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        prompts = [[{"role": "user", "content": str(r["prompt"])}] for r in batch]
        gen = batch_generate(
            model, tokenizer, prompts,
            max_prompt_length=dec["max_prompt_length"],
            max_new_tokens=dec["max_new_tokens"],
            temperature=dec["temperature"], top_p=dec["top_p"], do_sample=dec["do_sample"],
        )
        for row, resp, n_tok, trunc, term in zip(
            batch, gen["responses"], gen["response_lengths"], gen["truncated"], gen["terminated_with_eos"]
        ):
            append_jsonl(out_path, {
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": resp,
                "response_tokens": int(n_tok),
                "truncated": bool(trunc),
                "terminated": bool(term),
            })
    print(f"[{policy_name}] done in {timer():.1f}s -> {out_path}")


def run_generation(config_path, policies=None, batch_size=8, resume=True):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    dec = _resolve_decoding(cfg)
    policies = list(policies) if policies else list(POLICY_ORDER)
    for p in policies:
        if p not in POLICY_ORDER:
            raise KeyError(f"Unknown policy {p!r}; choose from {POLICY_ORDER}.")

    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)

    save_json(outdir / "generation_meta.json", {
        "base_model": cfg["base_model"],
        "policies": {p: policy_adapter(cfg, p) for p in POLICY_ORDER},
        "xstest": cfg["paths"]["xstest"],
        "decoding": dec,
        "seed": int(cfg["seed"]),
        "note": "four fixed policies; one deterministic greedy response per XSTest prompt per policy",
    })

    for p in policies:
        generate_for_policy(cfg, p, dec, outdir / f"generated_{p}.jsonl", batch_size=batch_size, resume=resume)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="+", choices=list(POLICY_ORDER),
                    help="subset to run (default all four). Run 'sft' first: it needs no adapter.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--fresh", action="store_true", help="ignore/overwrite existing generations")
    args = ap.parse_args()
    run_generation(args.config, policies=args.policies, batch_size=args.batch_size, resume=not args.fresh)


if __name__ == "__main__":
    main()
