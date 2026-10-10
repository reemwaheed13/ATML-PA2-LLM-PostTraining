from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import append_jsonl, save_json, wall_timer
# Released judge: loader, fixed prompt, parser, single-example scorer. We DO NOT write a judge
# prompt -- judge_one uses the supplied JUDGE_PROMPT and parse_json clamps to the 5 LABELS.
from task4_safety.judge_responses import LABELS, judge_one, load_judge
from task4_safety.generate import POLICY_ORDER, _done_ids

AUDIT_CONFIDENCE_NOTE = "confidence is stored for auditing only; it is NEVER aggregated into a rate"


def judge_for_policy(cfg, tok, model, policy, out_path, resume=True):
    gen_path = repo_path(cfg["results_dir"]) / "task4_safety" / f"generated_{policy}.jsonl"
    if not gen_path.exists():
        raise FileNotFoundError(
            f"No generations for {policy!r}: {gen_path}. Run: python -m task4_safety.generate --policies {policy}"
        )
    if not resume and out_path.exists():
        out_path.unlink()
    done = _done_ids(out_path)

    rows = read_jsonl(gen_path)
    pending = [r for r in rows if int(r["xstest_id"]) not in done]
    print(f"[judge {policy}] total={len(rows)}  done={len(done)}  pending={len(pending)}")

    max_new = int(cfg.get("judge_max_new_tokens", 64))
    timer = wall_timer()
    for r in pending:
        verdict = judge_one(tok, model, r["prompt"], r["response"], max_new_tokens=max_new)
        label = str(verdict["label"]).upper()
        if label not in LABELS:  # parse_json already guarantees this; belt-and-braces
            label = "AMBIGUOUS"
        append_jsonl(out_path, {
            "xstest_id": int(r["xstest_id"]),
            "policy": policy,
            "benchmark_class": r["benchmark_class"],
            "type": r["type"],
            "label": label,
            "confidence": float(verdict["confidence"]),  # AUDIT ONLY -- never a rate
            "rationale_tag": verdict.get("rationale_tag", ""),
        })
    print(f"[judge {policy}] done in {timer():.1f}s -> {out_path}")


def run_judge(config_path, policies=None, resume=True):
    cfg = load_yaml(config_path)
    policies = list(policies) if policies else list(POLICY_ORDER)
    for p in policies:
        if p not in POLICY_ORDER:
            raise KeyError(f"Unknown policy {p!r}; choose from {POLICY_ORDER}.")

    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)
    tok, model = load_judge(cfg)
    print("Fixed Task 4 judge loaded:", cfg["ai_judge_model"])

    save_json(outdir / "judge_meta.json", {
        "ai_judge_model": cfg["ai_judge_model"],
        "judge_max_new_tokens": int(cfg.get("judge_max_new_tokens", 64)),
        "labels": sorted(LABELS),
        "confidence_note": AUDIT_CONFIDENCE_NOTE,
        "judge_prompt_source": "task4_safety.judge_responses.JUDGE_PROMPT (released, unmodified)",
    })

    for p in policies:
        judge_for_policy(cfg, tok, model, p, outdir / f"judged_{p}.jsonl", resume=resume)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="+", choices=list(POLICY_ORDER),
                    help="subset to judge (default all four)")
    ap.add_argument("--fresh", action="store_true", help="ignore/overwrite existing judgements")
    args = ap.parse_args()
    run_judge(args.config, policies=args.policies, resume=not args.fresh)


if __name__ == "__main__":
    main()
