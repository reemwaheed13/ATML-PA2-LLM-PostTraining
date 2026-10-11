from __future__ import annotations

import argparse
import csv
import gc
import json

from common.data import load_yaml, read_jsonl, repo_path, prompt_messages
from common.generation import batch_generate
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import load_policy, load_tokenizer
from task5_feedback.rlvr import exact_reward, extract_designated_final
from task5_feedback.rlaif import PairwiseAIJudge
# Reuse Step-2 tools so in-domain and diagnostic share ONE gold cast and ONE judge
# protocol (both-order, position-bias-controlled).
from task5_feedback.score_perturbations import normalize_gold, judge_semantic_preference

# SFT first: it is the pairwise reference for both other policies and needs no adapter.
POLICY_ORDER = ("sft", "rlvr", "rlaif")


def policy_specs(cfg):
    return {"sft": None, "rlvr": cfg["policies"]["rlvr"], "rlaif": cfg["policies"]["rlaif"]}


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(f"unknown dataset {dataset!r}")


def _resolve_decoding(cfg):
    # Same fail-loud greedy convention as task4_safety.generate / task3 evaluate: no silent
    # default for eval_do_sample, so Tasks 2/3/4/5 cannot diverge on decoding.
    if "eval_do_sample" not in cfg:
        raise KeyError("configs/feedback.yaml must set 'eval_do_sample' (bool); no default.")
    return {
        "do_sample": bool(cfg["eval_do_sample"]),
        "temperature": 0.0,   # ignored when do_sample is False; recorded for provenance
        "top_p": 1.0,
        "max_new_tokens": int(cfg["math_max_new_tokens"]),
        "max_prompt_length": int(cfg.get("math_max_prompt_length", cfg.get("max_prompt_length", 512))),
    }


def row_id(row, idx):
    """Stable per-row id. transfer has source_index; otherwise fall back to the (order-stable)
    read_jsonl index so resume keys line up deterministically across restarts."""
    return int(row["source_index"]) if "source_index" in row else int(idx)


def problem_text(row):
    """Problem shown to the judge. Prefer an explicit question field; else the last user turn."""
    if row.get("question"):
        return str(row["question"])
    for m in reversed(prompt_messages(row)):
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", ""))
    return ""


def _done_ids(path):
    """row ids already written. Crash-tolerant: a truncated final line is skipped and that row
    is simply recomputed (same pattern as task4_safety.generate._done_ids)."""
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["id"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return done


# ---------------------------------------------------------------------------
# Generate phase (policy models; the judge is NOT loaded here -> no co-residency)
# ---------------------------------------------------------------------------

def generate_for_policy(cfg, dataset, policy, dec, out_path, batch_size=8, resume=True):
    if not resume and out_path.exists():
        out_path.unlink()
    done = _done_ids(out_path)

    adapter = policy_specs(cfg)[policy]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)

    rows = read_jsonl(dataset_path(cfg, dataset))
    pending = [(row_id(r, i), r) for i, r in enumerate(rows) if row_id(r, i) not in done]
    print(f"[gen {dataset}/{policy}] adapter={adapter}  total={len(rows)}  done={len(done)}  pending={len(pending)}")

    timer = wall_timer()
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        prompts = [prompt_messages(r) for _, r in batch]
        gen = batch_generate(
            model, tokenizer, prompts,
            max_prompt_length=dec["max_prompt_length"], max_new_tokens=dec["max_new_tokens"],
            temperature=dec["temperature"], top_p=dec["top_p"], do_sample=dec["do_sample"],
        )
        for (rid, r), resp, n_tok, trunc, term in zip(
            batch, gen["responses"], gen["response_lengths"], gen["truncated"], gen["terminated_with_eos"]
        ):
            append_jsonl(out_path, {
                "id": rid, "policy": policy, "dataset": dataset,
                "gold_final": normalize_gold(r["gold_final"]),
                "question": problem_text(r),
                "response": resp,
                "response_tokens": int(n_tok),
                "truncated": bool(trunc), "terminated": bool(term),
            })
    print(f"[gen {dataset}/{policy}] done in {timer():.1f}s -> {out_path}")

    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Score phase -- CPU metrics + (GPU) pairwise judge vs SFT
# ---------------------------------------------------------------------------

def policy_metrics(records):
    n = len(records)
    assert n > 0, "no generated records to score"
    acc = sum(exact_reward(r["response"], r["gold_final"]) for r in records) / n
    fmt = sum(extract_designated_final(r["response"]) is not None for r in records) / n
    toks = sum(int(r["response_tokens"]) for r in records) / n
    trunc = sum(bool(r["truncated"]) for r in records)
    # A truncated response with no designated final is a FORMAT miss, reported apart from accuracy.
    trunc_no_final = sum(bool(r["truncated"]) and extract_designated_final(r["response"]) is None
                         for r in records)
    return {"n": n, "exact_acc": acc, "format_compliance": fmt, "mean_response_tokens": toks,
            "truncated": trunc, "truncated_no_final": trunc_no_final}


def pairwise_vs_sft(cfg, dataset, policy, policy_recs, sft_by_id, out_jsonl, cache_path, resume=True):
    """Judge policy (A) vs SFT (B) per prompt, plus verifier ranking-agreement. Resumable per id."""
    if not resume and out_jsonl.exists():
        out_jsonl.unlink()
    done = _done_ids(out_jsonl)
    judge = PairwiseAIJudge(cfg, cache_path)

    timer = wall_timer()
    for r in policy_recs:
        rid = int(r["id"])
        if rid in done:
            continue
        sft = sft_by_id[rid]  # KeyError here would mean a lost row; see alignment assert in caller
        assert r["gold_final"] == sft["gold_final"], f"gold mismatch id={rid} between {policy} and sft"
        gold = r["gold_final"]
        pref, r_ab, r_ba, info = judge_semantic_preference(judge, r["question"], r["response"], sft["response"])
        rp, rs = int(exact_reward(r["response"], gold)), int(exact_reward(sft["response"], gold))
        verifier_rank = "A" if rp > rs else ("B" if rs > rp else "TIE")
        append_jsonl(out_jsonl, {
            "id": rid, "policy": policy, "dataset": dataset,
            "judge_pref": pref, "r_ab": r_ab, "r_ba_semantic": r_ba,
            "position_tested": bool(info["tested"]), "skipped_second_call": bool(info["skipped"]),
            "position_flip": bool(info["position_flip"]), "soft_disagreement": bool(info["soft"]),
            "reward_policy": rp, "reward_sft": rs, "verifier_rank": verifier_rank,
        })
    print(f"[judge {dataset}/{policy}] done in {timer():.1f}s -> {out_jsonl}")

    records = read_jsonl(out_jsonl)
    assert len(records) == len(policy_recs), \
        f"pairwise log for {policy} has {len(records)} rows, expected {len(policy_recs)}"
    return records


def aggregate_pairwise(records):
    n = len(records)
    wins = sum(r["judge_pref"] == "A" for r in records)
    losses = sum(r["judge_pref"] == "B" for r in records)
    ties = sum(r["judge_pref"] == "TIE" for r in records)
    assert wins + losses + ties == n, "win/loss/tie must partition the pairs"
    win_rate = (wins + 0.5 * ties) / n

    # Verifier-judge agreement: both rank policy-vs-SFT; count only where the verifier strictly
    # separates them (reward differs). A judge TIE there counts as disagreement.
    strict = [r for r in records if r["verifier_rank"] in ("A", "B")]
    agree = sum(r["judge_pref"] == r["verifier_rank"] for r in strict)
    agreement = agree / len(strict) if strict else None

    tested = [r for r in records if r["position_tested"]]
    return {
        "n": n, "win": wins, "loss": losses, "tie": ties,
        "win_rate_vs_sft": win_rate,  # win=1, tie=0.5, loss=0
        "verifier_judge_agreement": agreement,
        "verifier_strict_n": len(strict),
        "position_tested_n": len(tested),
        "position_untested_n": n - len(tested),
        "position_disagreement_count": sum(r["position_flip"] for r in tested),
        "soft_disagreement_count": sum(r["soft_disagreement"] for r in tested),
    }


def run_score(cfg, dataset, outdir, cache_path, resume=True):
    gen = {}
    for p in POLICY_ORDER:
        path = outdir / f"gen_{dataset}_{p}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"No generations for {p}: {path}. Run --phase generate first.")
        gen[p] = read_jsonl(path)

    sft_by_id = {int(r["id"]): r for r in gen["sft"]}
    result = {"dataset": dataset, "policies": {}}
    for p in POLICY_ORDER:
        result["policies"][p] = policy_metrics(gen[p])

    for p in ("rlvr", "rlaif"):
        ids_p, ids_s = {int(r["id"]) for r in gen[p]}, set(sft_by_id)
        assert ids_p == ids_s, (
            f"id mismatch {p} vs sft: {len(ids_p)} vs {len(ids_s)} "
            f"(only_{p}={sorted(ids_p - ids_s)[:5]}, only_sft={sorted(ids_s - ids_p)[:5]})"
        )
        recs = pairwise_vs_sft(cfg, dataset, p, gen[p], sft_by_id,
                               outdir / f"pairwise_{dataset}_{p}.jsonl", cache_path, resume=resume)
        result["policies"][p]["pairwise"] = aggregate_pairwise(recs)

    # Transfer "drop vs in-domain": only when the GSM result is already on disk.
    if dataset == "transfer":
        gsm_json = outdir / "task5_gsm.json"
        if gsm_json.exists():
            gsm = json.loads(gsm_json.read_text(encoding="utf-8"))
            for p in POLICY_ORDER:
                g = gsm["policies"].get(p, {}).get("exact_acc")
                if g is not None:
                    result["policies"][p]["exact_acc_drop_vs_gsm"] = g - result["policies"][p]["exact_acc"]
        else:
            result["drop_note"] = "run --dataset gsm --phase score first to populate exact_acc_drop_vs_gsm"

    save_json(outdir / f"task5_{dataset}.json", result)
    _write_csv(outdir / f"task5_{dataset}.csv", result)
    print(f"[score {dataset}] wrote task5_{dataset}.json / .csv")
    for p in POLICY_ORDER:
        m = result["policies"][p]
        pw = m.get("pairwise", {})
        print(f"  {p:5s} acc={m['exact_acc']:.3f} fmt={m['format_compliance']:.3f} "
              f"len={m['mean_response_tokens']:.0f} trunc={m['truncated']} "
              f"win_vs_sft={pw.get('win_rate_vs_sft')} vj_agree={pw.get('verifier_judge_agreement')}")
    return result


def _write_csv(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "policy", "n", "exact_acc", "format_compliance", "mean_response_tokens",
              "truncated", "truncated_no_final", "exact_acc_drop_vs_gsm",
              "win_rate_vs_sft", "win", "tie", "loss", "verifier_judge_agreement",
              "verifier_strict_n", "position_tested_n", "position_untested_n"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in POLICY_ORDER:
            m = result["policies"][p]
            pw = m.get("pairwise", {})
            w.writerow({
                "dataset": result["dataset"], "policy": p, "n": m["n"],
                "exact_acc": m["exact_acc"], "format_compliance": m["format_compliance"],
                "mean_response_tokens": m["mean_response_tokens"],
                "truncated": m["truncated"], "truncated_no_final": m["truncated_no_final"],
                "exact_acc_drop_vs_gsm": m.get("exact_acc_drop_vs_gsm", ""),
                "win_rate_vs_sft": pw.get("win_rate_vs_sft", ""),
                "win": pw.get("win", ""), "tie": pw.get("tie", ""), "loss": pw.get("loss", ""),
                "verifier_judge_agreement": pw.get("verifier_judge_agreement", ""),
                "verifier_strict_n": pw.get("verifier_strict_n", ""),
                "position_tested_n": pw.get("position_tested_n", ""),
                "position_untested_n": pw.get("position_untested_n", ""),
            })


def peek(cfg):
    for key, name in [("gsm_eval", "gsm"), ("math_transfer_eval", "transfer")]:
        path = cfg["paths"][key]
        try:
            rows = read_jsonl(path)
        except FileNotFoundError:
            print(f"=== {name} ({path}) : NOT PRESENT locally ===")
            continue
        print(f"=== {name} ({path})  n={len(rows)} ===")
        print("keys:", sorted(rows[0].keys()))
        print("row[0]:", json.dumps(rows[0], ensure_ascii=False, indent=2)[:2000])
        print("prompt_messages(row[0]):", json.dumps(prompt_messages(rows[0]), ensure_ascii=False)[:800])
        print("problem_text(row[0]):", problem_text(rows[0])[:300])
        print("gold_final:", repr(rows[0].get("gold_final")), "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--phase", choices=["generate", "score", "all"], default="all")
    ap.add_argument("--policies", nargs="+", choices=list(POLICY_ORDER),
                    help="subset to generate (default all three; score always needs sft present)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--peek", action="store_true", help="dump one record per eval set and exit")
    ap.add_argument("--fresh", action="store_true", help="ignore/overwrite existing generations/judgements")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.peek:
        peek(cfg)
        return

    set_seed(int(cfg["seed"]))
    dec = _resolve_decoding(cfg)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)
    cache_path = "cached/task5_math_judge_cache.json"

    if args.phase in ("generate", "all"):
        save_json(outdir / f"eval_meta_{args.dataset}.json", {
            "base_model": cfg["base_model"], "dataset": args.dataset,
            "policies": policy_specs(cfg), "decoding": dec, "seed": int(cfg["seed"]),
            "ai_judge_model": cfg["ai_judge_model"],
            "note": "greedy; one response/prompt/policy; pairwise vs SFT uses the Step-2 both-order judge",
        })
        for p in (args.policies or POLICY_ORDER):
            generate_for_policy(cfg, args.dataset, p, dec, outdir / f"gen_{args.dataset}_{p}.jsonl",
                                batch_size=args.batch_size, resume=not args.fresh)

    if args.phase in ("score", "all"):
        run_score(cfg, args.dataset, outdir, cache_path, resume=not args.fresh)


if __name__ == "__main__":
    main()
