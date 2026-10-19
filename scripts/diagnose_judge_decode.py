
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import torch

from common.data import load_yaml, repo_path
from common.logging_utils import save_json
from task5_feedback.rlaif import PairwiseAIJudge, PAIRWISE_RUBRIC
from task5_feedback.score_perturbations import load_diagnostic_groups, PAIRINGS

VERDICT_RE = re.compile(r"\b(A|B|TIE)\b")


def raw_generate(judge, problem, a, b, max_new_tokens):
    text = PAIRWISE_RUBRIC.format(problem=problem, a=a, b=b)
    ids = judge.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], return_tensors="pt", add_generation_prompt=True,
    ).to(next(judge.model.parameters()).device)
    with torch.no_grad():
        out = judge.model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=judge.tokenizer.eos_token_id, eos_token_id=judge.tokenizer.eos_token_id,
        )
    return judge.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def classify(decoded):
    s = decoded.strip()
    m = VERDICT_RE.search(s.upper())
    return {
        "has_explicit_verdict": m is not None,
        "parsed_token": m.group(1) if m else None,
        "is_empty": s == "",
        "compare_would_return": (m.group(1) if m else "TIE"),
    }


def select_pairs(groups, n_pairs):
    pids = sorted(groups, key=lambda x: int(x))
    by_pairing = {name: [] for name in PAIRINGS}
    for name, (va, vb) in PAIRINGS.items():
        for pid in pids:
            g = groups[pid]
            by_pairing[name].append({
                "pairing": name, "problem_id": pid, "variant_A": va, "variant_B": vb,
                "question": g[va]["question"], "a": g[va]["response"], "b": g[vb]["response"],
            })
    out, i = [], 0
    while len(out) < n_pairs and any(i < len(v) for v in by_pairing.values()):
        for name in PAIRINGS:
            if i < len(by_pairing[name]) and len(out) < n_pairs:
                out.append(by_pairing[name][i])
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--n-pairs", type=int, default=20)
    ap.add_argument("--extra-max-new-tokens", type=int, default=16,
                    help="second decode cap to test the truncation hypothesis")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    groups = load_diagnostic_groups(repo_path(cfg["paths"]["task5_diagnostics"]))
    pairs = select_pairs(groups, args.n_pairs)
    judge = PairwiseAIJudge(cfg, repo_path("results/task5_feedback/_judge_decode_scratch_cache.json"))

    records = []
    for p in pairs:
        raw4 = raw_generate(judge, p["question"], p["a"], p["b"], 4)
        raw_x = raw_generate(judge, p["question"], p["a"], p["b"], args.extra_max_new_tokens)
        records.append({
            **{k: p[k] for k in ("pairing", "problem_id", "variant_A", "variant_B")},
            "raw_maxnew4": raw4, "classify_maxnew4": classify(raw4),
            f"raw_maxnew{args.extra_max_new_tokens}": raw_x, "classify_extra": classify(raw_x),
        })

    n = len(records)
    explicit = [r for r in records if r["classify_maxnew4"]["has_explicit_verdict"]]
    tok = Counter(r["classify_maxnew4"]["parsed_token"] for r in explicit)
    recovered = sum(1 for r in records
                    if not r["classify_maxnew4"]["has_explicit_verdict"]
                    and r["classify_extra"]["has_explicit_verdict"])
    summary = {
        "n_pairs": n,
        "max_new_tokens_real": 4,
        "extra_max_new_tokens": args.extra_max_new_tokens,
        "n_with_explicit_verdict_at4": len(explicit),
        "explicit_token_breakdown_at4": dict(tok),
        "n_empty_at4": sum(r["classify_maxnew4"]["is_empty"] for r in records),
        "n_nonverdict_at4": n - len(explicit),
        "n_recovered_with_extra_tokens": recovered,
        "interpretation": (
            "A high n_nonverdict_at4 means the cached ~95% TIE is largely PARSE FAILURE (compare() "
            "defaults to TIE), not genuine ties; n_recovered_with_extra_tokens is how much the 4-token "
            "cap caused it. Inspect the raw strings: a parsed 'A'/'B' from prose (not a bare verdict "
            "token) is a false positive of the \\b(A|B|TIE)\\b parser."
        ),
    }

    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "judge_decode_diagnostic.json", {"summary": summary, "records": records})

    print(json.dumps(summary, indent=2))
    print("\n=== RAW decoded strings, verbatim (real 4-token cap | extra cap) ===")
    for r in records:
        print(f"[{r['pairing']}|{r['problem_id']}] "
              f"4tok={r['raw_maxnew4']!r}  extra={r[f'raw_maxnew{args.extra_max_new_tokens}']!r}")
    print(f"\nwrote {outdir / 'judge_decode_diagnostic.json'}")


if __name__ == "__main__":
    main()
