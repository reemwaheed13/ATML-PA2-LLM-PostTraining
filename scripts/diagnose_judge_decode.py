"""Is the RLAIF judge's ~95% TIE real, or parse failure? Diagnose the RAW decoded output.

`rlaif.PairwiseAIJudge.compare` (rlaif.py:91-92) returns `TIE` for BOTH the rubric's genuine TIE
label AND as the fallback when no A/B/TIE token is parsed; the cache stores only the final label, so
a high cached tie rate is unattributable. Compounding risks the run flagged: `max_new_tokens=4` may
truncate before a verdict, and the parser does `.upper()` + `\b(A|B|TIE)\b`, which also fires on a
stray standalone "A" inside prose.

This script RE-RUNS ~20 diagnostic pairs with the SAME rubric + greedy decoding, captures the RAW
decoded string per call, and BYPASSES the cache -- it calls `model.generate` directly and never calls
`compare()`, so no cached verdict is reused. It does NOT modify `rlaif.py` (fixed course utility).
Each pair is also decoded a second time at a larger cap to test the truncation hypothesis: if a
verdict appears with more room, the 4-token cap was cutting it off.

Reports: count with an explicit A/B/TIE token (and the A/B/TIE breakdown), count empty / non-verdict,
how many non-verdicts at 4 tokens recover a verdict at the larger cap, and every raw string verbatim.

Run (GPU):  python -m scripts.diagnose_judge_decode --config configs/feedback.yaml --n-pairs 20
"""

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

VERDICT_RE = re.compile(r"\b(A|B|TIE)\b")  # same token match as rlaif.compare (on .upper())


def raw_generate(judge, problem, a, b, max_new_tokens):
    """Mirror compare()'s prompt + greedy decoding exactly, but capture the RAW decoded text and never
    touch the cache. No orientation swap -- we are diagnosing the decode, not the preference."""
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
        "compare_would_return": (m.group(1) if m else "TIE"),  # rlaif.py:91-92 TIE fallback
    }


def select_pairs(groups, n_pairs):
    """Interleave the four pairings so a cap of n spans all perturbation types (not 20 of one kind)."""
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
    # Scratch cache path; we never call compare(), so the cache is never consulted or written.
    judge = PairwiseAIJudge(cfg, repo_path("results/task5_feedback/_judge_decode_scratch_cache.json"))

    records = []
    for p in pairs:
        raw4 = raw_generate(judge, p["question"], p["a"], p["b"], 4)                       # real decoding
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
