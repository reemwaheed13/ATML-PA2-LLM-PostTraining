from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import append_jsonl, save_json, wall_timer
from task5_feedback.rlvr import exact_reward, extract_designated_final
from task5_feedback.rlaif import PairwiseAIJudge


EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}

VARIANT_EXPECTED_REWARD = {
    "clean_correct": 1,
    "corrupt_reasoning_correct_final": 1,
    "persuasive_filler_correct": 1,
    "good_reasoning_wrong_final": 0,
    "gold_distractor_wrong_final": 0,
}

REFERENCE = "clean_correct"
PAIRINGS = {
    "S_reason": ("clean_correct", "corrupt_reasoning_correct_final"),
    "S_outcome": ("clean_correct", "good_reasoning_wrong_final"),
    "filler_control": ("clean_correct", "persuasive_filler_correct"),
    "gold_distractor": ("clean_correct", "gold_distractor_wrong_final"),
}


def normalize_gold(value) -> str:
    gold = str(value).strip()
    try:
        float(gold)
    except ValueError:
        raise ValueError(
            f"gold_final is not numeric: {value!r} (normalized {gold!r}). "
            "A '#### n' response string must never be passed as the gold."
        )
    return gold



def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def load_and_guard(path):
    rows = read_jsonl(path)
    assert isinstance(rows, list), f"expected a list of rows, got {type(rows).__name__}"
    assert rows, "diagnostic set is empty"
    assert isinstance(rows[0], dict), f"expected dict rows, got {type(rows[0]).__name__}"
    assert len(rows) == 100, f"expected n=100 diagnostic rows, got {len(rows)}"

    groups = load_diagnostic_groups(path)
    assert len(groups) == 20, f"expected 20 distinct problem_id, got {len(groups)}"

    variant_counts = Counter(r["variant_type"] for r in rows)
    for v in EXPECTED_VARIANTS:
        assert variant_counts[v] == 20, f"expected 20 x {v!r}, got {variant_counts[v]}"
    return rows, groups



def verifier_self_check(rows):
    ones = 0
    for row in rows:
        gold = normalize_gold(row["gold_final"])
        got = exact_reward(row["response"], gold)
        expected = int(row["expected_exact_reward"])
        if int(got) != expected:
            raise AssertionError(
                "verifier self-check FAILED "
                f"(problem_id={row['problem_id']}, variant={row['variant_type']!r}): "
                f"expected_exact_reward={expected}, exact_reward={int(got)}"
            )
        ones += expected
    assert ones == 60, f"expected 60 reward-1 rows, got {ones}"
    assert len(rows) - ones == 40, f"expected 40 reward-0 rows, got {len(rows) - ones}"

    assert extract_designated_final("#### 1,600") == "1600", "comma stripping regressed"
    print(f"[Step A] verifier self-check OK: {len(rows)}/{len(rows)} rows, "
          f"{ones} ones / {len(rows) - ones} zeros")


def discriminating_self_test(path):
    rows, _ = load_and_guard(path)
    verifier_self_check(rows)

    victim = next(r for r in rows if int(r["expected_exact_reward"]) == 1)
    vid = victim["problem_id"]
    corrupted = []
    for r in rows:
        if r is victim:
            bad = dict(r)
            bad["gold_final"] = str(int(normalize_gold(r["gold_final"])) + 7)
            corrupted.append(bad)
        else:
            corrupted.append(r)

    try:
        verifier_self_check(corrupted)
    except AssertionError as exc:
        assert f"problem_id={vid}" in str(exc), (
            f"assertion fired but did not name the corrupted problem_id={vid}: {exc}"
        )
        print(f"[discriminating] PASS: corrupting problem_id={vid} tripped the self-check:\n    {exc}")
        _gold_normalization_discriminating_test()
        return
    raise AssertionError("discriminating test FAILED: corrupting a gold_final did NOT trip the self-check")


def _gold_normalization_discriminating_test():
    try:
        normalize_gold("#### 42")
    except ValueError as exc:
        print(f"[discriminating] PASS: normalize_gold('#### 42') raised instead of scoring 0%:\n    {exc}")
        return
    raise AssertionError(
        "gold-normalization test FAILED: normalize_gold('#### 42') did not raise -- "
        "a malformed gold would silently score 0%"
    )



def rates_from_prefs(prefs):
    n = len(prefs)
    assert n > 0, "no preferences to aggregate"
    c = Counter(prefs)
    better = c.get("A", 0) / n
    tie = c.get("TIE", 0) / n
    wrong = c.get("B", 0) / n
    total = better + tie + wrong
    assert abs(total - 1.0) < 1e-9, f"better/tie/wrong must sum to 1.0, got {total} from {dict(c)}"
    return {"better": better, "tie": tie, "wrong": wrong, "n": n}



def verifier_preference(resp_a, resp_b, gold):
    ra = exact_reward(resp_a, gold)
    rb = exact_reward(resp_b, gold)
    if ra > rb:
        pref = "A"
    elif rb > ra:
        pref = "B"
    else:
        pref = "TIE"
    return pref, int(ra), int(rb)


def run_verifier_pairings(groups):
    per_pair_rows = []
    pairings = {}
    for name, (va, vb) in PAIRINGS.items():
        prefs = []
        for pid in sorted(groups, key=lambda x: int(x)):
            a, b = groups[pid][va], groups[pid][vb]
            gold = normalize_gold(a["gold_final"])
            assert gold == normalize_gold(b["gold_final"]), f"gold mismatch within problem {pid}"
            pref, ra, rb = verifier_preference(a["response"], b["response"], gold)
            prefs.append(pref)
            per_pair_rows.append({
                "pairing": name, "problem_id": pid,
                "variant_A": va, "variant_B": vb,
                "reward_A": ra, "reward_B": rb, "pref": pref,
                "style_B": b.get("style", ""), "reasoning_quality_B": b.get("reasoning_quality", ""),
            })
        assert len(prefs) == 20, f"[verifier/{name}] expected 20 pairs, got {len(prefs)}"
        pairings[name] = rates_from_prefs(prefs)

    assert pairings["S_reason"]["tie"] == 1.0 and pairings["S_reason"]["better"] == 0.0, \
        f"verifier S_reason must be all-tie, got {pairings['S_reason']}"
    assert pairings["filler_control"]["tie"] == 1.0, \
        f"verifier filler_control must be all-tie, got {pairings['filler_control']}"
    assert pairings["S_outcome"]["better"] == 1.0, \
        f"verifier S_outcome must be all-better, got {pairings['S_outcome']}"
    assert pairings["gold_distractor"]["better"] == 1.0, \
        f"verifier gold_distractor must be all-better (parser not fooled), got {pairings['gold_distractor']}"

    summary = {
        "scale": "binary exact reward in {0,1}; prefer A iff R(A) > R(B)",
        "pairings": pairings,
        "S_reason": pairings["S_reason"]["better"],
        "S_outcome": pairings["S_outcome"]["better"],
    }
    return summary, per_pair_rows



REMAP = {"A": "B", "B": "A", "TIE": "TIE"}


def _phys_swap(judge, problem, a, b):
    return int(judge._key(problem, a, b)[:8], 16) % 2 == 1


def judge_semantic_preference(judge, problem, resp_a, resp_b):
    tested = _phys_swap(judge, problem, resp_a, resp_b) == _phys_swap(judge, problem, resp_b, resp_a)
    r_ab = judge.compare(problem, resp_a, resp_b)
    if not tested:
        return r_ab, r_ab, r_ab, {"tested": False, "skipped": True,
                                  "position_flip": False, "soft": False}
    r_ba = REMAP[judge.compare(problem, resp_b, resp_a)]
    if r_ab == r_ba:
        return r_ab, r_ab, r_ba, {"tested": True, "skipped": False,
                                  "position_flip": False, "soft": False}
    flip = {r_ab, r_ba} == {"A", "B"}
    return "TIE", r_ab, r_ba, {"tested": True, "skipped": False,
                               "position_flip": flip, "soft": not flip}


def _done_pair_keys(path):
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(f"{rec['pairing']}|{rec['problem_id']}")
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def run_judge_pairings(cfg, groups, out_jsonl, cache_path, resume=True):
    if not resume and out_jsonl.exists():
        out_jsonl.unlink()
    done = _done_pair_keys(out_jsonl)
    judge = PairwiseAIJudge(cfg, cache_path)

    timer = wall_timer()
    skipped = 0
    for name, (va, vb) in PAIRINGS.items():
        for pid in sorted(groups, key=lambda x: int(x)):
            key = f"{name}|{pid}"
            if key in done:
                continue
            a, b = groups[pid][va], groups[pid][vb]
            problem = a["question"]
            assert problem == b["question"], f"question mismatch within problem {pid}"
            pref, r_ab, r_ba, info = judge_semantic_preference(judge, problem, a["response"], b["response"])
            skipped += int(info["skipped"])
            append_jsonl(out_jsonl, {
                "pairing": name, "problem_id": pid,
                "variant_A": va, "variant_B": vb,
                "r_ab": r_ab, "r_ba_semantic": r_ba, "pref": pref,
                "position_tested": bool(info["tested"]),
                "skipped_second_call": bool(info["skipped"]),
                "position_flip": bool(info["position_flip"]),
                "soft_disagreement": bool(info["soft"]),
                "style_B": b.get("style", ""), "reasoning_quality_B": b.get("reasoning_quality", ""),
            })
    elapsed = timer()
    print(f"[Step D] judge pairings done in {elapsed:.1f}s -> {out_jsonl} "
          f"(skipped {skipped} redundant second calls this run)")

    records = read_jsonl(out_jsonl)
    expected = len(PAIRINGS) * len(groups)
    assert len(records) == expected, f"judge log has {len(records)} rows, expected {expected}"

    calls_by_pairing = defaultdict(int)
    for r in records:
        calls_by_pairing[r["pairing"]] += 2 if r.get("position_tested") else 1
    save_json(out_jsonl.parent / "diagnostic_judge_meta.json", {
        "ai_judge_model": cfg["ai_judge_model"],
        "elapsed_s_last_run": elapsed,
        "n_pairs": len(records),
        "n_compare_calls_total": int(sum(calls_by_pairing.values())),
        "n_compare_calls_by_pairing": {k: int(v) for k, v in calls_by_pairing.items()},
        "note": "compare() calls = model calls on a cold cache; elapsed excludes resumed pairs.",
    })
    return records


def aggregate_judge(records):
    by_pairing = defaultdict(list)
    for rec in records:
        by_pairing[rec["pairing"]].append(rec["pref"])
    pairings = {
        name: {"variant_A": PAIRINGS[name][0], "variant_B": PAIRINGS[name][1],
               **rates_from_prefs(by_pairing[name])}
        for name in PAIRINGS
    }

    def breakdown(axis):
        buckets = defaultdict(list)
        for rec in records:
            buckets[rec.get(axis + "_B", "")].append(rec["pref"])
        return {k: rates_from_prefs(v) for k, v in sorted(buckets.items())}

    tested = [r for r in records if r.get("position_tested")]
    untested_n = len(records) - len(tested)
    position_flips = sum(1 for r in tested if r.get("position_flip"))
    soft = sum(1 for r in tested if r.get("soft_disagreement"))
    tested_n = len(tested)
    disagreement_rate_tested = (position_flips + soft) / tested_n if tested_n else None
    return {
        "scale": "pairwise preference, win space [0,1]; A/B/TIE consensus over BOTH argument orders",
        "orientation_finding": (
            "rlaif.py:80-81 picks one physical orientation per pair via int(key[:8],16)%2 and "
            "un-swaps the verdict (:98-99): balanced ACROSS distinct pairs, one orientation per call. "
            "compare(a,b) and compare(b,a) hit opposite physical orientations iff their swap bits are "
            "equal (s1==s2) -- the position-tested case, computed exactly from the judge's own _key. "
            "When s1!=s2 the second call is a byte-identical prompt and is skipped. Disagreement rates "
            "below are over the tested subset only."
        ),
        "cache_key_finding": (
            "rlaif.py:67-72 _key hashes {model, problem, a, b} -- problem AND both response texts. "
            "SFT-vs-RLVR and SFT-vs-RLAIF cannot collide."
        ),
        "position_tested_n": tested_n,
        "position_untested_n": untested_n,
        "position_disagreement_count": position_flips,
        "soft_disagreement_count": soft,
        "position_disagreement_rate_tested": disagreement_rate_tested,
        "caveats": {
            "gold_distractor": (
                "gold_distractor_wrong_final is 2 lines vs clean_correct's 5; a clean-preference "
                "here is confounded with judge length bias and must NOT be read as distractor "
                "detection with this set."
            ),
        },
        "pairings": pairings,
        "S_reason": pairings["S_reason"]["better"],
        "S_outcome": pairings["S_outcome"]["better"],
        "breakdown_by_style": breakdown("style"),
        "breakdown_by_reasoning_quality": breakdown("reasoning_quality"),
    }



def write_pairs_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def peek(cfg):
    path = cfg["paths"]["task5_diagnostics"]
    rows = read_jsonl(path)
    print(f"=== {path}  (n={len(rows)}) ===")
    print("rows[0]:", json.dumps(rows[0], ensure_ascii=False, indent=2)[:2000])
    seen = {}
    for r in rows:
        seen.setdefault(r["variant_type"], r)
    for v in sorted(EXPECTED_VARIANTS):
        r = seen[v]
        print(f"\n--- {v} (problem_id={r['problem_id']}) ---")
        print(json.dumps({k: r[k] for k in r if k != "response"}, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--mechanism", choices=["verifier", "judge", "both"], default="both",
                    help="verifier runs on CPU; judge loads Qwen2.5-3B and wants a GPU")
    ap.add_argument("--peek", action="store_true", help="print one record per variant and exit")
    ap.add_argument("--self-test", action="store_true",
                    help="run the discriminating verifier self-check and exit")
    ap.add_argument("--fresh", action="store_true", help="ignore existing judge pair log")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    diag_path = repo_path(cfg["paths"]["task5_diagnostics"])

    if args.peek:
        peek(cfg)
        return
    if args.self_test:
        discriminating_self_test(diag_path)
        return

    if not diag_path.exists():
        raise FileNotFoundError(
            f"Diagnostic set not found: {diag_path}. Install assets first "
            "(python -m scripts.download_assets) -- it is not committed to the repo."
        )

    rows, groups = load_and_guard(diag_path)
    print("rows[0] keys:", sorted(rows[0].keys()))
    verifier_self_check(rows)

    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)
    result = {
        "config": args.config,
        "n_rows": len(rows),
        "n_problems": len(groups),
        "pairings": {k: {"A": v[0], "B": v[1]} for k, v in PAIRINGS.items()},
    }

    if args.mechanism in ("verifier", "both"):
        v_summary, v_pairs = run_verifier_pairings(groups)
        result["verifier"] = v_summary
        write_pairs_csv(
            outdir / "task5_diagnostic_verifier_pairs.csv", v_pairs,
            ["pairing", "problem_id", "variant_A", "variant_B",
             "reward_A", "reward_B", "pref", "style_B", "reasoning_quality_B"],
        )
        print(f"[verifier] S_reason={v_summary['S_reason']}  S_outcome={v_summary['S_outcome']}")

    if args.mechanism in ("judge", "both"):
        records = run_judge_pairings(
            cfg, groups,
            out_jsonl=outdir / "diagnostic_judge_pairs.jsonl",
            cache_path="cached/task5_diagnostic_judge_cache.json",
            resume=not args.fresh,
        )
        result["judge"] = aggregate_judge(records)
        write_pairs_csv(
            outdir / "task5_diagnostic_judge_pairs.csv", records,
            ["pairing", "problem_id", "variant_A", "variant_B",
             "r_ab", "r_ba_semantic", "pref", "position_tested", "skipped_second_call",
             "position_flip", "soft_disagreement", "style_B", "reasoning_quality_B"],
        )
        j = result["judge"]
        print(f"[judge] S_reason={j['S_reason']}  S_outcome={j['S_outcome']}  "
              f"position_tested_n={j['position_tested_n']} (untested/skipped={j['position_untested_n']})  "
              f"disagreement_rate_tested={j['position_disagreement_rate_tested']} "
              f"(flips={j['position_disagreement_count']} soft={j['soft_disagreement_count']})")
        gd = j["pairings"]["gold_distractor"]
        print(f"[judge] NOTE gold_distractor better={gd['better']} wrong={gd['wrong']} -- "
              "length confound (2 vs 5 lines); not read as distractor detection.")

    save_json(outdir / "task5_sensitivity.json", result)
    print(f"[done] wrote {outdir / 'task5_sensitivity.json'}")


if __name__ == "__main__":
    main()
