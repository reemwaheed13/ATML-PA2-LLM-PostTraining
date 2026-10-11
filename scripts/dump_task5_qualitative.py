"""Dump candidate qualitative examples for the Task 5 report. Pure read; no model.

Selection criteria live HERE, in code -- nothing is hand-picked. All interpretation (is this a
judge error or a real quality difference?) is written by hand in the report.

Two CSVs to results/task5_feedback/:
  task5_qual_vj_disagreements.csv  real in-domain generations where the verifier and the AI judge
                                   rank policy-vs-SFT differently (the verifier strictly separates
                                   them but the judge prefers the other one, or ties). Criterion:
                                   verifier_rank in {A,B} and judge_pref != verifier_rank.
  task5_qual_diagnostic.csv        diagnostic examples spanning the perturbation categories. Per
                                   pairing, criterion (stated per row in selection_reason): prefer a
                                   judge WRONG-preference (pref==B, the most report-worthy failure),
                                   else a position-bias flip, else the lowest problem_id.

Run:  python -m scripts.dump_task5_qualitative --config configs/feedback.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def _df(path):
    return pd.DataFrame(read_jsonl(path))


def vj_disagreements(outdir):
    """In-domain verifier-judge ranking disagreements, joined to the real generations."""
    frames = []
    for policy in ("rlvr", "rlaif"):
        pw_path = outdir / f"pairwise_gsm_{policy}.jsonl"
        gen_p, gen_s = outdir / f"gen_gsm_{policy}.jsonl", outdir / "gen_gsm_sft.jsonl"
        if not (pw_path.exists() and gen_p.exists() and gen_s.exists()):
            continue
        pw = _df(pw_path)
        pol = _df(gen_p)[["id", "question", "gold_final", "response"]].rename(columns={"response": "response_policy"})
        sft = _df(gen_s)[["id", "response"]].rename(columns={"response": "response_sft"})
        m = pw.merge(pol, on="id", how="left", validate="one_to_one")
        if m["response_policy"].isna().any():
            raise ValueError(f"{policy}: pairwise rows without a matching generation (merge lost rows)")
        m = m.merge(sft, on="id", how="left", validate="one_to_one")
        if m["response_sft"].isna().any():
            raise ValueError(f"{policy}: pairwise rows without a matching SFT generation")
        dis = m[(m.verifier_rank.isin(["A", "B"])) & (m.judge_pref != m.verifier_rank)].copy()
        dis["policy"] = policy
        dis["disagreement_type"] = dis.apply(
            lambda r: "judge_ties_verifier_decisive" if r.judge_pref == "TIE" else "opposite_decisive", axis=1)
        frames.append(dis[["policy", "id", "question", "gold_final", "reward_policy", "reward_sft",
                            "verifier_rank", "judge_pref", "disagreement_type",
                            "response_policy", "response_sft"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def diagnostic_examples(outdir, diag_path, per_pairing):
    jp_path = outdir / "diagnostic_judge_pairs.jsonl"
    if not jp_path.exists():
        return pd.DataFrame()
    jp = _df(jp_path)
    diag = read_jsonl(diag_path)
    by_key = {(str(r["problem_id"]), r["variant_type"]): r for r in diag}

    def rank(row):  # lower sorts first -> most report-worthy failure first
        if row.judge_pref == "B":
            return (0, int(row.problem_id))          # judge wrong-preference
        if bool(row.get("position_flip", False)):
            return (1, int(row.problem_id))          # position-bias flip
        return (2, int(row.problem_id))

    def reason(row):
        if row.judge_pref == "B":
            return "judge wrong-preference (prefers perturbed)"
        if bool(row.get("position_flip", False)):
            return "position-bias flip"
        return "lowest problem_id (no failure to show)"

    jp["_rk"] = jp.apply(rank, axis=1)
    jp["selection_reason"] = jp.apply(reason, axis=1)
    picks = (jp.sort_values("_rk").groupby("pairing", group_keys=False).head(per_pairing))

    out = []
    for _, r in picks.iterrows():
        pid = str(r.problem_id)
        a, b = by_key[(pid, r.variant_A)], by_key[(pid, r.variant_B)]
        out.append({
            "pairing": r.pairing, "problem_id": pid,
            "variant_A": r.variant_A, "variant_B": r.variant_B,
            "judge_pref": r.judge_pref, "position_tested": r.get("position_tested"),
            "position_flip": r.get("position_flip"),
            "style_B": r.get("style_B"), "reasoning_quality_B": r.get("reasoning_quality_B"),
            "selection_reason": r.selection_reason,
            "question": a["question"], "gold_final": a["gold_final"],
            "response_A_clean": a["response"], "response_B_perturbed": b["response"],
        })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--per-pairing", type=int, default=3)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    diag_path = repo_path(cfg["paths"]["task5_diagnostics"])

    vj = vj_disagreements(outdir)
    if len(vj):
        vj.to_csv(outdir / "task5_qual_vj_disagreements.csv", index=False)
        print(f"verifier-judge disagreements: {len(vj)} -> task5_qual_vj_disagreements.csv")
    else:
        print("no in-domain pairwise logs yet -> skipped task5_qual_vj_disagreements.csv")

    dx = diagnostic_examples(outdir, diag_path, args.per_pairing)
    if len(dx):
        dx.to_csv(outdir / "task5_qual_diagnostic.csv", index=False)
        print(f"diagnostic examples: {len(dx)} ({args.per_pairing}/pairing) -> task5_qual_diagnostic.csv")
    else:
        print("no diagnostic judge log yet -> skipped task5_qual_diagnostic.csv")


if __name__ == "__main__":
    main()
