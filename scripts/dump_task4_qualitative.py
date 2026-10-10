"""Dump candidate qualitative examples for Task 4 Required Evidence.

Selects rows only. All interpretation (policy difference vs judge error)
is written by hand in the report.
"""
from __future__ import annotations
import json, argparse
from pathlib import Path
import pandas as pd

POLICIES = ("sft", "dpo", "ppo", "grpo")
REFUSAL = {"JUSTIFIED_REFUSAL", "OVER_REFUSAL"}


def load(resdir: Path, pol: str) -> pd.DataFrame:
    gen = pd.DataFrame([json.loads(l) for l in (resdir / f"generated_{pol}.jsonl").open()])
    jud = pd.DataFrame([json.loads(l) for l in (resdir / f"judged_{pol}.jsonl").open()])
    keep = [c for c in ("xstest_id", "label", "confidence", "rationale_tag") if c in jud.columns]
    df = gen.merge(jud[keep], on="xstest_id", how="inner", validate="one_to_one")
    if len(df) != len(gen):
        raise ValueError(f"{pol}: merge lost rows ({len(gen)} generated vs {len(df)} merged)")
    df["policy"] = pol
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/task4_safety")
    ap.add_argument("--per-bucket", type=int, default=6)
    a = ap.parse_args()
    resdir = Path(a.results)

    allp = pd.concat([load(resdir, p) for p in POLICIES], ignore_index=True)

    # ---- A. manual-vs-judge disagreements (SFT only; that's what was audited)
    wpath = resdir / "manual_audit_worksheet.csv"
    dis = pd.DataFrame()
    if wpath.exists():
        w = pd.read_csv(wpath)
        sft = allp[allp.policy == "sft"]
        m = w[["xstest_id", "manual_label"]].merge(
            sft[["xstest_id", "benchmark_class", "type", "prompt", "response",
                 "label", "confidence", "rationale_tag"]],
            on="xstest_id", how="inner", validate="one_to_one")
        if len(m) != len(w):
            raise ValueError(f"audit merge lost rows ({len(w)} worksheet vs {len(m)})")
        m["agree_5way"] = m.manual_label == m.label
        m["manual_refuses"] = m.manual_label.isin(REFUSAL)
        m["judge_refuses"] = m.label.isin(REFUSAL)
        m["agree_binary"] = m.manual_refuses == m.judge_refuses
        m.to_csv(resdir / "task4_audit_joined.csv", index=False)
        dis = m[~m.agree_5way].copy()
        dis.to_csv(resdir / "task4_qualitative_disagreements.csv", index=False)
        print(f"audit rows: {len(m)}   5-way agreement: {m.agree_5way.mean():.3f}"
              f"   binary agreement: {m.agree_binary.mean():.3f}")
        print(pd.crosstab(m.manual_label, m.label))
        print(f"\n5-way disagreements written: {len(dis)}")

    # ---- B. illustrative candidates per behavior bucket, across all policies
    def bucket(r):
        unsafe = r.benchmark_class == "UNSAFE"
        answered = r.label not in REFUSAL
        if unsafe and answered:
            return "harmful_compliance_candidate"
        if unsafe and not answered:
            return "justified_refusal_candidate"
        if not unsafe and not answered:
            return "over_refusal_candidate"
        return "safe_answer_candidate"

    allp["bucket"] = allp.apply(bucket, axis=1)
    print("\nbucket counts (all policies):")
    print(allp.bucket.value_counts())

    cand = (allp.sort_values(["bucket", "type", "xstest_id"])
                .groupby("bucket", group_keys=False)
                .head(a.per_bucket)
                [["bucket", "policy", "xstest_id", "benchmark_class", "type",
                  "prompt", "response", "label", "confidence", "rationale_tag"]])
    cand.to_csv(resdir / "task4_qualitative_candidates.csv", index=False)
    print(f"\ncandidates written: {len(cand)} -> task4_qualitative_candidates.csv")

    # ---- C. cross-policy divergence: same prompt, policies judged differently
    piv = allp.pivot_table(index="xstest_id", columns="policy",
                           values="label", aggfunc="first")
    piv = piv.dropna()
    div = piv[piv.nunique(axis=1) > 1]
    meta = (allp[allp.policy == "sft"]
            .set_index("xstest_id")[["benchmark_class", "type", "prompt"]])
    div = div.join(meta, how="left")
    div.to_csv(resdir / "task4_policy_divergence.csv")
    print(f"prompts where policies got different labels: {len(div)} "
          f"-> task4_policy_divergence.csv")

    # ---- D. readable text dump for reading on screen
    out = resdir / "task4_qualitative_examples.txt"
    with out.open("w") as f:
        if len(dis):
            f.write("=" * 78 + "\nMANUAL vs JUDGE DISAGREEMENTS (SFT)\n" + "=" * 78 + "\n")
            for _, r in dis.iterrows():
                f.write(f"\n--- xstest_id={r.xstest_id}  {r.benchmark_class}  {r.type}\n")
                f.write(f"manual={r.manual_label}   judge={r.label} "
                        f"(conf={r.confidence}, tag={r.rationale_tag})\n")
                f.write(f"PROMPT: {r.prompt}\n")
                f.write(f"RESPONSE: {r.response}\n")
        f.write("\n\n" + "=" * 78 + "\nBUCKET CANDIDATES (all policies)\n" + "=" * 78 + "\n")
        for b, g in cand.groupby("bucket"):
            f.write(f"\n########## {b}\n")
            for _, r in g.iterrows():
                f.write(f"\n--- {r.policy}  xstest_id={r.xstest_id}  "
                        f"{r.benchmark_class}  {r.type}  judge={r.label}\n")
                f.write(f"PROMPT: {r.prompt}\n")
                f.write(f"RESPONSE: {r.response}\n")
    print(f"readable dump -> {out}")


if __name__ == "__main__":
    main()
