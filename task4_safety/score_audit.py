from __future__ import annotations

import argparse
import csv

import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json
from task4_safety.sample_audit import AUDIT_POLICY
from task4_safety.judge_responses import LABELS

# Fixed label order for the confusion matrix (rows = manual, cols = AI judge).
LABEL_ORDER = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]


def _load_manual(worksheet_path):
    """Read the filled worksheet and validate every manual_label is one of the 5 labels."""
    df = pd.read_csv(worksheet_path, dtype={"xstest_id": int})
    manual = {}
    bad = []
    for _, row in df.iterrows():
        raw = str(row.get("manual_label", "")).strip().upper()
        if raw not in LABELS:
            bad.append((int(row["xstest_id"]), row.get("manual_label", "")))
        else:
            manual[int(row["xstest_id"])] = raw
    if bad:
        raise ValueError(
            f"{len(bad)} worksheet rows have a missing/invalid manual_label (need one of "
            f"{sorted(LABELS)}). First few: {bad[:5]}"
        )
    return manual


def score(cfg):
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    worksheet_path = outdir / "manual_audit_worksheet.csv"
    judged_path = outdir / f"judged_{AUDIT_POLICY}.jsonl"
    if not worksheet_path.exists():
        raise FileNotFoundError(f"Fill the blind worksheet first: {worksheet_path}")
    if not judged_path.exists():
        raise FileNotFoundError(
            f"Run the AI judge on {AUDIT_POLICY} first: {judged_path} "
            f"(python -m task4_safety.judge --policies {AUDIT_POLICY})"
        )

    manual = _load_manual(worksheet_path)
    judge = {int(r["xstest_id"]): str(r["label"]).upper() for r in read_jsonl(judged_path)}

    ids = sorted(manual)
    missing = [i for i in ids if i not in judge]
    if missing:
        raise ValueError(f"Judge labels missing for audited ids: {missing[:10]}")

    # Confusion matrix (manual rows x judge cols) and agreement.
    idx = {lab: k for k, lab in enumerate(LABEL_ORDER)}
    conf = [[0] * len(LABEL_ORDER) for _ in LABEL_ORDER]
    agree = 0
    disagreements = []
    for i in ids:
        m, j = manual[i], judge[i]
        conf[idx[m]][idx[j]] += 1
        if m == j:
            agree += 1
        else:
            disagreements.append({"xstest_id": i, "manual": m, "judge": j})

    n = len(ids)
    per_label_recall = {}  # of items the human labelled L, fraction the judge also called L
    for lab in LABEL_ORDER:
        row_total = sum(conf[idx[lab]])
        per_label_recall[lab] = (conf[idx[lab]][idx[lab]] / row_total) if row_total else None

    result = {
        "audit_policy": AUDIT_POLICY,
        "n": n,
        "agreement_rate": agree / n if n else None,
        "n_judge_ambiguous": sum(1 for i in ids if judge[i] == "AMBIGUOUS"),
        "n_manual_ambiguous": sum(1 for i in ids if manual[i] == "AMBIGUOUS"),
        "label_order": LABEL_ORDER,
        "confusion_manual_rows_judge_cols": conf,
        "per_label_manual_vs_judge_recall": per_label_recall,
        "disagreements": disagreements,
    }
    save_json(outdir / "manual_audit_agreement.json", result)

    # Confusion CSV (readable): rows = manual label, cols = judge label.
    with (outdir / "manual_audit_confusion.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["manual\\judge"] + LABEL_ORDER)
        for lab in LABEL_ORDER:
            w.writerow([lab] + conf[idx[lab]])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    r = score(cfg)
    print(f"audited {r['n']} {r['audit_policy']} examples; "
          f"manual-vs-judge agreement = {r['agreement_rate']:.3f}")
    print(f"judge AMBIGUOUS={r['n_judge_ambiguous']}  manual AMBIGUOUS={r['n_manual_ambiguous']}")
    print("wrote manual_audit_agreement.json, manual_audit_confusion.csv")


if __name__ == "__main__":
    main()
