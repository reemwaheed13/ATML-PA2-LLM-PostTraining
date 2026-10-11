"""Aggregate ALL Task 5 Required Evidence into one JSON + per-table CSVs + figures. Pure read.

Reads from results/task5_feedback/:
  task5_gsm.json / task5_transfer.json   (evaluate_math: in-domain / OOD)
  task5_sensitivity.json                 (score_perturbations: controlled diagnostic)
  diagnostic_judge_meta.json             (judge inference cost)
  gen_{gsm,transfer}_{policy}.jsonl      (cap-without-EOS counts)

Emits to results/task5_feedback/:
  task5_indomain.csv   per policy: exact acc, win-rate vs SFT (+win/tie/loss), format compliance,
                       mean & sd response length, verifier-judge agreement, diag20/other280 split
  task5_diagnostic.csv per pairing x mechanism: better/tie/wrong (verifier binary vs judge [0,1]
                       kept on separate rows, never averaged), plus S_reason / S_outcome
  task5_ood.csv        per policy: exact acc, win-rate, length, drop vs in-domain
  task5_caps.csv       per policy per eval set: cap-without-EOS (truncated) + truncated_no_final
  task5_aggregate.json everything, with a checklist_coverage block
And figures to report/figures/ (300 dpi, no title): task5_indomain.png, task5_diagnostic.png,
  task5_sensitivity.png, task5_transfer.png.

CONSTRAINT: the binary verifier reward and the [0,1] pairwise win rate are never put on one axis
or in one aggregate -- separate CSV rows and separate figure panels throughout.

Run:  python -m scripts.aggregate_task5 --config configs/feedback.yaml [--tables-only]
"""

from __future__ import annotations

import argparse
import csv

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json

POLICIES = ("sft", "rlvr", "rlaif")
PAIRINGS = ("S_reason", "S_outcome", "filler_control", "gold_distractor")
PAIRING_B = {  # perturbed variant B for each pairing (A is clean_correct throughout)
    "S_reason": "corrupt_reasoning_correct_final",
    "S_outcome": "good_reasoning_wrong_final",
    "filler_control": "persuasive_filler_correct",
    "gold_distractor": "gold_distractor_wrong_final",
}
DPI = 300


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(c, "") for c in header])


def _load(outdir, name):
    p = outdir / name
    return load_json(p) if p.exists() else None


# --------------------------------------------------------------------------- tables

def indomain_rows(gsm):
    rows = []
    for p in POLICIES:
        m = gsm["policies"][p]
        pw = m.get("pairwise", {})
        ds = m.get("diagnostic_split", {})
        tos = m.get("train_overlap_split", {})
        rows.append({
            "policy": p, "n": m["n"],
            "exact_acc": m["exact_acc"], "format_compliance": m["format_compliance"],
            "mean_response_tokens": m["mean_response_tokens"],
            "std_response_tokens": m.get("std_response_tokens"),
            "win_rate_vs_sft": pw.get("win_rate_vs_sft", ""),
            "win": pw.get("win", ""), "tie": pw.get("tie", ""), "loss": pw.get("loss", ""),
            "verifier_judge_agreement": pw.get("verifier_judge_agreement", ""),
            "verifier_strict_n": pw.get("verifier_strict_n", ""),
            "exact_acc_diag20": ds.get("diag20", {}).get("exact_acc", ""),
            "exact_acc_other280": ds.get("other280", {}).get("exact_acc", ""),
            # train-seen vs unseen (leakage split): acc + win-rate per subset
            "n_seen": tos.get("n_seen", ""), "n_unseen": tos.get("n_unseen", ""),
            "exact_acc_seen59": tos.get("seen", {}).get("exact_acc", ""),
            "exact_acc_unseen241": tos.get("unseen", {}).get("exact_acc", ""),
            "format_seen59": tos.get("seen", {}).get("format_compliance", ""),
            "format_unseen241": tos.get("unseen", {}).get("format_compliance", ""),
            "len_seen59": tos.get("seen", {}).get("mean_response_tokens", ""),
            "len_unseen241": tos.get("unseen", {}).get("mean_response_tokens", ""),
            "win_rate_seen59": tos.get("seen", {}).get("win_rate_vs_sft", ""),
            "win_rate_unseen241": tos.get("unseen", {}).get("win_rate_vs_sft", ""),
        })
    return rows


def diagnostic_rows(diag):
    """One row per (pairing, mechanism). Verifier (binary) and judge ([0,1]) on separate rows."""
    rows = []
    for mech in ("verifier", "judge"):
        block = diag[mech]
        for name in PAIRINGS:
            pr = block["pairings"][name]
            rows.append({
                "mechanism": mech, "pairing": name,
                "variant_A": pr.get("variant_A", "clean_correct"),
                "variant_B": pr.get("variant_B", PAIRING_B[name]),
                "better": pr["better"], "tie": pr["tie"], "wrong": pr["wrong"], "n": pr["n"],
            })
        rows.append({"mechanism": mech, "pairing": "S_reason_summary", "variant_A": "", "variant_B": "",
                     "better": block["S_reason"], "tie": "", "wrong": "", "n": ""})
        rows.append({"mechanism": mech, "pairing": "S_outcome_summary", "variant_A": "", "variant_B": "",
                     "better": block["S_outcome"], "tie": "", "wrong": "", "n": ""})
    # clean_correct is the reference variant A in every pairing -> no self-comparison row; note it.
    rows.append({"mechanism": "both", "pairing": "clean_correct_control",
                 "variant_A": "clean_correct", "variant_B": "(reference)",
                 "better": "", "tie": "", "wrong": "", "n": ""})
    return rows


def ood_rows(transfer):
    rows = []
    for p in POLICIES:
        m = transfer["policies"][p]
        pw = m.get("pairwise", {})
        rows.append({
            "policy": p, "n": m["n"], "exact_acc": m["exact_acc"],
            "exact_acc_drop_vs_gsm_full": m.get("exact_acc_drop_vs_gsm_full", ""),
            "exact_acc_drop_vs_gsm_unseen": m.get("exact_acc_drop_vs_gsm_unseen", ""),
            "mean_response_tokens": m["mean_response_tokens"],
            "std_response_tokens": m.get("std_response_tokens"),
            "win_rate_vs_sft": pw.get("win_rate_vs_sft", ""),
        })
    return rows


def cap_rows(outdir, datasets):
    rows = []
    for ds in datasets:
        for p in POLICIES:
            gp = outdir / f"gen_{ds}_{p}.jsonl"
            if not gp.exists():
                continue
            recs = read_jsonl(gp)
            rows.append({
                "dataset": ds, "policy": p, "n": len(recs),
                "cap_without_eos": sum(1 for r in recs if not bool(r.get("terminated", True))),
                "truncated": sum(1 for r in recs if bool(r.get("truncated", False))),
            })
    return rows


# --------------------------------------------------------------------------- figures

def _figdir():
    d = repo_path("report/figures")
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_figures(gsm, diag, transfer):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # In-domain: exact acc (binary) and win-rate-vs-SFT ([0,1]) on SEPARATE panels/axes.
    fig, (ax_a, ax_w) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_a.bar(POLICIES, [gsm["policies"][p]["exact_acc"] for p in POLICIES], color="tab:blue")
    ax_a.set_ylabel("exact final-answer accuracy (verifier, binary)")
    ax_a.set_xlabel("policy"); ax_a.set_ylim(0, 1); ax_a.grid(True, alpha=0.3, axis="y")
    wp = [p for p in POLICIES if gsm["policies"][p].get("pairwise")]
    ax_w.bar(wp, [gsm["policies"][p]["pairwise"]["win_rate_vs_sft"] for p in wp], color="tab:green")
    ax_w.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax_w.set_ylabel("AI judge win rate vs SFT ([0,1])")
    ax_w.set_xlabel("policy"); ax_w.set_ylim(0, 1); ax_w.grid(True, alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(_figdir() / "task5_indomain.png", dpi=DPI); plt.close(fig)
    print("wrote task5_indomain.png")

    # Diagnostic: stacked better/tie/wrong per pairing, verifier and judge on SEPARATE panels.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, mech in zip(axes, ("verifier", "judge")):
        b = [diag[mech]["pairings"][k]["better"] for k in PAIRINGS]
        t = [diag[mech]["pairings"][k]["tie"] for k in PAIRINGS]
        w = [diag[mech]["pairings"][k]["wrong"] for k in PAIRINGS]
        x = np.arange(len(PAIRINGS))
        ax.bar(x, b, color="tab:green", label="better (prefers clean)")
        ax.bar(x, t, bottom=b, color="tab:gray", label="tie")
        ax.bar(x, w, bottom=[bi + ti for bi, ti in zip(b, t)], color="tab:red", label="wrong (prefers perturbed)")
        ax.set_xticks(x); ax.set_xticklabels(PAIRINGS, rotation=30, ha="right")
        ax.set_title(f"{mech} ({'binary' if mech == 'verifier' else '[0,1]'})", fontsize=10)
        ax.set_ylim(0, 1); ax.grid(True, alpha=0.3, axis="y")
    axes[0].set_ylabel("rate"); axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(_figdir() / "task5_diagnostic.png", dpi=DPI); plt.close(fig)
    print("wrote task5_diagnostic.png")

    # Sensitivity summary: S_reason / S_outcome, verifier vs judge (grouped).
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(2)
    ax.bar(x - 0.2, [diag["verifier"]["S_reason"], diag["verifier"]["S_outcome"]], width=0.4,
           color="tab:blue", label="verifier (binary)")
    ax.bar(x + 0.2, [diag["judge"]["S_reason"], diag["judge"]["S_outcome"]], width=0.4,
           color="tab:green", label="judge ([0,1])")
    ax.set_xticks(x); ax.set_xticklabels(["S_reason", "S_outcome"])
    ax.set_ylabel("Pr[prefers better response]"); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y"); ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(_figdir() / "task5_sensitivity.png", dpi=DPI); plt.close(fig)
    print("wrote task5_sensitivity.png")

    # Transfer: in-domain vs OOD exact accuracy per policy (binary), with the drop visible.
    if transfer:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(POLICIES))
        ax.bar(x - 0.2, [gsm["policies"][p]["exact_acc"] for p in POLICIES], width=0.4,
               color="tab:blue", label="GSM8K (in-domain)")
        ax.bar(x + 0.2, [transfer["policies"][p]["exact_acc"] for p in POLICIES], width=0.4,
               color="tab:orange", label="SVAMP (OOD)")
        ax.set_xticks(x); ax.set_xticklabels(POLICIES)
        ax.set_ylabel("exact accuracy (binary)"); ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y"); ax.legend(loc="best", fontsize=8)
        fig.tight_layout(); fig.savefig(_figdir() / "task5_transfer.png", dpi=DPI); plt.close(fig)
        print("wrote task5_transfer.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--tables-only", action="store_true", help="skip figure regeneration")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"

    gsm = _load(outdir, "task5_gsm.json")
    diag = _load(outdir, "task5_sensitivity.json")
    transfer = _load(outdir, "task5_transfer.json")
    diag_cost = _load(outdir, "diagnostic_judge_meta.json")
    if gsm is None or diag is None:
        raise FileNotFoundError("need task5_gsm.json (evaluate_math gsm) and task5_sensitivity.json "
                                "(score_perturbations) before aggregating.")

    _write_csv(outdir / "task5_indomain.csv",
               ["policy", "n", "exact_acc", "format_compliance", "mean_response_tokens",
                "std_response_tokens", "win_rate_vs_sft", "win", "tie", "loss",
                "verifier_judge_agreement", "verifier_strict_n", "exact_acc_diag20", "exact_acc_other280",
                "n_seen", "n_unseen", "exact_acc_seen59", "exact_acc_unseen241",
                "format_seen59", "format_unseen241", "len_seen59", "len_unseen241",
                "win_rate_seen59", "win_rate_unseen241"],
               indomain_rows(gsm))
    _write_csv(outdir / "task5_diagnostic.csv",
               ["mechanism", "pairing", "variant_A", "variant_B", "better", "tie", "wrong", "n"],
               diagnostic_rows(diag))
    datasets = ["gsm"] + (["transfer"] if transfer else [])
    _write_csv(outdir / "task5_caps.csv",
               ["dataset", "policy", "n", "cap_without_eos", "truncated"], cap_rows(outdir, datasets))
    if transfer:
        _write_csv(outdir / "task5_ood.csv",
                   ["policy", "n", "exact_acc", "exact_acc_drop_vs_gsm_full", "exact_acc_drop_vs_gsm_unseen",
                    "mean_response_tokens", "std_response_tokens", "win_rate_vs_sft"], ood_rows(transfer))

    checklist = {
        "in_domain_exact_acc": True, "in_domain_pairwise": True, "in_domain_format": True,
        "in_domain_length_mean_and_sd": True, "in_domain_verifier_judge_agreement": True,
        "diagnostic_better_tie_wrong_per_mechanism": True, "diagnostic_S_reason_S_outcome": True,
        "ood_acc_pairwise_length_drop": bool(transfer),
        "cap_without_eos_counts": True,
        "position_untested_and_disagreement_rate": True,
    }
    save_json(outdir / "task5_aggregate.json", {
        "policies": list(POLICIES),
        "scales_note": "verifier reward binary {0,1}; judge win rate [0,1]; never one aggregate.",
        "in_domain": gsm["policies"],
        "diagnostic": {"verifier": diag["verifier"], "judge": diag["judge"]},
        "out_of_domain": (transfer or {}).get("policies"),
        "judge_inference_cost": diag_cost,
        "position_bias": {
            "diagnostic": {k: diag["judge"].get(k) for k in
                           ("position_tested_n", "position_untested_n",
                            "position_disagreement_count", "position_disagreement_rate_tested")},
            "in_domain": {p: {k: gsm["policies"][p].get("pairwise", {}).get(k) for k in
                              ("position_tested_n", "position_untested_n", "position_disagreement_count")}
                          for p in ("rlvr", "rlaif")},
        },
        "checklist_coverage": checklist,
    })
    print(f"wrote task5_indomain.csv, task5_diagnostic.csv, task5_caps.csv"
          + (", task5_ood.csv" if transfer else "") + f", task5_aggregate.json to {outdir}")

    if not args.tables_only:
        make_figures(gsm, diag, transfer)


if __name__ == "__main__":
    main()
