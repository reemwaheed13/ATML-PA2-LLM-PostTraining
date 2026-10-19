
from __future__ import annotations

import argparse

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import load_json, save_json

POLICIES = ("sft", "rlvr", "rlaif")
VERIFIER_TIE_PAIRINGS = ("S_reason", "filler_control")


def _require(path):
    p = repo_path(path)
    if not p.exists():
        raise FileNotFoundError(f"missing {p}; run the upstream step first (see README Task 5 sequence)")
    return p


def strict_subset_breakdown(pairwise_path):
    rows = read_jsonl(pairwise_path)
    strict = [r for r in rows if r["verifier_rank"] in ("A", "B")]
    agree = sum(1 for r in strict if r["judge_pref"] == r["verifier_rank"])
    judge_tie = sum(1 for r in strict if r["judge_pref"] == "TIE")
    disagree_decisive = len(strict) - agree - judge_tie
    n = len(strict)
    return {
        "verifier_strict_n": n,
        "agree": agree, "disagree_decisive": disagree_decisive, "judge_tie": judge_tie,
        "agreement_tie_is_disagree": (agree / n) if n else None,
        "agreement_ties_excluded": (agree / (n - judge_tie)) if (n - judge_tie) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"

    gsm = load_json(_require(outdir / "task5_gsm.json"))
    diag = load_json(_require(outdir / "task5_sensitivity.json"))
    transfer_p = outdir / "task5_transfer.json"
    transfer = load_json(transfer_p) if transfer_p.exists() else None
    diag_cost = load_json(outdir / "diagnostic_judge_meta.json") if (outdir / "diagnostic_judge_meta.json").exists() else None

    comparison = {
        "note": "verifier reward is binary {0,1}; judge win rate is [0,1]; never aggregated together.",
        "scales": {"verifier": "binary exact reward {0,1}", "judge": "pairwise win rate [0,1]"},
    }

    indomain = {}
    for p in POLICIES:
        m = gsm["policies"][p]
        pw = m.get("pairwise", {})
        ds = m.get("diagnostic_split", {})
        tos = m.get("train_overlap_split", {})
        entry = {
            "verifier": {
                "exact_acc": m["exact_acc"],
                "exact_acc_diag20": ds.get("diag20", {}).get("exact_acc"),
                "exact_acc_other280": ds.get("other280", {}).get("exact_acc"),
                "exact_acc_seen59": tos.get("seen", {}).get("exact_acc"),
                "exact_acc_unseen241": tos.get("unseen", {}).get("exact_acc"),
                "n_seen": tos.get("n_seen"), "n_unseen": tos.get("n_unseen"),
            },
            "format_compliance": m["format_compliance"],
            "mean_response_tokens": m["mean_response_tokens"],
            "std_response_tokens": m.get("std_response_tokens"),
        }
        if tos:
            entry["judge_seen_unseen"] = {
                "win_rate_seen59": tos.get("seen", {}).get("win_rate_vs_sft"),
                "win_rate_unseen241": tos.get("unseen", {}).get("win_rate_vs_sft"),
            }
        if pw:
            entry["judge"] = {
                "win_rate_vs_sft": pw["win_rate_vs_sft"],
                "win": pw["win"], "tie": pw["tie"], "loss": pw["loss"],
            }
            entry["verifier_judge_agreement"] = strict_subset_breakdown(outdir / f"pairwise_gsm_{p}.jsonl")
        indomain[p] = entry
    comparison["in_domain"] = indomain

    comparison["verifier_tie_region"] = {
        "description": "diagnostic pairings where the verifier ties (both parse to the same correct "
                       "final); the judge's better/tie/wrong here is resolution the binary verifier "
                       "structurally lacks.",
        "verifier": {k: diag["verifier"]["pairings"][k] for k in VERIFIER_TIE_PAIRINGS},
        "judge": {k: diag["judge"]["pairings"][k] for k in VERIFIER_TIE_PAIRINGS},
        "S_reason_verifier": diag["verifier"]["S_reason"],
        "S_reason_judge": diag["judge"]["S_reason"],
    }

    if transfer:
        comparison["out_of_domain"] = {
            p: {
                "exact_acc": transfer["policies"][p]["exact_acc"],
                "exact_acc_drop_vs_gsm_full": transfer["policies"][p].get("exact_acc_drop_vs_gsm_full"),
                "exact_acc_drop_vs_gsm_unseen": transfer["policies"][p].get("exact_acc_drop_vs_gsm_unseen"),
                "mean_response_tokens": transfer["policies"][p]["mean_response_tokens"],
                "win_rate_vs_sft": transfer["policies"][p].get("pairwise", {}).get("win_rate_vs_sft"),
            }
            for p in POLICIES
        }
    else:
        comparison["out_of_domain"] = {"note": "run evaluate_math --dataset transfer --phase all first"}

    j = diag["judge"]
    comparison["feedback_source"] = {
        "verifier_RLVR": {
            "coverage": "outcome only; blind to reasoning-only corruption (S_reason better="
                        f"{diag['verifier']['S_reason']}, all-tie by design)",
            "noise": "none (deterministic exact match)",
            "exploitability": {
                "gold_distractor_wrong_preference": diag["verifier"]["pairings"]["gold_distractor"]["wrong"],
                "note": "parser must score the discarded gold number 0; wrong-pref should be 0.0",
            },
            "inference_cost_model_calls": 0,
        },
        "judge_RLAIF": {
            "coverage": f"reasoning + style; S_reason better={j['S_reason']} (> verifier's "
                        f"{diag['verifier']['S_reason']})",
            "noise": {
                "position_disagreement_rate_tested": j.get("position_disagreement_rate_tested"),
                "position_tested_n": j.get("position_tested_n"),
            },
            "exploitability": {
                "filler_wrong_preference": diag["judge"]["pairings"]["filler_control"]["wrong"],
                "gold_distractor_wrong_preference": diag["judge"]["pairings"]["gold_distractor"]["wrong"],
                "gold_distractor_caveat": j.get("caveats", {}).get("gold_distractor"),
            },
            "inference_cost": {
                "diagnostic_compare_calls": (diag_cost or {}).get("n_compare_calls_total"),
                "diagnostic_compare_calls_by_pairing": (diag_cost or {}).get("n_compare_calls_by_pairing"),
                "diagnostic_elapsed_s_last_run": (diag_cost or {}).get("elapsed_s_last_run"),
                "indomain_compare_calls": {p: gsm["policies"][p].get("pairwise", {}).get("n_compare_calls")
                                           for p in ("rlvr", "rlaif")},
                "indomain_elapsed_s_last_run": {p: gsm["policies"][p].get("pairwise", {}).get("elapsed_s_last_run")
                                                for p in ("rlvr", "rlaif")},
            },
        },
    }

    save_json(outdir / "task5_comparison.json", comparison)
    print(f"wrote {outdir / 'task5_comparison.json'}")
    print("in-domain (verifier exact_acc | judge win-rate-vs-sft | V-J agreement[TIE=disagree]):")
    for p in POLICIES:
        e = indomain[p]
        jr = e.get("judge", {}).get("win_rate_vs_sft", "-")
        vj = e.get("verifier_judge_agreement", {}).get("agreement_tie_is_disagree", "-")
        print(f"  {p:5s} acc={e['verifier']['exact_acc']:.3f}  win_vs_sft={jr}  vj_agree={vj}")
    tr = comparison["verifier_tie_region"]
    print(f"RQ1 verifier-tie region: S_reason verifier={tr['S_reason_verifier']} judge={tr['S_reason_judge']}")


if __name__ == "__main__":
    main()
