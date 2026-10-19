
from __future__ import annotations

import argparse
import json

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json


def _source_indexes(path):
    p = repo_path(path)
    if not p.exists():
        return None, f"MISSING {p}"
    rows = read_jsonl(p)
    if not rows:
        return set(), "empty"
    key = "source_index" if "source_index" in rows[0] else None
    if key is None:
        return set(), f"no source_index field (keys={sorted(rows[0])})"
    return {r[key] for r in rows}, f"n={len(rows)} key=source_index"


def _scan_adapter_metadata(cfg):
    found = {}
    for name in ("rlvr_policy", "rlaif_policy"):
        d = repo_path(cfg["paths"][name])
        if not d.exists():
            found[name] = f"MISSING {d}"
            continue
        hits = {}
        for jf in d.glob("*.json"):
            try:
                obj = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, list) and len(v) > 5 and ("index" in k.lower() or "prompt" in k.lower()):
                        hits[f"{jf.name}:{k}"] = len(v)
        found[name] = hits or f"no index-list metadata in {d} (checked *.json)"
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    train_idx, train_note = _source_indexes(cfg["paths"]["gsm_train"])
    eval_idx, eval_note = _source_indexes(cfg["paths"]["gsm_eval"])
    diag_ids = None
    dp = repo_path(cfg["paths"]["task5_diagnostics"])
    if dp.exists():
        diag_ids = {r["problem_id"] for r in read_jsonl(dp)}

    result = {
        "gsm_train": train_note, "gsm_eval": eval_note,
        "adapter_metadata": _scan_adapter_metadata(cfg),
    }
    if train_idx is not None and eval_idx is not None:
        inter = train_idx & eval_idx
        result["train_eval_overlap"] = {
            "n_train": len(train_idx), "n_eval": len(eval_idx), "n_overlap": len(inter),
            "overlap_fraction_of_eval": (len(inter) / len(eval_idx)) if eval_idx else None,
            "overlap_ids_all": sorted(inter),
            "examples": sorted(map(str, inter))[:20],
            "verdict": ("LEAKAGE: in-domain accuracy is partly train accuracy" if inter
                        else "clean: training prompts disjoint from gsm8k_eval"),
        }
    if train_idx is not None and diag_ids is not None:
        inter_d = {str(x) for x in train_idx} & {str(x) for x in diag_ids}
        result["train_diagnostic_overlap"] = {"n_overlap": len(inter_d), "examples": sorted(inter_d)[:20]}

    outdir = repo_path(cfg["results_dir"]) / "task5_feedback"
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "train_eval_overlap.json", result)
    print(json.dumps(result, indent=2)[:3000])
    print(f"\nwrote {outdir / 'train_eval_overlap.json'}")


if __name__ == "__main__":
    main()
