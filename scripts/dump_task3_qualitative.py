"""Dump raw Task 3 qualitative material. Pure read; no model, no selection.

Reads the held-out generations (reward, length, response) for one GRPO policy, attaches
z-scored reward and z-scored length and their difference so reward-vs-length disagreement is
sortable, and emits EVERY record (no thresholding, no cherry-picking). Carries the source index
so the author can trace each case back to the fixed prompt set.

The author selects and interprets (reward and quality moving together vs disagreeing); this
script writes JSON only. Run it for any policy by passing its --name (standard, norm_grpo,
norm_dr_grpo, ...).

Run:  python -m scripts.dump_task3_qualitative --config configs/grpo.yaml --name standard
"""

from __future__ import annotations

import argparse

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json


def _z(values):
    a = np.array(values, dtype=float)
    mu, sd = a.mean(), a.std()
    sd = sd if sd > 0 else 1.0
    return [float((v - mu) / sd) for v in values]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])

    recs = read_jsonl(results_dir / f"{args.name}_generations.jsonl")
    idxs = list(range(len(recs)))

    rewards = [r["reward"] for r in recs]
    lengths = [r["length"] for r in recs]
    rz, lz = _z(rewards), _z(lengths)

    joined = []
    for k in idxs:
        r = recs[k]
        joined.append({
            "idx": r.get("idx", k),
            "source_index": r.get("source_index"),
            "prompt": r.get("prompt"),
            "response": r.get("response"),
            "reward": rewards[k],
            "reward_z": rz[k],
            "length": lengths[k],
            "length_z": lz[k],
            "disagreement": rz[k] - lz[k],  # +: high reward, short; -: low reward, long
            "entropy": r.get("entropy"),
            "truncated": r.get("truncated"),
            "terminated": r.get("terminated"),
        })
    # Sort by magnitude of disagreement for convenience; all records are retained.
    joined.sort(key=lambda r: abs(r["disagreement"]), reverse=True)

    out = results_dir / f"{args.name}_qualitative.json"
    save_json(out, {
        "name": args.name,
        "n_records": len(joined),
        "reward_vs_length": joined,
    })
    print(f"wrote {out}  ({len(joined)} held-out cases)")


if __name__ == "__main__":
    main()
