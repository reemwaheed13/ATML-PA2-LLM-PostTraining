"""Dump raw Task 1 qualitative material. Pure read/join; no model, no selection.

Joins the held-out generations (reward, length, response) with the teacher-forced
pairs (preference margin m_theta) on example idx, attaches z-scored reward and
margin and their difference so reward-vs-preference disagreement is sortable, and
emits EVERY joined record (no thresholding, no cherry-picking). Also passes through
the raw word-limit generations for instruction-compliance inspection.

The author selects and interprets; this script writes JSON only.

Run:  python -m scripts.dump_task1_qualitative --config configs/dpo.yaml --name standard
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
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])

    gens = {r["idx"]: r for r in read_jsonl(results_dir / f"{args.name}_generations.jsonl")}
    pairs = {r["idx"]: r for r in read_jsonl(results_dir / f"{args.name}_pairs.jsonl")}
    idxs = sorted(set(gens) & set(pairs))

    rewards = [gens[i]["reward"] for i in idxs]
    margins = [pairs[i]["m_theta"] for i in idxs]
    rz, mz = _z(rewards), _z(margins)

    joined = []
    for k, i in enumerate(idxs):
        joined.append({
            "idx": i,
            "prompt": gens[i].get("prompt"),
            "response": gens[i].get("response"),
            "reward": rewards[k],
            "reward_z": rz[k],
            "m_theta": margins[k],
            "margin_z": mz[k],
            "disagreement": rz[k] - mz[k],  # +: high reward, low margin; -: opposite
            "length": gens[i].get("length"),
            "chosen_text": pairs[i].get("chosen_text"),
            "rejected_text": pairs[i].get("rejected_text"),
        })
    # Sort by magnitude of disagreement for convenience; all records are retained.
    joined.sort(key=lambda r: abs(r["disagreement"]), reverse=True)

    wl_path = results_dir / f"{args.name}_wordlimit_gen.jsonl"
    word_limit = read_jsonl(wl_path) if wl_path.exists() else []

    out = results_dir / f"{args.name}_qualitative.json"
    save_json(out, {
        "name": args.name,
        "n_joined": len(joined),
        "reward_vs_preference": joined,
        "word_limit": word_limit,
    })
    print(f"wrote {out}  ({len(joined)} joined held-out cases, {len(word_limit)} word-limit gens)")


if __name__ == "__main__":
    main()
