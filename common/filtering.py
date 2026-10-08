"""The single data-loading entry point every DPO script calls.

read -> deterministic prompt-length filter (BEFORE shuffling) -> optionally record
the filter report (counts, fractions, per-stratum breakdown, and the exact
retained/dropped example indices the handout requires) to
`<results_dir>/<run_name>_filter.json`.

Putting the filter here guarantees it is identical across the standard run, all
three beta forks, the length-balanced run, and both eval sets: every caller passes
the one config value `max_sequence_length`, and the filter is on prompt length only
so it can never un-balance a length-stratified dataset.
"""

from __future__ import annotations

from pathlib import Path

from common.data import filter_overlength_prompts, read_jsonl, repo_path
from common.logging_utils import load_json, save_json


def load_filtered_rows(path, tokenizer, max_length, results_dir=None, run_name=None, tag=None):
    rows = read_jsonl(path)
    kept, report = filter_overlength_prompts(rows, tokenizer, int(max_length))
    if results_dir and run_name:
        tag = tag or Path(str(path)).stem
        out = repo_path(results_dir) / f"{run_name}_filter.json"
        doc = load_json(out) if out.exists() else {"datasets": {}}
        doc.setdefault("datasets", {})[tag] = {"path": str(path), **report}
        save_json(out, doc)
    return kept, report
