
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
