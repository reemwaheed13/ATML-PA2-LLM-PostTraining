from __future__ import annotations

import argparse

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import save_json
from common.models import load_policy, load_tokenizer
from common.precision import token_logprobs
from task2_ppo.continue_train import KL_CONVENTION, fork_name, run_fork
from task2_ppo.ppo import compute_gae, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


_TERMINAL_REWARD_KEYS = ("effective_terminal_reward", "terminal_reward", "raw_terminal_reward")
_SEQ_KEYS = ("sequences", "input_ids")
_PW_KEYS = ("prompt_width", "prompt_len", "prompt_length", "query_len")
_RESP_ID_KEYS = ("response_ids", "responses_ids")
_PROMPT_ID_KEYS = ("prompt_ids", "prompt_input_ids", "query_ids")


def _first_present(row, keys):
    for k in keys:
        if k in row and row[k] is not None:
            return k
    return None


def _as1d(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().flatten().float()
    return torch.tensor(np.asarray(x, dtype=float).flatten(), dtype=torch.float32)


def _as1d_long(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().flatten().long()
    return torch.tensor(np.asarray(x).flatten(), dtype=torch.long)


_INLINE_PROMPT_KEYS = ("prompt", "messages", "prompt_messages", "query", "question", "prompt_text")
_ID_FIELD_CANDIDATES = ("source_index", "id", "prompt_id", "idx", "index", "uid", "example_id", "qid", "sample_index")
_CACHE_ID_FIELDS = ("source_index", "prompt_id")


def _canon_id(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float) and float(v).is_integer():
        return int(v)
    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
        return int(v.strip())
    return v


def _inline_prompt(row):
    for k in _INLINE_PROMPT_KEYS:
        v = row.get(k)
        if isinstance(v, list) and v:
            return v, k
        if isinstance(v, str) and v.strip():
            return [{"role": "user", "content": v}], k
    return None, None


def _needs_prompt(row):
    if _first_present(row, _SEQ_KEYS) and _first_present(row, _PW_KEYS):
        return False
    if _first_present(row, _RESP_ID_KEYS) and _first_present(row, _PROMPT_ID_KEYS):
        return False
    return True


def _pool_lookup_maps(pool_rows):
    maps = {}
    for key in _ID_FIELD_CANDIDATES:
        if all(isinstance(r, dict) and key in r for r in pool_rows):
            maps[f"field:{key}"] = {_canon_id(r[key]): prompt_messages(r) for r in pool_rows}
    maps["positional"] = {i: prompt_messages(r) for i, r in enumerate(pool_rows)}
    return maps


def _resolution_diagnostics(cache_rows, pool_specs, needed, chosen_name):
    bar = "!" * 78
    first_src = [cache_rows[i].get("source_index") for i in range(min(8, len(cache_rows)))]
    first_pid = [cache_rows[i].get("prompt_id") for i in range(min(8, len(cache_rows)))]
    lines = [
        bar,
        "PROMPT RESOLUTION FAILED (Task 2 cached clipping study).",
        f"chosen pool strategy: {chosen_name!r} (None => no strategy covered every needed id)",
        f"cache rows: {len(cache_rows)}; rows needing a pool (no token ids, no inline prompt): {len(needed)}",
        f"first source_index values in cache: {first_src}",
        f"first prompt_id values in cache: {first_pid}",
        f"full key set of cache row 0: {sorted(cache_rows[0].keys())}",
        f"canonical ids needed but unresolved: {sorted(list(needed))[:12]}"
        + (" ..." if len(needed) > 12 else ""),
    ]
    for label, path, rows in pool_specs:
        if rows is None:
            lines.append(f"pool[{label}] {path}: MISSING (file not found)")
            continue
        idf = [k for k in _ID_FIELD_CANDIDATES if all(isinstance(r, dict) and k in r for r in rows)]
        samples = {k: [rows[i].get(k) for i in range(min(5, len(rows)))] for k in idf}
        lines.append(f"pool[{label}] {path}: {len(rows)} rows; positional ids 0..{len(rows) - 1}")
        lines.append(f"    id-like fields present on all rows: {idf}")
        lines.append(f"    sample id values: {samples}")
        lines.append(f"    pool row 0 keys: {sorted(rows[0].keys())}")
    lines.append("strategies tried (in order): configured(train) then eval pool, each by every")
    lines.append("id field above, then positional indexing. The cache row is keyed by source_index")
    lines.append(f"first then prompt_id ({list(_CACHE_ID_FIELDS)}). Inline prompt on the row wins first.")
    lines.append(bar)
    return "\n".join(lines)


def _build_prompt_resolver(cfg, cache_rows):
    needed_rows = [r for r in cache_rows if _needs_prompt(r) and _inline_prompt(r)[0] is None]

    if not needed_rows:
        def resolve_inline_only(row):
            msgs, key = _inline_prompt(row)
            if msgs is not None:
                return msgs, f"inline:{key}"
            raise ValueError("internal: row unexpectedly needs a pooled prompt")
        return resolve_inline_only, None

    pool_specs = []
    for label, key in (("train", "rl_prompt_train"), ("eval", "rl_prompt_eval")):
        path = cfg["paths"][key]
        try:
            rows = read_jsonl(path)
        except FileNotFoundError:
            rows = None
        pool_specs.append((label, path, rows))

    candidates = []
    for label, _path, rows in pool_specs:
        if not rows:
            continue
        maps = _pool_lookup_maps(rows)
        ordered = [f"field:{k}" for k in _ID_FIELD_CANDIDATES if f"field:{k}" in maps] + ["positional"]
        for suf in ordered:
            candidates.append((f"{label}:{suf}", maps[suf]))

    cache_fields = [f for f in _CACHE_ID_FIELDS if all(f in r for r in needed_rows)] or ["source_index"]

    chosen = None
    for cache_field in cache_fields:
        needed = {_canon_id(r.get(cache_field)) for r in needed_rows}
        for name, m in candidates:
            if needed.issubset(m.keys()):
                chosen = (name, cache_field, m)
                break
        if chosen:
            break

    if chosen is None:
        needed = {_canon_id(r.get(cache_fields[0])) for r in needed_rows}
        raise ValueError(_resolution_diagnostics(cache_rows, pool_specs, needed, None))

    pool_name, cache_field, chosen_map = chosen
    strategy = f"{pool_name} via cache.{cache_field}"

    def resolve(row):
        msgs, key = _inline_prompt(row)
        if msgs is not None:
            return msgs, f"inline:{key}"
        cid = _canon_id(row.get(cache_field))
        if cid in chosen_map:
            return chosen_map[cid], strategy
        raise ValueError(_resolution_diagnostics(cache_rows, pool_specs, {cid}, strategy))

    return resolve, strategy


def _reconstruct(row, tokenizer, resolve_prompt):
    seq_k = _first_present(row, _SEQ_KEYS)
    pw_k = _first_present(row, _PW_KEYS)
    resp_k = _first_present(row, _RESP_ID_KEYS)
    prompt_id_k = _first_present(row, _PROMPT_ID_KEYS)

    if seq_k and pw_k:
        seq = _as1d_long(row[seq_k])
        pw = int(row[pw_k])
        resp = _as1d_long(row[resp_k]) if resp_k else seq[pw:]
        source = f"cached:{seq_k}+{pw_k}"
    elif resp_k and prompt_id_k:
        pids = _as1d_long(row[prompt_id_k])
        resp = _as1d_long(row[resp_k])
        seq = torch.cat([pids, resp])
        pw = int(pids.numel())
        source = f"cached:{prompt_id_k}+{resp_k}"
    else:
        msgs, pstrat = resolve_prompt(row)
        pids = torch.tensor(
            tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True),
            dtype=torch.long,
        )
        if resp_k:
            resp = _as1d_long(row[resp_k])
        else:
            resp = torch.tensor(tokenizer(str(row["response"]), add_special_tokens=False)["input_ids"], dtype=torch.long)
        seq = torch.cat([pids, resp])
        pw = int(pids.numel())
        source = f"text:{'cached_resp_ids' if resp_k else 'retokenized'}|prompt={pstrat}"

    attn = torch.ones_like(seq)
    return seq.unsqueeze(0), attn.unsqueeze(0), pw, resp.unsqueeze(0), source


def _terminal_scalar(row, missing_eos_penalty):
    term_k = next((k for k in _TERMINAL_REWARD_KEYS if k in row), None)
    if term_k is None:
        raise ValueError(
            f"no terminal reward field {list(_TERMINAL_REWARD_KEYS)} in cache row "
            f"(keys={sorted(row.keys())}); cannot build the per-token reward."
        )
    value = float(_as1d(row[term_k]).flatten()[0])
    penalty_mode = "prefolded"
    if term_k.startswith("raw") and not bool(row.get("terminated_with_eos", True)):
        value -= float(missing_eos_penalty)
        penalty_mode = "reapplied_from_raw"
    return value, term_k, penalty_mode


def _verify_terminal_fields(rows, missing_eos_penalty):
    if not all(("effective_terminal_reward" in r and "raw_terminal_reward" in r) for r in rows):
        return {"checked": False, "reason": "effective_/raw_terminal_reward not both present on all rows"}
    max_abs_residual = 0.0
    rows_penalized = 0
    for r in rows:
        eff = float(_as1d(r["effective_terminal_reward"]).flatten()[0])
        raw = float(_as1d(r["raw_terminal_reward"]).flatten()[0])
        eos = bool(r.get("terminated_with_eos", True))
        if not eos:
            rows_penalized += 1
        expected = raw - (0.0 if eos else float(missing_eos_penalty))
        max_abs_residual = max(max_abs_residual, abs(eff - expected))
    return {
        "checked": True,
        "model": "effective = raw - missing_eos_penalty * (not terminated_with_eos)",
        "missing_eos_penalty": float(missing_eos_penalty),
        "rows_penalized": rows_penalized,
        "max_abs_residual": max_abs_residual,
        "consistent": max_abs_residual < 1e-2,
    }


def _cached_len(row, key):
    v = row.get(key)
    return int(_as1d(v).numel()) if v is not None else None


def _require_len(named_tensors, T, row):
    report = {name: (tuple(t.shape), int(t.numel())) for name, t in named_tensors}
    bad = [name for name, t in named_tensors if t.dim() != 1 or int(t.numel()) != T]
    if bad:
        raise ValueError(
            f"per-token ratio inputs must all be 1-D length T={T} (row "
            f"source_index={row.get('source_index')!r} prompt_id={row.get('prompt_id')!r}); "
            f"offenders={bad}; (shape, numel) per input = {report}"
        )


def _require_bt(named_tensors, T, row):
    report = {name: tuple(t.shape) for name, t in named_tensors}
    bad = [name for name, t in named_tensors if t.dim() != 2 or tuple(t.shape) != (1, T)]
    if bad:
        raise ValueError(
            f"compute_gae inputs must all be [1, T={T}] (row "
            f"source_index={row.get('source_index')!r} prompt_id={row.get('prompt_id')!r}); "
            f"offenders={bad}; shapes = {report}"
        )


def cached_clip_study(cfg, out_path):
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    eps_values = [float(e) for e in cfg["clip_values"]]
    gamma, lam = float(cfg["gamma"]), float(cfg["gae_lambda"])
    kl_beta = float(cfg["kl_beta"])
    missing_eos_penalty = float(cfg["missing_eos_penalty"])

    resolve_prompt, pool_strategy = _build_prompt_resolver(cfg, rows)

    tokenizer = load_tokenizer(cfg["base_model"])

    new_adapter = cfg["output"]
    if not repo_path(new_adapter).exists():
        raise FileNotFoundError(
            f"standard continuation adapter not found at {repo_path(new_adapter)}; it is the "
            f"'new' policy for the ratio. Run the standard PPO continuation before this study."
        )

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(
            "tokenizer has no eos_token_id; cannot reconstruct the terminal EOS token that the "
            "cached per-token arrays include."
        )

    prepared, retok_mismatches = [], []
    verify = {
        "n_rows": len(rows), "n_terminated_with_eos": 0, "n_clipped_at_max": 0,
        "raw_match_and_not_eos": 0, "raw_match_and_eos": 0,
        "raw_mismatch_and_eos": 0, "raw_mismatch_and_not_eos": 0,
        "eos_vs_clipped": {"eos&cap": 0, "eos&nocap": 0, "noeos&cap": 0, "noeos&nocap": 0},
        "hypothesis": "raw_match == (not terminated_with_eos)",
        "hypothesis_violations": [],
    }
    for i, row in enumerate(rows):
        old_len = int(_as1d(row["old_logprobs"]).numel())
        ref_len = int(_as1d(row["ref_logprobs"]).numel())
        val_len = int(_as1d(row["values"]).numel())
        if not (old_len == ref_len == val_len):
            raise ValueError(
                f"row {i}: cached per-token lengths disagree -- old_logprobs={old_len} "
                f"ref_logprobs={ref_len} values={val_len} (source_index={row.get('source_index')!r})"
            )
        T = old_len
        rt_count = int(_as1d(row["response_tokens"]).flatten()[0])
        if rt_count != T:
            raise ValueError(
                f"row {i}: response_tokens count {rt_count} != len(old_logprobs) {T} "
                f"(source_index={row.get('source_index')!r})"
            )

        eos = bool(row.get("terminated_with_eos", False))
        cam = bool(row.get("clipped_at_max", False))
        seq, attn, pw, resp, recon_src = _reconstruct(row, tokenizer, resolve_prompt)
        base_len = int(resp.shape[1])
        raw_match = (base_len == T)

        verify["n_terminated_with_eos"] += int(eos)
        verify["n_clipped_at_max"] += int(cam)
        verify[("raw_match" if raw_match else "raw_mismatch") + ("_and_eos" if eos else "_and_not_eos")] += 1
        verify["eos_vs_clipped"][("eos" if eos else "noeos") + ("&cap" if cam else "&nocap")] += 1
        if raw_match != (not eos):
            verify["hypothesis_violations"].append(
                {"row": i, "terminated_with_eos": eos, "raw_match": raw_match, "base_len": base_len, "cached_T": T}
            )
        print(f"[clip-study] row {i:2d}: terminated_with_eos={eos!s:>5} clipped_at_max={cam!s:>5} "
              f"retok_len={base_len} cached_T={T} raw_match={raw_match!s:>5}")

        if eos:
            eos_col = torch.tensor([[eos_id]], dtype=resp.dtype)
            resp = torch.cat([resp, eos_col], dim=1)
            seq = torch.cat([seq, eos_col.to(seq.dtype)], dim=1)
            attn = torch.cat([attn, torch.ones((1, 1), dtype=attn.dtype)], dim=1)

        final_len = int(resp.shape[1])
        if final_len != T:
            retok_mismatches.append(
                {"row": i, "source_index": row.get("source_index"), "terminated_with_eos": eos,
                 "retok_len_no_eos": base_len, "final_len": final_len, "cached_T": T}
            )
        prepared.append((row, T, seq, attn, pw, resp, recon_src))

    print(f"[clip-study] EOS/length verification over {len(rows)} rows: "
          f"terminated_with_eos={verify['n_terminated_with_eos']}, clipped_at_max={verify['n_clipped_at_max']}; "
          f"raw(no EOS): match&not_eos={verify['raw_match_and_not_eos']} match&eos={verify['raw_match_and_eos']} "
          f"mismatch&eos={verify['raw_mismatch_and_eos']} mismatch&not_eos={verify['raw_mismatch_and_not_eos']}; "
          f"eos_vs_clipped={verify['eos_vs_clipped']}; "
          f"hypothesis(raw_match==not eos) violations={len(verify['hypothesis_violations'])}")

    if retok_mismatches:
        raise ValueError(
            f"{len(retok_mismatches)}/{len(rows)} rows still mismatch AFTER EOS alignment (off by "
            f"more than the single terminal EOS). The cache stores response TEXT but no token IDs, "
            f"so new_logp cannot be aligned for these rows. Details: {retok_mismatches}"
        )

    print(f"[clip-study] pre-pass OK after EOS alignment: {len(prepared)} rows, 0 residual "
          f"mismatches, T=len(old_logprobs) in [{min(p[1] for p in prepared)}, {max(p[1] for p in prepared)}]")

    standard = load_policy(cfg, adapter_path=new_adapter, trainable=False)
    standard.eval()
    device = next(standard.parameters()).device

    ratios_all, adv_all = [], []
    recon_sources, term_fields_used, penalty_modes = set(), set(), set()
    resp_counts = []
    printed_row0 = False

    for i, (row, T, seq, attn, pw, resp, recon_src) in enumerate(prepared):
        recon_sources.add(recon_src)
        resp_counts.append(T)

        old_logp = _as1d(row["old_logprobs"])
        ref_logp = _as1d(row["ref_logprobs"])
        values = _as1d(row["values"])
        with torch.no_grad():
            new_logp = _as1d(token_logprobs(standard, seq.to(device), attn.to(device), pw, resp.to(device), cfg)[0])

        mask = torch.ones(T)

        if not printed_row0:
            print(f"[clip-study] row0 T=len(old_logprobs)={T}; 1-D shapes "
                  f"old_logp={tuple(old_logp.shape)} new_logp={tuple(new_logp.shape)} "
                  f"ref_logp={tuple(ref_logp.shape)} values={tuple(values.shape)} "
                  f"(response_tokens count == T, re-tokenized response len == T)")
            printed_row0 = True

        _require_len([("new_logp", new_logp), ("old_logp", old_logp), ("ref_logp", ref_logp),
                      ("values", values), ("mask", mask)], T, row)
        ratio = torch.exp(new_logp - old_logp)

        terminal_scalar, term_k, penalty_mode = _terminal_scalar(row, missing_eos_penalty)
        term_fields_used.add(term_k)
        penalty_modes.add(penalty_mode)
        terminal = torch.tensor([terminal_scalar], dtype=torch.float32)

        old2, ref2, val2, mask2 = (x.reshape(1, T) for x in (old_logp, ref_logp, values, mask))
        rewards = shaped_rewards(terminal, old2, ref2, mask2, kl_beta)
        _require_bt([("rewards", rewards), ("values", val2), ("mask", mask2)], T, row)
        adv, _ = compute_gae(rewards, val2, mask2, gamma, lam)

        ratios_all.append(ratio)
        adv_all.append(_as1d(adv))

    ratios = torch.cat(ratios_all)
    advs = torch.cat(adv_all)
    n_tokens = int(ratios.numel())

    per_eps = {}
    for eps in eps_values:
        clipped = (ratios < (1.0 - eps)) | (ratios > (1.0 + eps))
        binding = ((advs > 0) & (ratios > (1.0 + eps))) | ((advs < 0) & (ratios < (1.0 - eps)))
        surr1 = ratios * advs
        surr2 = ratios.clamp(1.0 - eps, 1.0 + eps) * advs
        per_eps[f"{eps:.2f}"] = {
            "clip_fraction": float(clipped.float().mean()),
            "affected_token_fraction": float(binding.float().mean()),
            "clipped_surrogate_mean": float(torch.minimum(surr1, surr2).mean()),
        }

    cam = [bool(r.get("clipped_at_max")) for r in rows if "clipped_at_max" in r]
    clipped_at_max_fraction = (sum(cam) / len(cam)) if cam else None
    term_check = _verify_terminal_fields(rows, missing_eos_penalty)

    result = {
        "study": "cached_clip",
        "n_rows": len(rows),
        "n_valid_tokens": n_tokens,
        "response_token_count": {"min": min(resp_counts), "max": max(resp_counts), "total": int(sum(resp_counts))},
        "reconstruction_source": sorted(recon_sources),
        "prompt_resolution": {"pool_strategy": pool_strategy},
        "policies": {
            "old_logprobs": "cached per-token (midpoint/rollout policy)",
            "new_logprobs": f"standard continuation forward over re-tokenized response: {new_adapter}",
            "ref_logprobs": "cached per-token (reference)",
            "values": "cached per-token (critic baseline)",
        },
        "mask_source": "derived:ones(T); every response token is valid",
        "cache_per_token_fields": {
            "status": "per-token tensors (length T); used directly",
            "row0_lengths": {
                "old_logprobs": _cached_len(rows[0], "old_logprobs"),
                "ref_logprobs": _cached_len(rows[0], "ref_logprobs"),
                "values": _cached_len(rows[0], "values"),
                "response_tokens_count": int(_as1d(rows[0]["response_tokens"]).flatten()[0]),
            },
            "note": "old_logprobs/ref_logprobs/values are per-token (length T); response_tokens is an "
                    "integer count asserted == T. Only new_logp is recomputed (standard forward).",
        },
        "retokenization": {
            "reason": "cache stores response TEXT, not token IDs; the response is re-tokenized for "
                      "the new_logp forward pass",
            "eos_alignment": "cached per-token arrays include the terminal EOS; eos_token_id is "
                             "appended to the re-tokenized ids iff terminated_with_eos is True",
            "assert": "final (EOS-aligned) length == len(old_logprobs) per row; raises otherwise",
            "rows_with_residual_mismatch": 0,
            "verification": verify,
        },
        "alignment": "per row: len(old_logp)==len(new_logp)==len(ref_logp)==len(values)==T="
                     "len(old_logprobs) enforced (raises on mismatch); ratio=exp(new-old)",
        "clip_epsilons": eps_values,
        "kl_convention": KL_CONVENTION,
        "advantage_reconstruction": {
            "reconstructed": True,
            "route": "gae(shaped_rewards(terminal, old_logp, ref_logp, mask, kl_beta), values); "
                     "old_logp/ref_logp/values are cached per-token, new_logp is a standard-policy "
                     "forward over the re-tokenized response",
            "terminal_reward_field": sorted(term_fields_used),
            "terminal_penalty_mode": sorted(penalty_modes),
            "terminal_field_check": term_check,
            "terminal_placement": "last valid response index (mask.sum()-1) per shaped_rewards; the "
                                  "missing-EOS penalty is already folded into effective_terminal_reward, "
                                  "so clipped_at_max / terminated_with_eos do not move the deposit point.",
            "kl_beta": kl_beta,
            "missing_eos_penalty": missing_eos_penalty,
            "gamma": gamma,
            "gae_lambda": lam,
            "clipped_at_max_fraction": clipped_at_max_fraction,
        },
        "affected_token_available": True,
        "per_eps": per_eps,
    }

    if term_check.get("checked") and not term_check.get("consistent"):
        bar = "!" * 78
        print(
            "\n" + bar + "\n"
            "WARNING (Task 2 clipping study): effective_terminal_reward does NOT match\n"
            "raw_terminal_reward - missing_eos_penalty*(not terminated_with_eos).\n"
            f"max_abs_residual={term_check['max_abs_residual']:.4g}, "
            f"rows_penalized={term_check['rows_penalized']}. Reconstructed advantages use\n"
            f"terminal field(s) {sorted(term_fields_used)}; verify the penalty semantics first.\n"
            + bar + "\n"
        )

    save_json(out_path, result)
    print(f"wrote {out_path} (clip + affected fractions; old=cached midpoint per-token, "
          f"new={new_adapter} forward; {n_tokens} per-token ratios over {len(rows)} rows)")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true", help="retrain forks even if already done")
    ap.add_argument("--skip-forks", action="store_true", help="only run the cached clip study")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_forks:
        kl = float(cfg["kl_beta"])
        for eps in [float(e) for e in cfg["clip_values"]]:
            print(f"=== clip fork {fork_name(eps, kl)} (eps={eps}, kl={kl}, {cfg['fork_updates']} updates) ===")
            run_fork(args.config, eps, kl, resume=args.resume, force=args.force)

    cached_clip_study(cfg, results_dir / "clip_cached.json")


if __name__ == "__main__":
    main()
