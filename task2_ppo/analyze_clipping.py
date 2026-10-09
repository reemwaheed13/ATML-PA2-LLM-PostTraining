from __future__ import annotations

import argparse

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import save_json
from common.models import load_policy, load_tokenizer
from common.precision import token_logprobs
from task2_ppo.continue_train import KL_CONVENTION, fork_name, run_fork
from task2_ppo.ppo import compute_gae


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
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


# ----------------------------- field detection ------------------------------
_ADV_KEYS = ("advantages", "advantage", "gae", "gae_advantages")
_REWARD_KEYS = ("rewards", "reward", "token_rewards")
_VALUE_KEYS = ("values", "value", "vpred", "token_values")
_SEQ_KEYS = ("sequences", "input_ids")
_PW_KEYS = ("prompt_width", "prompt_len", "prompt_length", "query_len")
_RESP_ID_KEYS = ("response_ids", "response_tokens", "responses_ids")
_PROMPT_ID_KEYS = ("prompt_ids", "prompt_input_ids", "query_ids")
_MASK_KEYS = ("response_mask", "mask", "action_mask")


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


# ----------------------------- prompt resolution ----------------------------
# The cached rollout rows index into a prompt pool by source_index, but the staff cache
# does not record WHICH pool (train vs eval) nor guarantee that source_index is the pool's
# own id field -- it may be a positional index, or the pool may expose a differently named
# id. Resolution is therefore adaptive: try the configured (train) pool, then eval, each by
# every id-like field the pool actually exposes, then by positional index. The winning
# strategy is logged into clip_cached.json so the report can state how prompts were matched.
_INLINE_PROMPT_KEYS = ("prompt", "messages", "prompt_messages", "query", "question", "prompt_text")
_ID_FIELD_CANDIDATES = ("source_index", "id", "prompt_id", "idx", "index", "uid", "example_id", "qid", "sample_index")


def _canon_id(v):
    """Canonicalize an id/index so int 124, float 124.0 and str '124' all compare equal."""
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
    """Return (messages, key) if the row carries its prompt text inline, else (None, None)."""
    for k in _INLINE_PROMPT_KEYS:
        v = row.get(k)
        if isinstance(v, list) and v:
            return v, k
        if isinstance(v, str) and v.strip():
            return [{"role": "user", "content": v}], k
    return None, None


def _needs_prompt(row):
    """True iff _reconstruct would fall through to the text path (no cached token ids),
    i.e. the only branch that needs a resolved prompt."""
    if _first_present(row, _SEQ_KEYS) and _first_present(row, _PW_KEYS):
        return False
    if _first_present(row, _RESP_ID_KEYS) and _first_present(row, _PROMPT_ID_KEYS):
        return False
    return True


def _pool_lookup_maps(pool_rows):
    """For one pool: {strategy_suffix: {canon_id: messages}} -- one map per scalar id-like
    field present on ALL rows, plus a positional map. Keys are canonicalized for matching."""
    maps = {}
    for key in _ID_FIELD_CANDIDATES:
        if all(isinstance(r, dict) and key in r for r in pool_rows):
            maps[f"field:{key}"] = {_canon_id(r[key]): prompt_messages(r) for r in pool_rows}
    maps["positional"] = {i: prompt_messages(r) for i, r in enumerate(pool_rows)}
    return maps


def _resolution_diagnostics(cache_rows, pool_specs, needed, chosen_name):
    """Human-readable dump carrying everything needed to debug a resolution failure:
    which pools were loaded + their sizes, the cache row key set, the first source_index
    values in the cache, and the id field names/values the pools actually expose."""
    bar = "!" * 78
    first_src = [cache_rows[i].get("source_index") for i in range(min(8, len(cache_rows)))]
    lines = [
        bar,
        "PROMPT RESOLUTION FAILED (Task 2 cached clipping study).",
        f"chosen pool strategy: {chosen_name!r} (None => no strategy covered every needed id)",
        f"cache rows: {len(cache_rows)}; rows needing a pool (no token ids, no inline prompt): {len(needed)}",
        f"first source_index values in cache: {first_src}",
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
    lines.append("id field above, then positional indexing. Inline prompt on the cache row wins first.")
    lines.append(bar)
    return "\n".join(lines)


def _build_prompt_resolver(cfg, cache_rows):
    """Return (resolve_fn, pool_strategy). resolve_fn(row) -> (messages, strategy_str).

    Picks ONE pool strategy that resolves every row that needs a pool, preferring the
    configured train pool, its source_index field first. Raises a rich diagnostic early
    (before the policy is loaded) when no strategy covers the needed ids."""
    # Which cache rows actually require a pooled prompt (text reconstruction path, no inline)?
    needed = set()
    for row in cache_rows:
        if _needs_prompt(row) and _inline_prompt(row)[0] is None:
            needed.add(_canon_id(row.get("source_index")))

    # Nothing to resolve from a pool (all rows carry token ids or inline prompts): skip pool IO.
    if not needed:
        def resolve_inline_only(row):
            msgs, key = _inline_prompt(row)
            if msgs is not None:
                return msgs, f"inline:{key}"
            raise ValueError("internal: row unexpectedly needs a pooled prompt")
        return resolve_inline_only, None

    # Load the configured pool first, then the other one.
    pool_specs = []
    for label, key in (("train", "rl_prompt_train"), ("eval", "rl_prompt_eval")):
        path = cfg["paths"][key]
        try:
            rows = read_jsonl(path)
        except FileNotFoundError:
            rows = None
        pool_specs.append((label, path, rows))

    # Ordered candidate strategies: per pool, each id field (source_index first), then positional.
    candidates = []  # (name, {canon_id: messages})
    for label, _path, rows in pool_specs:
        if not rows:
            continue
        maps = _pool_lookup_maps(rows)
        ordered = [f"field:{k}" for k in _ID_FIELD_CANDIDATES if f"field:{k}" in maps] + ["positional"]
        for suf in ordered:
            candidates.append((f"{label}:{suf}", maps[suf]))

    chosen_name, chosen_map = None, None
    for name, m in candidates:
        if needed.issubset(m.keys()):
            chosen_name, chosen_map = name, m
            break

    if chosen_map is None:
        raise ValueError(_resolution_diagnostics(cache_rows, pool_specs, needed, chosen_name))

    def resolve(row):
        msgs, key = _inline_prompt(row)
        if msgs is not None:
            return msgs, f"inline:{key}"
        cid = _canon_id(row.get("source_index"))
        if cid in chosen_map:
            return chosen_map[cid], chosen_name
        # Should not happen (chosen_map covers all needed), but fail loudly if it does.
        raise ValueError(_resolution_diagnostics(cache_rows, pool_specs, {cid}, chosen_name))

    return resolve, chosen_name


def _reconstruct(row, tokenizer, resolve_prompt):
    """Return (sequence[1,L], attn[1,L], prompt_width, response_ids[1,G], source_str).

    Prefers exact cached token ids; falls back to re-encoding prompt+response text (which
    may differ from the rollout tokenization by a token, handled by min-length alignment).
    Batch size 1 per row, so no padding is involved.
    """
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


def _row_mask(row, n):
    mk = _first_present(row, _MASK_KEYS)
    if mk:
        m = _as1d(row[mk])[:n]
        if m.numel() == n:
            return m, f"cached:{mk}"
    return torch.ones(n), "derived:ones(len(old_logprobs))"


def _row_advantages(row, mask, n, gamma, lam):
    """Return (advantages[n] or None, source_str). Per the resolution rules:
    per-token advantage field > compute_gae(rewards, values) > unavailable."""
    adv_k = _first_present(row, _ADV_KEYS)
    if adv_k:
        a = _as1d(row[adv_k])[:n]
        if a.numel() == n:
            return a, f"field:{adv_k}"
    rew_k = _first_present(row, _REWARD_KEYS)
    val_k = _first_present(row, _VALUE_KEYS)
    if rew_k and val_k:
        values = _as1d(row[val_k])
        if values.numel() == n:
            rew_raw = _as1d(row[rew_k])
            if rew_raw.numel() == n:
                rewards = rew_raw
            else:  # scalar terminal reward -> place at last valid response token
                rewards = torch.zeros(n)
                last = int(mask.nonzero().max()) if mask.sum() > 0 else n - 1
                rewards[last] = float(rew_raw.flatten()[0])
            adv, _ = compute_gae(rewards.unsqueeze(0), values.unsqueeze(0), mask.unsqueeze(0), gamma, lam)
            return adv.squeeze(0), f"gae({rew_k},{val_k})"
    return None, "unavailable"


def cached_clip_study(cfg, out_path):
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    eps_values = [float(e) for e in cfg["clip_values"]]
    gamma, lam = float(cfg["gamma"]), float(cfg["gae_lambda"])

    # Resolve prompts BEFORE loading the policy: a cache/pool mismatch should fail fast and
    # cheap (and, with main() reordered, never after the GPU forks already ran).
    resolve_prompt, pool_strategy = _build_prompt_resolver(cfg, rows)

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=False)
    device = next(policy.parameters()).device

    ratios_all, adv_all, mask_all = [], [], []
    adv_sources, mask_sources, recon_sources = set(), set(), set()
    adv_available = True

    for row in rows:
        seq, attn, pw, resp, recon_src = _reconstruct(row, tokenizer, resolve_prompt)
        recon_sources.add(recon_src)
        with torch.no_grad():
            new_logp = token_logprobs(policy, seq.to(device), attn.to(device), pw, resp.to(device), cfg)[0]
        new_logp = new_logp.squeeze(0).detach().cpu().float()
        old_logp = _as1d(row["old_logprobs"])
        n = int(min(new_logp.numel(), old_logp.numel()))
        ratio = torch.exp(new_logp[:n] - old_logp[:n])
        mask, mask_src = _row_mask(row, n)
        mask_sources.add(mask_src)

        adv, adv_src = _row_advantages(row, mask, n, gamma, lam)
        adv_sources.add(adv_src)
        if adv is None:
            adv_available = False

        m = mask.bool()
        ratios_all.append(ratio[m])
        mask_all.append(mask[m])
        adv_all.append(adv[m] if adv is not None else None)

    ratios = torch.cat(ratios_all)
    n_tokens = int(ratios.numel())

    # Clip fraction only needs ratios -> compute and report unconditionally.
    per_eps = {}
    for eps in eps_values:
        clipped = (ratios < (1.0 - eps)) | (ratios > (1.0 + eps))
        per_eps[f"{eps:.2f}"] = {"clip_fraction": float(clipped.float().mean())}

    result = {
        "study": "cached_clip",
        "n_rows": len(rows),
        "n_valid_tokens": n_tokens,
        "advantage_source": sorted(adv_sources),
        "mask_source": sorted(mask_sources),
        "reconstruction_source": sorted(recon_sources),
        "prompt_resolution": {"pool_strategy": pool_strategy},
        "alignment": "per row: min(len(new_logp), len(old_logprobs)); ratio=exp(new-old)",
        "clip_epsilons": eps_values,
        "kl_convention": KL_CONVENTION,
        "per_eps": per_eps,
    }

    if adv_available:
        advs = torch.cat([a for a in adv_all])
        for eps in eps_values:
            e = per_eps[f"{eps:.2f}"]
            # Affected = clipping is BINDING on the min() objective: A>0 & rho>1+eps, or A<0 & rho<1-eps.
            binding = ((advs > 0) & (ratios > (1.0 + eps))) | ((advs < 0) & (ratios < (1.0 - eps)))
            e["affected_token_fraction"] = float(binding.float().mean())
            surr1 = ratios * advs
            surr2 = ratios.clamp(1.0 - eps, 1.0 + eps) * advs
            e["clipped_surrogate_mean"] = float(torch.minimum(surr1, surr2).mean())
        result["affected_token_available"] = True
        save_json(out_path, result)
        print(f"wrote {out_path} (clip + affected fractions, adv source={sorted(adv_sources)})")
        return result

    # Advantage path unavailable: write the partial output (clip fractions only) and return
    # NON-FATALLY so the matched eps forks still train. Loud, Required-Evidence-aware message.
    result["affected_token_available"] = False
    save_json(out_path, result)
    keys_found = sorted(rows[0].keys())
    bar = "!" * 78
    print(
        "\n" + bar + "\n"
        "MISSING REQUIRED EVIDENCE (Task 2, clipping study): the affected-token fraction\n"
        "could NOT be computed from this cache. It is a Required Evidence item, not optional.\n"
        f"clip_cached.json was written with clip_fraction ONLY ({out_path}).\n"
        "To satisfy it, each cached row must provide, aligned to len(old_logprobs):\n"
        "  - a per-token advantage field: one of ['advantages','advantage','gae','gae_advantages']\n"
        "    OR\n"
        "  - rewards: one of ['rewards','reward','token_rewards']  AND\n"
        "    per-token values: one of ['values','value','vpred','token_values']\n"
        f"Keys actually found on row 0: {keys_found}\n"
        + bar + "\n"
    )
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

    # (A) matched short forks FIRST -- this is the expensive GPU work. It must not be
    # gated on the cached study: a cache/pool problem used to crash before the forks ran
    # and cost all three. The (eps=0.20, kl=0.10) fork is shared with ablate_kl and runs
    # once (run_fork skips if already trained).
    if not args.skip_forks:
        kl = float(cfg["kl_beta"])
        for eps in [float(e) for e in cfg["clip_values"]]:
            print(f"=== clip fork {fork_name(eps, kl)} (eps={eps}, kl={kl}, {cfg['fork_updates']} updates) ===")
            run_fork(args.config, eps, kl, resume=args.resume, force=args.force)

    # (B) cached-rollout geometric study LAST -- cheap and cannot block the forks above.
    cached_clip_study(cfg, results_dir / "clip_cached.json")


if __name__ == "__main__":
    main()
