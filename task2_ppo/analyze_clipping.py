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


def _resolve_prompt(row, prompt_pool):
    p = row.get("prompt")
    if isinstance(p, list) and p:
        return p
    if isinstance(p, str) and p:
        return [{"role": "user", "content": p}]
    src = row.get("source_index")
    if src in prompt_pool:
        return prompt_pool[src]
    raise ValueError(f"cannot resolve prompt for row source_index={src!r}")


def _reconstruct(row, tokenizer, prompt_pool):
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
        msgs = _resolve_prompt(row, prompt_pool)
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
        source = f"text:{'cached_resp_ids' if resp_k else 'retokenized'}"

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

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=False)
    device = next(policy.parameters()).device
    prompt_pool = {}
    for i, r in enumerate(read_jsonl(cfg["paths"]["rl_prompt_train"])):
        prompt_pool[r.get("source_index", i)] = prompt_messages(r)

    ratios_all, adv_all, mask_all = [], [], []
    adv_sources, mask_sources, recon_sources = set(), set(), set()
    adv_available = True

    for row in rows:
        seq, attn, pw, resp, recon_src = _reconstruct(row, tokenizer, prompt_pool)
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

    # (A) cached-rollout geometric study
    cached_clip_study(cfg, results_dir / "clip_cached.json")

    # (B) matched short forks: fixed kl_beta, sweep epsilon. The (eps=0.20, kl=0.10) fork
    # is shared with ablate_kl and runs once (run_fork skips if already trained).
    if not args.skip_forks:
        kl = float(cfg["kl_beta"])
        for eps in [float(e) for e in cfg["clip_values"]]:
            print(f"=== clip fork {fork_name(eps, kl)} (eps={eps}, kl={kl}, {cfg['fork_updates']} updates) ===")
            run_fork(args.config, eps, kl, resume=args.resume, force=args.force)


if __name__ == "__main__":
    main()
