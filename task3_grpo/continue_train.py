from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import append_jsonl, load_json, save_json, set_seed, wall_timer
from common.metrics import mean_response_length, sampled_kl
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from common.precision import autocast_context, make_grad_scaler, token_logprobs
from task3_grpo.grpo import (
    group_relative_advantages,
    grpo_policy_loss,
    mask_truncated_sequences,
)

# Same KL averaging convention string Task 1/2 record, kept identical so Task 1/2/3 KL numbers
# are comparable. sampled_kl == masked_mean, i.e. sum_t(logp_policy - logp_ref) over response
# tokens / total response tokens. This is the REPORTED KL. The k3 penalty actually added to the
# GRPO objective (exp(d)-d-1, d=ref-policy) is logged separately as kl_penalty_k3.
KL_CONVENTION = "sampled_per_token_mean(sum_tokens/sum_response_tokens)"
PRECISION = "base_fp16+lora_fp32_peft_native+autocast_fp16+gradscaler"


def _report_trainable_dtypes(model, label):
    """One-line trainable-parameter dtype histogram at load time -- a cheap sanity check.
    The hard guard is the fp32 assertion inside common.models load_policy."""
    from collections import Counter

    counts = Counter()
    n = 0
    for _, p in model.named_parameters():
        if p.requires_grad:
            n += 1
            counts[str(p.dtype)] += 1
    fp16 = counts.get("torch.float16", 0)
    fp32 = counts.get("torch.float32", 0)
    print(f"[dtype-report] {label}: {n} trainable | fp16={fp16} fp32={fp32} "
          f"other={n - fp16 - fp32} | dtypes={dict(counts)}")


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    _report_trainable_dtypes(policy, "policy")

    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


# ---------------------------------------------------------------------------
# Atomic checkpoint (policy + optimizer + scaler + RNG + progress). Critic-free,
# so this is the single-model version of task2_ppo/continue_train._save_state: a
# dying Colab session resumes exactly from the last saved update.
# ---------------------------------------------------------------------------
def _save_state(path: Path, policy, optimizer, scaler, update):
    state = {
        "policy_trainable": {n: p.detach().cpu() for n, p in policy.named_parameters() if p.requires_grad},
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "progress": {"update": int(update)},
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = str(path) + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, str(path))


def _load_state(path: Path, policy, optimizer, scaler, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    own = dict(policy.named_parameters())
    with torch.no_grad():
        for n, t in state["policy_trainable"].items():
            if n in own:
                own[n].copy_(t.to(dtype=own[n].dtype, device=own[n].device))
    optimizer.load_state_dict(state["optimizer"])
    for st in optimizer.state.values():
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                st[k] = v.to(device)
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["progress"]["update"])


def _group_reward_stats(rewards: torch.Tensor, group_ids: torch.Tensor, tol: float):
    """Per-group reward dispersion. Returns (mean within-group std, fraction of uninformative
    groups) where a group is uninformative iff its reward std <= tol (the same clamp floor used
    inside group_relative_advantages, so 'uninformative' <=> advantage is exactly 0)."""
    stds = []
    for g in torch.unique(group_ids):
        r = rewards[group_ids == g]
        stds.append(float(r.std(unbiased=False)))
    mean_std = float(sum(stds) / len(stds)) if stds else 0.0
    uninformative = float(sum(1 for s in stds if s <= tol) / len(stds)) if stds else 0.0
    return mean_std, uninformative, stds


def run_grpo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    loss_type: str = "grpo",
    run_name: str = "standard",
    resume: bool = False,
):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    policy = bundle["policy"]
    reward_model = bundle["reward_model"]
    reward_tok = bundle["reward_tokenizer"]
    tokenizer = bundle["tokenizer"]
    prompt_rows = bundle["prompt_rows"]
    optimizer = bundle["optimizer"]
    device = next(policy.parameters()).device

    n_updates = int(updates if updates is not None else cfg["updates"])
    eps = float(cfg["clip_epsilon"])
    beta_kl = float(cfg["kl_beta"])
    K = int(cfg["num_generations"])
    policy_epochs = int(cfg["policy_epochs"])
    prompts_per_update = int(cfg["prompts_per_update"])
    max_grad_norm = float(cfg["max_grad_norm"])
    max_completion_length = int(cfg["max_completion_length"])
    max_prompt_length = int(cfg["max_prompt_length"])
    reward_max_length = int(cfg["reward_max_length"])
    mask_truncated = bool(cfg.get("mask_truncated_completions", True))
    adv_eps = float(cfg.get("advantage_eps", 1e-6))
    group_tol = float(cfg.get("group_std_tolerance", adv_eps))

    gen_block = cfg.get("generation", {})
    temperature = float(gen_block.get("temperature", 0.7))
    top_p = float(gen_block.get("top_p", 0.9))
    do_sample = bool(gen_block.get("do_sample", True))

    if output:
        out = repo_path(output)
    elif run_name == "standard":
        out = repo_path(cfg["output"])
    else:
        out = repo_path(f"outputs/task3_grpo/{run_name}")
    out.mkdir(parents=True, exist_ok=True)

    results_dir = repo_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"{run_name}_train.jsonl"
    meta_path = results_dir / f"{run_name}_meta.json"
    summary_path = results_dir / f"{run_name}_summary.json"
    state_path = out / "train_state.pt"

    if not resume and log_path.exists():
        log_path.unlink()  # fresh start: never mix stale rows with a rerun

    scaler = make_grad_scaler(cfg)

    # Deterministic prompt schedule: consume rl_prompt_train in fixed order, cycling if fewer
    # prompts than updates. Recorded in meta so the exact prompt IDs are preserved. Each prompt
    # spawns K completions (one group); group_ids tag which scheduled prompt each belongs to.
    def prompt_index(update, j):
        return (update * prompts_per_update + j) % len(prompt_rows)

    planned_ids = [
        prompt_rows[prompt_index(u, j)].get("source_index", prompt_index(u, j))
        for u in range(n_updates)
        for j in range(prompts_per_update)
    ]

    start_update = 0
    if resume and state_path.exists():
        start_update = _load_state(state_path, policy, optimizer, scaler, device)

    save_json(meta_path, {
        "run_name": run_name,
        "loss_type": loss_type,
        "seed": int(cfg["seed"]),
        "base_model": cfg["base_model"],
        "reward_model": cfg["reward_model"],
        "midpoint_policy": cfg["paths"]["grpo_midpoint_policy"],
        "updates": n_updates,
        "policy_epochs": policy_epochs,
        "prompts_per_update": prompts_per_update,
        "num_generations": K,
        "clip_epsilon": eps,
        "kl_beta": beta_kl,
        "max_grad_norm": max_grad_norm,
        "mask_truncated_completions": mask_truncated,
        "advantage_eps": adv_eps,
        "group_std_tolerance": group_tol,
        "learning_rate": float(cfg["learning_rate"]),
        "max_completion_length": max_completion_length,
        "decoding": {
            "temperature": temperature, "top_p": top_p, "do_sample": do_sample,
            "max_new_tokens": max_completion_length, "max_prompt_length": max_prompt_length,
            "completions_per_prompt": K,
        },
        "precision": PRECISION,
        "kl_convention": KL_CONVENTION,
        "budget_note": "standard" if run_name == "standard" else "short fork",
        "prompt_source_indices": planned_ids,
    })

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    timer = wall_timer()
    opt_iter = start_update * policy_epochs

    for update in range(start_update, n_updates):
        rows = [prompt_rows[prompt_index(update, j)] for j in range(prompts_per_update)]
        src_ids = [r.get("source_index", prompt_index(update, j)) for j, r in enumerate(rows)]

        # Expand each scheduled prompt into K identical copies; sampling makes the completions
        # differ. group_ids[i] = which scheduled prompt produced completion i.
        expanded_prompts, group_ids_list = [], []
        for gi, r in enumerate(rows):
            msgs = prompt_messages(r)
            for _ in range(K):
                expanded_prompts.append(msgs)
                group_ids_list.append(gi)
        group_ids = torch.tensor(group_ids_list, device=device)

        # --- Rollout (on-policy, no grad) --------------------------------------
        gen = batch_generate(
            policy, tokenizer, expanded_prompts, max_prompt_length, max_completion_length,
            temperature=temperature, top_p=top_p, do_sample=do_sample,
        )
        sequences = gen["sequences"]
        attn = gen["attention_mask"]
        P = gen["prompt_width"]
        response_ids = gen["response_ids"]
        eos_mask = gen["response_mask"].to(device)  # EOS-trimmed; used for reported KL/length
        responses = gen["responses"]

        # Fixed rollout quantities (old/ref log-probs) in eval mode so LoRA dropout is OFF:
        # old_logp is the deterministic behavior log-prob. With policy_epochs=1 the single
        # update-epoch ratio is ~1 up to the dropout applied in the training-mode new_logp.
        policy.eval()
        with torch.no_grad():
            old_logp = token_logprobs(policy, sequences, attn, P, response_ids, cfg)[0].detach()
            with reference_mode(policy):
                ref_logp = token_logprobs(policy, sequences, attn, P, response_ids, cfg)[0].detach()
            rm_scores = score_reward_pairs(
                reward_model, reward_tok, expanded_prompts, responses, reward_max_length
            ).to(device)

        # Terminal reward = RM score. GRPO does NOT shape per-token KL into the reward (unlike
        # PPO); the KL penalty is a separate term inside grpo_policy_loss. Group-relative
        # advantage is one scalar per completion, standardized within each prompt group.
        seq_adv = group_relative_advantages(rm_scores, group_ids, eps=adv_eps).detach()

        # Loss mask: start from the EOS-trimmed response mask, then (optionally) zero out any
        # completion that hit max length without terminating, so truncated completions get no
        # gradient. They still inform the group baseline above (reward + advantage computed over
        # all K), matching "max-length completions masked from the training loss".
        loss_mask = eos_mask
        if mask_truncated:
            loss_mask = mask_truncated_sequences(loss_mask, gen["truncated"])

        mean_group_std, uninformative_frac, _ = _group_reward_stats(rm_scores, group_ids, group_tol)
        reward_mean = float(rm_scores.mean())
        kl_value = float(sampled_kl(old_logp, ref_logp, eos_mask))
        length_mean = float(sum(gen["response_lengths"]) / len(gen["response_lengths"]))
        masked_length_mean = mean_response_length(loss_mask)
        adv_var = float(seq_adv.var(unbiased=False)) if seq_adv.numel() > 1 else 0.0
        n_masked_completions = int((loss_mask.sum(-1) == 0).sum())

        # --- GRPO update epochs (one forward per opt step) ---------------------
        for epoch in range(policy_epochs):
            policy.train()
            with autocast_context(cfg):
                new_logp = token_logprobs(policy, sequences, attn, P, response_ids, cfg)[0]
                loss, stats = grpo_policy_loss(
                    new_logp, old_logp, seq_adv, loss_mask, ref_logp,
                    eps=eps, beta=beta_kl, loss_type=loss_type,
                    max_completion_length=max_completion_length,
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm)
            step_skipped = not bool(torch.isfinite(grad_norm))
            scaler.step(optimizer)  # GradScaler no-ops the step whose grads held inf/nan
            scaler.update()

            rec = {
                "run": run_name,
                "loss_type": loss_type,
                "update": update,
                "policy_epoch": epoch,
                "opt_iter": opt_iter,
                "source_indices": src_ids,
                "num_generations": K,
                "reward_mean": reward_mean,
                "within_group_reward_std": mean_group_std,
                "uninformative_group_fraction": uninformative_frac,
                "advantage_var": adv_var,
                "kl_sampled": kl_value,
                "kl_penalty_k3": float(stats["sampled_kl"]),
                "response_length_mean": length_mean,
                "masked_response_length_mean": masked_length_mean,
                "n_masked_completions": n_masked_completions,
                "loss": float(loss.detach()),
                "policy_term": float(stats["policy_term"]),
                "entropy": float(stats["sample_entropy"]),
                "clip_fraction": float(stats["clip_fraction"]),
                "ratio_mean": float(stats["ratio_mean"]),
                "grad_norm": float(grad_norm),
                "step_skipped": step_skipped,
                "scale": float(scaler.get_scale()),
                "lr": optimizer.param_groups[0]["lr"],
                "clip_epsilon": eps,
                "kl_beta": beta_kl,
                "wall_s": timer(),
            }
            if step_skipped:
                rec["event"] = "nonfinite_grad"  # step discarded by GradScaler; weights unchanged
            append_jsonl(log_path, rec)
            opt_iter += 1

        # Periodic + final checkpoint.
        if (update + 1) % max(1, n_updates // 4) == 0 or update + 1 == n_updates:
            _save_state(state_path, policy, optimizer, scaler, update + 1)
            policy.save_pretrained(str(out))

    policy.save_pretrained(str(out))

    summary = {
        "run_name": run_name,
        "loss_type": loss_type,
        "output": str(out),
        "updates": n_updates,
        "policy_epochs": policy_epochs,
        "num_generations": K,
        "clip_epsilon": eps,
        "kl_beta": beta_kl,
        "wall_seconds": timer(),
        "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else None,
        "kl_convention": KL_CONVENTION,
        "precision": PRECISION,
    }
    save_json(summary_path, summary)
    return summary


def fork_name(loss_type: str) -> str:
    return f"norm_{loss_type}"


def run_fork(config_path: str, loss_type: str, resume: bool = False, force: bool = False):
    """Short continuation fork from the identical midpoint, switching only the sequence
    normalization (loss_type). Skips if already trained so each normalization runs once."""
    cfg = load_yaml(config_path)
    name = fork_name(loss_type)
    summary_path = repo_path(cfg["results_dir"]) / f"{name}_summary.json"
    if summary_path.exists() and not force and not resume:
        print(f"skip {name}: already trained ({summary_path.name} exists)")
        return load_json(summary_path)
    return run_grpo(
        config_path,
        output=f"outputs/task3_grpo/{name}",
        updates=int(cfg["fork_updates"]),
        loss_type=loss_type,
        run_name=name,
        resume=resume,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name, args.resume)


if __name__ == "__main__":
    main()
