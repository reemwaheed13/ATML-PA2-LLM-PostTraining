from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import append_jsonl, load_json, save_json, set_seed, wall_timer
from common.metrics import mean_response_length, sample_entropy, sampled_kl
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from common.precision import autocast_context, make_grad_scaler, token_logprobs
from task2_ppo.ppo import compute_gae, ppo_policy_loss, shaped_rewards, value_mse_loss

# Same KL averaging convention string Task 1 records (task1_dpo/evaluate.py:233), kept
# identical so Task 1/2/3 KL numbers are comparable. sampled_kl == masked_mean, i.e.
# sum_t(logp_policy - logp_ref) over response tokens / total response tokens.
KL_CONVENTION = "sampled_per_token_mean(sum_tokens/sum_response_tokens)"
PRECISION = "base_fp16+lora_fp32_peft_native+autocast_fp16+gradscaler"


def _report_trainable_dtypes(model, label):
    """DIAGNOSTIC (temporary): print every trainable parameter's name+dtype and the
    fp16/fp32 counts, so the GradScaler fp16-gradient failure is diagnosed by measurement
    rather than assumption. Remove once the precision path is fixed and documented."""
    from collections import Counter

    counts = Counter()
    n = 0
    print(f"[dtype-report] {label}: trainable parameters")
    for name, p in model.named_parameters():
        if p.requires_grad:
            n += 1
            counts[str(p.dtype)] += 1
            print(f"    {name}: {p.dtype}")
    fp16 = counts.get("torch.float16", 0)
    fp32 = counts.get("torch.float32", 0)
    print(f"[dtype-report] {label}: {n} trainable tensors | fp16={fp16} fp32={fp32} "
          f"other={n - fp16 - fp32} | dtypes={dict(counts)}")


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    # DIAGNOSTIC (temporary): measure trainable-parameter dtypes for both models before
    # any training. GradScaler.unscale_ raises on fp16 grads, so any fp16 trainable param
    # breaks the fp32-master-weights path. Measured, not inferred (the value model loads
    # via AutoModelForSequenceClassification, a different path than the policy adapter).
    _report_trainable_dtypes(policy, "policy")
    _report_trainable_dtypes(value_model, "value_model")

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


# ---------------------------------------------------------------------------
# Atomic checkpoint (two models + two optimizers + scaler + RNG + progress).
# Same crash-safe pattern as task1_dpo/train.py:_save_train_state, extended to the
# PPO policy+critic pair so a dying Colab session resumes exactly.
# ---------------------------------------------------------------------------
def _save_state(path: Path, policy, value_model, policy_opt, value_opt, scaler, update):
    state = {
        "policy_trainable": {n: p.detach().cpu() for n, p in policy.named_parameters() if p.requires_grad},
        "value_trainable": {n: p.detach().cpu() for n, p in value_model.named_parameters() if p.requires_grad},
        "policy_opt": policy_opt.state_dict(),
        "value_opt": value_opt.state_dict(),
        "scaler": scaler.state_dict(),
        "progress": {"update": int(update)},
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = str(path) + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, str(path))


def _load_state(path: Path, policy, value_model, policy_opt, value_opt, scaler, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    for model, key in ((policy, "policy_trainable"), (value_model, "value_trainable")):
        own = dict(model.named_parameters())
        with torch.no_grad():
            for n, t in state[key].items():
                if n in own:
                    own[n].copy_(t.to(dtype=own[n].dtype, device=own[n].device))
    policy_opt.load_state_dict(state["policy_opt"])
    value_opt.load_state_dict(state["value_opt"])
    for opt in (policy_opt, value_opt):
        for st in opt.state.values():
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(device)
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["progress"]["update"])


def _response_values(value_model, sequences, attention_mask, prompt_width, resp_len, cfg):
    """Per-response-step critic values V_t, aligned to the policy log-probs / rewards.

    token_values returns a value for EVERY position: [B, P+G] (common/models.py:216-231).
    The hidden state at absolute index j is the representation after consuming tokens
    0..j, i.e. the state from which token j+1 is produced. Response token t (t=0..G-1)
    sits at absolute position P+t, so the state it is emitted FROM lives at index P+t-1.
    Sweeping t gives the slice [P-1 : P-1+G]. Prompts are LEFT-padded
    (common/models.py:30,34 -> load_tokenizer default padding_side="left"), so index
    P-1 is the last REAL prompt token for every example and this constant offset is
    correct regardless of prompt length. Off-by-one in either direction would feed GAE
    the value of the wrong state (the critic would baseline an action with a state that
    has already seen it, or lag one token), biasing advantages and the value target.
    """
    with autocast_context(cfg):
        vals_full = token_values(value_model, sequences, attention_mask)
    return vals_full[:, prompt_width - 1 : prompt_width - 1 + resp_len]


def run_ppo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    clip_epsilon: float | None = None,
    kl_beta: float | None = None,
    run_name: str = "standard",
    resume: bool = False,
):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    policy = bundle["policy"]
    value_model = bundle["value_model"]
    reward_model = bundle["reward_model"]
    reward_tok = bundle["reward_tokenizer"]
    tokenizer = bundle["tokenizer"]
    prompt_rows = bundle["prompt_rows"]
    policy_opt = bundle["policy_optimizer"]
    value_opt = bundle["value_optimizer"]
    device = next(policy.parameters()).device

    n_updates = int(updates if updates is not None else cfg["updates"])
    eps = float(clip_epsilon if clip_epsilon is not None else cfg["clip_epsilon"])
    beta_kl = float(kl_beta if kl_beta is not None else cfg["kl_beta"])
    ppo_epochs = int(cfg["ppo_epochs"])
    prompts_per_update = int(cfg["prompts_per_update"])
    gamma = float(cfg["gamma"])
    lam = float(cfg["gae_lambda"])
    value_coef = float(cfg["value_coef"])
    missing_eos_penalty = float(cfg["missing_eos_penalty"])
    max_grad_norm = float(cfg["max_grad_norm"])
    max_response_length = int(cfg["max_response_length"])
    max_prompt_length = int(cfg["max_prompt_length"])
    reward_max_length = int(cfg["reward_max_length"])

    gen_block = cfg.get("generation", {})
    temperature = float(gen_block.get("temperature", 0.7))
    top_p = float(gen_block.get("top_p", 0.9))
    do_sample = bool(gen_block.get("do_sample", True))

    if output:
        out = repo_path(output)
    elif run_name == "standard":
        out = repo_path(cfg["output"])
    else:
        out = repo_path(f"outputs/task2_ppo/{run_name}")
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

    # Deterministic prompt schedule: consume rl_prompt_train in fixed order, cycling if
    # fewer prompts than updates. Recorded in meta so the exact prompt IDs are preserved.
    def prompt_index(update, j):
        return (update * prompts_per_update + j) % len(prompt_rows)

    planned_ids = [
        prompt_rows[prompt_index(u, j)].get("source_index", prompt_index(u, j))
        for u in range(n_updates)
        for j in range(prompts_per_update)
    ]

    start_update = 0
    if resume and state_path.exists():
        start_update = _load_state(state_path, policy, value_model, policy_opt, value_opt, scaler, device)

    save_json(meta_path, {
        "run_name": run_name,
        "seed": int(cfg["seed"]),
        "base_model": cfg["base_model"],
        "reward_model": cfg["reward_model"],
        "midpoint_policy": cfg["paths"]["ppo_midpoint_policy"],
        "midpoint_value": cfg["paths"]["ppo_midpoint_value"],
        "updates": n_updates,
        "ppo_epochs": ppo_epochs,
        "prompts_per_update": prompts_per_update,
        "clip_epsilon": eps,
        "kl_beta": beta_kl,
        "gamma": gamma,
        "gae_lambda": lam,
        "value_coef": value_coef,
        "missing_eos_penalty": missing_eos_penalty,
        "max_grad_norm": max_grad_norm,
        "normalize_advantages": False,  # 1 response/update: zero-meaning a single trajectory kills the reward level
        "policy_learning_rate": float(cfg["policy_learning_rate"]),
        "value_lora_learning_rate": float(cfg["value_lora_learning_rate"]),
        "value_head_learning_rate": float(cfg["value_head_learning_rate"]),
        "value_train_mode": cfg.get("value_train_mode", "head_only"),
        "decoding": {
            "temperature": temperature, "top_p": top_p, "do_sample": do_sample,
            "max_new_tokens": max_response_length, "max_prompt_length": max_prompt_length,
            "samples_per_prompt": 1,
        },
        "precision": PRECISION,
        "kl_convention": KL_CONVENTION,
        "budget_note": "standard" if run_name == "standard" else "short fork",
        "prompt_source_indices": planned_ids,
    })

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    timer = wall_timer()
    opt_iter = start_update * ppo_epochs

    for update in range(start_update, n_updates):
        rows = [prompt_rows[prompt_index(update, j)] for j in range(prompts_per_update)]
        prompts = [prompt_messages(r) for r in rows]
        src_ids = [r.get("source_index", prompt_index(update, j)) for j, r in enumerate(rows)]

        # --- Rollout (on-policy, no grad) --------------------------------------
        gen = batch_generate(
            policy, tokenizer, prompts, max_prompt_length, max_response_length,
            temperature=temperature, top_p=top_p, do_sample=do_sample,
        )
        sequences = gen["sequences"]
        attn = gen["attention_mask"]
        P = gen["prompt_width"]
        response_ids = gen["response_ids"]
        mask = gen["response_mask"].to(device)
        resp_len = response_ids.shape[1]
        responses = gen["responses"]

        # Fixed rollout quantities (old/ref log-probs, rollout values) are computed in
        # eval mode so LoRA dropout is OFF: old_logp is the deterministic behavior log-prob
        # (generation itself ran in eval), not a dropout-perturbed one. The update epochs
        # below switch back to train(). This keeps the epoch-0 ratio ~1 up to the dropout
        # applied only inside the training-mode new_logp forward.
        policy.eval()
        value_model.eval()
        with torch.no_grad():
            old_logp = token_logprobs(policy, sequences, attn, P, response_ids, cfg)[0].detach()
            with reference_mode(policy):
                ref_logp = token_logprobs(policy, sequences, attn, P, response_ids, cfg)[0].detach()
            rm_scores = score_reward_pairs(reward_model, reward_tok, prompts, responses, reward_max_length).to(device)

        # Terminal learned reward, with the missing-EOS penalty (configs/ppo.yaml:22)
        # applied when a response was truncated without terminating.
        terminal = rm_scores.clone()
        for b in range(len(responses)):
            if not gen["terminated_with_eos"][b]:
                terminal[b] -= missing_eos_penalty

        with torch.no_grad():
            rewards = shaped_rewards(terminal, old_logp, ref_logp, mask, beta_kl)
            rollout_values = _response_values(value_model, sequences, attn, P, resp_len, cfg).float().detach()
            advantages, returns = compute_gae(rewards, rollout_values, mask, gamma, lam)
            advantages = advantages.detach()
            returns = returns.detach()
            # No advantage normalization (single trajectory). See meta.normalize_advantages.

        reward_mean = float(rm_scores.mean())
        terminal_mean = float(terminal.mean())
        kl_value = float(sampled_kl(old_logp, ref_logp, mask))
        length_mean = mean_response_length(mask)

        # --- PPO update epochs (one forward per opt step; each logged value is the
        # full-batch masked_mean, so there is no partial-window averaging issue). ---
        for epoch in range(ppo_epochs):
            policy.train()
            value_model.train()
            with autocast_context(cfg):
                new_logp = token_logprobs(policy, sequences, attn, P, response_ids, cfg)[0]
                policy_loss, _, clip_fraction = ppo_policy_loss(new_logp, old_logp, advantages, mask, eps=eps)
                pred_values = _response_values(value_model, sequences, attn, P, resp_len, cfg)
                value_loss = value_mse_loss(pred_values, returns, mask)
                loss = policy_loss + value_coef * value_loss

            policy_opt.zero_grad(set_to_none=True)
            value_opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # Single scaler, two optimizers (policy and critic share NO parameters:
            # 1.5B causal-LM policy vs 0.5B seq-cls critic, common/models.py:63-69,152-159).
            # Summed loss -> one backward -> unscale/clip/step each optimizer -> one update.
            scaler.unscale_(policy_opt)
            scaler.unscale_(value_opt)
            policy_gn = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_grad_norm)
            value_gn = torch.nn.utils.clip_grad_norm_(trainable_parameters(value_model), max_grad_norm)
            policy_skip = not bool(torch.isfinite(policy_gn))
            value_skip = not bool(torch.isfinite(value_gn))

            scaler.step(policy_opt)  # GradScaler no-ops the step whose grads held inf/nan
            scaler.step(value_opt)
            scaler.update()

            entropy = float(sample_entropy(new_logp.detach(), mask))

            rec = {
                "run": run_name,
                "update": update,
                "ppo_epoch": epoch,
                "opt_iter": opt_iter,
                "source_indices": src_ids,
                "reward_mean": reward_mean,
                "terminal_reward_mean": terminal_mean,
                "kl_sampled": kl_value,
                "response_length_mean": length_mean,
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "entropy": entropy,
                "clip_fraction": float(clip_fraction),
                "policy_grad_norm": float(policy_gn),
                "value_grad_norm": float(value_gn),
                "policy_step_skipped": policy_skip,
                "value_step_skipped": value_skip,
                "scale": float(scaler.get_scale()),
                "lr_policy": policy_opt.param_groups[0]["lr"],
                "clip_epsilon": eps,
                "kl_beta": beta_kl,
                "wall_s": timer(),
            }
            if policy_skip or value_skip:
                rec["event"] = "nonfinite_grad"  # step(s) discarded by GradScaler; weights unchanged
            append_jsonl(log_path, rec)
            opt_iter += 1

        # Periodic + final checkpoint.
        if (update + 1) % max(1, n_updates // 4) == 0 or update + 1 == n_updates:
            _save_state(state_path, policy, value_model, policy_opt, value_opt, scaler, update + 1)
            policy.save_pretrained(str(out))

    policy.save_pretrained(str(out))

    summary = {
        "run_name": run_name,
        "output": str(out),
        "updates": n_updates,
        "ppo_epochs": ppo_epochs,
        "clip_epsilon": eps,
        "kl_beta": beta_kl,
        "wall_seconds": timer(),
        "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else None,
        "kl_convention": KL_CONVENTION,
        "precision": PRECISION,
    }
    save_json(summary_path, summary)
    return summary


def fork_name(eps: float, kl: float) -> str:
    return f"fork_eps{eps:.2f}_kl{kl:.2f}"


def run_fork(config_path: str, eps: float, kl: float, resume: bool = False, force: bool = False):
    """Short continuation fork from the identical midpoint. Skips if already trained so
    the shared (eps=0.20, kl=0.10) fork runs exactly once across the clip and KL studies."""
    cfg = load_yaml(config_path)
    name = fork_name(eps, kl)
    summary_path = repo_path(cfg["results_dir"]) / f"{name}_summary.json"
    if summary_path.exists() and not force and not resume:
        print(f"skip {name}: already trained ({summary_path.name} exists)")
        return load_json(summary_path)
    return run_ppo(
        config_path,
        output=f"outputs/task2_ppo/{name}",
        updates=int(cfg["fork_updates"]),
        clip_epsilon=eps,
        kl_beta=kl,
        run_name=name,
        resume=resume,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name, args.resume)


if __name__ == "__main__":
    main()
