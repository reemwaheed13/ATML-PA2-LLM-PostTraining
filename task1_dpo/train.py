from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    repo_path,
)
from common.filtering import load_filtered_rows
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from common.precision import make_grad_scaler, sequence_logprobs, upcast_trainable_to_fp32
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None, run_name: str = "standard"):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]

    tokenizer = load_tokenizer(cfg["base_model"])
    # Filter over-length prompts at load, BEFORE shuffling; record the report to
    # results/task1_dpo/<run_name>_filter.json (exact retained/dropped indices).
    rows, filter_report = load_filtered_rows(
        path, tokenizer, int(cfg["max_sequence_length"]),
        results_dir=cfg["results_dir"], run_name=run_name,
    )
    # Fork/smoke subset is drawn AFTER the filter, so --max-examples yields that
    # many CLEAN (fit-in-window) examples with reproducible indices.
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    model = load_policy(cfg, trainable=True, fresh_lora=True)
    # Defensive guard (see README "Precision path"): PEFT 0.17.1 already keeps the
    # LoRA adapter in fp32, so this is normally a no-op (n_fp32 == 0). It only acts
    # if a version/config change produced fp16/bf16 trainable params.
    n_fp32 = upcast_trainable_to_fp32(model)

    # Deterministic, resume-safe shuffle: fixed generator -> same order every launch,
    # so skipping already-seen micro-batches on --resume is exact.
    gen = torch.Generator()
    gen.manual_seed(int(cfg["seed"]))
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
        generator=gen,
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "dataset_path": path,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
        "n_fp32": n_fp32,
        "filter_report": filter_report,
    }


def _save_train_state(path: Path, model, optimizer, scaler, micro, opt_step, seen):
    """Atomic checkpoint: fp32 trainable weights + optimizer + scaler + RNG + progress."""
    state = {
        "trainable": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "progress": {"micro_step": micro, "opt_step": opt_step, "seen_examples": seen},
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = str(path) + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, str(path))  # atomic: a crash mid-write cannot corrupt the live file


def _load_train_state(path: Path, model, optimizer, scaler, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    own = dict(model.named_parameters())
    with torch.no_grad():
        for n, t in state["trainable"].items():
            if n in own:
                own[n].copy_(t.to(dtype=own[n].dtype, device=own[n].device))
    optimizer.load_state_dict(state["optimizer"])
    # Move optimizer state tensors onto the param device (load_state_dict leaves them on CPU).
    for st in optimizer.state.values():
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                st[k] = v.to(device)
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["torch_rng"])
    if state.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    p = state["progress"]
    return int(p["micro_step"]), int(p["opt_step"]), int(p["seen_examples"])


def run_training(
    config_path: str,
    run_name: str = "standard",
    dataset_path: str | None = None,
    output_path: str | None = None,
    beta: float | None = None,
    max_examples: int | None = None,
    resume: bool = False,
):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples, run_name)
    cfg = bundle["cfg"]
    model = bundle["model"]
    loader = bundle["loader"]
    optimizer = bundle["optimizer"]
    beta = bundle["beta"]
    device = next(model.parameters()).device

    accum = int(cfg["grad_accum_steps"])
    max_norm = float(cfg["max_grad_norm"])
    epochs = int(cfg.get("epochs", 1))
    batch_size = int(cfg["batch_size"])

    if output_path:
        out = repo_path(output_path)
    elif run_name == "standard":
        out = repo_path(cfg["standard_output"])
    else:
        out = repo_path(f"outputs/task1_dpo/{run_name}")
    out.mkdir(parents=True, exist_ok=True)

    results_dir = repo_path(cfg["results_dir"])
    log_path = results_dir / f"{run_name}_train.jsonl"
    meta_path = results_dir / f"{run_name}_meta.json"
    summary_path = results_dir / f"{run_name}_summary.json"
    state_path = out / "train_state.pt"

    # Fresh (non-resume) start: rotate the train log so a rerun never mixes stale
    # rows with live ones. On --resume we keep appending to the existing log.
    if not resume and log_path.exists():
        log_path.unlink()

    scaler = make_grad_scaler(cfg)

    microbatches_per_epoch = math.ceil(len(bundle["rows"]) / batch_size)
    total_micro = microbatches_per_epoch * epochs
    total_opt_steps = max(1, math.ceil(total_micro / accum))
    ckpt_every = max(1, total_opt_steps // 4)

    start_micro, opt_step, seen = 0, 0, 0
    if resume and state_path.exists():
        start_micro, opt_step, seen = _load_train_state(state_path, model, optimizer, scaler, device)

    save_json(meta_path, {
        "run_name": run_name,
        "seed": int(cfg["seed"]),
        "beta": beta,
        "base_model": cfg["base_model"],
        "dataset": bundle["dataset_path"],
        "num_examples": len(bundle["rows"]),
        "learning_rate": float(cfg["learning_rate"]),
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
        "batch_size": batch_size,
        "grad_accum_steps": accum,
        "effective_batch": batch_size * accum,
        "epochs": epochs,
        "max_sequence_length": int(cfg["max_sequence_length"]),
        "max_grad_norm": max_norm,
        "dtype": cfg.get("dtype"),
        "precision": "base_fp16+lora_fp32_peft_native+autocast_fp16+gradscaler",
        "trainable_fp32_tensors": bundle["n_fp32"],
        "total_opt_steps_planned": total_opt_steps,
        "filtered_total": bundle["filter_report"]["total"],
        "filtered_kept": bundle["filter_report"]["kept"],
        "filtered_dropped": bundle["filter_report"]["dropped"],
        "dataset_indices_used": bundle["filter_report"]["kept_idx"][: len(bundle["rows"])],
    })

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    timer = wall_timer()

    micro = 0
    accum_count = 0
    window = _new_window()
    optimizer.zero_grad(set_to_none=True)

    for _epoch in range(epochs):
        for cb, rb in loader:
            if micro < start_micro:  # fast-forward over already-trained micro-batches on resume
                micro += 1
                continue
            cb = {k: v.to(device) for k, v in cb.items()}
            rb = {k: v.to(device) for k, v in rb.items()}

            # Policy log-probs (grad on), reference log-probs (adapter disabled, no grad).
            pc = sequence_logprobs(model, cb, cfg)[0]
            pr = sequence_logprobs(model, rb, cfg)[0]
            with torch.no_grad(), reference_mode(model):
                rc = sequence_logprobs(model, cb, cfg)[0]
                rr = sequence_logprobs(model, rb, cfg)[0]

            loss, metrics = dpo_loss(pc, pr, rc, rr, beta)

            if not torch.isfinite(loss):
                append_jsonl(log_path, {
                    "run": run_name, "event": "nonfinite_loss",
                    "micro_step": micro, "opt_step": opt_step, "loss": str(loss.item()),
                })
                micro += 1
                continue

            scaler.scale(loss / accum).backward()
            n = int(pc.shape[0])
            _accumulate_window(window, loss, metrics, n)  # example-weighted across the window
            accum_count += 1
            seen += n
            micro += 1

            if accum_count == accum:
                opt_step = _optimizer_step(
                    model, optimizer, scaler, max_norm, log_path, run_name,
                    opt_step, micro, seen, window, timer,
                )
                accum_count, window = 0, _new_window()
                if opt_step % ckpt_every == 0:
                    _save_train_state(state_path, model, optimizer, scaler, micro, opt_step, seen)
                    model.save_pretrained(str(out))

    # Flush a trailing partial accumulation window, if any.
    if accum_count > 0:
        opt_step = _optimizer_step(
            model, optimizer, scaler, max_norm, log_path, run_name,
            opt_step, micro, seen, window, timer,
        )

    _save_train_state(state_path, model, optimizer, scaler, micro, opt_step, seen)
    model.save_pretrained(str(out))

    summary = {
        "run_name": run_name,
        "output": str(out),
        "opt_steps": opt_step,
        "seen_examples": seen,
        "wall_seconds": timer(),
        "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else None,
    }
    save_json(summary_path, summary)
    return summary


def _new_window():
    return {"examples": 0, "loss": 0.0, "logit_mean": 0.0, "policy_margin_mean": 0.0, "preference_accuracy": 0.0}


def _accumulate_window(w, loss, metrics, n):
    """Accumulate example-weighted sums over an accumulation window. dpo_loss returns
    per-micro-batch MEANS, so weight each by its example count n to recover the true
    window average (not the last-micro-batch value)."""
    w["examples"] += n
    w["loss"] += float(loss.detach()) * n
    w["logit_mean"] += float(metrics["logit_mean"]) * n
    w["policy_margin_mean"] += float(metrics["policy_margin_mean"]) * n
    w["preference_accuracy"] += float(metrics["preference_accuracy"]) * n


def _optimizer_step(model, optimizer, scaler, max_norm, log_path, run_name, opt_step, micro, seen, window, timer):
    """Unscale -> clip -> (scaler) step. Logs grad_norm and the example-weighted
    window-average metrics every step; non-finite grads are logged and the poisoned
    update is skipped by the scaler, not applied."""
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_norm)
    found_inf = not bool(torch.isfinite(grad_norm))

    scaler.step(optimizer)   # no-op if unscale_ found inf/nan grads
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    ex = max(window["examples"], 1)
    rec = {
        "run": run_name,
        "opt_step": opt_step,
        "micro_step": micro,
        "seen_examples": seen,
        "window_examples": window["examples"],
        "loss": window["loss"] / ex,
        "logit_mean": window["logit_mean"] / ex,
        "policy_margin_mean": window["policy_margin_mean"] / ex,
        "preference_accuracy": window["preference_accuracy"] / ex,
        "grad_norm": float(grad_norm),
        "lr": optimizer.param_groups[0]["lr"],
        "scale": float(scaler.get_scale()),
        "wall_s": timer(),
    }
    if found_inf:
        rec["event"] = "nonfinite_grad"  # update skipped by GradScaler; weights unchanged
        append_jsonl(log_path, rec)
        return opt_step
    append_jsonl(log_path, rec)
    return opt_step + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples, args.resume)


if __name__ == "__main__":
    main()
