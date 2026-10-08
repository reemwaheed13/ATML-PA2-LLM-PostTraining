"""Empirically confirm the DPO objective behaviour at initialization.

Run (in Colab, after `pip install -r requirements.txt`):

    python -m scripts.verify_dpo_init                 # objective-only, instant
    python -m scripts.verify_dpo_init --with-model    # also loads the policy

At a freshly initialized LoRA adapter, lora_B == 0, so the adapter is a no-op
and policy == reference. Therefore (policy_margin - ref_margin) == 0 exactly,
and a CORRECT DPO loss must return log(2) = 0.6931472 for every beta.
The current (defective) code uses (policy_margin + ref_margin) instead, which
equals 2 * policy_margin != 0 and scales with beta, so it returns three
different values, none equal to log(2).

This script only REPORTS; it does not modify the objective.
"""

from __future__ import annotations

import argparse
import math

import torch

from task1_dpo.dpo import dpo_loss

BETAS = [0.03, 0.10, 0.30]
LOG2 = math.log(2.0)


def check_objective_only():
    """Feed equal policy/ref log-probs (the exact init condition) with nonzero margins."""
    # Hand-picked so chosen != rejected => nonzero, non-uniform margins.
    policy_chosen = torch.tensor([-10.0, -8.0, -15.0])
    policy_rejected = torch.tensor([-12.0, -7.0, -14.0])
    # policy == reference at init:
    ref_chosen = policy_chosen.clone()
    ref_rejected = policy_rejected.clone()

    pm = policy_chosen - policy_rejected
    rm = ref_chosen - ref_rejected
    print("[objective-only] policy_margin      :", pm.tolist())
    print("[objective-only] ref_margin         :", rm.tolist())
    print("[objective-only] margin DIFFERENCE  :", (pm - rm).tolist(), "(must be all 0)")
    print(f"[objective-only] target log(2)      : {LOG2:.6f}")
    for beta in BETAS:
        loss, _ = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
        flag = "OK" if abs(loss.item() - LOG2) < 1e-6 else "DIFFERS"
        print(f"[objective-only] beta={beta:<4}  loss={loss.item():.6f}  [{flag}]")


def check_with_model(config_path: str, max_length: int = 256):
    from common.data import encode_prompt_response, load_yaml, pad_batch
    from common.generation import response_sequence_logprobs
    from common.models import load_policy, load_tokenizer, reference_mode

    cfg = load_yaml(config_path)
    tok = load_tokenizer(cfg["base_model"])
    # fresh_lora=True => brand-new adapter with lora_B == 0 => policy == reference.
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    device = next(model.parameters()).device

    pairs = [
        ([{"role": "user", "content": "Explain gradient descent in one sentence."}],
         "Gradient descent iteratively steps parameters opposite the loss gradient to reduce error.",
         "It is a kind of soup you eat with a spoon on cold days."),
        ([{"role": "user", "content": "Name a prime number greater than ten."}],
         "Thirteen is a prime number greater than ten.",
         "Twelve is a prime number greater than ten."),
    ]
    chosen = [encode_prompt_response(tok, p, yc, max_length) for p, yc, _ in pairs]
    rejected = [encode_prompt_response(tok, p, yr, max_length) for p, _, yr in pairs]
    cb = {k: v.to(device) for k, v in pad_batch(tok, chosen).items()}
    rb = {k: v.to(device) for k, v in pad_batch(tok, rejected).items()}

    pc, _, _ = response_sequence_logprobs(model, cb)
    pr, _, _ = response_sequence_logprobs(model, rb)
    with torch.no_grad():
        with reference_mode(model):
            rc, _, _ = response_sequence_logprobs(model, cb)
            rr, _, _ = response_sequence_logprobs(model, rb)

    print("[with-model] max|policy_chosen - ref_chosen|  :", (pc - rc).abs().max().item())
    print("[with-model] max|policy_reject - ref_reject|  :", (pr - rr).abs().max().item())
    print("[with-model] policy_margin                    :", (pc - pr).detach().tolist())
    print("[with-model] margin DIFFERENCE (pm - rm)      :",
          ((pc - pr) - (rc - rr)).detach().tolist(), "(must be ~0)")
    print(f"[with-model] target log(2)                    : {LOG2:.6f}")
    for beta in BETAS:
        loss, _ = dpo_loss(pc, pr, rc, rr, beta)
        flag = "OK" if abs(loss.item() - LOG2) < 1e-4 else "DIFFERS"
        print(f"[with-model] beta={beta:<4}  loss={loss.item():.6f}  [{flag}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--with-model", action="store_true")
    args = ap.parse_args()
    check_objective_only()
    if args.with_model:
        print("-" * 60)
        check_with_model(args.config)


if __name__ == "__main__":
    main()
