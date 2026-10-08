"""Single source of truth for the DPO/RL precision path.

Precision facts for this codebase (PEFT 0.17.1, empirically verified):
- The frozen base model loads in fp16 per the config (`dtype: float16`).
- PEFT keeps the trainable LoRA adapter parameters in fp32; it does NOT cast them
  to the fp16 base dtype. The trainable params are already fp32, so there is no
  fp16-AdamW optimizer-underflow problem to fix.
- The forward passes run under autocast(fp16) for speed/memory: base and LoRA
  matmuls execute in fp16. There is no dtype mismatch in the LoRA path - PEFT
  casts the activation to the adapter dtype and casts the result back, and
  autocast governs op precision (holds with or without autocast active).
- A GradScaler guards the fp16 activation gradients produced inside the autocast
  region from underflow (grads flowing through fp16 matmuls before they accumulate
  into the fp32 param .grad). It is not about fp16 parameters.

EVERY entry point (train, evaluate, ablations) must compute log-probs through the
wrappers here so the numerical path is identical across training, evaluation, the
standard run, and all ablation forks. KL, preference accuracy, and DPO loss are
all differences of log-probs; a mismatched precision path between conditions
would invalidate the comparison.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from common.generation import response_sequence_logprobs, response_token_logprobs


def amp_dtype(cfg: dict) -> torch.dtype:
    name = str(cfg.get("dtype", "float16")).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float16


def autocast_context(cfg: dict):
    """Autocast region used around every forward pass. No-op on CPU."""
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=amp_dtype(cfg))
    return nullcontext()


def upcast_trainable_to_fp32(model) -> int:
    """Defensive guard: ensure trainable params are fp32 before building the
    optimizer. Under PEFT 0.17.1 the LoRA adapter is ALREADY fp32, so this is a
    no-op and returns 0 (surfaced as `trainable_fp32_tensors` in the run meta).
    It only acts if a different PEFT version or config ever produced fp16/bf16
    trainable params, which would be a poor choice for AdamW. Returns the count."""
    n = 0
    for p in model.parameters():
        if p.requires_grad and p.dtype in (torch.float16, torch.bfloat16):
            p.data = p.data.float()
            n += 1
    return n


def make_grad_scaler(cfg: dict):
    """GradScaler, enabled only for CUDA + fp16 (bf16 and CPU do not need it)."""
    enabled = torch.cuda.is_available() and amp_dtype(cfg) == torch.float16
    return torch.amp.GradScaler("cuda", enabled=enabled)


def sequence_logprobs(model, batch: dict, cfg: dict):
    """Response-token summed log-probs under the shared autocast path.

    Returns (seq_logp, per_token_logp, response_mask) exactly like
    `common.generation.response_sequence_logprobs`, but wrapped in autocast so
    train and eval share one numerical path. The caller controls grad: wrap in
    `torch.no_grad()` + `reference_mode(model)` for the reference pass.
    """
    with autocast_context(cfg):
        return response_sequence_logprobs(model, batch)


def token_logprobs(model, sequences, attention_mask, prompt_width, response_ids, cfg: dict):
    """Per-token log-probs of already-generated response tokens, under the shared
    autocast path. Returns (per_token_logp, logits) like
    `common.generation.response_token_logprobs`. Used for the sampled-KL estimator
    so generation-time log-probs match the training/eval numerics. Caller controls
    grad / `reference_mode`.
    """
    with autocast_context(cfg):
        return response_token_logprobs(model, sequences, attention_mask, prompt_width, response_ids)
