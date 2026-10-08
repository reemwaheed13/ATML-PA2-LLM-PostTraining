"""Single source of truth for the DPO/RL precision path.

The released configuration specifies `dtype: float16` and ships no loss scaler or
autocast context. With a fp16 base model, PEFT creates the trainable LoRA
adapters in fp16 as well; optimizing fp16 master weights with AdamW underflows
(the optimizer moments and the `w -= lr*grad` update are smaller than the fp16
ULP), and `clip_grad_norm_` over fp16 gradients can overflow to inf/nan.

Fix (see README "Precision deviation"): keep the frozen base in fp16, upcast the
trainable LoRA master weights to fp32, run every forward under autocast(fp16),
and use a GradScaler. EVERY entry point (train, evaluate, ablations) must compute
log-probs through `sequence_logprobs` here so the numerical path is identical
across training, evaluation, the standard run, and all ablation forks. KL,
preference accuracy, and DPO loss are all differences of log-probs; a mismatched
precision path between conditions would invalidate the comparison.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from common.generation import response_sequence_logprobs


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
    """Cast trainable (LoRA) params to fp32 master weights. Call BEFORE building
    the optimizer so AdamW moments and gradient clipping are computed in fp32.
    Returns the number of tensors upcast."""
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
