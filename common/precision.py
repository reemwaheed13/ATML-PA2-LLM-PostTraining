
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
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=amp_dtype(cfg))
    return nullcontext()


def upcast_trainable_to_fp32(model) -> int:
    n = 0
    for p in model.parameters():
        if p.requires_grad and p.dtype in (torch.float16, torch.bfloat16):
            p.data = p.data.float()
            n += 1
    return n


def make_grad_scaler(cfg: dict):
    enabled = torch.cuda.is_available() and amp_dtype(cfg) == torch.float16
    return torch.amp.GradScaler("cuda", enabled=enabled)


def sequence_logprobs(model, batch: dict, cfg: dict):
    with autocast_context(cfg):
        return response_sequence_logprobs(model, batch)


def token_logprobs(model, sequences, attention_mask, prompt_width, response_ids, cfg: dict):
    with autocast_context(cfg):
        return response_token_logprobs(model, sequences, attention_mask, prompt_width, response_ids)
