"""Empirically confirm the PPO clipped-surrogate geometry.

With advantage A>0 and ratio rho past the clip boundary, the CORRECT clipped
surrogate is capped at (1+eps)*A, so the loss must pin at -(1+eps)*A and be
INVARIANT to how large rho gets. The defective max() version grows with rho.

Run (CPU, seconds):  python -m scripts.verify_ppo_clip

Imports the real task2_ppo.ppo.ppo_policy_loss; it does not reimplement it.
"""

from __future__ import annotations

import math

import torch

from task2_ppo.ppo import ppo_policy_loss

EPS = 0.2
RHOS = [1.5, 2.0, 5.0]


def main():
    advantage = torch.ones(1, 1)   # A = +1
    mask = torch.ones(1, 1)
    target = -(1.0 + EPS)          # correct loss for A>0, rho>1+eps
    print(f"advantage=+1  eps={EPS}  target (correct) loss = {target:.6f} for every rho")
    for rho in RHOS:
        new_logp = torch.log(torch.tensor([[float(rho)]]))
        old_logp = torch.zeros(1, 1)
        loss, ratio, clip_fraction = ppo_policy_loss(new_logp, old_logp, advantage, mask, eps=EPS)
        flag = "OK" if abs(loss.item() - target) < 1e-5 else "DIFFERS"
        print(f"  rho={rho:<4}  loss={loss.item():.6f}  (defective max would give {-rho:.6f})  [{flag}]")


if __name__ == "__main__":
    main()
