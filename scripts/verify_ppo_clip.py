
from __future__ import annotations

import math

import torch

from task2_ppo.ppo import ppo_policy_loss

EPS = 0.2
RHOS = [1.5, 2.0, 5.0]


def main():
    advantage = torch.ones(1, 1)
    mask = torch.ones(1, 1)
    target = -(1.0 + EPS)
    print(f"advantage=+1  eps={EPS}  target (correct) loss = {target:.6f} for every rho")
    for rho in RHOS:
        new_logp = torch.log(torch.tensor([[float(rho)]]))
        old_logp = torch.zeros(1, 1)
        loss, ratio, clip_fraction = ppo_policy_loss(new_logp, old_logp, advantage, mask, eps=EPS)
        flag = "OK" if abs(loss.item() - target) < 1e-5 else "DIFFERS"
        print(f"  rho={rho:<4}  loss={loss.item():.6f}  (defective max would give {-rho:.6f})  [{flag}]")


if __name__ == "__main__":
    main()
