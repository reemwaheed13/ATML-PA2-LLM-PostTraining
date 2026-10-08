"""Empirically confirm GRPO group-relative advantages are per-group.

Two signatures of correctness:
1. A group whose completions all got the SAME reward is uninformative -> advantage
   exactly 0; and each group's advantages are mean-zero within the group.
2. Shift invariance: adding a constant to one group's rewards must NOT change any
   within-group advantage.

The defective (global) normalization fails both (it leaks the cross-prompt mean).

Run (CPU, seconds):  python -m scripts.verify_grpo_groups

Imports the real task3_grpo.grpo.group_relative_advantages; no reimplementation.
"""

from __future__ import annotations

import torch

from task3_grpo.grpo import group_relative_advantages


def main():
    rewards = torch.tensor([5.0, 5.0, 20.0, 30.0])
    groups = torch.tensor([0, 0, 1, 1])
    adv = group_relative_advantages(rewards, groups)
    print("rewards           :", rewards.tolist())
    print("groups            :", groups.tolist())
    print("advantages        :", [round(x, 4) for x in adv.tolist()])
    print("  correct   -> [0.0, 0.0, -1.0, 1.0]")
    print("  defective -> [-0.9428, -0.9428, 0.4714, 1.4142]")

    g0_zero = torch.allclose(adv[:2], torch.zeros(2), atol=1e-4)
    g0_mean0 = abs(float(adv[:2].mean())) < 1e-4
    g1_mean0 = abs(float(adv[2:].mean())) < 1e-4
    print(f"\n[check] constant group g0 == 0        : {g0_zero}")
    print(f"[check] g0 mean ~ 0 and g1 mean ~ 0   : {g0_mean0 and g1_mean0}")

    # Shift invariance: add +100 to group 1 only.
    shifted = rewards.clone()
    shifted[2:] += 100.0
    adv2 = group_relative_advantages(shifted, groups)
    invariant = torch.allclose(adv, adv2, atol=1e-5)
    print(f"[check] shift group1 by +100 -> advantages unchanged : {invariant}")
    print("  shifted advantages:", [round(x, 4) for x in adv2.tolist()])


if __name__ == "__main__":
    main()
