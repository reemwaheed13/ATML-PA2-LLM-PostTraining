
from __future__ import annotations

import math
import sys

import torch

from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss

ATOL = 1e-4


def _loss(new, old, adv, mask, ref, eps=0.2, beta=0.0, loss_type="grpo", lmax=None):
    t = lambda x: torch.tensor(x, dtype=torch.float32)
    return grpo_policy_loss(t(new), t(old), t(adv), t(mask), t(ref),
                            eps=eps, beta=beta, loss_type=loss_type, max_completion_length=lmax)


def main():
    results = []

    def check(name, passed, detail=""):
        results.append((name, bool(passed), detail))
        print(f"[{'PASS' if passed else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")

    c = 1.273438
    adv = group_relative_advantages(torch.tensor([c, c, c, c]), torch.tensor([0, 0, 0, 0]))
    check("constant group [c,c,c,c] -> exactly 0",
          torch.allclose(adv, torch.zeros(4), atol=1e-8), f"adv={[round(x,6) for x in adv.tolist()]}")

    adv = group_relative_advantages(torch.tensor([0.0, 0.0, 1.0, 1.0]), torch.tensor([0, 0, 0, 0]))
    check("[0,0,1,1] -> [-1,-1,+1,+1]",
          torch.allclose(adv, torch.tensor([-1.0, -1.0, 1.0, 1.0]), atol=ATOL),
          f"adv={[round(x,5) for x in adv.tolist()]}")

    adv = group_relative_advantages(torch.tensor([3.0, 7.0]), torch.tensor([0, 0]))
    check("K=2 pair [3,7] -> [-1,+1]",
          torch.allclose(adv, torch.tensor([-1.0, 1.0]), atol=ATOL),
          f"adv={[round(x,5) for x in adv.tolist()]}")

    rewards = torch.tensor([0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0, 11.0])
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    adv = group_relative_advantages(rewards, groups)
    g0, g1 = adv[:4], adv[4:]
    per_group_zero = abs(float(g0.sum())) < ATOL and abs(float(g1.sum())) < ATOL
    per_group_vals = (torch.allclose(g0, torch.tensor([-1.0, -1.0, 1.0, 1.0]), atol=ATOL)
                      and torch.allclose(g1, torch.tensor([-1.0, -1.0, 1.0, 1.0]), atol=ATOL))
    global_adv = (rewards - rewards.mean()) / rewards.std(unbiased=False)
    check("two groups w/ different means -> each group sums to 0 (per-group, not global)",
          per_group_zero and per_group_vals,
          f"g0.sum={float(g0.sum()):.2e} g1.sum={float(g1.sum()):.2e}; "
          f"global would give g0.sum={float(global_adv[:4].sum()):.3f} (defect signature)")

    clip_ok, losses = True, {}
    for rho in (1.5, 2.0, 5.0):
        loss, _ = _loss([[math.log(rho)]], [[0.0]], [1.0], [[1.0]], [[math.log(rho)]],
                        eps=0.2, beta=0.0, loss_type="grpo")
        losses[rho] = round(float(loss), 5)
        clip_ok = clip_ok and abs(float(loss) + 1.2) < 1e-5
    check("clipped surrogate (A=+1, eps=0.2) pins at -1.2 for rho in {1.5,2,5}",
          clip_ok, f"losses={losses} (defect max -> -rho)")

    _, s0 = _loss([[-0.5]], [[-0.5]], [0.0], [[1.0]], [[-0.5]], beta=1.0)
    l1, s1 = _loss([[0.0]], [[0.0]], [0.0], [[1.0]], [[-2.0]], beta=1.0)
    l2, _ = _loss([[0.0]], [[0.0]], [0.0], [[1.0]], [[-2.0]], beta=2.0)
    kl_expected = math.exp(-2.0) + 2.0 - 1.0
    kl_ok = (abs(float(s0["sampled_kl"])) < 1e-6
             and float(s1["sampled_kl"]) > 0
             and abs(float(s1["sampled_kl"]) - kl_expected) < 1e-5
             and abs(float(l2) - 2.0 * float(l1)) < 1e-5)
    check("KL penalty: 0 at ref==policy, k3-correct and +beta-scaled otherwise",
          kl_ok, f"kl(ref!=pol)={float(s1['sampled_kl']):.5f} expected {kl_expected:.5f}")

    mask = [[1, 1, 1, 1], [1, 1, 0, 0]]
    logp = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    l_grpo, _ = _loss(logp, logp, [1.0, 1.0], mask, logp, loss_type="grpo")
    l_dr, _ = _loss(logp, logp, [1.0, 1.0], mask, logp, loss_type="dr_grpo", lmax=4)
    norm_ok = abs(float(l_grpo) + 1.0) < 1e-6 and abs(float(l_dr) + 0.75) < 1e-6
    check("grpo normalizes by response T_k (=-1.0); dr_grpo by constant L_max (=-0.75)",
          norm_ok, f"grpo={float(l_grpo):.4f} dr_grpo={float(l_dr):.4f}")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed.")
    if n_fail:
        print("GRPO objective FAILED verification -- do not run training until fixed.")
        sys.exit(1)
    print("GRPO objective verified: per-group reduction, min-clip surrogate, k3 KL, "
          "response-token normalization, constant-vs-length normalization.")


if __name__ == "__main__":
    main()
