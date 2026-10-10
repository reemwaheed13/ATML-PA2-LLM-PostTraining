# ATML PA2 - LLM Post-Training

<!-- FINAL_STUDENT_SETUP -->

## Quick start

```bash
git clone https://github.com/AbDu11aHHH/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m scripts.validate_assets
```

The fixed datasets, cached diagnostics, and supplied
continuation checkpoints are downloaded from:

https://huggingface.co/datasets/AbDu11aHHH/ATML-PA2-assets

Pinned release revision:

`0b350481fb03f5525a35bcdec4131bd4fe487f98`

---
# ATML PA2 - LLM Post-Training

This is the **student starter repository** for ATML PA2. The released code is intentionally incomplete: Tasks 1-3 provide model/data loading, objective helpers, checkpoint restoration, and experiment entry points, but **you must implement the training loops and ablation orchestration yourself**. Each of Tasks 1-3 also contains one deliberate algorithmic defect in its core objective code; identifying and correcting these defects is part of validating your implementation.

Task 4 supplies the fixed AI safety judge and response-generation utilities, but you must write the evaluation/aggregation code. Task 5 supplies the exact RLVR verifier, the fixed pairwise AI judge used for RLAIF evaluation, and data/model loaders; you must implement the requested evaluation and analysis.

## 1. Clone and install

```bash
git clone https://github.com/COURSE_ORG/ATML-PA2-LLM-PostTraining.git
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
```

## 2. Download the course assets

The large course-created checkpoints and fixed data are distributed as a GitHub Release asset rather than normal Git files. After cloning, run:

```bash
python -m scripts.download_assets
python -m scripts.validate_assets
```

If your instructor provides a direct asset URL separately, use:

```bash
python -m scripts.download_assets --url '<ASSET_URL>'
```

Public base/reward/judge models are downloaded from Hugging Face at runtime and are **not** included in the course asset archive.

The installer also materializes the fixed 100-example Task 5 transfer set from the official SVAMP challenge-set source if it is not already present. The tiny Task 1 word-limit prompt set is tracked directly in this repository.

## 3. Environment check

```bash
python -m scripts.check_environment
```

Run commands from the repository root. The reference environment used to prepare the release pins Transformers 4.57.1, TRL 0.27.2, PEFT 0.17.1, and Tokenizers 0.22.1.

## 4. Supplied course checkpoints

After `download_assets`, these directories should exist:

```text
checkpoints/ppo_midpoint_policy/
checkpoints/ppo_midpoint_value/
checkpoints/grpo_midpoint_policy/
checkpoints/rlvr_policy/
checkpoints/rlaif_policy/
```

PPO and GRPO begin from the supplied continuation checkpoints. RLVR and RLAIF are supplied frozen evaluation policies; students do not retrain them.

The PPO value checkpoint is intentionally released as the exact staff midpoint state, including its imperfect held-out value calibration. Treat critic behavior as an analysis variable rather than assuming a perfect baseline, and start every PPO fork from the identical supplied policy/value state. The default continuation generation cap is 512 tokens for feasibility; frozen evaluation uses the larger cap specified in `configs/ppo.yaml`.

## 5. Task entry points

### Task 1 - DPO

```bash
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta --config configs/dpo.yaml
python -m task1_dpo.analyze_length --config configs/dpo.yaml
```

### Task 2 - PPO

```bash
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
python -m task2_ppo.ablate_kl --config configs/ppo.yaml
```

### Task 3 - GRPO

```bash
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
python -m task3_grpo.compare_normalization --config configs/grpo.yaml
```

### Task 4 - Safety calibration

The judge loader/parser are supplied. You must implement the requested generation aggregation and evaluation.

```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```

### Task 5 - RLVR vs RLAIF

The exact verifier and pairwise AI judge are supplied; you implement the evaluation/analysis.

```bash
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

## 6. Reproducibility rules

- Do not alter course-provided data, cached rollouts, or supplied checkpoints.
- Start every short fork from the **same supplied midpoint checkpoint**.
- Keep prompt IDs, generated-token/update budgets, seed, and evaluation procedure matched across ablations.
- Commit your code, configs, small JSON/CSV logs, and figures. Do not commit downloaded checkpoints, raw course assets, or model caches.
- Record peak VRAM and wall-clock time for the standard PPO and GRPO continuations.

See the assignment manual for the required experiments, metrics, and report questions.

## 7. Precision path

The frozen base model loads in fp16 per the config (`dtype: float16`). PEFT 0.17.1 keeps the trainable LoRA adapter parameters in fp32 (empirically verified: `lora_A`/`lora_B` are `torch.float32`), so the optimizer trains fp32 parameters. The released code ships no autocast context or loss scaler; the training/evaluation loops add both: forward passes run under **autocast(fp16)** for speed and memory, and a **GradScaler** guards the fp16 activation gradients produced inside the autocast region from underflow. There is no dtype mismatch in the LoRA forward path — PEFT casts the activation to the adapter dtype and casts the result back, and autocast governs op precision. The identical precision path is centralized in `common/precision.py` and applied in training, evaluation, and every ablation fork, because KL, preference accuracy, and DPO loss are all differences of log-probs and a mismatched path between conditions would invalidate the comparison.

**Task 2 (PPO critic) precision deviation — measured, not claimed.** With the `[dtype-report]` diagnostic in `task2_ppo/continue_train.py`, immediately after both models load: the **policy** has 112/112 trainable tensors in fp32, and the **value model** has 96/97 in fp32. The single fp16 trainable tensor is the value head `base_model.model.score.modules_to_save.default.weight`. Cause: PEFT trains the value head via `modules_to_save` (not a LoRA adapter), i.e. a **copy of the fp16 base score head**, so it inherits fp16; the value LoRA adapters, like the policy's, load fp32 natively. GradScaler unscales gradients in place before the optimizer step and raises `ValueError("Attempting to unscale FP16 gradients.")` on any fp16 gradient. Fix (in `common/models.py` `load_value_model`): cast **all** trainable params to fp32 (`_cast_trainable_to_fp32`, general rather than hardcoding `score`), leaving the frozen fp16 base untouched, then assert no trainable param remains non-fp32. The same `_assert_trainable_fp32` guard is applied to `load_policy` (assert only, no cast — the policy is already fp32) so any future load that silently produces an fp16 trainable param fails at load time, not at the first `unscale_`. Cost: the value head is roughly `hidden_size × 1`, a few thousand parameters — negligible VRAM; the frozen base stays fp16 per config. This cast is also correct on its own terms: the value head regresses unbounded returns, where fp16's ~3 decimal digits of precision are inadequate.

## 8. Over-length prompt handling (DPO)

We raise `max_sequence_length` from the released 768 to **1152** and then filter residual over-length examples on **prompt length only**, identically across all conditions (standard run, the three beta forks, the length-balanced run, and both eval sets), recording the exact retained/dropped indices per run in `results/task1_dpo/<run_name>_filter.json`.

Rationale: at 1152 the length-stratified eval set has **zero** over-length prompts (longest is 1109), so the 82/82/82 stratum balance that Task 1 Step 3 depends on survives fully intact. Residual drops elsewhere are ~1%: 17/1500 standard train, 3/300 standard eval, 11/1500 length-balanced train. At the released 768 the stratified eval would lose 9 examples unevenly (3/4/2 across strata), which would compromise the length-confounding comparison. The filter is on prompt length only — never response length — so it cannot un-balance a stratified set, and the fork/smoke subset is drawn **after** filtering so `--max-examples N` always yields N clean examples with reproducible indices.

The generation cap at evaluation (`max_generation_tokens = 256`) is a **separate** config key, left unchanged, so response-length statistics remain comparable across conditions. `max_prompt_length` is set explicitly to 1152 (not derived from `max_sequence_length`) so that no prompt which passes the over-length filter is truncated during evaluation generation.

## 9. Code attribution

Portions of the training/evaluation/ablation scaffolding in this repository were implemented with coding assistance from an LLM (Anthropic Claude). All submitted code was reviewed, tested, and is understood by the author, who is responsible for every line. The PDF report is written entirely by the author without AI assistance.

## 10. Task 1 run sequence (ordered)

All runnable as `python -m ...` from the repo root; every run writes machine-readable output to `results/task1_dpo/`. The standard one-epoch run is already done.

```bash
# Training: three matched beta forks (600 clean examples each, same init/seed, ~6 min/fork on L4)
python -m task1_dpo.train --config configs/dpo.yaml --run-name beta_0.03 --beta 0.03 --max-examples 600
python -m task1_dpo.train --config configs/dpo.yaml --run-name beta_0.10 --beta 0.10 --max-examples 600
python -m task1_dpo.train --config configs/dpo.yaml --run-name beta_0.30 --beta 0.30 --max-examples 600
#   (equivalently, one command: python -m task1_dpo.ablate_beta --config configs/dpo.yaml)
# Training: length-balanced condition (full epoch, default beta, ~14 min)
python -m task1_dpo.train --config configs/dpo.yaml --run-name length_balanced --dataset data/dpo_length_balanced_train.jsonl

# Evaluation on the standard held-out set (generation-dominated; ~equal cost per run)
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard        --name standard        --beta 0.10
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/beta_0.03       --name beta_0.03       --beta 0.03
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/beta_0.10       --name beta_0.10       --beta 0.10
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/beta_0.30       --name beta_0.30       --beta 0.30
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/length_balanced --name length_balanced --beta 0.10

# Length-confounding analysis: per-stratum preference accuracy on the stratified eval + word-limit compliance
python -m task1_dpo.analyze_length --config configs/dpo.yaml --adapter outputs/task1_dpo/standard        --name standard
python -m task1_dpo.analyze_length --config configs/dpo.yaml --adapter outputs/task1_dpo/length_balanced --name length_balanced

# Aggregation, figure, qualitative dump (CPU, pure read)
python -m scripts.aggregate_task1 --config configs/dpo.yaml
python -m scripts.plot_task1 --config configs/dpo.yaml --name standard
python -m scripts.dump_task1_qualitative --config configs/dpo.yaml --name standard
```

Cost ordering: the five `evaluate` runs dominate (one 256-token generation per held-out prompt, ~297 prompts each); `analyze_length` is cheap (stratified pass is teacher-forced, plus 10 word-limit generations); aggregation/plot/dump are CPU-only. `evaluate.py` records `wall_seconds` and `sec_per_generation` in each `<name>_eval.json` so the measured per-example cost is available after the first run.

## 11. Task 2 run sequence (ordered)

All runnable as `python -m ...` from the repo root; every run writes machine-readable output to `results/task2_ppo/`. PPO continues from the supplied midpoint checkpoint, and **every short fork restarts from that same midpoint**. Held-out eval is greedy (`eval_do_sample: false` in `configs/ppo.yaml`); `evaluate.py` fails loudly if that key is unset. `continue_train.py` supports `--resume` (atomic `train_state.pt` for both policy and critic); the fork orchestrators skip any fork whose `<name>_summary.json` already exists.

```bash
# Validate the clipped-surrogate objective (CPU, seconds) BEFORE any GPU run
python -m scripts.verify_ppo_clip

# Standard 20-update continuation (peak VRAM + wall-clock -> standard_summary.json)
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard

# Clipping study: cached-rollout clip/affected fractions (clip_cached.json) + three matched
#   eps forks (kl_beta fixed at 0.10, 8 updates each). The (eps=0.20, kl=0.10) fork is shared
#   with the KL study and trains only once.
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml

# KL-pressure study: three matched beta forks (eps fixed at 0.20, 8 updates each);
#   the shared (eps=0.20, kl=0.10) fork is reused, not retrained.
python -m task2_ppo.ablate_kl --config configs/ppo.yaml

# Held-out evaluation (--name MUST match the adapter dir so results land in
#   results/task2_ppo/<name>_eval.json): standard + the five unique forks.
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard            --name standard
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/fork_eps0.05_kl0.10 --name fork_eps0.05_kl0.10
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/fork_eps0.20_kl0.10 --name fork_eps0.20_kl0.10
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/fork_eps0.50_kl0.10 --name fork_eps0.50_kl0.10
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/fork_eps0.20_kl0.00 --name fork_eps0.20_kl0.00
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/fork_eps0.20_kl0.20 --name fork_eps0.20_kl0.20

# Report artifacts: tables + figures in one run (CPU, pure read). aggregate_task2 also
#   (re)generates the figures unless --tables-only is passed; plot_task2 is the standalone
#   figures-only entry point. reward-vs-KL needs the six fork *_eval.json files above.
python -m scripts.aggregate_task2 --config configs/ppo.yaml --name standard
#   tables only, no figure regeneration:
#   python -m scripts.aggregate_task2 --config configs/ppo.yaml --name standard --tables-only
#   figures only (standalone):
#   python -m scripts.plot_task2 --config configs/ppo.yaml --name standard
python -m scripts.dump_task2_qualitative --config configs/ppo.yaml --name standard
```

`aggregate_task2.py` writes `task2_standard_summary.csv` (one-row continuation summary: updates, wall-clock, peak VRAM, first/last reward·KL·entropy·length, mean clip fraction, count of GradScaler-skipped updates), `task2_continuation.csv` (per-update trajectory), `task2_clip_table.csv` (cached clip/affected fraction + clipped surrogate joined to each eps fork's held-out reward/KL/length and `policy_loss_std_over_updates`), `task2_kl_table.csv` (per-beta held-out reward/KL/entropy/length, with `shared_with_clip_study` flagging the reused `eps=0.20, β=0.10` fork), and `task2_summary.json`. When the cache lacked advantages, `cached_affected_frac` is written as `NA` with a `cached_affected_note` naming why — never dropped, since it is Required Evidence. Figures: `task2_standard_trajectory.png`, `task2_standard_diagnostics.png`, `task2_reward_vs_kl.png` (reward-vs-KL drift/gain trade-off across the eps and beta forks).

`dump_task2_qualitative.py` reads `<name>_generations.jsonl` and emits `<name>_qualitative.json`: every held-out case with z-scored reward, z-scored length, and their difference (`disagreement`), carrying `source_index`. Nothing is thresholded or pre-selected — the author picks the reward-and-quality-agree and reward-and-quality-disagree cases by hand. (Run it for any policy by passing its `--name`, e.g. a fork name, not just `standard`.)

Every table/summary output records the KL convention string `sampled_per_token_mean(sum_tokens/sum_response_tokens)` where KL is reported, matching Task 1. Standard-continuation peak VRAM and wall-clock live in `standard_summary.json` and `task2_standard_summary.csv`.

Code attribution: portions of the Task 2 training/evaluation/ablation code were implemented with coding assistance from an LLM (Anthropic Claude); all submitted code was reviewed, tested, and is understood by the author, who is responsible for every line. The PDF report is written entirely by the author without AI assistance (see §9).

## 12. Task 3 run sequence (ordered)

All runnable as `python -m ...` from the repo root; every run writes machine-readable output to `results/task3_grpo/`. GRPO is critic-free: a single trainable policy, a frozen reference (the same adapter with LoRA disabled), and the frozen reward model — no value model. GRPO continues from the supplied midpoint checkpoint, and **both normalization forks restart from that same midpoint**. Held-out eval is greedy (`eval_do_sample: false` in `configs/grpo.yaml`, the shared Task 2/3 convention); `evaluate.py` fails loudly if that key is unset. `continue_train.py` supports `--resume` (atomic `train_state.pt` for the policy + optimizer + scaler + RNG); the fork orchestrator skips any fork whose `<name>_summary.json` already exists.

```bash
# Validate the group-relative advantage objective (CPU, seconds) BEFORE any GPU run
python -m scripts.verify_grpo_groups

# Standard 20-update continuation (peak VRAM + wall-clock -> standard_summary.json).
# Each update generates K=4 completions for one scheduled prompt, computes within-group
# advantages, masks max-length completions from the loss, applies the clipped GRPO objective
# with the k3 KL penalty, and logs reward / within-group reward std / uninformative-group
# fraction / KL / entropy / clip fraction / grad norm / response length.
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard

# Group-size study: equal-generation regrouping of the supplied K=8 completion/reward cache
# into K in {2,4,8}. No training, no model (CPU, seconds). Writes group_size_study.json +
# task3_group_size[_by_difficulty].csv.
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml

# Length-normalization study: two matched 8-update forks from the identical midpoint
# (canonical GRPO vs Dr. GRPO), then the controlled length-conditioned gradient diagnostic
# on the fixed K-cache. --diagnostic-only runs just the analytic part (no training).
python -m task3_grpo.compare_normalization --config configs/grpo.yaml

# Held-out evaluation (--name MUST match the adapter dir so results land in
#   results/task3_grpo/<name>_eval.json): standard + the two normalization forks.
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard     --name standard
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/norm_grpo     --name norm_grpo
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/norm_dr_grpo  --name norm_dr_grpo

# Report artifacts: tables + figures in one run (CPU, pure read). aggregate_task3 also
#   (re)generates the figures unless --tables-only is passed.
python -m scripts.aggregate_task3 --config configs/grpo.yaml --name standard
#   figures only (standalone): python -m scripts.plot_task3 --config configs/grpo.yaml --name standard
python -m scripts.dump_task3_qualitative --config configs/grpo.yaml --name standard
```

`aggregate_task3.py` writes `task3_standard_summary.csv` (one-row continuation summary: updates, K, wall-clock, peak VRAM, first/last reward·KL·entropy·length, mean within-group reward std, mean uninformative-group fraction, mean clip fraction, count of GradScaler-skipped updates), `task3_continuation.csv` (per-update trajectory), `task3_group_size.csv` + `task3_group_size_by_difficulty.csv` (informative-group fraction, within-group reward std, group-relative-signal variance for K∈{2,4,8}, overall and per difficulty tercile), `task3_normalization.csv` (canonical vs Dr. GRPO held-out reward/KL/length/entropy + `policy_term_std_over_updates` + the length-conditioned gradient statistic), and `task3_summary.json`. Figures: `task3_standard_trajectory.png`, `task3_standard_diagnostics.png`, `task3_group_size.png`, `task3_normalization.png`.

**GRPO objective defect.** The core defect in `task3_grpo/grpo.py` was that `group_relative_advantages` normalized **globally**, ignoring `group_ids`; it now standardizes **within each prompt group** and uses the manual's additive denominator `σ_r + ε` (`A_k = (r_k − μ_r)/(σ_r + ε)`), validated by `scripts/verify_grpo_groups.py` (within-group mean-zero, constant group → zero advantage, shift invariance). The reported KL uses the shared `sampled_per_token_mean(...)` log-ratio convention for cross-task comparability; the k3 estimator actually added to the objective (`exp(Δ)−Δ−1`) is logged separately as `kl_penalty_k3`.

**K-cache schema (confirmed)** `source_index:int, prompt_id:str, generation_index:int, completion:str, completion_tokens:int, terminated_with_eos:bool, clipped_at_max:bool, reward:float` — 192 rows, 24 prompts × 8 completions, no missing values. The group-size and normalization diagnostics use `reward` and `completion_tokens` directly (no field guessing) and compute the group-relative signal **locally** as a reconstruction of `A_k = (r_k − μ_r)/(σ_r + ε)` (`advantage_eps = 1e-6`), deliberately decoupled from the training helper so the analysis is correct regardless of it.

**Normalization diagnostic is an analytic illustration.** The length-conditioned gradient diagnostic (`norm_diagnostic.json`, `task3_norm_diagnostic.csv`) scores the fixed K-cache under both normalization formulas; because the K-cache comes from a different policy snapshot than `norm_grpo`/`norm_dr_grpo`, it shows what the two normalizers do to a fixed completion-length distribution — it is **not** gradient data from the actual forks (those supply held-out reward/KL/length via `evaluate.py`). This is flagged in both outputs. Clipped (`clipped_at_max`) completions are **included** in the illustration (the long tail is where the normalizers differ most; Step 1 masks them from the training loss only); degenerate groups (within-group reward std ≤ tolerance, e.g. the all-identical prompt) are **excluded** (zero advantage → zero gradient), with counts reported. The Dr. GRPO constant uses `cache_generation_cap` and cancels in every reported correlation/ratio.

Every table/summary output records the KL convention string `sampled_per_token_mean(sum_tokens/sum_response_tokens)` where KL is reported, matching Tasks 1–2. Standard-continuation peak VRAM and wall-clock live in `standard_summary.json` and `task3_standard_summary.csv`.

Code attribution: portions of the Task 3 training/evaluation/ablation code were implemented with coding assistance from an LLM (Anthropic Claude); all submitted code was reviewed, tested, and is understood by the author, who is responsible for every line. The PDF report is written entirely by the author without AI assistance (see §9).

## 13. Task 4 run sequence (ordered)

All runnable as `python -m ...` from the repo root; output goes to `results/task4_safety/`. Four **fixed** policies, no substitutions: **SFT** = untouched `Qwen2.5-1.5B-Instruct` (no adapter), **DPO** = `outputs/task1_dpo/standard`, **PPO** = `outputs/task2_ppo/standard`, **GRPO** = `outputs/task3_grpo/standard`. `generate.py`'s `policy_adapter()` rejects any adapter path not ending in `standard`, so a β-sweep / clipping-KL fork / normalization fork cannot be substituted. Decoding is deterministic greedy for all four (`eval_do_sample: false` in `configs/feedback.yaml`, `max_new_tokens = safety_max_new_tokens = 256`), one response per prompt; `generate.py` fails loudly if `eval_do_sample` is unset, exactly like `task3_grpo/evaluate.py`. All generation/judging is resumable (append per-prompt JSONL, skip ids already present), so a Colab disconnect costs only the current batch.

```bash
# 1. Generate SFT responses FIRST (needs no Task 1-3 adapter) so hand-labelling can start while
#    the DPO/PPO/GRPO adapters are still training.
python -m task4_safety.generate --config configs/feedback.yaml --policies sft

# 2. Emit the fixed, BLIND 60-example audit worksheet (30 SAFE + 30 UNSAFE SFT responses;
#    prompt + response + empty manual_label; NO AI label shown). Hand-label manual_label with
#    one of SAFE_ANSWER / JUSTIFIED_REFUSAL / UNSAFE_COMPLIANCE / OVER_REFUSAL / AMBIGUOUS.
python -m task4_safety.sample_audit --config configs/feedback.yaml

# 3. Generate the remaining three policies once their standard adapters exist.
python -m task4_safety.generate --config configs/feedback.yaml --policies dpo ppo grpo
#    (or all four at once: python -m task4_safety.generate --config configs/feedback.yaml)

# 4. Run the fixed AI judge over every policy's responses (released judge/prompt/parser; the
#    five labels only; confidence stored for auditing, never aggregated).
python -m task4_safety.judge --config configs/feedback.yaml

# 5. Score the manual audit against the judge (after BOTH hand-labelling and judging SFT).
python -m task4_safety.score_audit --config configs/feedback.yaml

# 6. Aggregate report tables (CPU, pure read).
python -m scripts.aggregate_task4 --config configs/feedback.yaml
```

`generate.py` writes `generated_<policy>.jsonl` (+ `generation_meta.json`); `judge.py` writes `judged_<policy>.jsonl` (+ `judge_meta.json`); `sample_audit.py` writes `manual_audit_worksheet.csv` (blind) + `manual_audit_ids.json`; `score_audit.py` writes `manual_audit_agreement.json` + `manual_audit_confusion.csv` (5×5 manual-vs-judge). `aggregate_task4.py` writes `task4_safety_rates.csv` (per policy: paired `safe_answer_rate`/`over_refusal_rate` over SAFE prompts, `unsafe_compliance_rate`/`justified_refusal_rate` over UNSAFE prompts, overall `ambiguous_rate`, `mean_response_length`), `task4_category_distribution.csv` (label counts per XSTest `type` for all four policies), and `task4_summary.json` (folds in the manual-audit confusion when present). The paired safe/unsafe rates deliberately separate genuine reduction in harmful compliance from indiscriminate refusal, which a single refusal metric would conflate.

These entry points (`generate.py`, `judge.py`, `sample_audit.py`, `score_audit.py`) implement the Task 4 evaluation on top of the released utilities: `judge.py` reuses `judge_responses.load_judge`/`judge_one`/`JUDGE_PROMPT` unmodified (no student-authored judge prompt), and `sample_audit.py` reuses `make_audit_sheet.fixed_audit_ids` for the identical reproducible subset.

Code attribution: portions of the Task 4 evaluation/aggregation code were implemented with coding assistance from an LLM (Anthropic Claude); all submitted code was reviewed, tested, and is understood by the author, who is responsible for every line. The AI safety judge, its prompt, and its parser are course-supplied. The PDF report is written entirely by the author without AI assistance (see §9).
