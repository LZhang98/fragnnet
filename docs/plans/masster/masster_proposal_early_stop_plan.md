# MASSter proposal early stop and proposal-only weight decay

> **Status:** CLOSED NULL · **Updated:** 2026-09-23
> `p1` freeze@32 (+0.0010) and `p3` proposal wd 0.1 (+0.0020) are cosine nulls at the 128-epoch screen against 0.7270 (gate +-0.005); `p2` was never submitted. `p1` showed the shared encoder moving under frozen heads, so a clean freeze needs `separate_proposal_encoder`; read [Results](#results-p1-and-p3-at-the-128-epoch-screen-2026-09-23) before proposing it.

## Objective

Let the proposal branch and the intensity branch stop training at different epochs, or carry
different weight decay, because the
[2026-09-22 overfit read-out](../../findings/masster/masster_overfit_readout_2026_09_22.md) measured
that they overfit on different schedules and the training loop gives them one schedule.

### Acceptance criteria

1. A configuration can early-stop the proposal branch at a chosen epoch, or give it its own
   weight decay, **without changing what the spectrum objective optimizes**. Verified by
   gradient test, not by inspection.
2. Both knobs default off, and off is bit-identical to the previous behavior — a single
   optimizer parameter group and a proposal branch that trains to the last epoch.
3. Each prepared arm resolves to **exactly one** key against `c4a_base_s1`.

## Established facts

| Fact | Source |
|---|---|
| The validation **proposal** term degrades +0.06 to +0.14 and turns at **epochs 26-47**; the spectrum term turns at **52-120** | [overfit read-out §3](../../findings/masster/masster_overfit_readout_2026_09_22.md) |
| The proposal branch is **44.8% of the parameters** — 4,128,642 of 9,217,181 at `c4a_base_s1` (`BondFocusedBreakHead` 3,865,473 + `AtomChargeAffinityHead` 263,169) | measured 2026-09-22, this work |
| Under `proposal_selection_only: true` the proposal reads `.detach()`ed atom embeddings and context, so `proposal_loss_weight: 0.1` updates the proposal heads **and nothing else** | [head-input asymmetry §4.4](../../findings/masster/masster_head_input_asymmetry_2026_09_22.md) |
| `weight_decay` is `0.01` and identical in **all** 232 `masster` runs; it has never been swept per branch | [regularization audit](../../findings/masster/masster_regularization_audit_2026_09_19.md) |
| Coverage is **not** the gap: train `wrecall` 0.8975 vs val 0.8911, worth ~0.0027 of cosine against a 0.151 train/val cosine difference | [overfit read-out §4.1](../../findings/masster/masster_overfit_readout_2026_09_22.md) |

The detach is what makes this a clean intervention rather than a joint-training change. It is
also why the existing staged-training knob is not the answer: `masster_proposal_only_until_epoch`
trains the proposal **first** and then joins the branches, which is the opposite schedule.

### Why not proposal warm-up instead — measured 2026-09-22

Asked directly: freeze, or warm up and lean on weight decay? The warm-up half is answered, and
the answer is no, for three reasons that are each independently sufficient.

**1. The three `c4c_pwarm` arms read at the matched screen for the first time.** They ran on
630-epoch configs and were killed at the wall clock near epoch 205, so their W&B summaries
(0.7316 / 0.7301 / 0.7181) are last-logged values at unmatched epochs and had never been placed
against the control. Re-read as the tail mean of epochs 118-127, same method as the
[epoch-128 read-out](../../findings/masster/masster_architecture_arms_epoch128_2026_09_22.md), against
the same control (`c4a_base_s1` 0.7266, `c4a_base_s2` 0.7273, mean **0.7270**, gate ±0.005):

| arm | `proposal_only_until_epoch` | tail mean 118-127 | vs control |
|---|---:|---:|---:|
| `c4c_pwarm_e2` (`1xl29zka`) | 2 | 0.7278 | +0.0008 |
| `c4c_pwarm_e4` (`7h1wx0ft`) | 4 | 0.7282 | +0.0012 |
| `c4c_pwarm_e15` (`c25ayosq`) | 15 | 0.7174 | **−0.0096** |

Two nulls and one rejection, non-monotone — the same shape as the merged-lineage warm-up sweep
the [campaign read-out](../../findings/masster/masster_campaign_readout_2026_09_21.md) called "measured
nothing", now with a cost that appears once the warm-up is long. The horizon does not confound
it: all four runs carry `lr_schedule: false` at a constant `lr: 1e-3`, so nothing reads
`max_epochs`, and every arm shares seed 397623 with `c4a_base_s1`. `c4c_pwarm_e9` was configured
and never submitted.

**2. The key does not do only what its name says.** `_apply_staged_training` calls
`set_proposal_input_gradients_enabled(proposal_only or proposal_encoder is None)`, and
`proposal_encoder is None` in every `ma_mi_merged` config, so setting the key to any integer
un-detaches the proposal branch for the **whole** run, not just the warm-up epochs. The one
clean same-seed test of that rewiring measured **−0.0113 at epoch 100 and −0.0132 at epoch 400**
([head-input asymmetry §7.1](../../findings/masster/masster_head_input_asymmetry_2026_09_22.md)). Every
warm-up arm above is therefore confounded with a change already measured as harmful, and a clean
warm-up test would first need `separate_proposal_encoder: true`.

**3. It is aimed at the wrong end of the run.** The defect is late: the proposal turns at epochs
26-47 and keeps degrading. Warm-up adds proposal-only training at the **start** and freezes the
intensity head while it does — which is the most likely reason `e15` is the worst of the three.
It treats initialization, not late overfitting.

None of this argues against the weight-decay half, which is `p3` and is independent of the
warm-up. The freeze and the proposal decay can also be combined; only freeze-plus-warm-up is
rejected at setup.

## What landed

Both keys are in [config/template.yml](../../../config/template.yml) and default to `null`.

### `masster_proposal_frozen_from_epoch`

From this zero-indexed epoch, `MassterPL._apply_proposal_early_stop` sets the proposal-only
modules to `requires_grad_(False)`, clears their gradients and puts them in evaluation mode,
reusing the existing `_set_modules_trainable` helper. The set comes from the new
`_proposal_branch_modules`, which is the existing `_proposal_modules` **minus** the two encoders
the intensity branch shares — `precursor_encoder` and `context_encoder` are never frozen, or the
spectrum objective would lose its encoder mid-run.

The loss expression is deliberately **unchanged**. Because selection-only mode detaches the
proposal's inputs, freezing these heads removes the proposal term's last trainable leaf, so the
term contributes no gradient anywhere and the logged `proposal_loss` stays comparable across the
boundary. A gradient test asserts exactly that, on both sides of the boundary.

Rejected at setup: a non-integer or negative epoch; `proposal_selection_only: false` (outside
selection-only mode the proposal probabilities multiply into the spectrum, so the primary loss
would keep pulling on the frozen heads and the flag would stop meaning "early stop one
objective"); a non-positive `proposal_loss_weight`; and combining it with
`masster_proposal_only_until_epoch`, which is an untested schedule, not a supported one.

### `masster_proposal_weight_decay`

`SpectrumPL.configure_optimizers` now asks `_weight_decay_parameter_groups()` for its parameter
groups. That helper returns `None` — and the optimizer keeps its historical single group —
unless `weight_decay_exclude_norm_bias` is on or a subclass reports per-parameter overrides
through the new `_weight_decay_overrides()` hook. `MassterPL` implements the hook over the
same `_proposal_branch_modules()` set. When both are on, the norm/bias exemption wins: a
decay-exempt tensor stays exempt whichever branch it belongs to.

### Prepared arms — none submitted

[config/masster/ma_mi_merged/proposal_early_stop_2026_09_22/](../../../config/masster/ma_mi_merged/proposal_early_stop_2026_09_22),
each a verified one-key diff from `c4a_base_s1` at 128 epochs, seed 397623, and
parameter-identical to it (9,217,181):

| arm | key | value | why this value |
|---|---|---|---|
| `p1_proposal_freeze_e32` | `masster_proposal_frozen_from_epoch` | 32 | early end of the measured 26-47 turn window |
| `p2_proposal_freeze_e64` | `masster_proposal_frozen_from_epoch` | 64 | well after it; `p1` vs `p2` brackets the turn |
| `p3_proposal_wd10` | `masster_proposal_weight_decay` | 0.1 | 10x global, the value `c4d_wd10` applies to the whole model |

`p1` against `p2` is the informative pair. If `p1` beats `p2`, the stopping epoch is doing the
work; if they tie each other, the epoch does not matter; if both tie the control, the proposal's
late drift never cost cosine.

## Validation

- `pytest tests/test_masster_pl.py tests/test_lr_schedule_forwarding.py
  tests/test_masster_attention_and_budget.py -q` — 198 passed before the config test, then the
  17 new/selected proposal cases pass; full suite run clean.
- `ruff check` clean on all four changed Python files. Black was **not** run: the three files it
  wants to reformat were already not Black-clean at `HEAD`, so running it would rewrite unrelated
  code against the surgical-change rule.
- New coverage: invalid-value and incompatible-config rejection for both keys; the proposal
  branch excludes the shared encoders; a forward/backward gradient test showing the proposal
  objective trains the branch before the boundary and updates **nothing** — not the heads, not
  the encoders — from it, while the spectrum objective still reaches the intensity head and both
  encoders; single-group-when-unset; group membership when set; and the norm/bias precedence.
- Configs: all three arms resolve to one non-metadata key against `c4a_base_s1` and construct.

## Limitations to carry into the read

1. **The freeze does not freeze the proposal function.** Only proposal-only modules stop
   updating; the shared molecular and context encoders keep training on the spectrum objective,
   and the proposal reads them. A genuinely static proposal needs
   `masster_separate_proposal_encoder: true`, which is off in every `ma_mi_merged` config.
   These arms therefore test "stop updating the proposal *heads*", which is the 44.8% of
   parameters the proposal loss owns, not "hold the candidate set fixed".
2. **Freezing also drops head dropout.** The frozen modules go to evaluation mode, and the
   proposal heads carry `head_dropout: 0.1` (their normalization is LayerNorm, so nothing else
   changes). Training-time candidate selection therefore becomes deterministic after the
   boundary. That is a second, unseparated change in the same arm — and plausibly a *help*,
   given the [regularization audit's](../../findings/masster/masster_regularization_audit_2026_09_19.md)
   train/eval candidate-set divergence. `p3` has no such side effect, which is one more reason
   to run it alongside the freeze arms rather than instead of them.
3. **The premise predicts a null.** Coverage is already flat train-to-val, so if these arms move
   cosine it is because the candidate set's *composition* improved — fewer hard distractors among
   the 256 retained fragments — not its coverage. Log `val_wrecall` next to cosine: flat coverage
   with moved cosine is the mechanism; both flat is a null, and it would say the proposal term's
   late degradation was calibration the ranking metric never saw, exactly as §3 of the read-out
   argues for the spectrum term.

## Results: p1 and p3 at the 128-epoch screen (2026-09-23)

Read-only W&B query, no GPU. Both runs finished all 128 epochs, both carry seed 397623 (the same
seed as `c4a_base_s1`), and both were read by the family's method: the tail mean of epochs
118-127 against `c4a_base_s1` `5qws8g9x` and `c4a_base_s2` `6plumx6r`.

| run | cosine | vs control 0.7270 | val proposal loss | train proposal loss | `val_wrecall` | val primary loss |
|---|---:|---:|---:|---:|---:|---:|
| `c4a_base_s1` `5qws8g9x` | 0.7266 | — | 2.2769 | 1.7709 | 0.8911 | −2.7251 |
| `c4a_base_s2` `6plumx6r` | 0.7273 | — | 2.2776 | 1.7778 | 0.8912 | −2.7264 |
| `p1` freeze@32 `s5i5qqlz` | 0.7280 | **+0.0010** | **2.2546** | **2.0166** | 0.8888 | −2.7166 |
| `p3` proposal wd 0.1 `w1dj2ahw` | 0.7290 | **+0.0020** | 2.2845 | 1.7651 | 0.8912 | −2.7318 |

**Both are cosine nulls.** The seed-matched deltas against `c4a_base_s1` are +0.0014 and
+0.0024, both inside the ±0.005 gate. `p2` is not worth a box: it was only informative if `p1`
showed a sign.

**`p3` barely touched the proposal.** Its train proposal loss (1.7651) matches the control
(1.77), and its validation proposal loss is the worst of the four. At `lr: 1e-3`, a 0.1 decay did
not constrain the fit.

**`p1` measured the shared encoder drifting under the frozen heads.** From epoch 32 nothing the
proposal owns updates, yet its **train** proposal loss rises in every 16-epoch block after the
boundary: 1.940, 1.951, 1.965, 1.980, 1.998, 2.014. That is +0.082 from the epoch 32-35 mean to
the tail. The first-epoch drop (1.974 → 1.934 at epoch 32) is the head dropout switching off
(limitation 2). The rise after it can only come from `precursor_encoder` and `context_encoder`,
which keep training on the spectrum objective. Limitation 1 predicted this, and here is its
size. Relative to the epoch 28-35 mean:

| run | Δ val proposal loss | Δ `val_wrecall` |
|---|---:|---:|
| `c4a_base_s1` | +0.041 | +0.0021 |
| `c4a_base_s2` | +0.050 | +0.0023 |
| `p1` | **+0.016** | **−0.0004** |

Freezing the heads removes about two thirds of the late validation-proposal degradation. The
remaining third is encoder drift. It also forfeits the +0.002 coverage gain that the control's
heads earn by continuing to track the encoder. None of this reaches cosine.

**What this does and does not say about a separate proposal backbone.** It shows that, with a
shared encoder, the proposal cannot be held fixed or scheduled independently. A clean freeze
therefore needs `masster_separate_proposal_encoder: true`. It does **not** show that the
shared encoder costs cosine, for three reasons:

1. **Proposal loss and cosine are decoupled across these four runs.** `p3` has the worst
   validation proposal loss and the best cosine; `p1` has the best proposal loss and the second
   best. Removing a drift that cosine does not see is not a lever on cosine.
2. **The one direct test is a null.** `esb2fqhg` (separate encoder) against `d7s0yb72` (shared)
   is the 2026-08-21 merged pair from the [separate proposal encoder
   ablation](masster_plan.md#separate-proposal-encoder-ablation-2026-08-23). The two runs share
   seed 397623, and their resolved W&B configs differ only in
   `masster_separate_proposal_encoder`. Both crashed, at epochs 136 and 143. Tail mean over
   the last ten common epochs (128-137): 0.7409 against 0.7417, **−0.0008**. Validation proposal
   loss matched too (3.236 against 3.257 at epoch 135). The pair predates the 2026-09-03 target
   change, uses `gine_efa384` with learned OOS, and was not previously recorded. Take its sign as
   the prior, not its value.
3. **The shared part is small.** The proposal already has its own message passing:
   `BondFocusedBreakHead` runs three bond-centric update layers (3,865,473 parameters). What it
   shares is the two-layer atom encoder (1,808,256 parameters, per the
   [regularization audit §2.5](../../findings/masster/masster_regularization_audit_2026_09_19.md)).
   A separate encoder adds a second copy of that, +20% on 9,217,181. It is also detached today, so
   the proposal is a readout of a spectrum-trained encoder. Under V3 the proposal loss would train
   its own encoder, which is the substantive change. The context vector would still be shared
   ([head-input asymmetry §4.6](../../findings/masster/masster_head_input_asymmetry_2026_09_22.md)).

## Next concrete check

Nothing in this family is queued. If the separate-encoder question is worth a box, the test is
one key: `masster_separate_proposal_encoder: true` against `c4a_base_s1` at 128 epochs, read
by tail mean with `val_wrecall`. Add a second arm with the freeze on top only if a truly static
candidate set is itself the question. The prior from `esb2fqhg`/`d7s0yb72` is null. The
`a1_prenorm` replicate and the `d3` LR-schedule replicate are the live candidates, and both
rank ahead of it.
