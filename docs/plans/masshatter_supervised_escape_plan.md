# MASShatter supervised escape channel

> **Status:** IMPLEMENTED, SCREENED NULL · **Updated:** 2026-09-22
> Design of record for giving MASShatter somewhere to put intensity it cannot place on a
> candidate. **Two escapes, two targets, two purposes.** The *intensity* escape (§3) is the one
> that changes training: `wrecall` is 0.891, so 10.9% of observed intensity sits at m/z values
> no candidate reaches, and a predicted distribution normalized over candidates alone is forced
> to over-allocate everything it *can* reach. **It is fully implemented end to end as of
> 2026-09-22** behind `mass_shatter_intensity_escape`, default off. **It has now been screened
> and is an exact null so far**: `e1_escape` vs `e0_independent`, two seeds each, read at their
> common epoch 118 while still running — 0.7225 against 0.7225, **Δ +0.0000**, seed spreads
> 0.0001 and 0.0003. The tightest null in this family. See the
> [epoch-128 read-out §2](../findings/masshatter_architecture_arms_epoch128_2026_09_22.md);
> re-read at completion.
> The *contaminant* escape (§4) is `predict_oos` supervised by the subformula test, and remains
> unimplemented.
> Sizing comes from [the unexplained-peak split](../findings/unexplained_peak_split_2026_09_22.md).

## 1. Why supervise it

FraGNNet already carries an escape: `predict_oos`, a per-spectrum sigmoid head whose output
shifts every fragment log-probability by `log(1 - p_oos)`. It is on in **466 of 466** FraGNNet
runs and has never been ablated, so its value is unmeasured. MASShatter hard-disables it in
three places even though the head exists in its own head module:

| site | what it does |
|---|---|
| `bond_fragment_heads.py:678` | builds `oos_head = nn.Linear(model_dim, 1)`, zero weight, bias `logit(0.1)` |
| `mass_shatter_model.py:1508` | passes `predict_oos=False`, hard-coded |
| `mass_shatter_pl.py:894-914` | force-disables the hparam, strips `true_oos_prob`/`true_oos_e` |
| `mass_shatter_pl.py:1556` | discards the `oos_ce` that `sparse_cross_entropy_seq` returns |

FraGNNet's OOS target is implicit: whatever true intensity its predicted support did not
match. That conflates two different failures. **73% of MASShatter's unmatched intensity is a
real analyte fragment the enumeration missed**, and supervising an escape on that teaches the
model to give up on chemistry it should be learning.

The subformula test separates them model-free. It is what this plan supervises on.

## 2. What the escape is worth

Measured on the merged MA/MI val split, 15,257 spectra:

| population | intensity | treatment |
|---|---|---|
| coverage: feasible formula, no candidate | 0.0798 | **not** an escape problem |
| isotopologues of explainable peaks | 0.0135 | isotope handling, deferred |
| above the precursor | 0.0090 | **filtered** as of 2026-09-22 |
| genuine contaminant / noise | **0.0156** | **what this escape addresses** |

(The isotopologue and above-precursor rows overlap by 0.0065, the precursor's own ladder,
which the new filter now removes.)

So the escape's addressable population is **1.56% of target intensity**. This plan should not
be sold as a cosine win, and §4 explains why it provably is not one.

## 3. The intensity escape (dump)

### 3.0 Why it is needed, and what it actually fixes

`wrecall` is 0.8911, so for the average spectrum 10.9% of the observed intensity lies at m/z
values the candidate set does not reach. The predicted intensity vector normalizes over
candidates and therefore sums to 1 across *reachable* positions only, while the observed
vector spends 10.9% of its mass elsewhere. The model cannot place that mass, so it
systematically inflates everything it can place by roughly `1/0.891`.

Working the optimum out exactly, for Hellinger on a spectrum with reachable mass `t(S)`:

- **without an escape**, the best achievable Bhattacharyya coefficient is `sqrt(t(S))`, so the
  loss has an irreducible floor of `1 - sqrt(t(S))`;
- **with an escape**, the optimum is `p = 1 - t(S)` on the dump and the loss reaches 0.

In both cases the optimal *relative* allocation over the reachable peaks is the same,
`q_i ∝ t_i`. **So the escape does not change the ranking the model should learn.** What it
removes is a per-spectrum loss floor that varies with how well the enumeration happened to do
on that molecule — which currently weights the training signal by coverage, spending gradient
on a gap no model output can close. It also makes `kl_distance` finite on a coverage gap
instead of relying on `kl_epsilon` smoothing.

This corrects an argument in an earlier draft of this plan: supervising an escape on coverage
failure does **not** teach the model to "give up" on chemistry, because coverage is fixed by
enumeration and is not something any model output can influence.

### 3.1 Target and implementation status

The dump target is the unreachable observed mass, read from the tolerance `match_mask` that
the loss already computes — **not** from the predicted values, and **not** from a precomputed
column. A peak a candidate could explain stays in the support even when the model scores it
zero or a Hungarian assignment hands it to a neighbour. This dissolves the earlier "where is
the target computed" decision entirely: it is available at batch time from machinery that is
already there, with no `spec_df` column, no subformula enumeration and no merge-ordering
problem.

**Implemented end to end (2026-09-22), behind `mass_shatter_intensity_escape`, default off.**

*Loss.* `sparse_peak_list_distance` takes `pred_dump_intensities: th.Tensor | None`, shape
`(batch_size,)`, in the same units as `pred_intensities`. When given, every unreachable
observed peak is collapsed into one appended coordinate carrying that mass, the model's dump
prediction is appended to the predicted vector, and all four objectives plus the entropy term
run unchanged on the extended vectors. `None` reproduces the previous behaviour bit-identically.

*Model.* `MassShatterModel` takes `intensity_escape` and passes it to the intensity head's
existing `predict_oos`, which was previously hard-coded `False`. The head emits one logit per
spectrum from its mean-pooled molecule token. The model converts it to a mass and returns
`pred_escape_intensities`.

*Parameterization.* `sigmoid(logit)` is the escape's share of `candidates + escape`, so the
mass it needs is that share's odds times the candidate mass — and the odds of a sigmoid are
just `exp(logit)`, so the model computes `escape = candidate_mass * exp(logit)` with the logit
clamped at 9. Going through `sigmoid` and back was the first implementation and was **wrong
under reduced precision**: fp16 and bf16 round any guard of the form `1 - 1e-4` to exactly
`1.0`, so the share saturated and the odds became `inf`. Clamping the logit bounds the odds at
~8.1e3, which fp16 holds comfortably. The head's bias initializes at `logit(0.1)`, so training
starts with the escape holding 10% of the spectrum — close to the measured unreachable fraction
of 0.109 — and a test asserts that.

*Guards.* `MassShatterPL` raises at setup when the escape is requested with
`mass_shatter_intensity_output: probability` or a cross-entropy `loss_type`, rather than
silently ignoring it. Silent override is exactly how `predict_oos` went unnoticed on this model
for months, so the flag fails loudly instead. Note the escape is reachable only in
unique-fragment, proposal-selection-only mode, since that is the only mode `independent`
intensity output supports.

*Warm start.* A checkpoint trained before the escape existed carries no `oos_head`, so
`_load_from_state_dict` now restores the constructor's initialization for it, matching what the
neighbouring `ion_h_support_head` already did. Without this, switching the escape on over an
existing base checkpoint failed with a missing-key error and forced a run from scratch.

*Also fixed:* the head's molecule pooling divided by a component count that can be zero. That
path had never executed, because `predict_oos` was hard-coded off at its only call site.

*Monitoring.* `escape_share` is logged per spectrum whenever the escape is on. Without it the
escape is invisible: the loss alone cannot say whether the head learned the coverage deficit or
parked at its initial 0.1. This model family has repeatedly shipped arms whose key quantity was
never logged, so the metric lands with the feature.

### 3.2 Smoke result (2026-09-22)

`config/debug/debug_tiny_mass_shatter_intensity_escape.yml` and its matched control differ only
in the flag. 64 toy spectra, 8 epochs, `detect_anomaly: true`, local RTX 2070S, both exit 0.
**This is a smoke run: it checks the path trains and the mechanism behaves, and measures
nothing about quality.**

| train metric @ ep7 | escape ON | control OFF | delta |
|---|---|---|---|
| `spec_peak_distance` (Hellinger) | 0.3616 | 0.7464 | **−0.3848** |
| `spec_entropy_match` | 1.8840 | 2.5065 | −0.6226 |
| `escape_share` | **0.5911** | — | — |
| `wrecall` | 0.4717 | 0.4717 | +0.0000 |
| `recall` | 0.2808 | 0.2808 | +0.0000 |
| `precision` | 0.0778 | 0.0778 | +0.0000 |
| `cos_sim_0.01` | 0.2022 | 0.2236 | **−0.0214** |

Three things this establishes and one it warns about.

1. **The head learns the coverage deficit.** It starts at 0.1 by construction and reaches
   `escape_share` 0.591 against a true unreachable fraction of `1 - wrecall` = 0.528. It is
   tracking the right quantity, not drifting.
2. **The escape does not touch coverage.** `wrecall`, `recall` and `precision` are identical to
   the last digit, which is the control this design requires: the escape must absorb intensity,
   never change which peaks are reachable.
3. **The loss floor is what moved.** Hellinger halves. The floor predicted by
   `1 - sqrt(t(S))` at this coverage is 0.313, the right order for the 0.385 observed; neither
   arm has converged, so the numbers are not expected to match exactly.
4. **The warning: the loss drop is not a quality gain.** With the escape the objective includes
   a coordinate the model can trivially get right, so a lower number is guaranteed and means
   nothing on its own. **Cosine, the one metric whose definition did not change, went slightly
   down.** On 64 toy spectra at 8 epochs that is noise, but it is exactly what §5 predicts:
   the escape is not a ranking change. Any real comparison must be on cosine at matched epoch
   with replicates, never on the loss.

### 3.3 Merged arms (prepared 2026-09-22, not submitted)

`config/massshatter/ma_mi_merged/intensity_escape_2026_09_22/`, derived from the loss-objective
arm `h1_hellinger_closest.yml` (run `oykkffls`, val cosine 0.7247 @ep121).

**Why a ladder rather than one arm.** `h1` deliberately keeps
`mass_shatter_intensity_output: probability` to avoid the CE-versus-direct-loss bundle confound,
and the escape requires `independent`. Turning the escape on against `h1` would move two keys at
once, so the campaign takes two one-knob steps:

| arm | output | escape | seeds | reads against |
|---|---|---|---|---|
| `h1` (already run, `oykkffls`) | probability | off | 397623 | — |
| `e0_independent_s1/s2` | **independent** | off | 397623, 811237 | `h1` |
| `e1_escape_s1/s2` | independent | **on** | 397623, 811237 | `e0` |

Verified by resolving each config and constructing the model: the resolved-config difference
between `e0` and `h1` is exactly `mass_shatter_intensity_output`, and between `e1` and `e0`
exactly `mass_shatter_intensity_escape` (plus the intended name, tags and seed). The escape adds
**257 parameters** — one 256-wide linear plus bias — to a 9,217,181-parameter model, and one
appended coordinate per spectrum in the loss, so the memory shape is unchanged and the
`fir:h100-2g.20gb` profile this lineage is sized for still applies.

n=2 seeds per arm because this family's replicate floor is 0.003-0.005 and an n=1 arm cannot
resolve what it produces. 128 epochs, matching `h1` and the 2026-09-21 saturation finding.

Generation: `scripts/manual_slurm/fir/submit_masshatter_intensity_escape.sh [ladder|control]`,
which generates without submitting and validates configs and wheels first. Syntax-checked; no
job generated into `slurm_scripts/` and nothing submitted.

**How to read it.** Select on `val_cos_sim_0.01_epoch/mean` only. `spec_peak_distance` is **not**
comparable across `e0`/`e1`: the escape adds a coordinate the model can trivially get right, so
the loss is guaranteed to fall and a drop means nothing. Check that `escape_share` left its 0.1
initialization and tracks `1 - wrecall` (about 0.109 on this split). If `wrecall`, `recall` or
`precision` differ between `e0` and `e1`, that is a bug, not an improvement.

**Not yet done:** no real training run; the four arms above are prepared and unsubmitted. The
objective changes with this flag, so loss values are not comparable across it — see §7.

## 4. The contaminant escape (`predict_oos`)

### 4.1 Target

Per spectrum, `oos_target` = intensity fraction carrying no precursor-bounded,
nonnegative-RDBE subformula at instrument tolerance — the same test as
`filter_subformula_peaks`, measuring instead of filtering.

It composes correctly with the filters: with `filter_subformula_peaks: True` the target goes
to ~0 and the escape is inert, which is the right behaviour, since the population has already
been removed from the target.

### 4.2 Model

Thread `predict_oos` from config through `mass_shatter_model.py:1508` instead of the hard-coded
`False`, consume the head's extra return value, and emit `pred_oos_logprobs`. Stop the
force-disable in `MassShatterPL`. The head itself needs no change.

### 4.3 Loss

The escape enters as an explicit supervised term, **not** through `sparse_cross_entropy_seq`'s
existing `oos_ce`, whose target is the coverage residual this plan is specifically avoiding.

## 5. What each escape will and will not do

**It will not move `cos_sim`, and that is provable rather than pessimistic.** Under
cross-entropy with an escape,

```
-Σᵢ tᵢ·log[(1-p)qᵢ] - t_oos·log p  =  -(Σtᵢ)log(1-p) - Σᵢ tᵢ·log qᵢ - t_oos·log p
```

the gradient on the candidate ranking `q` is unchanged and `p` decouples. MASShatter's
candidate m/z set is fixed by enumeration, so the in/out partition does not drift during
training and the factorization is exact. Under `cosine_distance` and `hellinger_distance` the
metric is scale-invariant, so a scalar deficit changes nothing by itself.

**What it does buy:**

1. A calibrated per-spectrum "this fraction is not the analyte" output, which is real value at
   inference for annotation confidence and for flagging contaminated references.
2. Removal of a per-spectrum nuisance from the objective without changing the evaluation
   target — the property a hard target filter cannot have, since a filtered arm is scored
   against its own cleaned target.
3. It closes the config lie: MASShatter configs that set `predict_oos: True` are silently
   overridden today.

## 6. Open decisions

**Decision A is resolved.** The dump target comes from the batch-time `match_mask`, so there
is no `spec_df` column to compute and no interaction with the `tolerance_consensus` merge.
The question only ever applied to the contaminant escape of §4, whose subformula-derived
target does still need a home; that is now the only part of A left open, and it is not on the
critical path for the intensity escape.

**Decision B — does the dump share a head with the contaminant escape, or get its own?**
They have different targets: the dump absorbs *all* unreachable intensity (0.1089), the
contaminant escape only the part carrying no feasible subformula (0.0156), and the second is a
subset of the first. Options:

| option | effect |
|---|---|
| B1. one head, dump target only | simplest; loses the contaminant/coverage distinction at inference |
| B2. one head for the dump, a second predicting what fraction of it is contaminant | keeps both readings; the second head is pure diagnosis and cannot affect the main loss |
| B3. two independent escapes with overlapping targets | double-counts; rejected |

Recommendation: **B1 now, B2 when the contaminant escape lands.** Do not build B3.

## 7. Acceptance

- Escape is off by default; every existing config resolves unchanged. **Met for the loss
  primitive:** `pred_dump_intensities=None` is bit-identical to the previous behaviour on all
  four objectives, under test.
- With `filter_subformula_peaks: True`, `oos_target` is ~0 and the escape is inert.
- `p_oos` calibration measured against `oos_target` on val: report mean absolute error, not
  just the loss.
- A matched control at 128 epochs (the screen length established on 2026-09-21), n≥2, since
  this family's replicate floor is 0.003-0.005 and an n=1 arm cannot resolve what it produces.
- No cosine claim without an eval on the **unfiltered** target.
