# Ion × collision-energy matrix experiment plan

> **Status:** PROPOSED — PRECURSOR AND CE-SWITCH LOSSES SHIPPED; MATRIX AUDIT, TRAJECTORY LOSS, AND V2 NOT IMPLEMENTED
> &nbsp;·&nbsp; **Updated:** 2026-08-13
>
> Use an ion × collision-energy matrix as the diagnostic contract while keeping V1 training sparse.
> The precursor contender loss already exists; V1 adds an eligible-normalized reduction and tests a
> new full-trajectory objective against the shipped adjacent-CE switch loss. V2 is gated on V1.

Empirical basis: [dominant-peak outcome decomposition](../../findings/fragnnet/dominant_peak_outcome_decomposition.md).

The precursor default and general fragment-intensity misranking can be represented in one object
without assuming they have one cause. For each acquisition `group_id`, construct a ragged matrix
whose rows are distinguishable DAG-supported ions and whose columns are the collision energies
observed for that group:

| ion        |      CE 1 |      CE 2 | ... |      CE E |
| ---------- | --------: | --------: | --: | --------: |
| precursor  | intensity | intensity | ... | intensity |
| fragment 1 | intensity | intensity | ... | intensity |
| ...        |       ... |       ... | ... |       ... |
| fragment I | intensity | intensity | ... | intensity |

The dense table is conceptual in V1 and may be materialized per group for diagnostics and metrics.
V1 training operates on sparse `(group, ion_row, ce, intensity)` entries; it does not allocate a
batch-wide `[num_groups, max_ions, max_ce_levels]` tensor or change the model architecture. V2
predicts the multi-CE matrix jointly and is gated on V1 showing validation value.

## Shared representation contract

The diagnostic/evaluation view exposes the following logical arrays. They may remain ragged or be
padded only inside bounded evaluation chunks:

```text
true_intensity: [num_groups, max_ions, max_ce_levels]
pred_intensity: [num_groups, max_ions, max_ce_levels]
ion_mask:       [num_groups, max_ions]
ce_mask:        [num_groups, max_ce_levels]
ce_values:      [num_groups, max_ce_levels]
precursor_row:  [num_groups]
```

The representation obeys the following rules:

1. A matrix represents one `group_id`; V1 begins with FT spectra and NCE only.
2. Columns are the group's observed CE levels, ordered by the actual CE values. There is no fixed
   global CE grid in V1, and irregular CE gaps are retained.
3. Rows are instrument-tolerance-merged theoretical m/z candidates, not raw DAG-node IDs. Multiple
   DAG nodes at experimentally indistinguishable masses are summed into one row.
4. The precursor row is identified explicitly rather than inferred from row order.
5. Observed peaks are mapped to rows at the instrument tolerance. Ambiguous matches are excluded
   and counted rather than resolved silently.
6. OOS intensity stays separate because it has no stable ion identity across CE columns.
7. Padding is masked. A missing measurement is not treated as an observed zero.

For V1 training, store only occupied/eligible sparse entries plus group, row, and CE pointers. The
audit reports median, p95, and maximum ion-row and CE counts, realized `(I, E)` products, and the
memory that a padded representation would require. Any later dense V2 implementation must declare a
hard ion/CE bound or chunking policy before training.

## V1: matrix losses over the existing model

### V1.1 — Build and audit the matrix view

Add a read-only sparse row-indexing utility over the existing `m/z`, intensity, and batch-index
tensors, plus an optional per-group matrix materializer for plots and metrics. Before enabling a new
loss, run a diagnostic reporting:

- eligible multi-CE groups and spectra;
- CE levels and ion rows per group;
- fraction of true intensity mapped to supported rows and assigned to OOS;
- precursor-row coverage;
- ions observed at two or more CE levels;
- ambiguous tolerance matches and replicate-CE frequency;
- realized CE columns per group chunk under the configured sampler, including the fraction reduced
  to `E = 1` by CE-count or fragment-edge limits;
- median, p95, and maximum `(ion rows, CE columns)` and estimated padded-matrix memory.

The sparse → diagnostic matrix → sparse round trip must conserve supported intensity, preserve CE
order, merge aliases consistently, and reproduce the current precursor-contender selection on
single columns. Synthetic tests cover duplicate aliases, missing precursors, missing CE columns,
irregular CE gaps, ambiguous matches, and groups with no eligible comparison.

### V1.2 — Reuse CE-context batching

Use the existing `CEContextDynamicBatchSampler` rather than introducing another sampler:

```yaml
batch_sampler_type: ce_context
ce_context_spectra_per_context: 5
ce_context_ce_key_by_inst:
  FT: nce
```

The sampler does **not** emit every observed CE level for a group. It emits bounded contiguous
chunks of at most `ce_context_spectra_per_context` primary CE levels, rotates the chunk boundary by
epoch, and can split a chunk again at the fragment-edge budget. Same-CE non-primary replicates are
emitted as singleton atoms. Consequently, normalizing an ion trajectory over whatever columns happen
to be in a chunk would give the same ion a moving target across epochs.

V1 resolves this by enabling the trajectory loss only when the realized primary chunk contains the
group's complete set of distinct observed CE levels and `E >= 2`. This requires the group to fit both
the configured CE-count limit and fragment-edge budget. Other chunks still receive the primary
spectrum loss and any independently eligible within-spectrum loss. The coverage audit in V1.1 is a
go/no-go gate: if complete-axis chunks are too rare or unrepresentative, do not train the proposed
trajectory objective; revisit a fixed offline full-axis target with a mathematically matched sampled
loss instead.

Precompute the distinct CE count for each eligible `group_id` from dataset metadata. The sampler or
collate path must attach an explicit `ce_context_complete` indicator only to a primary atom whose
realized unique CE count equals that stored total; singleton replicate atoms are always false. The
loss must consume this indicator rather than infer completeness from batch size.

Tests lock in two existing sampler invariants: `batch_groups` prevents a `group_id` from occurring
twice in one batch, and non-primary same-CE replicates remain singleton atoms rather than new CE
columns. Replicate rotation may change the measured intensities at a fixed CE, but it cannot change
the CE support used in the trajectory denominator.

### V1.3 — Shipped precursor contender; change only its reduction

This objective is already implemented as `precursor_contender_bce` in
[`loss.py`](../../../src/fragnnet/model/loss.py), wired through
[`fragnnet_pl.py`](../../../src/fragnnet/pl_model/fragnnet_pl.py), exposed by the
`precursor_pair_*` template keys, covered by focused tests, and exercised by production-pilot and
debug configurations. It compares the precursor with the strongest supported non-precursor mass,
merges predicted aliases at instrument tolerance, rejects ambiguous observed groups, includes both
winner directions, and uses the configured base-peak-relative floor when the precursor is absent:

\[
L_{\mathrm{precursor}} =
\operatorname{BCEWithLogits}\!\left(
\log \hat I_{\mathrm{prec}}-\log \hat I_{\mathrm{frag}},
\frac{I_{\mathrm{prec}}}{I_{\mathrm{prec}}+I_{\mathrm{frag}}}
\right).
\]

The remaining V1.3 change is the denominator. The shipped function returns a per-spectrum vector
whose non-eligible entries are zero, after which the generic reducer divides by the full batch's
total sample weight (the batch size under the default `train_sample_weight: none`). Replace that
applied reduction with

\[
L_{\mathrm{precursor,eligible}} =
\frac{\sum_s w_s \ell_s}{\max(\epsilon, \sum_s w_s m_s)},
\]

where `m_s` is the eligible-pair indicator and `w_s` is the configured sample weight. A batch with
no eligible pairs has a zero numerator and contributes an exact zero. This makes the configured
weight independent of low-CE prevalence while preserving sample weighting. Preserve the shipped
`precursor_pair_eligible` and `precursor_pair_true_precursor_wins` metrics; add predicted precursor
wins if it is needed for online calibration monitoring. Test equivalence to the current result when
every sample is eligible and the correct zero-loss behavior when none is eligible.

The related shipped adjacent-CE contender objective and its sampler controls are documented in
[`ft_contender_pair_minimal_training_plan.md`](ft_contender_pair_minimal_training_plan.md). Do not
create a second precursor target or require a matrix allocation for the V1.3 reduction change.

### V1.4 — Ion-trajectory objective

For every reliable ion row observed at two or more CE levels, normalize its intensity across the
group's **complete distinct observed CE set** `E_g`:

\[
q_{i,e}=\frac{I_{i,e}}{\sum_{e' \in E_g}I_{i,e'}}, \qquad
L_{\mathrm{trajectory}}=-\sum_{i,e}q_{i,e}\log \hat q_{i,e}.
\]

Compute both target and prediction normalization only for complete-axis chunks admitted by V1.2;
never renormalize over a rotating partial chunk. Implement the loss with sparse segment operations
over `(group, ion_row, ce)` entries: gather the eligible entries for one row, compute its CE-axis
normalizers, and accumulate the row loss. The conceptual dense matrix is not part of this training
path.

This asks whether an ion appears at the right energies while leaving its within-spectrum competition
to the existing column loss. It is genuinely new and differs from the shipped `ce_switch_pair_bce`,
which supervises tolerance-merged contenders only at adjacent-CE winner switches. The trajectory
objective does **not** require monotonic behavior: rising, falling, and unimodal trajectories are all
valid. Rows with ambiguous mapping, insufficient CE coverage, or unreliable targets are masked.
Because each measured spectrum is normalized independently, this models an ion's relative spectral
share across CE, not absolute ion yield. OOS trajectories are out of scope for V1.

The complete V1 objective is

\[
L = L_{\mathrm{column}}
  + \lambda_p L_{\mathrm{precursor}}
  + \lambda_s L_{\mathrm{switch}}
  + \lambda_t L_{\mathrm{trajectory}}.
\]

Here `L_column` is the current per-spectrum loss and `L_switch` is the shipped adjacent-CE objective.
The main comparison substitutes switch versus trajectory rather than enabling every term at once.
For a scale-matched comparison, log every auxiliary loss over its eligible units before weighting;
if the shipped switch loss retains its fixed batch denominator, record its applied contribution and
tune its weight independently rather than comparing nominal lambda values.

### V1.5 — Debug and matched ablations

Retain the existing deterministic precursor-contender overfit as a regression test and rerun it after
the eligible-denominator change. Add a trajectory-only overfit whose realized chunk contains the
complete group CE axis, including at least one rising and one falling ion trajectory. Eligibility
must remain stable, losses must stay finite, and each isolated loss must decrease. Exercise the
shipped CE-switch loss on the same sampler configuration as a wiring control.

Then run matched experiments with the same split, seed, sampler, and training budget:

| run         | column loss | precursor contender | CE-switch pair | ion trajectory |
| ----------- | ----------: | ------------------: | -------------: | -------------: |
| M0 baseline |         yes |                  no |             no |             no |
| M1          |         yes |                 yes |             no |             no |
| M2          |         yes |                  no |            yes |             no |
| M3          |         yes |                  no |             no |            yes |
| M4          |         yes |                 yes |            yes |             no |
| M5          |         yes |                 yes |             no |            yes |

Start eligible-normalized auxiliary-weight pilots at 0.1 and 0.3; calibrate any fixed-denominator
CE-switch control by its measured applied contribution rather than assuming the same nominal weight
has the same scale. Select by validation outcomes, not training loss. Report molecule-grouped
bootstrap confidence intervals.

The pre-training reference values are the `9t63zils` NIST20 validation evaluation half: 0.376
non-precursor-winner hit and 0.953 precursor-winner hit at FT NCE 0–20 (n=4,935 total). M0 establishes
the matched baseline for M1–M5 and is the only denominator for ablation comparisons. Other primary
metrics are precursor-base calibration gap, low-CE intensity-weighted mean-m/z bias, and low-CE
cosine. General matrix metrics are non-precursor top-1 accuracy by CE, ion-trajectory correlation,
adjacent-CE intensity-change error, fragment appearance/disappearance accuracy, cosine, and
intensity-weighted mean m/z in every CE bin.

The V1 decision separates the hypotheses:

- M1 improves only precursor metrics: the precursor default is specialized.
- M3 exceeds M2: full-trajectory normalization adds value beyond the shipped adjacent-switch target.
- M2 matches or exceeds M3: retain the cheaper shipped switch objective; the matrix remains a
  diagnostic representation.
- M5 exceeds M4: the trajectory objective adds value conditional on precursor correction.
- M3 improves precursor and general ranking without M1: both failures reflect a shared CE-trajectory
  representation problem.
- Training improves but validation does not: target reliability or regularization is limiting.
- No V1 loss improves validation: stop the loss work before attempting V2.

## V2: joint multi-CE matrix prediction

V2 is justified only if V1 demonstrates that multi-CE supervision improves held-out metrics. Its
goal is to share molecule and ion computation across CE levels and model a coherent intensity
trajectory rather than issuing unrelated per-spectrum predictions.

### V2.0 — Grouped-forward parity control

First stack the existing per-spectrum forward calls by `group_id` and materialize their outputs as
one matrix without changing where CE enters the network. For fixed weights and evaluation mode,
each matrix column must match the corresponding independent forward call within numerical tolerance.
This isolates grouped batching, row merging, column normalization, and output plumbing from the
architectural change below.

The production configurations in scope set `ce_insert_location: mol` and `ce_scale: True`: CE enters
the molecular encoder and the enabled `CEScaler` applies CE-conditioned FiLM to the output logits.
`depth_ce_bias` is disabled and is not part of this claim. Encoder-level CE insertion alone is enough
to prevent encoding the production molecule representation once for all CE levels. Any production
shared-encoder decoder is therefore a genuine factorization change and must be trained and compared
as a new architecture, not presented as a caching refactor.

There is also a real caching-parity control: train or instantiate a control with
`ce_insert_location: mlp` only while leaving its other enabled output conditioning unchanged. Its
molecule encoding is CE-independent, so cached-once and repeated-per-CE executions of that same
control must agree numerically. This validates encoder caching separately from the grouped-output
parity check above; it does not claim parity with the production `mol`-insertion weights.

### V2.1 — Minimal shared-encoder joint decoder

Encode the molecule and fragmentation DAG once per group to obtain one embedding per merged ion row,
`H_ion [I, D]`. Encode all requested CE values as `H_ce [E, D]`, broadcast the two axes, and apply a
shared interaction head:

\[
s_{i,e}=f\!\left(h_i, c_e, h_i\odot c_e\right).
\]

This produces the full score matrix `S [I, E]` in one forward pass. Normalize independently within
each CE column, because every observed spectrum is a separate intensity distribution. Keep the same
per-column OOS head, target, and loss used by the matched V1 arm, exposed as a separate length-E
output rather than an ion row. This V2.1 model is the architecture control: it shares encoding and
parameters but introduces no explicit communication between CE columns. Train it from scratch at a
matched parameter count; do not initialize it by silently dropping the current encoder-level CE
conditioning.

### V2.2 — Continuous trajectory decoder

If V2.1 is sound, make each ion emit a continuous, non-monotonic CE response rather than adding
attention whose output depends on which other CE queries happen to be present. Let the ion embedding
produce coefficients over a small continuous CE basis (for example learned radial basis functions),
with an optional residual interaction:

\[
s_i(e)=b_i + a_i^\top\phi(e) + r(h_i,\phi(e)).
\]

This defines one coherent trajectory that can be evaluated at arbitrary CE values, supports
non-monotonic shapes, respects irregular CE spacing, and makes a prediction at CE 20 independent of
whether CE 10 was included in the same query. The number of basis channels is selected on validation
and kept small enough that V2 cannot simply memorize every acquisition energy.

Observed columns receive the existing column loss plus the V1 matrix losses. Unobserved CE values
are queryable at inference but receive no fabricated interpolation target. Replicate measurements at
one CE supervise the same trajectory point through the existing replicate rotation or a separately
audited replicate aggregate.

### V2.3 — V2 experiments and gate

First require both V2.0 parity checks: current grouped versus independent output, and cached versus
repeated execution of the `mlp`-only control. They are verification checks, not separately trained
quality arms. Then compare, at matched parameter count and training budget:

| run | decoder                             | matrix losses         |
| --- | ----------------------------------- | --------------------- |
| A   | current independent spectrum        | selected V1 objective |
| B   | V2.1 shared-encoder joint broadcast | selected V1 objective |
| C   | V2.2 continuous trajectory          | selected V1 objective |

In addition to the V1 metrics, test CE interpolation by holding out one interior CE level from
eligible validation groups, conditioning only on its numeric CE at inference, and scoring the held-out
spectrum. Also check that predictions at a fixed CE are invariant to the set and ordering of other CE
queries. Keep OOS handling identical across A–C. Adopt V2 only if it improves held-out CE and
ordinary test spectra without degrading single-CE groups or high-CE coverage-limited strata.

Dense global m/z grids, monotonic trajectory constraints, fabricated CE interpolation labels, and
OOS trajectory identities remain out of scope. A CE-axis transformer is also deferred because it can
make a prediction depend on the arbitrary set of CE queries supplied at inference; it requires a
separate justification beyond this experiment.
