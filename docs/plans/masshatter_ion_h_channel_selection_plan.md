# MASShatter Ion-Mode Proposal, H Selection, and Intensity-Head Plan

> **Status:** ACTIVE &nbsp;·&nbsp; **Updated:** 2026-09-15
>
> First remove CMF states duplicated by reachable CRF H shifts. Keep atom-set
> retrieval H-independent, add an explicit CRF/CMF proposal stage only on
> non-degenerate adduct scopes, split H-support selection from intensity prediction,
> and emit a compact CE-aware channel set per retained fragment.
>
> Measured 2026-09-14: no trained arm exercises channel selection at all (retained channels
> equal valid channels), the soft support gate has no measured spectral or precision benefit,
> and the support BCE sharpens channel ranking without moving cosine. The blocking next step is
> the Phase 2 fixed-checkpoint sweep, which needs a recoverable checkpoint.

## CMF candidate-universe alignment (2026-09-14)

**Implemented; no training or oracle-coverage evaluation run.** User authorization
superseded the prior proposal-only restriction: MASShatter now emits the full configured
H-transfer ladder for chemically valid, non-degenerate CMF adducts, matching FraGNNet.
It still removes CMF entirely for `[M+H]+` and `[M-H]-` when their base CMF ion duplicates
a reachable CRF H shift, and it retains the shared exclusion of in-source-loss precursors.
The pre-batch chemical-support filter uses the same mode/H validity contract.

This changes the MASShatter candidate universe for carrier-switching adducts such as
`[M+Na]+` and `[M+Cl]-`. Existing checkpoints did not train the newly valid CMF-H channels,
so they are diagnostic-only; rebaseline candidate coverage and train matched controls before
scientific comparison. The adduct-stratified zero-only versus full-range oracle measurement
remains required to quantify the effect.

## H-support ablation measurement (2026-09-14)

**Read from the authenticated W&B API on 2026-09-14. No training, evaluation, or code ran for
this record.** The matched arms were launched 2026-09-08 and every one of them terminated at the
three-hour scheduler limit after 43 epochs (epochs 0-15 proposal-only), single seed `397623`,
merged NIST20 `[M+H]+` scope, U256/E1024, dense evaluation, hard thresholds zero.

| Arm | Config | Dense run | Subformula run |
| --- | --- | --- | --- |
| Gated | [warmup_loss_transform](../../config/masshatter/merged/warmup_loss_transform/nist20_merged_independent_jss_warmup_e0_15_dense.yml) | [`gbdszzlw`](https://wandb.ai/frag-gnn/frag-gnn/runs/gbdszzlw) | [`elfzwq6s`](https://wandb.ai/frag-gnn/frag-gnn/runs/elfzwq6s) |
| H supervision only | [support_ablation](../../config/masshatter/merged/support_ablation/nist20_merged_independent_jss_h_supervision_only_warmup_e0_15_dense.yml) | [`sr9xfkk8`](https://wandb.ai/frag-gnn/frag-gnn/runs/sr9xfkk8) | [`1fp53ss4`](https://wandb.ai/frag-gnn/frag-gnn/runs/1fp53ss4) |
| No H support | [support_ablation](../../config/masshatter/merged/support_ablation/nist20_merged_independent_jss_no_h_support_warmup_e0_15_dense.yml) | [`s23w0lt2`](https://wandb.ai/frag-gnn/frag-gnn/runs/s23w0lt2) | [`ynhectw6`](https://wandb.ai/frag-gnn/frag-gnn/runs/ynhectw6) |

### Channel selection is inert in every trained arm

Final-epoch validation summaries, dense arms:

| Validation metric | Gated | Supervision only | No H support |
| --- | ---: | ---: | ---: |
| `valid_ion_h_per_fragment_mean` | 8.5433 | 8.5438 | 8.5414 |
| `attributed_ion_h_per_fragment_mean` | 1.0198 | 1.0168 | 1.0169 |
| `retained_ion_h_count` / `valid_ion_h_count` | 1475.5 / 1475.5 | 1496.6 / 1496.6 | 1478.7 / 1478.7 |
| `retained_ion_h_probability` | 1.0000 | 1.0000 | 1.0000 |
| `ion_h_cutoff_pruned_fraction` | 0.0000 | 0.0000 | 0.0000 |

Retained equals valid in every arm. With `min_ion_h_relative_probability: 0.0`,
`min_ion_h_absolute_probability: 0.0`, and `mass_shatter_apply_channel_selection_in_training:
False`, what has been trained so far is an H-**support** head; the hard channel-selection stage of
the hierarchy has never been exercised in training or validation. About 8.5 channels are emitted
per fragment where roughly 1.02 are attributed. Phase 2's fixed-checkpoint sweep remains the first
measurement of the selection itself, not a refinement of an existing measurement.

### Spectral quality does not separate the arms

`val_cos_sim_0.01`, mean over the last five shared epochs (38-42), with the per-epoch standard
deviation over the last ten epochs as the noise scale:

| Arm | Dense | sd | Subformula | sd |
| --- | ---: | ---: | ---: | ---: |
| Gated | 0.7630 | 0.0047 | 0.7692 | 0.0049 |
| H supervision only | 0.7655 | 0.0055 | 0.7693 | 0.0043 |
| No H support | 0.7631 | 0.0054 | 0.7504 | 0.0132 |

Every dense difference is smaller than one per-epoch standard deviation of a single seed. The soft
intensity gate has no measured spectral benefit in either data variant, and no benefit in
validation `fragment_weighted_precision` either (0.6513 gated, 0.6517 supervision only, 0.6547 no
support). The one large gap, subformula without H support, comes from the least stable arm in the
set and is not yet a result. Best-epoch values (0.7668, 0.7706, 0.7694 dense) reorder the arms
relative to the five-epoch means, which is itself evidence that best-epoch selection sits inside
the noise band here. Paired seeds are required before the gate or the support BCE is credited or
removed on quality grounds.

### The support loss does sharpen channel ranking

Final-epoch validation, dense arms:

| Validation metric | Gated | Supervision only | No H support |
| --- | ---: | ---: | ---: |
| `required_ion_h_top_k_95_mean` | 3.2721 | 3.2842 | 3.6623 |
| `required_ion_h_top_k_95_p90` | 5.5665 | 5.5731 | 6.3422 |
| `required_ion_h_top_k_95_max` | 7.2532 | 7.2250 | 7.8719 |

Supervision concentrates 95% of per-fragment channel mass into about 11% fewer channels while
cosine and weighted precision stay flat. This is the separation the plan's goal section predicts:
a mechanism diagnostic improves while thresholded spectral quality does not. It supports carrying
the supervision into Phase 2 as a ranking signal, not as a spectral-quality intervention.

**Metric check required before these numbers gate anything.** `required_ion_h_top_k_95_mean` and
`required_ion_h_top_k_99_mean` agree to four decimals in all three arms (3.2721 versus 3.2722 in
the gated arm). Two different cumulative thresholds yielding the same mean needs verification in
the metric implementation before either is used as a Phase 4 gate.

### Clone initialization and self-gating (code check 2026-09-15)

The separate support projection starts with an exact copy of the intensity projection's
weight and bias. Loading an old checkpoint without support keys also copies its intensity
weights into the newly enabled support head. Both projections consume the same `tokens`:
see [head construction and checkpoint migration](../../src/fragnnet/model/bond_fragment_heads.py)
and the copy-initialization requirement in this plan's experiment matrix. This is deliberate
RNG-preserving initialization, not an independently initialized support signal.

At that point, `ion_h_support_logits == ion_h_logits == z`, so the independent intensity
path computes `softplus(c + z) * sigmoid(z)`, where `c` is the corrected fragment logit;
see [spectral weighting](../../src/fragnnet/model/mass_shatter_model.py). The initial gate is
therefore a self-gate. The parameters remain separate and can diverge through support BCE
and through the different spectral gradients entering the intensity and gate branches.
Clone initialization is a candidate confound for the measured gate null, not an established
explanation: the recorded runs do not measure projection divergence or logit correlation.
An initialization comparison would need matched RNG isolation and separate treatment of
legacy-checkpoint migration; no initialization or compatibility behavior changed here.

### Scope consequences

- `spec_params.prec_types` is `['[M+H]+']` only, and its CMF is removed as degenerate with
  reachable CRF `delta_h = -1`. The CMF candidate-universe alignment above therefore does not
  change this screen's candidate set, and cannot explain metric differences between 2026-09-08
  and 2026-09-14 runs on it.
- Arm C stays untestable in this scope, exactly as Phase 1 predicts. Ion-mode work requires
  `[M+Na]+` or `[M+Cl]-` in `spec_params.prec_types`.
- In this six-run ablation, `cmf_precursor_valid` is false for the only configured adduct.
  With `max_h_transfer = 4`, each fragment has at most nine valid CRF H channels (-4 through
  +4), with chemical validity reducing the measured mean to about 8.5. The `ion_h` metric
  names describe a general tensor interface; these results measure H ranking/support only,
  and provide no evidence of learned ion-mode discrimination. This scope statement does
  not establish the adduct scope or validity behavior of every historical trained run.

### Next concrete check

Script update (2026-09-15):
[`mass_shatter_ion_h_top_k_sweep.py`](../../scripts/eval/mass_shatter_ion_h_top_k_sweep.py)
now accepts `--support_absolute_thresholds` separately from final-intensity
`--joint_relative_thresholds`. It uses the model's existing one-channel-per-spectrum
fallback, requires a separate binary support head, includes selected-output weighted
precision/recall, and reseeds each rule. Use the newer checkpoint's matching training
YAML to preserve architecture and target construction. Disable fragment pruning with
`--fragment_relative_thresholds 0`; compare `--ion_h_top_ks 0 4 5` against a support-floor
sweep, with `--joint_relative_thresholds` supplied without values to omit intensity-floor
rules. Metrics are validation-equivalent and do not include a separate exported-intensity
cutoff pass. No checkpoint/data evaluation ran for this update.

1. Set `mass_shatter_gate_independent_intensity_with_support: False` in the H-support configs. No
   measured outcome pays for the gate, and it confounds every later support comparison.
2. Recover one trained checkpoint from the cluster and run the Phase 2 staged selection sweep on
   it. This is blocked, not deprioritized: W&B lists only `run-*-history` artifacts for these
   runs, so no checkpoint is recoverable from the run records, and the earlier `tq4bdn3g` sweep
   predates the chemistry corrections.
3. Keep the support BCE at `0.1` until Phase 2 measures whether its sharper ranking survives hard
   selection.

## Decision

Do not fold hydrogen transfer into the bond-mask proposal. Use a hierarchical model:

```text
bond-mask proposal
    -> unique-fragment selection
    -> optional CE-aware CRF/CMF proposal head on non-degenerate adducts
    -> CE-aware H-support head within retained ion modes
    -> hard ion/H channel selection
    -> CE-aware ion/H intensity head
    -> sparse predicted peak list
```

The hierarchy separates three questions. Fragment proposal asks whether an atom set
can support any observed peak. Ion-mode proposal asks whether that fragment is emitted
through charge retention (CRF), charge migration (CMF), or both. H selection asks
which hydrogen-shift realization within each retained mode should be emitted. CRF/CMF
is a chemically meaningful proposal decision, while the exact H shift remains a
downstream channel decision. Neither should be folded into bond-mask enumeration.

The current `CandidateIntensityTransformerHead` emits one `ion_hydrogen_head` tensor
that simultaneously determines conditional ion/H probability and relative intensity,
while the proposal path has no CRF/CMF score. After the validity fix, the architecture
separates these roles where the configured adduct scope provides identifiable signal:

- optional `ion_mode_proposal_logits[f, i]` rank non-degenerate CRF/CMF modes before
  H-transfer selection;
- `ion_h_support_logits[f, i, h]` rank H configurations within retained ion modes;
- `ion_h_intensity_logits[f, i, h]` assign relative signal among the selected
  channels;
- fragment intensity remains the existing contextualized fragment score;
- proposal/support logits never multiply final intensity, and intensity logits never
  decide fragment or ion-mode proposal probability.

Existing-head selection is a mandatory control. On non-degenerate adducts, first
aggregate the already-trained joint logits over valid H shifts to obtain a CRF/CMF
ranking, then use the within-mode logits for H ranking. On degenerate lead-scope
`[M+H]+`, remove CMF and test only H ranking. This prevents arithmetic deduplication
from being credited to either new head.

## Joint fragment/channel selection-loss experiment (2026-09-13)

**Status: proposed; no implementation or run.** This is an alternative *training
objective* for the existing selection-only hierarchy, not a change to bond-mask
enumeration or a claim that the current support BCE is unnecessary. The current
[merged subformula JSS warmup 0–4 setup](../../config/masshatter/merged/warmup_duration/nist20_merged_independent_jss_warmup_e0_4_dense_subformula.yml)
trains with JSS + 0.1 fragment proposal coverage loss + 0.1 balanced ion/H support
BCE. It gates independent intensity with the support sigmoid. Its
[W&B run](https://wandb.ai/frag-gnn/frag-gnn/runs/8v0i0fq7) leads the inspected
single-seed filtered screen at best validation binned cosine 0.80099, but reports
crashed. See the [proposal review](../findings/masshatter_proposal_review_2026_09_13.md#selection-only-versus-joint-training-live-wb-comparison)
for selection-only and intensity-head comparisons and their provenance limits.

**Question:** Can one joint selection objective train both the fragment proposal
probabilities and ion/H support probabilities, while leaving the spectral intensity
objective to allocate signal over the retained candidates? The current fragment
proposal loss gives a fragment credit when *any* valid channel matches a peak; its
gradient does not distinguish the matching channel. The support BCE supplies a
separate channel target. A shared coverage objective would connect these decisions.

For observed peak `i`, let `P(f)` be the existing normalized proposal probability
over selected fragments, `g(f,c)` the sigmoid support probability of valid ion/H
channel `c` (including ion mode), and `M(i,f)` the channels within the instrument's
mass tolerance. A candidate joint coverage probability is:

```text
C_i = sum_f P(f) * [1 - product_{c in M(i,f)} (1 - g(f,c))]
L_positive = -sum_{i with at least one reachable channel} y_i * log(C_i)
```

`y_i` uses the same observed-intensity weighting as the current proposal loss. The
within-fragment union permits multiple compatible channels, and the outer sum
marginalizes over alternative fragments instead of asserting a unique structural
label. `C_i` is at most one because fragment probabilities sum to one. Compute it
stably in log space and preserve gradients to both heads. Do not multiply the final
intensity by `P(f)`; selection-only prediction remains the comparison contract.

Positive coverage alone is degenerate: it rewards opening every valid channel.
Complete **one joint selection loss** with a negative/sparsity term on valid channels
that have little observational support, using the existing Gaussian mass-attribution
semantics and explicit class/candidate-count normalization. Define the negative term
and coefficient on a frozen calibration batch before a training comparison. Avoid
turning one ambiguous observed mass into hard negatives for its other plausible
channels. A target peak with no chemically reachable candidate has no gradient under
this objective; log its excluded intensity and candidate weighted recall separately.
Neither a constant penalty nor a larger coefficient can create a missing candidate.

Compare on the same output architecture and data: retain the existing proposal and
support heads, sigmoid gate, independent intensity head, JSS spectral loss, dense
training support, warmup/routing policy, candidate budget, seed list, and tolerances.
Replace only the two selection auxiliaries with the calibrated joint selection loss.
The current separate-loss configuration is the control. Verify on synthetic ambiguous
and uncovered-peak examples that both heads receive the intended gradients and that
opening all valid channels is not optimal. Then compare the full training trajectory,
weighted candidate/channel recall, support precision by threshold, emitted peak count,
support oracle, and raw-intensity spectral quality on the same validation IDs. A simpler
intensity task is supported only if attainable quality and coverage are preserved and
the learned intensity model closes more of the oracle gap. Use the roadmap's paired-seed
and retrieval gates before promotion. No new loss weight or config is declared here;
those belong to implementation with checkpoint/config compatibility work.

## Goal

The goal is not to increase unthresholded raw precision for its own sake. That metric
counts every emitted channel equally even when its predicted intensity is negligible.
The intervention is useful only if it produces a smaller practical spectrum or lowers
matching/output cost while preserving spectral quality.

Use two primary outcomes:

1. **Practical spectral utility.** Normalize predictions to the existing 0-1000
   inference scale, apply the standard `--min_intensity 1.0` cutoff (0.1% of the base
   peak), and report thresholded weighted precision, precision, recall, weighted
   recall, cosine, and peak count.
2. **Systems utility.** Report pre-threshold emitted peak count, peak-matching/loss
   wall time, peak-tensor memory, serialized output size, and end-to-end inference
   time. The mode head runs before the fragment transformer and H selection runs after
   it, but neither changes the number of fragment tokens. Do not claim transformer
   savings; possible savings are in channel expansion, spectral matching, metrics,
   and output handling.

A secondary modeling outcome is a simpler intensity task. Once fragment existence,
CRF/CMF support, and H support are handled by separate objectives, the intensity head
only has to distribute signal among supported channels. Measure this benefit through
optimization and stability diagnostics; do not infer it solely from a lower emitted
peak count.

Unthresholded raw precision and channel count remain mechanism diagnostics. A change
that improves only raw precision but leaves the standard thresholded spectrum and
systems costs unchanged does not justify the new model path.

## Evidence and corrected metric interpretation

Sweep `0svt99d9` searched fragment target, ranked-mask budget, mask cutoff,
fragment cutoff, and proposal mixture count. The strongest balanced run was
`ka68eu6p`:

| Metric at best-cosine epoch | Value |
|---|---:|
| `val_cos_sim_0.01` | 0.7396 |
| raw peak precision | 0.1917 |
| peak recall | 0.7347 |
| intensity-weighted recall | 0.9104 |
| pre-cutoff unique fragments | about 238 |
| fragment cutoff fraction | 0.4087 |
| retained fragment probability | 0.9741 |

These values were queried from the authenticated W&B API on 2026-08-30 and aligned
at run `ka68eu6p`'s best-cosine epoch. They are not derived from a local run artifact.

The 0.01 MAP-relative fragment floor removed about 41% of the post-enumeration
unique-fragment tail while discarding only about 2.6% of proposal probability. That
result supports fragment pruning, but it does not show that fragment selection has
19% precision.

The logged `precision` is computed after each retained fragment expands into all
chemically valid ion/H channels. A fragment may emit CRF shifts from -4 through +4
and one CMF tensor channel. Before the degeneracy fix, the count can approach ten
entries per fragment; for lead-scope `[M+H]+`, however, CMF duplicates CRF at shift
`-1`, leaving only nine distinct masses. The raw metric counts the duplicate as a
second prediction, so it can distort both matched and predicted counts. If one distinct
channel matches, the other emitted entries still enter the raw precision denominator
even when their predicted intensity is nearly zero.

Approximately,

```text
peak_precision
    ~= fragment_precision
       * matching_channels_per_positive_fragment
       / valid_channels_per_fragment
```

Therefore the current raw peak precision is not a fragment-selection-purity metric.
The existing evidence says fragment coverage and probability ranking are useful; it
does not measure the fraction of selected fragments that can explain an observed
peak. That missing metric must be added before changing the fragment proposer.

### CMF/CRF degeneracy in the lead adduct

The lead sweep config does not override `spec_params.prec_types`, so it inherits the
template's `[M+H]+`-only scope while enabling CMF and `max_h_transfer = 4`. Under the
current mass/formula tables, its CMF state is the same chemical ion as CRF at
`delta_h = -1`:

```text
CRF([M+H]+, delta_h=-1): H1 - H1 = neutral direct cation
CMF([M+H]+, delta_h=0):  no formula delta = neutral direct cation
```

The same degeneracy occurs for `[M-H]-`, where CMF equals CRF at `delta_h = +1`.
The four-adduct `ma_mi_unmerged` family therefore has this static scope:

| Adduct | Equivalent CRF shift for CMF | Mode-identifying signal at ±4 H |
|---|---:|---|
| `[M+H]+` | `-1` | none; CMF is redundant |
| `[M-H]-` | `+1` | none; CMF is redundant |
| `[M+Na]+` | no integer shift | non-degenerate |
| `[M+Cl]-` | no integer shift | non-degenerate |

Consequences for the current lead checkpoint are predetermined:

- `y_mode[f, CMF] = 1` implies `y_mode[f, CRF] = 1`; a CMF-only target cannot occur;
- a CRF/CMF mode head has no identifiable label and Arm C is invalid on this scope;
- marginalizing nine CRF channels versus one CMF channel can make checkpoint mode
  pruning look successful for arithmetic reasons rather than learned chemistry;
- the nominal maximum is ten valid tensor entries, but only nine distinct ion masses
  remain after removing the redundant CMF state;
- raw precision `0.1917` and the channel-count denominator must be remeasured after
  fixing validity rather than used as the selector baseline.

Treat this as a static chemistry prediction that Phase 1 confirms, not a pattern that
Phase 1 must discover.

### Phase 0: remove CMF channels duplicated by reachable CRF shifts

Before any selector metric or new-head run, make CMF precursor validity depend on
non-degeneracy. For each configured precursor type, compute:

```text
shift = (cmf_mass_delta - crf_mass_delta) / H_MASS
k = round(shift)
mass_equivalent = abs(shift - k) <= degeneracy_tolerance
reachable = -max_h_transfer <= k <= max_h_transfer
formula_equivalent = cmf_formula_delta == crf_formula_delta + k * H
cmf_non_degenerate = not (mass_equivalent and reachable and formula_equivalent)
```

Use an explicit tight numerical tolerance; do not depend on floating-point `==`.
Checking formula equivalence makes precursor-wide invalidation safe by proving that
the corresponding CRF state is chemically valid whenever CMF is valid. If a future
adduct is mass-degenerate but not formula-equivalent, deduplicate per fragment after
chemical validity rather than disabling CMF globally.

Apply this guard when constructing `cmf_precursor_valid`, in addition to checking that
the adduct has a CMF table entry. Then rerun the unchanged checkpoint evaluation and
report separately:

- channel reduction from chemistry-validity deduplication;
- any changes to precision, candidate coverage, loss, and cosine;
- additional channel reduction from learned selection, measured relative to the
  deduplicated baseline.

The Phase 4 `>= 50%` channel-reduction gate uses the post-fix dense baseline. The
roughly one-of-ten reduction available from deleting redundant `[M+H]+` CMF entries
must never be credited to a selector.

### CMF full H-range alignment on non-degenerate adducts

MASShatter now permits every configured CMF H shift for non-degenerate adducts, matching
FraGNNet. Direct-charge `[M+H]+`/`[M-H]-` CMF remains suppressed when its base state
duplicates a reachable CRF state. For carrier-switching adducts such as `[M+Na]+`, this
adds the previously unenumerated distinct CMF-H masses at ±4 H.

This is a candidate-universe change, not part of the selector experiment. Report an
adduct-stratified oracle coverage comparison against the former zero-only universe,
then rebaseline candidate counts and train matched controls before attributing any
metric difference to CMF support.

## Scope

The first implementation targets the lead path:

```yaml
mass_shatter_unique_fragment_mode: True
mass_shatter_proposal_selection_only: True
```

Because that lead path inherits `[M+H]+` only, it covers the Phase 0 validity fix and
the H-support/intensity split, not Arm C. A CRF/CMF proposal-head experiment requires a
separately gated multi-adduct configuration with non-degenerate CMF examples.

It must support the current `probability` intensity output. The channel-mask helper
must also preserve the existing `independent` output contract, but no independent-
output training experiment is part of the first run matrix.

In scope:

- static CMF/CRF degeneracy detection and a post-fix baseline remeasurement;
- fragment-level purity diagnostics on the full chemical channel set;
- CRF/CMF purity and coverage diagnostics after collapsing H shifts within mode;
- a lightweight CRF/CMF proposal head on the proposal representation;
- a separate H-support head in `CandidateIntensityTransformerHead`;
- separate MAP-relative floors for ion-mode proposal and downstream ion/H channels;
- deterministic fixed-k and cumulative-mass checkpoint controls across the joint
  ion-mode/H-shift axis;
- dense spectral training and sparse validation/inference;
- support supervision from tolerance-matched observed peaks;
- matched controls using the existing intensity logits as support logits;
- one conditional sparse-training follow-up of the winning new-head arm;
- probability retention, channel count, precision, recall, and spectral metrics.

Out of scope for the first implementation:

- joint enumeration of `(bond mask, ion mode, H shift)` states;
- further changes to chemical validity rules, CMF mass/formula conventions, or the allowed
  H-transfer range;
- forcing one channel per fragment;
- sparsemax, entmax, Gumbel-softmax, straight-through estimators, or reinforcement
  learning;
- a second channel transformer or explicit ion/H channel tokens;
- global spectrum top-k or cumulative-intensity truncation;
- changing fragment proposal loss, mask target, or fragment cutoff;
- additional candidate-universe expansions in the selector experiment;
- applying hard channel selection during the first split-head training run.

Those are escalation options only after the minimal split-head experiment identifies
a measured limitation.

## Current tensor flow and failure mode

For each unique fragment `f`, the current model returns:

```text
fragment_residual[f]
ion_h_logits[f, ion_mode, h_shift]
valid_channels[f, ion_mode, h_shift]
```

It computes a masked conditional softmax over every chemically valid channel and
then flattens every valid entry into `pred_mzs` and `pred_logprobs`:

```text
P(peak[f, c]) = P(fragment[f]) * P(channel[c] | fragment[f])
```

Softmax assigns nonzero probability to every valid channel. Consequently, a channel
with negligible probability still counts as one full false prediction in raw
precision. The spectral loss can learn to suppress its intensity but cannot make it
leave the emitted support.

There is a second coupling to avoid when hard channel selection is introduced. The current
`proposal_candidate_mzs` and `cross_ev_candidate_mzs` tensors are both derived from
the same flattened peak list used for spectral output. If hard ion/H selection simply
replaces `valid_channels` everywhere, fragment proposal coverage and cross-EV fragment
consistency would be evaluated only through the selected H channels. That would
silently make both structural objectives depend on channel selection and violate the
intended separation.

The implementation must maintain two masks:

```text
chemical_valid_channels  # full chemistry-valid support
spectral_output_channels # selected subset used for emitted peaks
```

Fragment proposal loss, cross-EV fragment loss, and fragment-purity diagnostics use
`chemical_valid_channels`. Spectral prediction and raw peak metrics use
`spectral_output_channels`.

That summary has five concrete existing consumers. Route each explicitly:

| Consumer | Required support | Reason |
|---|---|---|
| `pred_mzs` plus `pred_logprobs`/`pred_intensities` | selected | Primary spectral output under test |
| `candidate_mzs_by_cut`, batch indices, and cut counts | full chemical | Historical candidate recall/weighted recall and cut-bucket metrics |
| `cross_ev_candidate_mzs` and fragment indices | full chemical | Cross-EV fragment loss must remain a structural objective |
| `proposal_candidate_mzs` and fragment indices | full chemical | Fragment proposal coverage must remain H-selection-independent |
| CE-path `candidate_wrecall` | full chemical | Preserve its historical enumerated-support meaning |

The new mode-proposal target is an additional full-support consumer. Build its
`y_mode[f, mode]` labels from all chemically valid H channels before applying either
the mode or downstream channel selector.

The last consumer requires a loss-path change. Cross-entropy currently derives
`candidate_wrecall` from `unmatched_true_logprobs` returned by the primary CE on
`pred_mzs`. Once `pred_mzs` becomes selected support, that byproduct no longer measures
full candidate coverage. Compute `candidate_wrecall` from the full-support match data
instead, reusing the proposal/coverage match matrix when its grouping semantics agree
or calling the shared dense-coverage helper otherwise. Both CE and direct peak-list
losses must report the same full-support definition so new runs remain comparable with
prior MASShatter sweeps.

Build a full flattened channel view once from `chemical_valid_channels` and derive
`candidate_mzs_by_cut`, cross-EV tensors, proposal tensors, and dense coverage from
that view. Build the selected flattened view separately for spectral output. Do not
construct full tensors by attempting to recover dropped entries from `pred_mzs`.
Carry explicit aligned fragment, ion-mode, and H-shift indices in the full view. The
mode/channel losses must scatter mass matches with these indices rather than infer
mode identity from positions after invalid channels have been filtered.

## Proposed hierarchical architecture

### Shared fragment contextualization

Keep the current candidate transformer, inducing tokens, fragment-atom decoder, and
fragment residual head unchanged. The transformer already provides a CE-aware,
molecule-contextualized token `t_f` for every retained fragment.

Do not add explicit channel embeddings in the first implementation. The existing
linear ion/H projection has independent output parameters for every `(ion mode,
H shift)` slot and is the smallest appropriate baseline.

### CRF/CMF proposal branch

Add a lightweight `ion_mode_proposal_head` to the proposal path after unique atom sets
and their fragment/complement/boundary features are available, but before the
candidate-intensity transformer. It consumes proposal-encoder features plus the same
precursor/adduct and CE context used by the proposer:

```text
ion_mode_proposal_logits[f, mode] = proposal_mode_head(
    proposal_fragment[f],
    proposal_complement[f],
    proposal_boundary[f],
    proposal_context[f],
)
```

Do not duplicate bond masks or atom-set fragments for CRF and CMF during enumeration.
The existing `P(fragment)` remains the atom-set retrieval probability and its cutoff
remains unchanged. The new branch makes a second proposal decision over the valid
ion modes of each retained fragment:

```text
valid_ion_mode[f, mode] = chemical_valid_channels[f, mode, :].any(dim=-1)
```

Apply a per-fragment MAP-relative floor to valid mode logits. If only CRF is valid, it
survives automatically. If CRF and CMF are valid, the branch may retain either or
both. The first implementation does not feed mode logits back into the atom-set score;
that preserves the existing fragment budget and isolates whether explicit mode
proposal is useful. Joint `(fragment, mode)` budgeting is a later experiment only if
mode-aware scoring works but the fixed fragment budget is demonstrably limiting.

Train `ion_mode_proposal_logits` with independent binary targets because CRF and CMF
can both be supported. For selection and `retained_ion_mode_probability` only, form a
temporary masked softmax across valid modes and apply the relative floor. This
normalization is a ranking device, not a claim that ion modes are mutually exclusive
or that balanced-BCE logits are calibrated probabilities.

This head belongs to the proposal model, not the intensity transformer. It can use the
separate proposal encoder when configured, and its parameters receive only the new
mode-proposal objective. Construct it conditionally so disabled configurations and old
state dictionaries stay unchanged.

Move chemical mass/validity construction ahead of the candidate transformer so the
mode head can mask invalid CMF states before its loss and selector. This is an ordering
change only: when both new heads are disabled, the same masses and masks must reach all
existing consumers. With `mass_shatter_share_proposal_across_ev`, share atom-set
enumeration but compute mode logits for each spectrum's CE context by repeating the
pooled proposal features through the existing spectrum-to-shared mapping. Do not make
the full-support cross-EV tensors depend on those CE-specific mode decisions.

Arm C has an adduct-scope prerequisite. At least one configured adduct must have a
chemically valid, non-degenerate CMF state, and the training/validation data must
contain enough such spectra to evaluate CRF-versus-CMF ranking separately by adduct.
The four-adduct `ma_mi_unmerged` family includes `[M+Na]+` and `[M+Cl]-`, but only two
of its four adducts carry mode-identifying signal after degeneracy removal. Existing
three-hour runs in that family have not yielded a converged recoverable baseline, so
Arm C is blocked until that convergence/wall-time prerequisite is satisfied.

In a mixed scope, mark degenerate CMF states invalid per precursor before mode loss,
selection, and metrics. Train and evaluate the mode head only on remaining two-mode
cases; report sample counts and results per adduct so `[M+H]+`/`[M-H]-` examples cannot
dominate or make the task look artificially easy.

### Split H-support and ion/H intensity heads

Replace the single semantic role of `ion_hydrogen_head` with two outputs from the
same fragment token:

```text
ion_h_intensity_logits[f] = intensity_projection(t_f)
ion_h_support_logits[f]   = support_projection(t_f)
```

Preserve the existing `ion_hydrogen_head` parameter name for the intensity projection
so disabled-mode checkpoints and state dictionaries retain their current meaning.
Add `ion_h_support_head` only when the separate support-head feature is enabled.

When the separate head is disabled:

```text
ion_h_support_logits = ion_h_intensity_logits
```

This gives the no-new-parameters checkpoint controls and keeps the disabled path
minimal.

When the separate head is enabled, initialize its weight and bias as copies of the
intensity projection at construction time. They are independent parameters after
initialization. This starts both arms with identical channel ranking while allowing
support supervision and spectral intensity supervision to specialize them.

The support loss updates the shared fragment tokens in the first experiment. Its
coefficient must be calibrated small enough that the primary spectral loss remains
dominant. Do not add a detach switch in the initial implementation; add one only if
gradient diagnostics show that support supervision harms intensity learning.

### Why the factorization should simplify intensity learning

The current joint head must solve three tasks with one set of logits:

```text
is this fragment observable?
which ion mode is supported?
which H configuration is supported, and at what relative intensity?
```

The first question remains fragment-level. The proposal-mode and H-support heads take
over the next two support decisions. The intensity head is then conditional on the
selected support:

```text
P_intensity(channel | fragment, selected_mode, selected_H_support)
```

This removes two avoidable competitions from intensity normalization: CMF's single
zero-shift channel no longer has to compete as a support decision against as many as
nine CRF shifts, and unsupported H shifts no longer need near-zero intensity merely to
stay out of the spectrum. Preserve the rectangular ion/H output tensor for checkpoint
and downstream compatibility, but mask unsupported modes/channels before the selected
conditional intensity normalization.

The first dense B/C runs isolate representation and auxiliary-loss effects, so their
intensity loss still sees full chemical support during training. The Phase 3c matched
sparse-training run is the direct test of the easier conditional task: only selected
channels enter the spectral intensity loss, while dense proposal/support losses still
allow excluded states to recover.

Evaluate the claim with:

- validation intensity loss and cosine convergence versus optimizer step;
- late-epoch variance and collapse frequency across matched seeds;
- conditional intensity entropy over selected channels;
- gradient cosine similarity on shared fragment tokens between the primary intensity
  loss, mode-proposal loss, and H-support loss;
- the fraction of intensity gradient previously spent on mode/H-negative channels;
- time or epochs to reach the dense baseline's best validation cosine.

Call the intensity model easier only if optimization or stability improves under
matched data, seed, parameter budget, and checkpoint selection. A smaller denominator
or better raw precision alone is insufficient evidence.

### Hierarchical channel selection

First select valid CRF/CMF modes using the proposal-mode floor. Then flatten the
remaining ion mode and H shift entries into one channel axis per fragment, mask
invalid or mode-pruned entries to `-inf`, and apply the downstream ion/H MAP-relative
floor:

```text
mode_keep[f, mode] = relative_keep(
    ion_mode_proposal_logits[f, mode],
    valid_ion_mode[f, mode],
    min_ion_mode_relative_probability,
)
scores[f, c] = log_sigmoid(ion_h_support_logits[f, c])  # Bernoulli support head
# Existing categorical intensity controls use their logits directly instead.
scores[f, invalid or mode-pruned] = -inf
map_logscore[f] = max(scores[f])
selected[f, c] = chemical_valid_channels[f, c]
                 and mode_keep[f, mode(c)]
                 and scores[f, c]
                     >= map_logscore[f]
                        + log(min_ion_h_relative_probability)
spectral_output_channels = chemical_valid_channels & selected
```

Requirements:

- the mode head ranks CRF/CMF before exact H-transfer configurations are considered;
- the downstream support head ranks channels jointly across the modes retained by the
  proposal head;
- a zero mode floor keeps every chemically valid ion mode, and a zero ion/H floor
  keeps every valid channel in each retained mode, without evaluating `log(0)`;
- positive floors are relative to the appropriate per-fragment MAP score, mirroring
  the existing fragment MAP-relative cutoff;
- guarantee at least one retained channel for every retained fragment; the MAP
  mode and MAP channel necessarily survive every valid floor in `(0, 1]`;
- do not force CRF zero-shift, CMF, or one channel per ion mode;
- detach only the discrete selected mask, not retained intensity logits;
- selection disabled must reproduce the current support exactly.

Fixed-k and cumulative-mass rules remain fixed-checkpoint diagnostics at both the
ion-mode level and the joint ion/H level. For both, sort valid entries with
`torch.sort(..., descending=True, stable=True)` and slice the result. Do not use
`torch.topk`: its ordering among equal values is not guaranteed to be stable, so tied
logits can select different identities on CPU and CUDA. The original flattened index
is the deterministic secondary order.

For fixed-k, use `k_eff = min(k, valid_channel_count[f])`. For cumulative mass, form
a normalized distribution over valid support scores and retain the smallest
descending-score prefix whose cumulative probability reaches `tau`. Include the
channel that crosses `tau` and always retain at least one channel. These rules use
relative mass within one fragment; they are not absolute sigmoid thresholds and do
not assume the balanced BCE preserves the empirical class prior.

For probability output, recompute the masked conditional softmax over the selected
channels before forming pathway probabilities. Also compute the sum of the original
dense conditional probabilities on the retained set before renormalization; log it as
`retained_ion_h_probability`.

For independent output, apply the same selected-channel mask to the nonnegative
scores. Do not introduce a channel softmax solely because selection is enabled.

### Dense-first training and a matched sparse-training follow-up

The first split-head run applies channel selection only when the module is not in
training mode:

```text
training:   spectral_output_channels = chemical_valid_channels
validation: spectral_output_channels = support_selection(...)
inference:  spectral_output_channels = support_selection(...)
```

For Arm A, applying hard selection during training is risky: the existing intensity
logits receive no separate dense support objective, so an initially excluded correct
channel loses its spectral gradient and can remain excluded. Dense training avoids
that self-reinforcing failure.

Arms B and C are different. Their H-support loss is computed densely over every
chemically valid channel even when spectral output is sparse; Arm C's mode loss is also
dense over every valid CRF/CMF state. An excluded state therefore still receives direct
supervision and can re-enter the selected set. Existing detached fragment pruning is
useful evidence that a hard selection boundary can coexist with training, but it is
not an exact precedent: a pruned fragment is removed before proposal tensors are built
and its underlying proposal parameters receive only indirect shared-parameter
gradients.

Train the first B and C arms densely to isolate each head change and obtain all
selection curves from fixed checkpoints. If a new-head arm succeeds under
validation/inference selection, run one matched follow-up of the winning architecture
with its winning MAP-relative floors active during training. This is a cheap immediate
ablation, not an open-ended architecture escalation. Report the dense-versus-sparse
train/eval mismatch explicitly.

## Support target and loss

Attribute every retained observed peak over the chemically valid channels using the
same per-spectrum instrument tolerance used by the primary loss. The default
`gaussian_posterior` method uses the detached predicted channel weight and Gaussian
mass likelihood, normalized over the channels matching that peak:

```text
r[peak, channel] ∝ stop_gradient(predicted_channel_weight[channel])
                   * Gaussian(observed_mz[peak] | channel_mz[channel], sigma)
y_channel[channel] = max_peak r[peak, channel]
y_mode[f, mode] = min(1, sum_h y_channel[f, mode, h])
```

This is a soft mass-observational attribution rather than a structure annotation. It
preserves uncertainty between indistinguishable formulas while preventing every
coincidentally matching channel from receiving a full positive label. The optional
`hungarian` method uses the same pairwise likelihood but produces a hard one-to-one
peak/channel assignment for ablation experiments.

Use separate balanced binary losses per spectrum:

```text
L_mode = balanced_bce(ion_mode_proposal_logits, y_mode, valid_ion_mode)
L_support = balanced_bce(ion_h_support_logits, y_channel, chemical_valid_channels)
```

Handle an empty positive or negative set by omitting that half and using the available
term, not by inserting a detached zero. Average spectra equally after the existing
evaluation/training sample weighting contract.

Balanced positive and negative means prevent the much larger negative channel set
from making an all-negative classifier appear successful. Do not add focal gamma,
label smoothing, positive-class weights, or pair sampling in the first implementation.
Those are justified only if calibration or optimization diagnostics fail.

This balanced loss is designed for ranking, not absolute sigmoid calibration. With a
large negative-to-positive class ratio, its logits intentionally need not represent
the empirical probability that a channel is positive. Do not apply an absolute
sigmoid threshold to this head without a separate held-out calibration step or a
loss whose class prior is preserved. Fixed top-k and per-fragment cumulative mass
under a normalized distribution remain valid ranking experiments, as do the
production MAP-relative floors because they compare scores only within a fragment.

The total training loss becomes:

```text
L_total = L_primary_spectrum
          + proposal_loss_weight * L_fragment_proposal
          + ion_mode_proposal_loss_weight * L_mode
          + ion_h_support_loss_weight * L_support
```

Calibrate `ion_mode_proposal_loss_weight` and `ion_h_support_loss_weight` on fixed
batches. Target approximately 5% of the primary spectral gradient norm from each
auxiliary loss near initialization, with their combined norm near 10%. Measure the
mode loss on proposal features/encoder parameters and the H-support loss on candidate
tokens. Record the batches, seed, raw losses, gradient norms, and chosen coefficients.
Do not launch a broad two-dimensional loss-weight sweep before this calibration.

The fragment proposal loss continues to use the full chemical channel set and its
existing fragment-collapsed coverage calculation. Cross-EV fragment consistency also
continues to use the full chemical channel set. Hard channel selection must have no
effect on either structural loss.

## Metrics

Dense observational channel matching is opt-in through
`mass_shatter_channel_diagnostics`. Support-head runs enable the same target-building
path implicitly because their losses require it. Keep the flag disabled for ordinary
selection-only runs so validation does not add a second candidate-matching pass.

### Fragment-selection metrics

Add metrics before interpreting proposal purity:

- `fragment_precision`: fraction of pre-channel-selection unique fragments with at
  least one chemically valid channel matching an observed peak;
- `fragment_weighted_precision`: sum of proposal fragment probability assigned to
  such fragments;
- existing `candidate_recall` and `candidate_wrecall` on full chemical support;
- fragment count before and after the existing MAP-relative fragment cutoff.

These metrics use each unique fragment once, regardless of how many pathways or
channels match.

Compute them in the existing proposal-coverage loop. That loop already constructs
`fragment_match` with shape `(num_true_peaks, num_fragments)` for each spectrum:

```text
positive_fragment = fragment_match.any(dim=0)
fragment_precision = mean(positive_fragment)
fragment_weighted_precision = sum(
    exp(fragment_logprob) * positive_fragment
)
```

Return these detached diagnostics with the proposal loss rather than running another
mass-matching pass. When proposal loss is disabled but the diagnostics are requested,
call the shared fragment-match helper directly. Phase 1 should therefore add almost
no cost on the lead configuration, where proposal supervision is already enabled.
Before the helper collapses `channel_match` to `fragment_match`, also scatter by the
aligned `(fragment, ion_mode)` indices to obtain `mode_match`. This supplies mode
targets and diagnostics from the same tolerance calculation; do not add a second
peak-matching pass for CRF/CMF.

Fragment precision alone is not a sufficient bound on peak precision because valid
channel counts differ by fragment and several channels or fragments can match one
observed mass. From the same match matrix, compute oracle channel-selection curves:

- `oracle_selected_precision@k`;
- `oracle_selected_recall@k`;
- `oracle_selected_wrecall@k`.

For oracle top-k, choose matching channels first for each fragment, up to `k`, then
fill any remaining required slots with nonmatching valid channels. This is not a
trainable target or a reported model result; it is an attainable-support diagnostic.
If no oracle rule can produce the required channel-count reduction while satisfying
the Phase 4 recall constraints, skip the split-head training run.

Mass-observational labels cannot distinguish two structural fragments whose channels
fall within tolerance of the same true peak. That limits structural identifiability
and can retain redundant explanations. It is not itself a raw-precision penalty under
the current metric, because every candidate inside tolerance is counted as matched.

### Ion-mode proposal metrics

Collapse the existing dense channel-match tensor over H and report:

- valid and retained ion modes per fragment and per spectrum;
- `ion_mode_precision` over valid `(fragment, mode)` states;
- CRF and CMF precision, recall, and weighted recall separately by adduct;
- on non-degenerate two-mode cases, fractions of positive fragments supported by CRF
  only, CMF only, both, or neither;
- mode-head AUROC and average precision when both classes are present;
- top-1 ion-mode coverage when both CRF and CMF are chemically valid;
- `retained_ion_mode_probability` before mode-selection renormalization;
- `ion_mode_cutoff_pruned_fraction`;
- selected-mode candidate recall and weighted recall before H selection.

These diagnostics answer whether CMF/CRF ambiguity is a meaningful part of the raw
channel multiplicity. They are also a Phase 1 gate on eligible adducts: if almost every
positive fragment requires both non-degenerate modes, a learned mode proposer is
unlikely to justify a separate training arm. An all-degenerate scope bypasses this
measurement and rejects the mode head statically.

### Channel-selection metrics

Log both dense and selected values:

- valid ion/H channels per fragment and per spectrum before selection;
- retained ion/H channels per fragment and per spectrum after selection;
- `retained_ion_h_probability` before selected-channel renormalization;
- `ion_h_cutoff_pruned_fraction`, matching the naming convention of
  `fragment_cutoff_pruned_fraction`;
- support positive rate;
- support AUROC and average precision, if both classes are present;
- top-1, top-2, and top-3 positive-channel coverage per positive fragment;
- `selected_candidate_recall` and `selected_candidate_wrecall` after channel
  selection, kept distinct from the full-support fragment-proposal metrics;
- `selected_candidate_precision` plus final raw peak precision and weighted precision
  after channel selection.

Support-head AUROC is diagnostic. Top-k coverage and the fixed-k/cumulative curves
measure ranking quality; the production action is a per-fragment MAP-relative floor.

### Spectral and systems metrics

Retain:

- `val_cos_sim_0.01`;
- primary validation loss;
- raw dense and selected peak counts before the inference intensity threshold;
- weighted precision, precision, recall, weighted recall, cosine, and peak count after
  the standard normalized `--min_intensity 1.0` threshold;
- peak matching/loss wall time;
- peak tensor memory where practical;
- serialized prediction size and end-to-end inference time;
- NaN/Inf and late-epoch stability monitoring.

The two prior sweep trials with late spectral collapse while candidate recall stayed
stable show that proposal and intensity failures can diverge. Always align channel,
intensity, and spectral metrics at the same checkpoint epoch.

## Configuration contract

Declare every new field in `config/template.yml`; do not use fallback access:

```yaml
mass_shatter_separate_ion_mode_proposal_head: False
mass_shatter_ion_mode_proposal_binary_logits: False
mass_shatter_min_ion_mode_relative_probability: 0.0
mass_shatter_ion_mode_proposal_loss_weight: 0.0
mass_shatter_separate_ion_h_support_head: False
mass_shatter_ion_h_support_binary_logits: False
mass_shatter_min_ion_h_relative_probability: 0.0
mass_shatter_ion_h_support_loss_weight: 0.0
mass_shatter_apply_channel_selection_in_training: False
mass_shatter_channel_support_attribution: gaussian_posterior
```

Semantics:

- both heads `False`, both floors `0.0`, and both loss weights `0.0`: selector-disabled
  path, bit-for-bit relative to the post-Phase-0 chemistry baseline;
- mode head `True`, mode binary-logit semantics `True`, and a positive mode-loss
  weight: learn CRF/CMF proposal support with balanced BCE;
- mode head `True` with mode floor `0.0`: train and diagnose the head without pruning;
- separate H-support head `True`, H-support binary-logit semantics `True`, and a
  positive support-loss weight: split H support from ion/H intensity with balanced
  BCE;
- `mass_shatter_apply_channel_selection_in_training: False` applies positive floors
  only during validation/inference;
- setting it `True` applies active mode and ion/H floors to the training spectral path,
  while mode/support targets and auxiliary losses remain dense;
- `mass_shatter_channel_support_attribution` controls only proposal/support targets;
  `mass_shatter_peak_matching` remains the independent direct intensity-loss setting;
- training-time selection is invalid for Arm A or any active selector without its own
  positive dense auxiliary-loss weight;
- either new head enabled with its loss weight zero is invalid because an unsupervised
  random selector has no defined purpose;
- either new loss weight positive with its head disabled is invalid;
- either binary-logit flag must agree with its current balanced-BCE loss configuration;
- mode-head configuration is invalid when CMF emission is disabled or when CMF is
  degenerate for every configured adduct at the configured `max_h_transfer`, because
  the proposer has no identifiable two-mode decision to learn;
- fixed-checkpoint controls may derive mode and H scores from existing intensity
  logits without enabling either trainable head.

Validate both relative floors as finite and in `[0, 1]`. Zero is the explicit disabled
value; the MAP valid mode and MAP valid channel remain selected at every positive
value.

Keep fixed-k and cumulative-mass thresholds in the fixed-checkpoint evaluation script.
They diagnose whether a failure comes from ranking or from the production cutoff
shape; do not commit competing selection interfaces to the model schema.

## Compatibility contract

Phase 0 is the sole intentional compatibility break in predicted support: degenerate
CMF entries disappear from flattened peak tensors for affected adducts. Record that
one-time delta explicitly and establish the post-Phase-0 checkpoint evaluation as the
new dense reference. Do not describe its channel reduction as a learned-model gain.

Relative to that post-fix reference, the disabled selector configuration must preserve:

- model input/output tensor shapes;
- `pred_mzs`, `pred_logprobs`, and `pred_intensities` values and semantics;
- proposal candidate tensors and fragment proposal loss;
- cross-EV candidate tensors and cross-EV fragment loss;
- state-dict keys and checkpoint loading;
- current training and evaluation results within exact deterministic equality where
  the existing kernels are deterministic.

Old checkpoints are evaluated in the existing-head checkpoint controls with
both new-head flags `False`. Do not silently construct an untrained proposal or
support head while loading an old checkpoint. New-head arms train from a new
configuration and checkpoint. Old checkpoint metrics evaluated after Phase 0 are not
bit-for-bit comparable to their historical W&B values; report both values and label
the chemistry version.

The model must build `candidate_mzs_by_cut`, proposal tensors, cross-EV tensors, and
the full-support CE `candidate_wrecall` inputs from `chemical_valid_channels` before
applying spectral selection. This five-consumer routing is a compatibility
requirement, not merely a metric choice.

## Implementation phases

### 0. Fix degenerate CMF validity and rebaseline

Implement the static adduct/max-H degeneracy helper and apply it to
`cmf_precursor_valid` without adding either selector head. Re-evaluate the lead fixed
checkpoint before implementing selection.

That old-checkpoint evaluation measures the immediate metric artifact only. Because
the original checkpoint trained with probability and gradient assigned to the
duplicate CMF state, it cannot serve as the matched Arm A training control after the
chemistry fix. Any new Arm B/C training comparison requires a fresh Arm A trained with
the same post-Phase-0 validity rules.

Verify:

- `[M+H]+` CMF is removed because it duplicates reachable CRF `delta_h = -1`;
- `[M-H]-` CMF is removed because it duplicates reachable CRF `delta_h = +1`;
- `[M+Na]+` and `[M+Cl]-` CMF remain valid at `max_h_transfer = 4`;
- candidate coverage and spectral metrics are remeasured rather than assumed stable;
- chemistry-only and learned-selection channel reductions are logged separately.

### 1. Add metrics without changing predictions

Implement fragment purity, CRF/CMF purity, and dense channel diagnostics while leaving
all selections disabled.

Verify:

- raw predictions and losses are unchanged from the post-Phase-0 dense reference;
- fragment precision counts unique fragments, not pathways or ion/H channels;
- mode precision counts each valid `(fragment, CRF/CMF)` state once after collapsing
  over H shifts;
- CRF-only, CMF-only, and joint-support rates on eligible adducts reconcile with the
  channel labels;
- dense channel counts reconcile with the flattened prediction count;
- fragment proposal metrics match the post-Phase-0 dense reference;
- CE and non-CE `candidate_wrecall` retain the historical full-support definition.

This phase establishes whether fragment selection is actually impure, separates
CRF/CMF multiplicity from within-mode H multiplicity, and supplies the oracle go/no-go
gates for Phase 3. Skip the mode-proposal arm if mode-level oracle pruning cannot
reduce states while preserving recall. Skip the H-support arm if channel-level oracle
selection cannot meet the Phase 4 channel-count and recall constraints.

For the lead `[M+H]+` scope, Phase 1 must confirm the static prediction: zero valid CMF
states after Phase 0 and therefore no mode-selection curve or Arm C. On multi-adduct
data, compute mode gates only over non-degenerate two-mode cases and report them by
adduct.

### 2. Add fixed-checkpoint selection controls

Use `ion_h_intensity_logits` as support scores and keep training unchanged. Derive the
existing-head ion-mode control by marginalizing its normalized channel probability
over valid H shifts within CRF and CMF:

```text
existing_mode_logprob[f, mode]
    = logsumexp(ion_h_logprob[f, mode, valid_h], dim=h)
```

When the fallback source is a binary H-support head instead, preserve its independent
Bernoulli semantics and derive each mode's probability as
`1 - product_h(1 - sigmoid(ion_h_support_logit))`. The explicit binary-logit config
flags select these conversions; head presence alone does not determine them.

Evaluate selection only from one fixed checkpoint.

**Blocking status (2026-09-14):** this is the next step and it is blocked on checkpoint
recovery, not on analysis. The
[H-support ablation measurement](#h-support-ablation-measurement-2026-09-14) shows that every
trained arm retains all valid channels, so no run to date measures any selection rule. W&B lists
only `run-*-history` artifacts for those runs, and the earlier `tq4bdn3g` sweep predates the
chemistry corrections. Recover one checkpoint from the cluster before planning further training.

Do not run the mode-control sweep on an all-degenerate scope. On the lead checkpoint,
Phase 2 starts directly with post-fix CRF H-channel controls. On mixed scopes, exclude
degenerate CMF states before marginalization and report every curve per adduct as well
as in aggregate.

On the same checkpoint, evaluate in two steps:

1. With H selection disabled, evaluate the ion-mode floor disabled and in
   `{0.01, 0.03, 0.10, 0.30}`, plus fixed top-1 mode.
2. With mode selection disabled and then with the best feasible mode floor, evaluate
   ion/H floors in `{0.01, 0.03, 0.10, 0.30}`, fixed `k = 1, 2, 3`, and the smallest
   descending-score prefix whose normalized per-fragment cumulative channel mass
   reaches `tau` in `{0.90, 0.95, 0.99}`.

Always retain at least one channel. These are per-fragment channel-selection rules,
not the out-of-scope global spectrum truncation, and require no training runs. The
staged evaluation avoids a broad Cartesian sweep. Use stable sorting for the fixed-k
and cumulative rules.

Verify:

- `candidate_mzs_by_cut` and its cut/batch metadata are identical for every rule;
- proposal and cross-EV candidate outputs and losses are identical for every rule;
- CE-path full-support `candidate_wrecall` is identical for every rule;
- selected conditional probabilities renormalize correctly;
- the mode selector never retains a chemically invalid mode and always retains the
  valid MAP mode;
- no rule retains an invalid channel and the MAP-relative floor always retains the
  MAP channel;
- cumulative-mass selection includes the channel that crosses `tau`, resolves ties
  deterministically, and retains at least one channel;
- channel count, retained probability, thresholded spectral metrics, and systems
  metrics form coherent tradeoff curves for all three rule families.

If marginalized existing logits plus MAP-relative floors already meet all success
criteria, that is the simplest production candidate. Fixed-k and cumulative-mass
results explain its ranking behavior but remain diagnostics. The new mode-proposal and
H-support heads are still run as matched scientific comparisons when their Phase 1
oracle gates pass; extra model complexity requires a measurable improvement.

### 3. Add the split support/intensity head

Add the optional support projection, balanced support loss, and dense support metrics.
Proceed only if the Phase 1 oracle clears the Phase 4 gates. Run the matched arms with
`mass_shatter_gate_independent_intensity_with_support: False`: the
[H-support ablation measurement](#h-support-ablation-measurement-2026-09-14) found no spectral or
weighted-precision benefit from the soft gate in either data variant, so keeping it only confounds
the support comparison. Train the split-head arm
from scratch using the `ka68eu6p` fragment-selection configuration. If there is no
recoverable control checkpoint produced by the same code, data, seed, and training
schedule, also train a fresh existing-head control. Keep channel selection disabled
during this first training run.

After training, evaluate the Phase 2 MAP-relative, fixed-k, and cumulative rules from
the same best checkpoint.

Verify:

- support loss has finite gradients to the support head and shared candidate tokens;
- spectral gradients still reach every valid intensity channel during training;
- support ranking improves top-k channel coverage over the existing-head control;
- the intensity distribution does not collapse or become less stable;
- the best checkpoint is uploaded before the cluster wall-time limit.

### 3b. Add the CRF/CMF proposal head

Add the conditional proposal-mode head, dense mode target/loss, and mode diagnostics.
Train Arm C from the same initialization/data/schedule as Arm B, adding only the mode
branch and its calibrated loss. Keep mode and H selection disabled during this first
training run, then evaluate the staged Phase 2 rules from the same best checkpoint.

Verify:

- invalid CMF states are masked before mode normalization and loss;
- mode loss has finite gradients to the mode head and intended proposal encoder;
- CRF and CMF each contribute one target per fragment regardless of H-channel count;
- mode ranking improves selected-mode coverage over marginalized intensity logits;
- full-support fragment proposal, cross-EV, and candidate-coverage tensors still use
  both chemically valid modes regardless of selection;
- cross-arm fragment metrics are reported rather than assumed invariant, because the
  mode loss updates the shared proposal representation.

### 3c. Test sparse training if a new-head arm succeeds

If Arm B or C clears the validation gates with inference-only selection, train one
matched follow-up of the winning architecture with its winning MAP-relative floors
active in both training and evaluation by setting
`mass_shatter_apply_channel_selection_in_training: True`. Keep mode and H-support
losses dense over all chemically valid states. This tests whether aligning training
support with inference improves the intensity model without removing the re-entry
signals.

Do not add this follow-up for Arm A: without independent dense proposal/support losses,
early selection errors can remove both the spectral learning signal and the mechanism
that would correct the ranking.

### 4. Select the MAP-relative floors

Choose the strongest ion-mode floor, then the strongest downstream ion/H floor,
satisfying all validation constraints. Do not select either on the test split. Use
fixed-k and cumulative-mass curves to interpret the choices, not as additional
production knobs.

Primary constraints relative to the same post-Phase-0 dense checkpoint. Apply the mode
constraint only to non-degenerate two-mode adducts:

```text
standard-cutoff cosine drop <= 0.005
selected_mode_candidate_wrecall drop <= 0.01
selected_candidate_wrecall >= 0.90
standard-cutoff weighted-recall drop <= 0.01
standard-cutoff weighted precision does not decrease beyond measured validation noise
pre-threshold emitted ion/H channel count reduction >= 50%
at least one matching/output cost decreases beyond measured run-to-run noise
fragment-level metrics remain unchanged
```

Measure dense-baseline wall time repeatedly before setting the cost confidence
interval; do not treat timing jitter as a gain. Also report standard-cutoff precision,
peak count, serialized size, and end-to-end inference time. Raw unthresholded precision
is diagnostic only and has no ship gate. Reject a selector that improves only raw
precision while leaving practical spectral output and systems cost unchanged.

The fragment-invariance gate applies only when comparing selection rules on one fixed
checkpoint. Hard mode/channel selection must not change that checkpoint's full-support
fragment tensors or metrics. Do not assume Arms A, B, and C have identical fragment
metrics: they are independently trained models and must report their fragment
precision, weighted fragment precision, recall, and proposal distributions side by
side.

In the lead `proposal_selection_only` path, proposal inputs are detached from the
intensity encoder (and `ka68eu6p` also uses the separate proposal encoder), so the
H-support loss should not directly update proposal parameters. Arm C's mode loss is
different: it intentionally updates the mode head and proposal representation, and
can indirectly change bond/charge proposal behavior through shared proposal-encoder
features. Treat cross-arm fragment differences as data, not as failed within-checkpoint
invariance assertions.

The selector may ship for a systems benefit without improving thresholded weighted
precision, but only if all quality guardrails hold. A spectral-quality claim requires
an improvement after the standard inference threshold, not merely deletion of
near-zero channels from the raw denominator.

### 5. Escalate only on a measured failure

If the split head cannot rank matching channels into top-3 while dense intensity
metrics remain good, add explicit learned ion-mode and H-shift embeddings and score
channel tokens with a small shared MLP. This tests whether the flat output projection
lacks a useful representation of channel identity.

If the MAP-relative rule is poor while fixed-k and cumulative-mass ranking are good,
inspect the per-fragment score-gap distribution before changing the interface. If an
absolute sigmoid support threshold is still desired, first calibrate the support logits
on held-out validation data or replace balanced BCE with a class-prior-preserving
objective. Do not interpret balanced-BCE sigmoid output as a calibrated probability.

If the Phase 3c sparse run is unstable, test a dense warm-up followed by the same
MAP-relative floor. Do not use straight-through estimators unless ordinary
selected-channel training demonstrably fails.

## Test plan

### Unit tests

- degeneracy detection marks `[M+H]+` at shift `-1` and `[M-H]-` at shift `+1`;
- `[M+Na]+` and `[M+Cl]-` remain non-degenerate at `max_h_transfer = 4`;
- degeneracy detection uses numerical tolerance plus formula equivalence rather than
  raw float equality;
- mode-head config fails when every configured adduct is CMF-degenerate or CMF is
  disabled;
- joint-axis selectors rank CRF/CMF and all H shifts together;
- mode validity is `any` over chemically valid H shifts;
- mode targets collapse all matching H shifts to one binary CRF/CMF target;
- invalid CMF modes are excluded from mode loss and selection;
- a zero mode floor retains all valid modes and a positive floor retains the MAP mode;
- invalid high-logit channels are never selected;
- fragments with fewer than `k` valid channels retain all valid channels;
- stable sorting resolves ties by original flattened channel index;
- CPU and CUDA stable-sort paths retain the same identities for tied scores;
- MAP-relative floor `0.0` returns the original valid mask exactly;
- every positive valid MAP-relative floor retains the MAP channel;
- non-finite floors and values outside `[0, 1]` fail validation;
- probability mode renormalizes selected channels to one per fragment;
- `retained_ion_h_probability` is measured before renormalization;
- `ion_h_cutoff_pruned_fraction` uses dense versus selected valid counts;
- cumulative-mass selection retains the threshold-crossing channel and at least one
  channel per fragment;
- independent mode filters channels without adding a softmax;
- balanced support loss handles mixed, all-positive, and all-negative spectra;
- balanced mode loss handles mixed, all-positive, and all-negative valid modes;
- support labels honor absolute and relative per-instrument tolerance;
- support-loss gradients reach support logits and shared tokens;
- fragment precision counts a unique fragment once when several channels match.

### Integration tests

- Phase 0 removes exactly the degenerate CMF states while leaving non-degenerate CMF
  support intact;
- selector-disabled forward output is identical to the post-Phase-0 dense reference;
- `candidate_mzs_by_cut`, batch indices, and cut counts are identical across channel
  selection rules;
- proposal candidate tensors and loss are identical across channel selection rules;
- cross-EV candidate tensors and loss are identical across channel selection rules;
- CE-path full-support `candidate_wrecall` is identical across selection rules;
- the first split-head training arm stays dense when a validation floor is configured;
- `mass_shatter_apply_channel_selection_in_training: False` keeps the spectral path
  dense during training and `True` applies the configured floors;
- training-time selection is rejected for Arm A;
- the Phase 3c dense and sparse-spectral paths produce identical mode/channel target
  tensors, chemically valid masks, and valid-state counts for auxiliary losses;
- Phase 3c auxiliary losses receive every chemically valid state even when the
  spectral output contains only selected states;
- disabling the mode head preserves old checkpoint keys and outputs;
- mode selection can remove CRF or CMF only from spectral output, never from the four
  full-support structural/coverage consumers;
- shared-proposal batches reuse atom-set enumeration while producing CE-specific mode
  logits with correctly aligned spectrum indices;
- evaluation mode emits at most `k` valid channels per fragment;
- output indices, masses, scores, fragment indices, and batch indices remain aligned;
- chemistry-validity and learned-selection channel reductions are logged separately;
- old checkpoints load in the existing-head control;
- probability and independent intensity output contracts both remain valid;
- CPU and CUDA paths produce the same selected channel identities, including ties.

### Focused verification commands

Use the configured `FRAGNNET-GPU` interpreter. Run the smallest affected test modules
first, then the full suite when practical:

```text
tests/test_bond_fragment_heads.py
tests/test_mass_shatter_model.py (or the existing model-forward test module)
tests/test_mass_shatter_pl.py
tests/test_loss.py
```

Add a fixed tiny-batch overfit that logs dense and selected channel metrics for one
molecule with known multiple H-transfer support.

## Experiment matrix

Use the `ka68eu6p` fragment-selection settings as the fixed structural base after
Phase 0:

```yaml
mass_shatter_proposal_num_mixtures: 3
mass_shatter_min_mask_relative_probability: 0.001
mass_shatter_min_fragment_relative_probability: 0.01
mass_shatter_unique_fragment_target: 512
mass_shatter_ranked_mask_budget: 1024
```

The lead `[M+H]+` experiment runs Arms A and B only; Arm C is structurally invalid
because no non-degenerate CMF state remains. To test Arm C, use a converged multi-adduct
family and train matched A/B/C controls within that same family. Do not compare Arm C
on `ma_mi_unmerged` directly with A/B metrics from the lead merged dataset.

Run these matched architecture arms when their Phase 1 oracle gates pass:

| Arm | Ion-mode score | H/channel score | Auxiliary losses | Training support |
|---|---|---|---|---|
| A | marginalized existing intensity logits | existing intensity logits | none | dense |
| B | marginalized H-support logits | separate H-support head | H support | dense |
| C | proposal-mode head | separate H-support head | mode proposal + H support | dense |

Evaluate every checkpoint with the staged disabled, MAP-relative, fixed-k, and
cumulative controls. If a new-head arm clears the gates, add only the Phase 3c matched
follow-up of the winning architecture, with its winning floors applied to the spectral
path during training while both auxiliary losses remain dense.

An existing compatible checkpoint is sufficient for Phase 2's within-checkpoint
selection-rule diagnostic. It is sufficient as Arm A for the architectural comparison
only when it was trained after Phase 0 and its code, data, seed, and training schedule
match the new-head arms. Historical checkpoints trained with degenerate CMF are
diagnostic-only; otherwise train fresh matched controls. Do not call independently
trained checkpoints identical controls merely because their YAML values match. Run
Arm C only if all of these prerequisites hold:

- at least one configured adduct has non-degenerate CMF at the configured H range;
- train and validation contain enough non-degenerate two-mode examples per adduct to
  report meaningful mode metrics;
- a matched dense A/B baseline converges and produces a recoverable checkpoint under
  the available wall time;
- the adduct-stratified mode oracle/control gates pass.

The current four-adduct `ma_mi_unmerged` scope supplies signal only through `[M+Na]+`
and `[M+Cl]-` and has not converged within the three-hour jobs. Arm C remains blocked
until that baseline issue is resolved. Repeat arms with additional seeds only after a
new-head arm clears the validation gates.

Constructing either optional head must not perturb the global initialization or
training RNG stream merely by consuming extra random draws. Initialize the H-support
head by copying the intensity projection under an isolated RNG context, and isolate
mode-head construction similarly. The auxiliary losses are the intended treatments;
an unrelated dropout/data-order RNG shift is not.

The current three-hour jobs terminate before `max_epochs` and may not upload a usable
best checkpoint. Before running the comparison, ensure the job wall time covers the
chosen training schedule or upload best checkpoints periodically. Do not compare
selection rules using independently trained checkpoints; each rule curve must reuse
one fixed checkpoint.

## Expected interpretation

Possible outcomes have distinct meanings:

1. **Existing-head hierarchical selection succeeds on non-degenerate adducts.**
   Marginalized intensity logits already rank CRF/CMF and H support; prefer the
   existing head unless a new head adds a clear gain. The lead `[M+H]+` checkpoint can
   support only the H-ranking part of this conclusion.
2. **Arm B succeeds but Arm C does not improve it.** Separating H support from
   intensity is useful, but a proposal-side CRF/CMF branch is unnecessary.
3. **Arm C improves mode coverage or sparsity over Arm B.** CRF/CMF is a genuine
   proposal decision; keep the proposal-mode head and downstream H/intensity split.
4. **Mode selection works but H selection loses recall.** Mode multiplicity was
   removable for the evaluated non-degenerate adducts, while several H transfers are
   genuinely needed. Keep mode proposal and weaken or disable the downstream H floor;
   do not generalize the result to unevaluated adducts.
5. **All arms preserve weighted recall but raw precision remains low.** Remaining
   false peaks arise primarily from impure fragments or duplicate/ambiguous masses,
   not mode/H multiplicity. Use the fragment metrics before changing retrieval.
6. **All arms lose substantial weighted recall by top-3.** Multiple H channels per
   fragment are genuinely required or the current fragment token cannot rank them.
   Inspect coverage by ion mode and H shift before adding capacity.
7. **Proposal/support metrics improve but cosine degrades.** The hard targets or their
   gradients conflict with relative intensity learning. Reduce the calibrated
   auxiliary weights as one isolated follow-up.

## Key files

- `src/fragnnet/utils/formula_utils.py`
  - CRF/CMF mass and formula deltas used by degeneracy detection
- `src/fragnnet/model/bond_fragment_heads.py`
  - lightweight proposal-side CRF/CMF scoring head
  - `CandidateIntensityTransformerHead`
  - split H-support and ion/H intensity projections
  - masked ion/H normalization helper
- `src/fragnnet/model/mass_shatter_model.py`
  - adduct/max-H-aware CMF degeneracy and chemical channel validity
  - dense proposal tensors
  - selected spectral tensors
  - channel diagnostics
- `src/fragnnet/pl_model/mass_shatter_pl.py`
  - mode/channel labels and balanced auxiliary losses
  - fragment/mode/channel metrics and logging
- `config/template.yml`
  - required compatibility-safe hyperparameters
- `scripts/eval/`
  - fixed-checkpoint MAP-relative, fixed-k, and cumulative evaluation sweep
- `tests/test_bond_fragment_heads.py`
- `tests/test_mass_shatter_pl.py`

## Completion criteria

This plan is complete only when:

- degenerate CMF states are removed before selector evaluation and the chemistry-only
  metric delta is recorded separately;
- fragment, CRF/CMF mode, and H-channel precision are reported separately;
- selector-disabled mode is compatibility-verified against the post-Phase-0 baseline;
- all five `pred_mzs` consumers are routed explicitly: selected spectral output and
  full-support cut-bucket, proposal, cross-EV, and CE candidate-coverage paths;
- proposal coverage, cross-EV consistency, and full-support `candidate_wrecall` are
  invariant to ion-mode and H selection;
- Arms A, B, and C use matched code, data, seeds, schedules, and checkpoint-selection
  rules whenever their oracle gates justify training;
- Arm C runs only on a converged scope with sufficient non-degenerate CMF examples and
  reports adduct-stratified results;
- MAP-relative, top-1/2/3, and cumulative curves report thresholded and raw spectral
  metrics, retained probability, channel count, and systems costs;
- ion-mode and downstream ion/H MAP-relative floors are selected on validation under
  the stated gates;
- the Phase 3c training-selection switch is explicit and auxiliary valid-state counts
  are invariant between dense- and sparse-spectral paths;
- raw precision is treated as a mechanism diagnostic, while the ship decision uses
  standard-threshold spectral quality and measured systems benefit;
- the best checkpoint is recoverable despite scheduler wall time;
- the result, including a null result, is recorded in this document.
