# MASSter fragment and ion/H channel-selection findings

> **Status:** FINDING · **Updated:** 2026-09-05
> Historical (2026-09-05) results; chemistry revalidation and the barcode control are pending, so numerical conclusions are unverified under the corrected rules.

## Experiment

**Chemistry correction (2026-09-05):** MASSter now interprets negative adduct element
counts as removals when validating CRF and CMF channels. It omits nonpositive masses and
fragments with no valid ion channels, renormalizing probability outputs over the remaining
fragments. Dataset setup filters incompatible molecule/adduct pairs before batching and logs
the removed count. Direct model calls return empty sparse output for incompatible spectra.
Legacy checkpoints still load strictly: the element-count buffers are regenerated from the
configured adducts, while learned weights are preserved. Predictions for affected adducts
therefore change intentionally; the historical results below have not been recomputed.
Both checkpoint sweeps use Lightning's batch-transfer hook so SMARTS-enabled checkpoints
receive the same template preparation as validation. Recoverable invalid batches are logged
and counted as skipped, rather than included in metric denominators.

The completed validation sweep used checkpoint `tq4bdn3g` with:

- config
  [`nist20_merged_u256_e1024_mix2_no_barcode_h_support_w01.yml`](../../../config/masster/merged/channel_selection/nist20_merged_u256_e1024_mix2_no_barcode_h_support_w01.yml);
- results
  [`h_support_fragment_channel_sweep_tq4bdn3g.csv`](../../../output/eval/h_support_fragment_channel_sweep_tq4bdn3g.csv);
- 4,222 merged NIST20 validation spectra;
- 256 unique-fragment target, 1,024-mask budget, two proposal mixtures;
- barcode proposal supervision disabled;
- dense training with a separate H-support head weighted by `0.1`.

The conclusions below apply to this checkpoint. They must not yet be generalized to
barcode-supervised proposal models.

## Metric interpretation

The reported spectrum `precision` is

```text
number of emitted fragment/ion/H masses matching any observed mass
------------------------------------------------------------------
number of emitted fragment/ion/H masses
```

Every emitted channel counts equally, including negligible-intensity channels. The metric
therefore measures final peak-list clutter, not the correctness of the underlying fragment.
Mass matching is also not a unique structural assignment: several fragment/ion/H channels can
match the same observed peak.

The fragment diagnostics have different meanings:

- `fragment_precision` is a **soft posterior-attribution mean**. Gaussian responsibility is
  divided among compatible channel aliases, summed within each fragment, clipped to one, and
  then averaged. It is not the literal fraction of correct fragments.
- `fragment_weighted_precision` weights that soft fragment attribution by predicted fragment
  probability.
- `ion_h_supported_fragment_fraction` is the fraction of proposed fragments having at least one
  ion/H channel with nonzero observational attribution. This is the closest available binary
  any-match diagnostic, but still indicates m/z compatibility rather than structural truth.

An earlier interpretation that `fragment_precision = 0.240` meant only 24% of fragments had any
matching channel was incorrect. The corresponding binary any-match value is 47.55%.

## Dense result

The dense checkpoint retained, per spectrum on average:

| Diagnostic                                     |   Result |
| ---------------------------------------------- | -------: |
| fragments                                      |   181.61 |
| chemically valid ion/H channels per fragment   |     8.44 |
| emitted fragment/ion/H channels                | 1,534.93 |
| attributed ion/H channels per fragment         |     1.28 |
| fragments with any attributed ion/H channel    |   47.55% |
| soft fragment attribution                      |   24.05% |
| probability-weighted soft fragment attribution |   75.90% |
| final channel precision                        |   15.11% |
| final peak recall                              |   72.15% |
| binned cosine (`0.01`)                       |  0.74089 |

The large difference between 47.55% fragment any-match and 15.11% final channel precision is
mostly explained by expanding each fragment into approximately 8.44 ion/H alternatives. The low
final precision must not be presented as evidence that fragment precision is 15%.

The weighted fragment result also matters: although the proposal list is broad, 75.90% of its
probability mass falls on fragments with strong posterior attribution. The model ranks fragments
substantially better than the unweighted candidate list suggests.

## Why per-fragment top-k is only a diagnostic

Ion/H top-4 reduced channels from 1,534.93 to 726.46 and increased count precision from 15.11% to
25.32%, while recall fell from 72.15% to 68.75% and cosine changed from 0.74089 to 0.74045.

This does show that the H-support head contains ranking signal. It does **not** establish top-4 as
the correct production policy:

- four states are allocated to every fragment, including weak or unsupported fragments;
- the precision increase is partly mechanical because the denominator shrinks;
- top-4 is per fragment rather than global across the spectrum;
- count precision cannot identify unique structural explanations.

Top-4 should therefore be described as a compute/coverage tradeoff, not as a chemically calibrated
selector.

## Fragment pruning result

The mildest tested fragment cutoff was already too aggressive for the current quality target:

| Fragment cutoff | Fragments pruned | Proposal mass retained | Top-k |  Cosine | Cosine change | Precision | Recall |
| --------------: | ---------------: | ---------------------: | ----: | ------: | ------------: | --------: | -----: |
|               0 |               0% |                   100% |     4 | 0.74045 |            — |    0.2532 | 0.6875 |
|           0.001 |            43.3% |                 99.58% |     4 | 0.73299 |      -0.00747 |    0.3216 | 0.6443 |
|           0.003 |            57.0% |                 98.58% |     4 | 0.72642 |      -0.01404 |    0.3556 | 0.6042 |
|           0.010 |            72.5% |                 95.50% |     4 | 0.71133 |      -0.02912 |    0.4050 | 0.5285 |

Retaining 99.58% of proposal probability did not retain equivalent spectral coverage. The
low-probability fragment tail contains candidates that explain observed peaks. No tested positive
fragment threshold met a `0.001` absolute cosine-loss target. If this rule is revisited, thresholds
below `0.001` should be tested first.

## Better selection variable

For probability output, MASSter already forms the final channel distribution as

```text
P(fragment, ion/H | spectrum)
  = P(fragment | spectrum) * P(ion/H | fragment, spectrum)
```

Selection is nevertheless staged separately: a fragment threshold is applied first and top-k is
then applied independently within each retained fragment. A better inference diagnostic is to rank
all channels in a spectrum by their joint probability and apply a global cumulative-mass or global
top-k rule. This can be tested with the existing checkpoint before training a direct joint-support
head.

The current spectrum-relative joint-threshold implementation preserves at least one channel for
every fragment to satisfy conditional normalization. That fallback prevents truly global pruning
of weak fragments and should be accounted for when interpreting a joint-threshold sweep.

## Learning zero intensity

“Predict zero” has three distinct requirements:

1. the spectral loss must penalize intensity assigned to peaks absent from the target;
2. the model must receive a per-candidate signal that a channel is unsupported; and
3. inference must be able to emit an exact zero, normally by omitting the rejected sparse peak.

The original independent MASSter intensity path satisfied only the first requirement. For a
valid fragment/ion/H candidate it computed

```text
z_i = corrected_fragment_logit_i + ion_h_logit_i
a_i = softplus(z_i)
```

in
[`masster_model.py`](../../../src/fragnnet/model/masster_model.py). `softplus(z)` is
strictly positive for every finite `z`, so every chemically valid candidate receives nonzero
intensity. A sufficiently negative logit produces a very small value but never an exact zero, and
its gradient also becomes small because the derivative of softplus is sigmoid.

### Why cosine favors many small false peaks

Suppose the target contains one unit-intensity peak. The model predicts that peak correctly and
distributes a fixed total false intensity `F` over `N` unmatched peaks. Its cosine similarity is

```text
                 1
cosine = -----------------
         sqrt(1 + F^2 / N)
```

The score approaches one as `N` increases. Cosine therefore rewards dividing the same false
intensity among more candidates. Hungarian matching removes many-to-one assignment ambiguity but
does not remove this L2 splitting incentive.

Jensen–Shannon distance does not have this particular failure: the unmatched contribution is
additive in probability mass, so splitting fixed false mass among more unmatched bins does not
reduce the loss. The fixed-total-smoothed `KL(true || predicted)` is also invariant to that split.
KL still mainly penalizes the *total* probability displaced from observed peaks, however; it does
not independently classify which candidate should be absent. The optional square-root intensity
transform makes small peaks more important to the spectral loss, but likewise does not define an
explicit zero decision.

### ICEBERG and GLACIER comparison

[ICEBERG](https://github.com/coleygroup/ms-pred/blob/main/src/ms_pred/iceberg/inten_model.py)
uses sigmoid intensity outputs, then explicitly masks chemically invalid or padded positions to
zero. A sigmoid is also strictly positive for finite logits. ICEBERG obtains effective sparsity
primarily from its upstream fragment generator and inference-time fragment threshold rather than
from the intensity activation. The vendored implementation has the same activation-and-mask
pattern in [`iceberg/model.py`](../../../src/fragnnet/iceberg/model.py).

[GLACIER](https://github.com/coleygroup/ms-pred/blob/main/src/ms_pred/glacier/joint_model.py)
similarly separates fragment detection from intensity prediction. Its detected fragment objects
receive sigmoid intensities, invalid positions are masked, and sparse inference retains top-k
outputs. Its important lesson is therefore the separation of *existence* from *magnitude*, not the
choice of sigmoid itself. See also the
[GLACIER paper](https://arxiv.org/abs/2606.29161).

### Implemented deterministic support gate

MASSter already has a separate ion/H support head with observational targets. Hungarian
attribution produces hard zero/one targets, while Gaussian-posterior attribution produces
fractional responsibilities. The balanced binary support loss directly teaches whether each
chemically valid channel is observed. The independent intensity path now uses this head for both
differentiable training gates and evaluation selection whenever the separate binary support head
is enabled.

The output is

```text
g_i = sigmoid(support_logit_i)
a_i = softplus(intensity_logit_i)
training_intensity_i = g_i * a_i
```

with total training loss

```text
spectral_distance(normalize(true), normalize(g * a))
    + support_loss_weight * balanced_BCE(support_logits, support_targets).
```

This remains a non-probabilistic independent intensity model: `a_i` is an unconstrained positive
magnitude and is not normalized by the model. The only probability is the already-existing binary
support classifier. JSS or KL normalizes the assembled spectrum internally for comparison.

For an unsupported candidate, BCE pushes its support logit negative, making `g_i` approach zero;
the spectral loss simultaneously penalizes its unmatched gated intensity. For a supported
candidate, BCE pushes `g_i` toward one and the magnitude head learns its relative intensity. This
gives each false candidate a direct zero target instead of asking a single softplus output to learn
both existence and magnitude.

Sigmoid gating is differentiable but still does not produce literal floating-point zero. Exact
zero should be introduced at inference:

```text
keep_i = support_score_i >= tuned_threshold
output_intensity_i = keep_i * g_i * a_i
```

Rejected candidates should be omitted from the sparse peak list. Hard gating during early
training is not recommended because a mistakenly rejected true channel would lose its spectral
gradient. ReLU intensity outputs can generate exact zeros directly, but negative logits have zero
gradient and can create permanently dead channels.

`masster_min_ion_h_absolute_probability` configures the evaluation-only exact-zero cutoff;
`0.0` disables it. The absolute selector is restricted to independent intensity with a separate
binary support head. It may remove all channels of an unsupported fragment, but preserves the
highest-support remaining channel when a whole spectrum would otherwise be empty. The six
independent JSS/JSS-sqrt/KL Gaussian/Hungarian configs use a conservative `0.01` absolute floor.

### Threshold and attribution caveats

The existing `masster_min_ion_h_relative_probability: 0.01` rule compares each support score
with one percent of the maximum score for that fragment. It always preserves the fragment's
highest-scoring valid channel. Consequently, it cannot reject an entirely unsupported fragment:
if all of its support scores are uniformly small, at least the largest still survives. The
implemented absolute support floor supplies that second decision. It should still be tuned on
validation data; a genuinely global final intensity threshold or fragment-level support gate may
be worth testing if the support floor alone leaves too many weak fragments.

Hungarian attribution is the clearest primary experiment for sparse gating because it provides
hard zero/one labels. Its limitation is mass aliasing: when multiple structural pathways have
indistinguishable m/z, the observed spectrum cannot identify which pathway generated the peak,
but Hungarian assigns one and labels the alternatives zero. Gaussian-posterior attribution is
more honest about this ambiguity, but its fractional targets naturally produce softer, less sparse
gates. Results should therefore be evaluated after equal-mass contributions are coalesced, and the
Gaussian version should remain an ambiguity-aware ablation.

### Preserve the recall advantage

MASSter's broad candidate enumeration produces substantially better recall and weighted recall
than a more aggressively pruned detector. That coverage is valuable and is a reason not to copy
ICEBERG or GLACIER's upstream pruning policy directly. Some improvement in ordinary recall is
mechanical—a dense list has more opportunities to match each observed peak—but improved weighted
recall also shows that the candidate set covers important high-intensity observations. Neither
metric penalizes false predictions, so they must be interpreted together with spectral distance,
precision, and emitted peak count.

The support gate should therefore attenuate intensity continuously during training without
removing candidates:

```yaml
masster_apply_channel_selection_in_training: False
```

This preserves spectral-loss gradients for a true candidate whose support score is temporarily
wrong, allows it to recover later, and retains the full candidate coverage available to JSS or KL.
Balanced BCE pushes unsupported gates toward zero, while the spectral objective trains the gated
magnitudes. Literal zeros are unnecessary during optimization.

At inference, threshold selection should include the dense `threshold = 0` reference and sweep
only conservative positive thresholds. The production threshold should be selected on a Pareto
curve rather than from peak-count precision alone. A suitable acceptance rule is, for example:

```text
weighted-recall decrease <= 0.005
absolute cosine decrease <= 0.001
```

Among thresholds satisfying those coverage constraints, select the one giving the largest
reduction in emitted peak count or the best count precision. The evaluation should report at least
weighted recall, ordinary recall, JSS or cosine similarity, count precision, and peaks per
spectrum. This uses exact-zero inference only where validation shows it does not sacrifice
MASSter's principal recall advantage.

## FraGNNet formula-support gate

The same low-intensity tail exists in FraGNNet because softmax assigns positive probability to
every finite candidate. The standard FraGNNet baseline uses cross entropy rather than cosine, so
it does not reward splitting fixed false mass, but it is indifferent to how that mass is divided.
Each false candidate receives a gradient proportional to its own probability, which becomes very
small when the false mass is spread over many candidates. OOS handles observed intensity outside
the candidate set, while the null formula handles invalid candidate states; neither supervises
the support of valid formulas.

FraGNNet now has an opt-in, separately parameterized support head. Its sigmoid probability gates
the existing joint candidate logit before the normalizing softmax:

```text
gate_i = sigmoid(support_logit_i)
effective_joint_logit_i = intensity_logit_i + log(gate_i)
```

Thus the intensity head remains a normalized probabilistic model, while support has independent
binary semantics. The support head starts at probability `0.99`; because the same initial gate is
applied to real and null candidates, enabling it initially preserves the legacy distribution.
Training uses soft gates. Evaluation and inference apply an absolute sigmoid-probability threshold
and always retain the highest-support real joint state in each spectrum, preventing an empty
prediction. The existing exported-intensity threshold remains a second, downstream filter.

Targets are constructed against all emitted isotope masses using the existing instrument-aware
tolerance, then reduced existentially to their parent formula. No intensity prior is used for
support attribution, keeping support and magnitude supervision separate. Gaussian-posterior and
Hungarian attribution are both supported. Gaussian targets preserve mass ambiguity; Hungarian
targets are harder and potentially sparser.

The first matched experiments now use the established NIST20 MA/MI depth-3 baseline: four adducts,
per-CE spectra, CMF candidates, instrument/fragmentation-mode conditioning, and instrument-aware
mass tolerances. They keep its cross-entropy objective and add support BCE with weight `0.1`:

- [`fraggnn_d3_ma_mi_nist20_amolc_all_formula_support_gaussian.yml`](../../../config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_formula_support_gaussian.yml)
- [`fraggnn_d3_ma_mi_nist20_amolc_all_formula_support_hungarian.yml`](../../../config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_formula_support_hungarian.yml)

Compare them with
[`fraggnn_d3_ma_mi_nist20_amolc_all.yml`](../../../config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all.yml)
using cosine similarity, recall, weighted recall, emitted peak count, and threshold curves.

## Barcode follow-up

Barcode proposal supervision may concentrate fragment probability on observationally supported
fragment identities, so the same diagnostics must be repeated on a matched barcode model. The
new training config is:

[`nist20_merged_u256_e1024_mix2_barcode_h_support_w01.yml`](../../../config/masster/merged/channel_selection/nist20_merged_u256_e1024_mix2_barcode_h_support_w01.yml)

It matches the no-barcode U256/E1024/Mix2 H-support configuration except for experiment identity,
tags, `masster_proposal_barcode_target: True`, and the corrected merged-target construction
described below. Checkpoint uploading remains disabled. This comparison therefore tests the
barcode model on the intended consensus target; it is not a strict one-variable ablation against
the historical exact-union checkpoint.

### Corrected merged target

The historical merge pooled every spectrum sharing molecule, precursor type, and instrument. It
summed intensities only at exactly equal m/z values and retained every union peak. On the 4,222
validation groups, each target combined 12.32 source spectra on average and contained 208.49
peaks. Although 66.7% of those peaks were below 1% of the merged base peak, they contributed only
4.32% of total intensity. This explains why top-2 caused a large count-recall loss with a much
smaller cosine loss.

The new config uses `spec_params.merge_method: tolerance_consensus`. It:

1. L1-normalizes every source spectrum;
2. clusters peaks using the instrument tolerance and a common-overlap rule;
3. averages replicate spectra within each CE schedule;
4. gives distinct CE schedules equal weight;
5. removes merged peaks below 0.1% of the base peak through
   `target_rel_ints_thresh: 0.001`;
6. reports `wrecall` alongside count recall.

On the same 4,222 validation groups, the implemented tolerance consensus contains 95.87 peaks per
target on average. Applying the 0.1% target floor leaves 71.00 peaks, removing 25.94% of the
tolerance-merged peaks. Relative to the historical 208.49-peak exact union, the new training target
contains about 66% fewer peak entries; most of that reduction is consolidation of masses that are
indistinguishable at the configured FT tolerance rather than intensity filtering.

The legacy `exact_sum` method remains the template default for checkpoint reproducibility. Because
the consensus construction changes training targets, it requires a newly trained checkpoint.

The matched no-barcode control is
[`nist20_merged_u256_e1024_mix2_no_barcode_consensus_h_support_w01.yml`](../../../config/masster/merged/channel_selection/nist20_merged_u256_e1024_mix2_no_barcode_consensus_h_support_w01.yml).
It differs from the barcode consensus arm only in experiment metadata and
`masster_proposal_barcode_target: False`, allowing the barcode effect to be isolated.

The matched follow-up should determine whether barcode supervision changes:

1. soft fragment attribution;
2. fragment any-match rate;
3. probability-weighted fragment attribution;
4. the gap between fragment diagnostics and expanded-channel precision;
5. the cosine/recall cost of fragment and global joint-channel pruning.

Historical barcode run `ka68eu6p` and recent barcode channel-selection runs `lbza2x9w` and
`rlz4xf63` did not expose usable uploaded checkpoints when checked on 2026-09-02. The barcode
conclusion remains pending until the matched checkpoint completes.
