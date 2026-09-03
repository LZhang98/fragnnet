# MASShatter fragment and ion/H channel-selection findings

> **Status:** FINDING — PROVISIONAL, BARCODE CONTROL PENDING &nbsp;·&nbsp; **Updated:** 2026-09-02
>
> The current count-based spectrum precision conflates fragment selection with ion/H
> expansion. It is useful for measuring output clutter, but it is not fragment precision.

## Experiment

The completed validation sweep used checkpoint `tq4bdn3g` with:

- config
  [`nist20_merged_u256_e1024_mix2_no_barcode_h_support_w01.yml`](../../config/masshatter/merged/channel_selection/nist20_merged_u256_e1024_mix2_no_barcode_h_support_w01.yml);
- results
  [`h_support_fragment_channel_sweep_tq4bdn3g.csv`](../../output/eval/h_support_fragment_channel_sweep_tq4bdn3g.csv);
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

For probability output, MASShatter already forms the final channel distribution as

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

The independent MASShatter intensity path currently satisfies only the first requirement. For a
valid fragment/ion/H candidate it computes

```text
z_i = corrected_fragment_logit_i + ion_h_logit_i
a_i = softplus(z_i)
```

in
[`mass_shatter_model.py`](../../src/fragnnet/model/mass_shatter_model.py). `softplus(z)` is
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
pattern in [`iceberg/model.py`](../../src/fragnnet/iceberg/model.py).

[GLACIER](https://github.com/coleygroup/ms-pred/blob/main/src/ms_pred/glacier/joint_model.py)
similarly separates fragment detection from intensity prediction. Its detected fragment objects
receive sigmoid intensities, invalid positions are masked, and sparse inference retains top-k
outputs. Its important lesson is therefore the separation of *existence* from *magnitude*, not the
choice of sigmoid itself. See also the
[GLACIER paper](https://arxiv.org/abs/2606.29161).

### Proposed deterministic support gate

MASShatter already has a separate ion/H support head with observational targets. Hungarian
attribution produces hard zero/one targets, while Gaussian-posterior attribution produces
fractional responsibilities. The balanced binary support loss directly teaches whether each
chemically valid channel is observed. At present, this head controls evaluation selection but is
disconnected from the independent training intensity.

The proposed output is

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

### Threshold and attribution caveats

The existing `mass_shatter_min_ion_h_relative_probability: 0.01` rule compares each support score
with one percent of the maximum score for that fragment. It always preserves the fragment's
highest-scoring valid channel. Consequently, it cannot reject an entirely unsupported fragment:
if all of its support scores are uniformly small, at least the largest still survives. A production
zero policy therefore needs a validation-tuned absolute support floor, a genuinely global final
intensity threshold, or a fragment-level support gate in addition to the relative H-selection
rule.

Hungarian attribution is the clearest primary experiment for sparse gating because it provides
hard zero/one labels. Its limitation is mass aliasing: when multiple structural pathways have
indistinguishable m/z, the observed spectrum cannot identify which pathway generated the peak,
but Hungarian assigns one and labels the alternatives zero. Gaussian-posterior attribution is
more honest about this ambiguity, but its fractional targets naturally produce softer, less sparse
gates. Results should therefore be evaluated after equal-mass contributions are coalesced, and the
Gaussian version should remain an ambiguity-aware ablation.

### Preserve the recall advantage

MASShatter's broad candidate enumeration produces substantially better recall and weighted recall
than a more aggressively pruned detector. That coverage is valuable and is a reason not to copy
ICEBERG or GLACIER's upstream pruning policy directly. Some improvement in ordinary recall is
mechanical—a dense list has more opportunities to match each observed peak—but improved weighted
recall also shows that the candidate set covers important high-intensity observations. Neither
metric penalizes false predictions, so they must be interpreted together with spectral distance,
precision, and emitted peak count.

The support gate should therefore attenuate intensity continuously during training without
removing candidates:

```yaml
mass_shatter_apply_channel_selection_in_training: False
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
MASShatter's principal recall advantage.

## Barcode follow-up

Barcode proposal supervision may concentrate fragment probability on observationally supported
fragment identities, so the same diagnostics must be repeated on a matched barcode model. The
new training config is:

[`nist20_merged_u256_e1024_mix2_barcode_h_support_w01.yml`](../../config/masshatter/merged/channel_selection/nist20_merged_u256_e1024_mix2_barcode_h_support_w01.yml)

It matches the no-barcode U256/E1024/Mix2 H-support configuration except for experiment identity,
tags, and `mass_shatter_proposal_barcode_target: True`. Checkpoint uploading is an operational
difference and remains disabled. This matched comparison should determine whether barcode
supervision changes:

1. soft fragment attribution;
2. fragment any-match rate;
3. probability-weighted fragment attribution;
4. the gap between fragment diagnostics and expanded-channel precision;
5. the cosine/recall cost of fragment and global joint-channel pruning.

Historical barcode run `ka68eu6p` and recent barcode channel-selection runs `lbza2x9w` and
`rlz4xf63` did not expose usable uploaded checkpoints when checked on 2026-09-02. The barcode
conclusion remains pending until the matched checkpoint completes.
