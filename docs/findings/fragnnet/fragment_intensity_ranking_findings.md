# Fragment intensity ranking findings

> **Status:** FINDING — PROVISIONAL NODEMLP EVIDENCE &nbsp;·&nbsp; **Updated:** 2026-08-11
>
> Empirical findings for complementary cuts, sibling competition, global base-peak selection,
> and collision-energy-dependent fragment ranking. Experiment specifications and promotion gates
> live in the separate
> [`complementary-cut`](../../plans/fragnnet/complementary_cut_supervision_experiment_plan.md) and
> [`global-ranking`](../../plans/fragnnet/global_fragment_intensity_ranking_experiment_plan.md) plans.

## Bottom line

The current responsibility-weighted complementary-cut objective is a soft negative: it concentrates
on high-confidence pairs the model already predicts well and recovers only 3.37% of exact
current-DAG intensity headroom in the absolute oracle edit. The larger unresolved problem is global
intensity ranking, especially selecting the correct base peak as collision energy changes.

The analyzed checkpoint, run `9t63zils`, is a NodeMLP rather than the required matched GINE-1
control. These results identify bottlenecks and prioritize experiments; they do not satisfy the
formal promotion gates.

## Complementary-cut target volume and local performance

The corrected target builder sampled 500 molecules with seed 17 and produced:

- 9,810 spectra;
- 2,844,736 retained complementary pairs;
- 133 validation molecules and 2,481 validation spectra overlapping saved predictions;
- 626,890 validation pairs;
- Kish effective sample size 26,066 after responsibility and overlap weighting.

On the prediction overlap:

| Complementary-cut diagnostic                            | Result           |
| ------------------------------------------------------- | ---------------: |
| responsibility-weighted mean TV                         |           0.0591 |
| unweighted mean TV                                      |           0.1902 |
| molecule-cluster 95% interval for weighted TV           | [0.0492, 0.0695] |
| responsibility-weighted dominant-side accuracy          |            95.4% |
| unweighted dominant-side accuracy                       |            86.5% |
| high-confidence (`max(q_cut_evidence) >= 0.8`) mean TV  |           0.0529 |
| high-confidence dominant-side accuracy                  |            96.0% |

Residual error is concentrated in ambiguous targets that carry little effective weight. Mean TV
was 0.314 for target confidence `[0.5, 0.6)` and 0.037 for `[0.99, 1.0]`. Kish ESS was only 4.2% of
the raw pair count. Across equal-count weight deciles, unweighted TV fell from 0.310 in the lowest
decile to 0.062 in the highest. The weighted headline therefore describes a narrow,
high-responsibility subset; it does not establish that complementary ranking is learned uniformly.

A cut-blind atom-count-ratio baseline had weighted TV 0.335 and unweighted TV 0.305. Uniform
baselines had weighted and unweighted TV 0.479 and 0.357. The model contains cut-relevant
information beyond these simple nulls, but the same-context replicate-TV floor is still missing.

## Complementary-cut oracle controls

A five-step regularized inference-time edit of node logits toward the observed cut targets changed
mean reconstructed cosine by `+0.00675`, with molecule-cluster interval
`[+0.00388, +0.00977]`.

| Oracle target                                      | Mean cosine change |
| -------------------------------------------------- | -----------------: |
| observed cut direction                             |           +0.00675 |
| within-pair shuffled direction                     |           -0.00542 |
| model direction with observed confidence           |           +0.00297 |
| exact current model pair ratio                     |           +0.00002 |

Observed direction exceeded the matched-confidence model-direction control by `+0.00377`, with
molecule-cluster interval `[+0.00261, +0.00504]`. Generic sharpening explains about 44% of the
observed-target edit; the observed cut direction supplies a smaller residual channel. The shuffled
contrast is not a practical effect size because shuffled targets actively damage predictions.

The gain was concentrated in the worst current-cosine quartile (`+0.0162`) and pair-rich spectra
(`+0.0112`), with approximately no gain in the best quartile or pair-sparse spectra. Stronger edits
degraded cosine.

Exact current-DAG intensity headroom on the same overlap was 0.19998. The absolute observed edit
recovered 3.37% of it, below the plan's 5% gate. Observed minus shuffled recovered 6.08%, but that
number is inflated by harm from the shuffled control and must not be used as recovered headroom.

## Global intensity ranking is the larger error

Across all 96,153 spectra in the saved validation export:

- mean `cos_hun_inst_tol` was 0.752;
- base-peak accuracy was 57.4%;
- mean matched-peak intensity Spearman was 0.613;
- predicted entropy exceeded observed entropy by 0.688 nats on average.

A **base peak** is the highest-intensity observed peak. A hit requires the largest predicted peak
to match its mass under the instrument-specific tolerance. Mean cosine was 0.912 on base-peak hits
and 0.535 on misses. This is a diagnostic association, not a causal decomposition: a generally bad
prediction can cause both errors.

The exact cut-overlap cohort reproduced the pattern: base-peak accuracy 59.5%, Spearman 0.627,
entropy excess 0.676, and mean cosine 0.910 versus 0.537 for base-peak hits versus misses.

### Support versus ranking decomposition

| Base-peak outcome                                  | Spectra | Fraction |
| -------------------------------------------------- | ------: | -------: |
| correct base peak                                  |  55,201 |    57.4% |
| correct mass exported but ranked below another     |  34,439 |    35.8% |
| correct mass absent from exported prediction support | 6,513 |     6.8% |

Thus 84.1% of the 40,952 base-peak misses retain the correct mass in exported support and are
ranking failures under this definition. In those supported misses, the observed base peak receives
47.9% of the predicted winner's intensity on average. Their mean DAG-oracle cosine is 0.979, while
their model cosine is 0.578.

“Absent from exported support” is not equivalent to absent from the raw DAG: inference applied the
default `min_peak_int=1.0` filter, so this category mixes DAG coverage failures with candidates to
which the model assigned effectively zero intensity.

**Resolved 2026-08-12** in
[`dominant_peak_outcome_decomposition.md`](dominant_peak_outcome_decomposition.md), which
reclassifies against the unfiltered DAG support on the NIST20 test export: the 6.8% splits into
5.2% genuinely outside the DAG and 0.8% DAG-covered but predicted below the export floor. The
ranking-versus-coverage ratio is therefore ~7:1, not the 84%/16% the export-only split implied.

## Collision energy changes the winner, but the model misses the transition

Within each `(molecule, adduct, instrument, fragmentation mode)` context, the lowest and highest
available collision energies were compared. This holds molecular identity and acquisition context
fixed instead of relying on a cross-sectional CE trend.

For 7,519 FT contexts:

| Diagnostic                         | Lowest CE | Highest CE |
| ---------------------------------- | --------: | ---------: |
| base-peak accuracy                 |     92.4% |      42.3% |
| observed base mass in export       |     99.5% |      83.3% |
| accuracy conditional on support    |     92.8% |      50.8% |
| model cosine                       |     0.933 |      0.642 |
| DAG-oracle cosine                  |     0.998 |      0.896 |

The molecule-balanced paired base-peak change was -51.7 percentage points, with 95% interval
`[-53.1, -50.3]`. High-energy support loss is real, but it does not explain the full decline: even
when the observed mass remains available, selection accuracy falls sharply.

For QTOF contexts, export support and DAG-oracle cosine were nearly unchanged from low to high CE,
while base-peak accuracy fell from 84.3% to 64.4%. That is a cleaner CE-conditioned ranking defect.

For adjacent FT energy levels:

- the observed base peak changed in 28.1% of transitions;
- conditional on an observed change, the model changed its winner in 38.5%;
- it followed the correct mass direction in 34.8%;
- it selected the correct winner at both endpoints in 13.1%;
- the observed winner moved downward by a median 44 Da, while the median predicted change was 0 Da.

Across large CE gaps the model eventually changes a winner more often, but the endpoint identities
remain unreliable. The model is therefore not CE-blind; it learns a coarse energy response without
reliably learning which structural channel should dominate or where the transition occurs.

Base-peak accuracy also declined from 61.5% in the smallest atom-count quartile to 53.4% in the
largest. Alkaloids, organic oxygen compounds, and organoheterocyclic compounds were among the
weaker broad structural classes. These class results are descriptive and confounded by acquisition
strata; the within-context CE comparison is stronger evidence.

## Static CE-average hypothesis is rejected; winner inertia remains

A follow-up test asked whether the model had learned nearly fixed fragment weights across all CEs,
equivalent to reusing a context-average spectrum. It used all 7,519 multi-CE FT contexts and 520
multi-CE QTOF contexts. For every CE level, it compared the prediction with both the CE-specific
observed spectrum and the average of all other CE levels in the same fixed context. It also measured
exact node-probability changes on a deterministic 1,500-molecule sample; local node indices aligned
across CE in the saved export.

| Diagnostic                                      |    FT | QTOF |
| ----------------------------------------------- | ----: | ---: |
| observed TV variation around context mean       | 0.441 | 0.371 |
| predicted TV variation around context mean      | 0.376 | 0.286 |
| predicted/observed variation                    | 85.8% | 79.2% |
| cosine of predicted versus observed CE residual | 0.710 | 0.616 |
| prediction closer to CE-specific target         | 79.8% | 71.3% |

For FT, mean TV from the prediction to its CE-specific target was 0.456, versus 0.614 to the
leave-one-CE-out context average. The corresponding QTOF values were 0.489 and 0.586. The model is
therefore not simply outputting one CE-averaged spectrum.

However, predicted variation was smaller than observed variation in 90.4% of FT contexts and 92.1%
of QTOF contexts. The shortfall was concentrated at winner changes:

| Adjacent transition     | Observed peak TV | Predicted peak TV |
| ----------------------- | ---------------: | ----------------: |
| FT, winner stable       |            0.186 |             0.164 |
| FT, winner changes      |            0.298 |             0.205 |
| QTOF, winner stable     |            0.153 |             0.112 |
| QTOF, winner changes    |            0.349 |             0.170 |

When the observed winner changed, adjacent node-distribution cosine remained 0.921 for FT and 0.930
for QTOF. The predicted top node changed in only 35.9% and 37.8% of those transitions,
respectively.

The smaller predicted TV does **not** imply that a uniform CE-response gain would help. The TV
ratio is L1-derived, while the least-squares gain must be computed in L2 geometry. For mass-aligned
adjacent change vectors, the diagnostic now computes

\[
\alpha^*=\frac{\sum \Delta p\cdot\Delta q}{\sum\lVert\Delta p\rVert_2^2}
\]

with molecule-cluster bootstrap intervals:

| Adjacent-transition diagnostic | FT | QTOF |
| --- | ---: | ---: |
| mean delta cosine, all transitions | 0.591 `[0.586, 0.597]` | 0.486 `[0.463, 0.508]` |
| optimal L2 gain, all transitions | 0.831 `[0.822, 0.841]` | 0.756 `[0.712, 0.800]` |
| mean delta cosine, observed winner switches | 0.593 `[0.587, 0.600]` | 0.478 `[0.449, 0.507]` |
| optimal L2 gain, observed winner switches | 1.010 `[0.996, 1.025]` | 0.847 `[0.760, 0.936]` |

Thus a uniform amplification is unsupported. Overall, the model is already optimally shrunk
relative to its directional error. At observed crossings, FT is statistically consistent with unit
gain rather than clear under-amplification, while QTOF still favors shrinkage. The missing signal is
primarily **which masses should rise and fall**, not a global response multiplier. This rules out a
generic CE-response temperature or residual rescaling as the next experiment.

Together these results support a weaker and more precise model:

```text
fragment score at CE = strong static structural prior + real but directionally imperfect,
                       appropriately shrunk CE correction
```

The model tracks much of the continuous CE-driven intensity movement, but the correction is often
assigned to the wrong masses and does not reliably reverse two contenders at the observed crossing.
The appropriate target is therefore the contender ordering at both endpoints—not generic CE
variation, uniform amplification, or another CE access path.

Mass-level calculations used 0.01-Da bins and inherit the filtered prediction export limitation.
The exact-node result does not depend on mass binning and independently shows the same winner
inertia. All reported intervals in the saved summary use molecule-cluster resampling.

## Replicate noise floors: FT directional headroom is real; QTOF is unresolved

A model-free Step 0 measured two replicate limits before building the contender-pair target. It
used observed validation spectra only; the prior scan supplied spectrum IDs, controlled-context
fields, and the filled NCE values. Winner identity uses instrument-relative mass tolerance. Delta
cosine uses the same 0.01-Da grid as the provisional model diagnostic. The common grid removes a
representation mismatch, but it does not by itself make values from different cohorts or estimators
directly comparable. Intervals use 5,000 molecule-cluster bootstrap samples.

Coverage comes first because QTOF replication is concentrated in very few molecules:

| Instrument/adduct | Replicated CE levels | Same-CE pairs | Molecules | Winner-flip floor |
| --- | ---: | ---: | ---: | ---: |
| FT, all | 3,794 | 4,519 | 1,202 | 0.171 `[0.156, 0.186]` |
| FT, `[M+Cl]-` | 141 | 158 | 65 | 0.234 `[0.177, 0.290]` |
| FT, `[M+H]+` | 1,392 | 1,653 | 528 | 0.148 `[0.122, 0.172]` |
| FT, `[M+Na]+` | 222 | 290 | 80 | 0.155 `[0.100, 0.217]` |
| FT, `[M-H]-` | 2,039 | 2,418 | 739 | 0.184 `[0.166, 0.204]` |
| QTOF, all | 122 | 160 | 10 | 0.200 `[0.103, 0.319]` |
| QTOF, `[M+H]+` | 77 | 81 | 8 | 0.222 `[0.073, 0.385]` |
| QTOF, `[M+Na]+` | 4 | 4 | 1 | 0.000 `[0.000, 0.000]` |
| QTOF, `[M-H]-` | 41 | 75 | 3 | 0.187 `[0.083, 0.333]` |

The provisional validation scan contained no IT replicate cohort, so no IT floor is estimated.

The second audit retained adjacent CE pairs with at least two observed replicates at both
endpoints. For each transition it enumerated disjoint replicate choices, formed independent
\(\Delta q\) estimates, averaged their cosine within the transition, and then bootstrapped
molecules:

| Cohort | Adjacent transitions | Molecules | Replicate delta cosine |
| --- | ---: | ---: | ---: |
| FT, all transitions | 1,488 | 555 | 0.831 `[0.799, 0.862]` |
| FT, stable winner | 916 | 414 | 0.797 `[0.756, 0.838]` |
| **FT, winner switches** | **572** | **386** | **0.885 `[0.856, 0.913]`** |
| QTOF, all transitions | 103 | 7 | 0.069 `[-0.185, 0.392]` |
| QTOF, stable winner | 88 | 7 | 0.075 `[-0.193, 0.411]` |
| QTOF, winner switches | 15 | 7 | 0.035 `[-0.193, 0.302]` |

Switch-only delta reliability by adduct is:

| Instrument/adduct | Switching transitions | Molecules | Replicate delta cosine |
| --- | ---: | ---: | ---: |
| FT, `[M+Cl]-` | 24 | 23 | 0.900 `[0.819, 0.961]` |
| FT, `[M+H]+` | 196 | 119 | 0.840 `[0.771, 0.902]` |
| FT, `[M+Na]+` | 41 | 30 | 0.830 `[0.712, 0.935]` |
| FT, `[M-H]-` | 311 | 261 | 0.920 `[0.893, 0.943]` |
| QTOF, `[M+H]+` | 10 | 5 | 0.094 `[-0.241, 0.504]` |
| QTOF, `[M+Na]+` | 0 | 0 | unavailable |
| QTOF, `[M-H]-` | 5 | 3 | -0.085 `[-0.340, 0.559]` |

The FT `[M+Cl]-` and `[M+Na]+` switch subsets are directionally consistent with the larger FT
strata but remain below the predeclared 50-transition coverage threshold on their own.

The model comparison on these exact transitions is:

| FT cohort | Model delta cosine | Replicate-minus-model difference | Interpretation |
| --- | ---: | ---: | --- |
| all switches | 0.765 `[0.737, 0.793]` | 0.120 `[0.091, 0.149]` | clears zero |
| `[M+Cl]-` | 0.687 `[0.582, 0.789]` | 0.213 `[0.088, 0.335]` | below coverage threshold |
| `[M+H]+` | 0.743 `[0.699, 0.793]` | 0.097 `[0.036, 0.154]` | clears zero |
| `[M+Na]+` | 0.720 `[0.618, 0.829]` | 0.110 `[-0.017, 0.240]` | does not clear zero |
| `[M-H]-` | 0.791 `[0.759, 0.822]` | 0.129 `[0.095, 0.163]` | clears zero |

The full-cohort model value cannot be compared directly with the double-replicated subset. On the
same 572 FT switching transitions, replicate delta cosine is 0.885 `[0.856, 0.913]`, the model reaches
0.765 `[0.737, 0.793]`, and their paired molecule-clustered difference is 0.120
`[0.091, 0.149]`. The model reaches only 0.593 `[0.587, 0.600]` on the full FT switch cohort, but
the difference is not explained by easier molecules. Restricting both groups to the 341 molecules
that contribute matched and unmatched transitions gives 0.779 `[0.752, 0.807]` on matched
transitions and 0.604 `[0.580, 0.628]` on unmatched transitions.

The measured cohort differs primarily in transition signal and target aggregation:

| FT switch cohort | Transitions | Median \(\lVert\Delta q\rVert_2\) | Median CE gap | Model delta cosine |
| --- | ---: | ---: | ---: | ---: |
| double-replicated | 572 | 0.396 | 5.0 | 0.765 |
| not double-replicated | 20,001 | 0.235 | 10.0 | 0.588 |

The matched cohort remains easier within every full-population \(\lVert\Delta q\rVert_2\)
quartile:

| Change-vector quartile | Matched transitions / cosine | Unmatched transitions / cosine |
| --- | ---: | ---: |
| Q1, smallest change | 80 / 0.539 | 5,064 / 0.412 |
| Q2 | 59 / 0.668 | 5,084 / 0.563 |
| Q3 | 107 / 0.791 | 5,036 / 0.675 |
| Q4, largest change | 326 / 0.830 | 4,817 / 0.710 |

Crucially, the replicate ceiling falls at small switch magnitudes without eliminating the paired
headroom:

| FT switch magnitude | Transitions | Replicate delta cosine | Model delta cosine | Paired gap |
| --- | ---: | ---: | ---: | ---: |
| Q1, \(\lVert\Delta q\rVert_2\leq0.152\) | 80 | 0.716 | 0.539 | 0.177 `[0.068, 0.279]` |
| Q2, \(\leq0.238\) | 59 | 0.774 | 0.668 | 0.106 `[-0.031, 0.234]` |
| Q3, \(\leq0.351\) | 107 | 0.899 | 0.791 | 0.109 `[0.061, 0.155]` |
| Q4, \(>0.351\) | 326 | 0.942 | 0.830 | 0.113 `[0.084, 0.143]` |
| all | 572 | 0.885 | 0.765 | 0.120 `[0.091, 0.149]` |

Q2 is inconclusive on only 59 transitions, but its positive point estimate lies between the
neighboring strata; there is no observed sign reversal. The corrected 0.120 gap is therefore not
only a large-change artifact. This strengthens the FT switch decision even though 0.885 remains an
invalid universal ceiling.

Stable transitions behave differently:

| FT stable magnitude | Transitions | Replicate delta cosine | Model delta cosine | Paired gap |
| --- | ---: | ---: | ---: | ---: |
| Q1, \(\lVert\Delta q\rVert_2\leq0.093\) | 170 | 0.556 | 0.554 | 0.002 `[-0.097, 0.098]` |
| Q2, \(\leq0.149\) | 150 | 0.680 | 0.615 | 0.065 `[-0.013, 0.141]` |
| Q3, \(\leq0.225\) | 184 | 0.838 | 0.733 | 0.105 `[0.044, 0.163]` |
| Q4, \(>0.225\) | 412 | 0.920 | 0.828 | 0.093 `[0.062, 0.123]` |
| all | 916 | 0.797 | 0.723 | 0.074 `[0.034, 0.110]` |

The smallest-change stable anchors have no measurable directional headroom, and Q2 remains
inconclusive. These anchors may still regularize against a flip-happy solution, but they should not
be counted as evidence that the model learned switch direction. Apply the v1 confidence eligibility
floor to stable anchors as well as switches, and never pool stable and switch metrics when deciding
whether the contender-pair arm worked.

Larger observed changes explain part of the cohort difference: 57.0% of matched transitions are in
Q4. The remaining within-quartile gap is consistent with target denoising because the matched
\(\Delta q\) averages at least two spectra per endpoint, whereas unmatched transitions generally
use one spectrum. That interpretation is plausible but not identified separately from other
acquisition differences. The replicate estimate therefore does not transport to the full FT
population. The FT kill criterion nevertheless does **not** fire on the valid paired statistic.

Replicate-to-replicate cosine is also not literally the ceiling for model-versus-observed cosine.
Under the approximate independent-error model \(\Delta q_a=T+e_a\) and
\(\Delta q_b=T+e_b\), replicate agreement estimates \(\rho^2\), so 0.885 implies a
single-observation reliability of approximately \(\sqrt{0.885}=0.941\). The model diagnostic
averages observed replicates at each endpoint, which can raise its effective ceiling further. This
correction is heuristic rather than a strict bound:
same-lab or same-run replicate errors may be correlated and thereby inflate the measured replicate
agreement. Neither effect is quantified here, but neither erases the observed paired gap of 0.120.

The ceiling audit covers only 572 of 20,573 FT winner switches, or 2.8%, and is concentrated in the
large-change regime where the model is already strongest. It provides positive evidence for
large, well-measured FT crossings, not a measured ceiling for the small-change, wider-gap majority.
Part of the full-cohort 0.593 attenuation may be target measurement noise rather than model error.
Consequently, confidence cannot be treated only as a smooth example weight: the first target
population needs a frozen minimum-confidence eligibility rule, and the audit must verify that this
rule actually selects the larger-\(\lVert\Delta q\rVert_2\), better-measured regime rather than
assuming that endpoint margin and change-vector magnitude are interchangeable.

QTOF is inconclusive, not negative evidence: its 15 switching transitions come from only seven
molecules. Do not extrapolate the FT ceiling to QTOF. A QTOF contender-pair arm requires more
replicate coverage or an explicitly FT-only target population.

The existing adjacent winner-flip point estimates exceed the same-CE floors by more than 0.05 in
every FT adduct: `[M+Cl]-` 0.331 versus 0.234, `[M+H]+` 0.282 versus 0.148, `[M+Na]+` 0.293 versus
0.155, and `[M-H]-` 0.273 versus 0.184. These numbers are descriptive rather than a formal Gate 2
comparison: the same-CE estimator uses raw per-spectrum winners and instrument-relative tolerance,
whereas the existing cross-CE estimator uses consensus winners and a fixed 0.02-Da threshold. The
pending joint molecule bootstrap must first recompute both sides with the same winner definition,
tolerance, aggregation level, and eligible cohort. The current mismatch is directionally
conservative, but intervals alone would not repair it.

**Decision: FT is a go for the next stage of contender-pair supervision.** The cohort-matched
directional gap is positive and material, so the target is not merely fitting replicate noise. On
the paired statistic, `[M+H]+` and `[M-H]-` clear zero; `[M+Na]+` does not, while `[M+Cl]-` is below
the predeclared coverage threshold. Proceed with an explicitly high-confidence FT target population
and the structural/coverage audit; do not extrapolate the ceiling to all FT transitions. Gates 1,
3, and 4 on the matched unfiltered GINE-1 remain open, as does the estimator-aligned winner-flip
part of Gate 2. QTOF is excluded because its replicate coverage is insufficient.

## Why `ce_insert_location: mol,mlp` changed little

The follow-up direct-insertion experiment changed performance very little. This rules out the
simple explanation that collision energy cannot reach the fragment readout. Four mechanisms remain:

1. Spectra are optimized independently, so the loss does not explicitly compare the same molecule
   below and above a winner transition.
2. Spectrum cross-entropy spreads supervision over many peaks, diluting the top-winner error.
3. Flat fragment scores are mostly computed independently and contain limited explicit comparison
   against candidates on distant DAG branches.
4. Global softmax normalizes existing logits but cannot change their ordering or create the missing
   competition features: `argmax(softmax(s)) == argmax(s)`.

The next informative target is a reliable adjacent-CE global contender comparison, not another CE
insertion or uniform CE-rescaling path. Include observed winner switches, supervising `A > B` at
low CE and `B > A` at high CE, plus stable-winner pairs that preserve `A > B` at both endpoints as
anchors against a flip-happy solution. Exclude unsupported, near-tied, or replicate-unstable
transitions.

## Sibling comparison remains provisional

Existing full-validation sibling `q_max` diagnostics reported mean TV 0.377 and dominant-child
accuracy 52.2%, but they are not comparable with binary responsibility-weighted cut metrics.
Sibling groups are multiway and unweighted, use an older matcher with broader CMF attribution, and
have cardinality-dependent chance accuracy. Rebuild both target families with one matcher,
weighting rule, confidence stratum, and cardinality-aware null before ranking their headroom.

## Reproduction

Build the 500-molecule cut cache:

```bash
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python \
  scripts/eval/build_cut_pair_targets.py \
  --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags \
  --splits train val \
  --limit_mols 500 \
  --seed 17 \
  --n_jobs 8 \
  --out_fp /tmp/cut-pair-review-BJ5z6X/cache_500mol.pt \
  --report_fp /tmp/cut-pair-review-BJ5z6X/audit_500mol.json
```

Then run:

```bash
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python \
  scripts/debug/debug_cut_pair_large_stats.py
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python \
  scripts/debug/debug_cut_pair_oracle_controls.py
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python \
  scripts/debug/debug_cut_pair_overlap_dag_oracle.py
```

The base-peak CE supplement is saved as
`scripts/debug/base_peak_ce_supplement.py`. It consumes `per_spectrum.csv`, `ce_levels.csv`, and
`cross_ce_pairs.csv` from the larger base-peak analysis directory.

Test the static-fragment-weight hypothesis with:

```bash
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python \
  scripts/debug/debug_ce_static_fragment_weights.py
```

It writes context-, CE-level-, and adjacent-transition results plus molecule-cluster intervals to
`/tmp/ce-static-fragment-weight-9t63zils/`. The adjacent output includes delta dot products and L2
norms; `summary.json` reports switch-restricted delta cosine and the L2-optimal uniform gain.

Measure the model-free replicate floors with:

```bash
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python \
  scripts/debug/debug_ce_replicate_noise_floors.py
```

It writes same-CE pair rows, double-replicated adjacent-transition rows, the model comparison joined
to those exact transitions, the matched-versus-unmatched coverage audit, and the coverage-first
summary to `/tmp/ce-replicate-noise-floors-9t63zils/`.

All uncertainty intervals above resample molecules rather than individual spectra or overlapping
pairs. The cut oracle freezes `P(formula | node)`, jointly edits per-spectrum node logits for five
Adam steps at learning rate 0.05 with L2 coefficient `1e-3`, rebuilds the spectrum on retained
prediction support, and compares the result with the recomputed unedited spectrum.

## Interpretation and next decision

Do not start cut-loss training from the current weighting solely because the local TV is low or the
oracle edit is statistically positive. The absolute headroom recovery misses the formal gate, and
the loss emphasizes already-correct examples. First repeat the diagnostics on the matched GINE-1
control with an unfiltered export and a same-context replicate floor.

Global base-peak and CE-dependent intensity ordering are higher-priority. Test adjacent-CE
contender-pair supervision according to
[`global_fragment_intensity_ranking_experiment_plan.md`](../../plans/fragnnet/global_fragment_intensity_ranking_experiment_plan.md)
before building a set-level competition head; the head is justified only if the target contains
held-out signal the flat scorer cannot realize.
