# Global fragment intensity ranking experiment plan — log

> **Status:** REFERENCE · **Updated:** 2026-10-06
> Verbatim measured-results text moved from [the plan](../plans/fragnnet/global_fragment_intensity_ranking_experiment_plan.md) on 2026-10-06; the plan holds the current summary.

The plan contained no dated chronological log; these are its two measured-results blocks (provisional NodeMLP validation read, 2026-08-11/12, run `9t63zils` export; Step 0 replicate floors). Moved unchanged.

## Evidence that motivates the experiment

The provisional NodeMLP validation analysis found:

- base-peak accuracy 57.4% over 96,153 spectra;
- 84.1% of base-peak misses retained the observed mass in exported prediction support;
- supported misses had mean DAG-oracle cosine 0.979 but model cosine 0.578;
- within 7,519 fixed FT contexts, base-peak accuracy fell from 92.4% at the lowest available CE to
  42.3% at the highest;
- conditional on the observed mass remaining in support, FT accuracy fell from 92.8% to 50.8%;
- on adjacent FT CE transitions where the observed winner changed, the model changed its winner
  only 38.5% of the time and got both endpoints right only 13.1%;
- predictions captured 85.8% of observed FT variation around the fixed-context CE mean and were
  closer to the CE-specific target than the leave-one-CE-out average in 79.8% of cases, rejecting a
  pure static-average explanation;
- despite that continuous response, predicted FT variation at winner-changing adjacent steps was
  only 0.205 TV versus 0.298 observed, and the top node changed in only 35.9% of such transitions;
- the metric-consistent L2-optimal uniform gain at winner-changing steps was 1.010
  `[0.996, 1.025]` for FT and 0.847 `[0.760, 0.936]` for QTOF, so uniform amplification is not the
  missing fix; switch-only delta cosine was only 0.593 for FT and 0.478 for QTOF;
- the observed winner was stable in 71.9% of adjacent FT transitions, motivating stable-winner
  contender pairs as anchors rather than training only on reversals;
- adding CE directly at `mol,mlp` changed performance very little.

These are provisional because the inference export applied `min_peak_int=1.0`. Repeat every gate
below on an unfiltered matched NodeMLP export.


## Step 0 — model-free replicate noise floors

(Measured results; the method stays in the plan.)

The provisional validation cohort gives:

| Instrument | Same-CE pairs / molecules | Winner-flip floor | Double-replicated switches / molecules | Replicate cosine | Matched model cosine | Paired difference | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FT | 4,519 / 1,202 | 0.171 `[0.156, 0.186]` | 572 / 386 | 0.885 `[0.856, 0.913]` | 0.765 `[0.737, 0.793]` | 0.120 `[0.091, 0.149]` | continue |
| QTOF | 160 / 10 | 0.200 `[0.103, 0.319]` | 15 / 7 | 0.035 `[-0.193, 0.302]` | not interpreted | not interpreted | inconclusive coverage |

Thus the kill criterion does not fire for FT on a valid cohort-matched comparison. For the first
training arm, however, freeze the eligible population to FT `[M+H]+` and `[M-H]-`: their paired
directional gaps clear zero. FT `[M+Na]+` remains excluded because its interval crosses zero, and
FT `[M+Cl]-` remains excluded because it is below the coverage threshold. Admit another adduct only
if the matched NodeMLP audit passes the same predeclared gate; “FT-only” by itself is not a complete
population definition. The full-cohort
model value of 0.593 is not used in this decision because the double-replicated subset is easier.
The approximate independent-error correction gives \(\sqrt{0.885}=0.941\) for single-observation
reliability, but same-lab replicate correlation can bias 0.885 upward; treat both effects as
unquantified sensitivity considerations, not a corrected endpoint. QTOF cannot be admitted to the
target population on this evidence; obtain more replication before admitting it. The first
contender-pair experiment is specifically FT `[M+H]+` and `[M-H]-`, not all FT spectra.

The FT reliability estimate covers only 572 of 20,573 switching transitions (2.8%). It is not a
representative ceiling for the full target population: the matched transitions have median
\(\lVert\Delta q\rVert_2=0.396\) and CE gap 5, versus 0.235 and CE gap 10 for unmatched
transitions, and 57.0% of matched transitions lie in the largest-change quartile. The model reaches
0.765 on the matched cohort and 0.588 on the unmatched cohort. A shared-molecule restriction does
not remove the gap, while replicate averaging can denoise the matched target. Treat the Step 0 go
as evidence for high-confidence, well-measured FT crossings only; do not extrapolate it to weak
transitions without a measured ceiling.

The paired switch gap nevertheless persists after stratifying by full-population
\(\lVert\Delta q\rVert_2\): Q1 0.177 `[0.068, 0.279]`, Q2 0.106
`[-0.031, 0.234]`, Q3 0.109 `[0.061, 0.155]`, and Q4 0.113 `[0.084, 0.143]`.
Thus the FT switch go is not solely a large-change artifact; Q2 is underpowered rather than showing
a reversed point estimate. Stable anchors are different: their Q1 gap is 0.002
`[-0.097, 0.098]`, with measurable headroom appearing only in Q3 and Q4. Confidence eligibility
must therefore apply when stable anchors are first introduced, and stable and switch outcomes must
be evaluated separately.
