# Isotope Peak Detection for Training

> **Status:** REFERENCE &nbsp;·&nbsp; **Updated:** 2026-10-12
>
> Canonical note for isotope-derived peak cleanup. DAG-based alternatives were tested 2026-04-08 and did not beat this.

**Date:** 2026-04-20
**Scripts:**

- `scripts/debug/isotopes/debug_prec_iso_leakage_pyopenms.py`
- `scripts/debug/isotopes/debug_dag_iso_peak_detection.py`

This is the canonical note for isotope-derived peak cleanup in monoisotopic
training targets.  It merges the original tuning note with the DAG isotope peak
detection analysis.

## Goal

The training target should be monoisotopic.  Regular fragment isotope peaks and
co-isolation isotope peaks are both isotope-derived signal in this target space;
they can be added back later as a Fragnnet post-process instead of being learned
as ordinary fragment peaks.

Co-isolation isotope contamination occurs when the MS1 isolation window
co-selects the M+1 or M+2 isotopologue of the precursor alongside M+0.  Both
selected populations fragment, so every fragment ion can gain a satellite at
`+k * 1.003355 Da` with intensity proportional to a spectrum-level
co-isolation fraction `f`.

## Physics Model

For fragment ion `i`, a first-order model is:

$$
r_i^{(k)} = R_{\text{exp},i}^{(k)} \cdot f
$$

where `f` is the co-isolation fraction and
$R_{\text{exp},i}^{(k)}$ is the natural M+k/M+0 ratio for the fragment formula.
When the fragment formula is unavailable, estimate it by scaling the precursor
expected ratio by relative neutral mass:

$$
R_{\text{exp},i}^{(k)} \approx R_{\text{exp,prec}}^{(k)} \cdot
  \frac{m_i^{\text{neutral}}}{m_{\text{prec}}^{\text{neutral}}},
\qquad
m_i^{\text{neutral}} = \mathrm{mz}_i - \Delta m_{\text{adduct}}
$$

This scales the detection window per fragment.  A small fragment needs a much
lower absolute M+k/M+0 ratio than the intact precursor to qualify as a
co-isolation satellite.

That equation is an approximation.  For `k=1`, an M+1 precursor population can
also add intensity to the observed fragment M+0 peak when the heavy isotope is
carried by the neutral loss rather than the charged fragment.  A better
single-isotope approximation is:

$$
\frac{I_i(M{+}1)}{I_i(M{+}0)}
\approx
\frac{f \cdot R_{\text{frag},i}^{(1)}}
{1 + f \cdot \left(R_{\text{prec}}^{(1)} - R_{\text{frag},i}^{(1)}\right)}
$$

Solving this for a single pair gives:

$$
f_i
\approx
\frac{r_i}
{R_{\text{frag},i}^{(1)} -
 r_i \cdot \left(R_{\text{prec}}^{(1)} - R_{\text{frag},i}^{(1)}\right)}
,\qquad
r_i = \frac{I_i(M{+}1)}{I_i(M{+}0)}
$$

This matters most when `f` is large, including the regular-isotope-like case
where `f ~= 1`.  The aggressive fallback should therefore be treated as an
approximate heuristic, not as a physically exact estimate.

When DAG fragment formulas are available, use them to compute
$R_{\text{frag},i}^{(k)}$ directly.  The mass-scaled precursor approximation
should be only a fallback when the candidate fragment formula is unknown.

## Algorithm

### Step 1 - Estimate `f` from above-precursor residual peaks

Look for precursor M+k residual peaks above `prec_mz` in the MS2 scan.  These
peaks are the cleanest local evidence that M+k precursor isotopologues leaked
through the isolation window.

For each detected precursor residual:

$$
\hat{f}^{(k)} = \frac{R_k^{\text{obs,prec}}}{R_{\text{exp,prec}}^{(k)}},
\qquad
R_k^{\text{obs,prec}} = \frac{I(M{+}k)_\text{prec}}{I(M{+}0)_\text{prec}}
$$

Accept only if the co-isolation fraction is physically plausible:

$$
f_{\min} \leq \hat{f}^{(k)} \leq f_{\max}
$$

When both M+1 and M+2 residuals are present, also compute the direct
above-precursor envelope ratio:

$$
R_{2/1}^{\text{obs,prec}}
= \frac{I(M{+}2)_\text{prec}}{I(M{+}1)_\text{prec}}
$$

Compare it with the expected precursor isotope envelope:

$$
R_{2/1}^{\text{exp,prec}}
= \frac{R_{\text{exp,prec}}^{(2)}}{R_{\text{exp,prec}}^{(1)}}
$$

If M+1 and M+2 came from the same isolation event and have the same
co-isolation fraction, then:

$$
\frac{R_{2/1}^{\text{obs,prec}}}{R_{2/1}^{\text{exp,prec}}}
\approx
\frac{\hat{f}^{(2)}}{\hat{f}^{(1)}}
\approx 1
$$

Use this M2/M1 ratio as an envelope-coherence check for peaks above the
precursor ion.  It catches cases where an unrelated peak near precursor M+2
would otherwise produce a plausible absolute M+2/M+0 ratio.  It can also serve
as weak support when M+0 is noisy, saturated, or missing, but it should not
produce an absolute `f` estimate by itself because the M2/M1 ratio only measures
relative consistency.

The existing consistency gate can be expressed either as
`f_hat[2] / f_hat[1]` or equivalently as the normalized M2/M1 envelope ratio:

$$
r_{\text{lo}} \leq
\frac{\hat{f}^{(2)}}{\hat{f}^{(1)}} \leq r_{\text{hi}}
$$

### Step 2 - Group-level `f` imputation

For NIST spectra, `group_id` is built from `mol_id + prec_type + inst_type`.
Spectra in the same group are more likely to share the same instrument and
isolation setting, so a reliable precursor-derived `f` from one spectrum can be
used as a fallback for another spectrum in the same group when that spectrum has
no visible precursor M+k residual peaks.

Workflow:

1. For each unmerged spectrum, call
   `estimate_coisolation_fraction_from_precursor(...)`.  This uses only the
   precursor M+0/M+k residual method and returns `{k: f}`.
2. Aggregate `f` values within each `group_id` and isotope order, preferably by
   median and with a count/dispersion sanity check.
3. When cleaning another spectrum in the same group, pass the aggregate as
   `detect_isotope_peaks_for_training(..., coiso_fraction_by_k=group_f)`.
   The local spectrum's own precursor-derived `f` still takes priority; the
   group-level value is used only for missing isotope orders.

This is specifically for co-isolation cleanup.  It should not be interpreted as
evidence that all NIST `ann_isotopes > 0` peaks are co-isolation artifacts.  For
monoisotopic targets, the important label is not whether a peak is regular
isotope or co-isolation; the important label is whether it is isotope-derived
signal that should be removed or downweighted during target construction.

### Step 3 - Aggressive fallback from cross-pair consistency

When no above-precursor reference peaks are found, the current fallback estimates
`f` from the formula-normalized ratios of all in-spectrum `(M, M+k)` candidate
pairs:

$$
\tilde{f}_i^{(k)} = \frac{r_i^{(k)}}{R_{\text{exp},i}^{(k)}},
\qquad
\hat{f}^{(k)} = \text{median}_i\!\left(\tilde{f}_i^{(k)}\right)
$$

Because genuine co-isolation makes `f` a spectrum-level constant, all normalized
values should cluster tightly.  Accept if the coefficient of variation is small:

$$
\mathrm{CV}\!\left(\tilde{f}_i^{(k)}\right)
= \frac{\sigma\!\left(\tilde{f}_i^{(k)}\right)}{\hat{f}^{(k)}}
\leq \mathrm{CV}_{\max}
$$

This fallback is the highest-risk path because it tries to infer `f` from
fragment peaks alone.  For monoisotopic-target cleanup, prefer this priority:

1. Use local precursor-derived `f` when visible.
2. Use group-level precursor-derived `f` from the same `mol_id + prec_type + inst_type` group when local precursor residuals are missing.
3. Use DAG formula support to downweight ambiguous isotope-derived peaks.
4. Use aggressive cross-pair fallback only as weak evidence, or disable it for
   hard removal.

### Step 4 - DAG-aware isotope scoring

If a DAG fragment candidate explains the M+0 peak, compute isotope expectations
from the fragment formula instead of estimating fragment composition from m/z:

$$
R_{\text{frag},i}^{(k)} =
\mathrm{isotope\_ratio}\!\left(\text{DAG formula}_i, k\right)
$$

Then score each candidate isotope-derived peak using:

1. mass error to `M+k`,
2. intensity-ratio agreement with $R_{\text{frag},i}^{(k)}$,
3. local precursor-derived or group-level `f` support,
4. the above-precursor M2/M1 envelope check when both residuals are visible,
5. whether the shifted peak is also explainable as a separate monoisotopic DAG
   fragment.

Recommended action for monoisotopic targets:

| Evidence                                                              | Action     |
| --------------------------------------------------------------------- | ---------- |
| Strong isotope explanation, weak monoisotopic DAG explanation         | remove     |
| Strong isotope explanation and plausible monoisotopic DAG explanation | downweight |
| Weak isotope explanation, strong monoisotopic DAG explanation         | keep       |

This avoids forcing a hard regular-isotope vs co-isolation label when the
observable spectrum cannot support that distinction.

### Step 5 - Per-peak acceptance

For each candidate peak, find its M+0 partner at
`mz - k * 1.003355 Da`.  Use 4 mDa tolerance for `k=1` and 8 mDa for `k >= 2`.
Accept as an isotope leak if:

1. **Monotonicity:** `I(M+0)_i > I(M+k)_i`
2. **Ratio window:** observed satellite intensity agrees with expected ratio
3. **Envelope coherence:** for `k=2`, the M+1 satellite should already be
   confirmed when it is visible

For the current first-order model, the ratio window is:

$$
r_{\text{lo}} \cdot R_{\text{exp},i}^{(k)} \cdot \hat{f}^{(k)}
\leq
\frac{I(M{+}k)_i}{I(M{+}0)_i}
\leq
r_{\text{hi}} \cdot R_{\text{exp},i}^{(k)} \cdot \hat{f}^{(k)}
$$

For `k=1`, the corrected denominator-aware expectation should be preferred when
$R_{\text{prec}}^{(1)}$ and a DAG fragment formula are available:

$$
R_{\text{expected},i}^{(1)}
=
\frac{\hat{f}^{(1)} \cdot R_{\text{frag},i}^{(1)}}
{1 + \hat{f}^{(1)}
\cdot \left(R_{\text{prec}}^{(1)} - R_{\text{frag},i}^{(1)}\right)}
$$

The 4 mDa tolerance is fixed absolute, not ppm, so H-transfer peaks
`(+1H sits 4.47 mDa above C13_SPACING)` are excluded regardless of m/z.

## DAG Experiment Summary

The DAG experiment tested whether the fragmentation DAG can improve
`detect_isotope_peaks_for_training` on NIST20 FT spectra.

### Data

- Dataset: NIST20 FT spectra only, 1,029,998 total; 120,393 with at least one
  NIST-annotated isotope peak
- Ground truth: `data/proc/nist20/ann_df.pkl`, using NIST `ann_isotopes`
- DAG: `data/frag/nist20_ma_mi_d3_h4_isoFalse_cmf/dags`, depth-3,
  H-transfer +/-4, CMF variant
- Evaluation: 500 randomly sampled FT spectra with at least one annotated
  isotope peak, 2440 GT isotope peaks total

### DAG structure

| Field                | Shape               | Description                                                                     |
| -------------------- | ------------------- | ------------------------------------------------------------------------------- |
| `formula_peak_mzs` | `(n_formulas, 5)` | M+0 through M+4 m/z per formula variant; M+1-M+4 columns are zero in this build |
| `idx_to_formula`   | `dict[int, str]`  | Formula index to formula string                                                 |
| `dag.x`            | `(n_nodes, 45)`   | Node features;`NODE_FEAT_TO_IDX["h_formulae_idx"]` points to formula index    |

Key observations:

- `formula_peak_mzs[:, 0]` gives monoisotopic M+0 m/z for each formula variant.
- M+k positions must be computed as `M+0 + k * C13_SPACING`.
- The DAG nodes typically use only 2-5 unique formula indices, even though
  `idx_to_formula` has many more entries.

### Methods tested

| Method          | Description                                                                                                    |
| --------------- | -------------------------------------------------------------------------------------------------------------- |
| `current`     | Existing detector with`formula` and `prec_type`; uses `frag_mz / prec_neutral_mass` as composition proxy |
| `dag_filter`  | Keep only flagged peaks whose parent M+0 is found in the DAG or is the precursor                               |
| `dag_protect` | Unflag any peak whose m/z directly matches a DAG M+0 node                                                      |
| `dag_formula` | Re-check flagged pairs using exact DAG fragment formula and precursor-derived`f`                             |

### Historical results

| Method      | TP   | FP  | FN   | Precision | Recall | F1    |
| ----------- | ---- | --- | ---- | --------- | ------ | ----- |
| Current     | 1895 | 237 | 545  | 0.889     | 0.777  | 0.829 |
| DAG-filter  | 1735 | 191 | 705  | 0.901     | 0.711  | 0.795 |
| DAG-protect | 369  | 39  | 2071 | 0.904     | 0.151  | 0.259 |
| DAG-formula | 1892 | 236 | 548  | 0.889     | 0.775  | 0.828 |

### Interpretation

The DAG did not improve isotope peak detection as a hard filter, protector, or
ratio-refinement step under this setup.

Important failure modes:

1. **DAG coverage gap:** around 10% of real annotated isotope peaks have a
   parent M+0 absent from the depth-3 DAG, and around 7% have the precursor
   itself as parent.
2. **DAG density vs. FT resolution:** many isotope satellite peaks happen to
   land near a different DAG M+0 node, so direct DAG protection incorrectly
   keeps genuine isotope artifacts.
3. **Formula proxy error is small relative to gate width:** M+1 proxy-vs-exact
   DAG formula error was roughly 10% median relative error and 13.3% mean
   relative error, while the ratio gate is much wider.

Conclusion: keep the baseline detector as the main mechanism.  Use DAG evidence
as supporting evidence for ambiguous decisions, not as a mandatory hard filter.

### Rerun with NIST20 annotations and `smpFalse` DAGs

Rerun date: 2026-04-20

Command:

```bash
conda run --no-capture-output -n FRAGNNET-GPU \
  python scripts/debug/isotopes/debug_dag_iso_peak_detection.py \
  --proc-dp data/proc/nist20 \
  --frag-dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags \
  --n-samples 500 \
  --n-plots 0 \
  --progress-interval 100
```

Data loaded:

- FT spectra: 1,029,998
- FT spectra with at least one NIST annotated isotope peak: 120,393
- Spectra analysed: 500
- Total GT isotope peaks: 2440
- Spectra skipped because DAG was unavailable: 1
- Group-level precursor `f`: 38,489 spectra with local precursor `f`,
  10,083 groups, 10,618 group/order estimates

Results:

| Method          |   TP |  FP |   FN | Precision | Recall |    F1 |
| --------------- | ---: | --: | ---: | --------: | -----: | ----: |
| Current         | 2008 | 482 |  432 |     0.806 |  0.823 | 0.815 |
| Group-f         | 2278 | 552 |  162 |     0.805 |  0.934 | 0.865 |
| DAG-native      | 1948 | 426 |  492 |     0.821 |  0.798 | 0.809 |
| DAG+Group-f     | 2293 | 509 |  147 |     0.818 |  0.940 | 0.875 |
| DAG-native-prot |  366 |  63 | 2074 |     0.853 |  0.150 | 0.255 |
| DAG-filter      | 1849 | 411 |  591 |     0.818 |  0.758 | 0.787 |
| DAG-protect     |  372 |  71 | 2068 |     0.840 |  0.152 | 0.258 |
| DAG-formula     | 2001 | 476 |  439 |     0.808 |  0.820 | 0.814 |
| DAG-form+Group  | 2271 | 545 |  169 |     0.806 |  0.931 | 0.864 |

Interpretation:

1. **Best method in this run:** `DAG+Group-f`, with F1 `0.875`.  Relative to
   `Current`, it recovered 285 more true positives and removed 27 fewer false
   positives overall, improving recall from `0.823` to `0.940`.
2. **Group-level precursor `f` is the main improvement:** `Group-f` improved F1
   from `0.815` to `0.865` by recovering 270 true positives, at the cost of 70
   extra false positives.
3. **DAG-native plus group `f` is better than group `f` alone:** `DAG+Group-f`
   gained 15 more true positives and removed 43 false positives relative to
   `Group-f`, increasing F1 from `0.865` to `0.875`.
4. **DAG-native alone is not enough:** `DAG-native` slightly improved precision
   but reduced recall enough to lower F1 (`0.809` vs `0.815`).
5. **DAG hard filtering/protection remains unsafe:** `DAG-native-prot`,
   `DAG-protect`, and `DAG-filter` all remove many true isotope peaks.  The
   protection variants are especially poor because isotope satellites often
   coincide with dense DAG monoisotopic nodes.
6. **DAG-formula post-check has little effect:** `DAG-formula` changed only a
   few peaks relative to `Current` and did not improve F1.

Recommendation from this rerun: use local precursor `f` first, group-level
precursor `f` second, and DAG formulas as soft supporting evidence.  The best
observed hard-removal variant is `DAG+Group-f`; avoid direct DAG protection and
mandatory DAG-parent filtering.

### Group-f quality sweep

Follow-up sweeps on the same 500-spectrum sample tested whether stricter
group-level `f` aggregation improves the `DAG+Group-f` result:

| `group_min_count` | `group_max_cv` | Group/order estimates | DAG+Group-f TP | DAG+Group-f FP | DAG+Group-f FN | Precision | Recall |    F1 |
| ------------------: | ---------------: | --------------------: | -------------: | -------------: | -------------: | --------: | -----: | ----: |
|                   1 |              1.0 |                10,618 |           2293 |            509 |            147 |     0.818 |  0.940 | 0.875 |
|                   1 |              0.5 |                10,168 |           2278 |            502 |            162 |     0.819 |  0.934 | 0.873 |
|                   2 |              1.0 |                 9,230 |           2279 |            504 |            161 |     0.819 |  0.934 | 0.873 |
|                   2 |              0.5 |                 8,780 |           2264 |            497 |            176 |     0.820 |  0.928 | 0.871 |

Stricter group filters reduce false positives slightly, but they remove more
true positives than they save.  Keep the permissive defaults
`group_min_count=1`, `group_max_cv=1.0` for now.

### Implementation decision

Dataset cleanup now implements the best observed path:

1. `SpecMolFragDataset` automatically provides its `frag_dp` to isotope cleanup
   when `remove_isotope_peaks=True` and `remove_isotope_use_dag_formula=True`.
2. `BaseDataset` caches per-molecule DAG formula arrays and passes
   `dag_formula_mzs` / `dag_formula_strs` into
   `detect_isotope_peaks_for_training_cleanup(...)`.
3. Group-level `f` remains enabled by default with permissive aggregation.
4. `protect_dag_mono` remains disabled by default because direct DAG protection
   was repeatedly shown to collapse recall.

## Evaluation vs NIST `ann_isotopes`

> Since 2026-09-20 the fragment composition can also come from subformula decomposition instead
> of a DAG, which matches the DAG source to 99.2% of peak decisions and needs no `frag_dp`:
> [subformula_isotope_detection_plan.md](../plans/subformula_isotope_detection_plan.md).

FT formula-mode sweep over `min_coiso_fraction`:

```text
   min_coiso    fired       TP       FP        FN     prec      rec       F1
        0.05  162,579  454,814  195,343   165,719    0.700    0.733    0.716
        0.10  157,656  444,042  194,917   176,491    0.695    0.716    0.705
        0.20  148,224  415,187  193,610   205,346    0.682    0.669    0.675
```

Lowering the minimum co-isolation fraction threshold to `0.05` recovered about
40,000 true positives while adding about 1,700 false positives, improving F1
from `0.675` to `0.716`.

## Bottlenecks

1. **Ratio window restricts detection:** Of peaks missed when the detector
   fired, 19.1% were rejected because the observed ratio fell outside the
   `[0.2, 5.0]` expected window.  60% of these misses were above the `5.0x`
   window.  Widening the ratio window to `[0.05, 10.0]` recovers 69.0% of these
   specific false negatives.
2. **Detector not firing:** For 65% of false negatives, the detector never
   turned on.  In 94.3% of these spectra, there was a candidate pair, but it was
   likely blocked by aggressive fallback constraints when a clean precursor M+k
   residual was missing.
3. **False positives are concentrated:** 12.3% of all false positives came from
   800 spectra.  The worst offenders generated 80-120 false positives per
   spectrum, usually suggesting polymers, oligomers, or repetitive structural
   losses spaced near 1 Da.

## Recommended Defaults

| Parameter                        | Default  |
| -------------------------------- | -------- |
| `min_coiso_fraction`           | `0.05` |
| `max_leak_factor`              | `1.5`  |
| `ratio_lo`                     | `0.05` |
| `ratio_hi`                     | `10.0` |
| `precursor_envelope_lo`        | `0.2`  |
| `precursor_envelope_hi`        | `5.0`  |
| `aggressive_cv`                | `0.6`  |
| `aggressive_single_pair_min_f` | `0.05` |

## Implementation Notes

- `_expected_mk_ratio` and `_parse_adduct_counts` are in
  `src/fragnnet/dataset/isotope_cleanup.py` (was `utils/isotope_utils.py`; the `spec_utils`
  aliases were removed on 2026-10-12).
- DAG M+0 lookup uses `formula_peak_mzs[:, 0]`; `idx_to_formula` keys align
  with the row indices of this array.
- The `DagInfo` helper class in `scripts/debug/isotopes/debug_dag_iso_peak_detection.py`
  wraps the DAG lookups used in the experiment.
- The precursor residual estimator now has an explicit above-precursor `M2/M1`
  envelope check before using M+2 evidence for hard removal.
