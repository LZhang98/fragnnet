# Isotope Peak Detection for Training

**Date:** 2026-04-09
**Script:** `scripts/debug/debug_prec_iso_leakage_pyopenms.py`

## Algorithm

### Physics model

The training target should be monoisotopic.  Regular fragment isotope peaks and
co-isolation isotope peaks are both isotope-derived signal in this target space;
they can be added back later as a Fragnnet post-process instead of being learned
as ordinary fragment peaks.

When the MS1 isolation window co-selects a fraction $f$ of the M+k precursor
isotopologue alongside M+0, the selected populations both fragment.  A simple
first-order model says that every fragment ion $i$ gains a satellite at
$+k \times 1.003355\,\text{Da}$ with intensity:

$$r_i^{(k)} = R_{\text{exp},i}^{(k)} \cdot f$$

where $f \in (0,1]$ is the **co-isolation fraction** — a spectrum-level
constant — and $R_{\text{exp},i}^{(k)}$ is the natural M+k/M+0 ratio for
fragment $i$'s formula, estimated by scaling the precursor's expected ratio by
the fragment's relative neutral mass:

$$R_{\text{exp},i}^{(k)} \approx R_{\text{exp,prec}}^{(k)} \cdot
  \frac{m_i^{\text{neutral}}}{m_{\text{prec}}^{\text{neutral}}},
\qquad
m_i^{\text{neutral}} = \mathrm{mz}_i - \Delta m_{\text{adduct}}$$

This is intended to scale the detection window per fragment: a small fragment
needs a much lower absolute M+k/M+0 ratio than the intact precursor to qualify
as a co-isolation satellite.

That equation is an approximation.  For $k=1$, an M+1 precursor population can
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

This matters most when $f$ is large, including the regular-isotope-like case
where $f \approx 1$.  The current aggressive fallback should therefore be
treated as an approximate heuristic, not as a physically exact estimate.

When DAG fragment formulas are available, use them to compute
$R_{\text{frag},i}^{(k)}$ directly.  The mass-scaled precursor approximation
should be only a fallback when the candidate fragment formula is unknown.

---

### Step 1 — Estimate $f$ from above-precursor residual peaks

Look for M+k peaks **above** `prec_mz` in the MS2 scan (precursor isotope
residuals that leaked through).  Compute:

$$\hat{f}^{(k)} = \frac{R_k^{\text{obs,prec}}}{R_{\text{exp,prec}}^{(k)}},
\qquad
R_k^{\text{obs,prec}} = \frac{I(M{+}k)_\text{prec}}{I(M{+}0)_\text{prec}}$$

Accept only if the co-isolation fraction is physically plausible:

$$f_{\min} \;\leq\; \hat{f}^{(k)} \;\leq\; f_{\max}$$

When both $k=1$ and $k=2$ are found, their estimates must be mutually
consistent (they come from the same isolation event, so $\hat{f}^{(1)} \approx
\hat{f}^{(2)}$):

$$r_{\text{lo}} \;\leq\; \frac{\hat{f}^{(2)}}{\hat{f}^{(1)}} \;\leq\; r_{\text{hi}}$$

---

### Step 2 — Aggressive fallback: $f$ from cross-pair consistency

When no above-precursor reference peaks are found, the current fallback estimates
$f$ from the formula-normalised ratios of all in-spectrum (M, M+k) candidate
pairs:

$$\tilde{f}_i^{(k)} = \frac{r_i^{(k)}}{R_{\text{exp},i}^{(k)}},
\qquad
\hat{f}^{(k)} = \text{median}_i\!\left(\tilde{f}_i^{(k)}\right)$$

Because genuine co-isolation makes $f$ a spectrum-level constant, all
normalised values should cluster tightly.  Accept if the coefficient of
variation is small:

$$\mathrm{CV}\!\left(\tilde{f}_i^{(k)}\right)
= \frac{\sigma\!\left(\tilde{f}_i^{(k)}\right)}{\hat{f}^{(k)}}
\;\leq\; \mathrm{CV}_{\max}$$

This fallback is the highest-risk path because it tries to infer $f$ from
fragment peaks alone.  For monoisotopic-target cleanup, prefer this priority:

1. Use local precursor-derived $f$ when visible.
2. Use group-level precursor-derived $f$ from the same `mol_id + prec_type +
   inst_type` group when local precursor residuals are missing.
3. Use DAG formula support to downweight ambiguous isotope-derived peaks.
4. Use aggressive cross-pair fallback only as weak evidence, or disable it for
   hard removal.

---

### Step 2b — Group-level $f$ imputation

For NIST spectra, `group_id` is built from `mol_id + prec_type + inst_type`.
Spectra in the same group are more likely to share the same instrument and
isolation setting, so a reliable precursor-derived $f$ from one spectrum can be
used as a fallback for another spectrum in the same group when that spectrum has
no visible precursor M+k residual peaks.

Workflow:

1. For each unmerged spectrum, call
   `estimate_coisolation_fraction_from_precursor(...)`.  This uses only the
   precursor M+0/M+k residual method and returns `{k: f}`.
2. Aggregate $f$ values within each `group_id` and isotope order, preferably by
   median and with a count/dispersion sanity check.
3. When cleaning another spectrum in the same group, pass the aggregate as
   `detect_isotope_peaks_for_training(..., coiso_fraction_by_k=group_f)`.
   The local spectrum's own precursor-derived $f$ still takes priority; the
   group-level value is used only for missing isotope orders.

This is specifically for co-isolation cleanup.  It should not be interpreted as
evidence that all NIST `ann_isotopes > 0` peaks are co-isolation artifacts.  For
monoisotopic targets, the important label is not whether a peak is regular
isotope or co-isolation; the important label is whether it is isotope-derived
signal that should be removed or downweighted during target construction.

---

### Step 2c — DAG-aware isotope scoring

If a DAG fragment candidate explains the M+0 peak, compute isotope expectations
from the fragment formula instead of estimating fragment composition from m/z:

$$
R_{\text{frag},i}^{(k)} =
\mathrm{isotope\_ratio}\!\left(\text{DAG formula}_i, k\right)
$$

Then score each candidate isotope-derived peak using:

1. mass error to $M+k$,
2. intensity-ratio agreement with $R_{\text{frag},i}^{(k)}$,
3. local precursor-derived or group-level $f$ support,
4. whether the shifted peak is also explainable as a separate monoisotopic DAG
   fragment.

Recommended action for monoisotopic targets:

| Evidence | Action |
|---|---|
| Strong isotope explanation, weak monoisotopic DAG explanation | remove |
| Strong isotope explanation and plausible monoisotopic DAG explanation | downweight |
| Weak isotope explanation, strong monoisotopic DAG explanation | keep |

This avoids forcing a hard regular-isotope vs co-isolation label when the
observable spectrum cannot support that distinction.

---

### Step 3 — Per-peak acceptance

For each candidate peak, find its M+0 partner at $\mathrm{mz} - k \times
1.003355\,\text{Da}$ (4 mDa tolerance for $k=1$, 8 mDa for $k \geq 2$).
Accept as an isotope leak if:

1. **Monotonicity** — $I(M{+}0)_i > I(M{+}k)_i$
2. **Ratio window** —

For the current first-order model, the ratio window is:

$$r_{\text{lo}} \cdot R_{\text{exp},i}^{(k)} \cdot \hat{f}^{(k)}
\;\leq\;
\frac{I(M{+}k)_i}{I(M{+}0)_i}
\;\leq\;
r_{\text{hi}} \cdot R_{\text{exp},i}^{(k)} \cdot \hat{f}^{(k)}$$

For $k=1$, the corrected denominator-aware expectation should be preferred when
$R_{\text{prec}}^{(1)}$ and a DAG fragment formula are available:

$$
R_{\text{expected},i}^{(1)}
=
\frac{\hat{f}^{(1)} \cdot R_{\text{frag},i}^{(1)}}
{1 + \hat{f}^{(1)}
\cdot \left(R_{\text{prec}}^{(1)} - R_{\text{frag},i}^{(1)}\right)}
$$

3. **Envelope coherence** — for $k=2$, the M+1 satellite must already be confirmed

---

### Default parameters

| Symbol | Parameter | Default |
|---|---|---|
| $f_{\min}$ | `min_coiso_fraction` | 0.05 |
| $f_{\max}$ | `max_leak_factor` | 1.5 |
| $r_{\text{lo}}$ | `ratio_lo` | 0.05 |
| $r_{\text{hi}}$ | `ratio_hi` | 10.0 |
| $\mathrm{CV}_{\max}$ | `aggressive_cv` | 0.6 |

The 4 mDa tolerance is **fixed absolute** (not ppm) so that H-transfer peaks
(+1H sits 4.47 mDa above C13_SPACING) are always excluded regardless of m/z.

---

## Evaluation vs NIST ann_isotopes (FT, formula mode, min_coiso sweep)

We evaluated the `detect_isotope_peaks_for_training` algorithm over a sweep of `min_coiso_fraction` thresholds:

```
   min_coiso    fired       TP       FP        FN     prec      rec       F1
        0.05  162,579  454,814  195,343   165,719    0.700    0.733    0.716
        0.10  157,656  444,042  194,917   176,491    0.695    0.716    0.705
        0.20  148,224  415,187  193,610   205,346    0.682    0.669    0.675
```

Lowering the minimum co-isolation fraction threshold to `0.05` is a clear win, recovering ~40,000 True Positives while only adding ~1,700 False Positives, driving the F1 score up from `0.675` to `0.716`.

## Bottlenecks and False Negatives

1. **Ratio Window restricts detection:** Of the peaks missed when the detector fired, 19.1% were rejected because the observed ratio fell outside the `[0.2, 5.0]` expected window. 60% of these misses were *above* the `5.0x` window. Widening the ratio window to `[0.05, 10.0]` recovers 69.0% of these specific false negatives.
2. **Detector Not Firing:** For 65% of the FN peaks, the detector never turned on. In 94.3% of these spectra, there was actually a candidate pair, but they were likely blocked by the aggressive fallback constraints when a pristine precursor M+k was missing. The `aggressive_cv` constraint and the `aggressive_single_pair_min_f` limit were too strict.

## False Positives Concentration

False positives are heavily concentrated:
- 12.3% of all False Positives come from just 800 spectra.
- The worst offenders are generating 80–120 FPs per spectrum. This highly concentrated failure mode usually indicates polymers, oligomers, or repetitive structural losses yielding peaks spaced apart by ~1 Da, which the algorithm misinterprets as carbon isotopes.

## Actions Taken

Updated the defaults in `detect_isotope_peaks_for_training` in `src/fragnnet/utils/spec_utils.py` to:
- `min_coiso_fraction = 0.05`
- `ratio_lo = 0.05`
- `ratio_hi = 10.0`
- `aggressive_cv = 0.6`
- `aggressive_single_pair_min_f = 0.05`
