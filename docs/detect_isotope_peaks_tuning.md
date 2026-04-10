# Isotope Peak Detection for Training

**Date:** 2026-04-09
**Script:** `scripts/debug/debug_prec_iso_leakage_pyopenms.py`

## Algorithm

### Physics model

When the MS1 isolation window co-selects a fraction $f$ of the M+k precursor
isotopologue alongside M+0, both fragment identically.  Every fragment ion $i$
gains a satellite at $+k \times 1.003355\,\text{Da}$ with intensity:

$$r_i^{(k)} = R_{\text{exp},i}^{(k)} \cdot f$$

where $f \in (0,1]$ is the **co-isolation fraction** — a spectrum-level
constant — and $R_{\text{exp},i}^{(k)}$ is the natural M+k/M+0 ratio for
fragment $i$'s formula, estimated by scaling the precursor's expected ratio by
the fragment's relative neutral mass:

$$R_{\text{exp},i}^{(k)} \approx R_{\text{exp,prec}}^{(k)} \cdot
  \frac{m_i^{\text{neutral}}}{m_{\text{prec}}^{\text{neutral}}},
\qquad
m_i^{\text{neutral}} = \mathrm{mz}_i - \Delta m_{\text{adduct}}$$

This correctly scales the detection window per fragment: a small fragment needs
a much lower absolute M+k/M+0 ratio than the intact precursor to qualify as a
co-isolation satellite.

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

When no above-precursor reference peaks are found, estimate $f$ from the
formula-normalised ratios of all in-spectrum (M, M+k) candidate pairs:

$$\tilde{f}_i^{(k)} = \frac{r_i^{(k)}}{R_{\text{exp},i}^{(k)}},
\qquad
\hat{f}^{(k)} = \text{median}_i\!\left(\tilde{f}_i^{(k)}\right)$$

Because genuine co-isolation makes $f$ a spectrum-level constant, all
normalised values should cluster tightly.  Accept if the coefficient of
variation is small:

$$\mathrm{CV}\!\left(\tilde{f}_i^{(k)}\right)
= \frac{\sigma\!\left(\tilde{f}_i^{(k)}\right)}{\hat{f}^{(k)}}
\;\leq\; \mathrm{CV}_{\max}$$

---

### Step 3 — Per-peak acceptance

For each candidate peak, find its M+0 partner at $\mathrm{mz} - k \times
1.003355\,\text{Da}$ (4 mDa tolerance for $k=1$, 8 mDa for $k \geq 2$).
Accept as an isotope leak if:

1. **Monotonicity** — $I(M{+}0)_i > I(M{+}k)_i$
2. **Ratio window** —

$$r_{\text{lo}} \cdot R_{\text{exp},i}^{(k)} \cdot \hat{f}^{(k)}
\;\leq\;
\frac{I(M{+}k)_i}{I(M{+}0)_i}
\;\leq\;
r_{\text{hi}} \cdot R_{\text{exp},i}^{(k)} \cdot \hat{f}^{(k)}$$

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
