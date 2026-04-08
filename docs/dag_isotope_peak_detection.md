# DAG-Based Isotope Peak Detection — Analysis

**Branch:** `do_we_have_iso_peak`
**Date:** 2026-04-08
**Script:** `scripts/debug/debug_dag_iso_peak_detection.py`

---

## Background

Co-isolation isotope contamination occurs when the instrument's isolation window
co-selects the M+1 or M+2 isotopologue of the precursor alongside M+0.  Both
species fragment identically, so every fragment ion gains a satellite peak at
`+k × 1.003355 Da` (C13 spacing) with intensity proportional to the co-isolation
fraction `f`.  Removing these satellite peaks from training spectra prevents the
model from learning spurious H-transfer patterns.

The existing function `detect_isotope_peaks_for_training` (in
`src/fragnnet/utils/spec_utils.py`) already handles this well for FT spectra.
This experiment asks whether the fragmentation DAG can improve on it.

---

## Data

- **Dataset:** NIST20, FT spectra only (1,029,998 total; 120,393 with ≥1 NIST-annotated isotope peak)
- **Ground truth:** `data/proc/nist20/ann_df.pkl` — NIST `ann_isotopes` field, giving M+k order per peak
- **DAG:** `data/frag/nist20_ma_mi_d3_h4_isoFalse_cmf/dags` — depth-3, H-transfer ±4, CMF variant
- **Evaluation:** 500 randomly-sampled FT spectra with ≥1 annotated isotope peak, 2440 GT isotope peaks total

---

## DAG Data Structure

Each DAG entry (one per molecule) contains:

| Field | Shape | Description |
|---|---|---|
| `formula_peak_mzs` | `(n_formulas, 5)` | M+0 through M+4 m/z per formula variant; M+1–M+4 columns are **all zero** in this DAG build |
| `idx_to_formula` | `dict[int, str]` | Maps formula index → formula string (e.g. `"C10H10BrN"`) |
| `dag.x` | `(n_nodes, 45)` | Node features; `NODE_FEAT_TO_IDX["h_formulae_idx"]` points each node to its formula index |

Key observations:
- `formula_peak_mzs[:, 0]` gives the monoisotopic M+0 m/z for every formula variant (~900 per molecule on average)
- Only M+0 is stored — M+k positions must be computed as `M+0 + k × C13_SPACING`
- `idx_to_formula` has ~264 entries per molecule, but the DAG nodes typically use only 2–5 unique formula indices (the rest are H-transfer variants across all possible fragment sizes)

---

## Methods Tested

All three DAG methods take the output of `detect_isotope_peaks_for_training` as
a starting point and modify it.

### 1. `current` (baseline)

`detect_isotope_peaks_for_training` with `formula` and `prec_type` provided.
Uses `frag_mz / prec_neutral_mass` as a proxy for the fragment's elemental
composition when computing the per-fragment expected M+k/M+0 ratio.

### 2. `dag_filter`

Keep only flagged peaks whose parent M+0 is found in the DAG (within 15 mDa),
or whose parent M+0 ≈ `prec_mz` (precursor own isotope peaks, within 50 mDa).

### 3. `dag_protect`

Un-flag any peak whose m/z directly matches a DAG M+0 node (within 15 mDa).
Logic: a peak confirmed as a real monoisotopic fragment by the DAG cannot be an
M+k isotope satellite.

### 4. `dag_formula`

Re-check each flagged (M+0, M+k) pair using the exact fragment formula from the
DAG instead of the m/z proxy.  Estimates the co-isolation fraction `f` from
precursor residual peaks above `prec_mz`, then gates on
`ratio_lo × R_exp_exact × f ≤ obs ≤ ratio_hi × R_exp_exact × f`.
Un-flags pairs that fail the exact gate.

---

## Results (500 spectra, 2440 GT isotope peaks)

| Method | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Current | 1895 | 237 | 545 | 0.889 | 0.777 | 0.829 |
| DAG-filter | 1735 | 191 | 705 | 0.901 | 0.711 | 0.795 |
| DAG-protect | 369 | 39 | 2071 | 0.904 | 0.151 | 0.259 |
| DAG-formula | 1892 | 236 | 548 | 0.889 | 0.775 | 0.828 |

---

## Analysis

### DAG-filter: bad trade-off

Removes 46 FP but loses 160 TP (F1: 0.829 → 0.795).

**Why it loses TP:** ~10% of NIST-annotated isotope peaks have a parent M+0 that
is not in the DAG.  This is a structural gap — the DAG is depth-3, so fragments
that require deeper cleavage sequences are absent.  Additional losses come from
tolerance mismatches (the DAG M+0 position can differ from the observed peak by
more than 15 mDa due to H-transfer ambiguity).

Breakdown of annotated isotope parent locations (100-spectrum sample):
- Parent fragment in DAG: **82.7%**
- Parent is precursor itself: **7.1%**
- Parent fragment NOT in DAG: **10.2%**

### DAG-protect: catastrophically bad

F1 collapses to 0.259.  The DAG contains ~900 M+0 nodes per molecule at FT
resolution.  Many isotope satellite peaks (M+k of real fragments) happen to land
within 15 mDa of a *different* fragment's M+0 position in the DAG, so they get
un-protected even though they are genuine isotope artifacts.  This approach is
fundamentally unsound at FT mass accuracy with dense DAG node coverage.

### DAG-formula: no practical effect

Removes only 1 FP and 3 TP (essentially unchanged from current).

**Why:** The m/z proxy error is 10–13% median relative error in the expected
M+k/M+0 ratio (mean absolute error 0.014).  The ratio gate uses a 5× window
(`ratio_lo=0.2`, `ratio_hi=5.0`).  A 10–13% error in the expected ratio is far
too small to move a peak across a 5× decision boundary.  The proxy is adequate
for the current gate width.

M+1 ratio proxy vs. exact DAG formula (200-spectrum sample, 586 matched pairs):
- Median relative error: **10%**
- Mean relative error: **13.3%**
- Fraction with >20% error: **20.8%**
- Fraction with >50% error: **1.7%**

---

## Conclusions

**The DAG does not improve isotope peak detection** as a filter, protector, or
ratio-refinement step under the current evaluation setup.

Key reasons:
1. **DAG coverage gap (~17%):** ~10% of real fragment M+0 peaks are not in the
   depth-3 DAG, and ~7% are the precursor itself (not a DAG node).  Any
   approach that requires DAG confirmation will lose these.
2. **DAG density vs. FT resolution:** With ~900 M+0 nodes per molecule and 15 mDa
   tolerance, the probability of a false coincidence is high.  At FT resolution
   (4 mDa m/z accuracy), isotope satellites and nearby DAG nodes overlap often.
3. **Proxy error is irrelevant at current gate width:** The 5× ratio window
   absorbs the 10–13% formula-estimation error.  Tightening the gate (e.g.
   `ratio_hi=2.0`) would make the proxy error matter, but would also reduce
   recall from the current method.

**The `detect_isotope_peaks_for_training` baseline is already well-calibrated.**
The 237 remaining FP in 500 spectra (11% false discovery rate) are the next
target; they appear to require approaches outside the DAG, such as:
- Examining molecule class distributions of FP spectra
- Checking whether FP peaks cluster near specific neutral losses
- Tightening `min_coiso_fraction` or `aggressive_cv` thresholds

---

## Implementation Notes

- `_expected_mk_ratio` and `_parse_adduct_counts` are importable from `spec_utils`
  (underscore-prefixed but accessible for scripts)
- DAG M+0 lookup uses `formula_peak_mzs[:, 0]` — the `idx_to_formula` dict keys
  align with the row indices of this array
- The `DagInfo` helper class in the debug script wraps all DAG lookups
