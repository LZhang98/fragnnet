# NIST20 Data-Quality Plan

Tracks measured data-quality issues in the NIST20 training set and the filters proposed to
address them. Scope so far: contaminant peaks (UWPR/MaConDa) and DAG-unexplainable spectra.
All numbers below are on the analysis set: FT (NCE 0–70) + QTOF (ACE 0–60 eV), adducts
`[M+H]+`/`[M-H]-`/`[M+Na]+`/`[M+Cl]-`, default D3/H4 DAGs, per-instrument tolerance
(FT 10 ppm, QTOF 20 ppm), CMF ∪ CRF, full H-transfer.

## Summary of issues

| Issue | Spectra | Severity | Action |
|---|---|---|---|
| Zero-coverage — precursor-mass mismatch (annotation error) | 16 | real, tiny | filter out |
| Zero-coverage — sparse (≤2 peaks) | 144 | low-information | flag / down-weight |
| Zero-coverage — genuine DAG coverage gap | 95 | model/DAG limit | not a data issue |
| UWPR contaminant peaks (peak-level) | 71,503 spectra with ≥1 | pervasive, low-intensity | strip contaminant peaks |

## 1. Zero-coverage spectra ("can't explain anything")

Spectra where the DAG matches **no** true peak (`recall == 0`).

- **255 spectra = 0.069%** of 370,957 (`wrecall < 0.01`: 331 = 0.089%). Negligible in aggregate.
- Only **7.5%** contain the precursor peak — these are fragment-only spectra (the intact-ion,
  which the DAG always explains, is absent).
- **93.3%** have a precursor m/z consistent with the assigned molecule **within the
  per-instrument matching tolerance** (FT 10 ppm / QTOF 20 ppm), so the molecule↔spectrum link
  is mostly sound. The error distribution is strongly bimodal — 91% at ≤0.002 Da, a clean gap,
  then a cluster at 4–12 Da — so the consistency count is insensitive to the exact threshold
  (93.3% at 0.005 Da through 93.7% at 1.0 Da). The threshold is set by the tolerance, not
  a hand-picked cutoff.

Breakdown by cause:

| category | n | % | data issue? |
|---|---|---|---|
| Sparse (≤2 peaks) | 144 | 56% | borderline — too few peaks to be informative |
| Genuine coverage gap | 95 | 37% | no — real fragmentation beyond the D3 DAG |
| Precursor-mass mismatch (gross, >0.1 Da) | 16 | 6% | **yes — annotation error** |
| Precursor-mass mismatch (mild, >inst-tol) | 1 | 0.4% | QTOF C5H8O5 at 142 ppm |

Data: `data/eval/oracle_cos_nist20_default/zero_coverage_spectra.csv`
(script: `scripts/eval/dag_oracle_scores.py` for recall, + zero-coverage profiling).

### 1a. Precursor-mass mismatch — lipid annotation slippage

The 16 mismatches are almost all **long-chain unsaturated fatty acids / oxylipins**, with the
reported precursor off by clean ~4–12 Da steps (≈ multiples of a degree-of-unsaturation gap):

| spec_id | adduct | prec_mz | expected | Δ (Da) | molecule |
|---|---|---|---|---|---|
| 1108496/7/5 | `[M-H]-` | 291.31 | 279.23 | **+12.07** | linoleic acid C18H32O2 |
| 1108223/2 | `[M-H]-` | 306.25 | 301.22 | +5.03 | C20H30O2 |
| 1108133 | `[M-H]-` | 332.26 | 327.23 | +5.03 | C22H32O2 |
| 1108579–81 | `[M-H]-` | 373.25 | 369.23 | +4.03 | C20H34O6 oxylipin |

These are NIST lipid entries whose precursor is tagged to a related-but-wrong degree of
unsaturation. **Action: filter spectra where `|prec_mz − (exact_mw + adduct_shift)|` exceeds the
per-instrument tolerance** (FT 10 ppm / QTOF 20 ppm). This is a general annotation-sanity filter,
not lipid-specific. Because the error distribution is bimodal (exact vs 4–12 Da), the 16 gross
mismatches are caught by any cut from ~0.005 to ~4 Da; using the matching tolerance (rather than
a flat 0.1 Da) additionally catches 1 borderline QTOF case (C5H8O5 at 142 ppm).

## 2. UWPR / MaConDa contaminant peaks

Peak-level contamination: NIST reference peaks that match a known contaminant m/z (solvent,
PEG/PPG, DMSO, phthalate, …) **and** cannot be explained by the molecule's DAG.

- **71,503 spectra** have ≥1 `contaminant_only` peak; of 561,651 DAG-evaluable matched peaks,
  **51% are contaminant_only**.
- Removing them lifts oracle coverage on affected spectra: recall **+3.3 pts**, wrecall **+0.9 pt**,
  opt-cos **+0.25 pt** — a clean but small, low-intensity-peak hygiene gain.
- Feasibility per peak: `scripts/eval/uwpr_feasibility.py` →
  `data/uwpr/nist20_uwpr_hits_feasibility.csv`. Denoised oracle scoring is wired into
  `dag_oracle_scores.py` via `--uwpr_feasibility_csv`.

**Action: strip `contaminant_only` peaks during preprocessing** (not whole-spectrum removal —
the rest of each spectrum is real).

## Proposed cleanup pipeline

1. **Precursor-mass sanity filter** — drop spectra where `|prec_mz − expected|` exceeds the
   per-instrument tolerance (FT 10 ppm / QTOF 20 ppm; ≈0.003/0.006 Da at m/z 300). Removes the
   16 lipid-annotation errors + 1 borderline QTOF case here, and similar mismatches at scale.
2. **Contaminant peak stripping** — remove `contaminant_only` peaks per spectrum via the UWPR
   feasibility map.
3. **Sparse-spectrum handling** (optional) — flag `n_peaks ≤ 2` spectra; consider down-weighting
   rather than dropping (they are low-information but not wrong).

Genuine coverage-gap spectra (37% of the zero-coverage set) are **out of scope for data cleanup**
— they are a DAG-depth/rule limitation, addressed by the fragmentation-coverage work, not filtering.
