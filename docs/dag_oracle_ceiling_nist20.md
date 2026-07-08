# DAG Oracle Ceiling on NIST20 (2026-07-08)

## TL;DR

Computed the **best possible cos_hun / jss_hun a model could ever achieve** given only a
DAG's fragment coverage (no training run required) on the full NIST20 `test` split
(96,128 spectra, 5,173 molecules, split `data/split/nist20_ma_mi_inchikey`). The
baseline d3 DAG's ceiling is **0.937 cos_hun** (fixed 10 ppm tolerance) / **0.959** under
the instrument-appropriate tolerance. Actual trained-model performance on comparable
data (NIST23, 9-fold CV, per-CE setup — see `[[project_alvhehe2_vs_pqx7ywc9_run_comparison]]`
in memory) is ~0.786 overall. These two numbers are **not from the same eval setup**
(different dataset, split, and possibly tolerance), so the gap is suggestive, not exact —
but it points at real, sizeable headroom, concentrated in specific places identified below.

## Method: what "oracle" means here

For each spectrum, the oracle prediction reproduces the true intensity **exactly** for every
true peak whose m/z is explainable by some fragment formula in the precomputed DAG (within
m/z tolerance), and predicts zero everywhere else. Scoring that oracle prediction with the
same `cos_hun_helper`/`jss_hun_helper` used in real training eval gives the ceiling imposed by
DAG fragment coverage alone, independent of how well any model predicts intensities.

- For `cos_hun` this is a real closed-form upper bound (Cauchy-Schwarz: the L2-normalized
  true-intensity mass restricted to DAG-achievable peaks).
- For `jss_hun` the same "reuse true intensities on the achievable support" construction is
  used — the standard weighted-recall-style ceiling, not a formally re-optimized
  JSS-maximizing prediction (no simple closed form exists there), but tractable and consistent
  with existing `wrecall`-style metrics in this codebase.

Script: [`scripts/eval/dag_oracle_scores.py`](../scripts/eval/dag_oracle_scores.py). Takes
~90s per DAG variant on the full test split with `--n_jobs 10`. Four tolerance conventions are
scored per spectrum:

| Column suffix | Tolerance |
|---|---|
| `_opt` | Fixed 10 ppm (ratio 1e-5), 200 Da floor — the default eval tolerance |
| `_inst_tol_opt` | Per-instrument: FT 10 ppm/200 Da, QTOF 20 ppm/750 Da (matches `inst_type_loss_tol` in production configs), other inst types fall back to 10 ppm |
| `_0p01da_opt` | Fixed absolute 0.01 Da |
| `_sqrt_opt` | Sqrt-transformed intensities at 10 ppm tolerance (de-emphasizes the dominant peak, matches `cos_hun_sqrt` eval variant) |

## Results — DAG variant comparison (fixed 10 ppm tolerance)

| DAG | cos_hun_opt | jss_hun_opt | peak coverage |
|---|---|---|---|
| d3 baseline (`smpFalse`) | 0.9371 | 0.9280 | 73.8% |
| d3 + SMARTS prepass (`smpTrue`) | 0.9377 | 0.9286 | 74.1% |
| d3 + ring-merge | **0.9519** | **0.9453** | 77.5% |
| d4 | **0.9518** | **0.9473** | 79.3% |

Ring-merge and d4 are tied on ceiling; ring-merge gets there without d4's combinatorial
blowup risk (a 1.3M-node outlier was observed at depth 4 on a 3,000-mol sample — see
`[[project_bond_breakability_mask_plan]]`). SMARTS prepass barely moves anything
(+0.0006), consistent with `[[feedback_prepass_rule_id]]`.

## Results — baseline DAG, all four tolerance conventions (full test split)

Overall (mean):

| Variant | cos_hun_opt | jss_hun_opt |
|---|---|---|
| Fixed 10 ppm | 0.9371 | 0.9280 |
| Per-instrument tol | 0.9591 | 0.9448 |
| Fixed 0.01 Da | 0.9666 | 0.9571 |
| Sqrt-transformed | 0.9266 | 0.8987 |

By instrument / mode:

| | n | cos_hun_opt | cos_hun_inst_tol_opt | cos_hun_0.01Da_opt | cos_hun_sqrt_opt |
|---|---|---|---|---|---|
| FT CID | 7,266 | 0.9778 | 0.9778 | 0.9828 | 0.9649 |
| FT HCD | 83,099 | 0.9559 | 0.9559 | 0.9652 | 0.9436 |
| QTOF CID | 5,763 | 0.6151 | **0.9818** | 0.9669 | 0.6323 |

**QTOF's low fixed-tolerance ceiling (0.615) is mostly a measurement-tolerance artifact, not
a fragment-coverage problem.** Under QTOF's proper 20 ppm/750 Da tolerance the ceiling jumps
to 0.982 — matching FT. This means the CID/QTOF gap noted in memory
(`[[project_alvhehe2_vs_pqx7ywc9_run_comparison]]`, actual QTOF cos_hun ~0.616 in the NIST23
per-CE setup) is very unlikely to close via better fragment/DAG modeling — it's a
loss-tolerance-configuration lever (confirm `inst_type_loss_tol` is actually wired into the
training loss, not just eval) or true measurement-precision noise, not "left on the table."

By adduct (prec_type):

| Adduct | n | cos_hun_opt | cos_hun_inst_tol_opt | cos_hun_0.01Da_opt | cos_hun_sqrt_opt |
|---|---|---|---|---|---|
| [M+Cl]⁻ | 1,497 | 0.9906 | 0.9906 | 0.9949 | 0.9739 |
| [M+H]⁺ | 62,177 | 0.9242 | 0.9532 | 0.9602 | 0.9145 |
| [M+Na]⁺ | 7,719 | 0.9426 | 0.9535 | 0.9748 | 0.9273 |
| [M-H]⁻ | 24,735 | 0.9647 | 0.9738 | 0.9785 | 0.9539 |

**[M+H]⁺ is both the largest adduct group (65% of spectra) and the lowest ceiling** at every
tolerance setting, including sqrt (0.914) where tolerance artifacts are minimized. Unlike the
QTOF case, this gap survives the tolerance relaxation — i.e. it looks like genuine DAG/fragment
coverage headroom, not a measurement artifact. [M+Cl]⁻ and [M-H]⁻ look near-ceiling (>0.96 even
under sqrt) — but see the CMF ablation below before concluding they're "not worth investment".

### Why [M+Cl]⁻ and [M+Na]⁺ score so high: it's the CMF convention, not easy fragmentation

Peak matching (both here and in production eval, `frag_utils.compute_frag_peak_stats` /
`spec_mol_frag_dataset._apply_fragment_mz_change`) checks a true peak against **two** ion-mass
conventions per DAG fragment and unions them: `std` (fragment retains the adduct atom
directly) and `CMF` (charge-migration-fragment: the adduct leaves, charge is carried by
proton loss/gain instead). Re-running the baseline DAG on the full test split with
`--no_cmf` (std-only) isolates how much of each adduct's ceiling comes from which mechanism:

| Adduct | cos_hun_opt (CMF ∪ std) | cos_hun_opt (std only) | Δ |
|---|---|---|---|
| [M+Cl]⁻ | 0.9906 | **0.2658** | **−0.725** |
| [M+Na]⁺ | 0.9426 | 0.7844 | −0.158 |
| [M+H]⁺ | 0.9242 | 0.9166 | −0.008 |
| [M-H]⁻ | 0.9647 | 0.9639 | −0.0008 |

**[M+Cl]⁻'s near-perfect ceiling is almost entirely the CMF hypothesis.** Drop it and [M+Cl]⁻
becomes the *worst*-explained adduct in the dataset (0.27) — worse than [M+H]⁺. This matches
the chemistry documented in `PREC_TYPE_TO_CMF_MASS_DIFF` (`formula_utils.py`): [M+Cl]⁻
fragments are really `[frag-H]⁻` — the chloride leaves during fragmentation entirely, and the
charge is carried the same way as in [M-H]⁻. Real chloride-*retaining* fragments are rare.
[M+Na]⁺ shows the same effect at roughly a fifth the size — a mix of genuine Na-retaining
fragments and H-mediated migration, not CMF-dominated like chloride.

This also rules out CMF∪std being a "cheap" ceiling inflation from doubling the candidate mass
hypotheses (i.e. more chances for a true peak to randomly land within tolerance of *something*):
if that were the mechanism, [M+H]⁺/[M-H]⁻ should drop comparably when one of their two candidate
sets is removed. They don't (−0.008, −0.0008) — for those adducts CMF is already the dominant,
chemically correct mechanism, and `std` is largely redundant with the DAG's own ±4 H-transfer
buckets. So **the CMF-inclusive ceilings reported above are the chemically faithful ones**, not
an artifact — [M+Cl]⁻ genuinely is well-explained once the correct fragmentation mechanism is
accounted for, matching how production eval already scores it.

## Where performance is actually left on the table

Ranking the headroom signals from this analysis, most to least actionable:

1. **[M+H]⁺ fragment coverage** — lowest ceiling across all tolerance settings, largest
   spectrum count. A genuine DAG-coverage gap, not a tolerance artifact. Worth auditing which
   [M+H]⁺ true peaks fall outside the DAG's achievable set (the `n_achievable_peaks` /
   `n_true_peaks` columns in the per-spectrum CSV make this queryable directly).
2. **Ring-merge / d4 DAG upgrade** — +0.015 cos_hun / +0.017-0.019 jss_hun ceiling over the
   current production baseline DAG, concentrated in FT-HCD (the bulk of the data, +0.017).
   Ring-merge reaches the same ceiling as d4 without d4's node-blowup risk — prefer it.
3. **QTOF/instrument-tolerance wiring** — not a DAG problem. If `inst_type_loss_tol` isn't
   already active in the training loss (only in eval), wiring it in could recover most of the
   QTOF gap directly, no DAG work needed.

## Caveats

- These ceilings are computed against `data/split/nist20_ma_mi_inchikey`'s `test` split; the
  ~0.786 actual-model figure in memory is from a **different dataset and eval protocol**
  (NIST23, 9-fold CV, per-CE). No apples-to-apples actual-vs-oracle number exists yet — to get
  one, run the trained model's real eval on this same NIST20 split with the same tolerance
  definitions used here.
- The jss_hun oracle is a tractable heuristic (reuse true intensities on the achievable
  support), not a formally re-optimized maximum — treat it as directionally correct, not exact.
- Ring-merge/d4/smpTrue numbers above are from the earlier 4-DAG comparison run at fixed 10 ppm
  tolerance only; the inst_tol/0.01Da/sqrt breakdown was only run for the baseline DAG so far.
- The CMF ablation (`--no_cmf`) was only run for the baseline DAG at fixed 10 ppm tolerance;
  the per-instrument/0.01Da/sqrt columns in that run (see script output) are included in the
  saved CSV but not tabulated above since they don't change the headline finding.

## Reproducing

```bash
python scripts/eval/dag_oracle_scores.py \
    --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags \
    --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse_ringmerge/dags \
    --split test --n_jobs 10 \
    --out_fp <output_path>.csv

# CMF ablation (isolates how much of an adduct's ceiling comes from the
# charge-migration-fragment ion-mass convention vs direct adduct retention)
python scripts/eval/dag_oracle_scores.py \
    --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags \
    --split test --n_jobs 10 --no_cmf \
    --out_fp <output_path>.csv
```
