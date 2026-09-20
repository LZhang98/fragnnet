# DAG Oracle Ceiling on NIST20 (2026-07-08)

> **Status:** FINDING &nbsp;·&nbsp; **Updated:** 2026-08-28
>
> Baseline d3 DAG ceiling = 0.937 cos_hun on NIST20 test. Model-free upper bound.

## TL;DR

Computed the **best possible cos_hun / jss_hun a model could ever achieve** given only a
DAG's fragment coverage (no training run required) on the full NIST20 `test` split
(96,128 spectra, 5,173 molecules, split `data/split/nist20_ma_mi_inchikey`). The
baseline d3 DAG's ceiling is **0.937 cos_hun** (fixed 10 ppm tolerance) / **0.959** under
the instrument-appropriate tolerance.

`output/inference_eval/{alvhehe2,pqx7ywc9}/predictions_test.hdf5` are real trained-model
inference outputs on this **exact same 96,128-spectrum test split** (verified: identical
spec_id sets) and already store a real `cos_hun_inst_tol` per spectrum — giving a genuine,
apples-to-apples actual-vs-oracle comparison (no dataset/protocol mismatch, unlike the NIST23
figures cited in earlier drafts of this doc):

| Run | actual cos_hun_inst_tol | oracle cos_hun_inst_tol_opt | **gap** |
|---|---|---|---|
| alvhehe2 | 0.7521 | 0.9591 | **0.2070** |
| pqx7ywc9 | 0.7555 | 0.9591 | **0.2036** |

**~20 points of cos_hun is genuinely left on the table**, and it is not evenly distributed. By
per-spectrum rate it looks concentrated in rare adducts ([M+Cl]⁻) and instruments (QTOF); by
**total error mass (gap × n), which is the actual prioritization signal, 87% sits in FT-HCD and
62% in [M+H]⁺** — see "Where performance is actually left on the table" below.

**Cool finding, drilling into that dominant FT-HCD ∩ [M+H]⁺ bucket**: the dramatic near-zero-score
tail (rare/novel scaffolds, an out-of-scope-detector blind spot) makes for a vivid story but
explains only ~1-2% of total error mass. **82% of spectra already score ≥0.5 and still account
for 55% of total error mass.** For that bulk, the model already *selects* nearly the right
fragments (71% match rate vs. the DAG's own 78% ceiling) but only moderately gets their
*relative intensity ranking* right (median Spearman rho 0.69, matched-peaks-only) — a
ranking/calibration problem, not a coverage problem. See "The dramatic tail is a red herring for
total impact" below.

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

Script: [`scripts/eval/dag_oracle_scores.py`](../../scripts/eval/dag_oracle_scores.py). Takes
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

### Provisional GLACIER coverage comparison

GLACIER reports NIST'20 peak **coverage** alongside spectral similarity in Appendix G
(Tables 15--17): for example, 0.887 for `[M+H]+` without contrastive finetuning and 0.843
for positive+negative mode with contrastive finetuning. The paper states that NIST'20 uses
0.1 Da binned spectra, but it does not define coverage in enough detail to establish a
top-20 evaluation protocol.

The released `ms-pred/analysis/spec_pred_eval.py` evaluator has a configurable
`--max-peaks` argument whose **default** is 20. With that setting, it keeps the 20 strongest
predicted bins and 20 strongest target bins and computes
`|target bins intersect predicted bins| / |target bins|`. This establishes what the public
evaluator does when invoked without an override; it does **not** establish that the reported
paper runs used 20. The authors could have supplied another value (including a value large
enough to retain all peaks), and no checked paper command or run metadata has yet confirmed
the actual argument. Therefore, do not describe GLACIER's published coverage as "top-20
coverage" without that additional provenance.

As a protocol sensitivity check, applying the evaluator's default-20 behavior to the local
NIST20 test split gives the following DAG oracle coverage:

| Bin width | d3 oracle | d4 oracle |
|---|---:|---:|
| 0.01 Da | 0.7674 | 0.8203 |
| 0.1 Da | 0.9334 | 0.9475 |

These are candidate-support ceilings, not trained-model predictions. They are also not yet a
strict comparison to GLACIER because the peak-count argument used for GLACIER's published
numbers is unverified, and the 0.01 Da row is not the NIST'20 resolution reported by GLACIER.
The safe conclusion is conditional: **if** the published GLACIER evaluation used the released
default of 20 peaks, d3 already provides high support at 0.1 Da and ranking/intensity allocation
is the more likely bottleneck. If GLACIER retained all peaks, the comparison must be recomputed
under that protocol.

**2026-09-20 update, from the PDF (`data/paper/GLACIER_arxiv_2606.29161.pdf`, gitignored):** the
0.1 Da resolution is now **confirmed** rather than inferred — Section 4.1 states "Binned spectral
vectors are used with 0.1 resolution on the NIST'20 dataset and 0.01 resolution on the
MassSpecGym dataset to align with prior work on both benchmarks." So the **0.1 Da row is the
right comparison row** and the 0.01 Da row is not. Under it, d3 oracle 0.9334 and d4 0.9475 both
sit above GLACIER's achieved 0.887, so candidate support is not the binding constraint at that
protocol. The `--max-peaks 20` provenance is still **unconfirmed** for the paper's own runs — the
caution above stands. Note also that GLACIER's FraGNNet row (0.717 random / 0.654 scaffold) is
adapted from Wang et al. (ICEBERG 2.0), not produced under our evaluation protocol.
See [`notes/glacier_object_detection.md`](../notes/glacier_object_detection.md).

**2026-09-20, SECOND update — the `--max-peaks` provenance is now RESOLVED, and it is 100, not 20.**
`run_scripts/glacier/02_predict_inten.py` (upstream `coleygroup/ms-pred`) passes
`--max-peaks 100 --min-inten 0 --formula-dir-name no_subform.hdf5`. Every model in the lineage
passes 100 (CFM-ID passes 1000); the argparse default of 20 is never used by any run script.
**The default-20 protocol-sensitivity table above is therefore at the wrong protocol** and its
d3/d4 oracle rows (0.7674/0.8203 at 0.01 Da, 0.9334/0.9475 at 0.1 Da) should be recomputed with
`max_peaks=100` before being compared to GLACIER's 0.887. Not yet redone.

Also retracted: an earlier note here and in the GLACIER note implied their evaluation target is
subformula-filtered. It is not — `no_subform` comes from `01_assign_subformulae.py --use-all`,
which writes `formula_mass_no_adduct = observed m/z - adduct mass` and drops nothing for lacking a
subformula. The only target filter is `max_inten_spec` (top-50 peaks above 0.001 relative
intensity).

Protocol-matched trained-model number for reference: our FraGNNet (`pqx7ywc9`) scores **0.757**
cosine at 0.1 Da under their exact settings on our test split (96,128 spectra), vs GLACIER's
published 0.838.

Sources: [GLACIER paper](https://arxiv.org/html/2606.29161v1),
[`spec_pred_eval.py`](https://github.com/coleygroup/ms-pred/blob/main/analysis/spec_pred_eval.py).

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
tolerance setting, including sqrt (0.914) where tolerance artifacts are minimized. [M+Cl]⁻ and
[M-H]⁻ look near-ceiling (>0.96 even under sqrt) — but see the CMF ablation below, and then the
real actual-vs-oracle gap section further down, before concluding low ceiling ⇒ where the real
model headroom is. It turns out **ceiling alone is the wrong signal** for that question — the
real per-adduct gap (actual model score vs. its own ceiling) inverts this ranking.

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

## Real actual-vs-oracle gap: alvhehe2 & pqx7ywc9 model runs

`output/inference_eval/{alvhehe2,pqx7ywc9}/predictions_test.hdf5` are real trained-model
inference outputs (see `[[project_alvhehe2_vs_pqx7ywc9_run_comparison]]` in memory) on
**exactly this NIST20 test split** — verified identical `spec_id` sets to
`data/split/nist20_ma_mi_inchikey/test_ids.csv` (96,128 spectra each) — and each already
stores a real per-spectrum `cos_hun_inst_tol`. Joining that to the baseline DAG's
`cos_hun_inst_tol_opt` oracle gives a genuine, same-split, same-tolerance actual-vs-ceiling
gap (no cross-dataset caveat needed, unlike an earlier draft of this doc that used NIST23
9-fold numbers as a rough proxy).

Overall:

| Run | actual | oracle | gap |
|---|---|---|---|
| alvhehe2 | 0.7521 | 0.9591 | 0.2070 |
| pqx7ywc9 | 0.7555 | 0.9591 | 0.2036 |

By instrument/mode (alvhehe2; pqx7ywc9 is within ~0.001-0.002 throughout):

| | n | actual | oracle | gap |
|---|---|---|---|---|
| FT CID | 7,266 | 0.7577 | 0.9778 | 0.2201 |
| FT HCD | 83,099 | 0.7473 | 0.9559 | 0.2086 |
| QTOF CID | 5,763 | 0.8141 | 0.9818 | 0.1676 |

By adduct:

| Adduct | n | actual (alvhehe2) | oracle | gap (alvhehe2) | actual (pqx7ywc9) | gap (pqx7ywc9) |
|---|---|---|---|---|---|---|
| [M+Cl]⁻ | 1,497 | 0.7163 | 0.9906 | **0.2742** | 0.7273 | **0.2632** |
| [M-H]⁻ | 24,735 | 0.7469 | 0.9738 | 0.2269 | 0.7497 | 0.2241 |
| [M+Na]⁺ | 7,719 | 0.7406 | 0.9535 | 0.2129 | 0.7485 | 0.2050 |
| [M+H]⁺ | 62,177 | 0.7564 | 0.9532 | 0.1968 | 0.7593 | 0.1939 |

**This inverts the ceiling-only ranking above.** [M+H]⁺ has the lowest *ceiling* but the
*smallest* actual gap — the model already captures a larger share of what's achievable for it.
[M+Cl]⁻ has the highest ceiling (via the CMF mechanism) but the *largest* gap: the model is
worst, in absolute terms, at recovering the ceiling that's chemically available to it. Ceiling
alone was the wrong signal for prioritization; the gap is.

## Where performance is actually left on the table

Per-spectrum gap (used above) rewards rare-but-broken buckets over common-but-slightly-broken
ones. Weighting by volume — **error mass = mean gap × n** — is the actual prioritization
signal, since it estimates each bucket's share of total unrealized cos_hun across the split
(alvhehe2 numbers; pqx7ywc9 is within ~1-2% relative throughout):

By instrument/mode:

| | gap | n | error mass | share |
|---|---|---|---|---|
| FT CID | 0.2201 | 7,266 | 1,599 | 8.0% |
| **FT HCD** | 0.2086 | 83,099 | **17,335** | **87.1%** |
| QTOF CID | 0.1676 | 5,763 | 966 | 4.9% |

By adduct:

| | gap | n | error mass | share |
|---|---|---|---|---|
| [M+Cl]⁻ | 0.2742 | 1,497 | 411 | 2.1% |
| [M-H]⁻ | 0.2269 | 24,735 | 5,612 | 28.2% |
| [M+Na]⁺ | 0.2129 | 7,719 | 1,643 | 8.3% |
| **[M+H]⁺** | 0.1968 | 62,177 | **12,236** | **61.5%** |

(The two totals agree to within 0.02% — 19,900 vs 19,903 — a useful cross-check that the
instrument and adduct breakdowns are self-consistent.)

**87% of total unrealized cos_hun is in FT-HCD; 62% is in [M+H]⁺.** [M+Cl]⁻, despite having the
largest *per-spectrum* gap, is only 2% of total opportunity — it's a rare adduct (1.6% of
spectra). Ranked by actual impact:

1. **FT-HCD / [M+H]⁺ model quality** — dominant by volume on both axes (these overlap heavily:
   [M+H]⁺ is mostly FT-HCD spectra). This is where model-side work (not just DAG coverage) would
   move the overall metric the most.
2. **Ring-merge / d4 DAG upgrade** — +0.015 cos_hun / +0.017-0.019 jss_hun *ceiling* over the
   current production baseline DAG (fixed 10 ppm tolerance), concentrated in **FT-HCD** (+0.017)
   and **[M+H]⁺** (+0.017 to +0.025) — see the by-instrument/by-adduct breakdown below. This
   directly targets the #1 error-mass bucket, not a low-volume one like [M+Cl]⁻. It raises the
   ceiling but doesn't by itself close the gap — whether it also reduces the *actual* gap (i.e.
   whether the model can use the new coverage) is untested until a model is trained on ring-merge
   DAGs. Ring-merge vs d4 is a tractability-vs-ceiling-quality tradeoff, not a clean win for
   either — see "Drilling into the single dominant bucket" below, which flips the initial
   "ring-merge ties d4, so prefer it" read once weighted by where the data actually is.
3. **[M-H]⁻ (28% of error mass)** — second-largest lever by volume, worth its own look; not
   examined further here.
4. **[M+Cl]⁻ / QTOF-CID (2-5% of error mass each)** — largest *rates*, smallest *impact*. Not
   worth prioritizing on this data despite how dramatic the per-spectrum numbers look.

## Is the ring-merge/d4 ceiling gain CMF-driven, like [M+Cl]⁻/[M+Na]⁺? No — the opposite

Given the CMF finding above, a natural hypothesis is that ring-merge/d4's higher average
ceiling is itself another CMF-side-effect (e.g. deeper cuts producing more
charge-migration-compatible neutral losses). Re-running the by-adduct breakdown for ring-merge
and d4 (fixed 10 ppm, CMF included, full test split) and diffing against the baseline directly
refutes this:

| Adduct | baseline | ring-merge | Δ ring-merge | d4 | Δ d4 |
|---|---|---|---|---|---|
| [M+Cl]⁻ | 0.9906 | 0.9929 | +0.0023 | 0.9766 | **−0.0140** |
| [M+Na]⁺ | 0.9426 | 0.9575 | +0.0149 | 0.9618 | +0.0192 |
| [M-H]⁻ | 0.9647 | 0.9751 | +0.0104 | 0.9552 | −0.0095 |
| [M+H]⁺ | 0.9242 | 0.9410 | +0.0168 | 0.9487 | **+0.0245** |

By instrument/mode, the same pattern holds — the gain concentrates in the highest-error-mass
bucket (FT-HCD, 87% of total error mass above), not the highest-ceiling one:

| | baseline | ring-merge Δ | d4 Δ |
|---|---|---|---|
| FT CID | 0.9778 | +0.0020 | +0.0010 |
| **FT HCD** | 0.9559 | **+0.0166** | **+0.0163** |
| QTOF CID | 0.6151 | +0.0051 | +0.0092 |

**[M+Cl]⁻ — the most CMF-dependent adduct — gets the smallest gain from ring-merge and actually
*regresses* under d4.** [M+H]⁺ — the adduct with essentially no CMF dependence (see the ablation
above, Δ −0.008 when CMF is removed) — gets the *largest* gain under both DAG variants. The
ring-merge/d4 ceiling improvement is a genuine, broad new-fragment-formula effect from deeper
cuts, unrelated to the CMF ion-mass-convention mechanism that inflates [M+Cl]⁻/[M+Na]⁺. This is
good news for prioritization: DAG-depth investment (ring-merge in particular) targets exactly
the adduct ([M+H]⁺) already identified as the largest real-coverage gap, not a CMF artifact.

d4's small regression on [M+Cl]⁻/[M-H]⁻ is itself notable: unlike ring-merge (base d3 DAG +
strictly *added* leaf fragments, so per-spectrum coverage can only stay the same or improve),
d4 is an independently regenerated depth-4 DAG and is **not guaranteed to be a superset of d3**
per molecule — some fragment formulas present at d3 can apparently drop out at d4 (e.g. via
`max_num_dag_nodes`/pruning interactions specific to some molecules).

## Drilling into the single dominant bucket: FT-HCD ∩ [M+H]⁺

The instrument and adduct breakdowns above overlap heavily — [M+H]⁺ is mostly FT-HCD spectra.
Filtering to that intersection directly:

| | value |
|---|---|
| n | 53,058 — **55.2% of the entire test split** |
| Oracle ceiling, baseline d3 (`cos_hun_opt`) | 0.9486 |
| — without CMF | 0.9402 (Δ −0.0084 — confirms not CMF-dependent, consistent with the adduct-level ablation) |
| — ring-merge | 0.9676 (Δ +0.0191) |
| — d4 | **0.9759 (Δ +0.0273)** |
| Actual (alvhehe2 / pqx7ywc9) | 0.7491 / 0.7523 |
| Gap | 0.1994 / 0.1962 — essentially identical to the overall 0.207/0.204 gap |
| Share of total error mass | **53.2%** — over half the dataset's entire unrealized cos_hun is in this one slice |
| Peak coverage | 35.0 true peaks/spectrum on average; 23.4 achievable at baseline (67%) → 26.8 ring-merge (76%) → 27.8 d4 (79%) |

**This revises the ring-merge-vs-d4 recommendation above.** In this specific dominant slice, d4
beats ring-merge by +0.0082 (0.9759 vs 0.9676) — a bigger edge than their near-tie in the
unweighted aggregate. Weighting by error mass: d4's extra recovery here (~+0.0082 × 53,058 ≈
+435 units) exceeds its regression in the tiny [M+Cl]⁻/[M-H]⁻ buckets combined (~−0.014 × 1,497
+ −0.0095 × 24,735 ≈ −256 units), so **on pure error-mass-weighted ceiling quality, d4 slightly
edges out ring-merge** — the opposite of what the unweighted averages suggested. Ring-merge's
real advantage is *tractability* (avoiding d4's combinatorial-blowup outlier risk,
`[[project_bond_breakability_mask_plan]]`), not ceiling quality — these are two separate axes
that the earlier "ring-merge ties d4, so prefer it" framing had conflated. If blowup risk can be
managed/filtered, d4 may be the better ceiling target specifically because of its edge in this
one dominant bucket.

### Median vs mean, and what's in the tail

Within FT-HCD ∩ [M+H]⁺, mean and median diverge sharply for both the actual score and the
oracle ceiling — a long left tail of badly-scored spectra drags the mean well below what a
typical spectrum looks like:

| | mean | median |
|---|---|---|
| oracle ceiling (`cos_hun_inst_tol_opt`) | 0.9486 | **0.9993** |
| actual (alvhehe2) | 0.7491 | **0.8377** |
| gap | 0.1994 | **0.1616** |

A typical spectrum in this bucket is nearly fully DAG-explainable (median ceiling 0.9993) and
the model recovers most of it (median actual 0.8377) — the mean gap (0.199) overstates the
typical-spectrum gap (0.162).

Splitting the bottom 25% by actual score (n=13,265) from the rest characterizes the tail:

| | tail (bottom 25%, actual ≤ 0.598) | rest |
|---|---|---|
| oracle ceiling | 0.839 | 0.985 |
| n_true_peaks | 57.5 | 27.5 |
| peak coverage (achievable / true) | 0.666 | 0.786 |
| `pred_oos_prob` (model's own out-of-scope confidence) | 0.282 | 0.124 |

Two effects stack, one larger than the other:

1. **Genuinely harder chemistry** — the tail has ~2x the true peaks and a meaningfully lower DAG
   ceiling itself (0.839 vs 0.985). Correlation of n_true_peaks with actual score: **−0.39**.
2. **Model uncertainty/scaffold novelty is the stronger signal** — `pred_oos_prob` correlates
   with actual score at **−0.54**, stronger than peak count. The tail has ~2.3x the baseline
   out-of-scope probability.

The sharpest cut is the complete-failure sub-tail: **285 spectra (0.5%) score essentially zero**
actual cos_hun. Their mean oracle ceiling is still **0.478** (the DAG could explain roughly half
the true spectrum), but mean `pred_oos_prob` is **0.464** — about 4x the overall baseline
(~0.12). This reads as scaffold/structure-identification failure (the model doesn't recognize
the molecule class at all) rather than fine-grained intensity misprediction: the DAG ceiling
isn't the bottleneck here, and the model's own confidence signal already flags these cases.

This shape — a big spike near 1.0 plus a heavy left tail, median well above mean — matches the
full-test-split distribution/CDF plots already generated in `output/inference_eval/*/plots`
(e.g. `cos_hun_inst_tol` median ≈0.838, ≥0.7 for 63.7% of spectra, ≥0.9 for only 41.0%; JSS
distributions skew even further left, ≥0.7 for 0% since JSS-inst-tol tops out near 0.69 in that
plot's binning). Consistent with this doc's numbers, not an independent confirmation of the
FT-HCD∩[M+H]⁺-specific slice above (those plots aren't filtered to that slice).

### Digging deeper: is the tail rare-scaffold-driven?

Four follow-up checks on the near-zero-score sub-tail (285 spectra / 115 unique molecules),
within FT-HCD ∩ [M+H]⁺, alvhehe2 unless noted:

**1. Shared across independently different model configs, not per-model noise.** 71-77%
overlap between alvhehe2's and pqx7ywc9's near-zero-failure spec_ids (two runs with different
DAG/feature setups — prepass-DAG vs no-prepass, see `[[project_alvhehe2_vs_pqx7ywc9_run_comparison]]`).
Whatever drives this is structural, not architecture-specific noise.

**2. Partially molecule-level, not purely spectrum-level.** Mean within-molecule std of actual
score (0.168, over 4,237 molecules with ≥3 spectra in this slice) is well below the overall std
(0.250) — if one CE of a molecule fails, its other CEs often do too, though not perfectly.

**3. Molecule size is a weak driver.** num_atoms correlates with actual score at only −0.097,
far weaker than n_true_peaks (−0.39) or `pred_oos_prob` (−0.54). Rules out "just bigger
molecules are harder" as the main story.

**4. Ruled out: model defaults to intact precursor, true spectrum is fully fragmented.**
Several hand-picked, very-low-peak-count near-zero examples (a trichloromethyl carbamate, a
hydrazine-diacetic-acid, a tetrasulfide-ring compound) showed the model's top predicted peak
sitting exactly at the precursor m/z while the true spectrum had no precursor peak at all —
looked like a systematic "doesn't believe this fragments" bias. Testing across all 285 near-zero
spectra killed it: the model's top predicted peak matches the precursor m/z in only **11.2%** of
near-zero cases, actually *below* the **28.6%** baseline rate in a random sample of the same
slice. Those examples were unrepresentative anecdotes, not a general mechanism.

**5. Independent (non-model-derived) novelty check: rare/novel scaffolds are measurably
over-represented, but don't fully explain the tail.** `pred_oos_prob` is the model's own
self-assessment, so its correlation with failure is somewhat circular. Testing chemistry-based
novelty directly against the training set (Bemis-Murcko scaffold matching + Morgan-fingerprint
nearest-neighbor Tanimoto, RDKit) on the 115 near-zero-failure molecules vs. a random baseline
sample from the rest of the slice:

| | near-zero failures | baseline |
|---|---|---|
| Exact Murcko scaffold seen in training set | 56.5% | 73.9% |
| Mean NN-Tanimoto similarity to nearest training molecule | 0.546 | 0.591 |
| Fraction with NN-Tanimoto < 0.5 (quite novel) | 39.1% | 26.1% |

Both measures point the same direction: a real, moderate, independent effect (see below — it
turns out *not* to be the same signal as `pred_oos_prob`), but not a full explanation. 56.5% of
near-zero-failure molecules *do* have their scaffold represented in training, and mean Tanimoto
similarity (0.546) is still fairly high in absolute terms, just measurably lower than baseline.

### Why this happens: two separate, largely uncorrelated failure mechanisms

The natural hypothesis is that `pred_oos_prob` is simply proxying structural novelty, and that
the model appropriately hedges (spreads its predicted intensity distribution) when it detects
novelty. Testing this directly, on an 800-molecule sample of the FT-HCD ∩ [M+H]⁺ slice with
per-spectrum prediction entropy / top-1 intensity concentration computed from `pred_ints`:

| | correlation |
|---|---|
| Tanimoto-novelty ↔ `pred_oos_prob` | **−0.049** — essentially independent |
| Tanimoto-novelty ↔ actual score | 0.241 |
| `pred_oos_prob` ↔ actual score | **−0.509** |
| `pred_oos_prob` ↔ prediction concentration (top-1 intensity fraction) | **−0.488** |
| Tanimoto-novelty ↔ actual score, computed separately within low- vs high-`pred_oos_prob` halves | 0.252 / 0.218 — holds independently in both |

`pred_oos_prob` and Tanimoto-based structural novelty are **not the same signal** (r ≈ 0) —
`pred_oos_prob` is tracking something else (plausibly formula/DAG-size/precursor-mass
distribution shift rather than raw whole-molecule structural similarity; not isolated further
here). This splits the tail into two mechanistically distinct components:

1. **`pred_oos_prob`-driven failures — the model correctly hedges.** When `pred_oos_prob` is
   high, predictions genuinely become more diffuse: mean top-1 intensity concentration collapses
   from 0.513 (bottom decile) to 0.155 (top decile), and mean actual score drops from 0.903 to
   0.447 in step. This is calibration working as intended for whatever `pred_oos_prob` detects.
2. **Tanimoto-novelty-driven failures — a genuine blind spot.** This effect is independent of
   `pred_oos_prob` (holds at similar strength inside both the low- and high-`pred_oos_prob`
   halves) and does **not** trigger hedging: an earlier check (bottom-25%-Tanimoto-similarity
   vs top-25%) found near-identical prediction concentration (0.345 vs 0.329) despite a large
   actual-score gap (0.656 vs 0.810). For structurally novel molecules the OOS gate doesn't
   catch, the model stays just as confident and simply gets the dominant peaks wrong — this is
   the genuine overconfidence/miscalibration component.

So: is this a probabilistic-modeling calibration failure? **Partially, and specifically in
mechanism (2).** The model's uncertainty machinery works as intended for whatever
`pred_oos_prob` tracks — that's the majority of the correlation with failure (−0.51, vs
Tanimoto-novelty's 0.24) and it does the right thing (hedges). The real gap is that this
detector has a blind spot for whole-molecule structural novelty specifically: those cases don't
get flagged, don't get hedged, and the model commits confidently to the wrong answer.

## The dramatic tail is a red herring for total impact — the real gap is in the "boring" bulk

All of the novelty/OOS investigation above concerns the dramatic near-zero-score tail. Checking
how much of the *total* error mass that tail actually accounts for reframes the whole priority
list:

| | n | % of spectra | gap-mass | **% of total error mass** |
|---|---|---|---|---|
| Near-zero sub-tail (score < 0.05) | 285 | 0.5% | 129 | **1.2%** |
| Bottom-25% tail (score ≤ 0.60) | 13,265 | 25% | 6,151 | 58.1% |
| **"Reasonably OK" (score ≥ 0.5)** | **43,448** | **81.9%** | **5,796** | **54.8%** |

The novel-scaffold/OOS-blind-spot mechanism explains at most ~1-2% of total error mass directly
(the near-zero cases). **82% of spectra already score ≥0.5 and still account for 55% of total
error mass** — each falling short of its own ceiling by a modest, consistent ~0.13 on average.
That's not a novelty/OOD story; it's a broad, systematic shortfall across most of the data.

### Is the bulk gap a selection problem or a ranking problem?

For the "reasonably OK" bulk, checking whether the model fails to identify *which* fragments
matter (selection) vs. correctly identifies them but assigns the *wrong relative magnitudes*
(ranking) — these need different fixes (richer DAG coverage/node features vs. loss/calibration
changes targeting relative intensity order). Measured via: (a) selection rate = fraction of true
peaks matched by *any* predicted node within tolerance, vs. the DAG's own oracle match rate for
the same spectra; (b) Spearman rank correlation between true and predicted intensity, computed
only over the peaks that *are* matched (isolates ordering accuracy from coverage):

| | value |
|---|---|
| Actual selection rate (true peaks matched by some predicted node) | 71.0% |
| DAG's own oracle selection rate (n_achievable/n_true), same spectra | 77.8% |
| Selection shortfall | only ~7 points |
| Spearman rank correlation, true vs. predicted intensity among matched peaks | mean 0.672, median 0.691 |
| Spectra with rho > 0.7 (well-ranked) | 48.5% |
| Spectra with rho < 0.3 (poorly-ranked) | only 4.75% |

Selection is already close to the DAG's own ceiling (78% → 71%, a modest shortfall). But among
the fragments the model *does* correctly select, relative intensity ordering is only moderately
correlated with ground truth (median rho 0.69 — barely half the spectra clear even a 0.7
threshold). **This is predominantly a ranking/intensity-calibration problem, not a
coverage/selection problem** — concretely: the model usually knows *which* fragments exist and
matter (say A, B, C), but doesn't reliably know that A should dominate over B and C; `cos_hun`
normalizes both intensity vectors to unit length before comparing, so getting the fragments
right but their relative order/proportions wrong scores nearly as badly as missing fragments
outright.

**This reorders the improvement priority:**

1. **Ring-merge/d4 (more DAG coverage) has a lower payoff ceiling than the earlier ceiling-only
   analysis implied.** Selection is already near its own limit for the dominant bulk — more
   fragment coverage mostly closes the smaller ~7-point selection gap, not the larger ranking gap.
   Still worth testing whether a trained model on ring-merge DAGs improves things (open question
   below), but temper expectations: it targets the smaller of the two components.
2. **Confirmed: the training loss doesn't match the eval metric, and plausibly explains the
   ranking gap directly.** See the mechanism section immediately below — this is now a concrete,
   low-cost, ready-to-test lever rather than an open question.
3. The tail/novelty investigation above is real but now clearly secondary for total impact
   (~1-2% of error mass) — worth fixing on its own merits (it's egregious when it happens) but
   not where the bulk of the ~20-point gap lives.

### Why cross-entropy allows selection-right/ranking-wrong (and why the Hungarian losses fix it)

Traced into the actual loss code (`src/fragnnet/pl_model/fragnnet_pl.py`,
`src/fragnnet/model/loss.py`). Neither alvhehe2 nor pqx7ywc9's config overrides `loss_type`, so
both inherit `config/*_template.yml`'s default: **`loss_type: cross_entropy`** with
`output_distribution: gaussian`.

- **`cross_entropy` routes to `mog_ce_fn`** (`loss.py:484-524`, via `get_sparse_cross_entropy_fn`)
  — a mixture-of-Gaussians negative-log-likelihood loss computed **independently per true peak**:
  for each true peak, does the model place enough predicted probability density near that
  position? Peaks are scored roughly independently against the predicted distribution; nothing
  in the objective forces peak A's predicted magnitude to be proportionally bigger than peak B's
  just because the true spectrum says so. A model can satisfy every peak's presence-likelihood
  independently while completely scrambling their relative weights, with little penalty.
- **`log_cosine_distance_hungarian` / `jss_distance_hungarian` route to
  `sparse_cosine_distance_hungarian` (`loss.py:915`) / `sparse_jensen_shannon_divergence_hungarian`**
  — differentiable reimplementations of the exact eval metrics: same Hungarian one-to-one
  matching, same tolerance, same normalize-then-compare logic as `cos_hun_helper`/
  `jss_hun_helper`. Loss = 1 − cos_hun (or the JS analog), computed on the **whole intensity
  vector jointly**. Reducing this loss forces the model to get relative proportions across *all*
  peaks right simultaneously, because normalization ties every peak's optimal value to every
  other peak's — exactly the property cross-entropy's peak-independent formulation lacks, and
  exactly what's missing per the selection-vs-ranking diagnostic (selection near-ceiling at 71%
  vs 78%; ranking only moderate at median rho 0.69).

This isn't a naming-based guess — the independence-per-peak structure of `mog_ce_fn` and the
joint-normalization structure of the Hungarian losses map directly onto the selection-vs-ranking
split found in the data. `config/loss_funcs/fraggnn_d3_nist20_jss_hun*.yml` already exist as
prepared recipes for this; as far as this investigation found, neither alvhehe2 nor pqx7ywc9 used
them. **Training a comparison run with `loss_type: jss_distance_hungarian` (or
`log_cosine_distance_hungarian`) on the same data/DAG is the single highest-leverage,
lowest-cost next experiment** given everything in this document.

## What's still needed before a concrete improvement plan

- **Train the Hungarian-loss comparison run** (above) and check whether it closes the ranking
  gap specifically (selection-vs-ranking diagnostic re-run on its predictions) — the most direct
  test of the mechanism proposed here.
- **Does a model trained on ring-merge/d4 DAGs actually recover a smaller actual-vs-oracle gap**,
  or does the extra coverage go unused because the bottleneck is ranking, not selection? Untested.
- **What does `pred_oos_prob` actually key on?** Confirmed not to be raw Tanimoto structural
  similarity (r ≈ 0) — candidates (formula/elemental novelty, DAG size, precursor-mass range)
  are untested.
- **Is the Tanimoto-novelty blind spot fixable with more/better training data**, or is it an
  inherent generalization limit of the architecture? Would need a scaffold-holdout-style ablation
  to test causally rather than just correlationally.
- **Does the same selection-vs-ranking split hold outside FT-HCD ∩ [M+H]⁺** (e.g. QTOF, other
  adducts), or is the mechanism different elsewhere? All of this section's mechanistic work was
  done on the one (dominant) slice.

## Caveats

- The selection-vs-ranking check (n=2,866 spectra, random 3,000-spectrum sample of the "OK"
  bulk) used greedy nearest-candidate matching per true peak for the matched-predicted-intensity
  value (max predicted intensity among within-tolerance candidates), not the full Hungarian
  one-to-one assignment `cos_hun_helper` itself solves — a reasonable proxy for isolating
  ranking accuracy, but not bit-for-bit identical to the production metric's matching logic.
- The scaffold/Tanimoto novelty check (n=115 near-zero-failure molecules vs. a single random
  n=115 baseline draw, seed=0) is small-sample and not significance-tested — the direction is
  consistent across two independent measures (exact scaffold match and NN-Tanimoto), which is
  reassuring, but treat the exact percentages as indicative, not precise.
- The `pred_oos_prob` vs Tanimoto-novelty reconciliation (n=800-molecule sample) is correlational,
  not causal, and what `pred_oos_prob` is actually keyed on was not isolated here (candidates:
  formula/elemental-composition novelty, DAG size, precursor-mass range — untested). "The model
  correctly hedges when `pred_oos_prob` fires" describes the observed correlation between that
  signal and prediction diffuseness, not a mechanistic account of what triggers the signal itself.
- The DAG-variant comparison (baseline/ringmerge/smpTrue/d4) is ceiling-only — no trained model
  exists yet on ring-merge/d4/smpTrue DAGs, so whether their higher ceiling translates into a
  smaller actual-vs-oracle gap (rather than just a higher, still-unrecovered ceiling) is
  untested, not just unconfirmed-but-likely.
- smpTrue's inst_tol/0.01Da/sqrt/adduct breakdown was not run (only its fixed-10ppm overall
  number is in the DAG-variant table above) — low priority given it barely moved the ceiling.
- The jss_hun oracle is a tractable heuristic (reuse true intensities on the achievable
  support), not a formally re-optimized maximum — treat it as directionally correct, not exact.
- The CMF ablation (`--no_cmf`) was only run for the baseline DAG at fixed 10 ppm tolerance;
  the per-instrument/0.01Da/sqrt columns in that run (see script output) are included in the
  saved CSV but not tabulated above since they don't change the headline finding.
- The alvhehe2/pqx7ywc9 actual-vs-oracle gap is only computed at the `inst_tol` tolerance
  (the only real metric stored in those prediction files); the fixed-10ppm/0.01Da/sqrt actual
  scores aren't in the HDF5s and weren't recomputed from `pred_mzs`/`pred_ints` here.

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
