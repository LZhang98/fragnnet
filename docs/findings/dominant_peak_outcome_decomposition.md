# Dominant-peak outcome decomposition: ranking headroom vs unexplained fragments

> **Status:** FINDING &nbsp;·&nbsp; **Updated:** 2026-08-12
>
> Base-peak failure is ~7:1 a ranking problem, not a DAG-coverage problem. At K=1, 36.7% of
> spectra have the true base peak available in the prediction but ranked below something else,
> against 5.2% where the DAG cannot reach it at all. Resolves the `min_peak_int` ambiguity left
> open by [`fragment_intensity_ranking_findings.md`](fragment_intensity_ranking_findings.md).

**Date:** 2026-08-12
**Script:** `scripts/eval/dominant_peak_diagnostic.py`
**Prediction export:** `output/inference_eval/pqx7ywc9/predictions_test.hdf5` (NIST20 **test**,
96,128 spectra / 5,173 molecules, mean `cos_hun_inst_tol` 0.755)
**DAG:** `data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags`
**Outputs:** `output/diagnostics/dominant_peak_pqx7ywc9_test/` (3 figures, 3 CSVs)

## Question

For the peaks that actually carry the spectrum — the top-K most intense observed peaks — does the
model put its own mass there? And when it does not, is that because the fragmentation DAG never
generated the fragment (an **unexplained fragment**, no training fixes it), or because the fragment
was available and the model ranked it too low (**model headroom**)?

## Method

Each of the top-K observed peaks of every spectrum is followed into the prediction and lands in one
of four outcomes:

| Outcome | Meaning | Whose problem |
|---|---|---|
| `hit` | in the exported prediction, model ranks it inside its own top-K | — |
| `under_ranked` | in the exported prediction, ranked outside the model's top-K | model |
| `zero_pred` | DAG reaches the m/z, but the model gave it under the `min_peak_int` export floor | model |
| `no_dag` | no fragment formula reaches that m/z at instrument tolerance | DAG |

Matching uses the eval-time per-instrument tolerance (FT 10 ppm / floor 200 Da, QTOF 20 ppm /
floor 750 Da) and the `1_above_prec` filter tier, matching `dag_oracle_scores.py`.

### The `min_peak_int` correction — read this before reusing the numbers

**The exported `pred_mzs` is not the DAG support.** `run_save_inference_eval.py` defaults to
`--min_peak_int 1.0`, which drops every candidate peak below 0.1% of the base peak — in this export
the median spectrum keeps only **5.3%** of its candidate m/z (mean 8.4%, n=2,000 audited).
Classifying "not in DAG" against the
export alone therefore charges the DAG for candidates the *model* zeroed out. This is the ambiguity
`fragment_intensity_ranking_findings.md` flagged and left open.

This diagnostic rebuilds the true support from `formula_peak_mzs` / `formula_peak_probs` in both the
std and CMF ion-mass conventions, reproducing `_dag_ion_mzs` in `dag_oracle_scores.py`. The
correction moves 0.8 pp at K=1 and 5.1 pp at K=10 out of `no_dag` and into `zero_pred`.

## Result

| K | hit | under_ranked | zero_pred | no_dag | n peaks |
|---|---|---|---|---|---|
| 1 | **0.573** | **0.367** | 0.008 | **0.052** | 96,128 |
| 3 | 0.590 | 0.293 | 0.020 | 0.096 | 285,436 |
| 5 | 0.574 | 0.264 | 0.033 | 0.130 | 459,807 |
| 10 | 0.555 | 0.224 | 0.051 | 0.171 | 823,031 |

At K=1 the model headroom (`under_ranked` + `zero_pred` = 37.5%) exceeds the DAG headroom (5.2%) by
**7.2×**. The DAG reaches the true base peak in **94.8%** of spectra; the model ranks it #1 in only
**61.0%** of those.

**The hit rate does not improve with more room.** As K goes 1 → 10, DAG coverage falls 94.8% → 82.9%
while the model hit rate stays flat at 0.555–0.601. The model is not exploiting the coverage it
already has, at any depth.

Base-peak outcome is strongly associated with the score — mean `cos_hun_inst_tol` is 0.912 on hits,
0.584 on ranking misses, 0.309 when the base peak is outside the DAG. This is association, not a
causal decomposition: a generally bad prediction produces both.

## The misses are local, and biased toward under-fragmentation

Of the 35,262 base peaks that are present but out-ranked:

- **66.4%** land within ±60 Da of the true base peak, on **discrete neutral-loss offsets** — the
  offset histogram spikes at H, CH₃, NH₃, H₂O, C₂H₂, CO, CH₂O, C₂H₂O, CO₂, HCOOH. The model picks a
  DAG neighbour one loss away, not a random peak.
- **57.3%** sit at *higher* m/z than the true base peak: the model systematically under-fragments.
- **17.4%** sit exactly on the precursor ion, but only 6.9% of those spectra actually have the
  precursor as base peak. Across the whole set the model calls the precursor the base peak in 25.8%
  of spectra against a true 21.6%.

## Collision energy drives both failure modes

FT, base peak (K=1):

| NCE (%) | hit | under_ranked | zero_pred | no_dag | n |
|---|---|---|---|---|---|
| 0–20 | 0.781 | 0.211 | 0.004 | 0.004 | 9,947 |
| 20–40 | 0.640 | 0.338 | 0.006 | 0.016 | 29,237 |
| 40–60 | 0.574 | 0.386 | 0.007 | 0.033 | 16,538 |
| 60–80 | 0.501 | 0.429 | 0.007 | 0.063 | 11,918 |
| 80–100 | 0.459 | 0.441 | 0.008 | 0.093 | 7,648 |
| 100–120 | 0.409 | 0.430 | 0.015 | 0.146 | 6,291 |
| 120–200 | 0.381 | 0.432 | 0.016 | 0.171 | 8,786 |

Ranking failure saturates around 43% by NCE 60 and then stops growing; past that point the further
decline is bought almost entirely by DAG coverage (`no_dag` 0.4% → 17.1%). **Ranking is the binding
constraint at low-to-mid energy; coverage becomes co-binding above NCE ~100.**

QTOF (ACE, n=5,763 total — thin, read as indicative) shows the cleaner ranking defect: hit falls
0.763 → 0.567 from 0–10 eV to 30–40 eV while `no_dag` stays flat at 1.3–2.4%.

Instrument and adduct at K=1: QTOF 0.640 hit vs FT 0.569; `[M+Na]+` 0.596, `[M-H]-` 0.579,
`[M+H]+` 0.568, `[M+Cl]-` 0.556. `[M+Cl]-` has the highest ranking-failure share (42.6%) and
essentially no coverage failure (0.7%).

Worst ClassyFire classes by base-peak hit rate (n ≥ 500): benzopyrans 0.433, coumarins 0.484,
prenol lipids 0.484, steroids 0.489, diazanaphthalenes 0.523, azoles 0.525. Benzopyrans, coumarins
and isoflavonoids also carry the highest `no_dag` share (10–12%), consistent with the existing
benzopyranone / furanocoumarin ring-cleavage rule work. These class results are descriptive and
confounded by acquisition stratum.

## The unexplained-fragment side

Per-spectrum true intensity the DAG cannot reach: **median 1.8%, mean 8.7%**. 23.4% of spectra are
fully explained; only 4.1% have more than half their intensity unreachable. But the effect where it
exists is steep and near-linear — mean `cos_hun_inst_tol` falls 0.811 → 0.046 across the
unexplained-fraction range.

So unexplained fragments are a **tail problem, not a median problem**, which matches
[`unexplainable_spectra_analysis.md`](unexplainable_spectra_analysis.md) rather than adding to it.
The `zero_pred` share is small everywhere (0.8% at K=1, 5.1% at K=10): the export floor is not
hiding a large pool of DAG-covered-but-discarded dominant peaks.

## Validation

- Per-spectrum `n_true_peaks` reproduces `dag_oracle_scores.py`'s value **exactly on all 96,128
  spectra** — the `1_above_prec` filter is reproduced faithfully.
- Per-spectrum in-DAG peak count agrees exactly with the oracle's `n_achievable_peaks_inst_tol` on
  **99.75%** of spectra and is ≤ it on 99.91% (the oracle CSV was built over base **+ ringmerge**
  DAGs, a superset of the base DAG used here).
- Every exported `pred_mzs` value falls inside the reconstructed DAG support on a 800-spectrum
  audit (100.0%), confirming the support reconstruction is a superset of what inference emitted.

## What this says to do next

1. **A ranking/competition objective is the right target, and it is not CE-gated.** The dominant
   failure exists at every energy and saturates by NCE 60, so it is not primarily a CE-conditioning
   problem. This supports the
   [sibling-ranking target plan](../plans/sibling_ranking_target_plan.md) and the
   [global fragment intensity ranking plan](../plans/global_fragment_intensity_ranking_experiment_plan.md)
   over further CE-conditioned modules — consistent with
   [`ce_extent_headroom_finding.md`](ce_extent_headroom_finding.md) being null.
2. **The confusions are local and chemically structured.** Misses concentrate on single
   neutral-loss offsets between DAG-adjacent fragments, which is exactly the parent-local sibling
   comparison a ranking loss operates on. A ranking target restricted to same-parent / one-loss-apart
   pairs would cover two thirds of the observed base-peak error.
3. **Fix the precursor over-call.** The model calls the precursor the base peak 4.2 pp more often
   than it should, and 17.4% of wrong base peaks are the precursor. This is a cheap, directly
   measurable sub-target.
4. **Do not spend on DAG expansion for the base peak.** 5.2% at K=1 is the entire available prize,
   and it is concentrated in benzopyrans/coumarins/isoflavonoids and NCE > 100.

## Caveats

- Single checkpoint (`pqx7ywc9`), NIST20 test split, one DAG. Not a matched-control experiment.
- "Model's rank" is over the intensity-filtered export, so ranks are optimistic relative to ranking
  over the full candidate set; the `zero_pred` category bounds that effect at 0.8% (K=1).
- Base-peak outcome vs cosine is association, not causal decomposition.
- QTOF strata are thin (5,763 spectra total; QTOF·[M+Na]+ is n=226 at K=1).

## Reproduce

```bash
python scripts/eval/dominant_peak_diagnostic.py \
    --pred_fp output/inference_eval/pqx7ywc9/predictions_test.hdf5 \
    --spec_df_fp data/proc/nist20/spec_df.pkl \
    --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags \
    --classyfire_fp data/classyfire/nist20_mol_classyfire.pkl \
    --out_dp output/diagnostics/dominant_peak_pqx7ywc9_test
```

Runtime ~2 min. `--frag_dp` is required; without the true DAG support the `no_dag` category is
not meaningful (see the `min_peak_int` correction above).
