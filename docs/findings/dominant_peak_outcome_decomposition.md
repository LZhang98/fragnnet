# Dominant-peak outcome decomposition: ranking headroom vs unexplained fragments

> **Status:** FINDING &nbsp;·&nbsp; **Updated:** 2026-08-13
>
> Base-peak failure is ~7:1 a ranking problem, not a DAG-coverage problem. At K=1, 36.7% of
> spectra have the true base peak available in the prediction but ranked below something else,
> against 5.2% where the DAG cannot reach it at all. Resolves the `min_peak_int` ambiguity left
> open by [`fragment_intensity_ranking_findings.md`](fragment_intensity_ranking_findings.md).
>
> Companion proposed work: the
> [ion × collision-energy matrix experiment plan](../plans/ion_collision_energy_matrix_experiment_plan.md).

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

| Outcome          | Meaning                                                                           | Whose problem |
| ---------------- | --------------------------------------------------------------------------------- | ------------- |
| `hit`          | in the exported prediction, model ranks it inside its own top-K                   | —            |
| `under_ranked` | in the exported prediction, ranked outside the model's top-K                      | model         |
| `zero_pred`    | DAG reaches the m/z, but the model gave it under the`min_peak_int` export floor | model         |
| `no_dag`       | no fragment formula reaches that m/z at instrument tolerance                      | DAG           |

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

| K  | hit             | under_ranked    | zero_pred | no_dag          | n peaks |
| -- | --------------- | --------------- | --------- | --------------- | ------- |
| 1  | **0.573** | **0.367** | 0.008     | **0.052** | 96,128  |
| 3  | 0.590           | 0.293           | 0.020     | 0.096           | 285,436 |
| 5  | 0.574           | 0.264           | 0.033     | 0.130           | 459,807 |
| 10 | 0.555           | 0.224           | 0.051     | 0.171           | 823,031 |

At K=1 the model headroom (`under_ranked` + `zero_pred` = 37.5%) exceeds the DAG headroom (5.2%) by
**7.2×**. The DAG reaches the true base peak in **94.8%** of spectra; the model ranks it #1 in only
**60.4%** of those.

**The hit rate does not improve with more room.** As K goes 1 → 10, DAG coverage falls 94.8% → 82.9%
while the model hit rate stays flat at 0.555–0.601. The model is not exploiting the coverage it
already has, at any depth.

Base-peak outcome is strongly associated with the score — mean `cos_hun_inst_tol` is 0.912 on hits,
0.584 on ranking misses, 0.309 when the base peak is outside the DAG. This is association, not a
causal decomposition: a generally bad prediction produces both.

## The misses are local

Of the 35,262 base peaks that are present but out-ranked:

- **66.4%** land within ±60 Da of the true base peak, on **discrete neutral-loss offsets** — the
  offset histogram spikes at H, CH₃, NH₃, H₂O, C₂H₂, CO, CH₂O, C₂H₂O, CO₂, HCOOH. The model picks a
  DAG neighbour one loss away, not a random peak.
- **17.4%** sit exactly on the precursor ion, but only 6.9% of those spectra actually have the
  precursor as base peak. Across the whole set the model calls the precursor the base peak in 25.8%
  of spectra against a true 21.6%.

## It is not global under-fragmentation — it is a low-CE precursor over-call

57.3% of all-instrument **under-ranked** base-peak misses sit at higher m/z than the true one, which
looks like under-fragmentation. It is not, on two controls:

1. **Chance is not 50%.** Only **47.0%** of a spectrum's DAG candidate m/z lie above its true base
   peak (n=4,000 audited), so the unbiased rate is 47%, not 50%.
2. **The bias is entirely a low-CE effect.** Among under-ranked FT misses, the higher-m/z share runs
   0.785 (NCE 0–20) → 0.626 → 0.559 → 0.520 → 0.513 → 0.510 → **0.491** (NCE 120–200). Above NCE
   ~80 it is 0.513, 0.510, and 0.491: approaching the 47.0% candidate-set chance rate, but remaining
   above it in every quoted bin.

Intensity-weighted mean m/z agrees: predicted minus true is **+26.8 Da (7.6% of the precursor) at
NCE 0–20** and within ±1.4 Da for every bin above NCE 60. This reproduces the shape of
[`ce_extent_headroom_finding.md`](ce_extent_headroom_finding.md) — global fragmentation extent is
already correct; only the *argmax* is biased, and only at low energy.

The mechanism is a precursor default. FT, base peak:

| NCE      | true base = precursor | predicted base = precursor | over-call        | hit   |
| -------- | --------------------- | -------------------------- | ---------------- | ----- |
| 0–20    | 0.712                 | 0.827                      | **+0.116** | 0.781 |
| 20–40   | 0.304                 | 0.367                      | +0.062           | 0.640 |
| 40–60   | 0.156                 | 0.189                      | +0.033           | 0.574 |
| 60–80   | 0.047                 | 0.057                      | +0.009           | 0.501 |
| 120–200 | 0.001                 | 0.003                      | +0.002           | 0.381 |

The over-call is roughly a constant **×1.2 multiplicative overshoot** on the precursor at every
energy; it is large in absolute terms only where the base rate is large. Conditioning on it splits
the low-CE performance completely:

| NCE 0–20                                    | hit             | n     |
| -------------------------------------------- | --------------- | ----- |
| true base peak**is** the precursor     | **0.951** | 7,078 |
| true base peak is**not** the precursor | **0.362** | 2,869 |

**The model is playing the population prior "the precursor survives at low CE".** That is right 71%
of the time and it collects nearly all of it (95.1%); it fails two thirds of the remainder.

### Post-hoc scalar precursor correction is rejected

A post-hoc sweep tested whether this was merely a calibratable global precursor offset. On the
`9t63zils` NIST20 validation export, the 9,947 FT NCE 0–20 spectra were split by molecule into
deterministic calibration and evaluation halves (seed 17). For each candidate bias `b`, every
predicted peak matching the precursor was multiplied by `exp(-b)` before recomputing base-peak
outcomes, intensity-weighted mean m/z, and instrument-tolerance Hungarian cosine. The bias was
selected only by calibration-half cosine.

Held-out evaluation half (n=4,935):

| precursor log-bias | precursor-winner hit | non-precursor-winner hit | overall base hit | precursor calibration gap | mean m/z error (Da) |          cosine |
| -----------------: | -------------------: | -----------------------: | ---------------: | ------------------------: | ------------------: | --------------: |
|      **0.0** |      **0.953** |                    0.376 |  **0.777** |                    +0.120 |               +3.21 | **0.878** |
|                0.2 |                0.925 |                    0.422 |            0.772 |                    +0.081 |               -0.48 |           0.877 |
|                0.6 |                0.846 |                    0.503 |            0.742 |                    -0.005 |               -7.94 |           0.863 |

Calibration cosine selected **`b = 0.0`**. A bias large enough to close the precursor-frequency
gap improves the non-precursor minority only by sacrificing correctly predicted precursor winners,
overall base-peak accuracy, cosine, and mean-m/z calibration. A single global precursor penalty is
therefore a trade, not a correction. The next test must be structure-dependent: compare the
precursor with the strongest DAG-supported non-precursor contender using their observed soft
intensity ratio, while retaining both precursor-win and fragment-win examples.

The sweep operates on the intensity-filtered prediction export. Its reconstructed zero-bias cosine
was 0.879 versus 0.883 stored before export filtering (difference -0.0029), small enough not to alter
the selected zero-bias conclusion. Script:
[`precursor_bias_sweep.py`](../../scripts/eval/precursor_bias_sweep.py). Output:
`output/diagnostics/precursor_bias_9t63zils_val.csv`.

What beats the precursor in that failing cohort (NCE < 40, true base ≠ precursor, n=23,205, hit
0.523) is a single facile, structure-specific cleavage: H₂O 7.1%, CO₂ 5.6%, NH₃ 3.3%,
**C₆H₁₀O₅ (hexose/glycoside) 2.6%**, HCl 1.7%, C₂H₂O 1.6%, HCOOH 2.2%. Median loss is 97 Da and only
36% are under 60 Da, so this is not a small-loss effect — it is *labile-bond identity*. The model
knows the average molecule's behaviour and not this molecule's weakest bond, which is the same
generalization signature as [`cc_corruption_probe_finding.md`](cc_corruption_probe_finding.md)
(features are used) plus a capacity-ample / generalization-limited fit.

## Collision energy drives both failure modes

FT, base peak (K=1):

| NCE (%)  | hit   | under_ranked | zero_pred | no_dag | n      |
| -------- | ----- | ------------ | --------- | ------ | ------ |
| 0–20    | 0.781 | 0.211        | 0.004     | 0.004  | 9,947  |
| 20–40   | 0.640 | 0.338        | 0.006     | 0.016  | 29,237 |
| 40–60   | 0.574 | 0.386        | 0.007     | 0.033  | 16,538 |
| 60–80   | 0.501 | 0.429        | 0.007     | 0.063  | 11,918 |
| 80–100  | 0.459 | 0.441        | 0.008     | 0.093  | 7,648  |
| 100–120 | 0.409 | 0.430        | 0.015     | 0.146  | 6,291  |
| 120–200 | 0.381 | 0.432        | 0.016     | 0.171  | 8,786  |

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
3. **Fix the precursor over-call, and target it at low CE.** The model calls the precursor the base
   peak 4.2 pp more often than it should overall, but the defect is concentrated at NCE < 60
   (+11.6 pp at NCE 0–20) and takes the form of a ×1.2 multiplicative overshoot. The addressable
   cohort is sharply defined: **low CE, true base peak is not the precursor** — 2,869 spectra at
   NCE 0–20 where hit is 0.362 against 0.951 when the precursor does win. Because the loss is
   cosine on a precursor-dominant spectrum, betting on the precursor is close to free; an explicit
   penalty for the precursor losing to a fragment has a clean target. The post-hoc sweep rejects a
   global scalar precursor penalty: calibration cosine selects zero, while larger corrections only
   trade precursor-winner accuracy for non-precursor-winner accuracy. Use a structure-dependent
   precursor-versus-supported-contender target instead.
4. **Do not spend on DAG expansion for the base peak.** 5.2% at K=1 is the entire available prize,
   and it is concentrated in benzopyrans/coumarins/isoflavonoids and NCE > 100.

## Companion experiment plan

The proposed ion × collision-energy matrix work lives in
[`ion_collision_energy_matrix_experiment_plan.md`](../plans/ion_collision_energy_matrix_experiment_plan.md).
V1 keeps training sparse and uses the matrix as its diagnostic contract; V2 joint multi-CE decoding
is gated on a positive matched V1 result.

## Caveats

- The dominant-peak decomposition uses one checkpoint (`pqx7ywc9`) on NIST20 test; the post-hoc
  calibration sweep separately uses `9t63zils` on NIST20 validation. Neither is a matched training
  experiment, and their absolute metrics should not be compared as if they were the same model.
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

Post-hoc precursor-bias sweep:

```bash
python scripts/eval/precursor_bias_sweep.py \
    --pred_fp output/inference_eval/9t63zils/predictions_val.hdf5 \
    --spec_df_fp data/proc/nist20/spec_df.pkl \
    --out_fp output/diagnostics/precursor_bias_9t63zils_val.csv
```
