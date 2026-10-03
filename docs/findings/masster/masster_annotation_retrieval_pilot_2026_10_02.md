# Fragment and formula annotations for MSGYM retrieval: 400-query pilot

> **Status:** FINDING · **Updated:** 2026-10-02 · Pilot, not the plan's Gate 1. On formula
> val, `ll04i00m`'s fragment-cost scores add to linear cosine but add nothing once the query is
> scored by its likelihood under the full predicted spectrum. That likelihood beats cosine by
> +14.8 hit@1 points; most of that gain is small-peak weighting.
> GLACIER's released MSG checkpoint is level with `ll04i00m` on every spectrum-level score
> (cosine hit@1 0.465 against 0.477), and the likelihood lifts it by the same amount.

## Question

Can a candidate's predicted fragment annotations rank MassSpecGym retrieval candidates better
than cosine? The annotations are the per-peak ion formula plus the fragment's atom set over
the parent molecule. The design and gates are in the
[annotation-aware retrieval plan](../../plans/annotation_aware_retrieval_plan.md). This pilot
runs that plan's Step 1 scores and its Step 2 fragment scores together on a small sample, to
see whether fragment-level scoring is worth building at full scale.

## Setup

- **Checkpoint.** `ll04i00m` at epoch 60 (MASSter, MSGYM benchmark 1.5).
  - Its hparams name the retired `FT` label, so it was copied with `FT` renamed to `ORB`. MSGYM
    has one Orbitrap label, and the sorted embedding rows (FT, QTOF) → (ORB, QTOF) are
    unchanged. The weights are not modified.
  - **Re-check under current code:** MSGYM test mean `cos_sim_0.01` is **0.571272**, against
    0.571277 recorded ([GLACIER comparison](masster_msgym_vs_glacier_released_ckpt_2026_09_24.md)).
    87 of 9,954 spectra moved by more than 1e-4 (max 0.0145).
- **Queries.** 400 molecules sampled with seed 20261002 from the 2,331 molecules of the
  `ms_gym_benchmark_1_5` val split (all `[M+H]+` with a CE), one spectrum each. That gives 375
  ORB and 25 QTOF queries, with median nearest-train Tanimoto 0.36.
- **Candidates.** MSGYM 1.5 formula val candidates: 72,946 spectra, median 235 per query
  (cap 256).
  - Each candidate row is a copy of its query's benchmark `spec_df` row with only the molecule
    swapped. CE, instrument and dual-CE extras are therefore exactly what the benchmark fed the
    model, and the in-model cosine is computed against the query.
  - The truth candidate's in-model cosine reproduces the benchmark val export (median
    |Δ| 1.2e-7 over 379 queries).
- **Export.** `--min_peak_int -1` with annotation: a mean 952 predicted peaks per candidate, 5.1 GB.
- **Scores.** All parameters were fixed in `score_pilot.py`'s docstring before the first read.
  - Tolerance is the checkpoint's instrument rule: ORB 10 ppm with a 200 m/z floor, QTOF 20 ppm
    with a 750 m/z floor.
  - (a) `cos`: in-model `cos_sim_0.01`.
  - (b) `supp`: the L2 share of non-precursor query peaks that the candidate's predicted support
    reaches.
  - (c) `ll`: Σ y·log max(p̂, 1e-5), with p̂ the predicted intensity within tolerance.
  - (F1) `cost`: Σ_k w_k · max over the candidate's fragments explaining peak k of
    2^−(cuts + |ΔH|). `cuts` counts parent bonds with one end in the atom set; ΔH = ion H −
    (atom-set H + 1). This is a MetFrag-style explanation cost on the model's own fragments.
  - (F3) `pars`: the same, but peaks are taken in decreasing intensity and each is charged only
    for cut bonds no earlier peak used (shared cleavage sites).
  - `_c` variants subtract the mean score of the query shifted by ±5, ±7 and ±11 Da.
  - Blends with cosine choose α on even-hash molecules (224) and report on odd-hash
    molecules (176).
  - Ties count as expected hits under random tie-breaking. Intervals are 2,000-resample query
    bootstraps; with one query per molecule, this is a molecule bootstrap.

On the formula track every candidate has the query's formula, so each query peak has the same
subformula candidates for all of them. Formula-level evidence therefore reduces to support
(b): the formula string adds nothing that the m/z at tolerance does not already carry.

## Results

### Single scores (400 queries; random hit@1 = 0.025)

| score | hit@1 [95% CI] | hit@5 | hit@20 | Δhit@1 vs `cos` |
| --- | --- | ---: | ---: | --- |
| (a) `cos` | 0.478 [0.428, 0.528] | 0.760 | 0.888 | — |
| `cos_hun` | 0.480 | 0.758 | 0.880 | +0.003 [−0.015, +0.018] |
| (b) `supp` | 0.239 | 0.540 | 0.808 | −0.239 |
| (b) `supp_c` | 0.187 | 0.446 | 0.713 | −0.291 |
| **(c) `ll`, ε 1e-5** | **0.625 [0.578, 0.668]** | **0.908** | **0.983** | **+0.148 [+0.100, +0.195]** |
| (c) `ll`, ε 1e-4 / 1e-6 | 0.618 / 0.630 | 0.888 / 0.908 | 0.978 / 0.983 | +0.140 / +0.153 |
| (F1) `cost` / `cost_c` | 0.342 / 0.340 | 0.705 / 0.688 | 0.932 / 0.915 | −0.136 / −0.138 |
| (F3) `pars` / `pars_c` | 0.330 / 0.314 | 0.703 / 0.692 | 0.935 / 0.940 | −0.147 / −0.164 |

### Blends, reported on the 176 odd molecules

| blend | α (even) | hit@1 | Δhit@1 [95% CI] | Δhit@5 | Δhit@20 |
| --- | ---: | ---: | --- | --- | --- |
| `cos` + `supp_c` | 0.9 | 0.432 | +0.000 [−0.023, +0.023] | +0.000 | +0.011 |
| `cos` + `cost_c` | 0.7 | 0.466 | +0.034 [−0.006, +0.080] | +0.074 [+0.028, +0.125] | +0.063 [+0.028, +0.102] |
| `cos` + `cost` | 0.7 | 0.483 | +0.051 [+0.006, +0.097] | +0.063 [+0.017, +0.108] | +0.063 [+0.023, +0.102] |
| `cos` + `pars_c` | 0.8 | 0.466 | +0.034 [−0.006, +0.074] | +0.051 [+0.011, +0.097] | +0.046 [+0.011, +0.085] |
| `cos` + `ll` | 0.2 | 0.585 | +0.153 [+0.091, +0.216] | +0.125 [+0.074, +0.176] | +0.080 [+0.046, +0.125] |
| `ll` + `cost_c` *(post hoc)* | 0.9 | 0.608 | −0.011 [−0.034, +0.011] | +0.006 | +0.000 |
| `ll` + `pars_c` *(post hoc)* | 0.9 | 0.608 | −0.011 [−0.034, +0.011] | +0.000 | +0.000 |
| `ll` + `supp_c` *(post hoc)* | 0.9 | 0.625 | +0.006 [−0.011, +0.028] | −0.006 | +0.000 |

The odd-half cosine baseline is 0.432 / 0.767 / 0.898. The `ll` base is 0.619 / 0.909 / 0.983.

### What the likelihood gain is (post hoc, 400 queries)

| score | hit@1 | hit@5 | hit@20 | Δhit@1 vs `ll` |
| --- | ---: | ---: | ---: | --- |
| `cos_sqrt` (0.01 Da bins, sqrt intensities) | 0.588 | 0.848 | 0.935 | −0.038 [−0.078, +0.000] |
| `ent` (unweighted spectral entropy similarity) | 0.573 | 0.845 | 0.930 | −0.053 [−0.090, −0.015] |
| `ll` without the precursor term | 0.628 | 0.895 | 0.981 | +0.003 [−0.020, +0.023] |

The precursor carries a median 0.4% of query intensity.

### Controls

- **Spectrum-blind consensus pseudo-query.** This is the top 30 bins of the mean predicted
  spectrum of the query's own candidates. Scored against it, every score stays between 0.018
  and 0.033 hit@1, close to random: `cos` 0.033, `ll` 0.028, `cost_c` 0.018, `pars_c` 0.031.
- **Same-formula partner query.** Only 49 queries have a partner (a val query with the same
  formula and another molecule). Under it the truth gets `cos` 0.102, `ll` 0.061 and
  `cost_c` 0.041 hit@1.
- **Support size and bond count** score 0.019–0.028 hit@1 in either direction.
- **Candidate list position is degenerate.** MassSpecGym's candidate JSON stores the truth at
  index 0 (3,000 of 3,000 sampled query SMILES), and the build keeps that order, so
  "first wins" gives hit@1 = 1.000. This build cannot measure the PubChem-order bias
  ([MassSpecGym audit](../../notes/massspecgym_in_the_wild_audit.md)); that needs PubChem CIDs.
  Any scorer that breaks ties by position leaks the answer.

### Subgroups (hit@1)

| group | n | `cos` | `ll` | `cost_c` | `pars_c` |
| --- | ---: | ---: | ---: | ---: | ---: |
| ORB | 375 | 0.493 | 0.635 | 0.348 | 0.324 |
| QTOF | 25 | 0.240 | 0.480 | 0.213 | 0.153 |
| nearest-train Tanimoto < 0.3 | 64 | 0.297 | 0.438 | 0.264 | 0.212 |
| 0.3–0.4 | 203 | 0.537 | 0.675 | 0.372 | 0.354 |
| ≥ 0.4 | 133 | 0.474 | 0.639 | 0.327 | 0.302 |

The likelihood gain over cosine is about +0.14 to +0.17 in every novelty bin. It is not
concentrated on novel molecules, as the plan hypothesized.

Mean within-query Spearman: `cos`–`ll` 0.75, `cost_c`–`pars_c` 0.92, `supp_c`–`cost_c` 0.73,
`cos`–`cost_c` 0.34.

### GLACIER's released MSG checkpoint on the same 400 queries (added 2026-10-02)

The same 72,946 candidate spectra (each under its query's CE, instrument and precursor) were run
through ms-pred's `predict_smis_joint.py` with the checkpoint, ms-pred checkout and
`--sparse-k 256` of the [GLACIER comparison](masster_msgym_vs_glacier_released_ckpt_2026_09_24.md).
Its output is squared to linear intensity. Every candidate got a prediction.

- `cos` uses that comparison's exact CUDA scorer.
- The other scores use the definitions above, applied to GLACIER's top-256 peaks.
- GLACIER exposes no fragment annotations in this output, so only spectrum-level scores apply.

| score | MASSter hit@1 / @5 / @20 | GLACIER hit@1 / @5 / @20 | GLACIER − MASSter hit@1 [95% CI] |
| --- | --- | --- | --- |
| `cos` | 0.477 / 0.760 / 0.887 | 0.465 / 0.748 / 0.875 | −0.013 [−0.068, +0.037] |
| `cos_sqrt` | 0.588 / 0.848 / 0.935 | 0.562 / 0.858 / 0.948 | −0.025 [−0.080, +0.028] |
| `ent` | 0.573 / 0.845 / 0.930 | 0.565 / 0.845 / 0.940 | −0.007 [−0.060, +0.045] |
| `ll` (ε 1e-5) | 0.625 / 0.907 / 0.983 | 0.615 / 0.887 / 0.958 | −0.010 [−0.060, +0.037] |
| `ll` without precursor | 0.627 / 0.895 / 0.980 | 0.600 / 0.877 / 0.958 | −0.027 [−0.077, +0.023] |

- **The two models are statistically level on every score.** The truth's mean `cos` is 0.615 for
  GLACIER and 0.634 for MASSter, the same direction as the test-set forward gap.
- **The likelihood gain is not model-specific.** It lifts GLACIER from 0.465 to 0.615 hit@1, as
  it lifts MASSter from 0.477 to 0.625.
- **GLACIER's `ll` is computed on its top 256 peaks** against MASSter's mean 952, so more of
  GLACIER's query peaks fall to the ε floor. The `ll` comparison partly measures that truncation.
- QTOF (n = 25): GLACIER `cos` hit@1 0.240, the same as MASSter.
- The first GLACIER pass was OOM-killed on its last 1,026 (largest) molecules: 8 workers each
  held a featurization cache of large molecules, pushing the machine into swap. Those 1,026
  were rerun with 3 workers and no cache in 24 s.

## Interpretation

1. **Fragment identity carries signal that linear cosine discards, and the full predicted
   distribution already holds it.** The explanation cost improves cosine at hit@5 and hit@20
   (+5 to +7 points, intervals above zero) but changes nothing on top of `ll`. A query peak that
   only a cheap fragment explains is one the model already gives non-negligible intensity, and
   cosine ignores it because it is small.
2. **Structural parsimony adds nothing beyond per-peak cost** (ρ 0.92, identical blend gains).
   The literature sweep found no ranking test of joint explanation coherence; this pilot is a
   negative one at this scale.
3. **Support alone is uninformative on the formula track.** MASSter's selected support averages
   952 peaks per candidate, so nearly every isomer reaches most query peaks.
   Chance-correction lowers it further. Formula-level matching adds nothing here (see Setup).
4. **The ranking lever is the similarity's intensity weighting.** Linear cosine is dominated by
   the few largest peaks, which are where the model's novel-molecule errors concentrate.
   Sqrt-cosine recovers +11.0 of the +14.8 points, entropy +9.5. The likelihood stays ahead at
   hit@5 and hit@20.

## Pipeline findings

- **Train/serve molecule skew.** Training molecules go through `02_prepare_proc.py` with
  tautomer canonicalization on by default ([02_prepare_proc.py:104](../../../preproc_scripts/02_prepare_proc.py#L104)).
  MSGYM retrieval candidates are built with `ml_standardize=False`
  ([01_prepare_msgym_retrieval.py:50](../../../preproc_scripts/msgym_ms2c/01_prepare_msgym_retrieval.py#L50)).
  - For 95 of 2,331 val molecules (660 of 9,731 queries), the benchmark's standardized molecule
    has a different 2D InChIKey from MSGYM's source molecule (anthracyclines as hydroquinone
    tautomers, for example).
  - Truth must be identified by the source SMILES' key, not the benchmark `mol_df` key.
  - In the 21 such pilot queries the truth's own prediction moved (median |Δcos| 0.034 against
    the benchmark).
  - Standardizing candidates the way training data is standardized is an untested choice.
- **The retrieval `spec_df` files still say `FT`.** The current loader refuses them. The pilot
  relabelled its own subset; the files in `data/ms2c/msgym/v1_5/*.zip` are unchanged.

## Limits

- One checkpoint, which is not the best MSGYM recipe; formula track only; val only. 400 queries
  overall and 176 for blends. Only 25 QTOF and 49 partner queries.
- No minimum useful gain was fixed, so this is not the plan's Gate 1. The `ll`-based blends,
  `cos_sqrt`, `ent` and the precursor split were added after the first read.
- The fragment scores use the model-selected support, not a structural enumeration. The cost
  model `2^−(cuts+|ΔH|)` is untuned and has no bond energies.
- Val hit@1 is not comparable to published MSGYM test numbers. The pilot molecules must be
  excluded from any Gate 1 read.

## Artifacts

All under `output/annotation_retrieval_pilot_2026_10_02/`:

- `patch_ft_to_orb.py` and `ckpt/ll04i00m_ep060_orb.ckpt`; `test_recheck/` and its log.
- `build_pilot_inputs.py`, `data/pilot/` (spec/mol/split, `pilot_queries.csv`,
  `pilot_candidates.csv`) and `query_novelty.py`.
- `pred/ll04i00m_ep060_orb/predictions_test.hdf5` (5.1 GB, annotated, `--min_peak_int -1`) and
  `pred.log`.
- `score_pilot.py` (the fixed protocol), `analyze_pilot.py`, `analyze_pilot.log`,
  `scores/part_*.pkl` and `scores/summary_*.csv`.
- `score_extra.py`, `analyze_extra.py`, `analyze_extra.log` and `scores/extra_*.pkl`.
- `glacier/`: `build_labels.py`, `labels_pilot*.tsv`, `preds_k256*/preds.hdf5` and logs,
  `score_glacier_pilot.py`, `scores_k256.pkl`, `compare.py`, `compare_k256.csv` and `.log`.
