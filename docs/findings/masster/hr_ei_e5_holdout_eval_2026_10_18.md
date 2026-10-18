# Held-out e5 EI checkpoint on measured HR-EI spectra

> **Status:** FINDING · **Updated:** 2026-10-18
> e5 (trained without the HR-EI identities) scores below contaminated e1 only where e1 had trained on the molecule; on records e1 never saw, HREI-MSDB and Exposome are unchanged. Metabolome drops on both strata.

## Setup

Model: e5 `e5_hr_ei_holdout_nominal`, [W&B `eugy8s9n`](https://wandb.ai/frag-gnn/frag-gnn/runs/eugy8s9n),
best epoch 45 (val `cos_sim_1.0` 0.86896; the only `model-epoch` file on the run), trained
from scratch on `data/split/nist23_ei_inchikey_hr_ei_excluded_2026_10_17`
([plan](../../plans/masster/hr_ei_isotopes_holdout_plan.md)). Local checkpoint
`saved_ckpts/e5_hr_ei_holdout_nominal_eugy8s9n.ckpt`, SHA-256
`307f89467f60e708f344b38804604e395987665f3a31954dd3ceba79a66eb4cd`.

Evaluation is identical to the [frozen e1 evaluation](hr_ei_checkpoint_eval_2026_10_17.md):
the same 962 mapping-consistent records, the same scorer, the same seed, and the same mass windows,
at Git `731fe493`. It was run on the local RTX 2070 SUPER (CUDA candidates), whereas the e1 run used CPU.
Exact mass mode, with natural fine-structure isotopes and also monoisotopic output.

`nominal_train_overlap` is the raw audit's flag for e1's original split. All 962 records
are absent from e5's training, validation, and internal test partitions.

## Results

Headline tolerance is 10 ppm with a 1 mDa floor (user decision 2026-10-18). Isotopes are natural exact-mass
fine structure (`masster_isotope_envelope: natural`, order 4, minimum ratio 0.001) unless stated.

| Reference | Spectra | e1 (contaminated) | e5 (held out) | e5 median | e5 monoisotopic | e5 5 ppm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HREI-MSDB | 329 | 0.8618 | 0.8497 | 0.8712 | 0.8272 | 0.8495 |
| RECETOX Exposome | 383 | 0.7549 | 0.7062 | 0.7772 | 0.6127 | 0.6315 |
| RECETOX Metabolome | 250 | 0.5973 | 0.5726 | 0.5743 | 0.5764 | 0.5724 |

Isotopes on minus off for e5 at 10 ppm, paired: HREI-MSDB +0.0225 [+0.0151, +0.0310], Exposome
+0.0935 [+0.0759, +0.1104], Metabolome −0.0038 [−0.0057, −0.0021]. Mean measured intensity with a
predicted match (10 ppm, isotopes on) is 0.973, 0.844 and 0.768.

Paired e5 − e1 at 10 ppm with isotopes on, split by e1 training overlap (bootstrap 95% CI from 2,000 resamples):

| Reference | e1 overlap | n | e1 | e5 | Δ | 95% CI |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| HREI-MSDB | no | 132 | 0.8501 | 0.8466 | −0.0036 | [−0.0137, +0.0067] |
| HREI-MSDB | yes | 197 | 0.8697 | 0.8518 | −0.0179 | [−0.0298, −0.0070] |
| Exposome | no | 163 | 0.7152 | 0.7119 | −0.0034 | [−0.0269, +0.0214] |
| Exposome | yes | 220 | 0.7844 | 0.7019 | −0.0824 | [−0.1028, −0.0622] |
| Metabolome | no | 136 | 0.5702 | 0.5467 | −0.0234 | [−0.0405, −0.0057] |
| Metabolome | yes | 114 | 0.6296 | 0.6034 | −0.0262 | [−0.0481, −0.0029] |

The 5 ppm and monoisotopic comparisons show the same overlap pattern, within 0.01.

### By molecular-ion m/z (e5, 10 ppm, isotopes on)

`output/hr_ei_e5_eugy8s9n_eval_2026_10_18/natural_exact/mz_bin_sina_10ppm.{png,csv}`, from
[plot_hr_ei_mz_sina.py](../../../scripts/debug/ei/plot_hr_ei_mz_sina.py). Median cosine per bin:

| m/z bin | HREI-MSDB | Exposome | Metabolome |
| --- | --- | --- | --- |
| <150 | 0.93 (31), 0.91 (119) | 0.82 (10) | 0.82 (1) |
| 150–200 | 0.80 (102) | 0.85 (34) | 0.71 (3) |
| 200–250 | 0.82 (47) | 0.87 (69) | 0.67 (12) |
| 250–300 | 0.96 (17) | 0.77 (69) | 0.63 (41) |
| 300–400 | 0.94 (5) | 0.79 (142) | 0.64 (90) |
| 400–500 | 0.78 (8) | 0.54 (50) | 0.63 (57) |
| ≥500 | — | 0.35 (9) | 0.41 (46) |

Above m/z 400, every source drops. Mass explains part of the source gap: the median Metabolome molecular
ion is about 400, versus about 167 for HREI-MSDB. Mass does not explain all of it, because Metabolome is
0.13–0.20 below Exposome in each bin from 200 to 400. Exposome has a low tail (cosine <0.3) in every
bin from 200 to 500; those records have not been examined.

### Why RECETOX Metabolome is lowest

243 of the 250 Metabolome records are TMS derivatives (GC-Orbitrap, methoximation/silylation).
At 10 ppm, the model leaves 23% of measured intensity unmatched, versus 3% on HREI-MSDB, and 45%
of predicted intensity has no measured match. Three causes are measured; the counterfactual scores
change only the stated ions:

1. **m/z 149.045 (C4H13O2Si2+) looks like a reference artifact.** It is ≥10% of base in 145 of 250 spectra
   and the base peak in 12%. The 176 records whose connectivity is in NIST23 have it at mean 30.5%
   of base (median 17%), while the NIST23 spectra of the same molecules have mean 3.5%. It is the base peak in
   the glucose, mannose, galactose, succinic and maleic acid TMS spectra. Removing that one measured peak raises
   the mean from 0.5726 to 0.5927.
2. **m/z 147 gets the wrong composition.** The measured ion is 147.0656 (C5H15OSi2+, the TMS-migration ion
   Me3Si–O=SiMe2+). It contains two Si and no backbone carbon, so it is not a connected substructure and
   the bond-cut candidates cannot generate it. The nominal-trained model instead puts that intensity at
   147.0836 (C6H15O2Si+), a reachable ion with the same nominal mass. That scores at 1 Da but misses at 10 ppm,
   and it explains part of the nominal-to-HR drop (0.638 to 0.573). Moving the predicted nominal-147 intensity
   to 147.0656 gives 0.5896; doing both 1 and 2 gives 0.6172.
3. **m/z 73 (Me3Si+) is over-allocated.** Pooled predicted share is 13.6% versus 7.0% measured; the
   mass is right, so this is intensity allocation.

The remaining gap to Exposome (0.706) also includes larger molecules (median molecular ion about
400, see the bins above). The 147 composition problem would need Si–O–Si migration candidates; the 149
artifact needs a reference-cleaning decision, not a model change. Diagnostic script:
`/tmp/claude-1002/-home-feiw-gitRepos-frag-gnn/f9050d06-e4a4-4b6e-92ba-b61bdf7696c3/scratchpad/metabolome_diag.py`
(session scratch; numbers above are reproduced inline).

### TMS vs non-TMS

A TMS group is matched by SMARTS `[O,N,S][Si]([CH3])([CH3])[CH3]`, counted per molecule. TMS molecules
make up 2.7% of e5 training (4,224 of 154,206 spectra; 2,031 with ≥2 TMS groups).

**NIST23 internal test (nominal, 1 Da).** Scored with
[analyze_masster_ei_checkpoint.py](../../../scripts/debug/ei/analyze_masster_ei_checkpoint.py), which now
has `--config` (e5 config, so e5's own test partition) and `--spec-ids`. The set is all 1,436 TMS test
spectra plus a seeded sample of 6,000 non-TMS test spectra; output is in `output/ei_tms_vs_non_tms_e5_2026_10_18/`.

| TMS groups | Spectra | Mean cosine | Mean Stein | Median exact mass |
| --- | ---: | ---: | ---: | ---: |
| 0 | 6,000 | 0.868 | 0.889 | 281 |
| 1 | 696 | 0.846 | 0.865 | 280 |
| 2 | 429 | 0.817 | 0.815 | 340 |
| 3 | 165 | 0.794 | 0.772 | 429 |
| ≥4 | 146 | 0.806 | 0.708 | 627 |

TMS spectra are heavier. Reweighting non-TMS spectra to the TMS mass distribution (six exact-mass bins)
gives a TMS penalty of −0.035 cosine [−0.046, −0.024] and −0.044 Stein [−0.055, −0.034] (500 bootstrap
resamples). The penalty grows with mass: within a bin, TMS minus non-TMS is −0.005 at 150–250, −0.028 at 250–350,
−0.051 at 350–450 and −0.093 at 450–600.

**HR-EI references (10 ppm, natural isotopes).** Every TMS record is in Metabolome, so source is confounded.
Within Metabolome, scores by TMS count 0/1/2/3/≥4 are 0.679/0.683/0.601/0.557/0.473 (n 12/43/71/52/72).
The nominal-control score minus the 10 ppm score is 0.009/0.035/0.073/0.056/0.091, so the exact-mass loss
grows once two TMS groups can form the Si2 ions. On the same model, ≥4-TMS spectra score 0.806 nominal on
NIST23 versus 0.564 nominal on RECETOX. Most of the Metabolome deficit is therefore specific to these
references and their exact masses, not to TMS chemistry alone.

### Measured HR-EI vs NIST23 nominal EI for the same molecules

[compare_nist_hr_ei_libraries.py](../../../scripts/debug/ei/compare_nist_hr_ei_libraries.py) matches each
HR record to NIST23 EI spectra with the same 2D connectivity: 860 of 962 records (321/363/176 by source).
e5 never trained on these NIST spectra. All cosines are nominal and linear, with HR spectra and exact
predictions rounded and summed. Everything is cut to the source window: 40–500 for HREI-MSDB, and ≥67 for
Exposome and ≥70 for Metabolome, which are the lowest peaks seen in any record of those sources. Means are
over the NIST spectra of each molecule. Output is in `output/hr_ei_vs_nist23_2026_10_18/` (`pairs.csv`, `summary.csv`).

| Mean nominal cosine (window) | HREI-MSDB | Exposome | Metabolome |
| --- | ---: | ---: | ---: |
| NIST23 replicate vs replicate | 0.955 (84) | 0.947 (59) | 0.947 (53) |
| HR library vs NIST23 | 0.904 | 0.908 | 0.759 |
| e5 vs NIST23 | **0.934** | 0.846 | **0.882** |
| e5 vs HR library | 0.862 | 0.778 | 0.676 |

Without the window, the HR-vs-NIST values are 0.884, 0.874 and 0.750. With nominal m/z 149 removed from every
Metabolome spectrum, HR vs NIST is 0.794, e5 vs NIST 0.882 and e5 vs HR 0.704.

- On HREI-MSDB and Metabolome, e5 predicts the NIST23 spectrum better than the measured HR library
  matches NIST23. The low HR scores on those sources mostly measure the difference between libraries.
- On Exposome, e5 vs NIST23 (0.846) is below library agreement (0.908), so that gap is model error.
- The difference between libraries is systematic. For matched pairs, the mean intensity share by m/z as a
  fraction of the molecular ion (<20%, 20–40%, 40–60%, 60–80%, 80–95%, ±5% of M) shows the HR spectra put
  less intensity below 40% of M and more at 60–95% than NIST23 does. For example, Metabolome HR is 0.062/0.276/0.316/0.185/0.079/0.082
  vs NIST23 0.120/0.356/0.271/0.133/0.053/0.066. e5 tracks NIST23 within 0.01 in every bin and every source.
  The direction is the same in all three HR sources, consistent with an instrument or tuning mass
  bias relative to NIST23. This was not tested further.

### Are the Orbitrap HR spectra softer than NIST23?

All three HR libraries are Orbitrap GC: Q Exactive GC for RECETOX, and Exploris GC for HREI-MSDB according to its
Figshare description. The test pairs each HR record with its first NIST23 spectrum, cut to the source window,
and uses Wilcoxon paired tests (n 321/363/176):

| Median, HR vs NIST23 | HREI-MSDB | Exposome | Metabolome |
| --- | --- | --- | --- |
| M+• share of total intensity | 0.020 vs 0.040 (p 0.66) | 0.014 vs 0.029 (p 9e−7) | 0.000 vs 0.002 (p 5e−3) |
| M+• relative to base peak | 0.103 vs 0.243 (p 3e−16) | 0.064 vs 0.148 (p 1e−22) | 0.000 vs 0.006 (p 8e−8) |
| Intensity-weighted m/z ÷ M | 0.630 vs 0.588 (p 3e−42) | 0.642 vs 0.601 (p 3e−24) | 0.521 vs 0.454 (p 1e−24) |
| Spectral entropy (nats) | 2.67 vs 2.85 | 2.93 vs 3.24 | 2.91 vs 3.13 |
| Peaks ≥1% of base | 30 vs 33 | 46 vs 51 | 45 vs 46 (p 0.60) |

The HR spectra are not softer in the usual sense, because the molecular ion is equal or weaker. Instead,
intensity moves from deep low-mass fragments to higher-mass fragments, and the spectra are more concentrated (lower
entropy, fewer small peaks). For peaks ≥2% of base in both spectra, a joint regression of log2(HR/NIST) intensity
on absolute m/z and on m/z ÷ M gives positive slopes for both. Relative position is stronger for HREI-MSDB
(+0.24 vs +0.10 log2 per SD) and Exposome (+0.14 vs +0.03); the two are equal for Metabolome (+0.11 each). So a pure
absolute-mass transmission bias does not explain the shift. One untested explanation: slow (metastable) M+• decay while ions
are stored in the C-trap would lower M+• and raise primary fragments. Low-mass transmission or thresholding during library
curation may add to it. Script: session scratch `softness.py` (numbers reproduced here).

### Chemical space: training vs references

[hr_ei_umap.py](../../../scripts/debug/ei/hr_ei_umap.py) fits one joint UMAP (Morgan r=2, 2048 bits,
Jaccard) on a seeded 40,000-molecule sample of the 149,848 unique e5 training molecules plus all
845 unique reference molecules (by source and 2D InChIKey). Outputs are in `output/hr_ei_umap_2026_10_18/`
(`umap_train_vs_hr_ei.png`, coordinates, `hr_mols.parquet` with nearest-training Tanimoto).

| Reference | Molecules | Median nearest-train Tanimoto | ≥0.7 | <0.4 | Spearman (Tanimoto, e5 cosine) |
| --- | ---: | ---: | ---: | ---: | ---: |
| HREI-MSDB | 278 | 0.667 | 42% | 6% | −0.02 (p 0.68) |
| Exposome | 350 | 0.640 | 30% | 9% | 0.31 (p 5e−9) |
| Metabolome | 217 | 0.706 | 52% | 3% | 0.03 (p 0.69) |

All three sets lie inside the training cloud. Metabolome is the *closest* to training, yet it scores lowest,
and its scores do not track similarity. That is consistent with the reference and composition causes above,
not with chemical novelty. Its worst molecules form one tight cluster: small poly-TMS acids, amino acids and
urea (succinic, fumaric, glutaric, itaconic, aspartic, urea; cosine 0.0–0.3 at Tanimoto 0.5–0.85). In those
spectra, TMS ions such as 73, 147 and 149 dominate the measured intensity. Glutamic acid_2TMS scores 0.00; glutamic acid is known to
cyclize to pyroglutamate during silylation, so its mapped structure may be wrong (not checked). Exposome is the only
source where score rises with similarity (0.50 below 0.4, 0.78 at 0.7–0.999).

Some molecules share an identical fingerprint with a training molecule (9% HREI-MSDB, 6% Metabolome)
despite different 2D InChIKeys. The split excludes by connectivity key, so these are fingerprint
collisions between different connectivities (likely positional isomers; not inspected).

## Interpretation

- The e1-to-e5 drop on HREI-MSDB and Exposome is confined to molecules e1 trained on, so it is
  memorization. On molecules neither model saw, the two models agree within noise.
- The e5 numbers are the first held-out HR-EI result: at 10 ppm with natural isotopes, HREI-MSDB 0.850,
  Exposome 0.706, and Metabolome 0.573.
- Metabolome drops on non-overlap records too. Untested explanation: the e1 overlap flag uses
  the measured (derivatized) structures. The e5 exclusion also removed native parent structures
  from the RECETOX SDFs, so e1 may have trained on parents of "non-overlap" derivatives. One seed per model;
  run-to-run variance at this n is unmeasured.
- Exposome loses 0.075 from 10 to 5 ppm in e5 too, which is why 10 ppm is the headline. It is a property of
  the reference masses or matching floor, not of the checkpoint.

## Outputs and reproduction

`output/hr_ei_e5_eugy8s9n_eval_2026_10_18/{natural_exact,monoisotopic}/`, with the same file
layout as the e1 evaluation (`scores.csv`, `metric_summary.csv`, `provenance.json`, plots,
`cache/exact_predictions.npz`). Runtime on GPU: 57 s natural, 22 s monoisotopic.

```bash
FRAGNNET_JOBLIB_N_JOBS=1 MPLCONFIGDIR=/tmp/fragnnet_mpl \
  /home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python -m scripts.debug.ei.evaluate_hr_ei_checkpoint \
  --checkpoint saved_ckpts/e5_hr_ei_holdout_nominal_eugy8s9n.ckpt --isotope-envelope natural \
  --output <fresh dir>
```

The plot script's figure title was hardcoded to "Frozen e1 checkpoint". It now uses the checkpoint
file name from `provenance.json`; e1 figures generated earlier keep the old title.

## Next

Test the Metabolome explanation by recomputing overlap against native parents. Add an e5 seed before
treating the Metabolome non-overlap drop as real. Castro still needs structure mapping before it can be scored.
