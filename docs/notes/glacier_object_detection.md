# GLACIER (Wang, Wang & Coley, MIT — arXiv 2606.29161)

> **Status:** NOTE &nbsp;·&nbsp; **Updated:** 2026-10-02
>
> Literature notes on "GLACIER: Rethinking Mass Spectrum Prediction as an Object Detection
> Problem". Read the **bin-width caveat** before quoting any of its NIST'20 numbers.

- arXiv: [2606.29161](https://arxiv.org/abs/2606.29161) · code in
  [`coleygroup/ms-pred`](https://github.com/coleygroup/ms-pred) (`src/ms_pred/glacier/`)
- Local copies (gitignored): `data/paper/GLACIER_arxiv_2606.29161.pdf` and the extracted `.txt`
  beside it. Line numbers below refer to the `.txt`.
- Companion note: [`aime_deepms2reasoner.md`](aime_deepms2reasoner.md). Same shape of protocol
  problem in a different place — GLACIER scores at a **bin width 100x coarser than the
  instrument**, AIMe cleans the **target spectra**. Neither set of numbers can sit beside ours
  without redoing the protocol.

## Bin-width caveat — READ FIRST

> "Binned spectral vectors are used with **0.1 resolution on the NIST'20 dataset** and **0.01
> resolution on the MassSpecGym dataset** to align with prior work on both benchmarks."
> (Section 4.1, line 445)

Every NIST'20 number in Tables 15-17 — cosine 0.838, entropy 0.781, coverage 0.887 — is at
**0.1 Da on Orbitrap data**, i.e. ~333 ppm at m/z 300. A single 0.1 Da bin merges CO (27.9949),
N2 (28.0062) and C2H4 (28.0313), so the standard neutral-loss ambiguities collapse and
fragment-identity error is invisible to the metric. The paper itself flags the hazard when it
belongs to someone else: MolSpecFlow "uses a different bin width (0.1 Dalton) instead of the bin
width adopted by MassSpecGym (0.01 Dalton); the comparison has to be made with extra care."

**The choice lands on the flattering side of each metric.** Their own Appendix F (Table 14,
MassSpecGym top-1 retrieval):

| train bin | test @ 0.1 Da | test @ 0.01 Da |
|---|---|---|
| 0.1 | 0.664 mass / 0.479 formula | **0.685 / 0.490** |
| 0.01 | 0.652 / 0.471 | **0.697 / 0.485** |

Retrieval is *better* at 0.01 Da (finer bins discriminate decoys). So NIST'20 reports **cosine +
coverage at 0.1 Da** while MassSpecGym reports **retrieval at 0.01 Da**.

**Measured 2026-09-20 (see "MEASURED" below): 0.1 Da is a real but SMALL relaxation on their
data.** Controlled sweep on our FraGNNet (identical peaks, identical Hungarian matching, only the
window changes): 10 ppm -> a 0.1 Da window is **+0.029 overall**, but only **+0.008 on FT/`[M+H]+`**
and **+0.38 on QTOF**. GLACIER's NIST'20 is HCD Orbitrap, so the bin width is worth roughly +0.008
of their 0.838 — not nothing, not the story. It stays small because a fragment model predicts
exact formula masses and Orbitrap targets are accurate to well under 10 ppm, so a wider window
finds few new matches. The stronger objection to 0.1 Da is **blindness to formula identity**
(CO / N2 / C2H4 share a bin), and that they cannot check it at an instrument tolerance because
ms-pred has no ppm anywhere in its loss or evaluator.

## NIST'20 results (Appendix G, Table 15, `[M+H]+`, 0.1 Da)

| Method | random split cosine | scaffold split cosine | coverage (random) |
|---|---|---|---|
| CFM-ID | 0.470 | 0.450 | 0.261 |
| GrAFF-MS | 0.565 | 0.470 | 0.740 |
| MassFormer | 0.632 | 0.526 | 0.787 |
| **FraGNNet** | **0.717** | **0.654** | - |
| ICEBERG 1.0 | 0.754 | 0.701 | 0.809 |
| ICEBERG 2.0 | 0.794 | 0.735 | 0.856 |
| GLACIER w/o CF | 0.838 | 0.788 | 0.887 |
| GLACIER w/ CF | 0.815 | 0.763 | 0.858 |

- **Our number in that table is not our protocol.** "Baseline results are adapted from Wang et al.
  [29]" (the ICEBERG 2.0 paper) — a third-party re-evaluation at 0.1 Da, with entropy and coverage
  left blank. Our own reported cosines are 0.01 Da binned (`val_cos_sim_0.01`) or
  instrument-tolerance Hungarian (`cos_hun`). Do not place our numbers next to this column.
- Contrastive finetuning *lowers* spectral similarity (0.838 -> 0.815) while raising retrieval —
  consistent with CF trading calibration for discriminability. But see the next section: the
  same pattern also fits a spectrum-independent structure prior.

## Contrastive finetuning as a §5.2 shortcut, not §4 leakage — ARGUMENT, UNMEASURED (2026-10-02)

> **Measured on MassSpecGym 2026-10-03, NULL there:** the released CF checkpoint shows no structure
> prior beyond MASSter under a wrong spectrum ([swap test](../findings/masster/glacier_cf_swap_test_2026_10_03.md)).
> The NIST'20 random-split case below remains an argument.

The "MassSpecGym in the Wild" audit ([`massspecgym_in_the_wild_audit.md`](massspecgym_in_the_wild_audit.md))
gives the framing. **Do not call GLACIER's CF "data leakage".** In the audit's §4 sense, leakage
means test structures or close analogs in training. GLACIER answers that charge in Appendix E,
Table 12: removing every retrieval-candidate structure from the CF negatives leaves NIST'20 random
top-1 unchanged (0.510 vs 0.505).

**The defensible claim is the §5.2 shortcut, which Table 12 does not test.**

- CF (Appendix E, line 893): the positive is always a *measured* molecule. Negatives are 50%
  same-formula PubChem structures and 50% single-step graph mutations. Retrieval candidates at
  test time are also PubChem structures.
- The audit's §5.2 shows that measured molecules and PubChem decoys come from different
  distributions. On MassSpecGym, PubChem's default ordering alone reaches 49.98% HR@1.
- CF trains the model to separate measured-like structures from PubChem-like or mutated ones.
  A forward model can do this without the spectrum: it predicts diffuse, mismatched spectra for
  off-distribution structures, so their entropy distance is large. Filtering exact candidates
  cannot rule this out, because the prior acts on the decoy *distribution*.
- GLACIER notes the risk in the main text (line 626, citing [11]). It dismisses it only on the
  basis of Table 12.

**GLACIER's own numbers fit this reading** (Tables 1, 2 and 4):

| | w/o CF | w/ CF | gain |
|---|---:|---:|---:|
| NIST'20 random top-1 | 0.390 | 0.525 | **+0.135** |
| NIST'20 scaffold top-1 | 0.384 | 0.460 | +0.076 |
| MSG formula top-1 | 48.49 | 49.88 | +1.4 pt |
| MSG mass top-1 | 69.69 | 69.95 | +0.3 pt |
| NIST'20 random cosine | 0.838 | 0.802 | -0.036 |

The gain shrinks as the split gets harder (random > scaffold > MSG's MCES split), and spectral
accuracy drops. GLACIER's own explanation (line 917) is that CF "is more effective for learning
on in-distribution data". On a random split, test truths have close training analogs, so that
explanation amounts to a learned "looks like the training molecules" prior. It is not evidence
of better spectrum reasoning. The harsher CF settings (Tables 8-9, lower delta) widen the random-split
gain and barely move the scaffold split, which is the same pattern again.

**Scope.** This argument only affects the CF rows. GLACIER without CF still leads every retrieval
table. Use the argument to justify comparing against the w/o-CF rows, not to dispute GLACIER's
SOTA claim. It also applies to ICEBERG 2.0, whose CF GLACIER adopted. ICEBERG 2.0's baseline rows
in GLACIER's Table 2 are with CF.

**The test that turns this into a finding** (the audit's §5.2 recommendation): score each test
query against its candidate set using the *wrong* spectrum. Swap in a spectrum from a different
test entry within the same precursor-mass window, then compare HR@1 for the CF and w/o-CF models.
A CF advantage that survives the swap is a structure prior. Under our model the question is the
same: what does a fully spectrum-blind ranker (PubChem order, the chiral-atom count, a structure-only
classifier) reach on the NIST'20 candidate sets? **Blocker (revised 2026-10-03):** the only released GLACIER weights
are the MassSpecGym **w/ CF** model
([released-ckpt finding](../findings/masster/masster_msgym_vs_glacier_released_ckpt_2026_09_24.md)).
The swap test can therefore run on the released CF model as is. The w/o-CF comparison arm, and
anything on NIST'20, still means retraining (`configs/glacier/joint_train_msg.yaml`, then
`joint_contr_finetune_*.yaml`).
Run on MassSpecGym 2026-10-03 (NULL, see the banner above). A NIST'20 checkpoint would need
ms-pred's MAGMa labelling pipeline run on NIST'20 first (judged not worth it, user, 2026-10-03),
so the NIST'20 case stays an argument.

## Coverage is top-100, NOT top-20 — RESOLVED 2026-09-20

**`run_scripts/glacier/02_predict_inten.py` passes `--max-peaks 100`.** Verbatim:

```
python analysis/spec_pred_eval.py \
--binned-pred-file {out_binned} \
--max-peaks 100 \
--min-inten 0 \
--formula-dir-name no_subform.hdf5 \
--dataset {dataset}
```

Every model in the lineage passes 100 (ICEBERG, SCARF, MassFormer, GrAFF-MS, GNN, FFN, MolNetMS);
CFM-ID passes 1000. **The argparse default of 20 is never used by any run script** — so the
earlier "top-20 coverage" reading of their 0.887 is WRONG, and any of our numbers computed under
default-20 behaviour are at the wrong protocol (this includes the protocol-sensitivity table in
[`dag_oracle_ceiling_nist20.md`](../findings/fragnnet/dag_oracle_ceiling_nist20.md), which should be
recomputed at 100).

`--max-peaks` truncates the **prediction** to its N strongest bins before cosine and before the
coverage intersection. At 100 this is nearly inert for us: our predictions average 81 bins per
spectrum at 0.1 Da.

### Their target is NOT subformula-filtered — RETRACTED 2026-09-20

An earlier version of this note claimed their ground truth is formula-assigned at theoretical
masses, "the same class of target cleaning as AIMe's". **That is wrong.** The eval path uses
`--formula-dir-name no_subform`, built by `01_assign_subformulae.py --use-all`, where

    "mz": spec_masses, "formula_mass_no_adduct": spec_masses - adduct_masses, "formula": [""], "mass_diff": [0]

i.e. **observed m/z minus the adduct mass**, with nothing dropped for lacking a subformula. The
theoretical-mass, subformula-assigned directory (`magma_subform_50`) is the *training-label*
folder, not the evaluation target.

The one real target filter is `max_inten_spec`: **top-50 peaks above 0.001 relative intensity**
(argparse defaults `--max-formulae 50`, `--inten-thresh 0.001`, confirmed upstream). That is an
intensity/count truncation, **not** a chemistry filter, so it is a different and much milder thing
than AIMe's 5 ppm subformula filter. Do not equate them.

## Their PREDICTION is unbinned — 0.1 Da is a readout choice, not a capability limit

Appendix D. The intensity head emits an **unbinned** value per (fragment, hydrogen shift):

    y(i)_delta = sigma( MLP_inten(H_i)_delta )                      (15)
    y_hat_m    = sum_i sum_delta y(i)_delta * 1[M(i,delta) = m]     (16)

where `M(i, delta)` is the **exact m/z** of fragment i at hydrogen shift delta. Binning is applied
after the model. They redesigned this away from ICEBERG deliberately — ICEBERG's attention softmax
was taken over predictions sharing a bin, which "will lead to inaccurate results if we want to
adapt the predicted fragment intensity into different bin-width setting" — and state the purpose
of eq. 16 outright: *"This formulation enables the model to adapt to different mass bin widths at
inference time, even when trained using a fixed binning resolution."*

**Consequences.** Their trained NIST'20 model could be re-read at 0.01 Da, or at exact masses,
**with no retraining**; Table 14 is exactly that experiment on MassSpecGym. So reporting NIST'20
at 0.1 Da is a comparability choice ("to align with prior work"), not a limitation of the model.

What the 0.1 Da *training loss* costs is narrower than it first appears: eq. 16 sums all fragments
in a bin, so the loss supervises only their **sum** — the split between CO / N2 / C2H4 inside one
bin receives no gradient. But Table 14 says that costs little: **test** bin width matters (0.01
beats 0.1 in both challenges) while **train** bin width is a wash (mass challenge favours
train@0.01 by +0.012, formula challenge favours train@0.1 by +0.005 — opposite directions,
noise-level). That is itself evidence that fragment masses are exact by construction and only the
readout grid matters.

So the standing criticism is not "they cannot evaluate at instrument tolerance" in a modelling
sense — they have a model emitting exact masses and an evaluator that only accepts uniform grids.
The missing piece is ppm-tolerance matching in `spec_pred_eval.py`, not a different architecture.

## The retrieval protocol contradicts the 0.1 Da scoring

**Candidates are selected at high mass precision, then discriminated at 0.1 Da.** MassSpecGym's
formula challenge draws candidates with the *same molecular formula* — exact isomers — so the only
signal that can separate them is which fragments form and at what intensity. Binning fragments at
0.1 Da merges CO (27.9949) / N2 (28.0062) / C2H4 (28.0313), i.e. exactly that signal. Their own
Appendix F Table 14 pays the price: retrieval is better at 0.01 Da in all four cells
(0.685 vs 0.664 mass, 0.490 vs 0.479 formula).

**The "10 ppm PubChem window" (AIMe, following FraGNNet) is not a difficulty control.** Our own
implementation is `preproc_scripts/ms2c_pubchem/02_prepare_ms2c_candidates.py`: query PubChem by
exact-mass range (`mass_tol = mw * tolerance * PPM`, default 10 ppm), then `filter_candidates`
keeps `top_k` by **Morgan-Tanimoto to the true structure**, after dedup by canonical SMILES and
2D InChIKey and dropping multi-component SMILES, radicals, oversized molecules and out-of-set
elements. In the dense 200-500 Da region a +/-0.003 Da window returns thousands of PubChem
compounds (one drug-like formula alone has tens of thousands of isomers), so the window width
barely affects the final 49 — the Tanimoto ranking does. **The benchmark is really "the 49 nearest
neighbours of the answer", not "10 ppm".**

Selection by precursor m/z rather than formula also makes the set *easier*: it admits
near-isobaric non-isomers whose fragment formulas differ systematically and are easy to reject.
AIMe flag this themselves. The harder and more meaningful variant is formula-based selection
(MassSpecGym's formula challenge; AIMe's CASMI all-PubChem-isomer setting, ~2.5M candidate
spectra, 35.6% top-1).

Not measured here: the actual occupancy of a 10 ppm PubChem window — no PubChem sqlite exists in
this checkout (`data/ms2c/pubchem/` holds only `predicted/`), so the "thousands" figure above is
an estimate from formula-isomer counts, not a measurement.

## Dataset scale and m/z limit

**They do not have more data than us — the difference is merged vs unmerged.**

| | spectra | molecules | spectra/mol |
|---|---:|---:|---:|
| GLACIER NIST'20 | 530,640 | 25,541 | 20.8 |
| ours `nist20_ma_mi_inchikey` | 480,002 | 25,859 | 18.6 |
| ours, merged | ~26,000 | 25,859 | 1.01 |

Molecule counts are within 1.2%. The 20x impression comes from comparing their unmerged count
against our merged setup (1.01 rows/mol, see [merged-vs-unmerged root cause](../findings/masster/masster_unmerged_overfitting_2026_09_16.md)).
The residual 10% is composition: theirs is all HCD in both polarities, ours spans FT/QTOF/IT over
4 adducts. Against the older ms-pred NIST20 subset (35,129 spectra in the checked-in
`split_1.tsv`, the ICEBERG 1.0 / SCARF era) GLACIER is ~15x larger — a change in extraction, not
a new data source.

**m/z limit is 1500 Da**, the same as ours. Not stated in the paper; from
`coleygroup/ms-pred@main` (verified upstream, not just the old local checkout):
`bin_spectra(spectras, num_bins=15000, upper_limit=1500, pool_fn="max")`, with the same defaults
in `MassSpec.ensure_binned_spectrum`, `_bin_spec_sparse` and `PredSpecDB.bin_spectrum`.
15,000 bins over [0, 1500) **is** the 0.1 Da resolution. Ours: `--max_peak_mz 1500.0`,
`--max_prec_mz 1500.0` (live in `data/split/nist20_ma_mi_inchikey/meta.json`).

Note the ceiling and the resolution are one constant pair: at a fixed 1500 Da ceiling, 0.01 Da
means 150,000 bins per spectrum instead of 15,000, a 10x blow-up across 530,640 spectra. That is
a plausible compute reason for leaving NIST'20 at 0.1 Da while MassSpecGym (231,104 spectra) got
0.01 — **inference, not stated in the paper**. It does not change that the reported cosine
benefits from the coarser bin.

## MEASURED: our FraGNNet under their binning (2026-09-20)

Scored `output/inference_eval/{alvhehe2,pqx7ywc9}/predictions_test.hdf5` (96,128-spectrum NIST20
test split) against raw `data/proc/nist20/spec_df.pkl` peaks, replicating `ms_pred.common.bin_spectra`
exactly: `np.linspace(0, 1500, num_bins)`, `np.digitize`, drop bins >= len(bins), max-pool per bin,
their `cos_sim_fn`. Script: `scratchpad/bin_eval2.py`.

| metric | alvhehe2 | pqx7ywc9 |
|---|---:|---:|
| stored `cos_hun_inst_tol` (10 ppm FT / 20 ppm QTOF, Hungarian) | 0.7521 | 0.7555 |
| binned cosine @ 0.1 Da | **0.7525** | **0.7579** |
| binned cosine @ 0.1 Da, top-20 predicted bins | 0.7488 | 0.7544 |
| binned cosine @ 0.01 Da | 0.6969 | 0.7016 |
| FT / `[M+H]+` slice (n=57,172) @ 0.1 Da | 0.7515 | 0.7577 |

### Controlled tolerance sweep — the number that actually answers "is 0.1 Da easier than 10 ppm?"

The table above is NOT a tolerance comparison: `cos_hun_inst_tol` uses **per-instrument**
tolerance (FT 10 ppm/200 Da, QTOF 20 ppm/750 Da) on a `rel_filter_spec`-filtered target with
Hungarian one-to-one matching, while the binned rows use raw peaks with max-pooling. Three
differences at once, which happened to cancel — do not read 0.7521 vs 0.7525 as "10 ppm approx
0.1 Da".

Controlled version (`scratchpad/tol_sweep.py`, `pqx7ywc9`, n=20,000 sampled): identical peaks,
identical Hungarian matching mirroring `cos_hun_helper`, **only the m/z window changes**:

| matching window | all | FT / `[M+H]+` | QTOF |
|---|---:|---:|---:|
| 10 ppm (200 Da floor) | 0.7286 | 0.7447 | 0.4383 |
| +/-0.005 Da (= 0.01 Da bin) | 0.7415 | 0.7449 | 0.6528 |
| +/-0.05 Da (= 0.1 Da bin) | **0.7574** | 0.7520 | 0.8202 |
| +/-0.1 Da | 0.7578 | 0.7524 | 0.8208 |

**Monotonic, as it must be — a wider window cannot lose matches. 0.1 Da is easier than 10 ppm by
+0.029 overall.** But the gain is almost entirely QTOF (+0.38, because 10 ppm is tighter than
that instrument's real accuracy); on **FT it is only +0.008**. GLACIER's NIST'20 is HCD Orbitrap,
so their bin width is worth about +0.008, not the headline. Reference: stored
`cos_hun_inst_tol` on the same spectra is 0.7560 vs 0.7286 for flat 10 ppm — that +0.027 is the
QTOF relief already built into our production instrument-tolerance table.

**Hard binning is harsher than tolerance matching at the same width.** +/-0.005 Da tolerance gives
0.7415 but 0.01 Da hard bins give 0.6969 — 0.045 lost purely to grid boundaries splitting matched
pairs. So our 0.01 Da binned row understates high-resolution performance; `cos_hun_inst_tol` is
our real high-resolution score.

What the coarse grid does instead is **hide formula identity**: CO (27.9949), N2 (28.0062) and
C2H4 (28.0313) share one 0.1 Da bin, so a wrong isobaric assignment is unpunished. That is why
their Appendix F Table 14 shows *retrieval* improving at 0.01 Da while cosine prefers the coarse
grid.

**They cannot check it at a real instrument tolerance.** `ppm` appears 25 times in the whole
ms-pred source and all 25 are in `src/ms_pred/magma/run_magma.py`, the MAGMa fragment-*labeling*
tool. Zero in `analysis/`, zero in the loss. Everything scored goes through
`bin_spectra`/`digitize_ar`, which accept only `(num_bins, upper_limit)`. A uniform grid over
[0, 1500) equals a given ppm at exactly one m/z:

| bin width | @ m/z 100 | @ m/z 300 | @ m/z 1500 | bins |
|---|---:|---:|---:|---:|
| 0.1 Da (their NIST'20) | 1000 ppm | 333 ppm | 67 ppm | 15,000 |
| 0.01 Da (their MSG) | 100 ppm | 33 ppm | 6.7 ppm | 150,000 |
| 0.001 Da (10 ppm at m/z 100) | 10 ppm | 3.3 ppm | 0.67 ppm | 1,500,000 |

Two further protocol notes measured here:

- **0.01 Da HARD binning is not "high resolution" done right.** It costs us 0.056 overall, but the
  damage is concentrated in QTOF (`[M+H]+` 0.8157 -> 0.4308, `[M+Na]+` 0.7613 -> 0.3587) where
  instrument mass error exceeds the bin; FT loses only 0.034. A fixed grid splits a matched pair
  across adjacent bins whenever a peak sits near a boundary, which tolerance matching forgives.
  Do not quote our 0.01 Da row as our high-resolution score — `cos_hun_inst_tol` is that.
- **`--max-peaks 20` truncates the PREDICTION, not just coverage.** `spec_pred_eval.py` does
  `pos_bins = pos_bins_sorted[:max_peaks]` and then scores that truncated vector. Measured on our
  predictions it *lowers* cosine by ~0.004 (our tail carries matched mass), so it is not a
  free inflation knob either way.
- **Their target is formula-assigned too.** `process_spec_file` builds the true spectrum from
  `subformulae/<formula_dir_name>/<spec>.json` using `output_tbl["formula_mass_no_adduct"]` —
  theoretical neutral formula masses, not observed m/z, and only peaks that received a subformula.
  Same class of target cleaning as AIMe's 5 ppm filter (see `aime_deepms2reasoner.md`).

### Fully protocol-matched number (2026-09-20)

`scratchpad/protocol_matched.py`, pqx7ywc9, full 96,128-spectrum test split, 0.1 Da bins, with
GLACIER's actual run-script settings (`--max-peaks 100`, `--min-inten 0`, `no_subform` target
truncated by `max_inten_spec` to top-50 peaks above 0.001 relative intensity):

| variant | ALL | FT `[M+H]+` |
|---|---:|---:|
| raw target, all predicted bins | 0.7579 | 0.7577 |
| raw target, top-100 predicted | 0.7577 | 0.7576 |
| top-50/0.1% target, all predicted | 0.7570 | 0.7568 |
| **both (fully matched)** | **0.7569** | **0.7566** |

**The protocol knobs are inert for us** — top-100 truncation costs 0.0002 (our predictions exceed
100 bins in only 26.7% of spectra; mean 80.4) and their target filter costs 0.0008. Use **0.757**
as our protocol-matched FraGNNet number.

### Prediction pooling: max vs sum (2026-09-20)

`bin_spectra` defaults to `pool_fn="max"`, but GLACIER's own intensity head **sums** fragment
contributions into a bin (Appendix D). Max-pooling therefore penalises a high-resolution model
that splits mass across isobaric candidates. Measured (`scratchpad/pool_test.py`, pqx7ywc9, full
test split):

| | ALL | FT `[M+H]+` |
|---|---:|---:|
| max-pooled predictions | 0.7579 | 0.7577 |
| **sum-pooled predictions** | **0.7594** | **0.7599** |
| gain | +0.0015 | +0.0022 |

33.8% of our predicted peaks share a 0.1 Da bin (121.5 peaks -> 80.4 bins), but max-pooling
discards only **5.7% of predicted intensity** — the colliding peaks are mostly small (isotopes,
minor H-shifts). Real handicap, worth +0.002.

### Every protocol knob is worth thousandths — the gap is NOT a scoring artifact

| knob | worth to us |
|---|---:|
| bin width, 10 ppm -> 0.1 Da (FT cosine) | +0.006 |
| sum- vs max-pooled predictions | +0.002 |
| their top-50/0.1% target filter | +0.0008 |
| `--max-peaks 100` truncation | +0.0002 |

Total well under 0.01 against a ~0.079 gap to GLACIER's 0.838. **Do not explain the gap by
protocol.** The "they trained at 0.1 Da" advantage is unmeasurable without training, but it is
bounded by the same 5.7% isobaric collision mass — optimising bins directly mainly buys freedom
to be wrong about which isobar, and little intensity rides on that.

### Against their Table 15

Protocol-matched, our FraGNNet (pqx7ywc9) on the FT/`[M+H]+` slice scores **0.757**, versus the
**0.717** their table attributes to FraGNNet, ICEBERG 1.0 at 0.754, ICEBERG 2.0 at 0.794 and
GLACIER at 0.838 — so we sit ~0.04 above their FraGNNet row and ~0.081 below GLACIER.
**Not a clean head-to-head** — different test split (our 60/20/20 vs their 80/10/10, so our model
saw 60% of the data), and our slice is our own test molecules. But our FraGNNet measured under
their own metric lands above the number published for it, and still clearly below GLACIER.

## Recall vs cosine: the coarse bin inflates COVERAGE, not cosine (2026-09-20)

`scratchpad/tol_recall.py`, pqx7ywc9, n=8,000 sampled, same Hungarian matching, only the window
changes. Intensity recall = matched TRUE intensity / total true intensity; peak recall = matched
true peaks / true peaks.

**A 0.1 Da BIN is not a +/-0.05 Da tolerance.** For random phase two peaks `d` apart share a bin
with probability `1 - d/w`, so a grid misses near pairs straddling an edge and accepts far pairs
sharing a bin. Our predicted masses are exact formula masses, so `d` is just the target's
measurement error (~1-3 ppm on FT = 0.0005-0.001 Da at m/z 300): predicted boundary loss ~1% at
w=0.1 Da but 5-10% at w=0.01 Da. Both confirmed below. Measured with the REAL grid
(`scratchpad/bin_recall.py`), support = does any predicted peak share the bin / fall in the window:

**FT (n=7,515), 32.2 true peaks and 120.8 predicted peaks per spectrum (0.50 predicted peaks/Da):**

| protocol | intensity recall | peak recall | cos_hun |
|---|---:|---:|---:|
| 10 ppm tolerance | 0.8895 | 0.6891 | 0.7470 |
| 0.01 Da bin | 0.8557 | 0.6770 | 0.6969 |
| **0.1 Da bin** | **0.9337** | **0.7873** | 0.7577 |
| +/-0.05 Da tolerance | 0.9412 | 0.7995 | 0.7534 |

**A 0.1 Da bin buys +9.8 points of peak recall and +4.4 of intensity recall on FT — but only
~+0.6 of cosine.** Note the **0.01 Da grid scores BELOW 10 ppm tolerance** on FT (0.8557 vs
0.8895) despite being nominally finer: that is entirely grid phase, and it is why our 0.01 Da
binned rows understate us. Recall counts a true peak as recovered as soon as *any* predicted peak falls in
the window; cosine credits the **product** of the two normalised intensities. On FT the extra
matches are spurious neighbours (a wrong fragment within 0.05 Da carrying near-zero predicted
intensity), so support rises and intensity agreement does not.

**QTOF (n=485) is the contrast** — there a wider window recovers *correct* matches that instrument
error had displaced, so recall and cosine move together:

| protocol | intensity recall | peak recall | cos_hun |
|---|---:|---:|---:|
| 10 ppm tolerance | 0.4823 | 0.3872 | 0.4383 |
| 0.01 Da bin | 0.5459 | 0.4720 | - |
| 0.1 Da bin | 0.9481 | 0.8793 | - |
| +/-0.05 Da tolerance | 0.9670 | 0.8967 | 0.8202 |

### Consequence for reading GLACIER's table

**It is the coverage column (0.887) that the 0.1 Da bin inflates, not the cosine (0.838).** Our FT
peak recall gains **+0.098** going from 10 ppm tolerance to a real 0.1 Da grid, and the d3 DAG
oracle gains 0.166 between 0.01 and 0.1 Da bins (`dag_oracle_ceiling_nist20.md`) — both support
metrics. So at an instrument tolerance their coverage would fall by roughly a tenth while their
cosine would move ~0.006. Caveat: our peak recall is any-match support; their coverage is
top-20 ∩ top-20, so the magnitudes are indicative, not a like-for-like restatement.
This also explains why the oracle-coverage jump (0.7674 -> 0.9334) and the small cosine effect are
both true: they are measuring different things.

## Feature setup (Appendix A, Tables 6-7; Appendix C)

Graphormer over the heavy-atom graph.

- **Atom**: type one-hot + "other"; **degree one-hot (excl. H) + overflow**; formal charge one-hot
  + overflow; hybridization one-hot + other; aromaticity; H-count one-hot + overflow; ring
  membership; **ring-size multi-hot**; **fused-ring**; chirality one-hot; optional element-group
  embedding; **Gasteiger charge**.
- **Bond**: type one-hot + other; conjugation; ring membership; **ring-size multi-hot**;
  **fused-ring**; stereo one-hot + other.
- **Missing CE** (added 2026-09-24): a NaN collision energy is replaced by a fixed all-zero CE
  embedding (`collision_embed_merged`, `requires_grad = False`, in `molecular_embedding`), with
  no imputation. `configs/glacier/joint_train_msg.yaml` trains on the base `msg` dataset, not
  the `msg_simulation` subset.
- **Metadata**: adduct one-hot, collision energy sinusoidally encoded, and an instrument one-hot
  (`embed-instrument: True` in the NIST'20 config). Section 3.1 reads as if metadata reaches
  "node embeddings and edge-level attention biases", but the code (`molecular_embedding`,
  `318bb7b`) concatenates it onto the **atom inputs only**; the attention bias is built from
  bond features and shortest-path distance alone. Corrected 2026-09-23.
- RDKit features are computed **once at molecule level and reused across fragments** — pitched
  explicitly as an advantage over ICEBERG's per-fragment embeddings.

**Intensity head inputs (Appendix C, eq. 14)** — the part that matters most for us:

```
t_k = MLP( G || psi(f_k) || psi(r_form - f_k) || OHE(n_broken_k) )
```

fragment formula, the **neutral-loss complement** (root formula minus fragment formula), and the
**broken-bond count**, with `psi` a sinusoidal formula embedder and `G` the root-molecule token.
That is exactly the two design gaps in our 2026-09-20 feature audit: our candidate intensity head
is mass- and composition-blind while `ion_mzs` is already in scope, and the cut bonds reach
intensity scoring only through the GNN.

### How the embeddings are built, read from code (`318bb7b`, 2026-09-23)

Source: `src/ms_pred/glacier/{joint_model,dataset}.py`, `src/ms_pred/graphormer/graphormer_layers.py`
and `src/ms_pred/nn_utils/form_embedder.py` at `samgoldman97/ms-pred` `318bb7b`, with the NIST'20
config. Widths were checked by instantiating `JointModel`.

1. **Atoms: computed once per molecule, and never specific to a fragment.** Each atom has 88 input
   features: element (18 + other), degree, formal charge, hybridization, aromaticity, H count,
   ring membership, ring sizes 3-8, fused ring, chirality, Gasteiger charge, element group (8) and
   a 14-step random-walk positional encoding. The metadata is concatenated onto every atom: adduct
   (one-hot + element-group multi-hot), CE (64-dim sinusoid; zeros when CE is missing) and
   instrument one-hot. That gives 178 columns, projected by `Linear(178, 256)` and added to a
   degree embedding.
2. **Bonds never become node inputs.** The 28 bond features pass through `Linear(28, 8)`, one scalar
   per attention head, and are averaged along the shortest path (at most 10 hops) into a per-head
   attention bias, plus a shortest-path-distance bias. Bond chemistry therefore only decides how
   strongly atom i attends to atom j. Five full-attention layers follow, plus a graph token `G`.
3. **The fragment token contains no atoms and no cut site.**
   `t_k = Linear(G ‖ psi(f_k) ‖ psi(r - f_k) ‖ onehot(n_broken, 0..6))`, i.e. 256 + 144 + 144 + 7 =
   551 → 256. `psi` is a **fixed** abs-sine code, 8 dims per element count over 18 elements (H
   included). It is close to a binary code of the exact integer, so "exactly one O and two H lost"
   is a distinct pattern. Two fragments of one molecule with the same formula and cut count start
   from the same token. The neutral-loss term `psi(r - f_k)` is the same vector in every molecule.
4. **Atoms enter only through attention restricted to the fragment's own atoms.** The intensity
   decoder has 4 layers. Each token cross-attends to the molecule's final atom embeddings, masked
   to its own atoms, and self-attends across all of the molecule's fragments in the same layers.
   Three full self-attention layers over the fragments follow. The atom embeddings do not depend
   on the fragment, so nothing marks which atoms sit at a cut. The cut site is visible only as
   `n_broken` and implicitly through which atoms are included. The pooling is learned and depends
   on the query, which carries the formula and `n_broken`.
5. **Output.** 26 sigmoids (±6 H × CRF/CMF) are masked to the shifts the fragment's own H count and
   the molecule's remaining H can supply (capped at 6), and placed at exact masses.

Versus the MASSter token (`c4a_base`):

| | GLACIER | MASSter |
|---|---|---|
| atoms | learned attention over its own atoms | mean pools of the fragment's atoms and of the complement's atoms |
| cut site | `n_broken` only | **boundary block**: `CutBondEncoder` MLP on `[in, out, in·out]` of the cut bonds' endpoint embeddings; plus `c_cut_count` |
| fragment formula | `psi(f_k)`, exact counts | none; `c_mass` (log1p mass, precursor fraction) |
| neutral loss | `psi(r - f_k)`, exact counts | none by default; `c_nl_composition` (log1p counts) built; `c4a_complement_composition` ran, null (+0.0012) |
| molecule | learned graph token `G` (carries the metadata) | metadata-only context vector |

GLACIER's token is **formula-first**. The parts that transfer across molecules (formula, neutral
loss, cut count) form the query, and molecule-specific atom detail enters only through attention.
Ours is **embedding-first**: three pooled 384-dim embeddings, one of them a sharp cut-site
embedding, and no formula. That cut-site embedding is where the
memorization probe (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_memorization_route_probe_2026_09_23.md`)
located most of the train/val gap. GLACIER is not immune to memorization either, since `G` plus
the formula can key a (molecule, formula) pair, and no train/val gap is reported for it.

### Versus our MASSter feature set

All 167 `model_type: masster` configs carry an identical set: `a_onehot, a_degree, a_hybrid,
a_formal, a_radical, a_ring` (ring + aromatic), `a_mass, a_chiral, a_ring_size, a_num_hs`; bonds
`b_degree` (type), `b_ring`, `b_conj`.

| GLACIER has | us |
|---|---|
| Degree one-hot + overflow | `a_degree`'s 12 one-hot columns were dead (list-vs-int); repaired in place 2026-09-20 at unchanged width |
| Fused-ring, atoms and bonds | **added 2026-09-22** as `a_fused_ring` / `b_fused_ring` |
| Bond ring-size multi-hot | **added 2026-09-22** as `b_ring_size`; was atoms only (`a_ring_size`) |
| Bond stereo | `b_stereo` registered, unused, and **measurably dead** — see below |
| Gasteiger charge | `a_charge`, on in the MASSter base since 2026-09-20 |
| - | we add `a_mass`, `a_radical` |

Note the `feature_set_audit` "gaps `a_ring_size`, `b_ring`" line is **template-only** — MASSter
configs enable both, plus `b_conj` and `a_num_hs`. `config/*_template.yml` still has them commented.

### Closing the head-input gap: the neutral-loss complement (2026-09-22)

GLACIER's intensity head (eq. 14) is

```
t_k = MLP( G || psi(f_k) || psi(r_form - f_k) || OHE(n_broken_k) )
```

MASSter already carried three of the four terms: the root context `G`, the candidate's own
pooled representation, and `cut_counts` (= `n_broken`). Two of those matches are loose
(2026-09-23). Our context vector is metadata only (adduct, instrument, frag mode, CE), whereas `G`
is a learned whole-molecule embedding. And `psi(f_k)` is the fragment's exact **formula**, while
ours is a pooled atom embedding plus `c_mass`; we have no fragment-composition feature at all. The missing term was
**`psi(r_form - f_k)`**, the elemental composition of the complementary neutral loss, now added
as `masster_candidate_composition_features` (off by default; arm
`..._c4a_complement_composition.yml`, a one-key diff from `c4a_base_s1`).

The gap was narrower than "absent" and is worth stating precisely. The head already receives a
mean-pooled **embedding** of the complement's atoms, and since 2026-09-20 the complement's
**mass** through `candidate_mass_features`. Neither is a composition: a mean of learned atom
vectors carries no exact counts, and the mass alone cannot separate a CO loss (27.9949) from a
C2H4 loss (28.0313) — one 0.1 Da bin apart, which is the resolution GLACIER scores NIST'20 at.
`test_complement_composition_separates_isobaric_neutral_losses` builds exactly that pair from
pentan-2-one, where the two candidates are themselves near-isobaric (C4H10 58.078 vs C3H6O
58.042) so the mass scalars barely move. The feature costs 18 columns on a 1,285-dim token.

FraGNNet has had the same input available all along as the `neutral_loss_formula` fragment
feature, with a written arm
(`config/fragnnet/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_neutral_loss.yml`).

*Corrected 2026-09-27* (this paragraph earlier said that arm had never run and that no
production arm had been trained):

- The FraGNNet arm ran on 2026-07-15 (`b7witrjj`, crashed at epoch 63) and has not been read.
- The MASSter arms `c4a_complement_composition` (`lxf97tck`) and `c4a_ring_topology`
  (`3gqgkl4x`) ran 128 epochs on 2026-09-23 and are null: +0.0012 and −0.0027 against the 0.7270
  control.
- The FraGNNet `graph_feature_parity` arm ran on 2026-09-23 (`m8qgi0ft`, crashed at epoch 77) and
  has not been read.

### Closing the graph-feature gap (2026-09-22)

The three features above were the only graph features GLACIER carries that we lacked **and** that
carry signal on our data. They are registered in `feat_utils.py`, commented into
`config/template.yml`, and enabled in two matched arms:
`config/masster/ma_mi_merged/nist20_ma_mi_merged_fast_batch512_cuda_lb64_c4a_ring_topology.yml`
(single-purpose diff from `c4a_base_s1`) and
`config/fragnnet/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_graph_feature_parity.yml`. Both resolve
to a **73-dim atom / 16-dim bond** input; the cost is ~9k parameters on FraGNNet (2.79M -> 2.80M)
and ~3k on MASSter, and no DAG regeneration, because mol graphs are built on the fly.
**No production arm has been trained**; these are inputs, not results.

#### Smoke coverage (2026-09-22)

Two debug configs stack **all four** of the 2026-09-22 features at toy width, so one short run
exercises every new code path -- both feature registries, the widened GNN input, and
`_element_counts_by_group` feeding the widened intensity token:
`config/masster/debug/debug_tiny_masster_glacier_feats.yml` (MASSter, unique-fragment mode and
the CUDA candidate backend as in the c4a arms, 2 epochs) and
`config/fragnnet/debug/debug_tiny_fraggnn_graph_feature_parity.yml` (FraGNNet, 5 epochs). Both ran green
on `debug_tinyv4` with `detect_anomaly: true`, loss finite and falling, no NaN.

The features are live rather than silently inert, confirmed three ways. Resolved widths move as
claimed, 71 -> 73 atom and 9 -> 16 bond. On naphthalene the columns carry the right chemistry:
2 of 10 atoms fused, 1 of 11 bonds fused, 11 of 11 bonds flagged `in6`. And the parameter counts
separate cleanly, each arm's cost landing where it should:

| arm | MASSter (debug width, `model_dim` 64) | FraGNNet (debug width, `mol_hidden_size` 128) |
|---|---|---|
| neither | 339,470 | 1,202,466 |
| ring topology | 340,494 (+1,024) | - |
| complement composition | 340,622 (**+1,152 = 18 x 64**, the whole feature on the token) | n/a |
| both | 341,646 (+2,176, exactly additive) | 1,205,538 (+3,072 = (13 + 11) x 128) |

These are shape and finiteness checks on 40-64 spectra. They say the code runs; they say nothing
about whether the features help.

Liveness, measured on 5,000 NIST20 molecules (`data/proc/nist20/mol_df.pkl`, seed 0):

| feature | coverage |
|---|---|
| `a_fused_ring` | 6.5% of atoms; **40.3% of molecules** contain at least one |
| `b_fused_ring` | 3.6% of bonds |
| `b_ring_size` | 50.3% of bonds are in a ring; 0.6 / 0.2 / 17.7 / 80.0 / 3.9% over [3, 4, 5, 6, >=7] |
| `b_stereo` | **0 of 127,249 bonds** |

**Bond stereo cannot be matched by enabling a feature.** Preprocessing strips stereochemistry, so
every bond is `STEREONONE` and every atom `CHI_UNSPECIFIED` — verified here on NIST20, NIST23 and
MassSpecGym alike, extending the atom-side finding in
[`masster_feature_set_audit.md`](../reference/masster_feature_set_audit.md) §7.6 to bonds and
to all three corpora. Enabling `b_stereo` would add 7 constant-zero columns. Matching GLACIER here
means keeping isomeric SMILES through preprocessing, which is a preprocessing change under the
authorization rule in [engineering rules](../rules/engineering.md) §6, and it would also revive the
5 dead `a_chiral` columns that every model already pays for.

## Split

The paper gives no ratios. It states only that it evaluates a random split "partitioned by InChI
key" and a Murcko scaffold split, over 530,640 HCD spectra / 25,541 unique structures. The repo
default (`data_scripts/make_splits.py`) is `--train-frac 0.8 --val-frac 0.1 --test-frac 0.1`
applied at the object level, which measures 0.809/0.090/0.101 on the checked-in `split_1.tsv`.
Ours is 60/20/20 — see [our split ratios](../reference/dataloader_data.md#split-ratios-from-local-memory-re-measured-2026-10-06).

## Method

Single-stage fragment "object detection": a transformer predicts fragment subgraph masks directly
from the molecular graph, replacing the enumerate-then-score two-stage pipeline (their own
ICEBERG). Affinity matrix -> binary subgraph masks with a temperature-controlled relaxation;
binary attention masks restrict each fragment's attention to its own atoms. Losses: cosine
spectral distance plus an auxiliary loss on heuristically-labeled fragment patterns. Claims ~8x
inference speedup over the two-stage model. Intensities are predicted unbinned and re-binned at
inference, which they argue lets one model serve multiple bin widths.

### No pretraining (checked 2026-09-27)

Every GLACIER model trains from random initialization on its own benchmark split. It uses no
outside-data pretraining and no pretrained molecular encoder.

- **Backbone:** Graphormer with `apply_graphormer_init=True`, which only applies a
  normal(0, 0.02) init (`src/ms_pred/graphormer/graphormer_graph_encoder.py:121`, `318bb7b`).
  No PCQM4M weights are loaded. `configs/glacier/joint_train_{nist20,msg}.yaml` pass no
  checkpoint, so the MassSpecGym model trains from scratch on MassSpecGym, not from NIST'20.
- **The MAGMa warm-start happens inside that same run.** `w_magma = 1` for 2,500 steps, then
  x0.9 every 2,500 steps (eq. 10, line 404; `magma-warmup-steps`/`magma-decay-steps: 2500`).
  In Table 5 (line 588), training on the intensity objective alone does not converge, and
  keeping MAGMa on throughout is worse (test loss 0.243 vs 0.177).
- **Contrastive finetuning is the only second stage.** `train_contr_joint.py` loads the stage-1
  `best.ckpt` from the same split (`train-checkpoint:` in `joint_contr_finetune_*.yaml`).
  Appendix E calls stage 1 "the pretraining stage" (line 908). CF lowers cosine
  (0.838 -> 0.815, Table 15), so **the 0.838 is the single-stage model without CF**.

## Design comparison — FraGNNet / MASSter vs GLACIER (2026-09-21)

Read against `research/masshatter-ion-h-selection` @ `92a0943a`, the production FraGNNet config
`config/fragnnet/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all.yml`, and GLACIER Sections 3.2-3.4
plus Appendices C-D. Architecture only — none of this is a measured result.

| stage | FraGNNet (prod, `pqx7ywc9`) | MASSter | GLACIER |
|---|---|---|---|
| fragment proposal | offline combinatorial BFS depth 3 + 16 SMARTS / 7 NL prepass rules, cached DAG | learned **per-bond** Bernoulli -> rank whole cut masks -> connected components | learned **per-atom** boundary detection: DETR query tokens -> affinity matrix `S` -> LinSATNet top-k' (k<=3) -> delete breakpoint atoms -> connected components |
| proposal supervision | none (enumeration) | proposal-coverage loss + support BCE | Hungarian set-matching to MAGMa-derived boundary patterns, per-decoder-layer aux losses decayed 0.9 |
| backbone | GINE x4 @384 on the mol graph | GINE_GSA x2 @384 (1.81M, a measured plateau) | Graphormer, shared, one pass |
| fragment embedding | pool its own atoms' embeddings via the `cc` mask; `cc_interstage: direct`, `cc_interstage_use_rest: False` (no molecule or complement context) | 3x384 pooled + 128 context | "dynamic embedding" pool + masked TransformerDecoder over its own atoms (eq. 7) |
| **fragment <-> fragment** | `frag_gnn_type: NodeMLP`, `pyg_edges: False` -> no DAG message passing. Only interaction is **subtracting the per-spectrum max-pooled fragment vector** (`src/fragnnet/models/fragnnet/fragnnet_model.py:1925-1930`) | **ISAB**, 64 inducing points, 3 layers (49.5% of the intensity head) | **full TransformerEncoder over the fragment set** (eq. 8) |
| intensity-head inputs | psi(formula) via `base_formula`, `depth`, `boundary_pair`, h-shift index; `neutral_loss_formula` implemented, off; its arm ran 2026-07-15 (`b7witrjj`), not read | pooled embeddings + `cut_counts` + log1p(size) + component count, **plus log1p(neutral mass), mass/precursor fraction and summed Gasteiger charge since `86268615` (2026-09-20)**, **plus the complement composition since 2026-09-22 (opt-in)** | psi(formula) \|\| **psi(root - formula)** \|\| **OHE(n_broken)** \|\| root token `G` (eq. 14) |
| output law | **softmax over the joint (fragment x formula) grid** + OOS sink; 9 H-shifts (+/-4) x {CRF, CMF} | charge head -> ion-mode / H-channel head | **independent sigmoids**, 13 H-shifts (+/-6) x {CRF, CMF} = 26, summed at exact m/z then binned (eqs. 15-16) |
| loss | Gaussian-density CE at **instrument tolerance** (FT 10 ppm / QTOF 20 ppm) | spectral CE (JSS ~= CE) + support BCE | cosine on **0.1 Da bins** + MAGMa teacher-forced auxiliary, weight `w_magma` annealed 1 -> 0 |
| conditioning | CE (NCE+ACE), adduct, instrument, frag-mode at encoder input + CE FiLM + mode affine on logits | same context encoder | adduct (one-hot + element-group multi-hot), CE (64-dim sinusoid) and instrument one-hot, concatenated onto every atom's input features; nothing in the attention biases (code, corrected 2026-09-23) |
| params | **2.77M measured** (2,772,290 trainable; 2.78M with the 2026-09-22 features) | **9.22M measured** (9,215,573 trainable) at the `c4a_base_s1` geometry: bond head 3.87M, intensity head 2.46M, encoder 1.81M, boundary encoder 0.79M | not in the paper, **15.11M measured** (15,105,694 trainable, 2026-09-23): upstream `ms_pred.glacier.joint_model.JointModel` at `samgoldman97/ms-pred` `318bb7b` with `configs/glacier/joint_train_nist20.yaml`. Fragment decoder (200 slots, 4 layers) 4.26M, intensity decoder 4.21M, Graphormer x5 @256 4.01M, fragment-set encoder 2.37M. ICEBERG 2.0 at the same commit and its NIST'20 configs: **7.06M** (gen 3.64M + inten 3.43M) |
| training recipe | — | AdamW wd 0.01, dropout 0.1, **constant** lr 1e-3, batch 512 merged spectra, fixed 128 epochs, best-val-cosine checkpoint | wd **0**, dropout 0.1, lr 4e-4 with 2000-step warmup and 0.9 decay, batch **16**, early stopping on val loss (patience 5), grad clip 5 (hardcoded in `train_joint.py`). ICEBERG: wd 0 / 1e-7, dropout 0.2 / 0.1, warmup + exponential decay, batch 32, patience 5 |

### 1. Fragment competition is the gap

GLACIER gives every fragment full self-attention over the whole candidate set *before* emitting
intensity. MASSter gives it rank-64 induced attention. FraGNNet gives it **one max-pooled
vector, subtracted** — a rank-1 set statistic — after which every logit is computed independently
and the joint softmax normalizes them. A softmax is a *constraint* on allocation, not a
*mechanism* for it.

[`findings/fragnnet/glacier_gap_ce_decomposition_2026_09_21.md`](../findings/fragnnet/glacier_gap_ce_decomposition_2026_09_21.md)
measured that the entire 0.077 gap is CE-dependent **allocation** against a ceiling we already
hold (oracle 0.9492; realization 88.4% at NCE<20 falling to 75.6% at 120+). FraGNNet has no
architectural site where "this fragment wins at NCE 120 and that one does not" can be computed.
That is the cleanest design-to-measurement match in this comparison.

### 2. Head inputs — mostly already built, off by default

GLACIER eq. 14's three inputs map onto existing code:

| GLACIER eq. 14 | FraGNNet | MASSter |
|---|---|---|
| psi(fragment formula) | `base_formula` — **in prod** | absent (no composition vector) |
| psi(root - fragment formula) | `neutral_loss_formula` — implemented, only in the `_neutral_loss` config arm | absent |
| OHE(n_broken) | `num_cut_bonds` — implemented, only in the `_cut_bond` arms | `cut_counts` (raw scalar, not one-hot) |
| root token `G` | `cc_interstage_use_rest` / context embedding, both off in prod | 128-d context |

So for FraGNNet this is a **config edit, not a build**. For MASSter, absolute mass arrived
2026-09-20 as a template default; composition, the neutral-loss complement, a sinusoidal/RBF mass
encoding and any pairwise `dm_ij` attention bias are still absent — see
architecture review (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_architecture_review_todo.md`), whose "mass-blind" section is amended for this.

### 3. Proposal: ours is strictly more expressive, and it is not the bottleneck

Their k<=3 **breakpoint-atom** parameterization cannot express two cuts landing on the same
boundary atom; deleting boundary atoms recovers the queried fragment exactly while mangling its
complement; and the reachable space is bounded by MAGMa labels, with no analog of the xanthone
4-cut, morphinan 5-ring, retro-Diels-Alder or the 7 neutral-loss rules. Their gain here is the
~8x inference speedup, not accuracy, and our own oracle says coverage is not binding.
**Do not trade the DAG for a learned proposer to chase this number.**

### 3b. MAGMa is a weak teacher, and it bounds more of GLACIER than the paper admits

Their whole fragment branch is supervised by MAGMa labels (eq. 2 derives the breakpoint patterns
from MAGMa subgraphs; `L_fragment` matches queries to them). Read what MAGMa actually is, from the
implementation they use — `ms-pred/src/ms_pred/magma/fragmentation.py`, parameters
`FRAGMENT_ENGINE_PARAMS = {"max_broken_bonds": 6, "max_tree_depth": 3}`:

- Fragments are **atom bitmasks of the intact molecule reached by deleting bonds** — connected
  substructures only. MAGMa **cannot form a bond**, so no skeletal rearrangement, no
  retro-Diels-Alder product, no ring contraction, nothing in our 16 SMARTS prepass rules or the
  7 neutral-loss rules. Hydrogens are not modelled at all; they enter as a `+/-h` correction on the
  formula with a bond-order-weighted penalty (`score_fragment`, `atom_pass_stats`).
- **`max_tree_depth: 3`.** Their labels are depth-3, the same depth as our d3 DAG. The claim to
  "generate fragments with various depth in a single forward pass" is about inference cost; the
  supervision is not deeper than our enumeration.
- **`max_broken_bonds: 6` is where their "13 hydrogen shift states" comes from** (`2*6+1`, and
  `shift_bucket_inds` is literally that array). Their `+/-6` window is an inherited MAGMa constant,
  not a chemical choice — worth knowing before treating our `num_hs: 4` (9 states) as the
  under-specified side.
- MAGMa labels are **peak-driven assignments over a mass-degenerate substructure set**: many
  substructures share a formula, so the Hungarian matching in eq. 5 optimizes against an
  ambiguous target.

The `w_magma` anneal is supposed to free the detector from this. But once `w_magma -> 0` the only
remaining gradient is **cosine on 0.1 Da bins**, which cannot tell a rearranged fragment from an
isobaric non-rearranged one — the coarse grid and the weak teacher fail in the same direction.
So end-to-end training can re-weight fragments MAGMa already proposes; it has no signal that would
teach a rearrangement MAGMa cannot express. This is the strongest form of the section-3 point:
their proposal stage is not merely less expressive than a DAG, it is trained toward a labeling
that excludes the chemistry a DAG plus prepass rules exists to capture.

### 4. Output law and supervision resolution — our real advantage, invisible under their metric

We fit a normalized distribution against exact masses at instrument tolerance, so fragment
*identity* receives gradient. Their eq. 16 sums fragments into a bin, so at 0.1 Da the
CO / N2 / C2H4 split gets **no gradient at all**, and independent sigmoids never have to allocate
a fixed budget. Eq. 16 lets them re-read at any bin width post hoc, but they never trained the
distinction — which is why their Appendix F shows retrieval improving at 0.01 Da while cosine
prefers the coarse grid.

Two smaller asymmetries: their CE reaches only the encoder input, and through it every atom
embedding and the root token `G` that seeds each fragment token. The paper's Section 3.1 wording
suggested CE also reaches the edge-level attention bias, but the code does not do that (corrected
2026-09-23). Ours reaches the encoder input and an output FiLM. Neither injects CE into
fragment-to-fragment attention directly, which is where the CE decomposition points. And their
MAGMa -> end-to-end anneal is a curriculum we have only as a stub (`dag_curriculum` configs).

### Consequence for the ranked plan

This adds an item 0 to the ranking in
[`findings/fragnnet/glacier_gap_ce_decomposition_2026_09_21.md`](../findings/fragnnet/glacier_gap_ce_decomposition_2026_09_21.md):

0. **Give FraGNNet a real fragment-set interaction.** Replace `frag_pool_combine: subtract` with
   attention over the fragment set — `set_transformer` already exists in `build_pool_module`, and
   the `concat` readout cell from the [successor-lines note](../plans/fragnnet/false_peak_loss_plan.md#successor-output-head-lines-all-null-from-local-memory-verified-2026-10-06) is built and untrained.
   It is the only one of GLACIER's three mechanisms with no production analog on our side, and the
   one the CE finding points at.
1. Marginal sibling-ranking loss on P(n) — unchanged; attacks the same allocation defect from the
   loss side rather than the architecture side.
2. Composition-aware head: enable `neutral_loss_formula` + `num_cut_bonds` on FraGNNet
   (config-only); add a formula vector and a `dm_ij` bias to MASSter's candidate tokens.
