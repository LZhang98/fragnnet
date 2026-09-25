# Splitting MASSter's unexplained intensity: escape versus coverage, 2026-09-22

> **Status:** FINDING · **Updated:** 2026-09-22
> The ~11% of validation intensity MASSter's candidate set cannot reach is **73% coverage
> and 27% escape**. Only **2.91 points** of the 10.89-point residual carry no feasible
> subformula and therefore cannot come from the molecule at all; **7.98 points** are real
> analyte fragments the enumeration misses. The escape population is a **diffuse small-peak
> background** — an infeasible peak carries 21% the intensity of a feasible one and half the
> mass is spread over 7.9% of spectra — so a per-spectrum scalar gate cannot address it.
> **46.5% of that escape population is isotopologues, not contaminants**: real analyte signal a
> monoisotopic candidate set can never match. Nothing in the data path removes peaks above the
> precursor, and the 0.90 points sitting there are 72% precursor isotopologue ladder, so cutting
> them deletes signal rather than noise. Model-free; no checkpoint, no training.

## Question

The [2026-09-22 overfit read-out](masster_overfit_readout_2026_09_22.md) established that
the train/validation gap is almost entirely intensity allocation, and that the only genuine
coverage ceiling is the ~11% of target intensity no candidate reaches (`val_wrecall` 0.8911).
That residual holds two populations that need opposite fixes:

- A peak whose m/z admits **no** precursor-bounded, nonnegative-RDBE subformula cannot come
  from this molecule. It is co-isolation, background or noise, and the model is currently
  graded against it. An **escape** is the right tool.
- A peak that **does** admit a feasible subformula is a real analyte fragment the candidate
  enumeration misses. An escape would only excuse it; the fix is **coverage** — enumeration
  budget, cut depth, or rearrangement chemistry.

Before building either, the question is how the 11% divides.

## Method

`scripts/debug/debug_unexplained_peak_split.py`. It resolves a run's own configuration and
builds the target through `BaseDataset._setup_dfs`, so the probed spectra are exactly the ones
the run trains against — same cleanup, same `tolerance_consensus` merge, same four-adduct
FT+QTOF HCD+CID scope. Feasibility reuses production `_subformula_masses`, the same
enumeration `filter_subformula_peaks` applies, at per-instrument tolerance (FT 10 ppm floor
200 Da, QTOF 20 ppm floor 750 Da). The probe **measures and never filters**; the target is
untouched.

Population: the full `val` split of `data/split/nist20_ma_mi_inchikey` under
`nist20_ma_mi_merged_fast_batch512_cuda_lb64_c4a_base_s1.yml` — **15,257 merged spectra,
5,172 molecules, 792,217 peaks**. `val_wrecall` is taken from the runs themselves (0.8911 in
`wttzk2ot` and `oykkffls`, 0.8923 in `oqobubu7`); the split below uses 0.8911 and is
insensitive to that 0.001.

Every enumerated candidate is by construction a precursor-bounded subformula, so the feasible
fraction is an upper bound on what any candidate set could reach. The gap between it and the
measured `wrecall` is the coverage population.

### The feasibility test has real power

A large precursor admits many subformulas, so "has a feasible subformula" could in principle
be vacuous. `--null_control` re-scores every spectrum against a donor molecule drawn from its
own precursor-mass decile:

| | infeasible intensity | infeasible peaks |
|---|---|---|
| true formula | 0.0291 | 0.1248 |
| donor formula, matched mass band | **0.5993** | 0.4152 |
| discrimination | **20.6x** | 3.3x |

Measured on the same full 15,257-spectrum split. A wrong molecule of similar mass rejects 60%
of the intensity where the right one rejects 2.9%, so the verdict is chemistry, not
arithmetic.

## 1. The split

| | intensity fraction | share of residual |
|---|---|---|
| reached by candidates (`val_wrecall`) | 0.8911 | — |
| **unexplained residual** | **0.1089** | 100% |
| — no feasible subformula (**escape**) | **0.0291** | **26.7%** |
| — feasible but unreached (**coverage**) | **0.0798** | **73.3%** |

By peak count rather than intensity, 12.48% of peaks (98,849 of 792,217) carry no feasible
subformula — so the infeasible population is numerous but small: **an infeasible peak carries
0.210x the mean intensity of a feasible one**.

**The escape is the minority problem.** Three quarters of what MASSter cannot explain is
real analyte fragmentation its enumeration does not reach.

## 2. Shape of the escape population

This decides what an escape would have to look like.

**Per-spectrum infeasible fraction:**

| p50 | p75 | p90 | p95 | p99 |
|---|---|---|---|---|
| 0.0066 | 0.0330 | 0.0776 | 0.1197 | 0.2637 |

23.3% of spectra are entirely clean. The mean spectrum carries 51.9 peaks of which 6.5 are
infeasible.

**Concentration of the infeasible mass:**

| share of all infeasible mass | sits in |
|---|---|
| 10% | 64 spectra (0.42% of the split) |
| 25% | 320 spectra (2.10%) |
| 50% | 1,201 spectra (7.87%) |

| threshold | spectra above it |
|---|---|
| >5% infeasible intensity | 17.56% |
| >10% | 6.64% |
| >25% | 1.10% |
| >50% | 0.36% |

**Largest infeasible peak, relative to its spectrum's base peak:**

| p50 | p90 | p95 | p99 |
|---|---|---|---|
| 0.0087 | 0.1258 | 0.2134 | 0.7427 |

12.93% of spectra contain an infeasible peak at ≥10% of base, 1.63% at ≥50%, and **0.67% have
an infeasible peak AS the base peak**.

**Reading.** Two regimes. The bulk is a diffuse low-level background: many small peaks,
present in three quarters of spectra, half the mass spread over 7.9% of them. On top sits a
thin tail — roughly 1-2% of spectra — where a dominant peak is not from the analyte at all,
which is what co-isolation looks like. **A per-spectrum scalar cannot represent the first
regime**, since the mass is distributed across peaks within a spectrum rather than
concentrated in identifiable bad spectra.

## 3. Nothing removes peaks above the precursor

There is no above-precursor peak filter anywhere in the training or validation data path.
`spec_params.preprocess` only pre-materializes tensors; `max_peak_mz` in `proc_utils` is a
spectrum-level inclusion test applied when the processed dataset is built, not a peak trim;
and `eval_bin_remove_prec_peaks` / `eval_hun_remove_prec_peaks` in `SpectrumPL` produce an
*alternate metric variant* that drops the precursor peak itself, not a target filter.

This is asymmetric with how the repo's oracle numbers were computed. `dag_oracle_scores.py`
defines a `1_above_prec` tier that drops every true peak above `prec_mz + tolerance`, and it is
the default tier for `dag_oracle_decomposition.py`, `dominant_peak_diagnostic.py` and
`export_zero_cos.py`. **Every published oracle ceiling was measured on a target the model does
not train on.**

What is actually up there:

| | intensity fraction |
|---|---|
| the precursor peak itself | 0.1242 |
| precursor isotopologue ladder, M+1..M+3 | **0.0065** |
| any other peak above the precursor | **0.0025** |
| total above the precursor | 0.0090 |

**Cutting above the precursor would remove signal, not noise.** 72% of that 0.0090 is the
precursor's own isotopologue ladder — real analyte ions. Only 0.0025 (0.25% of target
intensity, 8.7% of the escape population) is genuine co-isolation, contaminant or
multiply-charged material. 20.3% of spectra carry at least one such peak, averaging 0.38 per
spectrum.

The precursor peak itself carries **12.4% of all target intensity**, which is modelled (the
no-cut candidate) and is not at issue here.

## 4. Most of the escape population is isotopologues

The same reasoning applies below the precursor: a fragment's own M+1/M+2 is real analyte
signal that a monoisotopic candidate set can never match, and it is indistinguishable from a
contaminant to any test that only asks whether an m/z has a subformula. Asking instead whether
a *feasible* peak sits one or two neutron masses below each infeasible peak:

| | intensity fraction | share of escape population |
|---|---|---|
| one/two neutrons above a feasible peak | 0.0129 | 44.4% |
| — of which below the precursor | 0.0070 | |
| precursor isotopologue ladder | 0.0065 | 22.2% |
| **union of both tests** | **0.0135** | **46.5%** |
| **not explained by any isotopologue** | **0.0156** | **53.5%** |

The two tests overlap by 0.0059, because the precursor's own M+1 satisfies both; the union is
taken rather than the sum. By peak count, 36,223 of 98,849 infeasible peaks (36.6%) are
flagged as isotopologues and 62,123 are not.

**So the contaminant/noise population a true escape would address is 0.0156 — 1.56% of target
intensity, 14.3% of the unexplained residual.** The other 46.5% of the escape population wants
isotope handling, which is a different mechanism and already scoped: `remove_isotope_peaks`,
`remove_regular_isotope_peaks` and `remove_isotope_formula_source: subformula` exist in
`config/template.yml`, whose own comment records them as *"NOT HOOKED UP: the feature is off
everywhere and no run has been evaluated with it on"*. Isotopes have never been exercised on
the model side either — `num_isotopes` is 1 everywhere and every DAG is `isoFalse` — so the
model is being graded on peaks it structurally cannot produce.

This test is mass-only: it does not check that the intensity ratio matches the carbon count,
so 46.5% is an **upper** bound on the isotopologue share and 0.0156 a **lower** bound on the
contaminant population.

## 5. Stratification: flat

Intensity-weighted infeasible share.

| adduct | n | share | | instrument | n | share | | mode | n | share |
|---|---|---|---|---|---|---|---|---|---|---|
| `[M+H]+` | 8,865 | 0.0299 | | FT | 14,729 | 0.0291 | | CID | 7,654 | 0.0312 |
| `[M-H]-` | 4,222 | 0.0258 | | QTOF | 528 | 0.0270 | | HCD | 7,603 | 0.0269 |
| `[M+Na]+` | 1,919 | 0.0339 | | | | | | | | |
| `[M+Cl]-` | 251 | 0.0181 | | | | | | | | |

Everything sits in 0.018-0.034. This is **not** a QTOF or CID artifact, and not an Na/Cl
artifact either — the CMF-carrier fix of 2026-09-15 is in this checkout, and the enumeration
includes the charge-migration budget, so `[M+Na]+` at 0.0339 is a real if slightly elevated
number rather than the old bug. The escape population is uniform background.

## 6. What this implies

1. **The escape channel is worth building, but it is not where the 11% is.** It addresses
   2.91 points; coverage addresses 7.98.
2. **A per-spectrum scalar escape is the wrong shape** for the population it would target.
   This is independent of, and agrees with, the separate proof in the overfit read-out that a
   scalar `predict_oos` cannot move a scale-invariant metric: under cross-entropy the loss
   factorizes and leaves the candidate ranking gradient unchanged.
3. **Half the escape population is an isotope problem, not a noise problem.** 0.0135 of
   target intensity is isotopologues of peaks the model can already explain. The right
   response is isotope handling — either removing them from the target or predicting the
   envelope — not an escape channel. The machinery exists and has never been switched on.
4. **The genuine contaminant population is 0.0156**, 1.56% of target intensity and 14.3% of
   the unexplained residual. That is the honest size of the prize for a noise escape.
5. **Do not blanket-cut above the precursor.** It removes 0.0090 of which 72% is the
   precursor isotope ladder. Cutting only the non-isotopologue material above the precursor
   removes 0.0025 and is defensible; cutting the ladder is an isotope decision, not a
   denoising one, and should be made together with the rest of the isotope question.
6. **For the contaminant remainder, the per-peak shape still holds.**
   `filter_subformula_peaks` is a hard per-peak version of exactly this test, is implemented,
   and has never been run on this base (0 of 58 `nist20_ma_mi_inchikey` runs). Its measured
   +0.0061 on the narrow split is not a matched eval, since each arm is scored against its own
   cleaned target; a soft responsibility weight would keep the evaluation target intact.
7. **The priority is still coverage, not escape.** 7.98 points of analyte intensity have a
   feasible formula and no candidate — five times the contaminant population.

## 7. Limits

- "Feasible subformula" is **necessary, not sufficient**. A contaminant whose m/z happens to
  admit a subformula is counted as coverage, so **2.91 points is a lower bound on the escape
  population and 7.98 an upper bound on coverage**. The 20.6x null control bounds how loose
  this can be, but does not eliminate it.
- **Feasible does not mean reachable by bond cuts.** Part of the 7.98 points needs
  rearrangement chemistry rather than more enumeration. Separating "more cuts would reach it"
  from "no cut can" requires running the candidate enumeration and is the natural follow-up;
  this probe deliberately avoids it so the result needs no checkpoint.
- Measured on `val` only. The train-side number is not measured here, so this says nothing
  about whether target noise differs across the split.
- `val_wrecall` is imported from run summaries rather than recomputed, so the coverage
  figure inherits those runs' tolerance settings.

## Reproduce

```bash
python scripts/debug/debug_unexplained_peak_split.py \
    --custom_fp config/masster/ma_mi_merged/nist20_ma_mi_merged_fast_batch512_cuda_lb64_c4a_base_s1.yml \
    --split val --measured_wrecall 0.8911 --null_control
```
