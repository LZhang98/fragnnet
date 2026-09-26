# MASSter isotope envelope

> **Status:** PROPOSED · **Updated:** 2026-09-26
> Nothing is built. MASSter predicts monoisotopic pathways only, and `num_isotopes` does not
> reach it. Stage 0 needs no training: a model-free ceiling, then a post-hoc envelope on existing
> checkpoints. It decides whether the model change is worth building.

## Objective and acceptance criteria

Let MASSter emit each pathway's isotope envelope (M+1, M+2), splitting the pathway's probability
by natural abundance. Isotope satellites then become reachable by the peak that caused them,
instead of being fitted by hydrogen-shift channels or left unexplained.

- **Gate 0a (model-free ceiling).** A monoisotopic target reconstruction plus its natural envelope
  must recover cosine that the monoisotopic reconstruction alone loses. On Enveda val it must
  recover at least 0.005 in positive mode, which is above the 0.0042 seed spread. Report
  negative mode, NIST20 FT and MSGYM alongside.
- **Gate 0b (post-hoc on checkpoints).** Adding the natural envelope to an existing checkpoint's
  exported peaks must raise val cosine. The lower bound of the 95% molecule-bootstrap interval
  must be above 0 on at least one of NIST20 (`3bej6fq7`) or MSGYM (`ll04i00m`).
- **Gate 1 (training).** An envelope arm must beat its matched control by more than the 0.0042
  seed spread on the tail mean. If 0b fails on NIST and MSGYM, run Gate 1 only on Enveda.
- **Stop rule.** If gate 0a recovers under 0.005 on Enveda positive mode, isotopes do not limit
  MASSter there. Close the plan as `CLOSED NULL`.

## Established facts

- **MASSter has no isotope path.**
  - `num_isotopes` is a `frag_params` key ([template](../../../config/template.yml)), and only
    `SpecMolFragDataset` reads it. MASSter loads through `SpecMolDataset`
    (`setup_utils.get_dataset_cls`), and neither `masster_pl.py` nor `masster_model.py` reads
    `frag_params`. A MASSter config that sets `num_isotopes: 3` trains exactly as with 1.
  - `_fragment_masses_and_valid_channels` in `masster_model.py` emits one monoisotopic m/z per
    fragment × ion mode × H shift.
- **Hydrogen-shift channels absorb 13C satellites today.**
  - A +1 H channel sits 4.47 mDa above the 13C peak. The loss scores each target peak against
    every prediction within tolerance (`sparse_cross_entropy_seq` in `model/loss.py`).
  - 104 MASSter configs set the QTOF loss tolerance to 20 ppm with a 15 mDa floor below m/z
    750, so +1 H matches 13C at every QTOF m/z. At the default 10 ppm with a 200 m/z floor, it
    matches above m/z 447.
  - The `h_transfer_bound` comment in `masster_model.py` records this ("they currently absorb
    M+1 isotope peaks").
- **Enveda-180 measurements** ([finding §9](../../findings/enveda_180_dataset_2026_09_25.md#9-isotopes-2026-09-25)):
  - 13C satellites hold 7.7% of positive-mode and 3.9% of negative-mode training-target
    intensity. Halogen M+2 partners hold another 6.2% on Cl/Br molecules.
  - M+1 co-isolates at f1 = 0.934 of natural in positive mode and 0.480 in negative.
  - Halogen M+2 is a per-molecule switch: on for 38% of positive-mode halogen spectra, almost
    never in negative mode. The structure cannot predict it.
  - The timsTOF separates 13C from an H shift below m/z 300.
- **NIST:** 0.0065 of merged MA/MI val target intensity is the precursor's own isotopologue
  ladder ([unexplained-peak split](../../findings/masster/unexplained_peak_split_2026_09_22.md)).
  Fragment satellites on NIST are not measured.
- **Target filters remove isotope peaks.** `filter_subformula_peaks` alone removes 64.3% of
  NIST-annotated isotope peaks
  ([subformula isotope plan](../subformula_isotope_detection_plan.md)), and
  `filter_above_precursor_peaks` removes the precursor ladder. An envelope arm trained with
  either on would learn to predict peaks its target no longer has.
- **Element counts are missing.** MASSter counts only hydrogen and adduct-removable elements per
  fragment (`required_atomic_numbers` in `masster_model.py`). Isotope ratios also need C, N, O,
  S, Cl, Br and Si per fragment, plus the atoms the adduct adds.
- **FraGNNet precedent.** `num_isotopes > 1` splits a formula's probability across its
  isotopologues by a fixed natural-abundance prior, with no new parameters. It has never run,
  because every DAG is `isoFalse`.

## Plan

### Stage 0a: model-free ceiling (no checkpoint)

On Enveda val (`enveda_180_inchikey_811`), NIST20 FT val and MSGYM val, per polarity:

1. Split each target into satellite peaks and monoisotopic peaks. A satellite is a peak with a
   more intense partner below it at the 13C or halogen spacing: the §9.4 rule of
   `scripts/debug/debug_enveda_isotope_probe.py`.
2. Report cos(target, monoisotopic part). This is the ceiling of a perfect monoisotopic model.
3. Report cos(target, monoisotopic part + envelope × f). The envelope comes from each peak's
   subformula composition (`subformula_candidates`). Use two settings, f = 1 and f = the
   measured f1 per polarity.

This fixes the size of the prize before any code changes.

### Stage 0b: post-hoc envelope on existing checkpoints

- Take `3bej6fq7` (NIST20) and `ll04i00m` (MSGYM) val exports with per-peak annotations
  (`--return_formula_str`).
- Add each exported peak's natural envelope, computed from its annotated ion formula, then
  re-score cosine against the unfiltered target.
- This tests whether the model's monoisotopic mass would be better spread across its envelope.
  It needs no new model code.

### Stage 1: implementation behind default-off keys

- **Keys** in `config/template.yml`:
  - `masster_isotope_envelope: none | natural | scaled` (default `none`);
  - `masster_isotope_max_order: 2`;
  - `masster_isotope_min_ratio: 0.01`, which drops isotope rows below that share of their
    parent.
- **Composition.** In `_fragment_masses_and_valid_channels`, add a count of isotope-bearing
  elements per fragment, beside the existing tracked counts. Add the adduct's added atoms per
  channel.
- **Rows.** At the output assembly, where `pred_mzs` and `pred_logprobs` are selected, emit extra
  rows for each selected pathway:
  - The m/z is the parent plus the exact per-isotope offset. M+1 comes from 13C and 15N. M+2
    comes from 13C2, 18O, 34S, 37Cl and 81Br, which sit up to 9.7 mDa apart.
  - The log-probability is the parent's plus the log of the normalized envelope share, so each
    pathway's probability mass is conserved.
  - `round_aggregate_peaks` merges rows that coincide at 4 decimals.
- **`scaled` mode.** Learn one log-scalar per (polarity × instrument) for M+1 and one for M+2,
  initialized at natural (0). That can reach Enveda's f1 of about 0.48 in negative mode. For the
  halogen switch it can only learn the average.
- **Annotation export and diagnostics.**
  - Each isotope row needs an isotope label beside its pathway indices.
  - Report `candidate_wrecall` with and without isotope rows, so coverage stays comparable with
    past runs.
- **Tests:**
  - probability mass is conserved per pathway;
  - offsets and ratios match pyopenms on known formulas, including Br2 and Cl;
  - element counts match RDKit;
  - `none` gives bit-identical outputs;
  - existing checkpoints load strict.
- **Cost.** Measure step time and memory: the loss is linear in prediction rows, and envelopes
  add about 1-3 rows per pathway after the ratio cut.

### Stage 2: training arms

- On Enveda, once an Enveda MASSter baseline exists, run:
  - `i0`: the baseline;
  - `i1`: `natural`;
  - `i2`: `scaled`.
  All three set `filter_subformula_peaks: False` and `filter_above_precursor_peaks: False`.
- **Tolerance.** Use a 5 ppm loss tolerance for Enveda: timsTOF errors are about 3 ppm. At the
  15 mDa QTOF floor, an isotope row and a +1 H row share every window, so the test would not
  separate them.
- **NIST20.** Run the same arms against `c0` (`vcpho5qm`) only if gate 0b passed on NIST20.
- **Read.** Compare the tail mean against the seed spread. Also report the per-peak annotation
  share that moves from +1 H channels to isotope rows.

## Unresolved questions

- Whether NIST FT spectra carry fragment satellites at all, given narrow isolation windows.
  Stage 0a answers it.
- Whether to model the precursor's own ladder, which in MASSter is the intact-molecule pathway's
  envelope. The rows above would cover it for free.
- `h_transfer_bound`. Once isotope rows exist, the +1 H channels no longer have to stand in for
  13C, so the `none` versus `complement` choice should be re-measured.
- Halogen M+2 transmission is set per molecule by acquisition. No scalar can predict it, so up to
  6.2% of halogen-spectrum intensity stays out of reach either way.

## Current state and next check

- **Built:** nothing. The measurements behind the plan are in
  [finding §9](../../findings/enveda_180_dataset_2026_09_25.md#9-isotopes-2026-09-25).
- **Next:** Stage 0a on Enveda val and NIST20 FT val. It is read-only, needs no checkpoint and
  extends the existing probe script.
