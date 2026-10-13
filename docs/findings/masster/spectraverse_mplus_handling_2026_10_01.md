# MASSter `[M]+` handling on Spectraverse

> **Status:** FINDING · **Updated:** 2026-10-06
> Code read, SPV census and inference checks; `[M]+` is always a pre-charged cation on SPV; no code or config changed.

## Decision

On Spectraverse (SPV), `[M]+` means a pre-charged cation in every case. All 3,405 `[M]+` spectra
belong to formal-charge +1 molecules (457 molecules). No `[M]+` spectrum belongs to a neutral
molecule, so SPV has no odd-electron radical-cation precursors. Every precursor m/z matches the
atom-sum mass minus one electron within 0.01 Da. MASSter's `[M]+` candidate space covers these
spectra better than any other adduct, at 0.874 mean intensity recall. That figure comes from only
101 sampled spectra. The worst sampled spectra are data-quality failures, not chemistry gaps.

There is one real candidate gap. The `valence` hydrogen-transfer bound used by the SPV configs
blocks `[HN(CH3)3]+` (C3H10N+, m/z 60.0808), the diagnostic fragment of quaternary ammonium
compounds. A quaternary `[N+]` gets −1 free valence slots, so a trimethylamine piece cut from the
cation can gain no hydrogen. The ion reaches ≥1% of base peak in 18.9% of the 1,277 SPV
trimethylammonium `[M]+` spectra: 57.6% on OB and 6.5% on QTOF. It averages 1.9% of total ion
current (TIC), or 4.6% on OB. That is about 0.26% of SPV spectra. Any fix changes the candidate
chemistry, so it is not made here.

## How `[M]+` flows through MASSter

- **Tables.** In [`formula_utils.py`](../../../src/fragnnet/utils/formula_utils.py),
  `[M]+` has formula delta `""` and mass delta `−e`. It has no CMF entry, so it emits charge-retention
  (CRF) rows only. Each fragment's ion m/z is its neutral atom-sum mass (implicit hydrogens
  included), plus the ±`max_h_transfer` hydrogen ladder, minus one electron.
- **Charge site.** The ion-mass code applies one electron correction whatever the atoms' formal
  charges are (`_fragment_masses_and_valid_channels` in
  [`masster_model.py`](../../../src/fragnnet/model/masster_model.py)). The earlier
  double-electron bug ([code audit §2](masster_code_audit.md)) is no longer present. The charged
  atom is not required to be in the ion: either side of a cut can be the ion, and the hydrogen
  ladder covers hydride or proton bookkeeping. The encoder sees the charge site only through
  the `a_formal` atom feature.
- **Data guards.** [`proc_utils.py`](../../../src/fragnnet/utils/proc_utils.py) keeps molecules
  with charge 0 or +1, and drops charge +1 molecules paired with any adduct other than `[M]+`. The
  graph builder rejects any other net charge. A neutral molecule with `[M]+` would be treated as
  an odd-electron `M−e` precursor, the EI case in the
  [EI plan](../../plans/masster/masster_ei_nominal_plan.md). SPV has none.
- **Valence bound.** `_atom_free_valence_slots` in
  [`feat_utils.py`](../../../src/fragnnet/utils/feat_utils.py) subtracts the positive formal charge
  of a cationic atom that carries no hydrogen. That gives `C[N+](C)(C)C` −1 slots and aromatic
  `[n+]` 0 slots. Neutral pyridine N gets 1 slot.
- **Reproduction.** Checkpoint `ggifzj9m` (cv0) predicted acetylcholine, myristoylcarnitine and
  cetrimonium at 30 eV on OB and QTOF with `--min_intensity 0`. Each output contains 59.0730
  (C3H9N+•) and 58.0651 (C3H8N+). None contains 60.0808. Aromatic `[n+]` is not affected in
  practice, because the other ring atoms supply slots. The thiazole `C6H10NOS+` (144.0478, +1 H)
  is predicted for every thiazolium below.

## SPV `[M]+` census

| Item | Value |
| --- | --- |
| Spectra / molecules | 3,405 / 457. QTOF 2,362, OB 1,022, NA 20, Other 1 |
| Charge site (spectra) | Quaternary N+ 1,937; aromatic n+ 575; P+ 564; aromatic o+ (flavylium) 251; S+ 27; other 51 |
| Zwitterions / radicals / charged atom carrying H | 5 / 0 / 6 molecules |
| Subformula retention at 1× instrument tolerance | 93.8% of intensity ([subformula census](spectraverse_subformula_filter_2026_10_01.md)) |
| Base peak equals precursor (±0.02 Da) | 19.6% of spectra |

Candidate recall below joins the per-spectrum tables from the
[recall finding](spectraverse_recall_tolerance_2026_10_01.md) (4,096 validation spectra per fold,
cv0 + cv1) to precursor type. It is mean intensity-weighted candidate recall at instrument
tolerance. The caveat in the last section applies to OB rows.

| Adduct | n | Mean | Median |
| --- | ---: | ---: | ---: |
| `[M+H]+` | 4,080 | 0.761 | 0.925 |
| `[M-H]-` | 1,947 | 0.809 | 0.923 |
| `[M+Na]+` | 988 | 0.388 | 0.283 |
| `[M+NH4]+` | 559 | 0.725 | 0.856 |
| `[M]+` | 101 | 0.874 | 0.939 |

The four lowest `[M]+` spectra are not chemistry gaps. Spectra 227874 and 213903 are QTOF
acylcarnitines whose top peaks have negative mass defects (281.9445, 262.8293), and they lack the
carnitine ions 85.028 and 60.081. Spectrum 414427 (cetrimonium, QTOF) is offset by about +0.006 Da
at low m/z (60.0862 vs 60.0808). Spectrum 322279 (acetylcholine, OB) matches 43.0178, 87.044 and
60.0807, but its unmatched peaks are inorganic-defect noise.

## Thiazolium predictions and m/z 123.08

The two queried thiamine analogues are isomers, C14H18NO3S+ (280.1002). Neither is in SPV.
Thiamine (`Cc1ncc(C[n+]2csc(CCO)c2C)c(N)n1`) and its mono- and diphosphates are in the
**training** split of both cv0 and cv1, so their predictions are fits, not held-out results. Below,
the two fold checkpoints are averaged and ACE 10–40 eV is shown. The shared label that both
query rows carry, C8H9O2+, is the same mass for both, but the structure differs. For the
dihydroxyphenethyl analogue it is the 3,4-dihydroxyphenethyl cation. For the vanillyl analogue it
is the vanillyl cation.

| Molecule | Top predicted ions | Candidates within 0.06 Da of 123.08 |
| --- | --- | --- |
| `Cc1c(CCO)sc[n+]1CCc1ccc(O)c(O)c1` | 137.0597 C8H9O2+ (base), 144.0478 C6H10NOS+ 0.15–0.25; at 40 eV 108.0570, 109.0648, 126.0372, 114.0372 | 123.0441 C7H7O2+ (dihydroxybenzyl) rises with CE to 0.17 (QTOF 40 eV); 123.0804 C8H11O+ ≈ 0 (rank 380–730) |
| `COc1cc(C[n+]2csc(CCO)c2C)ccc1O` | 137.0597 C8H9O2+ (base), 143.0399 C6H9NOS+• 0.1–0.3, 144.0478 0.06–0.19; at 40 eV 122.0362 C7H6O2+• (−CH3•) to 0.51, 107.0491 to 0.36 | 123.0441 C7H7O2+ to 0.12 (QTOF 40 eV); 123.0804 C8H11O+ ≈ 0 (rank 150–440) |
| Thiamine (train) | 122.0713 C6H8N3+ (base), 144.0478, 81.0447 | 123.0791 C6H9N3+• 2–3% on QTOF, ~0.1% on OB |

**What 123.08 is.** In the observed SPV thiamine OB spectra, the ~123.08 peak is 123.0747–123.0751
at ~10% of base. That is the ¹³C isotope of 122.0713 (123.0746 expected). It is not the C6H9N3+•
radical (123.0791, 36 ppm away). Isotope rows are off (`masster_isotope_envelope: none`). The OB
loss window is ±2 mDa at that mass and excludes the radical channel. The QTOF window is ±15 mDa
(`min_mz` 750) and includes it. The QTOF predictions therefore put the isotope's intensity on
123.0791: the model learned a +1 H stand-in for ¹³C on QTOF only. The query predictions repeat the
pattern at 138.0675 (C8H10O2+•, 2–3% on QTOF), 4.4 mDa from the 138.0631 ¹³C isotope of 137.0597.
For the two query molecules, C8H11O+ at 123.0804 would require losing a phenolic oxygen and
gaining 2 H. Chemically it is implausible, and MASSter correctly gives it near-zero intensity. A
measured 123.08 peak from either compound would more likely be miscalibrated C7H7O2+ (123.0441)
or a contaminant. Checking needs the observed spectrum.

## Old SPV checkpoints carry the retired `FT` label

The `spectraverse_masster` checkpoints `ggifzj9m` (cv0) and `u46v0k0g` (cv1) were trained with
`inst_types` FT/QTOF/Other/NA. Instrument indices are sorted, so training used FT 0, NA 1, Other 2,
QTOF 3. The current YAML says OB/QTOF/Other/NA, which sorts to NA 0, OB 1, Other 2, QTOF 3. Loading
these checkpoints with the current config therefore gives OB spectra the trained NA row and NA
spectra the FT row. The swap is silent: strict loading checks keys and shapes, not what each row
means.
[`analyze_spectraverse_recall.py`](../../../scripts/debug/spectraverse/analyze_spectraverse_recall.py) builds
the model this way. The OB rows of the [recall finding](spectraverse_recall_tolerance_2026_10_01.md)
were therefore scored with the NA embedding. Its QTOF rows are unaffected. The predictions above
used copies with rows 0 and 1 swapped and hparams relabelled, made by
`output/thiazolium_pred_2026_10_01/patch_ft_to_ob_ckpt.py`. They were run through
`scripts/run_inference.py --template_fp config/template.yml --custom_fp config/spectraverse/masster/cv{0,1}.yml`.
The fold config supplies `fill_missing_ace`, which postdates the checkpoints.

## Artifacts

Everything is in `output/thiazolium_pred_2026_10_01/` (ignored artifact store): input data frames
for the queries and thiamine (OB/QTOF × ACE 10/20/30/40, CID, `ce_type` single); the patched
checkpoints `*_ob.ckpt`; predictions `pred_{ggifzj9m,u46v0k0g}/` and the long table
`pred_peaks_long.csv`; and the quaternary-ammonium check under `quat/`. The thiazolium runs
finished at 18:55, before concurrent working-tree edits to the MASSter modules started at 18:56.
The quaternary check ran after them, on edits that only add an opt-in JEPA path.

Next: if the 60.0808 gap matters, decide whether the `valence` slot for a hydrogen-free cationic
atom should count its charge as a heavy-bond slot (currently −1 for quaternary N) or keep the
neutral-valence convention (0). The same quantity is used in FraGNNet's `compute_cc_h_cap`.
Re-run the OB recall rows with an index-corrected checkpoint before relying on them.
