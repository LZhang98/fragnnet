# MASSter isotope envelope

> **Status:** CLOSED NULL · **Updated:** 2026-10-03
> Closed at gate 0a by its stop rule. On Enveda val, even a perfect monoisotopic model plus the
> best scalar envelope recovers only 0.0047 cosine in positive mode (95% CI 0.0045-0.0050)
> against the 0.005 gate, and 0.0010 in negative mode. The natural prior alone recovers nothing
> in positive mode and costs 0.013 in negative mode. Nothing was built. The result and what would
> reopen the plan are under [Stage 0a step 3](#step-3-on-enveda-val-2026-09-27-gate-0a-fails).
> On Spectraverse even a perfect envelope recovers at most 0.0042 in positive mode
> ([SPV step 2](#step-2-on-spectraverse-2026-10-03-ceiling-below-the-gate)). NIST20 and MSGYM
> were not measured.

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

**Step 2 on Enveda val (2026-09-26).** All 79,881 val spectra of `enveda_180_inchikey_811`, raw
intensities as the eval cosine uses them (`ints_transform: none`). A perfect monoisotopic
prediction reaches cos = ‖target without satellites‖ / ‖target‖:

| Val spectra | n | 1 − cos, 13C satellites | 1 − cos, 13C + halogen M+2 | Satellite intensity share |
| --- | ---: | ---: | ---: | ---: |
| positive | 61,431 | 0.0052 | 0.0086 | 7.4% |
| negative | 18,450 | 0.0020 | 0.0025 | 3.2% |
| Cl/Br molecules | 11,886 | 0.0036 | 0.0220 | 10.0% |
| all | 79,881 | 0.0044 | 0.0072 | 6.5% |

- The satellites are 6.5% of intensity but 0.0072 of cosine, because cosine squares intensities
  and satellites are small peaks. A perfect envelope therefore gains at most 0.0086 in positive
  mode, above gate 0a's 0.005. Step 3 decides how much of that a natural envelope reaches.
- Negative mode cannot pass a 0.005 gate at any f: its whole ceiling is 0.0025.
- Most of the halogen share (0.022 on Cl/Br molecules) is the per-molecule M+2 switch, which no
  scalar f reaches.
- Recall-type metrics (`wrecall`) see the 6.5% intensity share, not the cosine number.

#### Step 3 on Enveda val (2026-09-27): gate 0a fails

Probe: [debug_enveda_isotope_envelope_ceiling.py](../../../scripts/debug/debug_enveda_isotope_envelope_ceiling.py),
53 s on 10 workers. Per-spectrum dot products are in
`output/enveda_180_isotope_probe_2026_09_25/envelope_ceiling_val.pkl`, and the log sits beside
them.

- **Population.** The 77,320 `[M+H]+`/`[M-H]-` val spectra of `enveda_180_inchikey_811` (the
  population of the baseline config).
- **Metric.** The val metric's cosine: raw intensity summed in floor(m/z / 0.01) bins.
- **Prediction.** The target's monoisotopic part, which a perfect monoisotopic model predicts,
  plus f1 × M+1 and f2 × M+2.
  - The envelope is the exact natural isotopologue ratio of each peak's precursor-bounded
    subformula compositions (`subformula_candidates`, 10 ppm), averaged over candidates.
  - It carries per-element offsets for M+1 and for heavy M+2 isotopes, and two 13C spacings for
    M+2 pairs.
  - 97.8% of monoisotopic intensity has a subformula.
- **Fit.** f is fitted per polarity on even `mol_id` (grid step 0.05) and reported on odd.

| Odd-molecule val spectra | Perfect mono, 1 − cos | + natural (f1 = f2 = 1) | + fitted scalar f | + per-spectrum best f2 ∈ {0, 1} |
| --- | ---: | ---: | ---: | ---: |
| positive (29,403) | 0.0085 | 0.0085 | 0.0038 (f1 0.80, f2 0.30) | 0.0019 |
| negative (9,191) | 0.0025 | 0.0155 | 0.0015 (f1 0.40, f2 0.05) | 0.0015 |
| positive, Cl/Br (4,268) | 0.0274 | 0.0479 | 0.0182 | 0.0054 |
| negative, Cl/Br (1,464) | 0.0045 | 0.0762 | 0.0037 | 0.0035 |

- **Recovered cosine at the fitted f**, 95% molecule-bootstrap:
  - positive **0.0047** (0.0045-0.0050), of which the 13C part alone (f2 = 0) is 0.0039;
  - negative **0.0010** (0.0009-0.0010).

  Positive mode is under the 0.005 gate, so the stop rule applies. This is the ceiling for a
  perfect monoisotopic model. A trained model's mono peaks are imperfect, so it would realize
  less, below the +0.003 read gate.
- **The natural prior is harmful**, which is what `natural` mode and FraGNNet's `num_isotopes > 1`
  would apply.
  - It recovers +0.0000 in positive mode and costs −0.013 in negative mode.
  - It adds halogen M+2 peaks at full strength, and negative mode never transmits them (§9.3 of
    the finding).
  - FraGNNet on Enveda must therefore keep `num_isotopes: 1`.
- **Out of reach for any scalar.** The per-molecule halogen M+2 switch is the rest of the
  positive-mode ceiling: a per-spectrum oracle f2 would add 0.0019. The structure cannot predict
  it.
- **What would reopen the plan:**
  - a metric that weights small peaks more, such as sqrt or entropy-weighted similarity for
    retrieval;
  - a need for correct per-peak annotation, since +1 H channels still stand in for 13C above
    m/z 447 at 10 ppm;
  - a dataset whose isolation passes the full envelope.

  NIST20 FT and MSGYM were not measured.

#### Step 2 on Spectraverse (2026-10-03): ceiling below the gate

Probe: [debug_spectraverse_isotope_probe.py](../../../scripts/debug/debug_spectraverse_isotope_probe.py),
2 min on 6 workers, all 488,797 spectra of `data/proc/spectraverse/spec_df.pkl` (every fold, not
val only). The log and per-spectrum table are in
`output/analysis/spectraverse_isotope_probe_2026_10_03/`. Satellites use the §9.4 rule. The
cosine is computed peak by peak, not in the val metric's 0.01 Da bins. A halogen M+2 partner
counts only if its −2 Da partner is more intense.

| Spectra | n | 13C satellite share | 1 − cos, 13C | 1 − cos, 13C + halogen M+2 |
| --- | ---: | ---: | ---: | ---: |
| positive, all | 380,930 | 2.0% | 0.0039 | 0.0042 |
| negative, all | 107,867 | 1.4% | 0.0021 | 0.0022 |
| QTOF positive | 139,261 | 3.4% | 0.0067 | 0.0074 |
| ORB positive | 222,635 | 1.3% | 0.0024 | 0.0024 |
| Other (both modes) | 16,181 | 0.1% | 0.0002 | 0.0002 |
| QTOF positive, Cl/Br | 5,799 | — | — | 0.0211 |

- **Gate 0a cannot pass on SPV.** A perfect envelope would recover at most 0.0042 in positive
  mode. That is below the 0.005 gate before any scalar f, and Enveda realized about half its
  ceiling. Only QTOF positive mode (0.0074) has a ceiling above the gate.
- **The ORB rows mostly isolate the monoisotopic precursor alone.** Where the precursor holds ≥5%
  of base peak, M+1 is absent in 85% of ORB `[M+H]+` spectra. ORB Cl/Br spectra pass the halogen
  M+2 in 0.2% of cases. ORB spectra without the precursor M+1 put 0.16% of intensity on
  13C-spaced peaks, which is the false-pair floor. The 15% that keep it put 5.9% there, at a
  median f1 of 0.45.
- **QTOF passes more of the envelope, and it varies.** M+1 is present in 42% of `[M+H]+`
  spectra, at a median f1 of 0.85 where present. Halogen M+2 passes in 47% of positive-mode Cl/Br
  spectra. Across Cl/Br molecule × polarity groups with ≥2 spectra, 3,788 never pass M+2, 28
  always do and 223 are mixed. So within SPV the halogen M+2 depends on the spectrum, not on
  the molecule.
- **`Other` is effectively de-isotoped.** 0.1% satellite share, and the precursor is in only
  5-8% of its spectra.
- **Resolution.** ORB separates 13C from a +1 H shift below m/z 600: the trough between them is
  3-9% of the 13C peak. QTOF does not separate them below m/z 300, where the trough is 58%.

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

- On Enveda trained alone (user decision, 2026-09-26: standalone for now, no NIST or MSGYM mixed
  in). The baseline `i0` is
  `config/masster/enveda/enveda_180_inchikey_811_unmerged_mh_mmh_ace_fourier.yml`
  (`[M+H]+`/`[M-H]-`, 20 epochs, not submitted; renamed from `…_qet_…` 2026-09-27 and from
  `…_dual_ce_fourier_ema999` 2026-09-28). Never run,
  since the plan closed at gate 0a. Planned arms:
  - `i0`: the baseline;
  - `i1`: `natural`;
  - `i2`: `scaled`.
  All three set `filter_subformula_peaks: False` and `filter_above_precursor_peaks: False`.
- **Tolerance (corrected 2026-09-26).** Use 10 ppm with a 200 m/z floor, not 5 ppm.
  - Enveda's per-spectrum mean fragment error (`peaks_abs_ppm`) is 3.8 ppm median, 6.7 p90 and
    10.1 p99. The signed mean is nearly as large, so most of it is a per-spectrum offset, and
    5 ppm would drop real matches in many spectra.
  - At 10 ppm an isotope row and a +1 H row separate below m/z 447, which covers most fragment
    peaks (§9.1 found the 13C/H split resolvable below m/z 300). The QTOF floor of 15 mDa that
    most MASSter configs use would put them in the same window at every m/z, so the test could
    not separate them.
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

- **Closed 2026-09-27 at gate 0a** ([step 3](#step-3-on-enveda-val-2026-09-27-gate-0a-fails)).
  No model code was built, and none should be without one of the reopening conditions listed
  there. The measurements behind the plan are in
  [finding §9](../../findings/enveda_180_dataset_2026_09_25.md#9-isotopes-2026-09-25).
- **The Enveda baseline stays** as the first standalone Enveda run, no longer as an isotope
  control:
  `config/masster/enveda/enveda_180_inchikey_811_unmerged_mh_mmh_ace_fourier.yml`,
  not submitted.
  - **ACE only, no EMA (user decision, 2026-09-28).** `spec_params.nce: false` and no
    `ema_decay`. Enveda records only ACE, and the dataset's NCE is derived from it. Renamed
    from `…_dual_ce_fourier_ema999`; the fork origin is now the MSGYM non-EMA arm
    (`msgym_benchmark_1_5_unmerged_dual_ce_fast_batch512_cuda.yml`). A third smoke with the
    same recipe exits 0 (`smoke_ace_2026_09_28.log`). Its checkpoint holds one ACE table
    (301 × 22 fourier, Linear(22, 32)), no NCE table and no EMA state.
  - **CE input.** On 2026-09-27 it moved from `qet` to `ce_insert_type: embed` with
    `int_embedder: fourier`, as the Spectraverse folds did, after `qet` arms `n1`-`n3` trailed
    `c0` on NIST20. The file and run were renamed from `…_qet_…`.
  - **Resolve check.** Before 2026-09-28 it resolved against the MSGYM EMA arm with only its documented deltas:
    data paths, adducts with `filter_prec_types`, the fourier table, a 10 ppm QTOF loss
    tolerance, and 20 epochs with `lr_total_steps` 24,240.
  - **Smokes.** Two one-epoch local smokes on 2,000 spectra per split (batch 64) exit 0: the
    `qet` version on 2026-09-26 and the fourier version on 2026-09-27.
    - The fourier checkpoint holds a 301 × 22 fourier table and a Linear(22, 32) per CE channel.
    - Logs are in `output/enveda_masster_smoke_2026_09_26/`.
  - **Local `TMPDIR`.** On this workstation, set `TMPDIR` to a short directory on the root disk.
    - `/tmp` is 3.7 GB and was full, so Lightning's atomic checkpoint write failed there.
    - A long `TMPDIR` path breaks the dataloader workers' sockets (`AF_UNIX path too long`).
  - **Before reading it.** It needs a second seed, since no Enveda seed spread exists. Submitting
    it is a separate decision, because this workstation has no Fir access.
