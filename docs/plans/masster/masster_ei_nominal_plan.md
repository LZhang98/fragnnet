# MASSter on NIST23 EI (nominal mass, natural isotopes)

> **Status:** ACTIVE · **Updated:** 2026-09-30
> Stage 0 (feasibility) measured. Stage 1 built in the working tree (uncommitted): nominal mass
> mode, natural isotope rows, the `e1`/`e0` configs, 16 tests, GPU smokes of both arms.
> User decisions: nominal `nist23_ei`, not the NIST23 SHR set; select on `val_cos_sim_1.0`,
> not `val_cos_sim_0.01`; model isotopes; `loss_tolerance_abs` 0.5.
> No EI model has been trained. Next: submit `e1` and `e0` when the user decides.

## Objective and acceptance criteria

Train MASSter on the NIST23 electron-ionization library (`[M]+`, 70 eV, unit resolution).

- **Stage 1:**
  - MASSter emits nominal-mass channels, and each channel carries its natural isotope envelope.
  - Both are default-off keys, bit-identical to the current model when off.
  - Checks: tests for integer masses against RDKit, mass conservation, envelopes against a
    reference convolution, and old checkpoints loading strict.
  - A GPU smoke on `nist23_ei` must show `val_candidate_wrecall` near the Stage 0 ceiling, not 0.02.
- **Stage 2:** an EI run selected on `val_cos_sim_1.0_epoch/mean`, compared with a matched
  control: `e0`, the same config with `masster_isotope_envelope: none`. NEIMS on the same split
  is the natural external baseline and is not built.

## Established facts (2026-09-30)

### Data

- **Source.** `data/proc/nist23_ei/{spec_df,mol_df}.pkl` and `data/split/nist23_ei_inchikey`,
  built 2025-09-27 by `process_nist_ei_csv` in `preproc_scripts/01_prepare_df.py`.
  - The raw CSV is `data/df/nist23_ei_df.csv`; `data/raw/nist_23_ei/` is empty.
  - 261,424 spectra of 253,649 molecules (1.03 spectra per molecule).
  - The split is random by 2D InChIKey: 154,981 / 51,700 / 51,685.
  - Split elements: C, H, N, O, P, S, F, Cl, Br, I, Se, Si.
- **Every m/z is an integer** (0 of 26,143 sampled spectra have a fractional one), and the
  intensity maximum is 999.
  - Median 98 peaks per spectrum; p95 229.
  - The molecular ion holds a mean 5.3% of L1 intensity; peaks below m/z 50 hold 11.4%.
  - 5% of spectra have no peak within 79 Da of M.
- **Metadata from the 2025 pipeline.** `prec_type` `[M]+`, `spec_type` `MS1`, `inst_type` `Other`.
  - `frag_mode`, `ace` and `nce` are NaN; `prec_mz` is the neutral exact mass.
  - `mol_df` was pickled by a newer RDKit (depickle warnings), but 3,000 sampled molecules
    round-trip to their SMILES.
- **Replaced by user decision:** the NIST23 SHR set, 292,702 spectra in
  `data/raw/nits23_ei_shr/` and `data/df/nist23_shr_ei_df.csv`.
  - Its "high-res" m/z are NIST formula annotations of nominal peaks, one formula per peak.
  - They are given as the neutral formula mass (43.0184 = C2H3O), without the electron.

### The current model on EI data trains on 2% of the target

A 1-epoch smoke of `i1`'s config on a 512-spectrum EI subsample finished with exit 0.
- Changes from `i1`: `nist23_ei` paths, `prec_types: [M]+`, `nce`/`ace`/`inst_type` off.
- `val_candidate_wrecall` was 0.021: exact-mass channels against integer targets at 10 ppm
  (0.002 Da floor) match almost nothing.
- Only the metric changes between the 0.01 Da and 1 Da smokes (0.020 and 0.122).
- **Nothing fails loudly.** The loss needs nominal channels.

### Nominal reach of the `[M]+` candidate space

Probe: [debug_masster_ei_nominal_coverage.py](../../../scripts/debug/debug_masster_ei_nominal_coverage.py),
on 5,000 val molecules, 74 s on 6 workers. It carries its own copy of the SMARTS G0 gate's pool
and channel helpers, so it runs without that script; the copy reproduces the rows byte for byte. Rows are in
`output/masster_ei_nominal_2026_09_30/nominal_coverage_rows.csv`.
- Pool: the G0 stand-in pool, the connected components of every 0-2-cut mask; median 108 sets.
- Channels: `[M]+`, CRF only, valence bound.
- Nominal m/z: the integer mass sum over atoms (most abundant isotope), plus the H shift.

| Channels reaching the peak | L1 share | L2 share | mean sqrt(L2), cosine bound |
| --- | ---: | ---: | ---: |
| H shift 0 only | 0.497 | 0.643 | 0.771 |
| \|H\| ≤ 1 | 0.773 | 0.883 | 0.932 |
| \|H\| ≤ 2 | 0.852 | 0.926 | 0.958 |
| \|H\| ≤ 4 | 0.911 | 0.952 | 0.973 |
| \|H\| ≤ 4, plus +1/+2 isotopes | 0.948 | 0.973 | 0.985 |
| Null: \|H\| ≤ 4 channels of the nearest-mass other molecule | 0.819 | 0.836 | 0.893 |

- **At unit resolution, reach is mostly not specific to the structure.** A different molecule
  of similar mass reaches 82% of the intensity. Coverage does not bind; ranking among
  ambiguous channels does.
  - The intensity-weighted mean is 4.3 valid channels per observed integer peak.
  - Channels reach 75% of the integers between their lowest m/z and M+4.
- **Isotope peaks above M** hold 1.6% of L1. Uncovered peaks below m/z 50 hold 0.5%.
- **Floor binning misplaces exact masses.** `batched_bin_func` puts m/z in `[k, k+1)`.
  - Floored exact m/z equals the nominal m/z for 86.6% of valid channels (median per
    molecule 94.7%); negative mass defects fall one bin low (CF3+ 68.995 → 68).
  - Rounding agrees for 100.0% (minimum 95.4% on one molecule).
  - So `cos_sim_1.0` is exact only when the channels are nominal.

### EI isotope envelopes are natural

The observed M+1/M and M+2/M were compared with the formula's natural nominal pattern. The
sample is 12,158 spectra whose molecular ion is at least 30% of the base peak.

| Molecules | n | M+1 obs/theory, median (IQR) | M+2 obs/theory, median (IQR) | M+2 within ±20% |
| --- | ---: | --- | --- | ---: |
| all | 12,158 | 1.006 (0.950-1.081) | 0.974 (0.826-1.045) | 64.0% |
| no Cl/Br | 10,067 | 0.999 (0.942-1.056) | 0.958 (0.760-1.066) | 57.0% |
| Cl/Br | 2,091 | 1.079 (1.003-1.341) | 0.994 (0.972-1.012) | 97.7% |

- The correlation of observed and theoretical M+2/M is 0.997.
- EI forms every ion from the whole natural population, with no isolation step. So each
  fragment should carry its full natural envelope.
- This is the opposite of Enveda MS/MS. There M+1 co-isolates at 0.93 (positive) and 0.48
  (negative), and halogen M+2 is a per-molecule switch, which closed the
  [isotope envelope plan](masster_isotope_envelope_plan.md).
- **A fixed natural prior, with no learned scale, is the EI model.**

## Design (Stage 1)

- **`masster_mass_mode: exact | nominal`**, default `exact`.
  - `nominal` makes each channel's m/z the integer sum of most-abundant-isotope atom masses
    (implicit H included), plus the H shift, plus the adduct's nominal delta; the electron is
    dropped.
  - It applies to the pool channels too, so the proposal loss scores the same channels.
  - Candidate features (`c_mass`) keep exact masses.
- **`masster_isotope_envelope: none | natural`**, default `none`; `natural` requires nominal.
  - Each emitted pathway row is split over nominal offsets by the natural isotope pattern of
    its channel composition: fragment atoms, the channel's hydrogens and the adduct atoms.
  - Each pathway's probability is conserved.
  - `masster_isotope_max_order` bounds the offsets. Rows below `masster_isotope_min_ratio` of
    the pathway are dropped and the rest renormalised.
  - Candidate, proposal and dense channel tensors stay monoisotopic.
  - Annotation rows gain `pred_pathway_isotope_offsets`.
- **Row merge.** In nominal mode, when no annotation is requested, a spectrum's prediction rows
  that share an m/z are merged: log-sum-exp of probabilities (`round_aggregate_peaks`).
  - This is exact for the Gaussian loss, the binned cosine and recall, which see only
    (m/z, probability).
  - Without it `e1` ran out of memory in the backward pass of the smoke below. The loss is
    targets × prediction rows, and isotope rows multiply the rows about 3-4×.
- **EI config, following `i1`:**
  - Data and conditioning: `nist23_ei` paths; `prec_types: [M]+`; `nce`, `ace`, `inst_type`
    and `frag_mode` off; no CMF.
  - Loss: `loss_tolerance_rel: null`, `loss_tolerance_abs: 0.5`, `inst_type_loss_tol: null`.
    With integer channels and targets, any tolerance below 1 matches exact integers only, and
    σ changes the loss by a constant.
  - Metric: `eval_mz_bin_res: [1.0]` and `checkpoint_metric: val_cos_sim_1.0_epoch/mean`.
  - `cos_hun_inst_tol` is not requested.
  - Isotopes: `masster_isotope_envelope: natural`.

## Implementation state (2026-09-30, working tree, uncommitted)

- **Keys** in [template.yml](../../../config/template.yml), all default off:
  - `masster_mass_mode: exact`;
  - `masster_isotope_envelope: none`;
  - `masster_isotope_max_order: 4`;
  - `masster_isotope_min_ratio: 0.001`.
  - They pass through `MassterPL` to `MassterModel` (`mass_mode`, `isotope_envelope`,
    `isotope_max_order`, `isotope_min_ratio`).
- **Code:**
  - `nominal_isotope_power_table` in `formula_utils.py` gives an element's truncated natural
    pattern for 0-2048 atoms, from RDKit abundances.
  - In `masster_model.py`:
    - The nominal branch of `_fragment_masses_and_valid_channels`: rounded atom masses, H at
      1 Da, rounded adduct deltas.
    - Non-persistent isotope buffers in `_setup_chemistry_buffers`, so every existing checkpoint
      loads strict.
    - `_isotope_rows`: the heavy-atom pattern per fragment and ion mode, then hydrogen per row.
    - Composition at step 4, then row expansion and the nominal row merge before the output.
- **Export guard.** `script_utils.py` raises `NotImplementedError` when an explanation export
  meets isotope rows, instead of mislabelling them.
- **Configs** in [config/masster/ei/](../../../config/masster/ei/):
  - `e1_nominal_natural_isotopes.yml` (the arm);
  - `e0_nominal_no_isotopes.yml` (the control; one key differs, `masster_isotope_envelope`).
- **Tests.** [test_masster_nominal_isotopes.py](../../../tests/test_masster_nominal_isotopes.py),
  16 tests:
  - the tables against natural abundance (Cl, Br2, F, Se);
  - nominal channels against mass-number sums (CF3+ 69, CCl3+ 117, `[M+H]+`, `[M-H]-`);
  - `_isotope_rows` against a direct convolution with Se, Br and H shifts;
  - every pathway's probability conserved across its isotope rows;
  - each isotope row's pattern rebuilt from its fragment's atoms, in both the cut-mask and
    the unique-fragment paths;
  - merged rows equal the per-pathway rows summed by (spectrum, m/z);
  - strict checkpoint loads both ways;
  - the argument checks.
- **Existing suites.** `test_masster_model.py` and `_build_model` gained a `**model_kwargs`
  pass-through. The MASSter, inference-annotation, G0-probe and formula suites pass.
- **GPU smoke** (RTX 2070 SUPER; 512-spectrum subsamples; 1 epoch, batch 64; CUDA candidates;
  exit 0 for both arms):

  | Smoke | `val_candidate_wrecall` | `val_wrecall` | peak GPU (nvidia-smi) |
  | --- | ---: | ---: | ---: |
  | `i1` keys on EI data, exact masses | 0.021 | 0.021 | — |
  | `e0` (nominal) | 0.954 | 0.954 | 7.3 GiB |
  | `e1` (nominal + natural isotopes) | 0.954 | 0.983 | 5.5 GiB |

  - `val_candidate_wrecall` counts monoisotopic channels, and `val_wrecall` includes isotope rows.
  - The one-epoch cosines (0.215 and 0.208) say nothing about the arms.

## First Fir attempt (2026-09-30): crashed at the first checkpoint

- The job logged `val_cos_sim_1_epoch/mean`, so `ModelCheckpoint(monitor='val_cos_sim_1.0_epoch/mean')`
  raised `MisconfigurationException` at the end of epoch 0.
- **Cause.** `run_training` overwrote every config value with the one read back from
  `wandb.config`. W&B returns an integral float as an int (`[1.0]` comes back as `[1]`,
  reproduced offline), so the metric name lost its `.0`. The local smokes ran with W&B disabled.
  Every `0.01` config is unaffected.
- **Fix** (working tree): `merge_wandb_config` in `runner.py` replaces a value only when it
  differs. Test: `test_merge_wandb_config_keeps_integral_floats`. An offline W&B round trip now
  keeps `val_cos_sim_1.0_epoch/mean` equal to the monitor.

## Open

- A NEIMS baseline on `nist23_ei_inchikey` at 1 Da.
- EI library metrics (the NIST m/z-weighted composite cosine, retrieval recall) are not built.
- Whether ±4 H is still wanted once isotopes stop needing the +1/+2 H channels.
- The inference annotation export does not label isotope rows (it raises).
- `data/proc/nist23_ei` and `data/split/nist23_ei_inchikey` are not known to be on Fir; no
  submitter stage exists for `config/masster/ei/`. Full-batch (512) memory and epoch time on
  an H100 slice are unmeasured.
- Isotope-labelled atoms (for example deuterium) get their element's most abundant mass in
  both mass modes, as before.


## Code audit (2026-09-30)

Audit only; no runtime code or experiment configuration changed. Scope: nominal channels,
natural isotope expansion, row aggregation, EI loss configuration, checkpoint metric naming,
and EI ingestion/export. No training, cluster access, or whole-dataset evaluation performed.

- **P2 — intermediate isotope truncation changes the natural envelope.**
  `nominal_isotope_power_table` in `src/fragnnet/utils/formula_utils.py` drops single-atom
  offsets outside the final window and truncates every convolution; `_isotope_rows` in
  `src/fragnnet/model/masster_model.py` also truncates between elements. With negative Se
  offsets, discarded probability can later return inside the window through positive
  offsets. An independent dictionary convolution using all RDKit isotope abundances,
  truncated only after the complete composition, reproduces the error at default order 4
  and minimum ratio 0.001. For composition SeBr3, the expected +3 share is 0.010203 but
  production emits none; the +4 share is 0.273381 versus 0.257881. For C2H6SeBr, the
  maximum absolute share error is 0.004109. Existing reference tests reuse the production
  power-table helper and repeat intermediate truncation, so passing them does not establish
  the full natural envelope. Next: retain sufficient intermediate support for cancelling
  offsets, crop the final pattern, and add an independent full-convolution regression.
- **P3 — two-column SHR peaks silently use intensity as m/z.**
  `parse_peak_line` in `preproc_scripts/eims/prepare_nist23_shr_eims.py` accepts two tokens
  but parses the final token as exact mass: `43 999` returns `(999.0, 999.0)` rather than
  `(43.0, 999.0)`. Require an actual exact-mass field before using the final token, or
  reject unsupported short lines. This does not affect the nominal EI arm. Sampling the
  first 1,000 entries in each of the four local SHR archives found 396,053 peak lines,
  all with four tokens; no current archive impact was demonstrated.

Validation: `tests/test_masster_nominal_isotopes.py` **16 passed**;
`tests/test_runner_config.py` **94 passed**, including integral-float preservation. Independent
isotope and parser probes ran with the configured FRAGNNET-GPU interpreter. Source inspection
confirmed MASSter supplies an OOS tolerance multiple of 1.0 to the loss, so the configured
0.5 Da window excludes neighboring integer peaks. Isotope annotation export remains explicitly
unsupported and raises, as documented above.

Provenance correction: the earlier “uncommitted/not submitted” banners are stale. The plan
already records a Fir attempt; local Git history includes `fe5dbc4c` (W&B EI fix) followed by
`a0131ddb`. The current dirty-file list contains unrelated MASSter SMARTS and analysis work;
this audit does not establish the latest remote run status. Next: address the P2 finding
before treating the Se-containing predictions as a natural-envelope calculation.


## Audit fixes (2026-09-30; implemented, uncommitted)

Both code-audit findings above are fixed in the working tree.

- Natural distributions now use the lightest isotope as their polynomial origin. Indices
  are nonnegative, so only coefficients above the final upper index are discarded before
  the complete composition has been convolved. Power tables likewise retain intermediate
  support that can return to the requested window.
- `_isotope_rows` keeps its tensor path for ions with nonnegative isotope offsets. Ions
  containing negative-offset elements (Se, B, Hg, Sn, Pb) use cached full-composition patterns,
  including the ion's adduct and hydrogen shift, before final cropping and normalization.
  Unique affected compositions are transferred to CPU for this calculation. CUDA correctness
  is tested; full-batch throughput and the cost on element-rich datasets remain unmeasured.
- The SHR parser uses a trailing exact-mass field only when at least four fields exist.
  Two-column and annotation-only rows keep their nominal m/z. Existing four-column inputs
  retain their behavior; no processed dataset was regenerated.
- The reference tests now independently convolve all RDKit abundances. Added regression
  cases cover within-element and cross-element cancellation, orders 1/4/6, H shifts, mixed
  affected/unaffected ions, CPU/CUDA forward/backward and probability conservation, and
  two-column SHR entry parsing.

Validation: **423 passed** across `test_masster_nominal_isotopes.py` (28),
`test_prepare_nist23_shr_eims.py` (5), `test_formula_utils.py` (49),
`test_masster_model.py`, `test_masster_pl.py`, and `test_inference_masster_annotation.py`
(the latter three total 341). CPU and CUDA tests both ran. Ruff on the five changed Python
files and `git diff --check` passed. Existing strict checkpoint-loading tests pass; derived
buffers remain non-persistent and default-off models retain their existing path.

Reproducibility: predictions from natural-envelope models containing negative-offset elements
change intentionally; compare old and corrected runs with their source revisions recorded.
No model training, cluster submission, full test suite, or validation-set evaluation ran.
Next: use the corrected code for the next authorized EI run and measure full-batch throughput.
