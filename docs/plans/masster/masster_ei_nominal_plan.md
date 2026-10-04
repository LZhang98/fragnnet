# MASSter on NIST23 EI (nominal mass, natural isotopes)

> **Status:** ACTIVE · **Updated:** 2026-10-01
> Stage 0 (feasibility) measured. Stage 1 built in the working tree (uncommitted): nominal mass
> mode, natural isotope rows, the `e1`/`e0` configs, 16 tests, GPU smokes of both arms.
> User decisions: nominal `nist23_ei`, not the NIST23 SHR set; select on `val_cos_sim_1.0`,
> not `val_cos_sim_0.01`; model isotopes; `loss_tolerance_abs` 0.5.
> Both EI arms completed 64 epochs in W&B; see the completion check below.
> Local isotope audit fixes are not verified in those remote runs.

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

Probe: [debug_masster_ei_nominal_coverage.py](../../../scripts/debug/ei/debug_masster_ei_nominal_coverage.py),
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
- Whether ±4 H improves trained cosine after isotope modelling; the coverage screen below
  says it should be retained for now, but a matched trained ablation has not run.
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


## W&B completion check (2026-09-30 session)

Read-only W&B API inspection found both arms **finished**, with complete histories for
`val_cos_sim_1.0_epoch/mean` over epochs 0–63. Each uploaded output log ends with
`Trainer.fit` stopping because `max_epochs=64` was reached. The only saved-config differences
are isotope-envelope mode, W&B name and tags; both use the same seed, nominal NIST23 EI,
0.5 Da loss tolerance, batch 512 and CUDA candidates.

| Arm | Run | Best 1 Da cosine (epoch) | Epoch 63 | Mean epochs 54–63 | Final weighted recall |
| --- | --- | --- | --- | --- | --- |
| e1, natural isotopes | [jeljdikj](https://wandb.ai/frag-gnn/frag-gnn/runs/jeljdikj) | 0.870173 (37) | 0.866724 | 0.867738 | 0.996870 |
| e0, no isotopes | [he3j5y3o](https://wandb.ai/frag-gnn/frag-gnn/runs/he3j5y3o) | 0.861611 (57) | 0.859522 | 0.859883 | 0.974399 |

The pre-specified last-ten-epoch comparison favors e1 by **0.007856** cosine. This is one
matched-seed pair, not a seed-replicated effect. Final monoisotopic candidate weighted recall
is similar: 0.974254 (e1) versus 0.974399 (e0). Do not compare their coverage-weighted CE losses
as if they had identical target support; e1's isotope rows cover additional target intensity.

Both metadata files identify NVIDIA H100 80GB HBM3 GPUs. W&B `_runtime` reports 10,672 s
(e1) and 10,292 s (e0); these resumed run histories also contain the earlier metric-name
failure, so those values are not clean compute benchmarks. The stale `val_cos_sim_1_*`
summary values coexist with the complete corrected `val_cos_sim_1.0_*` history; use the latter.

Neither saved config nor uploaded metadata supplies a Git revision. Inclusion of the local
isotope correction is therefore **unverified**; do not label this a validated fixed-code run.
Temporary read artifacts: `/tmp/eims_wandb_check/runs.json` (configs and full cosine histories),
and each run ID's subdirectory (metadata and output log). No W&B state, files or runs changed.
Next: establish remote source provenance before attributing results to the correction;
repeat seeds if a replicated isotope benefit is required.


## EI H-transfer width screen (2026-09-30 session; retain ±4 for now)

Repeated the Stage 0 5,000 validation-molecule coverage probe on the identical seeded sample.
The stand-in candidate pool consists of connected fragments from 0–2 cuts, with production
`[M]+` valence validity. This screen allows every valid nominal channel to reach m/z +1/+2
as a generous isotope **reach ceiling**, without checking actual isotopic abundance or the
0.001 row cutoff. It therefore tests whether narrower H ladders might suffice; it is not a
trained-model score or an exact e1 isotope simulation.

| Max absolute H shift | L1 target reach, no isotope | L1 target reach, +1/+2 reach ceiling | Mean per-spectrum cosine ceiling |
| --- | ---: | ---: | ---: |
| 0 | 0.4972 | 0.7212 | 0.8971 |
| 1 | 0.7735 | 0.8578 | 0.9630 |
| 2 | 0.8520 | 0.9040 | 0.9744 |
| 3 | — | 0.9339 | 0.9817 |
| 4 | 0.9108 | 0.9482 | 0.9847 |

At ±2, the generous isotope reach still leaves 4.42 percentage points of target L1
intensity that ±4 reaches, and a 0.0103 gap in the mean per-spectrum cosine ceiling. ±3
recovers most of the coverage, but ±4 adds another 1.43 points L1. The real isotope envelope
may narrow the lower ladders' reach further. Thus the existing ±4 setup retains measurable
coverage; dropping it is not supported by this screen. The upper-bound search space also
contains 9 H positions at ±4 versus 5 at ±2. The trained model's intensity assigned to
outer H channels was not logged, and no matched ±2/±3 retraining ran, so the effect on
actual validation cosine and runtime is still unknown.

Probe file and rows: `/tmp/eims_hshift_coverage.py` and
`/tmp/eims_hshift_coverage_5000/nominal_coverage_rows.csv`. This is a read-only analysis of
EI data; it did not change training configs or launch a run. Next decision, if optimizing
compute: compare trained ±3 and ±4 with matched seed and current corrected isotope code.


## EI SMARTS screen (2026-09-30 to 2026-10-01; complete, no admission)

User-authorized screen of three literature families: N-H N-benzoyl arylsulfonamide
carbonyl-O transfer, anisoyl fluoride CO/F transfer, and para-hydroxyazobenzene N2
extrusion. Research definitions stay in the probe pending evidence.

- Census train/val/test structures; score train and validation spectra only.
- Baseline: all connected components of 0–2 bond cuts; production `[M]+` CRF channels,
  valence-bounded ±4 H, natural isotopes (order 4, cutoff 0.001).
- Measure added L1 intensity, nominal collisions, product load, and 32 size-matched
  disconnected-assembly controls. Bootstrap by molecule; seek train/val replication.
- A positive bounded-pool screen is not proof of novelty against all connected atom
  sets, or a trained-model gain; stronger checks remain necessary before admission.
- Validate positive/negative SMARTS fixtures, product maps/formulas, isotope overlap
  and production-channel parity. Preserve test spectra.
- Starting revision: `e4c9496948b08085b124f91a1ceea0a6d7ce3429`. Existing edits in this
  plan, project state and the base-peak-errors script/tests belong to other work.
- Planned command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  /home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python
  scripts/debug/ei/debug_masster_ei_smarts_screen.py --workers 6`.
- Artifacts: `output/masster_ei_smarts_2026_09_30/`; seed 20260930; CPU only. The probe
  records input/source SHA256 hashes, package versions, starting diff and UTC times.
- Complete: no admission. Anisoyl and narrow azo products add zero validation mass support;
  sulfonamide transfer has seven train examples and none in validation. Final artifacts are
  `output/masster_ei_smarts_2026_10_01/`; the earlier directory is a superseded pilot.
- [Results, exact scope, validation and limitations](../../findings/masster/masster_ei_smarts_screen_2026_10_01.md).
  Next: retain candidates outside production; any intensity/pathway experiment needs a
  separate gate.


## EI ±6 H screen (2026-10-01; retain ±4)

Extended the same seeded 5,000-molecule validation coverage probe to ±5 and ±6 H.
The 0–2-cut stand-in pool, `[M]+` production valence rule, and generous +1/+2 isotope
reach ceiling are the same as in the preceding screen. This is candidate reach, not a
trained prediction or an exact natural-isotope threshold calculation.

| H limit | L1 reach without isotope offsets | L1 reach with +1/+2 reach ceiling | Mean per-spectrum cosine ceiling |
| --- | ---: | ---: | ---: |
| ±4 | 0.9108 | 0.9482 | 0.9847 |
| ±5 | 0.9228 | 0.9576 | 0.9870 |
| ±6 | 0.9289 | 0.9613 | 0.9878 |

The generous-isotope gains are +0.94 L1 points for ±5 and +0.37 for ±6;
together the mean per-spectrum cosine ceiling rises only 0.00314. The added nominal
channels reach some new intensity in 63.6% and 42.0% of spectra respectively, but
usually little per spectrum. Moving from ±4 to ±6 creates 13 H positions per fragment
instead of 9 (44% more dense channel slots), before isotope expansion and row merging.

The actual trained e1 run at ±4 already reports `val_wrecall_epoch/mean = 0.996870` with
natural isotope rows and selected fragments, leaving only 0.313 L1 percentage points
unreached by that metric. Wider channels could still change ranking of peaks already
reachable; that requires a matched trained ablation. The stand-in reach gain cannot be
substituted directly for e1's gain because its candidate pool is smaller and fixed.

**Decision:** keep ±4 in the EI config; do not promote ±6 on this evidence. If evaluating
wider H transfer later, compare a matched ±6 arm against the corrected ±4 isotope code
and record both actual cosine and time/memory. Read-only probe:
`/tmp/eims_hshift_coverage_6.py` and
`/tmp/eims_hshift_coverage_6_5000/nominal_coverage_rows.csv`. No config or model changed.


## e1 checkpoint annotation read (2026-10-01; ±4 used, ±6 unresolved)

Downloaded the best e1 checkpoint, [W&B `jeljdikj` epoch 37](https://wandb.ai/frag-gnn/frag-gnn/runs/jeljdikj):
`saved_ckpts/e1_nominal_natural_isotopes_jeljdikj/ckpt/model-epoch=037.ckpt`
(110,736,938 bytes; SHA-256
`b281a13a892fa5e3e4ed6a268c06284b1166a79dc60893e3546713d205b53831`).
Saved W&B config and download provenance alongside it. Weights load with `strict=True`.
The W&B config and locally resolved recipe agree except for the current template's empty
`spec_params.split_filters`; its absence in the run config does not select spectra. W&B still
has no source Git revision, so the source used in the original run is not established.

Ran annotated inference with the local corrected code (revision `e4c94969`) on 512
validation spectra, one per molecule, sampled at seed 20261001. Raw pathway rows were
pooled by integer m/z; each fragment ID counts once per peak regardless of its H shifts
or isotope rows. A 0.1%-of-base-peak cutoff applies only to the reported peak subset.

| Predicted-peak measure, peaks ≥0.1% of base | Value |
| --- | ---: |
| Peaks | 69,537 |
| Distinct fragments per peak, mean / median / p90 | 10.62 / 10 / 18 |
| Fragments contributing ≥1% of their peak, mean | 6.87 |
| Effective fragments `1 / sum(within-peak share²)`, mean | 3.93 |

Across all 129,900 emitted integer peaks, the raw mean is 8.81 fragments/peak.
At predicted m/z matching a measured peak, the measured-intensity-weighted mean is
10.82 fragments/peak. These are model predictions, distinct from the Stage 0 figure of
4.3 valid channels at observed peaks in a 0–2-cut stand-in pool.

The mean predicted intensity by H shift is 48.26% at H0, 3.10% at H±4 (1.996% at -4 and
1.105% at +4), and 8.34% at H±3/4 combined. On the same fixed predictions, removing
H±4 rows lowers the sample mean 1 Da cosine from **0.86905 to 0.86805**; removing all
H±3/4 rows lowers it to **0.86284**. H±4 improves cosine in 63.7% of sampled spectra.
This ablation removes rows after inference; it does not retrain or re-normalize channel
scores, and cannot tell what an expanded ±6 model would learn. The full sample cosine
is close to W&B's best-validation 0.87017, but this is a sample and uses current source.

Script: `scripts/debug/ei/analyze_masster_ei_checkpoint.py` (Ruff passed). Artifacts:
`output/masster_ei_jeljdikj_analysis_2026_10_01/sample512_cosine/` contains the sampled
inputs, per-peak rows, per-spectrum H shares/cosines, summary and checkpoint/sample
provenance. A 16-spectrum pilot and a first 512-spectrum pass passed row-alignment,
integer-mass and probability-conservation checks. No training, W&B write or test split use.

The per-peak table also answers whether a single fragment dominates a predicted peak.
For the 69,537 peaks at least 0.1% of the predicted base peak, the largest distinct
fragment supplies a median **45.4%** of that peak's intensity. It supplies >50% for
**43.0%** of peaks, ≥80% for **20.8%**, and ≥90% for **15.3%**. Weighting peaks by their
predicted intensity raises the mean largest-fragment share to **69.7%**. For the one
predicted base peak per spectrum (512), **82.2%** have a fragment above 50%, **73.0%**
above 80%, and the median largest-fragment share is **99.6%**. All H and isotope
pathways of the same fragment were combined before taking its share; the statistic does
not identify that fragment as the experimentally correct structure.
The distribution and threshold comparison are plotted in
`output/masster_ei_jeljdikj_analysis_2026_10_01/sample512_cosine/fragment_dominance.png`
(with a PDF beside it); regenerate with
`scripts/debug/ei/plot_masster_ei_fragment_dominance.py`.

At the user-requested **1%-of-predicted-base-peak** intensity filter, 31,441 peaks remain
(61.4 per spectrum) and retain 93.1% of predicted intensity. The largest fragment's
median share is 47.7%; 46.0% of peaks have one fragment above 50%, 24.5% above 80%,
and 18.6% above 90%. The raw mean number of fragments is 11.44, while the mean
effective number is 3.85. The predicted-base-peak statistics are unchanged.
The filtered figure is `output/masster_ei_jeljdikj_analysis_2026_10_01/sample512_cosine/fragment_dominance_1pct.png`
(with a PDF beside it). The plotting script accepts `--min-relative-intensity 0.01`.

**Decision:** keep ±4. Its outer channels carry material probability and improve the
fixed-model cosine. A ±6 trial would require a matched retraining to measure benefit;
the earlier reach screen alone does not establish one.


## Fragment-count penalty decision (2026-10-01; do not add yet)

On the e1@37 checkpoint's 512-molecule val sample, 1%-of-base predicted peaks have a
mean 11.44 distinct fragment contributors but a mean effective count of
3.85, computed from the fragment intensity shares within each peak. A leading fragment exceeds 50% on 46.0% of peaks. Peaks with a
leading fragment above 50% coincide with an observed m/z 86.2% of the time, versus 80.9%
for diffuse peaks. The former peaks also have higher predicted relative intensity on
average (11.8% versus 6.6% of base), so this association does not isolate a beneficial
fragment-sparsity effect.

The spectral target labels integer m/z intensity, not the connected fragment behind it.
MASSter merges the fragment rows at each integer before the spectral loss and metric.
Thus redistributing one predicted peak's probability among its fragment contributors
leaves that peak's spectral match unchanged; a conditional fragment-count or entropy
penalty would choose an explanation without supervision. The fixed natural isotope
patterns couple a fragment's intensities across several m/z, so forcing one contributor
could also perturb correct isotope envelopes. Do not add a fragment-count penalty from
this diagnostic alone.

If chemically specific fragment explanations become the objective, establish independent
fragment-level labels or a validated physical prior and measure explanation accuracy.
If spectral accuracy is the objective, first test a post-hoc concentration change at
fixed peak masses, then use a matched training ablation and validation cosine; candidate
count alone is not a success metric. No code or training config changed in this decision.


## Full test performance by molecule size (2026-10-01)

The e1 `jeljdikj` epoch-37 checkpoint was evaluated on **all 51,685 spectra** in the
`nist23_ei_inchikey` test split. Each spectrum contributes once. The mean 1 Da cosine
is **0.869984**. The analysis joins the test spectra to processed molecule `mw` and
RDKit total atom count after adding hydrogens. Molecular-weight bins are quintiles;
total-atom bins are 2–16, 17–32, 33–48, 49–64 and 65+ to expose the 64-atom cutoff.
The [PNG](../../../output/masster_ei_jeljdikj_analysis_2026_10_01/test_all_v2/size_performance.png),
[PDF](../../../output/masster_ei_jeljdikj_analysis_2026_10_01/test_all_v2/size_performance.pdf),
and [bin table](../../../output/masster_ei_jeljdikj_analysis_2026_10_01/test_all_v2/size_performance_bins.csv)
are generated by `scripts/debug/ei/plot_masster_ei_size_performance.py` from the
per-spectrum score table in the same directory.

| Descriptor | Smallest bin mean | Largest bin mean | Range across five bin means |
| --- | ---: | ---: | ---: |
| Molecular weight | 0.8729 (32–200 Da) | 0.8703 (371–1485 Da) | 0.8654–0.8754 |
| Total atoms including H | 0.8730 (2–16) | 0.8887 (65–209) | 0.8614–0.8887 |

There is no monotonic drop in mean cosine across these broad size bins. The largest
bins cover wide descriptor ranges and have higher score spread, so the plot does not
rule out a sparse extreme-size failure. The bars are normal-approximation 95% CIs
computed from per-spectrum standard errors; repeated spectra from a molecule are not
clustered. The test set contains 50,128 molecules. Twenty-two predicted intensity
totals differed from one by more than `1e-5` (range 0.999992–1.000032); the analyzer
renormalizes these before cosine. There were no zero-total predictions. No model or
data changed.


## Matched intensity configs (2026-10-01; prepared, not launched)

Three standalone configs isolate intensity-head changes from the current natural-isotope
baseline. Use the same source revision containing the isotope-offset correction for all arms.

| Arm | Config | Change from e2 |
| --- | --- | --- |
| e2 | [Fresh control](../../../config/masster/ei/e2_nominal_control.yml) | None; e1 recipe on current code |
| e3 | [Full attention](../../../config/masster/ei/e3_nominal_full_attention.yml) | `masster_intensity_transformer_attention: full` |
| e4 | [EMA](../../../config/masster/ei/e4_nominal_ema999.yml) | `ema_decay: 0.999` |

All use seed 397623, the existing InChIKey split, natural isotope envelopes, ±4 H,
256 fragments, batch 512, and 64 epochs. W&B group: `nist23_ei_intensity_2026_10_01`.
Attention and EMA defaults are explicitly pinned; no extra dropout or fragment-count
penalty is introduced. Compare mean validation 1 Da cosine over epochs 54–63,
best validation score, a common control-defined worst decile, and runtime/peak memory.

Validation: all three resolved through the production config loader and constructed
successfully on CPU. Excluding W&B labels, e2 equals e1, e3 differs only in attention,
and e4 differs only in EMA. No training or GPU batch-memory test was run. Next: when
training is requested, record source/environment provenance and check full-attention
memory at batch 512 before completing the matched runs.


## Stein weighted similarity on the full test split (2026-10-01)

The same e1 `jeljdikj` epoch-37 checkpoint was rerun on all 51,685
`nist23_ei_inchikey` test spectra. The new score uses the [RASSP Stein implementation](https://github.com/thejonaslab/rassp-public/blob/main/rassp/metrics.py)'s weighted
cosine: each nominal peak is transformed to `m/z**3 * intensity**0.6`, then the
predicted and measured vectors are compared by cosine. Both spectra are grouped at
integer m/z before the transform. The mean is **0.886845**, median **0.938095**;
10th and 90th percentiles are **0.715053** and **0.988242**. The [Stein size plot](../../../output/masster_ei_jeljdikj_analysis_2026_10_01/test_all_stein/size_performance_stein.png)
([PDF](../../../output/masster_ei_jeljdikj_analysis_2026_10_01/test_all_stein/size_performance_stein.pdf),
[bin table](../../../output/masster_ei_jeljdikj_analysis_2026_10_01/test_all_stein/size_performance_stein_bins.csv))
shows a monotonic fall from 0.9066 to 0.8495 across molecular-weight quintiles and
from 0.9202 to 0.8556 across total-atom groups (including H). This differs from the nearly flat
unweighted cosine curve because Stein's mass power emphasizes large m/z peaks.
Among the 46,435 test spectra with at most 64 total atoms, mean Stein similarity is
0.890374; among the 5,250 above 64 atoms, it is 0.855634. The
[RASSP inference code](https://github.com/thejonaslab/rassp-public/blob/main/rassp/run_rassp.py)
adds hydrogens before counting atoms. Its public FormulaNet limit is **48 total
atoms**; SubsetNet permits **64 total atoms**, although both were trained on molecules
with at most 48. The published 0.929 Stein score belongs to FormulaNet, so the
64-atom cutoff is not its inference limit. RASSP also accepts only H, C, N, O, F,
P, S and Cl. The NIST17 paper also filtered to at most **4096 unique fragment
formulae**, computed in [RASSP util.py](https://github.com/thejonaslab/rassp-public/blob/main/rassp/util.py)
as the product of `(element count + 1)` across the molecular formula, including H.
On our test split, **26,372 spectra of 25,558 molecules** satisfy these three
published NIST17 filters; their MASSter mean Stein score is **0.899167**. The
formula limit excludes 7,404 spectra beyond the earlier atom-and-element screen.
The public `run_rassp.py` overrides FormulaNet's inference formula cap to 32,768,
which is a different limit from the paper's NIST17 dataset filter. Per-spectrum
counts are in `output/masster_ei_jeljdikj_analysis_2026_10_01/test_all_stein/rassp_nist17_prefilter_rows.csv.gz`.
This remains a prefilter census, not a matched RASSP benchmark: training-set overlap
has not been checked and RASSP was not run on our test spectra.

The score table contains exactly the 51,685 test IDs. The rerun's mean unweighted
cosine is 0.8699833 versus 0.8699837 in the earlier full run; five individual spectra
differ by over 0.01, with a maximum 0.0475. The cause of those small-scope inference
differences is unresolved; aggregate means agree to 4.2e-7. Two focused metric tests,
Ruff, and whitespace checks passed. RASSP's published 0.929 FormulaNet Stein score
uses NIST17 and its own filtering/split, so it is not a matched model comparison.

## EI test similarity distributions by molecular weight (2026-10-01)

The [CE-style violin plot](../../../output/masster_ei_jeljdikj_analysis_2026_10_01/test_all_stein/ei_weight_bin_violin.png)
([PDF](../../../output/masster_ei_jeljdikj_analysis_2026_10_01/test_all_stein/ei_weight_bin_violin.pdf),
[bin summary](../../../output/masster_ei_jeljdikj_analysis_2026_10_01/test_all_stein/ei_weight_bin_violin_summary.csv))
The matching [sina plot](../../../output/masster_ei_jeljdikj_analysis_2026_10_01/test_all_stein/ei_weight_bin_sina.png) ([PDF](../../../output/masster_ei_jeljdikj_analysis_2026_10_01/test_all_stein/ei_weight_bin_sina.pdf)) shows density-jittered spectrum points in the same bins; the plotting helper samples at most 4,000 points per bin for legibility while the summary markers use all spectra.
shows the per-spectrum distribution of 1 Da cosine and Stein weighted similarity for the
full 51,685-spectrum EI test split. It uses 50 Da bins from 100 to 500 Da, with wider
bins in the sparse tails. Each violin follows `plot_predictions_hdf5.py`'s CE-bin style:
IQR bar, median circle, mean diamond and the number of spectra. Molecular weight is
the processed molecule table's `mw`; each spectrum contributes once. The reproducible
command is:

```bash
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python scripts/plotting/plot_masster_ei_weight_bins.py \
  --input-dir output/masster_ei_jeljdikj_analysis_2026_10_01/test_all_stein \
  --output-dir output/masster_ei_jeljdikj_analysis_2026_10_01/test_all_stein
```

For both metrics, all bin counts sum to 51,685 and their spectrum-weighted means match
the existing full-test means. The far-right bin contains only 72 spectra and should be
read as a sparse tail, not a precise population estimate.
