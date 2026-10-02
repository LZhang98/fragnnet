# MASSter JEPA spectrum-embedding auxiliary

> **Status:** ACTIVE · **Updated:** 2026-10-01 · Gate 0 accepted with the 1% floor (user,
> 2026-10-01). [Step 1 is built](#step-1-result-2026-10-01): an audited target table, default-off
> code with tests, and arm configs `j1a`/`j1b`/`j2`, smoke-tested on real data. No training run;
> Gate 1 needs the user's go-ahead.

## Objective and acceptance

Test whether a JEPA-style latent target improves MASSter's held-out spectra, mainly the
worst-predicted floor. The model predicts a fixed embedding of the measured spectrum from its own
molecule and context representation. The term is added to the `i1` loss, the predictor is dropped
at inference, and weight 0 must reproduce `i1` exactly.

- **Base and control.** `i1_inst_ob_obf` (`i50ig7rf`, NIST20 `[M+H]+`, `ORB`/`ORB_F`, 64 epochs,
  tail 0.7607), config `config/masster/mh_ft_memorization_route_2026_09_24/i1_inst_ob_obf.yml`.
- **Primary read.** Tail mean of `val_cos_sim_0.01_epoch/mean` over epochs 54-63, against `i1`.
  The seed spread is 0.0042, so a single-seed delta below +0.005 is unresolved
  ([memorization route plan](masster_memorization_route_plan.md)).
- **Floor read.** Mean `cos_hun_inst_tol` on a floor chosen by an independent checkpoint
  (`fa1`@63, 290 spectra), never by the arm's own run
  ([worst predictions](../../findings/masster/masster_i1_worst_predictions_2026_09_30.md)).
- **Promotion.** Two seeds, mean tail ≥ +0.005 over `i1`, both seeds positive, and no floor loss.
  Also compare by wall clock: best-by-hour first, epoch-matched second.

Out of scope: retrieval embeddings (MassSpecGym/Spectraverse) and the combined PCDL + NIST23 +
Enveda-180 split, which the runner cannot load yet
([combined dataset plan](../combined_pcdl_nist23_enveda_180_dataset.md)).

## Established facts

**Literature (checked 2026-10-01).** Graph-JEPA, Mol-JEPA and DreaMS were read from abstracts or
repository pages only.

| Source | Relevant result |
| --- | --- |
| [I-JEPA](https://ar5iv.labs.arxiv.org/html/2301.08243) | EMA target encoder (momentum 0.996 → 1.0), L2 in representation space. Masking decides the result: multi-block 54.2% top-1, random patches 17.6%, single block 20.2%. Representation targets 66.9% against pixel targets 40.7%. |
| [LeJEPA](https://arxiv.org/html/2511.08544) | SIGReg (isotropic-Gaussian regularizer from 1,024 random 1-D projections and an Epps-Pulley test) replaces EMA, stop-gradient and the predictor. Default λ = 0.05, batch ≥ 128. In-domain pretraining beat DINOv2/v3 transfer on small domain datasets. Training loss tracks downstream accuracy (Spearman ≈ 0.85). |
| [LLM-JEPA](https://arxiv.org/html/2509.14252) | Next-token loss plus λ · cosine distance between the predicted and actual embeddings of two paired views. This is the closest template for an auxiliary term on a supervised model. It reports regex 71.10% against 51.95% and less overfitting, needs "non-trivial views", and doubles training compute. |
| [JEPA fine-tuning audit](https://pith.science/paper/2605.15394) | Independent replication on the same regex task (Llama-3.2-1B, 22 auxiliaries, 5 seeds): no gain survives multiple-comparison correction (full fine-tuning +0.04 pp, p = 0.96; +0.52 pp, p = 0.28). JEPA changed hidden-state geometry, not accuracy. |
| [Graph-JEPA](https://arxiv.org/abs/2309.16014), [polymer JEPA](https://arxiv.org/pdf/2506.18194) | Masked METIS-subgraph prediction. Gains mainly when labels are scarce; NIST20 is not. |
| [DreaMS](https://github.com/pluskal-lab/DreaMS) ([paper](https://www.nature.com/articles/s41587-025-02663-3)) | Spectrum transformer pretrained without labels on GNPS GeMS spectra (30% masked peaks, intensity-proportional, plus retention order). Public weights; `dreams_embeddings()` returns 1,024-d vectors from `.mgf`. |
| [SpecBridge](https://arxiv.org/pdf/2601.17204) | DreaMS mapped into frozen ChemBERTa space for retrieval. **Withdrawn** over a preprocessing/evaluation bug; not evidence. |

No JEPA for predicting spectra from molecules was found.

**Project evidence.**

- The encoder is not where the floor comes from. Frozen `i1` encoder embeddings predict cut sites
  on new molecules; the bond head does not use that information on the floor
  ([worst predictions, encoder or head](../../findings/masster/masster_i1_worst_predictions_2026_09_30.md#result-encoder-or-head-follow-up-2026-09-30)).
- Of `i1`'s 0.238 val deficit, 64% is allocation among proposed fragments within a band
  ([residual structure](../../findings/masster/masster_i1_residual_structure_2026_09_28.md)).
- In `i1` the bond head reads `.detach()`ed encoder outputs, so the shared encoder is trained only
  through the intensity head ([head-input asymmetry](../../findings/masster/masster_head_input_asymmetry_2026_09_22.md)).
- The +0.003 gate used before 2026-09-27 is about one standard error
  ([memorization attempts audit](../../findings/masster/masster_memorization_attempts_audit_2026_09_27.md)).

**What follows for the design.** A target trained only on our NIST20 pairs re-encodes what
MASSter already fits peak by peak, which is the redundant setting where the audit found nothing.
A frozen DreaMS target brings in information from unlabelled spectra MASSter never trains on.
Because it is frozen it cannot collapse, so it needs no EMA, stop-gradient or SIGReg. Attach
points follow the deficit: the shared encoder, and the allocator where 64% of the deficit sits.

## Plan

### Gate 0 — offline headroom (no training)

1. **Environment.** Install DreaMS in its own conda environment, not `FRAGNNET-GPU`, after user
   confirmation. Record the package version and checkpoint hash. Use the self-supervised
   pretrained weights only. Check whether any fine-tuned DreaMS head saw NIST20 labels; if one did,
   it is excluded. Verify DreaMS's peak cap, precursor input and ion-mode coverage; `i1` is
   `[M+H]+` only.
2. **Embed.** Write `.mgf` files and embed: all NIST20 `ORB`/`ORB_F` `[M+H]+` measured spectra, plus
   `i1`@59 predicted val spectra (export with inference if no full-val export exists; check the
   workstation for other training runs first). Store a side table keyed by
   `spec_id` under `data/proc/nist20/`, not a new `spec_df`: one `spec_df` per dataset.
3. **Noise floor.** Val replicate pairs: same molecule, same instrument, same NCE, distinct
   spectra. Their DreaMS cosine distance is the floor.
4. **Pre-registered pass rule.** Both must hold, or the plan stops `CLOSED NULL`:
   - median DreaMS distance (`i1` predicted, measured) ≥ 2× the median replicate distance;
   - per-spectrum Spearman between that distance and `cos_hun_inst_tol` ≤ −0.4 on all val, and the
     distance is higher on the `fa1`@63 floor than off it.

   If DreaMS cannot see `i1`'s errors, a term built on it cannot target them.

### Gate 0 result (2026-10-01)

**By the rule as written, Gate 0 fails.** The pre-registered decision variant uses DreaMS's default
input, the top 100 peaks. It passes R1 and the floor rule but fails the correlation rule: ρ is
+0.140, the wrong sign. The cause is how predicted spectra were presented, not DreaMS. The variant
that drops peaks below 1% of the base peak on both sides was written before the read as a
sensitivity check. It passes every rule by a wide margin. Whether to accept it is open.

| Read | Valid | Median d(pred, meas) | Median d(replicate) [95% CI] | R1 ratio (≥ 2) | Spearman with `cos_hun_inst_tol` (≤ −0.4) | Median d, floor vs off-floor |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **top 100, raw (decision)** | 51,242 | 0.414 | 0.024 [0.022, 0.028] | 17.1 | **+0.140** [0.120, 0.159] | 0.526 vs 0.414 (n = 287, p = 2e-40) |
| 1% floor, raw | 46,618 | 0.172 | 0.024 [0.021, 0.029] | 7.1 | −0.766 [−0.775, −0.756] | 0.438 vs 0.170 (n = 274, p = 2e-105) |
| top 100, mean-centred | 51,242 | 0.491 | 0.028 [0.025, 0.032] | 17.5 | +0.121 [0.102, 0.139] | 0.625 vs 0.491 |
| 1% floor, mean-centred | 46,618 | 0.206 | 0.028 [0.025, 0.035] | 7.3 | −0.773 [−0.782, −0.764] | 0.537 vs 0.205 |
| post hoc: measured top 100, predicted 1% floor | 47,921 | 0.190 | — | — | −0.654 | 0.426 vs 0.189 |

d is cosine distance between DreaMS embeddings. Replicates are 1,426 same-molecule,
same-instrument, same-NCE val pairs (485 molecules); 1,156 pairs with the 1% floor. CIs are
1,000 bootstrap resamples over molecules. The floor is `fa1`@63 `cos_hun_inst_tol` < 0.05.
p is a one-sided Mann-Whitney test.

**Why the decision variant fails.**

- In their top 100, predicted spectra show DreaMS a median of 100 peaks; measured spectra show 21.
- A median of 87 of those predicted peaks are below 1% of the base peak. They carry 2.3% of the
  intensity.
- DreaMS embeds every peak as a token. This variant therefore compares MASSter's dense
  low-intensity tail with library spectra, not predicted intensities with measured ones.
- With the 1% floor the counts are 16 predicted and 13 measured. Flooring only the predicted side
  already gives ρ = −0.654, so the predicted tail alone explains the failure.
- The decision variant was mis-specified for predicted spectra: DreaMS's default suits library
  spectra, and it was chosen without checking MASSter's output density.

**Other reads.**

- **Scale.** Replicates are 0.024 apart, `i1`'s predictions 0.17 (1% floor), different molecules
  0.85. DreaMS separates NIST20 Orbitrap molecules cleanly, so there is no domain-shift failure
  and `j3`'s trigger does not fire.
- **Instrument.** The same on `ORB` and `ORB_F` (ρ −0.769 and −0.755, 1% floor).
- **Floor noise.** Replicates of floor molecules are twice as far apart as overall (0.045, 26
  pairs). That is still a tenth of the floor's prediction distance, so in DreaMS space the floor
  errors are not replicate noise.
- **Exclusions.** The 1% floor leaves 4,624 spectra with fewer than 3 peaks. They are easy (median
  `cos_hun_inst_tol` 0.994) and less often on the floor (0.28% against 0.59%).
- **Centring.** Removing the shared mean direction changes nothing.

**What Gate 0 does not show.** It does not show that a DreaMS target adds information beyond the
peak loss. ρ = −0.77 says DreaMS distance largely re-expresses the spectral dissimilarity the
loss already reduces. Only Gate 1 tests whether the target adds anything.

**For Step 1.** Training embeds measured spectra only. They have no dense tail (median 21 peaks),
so the artifact does not reach the targets.

**Decision (user, 2026-10-01): "do 1% floor".** Gate 0 is read on the 1% floor variant, and the
training targets use the same floor, so they live in the representation Gate 0 validated.

**Provenance.**

- **Environment.** Conda env `dreams` (Python 3.11.0, torch 2.2.1+cu121, dreams 1.0.0), separate
  from `FRAGNNET-GPU`. Editable install of `pluskal-lab/DreaMS` at `dbec3a0b`.
- **Weights.** `ssl_model.ckpt` from Hugging Face `roman-bushuiev/DreaMS`, 1,392,710,212 bytes,
  sha256 `4b73da583a4b4e4abef4bb3ab496dc12f716ed484ea0e4066ad45d6952856fef`.
- **Excluded weights.** `embedding_model.ckpt` is what `dreams_embeddings()` loads by default.
  The repository's `dreams/training/fine_tune.sh` trains it as a contrastive head on labelled MoNA
  `[M+H]+` spectra under DreaMS's own Murcko split, and a commented-out path names a NIST20 + MoNA
  set.
- **Embedding.** The 1,024-d last-layer precursor-token output (`embs[:, 0, :]`, as in DreaMS's
  `get_embeddings(precursor_only=True)`). Format A takes charge +1 only, m/z ≤ 1000 and at least
  3 peaks; the limits were applied by hand. No val precursor is above m/z 1000. 778 measured
  spectra with fewer than 3 peaks are excluded.
- **Inputs (no new inference).** `i1`@63 val predictions and targets come from
  `output/i1_worst_predictions_2026_09_30/fa1/i1_63_val.pkl` (52,020 spectra). This uses @63 in
  place of the planned @59 because that export already existed. @63 is the last epoch, and the
  `fa1`@63 selector stays independent of it.
- **Scores and metadata.** Scores from `i3/scores.pkl`; precursor m/z and instrument from
  `subset_spec_df_val_all.pkl`. The join was checked: the export's own reference cosine matches
  the score table to 1e-5 on every spectrum.
- **Coverage.** Gate 0 embedded val only. The train and val target table came later, in
  [Step 1](#step-1-result-2026-10-01).
- **Files.** Scripts and logs are in `output/dreams_gate0_2026_10_01/` (local, ignored):
  `prep_gate0_inputs.py`, `gate0_embed.py`, `gate0_analyze.py` and `gate0_followup.py`, with
  their `.log` files. Embeddings are in `gate0_embeddings.npz` (float16). One checkpoint and no
  seed replicate.

### Step 1 — implementation (default off)

- New template keys in `config/template.yml`: `masster_jepa_target_fp` (side table),
  `masster_jepa_loss_weight` (default 0), `masster_jepa_source` (`encoder` | `allocator`),
  `masster_jepa_predictor_hidden_dim`. The template fills new keys for old checkpoints, so no
  `.get(default)` reads.
- The dataset attaches the target embedding per spectrum. Training spectra only contribute to the
  loss; val logs the JEPA cosine as a diagnostic.
- Predictor: MLP to 1,024-d, loss λ · (1 − cos). Input for `encoder`: mean-pooled
  `precursor_encoder` atom embeddings per molecule concatenated with `context`. Input for
  `allocator`: mean over candidate tokens of the intensity-transformer output per spectrum.
- Term added in `MassterPL` `loss_fn` next to the other auxiliaries.
- Tests: weight 0 gives `i1`'s loss bit-identically; with `encoder`, gradient reaches
  `precursor_encoder` and `context_encoder`; the predictor is absent from the inference output.
- Smoke: one real-data CUDA forward/backward; log the ratio of auxiliary to primary gradient norm
  at initialization, and the epoch time against `i1`.

### Step 1 result (2026-10-01)

Built and validated; no training run. The design came from six code scouts, two independent
designs and a judge. One design's prototype was reviewed line by line and ported. Where the build
departs from the bullets above, it says so.

**Target table.** `data/proc/nist20/jepa_dreams_ssl_1pct_floor.npz` (434 MB, local only).

- **Contents.** One row per train and val spectrum of `nist20_inchikey`, 209,698 in all. No test
  spectra, so no test target can exist.
- **Rule.** On raw float64 `spec_df` peaks: precursor m/z ≤ 1000, top 100 peaks, m/z ≤ 1000,
  intensity ≥ 1% of the base peak (`>=`), at least 3 peaks.
- **Tie rule.** Every NIST20 base peak is exactly 999.0, so the cut is 9.99, and 12,158 train and
  4,127 val peaks sit exactly on it. `>=` keeps them. Gate 0's float32 inputs classified 16 val
  spectra differently, which does not change its conclusions.
- **Coverage.** 196,521 spectra have a target: train 147,762 of 157,678 (93.7%), val 48,759 of
  52,020. The rest get a zero row and are masked.
- **Why raw peaks.** A scout showed that `i1`'s training target equals the raw `spec_df` peaks
  up to scale and float32 rounding: every val spectrum and a 49,205-spectrum train sample, with
  no dropped or merged peaks.
- **Producers.** Built by `preproc_scripts/jepa_targets/prep_inputs.py` (FRAGNNET-GPU) and
  `build_targets.py` (`dreams` env, local `ssl_model.ckpt`). The builder spot-checks 64 rows
  against one-spectrum forwards and records the md5 of `spec_df` and the split files.
- **Audit.** `audit_targets.py` passed. It re-derives ids, peak counts, m/z sums and the rule
  from `spec_df`, and compares val rows with Gate 0: median cosine 1.0000000. All 1,835 rows
  below 0.9999 have a peak on the 1% cut. Log: `output/dreams_gate0_2026_10_01/canonical_build.log`.

**Code (default off).** Files: `config/template.yml`, `src/fragnnet/utils/jepa_utils.py` (new),
`src/fragnnet/model/masster_model.py`, `src/fragnnet/model/bond_fragment_heads.py`,
`src/fragnnet/pl_model/masster_pl.py`, `scripts/setup_job_cc.py`.

- **Keys.** `masster_jepa_loss_weight` (0), `masster_jepa_source` (`encoder`), `masster_jepa_target_fp`
  (null), `masster_jepa_predictor_hidden_dim` (512).
- **Weight 0.** No module, no table read, no extra key, and the same state_dict and RNG stream.
  The keys are still validated.
- **Target attach.** The target is attached in `MassterPL.on_before_batch_transfer`, not in the
  dataset as the bullets above said. There, ids and peaks are CPU tensors, so the lookup needs no
  device sync, and the dataset shared by 7 model types stays untouched.
- **Lookup check.** Each lookup checks the spectrum's peak count and float32 m/z sum against the
  table and raises on a mismatch or an unknown id.
- **Pooled input.** The model returns the pooled vector only on request
  (`jepa_source=` on `MassterModel.forward`, `return_pooled_tokens=` on the intensity head). The
  pooling uses `index_add`, not `bincount`, so it adds no CUDA sync.
- **Predictor.** `LayerNorm → Linear(512) → SiLU → Linear(1024)`, built last in `_setup_model`
  under `fork_rng`, and applied only inside `loss_fn`, so inference never touches it. It has
  788,992 parameters for `encoder` and 657,408 for `allocator`, against 9,215,701 in the model.
- **Training only.** The term is added to the training loss only. Validation logs `jepa_loss`,
  `jepa_cos` and `jepa_target_fraction` but its loss is unchanged.
- **Masking.** Masked spectra stay in SpectrumPL's fixed denominator, so the effective weight is
  λ × 0.937.
- **Guards.** A positive weight requires a table path, `merge: false`, `unique_id: true`, no
  proposal-only phase and `eval_test_split: False`. A training batch without a target raises.
- **Checkpoints.** Loading is strict, so a JEPA checkpoint loads only into a JEPA config.
- **Cluster staging.** `setup_job_cc.py` stages the table path for the node. The cluster project
  copy must hold the table first.

**Tests.** `tests/test_masster_jepa.py` (new) and three staging tests in
`tests/test_setup_job_cc.py`. They cover:

- table layout and lookup checks;
- masked rows with zero loss and zero gradient;
- bit-identical loss and gradients at weight 0;
- unchanged init and RNG when the term is enabled;
- gradient reach by source;
- shuffled batches whose own targets score cosine 1, with rolled targets as the negative control;
- the hook inside a real Lightning `fit`.

On the workstation, ruff is clean and 532 tests pass across the new file and the affected
MASSter, config and staging suites. Full-suite result: see Current state.

**Arms.** `config/masster/mh_jepa_2026_10_01/`: `j1a_jepa_encoder_w0p1.yml`,
`j1b_jepa_encoder_w1.yml` and `j2_jepa_allocator_w1.yml`. Each is `i1` plus W&B names and the
four JEPA keys, and nothing else differs. All three resolve and build.

**Real-data smoke.** CUDA, 8 val spectra, 2 of them without a target; fresh weights, no
optimizer step. Script: `output/dreams_gate0_2026_10_01/step1_smoke.py`; logs `step1_smoke.log`
and `step1_timing.log`.

- **Masking.** Targets arrive as float16 on the GPU, and masked spectra score exactly 0.
- **Loss.** The training loss rises by exactly λ × mean `jepa_loss`, and the eval loss is
  identical across arms.
- **Gradients.** At λ = 1 on a fresh model, the auxiliary-to-primary gradient-norm ratio is:

  | Source | Precursor and context encoders | Intensity head |
  | --- | ---: | ---: |
  | `encoder` | 0.120 | 0 |
  | `allocator` | 0.115 | 0.072 |

  All gradients are finite.
- **Cost.** A warm step at batch 8 on the RTX 2070 takes 147.5 ms for `i1`, and 157.3 ms and
  158.0 ms for `j1b` and `j2`. The real per-epoch cost has to be measured on Fir.

### Gate 1 — single-seed screen (seed 397623, 64 epochs)

| Arm | Change from `i1` |
| --- | --- |
| `j1a` | `source: encoder`, λ = 0.1 |
| `j1b` | `source: encoder`, λ = 1.0 |
| `j2` | `source: allocator`, λ = 1.0 |

An arm with a tail delta ≥ +0.005, or a floor gain beyond the independent-selector spread, goes
to Gate 2. Also report the train/val JEPA-cosine gap: the LLM-JEPA claim is less overfitting, so
a passing arm should narrow it.

### Gate 2 — replicate

Second seed 811237 for each passing arm, plus an `i1` replicate at that seed if none exists.
Apply the promotion rule above.

### Deferred

- **Own target encoder (`j3`).** Train the existing
  [`SpectrumEncoder`](../../../src/fragnnet/model/spectrum_encoder.py) jointly as the target with
  SIGReg (λ = 0.05) and no EMA, masking m/z windows or fragment subgraphs, not random peaks (the
  I-JEPA ablation). Run only if Gate 0 fails because of DreaMS domain shift. If DreaMS passes
  Gate 0 and fails Gate 1, `j3` is unlikely to help, given the audit, and is not run.
- **Molecule-side Graph-JEPA pretraining.** Not planned; NIST20 is not label-scarce.

## Constraints and risks

- **Indirect leakage.** GeMS may contain unlabelled spectra of NIST20 val/test molecules. Only
  training-spectrum targets enter the loss, so the exposure is through DreaMS's embedding geometry
  alone. Record it; do not use val JEPA cosine as a selection metric.
- **CE.** DreaMS takes no collision energy. The measured spectrum's embedding carries it
  implicitly, and the predictor receives it through `context`.
- **Cost.** Targets are precomputed, so the overhead is one MLP; LLM-JEPA's 2× compute applies to
  `j3` only.
- This plan does not authorize installation, inference exports or cluster jobs; each needs the
  user's go-ahead ([session workflow](../../workflow/session.md)).

## Current state and next check

Gate 0 is accepted with the 1% floor, and Step 1 is built ([result](#step-1-result-2026-10-01)).
Full test suite (2026-10-01, workstation): 4,969 passed, 1 skipped, 2 xfailed. The 34 failures
and 22 errors were all `tmp_path` writes after the 3.7 GB `/tmp` filled. Their five files pass
84/84 when rerun with `--basetemp` on the home disk.

Gate 1 approved by the user (2026-10-01) and prepared, not yet submitted.

- **Launcher.** `scripts/manual_slurm/fir/submit_masster_jepa.sh gate1` only generates scripts.
  It checks that each config matches `i1` and that the Fir checkout has the JEPA code, the table
  at md5 `576c115361781cf3f8b37df1d96fbad8`, the ORB / ORB_F `spec_df.pkl`
  (`ce029870badc9e9e05935a4424f44c8d`, the 2026-10-01 rename of `bdc009bb…`; same peaks, so
  the table still matches) and the wheels.
- **Jobs.** `msh_jepa_j1a_encoder_w0p1`, `msh_jepa_j1b_encoder_w1` and `msh_jepa_j2_allocator_w1`
  on `fir:h100-3g.40gb` for 5:00:00.
- **Blocker.** Fir needs MFA, so the agent cannot copy or submit non-interactively.
- **Next steps** (on Fir unless noted):
  1. Get the JEPA code into the Fir checkout.
  2. From the workstation, copy the table to `project/feiw/frag-gnn/data/proc/nist20/`.
  3. Run the launcher.
  4. Run `sbatch` for each generated script, and record the job IDs and W&B runs here.
  5. Read Gate 1 as above, including wall clock per epoch against `i1`.
