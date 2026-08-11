# Model Improvement Plan

> **Status:** ACTIVE — CANONICAL ROADMAP &nbsp;·&nbsp; **Updated:** 2026-07-24
>
> Start here. Defines what to run next; detailed analyses live in the linked docs.

**Status:** canonical roadmap as of 2026-07-24

**Scope:** NIST20, per-collision-energy spectra, InChIKey split

**Primary metric:** mean `cos_hun_inst_tol`

This document defines what to run next. Detailed analyses and historical experiments live in the
linked finding documents; they are not repeated here.

## Goal

Improve a single model's spectral similarity while protecting formula retrieval. Select experiments
on validation data, then evaluate the locked winner once on test data.

A candidate advances only when:

- its three-seed mean gain is at least **+0.005 `cos_hun_inst_tol`**;
- the paired per-spectrum bootstrap 95% confidence interval excludes zero;
- retrieval metrics do not materially regress;
- the result is attributable to a recorded config, checkpoint, code state, data split, and DAG.

One-seed runs may reject clear failures but may not promote a model.

## Current Diagnosis

The representative run `4ews8j6f` scores **0.7506**, while its same-split,
instrument-tolerance DAG oracle is about **0.9601**.

| Error source | Mean gap | Share of total error |
|---|---:|---:|
| Candidate/DAG coverage | 0.0399 | 16.0% |
| Intensity assignment and ranking | 0.2095 | 84.0% |

The main problem is choosing and ranking intensities over candidates already in the DAG:

- matched-peak Spearman is 0.613;
- the predicted base peak is wrong in 42.8% of spectra;
- 84.4% of predictions are flatter than their targets;
- 86.3% of spectra scoring below 0.3 have an oracle score of at least 0.3;
- global intensity sharpening does not improve the mean score;
- the model already tracks fragmentation extent across collision energy;
- FluxMPNN has no meaningful paired improvement over its baseline;
- deeper or more expressive fragment encoders increase overfitting in existing probes;
- ring-merge or depth 4 adds only about 0.015 oracle cosine under fixed tolerance.

Therefore, the next budget goes to reproducibility, objective alignment, regularization, and direct
ranking supervision—not larger encoders, more CE-depth machinery, or generic DAG expansion.

The measured global base-mass and adjacent-CE ranking experiment is specified in
[`global_fragment_intensity_ranking_experiment_plan.md`](global_fragment_intensity_ranking_experiment_plan.md).

**Immediate priority:** localize the ranking error using the local `9t63zils` baseline checkpoint.
This is read-only, requires no new training, and determines which later intervention is correctly
aimed. Run `4ews8j6f` is retained above as the measured oracle-decomposition reference, but its
checkpoint is not available locally.

## Evaluation Contract

Every experiment must use the same:

- train, validation, and test IDs;
- DAG and peak-matching tolerances;
- target preprocessing;
- inference and metric code;
- three-seed list.

Report:

- mean and median `cos_hun_inst_tol`;
- paired bootstrap 95% confidence interval versus the matching baseline seed;
- matched-peak Spearman and base-peak accuracy;
- results by instrument, fragmentation mode, and adduct;
- the project's formula-retrieval metrics, including correct- and wrong-candidate scores.

Every checkpoint and prediction HDF5 must store:

- full config snapshot and seed;
- git SHA;
- split and DAG fingerprints;
- preprocessing flags;
- checkpoint path;
- evaluation command.

Current saved HDF5 files have empty attributes, so run IDs alone are not adequate provenance.

## Ordered Plan

### 0. Localize the ranking error — immediate priority

Start with:

- config: `config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all.yml`;
- checkpoint: `saved_ckpts/nist20_inchikey_fraggnn_d3_ma_mi_all_9t63zils.ckpt`;
- split: validation.

Run:

```bash
python scripts/eval/node_vs_h_transfer_diagnostic.py ...
```

Classify true-intensity mass into:

- candidate coverage;
- `P(n)`: choosing the fragment node;
- `P(f|n)`: choosing its H-transfer/CMF/CRF ion channel;
- both node and ion channel ranked too low;
- both ranked adequately, indicating calibration or competition after candidate identification.

Repeat the classification for:

- all true peaks;
- the top 1, 3, and 5 true peaks;
- relative-intensity thresholds 0.01, 0.05, and 0.10;
- node/formula rank cutoffs 1, 5, and 10.

Weight the primary result by true intensity, not peak count. Report overall results and slices by
collision energy, instrument, fragmentation mode, and adduct.

The output already contains `spec_id`. Join it to `data/proc/nist20/spec_df.pkl` for fragmentation
mode and adduct slices; no second inference pass is needed.

If `P(n)` dominates, use the existing sibling diagnostic to split node error into:

- wrong ordering inside structurally related sibling sets;
- global node-ranking error outside measured sibling sets.

This second split matters because sibling targets currently explain only about 28.5% of intensity.
They are the right intervention only if they overlap a meaningful share of the measured node error.

**Decision gate:**

- sibling-local `P(n)` error dominates → direct sibling-ranking supervision;
- broader `P(n)` error dominates → fragment regularization and objective alignment;
- `P(f|n)` dominates → candidate-ion/factorized/adduct-gated modeling;
- candidates rank adequately but intensities are wrong → calibration/competition loss;
- the answer changes across major strata → use a stratum-specific correction instead of one global
  mechanism.

Finish this stage with one table assigning total true-intensity error mass to these branches and one
recommended training experiment. Do not start several architecture branches in parallel.

### 1. Freeze the baseline before new training comparisons

The ranking diagnostic does not need to wait for a new baseline. New training experiments do:

1. select the current production NIST20 config;
2. save an immutable experiment config for each of three seeds;
3. train and evaluate all three seeds under the evaluation contract;
4. replay one seed entirely from its recorded artifacts.

**Gate:** replayed metrics agree within numerical tolerance, and seed variance is known. Do not
promote a newly trained model before this passes.

### 2. Run existing low-cost experiments

#### 2.1 Objective alignment

Compare:

1. current cross-entropy baseline;
2. Hungarian cosine;
3. Hungarian JSS without square-root intensity transformation;
4. cross-entropy plus Hungarian auxiliary loss at weights 0.05, 0.1, and 0.2.

Keep cross-entropy as the primary retrieval objective in mixed-loss arms. Reconstruction objectives
can improve the correct spectrum while also raising wrong-candidate scores, so spectral similarity
alone is not a promotion criterion.

#### 2.2 Fragment-identity regularization

Run the existing configs:

- `fraggnn_d3_ma_mi_nist20_amolc_all_cc_atom_dropout_p00.yml`
- `fraggnn_d3_ma_mi_nist20_amolc_all_cc_atom_dropout_p01.yml`
- `fraggnn_d3_ma_mi_nist20_amolc_all_cc_atom_dropout_p02.yml`
- `fraggnn_d3_ma_mi_nist20_amolc_all_cc_atom_dropout_p03.yml`

If uniform dropout wins, add one carbon-preferential arm that preserves heteroatoms. Do not redesign
the representation before measuring this existing regularizer.

#### 2.3 Static radius-2 representation probe

Run:

```bash
python scripts/eval/fragment_static_r2_probe.py ...
```

Compare:

- current exact fragment-membership representation;
- static radius-2 bond environment;
- learned `k=1` bond encoder;
- lossy heteroatom charge descriptor.

Use the probe to select a training arm; do not treat an offline probe result as a model improvement.

**Stage gate:** promote only experiments satisfying the evaluation contract. Combine winners only
after each shows an independent gain.

### 3. Add direct sibling-ranking supervision when `P(n)` dominates

Use the design in [sibling_ranking_target_plan.md](sibling_ranking_target_plan.md):

1. build the offline alias-free target cache;
2. report cache size, group coverage, explained intensity, and retained-versus-dropped composition;
3. grade existing checkpoints with the sibling metric;
4. add soft `KL(q_obs || q_base)` loss at weights 0.01, 0.1, and 1.0;
5. run a shuffled-target control at the best weight.

The shuffled control is mandatory because an auxiliary loss may help only through generic
regularization.

**Gate:** dominant-sibling mismatch falls, validation cosine does not regress, and shuffled targets
do not reproduce the gain. Current alias-free groups explain about 28.5% of intensity, so this
intervention has bounded reach.

### 4. Pretrain the molecular encoder with ClassyFire atom labels

Before changing spectrum targets, evaluate large-scale molecular representation pretraining if the
ClassyFire molecule and atom-level corpus is ready. Follow
[classyfire_atom_pretraining_plan.md](classyfire_atom_pretraining_plan.md):

1. pretrain a FraGNNet-compatible `mol_embedder` on 1M molecules;
2. compare molecule-level, atom-level, combined, and combined-plus-cut/JEPA objectives;
3. transfer only compatible molecular-encoder weights;
4. compare frozen and reduced-learning-rate fine-tuning;
5. scale to 10M and then 100M only when downstream transfer continues improving.

**Gate:** three-seed downstream FraGNNet improvement over scratch, reduced train-validation gap,
and no retrieval regression. Pretraining accuracy is not a promotion metric.

### 5. Train against cleaner targets

Create separate, fingerprinted manifests for these arms:

1. remove or quarantine precursor-formula/adduct mass mismatches;
2. strip UWPR contaminant-only peaks from the 71,503 identified spectra;
3. apply the existing isotope-clean target option;
4. downweight extremely sparse spectra.

Measure isolated arms before combining filters. Do not change canonical preprocessing in place.

**Gate:** the model must improve on the unchanged validation set. Improvement only on cleaned
validation data does not establish a better production model.

### 6. Validate an ensemble, then distill

Ensemble the best three to five independent seeds under the current per-CE protocol. The previously
reported +4.2% result used merged spectra and a scaffold split, so it is not directly transferable.

If the current-protocol ensemble wins:

1. use it as a teacher;
2. distill into the best single-model architecture;
3. keep the student only if it retains most of the gain without retrieval regression.

## Execution Queue

| Order | Experiment | New model code? | Required output |
|---:|---|---:|---|
| 0 | Ranking-error localization on `9t63zils` | Diagnostic only | Error-mass decomposition and one next experiment |
| 1 | Three-seed attributable baseline | No | Reference mean, variance, replay |
| 2 | Loss/objective comparison | Mostly config | Best objective |
| 3 | `cc_atom_dropout` sweep | No | Regularization setting |
| 4 | Static radius-2 probe and selected arm | Probe exists | Representation decision |
| 5 | Sibling target cache and loss | Yes | Direct `P(n)` ranking test |
| 6 | ClassyFire atom-level molecular pretraining | Yes | Representation transfer gain |
| 7 | Clean-target arms | Data/config | Label-noise contribution |
| 8 | Ensemble and distillation | Distillation only | Deployment ceiling and student |

## Deferred Work

Do not spend the next experiment budget on:

- larger fragment encoders or deeper GNNs;
- more CE × depth, cut-count, or flow-stop modules;
- FluxMPNN or other edge-flow machinery;
- global post-hoc sharpening;
- depth 4, generic DAG expansion, or new ring rules as the first intervention;
- chemical-class oversampling without a class-specific causal diagnostic.

Re-open DAG work only when the intensity gap has materially shrunk or a subgroup is proven
coverage-limited under the correct instrument tolerance.

For retrieval-specific work, the next justified direction is a cross
real-spectrum/predicted-candidate contrastive objective with formula hard negatives. Start it only
after the reconstruction baseline is reproducible.

## Evidence

- [DAG oracle ceiling](../findings/dag_oracle_ceiling_nist20.md)
- [Intensity error diagnostic script](../../scripts/eval/intensity_error_diagnostic.py)
- [CE extent headroom finding](../findings/ce_extent_headroom_finding.md)
- [Fragment representation audit](fragment_bond_set_representation.md)
- [Sibling-ranking target design](sibling_ranking_target_plan.md)
- [ClassyFire atom-level pretraining plan](classyfire_atom_pretraining_plan.md)
- [FluxMPNN results and interpretation](fluxmpnn_plan.md)
- [NIST data-quality plan](nist_data_quality_plan.md)
- [Pairwise and retrieval loss findings](../reference/pairwise_loss.md)

Historical NIST23, merged-spectrum, and scaffold-split results use different protocols. They are
useful context but must not be compared numerically with this plan's NIST20 per-CE results.
