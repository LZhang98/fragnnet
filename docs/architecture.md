# Architecture map

> **Status:** REFERENCE · **Updated:** 2026-10-12 · Package layout and dependency rule after the
> reorganization (Phases 0-7 of the [plan](plans/repo_reorg_plan.md)); MASSter backend boundary
> updated 2026-09-17.

## Entry points and data flow

1. [Preprocessing](../preproc_scripts/README.md) converts source spectra and molecules into
   processed tables, fragment features, and splits. Input identity, cleanup, split assignment,
   and DAG settings affect experiment reproducibility.
2. [Training entry point](../scripts/run_pl_model_fit.py) passes template, custom config,
   overrides, logging mode, and job ID to [runner.py](../src/fragnnet/runner.py).
   The runner loads configuration, seeds the run, builds datasets/loaders and the Lightning
   model, and configures logging and checkpoints.
3. [Datasets](../src/fragnnet/dataset/) assemble molecule and spectrum inputs, and the shared
   [data_sampler.py](../src/fragnnet/dataset/data_sampler.py) holds the generic batch samplers.
   Model-specific datasets and samplers live with their model (see
   [Package layout](#package-layout-and-dependency-rule)). The
   [batch reference](reference/dataloader_data.md) is a starting point; check actual consumers
   before changing a field or shape.
4. [Models](../src/fragnnet/model/) implement prediction; [Lightning modules](../src/fragnnet/pl_model/)
   connect predictions to losses and training/evaluation. Those two directories hold the shared
   base classes and the core models (NEIMS, ChemLM, precursor, GNN, CLIP). FraGNNet,
   which uses fragmentation DAGs, lives in [models/fragnnet/](../src/fragnnet/models/fragnnet/);
   MASSter, with its own candidate construction, in [models/masster/](../src/fragnnet/models/masster/);
   GRAFF and MassFormer in [baselines/](../src/fragnnet/baselines/) ([reference](reference/baselines.md)).
   The local ICEBERG fork was retired on 2026-10-11 in favor of external ms-pred
   ([reorganization plan](plans/repo_reorg_plan.md#decided-external-iceberg-dgl-free-main-environment)).
   MCES pretraining (`mces_pretraining`, a Siamese molecule encoder meant to initialize FraGNNet
   fine-tuning) was retired on 2026-10-12 because it did not help; its code and configs are at
   revision `13b1599d`.
   MASSter was named MASShatter until 2026-09-25. Pre-rename configs, checkpoints and W&B runs
   declare `model_type: mass_shatter` and `mass_shatter_*` keys; `migrate_legacy_masster_name`
   in [pl_utils.py](../src/fragnnet/utils/pl_utils.py) renames them wherever a config is loaded.
   MASSter also offers [FastGNN-style connected-fragment proposals](plans/masster/masster_fast_gnn_proposal.md),
   sharing its chemistry and intensity model while changing proposal score semantics.
   MASSter defaults to CPU candidate construction; its optional CUDA backend performs exact
   selection and topology for the supported modes. See the
   [backend and benchmark guide](guides/benchmark_masster_cpu.md) for limits and validation.
   MASSter can also share its context-free precursor encoder with optional
   [separate RTHead and CCSHead](plans/masster/masster_rt_ccs_plan.md), independently enabled
   with separate label sets. Their per-spectrum
   scalar outputs accompany the sparse MS/MS prediction; they default off.
5. [Inference](../scripts/run_inference.py) is documented in the [inference guide](guides/run_inference.md).
   [Evaluation scripts](../scripts/eval/) and [findings](findings/) connect artifacts to analysis.

## Package layout and dependency rule

```
src/fragnnet/
  dataset/ model/ pl_model/ utils/   core: shared, model-agnostic code and the core models
  runner.py registry.py hooks.py     core: entry point, model registry, hook interface
  fragmentation/                     bond-cut chemistry shared by FraGNNet and MASSter
  models/fragnnet/  models/masster/  the two owner models
  baselines/graff/  baselines/massformer/
```

- Core imports no model, baseline or `fragmentation/` code. `fragmentation/` imports core only.
  Each model or baseline imports core and `fragmentation/`, never another model or baseline.
  Tests may import any layer. `tests/test_reorg_boundary.py` enforces the rule on every import,
  module- and function-level, with the layer of each module taken from its directory
  (`tests/reorg_layers.py`).
- Core reaches a model only through [registry.py](../src/fragnnet/registry.py): `model_type`
  maps to `"module:Class"` strings for the Lightning module, the dataset and a hooks class,
  imported on first use. The registry also holds the aliases (`fragnnet` for `frag_gnn`, `clip`
  for `spec_mol_clip_pretraining`) and the retired types (ICEBERG, MCES pretraining). The legacy
  `mass_shatter` name never reaches it: `migrate_legacy_masster_name` in `utils/pl_utils.py`
  renames it to `masster` wherever a config or checkpoint is loaded.
- Model-specific steps that core runs go through [hooks.py](../src/fragnnet/hooks.py).
  `ModelHooks`, looked up by `model_type`, covers config and checkpoint migration, config
  validation, batch samplers and dataloader/trainer options, the dataset filter and the
  inference-config overlay. FraGNNet, MASSter and GRAFF override what they need; the core models
  and MassFormer use the no-op defaults. `ExportHooks`
  is a mixin on `FraGNNetPL` and `MassterPL` for the inference exporters' annotation columns.
  Hook modules import nothing model-specific at module level, so validating a config loads no
  model or compiled extension.
- Anything core reads, such as the annotation output keys, the size constants and the DAG
  storage format, stays in core even when only the fragment models produce it.
- Frozen records (findings, plans, `docs/history/`) cite the pre-reorganization paths
  (`src/fragnnet/frag/`, `src/fragnnet/graff/`, model files under `model/` and `pl_model/`).
  Resolve them in git history at `13b1599d`.

## Boundaries and canonical references

- Configuration: [config/](../config/) holds one `template.yml` that declares every model's
  keys, so each run resolves the same schema and `model_type` selects code rather than config.
  [config README](../config/README.md) documents the resolution order, where a key goes, and validation.
- Fragment generation: FraGNNet's DAG builders (Cython `compute_frags.pyx` and
  `multi_cut_bfs.pyx`, and `frag_utils.py`) are in [models/fragnnet/](../src/fragnnet/models/fragnnet/);
  MASSter's candidate construction (Cython `masster_core.pyx`, the CUDA `.cu` kernels next to the
  modules that load them) in [models/masster/](../src/fragnnet/models/masster/); the shared
  SMARTS prepass, hydrogen capping and cut-bond encoding in
  [fragmentation/](../src/fragnnet/fragmentation/). The
  [DAG/prepass reference](reference/frag_gen_prepass.md) explains the FraGNNet workflow.
  Cython changes require checking the build in [setup.py](../setup.py) and rebuilding in the
  approved environment (`python setup.py build_ext --inplace`) before testing; an existing
  binary is not evidence for changed source.
- Shared chemistry, feature, and spectrum operations: [utils/](../src/fragnnet/utils/).
  Scientific assumptions must be checked against code and the relevant findings or references.
- Tests: [tests/](../tests/), discovery in [pytest.ini](../pytest.ini), tooling in
  [pyproject.toml](../pyproject.toml). See the [session workflow](workflow/session.md) for checks.
- Data and artifacts: `data/`, `wandb/`, `output/`, `ckpts/`, and `saved_ckpts/` are ignored
  stores, not portable evidence by themselves. Record the actual run-specific paths and input
  identities. Do not move raw data, logs, or checkpoints into documentation. Layout, naming
  and citation rules for these stores: [artifact storage rules](rules/artifacts.md).

This is a map of stable boundaries, not a full file inventory. Update it when entry points
or interfaces change; retain detailed explanations in the existing reference documents.
