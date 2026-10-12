# Architecture map

> **Status:** REFERENCE · **Updated:** 2026-10-08 · MASShatter renamed MASSter; MASSter backend
> boundary updated 2026-09-17; other boundaries last verified against source on 2026-09-07.

## Entry points and data flow

1. [Preprocessing](../preproc_scripts/README.md) converts source spectra and molecules into
   processed tables, fragment features, and splits. Input identity, cleanup, split assignment,
   and DAG settings affect experiment reproducibility.
2. [Training entry point](../scripts/run_pl_model_fit.py) passes template, custom config,
   overrides, logging mode, and job ID to [runner.py](../src/fragnnet/runner.py).
   The runner loads configuration, seeds the run, builds datasets/loaders and the Lightning
   model, and configures logging and checkpoints.
3. [Datasets](../src/fragnnet/dataset/) assemble molecule, spectrum, and fragment inputs;
   `data_sampler.py` controls batching. The [batch reference](reference/dataloader_data.md)
   is a starting point; check actual consumers before changing a field or shape.
4. [Models](../src/fragnnet/model/) implement prediction; [Lightning modules](../src/fragnnet/pl_model/)
   connect predictions to losses and training/evaluation. FraGNNet uses fragmentation DAGs;
   MASSter has separate candidate construction in [frag/](../src/fragnnet/frag/) and its
   own model and Lightning module. GRAFF and MassFormer have separate package directories.
   The local ICEBERG fork was retired on 2026-10-11 in favor of external ms-pred
   ([reorganization plan](plans/repo_reorg_plan.md#decided-external-iceberg-dgl-free-main-environment)).
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

## Boundaries and canonical references

- Configuration: [config/](../config/) holds one `template.yml` that declares every model's
  keys, so each run resolves the same schema and `model_type` selects code rather than config.
  [config README](../config/README.md) documents the resolution order, where a key goes, and validation.
- Fragment generation: [frag/](../src/fragnnet/frag/) includes Python and Cython code;
  [DAG/prepass reference](reference/frag_gen_prepass.md) explains the FraGNNet workflow.
  Cython changes require checking the build in [setup.py](../setup.py) and rebuilding in the
  approved environment before testing; an existing binary is not evidence for changed source.
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
