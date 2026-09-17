# Architecture map

> **Status:** REFERENCE · **Updated:** 2026-09-17 · MASShatter backend boundary updated;
> other boundaries last verified against source on 2026-09-07.

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
   MASShatter has separate candidate construction in [frag/](../src/fragnnet/frag/) and its
   own model and Lightning module. GRAFF, ICEBERG, and MassFormer have separate package directories.
   MASShatter also offers [FastGNN-style connected-fragment proposals](plans/masshatter_fast_gnn_proposal.md),
   sharing its chemistry and intensity model while changing proposal score semantics.
   MASShatter defaults to CPU candidate construction; its optional CUDA backend performs exact
   selection and topology for the supported modes. See the
   [backend and benchmark guide](guides/benchmark_masshatter_cpu.md) for limits and validation.
5. [Inference](../scripts/run_inference.py) is documented in the [inference guide](guides/run_inference.md).
   [Evaluation scripts](../scripts/eval/) and [findings](findings/) connect artifacts to analysis.

## Boundaries and canonical references

- Configuration: [template.yml](../config/template.yml) declares hyperparameters;
  [config README](../config/README.md) documents experiment configuration.
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
  identities. Do not move raw data, logs, or checkpoints into documentation.

This is a map of stable boundaries, not a full file inventory. Update it when entry points
or interfaces change; retain detailed explanations in the existing reference documents.
