# Codebase correctness audit

> **Status:** FINDING · **Updated:** 2026-10-13
> Six findings and the three rechecked configuration defects repaired, uncommitted; unfiltered CPU suite passes.

## Scope and checkout

Reviewed shared configuration, training orchestration, checkpoint restoration, inference
exporters, dataset/loading interfaces and selected model utilities. This is a bounded
correctness review, not an exhaustive security or scientific-model audit. No experiment runs,
remote W&B requests, dataset rebuilds or production checkpoint changes were performed.
Synthetic checkpoint fixtures were created under temporary directories in `/tmp`.

The initial checkout was clean on `feiw_v2`, revision `92f82860`. The project-state and
reorganization records describe another branch and several uncommitted changes; those
claims do not match this checkout. `.vscode/settings.json` is absent. Checks used the
interpreter explicitly named in the engineering rules:
`/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python`.
Installation/environment files changed concurrently during validation; this audit does
not own or modify those changes. Validation therefore describes the live workspace,
not an immutable checkout of the initial revision.

## Findings (original audit snapshot)

### 1. P1 — W&B seed overrides are applied after RNG initialization

Location: [runner.py](../../src/fragnnet/runner.py), lines 432–438.

`seed_everything(config_d["seed"], workers=True)` runs before `setup_wandb` merges sweep
or resumed-run values. If W&B changes the seed, model initialization and global RNGs
use the file's seed while saved hparams and the sampling setup name the W&B seed.
Seed replicates can consequently share initialization and report misleading provenance.

A local runner probe intercepted seeding, made `setup_wandb` change the seed from 7 to
11, and stopped at model construction, before datasets or output directories. Observed:
`[("seed_everything", 7), ("model_config_seed", 11)]`; no second seeding occurred.

Resolve the final configuration before seeding, and test a W&B seed override through
the runner rather than only testing config merging.

### 2. P1 — Small inference overlays overwrite unrelated checkpoint settings

Location: [checkpoint.py](../../src/fragnnet/inference/checkpoint.py), lines 238–249.

`get_config_from_ckpt` resolves the custom file against the entire current template,
then writes every resulting key over the checkpoint hparams. Supplying only
`num_workers: 0` also changes a saved `ints_transform: sqrt` to `none` and a saved
`mol_hidden_size: 128` to 256. A shape change makes strict restoration fail; a transform
change can load successfully and export incorrectly scaled intensities.

Confirmed with a synthetic MASSter hparams checkpoint and that one-key YAML overlay.
No trained checkpoint or predictions were needed. Without an overlay, saved settings
are retained.

This behavior is explicitly frozen by
`tests/test_reorg_migrations.py::test_get_config_from_ckpt_of_legacy_c0`; it is not
an uncovered implementation accident. Repair requires a deliberate compatibility
decision: preserve saved settings unless explicitly overridden, with explicit migrations
for missing legacy keys, or provide a clearly separate full-config replacement mode.

### 3. P1 — The W&B loader still accepts incomplete trained weights

Location: [checkpoint.py](../../src/fragnnet/inference/checkpoint.py), lines 72–76.

`init_wandb_ckpt` retries a failed strict load with `strict=False` and ignores the missing
and unexpected key lists. Missing layers retain fresh random parameters and evaluation
continues. The separate `init_model_from_ckpt` path correctly requires strict loading;
the strict-restoration tests exercise that path, not this one.

Confirmed using a two-input linear model checkpoint with `layer.weight` removed and a
mocked W&B API/download. The function returned a model with newly initialized weights
instead of rejecting the checkpoint. No network request was made.

Require complete restoration on this path too. If legacy weight migrations are needed,
apply named migrations before the strict check. This is a remaining loader-path gap
related to the [earlier checkpoint audit](fragnnet_masster_audit_2026_09_12.md), not a
claim that its repaired local-loader bug has returned.

### 4. P1 — Multi-seed evaluation exports only the final seed

Location: [run_save_inference.py](../../scripts/eval/run_save_inference.py), lines 185–231.

Only checkpoint initialization is inside `for seed in range(num_seeds)`. Moving the
model to its device, prediction, saving and cleanup are outside the loop. Once the API
blockers below are repaired, the default five-seed run would load all five models but
evaluate and save only seed 4. Earlier seeds disappear from the output set.

Executed the actual seed-loop AST and following statements with checkpoint/export/save
stubs for three seeds. Observed loads for 0, 1 and 2, followed by device setup, evaluation
and saving only for 2. This probe isolates the orchestration defect; the current script
cannot reach it unmodified because of finding 5.

Keep inference, saving and cleanup inside the seed loop; verify the set of saved seed
identities, not only the number of loaded checkpoints.

### 5. P2 — Three retained evaluation entrypoints use removed APIs

Locations:
[run_save_inference.py](../../scripts/eval/run_save_inference.py), lines 169–176 and 208–216;
[run_save_inference_ablations.py](../../scripts/eval/run_save_inference_ablations.py),
lines 120–127 and 155–163;
[run_save_entropy.py](../../scripts/eval/run_save_entropy.py), lines 11 and 59.

- The first two call `load_config(template_fp, custom_fp)` without the required `override`
  argument, then pass an unsupported `model_type` keyword to `apply_inference_config`.
- Both exporters receive the old loader-first positional layout, including `eval_split`.
  The current exporters take model first. Binding the actual calls against the current
  signatures raises `multiple values for argument 'output_set'`. The tabular caller also
  passes the removed `untransform_spec` argument.
- `run_save_entropy.py --help` fails immediately because it imports nonexistent
  `fragnnet.inference.runners.run_inference`; its later config call also omits `override`.

Confirmed the signature mismatches with `inspect.signature(...).bind(...)`, without
starting W&B. Confirmed the entropy import failure by executing `--help`.
Migrate the retained callers together, prefer keyword arguments at exporter boundaries,
and add a local entrypoint smoke check. If these scripts are retired, say so explicitly
and remove them from supported workflows instead of leaving runnable-looking entrypoints.

### 6. P2 — A stale test import prevents the entire suite from running

Location: [test_cut_bond_chunking.py](../../tests/test_cut_bond_chunking.py), line 19.

`from fragnnet.model import fragnnet_model` points at the removed package. The module now
lives at `fragnnet.models.fragnnet.fragnnet_model`. Standard `pytest tests -q` collects
5,822 other cases but exits with a collection error before executing any of them.
An exact legacy-package import search found this remaining reference in Python files
under `src/`, `scripts/`, `preproc_scripts/` and `tests/`.

Update the import and run the cut-bond chunking tests followed by the unfiltered suite.
The excluded-file validation below does not substitute for that check.

## Previously recorded issues rechecked (original audit)

The [configuration audit](config_audit_2026_10_12.md) remains canonical for its seven
findings. Three local probes reconfirmed the runtime issues in the current checkout:

- A W&B `inst_type_loss_tol.QTOF` entry is discarded when the existing mapping contains
  only ORB ([config.py](../../src/fragnnet/config.py), lines 293–308).
- `spec_params={typo: true}` replaces the full section and is accepted.
- `model_type=not_a_model` is accepted by `load_config` because overrides run after
  model validation/migration.

These are existing open findings, not counted among the six above. The configuration
inventory, GPU candidate backend and real-data split integrity were not re-audited here.

## Validation (original audit)

- `python -m pytest tests -q --disable-warnings --maxfail=5`: collection failed on the
  stale cut-bond import, exit 2; no test cases executed.
- `python -m pytest tests/test_config.py tests/test_runner_config.py
  tests/test_script_utils_checkpoint.py tests/test_reorg_boundary.py
  tests/test_reorg_registry.py -q --disable-warnings`: **123 passed**.
- `python -m pytest tests -q --disable-warnings
  --ignore=tests/test_cut_bond_chunking.py --maxfail=5`: **5,688 passed, 133 skipped,
  1 xfailed**, 1,100 warnings, 414.34 seconds. The xfail records the known dual-group
  sampler/dataset index-type mismatch in `tests/test_reorg_samplers.py`.
- Local probes confirmed findings 1–5 and the three previously recorded config behaviors.
- Test-suite CPU optimizer smoke steps ran; no GPU validation, real-checkpoint inference,
  model-quality evaluation, preprocessing rebuild or remote W&B operation.
  CUDA-dependent skips are not GPU evidence.
- Documentation local-link checks and `git diff --check`: passed.

## Repairs 2026-10-13

The user requested repairs after the audit. Acceptance checks are: final W&B seed reaches
all three RNGs before model construction; overlays preserve unspecified checkpoint values;
both loaders reject incomplete state dicts; all requested seeds produce exports; the three
legacy entrypoints reach the current exporter without API errors; and unfiltered test
collection works. The three reconfirmed config defects are also in the repair scope.

- **Seed:** seeding follows the W&B merge and final validation. A regression checks actual
  Python, NumPy and torch draws at model construction after W&B changes seed 7 to 11 and
  consumes random state during setup.
- **Overlay compatibility decision:** migrated checkpoint settings are the base, and only
  explicitly declared YAML keys merge over them. Nested sections merge recursively. The
  template validates keys but does not fill arbitrary missing fields. Named checkpoint
  migrations remain responsible for legacy keys. Different model types are refused;
  registry aliases for the same model are allowed. Full training configs remain valid
  explicit overlays. This intentionally replaces the earlier template-over-checkpoint
  policy in [inference pitfalls](../reference/inference_pitfalls.md#config-precedence-at-inference).
  The single affected golden value now preserves the legacy MASSter head dropout at 0.1;
  no other frozen record was regenerated.
- **Weights:** W&B restoration is strict after decompilation. Tests exercise complete,
  compiled, missing-key and unexpected-key checkpoints with mocked downloads.
- **Evaluation:** all three scripts call `run_inference_pkl` with keywords and evaluation
  enabled. The two multi-seed scripts retain the dictionary artifacts read by the statistics
  consumers. Prediction, saving and cleanup now run for every seed. Current tracked config
  paths replace missing old paths; legacy dataset CLI labels remain accepted. Device setup
  uses CUDA availability and falls back to CPU. `--untransform_spec` remains accepted but
  deprecated: the current intensity contract is already linear and normalized.
- **Test discovery:** the old chunking module referenced both a removed import and absent
  helper/constant. It now tests the actual cut-bond deduplication path through full model
  outputs and parameter gradients against per-row encoding for mean, sum and max pooling.
  Production model math was not changed to resurrect the obsolete helper.
- **Configuration:** W&B merges recursively, including new open-mapping entries and nested
  data fields, while preserving equal float values in metric settings. CLI mappings merge
  after recursive schema checking and reject scalar section replacements. The final CLI
  model selection is migrated and validated before selecting model hooks. Resumed W&B
  configs still skip retired schema keys. Findings 4–7 of the
  [configuration audit](config_audit_2026_10_12.md) remain outside this repair scope.

Repair validation:

- Focused config/runner/checkpoint/cut-bond/migration tests: **157 passed**.
- Entrypoint/config/seed checks after the last entrypoint regression was added:
  **46 passed**, including the three `--help` imports, the actual pickle exporter with
  three synthetic seed outputs, and the entropy entrypoint reaching its first export.
- Ruff on all changed Python files, shell syntax for the two comment-updated SYNCED
  wrappers, and `git diff --check`: passed.
- All **16 tracked default evaluation configs** resolve through the current loader.
- New/updated local link targets pass. Two pre-existing links in unchanged sections of
  `config/README.md` still omit the `fragnnet/` directory (the fragment-competition and
  CE per-fragment feature configs); they are outside these runtime repairs.
- `python -m pytest tests -q -o addopts='' --disable-warnings --maxfail=5`:
  **5,725 passed, 134 skipped, 1 xfailed**, 1,100 warnings, 406.30 seconds. No file
  exclusions. The known dual-group sampler index-type mismatch remains the expected failure.
- Final focused configuration rerun: **26 passed**.
- No GPU validation, remote W&B access, real-checkpoint quality evaluation or experiment
  submission. Tests use synthetic data and temporary checkpoint fixtures.

Next: review the working-tree repairs. GPU and real-data
evaluation remain separate validation steps; staging and commits remain with the user.
