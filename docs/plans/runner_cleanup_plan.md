# Training runner cleanup plan

> **Status:** ACTIVE · **Updated:** 2026-10-12
> Phases 1–2 implemented in the working tree, not committed ([progress](#progress)). Phase 0 was folded into Phase 1. Phases 3–5 wait on the branch decision ([below](#open-decisions)).

## Objective and baseline

`runner.py` (964 lines at `a2a2d116`) resolves configs, builds datasets and loaders, validates
sampler settings, and runs training. `run_training` alone is about 400 lines. This plan fixes the
correctness bugs the audit found, then splits the file into named phases and modules without
changing behavior.

Line numbers below are as of `a2a2d116`. Lightning is 2.4.0 in `FRAGNNET-GPU`.

The [reorganization plan](repo_reorg_plan.md) already moved model-specific sampler, validation
and logging code behind hooks ([`hooks.py`](../../src/fragnnet/hooks.py)). This plan finishes the
parts it left in core.

### Callers to keep working

- Training: [`run_pl_model_fit.py`](../../scripts/run_pl_model_fit.py) and
  [`run_wandb_sweep.py`](../../scripts/run_wandb_sweep.py) call `run_training`; the sweep
  script also calls `load_config` and passes `schedule_train_size`.
- `load_config` is imported from `fragnnet.runner` by `inference/checkpoint.py`, about a dozen
  tests and several debug scripts.
- `init_dataloader` is imported by `scripts/run_inference.py` and debug scripts;
  `init_dataset`/`init_dataloader` are the entry points of the reorg golden tests
  (`test_reorg_smoke`, `test_reorg_samplers`, `test_reorg_baselines`).
- `test_masster_backend_compatibility.py` monkeypatches `runner.WandbLogger`,
  `runner.log_cuda_info`, `runner.pandarallel`, `runner.get_model_cls` and
  `runner.ModelCheckpoint`. Moving those names breaks it unless the patches move too.

## Audit findings

### Correctness bugs

| # | Where | Failure |
| --- | --- | --- |
| B1 | [:935](../../src/fragnnet/runner.py#L935) | `upload_checkpoints=True` with W&B disabled: `wandb` is imported only inside the W&B branch ([:615](../../src/fragnnet/runner.py#L615)), so `wandb.save` raises `NameError` after training. |
| B2 | [:921](../../src/fragnnet/runner.py#L921) | `disable_checkpoints=True`, `eval_test_split=True`, `min_epochs>0`: `trainer.test(ckpt_path="best")` raises "`ModelCheckpoint` is not configured" after the full fit. The condition tests `min_epochs` where it should test whether a best checkpoint exists. |
| B3 | [:926](../../src/fragnnet/runner.py#L926) | `compile=True`: `model.dynamo_prof` is set only by FraGNNet and NEIMS, and only when the compile profiler is available ([fragnnet_pl.py:295](../../src/fragnnet/models/fragnnet/fragnnet_pl.py#L295)). Other models raise `AttributeError` at the end of the run. |
| B4 | [:914](../../src/fragnnet/runner.py#L914), [:929](../../src/fragnnet/runner.py#L929), [:951](../../src/fragnnet/runner.py#L951) | Lightning 2.4 calls `exit(1)` on KeyboardInterrupt and re-raises every other exception, SIGTERM included, so `fit` never returns with `trainer.interrupted=True`. The guards are dead code. If the process dies between `hold_checkpoints_from_wandb` and `restore_held_checkpoints`, checkpoints stay in `ckpt_hold_*` and resume does not find them. |
| B5 | [`merge_wandb_config`](../../src/fragnnet/runner.py#L509) | W&B keys the config does not declare are silently ignored, so a typo in a sweep YAML runs the base value. Custom configs fail on the same typo. Merged values skip `find_retired_inst_type_refs` and the `model_type` checks. |
| B6 | [:829](../../src/fragnnet/runner.py#L829) | `validate_sampler_config` runs after both datasets are built. Config errors appear minutes in for DAG datasets. |
| B7 | [:301](../../src/fragnnet/runner.py#L301), [:323-352](../../src/fragnnet/runner.py#L323) | `group_sampler` vs `simple_group_sampler` exclusivity is an `assert` (removed by `python -O`). `formula_prec_type_group_sampler` together with either is never checked; the `elif` chain silently uses the first one set. (`bond_size_curriculum` with any group sampler is already rejected.) |
| B8 | [`match_sweep_schedule`](../../src/fragnnet/runner.py#L553) | Excludes some samplers but not `bond_size_curriculum`, which shrinks early epochs, so the cosine schedule is too long. It also names `masster_batch_sampler` and `dynamic_batch_sampler` in core. |
| B9 | [:849](../../src/fragnnet/runner.py#L849) | The bf16/fp16 → fp32 fallback edits `config_d` after `wandb.init` and after the model is built, so W&B and the model's saved hparams record the precision that was not used. |
| B10 | [:940-956](../../src/fragnnet/runner.py#L940) | Found during Phase 1. With W&B on and `delete_checkpoints=True`, checkpoints are not held out of the run directory, so `wandb.finish()` uploads them before they are deleted. |

### Rule violations and smells

Against [Python and ML rules](../rules/python.md):

- `assert` for production checks: [:527](../../src/fragnnet/runner.py#L527),
  [:648](../../src/fragnnet/runner.py#L648), [:718](../../src/fragnnet/runner.py#L718),
  [:734](../../src/fragnnet/runner.py#L734), [:817](../../src/fragnnet/runner.py#L817),
  [:895](../../src/fragnnet/runner.py#L895).
- `print` mixed with `logging`: [:228](../../src/fragnnet/runner.py#L228),
  [:304](../../src/fragnnet/runner.py#L304), [:533](../../src/fragnnet/runner.py#L533),
  [:926](../../src/fragnnet/runner.py#L926); f-strings inside `logging` calls.
- Process-wide setup inside library code: `logging.basicConfig` and `PYTORCH_CUDA_ALLOC_CONF`
  in `run_training`.

Other issues:

- Stale comments: [:229](../../src/fragnnet/runner.py#L229) says "not safe way to load" over
  `yaml.safe_load`; [:316-317](../../src/fragnnet/runner.py#L316) describes a batch-size
  computation that does not exist; [:912](../../src/fragnnet/runner.py#L912) logs
  "callback metrics" and logs nothing.
- `registry.get_model_hooks` is called four times, twice with `strict=False` and twice strict.
- `init_dataset` starts a `Manager()` server process per split and never shuts it down.
  `{**data_sds, **config_d}` lets a config key silently replace a data dict of the same name.
- `validate_sampler_config` also validates `cross_ev_fragment_loss_weight` and foreign-model
  keys; the EMA check sits in callback setup ([:771](../../src/fragnnet/runner.py#L771)).
- The PyTorch profiler block uses private APIs (`th.profiler._utils._init_for_cuda_graphs`,
  `th._C._profiler._ExperimentalConfig`).
- Migrations are applied piecemeal: `retire_pair_bce_hparams`, `migrate_legacy_masster_name`
  and `migrate_masster_proposal_method` run separately on the custom config, the template, each
  override and the saved W&B config, partly in `merge_wandb_config` and partly in its caller.
  The override loop special-cases MASSter key names in core
  ([:211-218](../../src/fragnnet/runner.py#L211)).
- CLI overrides skip the unknown-key check; an unknown leaf raises `KeyError` from inside a
  `print` ([:228](../../src/fragnnet/runner.py#L228)).
- Resume ([:717-755](../../src/fragnnet/runner.py#L717)) loads every old checkpoint with
  `th.load` and no `map_location` (CUDA tensors go to the GPU), then writes a full copy, doubling
  disk use. The path rewrite itself is needed: `ModelCheckpoint` restores its best-k state only
  when `dirpath` matches.
- `run_training` does about 12 jobs in one body, and `wandb` is bound only on one branch,
  which is how B1 slipped through.

## Plan

Each phase is one reviewable change. Phases 2–5 must not change behavior.

### Phase 0 — tests that lock in current behavior

Add `tests/test_runner_*.py`:

- B1–B3 and B7 as failing tests (marked `xfail` until Phase 1).
- `merge_wandb_config` with an undeclared key and with a retired instrument type.
- Resume path rewrite on a small fake checkpoint (`best_model_path`, `best_k_models`, `dirpath`).
- `load_config` overrides: unknown section, unknown leaf, MASSter legacy keys.

Reuse the stop-after-model harness in `test_masster_backend_compatibility.py` for
`run_training` paths.

→ verify: new tests fail exactly where expected; the existing suite is unchanged.

### Phase 1 — bug fixes (behavior changes, need user approval)

1. B1: reject `upload_checkpoints=True` with `wandb_mode="disabled"` before training starts
   (nothing to upload to).
2. B2: use `"best"` only when a `ModelCheckpoint` with a monitor is configured; otherwise test
   the final weights.
3. B3: print the compile report only when the model has `dynamo_prof`.
4. B4: remove the dead `trainer.interrupted` guards; wrap hold/restore in `try`/`finally` so
   held checkpoints are always restored.
5. B5: reject W&B keys the config does not declare (see [open decisions](#open-decisions));
   re-run the retired-instrument check after the merge.
6. B6: run all config validation (sampler, EMA, sweep schedule) right after the config is
   final, before any dataset is built.
7. B7: replace the `assert` with a `ValueError` in `validate_sampler_config` and check that at
   most one group sampler is set.
8. B8: replace the sampler key list with a hook capability (e.g. `uses_fixed_size_batches`)
   and reject `bond_size_curriculum`.
9. B9: apply the precision fallback before `wandb.init` and before the model is built.

→ verify: Phase 0 `xfail` tests pass; full suite; one short FraGNNet and one short MASSter run
with `w=offline`.

### Phase 2 — split `run_training` into phases (same file, no behavior change)

Extract:

- `setup_wandb(config_d, wandb_mode, job_id) -> WandbRunState` (dataclass: `run`,
  `is_resume`, `job_id_fp`, `old_wandb_dp`)
- `migrate_resume_checkpoints(old_wandb_dp, new_run_dir, ckpt_dp)` with
  `map_location="cpu"`
- `apply_global_torch_settings(config_d)`: TF32, LOG_ZERO, determinism, NaN hooks, sharing
  strategy
- `build_callbacks(config_d, ckpt_dp)`; `build_profiler(config_d, profile_dp)` moved to
  [`trainer_utils.py`](../../src/fragnnet/utils/trainer_utils.py)
- `build_trainer(config_d, loggers, callbacks, profiler, train_dl)`
- `finalize_checkpoints(config_d, ckpt_dp, wandb_mode)`: upload, hold, delete, restore

`run_training` becomes a flat sequence of numbered steps
([engineering rules](../rules/engineering.md) §4). Keep the monkeypatched names importable from
`runner`.

→ verify: full suite; reorg goldens unchanged; offline FraGNNet and MASSter runs, plus a resume
by killing and restarting the same job ID.

### Phase 3 — config module

Move `resolve_template_path`, `find_unknown_config_keys`, `load_config`,
`merge_wandb_config` and the override parser to `src/fragnnet/config.py`; re-export them from
`runner`.

- One `migrate_config_dict(d)` applied the same way to the template, the custom config,
  overrides and the W&B config.
- MASSter override special cases go through `ModelHooks.migrate_template_config` or a new
  override hook.
- Overrides pass the same unknown-key check as custom configs.

→ verify: full suite; `tests/test_config_templates.py`; checkpoint loading through
`inference/checkpoint.py` on one FraGNNet and one MASSter checkpoint.

### Phase 4 — data loading module

Move `init_dataset`, `init_dataloader` and `validate_sampler_config` to
`src/fragnnet/dataset/loading.py`; re-export them from `runner`.

- A sampler table plus the exclusivity check replaces the `elif` chain.
- One shared `Manager` for all splits.

→ verify: `test_reorg_samplers`, `test_reorg_smoke` and `test_reorg_baselines` goldens identical.

### Phase 5 — cleanup

- `print` → `logging`; `assert` → `raise`; delete the stale comments.
- `logging.basicConfig` and `PYTORCH_CUDA_ALLOC_CONF` move to the two entry scripts.
- One `model_hooks` lookup per function, with consistent `strict`.

→ verify: full suite; Ruff on changed files.

## Open decisions

1. **Sweep typos (B5), applied with Phase 1.** A new run fails on undeclared W&B keys. Every
   sweep YAML in `sweep/` sets only declared keys (checked 2026-10-12). A resumed run only
   warns: its saved config may hold keys the template has since dropped, which
   `test_legacy_wandb_run_config_merges_into_resolved_config` pins.
2. **Branch.** Do Phases 3–4 (moving code out of `runner.py`) on
   `refactor/fragnnet-masster-reorg`, or as a separate branch after it merges? Recommended:
   after the merge, so the reorg goldens are not regenerated mid-review.

## Progress

- 2026-10-12: audit done and plan written; no code changed.
- 2026-10-12: Phases 1–2 implemented in the working tree (user asked for them), with the Phase 0
  tests written alongside.
  - Phase 1: B1–B10 fixed as listed. B5 is strict only on new runs (open decision 1). B8 adds
    the `ModelHooks.builds_own_batch_sampler` capability (FraGNNet: dynamic or dual-group
    sampler; MASSter: `masster_batch_sampler`). B6 also moves the EMA check before dataset
    construction. `trainer.test` uses `"best"` only when a checkpoint callback saved one; no
    config sets `min_epochs: 0`, so no run changes from the old `min_epochs > 0` rule.
  - Phase 2: `run_training` is ten numbered steps over `setup_wandb` (returns `WandbRunState`),
    `fall_back_precision`, `validate_run_config`, `apply_process_settings`,
    `migrate_resume_checkpoints` (now `map_location="cpu"`), `build_callbacks`,
    `build_trainer` and `finalize_checkpoints`; `build_profiler` moved to `trainer_utils.py`.
    W&B calls after setup go through the run object (`run.save`, `run.finish`,
    `run.config.update`), so the `wandb` module is no longer bound on one branch only.
  - Departures: the `assert`s in the moved code became `ValueError`/`RuntimeError` (planned for
    Phase 5); the old `group_sampler`/`simple_group_sampler` `assert` in `init_dataloader` is
    gone, as the check now lives in `validate_sampler_config`.
  - Tests: `tests/test_runner_training.py` (new: orchestration on fakes, checkpoint
    finalization, resume rewrite); new cases in `test_runner_config.py`,
    `test_masster_sweep_schedule.py`, `test_reorg_hooks.py`;
    `test_reorg_migrations.py` passes `resumed=True`. The new config, merge and schedule tests
    fail against the `a2a2d116` runner (10 failures) and pass now.
  - Validation:
    - Full suite: 5775 passed, 1 xfailed, 11 failed. All 11 are in `test_export_schema.py` (export
      column order now holds `pred_rt`/`pred_ccs`) and
      `test_debug_masster_postcut_information_gate.py` (candidate `sides` unpack); neither
      imports `runner`, `hooks` or `trainer_utils`.
    - End to end on the `tests/reorg_fixtures.py` synthetic data (CPU, 2 epochs, test split
      on): FraGNNet and MASSter with W&B offline, then a requeue of each through a restored
      `job_id` file. The requeue copied and rewrote the checkpoints into the new run directory,
      restored `last.ckpt` and tested the rewritten best checkpoint with an identical metric;
      no `ckpt_hold_*` directory or job file was left.
    - Behavior check: the same W&B-off runs under the `a2a2d116` runner and the new one give
      identical val and test metric tables for both models.
    - Not run: a GPU run and an online W&B run.
  - Found while validating, not fixed: `data/proc/debug_tinyv4/spec_df.pkl` still uses the
    retired `FT` label, so the `config/debug/debug_tiny_*` configs no longer load. Resuming
    with a larger `max_epochs` fails in `SpectrumPL._consolidate_results` (per-epoch buffers
    sized by the checkpoint's `max_epochs`); a real requeue keeps the config, so this only
    matters for extending a finished run.
