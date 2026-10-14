# Repository reorganization plan — log

> **Status:** REFERENCE · **Updated:** 2026-10-12
> Verbatim dated log moved from [the plan](../plans/repo_reorg_plan.md) on 2026-10-11 and 2026-10-12, when the plan passed 600 lines; the plan holds the current summary.

Line references in these entries point at the plan as it stood when each entry was written (commits `66379ba3`, `4180d267`, `d2e3b03e`), not at the current plan.

## Progress

- 2026-10-11: plan written from a static coupling analysis at `13b1599d`. No code changed.
- 2026-10-11: revised after review, with each point checked against the code:
  - The output schema stays in core; `fragmentation/` would otherwise be imported by core.
  - Added the `_validate_tensor_condition` dependency (FraGNNet → MASSter through
    `candidate_support`).
  - Phase 2 gained a composition contract and per-stage gradient checks.
  - `misc_utils` uses `DGLGraph`; added a core batch-type registry and a `sys.modules` gate.
  - Phase 0 now exercises enabled losses, samplers across epochs and resume, export schemas, and
    every accepted model type and alias.
  - MassFormer collation follows the explicit GRAFF/ICEBERG collate chain.
- 2026-10-11: second review, each point checked against the code:
  - Phase 0 MASSter cases split: RT/CCS heads cannot be combined with proposal-only warmup.
  - Element order and `cmf_duplicate_crf_h_shift` stay in core, because core reads them.
  - A full AST check of core imports found more edges than the review named: core `feat_utils`,
    `base_dataset` (isotope cleaning reads DAG files) and `script_utils` depend on `frag_utils`,
    and `script_utils` on `masster_templates`. `frag_utils` is now split, not moved.
  - Package initializers get a lazy `__getattr__` transition and a core-only import gate.
  - Corrected: cluster jobs copy `src/` when the job starts, not at submission.
- 2026-10-11: third review, both points checked against the code:
  - `torch_scatter` is kept: PyG 2.6.1 uses it for CUDA max scatter with gradients, and
    `frag_pool_type: max` is common. The earlier "unused" claim counted only direct imports.
  - The `frag_utils` split and the `get_frag_graph`/`batch_mols_frags` extraction moved from
    Phase 6 to Phase 3, done in place. The isolation gate is now staged by phase, and Phase 6
    only renames.
- 2026-10-11: fourth review, both points checked:
  - The isolation gate is now per entry module; `script_utils` is report-only until Phase 4. A
    transitive import calculation confirmed the Phase 3 entry modules are clean. The Phase 1
    registry is now explicitly lazy for every model type, which that result depends on.
  - The registry check uses an expected-mapping table. The MassFormer dataset change in Phase 1
    is a declared change, with an equivalence check.
- 2026-10-11: user chose external ms-pred for ICEBERG. Revised the target layout, Phase 1,
  compatibility exceptions, validation and packaging accordingly. This supersedes the earlier
  optional-DGL extra and batch-type-registry proposals in the review log above. Implementation
  and environment changes have not started.
- 2026-10-11: re-read after the ICEBERG decision; fixed:
  - the stale DGL-shim sentence in the Phase 1 gate;
  - an ICEBERG pattern reference that outlived its removal;
  - the coupling-table resolution for the baselines.
  Added the concrete non-ICEBERG callers of ICEBERG code, the missing DGL install sites
  (`install_lib.py`, the Apptainer definition) and a comparability risk.

## Implementation progress, Phases 0-5 (moved 2026-10-12)

- 2026-10-11: a 12-agent verification (six finders, six adversarial verifiers) checked every claim against the code: 99 findings, 98 confirmed or partly confirmed, 1 refuted. This revision applies them. The main changes:
  - The `proc_utils` → `compute_frags` edge and the core size constants.
  - A full `frag_utils` placement table.
  - The sampler classification.
  - The re-export policy (none in core files).
  - The string-path audit.
  - Phase 0 determinism, baseline records and the CE-pair fixture recipe.
  - Retirement-error placement.
  - Per-term Phase 2 helpers.
  - Exporter hook coverage.
  - The alias table in core.
  - Phase 6 build hazards.
  - Packaging facts.

  The user approved implementing Phases 0–5 with one commit per step.
- 2026-10-11: Phase 0 committed on `refactor/fragnnet-masster-reorg`.
  - Commits: `7b3437e6` (item 10, boundary ratchet), `69c2eb86` (item 1, registry), `a766b754`
    (shared fixture `tests/reorg_fixtures.py`), `624a00a9` (items 2, 9), `d5303e0d` (item 3),
    `c289dc2a` (items 4, 5), `05bf1481` (item 6), `be417637` (item 7), `102400a7` (item 8).
  - Each test file was written by one agent, then mutation-tested by an independent reviewer
    (patched behavior through a scratch pytest plugin; the test had to fail), then fixed until
    every mutation was caught. All eight files: 165 passed, 2 strict xfails, about 85 s, bit-identical
    across runs.
  - Two source defects found and pinned as strict xfails, not fixed (behavior preservation):
    `MCESPL._setup_model` does not pass `use_nce`, `use_ace`, `ace_max`, `ace_mean`, `ace_std` to
    `SiameseGNNModel`, so `mces_pretraining` cannot be built; `DualGroupDynamicBatchSampler` yields
    numpy ints that `SpecMolFragDataset.__getitem__` rejects, so the dual-group sampler cannot load
    a train batch.
  - Baseline of the existing suite at `13b1599d` code: 28 failed, 4963 passed, 133 skipped. The 28
    are stale tests in `test_debug_masster_postcut_information_gate.py` (3), `test_export_schema.py`
    (8, predate the `pred_rt` column), `test_masster_pl.py` (7) and `test_runner_config.py` (10,
    precursor-pair validation). Later steps must add no failures beyond these.
  - ICEBERG archive SHA: `13b1599d3cd85ef8f214799894e527adaf3ddd0a` (on `origin/masster`; the
    ICEBERG paths are unchanged since).
- 2026-10-11: Phase 1 committed.
  - Commits: `8f05d1dc` (step 1, lazy registry), `0cca948c` (step 2, ICEBERG retired),
    `4834b9fc` (step 3, `SpecMolMFDataset` and GRAFF collation), `1df261de` (step 4, DGL removed),
    `59f6b798` (step 5, lazy package exports).
  - Every step ran the reorg tests, then the full suite against the Phase 0 baseline: no new
    failures (the same 28 stale tests). The Phase 0 baseline records did not change.
  - The Phase 1 gate holds and is asserted directly (`test_phase1_isolation_gate`), not only
    through the golden: no entry module loads a baseline or DGL, and selecting `graff` or
    `massformer` loads only its own package. AST violations: 28 at the end of Phase 0, 11 now.
  - FraGNNet and MASSter loads that remain (report-only until Phases 3-4): every entry module
    except `pl_utils` and `proc_utils` loads `smarts_prepass`, `compute_frags`, `multi_cut_bfs`
    and `frag_utils`; `spectrum_pl` also loads `fragnnet_model` (through
    `EXPLAINABILITY_OUTPUT_KEYS`), and so does every model selection; `script_utils` loads both
    models; `frag_gnn` selection loads `bond_fragment_heads` through `candidate_support`.
  - Departures from the step text:
    - Step 3: the `mol_params.mf` guard sits in `SpecMolDataset._process_mol`, where the removed
      code read the flag. A check in `__init__` broke a test that builds the dataset with
      `mol_params={}` and stubbed preprocessing.
    - Step 4: the shell-continuation fix and the count of `fir/` scripts were corrected in the
      step text above. Each env script was dry-run with stubbed `pip`/`conda`/`module`, and each
      slurm script compared as joined logical lines: the same calls, minus dgl.
    - Step 5: `model/` and `pl_model/` have no `__all__`, so a star import from them returns
      whichever lazy names were already accessed. Nothing star-imports them.
  - Left outside the PR:
    - 49 generated `slurm_scripts/*.sh` still install `dgl==2.4.0` until regenerated.
    - The gitignored `containers/apptainer/fragnnet_cuda128.def` is marked stale.
    - The live `FRAGNNET-GPU` environment still has DGL installed, and its editable-install
      metadata lists it until `pip install --no-deps -e .` is re-run.
    - Bytecode of the deleted modules (`src/fragnnet/iceberg/`, `dgl_compat_utils*.pyc`) needs a
      user-run cleanup script, because agent `rm -rf` is blocked.
- 2026-10-11: Phase 2 committed (`30eb927d`).
  - The Phase 0 item 4-5 records are bit-identical for FraGNNet, MASSter proposal-only and MASSter
    joint training. Multiplying any helper's output by 1.001 fails them for both models, so
    both loss paths run through the helpers.
  - Full suite: no new failures. Six tests in `test_fragnnet_audit_fixes.py` call
    `FraGNNetPL._setup_loss_fn` on a `SimpleNamespace` stand-in. They now bind the inherited
    helpers onto it.
  - A three-lens review workflow (numerics, config paths, contract) with two skeptics per
    finding found no behavior change. It flagged two imprecise docstrings, now fixed.
  - `_spectral_ce_auxiliary` takes the caller's `sparse_ce_fn`, because the models build it
    with different `oos_tolerance_multiple` values. FraGNNet's `oos_loss` branch, which no
    record reaches, is covered by `tests/test_reorg_shared_objectives.py`.
- 2026-10-11: Phase 3 committed.
  - Commits: `0efe7e28` (step 1, validator to core `misc_utils`), `e0d74390` (step 2, core
    `model/output_schema.py`), `582a6d12` (step 3, size constants in `formula_utils`), `0bebecd0`
    (step 4, `fragmentation/` with `smarts_prepass`, `candidate_support`, `cut_bond_encoder`,
    `hydrogen_cap`), `c5f46353` (step 5, `frag_utils` split into `formula_utils`, the new core
    `utils/dag_format_utils.py` and the FraGNNet remnant), `9557c48d` (step 6, `get_frag_graph`
    and `batch_mols_frags` to the FraGNNet-owned `utils/frag_feat_utils.py`).
  - Checks at each step:
    - the full suite, with no new failures;
    - a byte-identity check of every moved definition;
    - an import audit (all unique `fragnnet` import statements in tracked files resolve, apart
      from `run_inference` in `scripts/eval/run_save_entropy.py`, which was already broken);
    - the string-path audit;
    - an attribute-access audit of removed names.
  - The Phase 3 gate is enforced by `test_phase3_isolation_gate`, in a stricter form: the eight
    Phase 3 entry modules load no module outside core. NEIMS, ChemLM, precursor, GNN, MCES and
    CLIP selections load core only. FraGNNet and MASSter load `fragmentation/` and their own
    modules, never each other's. AST violations: 11 at the end of Phase 1, now 4, all in
    `script_utils` (Phase 4).
  - Departures from the step text:
    - Step 1: the validator went to `misc_utils` with its name kept, so its 37 call sites in
      `bond_fragment_heads` are unchanged.
    - Step 2: the output-schema module holds the gated key set and the per-model switch defaults
      only. The exporter keeps its annotation-name literals (deriving them was optional), so no
      unused constants were added.
    - Step 5: the four functions with no callers were deleted.
    - Step 6: the DAG feature indexing (`NODE_FEAT_TO_IDX`, `EDGE_FEAT_TO_IDX`,
      `get_node_feats`, `get_edge_feats`) already sat in the FraGNNet remnant after step 5, so it
      did not move again.
    - The `multi_cut_bfs` stub in `tests/test_spectrum_pl_utils.py` was dropped. It could also
      leak a `MagicMock` module into later tests.
  - A three-lens review workflow (serialization, equivalence, consumers) with two skeptics per
    finding found no regression. Real DAG pickles and HDF5 groups load through the moved readers.
  - Behavior change: Ctrl-C. Importing the compiled `compute_frags` replaces the process SIGINT
    handler with one that only sets a flag, and only the `compute_frags` loop checks that flag.
    Before Phase 3 every core entry module imported it, so Ctrl-C was ignored in any training
    run. Now only processes that load FraGNNet code or `script_utils` ignore it. MASSter, NEIMS
    and the other core-only runs get a normal `KeyboardInterrupt`.
  - The untracked `.claude/commands/add-smarts-rule.md` was updated locally to the new
    `smarts_prepass` path.
- 2026-10-12: Phase 4 committed.
  - Commits: `361bc9ab` (step 1, `ModelHooks`, registry dispatch and MASSter's migrations),
    `2e668262` (step 2, config validation), `fc079022` (step 3, samplers and batching),
    `c057b136` (step 4, MASSter's dataset filter), `60bc92f6` (step 5a, inference-config
    overlays), `6057b11e` (step 5b, export hooks), `90d73518` (step 6, gate).
  - Hook classes: `FraGNNetHooks` (`frag/fragnnet_hooks.py`, registered for `frag_gnn` and
    `fragnnet`), `MassterHooks` (`frag/masster_hooks.py`) and `GrAFFHooks` (`graff/hooks.py`).
    Each imports core only at module level, so config validation and checkpoint migration load
    no model code or compiled extension. Moved modules: FraGNNet samplers to
    `dataset/frag_data_sampler.py`, MASSter's to `dataset/masster_data_sampler.py`, the MASSter
    filter to `frag/masster_dataset_filter.py`, its annotation tables to `frag/masster_export.py`.
    Every moved definition is byte- or AST-identical (the MASSter filter with `self` renamed).
  - Checks at each step:
    - the full suite, with no new failures;
    - the Phase 0 records (migrations, samplers, export, losses), all unchanged;
    - the import audit, with no new failures;
    - mutation checks: replacing each hook implementation with a no-op or empty result fails
      tests.
  - Validation equivalence against the old `validate_sampler_config`: 0 accept/reject
    differences over 6000 random configs, and `tests/test_reorg_validation.py` records all 1080
    single-change outcomes (golden `validation.json`).
  - The Phase 4 gate is enforced by `test_phase4_isolation_gate`: every entry module, including
    `script_utils`, loads no module outside core, and the AST boundary check finds no violation
    (4 at the end of Phase 3). The legacy-checkpoint gate is the Phase 0 migration record,
    unchanged.
  - Departures from the hook table:
    - `owned_config_keys` became `reject_foreign_config`: each model rejects a config that
      switches on its own settings for another model, with the old messages. A general
      owned-keys check seeded from the template banners was not built; it would reject configs
      that load today.
    - `sampler_epoch_reset` became the capability `advances_sampler_each_epoch`. Both
      `_seed_dataloader` branches stay in core. `_seed_dataloader` looks hooks up by
      `hparams.model_type`, so the Phase 0 sampler stubs gained `model_type`.
    - The export hooks are a class-dispatched `ExportHooks` mixin on `FraGNNetPL` and
      `MassterPL`, not registry hooks, because the exporters hold a model object, not a
      `model_type`. Objects without it get the peak-only defaults. `formula_row_deltas` and
      `node_hash_features` became `export_formula_row_deltas` and `export_node_ccs`; added were
      `export_node_rules` and `nb_index_vars` (FraGNNet's `nb_iso` index names).
    - `dataloader_options` covers persistent workers; the `num_workers` reload workaround went
      to a separate `trainer_options` hook.
    - The GRAFF `apply_inference_config` hook, planned for Phase 1, landed in step 5a.
    - The `frag_params.include_cmf` check moved from `run_training` into
      `FraGNNetHooks.validate_config`.
    - `match_sweep_schedule` keeps its list of sampler keys (including `masster_batch_sampler`)
      in core. It reads config keys only and imports nothing.
  - Behavior changes, all intended:
    - a config with several problems may report a different first error;
    - in a MASSter config that uses its own sampler, turning on a FraGNNet sampler reports
      "can only be used with frag_gnn model" instead of "mutually exclusive";
    - `apply_inference_config` no longer edits `frag_params` or `ann_params` for models that do
      not own them;
    - `include_cmf` is no longer checked for non-FraGNNet models, which never read it.
  - A four-lens review (config, samplers, export, consumers), each reviewer reproducing at
    `fa18f7f0` and `90d73518`, found no regression. It covered 5796 validation cases, 78
    synthetic checkpoints through `get_pl_hparams`, `apply_inference_config` for 10 model types,
    and the index-variable grid. `README.md` still pointed at `data_sampler.py` for the FraGNNet
    samplers; it now points at `frag_data_sampler.py`.
    By design, external code importing the moved samplers from `dataset.data_sampler`, or
    star-importing them from `fragnnet.dataset`, now fails (no re-exports in core).
- 2026-10-12: Phase 5 committed (`32e99fea`).
  - `pyproject.toml`: `requires-python = ">=3.10,<3.13"`; `omegaconf` dropped; `scikit-learn`
    moved into `dev` and a new `scripts` extra; `pyyaml`, `networkx` and `pillow` declared;
    `torch_scatter` kept.
  - `omegaconf` removed from the install lists of the four `env/` scripts, the eight
    `scripts/manual_slurm` scripts and `scripts/setup_job_cc.py`; each edit removes only that
    line. `INSTALL.md` and `README.md` describe the extras and the Python range.
  - Lock: `env/fragnnet_gpu_py311.lock.txt`, a pip freeze of `FRAGNNET-GPU` (Python 3.11.13,
    torch 2.4.1+cu121) with the PyTorch and PyG wheel indexes. It leaves out the project,
    ICEBERG's graph library, `omegaconf` with its antlr4 runtime, and pip. The live environment
    still has those installed.
  - Full suite: no new failures (28 baseline failures, 5349 passed).
  - Not done: the stale `env_setup_gpu_py310_cu118.sh` reference in `INSTALL.md` (Phase 7), and
    the `README.md` section on the formula hard-negative sampler and its
    `formula_hard_negative_*` keys, which no longer exist in `src/` or the template (stale
    before this reorg).
