# Repository reorganization: FraGNNet and MASSter at the center

> **Status:** ACTIVE · **Updated:** 2026-10-11
> Phases 0–5 approved for implementation on `refactor/fragnnet-masster-reorg`, one commit per step; Phase 6 (directory moves) waits for the live branches and the layout decision. ICEBERG is retired in favor of external ms-pred, and the main environment becomes DGL-free.
> Line references were measured at commit `13b1599d` (code is unchanged through `d2e3b03e`) and re-verified by a 12-agent review on 2026-10-11.

## Current summary

- **Goal.** FraGNNet and MASSter become owner packages over a model-agnostic core. GRAFF and MassFormer become baseline subpackages; ICEBERG moves to external ms-pred. The dependency rule below is enforced by tests.
- **Main finding.** Directories are the small part of the work. The coupling sits in shared code: runner, exporter, dataset, samplers, `SpectrumPL`, `pl_utils`, and `frag_utils`. `frag_utils` is FraGNNet's DAG module, but core reads it, so it is split rather than moved.
- **Order.** Phase 0 is a safety net of recorded, deterministic CPU outputs. Then Phase 1 (registry, ICEBERG/DGL retirement, baselines out of core), Phase 2 (shared loss helpers), Phase 3 (`fragmentation/`, core constants, `frag_utils` split), Phase 4 (hooks), Phase 5 (packaging). Phase 6 (renames) and Phase 7 (doc paths) come later.
- **Done.** Phase 0 (safety net) is committed: 8 test files under `tests/test_reorg_*.py`, goldens in `tests/golden/reorg/`, all mutation-tested (165 passed, 2 strict xfails for source defects).
- **Next action.** Phase 1 step 1: the lazy registry.
- Earlier dated entries are in the [plan log](../history/repo_reorg_plan_log.md).

## Goal

Center the repository on FraGNNet and MASSter. Each retained model and baseline (GRAFF,
MassFormer) becomes its own subpackage, and the shared layer stops naming any specific model.
Retire the local ICEBERG implementation and use external ms-pred for that baseline.

The folders are the smaller part of this. Most of the work is removing the model-specific branches
from the shared runner, inference export, dataset and sampler code. A directory move without that
step changes paths and leaves the coupling in place.

### Decided: external ICEBERG, DGL-free main environment

User decision, 2026-10-11: stop maintaining this repository's ICEBERG fork; use
[coleygroup/ms-pred](https://github.com/coleygroup/ms-pred) for future ICEBERG runs.

- **Archive.**
  - Record the full SHA of an existing commit that contains the unmodified fork. Before recording, confirm that `git status --short src/fragnnet/iceberg config preproc_scripts scripts/ms2c env` is clean; if it is not, the user commits those changes first.
  - Before Phase 1 merges, confirm with `git merge-base --is-ancestor <sha> <target branch>` that the SHA stays reachable from the branch that lands, or ask the user for a tag.
  - Git history is the archive. Do not keep an importable copy under `src/` or put source code in `docs/history/`.
  - The SHA is written into the retirement error message, `docs/reference/baselines.md` and the provenance note for the local ICEBERG checkpoints.
- **What goes.** Remove the local implementation, its integration branches, its configs and its ICEBERG-only entry points (inventory in Phase 1 step 2). Keep historical findings and run references.
- **Local checkpoints.** The ICEBERG checkpoints (`saved_ckpts/speed_benchmark_1000/*iceberg*.ckpt`) stay tied to the archive revision. Do not claim that ms-pred can load them.
- **Running ms-pred.** Use a separate checkout and environment, outside artifact stores. Pin the upstream commit, model version, config and checkpoint identity for each comparison. Its dependencies belong to that environment, not `FRAGNNET-GPU`.
- **Exchanging predictions.** Pass prediction files into this repository's evaluation tools. Before scoring, validate sample identity, split, adduct, collision energy, m/z units, intensity transform and binning.
  - Readers today: `scripts/eval/compare_msgym_predictions.py` (`load_glacier_peaks`), `scripts/eval/score_retrieval_candidates.py` (`--pred_format mspred`) and `src/fragnnet/utils/retrieval_utils.py` (`mspred_*` scoring, pinned to ms-pred `318bb7b`).
  - They read a PredSpecDB HDF5 with exactly one InChIKey and one collision energy per entry, treat intensities as square roots, and serve MassSpecGym retrieval only.
  - NIST20 per-seed ICEBERG rows need a new adapter to the `run_save_inference` pkl schema (`{split}/{model}/s{seed}/{split}.pkl`, read by `scripts/stats_scripts/calculate_similarity_stats.py`), proven on a fixture.
  - No training or upstream installation is needed for the retirement itself.
- **DGL.** Remove DGL requirements from package metadata, maintained environment and cluster scripts, shared utilities and installation docs. No `iceberg` extra, DGL shim or batch-type registry is needed. Do not uninstall packages from the current environment as part of a source reorganization.

This is an intentional exception to behavior preservation: `iceberg_gen` and `iceberg_inten` must
fail with an actionable message pointing to ms-pred and the archive revision, rather than silently
selecting another model. Other models keep their behavior.

## Target layout

```
src/fragnnet/
  dataset/ model/ pl_model/ utils/ runner.py   # core: model-agnostic, import paths unchanged
  registry.py                                  # model_type -> "module:Class" (lazy), aliases, retired types
  fragmentation/                               # bond-cut chemistry shared by both fragment models
  models/
    fragnnet/                                  # DAG model
    masster/                                   # MASSter
  baselines/
    graff/ massformer/                         # neims/ only if open decision 2 says so
```

Dependency rule:

- `models/fragnnet`, `models/masster` and each baseline may import `fragmentation/` and core, never one another.
- `fragmentation/` imports core only.
- Core imports none of them, `fragmentation/` included; it reaches models only through the registry, which holds strings. Anything core consumes, such as the annotation output keys, therefore lives in core, even when only fragment models produce it.
- Tests may import any layer. For example, MASSter's H-cap parity test may use FraGNNet's Cython `update_bonds` as its oracle.

Until Phase 6, owner modules still sit in core directories, so the boundary check classifies each
module through a checked-in module-to-layer table rather than by directory.

Shared modules stay where they are rather than moving into a `core/` package. They hold about
32k lines, and moving them would change nearly every import. `fragnnet.fragnnet` is avoided
because the top-level package already has that name. Shared `model/` sitting next to `models/` is
a cosmetic wart; renaming it to `nn/` later is optional.

## Current coupling

### Sizes

Lines of `.py`, `.pyx` and `.cu` under `src/fragnnet` (70.3k in total):

| Group | Lines | Files |
| --- | --- | --- |
| MASSter-only | ~17.6k | `frag/masster_*`, `masster_core.pyx`, `.cu` kernels, `bond_mask_generator`, `masster_model`, `bond_fragment_heads`, `candidate_features`, `rt_ccs`, `masster_pl` |
| FraGNNet-only | ~6.8k | `compute_frags.pyx`, `multi_cut_bfs.pyx`, `fragnnet_model`, `fragnnet_pl`, `*spec_mol_frag_dataset` |
| Split or moved to `fragmentation/` | ~4.0k | `utils/frag_utils.py` (2.6k, split below), `frag/smarts_prepass.py`, `model/candidate_support.py` |
| Shared | ~32k | `SpectrumPL`, loss, `nn_blocks`, encoders, utils, runner |
| Baselines | ~9.8k | ICEBERG 6.5k (retired), MassFormer 2.0k, GRAFF 1.4k |

`MassterPL` and `FraGNNetPL` are siblings: both subclass `SpectrumPL`. Their model-specific
modules have no direct imports of each other. MASSter reads `CANONICAL_ELEMENT_ORDER` (and
`candidate_features` also `CANONICAL_H_IDX`) from the shared `frag_utils`. There is one indirect
FraGNNet → MASSter path: `fragnnet_pl` imports `candidate_support`, which imports
`_validate_tensor_condition` from the MASSter module `bond_fragment_heads`
([candidate_support.py:8](../../src/fragnnet/model/candidate_support.py#L8)). The package
initializers also load each model with the other; see [Package initializers](#package-initializers).

### Shared layer depends on the models

| Where | What it knows about a model |
| --- | --- |
| [spectrum_pl.py:23](../../src/fragnnet/pl_model/spectrum_pl.py#L23) | Imports `EXPLAINABILITY_OUTPUT_KEYS` from `fragnnet_model` |
| [spectrum_pl.py:2454-2500](../../src/fragnnet/pl_model/spectrum_pl.py#L2454) | `_seed_dataloader` reads `dag_size_curriculum`, `bond_size_curriculum`, `batch_sampler_type == "ce_context"` and `masster_batch_sampler`, then takes either the `set_epoch` + Lightning batch-count refresh path or the `_pre_compute_batches` path |
| [spec_mol_dataset.py:57-62](../../src/fragnnet/dataset/spec_mol_dataset.py#L57) | Filters spectra inside `__init__` when `model_type == "masster"`, through `_filter_masster_spectra` and core `formula_utils.masster_has_valid_ion` |
| [spec_mol_dataset.py:7,237-280](../../src/fragnnet/dataset/spec_mol_dataset.py#L7) | MassFormer `mf` preprocessing and collation (imports `massformer.data_utils`, which loads the MassFormer Cython `algos`) and GRAFF `mol_graff` collation |
| [data_sampler.py](../../src/fragnnet/dataset/data_sampler.py) | FraGNNet-only (budget on fragment-DAG nodes or edges): `SpecMolFragDynamicBatchSampler`, `DualGroupDynamicBatchSampler`, `CEContextDynamicBatchSampler`, `validate_dag_size_curriculum`. MASSter-only: `MassterCEContextDynamicBatchSampler`. Generic: `_CEContextBatchSamplerBase`, `GroupSampler`, `FormulaPrecTypeGroupSampler`, `get_group_sampler`, `BondCountCurriculumSampler` and `validate_bond_size_curriculum` (need only `mol_df.num_bonds`; every config enabling the bond curriculum is MASSter) |
| [proc_utils.py:6](../../src/fragnnet/utils/proc_utils.py#L6) | Imports `MAX_NUM_NODES`, `MAX_NUM_EDGES` from FraGNNet's Cython `frag.compute_frags`; `base_dataset.py:34` imports `proc_utils`, so importing any core dataset loads `compute_frags` |
| [runner.py](../../src/fragnnet/runner.py#L218) | 8 `model_type` branches (218, 331, 345, 515, 556, 806, 1069, 1267), a hard-coded `VALID_MODEL_TYPES` (:50), module-level sampler and curriculum-validator imports (:20-34), MASSter-only checks (609-631), FraGNNet curriculum and sampler checks (773-808), shared CE-pair checks naming `use_masster_batch_sampler` (588, 640), and `masster_batch_sampler` in the sweep-schedule key list (924) |
| [setup_utils.py:1-16](../../src/fragnnet/utils/setup_utils.py#L1) | Imports every model, NEIMS and every baseline at module load, so dgl loads with core |
| [script_utils.py](../../src/fragnnet/utils/script_utils.py#L850) | 10 model `isinstance` checks. 850-913: `required_index_vars_for_outputs` and `get_idx_vars` (the function ends at 978). 1075: `run_inference_pkl`, which gates the explainability switch and FraGNNet-only index vars (1107-1187). In `process_batch_to_tabular_row`: 2112 (MASSter-only rule-provenance error), 2127 (explainability switch), 2149-2200 (MASSter peak aggregation and the `masster_annotation_tables` call at 2167), 2414 (FraGNNet formula-delta fallback). Also 2586 (`get_node_feats` for the cc hash) and module imports at 20-22 (`MASSTER_RULE_VOCABULARY`, `IcebergIntenPL`, `FraGNNetPL`, `MassterPL`) |
| [script_utils.py:462-515](../../src/fragnnet/utils/script_utils.py#L462) | Inference-config overlays: the `disable_preproc` loop over `spec`, `mol`, `frag`, `magma`, `ann` (462-471); FraGNNet `frag_params.formula_str/preload/preprocess` (477-479, 509-512); MAGMA (ICEBERG) at 481-483 and 514-515; GRAFF `ann_params` at 484-486. `spec_params.prec_type_str = True` (479) applies to every model, and the formula-delta export needs it |
| [script_utils.py:1754-2063](../../src/fragnnet/utils/script_utils.py#L1754) | MASSter annotation export (`MassterAnnotationTables`, `masster_annotation_tables`) |
| [pl_utils.py](../../src/fragnnet/utils/pl_utils.py) | `migrate_legacy_masster_key`, `migrate_legacy_masster_name`, `migrate_masster_proposal_method`, `migrate_masster_rt_ccs_hparams`; inline MASSter backfills in `get_pl_hparams` (507-539), four of which duplicate `runner.load_config` (218-227); `migrate_legacy_hparams` (72-160, called for every model from `SpectrumPL.__init__`) holds string-keyed FraGNNet, CLIP and MCES default tables |
| [dgl_compat_utils.py](../../src/fragnnet/utils/dgl_compat_utils.py) | Used by ICEBERG and by shared [misc_utils.py:42](../../src/fragnnet/utils/misc_utils.py#L42), whose `to_device` and `print_shapes` check `isinstance(v, DGLGraph)` |

### Lightning modules and their loss terms

Both modules build their loss in a large `_setup_loss_fn` closure (fragnnet_pl.py:555-1464,
masster_pl.py:1353-2393). They also override other `SpectrumPL` hooks:

- **FraGNNetPL:** `training_step`, `configure_optimizers`, `on_train_epoch_end`, `optimizer_step` (manual optimization).
- **MassterPL:** `__init__`, `forward`, `_setup_metric_fns`, `on_load_checkpoint`, `on_train_epoch_start`, `_accumulate_running_metrics`, `_weight_decay_overrides`. It also keeps about 530 lines of loss helpers at module level (masster_pl.py:303-830).

Only two things are straight copies between the modules: the CE-pair term and
`validate_hellinger_auxiliary_weights`. For the CE-pair term, both build the same eligibility
tensors from `ce_pair_eligible_*` ([fragnnet_pl.py:784-800](../../src/fragnnet/pl_model/fragnnet_pl.py#L784),
[masster_pl.py:1660-1676](../../src/fragnnet/pl_model/masster_pl.py#L1660)), make the same
`ce_switch_pair_bce` call, and weight and log it the same way
([fragnnet_pl.py:1016-1043](../../src/fragnnet/pl_model/fragnnet_pl.py#L1016),
[masster_pl.py:2185-2226](../../src/fragnnet/pl_model/masster_pl.py#L2185)). `SpectrumPL`
already prepares the `ce_pair_*` batch fields
([spectrum_pl.py:1460](../../src/fragnnet/pl_model/spectrum_pl.py#L1460)).

The spectral-leak and spectral-CE-auxiliary terms are **not** copies:

- **FraGNNet** applies the leak to `exp(pred_logprobs)` and adds an OOS CE term when `oos_loss` is on (fragnnet_pl.py:949-987).
- **MASSter** applies both to raw pathway intensities, renormalizes them in the closure (with clamps) for the CE, and passes a zero OOS vector (masster_pl.py:1830-1881).

The terms enter the total in different orders:

- **FraGNNet:** primary, leak, auxiliary, formula support, CE-pair, precursor pair, null loss, entropies, edge, nb, pairwise (fragnnet_pl.py:938-1089+). There is no epoch-dependent branch; `ce_pair_enabled` is train-only.
- **MASSter:** primary, entropy match, leak, auxiliary, ion-H support, ion-mode proposal, proposal, cross-EV, base CE, CE-pair. During proposal-only training the total is then **replaced** by the proposal loss (plus the ion-mode proposal loss), discarding all of those. Only afterwards are the RT/CCS losses added ([masster_pl.py:2227-2250](../../src/fragnnet/pl_model/masster_pl.py#L2227)). The spectral terms are still computed and logged in that stage but carry no gradient.

### Annotation output contract

- **`EXPLAINABILITY_OUTPUT_KEYS`** lives in `fragnnet_model.py`. It is the set of FraGNNet outputs that become `None` when `return_explainability_tensors=False`, not the full annotation schema: it excludes `pred_formula_batch_idxs` and `pred_node_batch_idxs`, which the metrics also read. Only `SpectrumPL` (`spectrum_pl.py:23`, `:609`) and two tests import it.
- **The exporter** spells the annotation key names as string literals (script_utils.py:645-657, 703-706, 783-844).
- **MASSter** emits none of these names from its model or Lightning module; the exporter synthesizes them.
- **The shared interface** also includes the batch switch `return_explainability_tensors`. Core sets it (spectrum_pl.py:1720-1722, script_utils.py:1111 and :2130), and the two models default it differently: FraGNNet `True` (fragnnet_model.py:1252), MASSter `False` (masster_pl.py:2601).

Because core consumes all of this, it belongs in a core output-schema module, not in `fragmentation/`.

### `smarts_prepass` is shared

It sits in `frag/` beside the MASSter modules, imports nothing from `fragnnet`, and has two
importers: FraGNNet's `frag_utils` ([frag_utils.py:34](../../src/fragnnet/utils/frag_utils.py#L34);
the `_apply_smarts_prepass`/`_apply_nl_prepass` calls sit inside `compute_dags`, :1210, :1261,
:1284) and MASSter's templates ([masster_templates.py:13](../../src/fragnnet/frag/masster_templates.py#L13)).
Not running SMARTS for FraGNNet (VRAM) is a run-time choice; the code dependency remains.

### Chemistry implemented more than once

| Rule | Implementations |
| --- | --- |
| Hydrogen cap | MASSter torch, `h_transfer_bound="valence"` ([masster_model.py:2429](../../src/fragnnet/model/masster_model.py#L2429)), and `frag_utils.compute_cc_h_cap` ([:394](../../src/fragnnet/utils/frag_utils.py#L394), NumPy). The production DAG builder calls the latter at :523, :647 and :1554. [test_masster_model.py:36](../../tests/test_masster_model.py#L36) uses it as MASSter's reference |
| `update_bonds`, `compute_cc_h_floor` | Each exists in Cython ([compute_frags.pyx:485](../../src/fragnnet/frag/compute_frags.pyx#L485), :443) and in Python ([frag_utils.py:578](../../src/fragnnet/utils/frag_utils.py#L578), :420). Only the Cython versions run in production. The Python `update_bonds` has no callers; the Cython one is about 100× faster per call and returns int32/uint8 instead of a bool mask |
| Fragment charge | MASSter `candidate_features._build_charge` ([:311](../../src/fragnnet/model/candidate_features.py#L311)) and FraGNNet's `frag_q` |
| Element order | `CANONICAL_ELEMENT_ORDER` (C, H, then sorted heavy elements) is used by MASSter (`masster_model`, `candidate_features`) and by core `AtomOneHot` ([feat_utils.py:335](../../src/fragnnet/utils/feat_utils.py#L335)); `CANONICAL_H_IDX` by `candidate_features` and `frag_utils`. A second order, `ELEMENT_TO_IDX`/`ELEMENTS` (H last), drives DAG formula arrays and the MASSter exporter (script_utils.py:1907-1990). The two orders must stay distinct |

### Two metadata encoders

FraGNNet uses the `CEModel`/`PrecModel`/`InstModel`/`FragModeModel` mixins, `ContextEmbedding`
and four scalers from [base_model.py](../../src/fragnnet/model/base_model.py). MASSter has its own
`MassterContextEncoder` ([masster_model.py:675](../../src/fragnnet/model/masster_model.py#L675)),
reusing only `QETEnergyEncoder` and `get_embedder`. Both read the same `nce_*`, `ace_*` and
`ce_insert_*` keys and the fragmentation mode. Keeping two encoders is a reasonable design and
this plan does not merge them. Whether both standardize CE identically is unverified.

### Placement of shared symbols

| Symbol | Now in | Destination |
| --- | --- | --- |
| `SpectrumPL`, `get_sparse_cross_entropy_fn`, `scatter_reduce`, `GNN`, `get_embedder`, `get_mol_feats_sizes` | core | stay |
| `ce_switch_pair_bce`, `sparse_outside_window_intensity_fraction`, `validate_hellinger_auxiliary_weights` | `model/loss.py` | stay; per-term helpers in `SpectrumPL` call them (Phase 2) |
| `PREC_TYPE_TO_MASS_DIFF`, `PREC_TYPE_TO_CMF_MASS_DIFF`, `cmf_duplicate_crf_h_shift` | `formula_utils.py` | stay in core: core `formula_row_prec_deltas` ([formula_utils.py:559](../../src/fragnnet/utils/formula_utils.py#L559)) calls `cmf_duplicate_crf_h_shift`, and the exporter uses it |
| `masster_has_valid_ion` | `formula_utils.py` | MASSter (Phase 4), together with `_filter_masster_spectra`; no re-export in core |
| `CutBondEncoder`, `encode_cut_bond_pairs` | `model/base_model.py` | `fragmentation/`; used only by `fragnnet_model` and `masster_model`; importers retargeted, no re-export in `base_model` |
| `candidate_support.py` (all) | `model/` | `fragmentation/`, after `_validate_tensor_condition` moves to core |
| `_validate_tensor_condition` | `model/bond_fragment_heads.py` (MASSter) | core (tensor-validation helper) |
| `EXPLAINABILITY_OUTPUT_KEYS`, annotation key names, `return_explainability_tensors` defaults | `fragnnet_model.py`, exporter literals | core output-schema module |
| `CANONICAL_ELEMENT_ORDER`, `CANONICAL_H_IDX`, the other element tables | `utils/frag_utils.py` | core `formula_utils.py` (see [the split](#frag_utils-is-split-not-moved)) |
| `MAX_NUM_NODES`, `MAX_NUM_EDGES`, `MASK_SIZE` | Cython `compute_frags` / `frag_utils` | core `formula_utils.py` (see [Phase 3](#phase-3--fragmentation-core-constants-and-the-frag_utils-split)) |

Placement rules:

1. **Place code by what it is, not by who calls it.** Twelve losses are imported only by `fragnnet_pl` today (Hungarian JSD, entropies, Hellinger, pairwise similarity, edge loss, contender BCE), and `sparse_peak_list_distance` and `conditional_in_support_cross_entropy` are MASSter-only. All are generic spectrum math and stay in `model/loss.py`. The same applies to the nominal isotope tables in `formula_utils` and the peak aggregation in `spec_utils`, which only MASSter uses today. `QETEnergyEncoder` is already shared through `CEModel` and stays in `base_model`.
2. **Anything core reads stays in core**, whatever its subject.
3. **Core files get no re-exports**, eager or lazy. A re-export only moves a dependency, and the boundary check counts function-level imports. When a name leaves a core file, its importers are retargeted in the same PR. Re-exports are allowed only in files that belong to the destination's layer or a higher one, such as the FraGNNet remnant of `frag_utils` re-exporting names that moved to core. The Phase 1 package-initializer `__getattr__` is the one listed exception (`COMPAT_SHIMS`), removed in Phase 6.
4. **String module paths are not covered by re-exports.** This covers `patch(...)` and `monkeypatch.setattr("pkg.mod.name")` targets, `sys.modules[...]` stubs, `find_spec` and `import_module`. Retarget them in the PR that moves the code; see [String module paths](#string-module-paths).

### Every core import of code the plan moves

An AST check of every `src/` import (module-level and function-level) against the planned
destinations lists these edges from a lower layer to a higher one:

| Importer (layer) | Imports from (destination) | Names | Resolution |
| --- | --- | --- | --- |
| `model/__init__`, `pl_model/__init__`, `dataset/__init__` (core) | FraGNNet, MASSter (and NEIMS) | model, PL, dataset and sampler classes | [Package initializers](#package-initializers) (Phase 1) |
| `utils.setup_utils` (core) | FraGNNet, MASSter, NEIMS, baselines | PL and dataset classes | Lazy registry for every model type (Phase 1) |
| `utils.proc_utils` (core) | `frag.compute_frags` (FraGNNet) | `MAX_NUM_NODES`, `MAX_NUM_EDGES` | Core size constants (Phase 3) |
| `dataset.base_dataset` (core) | `utils.frag_utils` | `load_frag_d` | Core DAG-format module (Phase 3); shared isotope cleaning reads DAG formulas ([base_dataset.py:744](../../src/fragnnet/dataset/base_dataset.py#L744)) |
| `dataset.spec_mol_dataset` (core) | `massformer.data_utils` | `mf_data_utils` | `SpecMolMFDataset` (Phase 1) |
| `pl_model.spectrum_pl` (core) | `model.fragnnet_model` | `EXPLAINABILITY_OUTPUT_KEYS` | Core output-schema module (Phase 3) |
| `utils.feat_utils` (core) | `utils.frag_utils` | `CANONICAL_ELEMENT_ORDER`, `NODE_FEAT_TO_IDX`, `EDGE_FEAT_TO_IDX`, `get_node_feats`, `get_edge_feats`, `th_long_to_mask` | Element tables and mask packing to core; `get_frag_graph`, `batch_mols_frags` and DAG feature indexing to a FraGNNet-owned module (Phase 3); 12 importers retargeted |
| `utils.script_utils` (core) | `utils.frag_utils` | `ELEMENT_TO_IDX`, `MASK_SIZE`, `NUM_ELEMENTS`, `PACKED_DAG_INDEX_DATASET`, `formula_arr_to_str`, `th_mask_to_long`, `get_node_feats` | Formula and DAG-format names to core (Phase 3); `get_node_feats` use goes to FraGNNet's export hook (Phase 4) |
| `utils.script_utils` (core) | `frag.masster_templates`; `pl_model` | `MASSTER_RULE_VOCABULARY`; `FraGNNetPL`, `MassterPL` | Export hooks (Phase 4) |
| `utils.script_utils` (core) | `iceberg.pl_model` | `IcebergIntenPL` | Removed with ICEBERG (Phase 1). `script_utils` imports no GRAFF or MassFormer class; GRAFF's `ann_params` overlay becomes a hook |
| `runner` (core) | FraGNNet and MASSter samplers, `validate_dag_size_curriculum` (via `dataset.data_sampler`) | sampler classes, validator | `build_batch_sampler`, `validate_config` hooks (Phase 4) |
| `utils.misc_utils` (core) | `utils.dgl_compat_utils` | `DGLGraph` | DGL checks and the shim deleted (Phase 1) |
| `model.candidate_support` (`fragmentation/`) | `model.bond_fragment_heads` (MASSter) | `_validate_tensor_condition` | Validator to core (Phase 3) |
| `model.masster_model`, `model.candidate_features` (MASSter) | `utils.frag_utils` (FraGNNet) | `CANONICAL_ELEMENT_ORDER`, `CANONICAL_H_IDX` | Element tables to core (Phase 3) |

### `frag_utils` is split, not moved

Only some `frag_utils` functions need the Cython builders, `compute_frags` constants or
`smarts_prepass`:

- `compute_dags`, `timed_get_dags` and `run_frag_gen_hdf5`;
- `get_fraggen_input_arrays`, `compute_approximate_formula` and `compute_approximate_cchs`;
- `_inject_smarts_pairs_into_dag` and `_inject_nl_masks_into_dag`;
- `cc_bitmask_to_formula_arr` and `convert_cc_int_to_np_mask`.

The reader, format code, mask packing, element tables and `compute_cc_h_cap` do not.

| Destination | Symbols |
| --- | --- |
| Core `formula_utils` | `ELEMENT_TO_VE`, `HEAVY_ELEMENTS`, `NUM_HEAVY_ELEMENTS`, `ELEMENTS`, `NUM_ELEMENTS`, `CANONICAL_ELEMENT_ORDER`, `CANONICAL_H_IDX`, `ELEMENT_TO_IDX`, `IDX_TO_ELEMENT`, `formula_arr_to_str`, `th_mask_to_long`, `th_long_to_mask`, and the size constants `MASK_SIZE = 128`, `MAX_NUM_NODES = 128`, `MAX_NUM_EDGES = 512`, with the asserts `MASK_SIZE >= MAX_NUM_NODES` and `MASK_SIZE % 64 == 0` |
| Core DAG-format module | every `PACKED_DAG_*` constant, `get_frag_name`, `get_frag_fp`, `_legacy_frag_names`, `get_dag_output_path`, `_dump_pickle_file`, `_load_pickle_file`, `dump_dag_pickle`, `dump_dag_hdf5`, `initialize_packed_dag_hdf5`, `dump_dag_hdf5_packed`, `_load_dag_hdf5_packed`, `load_dag_hdf5`, `load_dag_hdf5_from_group`, `save_frag_d`, `load_frag_d`. Reader and writer stay together so the format lives in one place |
| `fragmentation/` | `compute_cc_h_cap` (FraGNNet's DAG builder imports it from there) |
| FraGNNet-owned (`frag_utils` remnant) | Everything else, including `extract_mol_info`, `get_fraggen_input_arrays`, `compute_cc_h_floor` and `compute_approximate_*`; the `cc_*` and `convert_cc_*` helpers, `wl_hash` and `get_subgraph_hashes`; the `_inject_*` helpers, `compute_dags`, `timed_get_dags` and `run_frag_gen_hdf5`; `compute_frag_peak_stats`, `compute_boundary_mask` and `compute_boundary_pair_idxs`; and the dtype and DAG feature-index constants with `get_node_feats`/`get_edge_feats`.It asserts the core size constants equal `compute_frags.MAX_NUM_NODES/EDGES` and re-exports moved core names until Phase 6 |
| Deleted | The Python `update_bonds` (no callers). Also no callers anywhere: `convert_cc_mask_to_int`, `convert_cc_int_to_mask`, `convert_cc_int_to_np_mask` and `compute_approximate_cchs` (delete or keep, decided in the Phase 3 PR) |

The `.pyx` files keep their literal 128/512, because Cython needs compile-time sizes. A test
asserts they equal the core constants.

### Package initializers

`model/__init__.py`, `pl_model/__init__.py` and `dataset/__init__.py` import model-specific
classes when imported: `FraGNNetModel`, `MassterModel`, `NeimsModel`, `FraGNNetPL`,
`MassterPL`, `NeimsPL`, `SpecMolFragDataset`, `IterableSpecMolFragDataset`, and the FraGNNet and
MASSter samplers. Python runs a package's `__init__` before any of its submodules, so importing a
shared module such as `fragnnet.pl_model.spectrum_pl` loads every model. A lazy registry alone
cannot isolate core while these exports exist.

About 48 files import from these three packages. Nothing star-imports them, and
`dataset/__init__.py` is the only one with an `__all__`. The repo uses three import forms, and all
three work with a PEP 562 `__getattr__` that raises `AttributeError` for unknown names:

- `from pkg import Class`;
- `from pkg import submodule` (for example `from fragnnet.model import MassterModel, bond_fragment_heads`);
- `import pkg.sub as alias`.

### Config keys

`config/template.yml` has 474 top-level keys, already grouped by per-model section banners. Those
banners are enforced by `MODEL_SECTIONS` in `tests/test_config_templates.py`:

| Section | Keys |
| --- | --- |
| shared core | 201 |
| MASSter only | 123 (all `masster_*`) |
| FraGNNet only | 95 (mostly unprefixed: entropy weights, `nb_*`, `edge_*`, `null_loss`, `candidate_ion_*`, `ion_channel_*`, `finetune_*`, `context_embedding*`, `*_scale`) |
| ICEBERG only | 15 (`iceberg_*`, `magma_params`, `magma_dp`) |
| GrAFF only | 11 |
| MassFormer only | 10 |
| MCES / CLIP / ChemLM / NEIMS | 7 / 6 / 5 / 1 |

`template.yml` stays one file (recorded decision). FraGNNet keys are not renamed, because that
would be a schema change. Each model package declares `owned_config_keys`, seeded from these
section banners. That generalizes `MASSTER_SAMPLER_KEYS`
([runner.py:97](../../src/fragnnet/runner.py#L97)), so the runner can flag a key that is set for a
model that ignores it. After ICEBERG retirement, its 15 keys stay inert and unowned (see
[Unchanged](#unchanged)), and the owned-keys check accepts them. ChemLM reads the
`neims_bottleneck_factor` key, so the `neims_*` keys stay in core whatever happens to NEIMS.

### Baselines

Core reaches into the baselines through `setup_utils`, `spec_mol_dataset`, `script_utils` (ICEBERG
only) and `VALID_MODEL_TYPES`. In the other direction, baselines subclass
`SpectrumPL`/`BinnedPL`/`BaseDataset` and use loss, scatter and formula utilities, which is
allowed.

- **Tests.** No ICEBERG, GRAFF or MassFormer module has its own test. `tests/test_neims_model.py` covers NEIMS, and `tests/test_config_templates.py` touches the others.
- **ICEBERG callers.** All outside callers of ICEBERG code are inventoried in [Phase 1 step 2](#phase-1--registry-retire-iceberg-decouple-retained-baselines).
- **Retained-baseline callers that need Phase 6 path updates:**
  - `scripts/eval/run_save_inference.py` (also edited in Phase 1);
  - `notebooks/param_counts.ipynb`;
  - `notebooks/test_notebooks/test_fit_anns.ipynb`;
  - `notebooks/test_notebooks/test_graff_ckpts.ipynb`.
- **Local checkpoints.** NEIMS has two (`saved_ckpts/spectraverse/fragnnet_neims_original/`). GRAFF and MassFormer have none locally; their W&B runs use the prefixes `graff3_long` and `massformer_long`.

### String module paths

Import counts miss module and file paths held as strings. Known sites:

- **Patch targets:**
  - `tests/test_masster_pl.py:717-720` patches `fragnnet.model.candidate_support.linear_sum_assignment` (moves in Phase 3);
  - `tests/test_masster_pl.py:1775` and `:2332` patch `fragnnet.pl_model.masster_pl.<name>`;
  - `tests/test_hdf5_dag_storage.py:222` patches `fragnnet.utils.frag_utils.par_apply` (`run_frag_gen_hdf5` stays in the `frag_utils` remnant).
- **Module stubs and lookups:**
  - `tests/test_spectrum_pl_utils.py:23-27` stubs `sys.modules["fragnnet.frag.multi_cut_bfs"]` (can be dropped after Phase 3);
  - `tests/test_masster_full_training_benchmark.py:114` calls `find_spec("fragnnet.frag.masster_core")` and has an embedded script at :134;
  - `scripts/debug/bench/benchmark_masster_full_training.py:60-103` and `:489-509` use `importlib`, `sys.modules` and a `masster_core*.so` glob.
- **File paths:**
  - `tools/build_compute_frags_setup.py:6-7` and `scripts/profile_compute_frags.py:12`, `:82` (`src/fragnnet/frag/compute_frags.pyx`);
  - `scripts/debug/ei/debug_masster_ei_smarts_screen.py:269-270`.
- **Untracked slash command:** `.claude/commands/add-smarts-rule.md` names `fragnnet.frag.smarts_prepass` and `frag_utils` helpers.

Each moving PR runs `git grep -nE "[\"']fragnnet\."` and `git grep -n "src/fragnnet/"` outside
import statements and retargets what it finds.

### Callers and tests

Counts come from an AST scan of import statements on 2026-10-11:

- **MASSter-specific code** (the Sizes-table MASSter modules plus `MassterPL`/`MassterModel` through the packages): about 65 files (13 src, 18 scripts, 34 tests).
- **FraGNNet-specific code or `frag_utils`:** about 120 files (12 src, 39 scripts, 15 preprocessing scripts, 51 tests, 1 notebook).
- **Both models by module:** 5–8 test files, depending on whether `smarts_prepass` and `candidate_support` count. More touch both through `model_type` or config strings (hparam migration, config templates, DAG storage).

Re-run the scan before Phase 6.

Commits in the 30 days before 2026-10-11 were heaviest in the files carrying the cross-model
branches: `runner.py` 20, `script_utils.py` 20, `spectrum_pl.py` 19. `fragnnet_model.py` had 11
(`CutBondEncoder`, CMF fixes, `feiw_v2` merges), so FraGNNet code is not frozen and `feiw_v2` is
live.

## Plugin interface

Each retained model and baseline registers a `model_type` with these hooks; core models use
defaults. External ICEBERG is not registered; its predictions enter through file-based
evaluation.

**Resolved in core, before any hook runs.** The registry owns three things:

- the alias table: `mass_shatter` → `masster` plus the `mass_shatter_*` key prefix, `frag_gnn`/`fragnnet`, and `clip`/`spec_mol_clip_pretraining`;
- `RETIRED_MODEL_TYPES` (`iceberg_gen`, `iceberg_inten`, with a message naming ms-pred and the archive SHA);
- the model-type list.

Alias resolution runs at the existing call sites: runner.py:190-245, `get_pl_hparams` at
pl_utils.py:499, runner.py:1028 for the saved W&B config, and `MassterPL.__init__`. Alias pairs
get identical behavior from every hook. The string-keyed legacy default tables in
`migrate_legacy_hparams` (FraGNNet, CLIP, MCES) stay in core `pl_utils`: they import nothing, so
the dependency rule allows them.

| Hook | Replaces | Phase |
| --- | --- | --- |
| `model_cls`, `dataset_cls` | `setup_utils.get_model_cls`/`get_dataset_cls`. Thin wrappers stay importable as `setup_utils.get_*` and `runner.get_*`, because tests patch and assert them | 1 |
| `migrate_template_config` | MASSter backfills in `runner.load_config` 218-227 | 4 |
| `migrate_checkpoint_hparams` | MASSter backfills in `get_pl_hparams` 507-539 and `migrate_masster_rt_ccs_hparams` | 4 |
| `validate_config` | runner 515-560, 609-631, 773-808. `validate_dag_size_curriculum` moves with the FraGNNet samplers; `validate_bond_size_curriculum` stays core | 4 |
| `uses_ce_context_sampler` (capability) | model names in the shared CE-pair checks (runner 588, 640) and the sweep-schedule key list (924) | 4 |
| `owned_config_keys` | `MASSTER_SAMPLER_KEYS` (generalized) | 4 |
| `build_batch_sampler` | runner 331-470 and the module-level sampler imports (20-34) | 4 |
| `dataloader_options` | runner 345 (persistent workers), 1267 (`num_workers` workaround exemption) | 4 |
| `filter_dataset` | `spec_mol_dataset` 57-62. Called from **inside** `SpecMolDataset.__init__` through the registry, after `_base_init` and before preprocessing, so the 27 scripts that build datasets through `get_dataset_cls` directly keep the filter | 4 |
| `sampler_epoch_reset` | `spectrum_pl` 2454-2500, both branches: `set_epoch` with the Lightning batch-count refresh, and `_pre_compute_batches` | 4 |
| `apply_inference_config` | script_utils 462-471, 477-486, 509-515 (FraGNNet `frag_params`, GRAFF `ann_params`). `prec_type_str` stays core | 1 (GRAFF), 4 (FraGNNet) |
| `export_index_vars` | script_utils 850-870 and 893-978 | 4 |
| `inference_batch_options` | explainability switch at script_utils 1075/1107-1111 and 2127 | 4 |
| `supports_rule_provenance` | script_utils 2112 | 4 |
| `postprocess_export_peaks` | script_utils 2149-2200, including the `annotation_tables` call at 2167; MASSter's `annotation_tables` body is 1754-2063 | 4 |
| `formula_row_deltas` | script_utils 2414 | 4 |
| `node_hash_features` | script_utils 2586 (`get_node_feats` for the cc hash) | 4 |
| `log_resolved_settings` | runner 1069 | 4 |

The imports at script_utils.py:20-22 disappear only when every export hook has landed.

## Phases

Each phase is one or more small PRs, committed one plan step at a time on
`refactor/fragnnet-masster-reorg` (user decision, 2026-10-11). Every phase must leave the tests
passing and the retained-model records identical. ICEBERG retirement is the explicit exception.

### Phase 0 — Safety net

Each later phase moves a specific behavior, so the safety net exercises that behavior rather than
a default batch.

**Determinism.** Records are generated and compared under these settings:

- CPU only, in the same environment: `accelerator: cpu`, `precision: 32`, `compile: false`, `num_workers: 0`;
- `seed_everything`/`th.manual_seed` before every step, `th.use_deterministic_algorithms(True, warn_only=True)` and `th.set_num_threads(1)`; a non-deterministic-op warning counts as a failure;
- MASSter uses `masster_candidate_backend: cpu` and `masster_proposal_method: cut_masks`.

Records are regenerated deliberately when the torch or numpy build changes. GPU parity (the CUDA
backend, `connected_fragment_beam`) is checked separately with a stated tolerance, not
bit-for-bit. Records are stored under `tests/` and built from in-code synthetic fixtures, so they
do not depend on the untracked `data/` directory.

1. **Registry coverage.** Assert the resolved model and dataset classes against an expected-mapping table kept in the test.
   - **Inputs covered:** every `model_type` the runner accepts today (`frag_gnn`, `fragnnet`, `masster`, `neims`, `chemlm`, `massformer`, `iceberg_gen`, `iceberg_inten`, `graff`, `precursor`, `gnn`, `mces_pretraining`, `clip`, `spec_mol_clip_pretraining`) plus a legacy `mass_shatter` config.
   - **How the table changes:** it starts as today's `get_model_cls`/`get_dataset_cls` and changes only deliberately, in the same PR as the code:
     - Phase 1 replaces the ICEBERG rows with retirement-error checks for every path in Phase 1 step 1;
     - Phase 1 maps `massformer` to `SpecMolMFDataset`, with the equivalence shown against item 3;
     - Phase 6 changes module paths only.
   - **Alias pairs** also get identical `validate_config` and migration results.
2. **Smoke test per retained model type.** Build from the template plus a tiny config; run one forward pass and one training step.
3. **Baseline records (MassFormer, GRAFF)**, captured before any Phase 1 change:
   - per-item `mol_mf`/`mol_graff` preprocessing output;
   - the collated batch through the runner's `init_dataloader` (keys, shapes, dtypes, values);
   - forward outputs, loss and post-step parameters.

   The fixture builds the compiled `algos` extension for MassFormer and a small `ann_fp` library for GRAFF.
4. **Enabled-loss records (FraGNNet, MASSter).**
   - **Required config:**
     - `loss_type: hellinger_distance` (leak and auxiliary are rejected otherwise, loss.py:83-84);
     - the three shared weights above zero;
     - `ce_pair_min_endpoint_confidence` in (0, 1] and `target_rel_ints_thresh` in (0, 1);
     - `inst_type_loss_tol` set for each eligible instrument type, `train_sample_weight: none`, `devices: 1` (runner.py:588-690);
     - a CE-context sampler: for MASSter, `masster_batch_sampler: true` with `masster_batch_sampler_max`; for FraGNNet, `batch_sampler_type: ce_context`, `dynamic_batch_sampler: true`, `dynamic_batch_sampler_mode: frag_edge` and `automatic_optimization: false`, so the optimizer step is `_manual_opt`.
   - **Fixture.** In the train split, at least two spectra of one molecule, adduct and instrument share `group_id` and differ in CE. Their top peaks A ≠ B are both observed at both CEs, above the intensity floor and the confidence threshold. Predicted peaks lie within tolerance of A and B; build the fixture after a forward pass, or pin wide tolerances (eligibility rules: loss.py:3139-3248).
   - **Assert** `ce_pair_eligible_endpoints.sum() > 0`.
   - **Record** every named loss term, the total, the gradients and the parameters after one optimizer step.
5. **MASSter stages.** `MassterPL` rejects RT/CCS heads combined with proposal-only warmup ([masster_pl.py:901-903](../../src/fragnnet/pl_model/masster_pl.py#L901)), so three cases:
   - proposal-only training with property heads off: the item 4 records, plus an assertion that the spectral terms contribute no gradient;
   - joint training with RT/CCS heads on: the item 4 records;
   - a rejection test: RT/CCS heads with `masster_proposal_only_until_epoch` set still raise.
6. **Samplers.**
   - **Coverage:** FraGNNet's dynamic, dual-group and CE-context samplers; the bond-count curriculum; MASSter's CE-context sampler; and the generic group samplers.
   - **Recording:** record the batch-index sequence per epoch by driving `SpectrumPL._seed_dataloader(epoch)` through a stub trainer that exposes `train_dataloader`. Calling `set_epoch` directly bypasses the reseed.
   - **Resume check:** build a fresh dataloader, call `_seed_dataloader(N)` as a resumed run would, and require epochs N and N+1 to equal the uninterrupted run. Samplers hold no checkpointed state; this guards the `(hparams.seed, current_epoch)` derivation.
7. **Export.** For FraGNNet and MASSter, record the schema (columns, dtypes, HDF5 dataset names) and values of the pkl, tabular and HDF5 inference exports on a fixture, including the explainability-switch and rule-provenance paths.
8. **Migrations.** For one stub old checkpoint and one template per model family (including legacy `mass_shatter`), record the backfilled hparams after `get_pl_hparams` and `load_config`. Take the stub hparams from `saved_ckpts/c0_control_vcpho5qm.ckpt` into a small JSON fixture, so the test does not need the untracked file.
9. **Dataset construction path.** A dataset built through `get_dataset_cls("masster")(**config)`, bypassing `runner.init_dataset`, still applies the MASSter spectrum filter.
10. **Import boundary.**
    - **AST check:** the dependency rule, using the module-to-layer table and a `COMPAT_SHIMS` allowlist.
    - **`sys.modules` gate:** runs in fresh subprocesses.

    Both start report-only and enforce what each phase resolves (gate table below).

### Phase 1 — Registry, retire ICEBERG, decouple retained baselines

Phase 1 delivers `model_cls`, `dataset_cls`, the alias table, `RETIRED_MODEL_TYPES` and GRAFF's
`apply_inference_config`. Phase 4 delivers the remaining hooks.

1. **Lazy registry** in `registry.py` for **every** retained model type, FraGNNet and MASSter included, so `runner` → `setup_utils` no longer imports any model at load time. `VALID_MODEL_TYPES` is derived from it and excludes retired types. The retirement message is raised in two places:
   - `load_config`, after the custom/checkpoint/template resolution (runner.py:203-208) and before the membership check (:209). This covers custom configs, the template and the `model_type` argument that `get_config_from_ckpt` passes.
   - The registry's `get_model_cls`/`get_dataset_cls`, which covers `init_model_from_ckpt` and every script.
2. **Retire ICEBERG.** Record the archive SHA (see the decision section), then delete `src/fragnnet/iceberg`, its imports in `setup_utils` and `script_utils`, the `isinstance(IcebergIntenPL)` branches and the MAGMA inference overlays (script_utils 481-483, 514-515; the `magma` entry of the 465 loop is harmless and goes too).
   - **Configs:** delete the 51 config and sweep files: 45 per-seed configs under `config/{massspecgym_merged,nist20_merged_inchikey,nist20_merged_scaffold}/{iceberg_gen,iceberg_inten,iceberg_inten_opt}/`, 3 `config/sweeps_merged/iceberg_*` templates and 3 `sweep/merged/iceberg_*` files. Git history at the archive SHA is the pointer.
   - **ICEBERG-only files, deleted:**
     - `preproc_scripts/05_prepare_magma_feats.py`, `06_predict_magma_dags.py`;
     - `scripts/ms2c/run_iceberg_ms2c_predict_val.py` (already unimportable: it imports functions `script_utils` no longer defines);
     - `notebooks/test_notebooks/test_iceberg_formula.py`.
   - **Importers edited:**
     - `scripts/eval/run_save_inference.py`: remove the ICEBERG imports (10-11), the `iceberg_inten`/`iceberg_inten_opt` entries in its four tables (35-36, 109-110, 120-121, 131-132) and the default `--wandb_model_types`. It already fails at HEAD for unrelated reasons: `load_config` is called without `override`, and `apply_inference_config` is passed a `model_type` argument it does not have. Repairing that is out of scope.
     - `scripts/stats_scripts/calculate_formula_stats.py`: delete the unused import (5), the `iceberg_inten` default (29) and the ICEBERG branch (181-185). No port to core is needed.
     - `notebooks/param_counts.ipynb`: drop the ICEBERG cell or mark the notebook historical.
   - **Files that only name ICEBERG:**
     - `scripts/setup_job_cc.py:538-539`: drop the ICEBERG branch; the `magma_dp` keyword at 501 stays while the template key stays;
     - `scripts/ms2c/setup_ms2c_scripts.py`: drop the ICEBERG targets;
     - `scripts/stats_scripts/calculate_similarity_stats.py:17-18`: drop the ICEBERG labels;
     - `scripts/debug/dag/debug_global_cutoff_recall_eval.py`: the docstring cites the archive revision;
     - `.pre-commit-config.yaml`: drop `iceberg` from the comment at :7 and the regex at :15;
     - `config/template.yml:213-216` and `:1225-1226`: annotate as retired and point the stale `TEMPLATE_STEM_BY_MODEL_TYPE` comment at the registry.
   - **Tests:** `tests/test_config_templates.py::test_every_model_type_is_accounted_for` drops `iceberg_gen`/`iceberg_inten` from `owned` and asserts `RETIRED_MODEL_TYPES` is disjoint from `VALID_MODEL_TYPES`. Retired configs are deleted, so `test_every_tracked_config_still_resolves` keeps its 18 known failures. The `ICEBERG only` section, `magma_params` and `iceberg_*` template assertions stay.
   - **Docs:**
     - `docs/architecture.md:21`;
     - `docs/guides/run_msgym_retrieval.md:112-113`;
     - `config/README.md:114-120` (already-dead ICEBERG links → retirement note and ms-pred pointer);
     - `preproc_scripts/README.md` Steps 5-6 (mark retired, point to the archive SHA, renumber if needed);
     - `INSTALL.md:99` and `:207`;
     - `docs/findings/masster/masster_channel_selection_findings.md:191`: repoint to a permalink at the archive SHA. Path-only citations in plans stay as historical.
   - **Local checkpoints:** the two local ICEBERG checkpoints get a provenance note naming the archive SHA and are not loaded in the main environment.
3. **Move MassFormer and GRAFF molecule handling out of `SpecMolDataset`.** The `mol_params.mf`/`graff` keys stay.
   - **Why a subclass override is not enough.** Collation is dispatched statically, not by inheritance: `SpecMolDataset.get_collate_fn()` returns `SpecMolDataset.collate_fn`, which calls `SpecMolDataset._special_collate` by name ([spec_mol_dataset.py:242-298](../../src/fragnnet/dataset/spec_mol_dataset.py#L242)).
   - **MassFormer:** a new `SpecMolMFDataset(SpecMolDataset)` defines its own `get_collate_fn`, `collate_fn` and `_special_collate` (the last calling `SpecMolDataset._special_collate`), following GRAFF's pattern ([graff/dataset.py:264-306](../../src/fragnnet/graff/dataset.py#L264)). The registry maps `massformer` to it.
   - **GRAFF:** `SpecMolAnnDataset` already has that chain; the `mol_graff` block moves into its `_special_collate`.
   - **Callers to retarget:** `scripts/eval/run_save_inference.py:117` maps `massformer` to `SpecMolDataset`; change it to `SpecMolMFDataset`. A grep check asserts no module hard-codes `SpecMolDataset` for `massformer`. Its table keys are W&B labels, not registry types, so it keeps its own tables.
   - **Testing:** test through the DataLoader the runner's `init_dataloader` builds, against the Phase 0 item 3 records.
4. **Remove DGL.**
   - **Code:** delete the `DGLGraph` cases from `misc_utils.to_device` and `print_shapes`, then delete `dgl_compat_utils.py`.
   - **Tracked install sites:**
     - `pyproject.toml:24`;
     - `env/env_cc_setup_py310.sh:25`, `env_cc_setup_py311.sh:25`, `env_setup_gpu_py312_cu121_251105.sh:32-33`;
     - `install_lib.py:85`, `:100` and the `"dgl==1.0.4"` list element at `:104`;
     - `scripts/setup_job_cc.py:395`;
     - the `scripts/manual_slurm/` files (seven under `fir/`, `vulcan/inference_mas.sh:41`);
     - `INSTALL.md:99`, `:207`;
     - the `dgl.random_walk_pe` docstring in `feat_utils.py:1156-1158`.
   - **Shell-continuation trap.** In `env/env_setup_gpu_py311_cu121_250602.sh`, line 25 (`matplotlib \`) continues into the comment on line 26, which ends the first `pip install`; line 27 starts a second `pip install` that also installs omegaconf through h5py. End line 25 without `\`, delete line 26 and change line 27 to `pip install \`. Deleting line 26 alone would join `pip install` into the first command's package list.
   - **Shell checks:** check each edited shell script with `bash -n` and a stubbed-pip dry run that shows the same package set minus dgl.
   - **Untracked files:** these are not part of the PR: `containers/apptainer/*.def` (gitignored), generated `slurm_scripts/` and `jobs/`. Mark the `.def` stale locally and regenerate job scripts from `setup_job_cc.py`. The prebuilt `.sif` images still contain DGL.
   - **DGL-free check:** parse `pyproject.toml` (tomllib) and grep the tracked install files for `dgl`. Do not check the live environment: the editable install's metadata and the gitignored `src/fragnnet.egg-info` still report `dgl==1.0.4` until the user chooses to re-run `pip install --no-deps -e .`, which the refactor does not do.
5. **Package initializers.**
   - **Reduce to core exports.** Reduce `model/__init__.py`, `pl_model/__init__.py` and `dataset/__init__.py` to core exports. NEIMS stays a core export through Phase 5 (open decision 2). Remove the lazy names from `dataset/__init__.py`'s `__all__`.
   - **Transition.** The model-specific names stay reachable through a module-level `__getattr__` (PEP 562) that maps only the known lazy names, caches each result in `globals()` and raises `AttributeError` for anything else, so submodule imports keep working. These `__getattr__` functions are the `COMPAT_SHIMS` the boundary check exempts until Phase 6.
   - **Tests:** one case per import form (`from pkg import LazyClass`, `from pkg import submodule`, `import pkg.sub as m`, `monkeypatch.setattr(pkg, "LazyClass", fake)`).
   - **What it achieves:** importing a shared submodule then no longer loads any baseline. FraGNNet and MASSter loading through `spectrum_pl`, `feat_utils`, `proc_utils`, `script_utils` and `data_sampler` is removed in Phases 3-4.

**Isolation gate.** Fresh subprocesses import core entry modules and select models through the
registry, then check `sys.modules`. DGL imports are blocked by an import hook that raises. The
gate enforces only what has been resolved; the rest is report-only:

| After phase | Entry modules checked | Importing them must not load | Selecting a model must not load | Report-only |
| --- | --- | --- | --- | --- |
| 1 | `runner`, `script_utils`, `spectrum_pl`, `spec_mol_dataset` | any baseline module, `dgl*` | another baseline, `dgl*` | FraGNNet and MASSter modules |
| 3 | `runner`, `spectrum_pl`, `spec_mol_dataset`, `base_dataset`, `feat_utils`, `proc_utils`, `data_sampler`, `pl_utils` | additionally `frag_utils`, `compute_frags`, `multi_cut_bfs`, `smarts_prepass`, `fragnnet_model`, `bond_fragment_heads` | another model's modules; FraGNNet and MASSter may load `fragmentation/` | `script_utils` |
| 4 | all of the above plus `script_utils` | any FraGNNet or MASSter module | as above | none |

The Phase 3 row was checked with a transitive import calculation at `13b1599d`. It assumed lazy
initializers, a lazy registry and the Phase 3 edges removed, including the `proc_utils` →
`compute_frags` edge. Without that edge's removal, `runner`, `spec_mol_dataset`, `base_dataset`
and `data_sampler` still load `compute_frags`. With it removed, every Phase 3 entry module is
clean. Selection after Phase 3 behaves as follows:

- NEIMS loads no fragmentation or model-specific module.
- MassFormer and GRAFF load only their own packages.
- MASSter loads `smarts_prepass`, `candidate_support`, `bond_fragment_heads` and its own modules, but no FraGNNet module.
- FraGNNet loads `candidate_support`, `smarts_prepass`, `frag_utils` and no MASSter module.

The AST boundary check becomes enforcing at the end of Phase 4, with the `COMPAT_SHIMS` allowlist.
Phase 6 moves files without changing what either check enforces.

### Phase 2 — Shared spectrum objectives into `SpectrumPL`

Composition contract:

- **Separate helpers.** `SpectrumPL` provides one helper per term: CE-pair, spectral leak and spectral-CE auxiliary, plus the Hellinger weight validation. There is no single call that evaluates all three. The helpers never add anything to the total loss, and there is no base-class aggregation step after the model's loss function. Setup-time CE-pair eligibility tensors live in the CE-pair helper.
- **Inputs prepared by the caller.** Each helper takes prediction m/z, probabilities or intensities, an explicit OOS argument and the tolerances, so each model keeps its own preparation: FraGNNet's `exp(pred_logprobs)` and OOS handling, MASSter's renormalization and zero OOS.
- **Current call sites.** Each model calls the helpers where the inlined code sits today, in its current order (see [Lightning modules](#lightning-modules-and-their-loss-terms)). MASSter keeps the proposal-only replacement and the RT/CCS additions after it.
- **Names unchanged.** Logged names (`ce_pair_loss`, `ce_pair_eligible_endpoints`, `spectral_leak_loss`, `spectral_ce_auxiliary`) and the `loss_names` lists stay the same.

Gate: the Phase 0 item 4-5 records match exactly for FraGNNet, MASSter proposal-only and MASSter
joint training: per-term values, totals, gradients and post-step parameters. Because the call
sites and operation order are unchanged, gradients are expected to be bit-identical. If a combined
helper is ever preferred, values stay bit-for-bit, and gradients get a stated tolerance, because
backward accumulation order changes.

### Phase 3 — `fragmentation/`, core constants and the `frag_utils` split

1. **Validator to core.** Move `_validate_tensor_condition` from `bond_fragment_heads` to core; MASSter and `candidate_support` import it from there.
2. **Core output-schema module**, holding `EXPLAINABILITY_OUTPUT_KEYS` (documented as the gated set), the annotation key names and the `return_explainability_tensors` per-model defaults.
   - Point `SpectrumPL` (spectrum_pl.py:23, :609) at it.
   - `fragnnet_model` may keep a re-export, since it is FraGNNet-owned. Otherwise retarget `tests/test_explainability_demand.py` and `tests/test_skip_train_explainability.py`.
   - Optionally derive the exporter's ColumnSpec names (script_utils.py:640-660) from it.
3. **Core size constants.**
   - Define `MAX_NUM_NODES`, `MAX_NUM_EDGES` and `MASK_SIZE`, with their asserts, in core `formula_utils`.
   - Point `proc_utils.py:6` at them.
   - The FraGNNet `frag_utils` remnant asserts they equal `compute_frags.MAX_NUM_NODES/EDGES`, and a test asserts the `multi_cut_bfs.pyx` copies match.
   - Retarget `preproc_scripts/inference/01_inf_prepare_mol_df.py:31` and `preproc_scripts/ms2c_pubchem/02_prepare_ms2c_candidates.py:11`, which import `MAX_NUM_NODES` from `compute_frags`.
4. **`fragmentation/`.** Move `smarts_prepass`, `CutBondEncoder`/`encode_cut_bond_pairs`, `candidate_support` and `compute_cc_h_cap`.
   - **No re-exports** in `base_model` or the old `candidate_support` path. Retarget their importers in the same PR: `fragnnet_model.py`, `masster_model.py`, `pl_model/fragnnet_pl.py:18`, `pl_model/masster_pl.py:32-38`, `tests/test_cut_bond.py:28`, `test_masster_candidate_features.py:29`, `test_masster_memorization_route.py:21`, and the patch string at `tests/test_masster_pl.py:717-720`.
   - **`update_bonds`.** Delete the dead Python copy. The Cython `update_bonds`/`compute_cc_h_floor` stay in FraGNNet as the production implementation. Optionally add a NumPy reference `update_bonds`/`compute_cc_h_floor` (normalized to int32/uint8) in `fragmentation/` with a Cython-versus-reference parity test. Never make the Python version the production one: it is about 100× slower on the `compute_dags` hot path and returns a bool mask.
   - **MASSter parity test.** It keeps using `compute_cc_h_cap` and the Cython `update_bonds` as its oracle; a test may import FraGNNet.
5. **Split `frag_utils`** as described in [the split](#frag_utils-is-split-not-moved). The FraGNNet remnant re-exports the names that moved to core until Phase 6. Core gets none.
6. **`feat_utils`.** Move `get_frag_graph`, `batch_mols_frags` and the DAG feature indexing (`NODE_FEAT_TO_IDX`, `EDGE_FEAT_TO_IDX`, `get_node_feats`, `get_edge_feats`) to a FraGNNet-owned module in the current tree. There is no re-export in `feat_utils`. Retarget the 12 importers in the same PR:
   - nine tests: `test_fragnnet_audit_regressions`, `test_cc_atom_dropout`, `test_rdbe`, `test_num_cut_bonds`, `test_frag_q_feature`, `test_skip_train_explainability`, `test_dag_scalar_features`, `test_fragnnet_audit_fixes`, `test_feat_utils`;
   - three scripts: `scripts/debug/fragnnet/debug_cut_pair_large_stats.py`, `scripts/eval/build_cut_pair_targets.py`, `scripts/eval/branching_headroom.py`;
   - `spec_mol_frag_dataset.py:16`, which is FraGNNet-owned.

After this phase, core `feat_utils`, `base_dataset` and `proc_utils` import neither `frag_utils`
nor `compute_frags`. The string-path audit runs and the Phase 3 gate row becomes enforcing. Drop
the `multi_cut_bfs` stub in `tests/test_spectrum_pl_utils.py:23-27` once `spectrum_pl` no longer
loads the extension.

### Phase 4 — Hooks for FraGNNet and MASSter

Implement the remaining hooks and delete the model branches from:

- `runner.py` (including its sampler and validator imports);
- `script_utils.py` (every site in the coupling table);
- `spec_mol_dataset.py` (`_filter_masster_spectra` and the `masster_*` kwargs become MASSter's `filter_dataset`);
- `data_sampler.py`: the FraGNNet samplers and `validate_dag_size_curriculum` move to FraGNNet, and MASSter's sampler to MASSter. The generic samplers and the bond curriculum stay.
- `spectrum_pl.py` (`_seed_dataloader` checks become `sampler_epoch_reset`);
- `pl_utils.py` (MASSter backfills become `migrate_checkpoint_hparams`).

Move `masster_has_valid_ion` from core `formula_utils` into MASSter and retarget
`tests/test_masster_dataset.py:8`. These are the most frequently changed files, so use small PRs
merged quickly.

Gate:

- **Legacy checkpoint:** resolving the config of a legacy `mass_shatter` checkpoint (the Phase 0 item 8 fixture) gives identical hparams before and after.
- **Isolation:** the Phase 4 gate row and the AST boundary check enforce.

### Phase 5 — Packaging

- **Who resolves `pyproject.toml`.** The scripted installs (`env/*.sh`, `setup_job_cc.py:424`, most `scripts/manual_slurm` scripts) use `--no-deps`, so there `pyproject.toml` is metadata only. Other paths do resolve it: `INSTALL.md` (67, 72, 80, 137, 157, 187), `README.md` (45, 57) and `scripts/manual_slurm/fir/msgym_ms2c_{formula,mass}_pred_arr.sh:36` run plain `pip install -e .` and `.[dev]`/`.[chemlm]`. The edits below therefore change what those users get, so update `INSTALL.md` and `README.md` in the same PR.
- **DGL:** removal is completed in Phase 1; no `iceberg` extra.
- **Drop `omegaconf`**, which nothing imports, from `pyproject.toml` and from the install lists in `scripts/setup_job_cc.py:396`, the `env/` scripts and `scripts/manual_slurm/*`.
- **Move `scikit-learn` into the `dev` extra.** Six tests load scripts that import it; optionally also list it in a `scripts` extra.
- **Declare `pyyaml`, `networkx` and `pillow`.** Core imports them at module level, but today they arrive only through other packages; torch also requires networkx.
- **Keep `torch_scatter`.**
  - **Why:** no repo code imports it, but PyG 2.6.1 routes CUDA `max`/`min` scatter with gradients through it when installed (`torch_geometric/utils/_scatter.py:89-119`), and 154 configs set `frag_pool_type: max` (`aggr.MaxAggregation`, [nn_blocks.py:2361](../../src/fragnnet/model/nn_blocks.py#L2361)).
  - **What changes without it:** for tied maxima, `torch_scatter` routes all gradient to one element (`[1, 0]`) while native PyTorch splits it (`[0.5, 0.5]`), as a reviewer's probe in FRAGNNET-GPU showed. Removing it would break behavior preservation; evaluate it separately, with a tied-maximum gradient test.
  - **The other environment:** `env/env_setup_gpu_py312_cu121_251105.sh` installs PyG 2.7.0 and torch 2.6.0 without `torch_scatter`, so an environment built from it already takes the native path and can differ on tied maxima.
- **`requires-python`:** `<=3.13` excludes 3.13.x patch releases. The pinned torch 2.4.1 and rdkit ≤2023.09.5, and the compiled extensions (cp311, cp312 only), do not support 3.13 anyway. Recommended: declare the supported range, `>=3.10,<3.13`.
- **Lock file:** commit a lock (freeze) of FRAGNNET-GPU if `pyproject.toml` is to describe a real environment.

### Phase 6 — Directory moves (deferred)

Runs after the diverged branches (`feiw_v2`, `spectraverse`, others in use) are merged or retired,
and after open decisions 1-2.

1. **Per move, one PR with two commits.**
   - **Commit 1:** a pure `git mv`. The no-edits rule applies to this commit only, so git rename detection works across merges.
   - **Commit 2:** a mechanical rewrite of imports across `src`, `tests`, `scripts`, `preproc_scripts`, `tools` and notebooks, plus every string and dynamic module name (see [String module paths](#string-module-paths)).

   Gate: grep for the old dotted paths returns nothing outside `docs/history` and frozen findings.
2. **Build.**
   - Create `__init__.py` in every new package directory (`models/`, `models/masster/`, `models/fragnnet/`, `baselines/`, `baselines/graff/`, `baselines/massformer/`, `fragmentation/`) before building: cythonize derives extension names from `__init__.py` files, and `packages = find` skips directories without one.
   - Update the `.pyx` paths in `setup.py` and the package-data keys in `pyproject.toml`. The `.cu` files are declared only there, under `"fragnnet.frag"`, and each `.py`/`.cu` pair moves together because `masster_cuda_*.py` load `Path(__file__).with_suffix(".cu")`.
   - Verify that `python -m build --wheel` contains both `.cu` files next to their `.py`, and that a non-editable `pip install -I --no-deps .` imports and compiles the CUDA modules. Cluster installs are non-editable.
3. **Stale binaries.** Gitignored `.so`/`.c` files stay at the old paths after a move and keep old imports working locally, because extension suffixes win over `.py`. After each move:
   - hand the user a cleanup script for `src/fragnnet/**/*.so`, `*.c` (including the stale `mass_shatter_core` build) and the root `build/`; agent `rm -rf` is blocked;
   - rebuild;
   - run the gate, import tests and smoke fixtures from a fresh `git worktree`.

   Change the cleanup command at `INSTALL.md:127-131` to cover subdirectories.
4. **Hard-coded paths:** `tools/build_compute_frags_setup.py`, `scripts/profile_compute_frags.py`, the benchmark script and its test, the `.pre-commit-config.yaml` comment and regex, the `.gitignore` entries for generated `.c` files, `.claude/commands/add-smarts-rule.md` (untracked), and the notebooks listed under [Baselines](#baselines).
5. **Checkpoints.** Load one existing checkpoint per retained model type that has one locally, through `get_pl_hparams` and `load_state_dict(strict=True)`, before and after each move:
   - `frag_gnn`/`fragnnet`;
   - `masster`, including legacy `mass_shatter` (`saved_ckpts/c0_control_vcpho5qm.ckpt`);
   - `neims`.

   For GRAFF and MassFormer, fetch one checkpoint each from W&B (`graff3_long`, `massformer_long`) with `init_wandb_ckpt`. Where none is available (chemlm, precursor, gnn, clip, mces), record the `state_dict` key names and shapes of a randomly initialized model before the move and assert them afterwards. That proves less than loading a legacy checkpoint, and the plan says so.
6. **Remove shims.** Remove the `frag_utils` remnant's re-exports and the package-initializer `__getattr__` names, and retarget the scripts, preprocessing scripts and tests that still import moved names from old paths.

### Phase 7 — Docs (path sweep after Phase 6)

- **`docs/architecture.md`:** add the dependency rule, the plugin interface and a note that frozen records cite the old layout.
- **`INSTALL.md`:** extras, and the missing `env_setup_gpu_py310_cu118.sh` reference.
- **`docs/reference/baselines.md`:** REFERENCE banner and a row in `docs/README.md`. It covers running the retained baselines, their config keys, and external ICEBERG through ms-pred, including archive provenance and the evaluation-file contract.
- **Path and link sweep.** After each phase that moves or deletes a module, run a relative-link and path check over the live docs:
  - `README.md`, `docs/README.md`, `docs/architecture.md`;
  - `docs/reference/` (including `dataloader_data.md` after the MassFormer dataset change), `docs/guides/`, `docs/workstreams/`, `docs/notes/`;
  - the `config/template.yml` comments at :215 and :872.

  Repoint what is live. Frozen findings, plans and history keep their links; add one line to `docs/rules/documentation.md`: "paths in frozen records are as of `13b1599d`; use git history to resolve them".
- **Phase 1 docs** are done in Phase 1, not here.

## Unchanged

- Retained-model `model_type` strings, config keys, config paths, sweep files and W&B runs. ICEBERG types and configs are the explicit retirement exception. The `iceberg_*`/`magma_params`/`magma_dp` template fields stay inert, so retained-model config and checkpoint loading is not broken by a separate schema cleanup.
- `config/template.yml` stays a single file.
- **Checkpoints store no class or module paths.** A scan of all 166 local `*.ckpt` files under `ckpts/` and `saved_ckpts/` found only torch storages, numpy and `collections` globals. `hyper_parameters` values are plain types, and the custom writers (`spectrum_pl.py:2502`, `pl_utils.py:408-422`) add only RNG state, metrics and EMA tensors. No `pickle.dump`, `torch.save` or `to_pickle` exists in baseline code or MAGMA preprocessing.

## Risks

- **Changed computation paths.** Phase 1 step 3 and Phase 2 change which code computes a value. The records must be identical, not merely close, under the Phase 0 determinism settings.
- **Gradient leaking into proposal-only training.** Phase 2 could silently put gradient back into MASSter's proposal-only stage. The composition contract and the per-stage gradient checks exist for this.
- **Merge conflicts.** Phases 2-4 edit the three most-changed files, so long-lived PRs will conflict.
- **Phase 6 renames.**
  - Jobs that were generated for the old layout but start after the move break: a generated job copies `src/` when its script starts running ([setup_job_cc.py:419](../../scripts/setup_job_cc.py#L419)), not when it is submitted.
  - Any branch that edited a moved file will conflict.
  - Stale local binaries can hide a missed import.
- **Mixing ICEBERG implementations.** Local ICEBERG results and ms-pred ICEBERG results come from different implementations and checkpoints. Label which one a number came from.
- **Limits of the analysis.** Coupling counts come from static import analysis and config-banner and regex classification. Classes selected by name at runtime (the `nn_blocks` GNN types, featurizer registries) and string module paths are invisible to it; the string-path audit covers the latter.

## Open decisions

1. **Layout.** Recommended: `fragnnet.models.{fragnnet,masster}` and `fragnnet.baselines.*`. The alternative, separate top-level packages or pip distributions for the retained baselines, is only worth it if they will be published or installed separately. Gates Phase 6 only.
2. **NEIMS.** It is a baseline but lives in core `model/` and `pl_model/`. It stays a core export, and is not a "baseline module" for the Phase 1 gate, through Phase 5. Phase 6 moves it to `baselines/neims` only if this decision says so. The `neims_*` keys stay in core either way, because ChemLM reads them. `chemlm`, `gnn`, `precursor`, MCES and CLIP stay in core.
3. **Baseline config directories.** Recommended: do not move them into `config/baselines/`, since sweeps and W&B runs point at the current paths.
4. **Live branches.** Which diverged branches are still in use; this sets when Phase 6 can run.

ICEBERG ownership is settled: use external ms-pred and archive the local fork through Git
provenance.

## Progress

Entries before 2026-10-11 (second half) are in the [plan log](../history/repo_reorg_plan_log.md).

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
    modules, never each other's. AST violations: 11 at the end of Phase 1, now 5 (`script_utils`
    and `runner`, which go in Phase 4).
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
