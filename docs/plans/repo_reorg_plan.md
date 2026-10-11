# Repository reorganization: FraGNNet and MASSter at the center

> **Status:** PROPOSED · **Updated:** 2026-10-11
> Written up, not started. Layout and scope decisions are open (see [Open decisions](#open-decisions)).
> Line references below were measured at commit `13b1599d` on branch `masster`.

## Goal

Center the repository on FraGNNet and MASSter. Each model, and each baseline (GRAFF, ICEBERG,
MassFormer), becomes its own subpackage. The shared layer stops naming any specific model.

The folders are the smaller part of this. Most of the work is removing the model-specific branches
from the shared runner, inference export, dataset and sampler code. A directory move without that
step changes paths and leaves the coupling in place.

## Target layout

```
src/fragnnet/
  dataset/ model/ pl_model/ utils/ runner.py   # core: model-agnostic, import paths unchanged
  registry.py                                  # model_type -> "module:Class", imported lazily
  fragmentation/                               # bond-cut chemistry shared by both fragment models
  models/
    fragnnet/                                  # DAG model
    masster/                                   # MASSter
  baselines/
    graff/ iceberg/ massformer/
```

Dependency rule, enforced by a test:

- `models/fragnnet`, `models/masster` and each baseline may import `fragmentation/` and core,
  never one another.
- `fragmentation/` imports core only.
- Core imports none of them, `fragmentation/` included; it reaches models only through the
  registry. Anything core consumes (for example the annotation output keys) therefore lives in
  core, even when only fragment models produce it.

Shared modules stay where they are rather than moving into a `core/` package. They hold about
32k lines, and moving them would change nearly every import in the repo. `fragnnet.fragnnet` is
avoided because the top-level package already has that name. One cosmetic wart remains: shared
`model/` sits next to `models/`; renaming it to `nn/` later is optional.

## Current coupling

### Sizes

| Group | Lines | Files |
| --- | --- | --- |
| MASSter-only | ~17.9k | `frag/masster_*`, `masster_core.pyx`, `.cu` kernels, `bond_mask_generator`, `masster_model`, `bond_fragment_heads`, `candidate_features`, `rt_ccs`, `masster_pl` |
| FraGNNet-only | ~6.8k | `compute_frags.pyx`, `multi_cut_bfs.pyx`, `fragnnet_model`, `fragnnet_pl`, `*spec_mol_frag_dataset` |
| Shared | ~32k | `SpectrumPL`, loss, `nn_blocks`, encoders, utils, runner |
| Baselines | ~9.8k | ICEBERG 6.5k, MassFormer 2.0k, GRAFF 1.4k |

`MassterPL` and `FraGNNetPL` are siblings: both subclass `SpectrumPL`. Their direct imports of
each other are limited to `CANONICAL_ELEMENT_ORDER` (see the chemistry section), but there is one
indirect FraGNNet → MASSter dependency: `fragnnet_pl` imports `candidate_support`, which imports
`_validate_tensor_condition` from the MASSter module `bond_fragment_heads`
([candidate_support.py:8](../../src/fragnnet/model/candidate_support.py#L8)). The validator is
also used by `candidate_features` and `masster_model`.

### Shared layer depends on the models

| Where | What it knows about a model |
| --- | --- |
| [spectrum_pl.py:23](../../src/fragnnet/pl_model/spectrum_pl.py#L23) | Imports `EXPLAINABILITY_OUTPUT_KEYS` from `fragnnet_model` |
| [spectrum_pl.py:2469](../../src/fragnnet/pl_model/spectrum_pl.py#L2469) | `_seed_dataloader` checks `dag_size_curriculum`, `bond_size_curriculum`, `masster_batch_sampler` |
| [spec_mol_dataset.py:57](../../src/fragnnet/dataset/spec_mol_dataset.py#L57) | Filters spectra when `model_type == "masster"` via `formula_utils.masster_has_valid_ion` |
| [spec_mol_dataset.py:7,237-280](../../src/fragnnet/dataset/spec_mol_dataset.py#L7) | MassFormer `mf` preprocessing and GRAFF `mol_graff` collation |
| [data_sampler.py](../../src/fragnnet/dataset/data_sampler.py) | FraGNNet samplers (`SpecMolFragDynamicBatchSampler`, `DualGroupDynamicBatchSampler`, `BondCountCurriculumSampler`) and MASSter's `MassterCEContextDynamicBatchSampler`; only `_CEContextBatchSamplerBase` and the group samplers are generic |
| [runner.py](../../src/fragnnet/runner.py#L218) | 8 `model_type` branches (lines 218, 331, 345, 515, 556, 806, 1069, 1267) and a hard-coded `VALID_MODEL_TYPES` |
| [setup_utils.py:1-16](../../src/fragnnet/utils/setup_utils.py#L1) | Imports every model and baseline when the module is imported, so dgl loads with core |
| [script_utils.py:850-940](../../src/fragnnet/utils/script_utils.py#L850) | 7 `isinstance` branches (`FraGNNetPL`, `MassterPL`, `IcebergIntenPL`) |
| [script_utils.py:481-515](../../src/fragnnet/utils/script_utils.py#L481) | MAGMA (ICEBERG) and GRAFF inference-config edits |
| [script_utils.py:1754-2063](../../src/fragnnet/utils/script_utils.py#L1754) | MASSter annotation export (`MassterAnnotationTables`, `masster_annotation_tables`) |
| [pl_utils.py](../../src/fragnnet/utils/pl_utils.py) | `migrate_legacy_masster_name`, `migrate_masster_proposal_method`, `migrate_masster_rt_ccs_hparams` |
| [dgl_compat_utils.py](../../src/fragnnet/utils/dgl_compat_utils.py) | Used by ICEBERG and by shared [misc_utils.py:42](../../src/fragnnet/utils/misc_utils.py#L42), whose `to_device` and `print_shapes` check `isinstance(v, DGLGraph)` |

### Loss terms copied in both Lightning modules

The CE-pair objective is duplicated almost line for line: eligibility tensors built from
`ce_pair_eligible_*`, the `ce_switch_pair_bce` call, weighting and logging.

- FraGNNet: [fragnnet_pl.py:786-800](../../src/fragnnet/pl_model/fragnnet_pl.py#L786) and
  [1016-1043](../../src/fragnnet/pl_model/fragnnet_pl.py#L1016)
- MASSter: [masster_pl.py:1660-1675](../../src/fragnnet/pl_model/masster_pl.py#L1660) and
  [2185-2226](../../src/fragnnet/pl_model/masster_pl.py#L2185)

`SpectrumPL` already prepares the `ce_pair_*` batch fields
([spectrum_pl.py:1460](../../src/fragnnet/pl_model/spectrum_pl.py#L1460)); only the loss term is
copied. `spectral_leak_loss` and `spectral_ce_auxiliary` (with
`validate_hellinger_auxiliary_weights`) are also implemented in both. How closely those two match
has not been checked line by line. The two modules override only `_setup_model` and
`_setup_loss_fn`; each keeps all its loss logic in one closure.

The order in which terms enter the total differs by model and stage. In MASSter, the spectral
terms and the CE-pair term are added first; then, during proposal-only training, the total is
**replaced** by the proposal loss (plus the ion-mode proposal loss), and only afterwards are the
RT/CCS losses added ([masster_pl.py:2227-2250](../../src/fragnnet/pl_model/masster_pl.py#L2227)).
The spectral terms are still computed and logged in that stage but carry no gradient. Any
extraction must preserve this.

### Annotation output contract

FraGNNet defines the formula/node/joint index outputs (`pred_formula_*`, `pred_node_*`,
`pred_joint_*`). MASSter's annotations are exported under those names, and `SpectrumPL` metrics
(`pred_node_count`, `pred_formula_count`) read them. The key list `EXPLAINABILITY_OUTPUT_KEYS`
lives in `fragnnet_model.py`. The schema has no home of its own. Because core (`SpectrumPL`,
inference export) consumes it, it belongs in core, not in `fragmentation/`.

### `smarts_prepass` is shared

It sits in `frag/` beside the MASSter modules but has two importers: FraGNNet's DAG builder
([frag_utils.py:34](../../src/fragnnet/utils/frag_utils.py#L34), via
`_inject_smarts_pairs_into_dag`) and MASSter's templates
([masster_templates.py:13](../../src/fragnnet/frag/masster_templates.py#L13)). Not running SMARTS
for FraGNNet (VRAM) is a run-time choice; the code dependency remains.

### Chemistry implemented more than once

| Rule | Implementations |
| --- | --- |
| Hydrogen cap | MASSter torch, `h_transfer_bound="valence"` ([masster_model.py:2441](../../src/fragnnet/model/masster_model.py#L2441)), and `frag_utils.compute_cc_h_cap` ([:394](../../src/fragnnet/utils/frag_utils.py#L394)). [test_masster_model.py:36](../../tests/test_masster_model.py#L36) uses FraGNNet's as the reference |
| `update_bonds` | Twice inside FraGNNet: [compute_frags.pyx:485](../../src/fragnnet/frag/compute_frags.pyx#L485) and [frag_utils.py:578](../../src/fragnnet/utils/frag_utils.py#L578) |
| Fragment charge | MASSter `candidate_features._build_charge` ([:312](../../src/fragnnet/model/candidate_features.py#L312)) and FraGNNet's `frag_q` |
| Element order | `CANONICAL_ELEMENT_ORDER`/`CANONICAL_H_IDX` are defined in `frag_utils` and read by MASSter (`masster_model`, `candidate_features`) **and by core**: the `AtomOneHot` featurizer ([feat_utils.py:335](../../src/fragnnet/utils/feat_utils.py#L335)) |

### Two metadata encoders

FraGNNet uses the `CEModel`/`PrecModel`/`InstModel` mixins, `ContextEmbedding` and four scalers
from [base_model.py](../../src/fragnnet/model/base_model.py). MASSter has its own
`MassterContextEncoder` ([masster_model.py:675](../../src/fragnnet/model/masster_model.py#L675)),
reusing only `QETEnergyEncoder` and `get_embedder`. Both read the same `nce_*`, `ace_*` and
`ce_insert_*` keys. Keeping two encoders is a reasonable design and this plan does not merge them.
Whether both standardize CE identically is unverified.

### Symbols both models use

| Symbol | Now in | Destination |
| --- | --- | --- |
| `SpectrumPL`, `get_sparse_cross_entropy_fn`, `scatter_reduce`, `GNN`, `get_embedder`, `get_mol_feats_sizes` | core | stay |
| `ce_switch_pair_bce`, `sparse_outside_window_intensity_fraction`, `validate_hellinger_auxiliary_weights` | `model/loss.py` | stay; the calling code moves into `SpectrumPL` |
| `PREC_TYPE_TO_MASS_DIFF`, `PREC_TYPE_TO_CMF_MASS_DIFF` | `formula_utils.py` | stay |
| `CutBondEncoder`, `encode_cut_bond_pairs` | `model/base_model.py` | `fragmentation/` |
| `observational_channel_targets`, `balanced_binary_support_loss` (all of `candidate_support.py`) | `model/` | `fragmentation/`, after `_validate_tensor_condition` moves to core |
| `_validate_tensor_condition` | `model/bond_fragment_heads.py` (MASSter) | core (a small tensor-validation helper) |
| `EXPLAINABILITY_OUTPUT_KEYS` and the annotation key names | `model/fragnnet_model.py` | core (an output-schema module); read by `SpectrumPL` and the exporter |
| `cmf_duplicate_crf_h_shift` | `formula_utils.py` | stay in core: also called by core `formula_row_prec_deltas` ([formula_utils.py:559](../../src/fragnnet/utils/formula_utils.py#L559)), which the inference exporter uses, and by `masster_has_valid_ion` ([:188](../../src/fragnnet/utils/formula_utils.py#L188)) |
| `CANONICAL_ELEMENT_ORDER`, `CANONICAL_H_IDX` | `utils/frag_utils.py` | core `formula_utils.py`: read by the core featurizer |

Rule for code one model uses today: place it by what it is, not by who calls it. Thirteen losses
are FraGNNet-only today (Hungarian JSD, entropies, Hellinger, pairwise similarity). They are
generic spectrum math and stay in `model/loss.py`. The same applies to `QETEnergyEncoder`, the
nominal isotope tables in `formula_utils`, and the peak aggregation in `spec_utils`.

Second rule: anything core reads stays in core, whatever its subject. A temporary re-export at
the old path does not hide such a dependency; it only moves it.

### Every core import of code the plan moves

An AST check of every `src/` import (module-level and function-level) against the planned
destinations lists all edges from a lower layer to a higher one. Each needs a resolution before
the moves:

| Importer (layer) | Imports from (destination) | Names | Resolution |
| --- | --- | --- | --- |
| `model/__init__`, `pl_model/__init__`, `dataset/__init__` (core) | FraGNNet, MASSter | model, PL, dataset and sampler classes | Package initializers; see below |
| `pl_model.spectrum_pl` (core) | `model.fragnnet_model` | `EXPLAINABILITY_OUTPUT_KEYS` | Core output-schema module (Phase 3) |
| `utils.feat_utils` (core) | `utils.frag_utils` | `CANONICAL_ELEMENT_ORDER`, `NODE_FEAT_TO_IDX`, `EDGE_FEAT_TO_IDX`, `get_node_feats`, `get_edge_feats`, `th_long_to_mask` | Element order to core; `get_frag_graph`/`batch_mols_frags` and the DAG feature indexing move to FraGNNet together, in Phase 3 (in place); `frag_utils` itself loads the Cython builders when imported |
| `dataset.base_dataset` (core) | `utils.frag_utils` | `load_frag_d` | DAG file reader stays in core: shared isotope cleaning reads DAG formulas ([base_dataset.py:744](../../src/fragnnet/dataset/base_dataset.py#L744)) |
| `utils.script_utils` (core) | `utils.frag_utils` | `ELEMENT_TO_IDX`, `MASK_SIZE`, `NUM_ELEMENTS`, `PACKED_DAG_INDEX_DATASET`, `formula_arr_to_str`, `get_node_feats`, `th_mask_to_long` | Formula and DAG-format names to core; `get_node_feats` use to FraGNNet's export hook |
| `utils.script_utils` (core) | `frag.masster_templates` | `MASSTER_RULE_VOCABULARY` | MASSter's `annotation_tables` hook |
| `utils.script_utils`, `utils.setup_utils` (core) | baselines | PL and dataset classes | Registry and hooks (Phase 1) |
| `utils.misc_utils` (core) | `utils.dgl_compat_utils` (to ICEBERG) | `DGLGraph` | Core batch-type registry (Phase 1) |
| `model.candidate_support` (`fragmentation/`) | `model.bond_fragment_heads` (MASSter) | `_validate_tensor_condition` | Validator to core (Phase 3) |
| `model.masster_model`, `model.candidate_features` (MASSter) | `utils.frag_utils` (FraGNNet) | `CANONICAL_ELEMENT_ORDER`, `CANONICAL_H_IDX` | Element order to core |

So `frag_utils` is split, not moved:

- **To core `formula_utils`:** element tables and formula representation (`CANONICAL_ELEMENT_ORDER`,
  `CANONICAL_H_IDX`, `ELEMENT_TO_IDX`, `NUM_ELEMENTS`, `HEAVY_ELEMENTS`, `ELEMENT_TO_VE`,
  `formula_arr_to_str`) and the mask packing helpers (`th_mask_to_long`, `th_long_to_mask`,
  `MASK_SIZE`).
- **To a core DAG-format module:** the `dags.h5`/pickle readers and writers and their layout
  constants (`load_frag_d`, `load_dag_hdf5`, `PACKED_DAG_INDEX_DATASET`, path helpers). The
  shared isotope cleaning and the exporter read this format, and keeping reader and writer
  together keeps the format in one place.
- **To `models/fragnnet`:** DAG generation (`compute_dags` and the Cython builders), DAG node and
  edge feature indexing (`NODE_FEAT_TO_IDX`, `get_node_feats`, …) with `feat_utils.get_frag_graph`
  and `batch_mols_frags`.
- **To `fragmentation/`:** the reference `compute_cc_h_cap` and `update_bonds`.

### Package initializers

`model/__init__.py`, `pl_model/__init__.py` and `dataset/__init__.py` import model-specific
classes when imported: `FraGNNetModel`, `MassterModel`, `FraGNNetPL`, `MassterPL`,
`SpecMolFragDataset`, `IterableSpecMolFragDataset`, and the FraGNNet and MASSter samplers.
Python runs a package's `__init__` before any of its submodules, so importing a shared module
such as `fragnnet.pl_model.spectrum_pl` loads every model. A lazy registry alone cannot isolate
core while these exports exist. 48 files import from these three packages; the model-specific
names are used by 1–13 files each (`MassterPL` 13, `FraGNNetPL` 10, `SpecMolFragDataset` 6).

### Config keys

`config/template.yml` has 474 top-level keys. A regex classification of which code reads each
key (approximate):

| Read by | Keys | Notes |
| --- | --- | --- |
| MASSter only | 117 | All but 3 prefixed `masster_` (`gnn_residual_activation`, `ce_pair_min_endpoint_reliability`, `cross_ev_fragment_loss_weight`) |
| FraGNNet only | 111 | About 80 unprefixed: entropy weights, `nb_*`, `edge_*`, `null_loss`, `candidate_ion_*`, `ion_channel_*`, `finetune_*`, `context_embedding*`, `*_scale`. The count includes a few generic training keys the regex misattributes |
| Both | 45 | `mol_*` encoder keys, loss/output shape, CE statistics, `ce_pair_*`, leak and auxiliary weights, `predict_oos`/`oos_loss` (MASSter reads these only to force them off) |
| Baselines only | 35 | `iceberg_*`, `mf_*`, `graff_*`, `magma_params` |

`template.yml` stays one file (recorded decision). FraGNNet keys are not renamed, because that
would be a schema change. Instead each model package declares the keys it owns. That generalizes
`MASSTER_SAMPLER_KEYS` ([runner.py:97](../../src/fragnnet/runner.py#L97)), so the runner can flag
a key set for a model that ignores it, for every model.

### Baselines

Core reaches into the baselines in five places (listed in the coupling table above:
`setup_utils`, `spec_mol_dataset`, two spots in `script_utils`, `VALID_MODEL_TYPES`). In the other
direction, baselines subclass `SpectrumPL`/`BinnedPL`/`BaseDataset` and use loss, scatter and
formula utilities, which is allowed. No baseline has its own test; only
`tests/test_config_templates.py` touches them. Outside callers:

- [run_save_inference.py](../../scripts/eval/run_save_inference.py)
- `scripts/stats_scripts/calculate_formula_stats.py`
- `scripts/ms2c/run_iceberg_ms2c_predict_val.py`
- `preproc_scripts/05_prepare_magma_feats.py`, `06_predict_magma_dags.py` (ICEBERG-only)

### Change frequency and tests

- Commits in the 30 days before 2026-10-11: `runner.py` 20, `script_utils.py` 20,
  `spectrum_pl.py` 19. These are the files carrying the cross-model branches.
- `fragnnet_model.py` had 11 commits in 30 days (`CutBondEncoder`, CMF fixes, `feiw_v2` merges).
  FraGNNet code is not frozen, and `feiw_v2` is live.
- 17 test files import code from both models: prepass, DAG storage, hparam migration, config
  templates, and MASSter tests using FraGNNet as a reference.
- Callers to update when model files move: MASSter-specific modules are imported by 71 files
  (14 src, 18 scripts, 39 tests); FraGNNet-specific ones by about 113 (15 src, 36 scripts,
  49 tests, 13 preprocessing scripts).

## Plugin interface

Each model and baseline registers a `model_type` with these hooks. Core models use defaults.

| Hook | Replaces |
| --- | --- |
| `model_cls`, `dataset_cls` | `setup_utils.get_model_cls` / `get_dataset_cls` |
| `migrate_config`, `default_config` | `pl_utils` MASSter migrations; runner 215-226 |
| `validate_config` | runner 515-560, 806 |
| `owned_config_keys` | `MASSTER_SAMPLER_KEYS` (generalized) |
| `build_batch_sampler` | runner 331-470 |
| `dataloader_options` | runner 345 (persistent workers), 1267 (`num_workers` workaround exemption) |
| `filter_dataset` | `spec_mol_dataset` 57 |
| `sampler_needs_epoch_reset` | `spectrum_pl` 2469 |
| `apply_inference_config` | `script_utils` 481-515 |
| `export_index_vars` | `script_utils` 850-940 |
| `annotation_tables` | `script_utils` 1754-2063 |
| `log_resolved_settings` | runner 1069 |

## Phases

Each phase is one or more small PRs. Every phase must leave the tests passing and the golden
outputs identical.

### Phase 0 — Safety net

Each later phase moves a specific behavior, so the safety net must exercise that behavior, not
just run a default batch. The three shared loss weights (`ce_pair_loss_weight`,
`spectral_leak_loss_weight`, `spectral_ce_auxiliary_weight`) default to `0.0` in
`template.yml`, so a default-config smoke test would never touch them.

1. **Registry coverage.** For every `model_type` the runner accepts today — `frag_gnn`,
   `fragnnet`, `masster`, `neims`, `chemlm`, `massformer`, `iceberg_gen`, `iceberg_inten`,
   `graff`, `precursor`, `gnn`, `mces_pretraining`, `clip`, `spec_mol_clip_pretraining` — plus a
   legacy `mass_shatter` config, assert the resolved model and dataset classes match an
   expected-mapping table kept in the test. The table starts as today's
   `get_model_cls`/`get_dataset_cls` and changes only deliberately, in the same PR as the code:
   - Phase 1: `massformer` → `SpecMolMFDataset` instead of `SpecMolDataset`. That PR must also
     show the new class is equivalent: identical per-item `mol_mf` preprocessing, an identical
     collated batch (keys, shapes, dtypes, values) through the runner's DataLoader, and identical
     MassFormer forward outputs and loss against the Phase 0 records.
   - Phase 6: module paths change with the directory moves; class names and behavior do not.
   Any other difference fails the test.
2. **Smoke test per model type.** Build from template plus a tiny config; run one forward pass
   and one training step on a fixture.
3. **Enabled-loss fixtures.** For FraGNNet and MASSter, configs with the three shared weights
   non-zero, on a fixture batch that contains CE-pair-eligible spectra. Assert
   `ce_pair_eligible_endpoints > 0`, so the CE-pair term is actually exercised. Record every named
   loss term, the total, the gradients and the parameters after one optimizer step.
4. **MASSter stages.** `MassterPL` rejects RT/CCS heads combined with proposal-only warmup
   ([masster_pl.py:901-903](../../src/fragnnet/pl_model/masster_pl.py#L901)), so three cases:
   - proposal-only training with property heads off: the same records, plus an assertion that
     the spectral terms contribute no gradient;
   - joint training with RT/CCS heads on: the same records;
   - a rejection test: RT/CCS heads with `masster_proposal_only_until_epoch` set must still raise.
5. **Samplers.** For each sampler that Phase 4 moves (FraGNNet dynamic, dual-group and curriculum
   samplers; MASSter's CE-context sampler), record the batch-index sequence across at least two
   epoch transitions (`set_epoch`) and after resuming from a mid-training checkpoint.
6. **Export.** For FraGNNet, MASSter and ICEBERG, record the schema (columns, dtypes, HDF5 dataset
   names) and values of the pkl, tabular and HDF5 inference exports on a fixture.
7. **Import boundary.** An AST check of the dependency rule. It starts in report-only mode and
   becomes enforcing once a phase removes the violations it covers.

Later phases compare these records exactly: values bit-for-bit, schemas field-for-field.

### Phase 1 — Baselines decoupled in place

1. Lazy registry in `registry.py` for **every** model type, FraGNNet and MASSter included, so
   `runner` → `setup_utils` no longer imports any model at load time. `VALID_MODEL_TYPES` is
   derived from it.
2. Replace the `isinstance(IcebergIntenPL)` checks and the MAGMA/GRAFF inference edits with hooks.
3. Move the MassFormer and GRAFF molecule handling out of `SpecMolDataset`. The
   `mol_params.mf`/`graff` keys stay.
   - Collation is dispatched statically, not by inheritance: `SpecMolDataset.get_collate_fn()`
     returns `SpecMolDataset.collate_fn`, which calls `SpecMolDataset._special_collate` by name
     ([spec_mol_dataset.py:242-298](../../src/fragnnet/dataset/spec_mol_dataset.py#L242)).
     Overriding `_special_collate` or the preprocessing in a subclass is therefore not enough.
   - MassFormer: a new `SpecMolMFDataset(SpecMolDataset)` defines its own `get_collate_fn`,
     `collate_fn` and `_special_collate` (the last calling `SpecMolDataset._special_collate`),
     following the existing GRAFF and ICEBERG pattern ([graff/dataset.py:264-306](../../src/fragnnet/graff/dataset.py#L264)).
     The registry maps `massformer` to it.
   - GRAFF: `SpecMolAnnDataset` already has that chain; move the `mol_graff` block into its
     `_special_collate`.
   - Test through the DataLoader that the runner's `init_dataloader` builds, not by calling the
     collate function directly.

   This is the only step of the phase that changes which code computes a value; the Phase 0
   records guard it.
4. Decouple DGL. `misc_utils.to_device` and `print_shapes` check `isinstance(v, DGLGraph)`
   through the shim ([misc_utils.py:42](../../src/fragnnet/utils/misc_utils.py#L42)), so moving the
   shim alone would either break core or make core import ICEBERG. Replace those checks with a
   core-owned registry of extra batch types (how to move one to a device, how to report its
   size). ICEBERG registers `DGLGraph` when its package is imported. Then move
   `dgl_compat_utils.py` into ICEBERG.

5. Package initializers. Reduce `model/__init__.py`, `pl_model/__init__.py` and
   `dataset/__init__.py` to core exports. During the transition, the model-specific names stay
   reachable through a module-level `__getattr__` (PEP 562) that imports the class only when the
   name is accessed. Importing a shared submodule then no longer loads any model. Update the
   callers to import from the defining module (later the model package), then delete the lazy
   names in Phase 6.

Gate: an isolation test runs in fresh subprocesses. It imports core alone (`fragnnet.runner`,
`fragnnet.pl_model.spectrum_pl`, `fragnnet.dataset.spec_mol_dataset`,
`fragnnet.utils.script_utils`) and checks `sys.modules`. It then selects each model through the
registry and checks that only that model's code has loaded. Successful import with dgl absent is
not enough on its own, because the shim already swallows import failures.

The test enforces only boundaries that have actually been resolved; the rest stay report-only.
Assertions are made per entry module, because `script_utils` keeps its FraGNNet and MASSter
imports (`FraGNNetPL`, `MassterPL`, `masster_templates`, `frag_utils.get_node_feats`) until its
export hooks land in Phase 4:

| After phase | Entry modules checked | Enforced: importing them does not load | Report-only |
| --- | --- | --- | --- |
| 1 | `runner`, `script_utils`, `spectrum_pl`, `spec_mol_dataset` | any baseline module, `dgl*` | FraGNNet and MASSter modules |
| 3 | `runner`, `spectrum_pl`, `spec_mol_dataset`, `base_dataset`, `feat_utils`, `data_sampler`, `pl_utils` | additionally `frag_utils`, `compute_frags`, `multi_cut_bfs`, `smarts_prepass`, `fragnnet_model`, `bond_fragment_heads` | `script_utils` |
| 4 | all of the above, plus `script_utils` | any FraGNNet or MASSter module | none |

The Phase 3 row was checked with a transitive import calculation at `13b1599d`: package
initializers treated as lazy (Phase 1), the registry lazy for every model (Phase 1), and the
Phase 3 edges removed. Every Phase 3 entry module came out clean; `script_utils` still reaches
`fragnnet_model`, `bond_fragment_heads`, `frag_utils` and `smarts_prepass`.

Phase 6 then moves files without changing what this test enforces.

### Phase 2 — Shared spectrum objectives into `SpectrumPL`

Composition contract:

- `SpectrumPL` provides a helper that **computes** the shared terms (CE-pair, spectral leak,
  spectral CE auxiliary, with the Hellinger weight validation) and returns their values and
  eligibility counts. It never adds anything to the total loss, and there is no base-class
  aggregation step after the model's loss function.
- Each model's loss function calls the helper and keeps deciding which terms enter the total, in
  which order, and in which training stage. MASSter keeps its current order: spectral and CE-pair
  terms added; total replaced in proposal-only training; RT/CCS added after.
- Logged names (`ce_pair_loss`, `ce_pair_eligible_endpoints`, `spectral_leak_loss`,
  `spectral_ce_auxiliary`) and the `loss_names` lists stay unchanged.

Gate: the Phase 0 enabled-loss records match exactly for FraGNNet, MASSter proposal-only and
MASSter joint training: per-term values, totals, gradients and post-step parameters.

### Phase 3 — `fragmentation/` and the core output schema

1. Move `_validate_tensor_condition` from `bond_fragment_heads` to core first. Until then,
   `candidate_support` cannot move without carrying a dependency on MASSter.
2. Create a core output-schema module holding `EXPLAINABILITY_OUTPUT_KEYS` and the annotation
   key names. Point `SpectrumPL` and the exporter at it, removing `spectrum_pl.py`'s import of
   `fragnnet_model`.
3. Move into `fragmentation/`: `smarts_prepass`, `CutBondEncoder`/`encode_cut_bond_pairs`,
   `candidate_support`, the reference `compute_cc_h_cap` and a single `update_bonds`. Parity tests
   move here and check each model's fast implementation against the reference.
4. Split `frag_utils` in place, as described in
   [Every core import of code the plan moves](#every-core-import-of-code-the-plan-moves). Only
   `compute_dags` depends on the Cython builders and `smarts_prepass`; the reader, format, mask
   packing, feature indexing and reference chemistry do not, so they can be separated cleanly:
   - element tables, formula representation and mask packing to core `formula_utils`;
     `cmf_duplicate_crf_h_shift` stays there;
   - the DAG file format (readers, writers, layout constants) to a core module;
   - DAG generation (`compute_dags` and its helpers) and DAG feature indexing
     (`NODE_FEAT_TO_IDX`, `get_node_feats`, …) to FraGNNet-owned modules in the current tree,
     together with `get_frag_graph` and `batch_mols_frags` taken out of core `feat_utils`.

   After this step core `feat_utils` and `base_dataset` no longer import `frag_utils`, and
   importing core no longer loads `compute_frags`, `multi_cut_bfs` or `smarts_prepass`.

Leave temporary re-exports at the old import paths so `feiw_v2` merges keep working. Remove them
in Phase 6.

### Phase 4 — Hooks for FraGNNet and MASSter

Implement the remaining hooks and delete the branches from `runner.py`, `script_utils.py`,
`spec_mol_dataset.py`, `data_sampler.py` (model samplers move to their owners) and `pl_utils.py`.
These are the most frequently changed files, so use small PRs merged quickly.

### Phase 5 — Packaging

- `pyproject.toml`: drop `dgl` from core; add an `iceberg` extra. Drop the unused `omegaconf`.
  **Keep `torch_scatter`.** No repo code imports it, but PyG 2.6.1 routes CUDA `max`/`min`
  scatter with gradients through it when it is installed (`torch_geometric/utils/_scatter.py`,
  lines 89-119), and 154 configs set `frag_pool_type: max` (`aggr.MaxAggregation`,
  [nn_blocks.py:2360](../../src/fragnnet/model/nn_blocks.py#L2360)). For tied maxima the gradients
  differ: `torch_scatter` routes all gradient to one element (`[1, 0]`), native PyTorch splits it
  (`[0.5, 0.5]`), as a reviewer's probe in FRAGNNET-GPU showed. Removing it would break the
  behavior-preservation requirement; evaluate that separately, with a tied-maximum gradient test.
  [env_setup_gpu_py312_cu121_251105.sh](../../env/env_setup_gpu_py312_cu121_251105.sh) does not
  install `torch_scatter`, so an environment built from it already takes the native path.
  Declare `pyyaml`, `networkx`, `pillow`, which core imports at module level but
  which today arrive only through other packages. Move `scikit-learn` to a scripts/dev extra. Fix
  `requires-python` (`<=3.13` excludes 3.13.x; use `<3.14`).
- [setup_job_cc.py:384](../../scripts/setup_job_cc.py#L384): install dgl only for ICEBERG model
  types, as it already does for `chemlm`. It currently installs dgl 2.4.0 for every job.
- Note: every install path runs `pip install --no-deps -e .`, so `pyproject.toml` dependencies are
  never resolved. Commit a lock (freeze) of FRAGNNET-GPU if `pyproject.toml` is to describe a real
  environment.

### Phase 6 — Directory moves

1. Baselines → `src/fragnnet/baselines/`. Move `algos.pyx` in [setup.py](../../setup.py) and
   rebuild the `.so` files. Update the `.pre-commit-config.yaml` exclude regex. Move the MAGMA
   preprocessing and ms2c scripts with ICEBERG and renumber the preprocessing README.
2. MASSter and FraGNNet → `models/masster`, `models/fragnnet`, including `.pyx`/`.cu` paths in
   `setup.py` and package-data. Every module that moves was already separated in Phases 3–4:
   FraGNNet's DAG generation, DAG feature indexing and fragment graph construction,
   `fragment_competition`, its samplers and curricula. This step only renames. The DAG file format
   stays in core.
3. Remove the Phase 3 re-exports and the lazy `__getattr__` names in the package initializers.

Rules:

- Pure-rename commits with no edits mixed in, so git rename detection works across merges.
- Only after the diverged branches (`feiw_v2`, `spectraverse`, others in use) are merged or retired.
- Only when no cluster jobs are queued. A generated job copies `src/` from the checkout when its
  script starts running, not when it is submitted
  ([setup_job_cc.py:419](../../scripts/setup_job_cc.py#L419)), so a job queued before the move
  would run whatever layout the checkout has when it starts.

### Phase 7 — Docs

Update [architecture.md](../architecture.md) with the dependency rule and plugin interface;
update INSTALL.md (extras, the missing `env_setup_gpu_py310_cu118.sh` reference). Add
`docs/reference/baselines.md`: running each baseline, data prerequisites (MAGMA for ICEBERG),
config keys.

## Unchanged

- `model_type` strings, config key names, config file paths, sweep files and W&B runs.
- `config/template.yml` stays a single file.
- Checkpoints: Lightning checkpoints keep state dict and hparams, not class paths. No `pickle.dump`
  or `torch.save` was found in baseline code or MAGMA preprocessing. Before Phase 6 merges, load
  at least one existing checkpoint per model type; there are no baseline checkpoints locally.

## Risks

- Phase 1 step 3 and Phase 2 change which code computes a value. Golden outputs must be identical,
  not merely close.
- Phase 2 can silently put gradient back into a stage that excludes it (MASSter proposal-only
  training). The composition contract and the per-stage gradient checks exist for this.
- Phases 2-4 edit the three most-changed files; long-lived PRs will conflict.
- Phase 6 breaks cluster jobs that were generated for the old layout but start after the move,
  and conflicts with any branch that edited a moved file.
- All coupling counts come from static import analysis and a config-key regex. Classes selected
  by name at runtime (the `nn_blocks` GNN types, featurizer registries) are invisible to it.

## Open decisions

1. **Layout.** Recommended: `fragnnet.models.{fragnnet,masster}` and `fragnnet.baselines.*`.
   Alternative: separate top-level packages or pip distributions (`fragnnet_iceberg`, …); only
   worth it if the baselines will be published or installed separately.
2. **NEIMS.** It is a baseline but lives in core `model/` and `pl_model/`. Recommended: move it
   with the baselines. Keep `chemlm`, `gnn`, `precursor`, MCES and CLIP in core; they share
   encoders, and the MSGYM workstream uses them.
3. **Baseline config directories.** Recommended: do not move them into `config/baselines/`, since
   sweeps and W&B runs point at the current paths.
4. **Live branches.** Which diverged branches are still in use; this sets when Phase 6 can run.

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
