# Inference and DAG-Preprocessing Pitfalls

> **Status:** REFERENCE · **Updated:** 2026-10-12
> Fixed bugs, silent-failure traps and measured facts about the inference scripts, the exports and DAG storage, moved here from local agent memory (verified against code 2026-10-06). The usage guide is [run_inference.md](../guides/run_inference.md).

Items marked **(unverified)** came from 2026-09 sessions and were not re-checked on the current
branch.

## Fixed bugs worth remembering

| Date | Defect | Fix and how to avoid it |
| --- | --- | --- |
| 2026-09-16 | `MolGraph.get_mol_graph` raised `IndexError` on a molecule with **zero bonds** (one heavy atom plus implicit Hs). `01_inf_prepare_mol_df.py` filters multi-component, charged and radical molecules, so these are the only zero-bond inputs, but they are not rare: SYNCED has 19 (`C N S F Cl O P B I Br [AsH3] [SiH4]`, all with `mol_id` below 48,209, so only array task 0 of `infer_synced_arr_hdf5.sh` crashed). | `bond_tuples` is reshaped to `(-1, 2)` ([feat_utils.py](../../src/fragnnet/utils/feat_utils.py)); tests in `tests/test_feat_utils.py`. Each bond-free molecule predicts exactly one peak, the protonated precursor, which is correct. |
| 2026-09-16 | `random_walk_pe` let scipy infer the adjacency shape from the largest index. It failed on edgeless graphs, was non-square for `bigraph=False` (broken for every directed graph that is not a pure cycle) and would silently drop trailing isolated atoms. | One argument: `shape=(g.num_nodes, g.num_nodes)`. **Not on this branch** (`csr_matrix` at `feat_utils.py` ~line 1163 is still shapeless; fix lives on `feiw_v2` and `project/spectraverse_benchmark`). MASSter uses `pyg_bigraph: True`, so connected molecules work, but port the fix before setting `pyg_pe_embed_k > 0`; see [architecture review todo, item 3](../plans/masster/masster_architecture_review_todo.md). |
| 2026-09-16 | `combine_formulae` and `parse_annotations` called `Composition(s)` positionally. pyteomics reads a positional string as a **peptide sequence**, so any digit-free one-letter formula misparses (`"C"` is cysteine; Hill-notation `CHN`, `CHNO`, `CHNOS` precursors broke every derived product and loss). 6 of 1,399 fragment formulas (0.4%) in a 6-molecule d3 run were affected. | Every string-taking `Composition` in `data_utils.py` now passes `formula=`. Rule: grep for `Composition(` and check each string call is keyword. Dict/`Composition` arguments are fine positionally. The module-level `PREC_TYPE_TO_*` tables were already safe (every value has a digit or is empty). Tests: `TestCombineFormulae`, `TestParseAnnotationsPrecursorParsing`. The export reads the raw neutral `frag_formula_str`, so it was never affected. |
| 2026-09-16 | **Train/serve skew in inference DAGs.** `03_prepare_dag_feats.py --isotopes` defaulted True while training DAGs are `isoFalse` (`frag_gen_common.sh`, `ISO_FLAG=False`). `frag_utils.py` sorts `(peak_mz, prob)` by probability **before** truncating to `num_isotopes`, so `formula_peak_mzs[:, :1]` is the **most abundant** isotopologue, not M+0: +2.0067 Da (or +4.0134) off for >=3 Cl or >=2 Br. 35% of formulas in the first 9 (chlorinated) SYNCED molecules were wrong. | Default flipped to False in `preproc_scripts/03_prepare_dag_feats.py` (with a help string); AST test `test_isotopes_cli_default_is_false`. **Any SYNCED or cspirit `dags.h5` built before 2026-09-16 must be regenerated**; training DAGs were never affected. `scripts/run_compute_frags.py` still defaults `--isotopes True` at the CLI (line 78) while its internal `kwargs.get` default is False. Lesson: a residual that is an exact small multiple of an isotope spacing means column 0 holds the wrong isotopologue, not "expected isotopologue behaviour". |
| 2026-09-16 | `spec_params.drop_empty_spectra: True` (the template default) deleted every spectrum of a prediction-only run (`peaks == []` by construction); the script printed `Total predictions: 0` and exited 0. A second defect left CUDA tensors in the saved pickle. | `apply_inference_config(drop_empty_spectra=...)`, passed False by `run_inference.py` and applied only when the key already exists in the config (writing it into a legacy checkpoint config trips the missing-cleanup-params `ValueError`). Eval scripts keep cleanup. `run_inference.py` now raises on an empty dataset, and a `--frag_dp` with no DAGs raises naming the directory. Index and probability arrays are moved to CPU at the single row-assembly point. |
| 2026-09-16 | `cos_hun_inst_tol` / `jss_hun_inst_tol` exported **-1.0 for every spectrum** when `inst_type_loss_tol` was unset (its template entries are commented out) and the run exited 0. | `_setup_tolerance` raises when either score is in `auxiliary_scores` without the table (`TestInstTolScoresRequireConfig`). `run_save_inference_eval.py` defaults to `cos_hun`; `cos_hun_inst_tol` is opt-in for QTOF-containing evals. Defaulting the table in `template.yml` was tried and **reverted by the user** (it changes the training loss for configs that never set it and breaks comparability with every historical `cos_hun`); do not re-propose. On FT-only configs the two scores coincide. Any recorded `cos_hun` value came from a config that set `inst_type_loss_tol`. |
| 2026-09-15 | Annotation export: `min_peak_int` never pruned node arrays, so `pred_node_ccs` was the whole DAG at any threshold; `min_peak_int <= 0` returned early and **skipped the global-to-per-molecule rebase** (CUDA device-side assert from the second spectrum on); index arrays were requested only via a probability token, so `--return_formula_str` / `--return_cc_hash` gave NaN. | Fixed in `script_utils.py`; tests `tests/test_inference_annotation_without_probs.py`. The surviving behaviour is documented in [run_inference.md, fragment annotation](../guides/run_inference.md#fragment-annotation). |
| 2026-09-16 | `batch_input["spec_prec_type_str"]` was a hard subscript, but the key exists only when `spec_params.prec_type_str` is True (False in the template and nearly every config). | `.get()` plus a `UserWarning`; test `test_missing_prec_type_str_degrades_to_nan`. |
| 2026-09-18 | MASSter atom decoder died with `CUDA error: invalid configuration argument` in `scaled_dot_product_attention` when `train_batch_size * masster_unique_fragment_target` exceeded 65,535 (the fragment count is the attention batch dimension and the CUDA grid caps there; probed: 65,535 works, 65,536 fails). The error is a launch rejection, not a sticky assert, so a retry in-process works and the crash looks intermittent. | Fixed then by chunking at 32,768 sequences (bit-identical on GPU). **That code no longer exists on this branch** (`remove atom decoder` commit; no `atom_decoder` call or `_MAX_ATTENTION_SEQUENCES` in `src/`). Relevant only if an attention call over a per-fragment batch dimension is reintroduced. |

## Reviewer false positives (do not re-file)

Both concern the `pred_formula_adducts` export in `process_batch_to_tabular_row`.

- **"`frag_formula_cumsizes.cpu()` needs a None-check."** It cannot be None at the annotation
  block: `batch_filter_by_peak_intensity` runs unconditionally first and its rebase dereferences
  the field (popping it dies earlier in `inference/batch_rows.py`). The later None-check serves models
  with no formula axis (NEIMS). A guard was added and then removed as unreachable; a comment
  states the invariant.
- **"`pred_formula_strs` and `pred_formula_adducts` can disagree in the ragged family."** The two
  gates differ by design in `COLUMN_SPECS` (`return_formula_str` vs `return_annotation`, where
  `return_annotation = return_formula_str or return_cc_hash`). With `--return_cc_hash` alone,
  `pred_formula_strs` is not a selected column at all. Making the gates identical would break
  that shape.

Before filing either again, check whether `batch_filter_by_peak_intensity` already dereferences
the field on that path and read the column's `flag`.

## Annotation export, verified behaviour not in the guide

- `--return_formula_str` / `--return_cc_hash` without `pred_formula_node_probs` still exports the
  formula, cc and index columns bit-identically, but `drop_none_cols` removes the P(n|f) column,
  so there is no argmax and no peak-to-fragment link (only the candidate node set).
- At `--min_intensity 0` the NULL formula row (`""`) survives (one per spectrum) and 153 of
  12,036 peaks had intensity 0, P(f) 0 and no joint row. Consumers of a threshold-0 export must
  skip empty formulas and check that `jf == f_idx` is non-empty.
- `pred_ints` differs between with-probs and without-probs arms by about 8e-7 relative
  (float32 reduction order); m/z and every index column are bit-identical. Resume and dynamic
  batching show the same float32 batch-composition non-invariance (about 1e-6 relative); `pred_oos_prob`
  of the undertrained debug head is a constant.
- `formula_delta_to_signed_str` takes the canonical form (`"H-1"`), not the exported signed form
  (`"-H"`); code parsing `pred_formula_adducts` must read the signed spelling itself.
- `preproc_scripts/inference/02_inf_prepare_spec_df.py` allows only `[M+H]+` and `[M-H]-`;
  `[M+Na]+` inputs were made by rewriting `prec_type`/`prec_mz` on an existing `spec_df`.
- A debug checkpoint for smoke runs (`config/debug/debug_tiny_inference_smoke.yml`, about 90 s) needs
  `sparse_cosine_similarity: True` and `prec_types` / `inst_types` listing **every** value in the
  split, because `prec_type_to_idx` / `inst_type_to_idx` are direct dict lookups (unlike
  `frag_mode`, which has an unknown token). The `_cmf` variant (`include_cmf: True`, about 25 min
  on one 2070 SUPER) is needed for CMF arms.

## HDF5 export size

Measured on a 96k-spectrum eval (`jnokddr9`, 212 MB) and the 2.6 GB `9t63zils` val file.

- int64 index columns (`pred_joint_*_idxs`, `pred_node_node_idxs`, `frag_formula_peak_idxs`)
  compress 15-26x; float32 probability columns compress only 1.1-1.3x and are already float32.
  Even float64 metric columns compress below what a float32 downcast would give. **Decision:
  compress, never downcast.**
- `run_inference_hdf5` writes every dataset with gzip level 4, plus shuffle for fixed-width numeric
  dtypes only (`dtype.kind in "biufc"`; a no-op on variable-length strings), and
  `chunks=(65536,) + trailing`. An index column: 80.1 MB raw, 40.0 MB as int32, 9.3 MB with h5py's
  auto chunk, **5.1 MB with the 65,536 chunk**.
- The big chunk and the compression are one decision: HDF5 allocates a whole chunk once touched,
  so a 378-row column costs 528 KB raw and 7 KB compressed. With `compression=None` the writer
  must fall back to `chunks=True`. Use gzip, not lzf (h5py-only filter).
- int32 on top adds only about 5%. If revisited: check max/min before casting (numpy wraps
  silently), decide at the first flush (h5py cannot widen a dataset), gate on `rebase_indices=True`
  (rebased indices are bounded by one DAG, max 36,734 measured), and never touch
  `start_points`/`end_points` or `spec_id`/`mol_id`. At 1M spectra the joint/node offsets reach
  about 700M, only 3.1x under the int32 ceiling, and NIST20 `spec_id` already reaches 1,010,470.
- `--drop_none_cols` was read only by `run_inference.py`, so HDF5 evals carried a full NaN column per
  unrequested output (9 of 20 datasets). The writer now decides on the first flush and enforces
  afterwards, because it appends to one file.

## `dags.h5` compression

`dump_dag_hdf5_packed` ([dag/storage.py](../../src/fragnnet/models/fragnnet/dag/storage.py)) already
compresses each per-molecule payload with zlib before writing it into `dag_bytes`; do **not**
propose an HDF5 `compression="gzip"` filter as if the file were uncompressed. Measured on 185
drug-like d3/h4 DAGs (150.2 MB raw pickle; stdlib codecs only, since zstd is not installed and
the cluster installs `pip --no-index`):

| Codec | Size (MB) | Decompress | Note |
| --- | ---: | ---: | --- |
| zlib-4 | 13.5 | 0.92 ms/DAG | previous default |
| **zlib-6** | **12.2** | **0.83 ms/DAG** | current `PACKED_DAG_ZLIB_LEVEL = 6` |
| zlib-9 | 12.0 | n/a | 20x write CPU for 1.5%; not worth it |
| bz2-9 | 9.85 | 4.88 ms/DAG | what the pickle/tar path uses; 5.3x slower reads |
| lzma-6 | 6.38 | 1.58 ms/DAG | -53%, but about 110 ms/molecule write, roughly doubling preprocessing time |

`zlib.decompress` is level-agnostic, so old level-4 files still read and no format bump was
needed. End to end: 13.65 to 12.41 MB with unchanged wall time, all DAGs byte-identical on
round trip. This is distinct from the prediction-export size above.

## Batching and sizing

- Size-capped inference batching is done by `IterableSpecMolFragDataset`
  (`--dynamic_batch_max`, see the guide). The rival `GreedyInferenceBatchSampler` was deleted
  2026-09-15 (referenced everywhere, used nowhere). Its flaw: `__init__` did an up-front pass
  that reloaded every DAG just to plan batches, in the main process, which opens the lazily
  created `frag_h5` handle before workers fork. Do not re-add a size cache or sizing pass.
- Wiring: `DataLoader(stream_ds, batch_size=None, collate_fn=<module-level function>)` (a lambda
  cannot be pickled to workers; without `batch_size=None` batches are batched twice). No
  `__len__`, so `len(dataloader)` raises; `run_inference_tabular_iter` handles that but
  `run_inference_pkl` calls `len(dataloader)` unguarded and cannot take a streaming loader.
- Verified 2026-09-16 on 50 SYNCED molecules / 700 spectra: every cap (500 / 5,000 / 50,000),
  both DAG backends, chunked resume and `--start_mol_idx` slices match the fixed `--batch_size 16`
  baseline bit-identically except `pred_ints` (about 1e-6 relative). GPU peak 0.09 GB at cap
  5,000 against 0.53 GB at batch 16, at about 200 spectra/s. An oversized sample is still
  predicted as a single-sample batch. `--dynamic_batch_limited_by frag_edge` on a
  `pyg_edges: False` config raises the intended guard.
- The hard-coded `num_workers=0` comment in `run_inference.py` ("deadlocks before the first batch")
  **did not reproduce** with 2 workers on the packed backend (full script, CUDA on). It is
  unverified conservatism, not a measured constraint.
- **SYNCED DAG size** (400 random molecules, d3/h4): `frag_node` per molecule mean 1,549, p50 332,
  p90 1,160, p99 24,609, max 108,721. The `spec_df` holds exactly 14 consecutive rows per molecule
  (4,049,486 rows / 289,249 molecules), so a fixed batch of 64 spectra is about 4.6 molecules
  repeated 14 times; simulated batch `frag_node` has p50 29,150 but p99 1,165,156 and max
  1,552,120. `--dynamic_batch_max 100000` reproduces the mean (96,409) and cuts the peak 14x to
  108,721; below about 110,000 the peak stops falling because 2 of 400 molecules have a single DAG
  larger than that. Size a SYNCED job from frag nodes, not spectrum count, and re-measure with a
  random 400-molecule sample (a subset uniform in heavy-atom count misses the tail). The
  production config trains at `dynamic_batch_sampler_max: 400000`.
- **DAG longest path** (37 sampled biostructure DAGs, `max_depth=3`): median 16, p75 23, p90 39,
  p95 49, max 87 hops. `max_depth` bounds BFS generation depth, not path length; only about 50%
  of edges respect `nodes_min_depth`. Any algorithm iterating over DAG depth must not assume
  3-4 rounds. FluxMPNN Stage 3 computes a longest-path topological level per node by integer
  relaxation and sweeps edges by parent level so each fires once
  (see [fluxmpnn_plan](../plans/fragnnet/fluxmpnn_plan.md)).

## Config precedence at inference (user decision 2026-09-15, restated 2026-09-29)

- `get_config_from_ckpt` merges `config/template.yml` only when `--custom_fp` is given, and
  writes the result onto the checkpoint hparams with a shallow top-level assignment. Neither is a
  defect. The template supplies defaults, `custom_fp` overrides, and the **checkpoint hparams
  are the custom config**. `--custom_fp` (the config the checkpoint was trained with) is therefore
  mandatory; `infer_synced_arr_common.sh` enforces it via `CUSTOM_FP`.
- A new required key in `config/template.yml` is supplied by that path. Do not add `.get(key,
  default)`, backfills or a `custom > checkpoint > template` precedence (rejected twice).
  `MassterPL.load_from_checkpoint` and `get_pl_hparams` plus the constructor are also unsupported
  loaders; test only `get_config_from_ckpt(..., custom_fp, template_fp)` plus `init_model_from_ckpt`.
- Accepted side effect: when `custom_fp` does not restate a nested key, the template's value wins
  over the checkpoint's (e.g. `frag_params.pyg_node_feats` on `9t63zils`).

## Known broken or deferred (as of 2026-09-15, unverified since)

- `scripts/eval/run_save_inference_ablations.py` (and `run_save_inference.py`) use the
  pre-refactor positional argument order and raise `TypeError`. User: "very old script, fix later";
  leave it.
- Resume in `run_inference.py` filters `spec_df` by `spec_id`, but the saved column is `id_key`
  (`group_id` when `spec_params["merge"]`), so resume is wrong in merged-spectra mode.
- The zero-size guard in `IterableSpecMolFragDataset` aborts on a legitimately edgeless sample
  (harmless for the `frag_node` default, real under an explicit `frag_edge`). The 2026-09 audit
  never covered worker sharding or the `batch_size=None` / `pin_memory` contract.
- Resolved and not to be re-raised: `--save_format parquet` (removed), `--save_interval` (removed),
  and the `evaluate=True` intensity-squaring bug (gone; the two paths agree on `pred_ints` to
  2.2e-5 relative over 378 spectra).

## Local SYNCED packed-HDF5 pipeline test (no Slurm)

Run the real `scripts/manual_slurm/fir` scripts on a workstation: set `VENV_DP=<dir>` containing an
**empty `bin/activate`** (the common bodies then skip `pip install --no-index` and run the repo
source); `module load` lines fail harmlessly. `CHUNK_SIZE` is assigned unconditionally in the
`*_hdf5.sh` wrappers, so `sed` a scratch copy (it still sources the repo common body through the
`${FRAGNNET_DP}/...` fallback). Needed env: `FRAGNNET_DP`, `DEST_FRAG_DP`, `SLURM_TMPDIR`,
`VENV_DP`, `SLURM_ARRAY_TASK_ID`, `OUT_DP`, `MODEL_CKPT`, `CUSTOM_FP`, and `env -u SLURM_JOB_ID`.
Leave `SRC_PROC_DP` unset to exercise the `data/inference/synced` to `synced_smiles` fallback.

On the real `synced_smiles` (289,249 molecules / 4,049,486 spectra) loading both dataframes takes
about 8 s and **4.3 GB peak RSS per array task**. Passed 2026-09-16: per-shard slicing with global
`mol_id`s, out-of-range task exiting 0, `merge_dag_h5_shards.py` byte-identical DAGs, merged-job ==
3-shard inference (except float32 `pred_ints` noise, max 6.1e-4 on the 0-1000 scale), the
`NUM_FRAG_SHARDS` array-width guard (rejects too narrow and sparse), loud failure on a missing
shard, and resume. `--array=0-14` at `CHUNK_SIZE=20000` covers 289,249 molecules exactly.
`MODEL_TAG` being `last` for a hand-passed `last.ckpt` was raised and **dismissed by the user**:
`get_best_ckpt_from_wandb` always names checkpoints `{run.name}_{run_id}.ckpt`.

Bond-free smoke recipe (about 1 min, no training; reuses `ckpts/debug_tiny_inference_smoke/last.ckpt`):
`01_inf_prepare_mol_df.py --smiles_list "C" "N" "S" "F" "Cl" "O" "[SiH4]" "CCO" "c1ccccc1O"`, then
`02_inf_prepare_spec_df.py --prec_types "[M+H]+" --ft_ces 25.0NCE --inst_types FT`, then
`03_prepare_dag_feats.py` (needs uncompressed `mol_df.pkl` / `spec_df.pkl` beside the `.pkl.gz`),
then `run_inference.py`. Write test outputs under the repository's `output/`; the workstation
`/tmp` is a small volume.
