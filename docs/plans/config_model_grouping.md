# Config grouping by model

> **Status:** LANDED · **Updated:** 2026-10-12
> Model directories implemented; YAML preservation and 138 focused tests passed.

## Objective and acceptance

Group custom experiment configs by their owning model. Keep the shared `config/template.yml`,
model types and hyperparameter schema intact. Preserve experiment settings, filenames,
dataset/experiment subgroups, standalone analysis settings and existing known-invalid configs.
Update live callers and docs; confirm the same YAML inventory and equivalent settings.

## Implementation

- Moved 507 config files. The root contains the shared template, existing README and six model
  directories: `fragnnet`, `masster`, `neims`, `massformer`, `graff`, `precursor`.
- `frag_gnn` and `fragnnet` configs belong to `fragnnet/`. Debug and sweep training configs
  follow their model. MASSter standalone diagnostic YAML lives in `masster/analysis/`.
- Retained dataset and experiment variants; removed redundant exact model directory names
  immediately under dataset folders. FraGNNet D3/D4 and NEIMS original variants remain.
- Updated tracked scripts, examples, test allowlists and live documentation. Added a layout
  guard to `tests/test_config_templates.py`. W&B search definitions stay under `sweep/`.
- Frozen findings, earlier plans, run records, checkpoints and generated ignored job scripts
  retain their original paths. This record supersedes the earlier decision to retain dataset-first
  paths in the [reorganization plan](repo_reorg_plan.md#unchanged).

## Compatibility

The loader continues to use the single root template. This is a file-path migration with no
hyperparameter schema change. Commands must use the new custom-config path; there are no old-path
symlinks or runtime aliases. Regenerate existing cluster jobs against this checkout. External
W&B commands and saved custom paths must be updated when reused. Historical paths can be
translated with the table below (use the longest matching source prefix; retain the suffix).

The [config audit](../findings/config_audit_2026_10_12.md) remains the record of pre-existing
loader and YAML problems. Grouping does not repair those problems or change their semantics.

## Validation

- Inventory preserved: 806 `.yml` files (including the shared template) and 8 diagnostic
  `.yaml` files. Verified all tracked YAML text against its original content, allowing only
  mapped path substitutions; 810 parsed mappings are unchanged, 3 malformed files remain
  malformed, and one diagnostic `extraction_config` points to the same companion after moving.
- `pytest tests/test_config_templates.py tests/test_config.py tests/test_runner_config.py
  tests/test_setup_job_cc.py -q`: **138 passed**, one NVML warning.
- New model ownership guard passes for every custom training config.
- Functional Python changes pass Ruff. Across all Python files touched for path references,
  Ruff reports four baseline errors: three F821 names in
  `preproc_scripts/no_dag_exp/03_test_other_smarts.py` and I001 imports in
  `scripts/eval/dag_oracle_scores.py`. Confirmed the same errors in the original files.
- Changed Python parses, changed shell scripts pass `bash -n`, new local Markdown targets
  exist, and `git diff --check` passes.
- No migrated literal source paths remain in tracked live files. Historical migration tables
  and frozen records intentionally retain source paths. Dynamic inference/MS2C/entropy path
  builders were updated separately.
- Full suite, training, GPU execution, scheduler submissions and external W&B edits were not run.

## Next action

Review the working-tree moves and use the new paths for future commands. Repair loader/YAML
correctness findings separately, starting with W&B open-mapping merge and CLI validation.

## Path migration table

Directory entries preserve all relative descendants. Individual file entries cover mixed-model
folders such as `debug/` and the shared sweep training-config folder.

| Previous path | Current path |
| --- | --- |
| `config/ablations` | `config/fragnnet/ablations` |
| `config/ablations_merged` | `config/fragnnet/ablations_merged` |
| `config/analysis` | `config/masster/analysis` |
| `config/debug/debug_candidate_ion_smoke.yml` | `config/fragnnet/debug/debug_candidate_ion_smoke.yml` |
| `config/debug/debug_ion_channel_smoke.yml` | `config/fragnnet/debug/debug_ion_channel_smoke.yml` |
| `config/debug/debug_neims_org.yml` | `config/neims/debug/debug_neims_org.yml` |
| `config/debug/debug_tiny_adduct_inst_sampler.yml` | `config/fragnnet/debug/debug_tiny_adduct_inst_sampler.yml` |
| `config/debug/debug_tiny_ce_context_sampler.yml` | `config/fragnnet/debug/debug_tiny_ce_context_sampler.yml` |
| `config/debug/debug_tiny_dual_ce.yml` | `config/fragnnet/debug/debug_tiny_dual_ce.yml` |
| `config/debug/debug_tiny_dual_group_sampler.yml` | `config/fragnnet/debug/debug_tiny_dual_group_sampler.yml` |
| `config/debug/debug_tiny_frag_gine.yml` | `config/fragnnet/debug/debug_tiny_frag_gine.yml` |
| `config/debug/debug_tiny_fraggnn_graph_feature_parity.yml` | `config/fragnnet/debug/debug_tiny_fraggnn_graph_feature_parity.yml` |
| `config/debug/debug_tiny_fragnnet_formula_support.yml` | `config/fragnnet/debug/debug_tiny_fragnnet_formula_support.yml` |
| `config/debug/debug_tiny_gine_gsa.yml` | `config/fragnnet/debug/debug_tiny_gine_gsa.yml` |
| `config/debug/debug_tiny_inference_smoke.yml` | `config/fragnnet/debug/debug_tiny_inference_smoke.yml` |
| `config/debug/debug_tiny_inference_smoke_cmf.yml` | `config/fragnnet/debug/debug_tiny_inference_smoke_cmf.yml` |
| `config/debug/debug_tiny_masster.yml` | `config/masster/debug/debug_tiny_masster.yml` |
| `config/debug/debug_tiny_masster_channel_control.yml` | `config/masster/debug/debug_tiny_masster_channel_control.yml` |
| `config/debug/debug_tiny_masster_cut_features.yml` | `config/masster/debug/debug_tiny_masster_cut_features.yml` |
| `config/debug/debug_tiny_masster_glacier_feats.yml` | `config/masster/debug/debug_tiny_masster_glacier_feats.yml` |
| `config/debug/debug_tiny_masster_h_support.yml` | `config/masster/debug/debug_tiny_masster_h_support.yml` |
| `config/debug/debug_tiny_masster_independent_gate.yml` | `config/masster/debug/debug_tiny_masster_independent_gate.yml` |
| `config/debug/debug_tiny_masster_intensity_escape.yml` | `config/masster/debug/debug_tiny_masster_intensity_escape.yml` |
| `config/debug/debug_tiny_masster_intensity_escape_control.yml` | `config/masster/debug/debug_tiny_masster_intensity_escape_control.yml` |
| `config/debug/debug_tiny_masster_mode_h_support.yml` | `config/masster/debug/debug_tiny_masster_mode_h_support.yml` |
| `config/debug/debug_tiny_merged.yml` | `config/fragnnet/debug/debug_tiny_merged.yml` |
| `config/debug/debug_tiny_new_frag_mol_feats.yml` | `config/fragnnet/debug/debug_tiny_new_frag_mol_feats.yml` |
| `config/debug/debug_tiny_new_mol_feats.yml` | `config/fragnnet/debug/debug_tiny_new_mol_feats.yml` |
| `config/debug/debug_tiny_pool_concat.yml` | `config/fragnnet/debug/debug_tiny_pool_concat.yml` |
| `config/debug/debug_tiny_prepass.yml` | `config/fragnnet/debug/debug_tiny_prepass.yml` |
| `config/debug/debug_tiny_split_filters.yml` | `config/fragnnet/debug/debug_tiny_split_filters.yml` |
| `config/debug/debug_tiny_um_prec.yml` | `config/fragnnet/debug/debug_tiny_um_prec.yml` |
| `config/debug/debug_tiny_um_prec_bf16m.yml` | `config/fragnnet/debug/debug_tiny_um_prec_bf16m.yml` |
| `config/debug/debug_tiny_um_prec_sampler.yml` | `config/fragnnet/debug/debug_tiny_um_prec_sampler.yml` |
| `config/entropy` | `config/fragnnet/entropy` |
| `config/exp_fraggnn_ma_mi` | `config/fragnnet/exp_fraggnn_ma_mi` |
| `config/loss_funcs` | `config/fragnnet/loss_funcs` |
| `config/massspecgym` | `config/fragnnet/massspecgym` |
| `config/massspecgym_merged/fragnnet_d3` | `config/fragnnet/massspecgym_merged/fragnnet_d3` |
| `config/massspecgym_merged/fragnnet_d4` | `config/fragnnet/massspecgym_merged/fragnnet_d4` |
| `config/massspecgym_merged/massformer` | `config/massformer/massspecgym_merged` |
| `config/massspecgym_merged/neims` | `config/neims/massspecgym_merged` |
| `config/mix_percision` | `config/fragnnet/mix_percision` |
| `config/new_dags` | `config/fragnnet/new_dags` |
| `config/nist20_merged_inchikey/fragnnet_d3` | `config/fragnnet/nist20_merged_inchikey/fragnnet_d3` |
| `config/nist20_merged_inchikey/fragnnet_d4` | `config/fragnnet/nist20_merged_inchikey/fragnnet_d4` |
| `config/nist20_merged_inchikey/graff` | `config/graff/nist20_merged_inchikey` |
| `config/nist20_merged_inchikey/massformer` | `config/massformer/nist20_merged_inchikey` |
| `config/nist20_merged_inchikey/masster` | `config/masster/nist20_merged_inchikey` |
| `config/nist20_merged_inchikey/neims` | `config/neims/nist20_merged_inchikey` |
| `config/nist20_merged_inchikey/precursor` | `config/precursor/nist20_merged_inchikey` |
| `config/nist20_merged_scaffold/fragnnet_d3` | `config/fragnnet/nist20_merged_scaffold/fragnnet_d3` |
| `config/nist20_merged_scaffold/fragnnet_d4` | `config/fragnnet/nist20_merged_scaffold/fragnnet_d4` |
| `config/nist20_merged_scaffold/graff` | `config/graff/nist20_merged_scaffold` |
| `config/nist20_merged_scaffold/massformer` | `config/massformer/nist20_merged_scaffold` |
| `config/nist20_merged_scaffold/neims` | `config/neims/nist20_merged_scaffold` |
| `config/nist20_merged_scaffold/precursor` | `config/precursor/nist20_merged_scaffold` |
| `config/nps_pcdl_nist23` | `config/fragnnet/nps_pcdl_nist23` |
| `config/nps_pcdl_nist23_ft` | `config/fragnnet/nps_pcdl_nist23_ft` |
| `config/pairwise_loss` | `config/fragnnet/pairwise_loss` |
| `config/pcdl_nist23` | `config/fragnnet/pcdl_nist23` |
| `config/pcdl_nist23v2` | `config/fragnnet/pcdl_nist23v2` |
| `config/profiler` | `config/fragnnet/profiler` |
| `config/sampler` | `config/fragnnet/sampler` |
| `config/spectraverse/fragnnet` | `config/fragnnet/spectraverse` |
| `config/spectraverse/fragnnet_jss_sqrt` | `config/fragnnet/spectraverse/fragnnet_jss_sqrt` |
| `config/spectraverse/fragnnet_merged` | `config/fragnnet/spectraverse/fragnnet_merged` |
| `config/spectraverse/fragnnet_sqrt` | `config/fragnnet/spectraverse/fragnnet_sqrt` |
| `config/spectraverse/masster` | `config/masster/spectraverse` |
| `config/spectraverse/masster_abs005` | `config/masster/spectraverse/masster_abs005` |
| `config/spectraverse/masster_c0rerun` | `config/masster/spectraverse/masster_c0rerun` |
| `config/spectraverse/masster_dedup` | `config/masster/spectraverse/masster_dedup` |
| `config/spectraverse/masster_merged_nomsdial` | `config/masster/spectraverse/masster_merged_nomsdial` |
| `config/spectraverse/masster_short18` | `config/masster/spectraverse/masster_short18` |
| `config/spectraverse/masster_subceil05` | `config/masster/spectraverse/masster_subceil05` |
| `config/spectraverse/neims` | `config/neims/spectraverse` |
| `config/spectraverse/neims_org` | `config/neims/spectraverse/neims_org` |
| `config/sweeps` | `config/fragnnet/sweeps` |
| `config/sweeps_merged/d3_sweep_template.yml` | `config/fragnnet/sweeps_merged/d3_sweep_template.yml` |
| `config/sweeps_merged/d4_sweep_template.yml` | `config/fragnnet/sweeps_merged/d4_sweep_template.yml` |
| `config/sweeps_merged/graff_sweep_template.yml` | `config/graff/sweeps_merged/graff_sweep_template.yml` |
| `config/sweeps_merged/massformer_sweep_template.yml` | `config/massformer/sweeps_merged/massformer_sweep_template.yml` |
| `config/sweeps_merged/masster` | `config/masster/sweeps_merged` |
| `config/sweeps_merged/neims_sweep_template.yml` | `config/neims/sweeps_merged/neims_sweep_template.yml` |
