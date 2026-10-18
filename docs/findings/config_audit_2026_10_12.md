# Configuration audit

> **Status:** FINDING · **Updated:** 2026-10-18
> Runtime findings 1–3 repaired and committed in `e65339ed`; finding 6 resolved for MASSter by retiring
> 97 configs on 2026-10-13; findings 4, 5 and 7 remain open.

Scope: experiment YAML under `config/`, the shared template, CLI overrides, W&B merging,
and runner validation. Checks use the configured FRAGNNET-GPU interpreter. No training,
submission, checkpoint loading, or dataset mutation was performed. Files under `sweep/`
and infrastructure/editor configuration are outside this audit.

## Confirmed findings

1. **P1 — W&B silently discards new entries in open mappings.**
   `merge_wandb_config` in `src/fragnnet/config.py:296` iterates only existing sub-keys,
   although schema validation explicitly permits new instrument/split entries. Starting with
   `inst_type_loss_tol = {ORB: {rel: 1e-5, min_mz: 200}}` and merging a W&B config containing
   `inst_type_loss_tol = {QTOF: {rel: 2e-5, min_mz: 750}}` leaves only ORB. QTOF then uses
   the global loss tolerance instead of the requested instrument tolerance. The same loop
   handles `ce_context_ce_key_by_inst`; a new instrument entry there is also lost.
   Merge permitted open-mapping entries explicitly and cover a new instrument key in tests.

2. **P2 — Whole-section CLI overrides bypass recursive schema checking.**
   `apply_override` checks only the addressed key (`src/fragnnet/config.py:175`), then
   replaces its entire value. `load_config('config', None, ['spec_params={typo: true}'])`
   succeeds with `spec_params == {'typo': True}`. It accepts an undeclared nested key and
   removes the required spectrum settings, moving failure to a later consumer. Validate
   mapping-valued overrides recursively; explicitly define whether replacing a section is
   allowed or whether it must merge with defaults. Scalar replacements of sections also
   need a setup-time error.

3. **P2 — CLI model selection happens after model validation and migrations.**
   `load_config` validates and migrates the model before applying overrides
   (`src/fragnnet/config.py:227-239`).
   `load_config('config', None, ['model_type=not_a_model'])` returns successfully with the
   unknown model type. Retired types can similarly bypass the loader's retirement guard;
   subsequent registry access will fail instead. Resolve final model selection before
   invoking its template migration and validate the resulting type.

4. **P2 — Three tracked experiment files cannot be parsed.**
   `config/ablations_merged/fraggnn_d3_edges/s1.yml` and `s2.yml` omit `wandb_tags:`
   before their tag lists (line 3). In
   `config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_sqrt.yml:197`,
   `dynamic_batch_sampler_max:400000` lacks the separating space. The repository-wide
   test explicitly allows these failures through `KNOWN_MALFORMED_CONFIGS`.

5. **P2 — Duplicate YAML keys are silently overwritten.**
   `config/debug/debug_tiny_um_prec.yml:51` declares `mol_pool_type: mean`, but line 110
   declares `attention`; the latter wins. `debug_tiny_um_prec_sampler.yml` repeats the
   same conflicting declarations. Several FraGNNet ablation configs repeat `mol_params`.
   Their repeated values are identical in the examples inspected, but the loader still
   discards the first mapping rather than recursively merging it. The template test checks
   duplicate top-level template keys, not duplicate experiment keys. Reject duplicate keys
   at every YAML mapping level; resolve conflicting values using the experiment's intent.

6. **P2 — Ten tracked configs are intentionally accepted as broken by the test.**
   `tests/test_config_templates.py` pins one debug config with undeclared
   `frag_params.pyg_pe_embed_k` and nine legacy MASSter configs with removed
   `masster_explore_masks`. This preserves the reorganization baseline but does not establish
   that every shipped config is usable. Separate historical configs from supported runnable
   configs, or migrate them with explicit reproducibility decisions.
   **MASSter part resolved 2026-10-13:** the nine legacy configs were retired (see
   [MASSter config retirement](#masster-config-retirement-2026-10-13)); only the FraGNNet debug
   config remains allowlisted.

7. **P2 — Five charge-migration ablations load but fail runner validation.**
   Under `config/ablations/charge_migration_fragmentation/`, the `molc_cl_only`,
   `molc_cmf`, `molc_cmf_cl_only`, `molc_cmf_na_only`, and `molc_na_only` configs set
   `dynamic_batch_sampler: SpecMolFragDynamicBatchSampler` (e.g. line 106 of the first).
   Validation attempts a boolean conversion and raises `invalid literal for int()`.
   `test_every_tracked_config_still_resolves` only calls `load_config`, so it misses these
   setup failures. Migrate the sampler declaration to the current boolean and sampler-type
   fields after checking the intended grouping.

## Validation and limits

- `python -m pytest tests/test_config_templates.py -q`: **23 passed**, one NVML warning.
- `python -m pytest tests/test_config.py tests/test_runner_config.py -q`: **86 passed**.
- All three runtime probes above reproduced successfully without creating a run.
- Inventory: **806 YAML files**, including the template; **787** custom configs pass both
  loading and runner validation with `wandb_mode='offline'` (validation only; no W&B session).
  **3** are malformed, **10** fail schema loading, and **5** more fail runner validation.
  **27** files have duplicate keys; this overlaps the runtime inventory, which was checked
  separately using the production loader's last-value-wins semantics.
- The duplicate scan rejects repeated keys at any mapping level. A full path inventory follows.
- Existing tests passing does not invalidate the findings: the file failures are allowlisted,
  and the runtime probes exercise uncovered cases.
- No full suite, model construction, data/split integrity, GPU, or end-to-end training check.
- Initial Git status was clean. The project-state claim of an in-progress isotope utility move
  was stale relative to the inspected working tree.

## Next action

Findings 1–3 were repaired with regression coverage in the
codebase repair record (removed 2026-10-18; `git show 731fe493:docs/findings/codebase_audit_2026_10_13.md`).
Resolve malformed and duplicate YAML next. Preserve historical experiment semantics when
migrating removed settings; these runtime repairs do not choose new values for those experiments.

## MASSter config retirement (2026-10-13)

User decision: retire closed and unreferenced MASSter configs. The last commit holding all of
them is `e180f2c2`; recover any file with `git show e180f2c2:<path>`. Frozen findings and plans
keep their links to these paths.

Every config under `config/masster/` was loaded with `load_config`, its data paths were checked
(all present), and the repository (docs, scripts, tests, src, `sweep/`) was searched for its
path, filename and stem.

**Retired, 97 configs plus 5 companions:**

| Group | Configs | Reason |
| --- | --- | --- |
| `config/masster/legacy/` (`explore`, `bond_size_curriculum`) | 15 | 9 failed validation on removed exploration keys; all were historical records |
| `config/masster/mh_ft_clean_2026_09_22/` (`f0`, `f1`) | 2 | `f0`/`f1` retired; their findings are SUPERSEDED |
| `config/masster/sweeps_merged/` | 4 | Pre-`i1` merged sweeps; the 4 matching `sweep/masster/merged_*` definitions retired with them |
| `config/masster/merged/` (`baseline` 7, `channel_selection` 10, `proposal_warmup` 6, `subformula_filter` 7, `unique_fragment` 17, `warmup_duration` 5, `intensity_attention` 1) | 53 | Closed screens; no reference to the file, its name or its arm stem anywhere |
| `config/masster/ma_mi_unmerged/` (`baseline` 5, `cmf` 5, `unique_fragment` 13) | 23 | Same criterion |

Also removed: `scripts/manual_slurm/fir/submit_masster_mh_ft_clean.sh`. The usage example in
`scripts/eval/bond_type_ce_allocation.py` now points at the `i1` config and checkpoint. Every
cited, test-loaded or active-workstream config was kept. The remaining `merged/` and
`ma_mi_unmerged/` files are those that tests, guides or findings name. Validation:
`tests/test_config_templates.py`, `test_merged_ce_context.py`,
`test_masster_connected_fragment_beam.py`, `test_config.py` and `test_runner_config.py`:
**239 passed**.

## Files with duplicate keys

- `config/ablations/charge_migration_fragmentation/fraggnn_d3_ma_mi_nist20_molc_cl_only.yml`
- `config/ablations/charge_migration_fragmentation/fraggnn_d3_ma_mi_nist20_molc_cmf.yml`
- `config/ablations/charge_migration_fragmentation/fraggnn_d3_ma_mi_nist20_molc_cmf_cl_only.yml`
- `config/ablations/charge_migration_fragmentation/fraggnn_d3_ma_mi_nist20_molc_cmf_na_only.yml`
- `config/ablations/charge_migration_fragmentation/fraggnn_d3_ma_mi_nist20_molc_na_only.yml`
- `config/ablations/compare_interstage/fraggnn_d3_ma_mi_nist20_amolc_gated.yml`
- `config/ablations/compare_interstage/fraggnn_d3_ma_mi_nist20_amolc_mlp.yml`
- `config/ablations/compare_mol_pooling/fraggnn_d3_ma_mi_nist20_amolc_p_attation.yml`
- `config/ablations/compare_mol_pooling/fraggnn_d3_ma_mi_nist20_amolc_p_attation_rest.yml`
- `config/ablations/compare_mol_pooling/fraggnn_d3_ma_mi_nist20_amolc_p_mean_rest.yml`
- `config/ablations/compare_mol_pooling/fraggnn_d3_ma_mi_nist20_amolc_p_mms.yml`
- `config/ablations/compare_mol_pooling/fraggnn_d3_ma_mi_nist20_amolc_p_mms_rest.yml`
- `config/ablations/compare_mol_pooling/fraggnn_d3_ma_mi_nist20_amolc_p_sum.yml`
- `config/ablations/compare_mol_pooling/fraggnn_d3_ma_mi_nist20_amolc_p_sum_rest.yml`
- `config/debug/debug_tiny_um_prec.yml`
- `config/debug/debug_tiny_um_prec_sampler.yml`
- `config/loss_funcs/fraggnn_d3_nist20_hellinger.yml`
- `config/loss_funcs/fraggnn_d3_nist20_hellinger_ce_aux0p1.yml`
- `config/loss_funcs/fraggnn_d3_nist20_hellinger_leak1.yml`
- `config/loss_funcs/fraggnn_d3_nist20_jss.yml`
- `config/loss_funcs/fraggnn_d3_nist20_jss_hun.yml`
- `config/loss_funcs/fraggnn_d3_nist20_jss_hun_sqrt.yml`
- `config/loss_funcs/fraggnn_d3_nist20_jss_sqrt.yml`
- `config/mix_percision/fraggnn_d3_ma_mi_nist20_16_mixed.yml`
- `config/mix_percision/fraggnn_d3_ma_mi_nist20_bf16_mixed.yml`
- `config/pairwise_loss/fraggnn_d3_ma_mi_nist20_pairwise_cos.yml`
- `config/pairwise_loss/fraggnn_d3_ma_mi_nist20_pairwise_jss.yml`
