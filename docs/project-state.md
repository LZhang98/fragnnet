# Current project state

> **Status:** REFERENCE · **Updated:** 2026-10-18
> Dashboard: one block per workstream, replaced (not appended) on every handoff. Details live in
> the [workstream hubs](workstreams/); the pre-2026-10-06 dated handoff is frozen in
> [history/handoff_log.md](history/handoff_log.md).

Keep this file under about 200 lines. To hand off, update the hub's "Where it stands" and
"Open threads", then replace that workstream's block below
([documentation rules](rules/documentation.md#where-each-kind-of-fact-goes)).

## Workstreams

| Workstream | Hub | State |
|---|---|---|
| MASSter generalization on NIST20 (`i1` line) | [masster_generalization](workstreams/masster_generalization.md) | ACTIVE: computed-energetics gate is next |
| MASSter core (architecture, training, speed) | [masster_core](workstreams/masster_core.md) | Prepared arms; no scheduled work |
| MassSpecGym training and retrieval | [msgym](workstreams/msgym.md) | Many arms PREPARED, none submitted |
| Spectraverse v1.0.2 | [spectraverse](workstreams/spectraverse.md) | Five arms PREPARED, none submitted |
| NIST23 EI (nominal) | [nist23_ei](workstreams/nist23_ei.md) | e2/e3/e4 PREPARED |
| CASMI 2026, Enveda-180, ZenoTOF EAD, shared data quality | [casmi_enveda](workstreams/casmi_enveda.md) | ACTIVE: CASMI train filter decision; EAD libraries at steps 1-2 |
| FraGNNet and the DAG | [fragnnet](workstreams/fragnnet.md) | Dormant; pair BCE retired |

## MASSter generalization on NIST20

- **Best recipe:** `i4` = `i1` + `k1` dropout + EMA 0.999 + pre-norm, W&B `rd9rsdqx`, tail
  `val_cos_sim_0.01` 0.7729 (epochs 54-63): +0.0122 over `i1` (0.7607). Seed spread 0.0042.
- **Open problem is the floor:** 330 spectra / 119 molecules below 0.05 at instrument tolerance
  (train sample: 4); no tested arm moves it.
- **Prepared, not submitted:** [rewire rules in training](plans/masster/masster_rewire_rules_training_plan.md) — after-selection stage built and tested (uncommitted); G0 reach 27.7% vs null 7.3%, G0b 1.14x forward at cap 0.5; arms `a0`/`a0d1`/`a0null`, Fir stage `rewire` ([finding](findings/masster/masster_rearrangement_mining_2026_10_14.md)).
- **Prepared, not submitted:** `so1` (templates-on sulfonamide O transfer), `fc1`/`fc1p`
  (local screen not positive).
- **Next:** computed-energetics information gate. First fix a bounded state/calculation policy
  and a train/calibration/confirmation protocol.
- **Prepared:** [rare-element route 1](plans/masster/rare_chemistry_augmentation_plan.md#rare-element-handling-revised-2026-10-14) A0/E0/E1/E2/E3 on `i1` (`a_periodic`, hidden-Br test; E3 drops `a_onehot`); the A1/A3 sampling arms were removed 2026-10-14 without running.

## MASSter core

- **Speed audit:** [current opportunities](findings/masster/masster_code_path_audit_2026_10_16.md#speed-audit-of-the-current-path-2026-10-16): main speed switches already active in `i1`/`i4`; rewire host round-trips/repeated reductions and base packing/loss padding are next profiling targets. CPU regressions pass; current GPU timing unavailable.

- **Current path audit:** [2026-10-16 findings](findings/masster/masster_code_path_audit_2026_10_16.md): retired-config tests repaired; 177 MASSter PL tests pass. Four package lint violations remain. Rewire row-order and 14 new batch/padding cases pass. **Next:** package lint; GPU/real-checkpoint validation remains open.

- **Shared codebase audit:** all six findings and the three config runtime defects repaired and committed in `e65339ed`; unfiltered CPU suite passes (record removed 2026-10-18; `git show 731fe493:docs/findings/codebase_audit_2026_10_13.md`). **Next:** GPU/real-checkpoint evaluation unverified. Older branch/uncommitted descriptions below are stale for this checkout.

- **Configs grouped by model:** record (removed 2026-10-13; `git show e65339ed:docs/plans/config_model_grouping.md`); shared template and experiment settings retained. [Audit findings](findings/config_audit_2026_10_12.md) 1–3 repaired; 6 resolved for MASSter by retiring 97 configs (2026-10-13, recover from `e180f2c2`); 4, 5 and 7 open. **Next:** malformed/duplicate YAML.

- **Shared notebook audit:** [findings](findings/notebook_audit_2026_10_12.md); reporting errors and execution blockers confirmed. NIST20 duplicate consolidated; large saved table outputs cleared with figures preserved. Reporting repairs remain open.

- **Unit test audit:** all four findings fixed in `e65339ed` (record removed 2026-10-18; `git show 731fe493:docs/findings/unit_test_audit_2026_10_12.md`). **Next:** GPU validation.

- **Reorganization done** (Phases 0–9 on `refactor/fragnnet-masster-reorg`, not merged; [progress](plans/repo_reorg_plan.md#progress)): MASSter is in `src/fragnnet/models/masster/`; ICEBERG runs through external ms-pred; package metadata and install files no longer require DGL (the live `FRAGNNET-GPU` environment is unchanged). **Next:** review and merge the branch.
- **Code cleanup committed:** [audit items 1–4](findings/masster/masster_simplification_audit_2026_10_12.md#outcome-2026-10-12), the export fix (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_export_merge_untransform_2026_10_12.md`) for sqrt/log10 models with merged rows, and module renames so file names say what they hold: candidate generation in `models/masster/candidates/` (`builder`, `cut_mask_kernels.pyx`, `reference_cut_masks`, `cuda_mask_ranking`, `cuda_topology_hybrid`; `4e36bcae`), and `bond_fragment_heads.py` split into `candidate_transformer`, `proposal_heads`, `component_pooling` and `pathway_probabilities`. After checkout run `python setup.py build_ext --inplace` (the Cython extension is now `cut_mask_kernels`). **Next:** audit item 5.
- **Shared loss consolidation committed** (`1a63257b`, file merge and renames `72ad897e`): [plan](plans/loss_consolidation_plan.md#completion-and-validation-2026-10-12), [file merge and renames](plans/loss_consolidation_plan.md#file-merge-and-renames-2026-10-12). **Next:** GPU validation when available.

- **Screen baseline:** merged MA/MI 128-epoch screen, control `c4a_base_s1` 0.7270 (spread
  0.0007, gate ±0.005). Real levers: `a1_prenorm` +0.0056, cosine LR decay `d3` 0.7357 (n=1).
- **Prepared:** full-length `wd10` and its `trainaux` control (`fir:h100-80gb`); `a4`,
  `a5_ce_pair_bias`, `b4`. Ion-H Phase 2 is blocked on a recoverable checkpoint.
- **RT/CCS:** [Separate RTHead/CCSHead GPU debug run completed](plans/masster/masster_rt_ccs_plan.md); full Enveda arm prepared, heads default off.
- **Next:** an `a1_prenorm` seed replicate. Otherwise, core levers go through the `i1` stack.

## MassSpecGym

- **Best:** `ll04i00m` test `cos_sim_0.01` 0.5713 vs GLACIER released 0.5455. Formula-track
  retrieval: `pow075` (`q3oxbpgx`@55) hit@1 `loglik_cal` 0.542, `mspred_ent` 0.516 (+0.031 over
  `ll04i00m`); `wd10` within noise ([arm screen finding](findings/masster/masster_msgym_arm_screen_formula_retrieval_2026_10_07.md)).
- **Problem:** generalization, train 0.884 vs val 0.584; arms one seed each (control spread 0.0025).
- **Ran 2026-10-03:** k1/EMA/Fourier control ×3, `s2`, 13 arms, including the calibrated-likelihood arms. **Prepared, not submitted:** `pow075_s2`, `b1024`, `b256` rerun,
  5 U256/E1024 configs
  ([arms plan](plans/masster/masster_msgym_training_arms_plan.md)).
- **Sweep ran:** [tiwhun9q live read](plans/masster/msgym_val_loss_sweep_48_2026_10_03.md#live-results-read-2026-10-11); 15 finished / 33 interrupted. Leader re-run `f44tzxoc` ties `pow075` on formula-test retrieval ([read](plans/masster/msgym_val_loss_sweep_48_2026_10_03.md#leader-re-run-formula-test-retrieval-2026-10-13)).
- **Read 2026-10-18:** [short-schedule / budget screen](plans/masster/msgym_short_budget_screen_2026_10_14.md#formula-test-retrieval-2026-10-17): 32 epochs is enough; E1024 within noise on retrieval (pair it on Fir before adopting); U128 −0.016, floor 0.01 −0.041 `loglik_cal` hit@1.
- **Next:** seed replicates of `pow075` and the leader. Re-select `loglik_cal` on val.
- **Removed 2026-10-14:** [rare chemistry sampling](plans/masster/rare_chemistry_augmentation_plan.md) MSGYM arms, never run; route 1 is NIST20-only.

## Spectraverse

- **Stuck at** `cos_sim_0.01` 0.458 (cv0 `ggifzj9m`, cv1 `u46v0k0g`). The causes are intensity
  allocation (62% of the emitted-bin ceiling) and 14% unexplainable spectra (MS-DIAL is 46% of
  those).
- **Prepared, not submitted (cv0/cv1):** `masster_c0rerun`, `short18`, `subceil05`,
  `merged_nomsdial`, `dedup`, plus the ten-fold `abs005`.
- **Next:** copy `data/split/spectraverse_benchmark_{subceil05,nomsdial,dedup}` to Fir, launch,
  read against `masster_c0rerun`.

## NIST23 EI

- **Held-out HR-EI:** [e5 `eugy8s9n` finding](findings/masster/hr_ei_e5_holdout_eval_2026_10_18.md): e1's HR-EI advantage is memorization. e5 matches NIST23 better than two of the three HR libraries do, so most of the HR gap is between libraries. Next: reference-cleaning decision, e5 seed.
- **Best:** e1 `jeljdikj`@37, test 1 Da cosine 0.869984, Stein 0.886845. Isotopes are +0.0079
  over e0 (one seed). The gap is intensity allocation (support oracle 0.996).
- **Next:** record provenance, check full-attention batch-512 memory, run e2/e3/e4, then a
  matched RASSP comparison.

## CASMI 2026, Enveda-180, shared data

- **Preprocessing audit:** seven correctness findings (removed 2026-10-18; `git show 731fe493:docs/findings/preproc_scripts_audit_2026_10_12.md`) fixed and committed (`bbcdf00c`). Rebuild any merger/inference-merged outputs into new directories before use.
- **ZenoTOF CID + EAD (uncommitted):** [MultiMS2 and MS-DIAL 5 standards](findings/zenotof_ead_libraries_2026_10_15.md) preprocessed to `data/proc/{multims2,msdial5_ead}`; new `EAD` frag_mode, `frag_mode` added to the `group_id` key (existing groups unchanged). CIeaD deferred (no structures). Next: review, splits, EAD-aware config.

- **CASMI 2026:** extracted, not preprocessed. The visible test is a copy of train, so validate
  molecule-disjoint. Dedup drops 17.6% of train; NIST HR would add 34,295 molecules.
- **Enveda-180:** baseline (standalone, ACE only, no EMA) prepared, not submitted. The combined
  PCDL+NIST23+Enveda-180 split is a manifest only.
- **RT/CCS:** [MASSter RTHead and CCSHead](plans/masster/masster_rt_ccs_plan.md) with independent label sets trained for 10 debug epochs on GPU; selected and final checkpoints retained. Full Enveda run remains prepared.
- **Next:** decide the CASMI train filter, then preprocess; confirm CCS calibration before multitask evaluation.

## FraGNNet

- **Shared loss consolidation committed** (`1a63257b`): [plan](plans/loss_consolidation_plan.md#completion-and-validation-2026-10-12). Follow-up in the working tree: [`loss/` cut from 9 to 6 modules, misleading function names fixed](plans/loss_consolidation_plan.md#file-merge-and-renames-2026-10-12). **Next:** review and commit; GPU validation when available.
- **Code audit complete:** [simplification findings](findings/fragnnet/fragnnet_simplification_audit_2026_10_12.md). cc-mask padding leak (rest pooling, cc-corruption probe) fixed with a regression test (`e0197d8f`); probe magnitudes now unverified. **Next:** remove dead loss branches, then consolidate duplicated operations.
- **Shared reorganization done** (branch, not merged; [progress](plans/repo_reorg_plan.md#progress)): FraGNNet is in `src/fragnnet/models/fragnnet/`. MCES pretraining (and fine-tuning from it), ChemLM and CLIP were retired 2026-10-12.

- **Dormant since September.** Best `pqx7ywc9` 0.7555 `cos_hun_inst_tol` (NIST20 test). The DAG
  oracle is 0.937, so the gap is intensity ranking, not coverage.
- **Stopped:** precursor BCE and CE-switch BCE retired at user request; Stage A configs and generator removed. See the [loss plan](plans/loss_consolidation_plan.md#phase-1-retire-precursor-and-ce-switch-bce).

## Branch `rare-chemistry-sampling` (2026-10-14)

- Rare-element route 1 (`a_periodic`, `filter_exclude_elements`, A0/E0/E1/E2 configs, Fir launcher) and the element-swap census; the A1/A3 sampler was removed. Not submitted.
  State and verification are in the [plan handoff](plans/masster/rare_chemistry_augmentation_plan.md#implementation-handoff-and-current-state).

## Standing cross-workstream rules

- Merged and per-CE evaluation setups are not comparable; neither are runs across the
  2026-09-03 target change, the `FT` → `ORB`/`ORB_F` relabel, or different binning (0.01 vs 0.1 Da).
- Read a hub's "Settled" list before re-proposing a mechanism: most closed nulls are recorded there.
- Cluster procedure: [submit_cluster_runs guide](guides/submit_cluster_runs.md). Artifact locations:
  [artifact rules](rules/artifacts.md).
