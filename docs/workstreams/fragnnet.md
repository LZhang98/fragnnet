# FraGNNet and the DAG

> **Status:** HUB · **Updated:** 2026-10-12
> FraGNNet (NodeMLP over the bond-cut fragmentation DAG): DAG generation, recall and pruning, oracle ceilings, intensity ranking, FluxMPNN, contender-pair training, GLACIER gap, and the cross-model conditioning and reranking plans.

## Where it stands
- **Shared loss consolidation proposed:** [plan](../plans/loss_consolidation_plan.md), backed by the [audit](../findings/loss_simplification_audit_2026_10_12.md). First fixes are committed in `6b359293`; next establish the remaining numerical contracts, then share binning, padding and dispatch.
- **Simplification audit complete:** [findings](../findings/fragnnet/fragnnet_simplification_audit_2026_10_12.md). Its cc-mask padding leak (rest pooling and the cc-corruption probe pulled atoms from other molecules in the batch) is fixed in the working tree, not committed, with a regression test. Next: remove dead loss branches, then deduplicate DAG insertion and head conditioning.
- **Reorganization implemented, not merged:** [plan](../plans/repo_reorg_plan.md) Phases 0–9 are committed on `refactor/fragnnet-masster-reorg`. FraGNNet code is in `src/fragnnet/models/fragnnet/`, MASSter in `src/fragnnet/models/masster/`, GRAFF and MassFormer in `src/fragnnet/baselines/` ([architecture](../architecture.md#package-layout-and-dependency-rule)); ICEBERG runs externally via ms-pred and the main environment is [DGL-free](../plans/repo_reorg_plan.md#decided-external-iceberg-dgl-free-main-environment).
- **No active FraGNNet work.** project-state shows all work since September 2026 on MASSter (see [MASSter core](masster_core.md)); Stage A is STOPPED; the other plans are deferred, proposed, landed or closed.
- **Best FraGNNet numbers cited in docs:** `pqx7ywc9` scores 0.7555 `cos_hun_inst_tol` on the NIST20 test split (`nist20_ma_mi_inchikey`, 96,128 spectra) and 0.7612 on FT `[M+H]+` at 0.1 Da bins, 0.077 below GLACIER's 0.838 ([gap finding](../findings/fragnnet/glacier_gap_ce_decomposition_2026_09_21.md)). The DAG oracle ceiling is 0.937 cos_hun (0.959 at instrument tolerance), so the gap is intensity ranking and allocation, not coverage ([ceiling](../findings/fragnnet/dag_oracle_ceiling_nist20.md)).
- **Stopped:** precursor and CE-switch BCE retired at user request. Stage A configs and generator removed; see the [loss plan](../plans/loss_consolidation_plan.md#phase-1-retire-precursor-and-ce-switch-bce).
- **User decision:** no SMARTS for FraGNNet (VRAM); SMARTS work is MASSter only ([pathway plan](../plans/masster/masster_smarts_pathway_plan.md), CLOSED NULL for MASSter). Production FraGNNet DAGs are `..._smpFalse` ([usage audit](../findings/smarts_usage_audit_2026_09_23.md)).
- **Next:** review BCE retirement, then address remaining numerical loss findings.

## Open threads
| Thread | Status | Next concrete step |
| --- | --- | --- |
| [Shared loss consolidation](../plans/loss_consolidation_plan.md) | PROPOSED | Add Phase 0 numerical regressions, then consolidate preparation in bounded changes |
| [Code simplification audit](../findings/fragnnet/fragnnet_simplification_audit_2026_10_12.md) | FINDING | Padding leak fixed (uncommitted); remove dead loss branches, then deduplicate operations |
| [FT contender-pair training](../plans/fragnnet/ft_contender_pair_minimal_training_plan.md) | STOPPED | Objective retired; do not submit |
| [Global fragment intensity ranking](../plans/fragnnet/global_fragment_intensity_ranking_experiment_plan.md) | PROPOSED | Phase 2 gates on an unfiltered matched NodeMLP export; after Stage A |
| [Complementary-cut supervision](../plans/fragnnet/complementary_cut_supervision_experiment_plan.md) | PROPOSED | Matched GINE-1 gates; lower priority than global ranking |
| [Ion x CE matrix](../plans/fragnnet/ion_collision_energy_matrix_experiment_plan.md) | PROPOSED | Matrix audit; V2 gated on V1 |
| [Per-fragment context gate](../plans/fragnnet/fragment_context_gate_experiment_plan.md) | PROPOSED | Zero-code routing ablation |
| [ClassyFire atom pretraining](../plans/fragnnet/classyfire_atom_pretraining_plan.md) | PROPOSED | Not started |
| [Energy-based reranking](../plans/energy_based_spectrum_reranking_plan.md) | PROPOSED | Pick checkpoint, unfiltered export, pass Gate A over matched random pools; MASSter bond-cut EBM first |
| [Sibling-ranking target on P(n)](../plans/fragnnet/sibling_ranking_target_plan.md) | DEFERRED | Implement the ranking loss (coverage gate already passed) |
| [Bond-set representation](../plans/fragnnet/fragment_bond_set_representation.md) | DEFERRED | Run `cc_atom_dropout_p{00,01,02}` (never run: no checkpoint, `output/` file or doc result as of 2026-10-06; configs `p00`-`p03` exist) |
| [Conditioning refactor](../plans/conditioning_refactor_plan.md) | DEFERRED | Decide which of 3 mechanisms survives; see MASSter metadata audit |
| [DAG full-recall tractability](../plans/fragnnet/dag_full_recall_tractability_plan.md) | DEFERRED | Steps 5-7 (trained scorer) paused; static priority lost to random |

## Settled — read before re-proposing
**DAG recall and pruning**
- [Oracle ceiling](../findings/fragnnet/dag_oracle_ceiling_nist20.md) — d3 DAG ceiling 0.937 cos_hun; model sits ~0.20 below; 82% of spectra score >=0.5 and carry 55% of error mass.
- [Bond mask STOPPED](../plans/fragnnet/bond_breakability_mask_plan.md) — 30.28% edge but 1.50% node reduction; nodeMLP node count is the cost driver.
- [Fan-out cap / radius env SUPERSEDED](../plans/fragnnet/dag_fanout_cap_plan.md) — static priorities (hand-mined 95.00%, learned 93.56%) lost to random 96.21% recall at K=8.
- [Full-recall plan](../plans/fragnnet/dag_full_recall_tractability_plan.md) — 100% recall walked back; global cutoff worse; max-of-N-routes bias is the driver.
- [Unexplainable spectra](../findings/fragnnet/unexplainable_spectra_analysis.md) and [oracle by CE](../findings/fragnnet/oracle_recall_wrecall_by_ce.md) — tools and tables for DAG coverage by adduct, CE and class.

**Intensity ranking and CE**
- [Dominant-peak decomposition](../findings/fragnnet/dominant_peak_outcome_decomposition.md) — misses are 7:1 ranking vs coverage (36.7% vs 5.2% at K=1); pass `--frag_dp`.
- [Ranking findings](../findings/fragnnet/fragment_intensity_ranking_findings.md) — base-peak accuracy 57.4%; model follows CE winner switches in only 38.5% of cases.
- [CE-extent null](../findings/fragnnet/ce_extent_headroom_finding.md) — CE -> deeper-fragmentation prior has no headroom; kills CFM stop head and `depth_ce_bias`.
- [GLACIER gap](../findings/fragnnet/glacier_gap_ce_decomposition_2026_09_21.md) — 0.077 gap all at high CE; allocation not coverage; uniform low-CE rate would give 0.835.
- [cc-corruption probe](../findings/fragnnet/cc_corruption_probe_finding.md) — fragment identity IS used (cos 0.82 -> 0.61); gap is comparative scoring. Magnitudes unverified: the corruption leaked atoms across molecules (fixed 2026-10-12, not rerun).

**Models and losses**
- [FluxMPNN STOPPED](../plans/fragnnet/fluxmpnn_plan.md#stage-3-trained-verdict-and-ceilings-from-local-memory-verified-2026-10-06) — conserved DAG flow null-to-harmful (`replace` 0.6244 vs control 0.7118; `residual` exactly neutral); learned beta explains 72% of itself, 18% of real branching; edge-beta identifiability needs an arborescence (87.7% multi-parent after min-depth pruning).
- [False-peak loss CLOSED NULL](../plans/fragnnet/false_peak_loss_plan.md) — `log1p_huber` harmful above lambda=0.1.
- [Output-head and conditioning nulls](../plans/fragnnet/false_peak_loss_plan.md#successor-output-head-lines-all-null-from-local-memory-verified-2026-10-06) — sparsemax/entmax15 + `intensity_head` (0.685-0.705 vs 0.711), two-head presence is root-cause dead, formula-bag 0.7090-0.7119, per-fragment CE features -0.0007, candidate multiplicity null; competition head never trained (four NaN crashes); `frag_pool_combine: concat` built, untrained.
- [Gap decomposition](../findings/fragnnet/dag_oracle_ceiling_nist20.md#gap-decomposition-and-overfit-ladder-from-local-memory-verified-2026-10-06) — 83.6% of the 0.249 gap is intensity, temperature caps +0.035; regularization ladder 0.533 -> 0.740 (`a6s5it2m` -> `s3iimtqi`).
- [MCES pretraining retired](../plans/repo_reorg_plan.md#progress) — pretraining the molecule encoder on MCES distances (Siamese GNN), then fine-tuning FraGNNet from it (pretrain run `edd2j4yy`), did not help (user verdict, 2026-10-12). Model, configs and preprocessing removed; they are at `13b1599d`. Fine-tuning from a FraGNNet checkpoint is unaffected.
- [Bond-set representation](../plans/fragnnet/fragment_bond_set_representation.md) — cost argument refuted (+109%); goal is regularization; `het_only` has best dominant-sibling mismatch 0.405 vs cc 0.473.

**Audits**
- [FraGNNet/MASSter audit](../findings/fragnnet_masster_audit_2026_09_12.md) and [fixes](../plans/fragnnet/fragnnet_audit_fixes.md) — six defects fixed and landed; checkpoints before the fix may differ.

## All documents
| Doc | Status (updated) | One-line content |
| --- | --- | --- |
| [Shared loss consolidation](../plans/loss_consolidation_plan.md) | PROPOSED (2026-10-12) | Numerical contracts, shared preparation, dispatch and validation gates |
| [Shared loss audit](../findings/loss_simplification_audit_2026_10_12.md) | FINDING (2026-10-12) | Two reproduced correctness issues and bounded simplifications |
| [fragnnet_simplification_audit_2026_10_12](../findings/fragnnet/fragnnet_simplification_audit_2026_10_12.md) | FINDING (2026-10-12) | cc-mask padding leak (fixed) and prioritized behavior-preserving cleanup |
| [cc_corruption_probe_finding](../findings/fragnnet/cc_corruption_probe_finding.md) | FINDING (2026-10-12) | Fragment identity representation is used |
| [ce_extent_headroom_finding](../findings/fragnnet/ce_extent_headroom_finding.md) | CLOSED NULL (2026-07-16) | CE -> extent prior has no headroom |
| [dag_oracle_ceiling_nist20](../findings/fragnnet/dag_oracle_ceiling_nist20.md) | FINDING (2026-08-28) | Model-free DAG ceiling 0.937 |
| [dominant_peak_outcome_decomposition](../findings/fragnnet/dominant_peak_outcome_decomposition.md) | FINDING (2026-08-13) | Base-peak failure ~7:1 ranking |
| [fragment_intensity_ranking_findings](../findings/fragnnet/fragment_intensity_ranking_findings.md) | FINDING (2026-08-11) | Provisional NodeMLP ranking evidence |
| [glacier_gap_ce_decomposition_2026_09_21](../findings/fragnnet/glacier_gap_ce_decomposition_2026_09_21.md) | FINDING (2026-09-21) | GLACIER gap is high-CE allocation |
| [fragnnet_masster_audit_2026_09_12](../findings/fragnnet_masster_audit_2026_09_12.md) | FINDING (2026-10-06) | Reproduced correctness defects |
| [missing_peak_findings](../findings/fragnnet/missing_peak_findings.md) | CURATED ANALYSIS (2026-07-18) | Per-subclass DAG misses `[M+H]+`; append-only |
| [missing_peak_findings_m-h](../findings/fragnnet/missing_peak_findings_m-h.md) | CURATED ANALYSIS (2026-06-30) | Per-subclass DAG misses `[M-H]-`; append-only |
| [oracle_recall_wrecall_by_ce](../findings/fragnnet/oracle_recall_wrecall_by_ce.md) | FINDING (2026-07-18) | DAG recall/wrecall by CE and adduct |
| [unexplainable_spectra_analysis](../findings/fragnnet/unexplainable_spectra_analysis.md) | REFERENCE (2026-07-21) | Workflow for spectra the DAG cannot explain |
| [conditioning_refactor_plan](../plans/conditioning_refactor_plan.md) | DEFERRED (2026-10-06) | Audit of 3 conditioning mechanisms; MASSter GNN context |
| [energy_based_spectrum_reranking_plan](../plans/energy_based_spectrum_reranking_plan.md) | PROPOSED (2026-09-13) | Energy head reranking, oracle-vs-null Gate A |
| [repo_reorg_plan](../plans/repo_reorg_plan.md) | ACTIVE (2026-10-12) | Cross-model reorg: registry and hooks, `fragmentation/`, `frag_utils` split, `models/` and `baselines/` packages, external ICEBERG via ms-pred, DGL-free env; Phases 0–7 committed on `refactor/fragnnet-masster-reorg`, awaiting merge |
| [bond_breakability_mask_plan](../plans/fragnnet/bond_breakability_mask_plan.md) | STOPPED (2026-07-03) | Hard bond mask; bucket table source |
| [classyfire_atom_pretraining_plan](../plans/fragnnet/classyfire_atom_pretraining_plan.md) | PROPOSED (2026-07-25) | ClassyFire pretraining |
| [complementary_cut_supervision_experiment_plan](../plans/fragnnet/complementary_cut_supervision_experiment_plan.md) | PROPOSED (2026-08-10) | Complementary-cut KL supervision |
| [dag_bond_radius_env_plan](../plans/fragnnet/dag_bond_radius_env_plan.md) | SUPERSEDED (2026-07-05) | Learned radius-1/2 scorer, AUC 0.884 |
| [dag_fanout_cap_plan](../plans/fragnnet/dag_fanout_cap_plan.md) | SUPERSEDED (2026-07-05) | Per-parent fan-out cap |
| [dag_full_recall_tractability_plan](../plans/fragnnet/dag_full_recall_tractability_plan.md) | DEFERRED (2026-10-06) | Recall under node budget; mostly negative |
| [false_peak_loss_plan](../plans/fragnnet/false_peak_loss_plan.md) | CLOSED NULL (2026-07-16) | False-peak penalty and calibration |
| [fluxmpnn_plan](../plans/fragnnet/fluxmpnn_plan.md) | STOPPED (2026-07-22) | DAG-flow output law |
| [fragment_bond_set_representation](../plans/fragnnet/fragment_bond_set_representation.md) | DEFERRED (2026-10-06) | Bond sets, charge placement, regularization reframe |
| [fragment_context_gate_experiment_plan](../plans/fragnnet/fragment_context_gate_experiment_plan.md) | PROPOSED (2026-08-11) | Per-fragment context routing and gate |
| [fragnnet_audit_fixes](../plans/fragnnet/fragnnet_audit_fixes.md) | LANDED (2026-09-12) | Six audit fixes, plus formula-support crash fix |
| [ft_contender_pair_minimal_training_plan](../plans/fragnnet/ft_contender_pair_minimal_training_plan.md) | STOPPED (2026-10-12) | Pair BCE retired |
| [global_fragment_intensity_ranking_experiment_plan](../plans/fragnnet/global_fragment_intensity_ranking_experiment_plan.md) | PROPOSED (2026-10-06) | Global base-mass ranking and CE pair supervision; [log](../history/global_fragment_intensity_ranking_experiment_plan_log.md) |
| [ion_collision_energy_matrix_experiment_plan](../plans/fragnnet/ion_collision_energy_matrix_experiment_plan.md) | PROPOSED (2026-08-13) | Ion x CE matrix diagnostic |
| [sibling_ranking_target_plan](../plans/fragnnet/sibling_ranking_target_plan.md) | DEFERRED (2026-10-06) | Sibling ranking loss on P(n) |
| [finetune_fragnnet](../guides/finetune_fragnnet.md) | GUIDE (2026-10-01) | Fine-tuning a checkpoint |
| [run_inference](../guides/run_inference.md) | GUIDE (2026-10-01) | Inference end to end |
| [frag_gen_prepass](../reference/frag_gen_prepass.md) | REFERENCE (2026-04-05) | BFS + SMARTS prepass DAG build |
| [fragnnet_pl_memory_review](../reference/fragnnet_pl_memory_review.md) | REFERENCE (2026-04-21) | Training-loop memory pressure points |
| [pairwise_loss](../reference/pairwise_loss.md) | REFERENCE (2026-06-08) | Shipped pairwise loss and sampler |
| [rearrangement_smarts](../reference/rearrangement_smarts.md) | REFERENCE (2026-06-20) | Gas-phase rearrangement SMARTS survey |
| [smarts](../reference/smarts.md) | REFERENCE (2026-03-02) | 72-rule SMARTS coverage survey (pre-split) |
| [detect_isotope_peaks_tuning](../reference/detect_isotope_peaks_tuning.md) | REFERENCE (2026-04-20) | Isotope-peak cleanup for training |
| [dataloader_data](../reference/dataloader_data.md) | REFERENCE (2026-09-08) | Data mapping definitions |
| [lz_molecular_generation](../notes/lz_molecular_generation.md) | NOTE (2026-06-08) | Spectrum-conditioned generation literature, part 1 |
| [lz_part2](../notes/lz_part2.md) | NOTE (2026-06-08) | Part 2: spectrum-to-DAG generation |

Neighbouring hubs: [MASSter core](masster_core.md), [MASSter generalization](masster_generalization.md), [MSGYM/GLACIER](msgym.md), [Spectraverse](spectraverse.md), [NIST23 EI](nist23_ei.md), [CASMI/Enveda](casmi_enveda.md).

## Key artifacts
- **DAG data:** `data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse.tar` (Stage A needs it on Fir), `nist20_ma_mi_d4_h4_isoFalse_smpFalse.tar`, `nist23_ma_mi_d3_h4_isoFalse.tar`.
- **Baseline config:** `config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all.yml`; latest control run `jnokddr9` (2026-09-14, Vulcan L40S, 100 epochs, best val `cos_sim_0.01` 0.7093). `pqx7ywc9` and `alvhehe2` are the NIST20 test inference references; `9t63zils` is the NodeMLP behind the ranking findings.
- **Stage A:** retired; historical configs are available in Git history.
- **Oracle tables:** `data/eval/oracle_cos_nist20_default/` (per-spectrum CSVs), produced by `scripts/eval/dag_oracle_scores.py`; missing peaks by `scripts/eval/find_missing_peaks.py`.
- **Probes:** `scripts/eval/ce_extent_headroom.py`, `scripts/eval/stage0_branching_model_vs_oracle.py`, `scripts/eval/branch_beta_target_coverage.py`, `scripts/debug/fragnnet/debug_ce_replicate_noise_floors.py`, DAG pruning probes in `scripts/debug/dag/`.
