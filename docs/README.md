# FraGNNet Documentation Index

Every doc carries a status banner directly under its title. Check it before acting on a doc —
several plans here are **closed nulls or superseded**, and they are kept deliberately so the same
line does not get re-run.

| Status | Meaning |
|--------|---------|
| `GUIDE` / `REFERENCE` | Describes how something works or how to run it. Safe to follow. |
| `ACTIVE` | Live plan. Something is meant to happen next. |
| `PROPOSED` | Written up, not started. |
| `LANDED` | The plan shipped; the doc now describes production behavior. |
| `FINDING` | A measurement. `CLOSED` = the question is answered and settled. |
| `CLOSED NULL` | Measured, no headroom. **Read before re-proposing the same mechanism.** |
| `SUPERSEDED` / `STOPPED` / `PARTLY DEAD` | Do not resume without reading why it stopped. |
| `CURATED ANALYSIS` | Long-running analysis doc. Append via its slash command rather than hand-editing. |

**Start at [plans/model_improvement_plan.md](plans/model_improvement_plan.md)** — the canonical
roadmap. It links the analyses that justify each item.

---

## guides/ — how to run things

| Doc | What it covers |
|-----|----------------|
| [run_inference.md](guides/run_inference.md) | Data prep (`01_inf_prepare_mol_df.py`, `02_inf_prepare_spec_df.py`) and running prediction. |
| [finetune_fragnnet.md](guides/finetune_fragnnet.md) | Fine-tuning for spectrum-to-molecule ranking and retrieval; component freezing. |
| [nist_human_review.md](guides/nist_human_review.md) | Generating the ranked suspicious-spectrum queue and visual report. |

## reference/ — how the system works

| Doc | What it covers |
|-----|----------------|
| [dataloader_data.md](reference/dataloader_data.md) | Batch dictionary schema. **Oldest doc here (2025-12-10)** — verify against the code. |
| [frag_gen_prepass.md](reference/frag_gen_prepass.md) | How BFS + SMARTS prepass build the fragment DAG. |
| [feature_set_audit.md](reference/feature_set_audit.md) | Which features are actually wired. Open gaps: `a_ring_size`, `b_ring`. |
| [pairwise_loss.md](reference/pairwise_loss.md) | Design of the shipped pairwise loss + sampler. |
| [detect_isotope_peaks_tuning.md](reference/detect_isotope_peaks_tuning.md) | Canonical note on isotope-peak cleanup for training. |
| [smarts.md](reference/smarts.md) | 72-rule coverage survey with per-rule status codes. |
| [rearrangement_smarts.md](reference/rearrangement_smarts.md) | Literature survey + SMARTS validation for gas-phase rearrangements. |
| [fragnnet_pl_memory_review.md](reference/fragnnet_pl_memory_review.md) | GPU/CPU memory pressure points in the training loop. |

## findings/ — measured results

| Doc | Result |
|-----|--------|
| [dag_oracle_ceiling_nist20.md](findings/dag_oracle_ceiling_nist20.md) | Baseline d3 DAG ceiling = **0.937 cos_hun** on NIST20 test. Model-free upper bound. |
| [oracle_recall_wrecall_by_ce.md](findings/oracle_recall_wrecall_by_ce.md) | Coverage ceiling vs. collision energy and adduct. Figures in [figures/](figures/). |
| [unexplainable_spectra_analysis.md](findings/unexplainable_spectra_analysis.md) | Workflow for spectra the DAG cannot explain, grouped by adduct / molecule / ClassyFire class. |
| [cc_corruption_probe_finding.md](findings/cc_corruption_probe_finding.md) | **CLOSED.** Fragment identity *is* used (cos 0.82 → 0.61 under mask randomization) — the gap is comparative scoring, not features. |
| [ce_extent_headroom_finding.md](findings/ce_extent_headroom_finding.md) | **CLOSED NULL.** No headroom in the CE → "mass moves deeper" prior. Read before building any CE-conditioned module. |
| [fragment_intensity_ranking_findings.md](findings/fragment_intensity_ranking_findings.md) | Complementary-cut supervision is a soft negative under current weighting; global base-peak and CE-dependent ranking are the larger measured bottlenecks. |
| [dominant_peak_outcome_decomposition.md](findings/dominant_peak_outcome_decomposition.md) | Base-peak failure is **7:1 ranking, not DAG coverage**. Two-thirds of ranked misses are local neutral-loss neighbours; the apparent high-m/z bias resolves to a low-CE precursor over-call, and a global scalar correction is rejected. |

## plans/

### Live

| Doc | Where it stands |
|-----|-----------------|
| [model_improvement_plan.md](plans/model_improvement_plan.md) | **Canonical roadmap.** Start here. |
| [sibling_ranking_target_plan.md](plans/sibling_ranking_target_plan.md) | **Current lead.** Coverage gate passed (583,539 targets at margin 0.1); ranking loss on `P(n)` not yet implemented. |
| [fragment_bond_set_representation.md](plans/fragment_bond_set_representation.md) | All four gates measured, objective corrected to regularization (read §7 first). Next: run the written-but-never-run `cc_atom_dropout` sweep. |
| [nist_data_quality_plan.md](plans/nist_data_quality_plan.md) | Contaminant-mass and unexplainable-spectra diagnostics; evidence needed before filtering. |
| [conditioning_refactor_plan.md](plans/conditioning_refactor_plan.md) | Collapse 3 overlapping conditioning mechanisms into one. Plan + ablation config built, not run. |
| [dag_full_recall_tractability_plan.md](plans/dag_full_recall_tractability_plan.md) | Active hub for DAG pruning, but **mostly negative results** — read the step results before proposing another mechanism. |

### Proposed, not started

| Doc | Idea |
|-----|------|
| [complementary_cut_supervision_experiment_plan.md](plans/complementary_cut_supervision_experiment_plan.md) | Target definition, matched GINE-1 measurement gates, training arms, and stop rules. Read the separate findings before running it. |
| [global_fragment_intensity_ranking_experiment_plan.md](plans/global_fragment_intensity_ranking_experiment_plan.md) | Global base-mass and online adjacent-CE contender supervision with bounded, group-balanced context batching and matched sampler controls. |
| [ft_contender_pair_minimal_training_plan.md](plans/ft_contender_pair_minimal_training_plan.md) | Minimal cache-free FT experiment: bounded CE-context batching plus online confidence-weighted soft A/B ranking. |
| [ion_collision_energy_matrix_experiment_plan.md](plans/ion_collision_energy_matrix_experiment_plan.md) | Ion × CE diagnostics plus a proposed sparse full-trajectory loss compared with the shipped precursor and adjacent-switch losses; V2 joint decoding is gated on V1. |
| [fragment_context_gate_experiment_plan.md](plans/fragment_context_gate_experiment_plan.md) | Zero-code `mol` versus `mol,mlp` context-routing ablation; final-logit gate is conditional on a positive result. |
| [classyfire_atom_pretraining_plan.md](plans/classyfire_atom_pretraining_plan.md) | Pretrain a transferable encoder on ClassyFire labels; no annotations needed at inference. |
| [directional_bond_break_probability_plan.md](plans/directional_bond_break_probability_plan.md) | MASShatter: rank global bond-cut masks and charged fragment-ion outcomes without a depth-limited fragmentation tree. |

### Closed — read before reviving

| Doc | Why it stopped |
|-----|----------------|
| [fluxmpnn_plan.md](plans/fluxmpnn_plan.md) | Stage 0 gate GREEN, but conserved DAG flow trained null-to-harmful and the learned β collapsed to a bond-type lookup (72% self-explained, 18% of real branching). Unsupervised competition line **dead**. |
| [bond_breakability_mask_plan.md](plans/bond_breakability_mask_plan.md) | **Stopped 2026-07-03** — 30.28% edge reduction but only 1.50% node reduction, and nodeMLP (nodes) is the cost driver. Its bucket table survives as the source of truth for `BUCKET_LOSS_RANK`. |
| [dag_fanout_cap_plan.md](plans/dag_fanout_cap_plan.md) | Right mechanism, wrong ranking — the hand-mined table lost to the learned radius-1/2 scorer. |
| [dag_bond_radius_env_plan.md](plans/dag_bond_radius_env_plan.md) | Confirmed the learned scorer (held-out AUC 0.884); prototype and integration folded into the tractability plan. |
| [false_peak_loss_plan.md](plans/false_peak_loss_plan.md) | **Closed null** — `log1p_huber` is harmful above lambda=0.1. Cited by the two loss sweeps in `slurm_scripts/`. |
| [metal_support_tier1.md](plans/metal_support_tier1.md) | **Landed** — shipped in `utils/frag_utils.py` and `utils/data_utils.py`. |

## notes/ — scratch

| Doc | What it is |
|-----|-----------|
| [lz_molecular_generation.md](notes/lz_molecular_generation.md) | Literature notes on spectrum-conditioned molecular generation (§1–3). |
| [lz_part2.md](notes/lz_part2.md) | Continuation of the above, starting at §4 (spectrum-to-DAG generation). |

## Top level — curated analysis logs

These two are **append-only analysis logs**, kept at top level because they grow by subclass rather
than belonging to one category. Add sections with the `/analyze-subclass` and `/analyze-subclass-mh`
commands, which insert before the Summary section. The measurement they interpret comes from
`scripts/eval/find_missing_peaks.py`; the prose is written, not generated.
`src/fragnnet/frag/smarts_prepass.py` cites the first one by path.

| Doc | Scope |
|-----|-------|
| [missing_peak_findings.md](missing_peak_findings.md) | FT, `[M+H]+`, 2,655 lines, grouped by root cause. |
| [missing_peak_findings_m-h.md](missing_peak_findings_m-h.md) | FT, `[M-H]-`, 1,199 lines. |

`spectra_cleanup.md` is a **0-byte placeholder** committed to the repo — it needs either content or
deletion.

## figures/

PNGs for [findings/oracle_recall_wrecall_by_ce.md](findings/oracle_recall_wrecall_by_ce.md),
generated by `scripts/plotting/plot_oracle_ce_coverage.py`.
