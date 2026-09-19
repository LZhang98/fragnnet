# FraGNNet Documentation Index

> **Status:** REFERENCE

Start with [current project state](project-state.md), the [architecture map](architecture.md),
and the [session workflow](workflow/session.md). Shared constraints live in
[rules/engineering.md](rules/engineering.md).

## Shared workflow and rules

| Doc | Purpose |
| --- | --- |
| [project-state.md](project-state.md) | Current handoff, research navigation, and next task |
| [architecture.md](architecture.md) | Verified entry points and architectural boundaries |
| [workflow/session.md](workflow/session.md) | Development, experiment provenance, validation, and handoff |
| [rules/engineering.md](rules/engineering.md) | Core behavior, environment, compatibility, validation, and Git rules |
| [rules/python.md](rules/python.md) | Python formatting, imports, docstrings, testing, and hyperparameters |
| [rules/documentation.md](rules/documentation.md) | Documentation placement, banners, indexing, and preservation |

Research documents generally carry a status banner directly under their title. Check it before
acting on a doc — several plans here are **closed nulls or superseded**, and they are kept
deliberately so the same line does not get re-run.

| Status | Meaning |
|--------|---------|
| `GUIDE` / `REFERENCE` | Describes how something works or how to run it; verify against current code. |
| `ACTIVE` | Live plan. Something is meant to happen next. |
| `PROPOSED` | Written up, not started. |
| `LANDED` | The plan shipped; the doc now describes production behavior. |
| `FINDING` | A measurement. `CLOSED` = the question is answered and settled. |
| `CLOSED NULL` | Measured, no headroom. **Read before re-proposing the same mechanism.** |
| `SUPERSEDED` / `STOPPED` / `PARTLY DEAD` | Do not resume without reading why it stopped. |
| `CURATED ANALYSIS` | Long-running analysis doc. Append via its slash command rather than hand-editing. |

**Start at [plans/model_improvement_plan.md](plans/model_improvement_plan.md)** — the canonical
roadmap. It links the analyses that justify each item.

| Audit | Scope |
|-------|-------|
| [Connected-fragment beam reuse and equivalence](findings/connected_fragment_beam_reuse_and_equivalence_2026_09_17.md) | Measured on trained logits: the beam's captured mass is flat at 0.8695 vs 1.0000 for cut_masks because it is complete below half the molecule and limited above it; the size prior is isolated to the scoring change, and mask scoring alone trades it for a budget-efficiency loss. |
| [Connected-fragment beam speed audit](findings/connected_fragment_beam_speed_2026_09_17.md) | Synchronized real-graph GPU audit: correctness fixes raise batch-128/beam-256 candidate time 3.07× and forward/backward time 68%; complement connectivity dominates. |
| [Cross-adduct loss headroom](findings/cross_adduct_loss_headroom_2026_09_17.md) | Data-side gate for grouping MS/MS by adduct: 46.5% of NIST20 molecules carry >=2 adducts, 78,938 groups match on instrument/mode/NCE, splits are adduct-safe, and a molecule-independent neutral-loss transform closes only 9.3% of the gap to the replicate floor, so ~92% of the cross-adduct difference is molecule-specific. |
| [FastGNN proposal option](plans/masshatter_fast_gnn_proposal.md) | LANDED but EVIDENCE AGAINST CONTINUING: implemented and validated, then measured against `cut_masks` and found to have an architectural coverage ceiling; the prepared arms should not be submitted as configured. |
| [MASShatter GPU audit](plans/masshatter_gpu_audit.md) | Static + measured (2026-09-14) GPU utilization and CPU–GPU synchronization audit of the masshatter and ion-H-selection branches; branch-specific optimization priorities. |
| [Intensity-head bucketing sync plan](plans/masshatter_intensity_head_bucketing_sync_plan.md) | PROPOSED design for removing avoidable host syncs in `CandidateIntensityTransformerHead`'s per-molecule bucketing; not yet implemented. |

---

## guides/ — how to run things

| Doc | What it covers |
|-----|----------------|
| [submit_cluster_runs.md](guides/submit_cluster_runs.md) | Generate, inspect, and submit Fir/Vulcan jobs; CV arrays, staging, and output persistence. |
| [benchmark_masshatter_cpu.md](guides/benchmark_masshatter_cpu.md) | Freeze source and generate three-hour L40S CPU/hybrid/full-GPU benchmarks for batches 128/256/512. |
| [benchmark_masshatter_merged_control.md](guides/benchmark_masshatter_merged_control.md) | Merged MASShatter training arms on the CPU/CUDA candidate backends; NCE-only, CMF off, exposure-matched epochs, and the confounds this leaves. |
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
| [masshatter_smarts_rules.md](reference/masshatter_smarts_rules.md) | MASShatter-specific allowlist, literature additions, exclusions, and validation gate. |
| [fragnnet_pl_memory_review.md](reference/fragnnet_pl_memory_review.md) | GPU/CPU memory pressure points in the training loop. |

## findings/ — measured results

| Doc | Result |
|-----|--------|
| [masshatter_regularization_audit_2026_09_19.md](findings/masshatter_regularization_audit_2026_09_19.md) | Dropout, normalization and generalization audit over 156 configs and 226 W&B runs. Only two runs vary a regularization or optimization setting without also changing the architecture, and both predate the `tolerance_consensus` merged-target change. The train/eval candidate-set divergence is driven by BatchNorm, not dropout (Jaccard 0.317/0.611 at dropout 0 under `batch`, exactly 1.000 under `layer`/`graph`/`none`); branch normalization prevents residual attenuation for every normalization type; `lr_schedule_type: cosine` is still silently ignored by `SpectrumPL`. Post-LN gradients measure flat, so that is not a reason for warmup. Also records the 2026-09-19 decision to drop the atom decoder (no epoch-matched cosine cost, 1.8x faster) and its effect on parameter shares. Probes: `scripts/debug/probe_masshatter_regularization.py`. |
| [connected_fragment_beam_reuse_and_equivalence_2026_09_17.md](findings/connected_fragment_beam_reuse_and_equivalence_2026_09_17.md) | Beam vs `cut_masks` objective algebra (equal up to a charge prior, ring-reconnection terms and search), plus a measured size/topology prior: on trained logits the beam's captured mass is flat at 0.8695 vs 1.0000 for cut_masks because it is complete below half the molecule and blind above it; cut-cap and budget confounds excluded, and the prior isolated to the scoring change against the original prototype. Reproducible via `scripts/debug/analyze_proposal_fragment_coverage.py`. |
| [cross_adduct_loss_headroom_2026_09_17.md](findings/cross_adduct_loss_headroom_2026_09_17.md) | Model-free gate for a cross-adduct loss. Matched-pair census, InChIKey split safety, ~40% shared neutral-fragment intensity at 29x over control, and a held-out global-vs-molecule-specific decomposition showing 90.7% molecule-specific. The trained-model half is open: no multi-adduct MASShatter checkpoint exists in this checkout. Probe: `scripts/debug/probe_cross_adduct_headroom.py`. |
| [masshatter_training_speed_2026_09_17.md](findings/masshatter_training_speed_2026_09_17.md) | s3gy1ro4 speed audit: 44.4-minute epochs, 33% median GPU utilization; disabling monitoring-only channel attribution and base CE improves local batch-128 forward/backward throughput by 1.40×. |
| [masshatter_unmerged_overfitting_2026_09_16.md](findings/masshatter_unmerged_overfitting_2026_09_16.md) | All 27 unmerged W&B histories: widening loss gaps, limited late regression, incomplete train-cosine instrumentation, and corrected sampler/scheduler claims. |
| [subformula_filter_ft_qtof.md](findings/subformula_filter_ft_qtof.md) | Cleaning census and verified saved D3/D4 filtered oracle; D4 95.92% weighted recall, with explicit limits on comparison to MASShatter. |
| [masshatter_loss_audit_2026_09_19.md](findings/masshatter_loss_audit_2026_09_19.md) | Audit of the MASShatter loss path after the `86cbf98d` batching refactor. The padded rewrite is semantically faithful, but padding to the per-chunk tail dominates real cost: with the measured merged peak-count distribution, Hungarian gets *worse* as `loss_batch_size` grows (116 -> 442 ms at 8 -> 64) while the uniform-input benchmark showed the opposite, and the chunk-splitting guard cannot fire when predicted-peak counts are roughly equal. The two channel-diagnostic loops that every `channel_selection` config enables cost 81.6 ms per batch, more than the loss itself, and are faster on CPU. Also: batch size inferred from observed peaks silently shrinks the result for a trailing empty spectrum. Local RTX 2070S, SciPy assignment fallback; Hungarian magnitudes need a cluster re-measure. |
| [masshatter_smarts_review_2026_09_13.md](findings/masshatter_smarts_review_2026_09_13.md) | SMARTS registry and eligibility audit, 75 passing tests, pyrimidine-carbamate gap, and positive-ion literature priorities. |
| [masshatter_proposal_review_2026_09_13.md](findings/masshatter_proposal_review_2026_09_13.md) | Source review of proposal coverage, deterministic selection, warmup routing, diversity, and mixture capacity; prioritized diagnostics and untested improvements. |
| [loss_setup_review_2026_09_13.md](findings/loss_setup_review_2026_09_13.md) | Reproduced FraGNNet JSS vectorization, OOS, and support-target issues; MASShatter gradient, correspondence, proposal, and warmup findings. |
| [fragnnet_masshatter_audit_2026_09_12.md](findings/fragnnet_masshatter_audit_2026_09_12.md) | Reproduced checkpoint, inference, FraGNNet loss/explanation, and shared metric defects; full-suite source/binary validation. |
| [masshatter_code_audit.md](findings/masshatter_code_audit.md) | SMARTS/proposal-only-warmup construction guard removed and validated (2026-09-11); earlier fixes and pickle API fix implemented; 90 focused tests pass. CPU bfloat16 mixture issue deferred by user. |
| [masshatter_context_design_audit.md](findings/masshatter_context_design_audit.md) | Merged-context audit and implemented ACE range/weighted aggregation fixes; metadata coverage and deferred initialization suggestion. |
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
| [masshatter_topology_assembly_performance_plan.md](plans/masshatter_topology_assembly_performance_plan.md) | Measurement-gated topology and assembly optimization for the three-hour MA/MI-unmerged constraint. Stage 0 can abort the work; every retained stage must improve end-to-end optimizer-step throughput. |
| [masshatter_direct_loss_batching_plan.md](plans/masshatter_direct_loss_batching_plan.md) | **LANDED (working tree, uncommitted).** `sparse_peak_list_distance` was a per-spectrum Python loop; it now uses the vectorized cross-entropy's chunk/pad/reduce shape, and `round_aggregate_peaks` is a batched segment reduction that all twelve of its call sites (including FraGNNet's vectorized JSS and the eval path) inherit. Chunk-64, 52/512 peaks, fwd+bwd: `closest` 170.1 → 3.8 ms, `hungarian` 208.9 → 19.0 ms, syncs ~1800 → 19. Hungarian's residual is the local scipy fallback — cluster jobs install `torch_linear_assignment` and should land near 4 ms, so **re-measure there before quoting a Hungarian number**. Loss values unchanged over 144 randomized equivalence cases; 3827 tests pass. Records the Hungarian exact-tie ambiguity (two equally optimal assignments, losses 0.097 apart) that the comparison surfaced — pre-existing, not a regression. |
| [masshatter_candidate_backend_plan.md](plans/masshatter_candidate_backend_plan.md) | CPU improvements and opt-in exact CUDA construction implemented; local speed/parity checks passed, target-device gates open. |
| [inference_step_consolidation_plan.md](plans/inference_step_consolidation_plan.md) | **Landed (working tree, uncommitted).** The three `SpectrumPL` inference entry points are one `inference_step(batch, *, metrics=None)`; `resolve_pred_ints`/`step_untransformed` deleted. No model evaluation run. |
| [fragnnet_audit_fixes.md](plans/fragnnet_audit_fixes.md) | Six FraGNNet/shared audit fixes on feiw_v2; reproduced regressions and compatibility notes. |
| [model_improvement_plan.md](plans/model_improvement_plan.md) | **Canonical roadmap.** Start here. |
| [sibling_ranking_target_plan.md](plans/sibling_ranking_target_plan.md) | **Current lead.** Coverage gate passed (583,539 targets at margin 0.1); ranking loss on `P(n)` not yet implemented. |
| [fragment_bond_set_representation.md](plans/fragment_bond_set_representation.md) | All four gates measured, objective corrected to regularization (read §7 first). Next: run the written-but-never-run `cc_atom_dropout` sweep. |
| [nist_data_quality_plan.md](plans/nist_data_quality_plan.md) | Contaminant-mass and unexplainable-spectra diagnostics; evidence needed before filtering. |
| [conditioning_refactor_plan.md](plans/conditioning_refactor_plan.md) | MASShatter merged routing pair has audited CE-input corrections; FraGNNet routing/scaler screen prepared; no runs launched. |
| [masshatter_regularization_sweep_plan.md](plans/masshatter_regularization_sweep_plan.md) | **PROPOSED.** One-hour Fir arms (6 one-knob regularization/normalization + controls) executing the audit's "sweep second" half, with train-cosine instrumentation on for the first time. Measured replicate floor 0.003 means the screen resolves ~0.007. A paired 2g/3g MIG pilot picks the sweep shape: `loss_batch_size` is pure loss chunking, so batch 512 fits a 20 GB slice at `lb16`, but 2g carries half the memory bandwidth for 57% of the price. Read the epoch-matched protocol and the bias against the dropout arms before interpreting it. |
| [masshatter_improvement_plan.md](plans/masshatter_improvement_plan.md) | Unmerged CMF control running; measured reduced-monitoring config prepared, with remaining encoder, loss and support experiments proposed. |
| [dag_full_recall_tractability_plan.md](plans/dag_full_recall_tractability_plan.md) | Active hub for DAG pruning, but **mostly negative results** — read the step results before proposing another mechanism. |

### Proposed, not started

| Doc | Idea |
|-----|------|
| [energy_based_spectrum_reranking_plan.md](plans/energy_based_spectrum_reranking_plan.md) | Shared design of record: null-adjusted oracle gate, ranking objective, matched controls; FraGNNet anchor. |
| [masshatter_energy_based_model_plan.md](plans/masshatter_energy_based_model_plan.md) | EBM program: bond-cut modeling first, spectrum reranking second; MASShatter spectrum setup delta. |
| [masshatter_bond_cut_ebm_plan.md](plans/masshatter_bond_cut_ebm_plan.md) | Bond-cut-first diagnostic: D4 evidence limits the coverage premise; require support or efficiency headroom before pair-energy training. |
| [complementary_cut_supervision_experiment_plan.md](plans/complementary_cut_supervision_experiment_plan.md) | Target definition, matched GINE-1 measurement gates, training arms, and stop rules. Read the separate findings before running it. |
| [global_fragment_intensity_ranking_experiment_plan.md](plans/global_fragment_intensity_ranking_experiment_plan.md) | Global base-mass and online adjacent-CE contender supervision with bounded, group-balanced context batching and matched sampler controls. |
| [ft_contender_pair_minimal_training_plan.md](plans/ft_contender_pair_minimal_training_plan.md) | Minimal cache-free FT experiment: bounded CE-context batching plus online confidence-weighted soft A/B ranking. |
| [ion_collision_energy_matrix_experiment_plan.md](plans/ion_collision_energy_matrix_experiment_plan.md) | Ion × CE diagnostics plus a proposed sparse full-trajectory loss compared with the shipped precursor and adjacent-switch losses; V2 joint decoding is gated on V1. |
| [fragment_context_gate_experiment_plan.md](plans/fragment_context_gate_experiment_plan.md) | Zero-code `mol` versus `mol,mlp` context-routing ablation; final-logit gate is conditional on a positive result. |
| [classyfire_atom_pretraining_plan.md](plans/classyfire_atom_pretraining_plan.md) | Pretrain a transferable encoder on ClassyFire labels; no annotations needed at inference. |
| [masshatter_plan.md](plans/masshatter_plan.md) | MASShatter: rank global bond-cut masks, marginalize duplicate latent pathways into unique-fragment tokens, and predict charged fragment-ion outcomes without a depth-limited fragmentation tree. |
| [masshatter_candidate_pathway_refactor.md](plans/masshatter_candidate_pathway_refactor.md) | **LANDED.** 7 branches collapsed into 5 layers, exploration deleted, `mixtures=1` now threads. Surfaced two latent bugs. Read the `mask_selection_ms` semantics note before comparing timing metrics across runs, and the config note before running a `legacy/explore*` config. |
| [masshatter_nonprobabilistic_intensity_plan.md](plans/masshatter_nonprobabilistic_intensity_plan.md) | Replace MASShatter's Gaussian probability-density intensity objective with tolerance-aligned peak-list cosine/JSD plus optional entropy matching, and test independent Softplus intensities against the candidate softmax. Intensity only; fragment proposal is unchanged. |

### Closed — read before reviving

| Doc | Why it stopped |
|-----|----------------|
| [fluxmpnn_plan.md](plans/fluxmpnn_plan.md) | Stage 0 gate GREEN, but conserved DAG flow trained null-to-harmful and the learned β collapsed to a bond-type lookup (72% self-explained, 18% of real branching). Unsupervised competition line **dead**. |
| [bond_breakability_mask_plan.md](plans/bond_breakability_mask_plan.md) | **Stopped 2026-07-03** — 30.28% edge reduction but only 1.50% node reduction, and nodeMLP (nodes) is the cost driver. Its bucket table survives as the source of truth for `BUCKET_LOSS_RANK`. |
| [dag_fanout_cap_plan.md](plans/dag_fanout_cap_plan.md) | Right mechanism, wrong ranking — the hand-mined table lost to the learned radius-1/2 scorer. |
| [masshatter_rl_exploration_plan.md](plans/masshatter_rl_exploration_plan.md) | **Superseded 2026-08-19** — R0 moved into the main support diagnostics, R1 changes the objective, R2/R3 retain the independent inference scorer, and R4 is replaced by the main plan's normalized canonical autoregressive path. |
| [dag_bond_radius_env_plan.md](plans/dag_bond_radius_env_plan.md) | Confirmed the learned scorer (held-out AUC 0.884); prototype and integration folded into the tractability plan. |
| [false_peak_loss_plan.md](plans/false_peak_loss_plan.md) | **Closed null** — `log1p_huber` is harmful above lambda=0.1. Cited by the two loss sweeps in `slurm_scripts/`. |
| [metal_support_tier1.md](plans/metal_support_tier1.md) | **Landed** — shipped in `utils/frag_utils.py` and `utils/data_utils.py`. |

## notes/ — scratch

| Doc | What it is |
|-----|-----------|
| [lz_molecular_generation.md](notes/lz_molecular_generation.md) | Literature notes on spectrum-conditioned molecular generation (§1–3). |
| [lz_part2.md](notes/lz_part2.md) | Continuation of the above, starting at §4 (spectrum-to-DAG generation). |

## Curated analysis logs

These two are **append-only analysis logs** under `findings/`, organized by subclass. Add sections
with the `/analyze-subclass` and `/analyze-subclass-mh`
commands, which insert before the Summary section. The measurement they interpret comes from
`scripts/eval/find_missing_peaks.py`; the prose is written, not generated.
`src/fragnnet/frag/smarts_prepass.py` cites the first one by path.

| Doc | Scope |
|-----|-------|
| [missing_peak_findings.md](findings/missing_peak_findings.md) | FT, `[M+H]+`, 2,655 lines, grouped by root cause. |
| [missing_peak_findings_m-h.md](findings/missing_peak_findings_m-h.md) | FT, `[M-H]-`, 1,199 lines. |

`spectra_cleanup.md` is a **0-byte placeholder** committed to the repo — it needs either content or
deletion.

## figures/

PNGs for [findings/oracle_recall_wrecall_by_ce.md](findings/oracle_recall_wrecall_by_ce.md),
generated by `scripts/plotting/plot_oracle_ce_coverage.py`.
