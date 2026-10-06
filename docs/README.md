# FraGNNet Documentation Index

> **Status:** REFERENCE · **Updated:** 2026-10-06
> Map of the docs tree. Plans and findings are indexed in the workstream hubs, not here.

Start with [current project state](project-state.md), the [architecture map](architecture.md),
and the [session workflow](workflow/session.md). Status meanings and placement rules are in
[rules/documentation.md](rules/documentation.md#status-banners). Closed nulls and superseded
plans are kept deliberately; read a hub's "Settled" list before re-proposing a mechanism.

## Shared workflow and rules

| Doc | Purpose |
| --- | --- |
| [project-state.md](project-state.md) | Dashboard: one block per workstream, next actions, uncommitted work |
| [architecture.md](architecture.md) | Verified entry points and architectural boundaries |
| [workflow/session.md](workflow/session.md) | Development, experiment provenance, validation, and handoff |
| [rules/engineering.md](rules/engineering.md) | Core behavior, environment, compatibility, validation, and Git rules |
| [rules/python.md](rules/python.md) | Python formatting, imports, docstrings, testing, and hyperparameters |
| [rules/documentation.md](rules/documentation.md) | Placement, status vocabulary, hubs, project-state cap, history |
| [rules/artifacts.md](rules/artifacts.md) | Where checkpoints and run outputs go (`ckpts/`, `saved_ckpts/`, `output/`), naming, citing, and deletion |

## workstreams/ — one hub per line of work

Each hub has "Where it stands", "Open threads", "Settled — read before re-proposing", an
"All documents" table covering every plan and finding in the workstream, and key artifacts.

| Hub | Covers |
| --- | --- |
| [masster_generalization.md](workstreams/masster_generalization.md) | MASSter train/val gap on NIST20 (`i1`→`i4`): memorization routes, error analyses, intensity displacement, SMARTS pathway, QET/NCE input, ORB instruments, closed nulls (isotope envelope, JEPA) |
| [masster_core.md](workstreams/masster_core.md) | MASSter design, architecture and regularization screens, losses, candidate/proposal machinery, ion-H channels, speed/GPU, feature and information audits |
| [msgym.md](workstreams/msgym.md) | MassSpecGym training, formula-track retrieval, GLACIER comparisons, ChemBERTa-3, annotation-aware retrieval |
| [spectraverse.md](workstreams/spectraverse.md) | Spectraverse v1.0.2 pipeline, folds, data audits, oracle ceiling, prepared arms |
| [nist23_ei.md](workstreams/nist23_ei.md) | MASSter on nominal NIST23 EI |
| [casmi_enveda.md](workstreams/casmi_enveda.md) | CASMI 2026 data, Enveda-180, combined PCDL/NIST23/Enveda split, shared NIST data quality and isotope/subformula detection |
| [fragnnet.md](workstreams/fragnnet.md) | FraGNNet and the DAG: recall, oracle ceilings, intensity ranking, FluxMPNN, conditioning refactor, curated missing-peak logs |

## guides/ — how to run things

| Doc | What it covers |
|-----|----------------|
| [submit_cluster_runs.md](guides/submit_cluster_runs.md) | Generate, inspect, and submit Fir/Vulcan jobs; CV arrays, staging, and output persistence. |
| [benchmark_masster_cpu.md](guides/benchmark_masster_cpu.md) | Three-hour L40S CPU/hybrid/full-GPU MASSter benchmarks for batches 128/256/512. |
| [benchmark_masster_merged_control.md](guides/benchmark_masster_merged_control.md) | Merged MASSter arms on the CPU/CUDA candidate backends and the confounds left. |
| [run_inference.md](guides/run_inference.md) | Data prep, prediction, fragment annotation, MASSter peak rows and the peak filter. |
| [run_msgym_retrieval.md](guides/run_msgym_retrieval.md) | End-to-end MSGYM 1.5 retrieval for a MASSter checkpoint, locally or on Fir/Vulcan. |
| [finetune_fragnnet.md](guides/finetune_fragnnet.md) | Fine-tuning for spectrum-to-molecule ranking and retrieval; component freezing. |
| [nist_human_review.md](guides/nist_human_review.md) | The ranked suspicious-spectrum queue and visual report. |

Data preprocessing commands (including Spectraverse folds) are in
[preproc_scripts/README.md](../preproc_scripts/README.md); config resolution is in
[config/README.md](../config/README.md).

## reference/ — how the system works

| Doc | What it covers |
|-----|----------------|
| [dataloader_data.md](reference/dataloader_data.md) | Batch dictionary schema. Oldest doc here; verify against the code. |
| [frag_gen_prepass.md](reference/frag_gen_prepass.md) | How BFS + SMARTS prepass build the fragment DAG. |
| [feature_set_audit.md](reference/feature_set_audit.md) | Which FraGNNet features are wired. Its "open gaps" line is stale (`a_ring_size`/`b_ring` are enabled in 44 configs). |
| [masster_feature_set_audit.md](reference/masster_feature_set_audit.md) | Which atom/bond/candidate/conditioning features MASSter consumes. |
| [pairwise_loss.md](reference/pairwise_loss.md) | Design of the shipped pairwise loss + sampler. |
| [detect_isotope_peaks_tuning.md](reference/detect_isotope_peaks_tuning.md) | Isotope-peak cleanup for training. |
| [smarts.md](reference/smarts.md) | 72-rule coverage survey with per-rule status codes. |
| [rearrangement_smarts.md](reference/rearrangement_smarts.md) | Literature survey + SMARTS validation for gas-phase rearrangements. |
| [masster_smarts_rules.md](reference/masster_smarts_rules.md) | MASSter SMARTS allowlist, literature additions, exclusions, and validation gate. |
| [fragnnet_pl_memory_review.md](reference/fragnnet_pl_memory_review.md) | GPU/CPU memory pressure points in the training loop, and FraGNNet training memory levers. |
| [inference_pitfalls.md](reference/inference_pitfalls.md) | Fixed inference/DAG bugs, `dags.h5` and HDF5 export sizing, batching, config precedence (checkpoint hparams are the custom config), known-broken scripts. |

## notes/ — literature and scratch

| Doc | What it is |
|-----|-----------|
| [massspecgym_in_the_wild_audit.md](notes/massspecgym_in_the_wild_audit.md) | **Read before any MassSpecGym comparison.** Evaluation pitfalls; spectrum-blind PubChem ranking scores 49.98% HR@1. |
| [spurious_evaluations_spectraverse.md](notes/spurious_evaluations_spectraverse.md) | Spectrum-blind structure-only model gets 26.9% top-1 under a structure-disjoint split. |
| [spectraverse_curation_paper.md](notes/spectraverse_curation_paper.md) | Spectraverse dataset paper; its cosine > 0.99 dedup does not hold in v1.0.2. |
| [glacier_object_detection.md](notes/glacier_object_detection.md) | GLACIER: single-stage fragment detection; all its NIST'20 numbers are 0.1 Da binned. |
| [aime_deepms2reasoner.md](notes/aime_deepms2reasoner.md) | AIMe / DeepMS2Reasoner; its cosines use formula-filtered targets, not comparable. |
| [molspecflow.md](notes/molspecflow.md) | MolSpecFlow: joint molecule↔spectrum flow model; no code. |
| [memorization_literature_2026_09_24.md](notes/memorization_literature_2026_09_24.md) | Memorization literature mapped onto the MASSter route map. |
| [masster_generalization_methods_2026_09_27.md](notes/masster_generalization_methods_2026_09_27.md) | R-Drop, centered logits, SAM, FLAG, distillation and cross-molecule geometry. |
| [intensity_displacement_literature_2026_09_24.md](notes/intensity_displacement_literature_2026_09_24.md) | RRKM/QET and CFM-ID read against intensity displacement. |
| [masster_excision_literature_2026_09_28.md](notes/masster_excision_literature_2026_09_28.md) | Internal losses and O/F transfer evidence. |
| [lz_molecular_generation.md](notes/lz_molecular_generation.md), [lz_part2.md](notes/lz_part2.md) | Spectrum-conditioned molecular generation notes. |

## history/ — frozen logs

Verbatim dated logs moved out of live documents on 2026-10-06. Read the hub or plan first;
these are records, not current guidance.

| Doc | Moved from |
|-----|-----------|
| [handoff_log.md](history/handoff_log.md) | `project-state.md` handoff, research navigation and next-task sections |
| [masster_memorization_route_plan_log.md](history/masster_memorization_route_plan_log.md) | [memorization-route plan](plans/masster/masster_memorization_route_plan.md) |
| [masster_plan_log.md](history/masster_plan_log.md) | [MASSter plan](plans/masster/masster_plan.md) |
| [masster_ion_h_channel_selection_plan_log.md](history/masster_ion_h_channel_selection_plan_log.md) | [ion-H channel selection plan](plans/masster/masster_ion_h_channel_selection_plan.md) |
| [global_fragment_intensity_ranking_experiment_plan_log.md](history/global_fragment_intensity_ranking_experiment_plan_log.md) | [global intensity ranking plan](plans/fragnnet/global_fragment_intensity_ranking_experiment_plan.md) |

Plans removed from the tree in commit `bf134762` (`model_improvement_plan.md`,
`inference_step_consolidation_plan.md`, `metal_support_tier1.md`) are recoverable from git
history only; do not treat references to them as current guidance.

## Curated analysis logs

[missing_peak_findings.md](findings/fragnnet/missing_peak_findings.md) (FT `[M+H]+`) and
[missing_peak_findings_m-h.md](findings/fragnnet/missing_peak_findings_m-h.md) (FT `[M-H]-`)
are append-only. Add sections with `/analyze-subclass` and `/analyze-subclass-mh`; the
measurement comes from `scripts/eval/find_missing_peaks.py`.
`src/fragnnet/frag/smarts_prepass.py` cites the first one by path.

[`spectra_cleanup.md`](spectra_cleanup.md) is a placeholder with no content; it points at
`clean_spectra_utils.py` and [findings/subformula_filter_ft_qtof.md](findings/subformula_filter_ft_qtof.md).

## figures/

PNGs referenced by findings, e.g. [oracle_recall_wrecall_by_ce.md](findings/fragnnet/oracle_recall_wrecall_by_ce.md)
(generated by `scripts/plotting/plot_oracle_ce_coverage.py`).
