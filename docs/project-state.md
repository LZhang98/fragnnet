# Current project state

> **Status:** REFERENCE · **Updated:** 2026-09-17

## Current handoff

The shared agent workflow is established: start at [AGENTS.md](../AGENTS.md), follow the
[session workflow](workflow/session.md), and use the [architecture map](architecture.md).
Existing AGENTS and Claude rules are organized into [engineering](rules/engineering.md),
[Python and ML](rules/python.md), and [documentation](rules/documentation.md) rules.
The agent entry point selects the applicable files by task; the original rules are preserved.
This includes import aliases, formatting/docstrings, testing conventions, and documentation
placement. Where legacy Claude constraints differed, the original AGENTS policy is retained:
preprocessing changes require explicit authorization and compatibility work; PR validation runs
when practical, with omitted checks reported.
The adoption changes remain uncommitted for user review.

Validation: shared-rule preservation, relative documentation links, and Git whitespace checks
passed. The configured interpreter provides pytest and Ruff. No model code changed; no training,
scientific evaluation, or full test suite was run for this documentation change.

MASShatter CMF alignment (2026-09-14): MASShatter now emits the full configured H-transfer
ladder for chemically valid, non-degenerate CMF adducts, matching FraGNNet. It preserves
reachable CRF/CMF duplicate suppression and in-source-loss exclusion; the pre-batch support
filter shares the same contract. Existing checkpoints are diagnostic-only because their newly
valid CMF-H channels were not trained. Validation: 3,032 passed, 21 skipped, 2 xfailed; the
CMF plan records the required oracle-coverage rebaseline and matched-control training.

## Research navigation

- [Energy-based spectrum reranking](plans/energy_based_spectrum_reranking_plan.md) is a proposed
  shared design of record with a FraGNNet anchor (2026-09-13). Next: reconcile the prior correction
  null, verify export with `--min_intensity -1`, and require oracle excess over matched random pools
  before implementing/training an energy head. Documentation only; no runs or implementation.
- [MASShatter improvement plan](plans/masshatter_improvement_plan.md) records the proposed
  unmerged sequence: corrected CE control, proposal-encoder comparison, Gaussian support-gated
  JSS, conservative output pruning, and measured throughput improvements. Five merged warmup
  configs are prepared for a CE/JSS/KL loss and intensity-transform screen. Next: verify checkpoint
  persistence before launching; resolve the unmerged anchor/warmup policy before its matched
  control. The plan includes [Vulcan generation and submission commands](plans/masshatter_improvement_plan.md#vulcan-commands-for-the-five-config-screen)
  for all five configs with a three-hour allocation. No runs were launched for this plan.
- [MASShatter code audit](findings/masshatter_code_audit.md) records three reproduced
  issues at `0b5ad3f4`: standalone SMARTS inference, charged-component masses, and
  SMARTS proposal gradient isolation. All three fixes and regression tests are present
  at `77fdbea2`. A continued audit reproduced two additional inference blockers:
  unconditional DAG-file filtering/dataset selection in the CLI and independent-output
  tabular export expecting `pred_logprobs`. Those fixes and the mass-feasibility
  alignment are now in the working tree, along with CLI directory/row-layout fixes.
  Validation: 781 passed, 8 skipped; six CLI regressions passed after adding Parquet
  serializer-boundary coverage. Real Parquet serialization was not verified because
  its optional engine is unavailable; checkpoint loading and dataset preprocessing
  are mocked in the CLI tests. Next check: a target-environment real-checkpoint CLI
  smoke run. Review the linked compatibility
  notes and re-evaluate affected checkpoints before comparing experiment metrics.
  A third audit pass found two remaining API issues: prediction-only
  `run_inference_pkl` rejects its existing `pred_ints` output, and direct CPU bfloat16
  calls with multiple proposal mixtures fail the normalization check. Standard CPU
  runner fallback avoids the latter. The pickle API fix is now in the working tree:
  90 tests passed, including real-model prediction-only regressions for both output
  modes and batch offsets. The bfloat16 issue is deferred by user because it is not
  used in the current workflow. Separate failure reproductions remain in the audit.
  The checkout was clean at audit start, so the
  uncommitted-work description in the workflow-adoption handoff above is historical.

- [Model improvement roadmap](plans/model_improvement_plan.md) remains the canonical source
  for research priorities and acceptance criteria. Its banner is dated 2026-07-24; reconcile
  its proposed next action with current code and artifacts before resuming it.
- [Documentation index](README.md) lists live, proposed, landed, and closed work, including
  MASShatter plans. Those are recorded statuses, not verification of current execution.
- Read linked [findings](findings/) before reviving a rejected or closed experiment.

No training processes, scheduler jobs, checkpoints, or external run statuses were audited during
workflow adoption. No new research priority or scientific conclusion is inferred here.

## Next task

The [FastGNN proposal option](plans/masshatter_fast_gnn_proposal.md) is implemented as an
explicit alternative to cut-mask proposals, preserving the default and checkpoint shapes.
Full suite: 3,451 passed, 96 skipped, 2 xfailed; three separate GPU checks and a real-data
forward/backward smoke passed. The linked plan supplies the config and score semantics.
The connected-fragment option now aligns search and scoring with boundary-cut/internal-intact
probabilities from the updated prototype; see the linked plan for changed checkpoint semantics.
Three merged MA/MI configs (128/256/512 fragments and matching beams) and a generation-only
Vulcan launcher with three-hour jobs are prepared in the linked plan.
The [connected-fragment audit](plans/masshatter_fast_gnn_proposal.md#connected-fragment-audit-2026-09-17)
found two selection issues now fixed in the working tree: stable beam ties remove padding
dependence, and complements are scored before output-budget pruning. Permanent regressions
and model/backend checks: 220 passed, 11 skipped in the sandbox; the six proposal CUDA checks
subsequently passed outside it. The [real-graph speed audit](findings/connected_fragment_beam_speed_2026_09_17.md)
measured candidate time 238.7 → 732.1 ms and forward/backward time 744.4 → 1,252.3 ms at batch
128 / beam 256 on RTX 2070 SUPER. Complement connectivity dominates; peak whole-step allocation
stays near 2.83 GB. Next: reduce unnecessary connectivity checks without changing selection,
then measure L40S throughput and coverage. No production code changed during the speed audit.
All four connected-fragment arms now use `mass_shatter_fast_gnn_max_boundary_cuts: 5`: at 3 the
boundary limit, not the fragment target, was the binding candidate budget, so the 256 and 512
arms filled only 63% and 40% of their budgets and ran at roughly half the `cut_masks` candidate
count. Template and checkpoint-migration defaults stay at 3.
On real NIST20 batches the beam is **not** a throughput improvement: 1,024 ms median against
70 ms for `cut_masks` with `candidate_backend: cuda` and 157 ms for its CPU backend, at batch
128 and comparable fragment counts. Batch cost is set by the batch maximum heavy-atom count,
whose median is 94 at batch 128, and 77% of the time is the per-growth-round complement
reachability loop introduced by the audit fix. See the
[real-precursor measurement](plans/masshatter_fast_gnn_proposal.md#real-precursor-throughput-measurement-2026-09-17)
for the profile and the proposed restructuring. Next: screen this arm on accuracy only, or
restructure complement connectivity before making any speed claim; no training run was launched.

MASShatter training speed (2026-09-17): the
[s3gy1ro4 audit](findings/masshatter_training_speed_2026_09_17.md) identifies monitoring-only
channel attribution and base CE as removable overhead. A matched fast config and reusable
benchmark are prepared; batch-128 local GPU forward/backward throughput improved by 1.40×.
Scientific losses and checkpoint shapes are unchanged; the two diagnostic families are disabled
in validation too. Loss/gradient equivalence and checkpoint compatibility passed. The small
attention-layout experiment was removed after failing to demonstrate useful whole-batch gains.
A batch-256/accumulation-1 variant is also prepared, preserving effective batch 256. It fits the
local sampled GPU workload (7.10 GB) but shows no throughput gain over batch 128; see the same
finding for matched measurements. Next: compare 128×2 and 256×1 on L40S and verify resolved resume
flags; no remote run or cluster job was changed. The requested 512×1 config is also prepared
with LR unchanged; its effective batch doubles to 512 and its GPU validation remains pending.

The [candidate-backend plan](plans/masshatter_candidate_backend_plan.md) is active. CPU numeric
construction now releases the GIL and shares a persistent thread pool; local full-training
throughput improved about 9.3% with exact candidate/gradient checks. The score-free selector
specialization was removed after failing its throughput gate. Membership compaction and streaming
are deferred by the current projections. The same [speed finding](findings/masshatter_training_speed_2026_09_17.md#cpu-implementation-and-full-training-measurements-2026-09-17)
records results and limits. Batch256 fits locally; batch512 runs out of memory on the 8 GB GPU.
[Three-hour L40S benchmark commands](guides/benchmark_masshatter_cpu.md) are prepared; no jobs were
submitted. CPU remains the default. The exact optional CUDA backend now passes local candidate,
model-gradient and checkpoint checks, with 19–23% higher full-training throughput than
improved CPU in the final 128/256 matrix. Explicit CUDA configs are available for all three batch sizes; see the same guide
for supported modes and compiler requirements. Final CPU full suite: 3,422 passed, 93 skipped,
2 xfailed; separate CUDA suite: 118 passed. Batch512 hits the local 8 GB GPU limit with either
backend. Next: confirm throughput, memory and native compilation on L40S before production promotion.

Live fast-run follow-up (2026-09-17): the existing L40S runs `dny62x0i`, `h0ibnndb`, and
`du6cr9dx` still report the earlier separate candidate timing fields and no backend setting.
Four CPU selection threads are enabled, but the new parallel topology/CUDA implementation is
not reflected in those runs. Monitoring-only fast128 reduces epoch time from 44.5 to 33.5 minutes;
256/512 take 32.1/31.8 minutes, so larger batches alone leave the CPU bottleneck. The old CPU
batch512 run does fit L40S (about 14.76 GB median sampled allocation); new-backend L40S acceptance
remains open. See the [live follow-up](findings/masshatter_training_speed_2026_09_17.md#live-fast-run-follow-up-earlier-candidate-code-still-running-2026-09-17).
Next: launch a fresh job from transferred updated source; running job-local installs will not
pick up these working-tree changes. No remote jobs were modified by this read-only audit.

Unmerged generalization audit (2026-09-16): the
[27-run W&B census](findings/masshatter_unmerged_overfitting_2026_09_16.md) supports
widening train/validation loss gaps in mature runs, with modest continued cosine gains.
The current `s3gy1ro4` run was still improving at epoch 8. No train cosine is logged.
This audit corrects the anchor provenance below: `ef7fl5c6` is **merged**, not unmerged.
The claimed architecture-invariant 0.688–0.693 band excludes many of the 27 runs.
Current `GroupSampler` does resample through the epoch hook; MASShatter's inherited
optimizer setup ignores cosine scheduler options. Next: matched evaluation-mode train/val
checkpoint metrics, then separate fixed-objective LR and regularization controls.
Documentation and a sampler probe only; no training or model/config changes.

Unmerged MASShatter CMF anchor prepared (2026-09-16): the
[improvement plan](plans/masshatter_improvement_plan.md#corrected-unmerged-cmf-anchor-prepared-2026-09-16)
records a new config,
`config/masshatter/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_unique256_enum4096_gine_gsa384_bondcentric_a2_b3_transformer_full_context_dual_ce_proposal_selection_only_inst_tol.yml`,
and a generation-only launcher
`scripts/manual_slurm/vulcan/submit_masshatter_unmerged_cmf.sh`. It is the lead unmerged arm
(`ef7fl5c6`, 0.7515) plus dual CE, `inst_types` reduced to the FT/QTOF actually present in the
split, channel diagnostics, and `ace_max: 300` (the template's 100 clamps 3.49% of this split's
spectra into one saturated embedding row once ACE is on); a resolved-config comparison shows
`spec_params` and `ace_max` as the only differing keys. Two conditioning differences against the
FraGNNet MA/MI reference are left in place as unmeasured treatments rather than defects:
`mass_shatter_context_in_gnn` and `mass_shatter_smarts_templates`, both `False`. The prepared
merged `context_ablation` pair is the place to settle the first. A two-arm proposal-warmup
pilot (`_warmup_e0_{2,4}.yml` beside the anchor, launcher
`scripts/manual_slurm/vulcan/submit_masshatter_unmerged_cmf_warmup.sh`) supplies the bounded
matched pilot that step 3 of the improvement plan asks for; each arm differs from the anchor by
`mass_shatter_proposal_only_until_epoch` alone, and the anchor is the no-warmup control. A real
four-epoch smoke of `e0_2` crossed the boundary with validation cosine flat through the frozen
epochs and rising once the intensity head unfroze. This is the first config to train the 2026-09-14 CMF alignment, because only its
`[M+Na]+`/`[M+Cl]-` spectra (38,156 and 7,820 of 480,002) have a non-degenerate CMF ladder. The
twelve-hour allocation and `j=` resume address the wall-time half of the convergence blocker that
holds up [Arm C](plans/masshatter_ion_h_channel_selection_plan.md). `filter_subformula_peaks`
stays off so the target spectra do not change alongside the candidate universe. Validation:
config resolved, both arms constructed, generated job script passed `bash -n`, and one full
subsampled epoch ran end-to-end on the local GPU (exit 0, `val_cos_sim_0.01_epoch/mean` present,
`last.ckpt` written). No Vulcan job was generated or submitted from this workstation, which has
no `sbatch`. Next: run the launcher on the Vulcan login node, `sbatch` it, and record the
scheduler and W&B run IDs in the improvement plan.

Subformula filter hooked up to MASShatter (2026-09-16): the merged Cl/Na fix reaches MASShatter
through the shared `BaseDataset` cleanup with no code change, but it was **inert for every
MASShatter config** — all of them use `data/split/nist20_inchikey`, which is 209,698 `[M+H]+`/FT
spectra with no Na, Cl, QTOF, or IT. Added
`config/masshatter/merged/subformula_filter/nist20_ma_mi_merged_independent_jss_warmup_e0_15_dense_subformula.yml`,
identical to its `[M+H]+` sibling except for `split_dp: data/split/nist20_ma_mi_inchikey`,
`prec_types` (four adducts) and `inst_types: [FT, QTOF]`, so the two remain a matched A/B on adduct
scope. MASShatter needs no DAGs, so the split change costs no preprocessing. The
[subformula finding](findings/subformula_filter_ft_qtof.md#clna-budget-fix-measured-on-the-multi-adduct-split-2026-09-16)
records the measured pre-fix/post-fix delta (+90 peaks of 62,702, every adduct nonnegative),
per-adduct retention at instrument tolerance, and the model/filter agreement on CMF validity.
Validation: 344 passed, 2 xfailed across the filter, MASShatter, dataset, and split-filter suites;
config resolved through `load_config`; cleanup run end-to-end on 2,620 real multi-adduct spectra
with zero emptied spectra. No training run was launched and no checkpoint was evaluated.
Next: launch the new config as the matched multi-adduct arm, or first measure oracle coverage on
the Na/Cl population. Note the existing 4-adduct MASShatter configs (`context_ablation`, `smarts`)
declare Na/Cl in `prec_types` but sit on the `[M+H]+`-only split, so their extra adduct slots and
CMF channels have never been exercised.

Inference path consolidation merged from `feiw_v2` (2026-09-16): the
[inference step consolidation plan](plans/inference_step_consolidation_plan.md) is LANDED on that
branch and is now merged here. It collapses the three `SpectrumPL` inference entry points into one,
deletes `resolve_pred_ints` and its `step_untransformed` flag, and removes
`GreedyInferenceBatchSampler` in favour of the lazy `IterableSpecMolFragDataset`. The same merge
brings the subformula-filter fix for Cl and Na adducts, HDF5 output compression, and the
`drop_empty_spectra` prediction-only fix. Training and loss paths keep `pred_logprobs` and were not
touched. Two contract changes from `feiw_v2` win over this branch's prior behaviour: the exporters
now rescale each spectrum's base peak to 1000 on top of the L1-normalized `pred_ints`, and
`run_inference.py` no longer offers `--save_format parquet`. Two MASShatter tests that pinned the
old exporter scale and the parquet writer were updated accordingly; this branch's `requires_dags`
guard and its `pred_intensities` output contract are preserved. Validation: 3,234 passed, 1 skipped,
2 xfailed; Ruff on the resolved files matches the `feiw_v2` baseline. Next: a real-checkpoint
inference smoke check; no GPU run or evaluation was performed for the merge.

Candidate attention topology audit (2026-09-13): the
[audit section](findings/masshatter_code_audit.md#candidate-attention-topology-audit-2026-09-13)
records seven findings on the uncommitted `full` candidate-attention arm, all fixed in the
working tree: a hyperparameter fallback default plus a missing `get_pl_hparams` migration,
corrected A/B parameter counts, an unstated capacity confound, a missing unique-fragment guard,
a pre-existing CUDA bfloat16 `index_add` crash, two documented degenerate head inputs, and a
per-layer host sync in `_efa_linear_attn`. Bucketed attention masking was verified exact for both
topologies, and `full` is cheaper only up to about 256 candidates. Validation: full suite plus
274 focused tests, Ruff check and format on changed files, and construction of both A/B arms.
A pre-rebuild full-suite run gave 92 failed, 2,873 passed; every failure was the then-stale
multi-cut Cython extension, all carrying `compute_ccs_multi_cut()` unexpected-keyword errors
for `max_ring_cut_steps` or `ring_open_mask`. Both extensions were rebuilt in place at
2026-09-13 18:12, and `mass_shatter_core` was force-rebuilt at 22:38 to a byte-identical
binary, confirming the 18:12 build already matched its source. All four extensions are now
current and the full suite passes: 2,975 passed, 1 skipped, 2 xfailed.
Next: decide whether to remove the degenerate `component_residual_head` and constant mask-count
feature, which would invalidate existing MASShatter checkpoints and so needs explicit
authorization. No training ran and no run was launched for this arm.

D4 oracle verification (2026-09-13): the
[subformula-filter finding](findings/subformula_filter_ft_qtof.md#saved-subformula-filtered-d3d4-comparison-verified-2026-09-13)
records reaggregation of the saved D3/D4 artifacts. MASShatter's roughly 94% coverage is strong
relative to that unmatched D4 reference; the remaining 6% is not established cut-scoring headroom.
No preprocessing, DAG generation, or model inference reran. Next: compare current MASShatter
support-oracle and predicted cosine on identical targets. Bond-cut-first remains the requested
diagnostic order, with training conditional on useful support or efficiency headroom.

MASShatter H-support ablation (measurement 2026-09-14; code clarification 2026-09-15): the
[ion/H channel plan](plans/masshatter_ion_h_channel_selection_plan.md#h-support-ablation-measurement-2026-09-14)
now records the matched 2026-09-08 gated / supervision-only / no-support arms read from W&B. No
trained arm exercises channel selection: retained channels equal valid channels in every arm
(about 8.5 per fragment against roughly 1.02 attributed), so the hard-selection stage has never
been measured. The soft support gate shows no spectral or weighted-precision benefit in either
data variant, and all dense cosine differences fall below one per-epoch standard deviation of a
single seed. The plan now records clone-initialized self-gating as an untested explanation for
the gate null, and clarifies that the six-run scope tests only H support (one valid ion mode,
at most nine CRF H channels). The support BCE does concentrate 95% of per-fragment channel mass into about 11%
fewer channels without moving cosine. Next: disable the gate in the H-support configs, verify the
`required_ion_h_top_k_95`/`_99` metric that reports identical means, and recover one cluster
checkpoint so the Phase 2 selection sweep can run; W&B holds no checkpoint artifacts for these
runs. Documentation only; no training, evaluation, or code change.

Joint selection objective (2026-09-13): the
[ion/H channel plan](plans/masshatter_ion_h_channel_selection_plan.md#joint-fragmentchannel-selection-loss-experiment-2026-09-13)
now records a proposed single selection loss that couples fragment proposal and
ion/H support probabilities while retaining the independent spectral intensity loss.
The current JSS + proposal loss + support BCE configuration is the control. Next:
check joint-loss gradients and the all-channels-open failure mode on synthetic cases,
then compare on matched validation support and intensity metrics. Documentation only;
no implementation, config change, or training run.

MASShatter SMARTS review (2026-09-13): the [set/literature review](findings/masshatter_smarts_review_2026_09_13.md)
confirms 26 rules, 19 legacy rules without enforced ionization guards, and a pyrimidine
gap in the carbamate family. Positive-ion oxygen-transfer and prazole-sulfone candidates
are supported literature leads, not measured coverage gains. Validation: 75 SMARTS tests
passed on CPU plus registry/matching probes. Next: measure existing rules' incremental
coverage on filtered validation peaks before adding narrowly mapped rules. No model edits
or training ran.

MASShatter proposal review (2026-09-13): the [proposal review](findings/masshatter_proposal_review_2026_09_13.md)
prioritizes support diagnostics before encoder expansion and separates established loss/routing
limitations from exploration, diversity, and mixture hypotheses. A fresh 155-run W&B query
and eight full validation histories support selection-only in a close same-seed historical
pair: +1.54 percentage points candidate weighted recall and +0.01261 cosine at epoch 142.
A separate larger-intensity-head pair gains 0.00897 cosine at epoch 98 despite slightly
lower coverage. Recent matched JSS/KL results are close and do not establish a KL advantage.
Next for the intensity stage: preserve selection-only and diagnose fragment-versus-H ranking
on fixed support before another head or loss comparison; see the review for provenance limits.
A live W&B check of three
recent merged U256/E1024 histories confirms only 5.5–7.0% reach U256 at their latest logged
validation epochs. Two existing unmerged pairs match in logged configuration except E and
labels: E4096 adds about 43–46 fragments but only 0.66–0.67 percentage points of candidate
weighted recall at common epochs. A subformula-filter query found 14 runs, all U256/E1024;
late-run coverage is about 94%, with no filtered E4096 counterpart found. No new tests or
experiments ran. Next: inspect remaining
missed intensity and support-oracle headroom before proposing another budget run. The proposal review
is present but untracked, as is its linked loss review; include both in the eventual documentation
change set with these indexes. No files were staged or committed.

EBM modeling proposal (2026-09-13): the user selected both approaches, with
[bond-cut EBM modeling first](plans/masshatter_bond_cut_ebm_plan.md), then spectrum reranking.
The [MASShatter program](plans/masshatter_energy_based_model_plan.md) records this ordering and
the spectrum setup delta. Next: select a checkpoint and validation subset, measure missed-intensity
and finite-mask-pool headroom, then evaluate pairwise cut energies against search-only, unary,
random, and mixture controls. Preserve probability contracts; the existing exact heap guarantee
does not extend to pairwise energies. No implementation or experiments started for this program.
The spectrum stage retains its separate shared protocol and null-adjusted gate.

Loss setup review (2026-09-13): the [loss review](findings/loss_setup_review_2026_09_13.md)
reproduces new FraGNNet vectorized-JSS correspondence/backward defects, ignored OOS
supervision under JSS, and invalid formula slots stealing support targets. MASShatter
probes identify weak JSS recovery gradients, correspondence/support ambiguity, proposal-loss
coverage limits, and a no-warmup gradient-routing confound. No implementation changes or
training ran; 195 focused tests passed on CPU. Next: fix the verified FraGNNet contracts,
then inspect MASShatter gradients/coverage before an isolated matched JSS/KL comparison.
The prior audit's three FraGNNet loss/explanation fixes and MASShatter precursor-pair
rejection are already present in the reviewed source; their earlier pending status below
is historical. The W&B screen report is missing from this checkout; its historical summary below
is not reverified evidence.

Correctness audit (2026-09-12): the [combined model audit](findings/fragnnet_masshatter_audit_2026_09_12.md)
records reproduced checkpoint-loading, HDF5 cutoff, FraGNNet loss/explanation, and shared
test-metric defects, plus silently ignored MASShatter precursor-pair supervision.
No fixes were applied. The existing local multi-cut extension is
stale; an isolated rebuild from current source yielded 2,876 passed, 21 skipped, and
2 expected failures for the full suite. Next: address checkpoint loading and test-metric
initialization, then the remaining counterexamples. Pre-existing SMARTS warmup edits
were preserved; CUDA and real-data scientific behavior remain unverified.

W&B review (2026-09-11; historical summary, source report unavailable here): the recorded screen
superseded older “no runs launched” statements. `docs/findings/wandb_screen_2026_09_11.md`
is absent from this checkout; recover the report before using these comparisons as evidence.
JSS warmup 0–4 leads the single-seed filtered screen at 0.80099 best validation binned
cosine; CE remains close and finished sooner. Support supervision helps early training;
an additional soft-gate gain is unresolved. GNN context completed, but its matched
heads-only comparison is missing from the inspected runs. SMARTS warmup failed at the
known construction guard. Next: matched-seed confirmation and the missing controls;
recover the missing report for metric definitions, run links, and provenance limitations.

SMARTS merged readiness (2026-09-09): the [readiness audit](findings/masshatter_code_audit.md#smarts-merged-run-readiness-2026-09-09)
passed 494 CPU tests (5 skipped) and a synthetic SMARTS + merged NCE + GNN-context
forward/backward probe. The current merged configs are blocked by their proposal-only
warmup, which explicitly rejects SMARTS. The requested no-warmup SMARTS config is now
prepared and linked in the audit; full model construction and intended-only resolved config
differences passed. Next: perform a real-data GPU smoke check. No runs launched.
The subsequently requested [SMARTS warmup config](../config/masshatter/merged/smarts/nist20_merged_full_metadata_context_gnn_heads_smarts_warmup_e0_15.yml)
is also prepared, with proposal-only epochs 0–15. The SMARTS/proposal-only-warmup
construction guard is now removed and validated (2026-09-11): see the
[guard-removal finding](findings/masshatter_code_audit.md#smarts-proposal-only-warmup-guard-removed-2026-09-11)
for the new PL-level gradient-isolation regression tests. Construction now succeeds.
Next: a real-data GPU smoke check before launching this arm.

Context-conditioning screen (2026-09-08): the user selected controlled experiments before a new
layer. The [conditioning plan](plans/conditioning_refactor_plan.md#current-controlled-screen-2026-09-08)
now selects the full-metadata, dual-CE MASShatter mask1024 config as the conditioning baseline:
the user reports that NCE + ACE + adduct + instrument + mode already helps in existing runs.
The prepared NCE-only variant is an optional diagnostic outside the next run queue. Two new
FraGNNet configs complete the existing full-metadata routing pair with output scalers disabled.
Follow-up: the user requested context at the merged MASShatter GNN input. The optional
`mass_shatter_context_in_gnn` path is now implemented, default off with legacy-checkpoint
migration. A matched full-metadata merged heads-only versus GNN+heads pair is prepared;
existing experiment configs are unchanged. Validation: 182 passed, 5 skipped, both merged
configs constructed, intended-only resolved diffs and Ruff passed. No runs launched.
The [pre-run design audit](findings/masshatter_context_design_audit.md) now confirms homogeneous
groups, but all selected rows are `[M+H]+`/FT/HCD. It found ACE saturation at 100 eV in 21.42% of
selected training groups and row-order-dependent ACE selection for same-NCE replicates. Actual
GINE-EFA forward/backward checks at epochs 15/16 were finite; no quality evaluation ran.
CE fixes are now implemented: the new pair uses ACE up to 300 eV and order-invariant weighted
CE aggregation, with both models consuming the weights and legacy behavior preserved. The
[audit follow-up](findings/masshatter_context_design_audit.md) records regression and source-data
checks. GNN initialization remains unchanged. Next: a real-data GPU smoke check before training.

FraGNNet formula-support crash follow-up (2026-09-07): W&B runs
[kvrlaxof](https://wandb.ai/frag-gnn/frag-gnn/runs/kvrlaxof) and
[71qtthvw](https://wandb.ai/frag-gnn/frag-gnn/runs/71qtthvw) both failed with infinite
loss on batch 0 before backward. Local reproduction shows that positive support targets for
unreferenced formula slots can overflow the binary loss: their logits retain the float32
minimum sentinel. The working-tree fix in `fragnnet_model.py` derives support validity from
valid joint-state contributors and zeros invalid support logits. Tensor shapes, parameters,
configs, and spectral intensity computation are unchanged. Regression coverage verifies both
attribution methods, finite gradients, max pooling, and empty support. Validation: 40 focused
tests passed on CPU; Ruff and whitespace checks passed. No full training or validation-set
evaluation ran. Next check: run a short cluster training smoke test with the patched package,
checking individual loss terms if failure persists. The exact failing remote batch was not
replayed, so this is a reproduced defect and candidate crash fix, not a confirmed remote cure.

For the next requested implementation or experiment, verify the relevant plan against code and
available artifacts. Link that task here when it spans sessions, and record its next check and
blockers in the plan. Do not duplicate the roadmap or copy its full backlog here.

## Local context caveat

The ignored `.claude/` entry point and six rule files now link to the shared guidance.
These local pointer changes are not included in Git diffs. It still contains a legacy architecture inventory;
its state directory had no files during this audit. Shared records above are the portable
entry points. Local permissions, commands, and skills were not migrated or audited for runtime
behavior. Their presence is not evidence that another checkout has them.

## Cluster submission knowledge

The [Fir/Vulcan submission guide](guides/submit_cluster_runs.md) now provides the canonical
procedure linked from INSTALL.md. Commands and profiles were checked against repository source;
no cluster connection, submission, or training validation was performed.

Connected-fragment beam review (2026-09-17): the
[reuse and equivalence finding](findings/connected_fragment_beam_reuse_and_equivalence_2026_09_17.md)
records that the beam objective is `log P(mask = ∂f)` minus the complement's intact term, so it
matches `cut_masks` up to replacing the charge head with a fixed prior, ring-reconnection terms
that vanish for induced-acyclic fragments, and exact-versus-approximate search. Nine of about
thirteen search mechanisms already have MASShatter counterparts; the dominant measured cost,
dense-reachability complement connectivity, is the one with an existing union-find replacement
on CPU and CUDA, which would also make the half-depth-plus-mirror growth free rather than paid
for. The finding also flags likely per-iteration host synchronization in the reachability loop.
The matching `cut_masks` inefficiency is mask-space budget: the unmerged arm enumerates 4096
masks for a 256 unique-fragment target, and that redundancy is already logged as
`unique_components_per_mask` and `duplicate_component_pathway_fraction`, so the efficiency
comparison can be answered from existing run artifacts without a new benchmark.
Source and algebra review only: no measurement, training, test run or code change. Next: the two
recorded gap checks (ring-bond probability histogram and implicit-prior versus charge-head split)
before any further beam optimization or scientific comparison.

Beam direction set (2026-09-17): keep the connected-fragment beam as a proposal-only
enumerator and restore a learned charge component, reconciling the prototype's two scores
toward its search form (the isolating mask's log probability) rather than its output form.
This removes the size/topology prior, keeps the chemically correct ring penalty through
boundary-cut count, makes complements score-symmetric again and so removes most of the
447 ms complement cost. The one structural item is emitting each mirror pair as two
components of one mask so the existing charge split and unique-component marginalization
apply unchanged. Recorded as PROPOSED in
[the FastGNN plan](plans/masshatter_fast_gnn_proposal.md#proposed-proposal-only-beam-with-learned-charge-2026-09-17);
not implemented. The gate has now been run: at matched fragment target on identical bond logits
the beam retains the smaller side of a cut about 5:1 (fraction above half 0.167 versus 0.590),
proposes about four times fewer large fragments (0.079 versus 0.319 at or above 0.75 of the
molecule) and 1.5 times more acyclic fragments. Raising `max_boundary_cuts` from 5 to 20 leaves
the distribution unchanged and raising the fragment target makes the bias worse, so neither the
cut cap nor the budget explains it. Bond logits were synthetic; the checkpoint-logit
confirmation was not obtained. Measurements and limits are in the same finding. A budget-exhaustion run also separated ranking from coverage: fragments at or above 0.75 of the
molecule saturate near a fixed count as the budget grows 9.4x, because anything above half the
molecule arrives only as a connected mirror, so a connected fragment with a disconnected
complement is unreachable. That is a coverage hole, not a ranking effect, and the same scoring
change plus union-find closes it. The coverage-per-budget gate then closed the question against the beam. Scoring every
reference fragment by the objective the redesign would adopt, the beam's captured probability
mass saturates near 0.70 across an eightfold budget increase while `cut_masks` reaches 0.983,
and the beam costs roughly twenty times more per fragment on CPU. Decomposition by fragment size
locates the ceiling: the beam is complete below half the molecule (1.000 and 0.999 by count) and
reaches only 0.109 of fragments above 0.75 of the molecule, the band holding 47.2% of all mass.
Large fragments from a single bond cut are reachable because their complement is connected;
multi-cut large fragments generally leave a disconnected remainder and are missing by
construction. This is architectural, not a scoring artifact, so the proposal-only redesign would
not lift it. Repeating the coverage scan with **trained** bond logits from checkpoint `zlv0bli8` (epoch 41)
reproduces every direction and revises the magnitude: captured mass is flat at 0.8695 across a
fourfold budget increase against 1.0000 for `cut_masks`, so the cost of the limit is about 13% of
probability mass rather than 30%. A confident trained model concentrates mass on single-cut
masks, whose large fragment has a connected complement and is reachable; the multi-cut large
fragments remain missing. The pre-fix and post-fix beams emit identical fragment sets in 18 of 24
configurations, differing by one or two fragments in 4,001 elsewhere, so the 447 ms complement
pass buys determinism rather than materially different candidates. All measurements are
reproducible through
[scripts/debug/analyze_proposal_fragment_coverage.py](../scripts/debug/analyze_proposal_fragment_coverage.py).
Next: a decision on whether to stop the connected-fragment line; the four prepared arms should
not be trained as configured in any case.
