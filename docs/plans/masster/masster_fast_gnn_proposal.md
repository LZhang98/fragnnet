# Optional FastGNN-style MASSter proposals

> **Status:** LANDED, EVIDENCE AGAINST CONTINUING · **Updated:** 2026-09-18 · The option is
> implemented and locally validated. Comparative candidate measurement is now complete and
> argues against continuing the line: on trained bond logits the beam's captured probability
> mass is flat at 0.8695 across a fourfold budget increase against 1.0000 for `cut_masks`,
> because it is complete below half the molecule and structurally limited above it. See the
> [reuse and equivalence finding](../../findings/masster/connected_fragment_beam_reuse_and_equivalence_2026_09_17.md).
> Spectral accuracy remains unmeasured and no training run was performed. Whether to mark this
> plan `STOPPED` is a pending decision for the user; in the meantime the four prepared arms and
> the three-hour launcher should **not** be submitted as configured.

## Objective and scope

Add a selectable connected-fragment beam proposal inspired by the workspace
[FastGNN prototype](../../../others/fast_gnn.py), retaining MASSter's encoder options, chemistry,
intensity head, sparse prediction contract, and spectrum-based proposal supervision.
The existing `cut_masks` formulation and CPU/CUDA backends remain the default.
This is an alternative scientific model, not an exact acceleration of mask marginalization.

## Configuration and semantics

[the model templates](../../../config) declares:

```yaml
masster_proposal_method: connected_fragment_beam  # default: cut_masks
masster_unique_fragment_mode: true
masster_unique_fragment_target: 256
masster_fast_gnn_beam_width: 256
masster_fast_gnn_max_boundary_cuts: 5
```

The prepared [dual-CE arm](../../../config/masster/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_cmf_dual_ce_fast_gnn_proposal.yml)
retains the existing fast control's encoder, intensity head, chemistry, loss weights,
optimizer and batch settings. It uses selection-only proposal supervision. No run was launched.

- Grow connected atom sets, ranking expansions by boundary-cut plus internal-intact log probability. Keep a bounded
  beam after atom-set deduplication; consider at most four times the beam width before
  deduplication. Stable score ties follow parent beam slot and local atom index, independent
  of batch padding. This approximation can underfill the beam when many paths duplicate.
- Grow to `ceil(atom_count / 2)` per molecule, adding connected complements at each round
  before output-budget pruning. Rank both orientations by their own scores. Using each
  molecule's own limit avoids dependence on unrelated batch padding. This differs from
  the prototype's batch-maximum growth depth.
- Apply the boundary-cut limit to emitted proper fragments, not intermediate growth states.
  A temporarily oversized boundary can shrink as additional atoms join the fragment.
- Deduplicate final atom sets and enforce a strict final target including one reserved
  intact precursor. Unlike the prototype, mirrors cannot make the count exceed the cap.
  The intact precursor is retained even for disconnected inputs; all proper fragments
  must be connected. Large fragments with disconnected complements may be missed.
- Use multiword 63-bit atom masks, no fixed atom-count ceiling. Dense batch adjacency and
  beam expansion still impose practical memory limits. Discrete search runs in FP32 on
  the model device with autograd disabled; live score computation preserves gradients.
- A fragment score is `sum_boundary log(p_e) + sum_internal log(1-p_e)`; bonds entirely
  outside the fragment are unrestricted. Beam growth, candidate ranking, complementary
  fragment ranking and live differentiable scoring use this same objective. Normalize
  over retained fragments for the support-coverage objective. No charge factor or
  latent-pathway marginal is used. Internal bonds must all remain intact, so this is
  not the full connected-component marginal on rings. Coupled intensity mode can also
  consume this alternative fragment prior.
- No extra encoder or FastGNN intensity/chemistry modules are introduced. The existing
  optional separate proposal encoder remains usable. Charge-head tensors remain in state
  dictionaries for compatibility, but the head is frozen and skipped for FastGNN proposals.
- Initial support: unique mode, one mixture, SMARTS off, shared-acquisition proposal reuse
  off, zero mask-probability floor. Unsupported combinations raise at construction.
  The fragment-probability floor and ion/H support options retain their normal meaning.
- `candidate_backend`, ranked-mask budget, CPU selection threads and forced-single-cut
  settings belong to `cut_masks`. FastGNN search follows the model device, independently
  of `candidate_backend`. PyTorch CPU and CUDA are supported; compilation remains disabled.
  Runner summaries record the resolved proposal method and `torch_device` backend for this option.
- Legacy templates and checkpoints explicitly migrate missing proposal keys to `cut_masks`.
  Fresh Lightning configurations require the declared keys. Explicit saved/overridden
  values retain precedence. Existing parameter tensor names and shapes are preserved.

## Integration and diagnostics

[Search implementation](../../../src/fragnnet/frag/masster_fast_gnn.py) returns the existing
sparse membership adapter with one identity mask/mechanism row per unique fragment. These
adapter rows are not a latent cut-mask pool: fragment events can overlap, so summing their scores after exponentiation is not retained
probability mass. Complementary fragments generally have different internal-bond scores.

[Model integration](../../../src/fragnnet/model/masster_model.py) bypasses charge/pathway
marginalization for this option and shares the remaining intensity and chemistry path.
The [Lightning wrapper](../../../src/fragnnet/pl_model/masster_pl.py) omits meaningless
mask/pathway diagnostics and logs `fast_gnn_fragment_count`, `fast_gnn_mean_boundary_cut_count`,
`fast_gnn_boundary_cut_fraction_*`, and `fast_gnn_search_batch_ms` instead. Existing private
model-to-loss adapter names remain for interface compatibility. Timers measure host wall time,
not synchronized CUDA stage durations; use synchronized whole-step benchmarks for speed claims.

## Validation and next check

Acceptance checks: exhaustive small-graph membership and boundary/internal-score references; normalized
score gradients; rings, disconnected and zero-bond inputs, multiword masks, budgets and batch
indexing; CPU/CUDA parity; real-model output normalization and proposal gradients; CE/JSS
loss and warmup routing; config rejection; strict new/legacy checkpoint loading; regression suites.

Final local validation:

- Full CPU-visible suite: **3,451 passed, 96 skipped, 2 xfailed**, 90.38 seconds.
  Log: `/tmp/masshatter_fast_gnn_full_suite.log` (local artifact).
- New-option suite: **29 passed, 3 skipped** inside the sandbox. Separate GPU execution:
  **3 passed**, covering candidate/gradient parity and real-model CE/JSS backward through
  the staged-training transition. Counts overlap the full suite; do not sum them.
- Two existing runner/resume logging regressions passed after adding resolved-proposal logging.
- Ruff and Git whitespace checks passed; documentation relative links checked.
- Real-data GPU smoke completed on RTX 2070 SUPER, PyTorch 2.4.1+cu121, seed 397623,
  using the prepared config, subsample 128, batch 16, one fixed batch, one warmup and two
  measured steps per monitoring variant. No optimizer updates or W&B logging. Finite losses;
  peak allocated GPU memory below 445 MB. This short, initially concurrent smoke is not
  comparative throughput evidence. Exact input IDs, resolved config and measurements are in
  `/tmp/masshatter_fast_gnn_smoke.json`; log: `/tmp/masshatter_fast_gnn_smoke.log`.

Reproduce the smoke from the repository root with the configured interpreter:

```bash
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python scripts/debug/bench/benchmark_masster_training.py \
  --config config/masster/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_cmf_dual_ce_fast_gnn_proposal.yml \
  --device cuda --subsample-size 128 --batch-size 16 --batches 1 --warmup 1 --steps 2 \
  --output /tmp/masshatter_fast_gnn_smoke.json
```

The CE integration checks use the vectorized path selected by the prepared arm. A preliminary
non-vectorized Gaussian-CE harness hit the existing `LOG_ZERO(torch.float64)` limitation with
model mass tensors; that shared loss path was not changed in this task.

No optimization/training run or validation-set quality evaluation was performed. Next scientific
check: compare the prepared arm to the fast control at matched candidate count and runtime,
reporting coverage, spectral accuracy and peak memory. The retained intensity head remains a
possible runtime bottleneck; this option does not claim full FastGNN-model speed.

## Proposal method naming (2026-09-17)

The canonical selector is now `masster_proposal_method`, with
`cut_masks` (default) and `connected_fragment_beam`. Config files, CLI overrides,
saved-run configuration and Lightning/repository checkpoint loading convert the original
`masster_proposal_type` key and its `cut_masks`/`fast_gnn` values. If both keys occur
in the same mapping, the canonical key wins; normal template/custom/CLI precedence is
preserved. Saved hyperparameters use the canonical name. Beam settings, metric names and
parameter tensors are unchanged.

Validation: 355 focused tests passed, 8 GPU tests skipped in the sandbox. These cover
old-name YAML/CLI loading, strict old/new checkpoint roundtrips and canonical override
precedence, plus model, loss, backend compatibility and runner-config regressions.
Ruff and whitespace checks passed. No training or scientific behavior changed.


## Fragment probability alignment (2026-09-17)

The updated workspace prototype scores boundary cuts and internal intact bonds, leaving
outside bonds unrestricted. The connected-fragment option now uses this formulation for
both search and live scores, including independently rescored complements. Growth adds
`sum_incident log(p)` plus `sum_edges_to_fragment [log(1-p) - 2 log(p)]`.
The prototype itself still searches by summed boundary logits; this implementation aligns
search with the returned score. Finite beam pruning and complement generation remain
approximate: ranking consistency does not guarantee globally optimal fragment selection.

The final candidate cap and reserved precursor remain unchanged. `cut_masks` is unchanged.
Existing connected-fragment checkpoints still load, but predictions and resumed-training
semantics change; comparisons with the prior formulation need the prior code revision.
Tests cover outside-bond independence, normalized gradients, narrow-beam selection and
exhaustive chain rankings. The multiword complement test now uses a nonbinding cap because
large complements no longer inherit their small partners' scores.

Validation for the probability alignment: **136 passed, 8 skipped** across the proposal
and MASSter model suites; GPU checks skipped because CUDA was unavailable in the
sandbox. Ruff and `git diff --check` passed. Log:
`/tmp/connected_fragment_probability_tests.log`. No training, throughput benchmark or
validation-set quality evaluation ran for this change.

## Merged three-hour screen (2026-09-17)

Three configs under [merged/unique_fragment](../../../config/masster/merged/unique_fragment) use the same MA/MI
split, four adducts, FT/QTOF instruments, model, loss, seed and batch settings
as the unmerged source. These merged configs use NCE only (`nce: true`, `ace: false`)
and explicitly disable CMF.
They enable tolerance-consensus merging with retained collision
energies, common-overlap tolerance, per-spectrum normalization and equal CE weighting.

| Config suffix after `nist20_ma_mi_merged_fast_gnn_proposal` | Final fragments | Beam width |
| --- | ---: | ---: |
| `_fragments128.yml` | 128 | 128 |
| `.yml` | 256 | 256 |
| `_fragments512.yml` | 512 | 512 |

Batch size is 128 with accumulation 2 in all arms. These compare combined search/output
budgets, not isolated beam-width effects. The 512-fragment arm's memory fit is unverified.

On the Vulcan login node, generate the three jobs using:

```bash
CC_EMAIL=your_email@example.org bash scripts/manual_slurm/vulcan/prepare_masster_merged_fast_gnn_3h.sh
```

The [generation-only launcher](../../../scripts/manual_slurm/vulcan/prepare_masster_merged_fast_gnn_3h.sh)
uses `python`, the `vulcan:l40s` profile, account `${CC_ACCOUNT:-aip-rgreiner}` and
`--time 3:00:00` for each independent job. It prints three `sbatch` commands after
validating generated shell syntax. `JOB_PREFIX` can distinguish repeated screens.
Three hours is scheduler wall time, including environment setup; it is not a promise
of completed training or a graceful trainer stop. Epoch limits remain those of the source.
W&B is online and the normal best/last checkpoint settings remain enabled; follow the
[cluster guide](../../guides/submit_cluster_runs.md) for staging and checkpoint persistence.

Validation: all three configs load through the repository loader, pass sampler validation,
and differ only in run identity, fragment cap and beam width. Checkpoint settings checked;
launcher shell syntax passed. No cluster scripts were generated locally, jobs submitted,
or training/quality evaluations run. Next: generate on Vulcan, review staging, and compare
coverage, completed steps, spectral accuracy and memory at the three-hour budget.

## Connected-fragment audit (2026-09-17)

Audited the working tree at `dca3d0f9`, including existing uncommitted backend changes.
Two correctness issues were reproduced; both are now fixed in the working tree as described
in the resolution below. The initial audit did not edit production code.

1. **P2: tied beam selection depends on unrelated batch padding.** In
   [the search](../../../src/fragnnet/frag/masster_fast_gnn.py), the expansion `topk`
   operates before lexicographic deduplication and has no stable secondary key. With
   a five-atom cycle `(0,1),(1,2),(2,3),(3,4),(4,0)`, all-zero logits, beam width 1,
   fragment cap 8 and boundary limit 3, searching alone retains `{0,4}`. Batching
   the same graph with an unrelated six-atom zero-logit chain replaces that fragment
   with `{1,2}`. All other returned sets agree. This contradicts the stated batch-padding
   independence; changing inference batch composition can change candidate support.
   Resolve ties by a padding-independent atom-set/order key before truncation. The
   existing padding test uses unequal logits and misses this case.
2. **P2: pruning before complement scoring loses better candidates.** Both per-round
   recording and the global pre-complement cap rank only the grown fragment's score.
   Complement scores can rank very differently. For the chain `0-1-2-3-4-5`, ordered
   bond logits `[2,6,-2,-4,-1.5]`, beam width 128, cap 4 and boundary limit 3, the
   result is `{0}`, `{1}`, `{5}`, and the precursor. The higher-scoring connected
   complement `{2,3,4,5}` is omitted because its seed `{0,1}` was discarded first.
   Its log score is approximately -0.349 versus -1.701 for retained `{5}`. Raising
   the cap to 128 recovers it without changing the beam. This is independent of
   finite-beam underfill and the documented disconnected-complement limitation:
   both this fragment and its seed are connected. Score eligible complements before
   applying output-budget pruning, or retain seeds using both orientations' scores.

Validation with the configured `FRAGNNET-GPU` interpreter: existing proposal and model
suites **136 passed, 8 skipped**; CUDA was unavailable. Two additional targeted regression
probes **failed as expected**, one per finding. Local reproducer:
`/tmp/test_connected_fragment_beam_audit.py` (ephemeral; exact graph inputs and expected
behavior are recorded above). Run with `python -m pytest /tmp/test_connected_fragment_beam_audit.py -q`
using the configured interpreter. No full suite, training, throughput or quality evaluation ran.
The resolution below supersedes this initial failing-probe status.

### Selection fixes and regression coverage

The search now uses stable descending sorting before expansion truncation. Equal scores
follow parent beam slot and local atom index, so unrelated padding cannot change which tied
expansions enter the beam. Every recorded growth round emits and scores connected nonempty
complements before deduplication and output-budget pruning. Keeping the round's top K unique
fragments then preserves its possible global top-K contributors. The precursor retains its
reserved final slot. Code comments explain both ordering requirements.

Complement reachability checks use chunks of at most 256 seeds per molecule to bound dense
temporary storage; chunk boundaries do not discard candidates. This is a workspace limit,
not a new scientific budget. Existing finite-beam and disconnected-complement limitations
remain. Config keys, checkpoint parameter shapes and `cut_masks` behavior are unchanged;
affected connected-fragment checkpoints can produce different candidates with the fixes.

Permanent regressions in [the proposal tests](../../../tests/test_masster_fast_gnn.py) cover:

- Tied beams with several padding sizes, including multiword masks; CUDA cases are included.
- The original chain counterexample plus a stronger first logit that independently exposes
  premature pruning within a growth round.
- Exhaustive reachable-fragment score rankings on chains, cycles and branching graphs.
- A mixed batch whose dense graph produces more than 256 seeds in one growth round.

Validation: proposal tests **81 passed, 6 skipped**; model/backend-compatibility tests
**139 passed, 5 skipped**. Total **220 passed, 11 skipped**; CUDA was unavailable. The two
original regressions failed before the fix and pass afterwards. Ruff and changed-file whitespace
checks passed. No full suite, training or validation-set quality evaluation ran.

A small single-thread CPU candidate-only probe compared `dca3d0f9` search against the fixed
working-tree implementation on chains with extra five-atom ring-closing bonds, random logits
(seed 17), boundary limit 5, one CPU thread, and three calls per case (reported median):

| Atoms/molecule | Batch | Beam = target | Before | Fixed |
| --- | --- | --- | --- | --- |
| 20 | 4 | 64 | 20.8 ms | 26.0 ms |
| 40 | 8 | 256 | 121.5 ms | 169.8 ms |

These probes expose the extra sorting/connectivity cost; they do not measure CUDA or training
throughput. The local script is `/tmp/benchmark_connected_beam_fix.py` (ephemeral).
The [subsequent real-graph speed audit](../../findings/masster/connected_fragment_beam_speed_2026_09_17.md)
measures GPU candidate time 238.7 → 732.1 ms and forward/backward time 744.4 → 1,252.3 ms at
batch 128 / beam 256. Complement connectivity dominates; stable sorting is a smaller cost.
All six previously skipped CUDA regressions passed. Next: reduce unnecessary complement
connectivity work while preserving exact selection, then measure L40S throughput and coverage
before the scientific comparison. Earlier performance measurements predate the fixes.

## Boundary-cut budget correction (2026-09-17)

`masster_fast_gnn_max_boundary_cuts: 5` was the binding candidate budget in all four
prepared arms, not `masster_unique_fragment_target`. Measured on a synthetic batch of
126 NIST-like precursors (RTX 2070 SUPER, random bond logits, candidate construction only),
raising the limit is nearly free in search time and restores budget fill:

| target = beam | cuts = 3 | cuts = 5 |
| --- | --- | --- |
| 128 | 121 frag/mol (95% fill) | 128 frag/mol (100%) |
| 256 | 161 frag/mol (63%) | 251 frag/mol (98%) |
| 512 | 203 frag/mol (40%) | 396 frag/mol (77%) |

Search time changed by about 3% and peak allocation by about 40%. At the previous setting the
merged 128/256/512 screen varied a budget that was not binding above 128, so it would have
measured the boundary-cut cap rather than the fragment budget. `cut_masks` delivers 245-256
unique fragments per molecule under the same conditions, so the previous unmerged arm also ran
at roughly half the control's candidate count.

All four connected-fragment arms now set `masster_fast_gnn_max_boundary_cuts: 5`. The
`config/*_template.yml` default and the legacy checkpoint/config migration defaults stay at 3 so
existing connected-fragment checkpoints keep their trained search behavior.

Candidate-construction throughput was also compared against the CUDA `cut_masks` backend on the
same batches, which the earlier CPU-only comparison in this plan does not cover. The beam is
launch-bound and scales well in batch size but linearly in the batch's largest molecule: it wins
at 22-29 maximum atoms (41 vs 59 ms at batch 126; 57 vs 155 ms at batch 510) and loses at 49
maximum atoms (104 vs 76 ms at batch 126; 235 vs 168 ms at batch 510). Confirm the NIST20 unmerged
atom-count distribution before claiming a throughput win over the CUDA backend.

Validation: all four configs load through the repository loader with the expected resolved
method, beam width, boundary limit and fragment target; `tests/test_masster_fast_gnn.py`
**42 passed**. These are synthetic-logit construction measurements on one consumer GPU, not
whole-step training throughput, and no training or quality evaluation ran. These measurements
predate the selection fixes in the preceding audit section and need a new baseline.

## Real-precursor throughput measurement (2026-09-17)

The [speed audit](../../findings/masster/connected_fragment_beam_speed_2026_09_17.md) measures this
option before and after the complement-scoring fix. This section adds the missing baseline:
the same construction against `cut_masks` on real sampled precursors.

The boundary-cut section above measured synthetic batches of 13-49 heavy atoms. That
understates cost: `mol_df.pkl` for NIST20 has 35,025 molecules at mean 24.3 heavy atoms but
p90 44, p99 82 and max 186, and search cost is set by the batch maximum, not the mean. Batches
of 128 sampled without replacement have a median maximum of 94 heavy atoms (p10 70, p90 121);
at 256 and 512 the medians are 104 and 114.

Measured on five real sampled batches (batch 128, fragment target and beam 256, boundary limit 5,
RTX 2070 SUPER, random bond logits, candidate construction only), median over batches:

| backend | median time | relative |
| --- | --- | --- |
| connected-fragment beam | 1,024 ms | 1.0x |
| `cut_masks`, `candidate_backend: cuda` | 70 ms | beam is **14x slower** |
| `cut_masks`, `candidate_backend: cpu` | 157 ms | beam is **6x slower** |

Per-batch beam times ranged 846-2,144 ms and tracked the batch maximum atom count. Unique
fragments per molecule were comparable (beam 218-230, `cut_masks` 234-242), so this is not a
candidate-count difference.

Profiling one batch at 79 maximum heavy atoms: 706 of 922 ms (77%) is `_connected_complements`,
called once per growth round (40 calls, 184,294 seed rows), each running a dense reachability
loop of up to `max_atoms` iterations that does not converge early because its break tests the
whole chunk. Growth rounds are `ceil(max_atoms / 2)`, so complement connectivity costs
`O(max_atoms^2 / 2)` sequential batched matrix products: about 3,174 launches here.

This cost arrived with the fix for the second audit finding, which correctly moved complement
scoring ahead of output pruning. The complement *score* does not need reachability: it is a
gather over the padded edge arrays. Only the connectivity filter does. Ranking by the exact
complement score first and running reachability only on each molecule's surviving candidates
would preserve the fix and cut roughly 5-6x of the reachability rows; the remaining growth loop
alone is still 216 ms, above the whole CUDA `cut_masks` time.

Correction to the preceding section: the claim that raising the boundary limit costs about 3%
holds only for the small synthetic batches it was measured on. On real batches, boundary limit 3
takes 501 ms and limit 5 takes 954 ms at the same batch. The limit-5 setting is kept because the
candidate-parity problem it fixes is real and independent of throughput, but it is not free.

Conclusion: the connected-fragment beam is not currently a throughput improvement on NIST20 at
any measured batch size, and the gap is large enough that it cannot be closed by tuning. Treat it
as an alternative proposal formulation to screen on accuracy, and remove the speed framing from
this plan unless the complement and growth loops are restructured and re-measured. No training,
whole-step throughput or quality evaluation ran for this measurement.

## Proposed: proposal-only beam with learned charge (2026-09-17)

> **Status:** PROPOSED, EVIDENCE AGAINST · Design recorded from review, then measured on
> 2026-09-17 and not supported. Do not implement from this section without reading the
> [coverage ceiling measurement](../../findings/masster/connected_fragment_beam_reuse_and_equivalence_2026_09_17.md#coverage-per-unit-budget-and-the-architectural-mass-ceiling):
> the beam's captured probability mass saturates near 0.70 at any budget against 0.983 for
> `cut_masks`, because it is complete below half the molecule and structurally blind above it,
> where 64% of the mass lies. That ceiling is architectural, so the scoring and charge changes
> below would not lift it. The redesign remains correct as a description of what is wrong with
> the current score; it is no longer a recommended build.

Direction set by the user after the
[reuse and equivalence finding](../../findings/masster/connected_fragment_beam_reuse_and_equivalence_2026_09_17.md):
keep the connected-fragment beam as a **candidate enumerator**, and restore a learned charge
component instead of the implicit prior that the current score carries.

### Objective

Reconcile the prototype's two scores toward its **search** form rather than its output form.
The beam then enumerates masks of the shape "boundary of one connected atom set" — the
chemically meaningful subclass — and MASSter's existing scoring path assigns fragment
probability, as it already does for `cut_masks`.

### Scoring change

Replace `B + I_f` with the isolating mask's log probability `B + I_f + I_f̄`, which is what
`_score_mask_log_probabilities` already computes from boundary bonds and a per-molecule
baseline. Consequences, all recorded in the linked finding:

- The size and topology prior favouring small acyclic fragments disappears. Ring cleavage
  remains penalized through the correct channel — two or more boundary cuts, each costing
  `log p` — rather than being double-charged through internal-bond count.
- The chemically reasonable part of the prior, fragile boundary with stable interior, is
  retained unchanged.
- A fragment and its complement share the mask and therefore the score, so mirrors become free
  again and the audit's premature-pruning finding cannot arise. Most of the measured 447 ms
  complement cost is removed with it.
- Internal-bond index arrays (`internal_id`, `internal_owner`) are no longer needed; scoring
  requires the boundary alone.

**Measured caveat (2026-09-17), which was not anticipated when this section was written.**
Running the original prototype, which already ranks by the mask log probability, confirms the
first bullet — its emitted distribution is near-symmetric, 0.485 above half against the installed
beam's 0.174 and `cut_masks`' 0.500 — but it captures **less** probability mass, 0.757 against
0.869. Under mask scoring a fragment and its complement score identically, so the search spends
two budget slots per mask and covers roughly half as many distinct masks. **The scoring change
alone therefore trades one defect for another.** Recovering both requires emitting every
component of a mask from one enumeration slot, which is `cut_masks`' assembly, at which point the
method is `cut_masks` with a beam heuristic in place of exact top-K enumeration.

### Charge component

Unfreeze `AtomChargeAffinityHead`, currently `requires_grad_(False)` for this proposal method,
and route fragments through the existing `latent_pathway_charge_log_probabilities` with the
configured `charge_pooling`. The learned head then separates a fragment from its complement on
proton affinity and basic sites, replacing the current implicit "charge stays on the smaller
fragment" prior. Existing checkpoint tensor names and shapes are unaffected; the head is
already present in state dictionaries.

### Adapter change

The one structural item. Today every candidate adapter mapping is an identity, one row per
fragment. Emit each mirror pair as **two components of one mask** instead:
`mask_log_probabilities` holds one value per mask, and both siblings reference it through
`component_mask_indices`. This is exactly the shape `cut_masks` produces, so the charge split,
`component_unique_indices` deduplication and unique-component marginalization all work
unchanged. Where two distinct masks yield the same atom set, the existing
`_segment_logsumexp` then recovers ring-reconnection pathways that the current formulation
drops.

### Open questions and gates

1. Whether the beam should enumerate more than one mask per fragment. One boundary mask per
   atom set is the default; additional masks are what would narrow the ring-reconnection gap.
   Note this is a different question from emitting **all** components of one cut set, which the
   2026-09-17 measurement makes the higher priority: fragments above half the molecule saturate
   near a fixed count regardless of budget, because they arrive only as connected mirrors, so a
   connected fragment whose complement is disconnected is unreachable today. Union-find returns
   every component of a cut set and closes that hole with the same edit.
2. Whether `proposal_selection_only` remains the right supervision once fragment probability
   comes from mask probability plus a trained charge head.
3. Matched comparison against `cut_masks` at equal fragment target: coverage, spectral
   accuracy, candidate time and peak memory. The fragment size and ring-count half of this gate
   was run on 2026-09-17 and confirms the prior, with both obvious confounds excluded; see the
   [measured section](../../findings/masster/connected_fragment_beam_reuse_and_equivalence_2026_09_17.md#measured-emitted-fragment-distributions-2026-09-17).
   Synthetic logits were used, so a checkpoint-logit confirmation remains open.
4. Existing connected-fragment checkpoints become diagnostic-only under the scoring change, as
   the earlier fragment-probability alignment already required.

No code was changed and nothing was measured for this proposal.

## Candidate coverage logging on the prepared arms (2026-09-18)

The four prepared `connected_fragment_beam` configs now set
`masster_candidate_coverage_metrics: true`: the three
[merged screen arms](../../../config/masster/merged/unique_fragment) and the
[unmerged dual-CE arm](../../../config/masster/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_cmf_dual_ce_fast_gnn_proposal.yml).
They were written when `masster_channel_diagnostics` still implied coverage and carry
`channel_diagnostics: false`, so as configured they logged no candidate ceiling at all -
the one quantity the 2026-09-17 measurement puts in question. This logs `candidate_wrecall`
every training step and `candidate_recall` at validation, at one match-mask pass per spectrum.
`masster_candidate_cut_diagnostics` stays off; set it to `true` for the per-boundary-cut
buckets, which add one pass per bucket per spectrum during validation.

Nothing else changed: no model, loss, search or supervision code, and no other config key.
The coverage plumbing itself was already proposal-method agnostic - `candidate_mzs_by_cut`,
`candidate_batch_idxs_by_cut` and `candidate_cut_counts` come from the shared post-proposal
path, and the `fast_gnn_*` rename in `masster_pl.py` only drops mask/pathway diagnostics -
so this is a configuration change, not an implementation gap. Two tests in
[test_masster_fast_gnn.py](../../../tests/test_masster_fast_gnn.py) now pin that:
an end-to-end beam forward whose targets are drawn from its own candidates, asserting
`candidate_wrecall` and `candidate_recall` are exactly 1.0 and survive the rename filter, and a
parametrized check that all four prepared arms resolve with the coverage names in
`loss_names`/`metric_names`.

Validation: `tests/test_masster_fast_gnn.py`, `tests/test_masster_pl.py` and
`tests/test_masster_model.py` pass (314 passed, CUDA-only cases skipped in this sandbox);
Ruff check and `git diff --check` pass. Ruff format reports pre-existing reformatting in this
test file, outside the added block, and was deliberately not applied. No training run,
throughput benchmark or validation-set evaluation was performed, and the standing
recommendation not to submit these arms as a scientific bet is unchanged - the coverage curve
is now available if they are run anyway.

## Batched candidate coverage and the 256-arm 3h screen (2026-09-18)

Enabling coverage on the beam arms exposed `_candidate_weighted_recall` in
[masster_pl.py](../../../src/fragnnet/pl_model/masster_pl.py) as the last
per-spectrum loop on the every-step path. It is now batched through
`matched_peak_candidate_pairs`, the same rewrite `_proposal_coverage_loss` received, so the
coverage metric no longer costs a host-device synchronization per spectrum.

Measured on an RTX 2070 SUPER at `train_batch_size: 128`, merged-scale peak counts
(p50 25, max 1168) and ~2,500 candidates per spectrum:

| | GPU | CPU |
| --- | ---: | ---: |
| Per-spectrum loop | 45.0 ms/batch | 58.1 ms/batch |
| Batched | 1.3 ms/batch | 33.4 ms/batch |

CPU being within 30% of GPU in the loop form is the synchronization signature, not
arithmetic cost. The batched form agrees with the loop to within **2.98e-08 absolute**
across 288 randomized cases (CPU and CUDA, sorted and shuffled batch indices, spectra with
no candidates, scalar relative/absolute and per-instrument tolerances). It is not bitwise
identical: the index-add reduction reassociates the float32 sums. Two regression tests in
[test_masster_pl.py](../../../tests/test_masster_pl.py) pin it against a per-spectrum
reference and against the empty-spectrum error.

### Prepared run: the beam on the lead's split, not MA/MI

Every config listed above sits on `data/split/nist20_ma_mi_inchikey` with four adducts and a
`cross_entropy` objective, so none of them is comparable to the merged JSS arm that is the
current lead (W&B `8v0i0fq7`, best logged `val_cos_sim_0.01_epoch/mean` 0.800989 at epoch 140).
A beam result on the MA/MI split would not answer whether the beam costs anything against the
model actually being developed.

[nist20_merged_independent_jss_warmup_e0_4_dense_subformula_fast_gnn_proposal.yml](../../../config/masster/merged/warmup_duration/nist20_merged_independent_jss_warmup_e0_4_dense_subformula_fast_gnn_proposal.yml)
is the beam sibling of the `_fast` baseline. Resolved-key diff, pinned by a test:

| Key | `_fast` baseline | Beam arm |
| --- | --- | --- |
| `masster_proposal_method` | `cut_masks` (template default) | `connected_fragment_beam` |
| `masster_fast_gnn_beam_width` | - | 256 |
| `masster_fast_gnn_max_boundary_cuts` | - | 5 |
| `masster_candidate_coverage_metrics` | `False` | `True` |
| `masster_candidate_backend` | `cuda` | removed (cut_masks-only) |
| `wandb_name`, `wandb_tags` | baseline identity | arm identity |

Split, subformula filter, `tolerance_consensus` merging, encoder, intensity head,
`jss_distance` objective, warmup boundary, seed, LR, batch sizes and epoch budget are
byte-identical. Beam width equals `masster_unique_fragment_target`, so search and output
budgets agree.

**This is not a clean ablation of the search alone.** The beam bypasses charge/pathway
marginalization, so the arm also drops the learned charge split and freezes the charge head;
`masster_charge_pooling: logsumexp` is inert in it. A deficit against the baseline cannot
be attributed to the proposal search by itself.

Generate **on the Vulcan login node** from the cluster checkout; the generator bakes in
`FRAGNNET_DP` and a resolved W&B key, so a locally generated script is not portable:

```bash
python scripts/setup_job_cc.py \
  --frag_gnn_dp . \
  --python_script_fp scripts/run_pl_model_fit.py \
  --job_name msh_merged_jss_cfb_256_3h \
  --gpu_profile vulcan:l40s \
  --cc_account aip-rgreiner \
  --cc_email fw4@ualberta.ca \
  --time 3:00:00 \
  --script_kwargs w=online c=config/masster/merged/warmup_duration/nist20_merged_independent_jss_warmup_e0_4_dense_subformula_fast_gnn_proposal.yml \
  --no_submit=True
bash -n slurm_scripts/msh_merged_jss_cfb_256_3h.sh
sbatch slurm_scripts/msh_merged_jss_cfb_256_3h.sh
```

Read `candidate_wrecall` (train) and `candidate_recall` (val) against the 0.8695 captured-mass
ceiling the [2026-09-17 finding](../../findings/masster/connected_fragment_beam_reuse_and_equivalence_2026_09_17.md)
measured on trained `cut_masks` logits, and read `val_cos_sim_0.01_epoch/mean` against
`8v0i0fq7` at matched wall clock and matched epoch. Three hours is scheduler wall time against
`max_epochs: 200`, so this truncates well short of the baseline's epoch 140 best; it is a
wall-clock screen, not a completed run, and a truncated curve is not convergence.

The MA/MI three-budget launcher
[prepare_masster_merged_fast_gnn_3h.sh](../../../scripts/manual_slurm/vulcan/prepare_masster_merged_fast_gnn_3h.sh)
is unchanged and untouched by this arm.

No job was submitted from this session: the workstation has no scheduler.
