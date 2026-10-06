# Retained-fragment encoding and candidate-graph comparison

> **Status:** DEFERRED · **Updated:** 2026-09-29
> The post-cut information gate lowers the case for a retained-fragment GNN. The user then had the bounded retained-fragment branch built and screened locally (`fc1`, 2026-09-29): not positive (see the addendum under "Current state"). The candidate branch is still unbuilt.

## Objective and evidence

Test whether separating within-fragment connectivity from between-candidate comparison improves
MASSter's ranking on unseen molecules. Keep candidate generation, molecular context, ion/H
channels, peak aggregation, and the spectral objective unchanged in the first experiment.

The [reopening audit](../../findings/experiment_reopening_audit_2026_09_28.md) distinguishes
this from tested cut-descriptor and FluxMPNN nulls. The
[base-peak finding](../../findings/masster/masster_i1_base_peak_errors_2026_09_28.md) identifies
nested and alternative-fragment decisions as the largest populations of errors, but finds a
broad transfer deficit, not evidence that retained connectivity is missing from the input.
The updated finding reports complementary pairs as 1.3% of misses; the earlier audit's 1.4%
is an older read and does not change the prioritization.

The newer [post-cut information gate](../../findings/masster/masster_i1_postcut_information_gate_2026_09_29.md)
is the controlling evidence for the retained-fragment branch. A cross-fitted correction on
held-out molecules tested 80 post-cut descriptors, including retained connectivity, cut-site
chemistry, bond classes, and composition. It added −0.0008 (MLP), −0.0004 (linear), or +0.0010
(regularized MLP) candidate-space cosine over each fit's context-only baseline; against the best
context baseline no feature family added more than +0.0006. Top-candidate agreement stayed about
0.614. A learned graph might express functions those descriptors miss, but there is no measured
signal supporting it as the next expensive experiment. The correction is a downstream probe on
one checkpoint, so it does not prove an end-to-end branch must fail.

**Decision after the gate:** defer the retained-fragment branch and combined R/RC training arms.
First read the already-built full-attention `fa1` control against `i1`. Consider candidate
relations only if a separate held-out test shows they add information beyond full attention,
including nested and sibling decisions. Reopen the atom branch only with a concrete reason to
expect useful signal beyond the tested descriptors, and a cheap edge-vs-pool pilot. Do not treat
this design as an approved training campaign.

The [existing candidate-graph proposal](../../history/masster_memorization_route_plan_log.md#candidate-graph-allocator-arms-fa0-cg1-and-cg2-proposed-2026-09-27-not-built)
owns the earlier `fa0`/`cg1`/`cg2` design. This document specifies a new controlled extension
on `i1`, not authorization to submit those older `c0` configs unchanged.

## Conditional architecture if the gate is reopened

```text
precursor graph -> existing precursor encoder -> existing candidate tokens t_f

candidate retained atoms, internal bonds, cut-site markers
    -> shared small atom GNN -> pooling -> structural residual r_f

u_f = t_f + W r_f
    -> candidate relation attention -> existing intensity and ion/H heads
    -> existing mass aggregation -> predicted spectrum
```

The atom branch computes independently for each candidate. It never receives other candidates
or observed intensities. The candidate branch receives candidate tokens and target-independent
relations, but does not propagate atom states or conserved intensity. Normal spectral
normalization remains at the output. Complementarity never means mutual exclusion: a spectrum
aggregates many fragmentation events.

Both modules can overlap statistically. They are complementary only if measured held-out gains
justify the combination; the architectural distinction alone is insufficient.

## Retained-fragment branch: bounded first version

- Nodes are retained atoms, edges are precursor bonds with both endpoints retained. Start with
  one message-passing layer at width 64; these are proposed pilot settings, not measured optima.
- Use a shared projection of raw atom/bond features, plus per-candidate retained degree and
  number of boundary bonds incident on each atom. Do not seed this branch from deep
  precursor-contextualized embeddings in the first test. Existing tokens retain that context.
- Keep precursor aromaticity, formal charge, and other inherited descriptors explicitly labelled
  as precursor attributes. Cutting an aromatic ring does not yield a sanitized charged product.
  No recomputed protonation, valence completion, or rearranged bond order is claimed.
- Pool over all retained atoms and separately over marked cut-site atoms. Record boundary count;
  the zero-cut precursor has a defined empty-boundary vector. A pooled control gets the same
  features, markers, reductions, and comparable parameters, but no retained-edge messages.
- Project the result into the existing token width as an additive residual immediately after
  `input_projection` and before token dropout/contextualization. Zero-initialize only the final
  residual projection, not both a gate and projection. At initialization the baseline is
  preserved; internal branch gradients begin once the output projection moves from zero.
- Preserve baseline RNG initialization when adding modules. Default-off behavior must remain
  compatible with historical checkpoints; no new keys are introduced by this design document.

**Expressivity limit:** one layer sees one-hop retained neighborhoods. It does not reconstruct
arbitrary ring topology or long-range conjugation. With additive pooling it may mostly encode
local edge statistics; require nonlinear updates and a matched pooled control. Deeper layers
are a later hypothesis, not an automatic response to a null.

## Candidate branch: reuse graph attention, isolate the relations

The current induced allocator already exchanges candidate information. Implement the first
candidate GNN as relation-masked **full attention**, reusing the existing full-attention block.
Replace the induced interaction layers; do not stack a second allocator on top. Keep layer count,
head count, width, and normalization fixed between the full and masked variants.

Initial graph follows the existing bounded relation proposal:

1. Every candidate has a self-loop.
2. Direct containment in both directions: A is a strict subset of B with no retained candidate
   strictly between. Label the two directions separately.
3. Exact complements: disjoint atom sets whose union is the precursor.
4. The precursor can receive but does not send to other candidates; it keeps its self-loop.

Initially use the mask alone. Relation-specific biases and overlap-neighbor edges are later
ablations, not additional changes in the first arm. Complement edges are cheap but low priority
chemically; the small charge-choice error population prevents claiming they target most errors.
General sibling/overlap connections can make nearly every candidate reachable and should only
be added after a graph census. Non-neighbors still compete through the final spectral objective.

Measure isolated-node fraction, degree distribution, precursor dependence, and reach at the
actual number of attention layers. An isolated node remains valid through its self-loop and
existing token. Direct containment is relative to the retained candidate set; proposal churn
can change edges even when two endpoints remain, so measure graph stability under repeated
proposal construction. Containment is not a chemically verified reaction step.

**Cost caveat:** an attention mask does not make a dense implementation computationally sparse.
At K=256 it still forms 65,536 attention entries per head per layer before padding. A truly sparse
kernel is a separate optimization justified only after the mechanism shows value.

## Conditional experiments that identify each contribution

The previous four-arm sketch confounded candidate relations with induced-to-full topology.
Use one production anchor plus a 2 x 2 factorial on the same full-attention implementation:

| Arm | Retained-atom branch | Candidate interaction |
| --- | --- | --- |
| B | none | production `i1` induced attention |
| F | none | full attention |
| R | retained-fragment GNN | full attention |
| C | none | relation-masked full attention |
| RC | retained-fragment GNN | relation-masked full attention |

Read R-F for the atom branch, C-F for the candidate graph, and RC against both R and C for
combination value. The interaction contrast is `RC - R - C + F`; report it with paired
uncertainty, not as proof of biological or chemical mechanism. Also compare every arm with B.
A positive interaction does not rescue a combined model that loses to production.

If R or RC shows a useful gain, add the matched pooled branch at that same candidate topology.
If C or RC shows a useful gain, use a graph control matched on degree where feasible to test
whether chemistry-defined relations outperform generic restricted communication. Controls must
be defined without target labels, preserve self-loops, and report any unavoidable mismatches.

Do not require both individual arms to pass before ever checking RC: a modest interaction could
exist even when individual gains are small. Conversely, two individually positive arms do not
justify assuming additive gains. Stage the budget around feasibility and attributable results.

## Feasibility, validation, and stop criteria

1. **Cost and graph census before training.** Use fixed actual `i1` candidate batches. Count
   total retained atoms/bonds, unique atom sets across CE, relation edges, and graph reach.
   Benchmark construction plus forward/backward against B and F, including real candidate-size
   tails. No speed or memory estimate is established yet. Select a practical cost ceiling from
   the measured profile before launching a campaign.
2. **Safe reuse.** Cache topology/raw static features by molecule and retained atom set; do not
   persist learned embeddings across optimizer steps. Within a forward, CE-free embeddings may
   be reused only when atom features and topology agree. Sharing dropout realizations across CE
   is an explicit choice. Keep CE-dependent proposal membership unchanged.
3. **Correctness when implemented.** Candidate permutation equivariance, internal-edge masking,
   correct cut-site markers, empty boundary/one-atom cases, finite gradients, batch isolation,
   and exact disabled-path behavior. No target-defined edges or support selection. Preserve all
   isobaric explanations and sum at observed mass as before.
4. **Quality.** Use matched data, seed, training recipe, and candidate budgets on corrected
   instrument labels. Report production cosine, calibration-controlled cosine, base-peak hits,
   ranking deficit, nested/alternative-pair accuracy and novelty slices, coverage, and cost.
   Candidate-space tolerance cosine is a diagnostic, not a replacement headline metric.
5. **Decision.** Confirm useful gains across seeds, with molecule-clustered uncertainty and
   matched controls. No promotion based only on train fit or gap shrinkage. Reject expensive
   combinations that do not improve on the cheaper successful arm. If the atom branch only
   matches its pooled control, do not claim a connectivity benefit. If relations become nearly
   global, do not claim sparse relational inductive bias.

## Current state and next check

Inspected the existing candidate graph plan, updated `i1` findings,
[model pooling](../../../src/fragnnet/model/masster_model.py), and
[head projection/contextualization](../../../src/fragnnet/model/bond_fragment_heads.py).
No source/config changes. Documentation links and whitespace are the validation scope.

Next: read `fa1` against `i1` and test whether candidate relations explain held-out
ranking errors after full attention. The retained-graph cost census and R/RC arms are deferred
until a new information signal warrants them. No training, cluster submission, dataset
modification, or committed Git change is part of this update.

**2026-09-29 addendum: the retained-fragment branch was built and screened locally.**

- **Decision.** The user chose, in the step-3 session of the wrong-base-peak work, to build the
  bounded first version and screen it despite the deferral.
- **Branch.** One GINE layer, width 32, over each candidate's retained bonds.
  - Inputs: raw atom and bond features plus the retained degree and kept cut-bond count per atom.
  - Readout: all-atom and cut-site pools, added as a zero-initialized residual after
    `input_projection`.
  - Keys: `masster_fragment_connectivity_*`, default off.
  - Arms: `fc1`, and `fc1p` as the matched no-message control.
- **Result.** Two seeds per arm against two fresh `i1` local controls.
  - The val tail is +0.0088 and +0.0013, a mean of +0.0050, below the pre-registered 0.0076
    control spread.
  - Calibrated cosine is +0.0008.
  - Nested and sibling decisions do not move consistently. Train fit falls slightly.
  - Not positive, so `fc1p` was not run.
- **The candidate branch (relation-masked attention) is untouched by this.**
- [Finding](../../findings/masster/masster_fragment_connectivity_local_screen_2026_09_29.md).
