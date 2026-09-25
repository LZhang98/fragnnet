# MASSter is supervised entirely through m/z, and ~2.75 substructures share every m/z

> **Status:** FINDING — measurement stands, **its recommendation is RETRACTED** &nbsp;·&nbsp;
> **Updated:** 2026-09-22
>
> The measurement: nothing in MASSter's objective checks **which fragment** a candidate is,
> and **2.75 distinct connected fragments share each formula** (p90 3.97), in 82.3% of molecules.
> That much is solid.
>
> **The conclusion drawn from it — that MASSter needs a structural teacher — does not
> survive contact with the evidence, and is withdrawn.** See
> [§ Why this does not motivate a teacher](#why-this-does-not-motivate-a-teacher). In short:
> coverage already transfers (train `wrecall` 0.8975 vs validation 0.8911 against a 0.151 cosine
> gap), MASSter converges perfectly well without a teacher so GLACIER's "Not Converged" does
> not transfer, and same-formula isomers are summed at one m/z, so choosing between them changes
> the predicted spectrum by nothing.
>
> **See also:** what the objective can and cannot see is consolidated with the loss-side reviews
> in [MASSter proposal loss — consolidated audit](masster_proposal_loss_audit_2026_09_22.md).

MASSter-focused follow-up to [feature headroom](../feature_headroom_2026_09_22.md) and the
[unused information inventory](../unused_information_inventory_2026_09_22.md). Those measured
FraGNNet's DAG; MASSter has no DAG, so the conclusions do not transfer and are re-derived here.

## What the objective can and cannot see

| supervision | what it checks |
|---|---|
| `_proposal_coverage_loss` (`masster_pl.py:297`) | a proposed fragment's **m/z** falls inside the tolerance of an observed peak. Every unique fragment counts at most once; which fragment it is, is never checked |
| spectral loss (`loss_type: cross_entropy`, `inst_type_loss_tol`) | predicted vs observed **intensity at m/z** |
| support BCE / channel selection | ranking among already-proposed candidates |

There is no term anywhere that reads fragment identity.

## The two degeneracies are very different sizes

`scripts/debug/debug_masster_identity_degeneracy.py`, 300 molecules, 114,458 annotated
monoisotopic `[M+H]+` peaks, 10 ppm:

| degeneracy | size | resolvable by a formula label? |
|---|---|---|
| **composition** — two different formulas in one tolerance window | 5.2% of peaks, **3.2% of matched intensity** | yes, and NIST ships one |
| **substructure** — one formula reachable as several connected fragments | **mean 2.75 per formula**, p50 2.56, p90 3.97, max 8.87; **82.3% of molecules** average ≥2 | **no** — identical m/z and identical composition |

This **corrects the ranking in the [inventory](../unused_information_inventory_2026_09_22.md)** as it
applies to MASSter. NIST's annotations are the largest unused data in the repository, but they
are *formula* labels, and formula ambiguity is only 3.2% of MASSter's matched intensity. They
are worth little to its proposal head. The degeneracy that matters here is structural, and no
formula label touches it.

Both numbers are computed over the depth-3 DAG's support, which is a **subset** of what
MASSter's unrestricted multi-cut masks can build, so both are **lower bounds** for MASSter.

## What the closest published architecture does instead

> This section was originally the case *for* a teacher. It is kept because the comparison is
> useful, but read it against [§ Why this does not motivate a teacher](#why-this-does-not-motivate-a-teacher):
> the ablation below is a property of GLACIER's proposal mechanism, not of ours.

GLACIER is the nearest design to MASSter: a learned per-atom/per-bond fragment detector rather
than an enumerated DAG. It supervises that detector by **Hungarian set-matching to MAGMa-derived
boundary patterns**, with the teacher weight annealed 1 → 0 after 2,500 steps. Its own ablation
(Table 5, arXiv 2606.29161):

| `λ_decay` | `w_magma` | test loss |
|---|---|---|
| 0 | always 0 | **Not Converged** |
| 0.9 | decays from 1 | 0.177 |
| 1 | always 1 | 0.243 |

The teacher-free row *looks* like the analogue of MASSter's current setup — a spectral
objective with m/z-matched proposal coverage and no structural signal. It is not: MASSter
converges from that starting point and GLACIER does not, because the two proposal mechanisms
differ (see below). Note also that the annealed row beats the teacher-forced row, so where a
teacher is used at all it is a *warm start*, not a target to imitate.

## MASSter loads no DAG, so a stronger teacher than MAGMa sits unused

**No MASSter config sets `frag_dp`, and neither `masster_pl.py` nor `masster_model.py`
references a DAG.** The 1.2 GB depth-3 DAG — enumerated connected fragments with their atom masks,
plus 16 SMARTS and 7 neutral-loss prepass rules — is invisible to it.

That artifact is a strictly stronger teacher than the one GLACIER settles for. MAGMa
(`max_broken_bonds: 6`, `max_tree_depth: 3`) deletes bonds only: it cannot form one, so it
expresses no rearrangement, nothing from our SMARTS or neutral-loss rules, and its depth equals
our d3.

The MASSter plan already names exactly this, as step 2 of its staged training
([`masster_plan.md`](../../plans/masster/masster_plan.md) §7.1):

> **Optional supervised warm start.** If trusted fragment atom masks are available, supervise
> boundary-bond breaks, charge retention, and `(mode, delta_H)` directly. Existing D3/D4
> artifacts may be used as temporary teachers, but the trained generator must not depend on them
> at inference.

It was skipped — production runs go straight to spectrum-level training with
`masster_proposal_selection_only: true` and `proposal_loss_weight: 0.1`. The section
anticipates a failure mode: "Without warm-start labels, top-K selection is discrete and missed
candidates receive no gradient… deterministic top-K alone can lock into initially incorrect
bonds."

**That predicted failure did not materialise.** Coverage is 0.8911 `wrecall` on validation and
transfers from train to within 0.0064, so top-K did not lock into incorrect bonds in any way that
costs coverage. The plan's "optional" label looks correct, and the availability of a better
teacher than MAGMa is a fact about the repository rather than a reason to use one.

## Why this does not motivate a teacher

Three reasons, in order of decisiveness.

**1. Coverage already transfers; allocation is the whole gap.** The
[2026-09-22 overfit read-out](masster_overfit_readout_2026_09_22.md) measures train `wrecall`
0.8975 against validation 0.8911 — a **0.0064** difference — while cosine differs by **0.1507**.
The candidate set is equally good on both sides. A teacher improves the candidate set, which is
the one thing already not broken.

**2. GLACIER's "Not Converged" does not transfer.** Their proposal path is DETR-style query
tokens feeding an affinity matrix through a LinSATNet top-k relaxation — a discrete selection with
a genuine cold-start problem, which is what their teacher rescues. MASSter ranks enumerated cut
masks instead, so it has no such failure mode, and it demonstrably converges to 0.78–0.80 merged
cosine with `proposal_loss_weight: 0.1` and no structural supervision at all. Importing that
ablation across architectures was the error.

**3. The degeneracy is largely inert for the spectrum.** Same-formula isomers land at the *same*
m/z and the model sums their contributions at that peak, so moving probability between them
changes the predicted spectrum by exactly nothing. It is unidentifiable from the target *and*
costless under the metric. It can only bite three ways, none established:

- `h_transfer_bound: valence` makes channel validity depend on structure, so isomers differ in
  which `(ion, ΔH)` channels are legal — a second-order effect on which m/z are reachable;
- the annotation export, where peak → structure is the product, not the metric;
- generalization, if a structure → intensity rule fits train and fails out of scaffold. That is
  *consistent* with the 0.151 gap but is not evidence for it.

## Ranked next steps for MASSter

1. **Allocation, not proposal.** The read-out's own result — `d3` (cosine decay + 1000-step
   warmup) at **0.7357** against a control mean of 0.7286 whose replicates span 0.0007 — is the
   first non-null in this family since `tolerance_consensus`. That is where the movement is.
2. **The two feature arms already built and never run** — `c4a_ring_topology` and
   `c4a_complement_composition`, one-key diffs from `c4a_base_s1` at the same seed, ~5k
   parameters. Cheap screen fodder, not a bet.
3. **Not** a DAG-as-teacher warm start, on this evidence.
4. **Not** more descriptor features — closed in [feature headroom](../feature_headroom_2026_09_22.md).
5. **Not** NIST formula supervision for the proposal head — worth 3.2% of matched intensity here,
   though it stays valuable for the DAG-based models and as a rule-mining residual.

## What would revive the teacher argument

Only evidence that identity error costs accuracy: a corruption-intervention on a trained
checkpoint (the `cc`-corruption pattern used for FraGNNet), or an annotation-quality requirement
where peak → structure is itself the deliverable. Absent either, the measurement above is a
property of the model, not a defect. Nothing here has been trained.
