# Radius-N Bond Environment — Learned Fan-Out Ranking Plan

> **Status:** SUPERSEDED · **Updated:** 2026-07-05
> Confirmed the learned radius-1/2 scorer (held-out AUC 0.884) over the hand-mined table; prototype and remaining integration steps moved to `dag_full_recall_tractability_plan.md` (itself negative, see there).

## Motivation

`docs/plans/fragnnet/dag_fanout_cap_plan.md` ranks candidate cuts at each BFS expansion
step using a hand-mined `BUCKET_LOSS_RANK` table keyed on a coarse bond
descriptor: `(in_ring, element_pair, alpha_hetero)` — effectively a
radius-1 approximation (it only looks one bond out, to check for an
adjacent heteroatom). This can't distinguish e.g. the C-C bond in
`O=C-C=C` (conjugated, alpha to a carbonyl) from a bare alkyl C-C, even
though these likely have very different real breakability.

This doc explores generalizing that descriptor to an arbitrary-radius
circular bond environment (ECFP/Morgan-style), and asks two questions
before committing to either a finer hand-mined table or a learned scorer:

1. Does cardinality blow up so much that hand-mining starves for data
   (most buckets become singletons)?
2. If mining still works statistically, is a hand-authored lookup table
   still the right mechanism, or does the resulting table become too
   large to responsibly hand-curate — the point where a learned scorer
   (taking the same environment features as input, without needing
   enumerated rows) becomes the better mechanism?

## Descriptor design

For a candidate bond between atoms `a` and `b`:

- `atom_env(atom, radius, came_from)`: a rooted tuple environment —
  `(symbol, is_aromatic, formal_charge, children)`, where `children` is a
  sorted tuple of `(bond_type, atom_env(neighbor, radius-1, atom))` for
  each neighbor except `came_from` (the other side of the bond being
  described, excluded so the environment reflects "what's attached on
  this side of the cut," not the cut itself). Radius 0 terminates with
  empty children. Memoized per-molecule via a `(atom_idx, came_from,
  radius)` cache, since the same environment is revisited across the many
  redundant multi-parent routes to a fragment.
- Full bond bucket: `(in_ring, sorted_pair(atom_env(a, r, b), atom_env(b, r, a)))`
  — undirected, so the pair is order-independent.
- A human-readable renderer (`env_to_str`) turns the tuple back into a
  SMILES-like string, e.g. `C(=O,-C) -- C(-C,-C)`, for inspection.

This is a tree-unrolling (not a canonical WL/Morgan hash), so it doesn't
collapse graph-symmetric environments the way true Morgan invariants
would — acceptable for a diagnostic, not claimed to be a minimal/canonical
encoding.

## Step 1 Results (2026-07-03, 191 molecules, seed=42)

Built `scripts/debug/dag/debug_dag_bond_radius_env_stats.py`: reuses the same
DAG-traversal / min-depth-route logic as
`scripts/debug/dag/debug_dag_bond_descriptor_stats.py`, but records the raw
`(a, b)` atom-index bond pairs per route instead of pre-bucketing them, so
one tar scan can be re-bucketed at multiple radii without re-scanning.
Reports bucket cardinality and support (route count per bucket), plus
top single-bucket masking loss — not the full greedy combined-mask search
(that's O(bucket_count²) over formulas; at radius ≥ 1 bucket counts are
already in the hundreds-to-thousands, making that search impractically
slow and its output an unreviewable table).

Ran on `nist20_ma_mi_d4_h4_isoFalse_smpFalse.tar`, `--n_sample 200 --seed
42 --max_nodes 50000` (191 of 200 sampled molecules analyzed after
skipping 9 with >50,000 nodes; **60,556 formulas first-matched at depth
4**):

| Radius | Distinct buckets | Singletons | Median support | Mean support |
|---|---|---|---|---|
| 0 (fixed descriptor baseline, `debug_dag_bond_descriptor_stats.py`) | ~12 | 0 | high | high |
| 1 | 648 | 1 (0.15%) | 498 | 2,192 |
| 2 | 2,042 | 7 (0.34%) | 259 | 707 |

Max single-bucket masking loss also shrinks as radius increases (~0.34%
of formulas at radius 1, ~0.13% at radius 2, vs. double-digit percentages
for the coarsest buckets in the original bond-mask plan) — expected, since
finer buckets are individually less load-bearing.

**Finding**: cardinality grows fast (12 → 648 → 2,042) as expected, but
support per bucket does **not** collapse — even at radius 2, on only 191
molecules, the median bucket still has 259 backing formulas and only
0.34% of buckets are singletons. So **sparsity is not the blocker** for
hand-mining at radius 2 on this corpus size.

**Reframing**: the real obstacle is that a useful masking/priority policy
at radius 2 requires reasoning about ~2,000 buckets jointly (the greedy
combined-mask search from the original bond-mask plan), which is not a
table a person can responsibly author or verify by hand, even though the
data supports it statistically. This is the concrete case for a learned
scorer: not "we lack data" but "a human can't own a 2,000-row lookup
table, while a model conditioned on the same environment features can
absorb it directly without anyone enumerating rows."

## Step 1b — Scale check (2026-07-04, 2,868 molecules, seed=42)

Reran `debug_dag_bond_radius_env_stats.py` at `--radii 1,2` on the full
production-scale corpus (`--n_sample 2900`, matching the fan-out-cap/bond-mask
plans' sample size, vs. 191 molecules previously), on
`nist20_ma_mi_d4_h4_isoFalse_smpFalse.tar` (32 molecules skipped for
>50,000 nodes; **979,169 formulas first-matched at depth 4**):

| Radius | Distinct buckets | Singletons | Median support | Mean support | Max single-bucket loss |
|---|---|---|---|---|---|
| 1 | 2,174 | 9 (0.41%) | 816 | 12,607 | 0.34% |
| 2 | 14,839 | 66 (0.44%) | 317 | 1,881 | 0.06% |

**Confirms the 191-molecule finding at full scale.** Cardinality grows even
faster than the small-sample run suggested (2,042 → 14,839 buckets at radius
2 alone), but support does not collapse — median support and singleton rate
are essentially unchanged (0.34%→0.44% singletons), and max single-bucket
loss actually *shrinks* further (0.13%→0.06%), meaning individual buckets are
if anything less load-bearing at scale, not more fragile. Sparsity is
decisively not the blocker; a ~15,000-row table is decisively not
hand-curatable. Scale check complete — no need to re-run at higher `n_sample`
before proceeding.

## Decision

**Confirmed**: move to a learned scorer using radius-1/2 environment tuples
as input features, replacing `docs/plans/fragnnet/dag_fanout_cap_plan.md`'s hand-mined
`BUCKET_LOSS_RANK` table. See
`docs/plans/fragnnet/dag_full_recall_tractability_plan.md` for the scorer prototype
(Step 2) and remaining integration steps (3-5) — first prototype results
(2026-07-04, 400-molecule sample, logistic regression over hashed
radius-1/2 features): held-out AUC 0.884, ~8x precision lift over base rate
in the top 5% of scored bonds, top-ranked bonds are ring C-C bonds
(consistent with `bond_breakability_mask_plan.md`'s finding that ring-CC was
the single highest-loss hand-mined bucket).

## Related docs

- `docs/plans/fragnnet/dag_fanout_cap_plan.md` — per-parent fan-out cap mechanism this
  descriptor is meant to rank candidates for; its `BUCKET_LOSS_RANK` table
  is the radius-0 baseline this plan generalizes.
- `docs/plans/fragnnet/bond_breakability_mask_plan.md` — original coarse bond-bucket
  mining; superseded as the size-reduction mechanism (see
  `[[feedback_nodemlp_cost_metric]]`) but its corpus/methodology is the
  template this diagnostic follows.

## Key files

- `scripts/debug/dag/debug_dag_bond_radius_env_stats.py` — Step 1 diagnostic
  (radius-N cardinality/support + top single-bucket loss)
- `scripts/debug/dag/debug_dag_bond_descriptor_stats.py` — radius-0 baseline,
  reused DAG-traversal logic
- `src/fragnnet/frag/multi_cut_bfs.pyx` — eventual production integration
  target for either a finer hand-mined table or a learned scorer
