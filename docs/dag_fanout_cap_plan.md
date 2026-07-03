# Per-Parent Fan-Out Cap — DAG Node-Count Reduction Plan

## Motivation

`docs/bond_breakability_mask_plan.md` mined a fine per-bond descriptor
`(in_ring, element_pair, alpha_hetero)` and used it as a **hard binary
mask** at depth ≥ 4 (forbid cutting certain bond types entirely). Result:
30.28% edge reduction at 0.74% recall loss, but only **1.50% node
reduction**. That plan's "Step 4 Results" section traced this to why:
`compute_ccs_multi_cut` (`src/fragnnet/frag/multi_cut_bfs.pyx`) dedups
nodes by atom-subset (`css_to_id_dict`) — a fragment is one node no matter
how many redundant parent routes/bond-types reach it, so forbidding one
bond type usually just removes one of several redundant *routes*, not the
node itself.

This matters because FraGNNet's dominant cost is **nodeMLP**
(`src/fragnnet/model/nn_blocks.py:620-675`, invoked at
`src/fragnnet/model/fragnnet_model.py:~1345`), which scales with **node**
count, not edge count. Bond masking optimized the wrong metric. This doc
picks up the "Open problem" flagged at the end of
`bond_breakability_mask_plan.md`'s Step 4.

## Core idea

At each BFS expansion step, if a parent fragment has more than **K**
candidate cuts (combining bridge/cut=1 and ring-pair/cut=2 candidates),
keep only the top-K children, ranked by de-prioritizing the already-mined
"safe" bond buckets (drop non-ring CC/XX first, keep ring-CX/ring-CC
preferentially). This bounds how many *new* atom-subsets get discovered
from any single parent — unlike bond masking, this isn't defeated by
redundant multi-parent routes, because the cap is enforced at candidate
generation time, per parent, before the `css_to_id_dict` dedup lookup.

Full design rationale, rejected alternatives (global top-K budget, smaller
`max_cut_size`/`max_ring_cut_steps`), and the production-integration
sketch for `multi_cut_bfs.pyx` are in the plan file this doc formalizes:
see the "Recommended approach" and "Production integration point" sections
worked out during planning (per-parent cap chosen because node *creation*,
unlike node *dedup*, happens per-parent in the loops around
`multi_cut_bfs.pyx:648-720` (cut=1) and `~775-844` (cut=2); a correct cap
combines both loops' candidates before truncating).

**Design constraint inherited unchanged from the bond-mask plan**: one DAG
is reused across all CE spectra per molecule, so K and the priority
function must stay per-molecule/per-parent static properties, never
CE-conditioned.

## Step 1 Results (2026-07-03, 2,903 molecules, seed=42)

Built `scripts/debug/debug_dag_fanout_cap_node_reduction.py`: reuses
`decode_cc`/`boundary_buckets` from `debug_dag_bond_descriptor_stats.py`,
but inverts the grouping to `children_of[parent] -> [(child, bucket_set),
...]` (the existing scripts group by child to ask "does this fragment
survive," this one groups by parent because the cap truncates candidates
at the parent). For each parent whose candidate count exceeds K, ranks
candidates by `max(BUCKET_LOSS_RANK[b] for b in bucket_set)` — a route is
as load-bearing as its most load-bearing bond — and keeps only the top K.
A depth-4 node survives if any surviving edge (from any parent) reaches
it, mirroring the OR-across-routes logic from the bond-mask scripts.

**Caveat**: this is a lower bound on true generation-time fan-out, since
two different candidate cuts from the same parent that resolve to the same
child are invisible as separate candidates in the already-deduped cached
DAG (they'd both count against the parent's budget at real generation
time, before dedup).

Ran on the same D4 tar and sample as the bond-mask plan
(`nist20_ma_mi_d4_h4_isoFalse_smpFalse.tar`, `--n_sample 3000 --seed 42
--max_nodes 50000`; 33 pathological-outlier molecules skipped, matching
step 1 of the bond-mask plan):

**2,903 molecules | 4,400,648 total depth-4 nodes | 1,005,715 formulas
first-matched at depth 4**

| K | Nodes retained | Node reduction | Recall retained | Recall loss |
|---|---|---|---|---|
| 2 | 1,835,470 | 58.29% | 739,040 | 26.52% |
| 3 | 2,388,127 | 45.73% | 827,956 | 17.67% |
| 4 | 2,808,834 | 36.17% | 871,293 | 13.37% |
| 6 | 3,356,012 | 23.74% | 917,345 | 8.79% |
| 8 | 3,686,137 | 16.24% | 942,403 | 6.30% |
| unlimited | 4,400,648 | 0.00% | 1,005,715 | 0.00% |

**Head-to-head vs. the bond-mask plan's best policies** (same corpus, same
sample):

| Policy | Node reduction | Recall loss |
|---|---|---|
| Bond mask, 4-bucket | 1.50% | 0.74% |
| Bond mask, 6-bucket | 6.57% | 7.67% |
| **Fan-out cap, K=8** | **16.24%** | **6.30%** |
| Fan-out cap, K=6 | 23.74% | 8.79% |

**Fan-out cap at K=8 strictly dominates the bond mask's 6-bucket policy**
— better node reduction (16.24% vs. 6.57%) *and* lower recall loss (6.30%
vs. 7.67%) simultaneously. This confirms the core hypothesis: targeting
node creation directly (per-parent cap) succeeds where bond-type masking
failed, because it isn't defeated by redundant multi-parent routes.

**Decision so far**: fan-out capping is the right mechanism for the actual
cost driver (nodeMLP / node count). K=8 is a strong, low-risk starting
candidate (dominates bond masking outright); K=6 or K=4 trade more
aggressive reduction for higher recall loss if a larger node-count budget
is needed.

## Remaining steps (per the staged plan)

```
0. [Implicitly covered by Step 1's per-K sweep - a dedicated fan-out
   histogram/bridge-vs-ring-pair decomposition was not built separately;
   revisit only if step 2/3 surface a need to understand WHY specific
   parents have high fan-out.]

2. Sensitivity check: does bucket-priority ranking actually beat a
   naive/random top-K at matched K? Rerun step 1's simulation with an
   arbitrary tie-break instead of bucket priority, same K values.
   → verify: bucket-informed ranking should show lower recall loss than
   naive ranking at matched node-reduction. Not yet run.

3. Sanity-check against the blunter existing levers (max_ring_cut_steps=1)
   via actual regeneration of a small sample through
   preproc_scripts/03_prepare_dag_feats.py --max_depth 4
   --max_ring_cut_steps 1 (can't be simulated post-hoc). Not yet run.

4. DECISION GATE: choose K (confirm bucket-priority ranking earns its
   complexity per step 2) before touching multi_cut_bfs.pyx. Given step 1's
   results already dominate the bond-mask baseline, K=8 or K=6 are the
   leading candidates pending steps 2-3.

5-8. Freeze the priority function, integrate into compute_ccs_multi_cut as
   a new max_fanout/bond_priority parameter (two-pass restructure per
   parent - buffer candidates, rank, truncate, then register survivors),
   verify D3 is an unchanged no-op, regenerate a D4 validation subset with
   the cap on and compare wrecall + node count against D3 and the unmasked
   D4 diagnostic (checking high-CE/morphine specifically), then ship.
```

## Related docs

- `docs/bond_breakability_mask_plan.md` — bond-level bucket mining (input
  to this plan's priority ranking); its bond mask is superseded as the
  node-count mechanism but its bucket table remains the source of truth
  for `BUCKET_LOSS_RANK`.

## Key files

- `scripts/debug/debug_dag_fanout_cap_node_reduction.py` — Step 1 diagnostic
- `scripts/debug/debug_dag_bond_descriptor_stats.py` — reused
  `decode_cc`/`boundary_buckets`
- `src/fragnnet/frag/multi_cut_bfs.pyx` — eventual production integration
  target (lines ~390/396 for how static arrays are threaded in today via
  `ring_edge_mask`/`ring_open_mask`; ~648-720/775-844 for the per-parent
  registration loops that need restructuring)
- `preproc_scripts/03_prepare_dag_feats.py` — where `max_fanout` should be
  exposed as a new CLI flag, matching `max_cut_size`/`max_ring_cut_steps`
