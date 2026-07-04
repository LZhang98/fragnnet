# DAG Full-Recall Tractability Plan

## Requirement reversed (2026-07-04)

This doc originally targeted **literal 100% DAG structural recall** (every
true fragment/formula reachable by full enumeration must be reachable in the
generated DAG — no masking, no capping, no learned pruning of any kind). That
requirement has been walked back to **"as good recall as possible under a
fixed node-count/compute budget"** — keep DAG size (and thus nodeMLP training
cost) close to today's D3-class budget, and maximize recall within that
ceiling.

This re-legitimizes the mask/cap/learned-scorer track that the original
version of this doc disqualified: that track is precisely a recall-vs-budget
optimizer, not a 100%-recall mechanism, so it was wrong to rule out under the
old requirement and is the right primary lever under this one. Workstreams A
and B below (generation-time tractability engineering, training-time node
subsampling) were written to make literal full enumeration tractable; they
are demoted to **Deferred**, since the goal is no longer full enumeration.

## Primary plan: node-budget-constrained recall maximization

This picks up work already in progress in two sibling docs, not a fresh
start:

- `docs/dag_fanout_cap_plan.md` — per-parent fan-out cap mechanism.
  Already validated: K=8 gives 16.24% node reduction at 6.30% recall loss,
  dominating the best bond-mask policy (6.57% node reduction / 7.67% recall
  loss) on both axes simultaneously. Currently ranks candidates with a
  hand-mined `BUCKET_LOSS_RANK` table (coarse `(in_ring, element_pair,
  alpha_hetero)` descriptor, effectively radius-1).
- `docs/dag_bond_radius_env_plan.md` — already **green-lit** the move to a
  learned scorer: at radius 2, bucket cardinality grows fast (12 → 648 →
  2,042) but support does *not* collapse (median 259 backing formulas/bucket,
  only 0.34% singletons on a 191-molecule sample) — the blocker isn't data
  sparsity, it's that a ~2,000-row table can't be responsibly hand-authored
  even though the data supports it statistically. A learned scorer over the
  same environment features absorbs that table without anyone enumerating
  rows.

### Step 1 — Scale check (not yet done)

Rerun `scripts/debug/debug_dag_bond_radius_env_stats.py` at `--radii 1,2` on
the full ~2,900-molecule corpus (matching the scale of the fan-out-cap and
bond-mask plans' step-1 runs, vs. the 191 molecules used so far). Confirms
the support numbers hold at production scale before committing to the
learned-scorer design. Estimated ~35-40 min given the ~2.5 min / 191-molecule
runtime observed at small scale.
→ verify: singleton-bucket % and median support stay in the same range as
the 191-molecule run; if support collapses at full scale, fall back to
hand-mining an intermediate radius instead of building a learned scorer.

### Step 2 — Learned scorer prototype

Design not yet started. Constraints carried over from `dag_bond_radius_env_plan.md`:
- Input: radius-1/2 environment tuples (same descriptor already built for
  the mining diagnostic), not raw SMILES/graph — reuse the existing
  featurization rather than inventing a new one.
- Must be cheap enough to run **per-candidate-cut during DAG generation**
  inside `multi_cut_bfs.pyx`, not at train/inference time alongside nodeMLP —
  a small linear/shallow model over hashed/embedded environment features, not
  a GNN.
- Training signal: same recall-contribution label already used for the
  hand-mined buckets (does this bond appear in a route to a real
  matched-peak formula), reused from the existing mining scripts' route-
  finding logic rather than a new labeling pass.
- Output is a static, per-molecule, CE-agnostic priority score (no CE
  input) — inherited unchanged from both sibling plans, since one DAG is
  reused across all CE spectra per molecule.

### Step 3 — Validate against existing baselines (offline, no integration yet)

Compare the learned scorer's candidate ranking against two baselines on a
held-out mining sample:
- (a) the hand-mined `BUCKET_LOSS_RANK` fan-out cap at K=8 — 16.24% node
  reduction / 6.30% recall loss is the bar to beat.
- (b) naive/random tie-break at matched K — this is `dag_fanout_cap_plan.md`
  step 2, also not yet run; do it alongside step 3 here, since both are
  prerequisites for a credible "the learned ranking earns its complexity"
  claim.
→ verify: learned scorer must clearly dominate (b), and should meet or beat
(a) at the same node-count budget — otherwise the simpler hand-mined table
stays in production.

### Step 4 — Integrate

Plug the learned scorer in as the ranking function for the `max_fanout`
parameter already scoped in `dag_fanout_cap_plan.md` for
`compute_ccs_multi_cut` (`multi_cut_bfs.pyx`), replacing `BUCKET_LOSS_RANK`
wholesale rather than running both.

### Step 5 — Regenerate & verify

Regenerate a D4 validation subset with the learned-scorer-ranked fan-out cap;
compare wrecall + node count against: unmasked D4, the D3 baseline, and the
hand-mined K=8 cap. Check high-CE/morphine specifically (the recurring
validation target across both sibling plans).
→ decision gate: ship only if it beats the K=8 hand-mined baseline by a real
margin at the same node-count budget; otherwise this becomes a documented
"tried, didn't beat the simple baseline" entry and the hand-mined table
stays in production.

## Deferred (were primary under the literal-100%-recall framing)

These targeted making *full, unmasked* enumeration tractable. They're not
needed for the node-budget-constrained goal above, but stay documented in
case the budget-constrained approach proves insufficient later and full
enumeration becomes necessary after all:

- **Workstream A — generation-time tractability engineering.** Make
  `multi_cut_bfs.pyx` handle combinatorial-blowup outliers (33/3,000 sampled
  molecules exceeded 50,000 nodes at D4; worst case 1,346,800 nodes /
  9,720,683 edges) without OOM — better dedup data structures, streaming/
  on-disk storage, quarantining outliers into a separate slow-path
  preprocessing batch.
- **Workstream B — decouple training-time compute from DAG completeness.**
  Generate the full DAG once offline; subsample nodes per training batch for
  the nodeMLP pass (unbiased compute/variance tradeoff); keep eval/inference
  walking the complete DAG. Only matters if DAG size is allowed to exceed the
  fixed budget again.

## Related docs

- `docs/dag_fanout_cap_plan.md` — per-parent fan-out cap mechanism this plan
  builds on; its `BUCKET_LOSS_RANK` table is the baseline the learned scorer
  must beat, and its step 2 (naive tie-break sensitivity) is a shared
  prerequisite with this plan's step 3.
- `docs/dag_bond_radius_env_plan.md` — radius-N descriptor design and the
  step-1 mining results that green-lit the learned-scorer direction; this
  plan's step 1 is that doc's still-open "scale check" item.
- `docs/bond_breakability_mask_plan.md` — original coarse bond-bucket mining;
  superseded as the size-reduction mechanism (node reduction was negligible
  relative to nodeMLP's node-count cost driver), but its corpus/methodology
  is the shared template.

## Key files

- `scripts/debug/debug_dag_bond_radius_env_stats.py` — step 1 scale-check
  target.
- `src/fragnnet/frag/multi_cut_bfs.pyx` — step 4 integration target
  (`max_fanout` parameter, per-parent registration loops at ~648-720/
  775-844).
- `preproc_scripts/03_prepare_dag_feats.py` — where `max_fanout` gets exposed
  as a CLI flag, matching `max_cut_size`/`max_ring_cut_steps`.
