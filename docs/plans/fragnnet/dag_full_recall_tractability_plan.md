# DAG Full-Recall Tractability Plan

> **Status:** ACTIVE HUB — MOSTLY NEGATIVE RESULTS &nbsp;·&nbsp; **Updated:** 2026-07-05
>
> The 100%-recall requirement was walked back 2026-07-04. Both hand-mined and learned priority ranking lost to plain random capping; the global cutoff made things worse. Read the step results before proposing another pruning mechanism.

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

- `docs/plans/fragnnet/dag_fanout_cap_plan.md` — per-parent fan-out cap mechanism.
  Already validated: K=8 gives 16.24% node reduction at 6.30% recall loss,
  dominating the best bond-mask policy (6.57% node reduction / 7.67% recall
  loss) on both axes simultaneously. Currently ranks candidates with a
  hand-mined `BUCKET_LOSS_RANK` table (coarse `(in_ring, element_pair,
  alpha_hetero)` descriptor, effectively radius-1).
- `docs/plans/fragnnet/dag_bond_radius_env_plan.md` — already **green-lit** the move to a
  learned scorer: at radius 2, bucket cardinality grows fast (12 → 648 →
  2,042) but support does *not* collapse (median 259 backing formulas/bucket,
  only 0.34% singletons on a 191-molecule sample) — the blocker isn't data
  sparsity, it's that a ~2,000-row table can't be responsibly hand-authored
  even though the data supports it statistically. A learned scorer over the
  same environment features absorbs that table without anyone enumerating
  rows.

### Step 1 — Scale check (DONE 2026-07-04)

Reran `scripts/debug/debug_dag_bond_radius_env_stats.py` at `--radii 1,2` on
the full corpus (2,868 molecules analyzed, 979,169 formulas first-matched at
depth 4). **Confirms** the 191-molecule finding at production scale: radius-2
cardinality explodes (2,042→14,839 buckets) but support does not collapse
(median 317 backing formulas/bucket, only 0.44% singletons; max single-bucket
loss actually shrinks further, 0.13%→0.06%). Sparsity is not the blocker, a
~15,000-row table is not hand-curatable. Full results and table in
`docs/plans/fragnnet/dag_bond_radius_env_plan.md`'s "Step 1b" section.
→ verify: passed — proceeding to the learned scorer without a hand-mining
fallback.

### Step 2 — Learned scorer prototype (first pass DONE 2026-07-04)

Built `scripts/debug/debug_learned_bond_scorer_prototype.py`: logistic
regression over `FeatureHasher`-hashed radius-1/2 environment strings
(reusing `bond_env_bucket`/`analyze_dag` from the radius-env diagnostic),
predicting whether a bond is **critical** for some depth-4 matched formula —
defined as present in *every* route to that formula (masking it alone would
block the formula entirely), mirroring `bond_breakability_mask_plan.md`'s
per-bucket `bucket_loss` computation but at per-bond granularity.

**False start, corrected**: the first label attempt ("bond appears in *at
least one* route to *any* matched formula") was degenerate — 95.4% of real
bonds came out positive, because full D4 enumeration's redundant multi-parent
routes mean nearly every bond gets touched by *some* route to *some*
formula somewhere (consistent with `bond_breakability_mask_plan.md` Step 4's
"3.00 edges per node" finding). Switched to the stricter "critical in every
route" label, which dropped the positive rate to a sane 6.5% (336/8,217
bonds, 400-molecule sample) and produced a real signal.

**Results (400-molecule sample, 291 train / 97 test molecules, grouped
split)**: held-out AUC 0.884, average precision 0.435 vs. 6.5% base rate
(~6.7x lift), precision in the top 5% of scored bonds = 51.8% (~8x lift).
Top-scored bonds are ring C-C environments — consistent with
`bond_breakability_mask_plan.md`'s finding that ring-CC was the single
highest-loss hand-mined bucket (4.17%), a chemistry sanity check the model
passes.

Constraints carried over from `dag_bond_radius_env_plan.md`, still to be
respected as this moves past prototype: cheap enough to run
per-candidate-cut during DAG generation (a linear/shallow model, not a GNN,
already satisfied by logistic regression); static, per-molecule, CE-agnostic
output (no CE input) — not yet exercised since the prototype only scores
existing bonds, doesn't yet run inside `multi_cut_bfs.pyx`.

Not yet done: scale the prototype to the full ~2,900-molecule corpus (only
run on 400 so far) and try higher-capacity models (shallow MLP) if logistic
regression's ceiling turns out too low once compared against the K=8
baseline in Step 3.

### Step 3 — Validate against existing baselines (DONE 2026-07-04, negative result)

Built `scripts/debug/debug_fanout_cap_recall_eval.py` from scratch (the
`debug_dag_fanout_cap_node_reduction.py` script `dag_fanout_cap_plan.md`
cites for its K=8 baseline does not exist anywhere in this repo or its git
history — those numbers could not be reproduced or verified; this is an
independent implementation, not a rerun). Compared hand-mined
`BUCKET_LOSS_RANK` priority, the learned scorer, and naive/random tie-break,
all on the same 90 held-out test molecules, at K=2,3,4,6,8.

**Result: both hand-mined and learned priority ranking lost to plain random
tie-break at every K.** E.g. at K=8: hand-mined 95.00% recall retained,
learned 93.56%, naive random 96.21% (averaged over 3 seeds, std <0.1pp — not
noise). Full table in the script's output / conversation record.

**This invalidates the per-parent independent-static-priority mechanism**,
not just the specific ranking function. Hypothesis: depth-4 nodes average
~3 redundant parent routes; a deterministic per-bond priority ranks the same
bond type the same way at *every* parent, so it correlatedly drops the same
route-type everywhere a formula needs it. Independent random selection
decorrelates survival across a node's redundant routes, benefiting from the
very redundancy a systematic ranking works against. See
`docs/plans/fragnnet/dag_fanout_cap_plan.md`'s new "Superseded" section for the full
writeup — that plan's core mechanism is retired, not just its ranking
function.

## Pivot (2026-07-04): ICEBERG precedent found — new leading candidate

`src/fragnnet/iceberg/` contains an already-implemented, trained, and
evaluated model (cos_sim 0.8094, `docs/plans/model_improvement_plan.md`'s
comparison table) that does almost exactly the "learned generative extend"
mechanism `bond_breakability_mask_plan.md` flagged as deferred — and its
design plausibly explains why Step 3's static per-parent priority failed
where a different mechanism might not.

**How ICEBERG's autoregressive generation works** (`pl_model.py:predict_mol`):
at each depth step, a GNN scores a per-atom "leaving probability" over the
*current frontier* (not per-parent — the whole frontier at once), multiplies
into a cumulative `prob_gen` per candidate fragment, then keeps only the
**global top-`max_nodes`** candidates by probability before expanding to the
next depth (`max_nodes` defaults to 100, actively used in production eval via
`run_iceberg_ms2c_predict_val.py:97`). Redundant multi-parent routes to the
same fragment take `prob_gen = max(existing, new)` — a fragment only needs
its single best route to survive.

**Why this plausibly fixes Step 3's failure mode**: two mechanism
differences from our per-parent static cap, both pointed at the same
problem. (1) The cutoff is **global** across the whole frontier, not
independent per-parent decisions — it can't correlatedly starve one bond
type's routes everywhere while leaving budget unused elsewhere. (2) The
score comes from a model conditioned on **each specific local fragment**,
not a static population-average bucket score, and the max-over-routes rule
means redundancy actively helps survival instead of being a shared
vulnerability.

**Caveats**: (a) ICEBERG's own fragmentation is atom-removal (MAGMa-style),
not FraGNNet's bond-cut BFS — the mechanism needs reimplementing for
bond-cuts, not direct reuse of ICEBERG's code. (b) No config in this repo
sets `max_tree_depth` above 3 (`FRAGMENT_ENGINE_PARAMS` default) — the
"avoids blowup at D4+" property is architecturally plausible (global budget
bounds cost at any depth) but untested here. (c) ICEBERG is a separate model
type/pipeline (`iceberg_gen`), not integrated into FraGNNet's DAG generation.

### Step 4 — Cheap mechanism test before building a model (DONE 2026-07-04, negative — hypothesis rejected)

Built `scripts/debug/debug_global_cutoff_recall_eval.py`: reused the *same*
two priority signals as Step 3 (hand-mined `BUCKET_LOSS_RANK`, learned
logistic-regression scorer), unchanged, but scored each depth-4 node by
`max(priority over all its incoming edges)` and ranked **globally** across
the whole molecule, keeping the global top fraction of nodes by count
(mirroring ICEBERG's `prob_gen = max(existing, new)` + global `max_nodes`
cutoff), instead of Step 3's independent per-parent top-K.

**Result: the global mechanism made things worse, not better.** At every
retention level tested, the gap to naive random *grew* relative to Step 3's
per-parent numbers:

| Node budget (target) | Hand-mined recall | Learned recall | Naive recall |
|---|---|---|---|
| 85% | 85.47% | 90.41% | 95.94% |
| 79% | 80.09% | 85.11% | 94.14% |
| 66% | 75.14% | 75.22% | 89.45% |
| 56% | 70.62% | 66.30% | 84.98% |
| 44% | 62.90% | 54.31% | 78.40% |

(Compare Step 3's per-parent gap at a similar ~85% node budget: hand-mined
95.00% vs. naive 96.21%, only ~1.2 points — the global mechanism's gap here
is roughly 10x larger.)

**Hypothesis rejected, and here's the likely reason**: taking `max` over a
node's redundant incoming routes and then ranking *globally* systematically
inflates the score of nodes with *more* redundant parents, purely because a
max over more samples is stochastically larger — independent of whether the
node is actually important. A node reached by only one route gets a lower
expected max-score for purely statistical reasons, even though single-route
nodes are exactly the fragile ones recall depends on most. This is a flaw
specific to reusing an **uncalibrated proxy score** (a population-average
bucket percentage, or an arbitrary logistic-regression decision value)
inside a max-then-globally-rank mechanism. ICEBERG's actual `prob_gen` is a
genuinely *calibrated probability* trained end-to-end — comparable in
absolute terms across different fragments — which our proxy priority scores
are not. The mechanism (global cutoff, max-over-routes) may still be exactly
right *given a properly calibrated probability model*; this test only shows
it is not a free lunch when bolted onto an existing uncalibrated score.

**Per the verify criterion written before running this**: it lost to random,
so per the decision gate, *the problem is deeper than mechanism choice* —
building a full ICEBERG-style GNN is not yet justified by this evidence
alone. Steps 5-7 (build the real scorer, integrate, regenerate/verify) are
paused pending more diagnosis.

## Open question — resolved (2026-07-04): max-of-N-routes order-statistic bias confirmed as the dominant driver

Extended `debug_global_cutoff_recall_eval.py` with a `hand_mined_single` /
`learned_single` variant: same priority signals, same global ranking,
but score each node from **one uniformly-random incoming route** instead of
the max over all its routes. This isolates whether the bias is (a) an
order-statistics artifact of `max` favoring high-in-degree nodes regardless
of true importance, or (b) score incomparability across different
structural regions of the molecule (see the two hypotheses above).

**Result: single-route scoring recovers most of the gap to naive at every
budget level**, using the exact same (uncalibrated) priority scores:

| Node budget | max (hand-mined) | single-route (hand-mined) | max (learned) | single-route (learned) | naive |
|---|---|---|---|---|---|
| 85% | 85.47% | **94.77%** | 90.41% | **95.36%** | 95.94% |
| 79% | 80.09% | **91.56%** | 85.11% | **92.90%** | 94.14% |
| 66% | 75.14% | **85.70%** | 75.22% | **87.34%** | 89.45% |
| 56% | 70.62% | **80.28%** | 66.30% | **81.74%** | 84.98% |
| 44% | 62.90% | **71.15%** | 54.31% | **71.53%** | 78.40% |

At 85% budget the deficit to naive shrinks from ~10 points to ~0.6-1.2
points; even at the more aggressive 44% cut it shrinks from ~16-24 points to
~7 points. **Effect #2 (max-of-N-samples inflating high-redundancy nodes'
scores regardless of true importance) was the dominant driver of Step 4's
failure, not fundamentally broken score comparability.** A residual gap
remains, especially at aggressive budgets — plausibly effect #1 (score
comparability across molecule regions) or genuine signal weakness in the
proxy scores, not yet separately isolated.

**Implication (revised below)**: this is a fixable mechanical bug (naive
max-aggregation over redundant routes), not evidence that any
static/precomputed priority is fundamentally incapable of beating random
here. It doesn't retroactively justify skipping straight to a full trained
GNN (steps 5-7 are still paused), but it does mean the global-cutoff framing
itself isn't broken — only "max" as the way to combine redundant routes was.

## Follow-up (2026-07-04): "average over routes" tried, surprisingly does NOT help

Natural next ablation: average over all a node's incoming-route scores
instead of max (uses all the redundancy information, without max's
degree-inflation bias) or a single random route (uses only one, but zero
aggregation bias). Added `per_node_average_route_score` to the same script.

**Result: average is roughly tied with, or worse than, max — and clearly
worse than single-route — at every budget:**

| Node budget | max (hand-mined) | avg (hand-mined) | single (hand-mined) | naive |
|---|---|---|---|---|
| 85% | 85.47% | 86.59% | **94.77%** | 95.94% |
| 66% | 75.14% | 73.66% (worse than max) | **85.70%** | 89.45% |
| 44% | 62.90% | 54.49% (much worse than max) | **71.15%** | 78.40% |

This contradicts the natural expectation that averaging would split the
difference between max's bias and single-route's higher variance. Working
explanation: averaging across a node's routes pulls the score toward the
population-mean bond-type score — this doesn't just remove max's
degree-inflation, it also washes out the signal from any one genuinely
load-bearing route a node happens to have. Single-route keeps whichever one
route's raw score it samples, undistorted in either direction; average
actively destroys the useful part of the signal along with the bias.

**Revised implication**: single-route removes essentially all of the
*aggregation-method* bias (that was the point of the ablation), yet it still
trails naive by 4-7 points at every budget, worse at aggressive cuts. Since
the best of the three combination rules tried still doesn't close the gap,
weight shifts back toward **hypothesis #1 (score incomparability across
different molecule regions)** rather than "we just need a better combination
rule" — no combination rule can fix a score that isn't measuring the same
thing in different local contexts. This is now the stronger case for
needing an actually *calibrated* score (i.e. a trained model, not a
population statistic or an arbitrary logit) rather than further tuning how
redundant routes get combined. Steps 5-7 (build the real scorer) remain the
likely next real step, but the specific residual gap size (4-7 points, not
10-24) is now a concrete target to beat rather than an open unknown.

## Follow-up (2026-07-04): node-level scorer tried, hypothesis wrong — features matter more than aggregation

Since every bond-level combination rule (max, average, single-route) leaves
some residual gap, tried removing the aggregation step entirely: score
depth-4 **nodes** directly from features intrinsic to the fragment itself
(element counts, size, aromaticity, ring-atom count, boundary-bond count),
with no bond/route/aggregation step at all — one score per node by
construction. Built `scripts/debug/debug_node_level_scorer_prototype.py`,
same recall label, same 400-molecule sample/split.

**Result: this performed about as badly as the worst prior variant (max),
clearly worse than single-route.** Held-out AUC was only 0.6059 (vs. 0.884
for the bond-level scorer) — barely better than chance:

| Node budget | Node-level scorer | Single-route (bond-level, best so far) | Naive |
|---|---|---|---|
| 85% | 85.76% | **94.77%** | 95.94% |
| 66% | 71.72% | **85.70%** | 89.45% |
| 44% | 55.28% | **71.15%** | 78.40% |

**Hypothesis rejected**: removing the aggregation problem didn't help,
because the substitute features were too coarse. Element counts, size, and
summary ring/aromatic/boundary statistics can't distinguish, e.g., six
carbons in a chain from six carbons in a fused ring system — exactly the
kind of distinction the bond-level radius-1/2 environment encoding (with its
tree-unrolled `env_to_str` local connectivity) *can* make. The lesson isn't
"don't derive scores from bond-level detail" — it's that genuine
connectivity/structural detail matters more than which entity (bond or node)
gets scored, or how routes get combined.

**Current standing best result across this whole investigation remains the
bond-level single-route scorer** (94.77% recall at 85% node budget, ~1.2
points behind naive) — rich local bond-environment features, aggregation
bias removed by sampling one route instead of combining them. The
implication for steps 5-7: keep the bond-level environment richness, keep
the single-route (no max/average) selection, and only then evaluate whether
a genuinely trained model is needed to close the remaining small gap —
rather than trying to route around bond-level features toward coarser
node-level ones.

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

- `docs/plans/fragnnet/dag_fanout_cap_plan.md` — per-parent fan-out cap mechanism; its core
  mechanism is superseded (Step 3 above), but its `BUCKET_LOSS_RANK` table
  is reused as one of the two priority signals in step 4's mechanism test.
- `docs/plans/fragnnet/dag_bond_radius_env_plan.md` — radius-N descriptor design and the
  step-1 mining results that green-lit the learned-scorer direction; this
  plan's step 1 is that doc's still-open "scale check" item.
- `docs/plans/fragnnet/bond_breakability_mask_plan.md` — original coarse bond-bucket mining;
  superseded as the size-reduction mechanism (node reduction was negligible
  relative to nodeMLP's node-count cost driver), but its corpus/methodology
  is the shared template.
- `docs/plans/model_improvement_plan.md` — ICEBERG comparison numbers (cos_sim
  0.8094) referenced in the pivot section above.

## Key files

- `scripts/debug/debug_dag_bond_radius_env_stats.py` — step 1 scale-check
  target.
- `scripts/debug/debug_learned_bond_scorer_prototype.py` — step 2 learned
  scorer prototype.
- `scripts/debug/debug_fanout_cap_recall_eval.py` — step 3 mechanism
  evaluation (hand-mined vs. learned vs. naive per-parent cap).
- `scripts/debug/debug_global_cutoff_recall_eval.py` — step 4 mechanism
  evaluation and its follow-up ablations: max / single-random-route /
  average-over-routes, all with the same underlying priority signals.
- `scripts/debug/debug_node_level_scorer_prototype.py` — node-intrinsic
  (element/size/ring/aromatic/boundary) scorer; negative result, features
  too coarse relative to the bond-level radius-1/2 environment encoding.
- `src/fragnnet/iceberg/pl_model.py` — `predict_mol` (~line 179-410) is the
  autoregressive global-cutoff generation loop the step 4/5 mechanism is
  modeled on.
- `src/fragnnet/iceberg/fragmentation.py` — ICEBERG's `FragmentEngine`;
  atom-removal fragmentation, not directly reusable for FraGNNet's bond-cut
  BFS, but the reference for how `max_tree_depth`/`max_broken_bonds` bound
  its enumeration.
- `src/fragnnet/frag/multi_cut_bfs.pyx` — step 6 integration target (would
  replace the `max_fanout` per-parent design with a global per-round cutoff).
- `preproc_scripts/03_prepare_dag_feats.py` — where the new cutoff parameter
  gets exposed as a CLI flag.
