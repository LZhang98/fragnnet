# Bond Breakability Mask — DAG Depth Extension Plan

## Motivation

Manual SMARTS rule mining (`smarts_prepass.py`, 16 rules as of 2026-07-02) has
hit diminishing returns as a way to close DAG missing-peak gaps. Several
mined rules note the gap is exactly "one cut over the `max_depth=3` budget"
(e.g. `smarts_prepass.py:382`), suggesting the underlying problem is often
**depth**, not a specific unmodeled rearrangement.

Simply raising `max_depth` combinatorially (D3 → D4 → D5) is not viable as-is:
branching factor multiplies per depth level, so even D4 produces a candidate
pool where real continuations are a small minority — a signal-to-noise
problem, not just a compute-cost one.

## Key Findings (2026-07-02 discussion)

- **Existing rule library is almost entirely "+1 cut."** Every SMARTS rule
  comment that states a cut count says the gap was exactly 4 cuts vs. a
  3-cut budget. No mined rule so far needed 5+ cuts. This argues for scoping
  to D4 first, not committing to D5 up front.
- **Depth need is real and large for some scaffolds.** Morphine (oxide-
  bridged morphinan — previously flagged in
  `smarts_rule_desoxymorphinan_aromatic_core.md` as having "no clean
  topology" for hand-written rules) shows massive D3→D5 wrecall gain at
  higher CE (e.g. NCE 90: 0.102 → 0.966). This is likely recoverable by
  depth alone, not a bespoke rule.
- **Prior art**: Jin, Coley, Barzilay, Jaakkola, "Predicting Organic
  Reaction Outcomes with Weisfeiler-Lehman Network" (NeurIPS 2017,
  [arXiv:1709.04555](https://arxiv.org/abs/1709.04555)). Core idea:
  identify likely reaction-center bonds *first*, then only enumerate
  combinations among those — instead of enumerating everything and
  filtering after. This is the model for how to keep D4/D5 tractable: mask
  out bonds that are never real cleavage sites *before* BFS enumerates,
  rather than scoring the full combinatorial output.
- **A prototype of this already exists and hasn't been run**:
  `scripts/debug/debug_dag_boundary_bond_stats.py` classifies boundary cuts
  as C-C-only vs. heteroatom-containing and measures matched-formula recall
  lost by skipping C-C cuts at the deepest depth. It aggregates globally
  (no CE split).
- **A D4 dataset already exists**: `data/frag/nist20_ma_mi_d4_h4_isoFalse_smpFalse.tar`.
- **Design constraint: one DAG per molecule is reused across all CEs for
  that molecule** (CE conditioning happens downstream in scoring, e.g.
  `FragModeModel`/`CEModel` mixins — not in DAG structure). This means the
  bond mask must **not** be CE-conditioned: it has to be safe for whichever
  CE that molecule's spectra reach. A global (non-CE-stratified) "does this
  bond ever contribute to a matched fragment, anywhere in the corpus"
  aggregation is therefore the correct approach, not a simplification that
  loses information — CE-conditioning wouldn't be usable at generation time
  under this constraint anyway.

## Step 1 Results (2026-07-02, 2,967 molecules, seed=42)

Ran `debug_dag_boundary_bond_stats.py` on `nist20_ma_mi_d4_h4_isoFalse_smpFalse.tar`
(`--n_sample 3000 --max_nodes 50000`). Two fixes were needed to make this runnable:

- **Bug**: the script's heteroatom index list was built from the current
  (17-element) `CANONICAL_ELEMENT_ORDER`, but this D4 tar was generated under
  an older 12-element schema (before As/B/Hg/Sn/Pb were added to
  `ELEMENT_TO_VE`), causing an `IndexError`. Fixed by making the C-C check
  width-agnostic (`base_formula` layout is always `[C, ...heavy, H]`, so
  heteroatoms are just "everything strictly between the first and last
  column"), instead of hardcoding a specific element list.
- **Combinatorial blowup confirmed empirically**: of 3,000 sampled molecules,
  33 exceeded 50,000 DAG nodes at D4; the worst was **1,346,800 nodes /
  9,720,683 edges** for a single molecule — orders of magnitude beyond the
  ~500-4,000 node typical case. This is direct evidence for the
  signal-to-noise concern that motivated this whole plan (see Motivation).
  Added a `--max_nodes` skip guard so the diagnostic can finish; these
  outliers were excluded from the aggregate below.

**Global recall vs. depth, split by cut type at first-match depth:**

| Depth | New formulas | Cumulative recall | C-C % | Heteroatom % |
|---|---|---|---|---|
| d=0 | 14,546 | 0.44% | – | – |
| d=1 | 234,408 | 7.49% | 52.0% | 48.0% |
| d=2 | 861,556 | 33.41% | 44.3% | 55.7% |
| d=3 | 1,207,547 | 69.74% | 38.2% | 61.8% |
| d=4 | 1,005,715 | 100.00% | 35.3% | 64.7% |

**Masking scenarios:**

| Scenario | Recall | Loss |
|---|---|---|
| D3 only (no D4) | 69.74% | 30.26% |
| D4, skip C-C cuts at d=4 only | 89.31% | 10.69% |
| D4, skip C-C cuts at d≥3 | 75.45% | 24.55% |
| Full D4 | 100.00% | — |

**Decision gate outcome: do not stop at the coarse binary bucket.** A 10.7%
global recall loss from masking C-C cuts at just the last depth is too large
to call "good enough" — and the C-C share of newly-matched formulas doesn't
meaningfully taper with depth (52%→44%→38%→35%), so C-C bonds are not a
rare/negligible contributor we can write off with a blanket rule. Proceeding
to step 3: a finer bond descriptor is needed, not a global C-C/heteroatom
split alone.

## Step 3 Results (2026-07-03, 2,967 molecules, seed=42)

Built `scripts/debug/debug_dag_bond_descriptor_stats.py`: reconstructs the actual
broken bond(s) for every DAG edge by diffing parent/child atom bitmasks (the
`cc` node feature, decoded via the existing `th_long_to_mask`) against the
source RDKit mol (`data/proc/nist20/mol_df.pkl`, keyed by `mol_id`), then
classifies each bond into a `(in_ring, element_pair, alpha_hetero)` bucket:

- `in_ring`: is the bond part of a ring in the original molecule.
- `element_pair`: `CC` / `CX` / `XX` (X = any heteroatom) — the same axis as
  the coarse script, but now scoped to the *specific broken bond* rather than
  the whole complement fragment's composition.
- `alpha_hetero`: does either carbon endpoint have another heteroatom
  neighbor (i.e. is the bond "activated" by proximity to a heteroatom, e.g.
  the C-C bond alpha to a carbonyl).

A formula is only counted as *lost* under masking a bucket-set S if **every**
route to it (every tied min-depth parent edge, and every bond in a
multi-bond/ring-release cut) contains at least one bond in S — masking
removes candidate bonds before enumeration, so any route needing even one
masked bond can't occur at all.

**Per-bucket loss if masked alone** (1,005,715 formulas first-matched at d=4):

| in_ring | pair | alpha_hetero | loss if masked |
|---|---|---|---|
| False | XX | – | 0.03% |
| False | CC | True | 0.03% |
| True | XX | – | 0.03% |
| False | CC | False | 0.04% |
| False | CX | True | 0.28% |
| False | CX | False | 0.50% |
| True | CX | True | 0.93% |
| True | CX | False | 1.73% |
| True | CC | False | 1.83% |
| True | CC | True | 4.17% |

**Greedy combined masking** (add buckets lowest-marginal-loss first):

| Buckets masked | Cumulative loss |
|---|---|
| 4 (all non-ring XX/CC variants) | 0.74% |
| 6 (+ ring CX both variants) | 7.67% |
| 8 (+ ring CC both variants) | 25.71% |
| 9 (+ non-ring CX, alpha_hetero=True) | 43.88% |
| 10 (all) | 100.00% |

**Interpretation:** ring membership and alpha-heteroatom proximity are much
stronger discriminators than element-pair type alone. Non-ring C-C and X-X
bonds (of any activation state) can be masked almost for free. The two
buckets that are essentially load-bearing are non-ring C-heteroatom bonds —
chemically the expected "money bonds" for MS/MS fragmentation (ester, amide,
ether cleavage). This clears the step-3 decision gate: masking 4-6 fine
buckets beats masking the entire coarse C-C bucket (10.7% loss) while
touching a comparable or smaller share of bond-space.

## Step 4 Results (2026-07-03, 2,903 molecules, seed=42)

Built `scripts/debug/debug_dag_bond_descriptor_node_reduction.py` to measure the
node/edge-count side of the tradeoff for the two candidate policies from step 3,
applied post-hoc to the same (already fully-enumerated, unmasked) D4 tar: for
every depth-4 node, check whether at least one of its routes (parent edges /
multi-bond cuts) survives a given masking policy.

**Key finding: node count barely shrinks, but edge count shrinks a lot.**
Depth-4 fragments are frequently reachable via multiple redundant parent
routes (3.00 edges per node on average across the corpus), so masking one
bond type rarely eliminates a *node* outright — some other route to the same
fragment usually survives. But it does eliminate a large fraction of the
redundant *edges/routes* themselves, which is the more direct proxy for BFS
enumeration cost (the 1.3M-node pathological outlier from step 1 had ~7
edges per node — that ratio is exactly what this mechanism produces).

| Policy | Node reduction | **Edge reduction** |
|---|---|---|
| 4-bucket (non-ring CC/XX) | 1.50% | **30.28%** |
| 6-bucket (+ ring CX) | 6.57% | **39.43%** |

Combined with the step-3 recall-loss numbers:

| Policy | Recall loss | Edge reduction | Verdict |
|---|---|---|---|
| 4-bucket | 0.74% | 30.28% | excellent tradeoff |
| 6-bucket | 7.67% | 39.43% | much worse marginal rate (10x more recall loss for +9pp edge reduction) |

**SUPERSEDED — wrong target metric.** The above "decision" assumed edge
count (BFS enumeration cost) was the metric to optimize. It is not: the
actual downstream cost driver is **nodeMLP**, whose cost scales with DAG
**node** count, not edge count. Re-reading the table under that lens: bond
masking is largely **ineffective** at the metric that matters — 1.50% node
reduction for the 4-bucket policy, 6.57% for the 6-bucket policy, vs. 30-39%
edge reduction. The mechanism that made edge reduction look good (redundant
multi-parent routes to the same fragment) is precisely what makes node
reduction small: a fragment/node survives as long as *any one* of its
(usually several) routes is unmasked, so masking a bond type rarely
eliminates a node outright, even though it eliminates most of the redundant
routes to it.

**Open problem, unresolved as of 2026-07-03**: bond-level masking as
currently formulated does not solve the actual problem (nodeMLP cost /
node-count blowup). Options not yet explored:
- Mask at the **fragment/node** level instead of the bond level (e.g. only
  keep the first-discovered route to each fragment, or dedupe more
  aggressively — but this doesn't reduce which fragments *exist*, only how
  many ways they're reached, so it wouldn't touch node count either).
- A stronger per-molecule cap or heuristic that limits *which fragments*
  get materialized at depth ≥ 4 at all (closer to the originally-rejected
  "generate-then-filter" approach, or a genuine reaction-center/likely-node
  scorer rather than a bond mask).
- Revisit whether the mined bond-breakability signal can be applied
  earlier — e.g. to prune which depth-3 fragments are even allowed to
  *expand* into depth 4, rather than which bonds can be cut once expanding.

Stopped here per user direction (2026-07-03) — steps 5-8 below are paused
pending a rethink of what "mask" should target.

## Plan

```
1. [DONE — see Step 1 Results above] Run debug_dag_boundary_bond_stats.py on
   the existing D4 tar for a global baseline.

2. [DONE — see Decision Gate Outcome above] DECISION GATE: coarse binary
   bucket rejected (10.7% recall loss at d=4-only masking). Proceed to step 3.

3. [DONE — see Step 3 Results above] Extend the bond descriptor — ring
   membership, alpha-heteroatom proximity — and redo the mining, still
   globally aggregated. DECISION GATE: cleared — 4-6 fine buckets beat the
   coarse bucket's tradeoff. Proceed to step 4, but first confirm
   node-reduction (not yet measured) alongside recall-loss.

4. [DONE — see Step 4 Results above] Measure node/edge-reduction for the
   candidate policies on the same D4 tar. DECISION GATE: 4-bucket policy
   adopted (30.28% edge reduction for 0.74% recall loss; 6-bucket policy's
   marginal cost/benefit is much worse and dropped as fallback-only).

5. Freeze the chosen (4-bucket) policy into a single mask-construction
   function: molecule → bond mask (no CE input at all).
   → verify sanity check specifically against high-CE spectra (morphine
   included) — even though the mask itself isn't CE-conditioned, this is
   where under-sampling risk is highest (rare high-CE-only cleavage events
   could get washed out in a global aggregate), so validate there
   deliberately rather than trusting the average case.

6. Integrate as a static char[::1] array into compute_ccs_multi_cut,
   alongside the existing ring_edge_mask/base_edge_mask.
   → verify: regenerating the current D3 cache with the mask on is a
   no-op — identical wrecall (mask should only start binding at depth>=4).

7. Regenerate a validation subset at depth 4 (5 if warranted) with the
   mask on; compare wrecall + DAG node count against the unmasked
   diagnostic and the current D3 baseline, checking high-CE spectra
   specifically since that's where the depth gains concentrate.
   → verify: recall gain roughly matches steps 1-3's prediction, DAG size
   stays close to D3's size class.

8. DECISION GATE: if precision of retained D4/D5 nodes is high enough,
   ship as the new production depth budget directly — no learned model
   needed. If still noisy, this smaller masked pool feeds a scoring model
   instead of raw combinatorial D4.
```

## Explicitly Deferred

- CE-conditioned masking (rejected — see design constraint above; one DAG
  serves all CEs for a molecule).
- MCTS-style adaptive-depth search — considered as an alternative framing,
  but likely overkill relative to a static mask given the existing rule
  library shows most gaps are "one cut deeper," not open-ended.
- Learned generative "extend" model (propose specific next cuts rather than
  enumerate) — only relevant if the mined bond mask still leaves the
  candidate pool too noisy for direct inclusion (step 7 negative outcome).
