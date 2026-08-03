# FluxMPNN — FraGNNet-CFM: a directional neural flow model over the fragmentation DAG

> **Status:** PARTLY DEAD — READ BEFORE REVIVING &nbsp;·&nbsp; **Updated:** 2026-07-22
>
> Stage 0 headroom gate GREEN, but conserved DAG flow trained null-to-harmful and the unsupervised competition line is DEAD (learned beta collapsed to a bond-type lookup: explains 72% of itself, 18% of real branching). Superseded in practice by `sibling_ranking_target_plan.md`.

## One-line thesis

Unify FraGNNet's learned fragment/molecule representations and rich state space (H-shift,
CRF/CMF candidates) with CFM's competitive generative fragmentation, by making the **output
distribution itself flow-structured over the DAG** instead of a flat categorical over nodes.

- FraGNNet answers: *what fragment states are possible?* (the DAG + representations)
- CFM answers: *how does probability move between states?* (branching + flow)
- FluxMPNN answers: *how can neural representations learn that movement from data?*

## Framing check (what is and isn't true today)

**Correction to the "intensity head sidelined the DAG" framing.** The production model
(run `gdu9njqa`, `intensity_head: False`) does **not** have an intensity head weakening the DAG.
Its output is a *flat joint softmax over every (node, formula) pair in the DAG*
(`fragnnet_model.py:1545`), so every DAG node already receives probability, and there is even an
auxiliary edge loss (`skip_edge_loss`). The accurate gap is subtler and stronger:

> The DAG is used only for **message passing** (representation). The **probability model** that
> turns node embeddings into a spectrum is a flat categorical that ignores the DAG's branching/flow
> structure. FluxMPNN's contribution is making the output distribution *itself* DAG-flow-structured.

This is the more defensible claim; the intensity-head version invites pushback from anyone who
knows the baseline is already node-distributed.

## The one hard constraint our own measurements impose

A flow model has two knobs at every node: **branching** (which child receives the incoming flux)
and **stop/emission** (what fraction of flux halts here and emits a peak). Two model-free
measurements pin down where collision energy is allowed to act:

| Axis | Measurement | Verdict |
|---|---|---|
| *Which sibling* (branching) | `branching_headroom.py`, 2026-07-16 | **strongly CE-dependent** (dom-sibling flip: floor 0.11 → cross-NCE 0.43, monotonic in NCE gap to 0.68) |
| *How deep* / extent (path-level observable) | `ce_extent_headroom.py`, 2026-07-16 | extent **is** CE-dependent (0.838→0.350), but the model already reproduces it with **CE-flat residual error** (~10%/bin) → no headroom for a CE→halt knob; killed the CFM-flow stop head, `depth_ce_bias` |

Two knobs both drive observed extent the same way — CE in **branching** (route flux into deeper
subtrees) and CE in **local halt** `α_halt(p)` (stop less at high CE) — so for extent alone they are
**near-degenerate**. The measurement breaks the tie by *headroom, not physics*: branching has
residual headroom (the sibling-flip), extent does not. Therefore:

> **CE conditions the branching softmax ONLY. Local halt `α_halt(p)` stays CE-free (or nearly) —
> not because halt is energy-independent (it isn't), but because CE-conditioned branching already
> reproduces the extent trend, so a separate CE→halt term has no measured headroom.**

This is what makes FluxMPNN *evidence-shaped* rather than a CFM clone — vanilla CFM makes everything
energy-dependent, which here would spend the CE budget on a second knob for an observable already fit.

### Sharpening: flux fates — branch, sink, escape (two structural halt gates)

The "stop/emission" knob above is really **two structural halt gates** feeding four flux fates:

```text
F_in(p)
  ├─ BRANCH  → continue to a child (outgoing edge flux)          stage 1, CE-live (β)
  └─ HALT    → stop fragmenting here                             stage 1, structural (α_halt)
        ├─ SINK   → observed: emit a peak                        structural (α_sink), CE-free
        │            └─ OBSERVE = existing channel head P(h|node) stage 2 (H-shift/CRF/CMF)
        └─ ESCAPE → unobserved: flux leaves the modeled system   structural, CE-free → OOS/null
```

Two gates split the halted flux where the original plan had one.
`α_halt(p) = σ(halt_logit(h_p))` is halt-vs-branch; `α_sink(p) = σ(sink_logit(h_p))` is, of the
halted flux, the fraction that is *observed*:

- `F(p→c)     = (1 − α_halt(p)) · β_c · F_in(p)`      (β = CE-live sibling softmax)
- `F_sink(p)   = α_halt(p) · α_sink(p) · F_in(p)`      → peak, then OBSERVE channel head
- `F_escape(p) = α_halt(p) · (1 − α_sink(p)) · F_in(p)` → global escape / OOS bucket

They sum to `F_in(p)` exactly. **Why escape is its own gate, not folded into halt or a bolted-on
OOS term:** the DAG is depth-truncated and MS is lossy — neutral/uncharged products, sub-detection
fragments, and out-of-DAG continuations all carry real flux that must *not* be forced to emit a
peak. A single halt gate makes every stopped fragment observable. Total escape `Σ_p F_escape(p)` is
then a **flow-native home for the OOS/null-formula mass**: peaks + escape = 1 from the root by
conservation, so normalization falls out instead of being reconciled ad hoc.

**CE placement (extends the hard constraint to both gates).** `α_halt` is CE-free — it *is* extent
(extent-null). `α_sink` is CE-free to start: observed-vs-unobserved is a structural/ionization
property, and the extent-null buys no license to CE-condition it. It is the **one gate worth
relaxing later** if total observed coverage turns out CE-dependent — but only through its own
sigmoid, never a joint softmax over `{sink, escape, children}` (a shared `Z` recouples CE into
halt/extent). Flag, don't build.

So `sink (peak intensity) = HALT × SINK × OBSERVE`; the extent-null constrains **HALT** and says
nothing about OBSERVE's CE-dependence (a separate, pre-existing question).

**Why local halt stays CE-free — the precise claim.** The phrase "CE-invariant halt" conflates two
different quantities; only one is CE-free:

- **Extent** (observed, path-level): strongly CE-dependent, 0.838→0.350. Not disputed.
- **`α_halt(p)`** (latent, per-node — the fraction of flux arriving at `p` that stops here rather
  than continuing to a child): the knob we freeze CE-free.

**Physically `α_halt` *is* energy-dependent** (higher internal energy → lower survival → more
fragmentation; CFM models it so). We freeze it anyway on an **identifiability** argument, not a
physics one. Extent is **emergent from the whole flow**: CE-conditioned *branching* that routes flux
into smaller (deeper) subtrees at high CE lowers observed extent *even with every node's `α_halt`
CE-free*. So the two knobs are near-degenerate for the observable — and the measurement says the
model already reproduces the 0.838→0.350 curve (0.863→0.368) with CE-flat residual error, while
branching still shows headroom (the sibling-flip). Spend CE where there's headroom (branching);
freeze the degenerate knob to keep the halt normalizer clean and the high-NCE regime from getting a
second, redundant lever. **We never measured `α_halt` directly** (it's latent) — only that the
path-level observable it feeds has no residual headroom.

**Failure mode (why "or nearly").** In a linear chain, branching has no freedom, so the only way to
deepen penetration at high CE is to lower `α_halt` — which needs CE in halt. FraGNNet DAGs branch
richly, so branch-mediated extent is plausible but not guaranteed. **Verify** before locking CE out
of halt: confirm a CE-free-halt flux model reproduces the observed extent-vs-NCE curve
(0.84→0.35). If it can't, extent has gone CE-frozen and halt needs a *small* CE term.

## Verified substrate (feiw_v2, 2026-07-16)

- DAG edges are directed `(parent, child)`, deduped (`compute_frags.pyx:381`); `is_directed=True`.
- **Provably acyclic**: 100% of edges have the child with strictly fewer heavy atoms than the
  parent. Flow is exact; max depth 3 → cheap.
- **Topological order = descending heavy-atom count.** Do NOT layer by `nodes_min_depth`: only
  ~50% of edges respect it (multi-parent min-depth artifact), so depth-layering drops half the flow.
- Exactly **one root** (node 0 = precursor) — flux source.
- Training uses `pyg_bigraph: False` → `frag_edge_index` reaching the model is single-direction, so
  `scatter_logsoftmax(logits, src)` is a true per-parent sibling softmax. **Guardrail:** assert
  `pyg_bigraph == False` at init (it is a swept flag; reverse edges silently corrupt the softmax).
- Edge features already carry the transition: `cc` (broken bond), `complement` (lost/neutral-loss
  fragment), `base_formula`. Per-node CE embedding `ce_embed` exists (`fragnnet_model.py:792`).
- Per-node scalar residual injection point already exists: `fragnnet_model.py:1519`
  (`frag_joint_logits + per_node_bias.unsqueeze(-1)`).

**Reuse win:** FluxMPNN replaces only *how P(n) is computed* (flat softmax → flow). Everything
downstream of P(n) — P(f|n) formula head, H-shift states, CRF/CMF, isotopes, ion m/z, OOS/null,
the spectrum loss — is untouched.

## Staged plan (each stage is a decision gate)

### Stage 0 — headroom gate ✅ GREEN (ran 2026-07-18, `gdu9njqa` val)
Model-vs-oracle branching probe: `q_base(p→c) = softmax_siblings(pred_node_probs)` vs the observed
sibling target (`branching_headroom.py` machinery), NCE-stratified.
Script: `scripts/eval/stage0_branching_model_vs_oracle.py` (reads the node-grouped inference HDF5).

**Result** (4,945 mols, 90,904 spectra, 5.87M competable parent-spectra = 48.6%, matching
`branching_headroom` coverage exactly). Per-parent, over the explained-sibling support, by NCE:

| NCE | TV_model | TV_unif | dom-mismatch model | dom-mismatch unif |
|-----|----------|---------|--------------------|-------------------|
| 0–20  | 0.397 | 0.326 | 0.452 | 0.679 |
| 35–50 | 0.402 | 0.397 | 0.506 | 0.751 |
| ≥120  | 0.324 | 0.362 | 0.442 | 0.690 |

- **The model does NOT capture branching → PROCEED.** Its implied sibling split is *at or below* the
  uniform-over-siblings baseline on TV (worse than a coin flip at NCE ≤35), and it picks the wrong
  dominant sibling **~45–50%** of the time vs the **~0.11** same-NCE replicate floor. Parent-local
  competition is genuinely absent from the flat node-softmax. Large absolute headroom.
- **BUT the error is NOT concentrated at high NCE** — flat-to-U-shaped (worst mid-band 35–50, better
  at both extremes). Min/max brackets agree, so alias ambiguity doesn't hide it. The observed
  branching *is* CE-dependent (`branching_headroom`), but the *model's error* is CE-flat → the
  first-order gap is **structural sibling competition, not its CE-modulation**.
- **Implication for Stage 1:** build the sibling-softmax residual CE-conditioned (cheap) but **ablate
  against a CE-free sibling head** — the data predicts the structural head carries the gain and CE is
  a second-order correction, not the driver. Do not over-invest in CE-conditioning up front.

### Stage 1 — sibling edge-softmax residual (de-risk, non-flow)
Learned `EdgeMLP` over directed `frag_edge_index` → `scatter_logsoftmax(logits, src)` (per-parent
sibling softmax, CE-conditioned) → logsumexp-over-parents → per-node residual at `1519`. The residual
must be **identity at init**: a raw sibling log-softmax is *not* neutral, because zero logits imply a
uniform-over-children prior and, after parent aggregation, inject an in-degree/out-degree structural
bias before learning. Implement the correction as either:

- a baseline-centered edge residual, e.g. `log β_edge - log β_uniform(parent)`, then aggregate to the
  node; or
- a learned scalar gate multiplying the whole parent-competition residual, initialized exactly to 0.

Zero-init the edge scorer, KL-leash the final node distribution to the base distribution, and ablate
CE-conditioned vs CE-free sibling heads. Tests whether parent-local competition helps *at all* before
re-architecting the output. This is the minimal version of the user's directional-transition head.

### Stage 2 — directional transition MPNN
Maintain a state per directed edge `p→c` initialised from `[h_p, h_c, h_c−h_p, complement, cc,
ce]`; 1–2 layers of directional message passing where `p→c` aggregates the edge states that produced
`p` (ancestor→transition context). Edge score → same sibling softmax → same residual. Still a
correction on top of base P(n); no flow yet.

### Stage 3 — FluxMPNN (conserved probability flow)
Replace the flat P(n) with a flow:
1. `flux = 1` at the root (node 0).
2. Propagate over the **DAG directly** using a real topological order computed from
   `frag_edge_index` per graph (Kahn/CSR traversal is fine). The DAG is provably acyclic, so the
   traversal is exact; a node's `F_in` is ready once all its in-edges have been processed. Descending
   heavy-atom count is a useful verified shortcut/test oracle, but not the implementation contract.
   Do NOT layer by `nodes_min_depth`.
3. At each node, a **factored exit**. Think of every parent as having three kinds of **outgoing
   edge** — `p→child_c` (continue), `p→sink` (emit/observe here), `p→escape` (unobserved) — with
   flux conserved: `F_in(p) = Σ_c F(p→child_c) + F(p→sink) + F(p→escape)`. The split is a **nested
   (2-level) normalizer**, NOT one flat softmax over `{children, sink, escape}`:
   - CE-free **halt gate**: `α_halt(p) = σ(halt_logit(h_p))` (structural node repr only).
   - CE-free **sink gate**: `α_sink(p) = σ(sink_logit(h_p))` — observed fraction of halted flux.
   - CE-conditioned **sibling softmax**: `β_c = softmax_children(edge_logits(h_p, h_c, cc, ce))` —
     `β` is the branching *ratio* `P(child=c | p continues, CE)`, normalised over children **only**;
     it is the sole CE-carrier (an `EdgeMLP` over `(parent, child)` pairs).
   - combine: to child `F(p→c) = (1 − α_halt)·β_c·F_in`; observed `F_sink = α_halt·α_sink·F_in`
     (→ `P(n)`, then the channel head); escaped `F_escape = α_halt·(1 − α_sink)·F_in`.
   - **leaf rule:** for nodes with no children, force `α_halt = 1` (or equivalently route all
     incoming flux to halt before sink/escape). Otherwise `(1 − α_halt)` has no outgoing edge and
     conservation silently leaks mass.

   Three separate normalizers keep CE out of both halt and sink *by construction*. A **flat** softmax
   `[α_sink, α_escape, α_c1, …] = softmax(exit_logits)` would re-couple them: the shared partition
   function `Z` means any CE-driven change to a child logit moves `α_sink`/`α_escape` too, leaking
   the CE-live branching signal into the CE-flat extent axis. The nested form is what makes the
   all-edges picture safe.
4. Multi-parent: incoming flux to a node = `logsumexp` over its in-edges ("evidence combines" for
   different parents, competition only among siblings). **This is load-bearing:** the DAG is ~98%
   multi-parent (median in-degree 3, mean 8; `scripts/debug/debug_dag_indegree_stats.py`), so branching
   is irreducibly per-edge and a per-node projection of it loses the parent-conditional structure —
   the flow's per-edge `β` + this in-edge aggregation is the *only* place that structure survives.
5. **`P(n)` (from `F_sink`) feeds the existing formula→spectrum pipeline unchanged.** OOS handling —
   **v1 keeps the existing `oos_module` and does NOT model escape**: set `α_sink = 1` (all halted
   flux is observed), so `P(n) = F_sink(n)` sums to 1 over nodes (a valid distribution, since every
   leaf is explicitly forced to halt) and the current pooled OOS gate stays on top exactly as today.
   Rationale: `α_sink` is a CE-free knob on the **halt/extent** axis, which
   [[project_ce_extent_allocation_null]] proved has no residual headroom; and the per-node escape
   split is **under-identified** (the observed spectrum only constrains the *aggregate* off-peak mass,
   never *which* node leaked), so a rich escape model just regularises to a global constant with extra
   machinery. Spend all capacity on `β`.
   **Later (optional):** if conservation-native OOS is wanted for its own sake, turn `α_sink` back on
   with **low capacity** (global / near-constant) and **anchor** `Σ_p F_escape(p)` to the observed
   unexplained fraction — treat it as a normalization nicety, not a modeling lever.
All in log-space, differentiable, trained with the existing spectrum loss.

## Risks / watch-items

- **CE at branching only** (above) — the load-bearing constraint. Parameterise halt as a per-node
  scalar from a named **CE-free node representation**, not from `ce_embed` and not from any
  `frag_embed` tensor after CE/context has been concatenated. Concretely, expose/pass a structural
  node view for halt/sink (`h_struct`) and reserve the CE-conditioned edge view for `β`. Use a
  **factored exit** (halt gate ⊗ sink gate ⊗ sibling softmax), not a joint softmax over
  `{sink, escape, children}`, or the shared normalizer re-couples them. **Verify** extent still
  tracks CE (0.84→0.35) under CE-free halt before locking it in — chain-like subgraphs are the
  failure mode.
- **Gradient stability**: the earlier top-K competition head NaN'd from gradient backflow into the
  shared frag encoder (fixed via `frag_embed.detach()`). Flow over 3 layers is milder but start with
  the encoder detached, then relax.
- **OOS/normalization — do NOT unify escape with OOS in v1.** The current `oos_logprobs`
  (`fragnnet_model.py:1672`) is a *global pooled sigmoid*, not flow-derived; escape would be a
  per-node, under-identified latent on the no-headroom extent axis (see Stage 3 step 5). v1: keep the
  pooled `oos_module`, set `α_sink=1`, no escape. Only later, if you add escape, anchor its aggregate
  and keep it low-capacity — and wire it *into* the existing OOS slot, never as a second normalizer
  on top (that double-counts).
- **Scope discipline**: Stage 3 is a real re-architecture; Stages 0–2 must each pay their way first.
  Supersedes the top-K transformer competition head (`project_frag_competition`).

## Success metric

Overall `cos_hun_inst_tol`, but the axis this targets is **high-NCE** (enable
`log_adduct_inst_ce_metrics`). Interpretable diagnostics for free: `P(child|parent, CE)`, sibling
entropy, top-transition accuracy vs the `branching_headroom` oracle, branching-ratio shift with CE.

---

# POST-MORTEM ADDENDUM 2026-07-22 — β pooling tested, "no tying" hypothesis REFUTED

Hypothesis under test (proposed in discussion, never measured): Stage 3's β failed to learn because a
GNN-produced per-edge β can give every edge its own value, so evidence never pools across molecules
the way CFM's coarse feature-tied break parameters do. If true, the fix would be CFM-style tying —
β as a function of a coarse bond descriptor rather than a learned edge embedding.

**Tested offline with no retraining and no checkpoint**, via `scripts/eval/flux_beta_bucket_variance.py`.
β is not in the inference dump, but is exactly recoverable for single-parent children from the
already-exported node latents:

```
log_f_in       = log_f_sink - log_alpha_halt
log β(p→c)     = log_f_in(c) - log_f_in(p) - log(1 - alpha_halt(p))
```

400 val mols → 14,319 recoverable edges. Variance of excess-over-uniform `log β`
(`log β + log out_degree`, removing the sibling-softmax fan-out baseline) decomposed three ways:
within-edge (CE/context), within-bucket (edge-to-edge, same bond environment), between-bucket
(bond chemistry).

| radius | ICC (all) | n (all) | buckets | ICC (fixed@r2) | n | buckets |
|---|---|---|---|---|---|---|
| coarse 12-way | 0.1863 | 14,291 | 24 | — | — | — |
| 0 | 0.1990 | 14,022 | 42 | 0.5628 | 837 | 9 |
| 1 | 0.4865 | 5,289 | 113 | 0.7177 | 837 | 19 |
| 2 | 0.7219 | 837 | 25 | 0.7219 | 837 | 25 |
| 3 | 0.7251 | 223 | 9 | 0.7251 | 223 | 9 |
| 4 | 0.5735 | 75 | 3 | 0.5735 | 75 | 3 |

**Findings:**

- The hypothesis is **refuted**. Bond environment explains ~72% of edge-to-edge β variance where it
  can be measured. β is not arbitrary per edge; pooling works.
- The coarse 12-way ICC of 0.186 is **bucket crudeness, not per-edge arbitrariness** — the fixed-edge-set
  control shows the same 837 edges score 0.563 under the coarsest bucketing and 0.718 at radius 1.
- **Radius 1–2 saturates.** r3 adds +0.003 on a quarter of the edges (noise); r4's drop is a small-n
  artifact (75 edges, 3 buckets). The model encodes *local* bond chemistry in β, not long-range context.
- Parent fan-out is not a confound (ICC 0.203 raw vs 0.186 excess). Single-parent edges are **not**
  depth-1-biased (68.9% depth-3, 13.7% depth-1).
- Caveat: the fixed-set column is not truly fixed-n at r3/r4 — the ≥20-member filter re-applies within
  the subset. Only the r0/r1/r2 rows are a clean same-edges comparison. All results are single-parent
  edges only.

**Revised picture.** Only 5.8% of edges sit in a radius-2 bucket with ≥20 support. β is well-determined
where the environment recurs and *unmeasurable* where it does not. The binding constraint is **support
in the rare-environment tail**, not expressiveness, radius, or absence of tying. Coarse tying therefore
survives only as a way to borrow strength into the tail — a much smaller predicted effect than "fix
global arbitrariness", and not currently ranked above the marginal-level work in
`docs/plans/sibling_ranking_target_plan.md`.

**Open reference test.** The same decomposition on observed `q_obs` gives the ceiling: how much of REAL
branching variance bond environment explains at all. If real branching also saturates near 0.72 at
radius 1, β was about as good as bond chemistry allows and the tying line closes entirely. Shares all
its machinery with the sibling-ranking target builder.
