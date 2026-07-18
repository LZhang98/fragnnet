# FluxMPNN — FraGNNet-CFM: a directional neural flow model over the fragmentation DAG

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
sibling softmax, CE-conditioned) → logsumexp-over-parents → per-node residual at `1519`. Zero-init,
KL-leashed to base. Tests whether parent-local competition helps *at all* before re-architecting the
output. This is the minimal version of the user's directional-transition head.

### Stage 2 — directional transition MPNN
Maintain a state per directed edge `p→c` initialised from `[h_p, h_c, h_c−h_p, complement, cc,
ce]`; 1–2 layers of directional message passing where `p→c` aggregates the edge states that produced
`p` (ancestor→transition context). Edge score → same sibling softmax → same residual. Still a
correction on top of base P(n); no flow yet.

### Stage 3 — FluxMPNN (conserved probability flow)
Replace the flat P(n) with a flow:
1. `flux = 1` at the root (node 0).
2. Propagate over the **DAG directly** — a topological traversal of `frag_edge_index` itself (the DAG
   is provably acyclic, so its own edge structure gives the order). Do NOT reorder by a heavy-atom
   proxy; a node's `F_in` is ready once all its in-edges have been processed.
3. At each node, a **factored exit** with **two structural halt gates** — NOT one joint softmax over
   `{sink, escape, children}`:
   - CE-free **halt gate**: `α_halt(p) = σ(halt_logit(h_p))` (structural node repr only).
   - CE-free **sink gate**: `α_sink(p) = σ(sink_logit(h_p))` — observed fraction of halted flux.
   - CE-conditioned **sibling softmax**: `β_c = softmax_children(edge_logits(h_p, h_c, cc, ce))`.
   - combine: to child `F(p→c) = (1 − α_halt)·β_c·F_in`; observed `F_sink = α_halt·α_sink·F_in`
     (→ `P(n)`, then the channel head); escaped `F_escape = α_halt·(1 − α_sink)·F_in` (→ OOS/null).

   Three separate normalizers keep CE out of both halt and sink *by construction*. A joint softmax
   `[α_sink, α_escape, α_c1, …] = softmax(exit_logits)` would re-couple them: the shared partition
   function `Z` means any CE-driven change to a child logit moves `α_sink`/`α_escape` too, leaking
   the CE-live branching signal into the CE-flat extent axis.
4. Multi-parent: incoming flux to a node = `logsumexp` over its in-edges ("evidence combines" for
   different parents, competition only among siblings).
5. `P(n)` (from `F_sink`) feeds the existing formula→spectrum pipeline unchanged; the escape mass
   `Σ_p F_escape(p)` *is* the OOS/null-formula bucket — conservation makes peaks + escape = 1, so it
   reconciles by construction rather than via a bolted-on term.
All in log-space, differentiable, trained with the existing spectrum loss.

## Risks / watch-items

- **CE at branching only** (above) — the load-bearing constraint. Parameterise halt as a per-node
  scalar from structure, not from `ce_embed`; use a **factored exit** (halt gate ⊗ sink gate ⊗
  sibling softmax), not a joint softmax over `{sink, escape, children}`, or the shared normalizer
  re-couples them. Also guard
  against CE reaching halt *through* `h_p` (detach / use a CE-free view if `ce_embed` was fused
  upstream). **Verify** extent still tracks CE (0.84→0.35) under CE-free halt before locking it in —
  chain-like subgraphs are the failure mode.
- **Gradient stability**: the earlier top-K competition head NaN'd from gradient backflow into the
  shared frag encoder (fixed via `frag_embed.detach()`). Flow over 3 layers is milder but start with
  the encoder detached, then relax.
- **OOS/normalization reconciliation**: the escape gate *is* the OOS bucket (peaks + escape = 1 by
  conservation). Wire `Σ_p F_escape(p)` into the current `frag_null_formula_logprob` / `oos_logprobs`
  slot rather than adding a second normalizer — **do not double-count** a separate OOS term on top.
- **Scope discipline**: Stage 3 is a real re-architecture; Stages 0–2 must each pay their way first.
  Supersedes the top-K transformer competition head (`project_frag_competition`).

## Success metric

Overall `cos_hun_inst_tol`, but the axis this targets is **high-NCE** (enable
`log_adduct_inst_ce_metrics`). Interpretable diagnostics for free: `P(child|parent, CE)`, sibling
entropy, top-transition accuracy vs the `branching_headroom` oracle, branching-ratio shift with CE.
