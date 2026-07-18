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
| *How deep* / extent (cumulative stop) | `ce_extent_headroom.py`, 2026-07-16 | **CE-INVARIANT** (extent error ~10% in every NCE bin; killed the CFM-flow stop head, `depth_ce_bias`) |

Cumulative stop probability **is** fragmentation extent. Therefore:

> **CE conditions the branching softmax ONLY. The stop/emission must be CE-invariant (or nearly).**

This is what makes FluxMPNN *evidence-shaped* rather than a CFM clone — vanilla CFM makes
everything energy-dependent, which here would burn the CE budget re-modelling a measured null.

### Sharpening: three flux fates, and what "CE-invariant stop" actually claims

The "stop/emission" knob above is really **two stages**. Flux at a node `p` has three fates:

```text
F_in(p)
  ├─ BRANCH  → continue to a child (outgoing edge flux)          stage 1, CE-live
  └─ HALT    → stop fragmenting here (F_stop)                    stage 1, structural
                └─ OBSERVE → peak intensity via the existing
                             channel head P(h | node)             stage 2 (H-shift/CRF/CMF)
```

So `sink (peak intensity) = HALT × OBSERVE` (`F_sink · P(h|node)`). The exit softmax governs
**BRANCH vs HALT only**; OBSERVE is the current channel head, untouched. The extent-null constrains
**HALT**; it says nothing about OBSERVE's CE-dependence (a separate, pre-existing question).

**"HALT is CE-invariant" is a *headroom* claim, not a physics claim.** Extent *is* CE-dependent in
the data (retained-heavy-atom fraction 0.838→0.350). The point is the current model already
reproduces that trend (0.863→0.368) with CE-flat residual error, so an explicit CE→halt knob has
nothing to fix — the high-NCE cosine collapse is not coming from getting extent wrong.

**How extent still moves with CE while local `α_halt` stays CE-free.** Extent is a *path-level*
quantity; `α_halt(p)` is *local*. CE-conditioned *branching* that routes flux into smaller (deeper)
subtrees at high CE shifts emission mass toward low-heavy-atom nodes — so extent falls with CE even
when every node's halt fraction is CE-free. Extent is meant to be **emergent from branching**, not
set by a local CE→halt term.

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

### Stage 0 — headroom gate (in flight)
Model-vs-oracle branching probe: `q_base(p→c) = softmax_siblings(pred_node_probs)` vs the observed
sibling target (`branching_headroom.py` machinery), NCE-stratified, on run `gdu9njqa` val.
- **Read:** if `q_base` already tracks the observed cross-NCE shift → the model captures branching,
  STOP. If it diverges (and worse at high NCE) → real headroom, proceed.
- Necessary condition already holds (branching is CE-dependent in the data); this checks the model
  doesn't already have it. **Do not build Stages 1–3 until this is green.**

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
2. Process nodes in **topological order by heavy-atom count**.
3. At each node, a **factored exit** — NOT one joint softmax over `{sink, children}`:
   - CE-free **halt gate**: `α_halt(p) = σ(stop_logit(h_p))` (structural node repr only).
   - CE-conditioned **sibling softmax**: `β_c = softmax_children(edge_logits(h_p, h_c, cc, ce))`.
   - combine: `α_p→c = (1 − α_halt(p)) · β_c`, and `F_stop(p) = α_halt(p) · F_in(p)`.

   Separate normalizers keep CE out of halt *by construction*. A joint softmax
   `[α_sink, α_c1, …] = softmax(exit_logits)` would re-couple them: the shared partition function
   `Z` means any CE-driven change to a child logit moves `α_sink` too, leaking the CE-live branching
   signal into the CE-flat extent axis. Halted flux → node emission `P(n)`, then the existing
   channel head observes it.
4. Multi-parent: incoming flux to a node = `logsumexp` over its in-edges ("evidence combines" for
   different parents, competition only among siblings).
5. `P(n)` feeds the existing formula→spectrum pipeline unchanged; conservation reconciles with the
   existing OOS/null-formula mass.
All in log-space, differentiable, trained with the existing spectrum loss.

## Risks / watch-items

- **CE at branching only** (above) — the load-bearing constraint. Parameterise halt as a per-node
  scalar from structure, not from `ce_embed`; use a **factored exit** (halt gate ⊗ sibling softmax),
  not a joint softmax over `{sink, children}`, or the shared normalizer re-couples them. Also guard
  against CE reaching halt *through* `h_p` (detach / use a CE-free view if `ce_embed` was fused
  upstream). **Verify** extent still tracks CE (0.84→0.35) under CE-free halt before locking it in —
  chain-like subgraphs are the failure mode.
- **Gradient stability**: the earlier top-K competition head NaN'd from gradient backflow into the
  shared frag encoder (fixed via `frag_embed.detach()`). Flow over 3 layers is milder but start with
  the encoder detached, then relax.
- **OOS/normalization reconciliation**: flow mass + OOS must sum to 1; integrate with the current
  `frag_null_formula_logprob` / `oos_logprobs` path, don't double-count.
- **Scope discipline**: Stage 3 is a real re-architecture; Stages 0–2 must each pay their way first.
  Supersedes the top-K transformer competition head (`project_frag_competition`).

## Success metric

Overall `cos_hun_inst_tol`, but the axis this targets is **high-NCE** (enable
`log_adduct_inst_ce_metrics`). Interpretable diagnostics for free: `P(child|parent, CE)`, sibling
entropy, top-transition accuracy vs the `branching_headroom` oracle, branching-ratio shift with CE.
