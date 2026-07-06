# Directional Bond-Break Probability — Fragment Scoring Plan

## Motivation

FraGNNet's dominant training/inference cost is **nodeMLP**
(`src/fragnnet/model/nn_blocks.py:620` `NodeMLP`, invoked per DAG fragment
node around `src/fragnnet/model/fragnnet_model.py`), which scores every
enumerated fragment independently and scales with **node count** — the same
bottleneck that motivated `docs/bond_breakability_mask_plan.md` and
`docs/dag_fanout_cap_plan.md`. Those plans attack node count by pruning
*which fragments get enumerated*. This plan attacks the other side: replace
*how a fragment is scored* once it exists.

## Core idea

Factorize fragment probability through bonds instead of scoring each
fragment (node) directly:

```
P(fragment F) = intensity(F)
              = Π_{bonds crossing F's boundary} P(bond breaks, F's side keeps charge)
              × Π_{bonds internal to F}         P(bond stays intact)
```

Each molecular bond gets a **3-way categorical** state per direction, not a
binary break/no-break flag: `{intact, cut→charge retained on side A, cut→charge
retained on side B}`. This is necessary because "breaks toward A" and "breaks
toward B" are mutually exclusive outcomes of one event, not independent
sigmoids — and because for a fragment's boundary bond, only the term where
*that fragment's side* keeps the charge is the relevant one (compound
multi-cut rules like `xanthone_ring_cleavage`, which need 4 simultaneous
cuts, require directional consistency across all 4 or the fragment isn't the
one being scored).

The expensive step (full molecular-context GNN pass) runs once per molecule
to produce per-bond directional logits. Scoring any individual fragment then
costs O(boundary + internal bond count) — a lookup/sum over cached per-bond
log-probs — instead of a full nodeMLP forward pass per fragment.

## Relevant existing infrastructure

- **`NodeMLP`** (`nn_blocks.py:620`) — current per-fragment scorer this would
  replace or sit alongside.
- **Frag-DAG edge features already exist**, but note these are edges between
  *DAG nodes* (parent fragment → child fragment), not molecular bonds: `cc`
  (atom-membership bitmask, `fragnnet_model.py:865-887`), `base_formula`,
  `complement` (`fragnnet_model.py:917-969`). A bond-level model needs a new
  feature axis keyed to the **precursor molecule's bond index**, not the DAG
  edge index — these are easy to conflate since both are called "edges."
- **`frag_edge_logits` auxiliary loss** (`fragnnet_model.py:1444-1449`,
  gated by `skip_edge_loss`) already sums `frag_node_logprobs[src] +
  frag_node_logprobs[dst]` and log-softmaxes over DAG edges. This is a
  precedent for edge-level composition of node scores, but it's currently a
  secondary consistency loss on top of node-level scores, not the primary
  scoring path — this plan would make bond-level scoring primary.
- **CRF/CMF charge-mode formalism** (`src/fragnnet/utils/formula_utils.py:45-67`)
  already distinguishes charge-retention mechanisms (H-transfer complement
  vs. direct-charge/CMF) at the *whole-fragment* mass-shift level. **Open
  question**: does per-bond directional charge-retention duplicate this, or
  is it a finer-grained mechanism underneath it (CRF/CMF decides the overall
  adduct bookkeeping; per-bond direction decides which specific atoms end up
  on the charged side)? Needs to be resolved before implementation — see
  Step 0 below.

## Open questions (unresolved, need answers before this becomes an
implementation plan rather than an idea)

1. **Does intensity actually decay multiplicatively with cut count?** A
   4-cut compound fragment (xanthone-style) gets a product of 4 sub-1 terms.
   If real NIST spectra don't show intensity dropping with cut count, a raw
   independent-Bernoulli product is the wrong combination rule — likely need
   a small learned combiner over cached per-bond logit embeddings instead of
   literal multiplication (still far cheaper than per-fragment nodeMLP,
   since the expensive context pass is still amortized once per molecule).
2. **CRF/CMF overlap** (above) — scope this plan as complementary to or
   replacing that mechanism.
3. **H-transfer / rearrangement fragments** — mass ≠ simple bond-cut sum for
   many observed fragments (this is exactly what the CRF `h_range` machinery
   at `fragnnet_model.py:1451-1461` currently handles). Does the directional
   bond-break model need an additive rearrangement term on top, or is this
   out of scope (bond-cut-only fragments first, rearrangements deferred)?
4. **Independence assumption** — is bond-break conditionally independent
   given per-bond GNN embeddings, or do concerted multi-bond cuts (the
   existing 2-cut ring-release SMARTS rules) need explicit joint modeling
   rather than a product of marginals?
5. **Normalization** — per-bond probabilities won't sum to 1 over the
   fragment/DAG space automatically. Decide: calibrate as unnormalized
   energies feeding a downstream `scatter_masked_softmax` over the DAG (same
   pattern already used for `frag_formula_logprobs` at
   `fragnnet_model.py:1390-1403`), vs. some other renormalization.

## Staged plan

```
0. DECISION GATE (blocks everything else): read formula_utils.py's CRF/CMF
   code path end-to-end plus a handful of real CRF vs CMF training examples,
   and answer Open Question 2 — is per-bond charge direction a duplicate of
   CRF/CMF or a finer-grained layer under it? This determines whether step 1
   is "add a bond-direction head" or "replace part of the CRF/CMF pipeline."

1. Offline diagnostic (no model changes): for a small sample of molecules
   already in a cached DAG (reuse the D3/D4 tars from the sibling depth-
   extension plans), compute a naive per-bond empirical break-frequency
   (fraction of matched/true fragments in the corpus whose boundary includes
   that bond, cut in the observed direction) and check how well
   Π(per-bond frequency) approximates the *existing* nodeMLP-derived
   fragment probabilities on the same molecules. This directly tests Open
   Question 1 (does the product law even approximately hold) before writing
   any training code.
   → verify: correlation between naive bond-product estimate and current
   model's fragment probabilities, stratified by cut count (1 vs 2 vs 4+).

2. DECISION GATE: if step 1 shows the product law degrades badly with cut
   count, prototype the "learned combiner over cached bond logits" variant
   instead of literal multiplication, and re-run the same correlation check.

3. Only after 1-2 clear a gate: design the actual bond-head architecture
   (per-directed-bond 3-way softmax over the mol-graph GNN embeddings,
   analogous to how `edge_transform`/`GatedGraphConv` already consume
   `mol_edge_attr` in `nn_blocks.py:610-617`) and how it plugs into the
   existing DAG scoring path (replacing or feeding `frag_node_logprobs`).

4. Small-scale training comparison (same held-out split used elsewhere in
   this repo) — bond-factorized scorer vs. current NodeMLP — on cos_hun,
   before considering a full swap.
```

## Related docs

- `docs/bond_breakability_mask_plan.md`, `docs/dag_fanout_cap_plan.md`,
  `docs/dag_bond_radius_env_plan.md`, `docs/dag_full_recall_tractability_plan.md`
  — all attack DAG *node count* by controlling which fragments get
  enumerated. This plan is complementary: it changes how an already-
  enumerated fragment gets *scored*, and could in principle stack with any
  of them (fewer nodes, each scored more cheaply).

## Key files

- `src/fragnnet/model/nn_blocks.py:620` — `NodeMLP` (current scorer)
- `src/fragnnet/model/nn_blocks.py:600-617` — existing mol-graph edge-feature
  consumption pattern (`GatedGraphConv` + `edge_transform`), closest existing
  analog for a per-bond head
- `src/fragnnet/model/fragnnet_model.py:1444-1449` — existing edge-logit
  composition precedent (`frag_edge_logits`)
- `src/fragnnet/model/fragnnet_model.py:1390-1403` — existing
  `scatter_masked_softmax` normalization pattern over the DAG
- `src/fragnnet/utils/formula_utils.py:45-67` — CRF/CMF charge-mode tables,
  needs reconciling with per-bond directionality (Step 0)
