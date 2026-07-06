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

Each molecular bond gets a **joint (charge-direction × δH) categorical**
state — the full per-bond object is
`{intact} ∪ {cut × side∈{A,B} × δH∈h_range}` (see Open Question 3 for why δH
belongs here). The directional part is not two independent sigmoids because
"breaks toward A" and "breaks toward B" are mutually exclusive outcomes of one
event, and for a fragment's boundary bond only the term where *that fragment's
side* keeps the charge is the relevant one (compound multi-cut rules like
`xanthone_ring_cleavage`, which need 4 simultaneous cuts, require directional
consistency across all 4 or the fragment isn't the one being scored). The δH
axis folds *local* H-transfer (H migrating across the cleaving bond) directly
into the per-bond head; *remote* H-transfer/rearrangement does not factorize
this way and stays in a separate channel (Open Question 3).

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
3. **H-transfer — must be split into two regimes; they behave oppositely
   under factorization** (this replaces the earlier lazy framing of "one
   additive rearrangement term"):

   - **Local H-transfer (at the cleaving bond).** H migrates across the
     nascent boundary as the bond breaks; observed fragment is
     `[cut fragment ± k H]`. This is exactly the CRF `h_range` machinery at
     `fragnnet_model.py:1451-1461`. It *fits the per-bond model cleanly and
     strengthens it*: the correct per-directed-bond output is not just a
     3-way charge-direction categorical but a **joint (charge-direction × δH)
     softmax**, with k ranging over the bond's `h_range`. The once-per-
     molecule context GNN pass already sees the local environment that
     governs this (α-H availability, heteroatom acceptor on the far side),
     and this is precisely the kind of *local* chemistry that transfers
     across scaffolds — so local H-transfer is an asset for the
     generalization argument, not a liability. **Default: fold into the
     per-bond head.**

   - **Remote H-transfer / rearrangement (away from the cleaving bond).**
     McLafferty, retro-ene, and the new-bond-forming neutral-loss rules
     (`NL_RULES`: SO₂/CO/N₂O extrusion) draw an H (or reorganize bonds) from
     an atom that is *not* at the boundary. A product of local per-bond
     marginals **structurally cannot** represent "this bond breaks *and* an H
     arrives from a specific remote position" — the mass depends on a
     non-local degree of freedom correlated with the cut. This is exactly the
     family the `NL_RULES` and rearrangement SMARTS were hand-coded to
     capture because base BFS can't reach it either. **Keep this as a
     separate out-of-band channel** (SMARTS-injected nodes with an explicit
     H-delta, scored by the concerted-cut combiner), and **do not let the
     bond product try to explain these fragments** — it will either fail to
     reach them or assign nonsense probability.

   Sharp version: the per-bond state is
   `{intact} ∪ {cut × side∈{A,B} × δH∈h_range}`. Everything that fits inside
   that local object goes through the bond product; everything that doesn't
   (remote H, concerted multi-bond, new-bond-forming NL) is by definition the
   rearrangement channel and is modeled separately. **Consequence for
   Step 1**: the diagnostic must *exclude* known-rearrangement / NL-rule
   fragments from the correlation check, or they poison the fit and make the
   product law look worse than it is on the cases where it is actually valid.
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
   **Exclude known-rearrangement / NL-rule fragments** (SMARTS-prepass and
   `NL_RULES`-origin nodes) from this correlation — per Open Question 3 they
   are the remote-H channel the product law is not meant to explain, and
   leaving them in poisons the fit on the cases where it is valid.
   → verify: correlation between naive bond-product estimate and current
   model's fragment probabilities, stratified two ways:
   (a) by cut count (1 vs 2 vs 4+) — tests the multiplicative-bias risk;
   (b) by scaffold novelty (held-out scaffolds vs. seen scaffolds) — tests
   the *generalization* claim directly, since the whole case for this model
   over NodeMLP rests on bond-locality transferring to novel structures.
   If the product law holds *better* on held-out scaffolds than NodeMLP does,
   that is the strongest evidence to proceed; if it only matches on seen
   scaffolds, the generalization premise is not supported.

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

## Where the existing SMARTS rules fit

The two rule sets in `src/fragnnet/frag/smarts_prepass.py` map to the two
channels above very differently:

- **`FRAG_RULES` (ring-cleavage: xanthone, phenoxazine, morphinan, …)** are
  still *sets of real bond cuts*, so they belong to the bond product, in two
  roles: (1) a per-directed-bond **SMARTS-matched indicator** feeding the
  per-bond GNN head, so a bond recognized as part of a known release gets its
  own learned prior instead of looking like a generic C–O bond; and more
  importantly (2) a **concerted-cut group ID** shared across the bonds one
  rule cuts together — this is the exact input the learned combiner (Open
  Question 1 / Step 2) needs to treat a 4-bond ring release as one joint term
  instead of ANDing 4 independent marginals.
- **`NL_RULES` (SO₂/SO₃/N₂O/CO extrusion, …) do not fit the bond product at
  all** — they form new bonds between the remaining pieces, so the fragment
  is not a simple disconnection of existing bonds. These are the remote-H /
  rearrangement channel (Open Question 3), scored out-of-band, and excluded
  from the Step 1 diagnostic.

**Caution from prior work** ([[feedback_prepass_rule_id]]): rule-identity was
already tried as a *fragment-node* feature and did not help, because prepass-
generated *nodes* are noisy. This proposal puts the SMARTS signal on real
*molecular bonds*, a different granularity, so it is not automatically the
same failure — but it is close enough that Step 1 must check the SMARTS-match
feature's value explicitly (matched-vs-unmatched bond stratification) rather
than assuming it helps.

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
