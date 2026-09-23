# What the candidate token's cut description resolves, on MASShatter's own candidates

> **Status:** FINDING — measurement only, nothing trained &nbsp;·&nbsp; **Updated:** 2026-09-22
>
> Recomputes the [cut-set plan](../plans/masshatter/masshatter_cut_set_features_plan.md)'s premise on the
> right population and **changes what the plan should build**. The plan's item A led with
> ring-sharing cut pairs; on MASShatter's own candidates the pooled cut-bond sum already
> determines the ring-sharing flag (**0.0%** residual ambiguity) while it leaves the adjacency
> flag ambiguous for **48.9%** of candidates. Adjacency is the missing relation, it needs no ring
> identity, and therefore no new graph field and no cache regeneration.

`scripts/debug/debug_candidate_cut_descriptors.py`. 400 NIST20 molecules, **73,045** candidates.

## The premise, on the right population

The plan's 96.8% / 55.4% came from the depth-3 BFS DAG, where the cut count is bounded by
enumeration depth. These come from `build_mass_shatter_candidates` at the production
unique-fragment settings (target 256, ranked budget 4096, single cuts included).

| | depth-3 DAG (old) | MASShatter candidates |
|---|---:|---:|
| candidates cut at >= 2 bonds | 0.968 | **0.869** |
| of those, >= 2 cuts share a ring | 0.572 | **0.847** |
| of those, >= 2 cuts share an atom | 0.350 | **0.787** |
| as a share of all candidates: same-ring | 0.554 | **0.736** |
| as a share of all candidates: adjacent | 0.339 | **0.684** |

Cut counts per candidate: 0.131 / 0.159 / 0.238 / 0.223 / 0.151 / 0.098 for 1 / 2 / 3 / 4 / 5 /
>=6 cuts. The DAG overstated how rare single cuts are (3.2% there, **13.1%** here) and
**understated** both relational effects by roughly a factor of two. The plan's qualitative claim
— multi-cut candidates dominate and relations among their cuts are common — holds, and its
motivating relation is more common than it said, not less.

## What each descriptor resolves

Candidates are bucketed **within a molecule**, because that is the set the intensity head ranks
against one another. "Mean per bucket" is candidate-weighted: 1.00 would mean the descriptor
identifies every candidate.

| descriptor | mean per bucket | singletons |
|---|---:|---:|
| cut count (what the token had) | 47.48 | 0.001 |
| cut count + fragment size | 7.80 | 0.084 |
| **cut-bond sum** (`mass_shatter_candidate_cut_bond_features`) | **22.55** | 0.027 |
| cut-bond sum + adjacency count | 12.28 | 0.074 |
| **cut-bond sum + adjacency, split retained/lost** | **10.44** | 0.099 |
| cut-bond sum + adjacency + both ring terms | 10.41 | 0.102 |
| size + cut count + cut-bond sum | 4.30 | 0.237 |
| size + cut count + cut-bond sum + adjacency | 3.16 | 0.332 |
| size + cut count + cut-bond sum + adjacency + ring terms | 3.14 | 0.335 |

The cut-bond sum strictly contains the cut count (the bond-order one-hots sum to it), so the rows
are nested: the pooled bond features halve the bucket, **47.48 -> 22.55**, and adjacency halves it
again, **22.55 -> 10.44**. The two ring terms then move it by **0.03**, and by **0.02** once size
and cut count are in the key.

## Where the ring terms went

Residual ambiguity **given** the summed cut-bond vector — the share of candidates sharing a
molecule and a cut-bond sum with a candidate that disagrees about each property:

| property | ambiguous share |
|---|---:|
| candidates sharing a cut-bond sum at all | 0.973 |
| ring-sharing flag | **0.000** |
| adjacency flag | **0.489** |
| adjacency count | 0.795 |
| adjacency split, retained vs lost | 0.800 |
| ring-shared pair count | 0.020 |
| distinct rings cut | 0.021 |

The sum carries `b_ring` counts, so "two of the cuts were ring bonds" is already in the token, and
on this population that never disagrees with "two cuts share a ring" — a molecule whose cut set
contains two ring bonds in *different* rings is rare enough not to appear. Adjacency is the
opposite: no bond feature says whether two cut bonds touch, and the boundary block cannot recover
it either, because a mean over per-bond encodings cannot detect that the same atom appears twice.

The retained/lost split is worth its second column (12.28 -> 10.44) and is chemically distinct: a
shared **retained** atom means one atom of the fragment lost two substituents; a shared **lost**
atom means the fragment expelled a bridging atom, which is the CO / CH2 / S extrusion shape.

## What this changed

Built as `mass_shatter_candidate_cut_adjacency_features` (two columns, no parameters, no new graph
field), rather than the plan's three ring-topology scalars. The ring terms would need SSSR ring
identity per bond, which is **not** on the graph today — a preprocessing field and a cache
regeneration — to buy 0.02–0.03 of bucket resolution. That trade is closed by this measurement.

## Limits

- **Uniform bond logits.** Mask ranking then depends only on how many bonds a mask cuts, which is
  the model-independent part of the enumeration. A trained proposer concentrates on a few
  breakable bonds, which would push mass toward higher cut counts among them, so the cut-count
  distribution here is a lower bound rather than a mean. The relational shares are properties of
  the enumerated geometry and are less exposed to this.
- **Identifiability is not generalization, and its sign is not obvious.** The 2026-07-23 bond-set
  probe found the *least* identifiable fragment descriptor generalized best (`het_only`
  dom_mismatch 0.4046 vs cc 0.4732, and the smallest train/validation gap), and that a learned
  cut-bond encoder overfit *more* than the static one. So "resolves more candidates" is evidence
  that the head cannot otherwise obtain the information — not evidence that adding it raises
  validation cosine. That is what arms `e1`, `e4` and `e5` are for.
- 400 molecules, no model in the loop, no spectra: this measures the representation, not the fit.
