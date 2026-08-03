# Frag-Gen + SMARTS Prepass

> **Status:** REFERENCE &nbsp;·&nbsp; **Updated:** 2026-04-05
>
> How BFS + SMARTS prepass build the fragment DAG.

## Overview

The fragmentation generation (frag-gen) pipeline builds a DAG of molecular
fragments for each molecule.  Two mechanisms contribute fragments:

1. **BFS** (`compute_frags.compute_ccs`) — exhaustive single-bond-cut
   breadth-first search from the root molecule up to `max_depth` levels.
2. **SMARTS prepass** (`_apply_smarts_prepass`) — rule-based rearrangement
   reactions that require **at least 2 bond cuts plus 1 new bond formation**,
   and therefore cannot be produced by BFS alone.

This document describes the prepass mechanism, its depth semantics, and the
contract between the two stages.

---

## Why a Prepass?

BFS can only break existing bonds; it cannot form new ones.  Some
gas-phase rearrangement reactions in MS/MS involve a bond migration step
that BFS is structurally unable to reproduce:

| Rule | Reaction | Cuts | New bond |
|---|---|---|---|
| `crf12_0` | Acyclic 1,3-elimination (single bond) `A–B–C–D → A·D + B=C` | 2 (A–B, C–D) | A–D merged |
| `crf12_1` | Acyclic 1,3-elimination (double bond) `A–B=C–D → A·D + B≡C` | 2 (A–B, C–D) | A–D merged |
| `crf13_2` | Acyclic ester CO₂ loss `R–C(=O)–O–R' → CO₂ + R·R'` | 2 (R–C, O–R') | R–R' merged |

All three rules produce exactly **two fragments** from the root molecule.  The
"outer" fragment (containing A and D, or R and R') is a **virtual** fragment:
it is a connected component that only exists after the rearrangement because a
new bond is formed between the two outer atoms.

---

## Depth Semantics in the DAG

The DAG `nodes_depth_matrix` encodes a one-hot depth label per fragment.
Depth has the following meaning in the **single-cut BFS** (`multi_cut_bfs=False`)
context:

```
depth 0  →  root molecule (the full precursor)
depth 1  →  one bond cut from the root (2 CCs)
depth 2  →  one bond cut from a depth-1 fragment
depth d  →  one bond cut from a depth-(d-1) fragment
```

Depth is the **minimum number of single-bond-cut BFS steps** needed to reach a
fragment from the root.

### Prepass fragments belong at depth 2

The prepass rules require **two bond cuts** from the root.  Assigning them
depth 1 is incorrect: it conflates them with single-bond-cut fragments and
misrepresents their structural distance from the root.

Assigning depth 2 is semantically correct:
- Two bonds broken from root = two BFS steps = depth 2.
- The new bond formation is a rearrangement on top of the two cuts; it does not
  reduce the cut count.
- Depth 2 makes the model aware that prepass fragments are more "processed"
  than depth-1 fragments; they should receive lower prior probability as direct
  precursor peaks.

**DAG structure for a prepass fragment F produced from root R:**
```
R (depth 0)
└── F (depth 2)   ← edge min_depth = 1 (the edge appears at step 1 of
                    the prepass, but F's own depth label is 2)
```

The DAG edge (R → F) still exists at depth 1 of the DAG traversal (the edge
is a direct root-to-child edge), but the **node depth label** must be 2 to
correctly encode the two-cut requirement.

### Edge vs node depth

`dag_frag_meta["nodes_min_depth"]` stores the minimum depth at which each
node first appears.  For prepass-injected fragments this must be **2**, not 1.

`dag_frag_meta["edges_min_depth"]` stores the minimum depth at which each
edge first appears.  For the root → prepass-fragment edge, this is **1**
(the edge is introduced in the same BFS expansion pass as depth-1 fragments,
not depth 2, because the prepass is applied directly to the root).

---

## Prepass Scope

The prepass currently operates **only on the root molecule** (depth 0).
Rules are not applied recursively to depth-1 or deeper BFS fragments.

**Rationale:** the rules target whole-molecule rearrangements that are most
common before any bond-cleavage fragmentation has occurred.  Applying rules
to already-fragmented intermediates would require a recursive expansion that
has not been validated.

---

## Interaction with `multi_cut_bfs`

When `multi_cut_bfs=True`, SMARTS pairs are passed as `smarts_seed_pairs` to
the Cython BFS, which injects them as depth-1 seeds.  This is a different
depth convention: in `multi_cut_bfs`, depth encodes BFS *expansion rounds*, not
individual bond cuts, and a round can cut up to `max_cut_size` bonds.

The depth-2 rule described above applies to the **`multi_cut_bfs=False`**
(single-cut BFS + `_inject_smarts_pairs_into_dag`) path only.

---

## Implementation: `_inject_smarts_pairs_into_dag`

The relevant lines in [frag_utils.py](../../src/fragnnet/utils/frag_utils.py):

```python
# BEFORE (incorrect — depth 1):
depth_row[min(1, max_depth_cols - 1)] = 1
new_min_depths.append(1)
new_edge_min_depths.append(1)

# AFTER (correct — node depth 2, edge depth 1):
depth_row[min(2, max_depth_cols - 1)] = 1
new_min_depths.append(2)
new_edge_min_depths.append(1)   # edge stays at 1: root→fragment edge appears at step 1
```

**When `max_depth < 2`** (rare), the node depth is clamped to `max_depth_cols - 1`
so the one-hot vector remains valid.

---

## Summary Table

| Fragment source | Depth label | Edge min_depth | New bond? |
|---|---|---|---|
| BFS single cut from root | 1 | 1 | No |
| BFS single cut from depth-1 fragment | 2 | 2 | No |
| SMARTS prepass from root (crf12_x, crf13_2) | **2** | 1 | **Yes** |
