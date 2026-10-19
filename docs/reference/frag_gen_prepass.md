# Frag-Gen + SMARTS Prepass

> **Status:** REFERENCE &nbsp;·&nbsp; **Updated:** 2026-10-12
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

The relevant lines in [dag/builder.py](../../src/fragnnet/models/fragnnet/dag/builder.py):

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

---

## Rule design notes (from local memory, verified 2026-10-06)

Per-rule scaffolds and counts for `FRAG_RULES` / `NL_RULES` in
[smarts_prepass.py](../../src/fragnnet/fragmentation/smarts_prepass.py). Usage statistics are in
[smarts_usage_audit_2026_09_23.md](../findings/smarts_usage_audit_2026_09_23.md); MASSter's
adopted rules are in [masster_smarts_rules.md](masster_smarts_rules.md).

### Why the compound-cut rules exist

Each ring release costs two sequential cuts, so two independent ring events cost four, one over
`max_depth=3`. Each rule injects the 4-bond compound cut as one depth-2 leaf. Prepass fragments
are one-shot leaves off the root: there is **no further BFS expansion or rule chaining from an
injected node** (see Prepass Scope above), so a cascade such as fluorenone, then -CO, then -C2H2
needs one rule per distinct (C, O) heavy-atom composition. H-shift tolerance (h4) collapses peaks
that differ only by H count onto one mask.

| Rule (FRAG_RULES) | Scaffold and cut | NIST20 molecules |
| --- | --- | --- |
| `xanthone_ring_cleavage` (2026-06-30) | xanthone; cuts (2,3) (6,7) (9,10) (13,14), merges outer C4 pieces; keeps the central C5O2 pyranone | 56 of 56 scaffold molecules |
| `phenoxazine_ring_cleavage` | phenoxazine, same cut map; `[O+0]` because the bridge O is non-aromatic; keeps C4NO | 2 (mol_ids 1993, 13409) |
| `furanocoumarin_ring_cleavage` | linear psoralen; cuts (1,2) (3,4) (5,6) (7,8), merges [4, 8]; kept fragment is a 4-carbon open chain, not a ring; angelicin-type is not matched | 56 (43 bare psoralen) |
| `morphinan_ring_cleavage` | oxide-bridged morphinan; 4 cuts, n_ccs=2 so no `merge_outer_map_nums`; releases C10HxO2 | 14 classified plus 2 lookalikes; GT C10H8O2 in 20+ molecules; base BFS gap 28.8% vs mbfs 72.2% |
| `halogenated_quinoline_ring_cleavage` | quinoline plus any halogen (disconnected `.[F,Cl,Br,I]` component); excises ring N and one benzo carbon, merges [1, 7]; `min_num_atoms=11` | 96 |
| `benzopyranone_ring_cleavage` | coumarin, 4 cuts, `min_num_atoms=11`; `[OC]` is impossible in SMARTS, write `[O]` separately | 52 improved |
| `benzopyranone_bis_ch_loss`, `anthraquinone_ring_c4_plus_ch_loss`, `morphinan_amine_arm_loss`, `desoxymorphinan_ring_cleavage` (2026-07-01) | second cut topologies on existing scaffolds | 61 coumarins, 35 anthraquinones, 14 morphinans, 4 non-oxide morphinans |
| `desoxymorphinan_aromatic_core_release` (rule 16, 2026-07-02) | non-oxide morphinan; keeps the large N-free aromatic core (fluorenone/fluorenol C13O1) instead of the small N piece; `min_num_atoms=17`; +2 DAG nodes per molecule | 4 (1412, 15421, 20146, 34342) |

- **Extension method that works** (residual gap): for a ClassyFire subclass, filter to molecules
  where the rule already matches, rank by `mbfs_wrecall - smpTrue_wrecall`, take the top N, diff
  `_get_cc_mask_rows(mbfs_dag)` against the prepass DAG (helpers in
  `scripts/debug/dag/debug_compare_mbfs_dags.py`), group the mbfs-only masks by
  `(n_cut_bonds, n_ring_cut_bonds)` and count **distinct molecules**, not masks (combinatorial
  substituent cuts inflate mask counts). A signature whose molecule count equals N is the rule.
  Source notebook: `notebooks/frag_notebooks/nist20_mbfs_vs_prepass_h4.ipynb`.
- **Still open:** the oxide-bridged morphinans (about 29 of 33; morphine yields 19 mbfs-only
  C13O1 masks with no repeated topology). Needs aggregation across the full set before a rule can be
  written; the gap is moderate CE (NCE 75-90).
- **`uniquify` trap:** RDKit `GetSubstructMatches(uniquify=True)` keeps only one of two
  structurally valid embeddings for a query with internal ring symmetry matched against a molecule
  substituted on one side (also true of `anthraquinone_bis_co_loss`). Do not claim both mirror
  directions are covered without checking `uniquify=False`. Halogenated quinoline likewise returns
  1 or 2 matches depending on symmetry.
- **Do not force a rule the data does not support:** 2-aminobenzothiazole was asked for as a FRAG
  rule, but the 9-atom bicyclic core gives about 150 combinatorial "new" masks with no dominant
  topology, so the user agreed to switch to the NL workflow.

### Neutral-loss rules

NL rules fire when the product contains a **new bond**, so its atoms are disconnected in the
precursor graph (BFS cannot produce them). The retained atoms get an H cap at each free valence;
the correct formula lands at H_delta = -2. Rule principle from the NIST FT NCE 30-40 worst-case
audit (2026-06-20; case list in [smarts_surveys.md](smarts_surveys.md#gas-phase-msms-rearrangement-smarts-review)): the GNN only
re-ranks fragments already in the DAG, so any new-bond pathway needs a `NeutralLossRule` first.

| Rule | SMARTS and notes |
| --- | --- |
| `sulfoxide_so_loss` (47.9669) | `[#6:1]-[SX3:2](=[O:3])-[#6:4]`, `nl_map_nums=[2, 3]`; `SX3` excludes sulfones. **The original `=[O:3]` form was dead**: the standardizer rewrites sulfoxides to `[S+][O-]`, so `=[O:3]` matched 0 of 52,787 NIST23 molecules; the charge-tolerant form matches 171. The rule now uses `[#6:1]-[SX3:2](~[OX1:3])-[#6:4]`, which accepts both forms (see [masster_smarts_rules.md](masster_smarts_rules.md)). The unit tests passed only because they used raw SMILES. Test every pattern against `mol_utils.mol_from_smiles` output; N-oxides and sulfoxides are the hypervalent forms that bite. |
| `diarylvinyl_c2h2_loss` (26.0157) | `[c:1]-[CH;!R:2]=[CH;!R:3]-[c:4]`, `nl_map_nums=[2, 3]`, `min_num_atoms=8`. `!R` on both vinyl carbons is required, or indene's ring vinyl matches. Excludes chalcone, cinnamic acid, styrene, indene, acenaphthylene. |
| `diaryl_ketone_co_loss` (27.9949) | `[c:1]-[CX3;!R:2](=[O:3])-[c:4]`, `min_num_atoms=12`. For bridging ketones, cutting both Ar-CO bonds gives 3 CCs, and raw BFS accepts only n_ccs in {1, 2} (n_ccs=99 is discarded), so the merged `{Ar+Ar'}` mask is unreachable. `!R` excludes chromone, fluorenone and anthraquinone, whose 2-CC cut already reaches the mask. |
| `aminobenzothiazole_hcn_loss` (27.0109) | `[NX3H2]-[c:1]1[n:2]c2ccccc2s1`, `nl_map_nums=[1, 2]`, `min_num_atoms=10`. NL 27.011 seen in 13 of 14 halogenated 2-aminobenzothiazole spectra (mol_ids 27768-27795). The competing HNCS loss (58.983 Da, about 6 of 14) was not implemented. |

These four are disabled in production as of 2026-09-29 (all but the sulfoxide), see
[masster_smarts_rules.md](masster_smarts_rules.md).

### Ring-merge DAG postprocessing

[`scripts/debug/dag/run_dag_postproc_merge.py`](../../scripts/debug/dag/run_dag_postproc_merge.py) keeps every node of the plain d3 DAG tar and adds only the
leaf-depth nodes from the unrestricted-ring-opening multi-cut BFS tar (`..._mbfs_old`) whose
`(base_formula, h_min, h_max)` key is absent, after deduplicating the extra leaves among
themselves (about +21% nodes instead of 2x). Added nodes get a **placeholder root edge**, so it
is valid only for `frag_gnn_type: NodeMLP` / `pyg_edges: False`; do not point a `pyg_edges: True`
config at it without rebuilding the edges. Built 2026-07-08:
`data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse_ringmerge.tar` and
`config/fragnnet/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_ringmerge.yml`. Oracle ceilings are in
[dag_oracle_ceiling_nist20.md](../findings/fragnnet/dag_oracle_ceiling_nist20.md); a trained-model
result was still untested there.
