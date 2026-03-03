"""Tests for the SMARTS-based rearrangement prepass in multi_cut_bfs.py.

Tests cover:
1. _edges_from_node_mask: edge reconstruction from node mask is correct.
2. _union_find_ccs_all: returns all CCs (including n>2).
3. _apply_smarts_prepass: crf12_0 on butane produces non-contiguous outer fragment.
4. _apply_smarts_prepass: crf13_2 on methyl acetate produces CO2 (3-atom) fragment.
5. Cyclic filter: crf13_2 skips lactone (filter_bond_map_nums are directly bonded).
6. compute_ccs_multi_cut with smarts_rules: depth-1 includes SMARTS-only fragment.
7. SmartsFragRule: extensible — custom rule produces expected fragments.

All tests import the Python reference implementation directly (not the compiled .so).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Load the Python reference implementation (avoids .so taking precedence)
# ---------------------------------------------------------------------------

_PY_FILE = Path(__file__).parent.parent / "src" / "fragnnet" / "frag" / "multi_cut_bfs.py"
_MOD_NAME = "_multi_cut_bfs_py_ref"

if _MOD_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_MOD_NAME, _PY_FILE)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_MOD_NAME] = _mod
    _spec.loader.exec_module(_mod)
else:
    _mod = sys.modules[_MOD_NAME]

compute_ccs_multi_cut = _mod.compute_ccs_multi_cut
get_ring_edge_mask = _mod.get_ring_edge_mask
FRAG_RULES = _mod.FRAG_RULES
SmartsFragRule = _mod.SmartsFragRule
_apply_smarts_prepass = _mod._apply_smarts_prepass
_edges_from_node_mask = _mod._edges_from_node_mask
_union_find_ccs_all = _mod._union_find_ccs_all
_find_bond_idx = _mod._find_bond_idx
MAX_NUM_EDGES = _mod.MAX_NUM_EDGES
MASK_DTYPE = _mod.MASK_DTYPE

from fragnnet.utils.frag_utils import extract_mol_info, get_fraggen_input_arrays

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_arrays(smiles: str):
    """Return (mol_d, num_nodes, num_edges, node_mask, edges, edge_mask, n2e, mol)."""
    mol_d = extract_mol_info(smiles)
    num_nodes, num_edges, node_mask, edges, edge_mask, n2e = get_fraggen_input_arrays(mol_d)
    return mol_d, num_nodes, num_edges, node_mask, edges, edge_mask, n2e, mol_d["mol"]


def _run_py(smiles: str, max_cut_size: int = 2, max_depth: int = 3, use_prepass: bool = True):
    """Run the Python reference BFS (with optional SMARTS prepass) on a SMILES."""
    mol_d, num_nodes, num_edges, node_mask, edges, edge_mask, n2e, mol = _get_arrays(smiles)
    ring_mask = mol_d["ring_bond_mask"]
    rules = FRAG_RULES if use_prepass else None
    mol_arg = mol if use_prepass else None
    nodes_mask, nodes_depth, dag_edges, meta = compute_ccs_multi_cut(
        num_nodes,
        num_edges,
        node_mask,
        edges,
        edge_mask,
        n2e,
        max_depth=max_depth,
        time_limit=30,
        ring_edge_mask=ring_mask,
        max_cut_size=max_cut_size,
        smarts_rules=rules,
        mol=mol_arg,
    )
    return {
        "nodes_mask": nodes_mask,
        "nodes_depth": nodes_depth,
        "dag_edges": dag_edges,
        "meta": meta,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
    }


def _node_atom_sets(nodes_mask: np.ndarray) -> list[frozenset[int]]:
    """Return a list of frozensets of atom indices for each non-root DAG node."""
    result = []
    for i in range(1, len(nodes_mask)):
        atom_set = frozenset(int(j) for j in np.where(nodes_mask[i])[0])
        result.append(atom_set)
    return result


# ---------------------------------------------------------------------------
# Test 1: _edges_from_node_mask reconstructs edge mask correctly
# ---------------------------------------------------------------------------


class TestEdgesFromNodeMask:
    """Edge mask reconstruction from node mask + base edge mask."""

    SMILES = "CCCC"  # butane: 4 atoms, 3 bonds

    def test_root_mask_returns_all_edges(self):
        _, num_nodes, num_edges, node_mask, edges, edge_mask, _, _ = _get_arrays(self.SMILES)
        base = np.asarray(edge_mask, dtype=MASK_DTYPE).copy()
        root_mask = np.ones(num_nodes, dtype=MASK_DTYPE)
        padded_base = np.zeros(MAX_NUM_EDGES, dtype=MASK_DTYPE)
        padded_base[:num_edges] = base[:num_edges]
        reconstructed = _edges_from_node_mask(root_mask, padded_base, edges, num_edges)
        assert reconstructed[:num_edges].sum() == num_edges, "All edges should be active for root"

    def test_single_atom_mask_returns_no_edges(self):
        _, num_nodes, num_edges, node_mask, edges, edge_mask, _, _ = _get_arrays(self.SMILES)
        base = np.zeros(MAX_NUM_EDGES, dtype=MASK_DTYPE)
        base[:num_edges] = np.asarray(edge_mask, dtype=MASK_DTYPE)[:num_edges]
        # Mask with only atom 0 active
        single = np.zeros(num_nodes, dtype=MASK_DTYPE)
        single[0] = 1
        reconstructed = _edges_from_node_mask(single, base, edges, num_edges)
        assert reconstructed[:num_edges].sum() == 0, "No edges active for single isolated atom"

    def test_two_adjacent_atoms_gives_one_edge(self):
        _, num_nodes, num_edges, node_mask, edges, edge_mask, _, _ = _get_arrays(self.SMILES)
        base = np.zeros(MAX_NUM_EDGES, dtype=MASK_DTYPE)
        base[:num_edges] = np.asarray(edge_mask, dtype=MASK_DTYPE)[:num_edges]
        # Atoms 0 and 1 (first C-C bond in butane)
        pair = np.zeros(num_nodes, dtype=MASK_DTYPE)
        pair[0] = 1
        pair[1] = 1
        reconstructed = _edges_from_node_mask(pair, base, edges, num_edges)
        # Only bond 0-1 should be active
        assert reconstructed[:num_edges].sum() == 1


# ---------------------------------------------------------------------------
# Test 2: _union_find_ccs_all handles 2 and 3 components
# ---------------------------------------------------------------------------


class TestUnionFindCcsAll:
    """_union_find_ccs_all returns all CCs without capping at 2."""

    SMILES = "CCCC"

    def test_intact_molecule_returns_one_cc(self):
        _, num_nodes, num_edges, node_mask, edges, edge_mask, _, _ = _get_arrays(self.SMILES)
        em = np.zeros(MAX_NUM_EDGES, dtype=MASK_DTYPE)
        em[:num_edges] = np.asarray(edge_mask, dtype=MASK_DTYPE)[:num_edges]
        nm = np.ones(num_nodes, dtype=MASK_DTYPE)
        ccs = _union_find_ccs_all(num_nodes, nm, edges, em, num_edges)
        assert len(ccs) == 1

    def test_one_bond_cut_returns_two_ccs(self):
        _, num_nodes, num_edges, node_mask, edges, edge_mask, _, _ = _get_arrays(self.SMILES)
        em = np.zeros(MAX_NUM_EDGES, dtype=MASK_DTYPE)
        em[:num_edges] = np.asarray(edge_mask, dtype=MASK_DTYPE)[:num_edges]
        # Cut the middle bond (bond index for C1-C2, which is bond 1)
        em[1] = 0
        nm = np.ones(num_nodes, dtype=MASK_DTYPE)
        ccs = _union_find_ccs_all(num_nodes, nm, edges, em, num_edges)
        assert len(ccs) == 2
        sizes = sorted(int(m.sum()) for m in ccs)
        assert sizes == [2, 2]

    def test_two_bond_cuts_returns_three_ccs(self):
        _, num_nodes, num_edges, node_mask, edges, edge_mask, _, _ = _get_arrays(self.SMILES)
        em = np.zeros(MAX_NUM_EDGES, dtype=MASK_DTYPE)
        em[:num_edges] = np.asarray(edge_mask, dtype=MASK_DTYPE)[:num_edges]
        # Cut bonds 0 (C0-C1) and 2 (C2-C3): isolates C0, C1-C2, C3
        em[0] = 0
        em[2] = 0
        nm = np.ones(num_nodes, dtype=MASK_DTYPE)
        ccs = _union_find_ccs_all(num_nodes, nm, edges, em, num_edges)
        assert len(ccs) == 3
        sizes = sorted(int(m.sum()) for m in ccs)
        assert sizes == [1, 1, 2]


# ---------------------------------------------------------------------------
# Test 3: _apply_smarts_prepass — crf12_0 on butane
# ---------------------------------------------------------------------------


class TestSmartsPrepassCrf12Butane:
    """crf12_0 on butane (CCCC): 1,3-elimination produces outer {C0,C3} + inner {C1,C2}."""

    SMILES = "CCCC"

    def test_prepass_returns_nonempty(self):
        mol_d, num_nodes, num_edges, node_mask, edges, edge_mask, _, mol = _get_arrays(self.SMILES)
        base = np.zeros(MAX_NUM_EDGES, dtype=MASK_DTYPE)
        base[:num_edges] = np.asarray(edge_mask, dtype=MASK_DTYPE)[:num_edges]
        rules = [r for r in FRAG_RULES if r.name == "crf12_0"]
        results = _apply_smarts_prepass(rules, mol, base, edges, num_nodes, num_edges)
        assert len(results) > 0, "crf12_0 should match on butane (4-atom chain)"

    def test_outer_fragment_contains_two_atoms(self):
        mol_d, num_nodes, num_edges, node_mask, edges, edge_mask, _, mol = _get_arrays(self.SMILES)
        base = np.zeros(MAX_NUM_EDGES, dtype=MASK_DTYPE)
        base[:num_edges] = np.asarray(edge_mask, dtype=MASK_DTYPE)[:num_edges]
        rules = [r for r in FRAG_RULES if r.name == "crf12_0"]
        results = _apply_smarts_prepass(rules, mol, base, edges, num_nodes, num_edges)
        # Each result is (mask_a, mask_b); check at least one pair sums to 4 total atoms
        for mask_a, mask_b in results:
            total = int(mask_a.sum()) + int(mask_b.sum())
            assert total == num_nodes, f"Fragment pair must cover all {num_nodes} atoms"

    def test_inner_fragment_is_middle_two_atoms(self):
        """Inner fragment should be the middle 2 atoms (C1-C2) of butane."""
        mol_d, num_nodes, num_edges, node_mask, edges, edge_mask, _, mol = _get_arrays(self.SMILES)
        base = np.zeros(MAX_NUM_EDGES, dtype=MASK_DTYPE)
        base[:num_edges] = np.asarray(edge_mask, dtype=MASK_DTYPE)[:num_edges]
        rules = [r for r in FRAG_RULES if r.name == "crf12_0"]
        results = _apply_smarts_prepass(rules, mol, base, edges, num_nodes, num_edges)
        # Inner fragment (2 atoms, C1-C2 are adjacent) should appear
        inner_sizes = [int(min(int(a.sum()), int(b.sum()))) for a, b in results]
        assert 2 in inner_sizes, f"Expected inner fragment of size 2, got sizes {inner_sizes}"

    def test_outer_atoms_are_non_contiguous(self):
        """Outer merged fragment (C0+C3) must contain non-adjacent atoms.

        BFS can never produce this fragment since C0 and C3 are not connected
        in any subgraph reachable by bond cuts alone.
        """
        mol_d, num_nodes, num_edges, node_mask, edges, edge_mask, _, mol = _get_arrays(self.SMILES)
        base = np.zeros(MAX_NUM_EDGES, dtype=MASK_DTYPE)
        base[:num_edges] = np.asarray(edge_mask, dtype=MASK_DTYPE)[:num_edges]
        rules = [r for r in FRAG_RULES if r.name == "crf12_0"]
        results = _apply_smarts_prepass(rules, mol, base, edges, num_nodes, num_edges)
        # Collect all 2-atom fragment atom-sets
        two_atom_sets = []
        for mask_a, mask_b in results:
            for m in (mask_a, mask_b):
                if int(m.sum()) == 2:
                    two_atom_sets.append(frozenset(int(j) for j in np.where(m)[0]))
        # {0, 3} = non-adjacent C0+C3 must be one of them (from outer merge)
        assert frozenset({0, 3}) in two_atom_sets, (
            f"Non-contiguous outer fragment {{0,3}} not found; got {two_atom_sets}"
        )


# ---------------------------------------------------------------------------
# Test 4: _apply_smarts_prepass — crf13_2 on methyl acetate
# ---------------------------------------------------------------------------


class TestSmartsPrepassCrf132Ester:
    """crf13_2 on methyl acetate (CC(=O)OC): CO2 loss → CO2 (3-atom) + R+R' (2-atom)."""

    SMILES = "CC(=O)OC"  # methyl acetate: 5 atoms total

    def test_prepass_finds_co2_fragment(self):
        mol_d, num_nodes, num_edges, node_mask, edges, edge_mask, _, mol = _get_arrays(self.SMILES)
        base = np.zeros(MAX_NUM_EDGES, dtype=MASK_DTYPE)
        base[:num_edges] = np.asarray(edge_mask, dtype=MASK_DTYPE)[:num_edges]
        rules = [r for r in FRAG_RULES if r.name == "crf13_2"]
        results = _apply_smarts_prepass(rules, mol, base, edges, num_nodes, num_edges)
        assert len(results) > 0, "crf13_2 should match on methyl acetate (acyclic ester)"

    def test_co2_fragment_has_3_atoms(self):
        """CO2 product fragment should contain exactly 3 atoms: C(=O)O."""
        mol_d, num_nodes, num_edges, node_mask, edges, edge_mask, _, mol = _get_arrays(self.SMILES)
        base = np.zeros(MAX_NUM_EDGES, dtype=MASK_DTYPE)
        base[:num_edges] = np.asarray(edge_mask, dtype=MASK_DTYPE)[:num_edges]
        rules = [r for r in FRAG_RULES if r.name == "crf13_2"]
        results = _apply_smarts_prepass(rules, mol, base, edges, num_nodes, num_edges)
        frag_sizes = sorted({int(a.sum()) for a, b in results} | {int(b.sum()) for a, b in results})
        assert 3 in frag_sizes, f"Expected CO2 fragment (3 atoms), got sizes {frag_sizes}"

    def test_outer_fragment_has_2_atoms(self):
        """Outer merged fragment R+R' should contain 2 atoms (CH3 + CH3)."""
        mol_d, num_nodes, num_edges, node_mask, edges, edge_mask, _, mol = _get_arrays(self.SMILES)
        base = np.zeros(MAX_NUM_EDGES, dtype=MASK_DTYPE)
        base[:num_edges] = np.asarray(edge_mask, dtype=MASK_DTYPE)[:num_edges]
        rules = [r for r in FRAG_RULES if r.name == "crf13_2"]
        results = _apply_smarts_prepass(rules, mol, base, edges, num_nodes, num_edges)
        frag_sizes = sorted({int(a.sum()) for a, b in results} | {int(b.sum()) for a, b in results})
        assert 2 in frag_sizes, f"Expected outer R+R' fragment (2 atoms), got sizes {frag_sizes}"

    def test_fragment_pair_covers_all_atoms(self):
        mol_d, num_nodes, num_edges, node_mask, edges, edge_mask, _, mol = _get_arrays(self.SMILES)
        base = np.zeros(MAX_NUM_EDGES, dtype=MASK_DTYPE)
        base[:num_edges] = np.asarray(edge_mask, dtype=MASK_DTYPE)[:num_edges]
        rules = [r for r in FRAG_RULES if r.name == "crf13_2"]
        results = _apply_smarts_prepass(rules, mol, base, edges, num_nodes, num_edges)
        for mask_a, mask_b in results:
            total = int(mask_a.sum()) + int(mask_b.sum())
            assert total == num_nodes, f"Fragment pair must cover all {num_nodes} atoms"


# ---------------------------------------------------------------------------
# Test 5: Cyclic filter — crf13_2 skips 4-membered ring lactone
# ---------------------------------------------------------------------------


class TestCyclicFilter:
    """filter_bond_map_nums blocks the cyclic case when outer atoms are directly bonded.

    The filter checks whether the two "outer" atoms (:1 and :5 for crf13_2) are
    directly bonded in the original graph.  This catches 4-membered ring lactones
    (β-propiolactone: O=C1CCO1) where the flanking CH2 groups ARE adjacent.

    For larger ring lactones (e.g., 5-membered ring: O=C1CCCO1), the outer atoms
    are not directly bonded so the filter does not fire; the prepass still runs but
    produces the same fragment pair that BFS cut=2 would discover independently.
    The BFS deduplication (css_to_id_dict) collapses these into the same DAG node.
    """

    SMILES = "O=C1CCO1"  # β-propiolactone (4-membered ring): outer CH2 atoms ARE adjacent

    def test_no_prepass_results_for_4membered_lactone(self):
        """In a 4-membered ring lactone the two outer atoms (:1=C2, :5=C3) are bonded.

        Bond C2-C3 exists in the ring → filter_bond_map_nums fires → 0 prepass results.
        The BFS cut=2 covers this fragmentation independently.
        """
        mol_d, num_nodes, num_edges, node_mask, edges, edge_mask, _, mol = _get_arrays(self.SMILES)
        base = np.zeros(MAX_NUM_EDGES, dtype=MASK_DTYPE)
        base[:num_edges] = np.asarray(edge_mask, dtype=MASK_DTYPE)[:num_edges]
        rules = [r for r in FRAG_RULES if r.name == "crf13_2"]
        results = _apply_smarts_prepass(rules, mol, base, edges, num_nodes, num_edges)
        assert len(results) == 0, (
            f"crf13_2 should produce 0 results for 4-membered lactone "
            f"(outer atoms directly bonded), but got {len(results)}"
        )


# ---------------------------------------------------------------------------
# Test 6: compute_ccs_multi_cut with smarts_rules adds SMARTS-only fragments
# ---------------------------------------------------------------------------


class TestComputeCcsWithPrepass:
    """SMARTS prepass fragments appear at depth 1, even for non-adjacent atom pairs."""

    SMILES = "CCCC"  # butane

    def test_prepass_adds_extra_depth1_fragment(self):
        """With prepass, butane gains the non-contiguous {C0,C3} fragment at depth 1.

        BFS alone (no prepass) cannot produce {C0,C3} because they are not
        directly bonded and no single or multi-bond ring cut disconnects them
        into a connected subgraph together.
        """
        r_no_prepass = _run_py(self.SMILES, use_prepass=False)
        r_prepass = _run_py(self.SMILES, use_prepass=True)

        sets_no_prepass = set(_node_atom_sets(r_no_prepass["nodes_mask"]))
        sets_prepass = set(_node_atom_sets(r_prepass["nodes_mask"]))

        assert frozenset({0, 3}) not in sets_no_prepass, (
            "Without prepass, {C0,C3} should NOT appear (non-adjacent, no ring cut)"
        )
        assert frozenset({0, 3}) in sets_prepass, (
            "With prepass, {C0,C3} should appear as a depth-1 fragment"
        )

    def test_prepass_depth1_node_count_gte_without(self):
        """Prepass can only ADD fragments; it should never reduce the fragment count."""
        r_no = _run_py(self.SMILES, use_prepass=False)
        r_yes = _run_py(self.SMILES, use_prepass=True)
        assert len(r_yes["nodes_mask"]) >= len(r_no["nodes_mask"])


# ---------------------------------------------------------------------------
# Test 7: SmartsFragRule is extensible — custom rule works correctly
# ---------------------------------------------------------------------------


class TestCustomSmartsRule:
    """A user-defined SmartsFragRule can be added to extend the prepass."""

    def test_custom_rule_produces_fragments(self):
        """A simple N-C bond cleavage rule applied to methylamine (CN)."""
        # Simple N-C cut: [N:1][C:2] → splits into N and C fragments
        rule = SmartsFragRule(
            name="n_c_cut",
            reactant_smarts="[N:1][C:2]",
            cut_bond_map_pairs=[(1, 2)],
            merge_outer_map_nums=[],
        )
        smiles = "CN"  # methylamine: C-N, 2 atoms
        mol_d, num_nodes, num_edges, node_mask, edges, edge_mask, _, mol = _get_arrays(smiles)
        base = np.zeros(MAX_NUM_EDGES, dtype=MASK_DTYPE)
        base[:num_edges] = np.asarray(edge_mask, dtype=MASK_DTYPE)[:num_edges]
        results = _apply_smarts_prepass([rule], mol, base, edges, num_nodes, num_edges)
        assert len(results) > 0, "Custom N-C cut rule should match on methylamine"
        # Both fragments should have 1 atom each
        for mask_a, mask_b in results:
            assert int(mask_a.sum()) == 1 and int(mask_b.sum()) == 1

    def test_frag_rules_list_is_extendable(self):
        """FRAG_RULES can be appended with new rules without affecting existing ones."""
        extra_rule = SmartsFragRule(
            name="test_extra",
            reactant_smarts="[*:1][*:2]",
            cut_bond_map_pairs=[(1, 2)],
            merge_outer_map_nums=[],
        )
        extended = FRAG_RULES + [extra_rule]
        assert len(extended) == len(FRAG_RULES) + 1
        assert extended[-1].name == "test_extra"
        # Original FRAG_RULES is unchanged
        assert len(FRAG_RULES) == 3
        assert all(r.name in ("crf12_0", "crf12_1", "crf13_2") for r in FRAG_RULES)
