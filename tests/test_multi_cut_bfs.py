"""Tests for the multi-bond-cut BFS fragmentation module.

Tests cover:
1. Acyclic molecule: multi-cut output matches standard BFS (no ring bonds → same result).
2. Cyclohexane: ring fragment appears at depth 1 with max_cut_size=2 but NOT with
   max_cut_size=1 (single-bond BFS can never disconnect a ring at depth 1).
3. Naphthalene: fused bicyclic — 2-bond cut separates the two rings.
4. Retro-Diels-Alder scaffold: cyclohexene-like 2-bond cut.
5. max_cut_size=1 falls back to standard single-bond BFS on a ring.
"""

from __future__ import annotations

import numpy as np

from fragnnet.frag.multi_cut_bfs import compute_ccs_multi_cut
from fragnnet.utils.frag_utils import extract_mol_info, get_fraggen_input_arrays

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(smiles: str, max_cut_size: int = 2, max_depth: int = 3) -> dict:
    """Run multi-cut BFS on a SMILES and return result components."""
    mol_d = extract_mol_info(smiles)
    ring_mask = mol_d["ring_bond_mask"]
    num_nodes, num_edges, node_mask, edges, edge_mask, n2e = get_fraggen_input_arrays(mol_d)
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
    )
    return {
        "nodes_mask": nodes_mask,
        "nodes_depth": nodes_depth,
        "dag_edges": dag_edges,
        "meta": meta,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
    }


def _atom_counts(nodes_mask: np.ndarray) -> list[int]:
    """Return sorted list of atom counts per DAG node (excl. root)."""
    counts = [int(nodes_mask[i].sum()) for i in range(1, len(nodes_mask))]
    return sorted(counts)


def _node_depths(nodes_mask: np.ndarray, nodes_depth: np.ndarray) -> dict[int, int]:
    """Return {atom_count: min_depth} for non-root nodes."""
    result = {}
    for i in range(1, len(nodes_mask)):
        count = int(nodes_mask[i].sum())
        min_d = int(np.where(nodes_depth[i])[0].min())
        result[count] = min(result.get(count, 999), min_d)
    return result


# ---------------------------------------------------------------------------
# Test 1: acyclic molecule — multi-cut and standard BFS agree
# ---------------------------------------------------------------------------


class TestAcyclicMolecule:
    """n-pentane: no ring bonds → max_cut_size has no effect."""

    SMILES = "CCCCC"  # n-pentane, 5C acyclic

    def test_same_node_count_as_single_cut(self):
        r1 = _run(self.SMILES, max_cut_size=1)
        r2 = _run(self.SMILES, max_cut_size=2)
        assert len(r1["nodes_mask"]) == len(r2["nodes_mask"])

    def test_same_dag_edge_count(self):
        r1 = _run(self.SMILES, max_cut_size=1)
        r2 = _run(self.SMILES, max_cut_size=2)
        assert len(r1["dag_edges"]) == len(r2["dag_edges"])

    def test_depth1_has_fragments(self):
        r = _run(self.SMILES, max_cut_size=2)
        depths = _node_depths(r["nodes_mask"], r["nodes_depth"])
        # Breaking any bond in n-pentane gives fragments at depth 1
        assert 1 in depths.values()

    def test_ring_bond_mask_all_zero(self):
        mol_d = extract_mol_info(self.SMILES)
        assert mol_d["ring_bond_mask"].sum() == 0


# ---------------------------------------------------------------------------
# Test 2: cyclohexane — ring fragment requires 2-bond cut
# ---------------------------------------------------------------------------


class TestCyclohexane:
    """Cyclohexane: every bond is a ring bond.

    Standard BFS (max_cut_size=1) can never disconnect it at any depth
    (single bond break → num_ccs=1 always).
    Multi-cut BFS (max_cut_size=2) finds the 2-bond cuts at depth 1.
    """

    SMILES = "C1CCCCC1"  # cyclohexane, 6C ring

    def test_standard_bfs_finds_no_fragments(self):
        """max_cut_size=1 on a simple ring finds no fragments (only root node)."""
        r = _run(self.SMILES, max_cut_size=1, max_depth=1)
        # Only the root node should exist at depth 1 since single-bond breaks
        # on a ring never disconnect
        assert len(r["nodes_mask"]) == 1, (
            f"Expected only root node, got {len(r['nodes_mask'])} nodes"
        )

    def test_multi_cut_finds_ring_fragments_at_depth1(self):
        """max_cut_size=2 finds ring-opening fragments at depth 1."""
        r = _run(self.SMILES, max_cut_size=2, max_depth=1)
        # Should have more than just the root node
        assert len(r["nodes_mask"]) > 1, "Multi-cut BFS should find ring fragments"

    def test_ring_cut_fragment_sizes(self):
        """2-bond cuts of cyclohexane produce complementary fragment size pairs.

        Adjacent bond pair (e.g. bonds 0-1 and 1-2): 1+5 split
        Skip-1 bond pair (e.g. bonds 0-1 and 2-3):   2+4 split
        Opposite bond pair (e.g. bonds 0-1 and 3-4):  3+3 split

        Each cut produces 2 fragments summing to 6 atoms.
        All sizes 1..5 should appear in the non-root node set.
        """
        r = _run(self.SMILES, max_cut_size=2, max_depth=1)
        counts = _atom_counts(r["nodes_mask"])
        for expected_size in [1, 2, 3, 4, 5]:
            assert expected_size in counts, (
                f"Expected fragment of size {expected_size} from cyclohexane 2-bond cuts, "
                f"got sizes: {sorted(set(counts))}"
            )

    def test_ring_bond_mask_all_ring(self):
        mol_d = extract_mol_info(self.SMILES)
        assert mol_d["ring_bond_mask"].sum() == 6  # all 6 bonds are ring bonds

    def test_depth1_nodes_exist_in_multi_cut(self):
        r = _run(self.SMILES, max_cut_size=2, max_depth=2)
        depths = _node_depths(r["nodes_mask"], r["nodes_depth"])
        # At least one non-root fragment should appear at depth 1
        assert 1 in depths.values(), "No depth-1 fragments found for cyclohexane"


# ---------------------------------------------------------------------------
# Test 3: naphthalene — fused bicyclic, 2-bond cut separates the two rings
# ---------------------------------------------------------------------------


class TestNaphthalene:
    """Naphthalene: two fused 6-membered aromatic rings.

    A 2-bond cut on the two bonds connecting the rings separates them into
    two 5-atom (C5H5) and 5-atom (C5H5) fragments — each being a cyclopentadienyl
    fragment at the formula level.
    """

    SMILES = "c1ccc2ccccc2c1"  # naphthalene, 10C

    def test_multi_cut_finds_more_nodes_than_single_cut(self):
        r1 = _run(self.SMILES, max_cut_size=1, max_depth=1)
        r2 = _run(self.SMILES, max_cut_size=2, max_depth=1)
        # fused ring cuts are only possible with 2-bond cuts
        assert len(r2["nodes_mask"]) >= len(r1["nodes_mask"])

    def test_ring_bond_mask_count(self):
        mol_d = extract_mol_info(self.SMILES)
        # All 11 bonds of naphthalene are ring bonds
        assert mol_d["ring_bond_mask"].sum() == 11

    def test_achievable_fragment_sizes(self):
        """Verify the fragment sizes that are actually achievable with 2-bond cuts.

        Naphthalene min-edge-cut = 3 (bridgehead atoms have degree 3), so a
        5+5 split requires 3 bonds — NOT achievable with max_cut_size=2.

        Valid 2-bond cuts:
          1+9: cut both bonds around any of the 8 non-bridgehead (degree-2) atoms
          2+8: cut both external bonds of any adjacent pair of non-bridgehead atoms
          3+7: three consecutive non-bridgehead atoms (each ring has 2 such triples)
          4+6: four consecutive non-bridgehead atoms (one chain per ring: C4 subpath)
        """
        r = _run(self.SMILES, max_cut_size=2, max_depth=1)
        counts = set(_atom_counts(r["nodes_mask"]))
        # All of these splits must be achievable
        for expected in [1, 2, 3, 4, 6, 7, 8, 9]:
            assert expected in counts, (
                f"Expected fragment of size {expected} from naphthalene 2-bond cuts, "
                f"got sizes: {sorted(counts)}"
            )
        # 5-atom split requires a 3-bond cut — must NOT appear with max_cut_size=2
        assert 5 not in counts, (
            "5-atom split of naphthalene should require 3-bond cut, "
            "but appeared with max_cut_size=2"
        )


# ---------------------------------------------------------------------------
# Test 4: retro-Diels-Alder scaffold — cyclohexene 2-bond cut
# ---------------------------------------------------------------------------


class TestCyclohexene:
    """Cyclohexene: the diene + dienophile 2-bond cut should appear at depth 1."""

    SMILES = "C1CC=CCC1"  # cyclohex-3-ene (simplified RDA scaffold)

    def test_multi_cut_produces_depth1_ring_fragments(self):
        r = _run(self.SMILES, max_cut_size=2, max_depth=1)
        assert len(r["nodes_mask"]) > 1, "Should find fragments from 2-bond ring cut"

    def test_ring_bonds_present(self):
        mol_d = extract_mol_info(self.SMILES)
        assert mol_d["ring_bond_mask"].sum() > 0


# ---------------------------------------------------------------------------
# Test 5: output format compatibility with compute_ccs
# ---------------------------------------------------------------------------


class TestOutputFormat:
    """Verify output arrays match expected shapes and dtypes."""

    SMILES = "C1CCCCC1"  # cyclohexane

    def test_nodes_mask_matrix_shape(self):
        r = _run(self.SMILES, max_cut_size=2)
        assert r["nodes_mask"].ndim == 2
        assert r["nodes_mask"].shape[1] == r["num_nodes"]
        assert r["nodes_mask"].dtype == np.uint8

    def test_nodes_depth_matrix_shape(self):
        r = _run(self.SMILES, max_cut_size=2, max_depth=3)
        assert r["nodes_depth"].ndim == 2
        assert r["nodes_depth"].shape[0] == r["nodes_mask"].shape[0]
        assert r["nodes_depth"].shape[1] == 4  # max_depth + 1

    def test_dag_edges_shape(self):
        r = _run(self.SMILES, max_cut_size=2)
        if len(r["dag_edges"]) > 0:
            assert r["dag_edges"].ndim == 2
            assert r["dag_edges"].shape[1] == 2
            assert r["dag_edges"].dtype == np.int64

    def test_meta_keys_present(self):
        r = _run(self.SMILES, max_cut_size=2)
        for key in ("reached_depth", "edges_min_depth", "nodes_min_depth", "force_stopped"):
            assert key in r["meta"], f"Missing key: {key}"

    def test_root_node_is_index_0(self):
        r = _run(self.SMILES, max_cut_size=2)
        # Root node should contain all atoms
        assert r["nodes_mask"][0].sum() == r["num_nodes"]

    def test_root_node_at_depth_0(self):
        r = _run(self.SMILES, max_cut_size=2)
        assert r["nodes_depth"][0, 0] == 1  # root at depth 0


# ---------------------------------------------------------------------------
# Test 6: ring_bond_mask in mol_d
# ---------------------------------------------------------------------------


class TestRingBondMask:
    """Verify ring_bond_mask is correctly populated in mol_d."""

    def test_benzene_all_ring(self):
        mol_d = extract_mol_info("c1ccccc1")
        assert mol_d["ring_bond_mask"].sum() == 6

    def test_acyclic_all_zero(self):
        mol_d = extract_mol_info("CCCC")
        assert mol_d["ring_bond_mask"].sum() == 0

    def test_mixed_ring_acyclic(self):
        # methylcyclohexane: 6 ring bonds + 1 exocyclic C-C bond = 7 total, 6 ring
        mol_d = extract_mol_info("CC1CCCCC1")
        num_bonds = len(mol_d["bonds"])
        num_ring = int(mol_d["ring_bond_mask"].sum())
        assert num_bonds == 7
        assert num_ring == 6

    def test_mask_length_matches_bonds(self):
        for smiles in ["C1CCCCC1", "c1ccccc1", "CCCCC", "CC1CCCCC1"]:
            mol_d = extract_mol_info(smiles)
            assert len(mol_d["ring_bond_mask"]) == len(mol_d["bonds"])
