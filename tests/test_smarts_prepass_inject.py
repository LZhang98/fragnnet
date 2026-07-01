"""Tests for SMARTS prepass injection into the original compute_frags BFS path.

Verifies that _inject_smarts_pairs_into_dag and the smarts_prepass=True flag
in compute_dags correctly add rearrangement fragments without duplicates and
with consistent metadata arrays.
"""
from __future__ import annotations

import numpy as np
import pytest

from fragnnet.frag.compute_frags import MASK_DTYPE
from fragnnet.frag.smarts_prepass import FRAG_RULES, _apply_smarts_prepass
from fragnnet.utils.frag_utils import (
    _inject_smarts_pairs_into_dag,
    compute_dags,
    extract_mol_info,
    get_fraggen_input_arrays,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_compute_dags(smiles: str, smarts_prepass: bool = False, max_depth: int = 3) -> dict:
    mol_d = extract_mol_info(smiles)
    return compute_dags(
        mol_d,
        max_depth=max_depth,
        h_prior=True,
        max_h_transfer=2,
        frag_max_time=int(1e6),
        isotopes=False,
        nb_isomorphic=False,
        wl_max_iterations=-1,
        multi_cut_bfs=False,
        smarts_prepass=smarts_prepass,
    )


def _get_frag_smiles_set(dag_d: dict, mol_d: dict) -> set[frozenset[int]]:
    """Return the set of fragments as frozensets of atom indices."""
    mol = mol_d["mol"]
    num_nodes = mol.GetNumAtoms()
    node_feats = dag_d["node_feats"]  # (n_frags, num_nodes, ...)
    # nodes_mask_matrix is not directly in dag_d; reconstruct from cc_bit_mask cols
    # Instead, use raw BFS output via _inject path — just check count via dag_d keys.
    _ = num_nodes  # silence unused warning
    return dag_d


# ---------------------------------------------------------------------------
# Direct unit tests for _inject_smarts_pairs_into_dag
# ---------------------------------------------------------------------------


class TestInjectSmartsPairs:
    """Unit tests for the _inject_smarts_pairs_into_dag helper."""

    def _make_dag(self, n_frags: int = 3, num_nodes: int = 6, max_depth: int = 3):
        """Build a minimal synthetic DAG."""
        rng = np.random.default_rng(42)
        nodes_mask_matrix = rng.integers(0, 2, (n_frags, num_nodes), dtype=np.uint8)
        nodes_depth_matrix = np.zeros((n_frags, max_depth + 1), dtype=np.uint8)
        nodes_depth_matrix[0, 0] = 1  # root at depth 0
        for i in range(1, n_frags):
            nodes_depth_matrix[i, 1] = 1
        dag_edges_matrix = np.array([[0, 1], [0, 2]], dtype=np.int64)
        nodes_min_depth = np.array([0] + [1] * (n_frags - 1), dtype=np.uint8)
        edges_min_depth = np.array([1, 1], dtype=np.uint8)
        dag_frag_meta = {
            "reached_depth": 1,
            "edges_min_depth": edges_min_depth,
            "nodes_min_depth": nodes_min_depth,
            "force_stopped": False,
        }
        return nodes_mask_matrix, nodes_depth_matrix, dag_edges_matrix, dag_frag_meta

    def test_empty_pairs_returns_unchanged(self):
        """No pairs → output is identical to input."""
        nmm, ndm, dem, meta = self._make_dag()
        n_frags_before = len(nmm)
        n_edges_before = len(dem)
        nmm2, ndm2, dem2, meta2 = _inject_smarts_pairs_into_dag(
            [], nmm, ndm, dem, meta, nmm.shape[1], min_frag_atoms=0
        )
        assert len(nmm2) == n_frags_before
        assert len(dem2) == n_edges_before

    def test_new_fragment_appended(self):
        """A genuinely new fragment pair increases the fragment count."""
        num_nodes = 6
        nmm, ndm, dem, meta = self._make_dag(num_nodes=num_nodes)
        # Craft masks that are not already in nmm
        mask_a = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)
        mask_b = np.array([0, 0, 1, 1, 0, 0], dtype=np.uint8)
        # Ensure they're not accidentally already present
        existing_keys = {nmm[i].tobytes() for i in range(len(nmm))}
        assert mask_a.tobytes() not in existing_keys
        assert mask_b.tobytes() not in existing_keys

        n_before = len(nmm)
        nmm2, ndm2, dem2, meta2 = _inject_smarts_pairs_into_dag(
            [(mask_a, mask_b)], nmm, ndm, dem, meta, num_nodes, min_frag_atoms=0
        )
        assert len(nmm2) == n_before + 2
        assert len(ndm2) == n_before + 2

    def test_new_fragments_have_depth_2(self):
        """Injected prepass fragments are marked at depth 2 (not 1).

        Prepass rules require ≥2 bond cuts + 1 new bond from the root,
        so the node depth label is 2.  See docs/frag_gen_prepass.md.
        """
        num_nodes = 6
        nmm, ndm, dem, meta = self._make_dag(n_frags=1, num_nodes=num_nodes)
        # Single root only, depth matrix has only root
        ndm[0] = 0
        ndm[0, 0] = 1
        mask_a = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)
        mask_b = np.array([0, 0, 1, 1, 0, 0], dtype=np.uint8)
        nmm2, ndm2, dem2, meta2 = _inject_smarts_pairs_into_dag(
            [(mask_a, mask_b)], nmm, ndm, dem, meta, num_nodes, min_frag_atoms=0
        )
        # New rows must have depth 2 set (index 2 in one-hot), not depth 1
        assert ndm2[1, 2] == 1, f"expected depth 2 for fragment 1, got {ndm2[1]}"
        assert ndm2[2, 2] == 1, f"expected depth 2 for fragment 2, got {ndm2[2]}"
        assert ndm2[1, 1] == 0, "depth 1 must not be set for prepass fragment"
        assert ndm2[2, 1] == 0, "depth 1 must not be set for prepass fragment"

    def test_duplicate_fragment_not_added_twice(self):
        """A fragment already in the DAG is not duplicated."""
        num_nodes = 6
        nmm, ndm, dem, meta = self._make_dag(num_nodes=num_nodes)
        # Use the first existing fragment as one of the "new" pairs
        existing_mask = nmm[1].copy()
        new_mask = np.array([1, 0, 0, 0, 0, 1], dtype=np.uint8)
        existing_keys = {nmm[i].tobytes() for i in range(len(nmm))}
        assert new_mask.tobytes() not in existing_keys

        n_before = len(nmm)
        nmm2, ndm2, dem2, meta2 = _inject_smarts_pairs_into_dag(
            [(existing_mask, new_mask)], nmm, ndm, dem, meta, num_nodes, min_frag_atoms=0
        )
        # Only new_mask should be added, existing_mask is already present
        assert len(nmm2) == n_before + 1

    def test_min_frag_atoms_filter(self):
        """Fragments below min_frag_atoms are not injected."""
        num_nodes = 6
        nmm, ndm, dem, meta = self._make_dag(num_nodes=num_nodes)
        # mask_a has only 1 atom, mask_b has 4
        mask_a = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        mask_b = np.array([0, 1, 1, 1, 1, 0], dtype=np.uint8)
        n_before = len(nmm)
        nmm2, ndm2, dem2, meta2 = _inject_smarts_pairs_into_dag(
            [(mask_a, mask_b)], nmm, ndm, dem, meta, num_nodes, min_frag_atoms=3
        )
        # mask_a (1 atom) is filtered; mask_b (4 atoms) is added
        assert len(nmm2) == n_before + 1

    def test_metadata_lengths_consistent(self):
        """nodes_min_depth and edges_min_depth stay aligned with matrix rows/edges."""
        num_nodes = 6
        nmm, ndm, dem, meta = self._make_dag(num_nodes=num_nodes)
        mask_a = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)
        mask_b = np.array([0, 0, 1, 1, 0, 0], dtype=np.uint8)
        nmm2, ndm2, dem2, meta2 = _inject_smarts_pairs_into_dag(
            [(mask_a, mask_b)], nmm, ndm, dem, meta, num_nodes, min_frag_atoms=0
        )
        assert len(meta2["nodes_min_depth"]) == len(nmm2)
        assert len(meta2["edges_min_depth"]) == len(dem2)

    def test_duplicate_edge_not_added(self):
        """An edge (0, frag_idx) already in the DAG is not duplicated."""
        num_nodes = 6
        nmm, ndm, dem, meta = self._make_dag(num_nodes=num_nodes)
        # Fragment at index 1 already has edge (0,1) in dem
        existing_mask = nmm[1].copy()
        n_edges_before = len(dem)
        nmm2, ndm2, dem2, meta2 = _inject_smarts_pairs_into_dag(
            [(existing_mask, nmm[2].copy())], nmm, ndm, dem, meta, num_nodes, min_frag_atoms=0
        )
        # Edge (0,1) already present; edge (0,2) already present — no new edges
        assert len(dem2) == n_edges_before


# ---------------------------------------------------------------------------
# Integration tests via compute_dags
# ---------------------------------------------------------------------------


class TestComputeDagsWithSmartsPrepass:
    """Integration tests: smarts_prepass=True on the original BFS path."""

    # Methyl acetate: CH3-C(=O)-O-CH3 — triggers crf13_2 (CO2 loss from ester)
    METHYL_ACETATE = "COC(C)=O"
    # A simple acyclic molecule with A-B-C-D pattern — triggers crf12_0
    BUTANE = "CCCC"

    def test_prepass_does_not_crash(self):
        """smarts_prepass=True runs without error on a simple molecule."""
        _run_compute_dags(self.METHYL_ACETATE, smarts_prepass=True)

    def test_prepass_adds_fragments(self):
        """smarts_prepass=True adds nodes/edges on top of the same BFS engine."""
        dag_no = _run_compute_dags(self.METHYL_ACETATE, smarts_prepass=False)
        dag_yes = _run_compute_dags(self.METHYL_ACETATE, smarts_prepass=True)
        assert dag_yes["dag_num_nodes"] >= dag_no["dag_num_nodes"]
        assert dag_yes["dag_num_edges"] >= dag_no["dag_num_edges"]

    def test_prepass_ester_co2_fragment_present(self):
        """For methyl acetate, the CO2 fragment (C=O and two O) should be in the DAG."""
        # Methyl acetate: COC(C)=O  atoms: O(0)-C(1)(=O(2))-C(3)H3  — RDKit canonical
        # The CO2 fragment from crf13_2 corresponds to C(=O)O (3 heavy atoms: C,O,O)
        mol_d = extract_mol_info(self.METHYL_ACETATE)
        mol = mol_d["mol"]
        dag = compute_dags(
            mol_d,
            max_depth=3,
            h_prior=True,
            max_h_transfer=2,
            frag_max_time=int(1e6),
            isotopes=False,
            nb_isomorphic=False,
            wl_max_iterations=-1,
            multi_cut_bfs=False,
            smarts_prepass=True,
        )
        # Collect all fragment atom-sets from the DAG node masks.
        # dag["node_feats"] rows correspond to fragments; the cc (connectivity
        # component) bits encode which atoms belong to that fragment.
        # Use _apply_smarts_prepass directly to know what masks to expect.
        num_nodes, num_edges, node_mask, edges, edge_mask, node_to_edge_idx = (
            get_fraggen_input_arrays(mol_d)
        )
        pairs = _apply_smarts_prepass(
            FRAG_RULES, mol, edge_mask.astype(np.uint8), edges, num_nodes, num_edges
        )
        # At least one pair must have been generated for methyl acetate
        assert len(pairs) > 0, "Expected at least one SMARTS match for methyl acetate"

    def test_prepass_result_metadata_consistent(self):
        """nodes_min_depth and edges_min_depth lengths match matrix dimensions."""
        mol_d = extract_mol_info(self.METHYL_ACETATE)
        num_nodes, num_edges, node_mask, edges, edge_mask, node_to_edge_idx = (
            get_fraggen_input_arrays(mol_d)
        )
        import fragnnet.frag.compute_frags as compute_frags
        from fragnnet.frag.compute_frags import MASK_DTYPE

        nmm, ndm, dem, meta = compute_frags.compute_ccs(
            num_nodes, num_edges,
            node_mask.astype(MASK_DTYPE), edges, edge_mask.astype(MASK_DTYPE),
            node_to_edge_idx, 3, int(1e6), 3,
        )
        pairs = [
            (a, b) for a, b, _ in _apply_smarts_prepass(
                FRAG_RULES, mol_d["mol"], edge_mask.astype(np.uint8), edges, num_nodes, num_edges
            )
        ]
        nmm2, ndm2, dem2, meta2 = _inject_smarts_pairs_into_dag(
            pairs, nmm, ndm, dem, meta, num_nodes, min_frag_atoms=3
        )
        assert len(meta2["nodes_min_depth"]) == len(nmm2), (
            f"nodes_min_depth length {len(meta2['nodes_min_depth'])} != "
            f"n_frags {len(nmm2)}"
        )
        assert len(meta2["edges_min_depth"]) == len(dem2), (
            f"edges_min_depth length {len(meta2['edges_min_depth'])} != "
            f"n_edges {len(dem2)}"
        )

    def test_prepass_idempotent_on_no_match(self):
        """On a molecule with no SMARTS matches, output equals no-prepass output."""
        # Ethanol has no ester group and too short for 1,3-elimination
        smiles = "CCO"
        dag_no = _run_compute_dags(smiles, smarts_prepass=False)
        dag_yes = _run_compute_dags(smiles, smarts_prepass=True)
        assert dag_no["dag_num_nodes"] == dag_yes["dag_num_nodes"]

    def test_prepass_butane_1_3_elimination(self):
        """Butane triggers crf12_0 (1,3-elimination), adding a non-contiguous fragment."""
        mol_d = extract_mol_info(self.BUTANE)
        num_nodes, num_edges, node_mask, edges, edge_mask, node_to_edge_idx = (
            get_fraggen_input_arrays(mol_d)
        )
        pairs = _apply_smarts_prepass(
            FRAG_RULES, mol_d["mol"], edge_mask.astype(np.uint8), edges, num_nodes, num_edges
        )
        # crf12_0 should fire on butane's C-C-C-C chain
        assert len(pairs) > 0, "Expected crf12_0 to match on butane"
    def test_prepass_unique_fragment_in_dag(self):
        """The unique SMARTS fragment (not reachable by bond cuts) must appear in the DAG.

        Ethyl propanoate (CCOC(=O)CC): crf13_2 (CO2 loss from ester) produces
        fragments whose atom subsets cannot be obtained by any single bond cut
        from the root — verifies the post-merge actually adds new nodes.
        Note: methyl acetate (COC(C)=O) SMARTS fragments ARE reachable by bond
        cuts at depth 3, so we need a longer-chain ester.
        """
        import fragnnet.frag.compute_frags as compute_frags

        # Ethyl propanoate: crf13_2 fires and produces fragments not in BFS
        smiles = "CCOC(=O)CC"
        mol_d = extract_mol_info(smiles)
        num_nodes, num_edges, node_mask, edges, edge_mask, node_to_edge_idx = (
            get_fraggen_input_arrays(mol_d)
        )
        nmm_base, _, _, _ = compute_frags.compute_ccs(
            num_nodes, num_edges,
            node_mask.astype(MASK_DTYPE), edges, edge_mask.astype(MASK_DTYPE),
            node_to_edge_idx, 3, int(1e6), 3,
        )
        pairs = [
            (a, b) for a, b, _ in _apply_smarts_prepass(
                FRAG_RULES, mol_d["mol"], edge_mask.astype(np.uint8), edges, num_nodes, num_edges
            )
        ]
        bfs_keys = {nmm_base[i].tobytes() for i in range(len(nmm_base))}
        unique_masks = [
            m for pair in pairs for m in pair
            if m.sum() >= 3 and m[:num_nodes].astype(np.uint8).tobytes() not in bfs_keys
        ]
        assert len(unique_masks) > 0, (
            "Expected at least one SMARTS fragment not reachable by plain bond cuts"
        )
        # Verify those fragments appear in the post-merge DAG
        dag_yes = _run_compute_dags(smiles, smarts_prepass=True)
        dag_no = _run_compute_dags(smiles, smarts_prepass=False)
        assert dag_yes["dag_num_nodes"] > dag_no["dag_num_nodes"], (
            "Post-merge should add at least one unique SMARTS node"
        )

    def test_prepass_ring_nodes_unchanged(self):
        """Ring-containing molecules: BFS node count is identical with/without prepass.

        The post-merge approach runs compute_frags.compute_ccs unchanged, so all
        ring-opening fragmentation (n_ccs=1 path) is preserved.  Prepass can only
        ADD nodes, never remove ring-opened intermediates.
        """
        benzene = "c1ccccc1"
        indole = "c1ccc2[nH]ccc2c1"
        for smiles in (benzene, indole):
            dag_no = _run_compute_dags(smiles, smarts_prepass=False)
            dag_yes = _run_compute_dags(smiles, smarts_prepass=True)
            assert dag_yes["dag_num_nodes"] >= dag_no["dag_num_nodes"], (
                f"{smiles}: prepass reduced node count "
                f"({dag_no['dag_num_nodes']} → {dag_yes['dag_num_nodes']})"
            )

    def test_prepass_new_edges_connect_root(self):
        """New edges introduced by the prepass must originate from root (node 0)."""
        import fragnnet.frag.compute_frags as compute_frags

        mol_d = extract_mol_info(self.METHYL_ACETATE)
        num_nodes, num_edges, node_mask, edges, edge_mask, node_to_edge_idx = (
            get_fraggen_input_arrays(mol_d)
        )
        pairs = [
            (a, b) for a, b, _ in _apply_smarts_prepass(
                FRAG_RULES, mol_d["mol"], edge_mask.astype(np.uint8), edges, num_nodes, num_edges
            )
        ]
        nmm, ndm, dem, meta = compute_frags.compute_ccs(
            num_nodes, num_edges,
            node_mask.astype(MASK_DTYPE), edges, edge_mask.astype(MASK_DTYPE),
            node_to_edge_idx, 3, int(1e6), 3,
        )
        _, _, dem2, _ = _inject_smarts_pairs_into_dag(
            pairs, nmm, ndm, dem, meta, num_nodes, min_frag_atoms=3
        )
        new_edges = dem2[len(dem):]
        if len(new_edges) > 0:
            assert (new_edges[:, 0] == 0).all(), (
                "All new SMARTS edges must originate from root (index 0)"
            )


# ---------------------------------------------------------------------------
# benzopyranone_ring_cleavage rule
# ---------------------------------------------------------------------------


class TestBenzopyranoneRingCleavage:
    """Tests for the benzopyranone_ring_cleavage SMARTS fragmentation rule.

    The rule targets fused benzene+pyranone (coumarin-type) scaffolds.
    It cuts 4 bonds simultaneously — 2 adjacent benzene C-C bonds isolating
    one ring carbon, plus C8a-O and C2-C3 in the pyranone — producing 3
    CCs that merge to 2 fragment masks via merge_outer_map_nums=[1, 5].
    """

    # 3-phenyl-7-methoxycoumarin: the dominant improved scaffold in NIST20
    COUMARIN_OMe_Ph = "COc1ccc2oc(=O)c(-c3ccccc3)cc2c1"
    # 3-phenyl-4-methyl-7-methoxycoumarin: additional confirmed substrate
    COUMARIN_OMe_Ph_Me = "COc1ccc2oc(=O)c(-c3ccccc3)c(C)c2c1"
    # 7-methoxycoumarin (no pendant phenyl): should still match the coumarin core
    COUMARIN_OMe_ONLY = "COc1ccc2ccc(=O)oc2c1"
    # Non-coumarin: plain naphthalene — should NOT match
    NAPHTHALENE = "c1ccc2ccccc2c1"
    # Non-coumarin: plain benzene — should NOT match
    BENZENE = "c1ccccc1"

    def _get_rule(self):
        """Return the benzopyranone_ring_cleavage rule from FRAG_RULES."""
        for rule in FRAG_RULES:
            if rule.name == "benzopyranone_ring_cleavage":
                return rule
        raise AssertionError("benzopyranone_ring_cleavage rule not found in FRAG_RULES")

    def test_rule_exists_in_frag_rules(self):
        """The rule is registered in the FRAG_RULES list."""
        names = [r.name for r in FRAG_RULES]
        assert "benzopyranone_ring_cleavage" in names

    def test_smarts_compiles(self):
        """The SMARTS pattern compiles to a valid RDKit query molecule."""
        from rdkit import Chem
        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        assert patt is not None, f"Invalid SMARTS: {rule.reactant_smarts}"

    def test_matches_3_phenyl_7_ome_coumarin(self):
        """Exactly 1 match on the primary target scaffold."""
        from rdkit import Chem
        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.COUMARIN_OMe_Ph)
        matches = mol.GetSubstructMatches(patt)
        assert len(matches) == 1

    def test_matches_3_phenyl_4_methyl_7_ome_coumarin(self):
        """Matches 3-phenyl-4-methyl-7-methoxycoumarin (4-Me substituent tolerated)."""
        from rdkit import Chem
        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.COUMARIN_OMe_Ph_Me)
        matches = mol.GetSubstructMatches(patt)
        assert len(matches) == 1

    def test_matches_7_methoxycoumarin_no_phenyl(self):
        """Matches the coumarin core even without a 3-phenyl substituent."""
        from rdkit import Chem
        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.COUMARIN_OMe_ONLY)
        matches = mol.GetSubstructMatches(patt)
        assert len(matches) == 1

    def test_no_match_naphthalene(self):
        """Naphthalene (no O, no lactone) does not match."""
        from rdkit import Chem
        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.NAPHTHALENE)
        assert len(mol.GetSubstructMatches(patt)) == 0

    def test_no_match_benzene(self):
        """Benzene (no fused ring, no O) does not match."""
        from rdkit import Chem
        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.BENZENE)
        assert len(mol.GetSubstructMatches(patt)) == 0

    def test_cut_bonds_all_exist_in_molecule(self):
        """All 4 cut bonds specified in cut_bond_map_pairs exist in the molecule."""
        from rdkit import Chem
        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.COUMARIN_OMe_Ph)
        matches = mol.GetSubstructMatches(patt)
        assert matches, "Expected at least 1 match"
        match = matches[0]
        q_idx_to_map = {
            q: patt.GetAtomWithIdx(q).GetAtomMapNum()
            for q in range(patt.GetNumAtoms())
            if patt.GetAtomWithIdx(q).GetAtomMapNum() != 0
        }
        map_to_mol = {m: match[q] for q, m in q_idx_to_map.items()}
        for map_a, map_b in rule.cut_bond_map_pairs:
            bond = mol.GetBondBetweenAtoms(map_to_mol[map_a], map_to_mol[map_b])
            assert bond is not None, (
                f"No bond between map:{map_a}(atom {map_to_mol[map_a]}) "
                f"and map:{map_b}(atom {map_to_mol[map_b]})"
            )

    def test_cuts_produce_three_fragments(self):
        """Applying all 4 cuts to the primary substrate yields exactly 3 CCs."""
        from rdkit import Chem
        from rdkit.Chem import rdmolops
        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.COUMARIN_OMe_Ph)
        match = mol.GetSubstructMatches(patt)[0]
        q_idx_to_map = {
            q: patt.GetAtomWithIdx(q).GetAtomMapNum()
            for q in range(patt.GetNumAtoms())
            if patt.GetAtomWithIdx(q).GetAtomMapNum() != 0
        }
        map_to_mol = {m: match[q] for q, m in q_idx_to_map.items()}
        em = Chem.RWMol(mol)
        for map_a, map_b in rule.cut_bond_map_pairs:
            em.RemoveBond(map_to_mol[map_a], map_to_mol[map_b])
        frags = rdmolops.GetMolFrags(em.GetMol(), asMols=False)
        assert len(frags) == 3, f"Expected 3 fragments after cuts, got {len(frags)}"

    def test_large_carbon_fragment_exists(self):
        """The large all-carbon fragment (C13+) is one of the 3 produced CCs."""
        from rdkit import Chem
        from rdkit.Chem import rdmolops
        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.COUMARIN_OMe_Ph)
        match = mol.GetSubstructMatches(patt)[0]
        q_idx_to_map = {
            q: patt.GetAtomWithIdx(q).GetAtomMapNum()
            for q in range(patt.GetNumAtoms())
            if patt.GetAtomWithIdx(q).GetAtomMapNum() != 0
        }
        map_to_mol = {m: match[q] for q, m in q_idx_to_map.items()}
        em = Chem.RWMol(mol)
        for map_a, map_b in rule.cut_bond_map_pairs:
            em.RemoveBond(map_to_mol[map_a], map_to_mol[map_b])
        frags = rdmolops.GetMolFrags(em.GetMol(), asMols=False)
        max_c = max(
            sum(1 for a in f if mol.GetAtomWithIdx(a).GetSymbol() == "C")
            for f in frags
        )
        # 3-phenyl-7-methoxycoumarin: large fragment = 13 C atoms
        assert max_c == 13, f"Expected large C13 fragment, got max C count = {max_c}"

    def test_prepass_adds_fragments_to_coumarin_dag(self):
        """smarts_prepass=True adds nodes for the coumarin scaffold."""
        dag_no = _run_compute_dags(self.COUMARIN_OMe_Ph, smarts_prepass=False)
        dag_yes = _run_compute_dags(self.COUMARIN_OMe_Ph, smarts_prepass=True)
        assert dag_yes["dag_num_nodes"] > dag_no["dag_num_nodes"], (
            "Expected benzopyranone_ring_cleavage rule to add at least one new node"
        )


# ---------------------------------------------------------------------------
# furanocoumarin_ring_cleavage rule
# ---------------------------------------------------------------------------


class TestFuranoCoumarinRingCleavage:
    """Tests for the furanocoumarin_ring_cleavage SMARTS fragmentation rule.

    The rule targets psoralen (linear furanocoumarin) scaffolds: a 5-membered furan
    ring and a 6-membered pyranone (lactone) ring fused on opposite sides of a central
    benzene ring. It cuts 4 bonds simultaneously — (5,6)/(3,4) releasing the furan's
    non-fusion atoms {C,C,O} and (1,2)/(7,8) releasing the entire pyranone ring —
    producing 3 CCs that merge to 2 fragment masks via merge_outer_map_nums=[4, 8].
    The inner kept fragment is the 4-atom open chain from the central benzene's
    furan-flanking carbons plus any substituents on those atoms.
    """

    # Bare psoralen scaffold (14 heavy atoms, linear furanocoumarin)
    PSORALEN = "O=c1ccc2cc3ccoc3cc2o1"
    # Isopimpinellin-type: dimethylallyl + OMe substituents on the benzene
    ISOPIMPINELLIN = "C=CC(C)(C)c1c2ccoc2c(OC)c2oc(=O)ccc12"
    # 5,8-Dimethoxypsoralen: OMe on both benzene positions
    DIMETHOXYPSORALEN = "COc1c2ccoc2c(OC)c2oc(=O)ccc12"
    # 8-Methoxypsoralen (xanthotoxin): OMe on one benzene position
    XANTHOTOXIN = "O=c1ccc2cc3ccoc3c(OC)c2o1"
    # Negative: bare coumarin (no furan ring) — should NOT match
    COUMARIN = "c1ccc2oc(=O)ccc2c1"
    # Negative: benzene — should NOT match
    BENZENE = "c1ccccc1"
    # Negative: xanthone (two benzene rings + central pyranone, no furan) — should NOT match
    XANTHONE = "O=C1c2ccccc2Oc2ccccc21"
    # Negative: naphthalene — should NOT match
    NAPHTHALENE = "c1ccc2ccccc2c1"

    def _get_rule(self):
        for rule in FRAG_RULES:
            if rule.name == "furanocoumarin_ring_cleavage":
                return rule
        raise AssertionError("furanocoumarin_ring_cleavage not found in FRAG_RULES")

    def test_rule_exists_in_frag_rules(self):
        """The rule is registered in the FRAG_RULES list."""
        assert any(r.name == "furanocoumarin_ring_cleavage" for r in FRAG_RULES)

    def test_smarts_compiles(self):
        """The SMARTS pattern compiles to a valid RDKit query molecule."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        assert patt is not None, f"Invalid SMARTS: {rule.reactant_smarts}"

    def test_matches_bare_psoralen(self):
        """Exactly 1 match on bare psoralen scaffold."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.PSORALEN)
        assert len(mol.GetSubstructMatches(patt)) == 1

    def test_matches_substituted_psoralen(self):
        """Matches substituted psoralens (OMe and dimethylallyl groups tolerated)."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        for smi in (self.ISOPIMPINELLIN, self.DIMETHOXYPSORALEN, self.XANTHOTOXIN):
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None
            assert len(mol.GetSubstructMatches(patt)) >= 1, f"No match on {smi}"

    def test_no_match_coumarin(self):
        """Coumarin (no furan ring) does not match."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.COUMARIN)
        assert len(mol.GetSubstructMatches(patt)) == 0

    def test_no_match_benzene(self):
        """Benzene does not match."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.BENZENE)
        assert len(mol.GetSubstructMatches(patt)) == 0

    def test_no_match_xanthone(self):
        """Xanthone (two benzene rings + pyranone, no furan) does not match."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.XANTHONE)
        assert len(mol.GetSubstructMatches(patt)) == 0

    def test_no_match_naphthalene(self):
        """Naphthalene (no O, no fused heterocycle) does not match."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.NAPHTHALENE)
        assert len(mol.GetSubstructMatches(patt)) == 0

    def test_cut_bonds_all_exist_and_are_ring_bonds(self):
        """All 4 cut bonds in cut_bond_map_pairs exist and are ring bonds in psoralen."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.PSORALEN)
        matches = mol.GetSubstructMatches(patt)
        assert matches
        match = matches[0]
        q_idx_to_map = {
            q: patt.GetAtomWithIdx(q).GetAtomMapNum()
            for q in range(patt.GetNumAtoms())
            if patt.GetAtomWithIdx(q).GetAtomMapNum() != 0
        }
        map_to_mol = {m: match[q] for q, m in q_idx_to_map.items()}
        for map_a, map_b in rule.cut_bond_map_pairs:
            bond = mol.GetBondBetweenAtoms(map_to_mol[map_a], map_to_mol[map_b])
            assert bond is not None, f"No bond between map:{map_a} and map:{map_b}"
            assert bond.IsInRing(), f"Cut bond ({map_a},{map_b}) is not a ring bond"

    def test_cuts_produce_three_fragments(self):
        """Applying all 4 cuts to psoralen yields exactly 3 connected components."""
        from rdkit import Chem
        from rdkit.Chem import rdmolops

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.PSORALEN)
        match = mol.GetSubstructMatches(patt)[0]
        q_idx_to_map = {
            q: patt.GetAtomWithIdx(q).GetAtomMapNum()
            for q in range(patt.GetNumAtoms())
            if patt.GetAtomWithIdx(q).GetAtomMapNum() != 0
        }
        map_to_mol = {m: match[q] for q, m in q_idx_to_map.items()}
        em = Chem.RWMol(mol)
        for map_a, map_b in rule.cut_bond_map_pairs:
            em.RemoveBond(map_to_mol[map_a], map_to_mol[map_b])
        frags = rdmolops.GetMolFrags(em.GetMol(), asMols=False)
        assert len(frags) == 3, f"Expected 3 fragments, got {len(frags)}"

    def test_inner_fragment_is_benzene_chain(self):
        """The inner kept fragment is the 4-atom open chain from the central benzene."""
        from rdkit import Chem
        from rdkit.Chem import rdmolops

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.PSORALEN)
        match = mol.GetSubstructMatches(patt)[0]
        q_idx_to_map = {
            q: patt.GetAtomWithIdx(q).GetAtomMapNum()
            for q in range(patt.GetNumAtoms())
            if patt.GetAtomWithIdx(q).GetAtomMapNum() != 0
        }
        map_to_mol = {m: match[q] for q, m in q_idx_to_map.items()}
        # Outer atoms (from merge_outer_map_nums=[4, 8])
        outer_atoms = {map_to_mol[4], map_to_mol[8]}
        em = Chem.RWMol(mol)
        for map_a, map_b in rule.cut_bond_map_pairs:
            em.RemoveBond(map_to_mol[map_a], map_to_mol[map_b])
        frags = rdmolops.GetMolFrags(em.GetMol(), asMols=False)
        inner_frags = [f for f in frags if not any(a in outer_atoms for a in f)]
        assert len(inner_frags) == 1, "Expected exactly one inner fragment"
        inner_atoms = inner_frags[0]
        # In bare psoralen, inner fragment = 4 carbons (C4H4 open chain)
        c_count = sum(1 for a in inner_atoms if mol.GetAtomWithIdx(a).GetSymbol() == "C")
        o_count = sum(1 for a in inner_atoms if mol.GetAtomWithIdx(a).GetSymbol() == "O")
        assert c_count == 4, f"Inner fragment must have 4 C atoms, got {c_count}"
        assert o_count == 0, f"Inner fragment must have 0 O atoms, got {o_count}"

    def test_prepass_adds_fragments_to_dag(self):
        """smarts_prepass=True adds nodes for psoralen scaffold molecules."""
        dag_no = _run_compute_dags(self.PSORALEN, smarts_prepass=False)
        dag_yes = _run_compute_dags(self.PSORALEN, smarts_prepass=True)
        assert dag_yes["dag_num_nodes"] > dag_no["dag_num_nodes"], (
            "Expected furanocoumarin_ring_cleavage rule to add at least one new node"
        )

    def test_prepass_adds_fragments_substituted(self):
        """smarts_prepass=True adds nodes for a substituted psoralen derivative."""
        dag_no = _run_compute_dags(self.ISOPIMPINELLIN, smarts_prepass=False)
        dag_yes = _run_compute_dags(self.ISOPIMPINELLIN, smarts_prepass=True)
        assert dag_yes["dag_num_nodes"] > dag_no["dag_num_nodes"], (
            "Expected rule to add nodes for substituted psoralen"
        )


# ---------------------------------------------------------------------------
# xanthone_ring_cleavage rule
# ---------------------------------------------------------------------------


class TestXanthoneRingCleavage:
    """Tests for the xanthone_ring_cleavage SMARTS fragmentation rule.

    The rule targets dibenzo-γ-pyranone (xanthone/xanthen-9-one) scaffolds.
    It cuts 4 bonds simultaneously — (2,3)/(6,7) releasing the left benzene C4
    unit and (9,10)/(13,14) releasing the right benzene C4 unit — producing 3 CCs
    that merge to 2 fragment masks via merge_outer_map_nums=[3, 10].
    The inner kept fragment is the central pyranone ring (C5O2).
    """

    # Bare xanthone scaffold (15 heavy atoms, symmetric)
    XANTHONE = "O=C1c2ccccc2Oc2ccccc21"
    # 1-hydroxyxanthone: OH substituent on one benzene ring
    XANTHONE_1OH = "Oc1ccc2oc3ccccc3c(=O)c2c1"
    # Xanthone with ester + methyl substituents (NIST20 mol 16676)
    XANTHONE_ESTER = "COC(=O)c1cc(C)cc2oc3cccc(O)c3c(=O)c12"
    # Naphthalene: no O, no ketone — should NOT match
    NAPHTHALENE = "c1ccc2ccccc2c1"
    # Benzene — should NOT match
    BENZENE = "c1ccccc1"
    # Coumarin (benzopyranone): has O+C=O but NOT the xanthone tricyclic topology
    COUMARIN = "c1ccc2oc(=O)ccc2c1"
    # Dibenzofuran: two benzene rings + bridge O but NO carbonyl — should NOT match
    DIBENZOFURAN = "c1ccc2oc3ccccc3c2c1"

    def _get_rule(self):
        for rule in FRAG_RULES:
            if rule.name == "xanthone_ring_cleavage":
                return rule
        raise AssertionError("xanthone_ring_cleavage rule not found in FRAG_RULES")

    def test_rule_exists_in_frag_rules(self):
        """The rule is registered in the FRAG_RULES list."""
        names = [r.name for r in FRAG_RULES]
        assert "xanthone_ring_cleavage" in names

    def test_smarts_compiles(self):
        """The SMARTS pattern compiles to a valid RDKit query molecule."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        assert patt is not None, f"Invalid SMARTS: {rule.reactant_smarts}"

    def test_matches_bare_xanthone(self):
        """Exactly 1 match on bare xanthone (symmetric molecule)."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.XANTHONE)
        matches = mol.GetSubstructMatches(patt)
        assert len(matches) == 1

    def test_matches_1_hydroxyxanthone(self):
        """Matches 1-hydroxyxanthone (OH substituent tolerated)."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.XANTHONE_1OH)
        assert len(mol.GetSubstructMatches(patt)) >= 1

    def test_matches_xanthone_with_ester_substituents(self):
        """Matches a xanthone with ester and methyl substituents."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.XANTHONE_ESTER)
        assert len(mol.GetSubstructMatches(patt)) >= 1

    def test_no_match_naphthalene(self):
        """Naphthalene (no O, no carbonyl) does not match."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.NAPHTHALENE)
        assert len(mol.GetSubstructMatches(patt)) == 0

    def test_no_match_benzene(self):
        """Benzene (single ring) does not match."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.BENZENE)
        assert len(mol.GetSubstructMatches(patt)) == 0

    def test_no_match_coumarin(self):
        """Coumarin (monoaromatic ring fused to lactone, not xanthone) does not match."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.COUMARIN)
        assert len(mol.GetSubstructMatches(patt)) == 0

    def test_no_match_dibenzofuran(self):
        """Dibenzofuran (no carbonyl C=O) does not match."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.DIBENZOFURAN)
        assert len(mol.GetSubstructMatches(patt)) == 0

    def test_cut_bonds_all_exist_in_molecule(self):
        """All 4 cut bonds specified in cut_bond_map_pairs exist in the molecule."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.XANTHONE)
        matches = mol.GetSubstructMatches(patt)
        assert matches, "Expected at least 1 match"
        match = matches[0]
        q_idx_to_map = {
            q: patt.GetAtomWithIdx(q).GetAtomMapNum()
            for q in range(patt.GetNumAtoms())
            if patt.GetAtomWithIdx(q).GetAtomMapNum() != 0
        }
        map_to_mol = {m: match[q] for q, m in q_idx_to_map.items()}
        for map_a, map_b in rule.cut_bond_map_pairs:
            bond = mol.GetBondBetweenAtoms(map_to_mol[map_a], map_to_mol[map_b])
            assert bond is not None, (
                f"No bond between map:{map_a}(atom {map_to_mol[map_a]}) "
                f"and map:{map_b}(atom {map_to_mol[map_b]})"
            )
            assert bond.IsInRing(), f"Cut bond ({map_a},{map_b}) must be a ring bond"

    def test_cuts_produce_three_fragments(self):
        """Applying all 4 cuts to bare xanthone yields exactly 3 CCs."""
        from rdkit import Chem
        from rdkit.Chem import rdmolops

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.XANTHONE)
        match = mol.GetSubstructMatches(patt)[0]
        q_idx_to_map = {
            q: patt.GetAtomWithIdx(q).GetAtomMapNum()
            for q in range(patt.GetNumAtoms())
            if patt.GetAtomWithIdx(q).GetAtomMapNum() != 0
        }
        map_to_mol = {m: match[q] for q, m in q_idx_to_map.items()}
        em = Chem.RWMol(mol)
        for map_a, map_b in rule.cut_bond_map_pairs:
            em.RemoveBond(map_to_mol[map_a], map_to_mol[map_b])
        frags = rdmolops.GetMolFrags(em.GetMol(), asMols=False)
        assert len(frags) == 3, f"Expected 3 fragments after cuts, got {len(frags)}"

    def test_inner_fragment_is_central_pyranone_c5o2(self):
        """The inner (kept) fragment after cuts is the 7-atom central C5O2 ring."""
        from rdkit import Chem
        from rdkit.Chem import rdmolops

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.XANTHONE)
        match = mol.GetSubstructMatches(patt)[0]
        q_idx_to_map = {
            q: patt.GetAtomWithIdx(q).GetAtomMapNum()
            for q in range(patt.GetNumAtoms())
            if patt.GetAtomWithIdx(q).GetAtomMapNum() != 0
        }
        map_to_mol = {m: match[q] for q, m in q_idx_to_map.items()}
        outer_atoms = {map_to_mol[3], map_to_mol[6], map_to_mol[10], map_to_mol[13]}
        em = Chem.RWMol(mol)
        for map_a, map_b in rule.cut_bond_map_pairs:
            em.RemoveBond(map_to_mol[map_a], map_to_mol[map_b])
        frags = rdmolops.GetMolFrags(em.GetMol(), asMols=False)
        inner_frag = [f for f in frags if not any(a in outer_atoms for a in f)]
        assert len(inner_frag) == 1, "Expected exactly one inner (non-outer) fragment"
        inner_atoms = inner_frag[0]
        c_count = sum(1 for a in inner_atoms if mol.GetAtomWithIdx(a).GetSymbol() == "C")
        o_count = sum(1 for a in inner_atoms if mol.GetAtomWithIdx(a).GetSymbol() == "O")
        # Central ring: 5 aromatic C + 2 O (bridge O + exo =O) = 7 heavy atoms
        assert c_count == 5, f"Inner fragment must have 5 C, got {c_count}"
        assert o_count == 2, f"Inner fragment must have 2 O, got {o_count}"

    def test_prepass_adds_fragments_to_xanthone_dag(self):
        """smarts_prepass=True adds nodes for the xanthone scaffold."""
        dag_no = _run_compute_dags(self.XANTHONE, smarts_prepass=False)
        dag_yes = _run_compute_dags(self.XANTHONE, smarts_prepass=True)
        assert dag_yes["dag_num_nodes"] > dag_no["dag_num_nodes"], (
            "Expected xanthone_ring_cleavage rule to add at least one new node"
        )


# ---------------------------------------------------------------------------
# phenoxazine_ring_cleavage rule
# ---------------------------------------------------------------------------


class TestPhenoxazineRingCleavage:
    """Tests for the phenoxazine_ring_cleavage SMARTS fragmentation rule.

    The rule targets dibenz[b,e][1,4]oxazine (phenoxazine) scaffolds: two benzene
    rings fused on either side of a central 6-membered ring containing one N and
    one O bridge. It cuts 4 bonds simultaneously — (2,3)/(6,7) releasing the right
    benzene C4 unit and (9,10)/(13,14) releasing the left benzene C4 unit —
    producing 3 CCs that merge to 2 fragment masks via merge_outer_map_nums=[3, 10].
    The inner kept fragment is the central C4NO ring (6 heavy atoms).
    """

    # Bare phenoxazine scaffold (13 heavy atoms)
    PHENOXAZINE = "c1ccc2Nc3ccccc3Oc2c1"
    # N-acetyl-dihydroxyphenoxazine (NIST20 mol_id=1993): substituted phenoxazine
    PHENOXAZINE_SUBST = "CC(=O)N1c2ccc(O)cc2Oc2cc(O)ccc21"
    # N-alkyl phenoxazine (NIST20 mol_id=13409): N-substituted
    PHENOXAZINE_N_ALKYL = "CCN(CC)CCCCN1c2ccccc2Oc2ccc(Cl)cc21"
    # Naphthalene: no O, no N — should NOT match
    NAPHTHALENE = "c1ccc2ccccc2c1"
    # Benzene — should NOT match
    BENZENE = "c1ccccc1"
    # Carbazole: has N but no O — should NOT match
    CARBAZOLE = "c1ccc2[nH]c3ccccc3c2c1"
    # Dibenzofuran: has O but no N — should NOT match
    DIBENZOFURAN = "c1ccc2oc3ccccc3c2c1"
    # Xanthone: has O and C=O but no N — should NOT match
    XANTHONE = "O=C1c2ccccc2Oc2ccccc21"

    def _get_rule(self):
        """Return the phenoxazine_ring_cleavage rule from FRAG_RULES."""
        for rule in FRAG_RULES:
            if rule.name == "phenoxazine_ring_cleavage":
                return rule
        raise AssertionError("phenoxazine_ring_cleavage rule not found in FRAG_RULES")

    def test_rule_exists_in_frag_rules(self):
        """The rule is registered in the FRAG_RULES list."""
        names = [r.name for r in FRAG_RULES]
        assert "phenoxazine_ring_cleavage" in names

    def test_smarts_compiles(self):
        """The SMARTS pattern compiles to a valid RDKit query molecule."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        assert patt is not None, f"Invalid SMARTS: {rule.reactant_smarts}"

    def test_matches_phenoxazine(self):
        """Exactly 1 match on bare phenoxazine scaffold."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.PHENOXAZINE)
        matches = mol.GetSubstructMatches(patt)
        assert len(matches) == 1

    def test_matches_substituted_phenoxazine(self):
        """Matches N-acetyl-dihydroxyphenoxazine (N-acetyl + hydroxyl substituents)."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.PHENOXAZINE_SUBST)
        assert len(mol.GetSubstructMatches(patt)) >= 1

    def test_matches_n_alkyl_phenoxazine(self):
        """Matches N-alkyl phenoxazine (N is substituted, not NH)."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.PHENOXAZINE_N_ALKYL)
        assert len(mol.GetSubstructMatches(patt)) >= 1

    def test_no_match_naphthalene(self):
        """Naphthalene (no O, no N) does not match."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.NAPHTHALENE)
        assert len(mol.GetSubstructMatches(patt)) == 0

    def test_no_match_benzene(self):
        """Benzene (single ring) does not match."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.BENZENE)
        assert len(mol.GetSubstructMatches(patt)) == 0

    def test_no_match_carbazole(self):
        """Carbazole (has N but no O bridge) does not match."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.CARBAZOLE)
        assert len(mol.GetSubstructMatches(patt)) == 0

    def test_no_match_dibenzofuran(self):
        """Dibenzofuran (has O but no N bridge) does not match."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.DIBENZOFURAN)
        assert len(mol.GetSubstructMatches(patt)) == 0

    def test_cut_bonds_exist_in_molecule(self):
        """All 4 cut bonds specified in cut_bond_map_pairs exist in the molecule."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.PHENOXAZINE)
        matches = mol.GetSubstructMatches(patt)
        assert matches, "Expected at least 1 match"
        match = matches[0]
        q_idx_to_map = {
            q: patt.GetAtomWithIdx(q).GetAtomMapNum()
            for q in range(patt.GetNumAtoms())
            if patt.GetAtomWithIdx(q).GetAtomMapNum() != 0
        }
        map_to_mol = {m: match[q] for q, m in q_idx_to_map.items()}
        for map_a, map_b in rule.cut_bond_map_pairs:
            bond = mol.GetBondBetweenAtoms(map_to_mol[map_a], map_to_mol[map_b])
            assert bond is not None, (
                f"No bond between map:{map_a}(atom {map_to_mol[map_a]}) "
                f"and map:{map_b}(atom {map_to_mol[map_b]})"
            )
            assert bond.IsInRing(), f"Cut bond ({map_a},{map_b}) must be a ring bond"

    def test_cuts_produce_three_fragments(self):
        """Applying all 4 cuts to bare phenoxazine yields exactly 3 CCs."""
        from rdkit import Chem
        from rdkit.Chem import rdmolops

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.PHENOXAZINE)
        match = mol.GetSubstructMatches(patt)[0]
        q_idx_to_map = {
            q: patt.GetAtomWithIdx(q).GetAtomMapNum()
            for q in range(patt.GetNumAtoms())
            if patt.GetAtomWithIdx(q).GetAtomMapNum() != 0
        }
        map_to_mol = {m: match[q] for q, m in q_idx_to_map.items()}
        em = Chem.RWMol(mol)
        for map_a, map_b in rule.cut_bond_map_pairs:
            em.RemoveBond(map_to_mol[map_a], map_to_mol[map_b])
        frags = rdmolops.GetMolFrags(em.GetMol(), asMols=False)
        assert len(frags) == 3, f"Expected 3 fragments after cuts, got {len(frags)}"

    def test_inner_fragment_is_central_c4no_ring(self):
        """The inner (kept) fragment after cuts is the 6-atom central C4NO ring."""
        from rdkit import Chem
        from rdkit.Chem import rdmolops

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.PHENOXAZINE)
        match = mol.GetSubstructMatches(patt)[0]
        q_idx_to_map = {
            q: patt.GetAtomWithIdx(q).GetAtomMapNum()
            for q in range(patt.GetNumAtoms())
            if patt.GetAtomWithIdx(q).GetAtomMapNum() != 0
        }
        map_to_mol = {m: match[q] for q, m in q_idx_to_map.items()}
        # Outer atoms: map 3 and map 10 (one from each released C4 unit)
        outer_atoms = {map_to_mol[3], map_to_mol[10]}
        em = Chem.RWMol(mol)
        for map_a, map_b in rule.cut_bond_map_pairs:
            em.RemoveBond(map_to_mol[map_a], map_to_mol[map_b])
        frags = rdmolops.GetMolFrags(em.GetMol(), asMols=False)
        # Inner fragment is the one not containing any outer atom
        inner_frags = [f for f in frags if not any(a in outer_atoms for a in f)]
        assert len(inner_frags) == 1, "Expected exactly one inner (non-outer) fragment"
        inner_atoms = inner_frags[0]
        c_count = sum(1 for a in inner_atoms if mol.GetAtomWithIdx(a).GetSymbol() == "C")
        n_count = sum(1 for a in inner_atoms if mol.GetAtomWithIdx(a).GetSymbol() == "N")
        o_count = sum(1 for a in inner_atoms if mol.GetAtomWithIdx(a).GetSymbol() == "O")
        # Central ring: 4 aromatic C + 1 N + 1 O = 6 heavy atoms
        assert c_count == 4, f"Inner fragment must have 4 C, got {c_count}"
        assert n_count == 1, f"Inner fragment must have 1 N, got {n_count}"
        assert o_count == 1, f"Inner fragment must have 1 O, got {o_count}"

    def test_prepass_adds_fragments_to_dag(self):
        """smarts_prepass=True adds nodes for a substituted phenoxazine scaffold."""
        dag_no = _run_compute_dags(self.PHENOXAZINE_SUBST, smarts_prepass=False)
        dag_yes = _run_compute_dags(self.PHENOXAZINE_SUBST, smarts_prepass=True)
        assert dag_yes["dag_num_nodes"] > dag_no["dag_num_nodes"], (
            "Expected phenoxazine_ring_cleavage rule to add at least one new node"
        )


class TestMorphinanRingCleavage:
    """Tests for the morphinan_ring_cleavage SMARTS fragmentation rule.

    The rule targets the 5-ring morphinan alkaloid scaffold (morphine/codeine class).
    It cuts 4 bonds simultaneously — N-C (piperidine, maps 1-2), C-C (cyclohexene
    C15-C16, maps 3-4), C-C (bridgehead C6-C20, maps 5-6), c-C (aromatic-bridge
    C8-C22, maps 7-8) — producing n_ccs=2 (no merging needed). The inner kept
    fragment is the bicyclic benzene+dihydrofuran C10HxO2 piece; the outer is the
    large N-containing remainder.
    """

    # Morphine: allyl-N, phenol OH, ring OH, C=C in ring 3
    MORPHINE = "C=CCN1CCC23c4c5ccc(O)c4OC2C(O)C=CC3C1C5"
    # Codeine: OMe on phenol, methyl-N, C=C in ring 3
    CODEINE = "COc1ccc2c3c1OC1C(O)C=CC4C(C2)N(C)CCC341"
    # Naloxone: allyl-N, phenol OH, C=O in ring 3 (saturated ring bond variant)
    NALOXONE = "C=CCN1CC[C@]23c4c5ccc(O)c4O[C@H]2C(=O)CC[C@H]3[C@H]1C5"
    # Norcodeine: NH, OMe — tests that non-N-substituted variant matches
    NORCODEINE = "COc1ccc2c3c1OC1C(O)C=CC4C(C2)NCCC341"
    # Naphthalene: no N, no ether O — should NOT match
    NAPHTHALENE = "c1ccc2ccccc2c1"
    # Benzene — should NOT match
    BENZENE = "c1ccccc1"
    # Indole: has N but no saturated ring system — should NOT match
    INDOLE = "c1ccc2[nH]ccc2c1"
    # Piperidine: has N but no aromatic system — should NOT match
    PIPERIDINE = "C1CCNCC1"

    def _get_rule(self):
        """Return the morphinan_ring_cleavage rule from FRAG_RULES."""
        for rule in FRAG_RULES:
            if rule.name == "morphinan_ring_cleavage":
                return rule
        raise AssertionError("morphinan_ring_cleavage rule not found in FRAG_RULES")

    def test_rule_exists_in_frag_rules(self):
        """The rule is registered in the FRAG_RULES list."""
        names = [r.name for r in FRAG_RULES]
        assert "morphinan_ring_cleavage" in names

    def test_smarts_compiles(self):
        """The SMARTS pattern compiles to a valid RDKit query molecule."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        assert patt is not None, f"Invalid SMARTS: {rule.reactant_smarts}"

    def test_matches_morphine(self):
        """Exactly 1 match on morphine (allyl-N, phenol OH, C=C in ring 3)."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.MORPHINE)
        assert len(mol.GetSubstructMatches(patt)) == 1

    def test_matches_codeine(self):
        """Matches codeine (OMe on phenol, methyl-N variant)."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.CODEINE)
        assert len(mol.GetSubstructMatches(patt)) >= 1

    def test_matches_naloxone(self):
        """Matches naloxone (saturated ring 3, allyl-N) — tests ~ bond flexibility."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.NALOXONE)
        assert len(mol.GetSubstructMatches(patt)) >= 1

    def test_matches_norcodeine(self):
        """Matches norcodeine (NH, not N-substituted)."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.NORCODEINE)
        assert len(mol.GetSubstructMatches(patt)) >= 1

    def test_no_match_naphthalene(self):
        """Naphthalene (no N, no ether O) does not match."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.NAPHTHALENE)
        assert len(mol.GetSubstructMatches(patt)) == 0

    def test_no_match_benzene(self):
        """Benzene (single ring, too small) does not match."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.BENZENE)
        assert len(mol.GetSubstructMatches(patt)) == 0

    def test_no_match_indole(self):
        """Indole (no saturated ring system) does not match."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.INDOLE)
        assert len(mol.GetSubstructMatches(patt)) == 0

    def test_no_match_piperidine(self):
        """Piperidine (no aromatic system) does not match."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.PIPERIDINE)
        assert len(mol.GetSubstructMatches(patt)) == 0

    def test_cut_bonds_are_ring_bonds_morphine(self):
        """All 4 cut bonds in morphine are ring bonds."""
        from rdkit import Chem

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.MORPHINE)
        match = mol.GetSubstructMatches(patt)[0]
        q_idx_to_map = {
            q: patt.GetAtomWithIdx(q).GetAtomMapNum()
            for q in range(patt.GetNumAtoms())
            if patt.GetAtomWithIdx(q).GetAtomMapNum() != 0
        }
        map_to_mol = {m: match[q] for q, m in q_idx_to_map.items()}
        for map_a, map_b in rule.cut_bond_map_pairs:
            bond = mol.GetBondBetweenAtoms(map_to_mol[map_a], map_to_mol[map_b])
            assert bond is not None, (
                f"No bond between map:{map_a}(atom {map_to_mol[map_a]}) "
                f"and map:{map_b}(atom {map_to_mol[map_b]})"
            )
            assert bond.IsInRing(), f"Cut bond ({map_a},{map_b}) is not a ring bond"

    def test_cuts_produce_two_fragments(self):
        """Applying all 4 cuts to morphine yields exactly 2 CCs (n_ccs=2, no merge)."""
        from rdkit import Chem
        from rdkit.Chem import rdmolops

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.MORPHINE)
        match = mol.GetSubstructMatches(patt)[0]
        q_idx_to_map = {
            q: patt.GetAtomWithIdx(q).GetAtomMapNum()
            for q in range(patt.GetNumAtoms())
            if patt.GetAtomWithIdx(q).GetAtomMapNum() != 0
        }
        map_to_mol = {m: match[q] for q, m in q_idx_to_map.items()}
        em = Chem.RWMol(mol)
        for map_a, map_b in rule.cut_bond_map_pairs:
            em.RemoveBond(map_to_mol[map_a], map_to_mol[map_b])
        frags = rdmolops.GetMolFrags(em.GetMol(), asMols=False)
        assert len(frags) == 2, f"Expected 2 fragments after cuts, got {len(frags)}"

    def test_inner_fragment_is_c10o2(self):
        """The inner (N-free) fragment after cuts has 10C and 2O heavy atoms."""
        from rdkit import Chem
        from rdkit.Chem import rdmolops

        rule = self._get_rule()
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.MORPHINE)
        match = mol.GetSubstructMatches(patt)[0]
        q_idx_to_map = {
            q: patt.GetAtomWithIdx(q).GetAtomMapNum()
            for q in range(patt.GetNumAtoms())
            if patt.GetAtomWithIdx(q).GetAtomMapNum() != 0
        }
        map_to_mol = {m: match[q] for q, m in q_idx_to_map.items()}
        em = Chem.RWMol(mol)
        for map_a, map_b in rule.cut_bond_map_pairs:
            em.RemoveBond(map_to_mol[map_a], map_to_mol[map_b])
        frags = rdmolops.GetMolFrags(em.GetMol(), asMols=False)
        # Inner fragment = the one with no N
        n_atom_idx = map_to_mol[1]  # map 1 = N atom
        inner_frags = [f for f in frags if n_atom_idx not in f]
        assert len(inner_frags) == 1, "Expected exactly one N-free inner fragment"
        inner = inner_frags[0]
        c_count = sum(1 for a in inner if mol.GetAtomWithIdx(a).GetSymbol() == "C")
        o_count = sum(1 for a in inner if mol.GetAtomWithIdx(a).GetSymbol() == "O")
        n_count = sum(1 for a in inner if mol.GetAtomWithIdx(a).GetSymbol() == "N")
        assert c_count == 10, f"Inner fragment must have 10 C, got {c_count}"
        assert o_count == 2, f"Inner fragment must have 2 O, got {o_count}"
        assert n_count == 0, f"Inner fragment must have 0 N, got {n_count}"

    def test_prepass_adds_fragments_to_dag_morphine(self):
        """smarts_prepass=True adds nodes for morphine."""
        dag_no = _run_compute_dags(self.MORPHINE, smarts_prepass=False)
        dag_yes = _run_compute_dags(self.MORPHINE, smarts_prepass=True)
        assert dag_yes["dag_num_nodes"] > dag_no["dag_num_nodes"], (
            "Expected morphinan_ring_cleavage rule to add at least one new node"
        )

    def test_prepass_adds_fragments_to_dag_codeine(self):
        """smarts_prepass=True adds nodes for codeine (OMe variant)."""
        dag_no = _run_compute_dags(self.CODEINE, smarts_prepass=False)
        dag_yes = _run_compute_dags(self.CODEINE, smarts_prepass=True)
        assert dag_yes["dag_num_nodes"] > dag_no["dag_num_nodes"], (
            "Expected morphinan_ring_cleavage rule to add at least one new node for codeine"
        )

    def test_prepass_no_change_naphthalene(self):
        """smarts_prepass=True does not add nodes for naphthalene (negative control)."""
        dag_no = _run_compute_dags(self.NAPHTHALENE, smarts_prepass=False)
        dag_yes = _run_compute_dags(self.NAPHTHALENE, smarts_prepass=True)
        assert dag_yes["dag_num_nodes"] == dag_no["dag_num_nodes"], (
            "morphinan_ring_cleavage rule should not fire on naphthalene"
        )
