"""Tests for NeutralLossRule prepass (SO₂ extrusion, SO₃ loss, N₂O extrusion).

Verifies that:
- NL SMARTS patterns compile and match expected molecules.
- _apply_nl_prepass returns product masks with the correct atom count.
- _inject_nl_masks_into_dag adds new nodes only when the formula is absent from BFS.
- smarts_prepass=True in compute_dags correctly includes NL products.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from fragnnet.frag.smarts_prepass import NL_RULES, _apply_nl_prepass
from fragnnet.utils.frag_utils import (
    _inject_nl_masks_into_dag,
    compute_dags,
    extract_mol_info,
    get_fraggen_input_arrays,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nl_rules_by_name() -> dict:
    return {r.name: r for r in NL_RULES}


def _run_dag(smiles: str, smarts_prepass: bool) -> dict:
    mol_d = extract_mol_info(smiles)
    return compute_dags(
        mol_d,
        max_depth=3,
        h_prior=False,
        max_h_transfer=4,
        frag_max_time=int(1e6),
        multi_cut_bfs=False,
        smarts_prepass=smarts_prepass,
    )


# ---------------------------------------------------------------------------
# 1. SMARTS compilation and basic matching
# ---------------------------------------------------------------------------


class TestNLRulesCompile:
    def test_all_rules_compile(self):
        """All NL_RULES SMARTS patterns must parse without error."""
        for rule in NL_RULES:
            patt = Chem.MolFromSmarts(rule.reactant_smarts)
            assert patt is not None, f"Rule {rule.name!r} SMARTS failed to compile"

    def test_aryl_sulfonamide_matches_sulfadiazine(self):
        """Aromatic sulfonamide pattern matches N-aryl sulfonamide."""
        mol = Chem.MolFromSmiles("O=[N+]([O-])c1ccc(S(=O)(=O)Nc2ncccn2)cc1")
        results = _apply_nl_prepass(NL_RULES, mol, mol.GetNumAtoms())
        names = [NL_RULES[ridx].name for _, ridx in results]
        assert "aryl_sulfonamide_so2" in names

    def test_aryl_sulfonamide_no_match_aliphatic(self):
        """Aliphatic sulfonamide must NOT match aryl_sulfonamide_so2 (requires [c])."""
        mol = Chem.MolFromSmiles("CS(=O)(=O)NC")  # methanesulfonamide, no aromatic C–S
        results = _apply_nl_prepass(NL_RULES, mol, mol.GetNumAtoms())
        names = [NL_RULES[ridx].name for _, ridx in results]
        assert "aryl_sulfonamide_so2" not in names

    def test_aryl_sulfonate_matches_pyrene_sulfonic_acid(self):
        """Sulfonate pattern matches aromatic sulfonic acid."""
        mol = Chem.MolFromSmiles("O=S(=O)(O)c1ccc2ccc3cccc4ccc1c2c34")
        results = _apply_nl_prepass(NL_RULES, mol, mol.GetNumAtoms())
        names = [NL_RULES[ridx].name for _, ridx in results]
        assert "aryl_sulfonate_so2" in names
        assert "aryl_sulfonate_so3" in names

    def test_n_nitramine_matches_nitramine_ring(self):
        """N-nitramine pattern matches cyclic nitramine."""
        mol = Chem.MolFromSmiles("O=[N+]([O-])NC1=NCCN1")
        results = _apply_nl_prepass(NL_RULES, mol, mol.GetNumAtoms())
        names = [NL_RULES[ridx].name for _, ridx in results]
        assert "n_nitramine_n2o" in names

    def test_n_nitramine_no_match_nitrobenzene(self):
        """Nitrobenzene (no N-H adjacent) must not match n_nitramine_n2o."""
        mol = Chem.MolFromSmiles("O=[N+]([O-])c1ccccc1")
        results = _apply_nl_prepass(NL_RULES, mol, mol.GetNumAtoms())
        names = [NL_RULES[ridx].name for _, ridx in results]
        assert "n_nitramine_n2o" not in names


# ---------------------------------------------------------------------------
# 2. Product mask atom counts
# ---------------------------------------------------------------------------


class TestNLProductMasks:
    def test_sulfonamide_so2_removes_s_and_two_o(self):
        """SO₂ extrusion removes exactly S + 2 O atoms from the product mask."""
        smi = "O=[N+]([O-])c1ccc(S(=O)(=O)Nc2ncccn2)cc1"
        mol = Chem.MolFromSmiles(smi)
        n_total = mol.GetNumAtoms()  # 19
        results = _apply_nl_prepass(NL_RULES, mol, n_total)
        so2_results = [(m, r) for m, r in results if NL_RULES[r].name == "aryl_sulfonamide_so2"]
        assert len(so2_results) == 1
        product_mask, _ = so2_results[0]
        # 3 atoms removed (S + 2×O = SO₂)
        assert int(product_mask.sum()) == n_total - 3

    def test_sulfonate_so3_removes_s_and_three_o(self):
        """SO₃ loss removes exactly S + 3 O atoms."""
        smi = "O=S(=O)(O)c1ccc2ccc3cccc4ccc1c2c34"  # pyrene sulfonic acid
        mol = Chem.MolFromSmiles(smi)
        n_total = mol.GetNumAtoms()  # 20
        results = _apply_nl_prepass(NL_RULES, mol, n_total)
        so3_results = [(m, r) for m, r in results if NL_RULES[r].name == "aryl_sulfonate_so3"]
        assert len(so3_results) == 1
        product_mask, _ = so3_results[0]
        assert int(product_mask.sum()) == n_total - 4  # S + 3×O

    def test_nitramine_n2o_removes_two_n_and_one_o(self):
        """N₂O extrusion removes exactly 2 N + 1 O atoms (N₂O)."""
        smi = "O=[N+]([O-])NC1=NCCN1"
        mol = Chem.MolFromSmiles(smi)
        n_total = mol.GetNumAtoms()  # 9
        results = _apply_nl_prepass(NL_RULES, mol, n_total)
        n2o_results = [(m, r) for m, r in results if NL_RULES[r].name == "n_nitramine_n2o"]
        assert len(n2o_results) == 1
        product_mask, _ = n2o_results[0]
        assert int(product_mask.sum()) == n_total - 3  # N + O + N = N₂O

    def test_deduplication_same_nl_set(self):
        """Multiple SMARTS embeddings on the same atoms must be deduplicated."""
        # Symmetric molecule where both orientations of the NL match hit the same atoms
        smi = "O=S(=O)(O)c1ccccc1"  # benzenesulfonic acid
        mol = Chem.MolFromSmiles(smi)
        results = _apply_nl_prepass(NL_RULES, mol, mol.GetNumAtoms())
        # Only 2 unique NL products (SO₂ and SO₃), not 4 (not doubled by embedding symmetry)
        names = [NL_RULES[r].name for _, r in results]
        assert names.count("aryl_sulfonate_so2") == 1
        assert names.count("aryl_sulfonate_so3") == 1


# ---------------------------------------------------------------------------
# 3. DAG injection
# ---------------------------------------------------------------------------


class TestInjectNLMasks:
    def test_new_formula_adds_node_and_edge(self):
        """A genuinely new NL product mask adds 1 node and 1 edge to the DAG."""
        smi = "O=[N+]([O-])c1ccc(S(=O)(=O)Nc2ncccn2)cc1"
        dag_base = _run_dag(smi, smarts_prepass=False)
        dag_nl = _run_dag(smi, smarts_prepass=True)
        # At least 1 new node from the NL prepass (sulfonamide SO₂ product)
        assert dag_nl["dag_num_nodes"] > dag_base["dag_num_nodes"]

    def test_so3_product_not_duplicated_when_already_in_bfs(self):
        """SO₃ loss (simple bridge cut) already in BFS should not add an extra node."""
        smi = "O=S(=O)(O)c1ccc2ccc3cccc4ccc1c2c34"  # pyrene sulfonic acid
        dag_base = _run_dag(smi, smarts_prepass=False)
        dag_nl = _run_dag(smi, smarts_prepass=True)
        # SO₃ product (Ar fragment from direct S–C cut) is in BFS; SO₂ product is new.
        # Net new nodes = 1 (only SO₂ product).
        delta = dag_nl["dag_num_nodes"] - dag_base["dag_num_nodes"]
        assert delta == 1

    def test_nl_node_has_depth2(self):
        """Injected NL product nodes must have min_depth=2."""
        from fragnnet.frag.compute_frags import MASK_DTYPE

        smi = "O=[N+]([O-])c1ccc(S(=O)(=O)Nc2ncccn2)cc1"
        mol_d = extract_mol_info(smi)
        mol = mol_d["mol"]
        num_nodes, num_edges, node_mask, edges, edge_mask, _ = get_fraggen_input_arrays(mol_d)

        nl_results = _apply_nl_prepass(NL_RULES, mol, num_nodes)
        nl_masks = [m for m, _ in nl_results]

        # Build minimal DAG scaffold (root only)
        root_mask = np.ones((1, num_nodes), dtype=MASK_DTYPE)
        depth_matrix = np.zeros((1, 4), dtype=MASK_DTYPE)
        depth_matrix[0, 0] = 1
        dag_edges = np.empty((0, 2), dtype=np.int64)
        dag_meta = {
            "nodes_min_depth": np.array([0], dtype=MASK_DTYPE),
            "edges_min_depth": np.array([], dtype=MASK_DTYPE),
            "reached_depth": 0,
            "force_stopped": False,
        }

        _, _, _, out_meta = _inject_nl_masks_into_dag(
            nl_masks, root_mask, depth_matrix, dag_edges, dag_meta, num_nodes, 1
        )
        assert all(d == 2 for d in out_meta["nodes_min_depth"][1:])
