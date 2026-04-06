"""Tests for the prepass_rule_id fragment node feature and prepass depth semantics.

Verifies that:
1. prepass_rule_id is all-zero when smarts_prepass=False.
2. prepass_rule_id is all-zero for molecules that don't match any FRAG_RULE.
3. At least one fragment has a non-zero prepass_rule_id for molecules that match
   a FRAG_RULE (acyclic 1,3-elimination for crf12_0; acyclic ester CO2 loss for
   crf13_2).
4. The correct rule dimension is set (crf12_0 → dim 0, crf13_2 → dim 2).
5. prepass_rule_id shape is (num_frags, len(FRAG_RULES)) and all values are 0 or 1.
6. The root fragment (index 0) is never tagged.
7. _apply_smarts_prepass returns (mask_a, mask_b, rule_idx) triples.
8. Structural constraints: prepass only adds fragments, never removes them.
9. Prepass-injected fragments have node depth 2 (not 1): prepass rules require
   ≥2 bond cuts + 1 new bond, so depth 2 correctly encodes the two-cut distance
   from the root.  See docs/frag_gen_prepass.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from fragnnet.frag.smarts_prepass import FRAG_RULES, _apply_smarts_prepass
from fragnnet.utils.frag_utils import (
    NODE_FEAT_TO_IDX,
    compute_dags,
    extract_mol_info,
    get_fraggen_input_arrays,
    get_node_feats,
)

NUM_RULES = len(FRAG_RULES)
# Rule indices by name for readable assertions
RULE_IDX = {rule.name: i for i, rule in enumerate(FRAG_RULES)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(smiles: str, smarts_prepass: bool, multi_cut_bfs: bool = False, max_depth: int = 3) -> dict:
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
        multi_cut_bfs=multi_cut_bfs,
        smarts_prepass=smarts_prepass,
    )


def _get_rule_id(dag_d: dict) -> np.ndarray:
    """Extract prepass_rule_id from DAG node features. Shape: (num_frags, NUM_RULES)."""
    dag = dag_d["dag"]
    x = dag.x.long()
    node_feat_idxs = dag.node_feat_idxs.long()
    feat = get_node_feats(x, node_feat_idxs[0], "prepass_rule_id")
    return feat.numpy()


# ---------------------------------------------------------------------------
# 1. _apply_smarts_prepass returns (mask_a, mask_b, rule_idx) triples
# ---------------------------------------------------------------------------


class TestApplySmartsPrepasReturnType:
    """Verify _apply_smarts_prepass returns tuples of length 3 with valid rule_idx."""

    def test_return_tuple_length(self):
        """Each result tuple must have exactly 3 elements."""
        mol_d = extract_mol_info("CCCC")  # butane — matches crf12_0
        num_nodes, num_edges, _, edges, edge_mask, _ = get_fraggen_input_arrays(mol_d)
        results = _apply_smarts_prepass(
            FRAG_RULES, mol_d["mol"], edge_mask, edges, num_nodes, num_edges
        )
        assert len(results) > 0, "butane should match crf12_0"
        for item in results:
            assert len(item) == 3, f"expected 3-tuple, got {len(item)}-tuple"

    def test_rule_idx_is_int_in_range(self):
        """rule_idx must be a valid 0-based index into FRAG_RULES."""
        mol_d = extract_mol_info("CC(=O)OC")  # methyl acetate — matches crf13_2
        num_nodes, num_edges, _, edges, edge_mask, _ = get_fraggen_input_arrays(mol_d)
        results = _apply_smarts_prepass(
            FRAG_RULES, mol_d["mol"], edge_mask, edges, num_nodes, num_edges
        )
        assert len(results) > 0
        for _mask_a, _mask_b, rule_idx in results:
            assert isinstance(rule_idx, int)
            assert 0 <= rule_idx < NUM_RULES

    def test_crf13_2_rule_idx(self):
        """Methyl acetate must produce results with rule_idx == RULE_IDX['crf13_2']."""
        mol_d = extract_mol_info("CC(=O)OC")
        num_nodes, num_edges, _, edges, edge_mask, _ = get_fraggen_input_arrays(mol_d)
        results = _apply_smarts_prepass(
            FRAG_RULES, mol_d["mol"], edge_mask, edges, num_nodes, num_edges
        )
        rule_idxs = {r for _, _, r in results}
        assert RULE_IDX["crf13_2"] in rule_idxs

    def test_no_match_returns_empty(self):
        """A molecule that matches no rule returns an empty list."""
        mol_d = extract_mol_info("c1ccccc1")  # benzene
        num_nodes, num_edges, _, edges, edge_mask, _ = get_fraggen_input_arrays(mol_d)
        results = _apply_smarts_prepass(
            FRAG_RULES, mol_d["mol"], edge_mask, edges, num_nodes, num_edges
        )
        assert results == []


# ---------------------------------------------------------------------------
# 2. prepass_rule_id all-zero when smarts_prepass=False
# ---------------------------------------------------------------------------


class TestRuleIdAllZeroWhenDisabled:
    """prepass_rule_id must be all-zero when the prepass is not used."""

    @pytest.mark.parametrize("smiles", ["CCCC", "CC(=O)OC", "c1ccccc1"])
    def test_all_zero_no_prepass(self, smiles):
        feat = _get_rule_id(_run(smiles, smarts_prepass=False))
        assert np.all(feat == 0), f"expected all-zero prepass_rule_id for {smiles} without prepass"

    @pytest.mark.parametrize("smiles", ["CCCC", "CC(=O)OC"])
    def test_all_zero_no_prepass_multi_cut(self, smiles):
        feat = _get_rule_id(_run(smiles, smarts_prepass=False, multi_cut_bfs=True))
        assert np.all(feat == 0)


# ---------------------------------------------------------------------------
# 3. prepass_rule_id all-zero for molecules that don't match any rule
# ---------------------------------------------------------------------------


class TestRuleIdAllZeroNoMatch:
    """Molecules with no FRAG_RULE match should produce all-zero prepass_rule_id."""

    def test_benzene(self):
        feat = _get_rule_id(_run("c1ccccc1", smarts_prepass=True))
        assert np.all(feat == 0)

    def test_ethanol_too_small(self):
        """Ethanol has fewer atoms than any rule's min_num_atoms."""
        feat = _get_rule_id(_run("CCO", smarts_prepass=True))
        assert np.all(feat == 0)


# ---------------------------------------------------------------------------
# 4. Non-zero prepass_rule_id for matching molecules
# ---------------------------------------------------------------------------


class TestRuleIdSetForMatchingMolecules:
    """Molecules with rule matches must have at least one non-zero prepass_rule_id row."""

    def test_3methylpentane_crf12_0(self):
        """3-methylpentane (CCC(C)CC) matches crf12_0 (1,3-elimination).

        The branch on C3 gives the inner fragment 3 atoms, above min_frag_atoms=3.
        Linear chains (butane, pentane) always produce 2-atom inner fragments
        which are filtered out by min_frag_atoms.
        """
        feat = _get_rule_id(_run("CCC(C)CC", smarts_prepass=True))
        assert np.any(feat > 0), "3-methylpentane should have at least one prepass fragment"

    def test_methyl_acetate_crf13_2(self):
        """Methyl acetate (CC(=O)OC) matches crf13_2 (acyclic ester CO2 loss)."""
        feat = _get_rule_id(_run("CC(=O)OC", smarts_prepass=True))
        assert np.any(feat > 0), "methyl acetate should have at least one prepass fragment"

    def test_3methylpentane_multi_cut_bfs(self):
        feat = _get_rule_id(_run("CCC(C)CC", smarts_prepass=True, multi_cut_bfs=True))
        assert np.any(feat > 0)

    def test_methyl_acetate_multi_cut_bfs(self):
        feat = _get_rule_id(_run("CC(=O)OC", smarts_prepass=True, multi_cut_bfs=True))
        assert np.any(feat > 0)


# ---------------------------------------------------------------------------
# 5. Correct rule dimension is activated
# ---------------------------------------------------------------------------


class TestCorrectRuleDimension:
    """The correct column in prepass_rule_id must be set for each rule."""

    def test_methyl_acetate_sets_crf13_2_dim(self):
        """Methyl acetate fragments must have dim RULE_IDX['crf13_2'] set to 1.

        Note: methyl acetate's C-C(=O)-O-C chain also matches crf12_0 (any 4-atom
        path), so crf12_0 may also fire — we only assert crf13_2 is present.
        """
        feat = _get_rule_id(_run("CC(=O)OC", smarts_prepass=True))
        crf13_2_idx = RULE_IDX["crf13_2"]
        assert np.any(feat[:, crf13_2_idx] == 1), (
            f"expected dim {crf13_2_idx} (crf13_2) to be set for methyl acetate"
        )

    def test_3methylpentane_sets_crf12_0_dim(self):
        """3-methylpentane fragments must have dim RULE_IDX['crf12_0'] set to 1."""
        feat = _get_rule_id(_run("CCC(C)CC", smarts_prepass=True))
        crf12_0_idx = RULE_IDX["crf12_0"]
        assert np.any(feat[:, crf12_0_idx] == 1), (
            f"expected dim {crf12_0_idx} (crf12_0) to be set for 3-methylpentane"
        )
        # crf13_2 should not fire on a pure hydrocarbon
        crf13_2_idx = RULE_IDX["crf13_2"]
        assert np.all(feat[:, crf13_2_idx] == 0), (
            f"dim {crf13_2_idx} (crf13_2) should not be set for a hydrocarbon"
        )


# ---------------------------------------------------------------------------
# 6. Shape and value constraints
# ---------------------------------------------------------------------------


class TestRuleIdShapeAndValues:
    """prepass_rule_id shape is (num_frags, NUM_RULES) with values in {0, 1}."""

    @pytest.mark.parametrize(
        "smiles,prepass",
        [
            ("CCCC", False),
            ("CCCC", True),
            ("CC(=O)OC", True),
            ("c1ccccc1", True),
        ],
    )
    def test_shape(self, smiles, prepass):
        dag_d = _run(smiles, smarts_prepass=prepass)
        feat = _get_rule_id(dag_d)
        assert feat.ndim == 2
        assert feat.shape[1] == NUM_RULES
        assert feat.shape[0] == dag_d["dag"].num_nodes

    @pytest.mark.parametrize(
        "smiles,prepass",
        [
            ("CCCC", False),
            ("CCCC", True),
            ("CC(=O)OC", True),
            ("CCC(C)CC", True),
        ],
    )
    def test_binary_values(self, smiles, prepass):
        feat = _get_rule_id(_run(smiles, smarts_prepass=prepass))
        assert np.all((feat == 0) | (feat == 1)), "prepass_rule_id must be binary (0 or 1)"

    def test_feature_size_same_with_and_without_prepass(self):
        """prepass_rule_id is always stored (all-zero when disabled); feature dim must not change."""
        for smiles in ("CCCC", "CC(=O)OC", "c1ccccc1"):
            d_off = _run(smiles, smarts_prepass=False)
            d_on = _run(smiles, smarts_prepass=True)
            assert d_on["dag"].num_features == d_off["dag"].num_features, (
                f"{smiles}: feature dim changed between prepass=True and prepass=False"
            )


# ---------------------------------------------------------------------------
# 7. Root fragment (index 0) is never tagged
# ---------------------------------------------------------------------------


class TestRootNeverTagged:
    """The root node (full molecule, index 0) must always have all-zero prepass_rule_id."""

    @pytest.mark.parametrize("smiles", ["CCCC", "CC(=O)OC", "CCC(C)CC", "CCCCC"])
    def test_root_all_zero(self, smiles):
        feat = _get_rule_id(_run(smiles, smarts_prepass=True))
        assert np.all(feat[0] == 0), "root fragment must never be tagged as prepass"


# ---------------------------------------------------------------------------
# 8. Structural constraints
# ---------------------------------------------------------------------------


class TestStructuralConstraints:
    """Prepass only adds fragments/edges, never removes them."""

    @pytest.mark.parametrize("smiles", ["c1ccccc1", "CCO"])
    def test_no_match_node_count_unchanged(self, smiles):
        d_off = _run(smiles, smarts_prepass=False)
        d_on = _run(smiles, smarts_prepass=True)
        assert d_on["dag"].num_nodes == d_off["dag"].num_nodes

    @pytest.mark.parametrize("smiles", ["c1ccccc1", "CCO"])
    def test_no_match_edge_count_unchanged(self, smiles):
        d_off = _run(smiles, smarts_prepass=False)
        d_on = _run(smiles, smarts_prepass=True)
        assert d_on["dag"].num_edges == d_off["dag"].num_edges

    @pytest.mark.parametrize("smiles", ["CCC(C)CC", "CC(=O)OC", "CCCCC"])
    def test_prepass_never_removes_fragments(self, smiles):
        d_off = _run(smiles, smarts_prepass=False)
        d_on = _run(smiles, smarts_prepass=True)
        assert d_on["dag"].num_nodes >= d_off["dag"].num_nodes


# ---------------------------------------------------------------------------
# 9. Prepass fragment depth is 2 (not 1)
# ---------------------------------------------------------------------------


def _get_depth_onehot(dag_d: dict) -> np.ndarray:
    """Extract the depth one-hot matrix from DAG node features.

    Returns shape (num_frags, max_depth+1).
    """
    dag = dag_d["dag"]
    x = dag.x.long()
    node_feat_idxs = dag.node_feat_idxs.long()
    feat = get_node_feats(x, node_feat_idxs[0], "depth")
    return feat.numpy()


class TestPrepasFragmentDepth:
    """Prepass-injected fragments must carry node depth 2, not 1.

    Rationale: each prepass rule cuts ≥2 bonds from the root and forms 1 new
    bond.  Two cuts = two BFS steps = depth 2.  See docs/frag_gen_prepass.md.
    """

    def test_methyl_acetate_prepass_frags_at_depth2(self):
        """All fragments tagged as prepass must have depth-2 one-hot set."""
        dag_d = _run("CC(=O)OC", smarts_prepass=True, max_depth=4)
        rule_id = _get_rule_id(dag_d)           # (num_frags, NUM_RULES)
        depth = _get_depth_onehot(dag_d)        # (num_frags, max_depth+1)
        prepass_rows = np.where(rule_id.any(axis=1))[0]
        assert len(prepass_rows) > 0, "expected at least one prepass fragment"
        for row in prepass_rows:
            assert depth[row, 2] == 1, (
                f"fragment {row} tagged as prepass but depth one-hot is {depth[row]} "
                f"(expected depth index 2 to be 1)"
            )

    def test_3methylpentane_prepass_frags_at_depth2(self):
        """Same check for a crf12_0 match (3-methylpentane)."""
        dag_d = _run("CCC(C)CC", smarts_prepass=True, max_depth=4)
        rule_id = _get_rule_id(dag_d)
        depth = _get_depth_onehot(dag_d)
        prepass_rows = np.where(rule_id.any(axis=1))[0]
        assert len(prepass_rows) > 0
        for row in prepass_rows:
            assert depth[row, 2] == 1, (
                f"fragment {row} tagged as prepass but depth one-hot is {depth[row]}"
            )

    def test_prepass_frags_not_at_depth1(self):
        """No prepass fragment should have depth-1 one-hot set."""
        for smiles in ("CC(=O)OC", "CCC(C)CC"):
            dag_d = _run(smiles, smarts_prepass=True, max_depth=4)
            rule_id = _get_rule_id(dag_d)
            depth = _get_depth_onehot(dag_d)
            prepass_rows = np.where(rule_id.any(axis=1))[0]
            for row in prepass_rows:
                assert depth[row, 1] == 0, (
                    f"{smiles}: prepass fragment {row} has depth 1 "
                    f"(one-hot {depth[row]}); expected depth 2"
                )

    def test_bfs_depth1_frags_unchanged(self):
        """BFS-generated depth-1 fragments must still have depth 1."""
        dag_d = _run("CC(=O)OC", smarts_prepass=True, max_depth=4)
        rule_id = _get_rule_id(dag_d)
        depth = _get_depth_onehot(dag_d)
        bfs_rows = np.where(~rule_id.any(axis=1))[0]
        # Exclude the root (index 0, depth 0); remaining BFS rows should
        # include depth-1 fragments.
        bfs_non_root = bfs_rows[bfs_rows != 0]
        # At least some BFS fragments should still be at depth 1.
        has_depth1 = np.any(depth[bfs_non_root, 1] == 1)
        assert has_depth1, "BFS-generated depth-1 fragments should still exist"

    def test_nodes_min_depth_is_2_for_prepass(self):
        """nodes_min_depth in dag_frag_meta must be 2 for prepass fragments."""
        dag_d = _run("CC(=O)OC", smarts_prepass=True, max_depth=4)
        rule_id = _get_rule_id(dag_d)
        nodes_min_depth = dag_d["nodes_min_depth"]  # shape (num_frags,)
        prepass_rows = np.where(rule_id.any(axis=1))[0]
        assert len(prepass_rows) > 0
        for row in prepass_rows:
            assert nodes_min_depth[row] == 2, (
                f"fragment {row}: nodes_min_depth={nodes_min_depth[row]}, expected 2"
            )
