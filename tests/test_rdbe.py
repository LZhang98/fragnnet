"""Tests for the rdbe fragment feature (ring-and-double-bond equivalents).

RDBE = 1 + sum_i n_i * (v_i - 2) / 2 -- the standard generalization of "C - H/2 + N/2 + 1" to
arbitrary valences, derived from ELEMENT_TO_VE rather than hardcoded for CHNO.

Two facts drive the implementation and are pinned here:

1. base_formula carries NO hydrogen -- its H column is identically zero -- so H must come from
   h_counts column 0 (the delta-0 / unshifted count). test_base_formula_has_no_hydrogen guards
   this, because if base_formula ever gained H the feature would silently double-count it.
2. h_counts contributes 0 dims to frag_node_feats_size, so requiring it for rdbe does NOT bundle
   a second feature into the arm. test_h_counts_adds_no_width guards that.

The value itself is checked against hand-computable molecules rather than a reimplementation of
the same formula, which would only test that the code equals itself.
"""

from __future__ import annotations

import pytest
import torch as th
from test_ce_multi_location import _build as _build_fragnnet

from fragnnet.models.fragnnet.frag_feat_utils import get_frag_graph
from fragnnet.models.fragnnet.frag_utils import compute_dags, extract_mol_info, get_node_feats
from fragnnet.utils.formula_utils import ELEMENTS

BASE = ["cc", "base_formula", "depth", "h_formulae_idx"]
BASE_H = [*BASE, "h_counts"]

# (smiles, label, textbook RDBE of the intact molecule)
KNOWN = [
    ("c1ccccc1", "benzene C6H6", 4.0),  # 3 C=C + 1 ring
    ("CCO", "ethanol C2H6O", 0.0),  # fully saturated, acyclic
    ("CCOC(=O)c1ccccc1", "ethyl benzoate C9H10O2", 5.0),  # ring 4 + ester C=O
    ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", "caffeine C8H10N4O2", 6.0),
    ("CC(=O)Oc1ccccc1C(=O)O", "aspirin C9H8O4", 6.0),
    ("ClC(Cl)(Cl)Cl", "carbon tetrachloride CCl4", 0.0),  # halogens count as H
    ("C1CCCCC1", "cyclohexane C6H12", 1.0),  # ring only, no unsaturation
]


def _root_rdbe(smiles: str, model) -> float:
    """RDBE of the intact precursor, via the model's own coefficient buffer."""
    mol_d = extract_mol_info(smiles)
    dag_d = compute_dags(
        mol_d,
        max_depth=1,
        h_prior=True,
        max_h_transfer=2,
        frag_max_time=int(1e6),
        isotopes=False,
        nb_isomorphic=False,
        wl_max_iterations=-1,
        multi_cut_bfs=False,
    )
    g = get_frag_graph(
        dag_d["dag"], frag_node_feats=BASE_H, frag_edge_feats=[], edges=False, bigraph=False
    )
    bf = get_node_feats(g.x, g.node_feat_idxs[0], "base_formula").reshape(g.x.shape[0], -1)
    hc = get_node_feats(g.x, g.node_feat_idxs[0], "h_counts").reshape(g.x.shape[0], -1)
    counts = bf.clone()
    counts[:, ELEMENTS.index("H")] = hc[:, 0].clamp(min=0)
    rdbe_x2 = 2 + (counts * model._rdbe_coef_buf).sum(dim=1)
    return rdbe_x2[0].item() / 2


@pytest.fixture(scope="module")
def model():
    return _build_fragnnet(frag_node_feats=[*BASE_H, "rdbe"], frag_edge_feats=[])


class TestRdbeValue:
    @pytest.mark.parametrize(("smiles", "label", "expected"), KNOWN)
    def test_known_molecule(self, smiles, label, expected, model):
        assert _root_rdbe(smiles, model) == pytest.approx(expected), label

    def test_coefficients_are_valence_minus_two(self, model):
        """C=+2, N=+1, O=0, halogens=-1, H=-1 (doubled, so the sum stays integral)."""
        coef = model._rdbe_coef_buf
        for elem, expected in [("C", 2), ("N", 1), ("O", 0), ("Cl", -1), ("S", 0), ("H", -1)]:
            assert coef[ELEMENTS.index(elem)].item() == expected, elem


class TestRdbeAssumptions:
    """If either of these breaks, the feature is silently wrong rather than loudly broken."""

    def test_base_formula_has_no_hydrogen(self):
        """H must come from h_counts; base_formula's H column is identically zero."""
        mol_d = extract_mol_info("CCOC(=O)c1ccccc1")  # C9H10O2
        dag_d = compute_dags(
            mol_d,
            max_depth=2,
            h_prior=True,
            max_h_transfer=2,
            frag_max_time=int(1e6),
            isotopes=False,
            nb_isomorphic=False,
            wl_max_iterations=-1,
            multi_cut_bfs=False,
        )
        g = get_frag_graph(
            dag_d["dag"], frag_node_feats=BASE, frag_edge_feats=[], edges=False, bigraph=False
        )
        bf = get_node_feats(g.x, g.node_feat_idxs[0], "base_formula").reshape(g.x.shape[0], -1)
        assert th.all(bf[:, ELEMENTS.index("H")] == 0)
        assert bf[0, ELEMENTS.index("C")] == 9  # heavy atoms ARE there

    def test_h_counts_delta0_is_the_true_h_count(self):
        """Column 0 of h_counts is the unshifted H count: 10 for ethyl benzoate C9H10O2."""
        mol_d = extract_mol_info("CCOC(=O)c1ccccc1")
        dag_d = compute_dags(
            mol_d,
            max_depth=1,
            h_prior=True,
            max_h_transfer=2,
            frag_max_time=int(1e6),
            isotopes=False,
            nb_isomorphic=False,
            wl_max_iterations=-1,
            multi_cut_bfs=False,
        )
        g = get_frag_graph(
            dag_d["dag"], frag_node_feats=BASE_H, frag_edge_feats=[], edges=False, bigraph=False
        )
        hc = get_node_feats(g.x, g.node_feat_idxs[0], "h_counts").reshape(g.x.shape[0], -1)
        assert hc[0, 0].item() == 10

    def test_h_counts_adds_no_width(self):
        """rdbe requires h_counts, so h_counts must not smuggle in a second feature."""
        without = _build_fragnnet(frag_node_feats=BASE, frag_edge_feats=[])
        with_h = _build_fragnnet(frag_node_feats=BASE_H, frag_edge_feats=[])
        assert with_h.frag_node_feats_size == without.frag_node_feats_size


class TestRdbeWiring:
    def test_requires_base_formula(self):
        """Rejected at construction, not hours into a run."""
        with pytest.raises(ValueError, match="rdbe requires base_formula"):
            _build_fragnnet(
                frag_node_feats=["cc", "depth", "h_formulae_idx", "h_counts", "rdbe"],
                frag_edge_feats=[],
            )

    def test_requires_h_counts(self):
        """base_formula alone cannot supply H, so rdbe must refuse rather than assume H=0."""
        with pytest.raises(ValueError, match="rdbe requires h_counts"):
            _build_fragnnet(frag_node_feats=[*BASE, "rdbe"], frag_edge_feats=[])

    def test_adds_expected_width(self, model):
        base = _build_fragnnet(frag_node_feats=BASE_H, frag_edge_feats=[])
        added = model.frag_node_feats_size - base.frag_node_feats_size
        assert added == model.rdbe_embedder.num_dim
