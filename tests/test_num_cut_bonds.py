"""Tests for the num_cut_bonds fragment feature: how many bonds were broken to make a fragment.

mol_pool is a mean, so boundary_pair / cut_bond pool bond multiplicity away -- a fragment cut at
one bond and the same fragment cut at three give embeddings of the same magnitude. This feature
restores the count, derived from the same precomputed boundary_pair_* indices the batcher already
supplies on every forward, so it costs no stored DAG columns.

IMPORTANT -- the feature is a NO-OP on the current production data. Under plain single-bond BFS
every cut adds exactly one depth level, so cut count == depth and `depth` already carries it.
The production DAGs are exactly that (nist20_ma_mi_d3_h4_isoFalse_smpFalse: no _mbfs suffix,
smp=False). The count only decouples from depth where a single BFS step breaks several bonds --
multi_cut_bfs ring cleavage or the SMARTS prepass -- i.e. on the _mbfs / _ringmerge DAGs. Both
facts are pinned below (test_redundant_with_depth_on_plain_bfs,
test_decouples_from_depth_under_multi_cut) so the feature's applicability is checked, not assumed.

The other subtlety: mol_params.pyg_bigraph is True, so each cut bond arrives twice (once per
direction) as the same (frag, in, out) triple. The count must deduplicate or it is exactly 2x.
"""

from __future__ import annotations

import torch as th
from test_cc_atom_dropout import _forward_kwargs
from test_ce_multi_location import _build as _build_fragnnet

from fragnnet.utils.frag_feat_utils import get_frag_graph
from fragnnet.utils.frag_utils import (
    compute_boundary_pair_idxs,
    compute_dags,
    extract_mol_info,
    get_node_feats,
    th_long_to_mask,
)

BASE = ["cc", "base_formula", "depth", "h_formulae_idx"]


def _cut_counts_and_depth(
    smiles: str = "CCOC(=O)c1ccccc1", max_depth: int = 3, multi_cut: bool = False
):
    """Per-fragment DEDUPLICATED cut-bond count and depth, straight from the DAG."""
    mol_d = extract_mol_info(smiles)
    dag_d = compute_dags(
        mol_d,
        max_depth=max_depth,
        h_prior=True,
        max_h_transfer=2,
        frag_max_time=int(1e6),
        isotopes=False,
        nb_isomorphic=False,
        wl_max_iterations=-1,
        multi_cut_bfs=multi_cut,
    )
    g = get_frag_graph(
        dag_d["dag"], frag_node_feats=BASE, frag_edge_feats=[], edges=False, bigraph=False
    )
    mol_pyg = _forward_kwargs(smiles)["mol_pyg"]
    n_frag, n_mol = g.x.shape[0], mol_pyg.num_nodes
    frag_node_mask = th_long_to_mask(get_node_feats(g.x, g.node_feat_idxs[0], "cc"))
    frag_idxs, in_g, out_g = compute_boundary_pair_idxs(
        frag_node_mask,
        mol_pyg.edge_index,
        th.tensor([0, n_mol]),
        th.tensor([0, n_frag]),
    )
    # mirror the model: the bidirectional mol graph supplies each cut bond twice
    uniq = th.unique(th.stack([frag_idxs, in_g, out_g], dim=1), dim=0)
    counts = th.bincount(uniq[:, 0], minlength=n_frag)
    depth = get_node_feats(g.x, g.node_feat_idxs[0], "depth").reshape(n_frag, -1).argmax(dim=1)
    return counts, depth


class TestCutCountSemantics:
    def test_root_has_no_cut_bonds(self):
        """The intact precursor was made by cutting nothing."""
        counts, _ = _cut_counts_and_depth()
        assert counts[0].item() == 0

    def test_non_root_fragments_have_cuts(self):
        counts, _ = _cut_counts_and_depth()
        assert th.all(counts[1:] > 0)

    def test_counts_are_deduplicated_bond_counts(self):
        """A cut bond counts ONCE, though the bidirectional mol graph supplies it twice.

        mol_params.pyg_bigraph is True, so each cut bond yields one boundary-pair entry per
        direction and both map to the same (frag, in, out) triple. Without deduplication every
        count is exactly doubled.
        """
        counts, _ = _cut_counts_and_depth("CCO", max_depth=1)
        assert counts[0].item() == 0
        assert sorted(counts[1:].tolist()) == [1, 1, 1, 1]  # each single-bond cut -> 2 frags

    def test_redundant_with_depth_on_plain_bfs(self):
        """DOCUMENTS A LIMITATION, not a success.

        Under plain single-bond BFS -- which is what the production DAGs use
        (nist20_ma_mi_d3_h4_isoFalse_smpFalse: no _mbfs suffix, smp=False) -- every cut adds
        exactly one depth level, so cut count == depth and this feature carries NO information
        the model does not already have. It is only worth enabling on DAGs with multi-cut nodes
        (_mbfs / _ringmerge / SMARTS prepass). If this test ever fails, plain BFS has gained
        multi-cut nodes and the feature became useful on the baseline data too.
        """
        counts, depth = _cut_counts_and_depth()
        for d in depth.unique():
            vals = counts[depth == d].unique()
            assert vals.numel() == 1, f"depth {int(d)} has cut counts {vals.tolist()}"
            assert vals.item() == int(d), f"depth {int(d)} != cut count {vals.item()}"

    def test_decouples_from_depth_under_multi_cut(self):
        """Where the feature earns its place: ring cleavage breaks the depth == cuts identity.

        Naphthalene under multi_cut_bfs opens a fused ring by breaking 2 bonds in ONE BFS step,
        so fragments at the same depth differ in cut count.
        """
        counts, depth = _cut_counts_and_depth("c1ccc2ccccc2c1", max_depth=2, multi_cut=True)
        spreads = {
            int(d): sorted(counts[depth == d].unique().tolist())
            for d in depth.unique()
            if (depth == d).sum() > 1
        }
        assert any(
            len(v) > 1 for v in spreads.values()
        ), f"cut count never varies within a depth even under multi-cut: {spreads}"


class TestFraGNNetIntegration:
    @staticmethod
    def _model(feats):
        return _build_fragnnet(frag_node_feats=feats, frag_edge_feats=[])

    def test_adds_expected_width(self):
        base = self._model(BASE)
        cut = self._model([*BASE, "num_cut_bonds"])
        added = cut.frag_node_feats_size - base.frag_node_feats_size
        assert added == cut.num_cut_bonds_embedder.num_dim

    def test_forward_runs_and_changes_prediction(self):
        """Reuses boundary_pair's indices, so existing DAGs must work -- and it must be consumed."""
        from test_cut_bond import _forward_kwargs_with_cut_bonds

        base = self._model(BASE).eval()
        cut = self._model([*BASE, "num_cut_bonds"]).eval()
        kwargs = _forward_kwargs_with_cut_bonds()
        with th.no_grad():
            a = base(**kwargs)["pred_logprobs"]
            b = cut(**kwargs)["pred_logprobs"]
        assert th.isfinite(b).all()
        assert a.shape == b.shape
        assert not th.allclose(a, b, atol=1e-5)

    def test_composes_with_cut_bond(self):
        """num_cut_bonds restores the multiplicity that cut_bond's mean pooling discards."""
        m = self._model([*BASE, "cut_bond", "num_cut_bonds"])
        assert (
            m.frag_node_feats_size
            == self._model([*BASE, "cut_bond"]).frag_node_feats_size
            + m.num_cut_bonds_embedder.num_dim
        )
