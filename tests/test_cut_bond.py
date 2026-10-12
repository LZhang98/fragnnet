"""Tests for the cut_bond fragment feature and CutBondEncoder.

cut_bond replaces boundary_pair, whose encoder is Linear(cat(in, out)) == W_in @ in +
W_out @ out + b. That is degenerate twice over:

1. additive -- a bond can never be more than the sum of two independent atom terms, so
   "C-N bond" cannot differ from "C contribution + N contribution", even at one cut bond;
2. linear -- it commutes through mean/sum pooling, so for >= 2 cut bonds the pooled feature
   equals Linear(mean(inside), mean(outside)) and WHICH atom bonded to which is discarded
   exactly (the multi-cut ring-cleavage regime).

The load-bearing test here is test_distinguishes_rewired_bonds: cutting a1-b1 + a2-b2 vs
a1-b2 + a2-b1 uses identical atom sets on each side and differs only in the pairing. It is
the exact property boundary_pair provably fails, so it is asserted against both encoders --
if it ever passes for the linear one the test has stopped measuring anything.
"""

from __future__ import annotations

import pytest
import torch as th
import torch.nn as nn
from test_cc_atom_dropout import _forward_kwargs
from test_ce_multi_location import _build as _build_fragnnet
from torch_geometric.nn import global_mean_pool

import fragnnet.models.fragnnet.fragnnet_model as fragnnet_model_module
from fragnnet.layers.cut_bond_encoder import CutBondEncoder, encode_cut_bond_pairs
from fragnnet.models.fragnnet.dag.layout import compute_boundary_pair_idxs, get_node_feats
from fragnnet.utils.formula_utils import th_long_to_mask

ATOM_DIM = 8
HIDDEN = 16


def _atoms(n: int, seed: int = 0) -> th.Tensor:
    return th.randn(n, ATOM_DIM, generator=th.Generator().manual_seed(seed))


def _encoder() -> CutBondEncoder:
    return CutBondEncoder(atom_dim=ATOM_DIM, hidden_dim=HIDDEN, out_dim=ATOM_DIM)


class TestCutBondEncoder:
    def test_output_shape_is_per_bond(self):
        enc = _encoder()
        out = enc(_atoms(5, 0), _atoms(5, 1))
        assert out.shape == (5, ATOM_DIM)

    def test_bond_is_more_than_sum_of_its_atoms(self):
        """The additive failure of boundary_pair, at a SINGLE bond.

        For a linear encoder f(in, out) = W_in @ in + W_out @ out + b, superposition holds:
        f(a, y) + f(x, b) == f(a, b) + f(x, y). A real bond encoding must break it.
        """
        enc = _encoder()
        a, b, x, y = (_atoms(1, s) for s in range(4))
        lhs = enc(a, y) + enc(x, b)
        rhs = enc(a, b) + enc(x, y)
        assert not th.allclose(lhs, rhs, atol=1e-4)

    def test_asymmetric_in_out(self):
        """inside (retained) and outside (lost) atoms are chemically distinct roles."""
        enc = _encoder()
        a, b = _atoms(1, 0), _atoms(1, 1)
        assert not th.allclose(enc(a, b), enc(b, a), atol=1e-4)

    def test_rejects_mismatched_shapes(self):
        enc = _encoder()
        with pytest.raises(ValueError, match="must have the same shape"):
            enc(_atoms(3, 0), _atoms(4, 1))

    def test_rejects_wrong_atom_width(self):
        enc = _encoder()
        wrong = th.randn(3, ATOM_DIM + 1)
        with pytest.raises(ValueError, match="expected"):
            enc(wrong, wrong)

    @pytest.mark.parametrize("bad", ["atom_dim", "hidden_dim", "out_dim"])
    def test_rejects_nonpositive_dims(self, bad):
        dims = {"atom_dim": ATOM_DIM, "hidden_dim": HIDDEN, "out_dim": ATOM_DIM, bad: 0}
        with pytest.raises(ValueError, match=f"{bad} must be positive"):
            CutBondEncoder(**dims)

    def test_gradients_flow_to_both_atoms(self):
        enc = _encoder()
        a, b = _atoms(3, 0).requires_grad_(), _atoms(3, 1).requires_grad_()
        enc(a, b).sum().backward()
        assert a.grad is not None and th.any(a.grad != 0.0)
        assert b.grad is not None and th.any(b.grad != 0.0)


class TestPairingSurvivesPooling:
    """The regression that motivates the whole feature."""

    @staticmethod
    def _pooled(encode, in_atoms, out_atoms):
        frag_idx = th.zeros(in_atoms.shape[0], dtype=th.long)  # all bonds -> one fragment
        return global_mean_pool(encode(in_atoms, out_atoms), frag_idx)

    def test_distinguishes_rewired_bonds(self):
        """a1-b1 + a2-b2  vs  a1-b2 + a2-b1: same atom sets, different bonds."""
        enc = _encoder()
        a1, a2, b1, b2 = (_atoms(1, s) for s in range(4))
        inside = th.cat([a1, a2])
        real = self._pooled(enc, inside, th.cat([b1, b2]))
        rewired = self._pooled(enc, inside, th.cat([b2, b1]))
        assert not th.allclose(real, rewired, atol=1e-4)

    def test_boundary_pair_encoder_fails_the_same_check(self):
        """Pins the defect being fixed: the linear encoder cannot tell the two apart.

        If this ever fails, boundary_pair's collapse is gone and the comparison above no
        longer measures anything -- so this asserts the baseline stays broken.
        """
        proj = nn.Linear(ATOM_DIM * 2, ATOM_DIM)  # exactly boundary_pair_proj

        def encode(in_atoms, out_atoms):
            return proj(th.cat([in_atoms, out_atoms], dim=-1))

        a1, a2, b1, b2 = (_atoms(1, s) for s in range(4))
        inside = th.cat([a1, a2])
        real = self._pooled(encode, inside, th.cat([b1, b2]))
        rewired = self._pooled(encode, inside, th.cat([b2, b1]))
        assert th.allclose(real, rewired, atol=1e-6)

        # and it collapses exactly to Linear(mean(inside), mean(outside))
        collapsed = proj(
            th.cat([inside.mean(0, keepdim=True), th.cat([b1, b2]).mean(0, keepdim=True)], dim=-1)
        )
        assert th.allclose(real, collapsed, atol=1e-5)


class TestEncodeCutBondPairs:
    """``encode_cut_bond_pairs`` must equal per-row encoding while encoding each pair once."""

    # Seven rows over three distinct directed pairs; (0, 1) and (1, 0) are one bond's two
    # orientations and must stay distinct.
    IN = th.tensor([0, 1, 0, 2, 0, 1, 2])
    OUT = th.tensor([1, 0, 1, 3, 1, 0, 3])

    def test_matches_per_row_encoding_with_repeats(self):
        enc, atoms = _encoder(), _atoms(4)
        rows = encode_cut_bond_pairs(enc, atoms, self.IN, self.OUT)
        assert rows.shape == (self.IN.numel(), ATOM_DIM)
        assert th.allclose(rows, enc(atoms[self.IN], atoms[self.OUT]), atol=1e-6)
        assert not th.allclose(rows[0], rows[1], atol=1e-4)  # orientation kept

    def test_encoder_sees_each_distinct_pair_once(self):
        enc, atoms = _encoder(), _atoms(4)
        seen_rows = []
        handle = enc.register_forward_hook(lambda _m, args, _o: seen_rows.append(len(args[0])))
        try:
            encode_cut_bond_pairs(enc, atoms, self.IN, self.OUT)
        finally:
            handle.remove()
        assert seen_rows == [3]

    def test_gradients_match_per_row_encoding(self):
        enc, weights = _encoder(), th.randn(self.IN.numel(), ATOM_DIM)
        grads = []
        for encode in (
            lambda a: encode_cut_bond_pairs(enc, a, self.IN, self.OUT),
            lambda a: enc(a[self.IN], a[self.OUT]),
        ):
            enc.zero_grad()
            atoms = _atoms(4).requires_grad_(True)
            (encode(atoms) * weights).sum().backward()
            grads.append([atoms.grad, *(p.grad.clone() for p in enc.parameters())])
        for dedup, per_row in zip(*grads):
            assert th.allclose(dedup, per_row, atol=1e-5)

    def test_empty_rows(self):
        empty = th.zeros(0, dtype=th.long)
        assert encode_cut_bond_pairs(_encoder(), _atoms(4), empty, empty).shape == (0, ATOM_DIM)

    @pytest.mark.parametrize(
        "in_idx, out_idx",
        [
            (th.tensor([0, 1]), th.tensor([1])),
            (th.tensor([[0, 1]]), th.tensor([[1, 0]])),
            (th.tensor([0, 1], dtype=th.int32), th.tensor([1, 0], dtype=th.int32)),
        ],
    )
    def test_rejects_bad_indices(self, in_idx, out_idx):
        with pytest.raises(ValueError):
            encode_cut_bond_pairs(_encoder(), _atoms(4), in_idx, out_idx)


def _forward_kwargs_with_cut_bonds(smiles: str = "CCOC(=O)c1ccccc1") -> dict:
    """Real forward batch plus the cut-bond indices the production batcher supplies.

    cut_bond deliberately reuses boundary_pair's precomputed boundary_pair_* indices so that
    existing DAGs work unchanged; get_mol_frag_batch fills them in training. The lightweight
    harness borrowed here does not, so derive them with the same function the pipeline uses.
    """
    kwargs = _forward_kwargs(smiles)
    mol_pyg, frag_pyg = kwargs["mol_pyg"], kwargs["frag_pyg"]
    frag_node_mask = th_long_to_mask(get_node_feats(frag_pyg.x, frag_pyg.node_feat_idxs[0], "cc"))
    frag_idxs, in_global, out_global = compute_boundary_pair_idxs(
        frag_node_mask,
        mol_pyg.edge_index,
        kwargs["mol_num_nodes"],
        kwargs["frag_num_nodes"],
    )
    if frag_idxs.numel() == 0:
        raise AssertionError(f"{smiles} produced no cut bonds; the test would be vacuous")
    kwargs["boundary_pair_frag_idxs"] = frag_idxs
    kwargs["boundary_pair_in_idxs"] = in_global
    kwargs["boundary_pair_out_idxs"] = out_global
    return kwargs


class TestFraGNNetIntegration:
    @staticmethod
    def _model(feats):
        return _build_fragnnet(frag_node_feats=feats, frag_edge_feats=[], cut_bond_hidden_dim=32)

    _BASE = ["cc", "base_formula", "depth", "h_formulae_idx"]

    def test_cut_bond_matches_boundary_pair_width(self):
        """Same width => swapping the two feature names is a clean one-variable comparison."""
        pair = self._model([*self._BASE, "boundary_pair"])
        cut = self._model([*self._BASE, "cut_bond"])
        assert cut.frag_node_feats_size == pair.frag_node_feats_size

    def test_cut_bond_adds_width_over_base(self):
        base = self._model(self._BASE)
        cut = self._model([*self._BASE, "cut_bond"])
        added = cut.frag_node_feats_size - base.frag_node_feats_size
        assert added == cut.mol_embedder.hidden_size

    def test_encoder_built_only_when_feature_enabled(self):
        assert hasattr(self._model([*self._BASE, "cut_bond"]), "cut_bond_encoder")
        assert not hasattr(self._model(self._BASE), "cut_bond_encoder")

    def test_forward_runs_on_a_real_molecule(self):
        """cut_bond reuses boundary_pair's precomputed indices, so existing DAGs must work."""
        m = self._model([*self._BASE, "cut_bond"]).eval()
        with th.no_grad():
            out = m(**_forward_kwargs_with_cut_bonds())["pred_logprobs"]
        assert th.isfinite(out).all()

    def test_cut_bond_changes_prediction_vs_boundary_pair(self):
        """Sanity that the feature is actually consumed, not silently dropped."""
        cut = self._model([*self._BASE, "cut_bond"]).eval()
        pair = self._model([*self._BASE, "boundary_pair"]).eval()
        kwargs = _forward_kwargs_with_cut_bonds()
        with th.no_grad():
            a = cut(**kwargs)["pred_logprobs"]
            b = pair(**kwargs)["pred_logprobs"]
        assert a.shape == b.shape
        assert not th.allclose(a, b, atol=1e-5)

    def test_deduplicated_encoding_matches_per_row(self, monkeypatch):
        """The forward's per-pair dedup reproduces encoding every cut-bond row separately."""
        model = self._model([*self._BASE, "cut_bond"]).eval()
        kwargs = _forward_kwargs_with_cut_bonds()
        pairs = th.stack([kwargs["boundary_pair_in_idxs"], kwargs["boundary_pair_out_idxs"]])
        assert th.unique(pairs, dim=1).shape[1] < pairs.shape[1]  # repeats exist: not vacuous
        with th.no_grad():
            deduplicated = model(**kwargs)["pred_logprobs"]
            monkeypatch.setattr(
                fragnnet_model_module,
                "encode_cut_bond_pairs",
                lambda encoder, atoms, in_idx, out_idx: encoder(atoms[in_idx], atoms[out_idx]),
            )
            per_row = model(**kwargs)["pred_logprobs"]
        assert th.allclose(deduplicated, per_row, atol=1e-6)
