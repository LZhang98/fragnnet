"""Tests for context-gated mol-GNN message passing (mol_gnn_type="GINE_GATED").

Stock GINE shares one message function across every collision energy and adduct: context is
broadcast onto every atom's input features and concatenated, so the model must discover any
energy-by-bond interaction inside a shared MLP. GatedGINE makes it structural, scaling each
bond's message by a context-conditioned scalar g_ij = 1 + tanh(...) bounded to [0, 2].

The load-bearing property guarded here is the strict-superset claim: the gate readout is
zero-initialized, so g_ij == 1 exactly at init and GatedGINE must be bit-identical to GINE on
the first forward. If that breaks, "can only help" stops being true and a null result becomes
uninterpretable.
"""

from __future__ import annotations

import pytest
import torch as th

# The test_* imports reuse sibling harnesses rather than duplicating ~50 lines of DAG setup:
# _build_fragnnet is the smallest valid FraGNNetModel (its docstring invites reuse) and
# _forward_kwargs builds a real forward batch from a SMILES (ethyl benzoate). isort sorts them
# into the third-party block because known-first-party is just "fragnnet".
from test_cc_atom_dropout import _forward_kwargs
from test_ce_multi_location import _build as _build_fragnnet

from fragnnet.layers.gnn import GNN, GatedGINE

HIDDEN = 16
EDGE_FEATS = 4
CTX_DIM = 8
NUM_LAYERS = 3


def _graph(num_nodes: int = 6, num_edges: int = 10, num_graphs: int = 2, seed: int = 0):
    """Build a small batched graph: node feats, batch index, edge index/attr, context."""
    g = th.Generator().manual_seed(seed)
    x = th.randn(num_nodes, HIDDEN, generator=g)
    batch_index = th.arange(num_graphs).repeat_interleave(num_nodes // num_graphs)
    # keep every edge inside one graph (edges across graphs never occur in PyG batches)
    edge_index = th.stack(
        [
            th.randint(0, num_nodes // num_graphs, (num_edges,), generator=g),
            th.randint(0, num_nodes // num_graphs, (num_edges,), generator=g),
        ]
    )
    edge_attr = th.randn(num_edges, EDGE_FEATS, generator=g)
    ctx = th.randn(num_graphs, CTX_DIM, generator=g)
    return x, batch_index, edge_index, edge_attr, ctx


def _build(gate_mode: str, normalization: str = "layer") -> GatedGINE:
    return GatedGINE(
        hidden_size=HIDDEN,
        node_feats_size=HIDDEN,
        edge_feats_size=EDGE_FEATS,
        num_layers=NUM_LAYERS,
        dropout=0.0,
        normalization=normalization,
        ctx_dim=CTX_DIM,
        gate_mode=gate_mode,
        gate_hidden_dim=8,
    )


class TestGateInit:
    """The gate must start at exactly 1.0 so GatedGINE == GINE at init."""

    @pytest.mark.parametrize("gate_mode", ["static", "dynamic"])
    def test_gate_readout_is_zero_init(self, gate_mode):
        m = _build(gate_mode)
        for mlp in m.gate_mlps:
            assert th.all(mlp[-1].weight == 0.0)
            assert th.all(mlp[-1].bias == 0.0)

    @pytest.mark.parametrize("gate_mode", ["static", "dynamic"])
    def test_gate_value_is_exactly_one_at_init(self, gate_mode):
        """1 + tanh(0) == 1: the multiplicative identity, not merely close to it."""
        m = _build(gate_mode)
        feats = th.randn(10, m.gate_mlps[0][0].in_features)
        assert th.equal(m._gate(m.gate_mlps[0], feats), th.ones(10, 1))

    @pytest.mark.parametrize("gate_mode", ["static", "dynamic"])
    def test_gate_mlp_count(self, gate_mode):
        """static reuses one gate across layers; dynamic needs one per layer."""
        m = _build(gate_mode)
        assert len(m.gate_mlps) == (1 if gate_mode == "static" else NUM_LAYERS)

    def test_gate_input_dims_differ_by_mode(self):
        """dynamic additionally consumes x_src and x_dst (2 * hidden more than static)."""
        static, dynamic = _build("static"), _build("dynamic")
        assert static.gate_mlps[0][0].in_features == HIDDEN + CTX_DIM
        assert dynamic.gate_mlps[0][0].in_features == 3 * HIDDEN + CTX_DIM


class TestIdentityAtInit:
    """At init GatedGINE must reproduce GINE bit-for-bit, given the same weights."""

    @pytest.mark.parametrize("gate_mode", ["static", "dynamic"])
    def test_matches_stock_gine_at_init(self, gate_mode):
        x, batch_index, edge_index, edge_attr, ctx = _graph()
        gated = GNN(
            hidden_size=HIDDEN,
            num_layers=NUM_LAYERS,
            node_feats_size=HIDDEN,
            edge_feats_size=EDGE_FEATS,
            gnn_type="GINE_GATED",
            dropout=0.0,
            normalization="layer",
            ctx_dim=CTX_DIM,
            gate_mode=gate_mode,
            gate_hidden_dim=8,
        ).eval()
        stock = GNN(
            hidden_size=HIDDEN,
            num_layers=NUM_LAYERS,
            node_feats_size=HIDDEN,
            edge_feats_size=EDGE_FEATS,
            gnn_type="GINE",
            dropout=0.0,
            normalization="layer",
        ).eval()
        # copy the shared (non-gate) weights across so only the gate can cause a difference
        stock.load_state_dict(
            {k: v for k, v in gated.state_dict().items() if "gate_mlps" not in k}, strict=True
        )
        with th.no_grad():
            out_gated = gated(x, batch_index, edge_index, edge_attr, ctx=ctx)
            out_stock = stock(x, batch_index, edge_index, edge_attr)
        assert th.allclose(out_gated, out_stock, atol=1e-6)

    def test_context_is_inert_at_init_but_shapes_flow(self):
        """Two different contexts must give identical output while the gate is still 1.0."""
        x, batch_index, edge_index, edge_attr, ctx = _graph()
        m = _build("dynamic").eval()
        with th.no_grad():
            a = m(x, batch_index, edge_index, edge_attr, ctx)
            b = m(x, batch_index, edge_index, edge_attr, ctx + 5.0)
        assert th.allclose(a, b, atol=1e-6)
        assert a.shape == (x.shape[0], HIDDEN)


class TestGateIsLive:
    """Once the gate readout is nonzero, context must actually change the output."""

    @pytest.mark.parametrize("gate_mode", ["static", "dynamic"])
    def test_context_changes_output_after_gate_perturbation(self, gate_mode):
        x, batch_index, edge_index, edge_attr, ctx = _graph()
        m = _build(gate_mode).eval()
        with th.no_grad():
            for mlp in m.gate_mlps:
                mlp[-1].weight.normal_(0.0, 0.5)
            a = m(x, batch_index, edge_index, edge_attr, ctx)
            b = m(x, batch_index, edge_index, edge_attr, ctx + 5.0)
        assert not th.allclose(a, b, atol=1e-4)

    def test_zero_gate_kills_all_messages(self):
        """g == 0 must leave only the GINE self-term nn((1+eps) * x_i) -- i.e. no neighbor info.

        Drives the gate to its floor via a large negative bias (1 + tanh(-inf) -> 0) and checks
        the result equals passing an empty edge set.
        """
        x, batch_index, edge_index, edge_attr, ctx = _graph()
        m = _build("static").eval()
        with th.no_grad():
            m.gate_mlps[0][-1].bias.fill_(-50.0)  # tanh saturates -> gate == 0
            gated_off = m(x, batch_index, edge_index, edge_attr, ctx)
            empty = th.zeros(2, 0, dtype=edge_index.dtype)
            no_edges = m(x, batch_index, empty, edge_attr[:0], ctx)
        assert th.allclose(gated_off, no_edges, atol=1e-6)

    def test_gate_stays_in_bounds_under_extreme_input(self):
        """1 + tanh(...) is bounded to [0, 2] no matter how large the readout gets.

        The interval is closed, not open: tanh saturates to exactly +/-1 in float32, which is
        what makes the fully-closed gate in test_zero_gate_kills_all_messages reachable.
        """
        m = _build("static")
        with th.no_grad():
            m.gate_mlps[0][-1].weight.normal_(0.0, 100.0)
            feats = th.randn(64, m.gate_mlps[0][0].in_features) * 100.0
            g = m._gate(m.gate_mlps[0], feats)
        assert th.all(g >= 0.0) and th.all(g <= 2.0)
        assert th.isfinite(g).all()


class TestValidation:
    def test_bad_gate_mode_raises(self):
        with pytest.raises(ValueError, match="gate_mode must be 'static' or 'dynamic'"):
            _build("bogus")

    def test_nonpositive_ctx_dim_raises(self):
        with pytest.raises(ValueError, match="ctx_dim must be positive"):
            GatedGINE(
                hidden_size=HIDDEN,
                node_feats_size=HIDDEN,
                edge_feats_size=EDGE_FEATS,
                num_layers=1,
                dropout=0.0,
                normalization="none",
                ctx_dim=0,
                gate_mode="static",
                gate_hidden_dim=8,
            )

    def test_missing_ctx_raises(self):
        x, batch_index, edge_index, edge_attr, _ = _graph()
        gnn = GNN(
            hidden_size=HIDDEN,
            num_layers=1,
            node_feats_size=HIDDEN,
            edge_feats_size=EDGE_FEATS,
            gnn_type="GINE_GATED",
            dropout=0.0,
            normalization="none",
            ctx_dim=CTX_DIM,
            gate_mode="static",
            gate_hidden_dim=8,
        )
        with pytest.raises(ValueError, match="requires ctx"):
            gnn(x, batch_index, edge_index, edge_attr)

    def test_ungated_gnn_ignores_ctx(self):
        """Passing ctx to a non-gated type must be harmless, not an error."""
        x, batch_index, edge_index, edge_attr, ctx = _graph()
        gnn = GNN(
            hidden_size=HIDDEN,
            num_layers=1,
            node_feats_size=HIDDEN,
            edge_feats_size=EDGE_FEATS,
            gnn_type="GINE",
            dropout=0.0,
            normalization="none",
        ).eval()
        with th.no_grad():
            assert th.allclose(
                gnn(x, batch_index, edge_index, edge_attr, ctx=ctx),
                gnn(x, batch_index, edge_index, edge_attr),
            )


class TestGradients:
    @pytest.mark.parametrize("gate_mode", ["static", "dynamic"])
    def test_gate_receives_gradient(self, gate_mode):
        """The gate is zero-init but must not be gradient-dead, or it could never learn."""
        x, batch_index, edge_index, edge_attr, ctx = _graph()
        m = _build(gate_mode)
        m(x, batch_index, edge_index, edge_attr, ctx).sum().backward()
        for mlp in m.gate_mlps:
            assert mlp[-1].weight.grad is not None
            assert th.any(mlp[-1].weight.grad != 0.0)


class TestFraGNNetIntegration:
    """End-to-end wiring: the context vector must actually reach the gate inside a real model."""

    @staticmethod
    def _model(gnn_type: str, **overrides):
        return _build_fragnnet(
            mol_gnn_type=gnn_type,
            context_embedding=True,
            context_embedding_dim=12,
            context_insert_location="mlp",
            **overrides,
        )

    def test_requires_context_embedding(self):
        """Without context_embedding there is no vector to condition the gate on."""
        with pytest.raises(ValueError, match="requires context_embedding=True"):
            _build_fragnnet(mol_gnn_type="GINE_GATED", context_embedding=False)

    def test_flag_reflects_gnn_type(self):
        assert self._model("GINE_GATED").mol_msg_gate is True
        assert self._model("GINE").mol_msg_gate is False

    @pytest.mark.parametrize("gate_mode", ["static", "dynamic"])
    def test_forward_matches_stock_gine_at_init(self, gate_mode):
        """The whole model, on a real molecule, must be unchanged by adding the gate at init."""
        gated = self._model("GINE_GATED", mol_msg_gate_mode=gate_mode).eval()
        stock = self._model("GINE").eval()
        stock.load_state_dict(
            {k: v for k, v in gated.state_dict().items() if "gate_mlps" not in k}, strict=True
        )
        kwargs = _forward_kwargs()
        with th.no_grad():
            out_gated = gated(**kwargs)
            out_stock = stock(**kwargs)
        assert th.allclose(out_gated["pred_logprobs"], out_stock["pred_logprobs"], atol=1e-6)

    @pytest.mark.parametrize("gate_mode", ["static", "dynamic"])
    def test_forward_is_finite_and_gate_is_live(self, gate_mode):
        """Once the gate readout is nonzero it must move the real model's prediction."""
        m = self._model("GINE_GATED", mol_msg_gate_mode=gate_mode).eval()
        kwargs = _forward_kwargs()
        with th.no_grad():
            base = m(**kwargs)["pred_logprobs"]
            for mlp in m.mol_embedder.gnn.gate_mlps:
                mlp[-1].weight.normal_(0.0, 0.5)
            perturbed = m(**kwargs)["pred_logprobs"]
        assert th.isfinite(base).all() and th.isfinite(perturbed).all()
        assert not th.allclose(base, perturbed, atol=1e-5)

    def test_gate_is_ce_dependent_end_to_end(self):
        """The point of the whole change: with the gate live, collision energy must change the
        mol-encoder's atom embeddings, not merely rescale downstream outputs.

        context_insert_location="mlp" keeps context OUT of the mol input concat, so the gate is
        the only path by which CE can reach the mol GNN. Comparing pooled atom embeddings across
        two CEs therefore isolates the gate.
        """
        m = self._model("GINE_GATED", mol_msg_gate_mode="dynamic").eval()
        lo, hi = _forward_kwargs(), _forward_kwargs()
        hi["spec_ce"] = th.tensor([90.0])

        def _atom_embed(kwargs):
            captured = {}
            handle = m.mol_embedder.register_forward_hook(
                lambda _mod, _inp, out: captured.setdefault("out", out)
            )
            with th.no_grad():
                m(**kwargs)
            handle.remove()
            return captured["out"]

        # gate still at 1.0 -> mol encoder is provably CE-blind
        assert th.allclose(_atom_embed(lo), _atom_embed(hi), atol=1e-6)
        with th.no_grad():
            for mlp in m.mol_embedder.gnn.gate_mlps:
                mlp[-1].weight.normal_(0.0, 0.5)
        # gate live -> the same atoms now embed differently at 30 vs 90 NCE
        assert not th.allclose(_atom_embed(lo), _atom_embed(hi), atol=1e-5)
