"""Tests for ContextEmbedding -- the combined acquisition-metadata (stratum) embedding.

Scaffolded ahead of wiring (see docs/conditioning_refactor_plan.md). Verifies shape, that it
consumes ALL metadata inputs (a change to any input changes the output -- the property that
justifies one combined vector over four separate inserts), cross-metadata interaction capacity,
gradient flow, and the input-width guard against misordered/misconfigured metadata.
"""

from __future__ import annotations

import pytest
import torch as th

from fragnnet.model.base_model import ContextEmbedding


def _module(dims=(8, 4, 4, 4), hidden=16, out=12):
    return ContextEmbedding(input_dim=sum(dims), hidden_dim=hidden, out_dim=out), dims


class TestContextEmbedding:
    def test_output_shape(self):
        m, dims = _module()
        embeds = [th.randn(5, d) for d in dims]
        out = m(embeds)
        assert out.shape == (5, m.out_dim)

    def test_deterministic(self):
        m, dims = _module()
        m.eval()
        embeds = [th.randn(3, d) for d in dims]
        assert th.allclose(m(embeds), m(embeds))

    def test_depends_on_every_metadata_input(self):
        """Consolidation is only justified if the context reflects all metadata: perturbing
        any single input must change the output."""
        m, dims = _module()
        m.eval()
        base = [th.randn(4, d) for d in dims]
        out0 = m(base)
        for k in range(len(dims)):
            perturbed = [e.clone() for e in base]
            perturbed[k] = perturbed[k] + 1.0
            assert not th.allclose(m(perturbed), out0), f"output ignores metadata input {k}"

    def test_models_cross_metadata_interaction(self):
        """A nonlinear MLP over concatenated metadata can represent interactions: the effect of
        changing input A is not independent of input B (unlike a purely additive combiner)."""
        m, dims = _module()
        m.eval()
        a0 = [th.zeros(1, d) for d in dims]
        # effect of bumping input 0, measured at two different settings of input 1
        bump0 = a0[0] + 1.0
        e_lowB = m([bump0, a0[1], a0[2], a0[3]]) - m(a0)
        highB1 = a0[1] + 3.0
        e_highB = m([bump0, highB1, a0[2], a0[3]]) - m([a0[0], highB1, a0[2], a0[3]])
        # if the combiner were additive, the two deltas would be identical
        assert not th.allclose(e_lowB, e_highB, atol=1e-4)

    def test_gradient_flows_to_all_inputs(self):
        m, dims = _module()
        embeds = [th.randn(2, d, requires_grad=True) for d in dims]
        m(embeds).sum().backward()
        for k, e in enumerate(embeds):
            assert e.grad is not None and not th.allclose(e.grad, th.zeros_like(e.grad)), k

    def test_wrong_width_raises(self):
        m, dims = _module()
        embeds = [th.randn(2, d) for d in dims]
        embeds[0] = th.randn(2, dims[0] + 1)  # misconfigured width
        with pytest.raises(ValueError, match="expected"):
            m(embeds)

    def test_empty_raises(self):
        m, _ = _module()
        with pytest.raises(ValueError, match="at least one"):
            m([])

    def test_bad_input_dim_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            ContextEmbedding(input_dim=0, hidden_dim=8, out_dim=4)


class TestContextEmbeddingWiring:
    """Wiring of context_embedding into FraGNNetModel (dims + subsumption of the *_insert paths)."""

    def test_off_is_inert(self):
        from tests.test_ce_multi_location import _build

        m = _build("mol")  # context_embedding defaults False
        assert m.context_embedder is None
        assert m.context_mol_input_dim == 0 and m.context_mlp_input_dim == 0
        assert m.ce_mol_input_dim > 0  # individual CE injection intact

    def test_on_subsumes_individual_inserts_and_sizes_context(self):
        from tests.test_ce_multi_location import _build

        m = _build("mol", context_embedding=True, context_embedding_dim=24,
                   context_embedding_hidden_dim=16, context_insert_location="mol,mlp")
        assert m.context_embedder is not None
        # individual injection dims zeroed (context subsumes them)
        assert m.ce_mol_input_dim == m.ce_mlp_input_dim == 0
        assert m.prec_mol_input_dim == m.prec_mlp_input_dim == 0
        assert m.inst_mol_input_dim == m.inst_mlp_input_dim == 0
        assert m.frag_mode_mol_input_dim == m.frag_mode_mlp_input_dim == 0
        # context injects at both locations with the configured width
        assert m.context_mol_input_dim == 24 and m.context_mlp_input_dim == 24
        # nodeMLP input carries the context width (end-to-end dim check)
        base = _build("none")  # no CE, no context -> narrowest nodeMLP
        assert m.formula_module.input_layer.in_features == base.formula_module.input_layer.in_features + 24

    def test_mol_only_context_location(self):
        from tests.test_ce_multi_location import _build

        m = _build("mol", context_embedding=True, context_embedding_dim=16,
                   context_insert_location="mol")
        assert m.context_mol_input_dim == 16 and m.context_mlp_input_dim == 0

    def test_invalid_context_location_raises(self):
        from tests.test_ce_multi_location import _build

        with pytest.raises(ValueError, match="context_insert_location"):
            _build("mol", context_embedding=True, context_insert_location="mol,frag")
