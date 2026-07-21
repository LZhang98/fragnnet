"""Tests for the adduct-gated ion-channel residual head.

The static ``ion_channel_factorized`` head adds a node x channel residual that the formula MLP
can already express, and the existing ``AdductScaler`` adds a node-invariant adduct x channel
affine. The adduct-gated head is the meaningful addition: gating the low-rank node factor by
``spec_prec_type`` yields a node x channel x adduct *trilinear* residual that neither existing
component can represent. These tests pin the wiring, the identity start, gradient flow into the
node projection, and that the adduct index actually routes the residual.
"""

from __future__ import annotations

import pytest
import torch
from test_cc_atom_dropout import _build_model, _forward_kwargs
from test_ce_multi_location import _build

from fragnnet.model.fragnnet_model import FraGNNetModel

PREC_TYPES = ["[M+H]+", "[M+Na]+", "[M-H]-"]
RANK = 4


def _gated_model() -> FraGNNetModel:
    return _build(
        "mol",
        ion_channel_adduct_gated=True,
        ion_channel_rank=RANK,
        prec_types=PREC_TYPES,
    )


def _synthetic_inputs(model: FraGNNetModel, num_nodes: int = 6, batch_size: int = 3):
    """Build (logits, frag_embed, spec_prec_type, frag_node_batch_idxs) for the apply method."""
    torch.manual_seed(0)
    in_dim = model.ion_channel_node_proj.in_features
    logits = torch.randn(num_nodes, model.mlp_output_dim)
    frag_embed = torch.randn(num_nodes, in_dim)
    # Distinct adduct per sample so gating differences are observable.
    spec_prec_type = torch.arange(batch_size) % len(PREC_TYPES)
    frag_node_batch_idxs = torch.arange(num_nodes) % batch_size
    return logits, frag_embed, spec_prec_type, frag_node_batch_idxs


class TestIonChannelAdductGated:
    def test_builds_gate_and_omits_bias(self) -> None:
        model = _gated_model()
        assert model.ion_channel_node_proj.out_features == RANK
        assert model.ion_channel_embed.num_embeddings == model.mlp_output_dim
        assert model.ion_channel_embed.embedding_dim == RANK
        # Gate table has one row per known adduct plus an unknown fallback slot.
        assert model.ion_channel_adduct_gate.num_embeddings == len(PREC_TYPES) + 1
        assert model.ion_channel_adduct_gate.embedding_dim == RANK
        # AdductScaler owns the adduct x channel additive term, so this head carries no bias.
        assert model.ion_channel_bias is None

    def test_identity_start(self) -> None:
        """Zero-init node projection + ones-init gate => residual is exactly zero at init."""
        model = _gated_model()
        logits, frag_embed, prec, batch = _synthetic_inputs(model)
        out = model._apply_ion_channel_factorized(logits, frag_embed, prec, batch)
        assert torch.equal(out, logits)

    def test_gradient_flows_to_node_proj_at_init(self) -> None:
        """Ones gate and nonzero channel embedding keep node_proj learnable despite zero output."""
        model = _gated_model()
        logits, frag_embed, prec, batch = _synthetic_inputs(model)
        out = model._apply_ion_channel_factorized(logits, frag_embed, prec, batch)
        out.sum().backward()
        grad = model.ion_channel_node_proj.weight.grad
        assert grad is not None
        assert torch.count_nonzero(grad) > 0

    def test_adduct_index_routes_residual(self) -> None:
        """After perturbing node_proj and giving adducts distinct gates, the same node embedding
        gets different residuals under different adducts."""
        model = _gated_model()
        with torch.no_grad():
            torch.nn.init.normal_(model.ion_channel_node_proj.weight, std=0.5)
            # Make each adduct's gate distinct.
            model.ion_channel_adduct_gate.weight.copy_(
                torch.arange(len(PREC_TYPES) + 1, dtype=torch.float32).unsqueeze(1)
                * torch.ones(1, RANK)
            )
        in_dim = model.ion_channel_node_proj.in_features
        frag_embed = torch.randn(1, in_dim)
        base = torch.zeros(1, model.mlp_output_dim)
        node_batch = torch.zeros(1, dtype=torch.long)

        res_a = model._apply_ion_channel_factorized(base, frag_embed, torch.tensor([0]), node_batch)
        res_b = model._apply_ion_channel_factorized(base, frag_embed, torch.tensor([1]), node_batch)
        assert not torch.allclose(res_a, res_b)

    def test_gate_ones_makes_adducts_identical(self) -> None:
        """Sanity: with the ones-init gate, adduct choice cannot change the residual yet."""
        model = _gated_model()
        with torch.no_grad():
            torch.nn.init.normal_(model.ion_channel_node_proj.weight, std=0.5)
        in_dim = model.ion_channel_node_proj.in_features
        frag_embed = torch.randn(1, in_dim)
        base = torch.zeros(1, model.mlp_output_dim)
        node_batch = torch.zeros(1, dtype=torch.long)
        res_a = model._apply_ion_channel_factorized(base, frag_embed, torch.tensor([0]), node_batch)
        res_b = model._apply_ion_channel_factorized(base, frag_embed, torch.tensor([2]), node_batch)
        assert torch.allclose(res_a, res_b)

    def test_requires_prec_type(self) -> None:
        model = _gated_model()
        logits, frag_embed, _prec, batch = _synthetic_inputs(model)
        with pytest.raises(ValueError, match="requires spec_prec_type"):
            model._apply_ion_channel_factorized(logits, frag_embed, None, batch)

    def test_conflicts_with_factorized(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            _build("mol", ion_channel_factorized=True, ion_channel_adduct_gated=True)

    def test_conflicts_with_candidate_head(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            _build(
                "mol",
                candidate_ion_head=True,
                candidate_ion_formula_features=False,
                ion_channel_adduct_gated=True,
            )


class TestIonChannelAdductGatedForward:
    def test_forward_backward_grads_ion_channel(self) -> None:
        """End-to-end: a real forward pass routes through the gated head and its params learn."""
        model = _build_model(cc_atom_dropout=0.0, ion_channel_adduct_gated=True)
        model.train()
        out = model(**_forward_kwargs())
        loss = -out["pred_logprobs"].mean()
        loss.backward()
        grads = {
            n: p.grad
            for n, p in model.named_parameters()
            if n.startswith("ion_channel_") and p.grad is not None
        }
        # node_proj must receive a nonzero gradient at init despite the zero-valued residual.
        assert "ion_channel_node_proj.weight" in grads
        node_grad = grads["ion_channel_node_proj.weight"]
        assert torch.isfinite(node_grad).all() and torch.any(node_grad != 0)
