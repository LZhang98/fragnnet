"""Model-level equivalence tests for the current cut-bond deduplication path.

This legacy chunking test module previously referenced a removed helper and chunk
constant. The current model encodes each directed bond once and gathers the rows
before pooling; compare that path with per-row encoding through the full forward.
"""

import pytest
import torch as th
from test_ce_multi_location import _build as _build_fragnnet
from test_cut_bond import _forward_kwargs_with_cut_bonds

from fragnnet.models.fragnnet import fragnnet_model


@pytest.mark.parametrize("mol_pool_type", ["mean", "sum", "max"])
@pytest.mark.parametrize("training", [False, True])
def test_deduplicated_cut_bonds_match_per_row_forward_and_gradients(
    monkeypatch, mol_pool_type, training
):
    model = _build_fragnnet(
        frag_node_feats=["cc", "base_formula", "depth", "h_formulae_idx", "cut_bond"],
        frag_edge_feats=[],
        mol_pool_type=mol_pool_type,
        cut_bond_hidden_dim=32,
    ).eval()
    # Keep dropout disabled to isolate encoding equivalence; autograd remains enabled
    # in the training case, just as in the previous memory-optimization regression.
    kwargs = _forward_kwargs_with_cut_bonds()
    row_count = kwargs["boundary_pair_in_idxs"].numel()
    encoded_counts = []
    handle = model.cut_bond_encoder.register_forward_hook(
        lambda _module, args, _output: encoded_counts.append(args[0].shape[0])
    )
    try:
        with th.set_grad_enabled(training):
            optimized = model(**kwargs)["pred_logprobs"]
            if training:
                optimized.square().sum().backward()
                optimized_grads = {
                    name: parameter.grad.clone()
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                }
                model.zero_grad(set_to_none=True)
            monkeypatch.setattr(
                fragnnet_model,
                "encode_cut_bond_pairs",
                lambda encoder, atoms, inside, outside: encoder(atoms[inside], atoms[outside]),
            )
            reference = model(**kwargs)["pred_logprobs"]
            if training:
                reference.square().sum().backward()
    finally:
        handle.remove()

    assert 0 < encoded_counts[0] < row_count
    assert encoded_counts[1] == row_count
    th.testing.assert_close(optimized, reference, atol=1e-6, rtol=1e-5)
    if training:
        assert any(name.startswith("cut_bond_encoder.") for name in optimized_grads)
        for name, parameter in model.named_parameters():
            if name in optimized_grads:
                assert parameter.grad is not None and th.isfinite(parameter.grad).all()
                th.testing.assert_close(
                    parameter.grad, optimized_grads[name], atol=1e-5, rtol=1e-4
                )
