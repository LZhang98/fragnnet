"""Training-only atom masking in the shared GNN wrapper (MASSter arm am1).

Off, it must add no parameter and draw nothing. On, it replaces whole projected atom inputs by one
learned vector in training only, and leaves every other parameter's initialization unchanged.
"""

import pytest
import torch as th

from fragnnet.layers.gnn import GNN


def _encoder(**kwargs) -> GNN:
    th.manual_seed(31)
    return GNN(
        hidden_size=16,
        num_layers=2,
        node_feats_size=5,
        edge_feats_size=3,
        gnn_type="GINE_GSA",
        dropout=0.0,
        normalization="batch",
        num_heads=4,
        **kwargs,
    )


def _chain_inputs(num_atoms: int) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """One chain molecule with both directions of every bond."""
    th.manual_seed(32)
    forward = th.stack([th.arange(num_atoms - 1), th.arange(1, num_atoms)])
    edge_index = th.cat([forward, forward.flip(0)], dim=1)
    x = th.randn(num_atoms, 5)
    edge_attr = th.randn(edge_index.shape[1], 3)
    return x, th.zeros(num_atoms, dtype=th.long), edge_index, edge_attr


def test_atom_mask_off_adds_no_parameter_and_draws_no_random_numbers() -> None:
    encoder = _encoder().train()
    inputs = _chain_inputs(6)
    rng_before = th.get_rng_state()

    encoder(*inputs)

    assert th.equal(th.get_rng_state(), rng_before)
    assert "atom_mask_embedding" not in encoder.state_dict()


def test_atom_mask_adds_one_zero_vector_and_keeps_every_other_initialization() -> None:
    plain_state = _encoder().state_dict()
    masked_state = _encoder(atom_mask=0.3).state_dict()

    assert set(masked_state) - set(plain_state) == {"atom_mask_embedding"}
    assert [key for key in masked_state if key in plain_state] == list(plain_state)
    assert all(th.equal(masked_state[key], plain_state[key]) for key in plain_state)
    assert th.equal(masked_state["atom_mask_embedding"], th.zeros(16))


def test_atom_mask_is_training_only() -> None:
    plain, masked = _encoder(), _encoder(atom_mask=0.3)
    inputs = _chain_inputs(6)

    assert th.equal(masked.eval()(*inputs), plain.eval()(*inputs))
    th.manual_seed(5)
    assert not th.allclose(masked.train()(*inputs), plain.train()(*inputs))


def test_atom_mask_replaces_whole_projected_inputs_by_the_learned_vector() -> None:
    encoder = _encoder(atom_mask=0.5).train()
    with th.no_grad():
        encoder.atom_mask_embedding.fill_(7.0)
    seen = {}
    encoder.gnn.register_forward_pre_hook(lambda _module, args: seen.update(z=args[0]))
    inputs = _chain_inputs(400)
    th.manual_seed(6)

    encoder(*inputs).sum().backward()

    projected = encoder.input_project(inputs[0])
    replaced = (seen["z"] == 7.0).all(dim=1)
    kept = (seen["z"] == projected).all(dim=1)
    assert (replaced ^ kept).all()
    assert 140 < int(replaced.sum()) < 260
    # The mask vector learns from the masked atoms.
    assert encoder.atom_mask_embedding.grad.abs().sum() > 0


@pytest.mark.parametrize("atom_mask", [1.0, -0.1])
def test_atom_mask_rejects_invalid_settings(atom_mask: float) -> None:
    with pytest.raises(ValueError, match="atom_mask must be in"):
        _encoder(atom_mask=atom_mask)
