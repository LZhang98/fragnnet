"""The retained-fragment connectivity branch (arm ``fc1``).

``docs/plans/masster/masster_hierarchical_fragment_graph_plan.md`` adds a small
message-passing branch that runs a k-layer GINE over each candidate fragment's OWN retained
(post-cut) bond graph, seeded from raw atom/bond features -- never the shared encoder's
globally attended ``atom_embeddings`` -- plus a per-(fragment, atom) retained-bond degree and
boundary-cut-bond count. It is pooled twice (all retained atoms; cut-site atoms only) and added
to the intensity token as a zero-initialized residual, so at its default (``layers=0``) the
model is exactly unchanged. ``masster_fragment_connectivity_messages: False`` (arm ``fc1p``) is
the matched pooled control: the identical module, features and pooling, but over an EMPTY edge
set, so no message crosses a retained bond.

This file covers, in order: the two new join primitives in ``component_pooling.py``
(``fragment_internal_edges``, ``component_membership_rows``) against a hand-built fixture, then
the model-level branch (construction, safe-start, shapes, raw-feature sourcing, gradients,
backend parity) and the ``fc1p`` messages-off control.
"""

import pytest
import torch as th
import torch.nn as nn

from fragnnet import runner
from fragnnet.models.masster.masster_component_pooling import (
    component_membership_rows,
    fragment_internal_edges,
)
from fragnnet.models.masster.masster_pl import MassterPL
from tests.test_masster_model import _build_batch, _build_model

UNIQUE_KWARGS = {"max_masks": 4, "max_enumerated_masks": 8, "unique_fragment_mode": True}


@pytest.fixture
def batch():
    """Two small molecules, so per-spectrum effects can be told apart."""
    return _build_batch(("CCOC", "CCO"))


def _forward(model, batch, **kwargs) -> dict[str, th.Tensor]:
    return model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0, 0], device=batch.x.device),
        spec_nce=th.tensor([10.0, 40.0], device=batch.x.device),
        **kwargs,
    )


# 1. `fragment_internal_edges` / `component_membership_rows` against a hand-built fixture.
#
# 5 atoms (0-1-2-3 chain plus atom 4 branching off atom 1); 4 chemical bonds:
# b0=(0,1), b1=(1,2), b2=(2,3), b3=(1,4). Four fragments span the three kinds the spec asks for:
# frag0={1} (single atom), frag1={0,1} and frag2={1,2,3} (multi-atom-connected), frag3=all 5
# atoms (whole molecule).


def _internal_edges_fixture() -> tuple[th.Tensor, th.Tensor, th.Tensor, int]:
    chemical_bond_index = th.tensor([[0, 1, 2, 1], [1, 2, 3, 4]], dtype=th.long)
    atom_indices = th.tensor([1, 0, 1, 1, 2, 3, 0, 1, 2, 3, 4], dtype=th.long)
    atom_component_indices = th.tensor([0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3], dtype=th.long)
    return chemical_bond_index, atom_indices, atom_component_indices, 5


def _brute_force_internal_edges(
    chemical_bond_index: th.Tensor,
    atom_indices: th.Tensor,
    atom_component_indices: th.Tensor,
) -> set[tuple[int, int, int, int]]:
    """Every (fragment, atom_low, atom_high, bond_index) with both endpoints retained."""
    bonds = [
        (chemical_bond_index[0, bond].item(), chemical_bond_index[1, bond].item())
        for bond in range(chemical_bond_index.shape[1])
    ]
    membership: dict[int, set[int]] = {}
    for row in range(atom_indices.shape[0]):
        membership.setdefault(atom_component_indices[row].item(), set()).add(
            atom_indices[row].item()
        )
    expected = set()
    for fragment, atoms in membership.items():
        for bond_index, (u, v) in enumerate(bonds):
            if u in atoms and v in atoms:
                expected.add((fragment, min(u, v), max(u, v), bond_index))
    return expected


def _as_quadruples(
    row_u: th.Tensor,
    row_v: th.Tensor,
    bond_indices: th.Tensor,
    atom_indices: th.Tensor,
    atom_component_indices: th.Tensor,
) -> set[tuple[int, int, int, int]]:
    return {
        (
            atom_component_indices[row_u[i]].item(),
            atom_indices[row_u[i]].item(),
            atom_indices[row_v[i]].item(),
            bond_indices[i].item(),
        )
        for i in range(row_u.shape[0])
    }


def test_fragment_internal_edges_matches_brute_force() -> None:
    chemical_bond_index, atom_indices, atom_component_indices, num_atoms = _internal_edges_fixture()

    row_u, row_v, bond_indices = fragment_internal_edges(
        chemical_bond_index, atom_indices, atom_component_indices, num_atoms=num_atoms
    )

    actual = _as_quadruples(row_u, row_v, bond_indices, atom_indices, atom_component_indices)
    expected = _brute_force_internal_edges(
        chemical_bond_index, atom_indices, atom_component_indices
    )
    assert actual == expected
    # Membership-row invariants: both endpoints belong to the same fragment as their row.
    assert th.equal(atom_component_indices[row_u], atom_component_indices[row_v])
    assert (atom_indices[row_u] < atom_indices[row_v]).all()


def test_fragment_internal_edges_excludes_boundary_bonds() -> None:
    """No returned (fragment, bond) pair is also one of that fragment's own boundary bonds."""
    chemical_bond_index, atom_indices, atom_component_indices, num_atoms = _internal_edges_fixture()
    row_u, row_v, bond_indices = fragment_internal_edges(
        chemical_bond_index, atom_indices, atom_component_indices, num_atoms=num_atoms
    )
    internal_pairs = {
        (atom_component_indices[row_u[i]].item(), bond_indices[i].item())
        for i in range(row_u.shape[0])
    }
    # frag0={1}: bonds 0, 1, 3 all boundary (atom 1's whole degree). frag1={0,1}: bonds 1, 3.
    # frag2={1,2,3}: bonds 0, 3. frag3=all atoms: no boundary bonds.
    boundary_fragment_indices = th.tensor([0, 0, 0, 1, 1, 2, 2])
    boundary_bond_indices = th.tensor([0, 1, 3, 1, 3, 0, 3])
    boundary_pairs = {
        (boundary_fragment_indices[i].item(), boundary_bond_indices[i].item())
        for i in range(boundary_fragment_indices.shape[0])
    }
    assert internal_pairs.isdisjoint(boundary_pairs)


def test_fragment_internal_edges_deduplicates_each_bond_once_per_fragment() -> None:
    chemical_bond_index, atom_indices, atom_component_indices, num_atoms = _internal_edges_fixture()
    row_u, row_v, bond_indices = fragment_internal_edges(
        chemical_bond_index, atom_indices, atom_component_indices, num_atoms=num_atoms
    )
    pairs = [
        (atom_component_indices[row_u[i]].item(), bond_indices[i].item())
        for i in range(row_u.shape[0])
    ]
    assert len(pairs) == len(set(pairs))
    assert len(pairs) == 7  # 1 (frag1) + 2 (frag2) + 4 (frag3), by brute force above.


def test_fragment_internal_edges_empty_inputs() -> None:
    empty_long = th.empty(0, dtype=th.long)
    for chemical_bond_index, atom_indices, atom_component_indices in (
        (th.empty((2, 0), dtype=th.long), empty_long, empty_long),
        (_internal_edges_fixture()[0], empty_long, empty_long),
    ):
        row_u, row_v, bond_indices = fragment_internal_edges(
            chemical_bond_index, atom_indices, atom_component_indices, num_atoms=5
        )
        for tensor in (row_u, row_v, bond_indices):
            assert tensor.shape == (0,)
            assert tensor.dtype == th.long


def test_fragment_internal_edges_single_atom_fragment() -> None:
    """frag0={1} is single-atom and must contribute zero internal edges."""
    chemical_bond_index, atom_indices, atom_component_indices, num_atoms = _internal_edges_fixture()
    row_u, _, _ = fragment_internal_edges(
        chemical_bond_index, atom_indices, atom_component_indices, num_atoms=num_atoms
    )
    assert not (atom_component_indices[row_u] == 0).any()


def test_component_membership_rows_matches_brute_force() -> None:
    _, atom_indices, atom_component_indices, num_atoms = _internal_edges_fixture()
    query_atoms = th.tensor([2, 1, 4], dtype=th.long)
    query_components = th.tensor([2, 3, 3], dtype=th.long)

    rows = component_membership_rows(
        atom_indices, atom_component_indices, query_atoms, query_components, num_atoms=num_atoms
    )

    assert atom_indices[rows].tolist() == query_atoms.tolist()
    assert atom_component_indices[rows].tolist() == query_components.tolist()


def test_component_membership_rows_rejects_a_non_member_pair() -> None:
    _, atom_indices, atom_component_indices, num_atoms = _internal_edges_fixture()
    with pytest.raises(ValueError, match="not a member"):
        component_membership_rows(
            atom_indices,
            atom_component_indices,
            th.tensor([4], dtype=th.long),
            th.tensor([1], dtype=th.long),  # frag1={0, 1}; atom 4 is not a member.
            num_atoms=num_atoms,
        )


# 2. The branch at the model level (arm fc1 / control fc1p).


def test_fragment_connectivity_layers_zero_is_bit_identical_default(batch) -> None:
    """hidden_dim and messages are inert when layers == 0."""
    explicit = _build_model(
        batch,
        fragment_connectivity_layers=0,
        fragment_connectivity_hidden_dim=999,
        fragment_connectivity_messages=False,
        **UNIQUE_KWARGS,
    )
    implicit = _build_model(batch, **UNIQUE_KWARGS)

    assert list(explicit.state_dict()) == list(implicit.state_dict())
    assert all(
        th.equal(explicit.state_dict()[key], implicit.state_dict()[key])
        for key in implicit.state_dict()
    )
    assert th.equal(
        _forward(explicit.eval(), batch)["pred_logprobs"],
        _forward(implicit.eval(), batch)["pred_logprobs"],
    )


def test_fragment_connectivity_module_only_appears_when_enabled(batch) -> None:
    control = _build_model(batch, **UNIQUE_KWARGS).state_dict()
    arm = _build_model(
        batch,
        fragment_connectivity_layers=1,
        fragment_connectivity_hidden_dim=4,
        **UNIQUE_KWARGS,
    ).state_dict()

    new_keys = set(arm) - set(control)
    assert new_keys and all(
        key.startswith(("fragment_connectivity_encoder.", "intensity_head.connectivity_residual."))
        for key in new_keys
    )
    assert set(control) <= set(arm)
    assert all(th.equal(arm[key], control[key]) for key in control)


def test_fragment_connectivity_requires_unique_fragment_mode(batch) -> None:
    with pytest.raises(ValueError, match="requires unique_fragment_mode"):
        _build_model(batch, fragment_connectivity_layers=1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"fragment_connectivity_layers": -1},
        {"fragment_connectivity_layers": 1, "fragment_connectivity_hidden_dim": 0},
    ],
)
def test_fragment_connectivity_rejects_bad_dims(batch, overrides: dict) -> None:
    with pytest.raises(ValueError):
        _build_model(batch, **UNIQUE_KWARGS, **overrides)


def test_fragment_connectivity_residual_is_zero_at_initialization(batch) -> None:
    model = _build_model(
        batch,
        fragment_connectivity_layers=1,
        fragment_connectivity_hidden_dim=4,
        **UNIQUE_KWARGS,
    )
    residual = model.intensity_head.connectivity_residual
    assert th.equal(residual.weight, th.zeros_like(residual.weight))
    assert th.equal(residual.bias, th.zeros_like(residual.bias))


def test_fragment_connectivity_arm_matches_its_control_at_initialization(batch) -> None:
    control = _build_model(batch, **UNIQUE_KWARGS).eval()
    arm = _build_model(
        batch,
        fragment_connectivity_layers=1,
        fragment_connectivity_hidden_dim=4,
        **UNIQUE_KWARGS,
    ).eval()

    assert th.equal(
        _forward(arm, batch)["pred_logprobs"], _forward(control, batch)["pred_logprobs"]
    )


def test_fragment_connectivity_embeddings_shape_and_finite(
    batch, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _build_model(
        batch,
        fragment_connectivity_layers=1,
        fragment_connectivity_hidden_dim=4,
        **UNIQUE_KWARGS,
    ).eval()
    captured: list[th.Tensor] = []
    real_forward = model.intensity_head.forward

    def _spy(*args, **kwargs):
        captured.append(kwargs["connectivity_embeddings"])
        return real_forward(*args, **kwargs)

    monkeypatch.setattr(model.intensity_head, "forward", _spy)
    with th.no_grad():
        _forward(model, batch)

    connectivity = captured[-1]
    assert connectivity.shape[0] > 0
    assert connectivity.shape[1] == 8  # 2 * hidden_dim
    assert th.isfinite(connectivity).all()


def test_fragment_connectivity_reads_raw_features_not_atom_embeddings(
    batch, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _build_model(
        batch,
        fragment_connectivity_layers=1,
        fragment_connectivity_hidden_dim=4,
        **UNIQUE_KWARGS,
    ).eval()
    captured_atom_indices: list[th.Tensor] = []
    real_method = model._fragment_connectivity_embeddings

    def _spy_method(mol_pyg, fragment_atom_indices, *args):
        captured_atom_indices.append(fragment_atom_indices)
        return real_method(mol_pyg, fragment_atom_indices, *args)

    monkeypatch.setattr(model, "_fragment_connectivity_embeddings", _spy_method)
    captured_x: list[th.Tensor] = []
    real_encoder_forward = model.fragment_connectivity_encoder.forward

    def _spy_encoder(x, batch_index, edge_index, edge_attr, ctx=None):
        captured_x.append(x)
        return real_encoder_forward(x, batch_index, edge_index, edge_attr, ctx)

    monkeypatch.setattr(model.fragment_connectivity_encoder, "forward", _spy_encoder)

    with th.no_grad():
        _forward(model, batch)

    node_feature_dim = batch.x.shape[1]
    fragment_atom_indices = captured_atom_indices[-1]
    x_local = captured_x[-1]
    assert th.equal(x_local[:, :node_feature_dim], batch.x[fragment_atom_indices])


def test_single_atom_fragment_has_zero_internal_bonds_and_finite_connectivity() -> None:
    """A terminal single-atom fragment: connectivity_all is finite despite zero internal bonds."""
    batch = _build_batch(("CCO",))
    model = _build_model(
        batch,
        fragment_connectivity_layers=1,
        fragment_connectivity_hidden_dim=4,
        **UNIQUE_KWARGS,
    ).eval()
    bond = next(
        bond_index
        for bond_index in range(batch.chemical_bond_index.shape[1])
        if {
            batch.chemical_bond_index[0, bond_index].item(),
            batch.chemical_bond_index[1, bond_index].item(),
        }
        == {0, 1}
    )

    with th.no_grad():
        connectivity = model._fragment_connectivity_embeddings(
            batch,
            th.tensor([0], dtype=th.long),
            th.tensor([0], dtype=th.long),
            th.tensor([bond], dtype=th.long),
            th.tensor([0], dtype=th.long),
            num_fragments=1,
        )

    hidden = model.fragment_connectivity_dim
    assert connectivity.shape == (1, 2 * hidden)
    assert th.isfinite(connectivity[:, :hidden]).all()


def test_intact_precursor_has_zero_cut_site_pool() -> None:
    batch = _build_batch(("CCO",))
    model = _build_model(
        batch,
        fragment_connectivity_layers=1,
        fragment_connectivity_hidden_dim=4,
        **UNIQUE_KWARGS,
    ).eval()

    with th.no_grad():
        connectivity = model._fragment_connectivity_embeddings(
            batch,
            th.tensor([0, 1, 2], dtype=th.long),
            th.tensor([0, 0, 0], dtype=th.long),
            th.empty(0, dtype=th.long),
            th.empty(0, dtype=th.long),
            num_fragments=1,
        )

    hidden = model.fragment_connectivity_dim
    assert th.equal(connectivity[:, hidden:], th.zeros(1, hidden))


def test_fragment_connectivity_gradients_reach_the_encoder_and_residual(batch) -> None:
    model = _build_model(
        batch,
        fragment_connectivity_layers=1,
        fragment_connectivity_hidden_dim=4,
        **UNIQUE_KWARGS,
    ).train()
    # The residual is zero-initialized (see the init test above), which would also zero the
    # gradient flowing back INTO the encoder through it; perturb it first so this test exercises
    # the encoder's own gradient path, not just the residual's.
    with th.no_grad():
        nn.init.normal_(model.intensity_head.connectivity_residual.weight, std=0.1)

    _forward(model, batch)["pred_logprobs"].exp().square().sum().backward()

    encoder_grads = [
        parameter.grad
        for parameter in model.fragment_connectivity_encoder.parameters()
        if parameter.grad is not None
    ]
    assert encoder_grads and all(grad.abs().sum() > 0 for grad in encoder_grads)
    residual = model.intensity_head.connectivity_residual
    assert residual.weight.grad is not None and residual.weight.grad.abs().sum() > 0
    assert residual.bias.grad is not None and residual.bias.grad.abs().sum() > 0


@pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA not available")
def test_fragment_connectivity_cpu_and_cuda_backends_agree() -> None:
    batch = _build_batch(("CCOC", "CCO")).cuda()
    cpu_backend = (
        _build_model(
            batch,
            fragment_connectivity_layers=1,
            fragment_connectivity_hidden_dim=4,
            **UNIQUE_KWARGS,
        )
        .cuda()
        .eval()
    )
    with th.no_grad():
        nn.init.normal_(cpu_backend.intensity_head.connectivity_residual.weight, std=0.1)
    cuda_backend = (
        _build_model(
            batch,
            fragment_connectivity_layers=1,
            fragment_connectivity_hidden_dim=4,
            candidate_backend="cuda",
            **UNIQUE_KWARGS,
        )
        .cuda()
        .eval()
    )
    cuda_backend.load_state_dict(cpu_backend.state_dict(), strict=True)
    inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 0], device="cuda"),
        "spec_nce": th.tensor([10.0, 40.0], device="cuda"),
    }

    with th.no_grad():
        expected = cpu_backend(**inputs)
        actual = cuda_backend(**inputs)

    for name in ("pred_mzs", "pred_batch_idxs", "pred_logprobs"):
        th.testing.assert_close(expected[name], actual[name], atol=1e-6, rtol=1e-5)


def test_fragment_connectivity_pl_threads_both_hparams() -> None:
    config = runner.load_config("config", None, None, model_type="masster")
    config.update(
        model_type="masster",
        compile=False,
        sparse_cosine_similarity=True,
        oos_tolerance_multiple=1,
        gaussian_renormalize=True,
        mol_hidden_size=16,
        mol_num_layers=1,
        masster_context_dim=8,
        masster_metadata_embedding_dim=4,
        masster_head_hidden_dim=16,
        masster_intensity_transformer_dim=16,
        masster_intensity_transformer_num_layers=2,
        masster_intensity_transformer_feedforward_dim=32,
        masster_intensity_transformer_num_inducing_points=4,
        masster_unique_fragment_mode=True,
        masster_selected_mask_limit=2,
        masster_unique_fragment_target=2,
        masster_ranked_mask_budget=2,
        masster_max_h_transfer=1,
        masster_fragment_connectivity_layers=1,
        masster_fragment_connectivity_hidden_dim=6,
        masster_fragment_connectivity_messages=False,
    )

    model = MassterPL(**config).model

    assert model.fragment_connectivity_encoder is not None
    assert model.fragment_connectivity_encoder.num_layers == 1
    assert model.fragment_connectivity_encoder.hidden_size == 6
    assert model.fragment_connectivity_dim == 6
    assert model.fragment_connectivity_messages is False


# 3. The matched pooled control (arm fc1p, `masster_fragment_connectivity_messages: False`; A2).


def test_fragment_connectivity_messages_false_builds_identical_parameters_and_blocks_messages(
    batch, monkeypatch: pytest.MonkeyPatch
) -> None:
    with_messages = _build_model(
        batch,
        fragment_connectivity_layers=1,
        fragment_connectivity_hidden_dim=4,
        fragment_connectivity_messages=True,
        **UNIQUE_KWARGS,
    )
    without_messages = _build_model(
        batch,
        fragment_connectivity_layers=1,
        fragment_connectivity_hidden_dim=4,
        fragment_connectivity_messages=False,
        **UNIQUE_KWARGS,
    )

    # Same module, same parameters: both are built under the same seed and forked RNG, so the
    # `messages` flag (read only in `forward`) changes nothing about construction.
    assert list(with_messages.state_dict()) == list(without_messages.state_dict())
    assert all(
        th.equal(with_messages.state_dict()[key], without_messages.state_dict()[key])
        for key in with_messages.state_dict()
    )

    captured_edge_index: list[th.Tensor] = []
    real_forward = without_messages.fragment_connectivity_encoder.forward

    def _spy(x, batch_index, edge_index, edge_attr, ctx=None):
        captured_edge_index.append(edge_index)
        return real_forward(x, batch_index, edge_index, edge_attr, ctx)

    monkeypatch.setattr(without_messages.fragment_connectivity_encoder, "forward", _spy)
    without_messages.eval()
    with th.no_grad():
        _forward(without_messages, batch)
    assert captured_edge_index[-1].shape == (2, 0)

    # Perturb both residuals identically to a nonzero value, so the branch's own computation --
    # not the zero-initialized residual -- reaches the output.
    with_messages.eval()
    with th.no_grad():
        nn.init.normal_(with_messages.intensity_head.connectivity_residual.weight, std=0.1)
        without_messages.intensity_head.connectivity_residual.weight.copy_(
            with_messages.intensity_head.connectivity_residual.weight
        )

    with_output = _forward(with_messages, batch)["pred_logprobs"]
    without_output = _forward(without_messages, batch)["pred_logprobs"]
    assert with_output.shape == without_output.shape
    assert not th.allclose(with_output, without_output)
