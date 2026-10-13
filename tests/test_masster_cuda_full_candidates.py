"""Full CUDA selection preserves CPU candidate identities and live model gradients."""

import pytest
import torch as th
from rdkit import Chem
from test_masster_cuda_candidates import _assert_candidates_match, _fixture

from fragnnet.models.masster.candidates.builder import (
    MassterStaticGraph,
    build_masster_candidates,
)
from fragnnet.models.masster.candidates.cuda_candidates import (
    _prepare_graph_batch,
    _template_forced_masks,
    build_masster_candidates_cuda,
)
from fragnnet.models.masster.candidates.smarts_templates import (
    MassterTemplate,
    build_masster_templates,
)

# Real templates: a disconnected product (sulfonamide, diaryl ketone, sulfonate SO2), and none
# for xanthone, saccharin or ethanol, whose ring-rule and duplicate templates MASSter drops.
# Synthetic partitions keep every builder path covered: a single-bond cut already in the
# forced prefix, a multi-cut mask with two connected sides read from ordinary components, and
# a disconnected side that becomes a new unique fragment.
_SMARTS_SMILES = (
    "Cc1ccc(S(=O)(=O)Nc2ccccn2)cc1",
    "O=C(c1ccccc1)c1ccc(O)cc1",
    "Oc1ccc(S(=O)(=O)O)cc1",
    "O=c1c2ccccc2oc2ccccc12",
    "O=C1NS(=O)(=O)c2ccccc21",
    "CCO",
)


def _partition_template(molecule: Chem.Mol, side: set[int]) -> MassterTemplate:
    """Template splitting ``molecule`` into ``side`` and the rest, cutting every crossing bond."""
    rest = tuple(atom for atom in range(molecule.GetNumAtoms()) if atom not in side)
    cut = tuple(
        bond.GetIdx()
        for bond in molecule.GetBonds()
        if (bond.GetBeginAtomIdx() in side) != (bond.GetEndAtomIdx() in side)
    )
    return MassterTemplate(cut, tuple(sorted((tuple(sorted(side)), rest))), 1, "synthetic")


def _synthetic_templates(molecule: Chem.Mol) -> tuple[MassterTemplate, ...]:
    atoms = list(molecule.GetAtoms())
    terminal = next(atom.GetIdx() for atom in atoms if atom.GetDegree() == 1)
    ring = [atom.GetIdx() for atom in atoms if atom.IsInRing() and atom.GetDegree() == 2]
    pair = next(
        {left.GetIdx(), right.GetIdx()}
        for left in atoms
        for right in atoms
        if left.GetIdx() < right.GetIdx()
        and molecule.GetBondBetweenAtoms(left.GetIdx(), right.GetIdx()) is None
    )
    sides = [{terminal}, *([{ring[0]}] if ring else []), pair]
    return tuple(_partition_template(molecule, side) for side in sides)


def _smarts_graph(view_molecules: tuple[int, ...], synthetic: bool = False):
    """Pack molecules into one batch with their SMARTS templates, one view per index."""
    molecules = [Chem.MolFromSmiles(smiles) for smiles in _SMARTS_SMILES]
    edges, atom_batch, atom_ptr = [], [], [0]
    for view, molecule_index in enumerate(view_molecules):
        molecule = molecules[molecule_index]
        offset = atom_ptr[-1]
        edges.extend(
            (bond.GetBeginAtomIdx() + offset, bond.GetEndAtomIdx() + offset)
            for bond in molecule.GetBonds()
        )
        atom_batch.extend([view] * molecule.GetNumAtoms())
        atom_ptr.append(offset + molecule.GetNumAtoms())
    templates = tuple(
        build_masster_templates(molecules[index])
        + (_synthetic_templates(molecules[index]) if synthetic else ())
        for index in view_molecules
    )
    return (
        th.tensor(edges, dtype=th.long).T.contiguous(),
        th.tensor(atom_batch),
        th.tensor(atom_ptr),
        templates,
    )


def test_template_forced_masks_follow_the_cpu_forced_order() -> None:
    def template(bonds: tuple[int, ...]) -> MassterTemplate:
        return MassterTemplate(bonds, ((0,), (1,)), 1, "rule")

    templates = [template(()), template((2,)), template((1, 3)), template((0, 4)), template((1, 3))]
    assert _template_forced_masks(templates, 5, True) == (((1, 3), (0, 4)), (0, 3, 6, 7, 6))
    assert _template_forced_masks(templates, 5, False) == (
        ((2,), (1, 3), (0, 4)),
        (0, 1, 2, 3, 2),
    )


def test_static_graph_preparation_preserves_interleaved_bonds_and_group_order():
    _, edges, atom_batch, atom_ptr, groups, bonds, *_ = _fixture(3)
    actual = _prepare_graph_batch(edges, atom_batch, atom_ptr, groups, 3)
    assert actual.representatives == [0, 1, 2]
    assert actual.representative_of == [0, 1, 2, 0]
    assert actual.representative_bond_indices == [0, 3, 1, 4, 6]
    assert actual.comparison_view_bonds == [2, 5]
    assert actual.comparison_leader_bonds == [0, 3]
    assert actual.bond_indices_by_molecule == bonds
    assert actual.bond_batch_values == [0, 2, 3, 0, 2, 3, 2]
    assert [graph.num_atoms for graph in actual.graphs] == [3, 1, 3]
    assert actual.graphs[1].endpoints == ((), ())


def test_static_group_preparation_requires_identical_ordered_topology():
    _, edges, atom_batch, atom_ptr, groups, *_ = _fixture(3)
    edges[0, 5] = 7
    with pytest.raises(ValueError, match="identical molecular topology"):
        _prepare_graph_batch(edges, atom_batch, atom_ptr, groups, 3)


def test_static_group_preparation_rejects_cross_molecule_bonds():
    _, edges, atom_batch, atom_ptr, groups, *_ = _fixture(3)
    edges[0, 5] = 0
    with pytest.raises(ValueError, match="connect different molecules"):
        _prepare_graph_batch(edges, atom_batch, atom_ptr, groups, 3)


def test_full_cuda_candidates_reject_cpu_logits_explicitly():
    logits, edges, atom_batch, atom_ptr, *_ = _fixture(3)
    with pytest.raises(ValueError, match="one-dimensional CUDA float32 logits"):
        build_masster_candidates_cuda(
            logits, edges, atom_batch, atom_ptr, max_masks=3, include_single_bond_cuts=True
        )


@pytest.mark.parametrize(
    ("atom_batch", "atom_ptr", "message"),
    [
        ([0, 0], [1, 2], "does not describe"),
        ([0, 1], [0, 2], "contiguous molecule spans"),
        ([0], [0, 0, 1], "contiguous molecule spans"),
    ],
)
def test_static_graph_preparation_rejects_inconsistent_csr_even_without_bonds(
    atom_batch, atom_ptr, message
):
    with pytest.raises(ValueError, match=message):
        _prepare_graph_batch(
            th.empty((2, 0), dtype=th.long),
            th.tensor(atom_batch),
            th.tensor(atom_ptr),
            None,
            1,
        )


@pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA required for full candidate parity")
@pytest.mark.parametrize("target", [1, 3, 100])
@pytest.mark.parametrize("use_sidecar", [False, True])
@pytest.mark.parametrize("include_singles", [False, True])
def test_full_cuda_matches_fields_counts_and_gradients_without_host_logits(
    monkeypatch, target, use_sidecar, include_singles
):
    logits, edges, atom_batch, atom_ptr, groups, *_ = _fixture(target)
    graph_args = (edges.cuda(), atom_batch.cuda(), atom_ptr.cuda())
    kwargs = {
        "max_masks": target,
        "max_unique_components": target,
        "max_enumerated_masks": 128,
        "include_single_bond_cuts": include_singles,
        "candidate_group_ids": th.tensor(groups, device="cuda"),
        "static_graph": (
            MassterStaticGraph(edges, atom_batch, atom_ptr) if use_sidecar else None
        ),
    }
    expected_logits = logits.cuda().requires_grad_()
    actual_logits = logits.cuda().requires_grad_()
    expected = build_masster_candidates(expected_logits, *graph_args, **kwargs)

    # Static int64 graph metadata/count reads are allowed, floating logits are not.
    original_cpu = th.Tensor.cpu

    def reject_float_host_copy(tensor, *args, **copy_kwargs):
        assert not (tensor.is_cuda and tensor.is_floating_point()), "copied CUDA logits to CPU"
        return original_cpu(tensor, *args, **copy_kwargs)

    monkeypatch.setattr(th.Tensor, "cpu", reject_float_host_copy)
    statistics = []
    actual = build_masster_candidates(
        actual_logits, *graph_args, **kwargs, candidate_backend="cuda", _cuda_full_stats=statistics
    )
    _assert_candidates_match(expected, actual)
    weights = th.arange(actual.mask_log_probabilities.numel(), device="cuda") + 1.0
    (expected.mask_log_probabilities * weights).sum().backward()
    (actual.mask_log_probabilities * weights).sum().backward()
    th.testing.assert_close(expected_logits.grad, actual_logits.grad, atol=1e-6, rtol=1e-6)
    result = statistics[0]
    assert result["representative_indices"] == [0, 1, 2]
    assert result["learned_mask_counts"] == [4, 1, 8]
    assert result["learned_single_cut_counts"] == [2, 0, 3]
    assert result["selection_native_load_ms"] >= 0.0
    assert all(
        physical >= logical
        for physical, logical in zip(
            result["physical_mask_counts"], result["logical_scanned_mask_counts"], strict=True
        )
    )


@pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA required for full candidate parity")
def test_full_cuda_ungrouped_views_and_signed_zero_match_cpu():
    logits, edges, atom_batch, atom_ptr, *_ = _fixture(3)
    logits[[0, 2]] = th.tensor([0.0, -0.0])
    logits = logits.cuda()
    graph_args = (edges.cuda(), atom_batch.cuda(), atom_ptr.cuda())
    kwargs = {
        "max_masks": 3,
        "max_unique_components": 3,
        "max_enumerated_masks": 128,
        "include_single_bond_cuts": True,
    }
    expected = build_masster_candidates(logits, *graph_args, **kwargs)
    ungrouped = build_masster_candidates_cuda(logits, *graph_args, **kwargs)
    grouped = build_masster_candidates_cuda(
        logits,
        *graph_args,
        **kwargs,
        candidate_group_ids=th.tensor([3, 4, 5, 3], device="cuda"),
    )
    _assert_candidates_match(expected, ungrouped)
    _assert_candidates_match(expected, grouped)


@pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA required for SMARTS parity")
@pytest.mark.parametrize("target", [2, 8, 256])
@pytest.mark.parametrize("include_singles", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_full_cuda_smarts_templates_match_cpu_fields_and_gradients(
    target: int, include_singles: bool, grouped: bool
) -> None:
    # The sulfonamide repeats as a grouped view; saccharin sits after it, so its looked-up
    # unique indices must also shift past the earlier views' template-only fragments.
    views = (0, 5, 1, 2, 3, 0, 4)
    edges, atom_batch, atom_ptr, templates = _smarts_graph(views, synthetic=True)
    generator = th.Generator().manual_seed(target + 2 * include_singles)
    view_logits = {
        index: th.randn(Chem.MolFromSmiles(_SMARTS_SMILES[index]).GetNumBonds(), generator=generator)
        for index in set(views)
    }
    logits = th.cat([view_logits[index] for index in views])
    kwargs = {
        "max_masks": target,
        "max_unique_components": target,
        "max_enumerated_masks": max(target, 64),
        "include_single_bond_cuts": include_singles,
        "static_graph": MassterStaticGraph(edges, atom_batch, atom_ptr, templates),
    }
    groups = th.tensor([0, 1, 2, 3, 4, 0, 5], device="cuda") if grouped else None
    graph_args = (edges.cuda(), atom_batch.cuda(), atom_ptr.cuda())
    expected_logits = logits.cuda().requires_grad_()
    expected = build_masster_candidates(
        expected_logits, *graph_args, **kwargs, candidate_group_ids=groups
    )
    actual_logits = logits.cuda().requires_grad_()
    actual = build_masster_candidates(
        actual_logits, *graph_args, **kwargs, candidate_group_ids=groups, candidate_backend="cuda"
    )
    assert expected.num_mechanisms > expected.mask_log_probabilities.numel()
    assert (expected.mechanism_rule_indices > 0).sum().item() == sum(map(len, templates))
    _assert_candidates_match(expected, actual)
    weights = th.rand(expected.mask_log_probabilities.numel(), generator=generator).cuda()
    (expected.mask_log_probabilities * weights).sum().backward()
    (actual.mask_log_probabilities * weights).sum().backward()
    th.testing.assert_close(expected_logits.grad, actual_logits.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA required for SMARTS grouping")
def test_full_cuda_smarts_rejects_grouped_views_with_different_template_cuts() -> None:
    edges, atom_batch, atom_ptr, templates = _smarts_graph((0, 0))
    with pytest.raises(ValueError, match="identical molecular topology"):
        build_masster_candidates_cuda(
            th.zeros(edges.shape[1], device="cuda"),
            edges.cuda(),
            atom_batch.cuda(),
            atom_ptr.cuda(),
            max_masks=4,
            max_unique_components=4,
            max_enumerated_masks=16,
            include_single_bond_cuts=True,
            smarts_templates=(templates[0], ()),
            candidate_group_ids=th.tensor([0, 0], device="cuda"),
        )


@pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA required for invalid GPU inputs")
def test_full_cuda_requires_equal_grouped_logits_and_one_dimensional_m1():
    logits, edges, atom_batch, atom_ptr, groups, *_ = _fixture(3)
    logits[2] += 0.01
    kwargs = {
        "max_masks": 3,
        "max_unique_components": 3,
        "max_enumerated_masks": 128,
        "include_single_bond_cuts": True,
        "candidate_group_ids": th.tensor(groups, device="cuda"),
    }
    graph_args = (edges.cuda(), atom_batch.cuda(), atom_ptr.cuda())
    with pytest.raises(ValueError, match="identical proposal logits"):
        build_masster_candidates_cuda(logits.cuda(), *graph_args, **kwargs)
    with pytest.raises(ValueError, match="one-dimensional CUDA float32 logits"):
        build_masster_candidates_cuda(logits[:, None].cuda(), *graph_args, **kwargs)


@pytest.mark.parametrize(
    "overrides",
    [
        {"candidate_backend": "invalid"},
        {"unique_fragment_mode": False},
        {"proposal_num_mixtures": 2},
        {"min_mask_relative_probability": 0.1},
    ],
)
def test_public_cuda_model_rejects_unsupported_configuration(overrides):
    from test_masster_model import _build_batch, _build_model

    kwargs = {"candidate_backend": "cuda", "unique_fragment_mode": True, **overrides}
    with pytest.raises(ValueError, match="candidate_backend"):
        _build_model(_build_batch(), **kwargs)


def test_public_cuda_model_rejects_cpu_execution():
    from test_masster_model import _build_batch, _build_model

    batch = _build_batch()
    model = _build_model(batch, candidate_backend="cuda", unique_fragment_mode=True).eval()
    with pytest.raises(ValueError, match="CUDA float32 logits"):
        model(
            mol_pyg=batch,
            spec_prec_type=th.tensor([0, 0]),
            spec_nce=th.tensor([20.0, 25.0]),
        )


@pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA required for full model parity")
def test_public_cuda_model_preserves_predictions_loss_and_parameter_gradients():
    from test_masster_model import _build_batch, _build_model

    batch = _build_batch(("CCO", "C1CC1", "CCO")).cuda()
    model = (
        _build_model(
            batch,
            max_masks=3,
            unique_fragment_mode=True,
            max_enumerated_masks=16,
            share_proposal_across_ev=True,
        )
        .cuda()
        .eval()
    )
    inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 0, 0], device="cuda"),
        "spec_nce": th.tensor([20.0, 25.0, 40.0], device="cuda"),
        "proposal_group_ids": th.tensor([4, 5, 4], device="cuda"),
    }
    expected = model(**inputs)
    expected_loss = expected["pred_logprobs"].exp().square().sum()
    expected_loss.backward()
    expected_gradients = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    cuda_model = (
        _build_model(
            batch,
            max_masks=3,
            unique_fragment_mode=True,
            max_enumerated_masks=16,
            share_proposal_across_ev=True,
            candidate_backend="cuda",
        )
        .cuda()
        .eval()
    )
    assert cuda_model.candidate_backend == "cuda"
    assert model.state_dict().keys() == cuda_model.state_dict().keys()
    cuda_model.load_state_dict(model.state_dict(), strict=True)
    actual = cuda_model(**inputs)
    actual_loss = actual["pred_logprobs"].exp().square().sum()
    actual_loss.backward()
    for name in ("pred_mzs", "pred_batch_idxs", "pred_logprobs"):
        th.testing.assert_close(expected[name], actual[name], atol=1e-6, rtol=1e-5)
    th.testing.assert_close(expected_loss, actual_loss, atol=1e-6, rtol=1e-5)
    actual_gradients = {
        name: parameter.grad
        for name, parameter in cuda_model.named_parameters()
        if parameter.grad is not None
    }
    assert actual_gradients.keys() == expected_gradients.keys()
    assert any("bond" in name for name in actual_gradients)
    assert any("intensity" in name for name in actual_gradients)
    for name in expected_gradients:
        th.testing.assert_close(
            expected_gradients[name], actual_gradients[name], atol=2e-5, rtol=2e-4, msg=name
        )


@pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA required for full model parity")
def test_public_cuda_model_matches_cpu_backend_with_smarts_templates():
    from test_masster_model import _build_batch, _build_model

    smiles = (_SMARTS_SMILES[0], _SMARTS_SMILES[5], _SMARTS_SMILES[3], _SMARTS_SMILES[4])
    batch = _build_batch(smiles)
    static_graph = MassterStaticGraph(
        chemical_bond_index=batch.chemical_bond_index,
        atom_batch_indices=batch.batch,
        atom_ptr=batch.ptr,
        smarts_templates=tuple(
            build_masster_templates(molecule) + _synthetic_templates(molecule)
            for molecule in map(Chem.MolFromSmiles, smiles)
        ),
    )
    inputs = {
        "mol_pyg": batch.cuda(),
        "spec_prec_type": th.tensor([0, 0, 1, 0], device="cuda"),
        "spec_nce": th.tensor([20.0, 25.0, 40.0, 30.0], device="cuda"),
        "static_graph": static_graph,
    }
    models = {
        backend: _build_model(
            batch,
            max_masks=6,
            unique_fragment_mode=True,
            max_enumerated_masks=32,
            use_smarts_templates=True,
            candidate_backend=backend,
        )
        .cuda()
        .eval()
        for backend in ("cpu", "cuda")
    }
    models["cuda"].load_state_dict(models["cpu"].state_dict(), strict=True)
    outputs, gradients = [], []
    for model in models.values():
        output = model(**inputs)
        output["pred_logprobs"].exp().square().sum().backward()
        outputs.append(output)
        gradients.append(
            {
                name: parameter.grad.clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
        )
    expected, actual = outputs
    for name in ("pred_mzs", "pred_batch_idxs", "pred_logprobs"):
        th.testing.assert_close(expected[name], actual[name], atol=1e-6, rtol=1e-5)
    assert gradients[0].keys() == gradients[1].keys()
    assert any(name.startswith("mechanism_head") for name in gradients[1])
    for name in gradients[0]:
        th.testing.assert_close(
            gradients[0][name], gradients[1][name], atol=2e-5, rtol=2e-4, msg=name
        )
