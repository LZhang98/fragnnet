"""Hybrid candidate assembly must preserve grouped views and live probability gradients."""

from dataclasses import fields, replace
from time import perf_counter

import numpy as np
import pytest
import torch as th

from fragnnet.models.masster.candidates import cuda_topology
from fragnnet.models.masster.candidates.builder import (
    MassterStaticGraph,
    _build_local_topology,
    _merge_forced_masks,
    _select_masks_batch,
    _TopologyInput,
    build_masster_candidates,
)
from fragnnet.models.masster.candidates.cuda_topology import (
    CudaTopologyBatch,
    CudaTopologyCounts,
)
from fragnnet.models.masster.candidates.cuda_topology_hybrid import (
    _build_cuda_candidates,
    _validate_grouped_topology,
    build_masster_candidates_cuda_topology,
)


def _fixture(target):
    # Interleaved global bonds, a zero-bond graph, a ring, and a repeated chain view.
    edges = th.tensor([[0, 4, 7, 1, 5, 8, 6], [1, 5, 8, 2, 6, 9, 4]])
    atom_ptr = th.tensor([0, 3, 4, 7, 10])
    atom_batch = th.tensor([0, 0, 0, 1, 2, 2, 2, 3, 3, 3])
    bonds = [[0, 3], [], [1, 4, 6], [2, 5]]
    logits = th.tensor([-0.1, 0.3, -0.1, 0.2, -0.4, 0.2, 0.5])
    groups = [3, 4, 5, 3]
    selected = _select_masks_batch(
        [logits[indices].double().numpy().reshape(-1, 1) for indices in bonds],
        None,
        groups,
        128,
        0.0,
        1,
    )
    requests, fractions, selected_counts = [], [], []
    for index, (ptr, broken) in enumerate(selected):
        cuts = np.diff(ptr)
        fractions.append(float(np.count_nonzero(cuts == 1)) / cuts.size)
        forced = [(), *((bond,) for bond in range(len(bonds[index])))]
        ptr, broken, forced_count = _merge_forced_masks(
            ptr, broken, forced, cuts, cuts.size, len(forced), True
        )
        selected_counts.append(ptr.size - 1)
        requests.append(
            _TopologyInput(
                num_atoms=int(atom_ptr[index + 1] - atom_ptr[index]),
                endpoints=tuple(
                    tuple(row) for row in (edges[:, bonds[index]] - atom_ptr[index]).tolist()
                ),
                mask_ptr=ptr,
                broken_bonds=broken,
                forced_mask_count=forced_count,
                max_unique_components=target,
            )
        )
    return logits, edges, atom_batch, atom_ptr, groups, bonds, requests, fractions, selected_counts


def _cpu_topology(requests, device):
    caches = [_build_local_topology(request) for request in requests]
    names = (
        "cut_counts",
        "component_mask_indices",
        "component_cut_counts",
        "atom_indices",
        "atom_component_indices",
        "mask_indices",
        "broken_bond_indices",
        "boundary_bond_indices",
        "boundary_component_indices",
        "component_unique_indices",
        "unique_representative_component_indices",
    )
    counts = tuple(
        CudaTopologyCounts(
            num_masks=cache.cut_counts.size,
            num_components=cache.num_components,
            num_unique_components=cache.unique_component_count,
            scanned_mask_count=cache.scanned_mask_count,
            num_atom_memberships=cache.atom_indices.size,
            num_broken_bonds=cache.broken_bond_indices.size,
            num_boundary_memberships=cache.boundary_bond_indices.size,
        )
        for cache in caches
    )
    return CudaTopologyBatch(
        arrays=tuple(
            th.from_numpy(np.concatenate([getattr(cache, name) for cache in caches])).to(device)
            for name in names
        ),
        counts=counts,
        physical_mask_counts=tuple(count.scanned_mask_count for count in counts),
        scratch_bytes=0,
        native_load_ms=0.0,
    )


def _assert_candidates_match(expected, actual):
    for field in fields(expected):
        if field.name == "timings":
            continue
        left, right = getattr(expected, field.name), getattr(actual, field.name)
        if isinstance(left, th.Tensor):
            if left.is_floating_point():
                th.testing.assert_close(left, right, atol=1e-6, rtol=1e-6, msg=field.name)
            else:
                assert th.equal(left, right), field.name
        else:
            assert left == right, field.name


@pytest.mark.parametrize("target", [1, 3, 100])
def test_vectorized_group_expansion_matches_cpu_candidate_fields_and_gradients(monkeypatch, target):
    logits, edges, atom_batch, atom_ptr, groups, bonds, requests, fractions, selected_counts = (
        _fixture(target)
    )
    expected_logits = logits.clone().requires_grad_()
    expected = build_masster_candidates(
        expected_logits,
        edges,
        atom_batch,
        atom_ptr,
        max_masks=target,
        max_unique_components=target,
        max_enumerated_masks=128,
        include_single_bond_cuts=True,
        candidate_group_ids=th.tensor(groups),
    )
    monkeypatch.setattr(cuda_topology, "build_cuda_topology", _cpu_topology)
    actual_logits = logits.clone().requires_grad_()
    statistics = []
    actual = _build_cuda_candidates(
        bond_logits=actual_logits,
        topology_inputs=requests,
        topology_representatives=[0, 1, 2],
        representative_of=[0, 1, 2, 0],
        atom_ptr_values=atom_ptr.tolist(),
        bond_indices_by_molecule=bonds,
        bond_batch_values=atom_batch[edges[0]].tolist(),
        proposal_single_cut_fractions=fractions,
        selected_mask_counts=selected_counts,
        total_start=perf_counter(),
        input_transfer_ms=0.0,
        mask_selection_ms=0.0,
        forced_merge_ms=0.0,
        statistics=statistics,
    )
    _assert_candidates_match(expected, actual)
    weights = th.arange(actual.mask_log_probabilities.numel()) + 1.0
    (expected.mask_log_probabilities * weights).sum().backward()
    (actual.mask_log_probabilities * weights).sum().backward()
    th.testing.assert_close(expected_logits.grad, actual_logits.grad, atol=0, rtol=0)
    assert statistics[0]["representative_indices"] == [0, 1, 2]
    assert statistics[0]["native_load_ms"] == 0.0


def test_group_validation_checks_topology_and_retained_csr_prefix():
    *_, requests, _, _ = _fixture(3)
    counts = {
        index: _build_local_topology(request).cut_counts.size
        for index, request in enumerate(requests[:3])
    }
    changed = list(requests)
    changed[3] = replace(changed[3], endpoints=((0, 0), (1, 2)))
    with pytest.raises(ValueError, match="identical molecular topology"):
        _validate_grouped_topology(changed, [0, 1, 2, 0], counts)
    changed[3] = replace(requests[3], broken_bonds=requests[3].broken_bonds[::-1].copy())
    with pytest.raises(ValueError, match="identical molecular topology"):
        _validate_grouped_topology(changed, [0, 1, 2, 0], counts)


def test_private_cuda_prototype_rejects_cpu_inputs():
    logits, edges, atom_batch, atom_ptr, *_ = _fixture(3)
    with pytest.raises(ValueError, match="CUDA float32"):
        build_masster_candidates_cuda_topology(
            logits,
            edges,
            atom_batch,
            atom_ptr,
            max_masks=3,
            max_unique_components=3,
            max_enumerated_masks=128,
            include_single_bond_cuts=True,
        )


@pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA required for hybrid candidate parity")
@pytest.mark.parametrize("target", [1, 3, 100])
def test_cuda_candidates_match_cpu_topology_fields_and_probability_gradients(target):
    logits, edges, atom_batch, atom_ptr, groups, *_ = _fixture(target)
    static_graph = MassterStaticGraph(edges, atom_batch, atom_ptr)
    kwargs = {
        "max_masks": target,
        "max_unique_components": target,
        "max_enumerated_masks": 128,
        "include_single_bond_cuts": True,
        "static_graph": static_graph,
        "candidate_group_ids": th.tensor(groups, device="cuda"),
    }
    cpu_topology_logits = logits.cuda().requires_grad_()
    cuda_topology_logits = logits.cuda().requires_grad_()
    graph_args = (edges.cuda(), atom_batch.cuda(), atom_ptr.cuda())
    expected = build_masster_candidates(cpu_topology_logits, *graph_args, **kwargs)
    statistics = []
    actual = build_masster_candidates_cuda_topology(
        cuda_topology_logits, *graph_args, **kwargs, statistics=statistics
    )
    _assert_candidates_match(expected, actual)
    weights = th.arange(actual.mask_log_probabilities.numel(), device="cuda") + 1.0
    (expected.mask_log_probabilities * weights).sum().backward()
    (actual.mask_log_probabilities * weights).sum().backward()
    th.testing.assert_close(
        cpu_topology_logits.grad, cuda_topology_logits.grad, atol=1e-6, rtol=1e-6
    )
    assert all(
        physical >= logical
        for physical, logical in zip(
            statistics[0]["physical_mask_counts"],
            statistics[0]["logical_scanned_mask_counts"],
            strict=True,
        )
    )


@pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA required for hybrid model parity")
def test_cuda_hybrid_model_preserves_predictions_and_parameter_gradients(monkeypatch):
    from test_masster_model import _build_batch, _build_model

    import fragnnet.models.masster.masster_model as model_module

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
    model.zero_grad(set_to_none=True)
    def hybrid(*args, candidate_backend, **kwargs):
        assert candidate_backend == "cpu"
        return build_masster_candidates_cuda_topology(*args, **kwargs)

    monkeypatch.setattr(model_module, "build_masster_candidates", hybrid)
    actual = model(**inputs)
    actual_loss = actual["pred_logprobs"].exp().square().sum()
    actual_loss.backward()
    for name in ("pred_mzs", "pred_batch_idxs", "pred_logprobs"):
        th.testing.assert_close(expected[name], actual[name], atol=1e-6, rtol=1e-5)
    th.testing.assert_close(expected_loss, actual_loss, atol=1e-6, rtol=1e-5)
    actual_gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    assert actual_gradients.keys() == expected_gradients.keys()
    assert any("bond" in name for name in actual_gradients)
    assert any("intensity" in name for name in actual_gradients)
    for name in expected_gradients:
        th.testing.assert_close(
            expected_gradients[name], actual_gradients[name], atol=2e-5, rtol=2e-4, msg=name
        )
