"""Candidate-graph attention (arm ``cg1``).

``masster_intensity_candidate_graph`` masks full candidate attention to the candidate graph:
each candidate attends to itself, its direct parents and children by atom containment and its
complement, and the intact precursor receives but never sends. This file covers the edge builder
(``candidate_graph_edges``) against a hand-enumerated chain molecule, the head's per-bucket mask
(reach, padding, validation, equivalence to full attention on the complete graph) and the model
wiring (default off, unique-fragment requirement, a finite forward and backward).
"""

import pytest
import torch as th

from fragnnet.models.masster.masster_candidate_transformer import candidate_graph_edges
from tests.test_masster_attention_and_budget import _head_inputs, _moved_outputs, _mz_head
from tests.test_masster_model import _build_batch, _build_model

# 1. The edge builder on a 4-atom chain 0-1-2-3 (bonds 01, 12, 23).
#
# Every candidate of its three single cuts and three double cuts, plus the precursor:
CHAIN = [
    (0, 1, 2, 3),  # 0 precursor
    (0,),  # 1
    (1, 2, 3),  # 2
    (0, 1),  # 3
    (2, 3),  # 4
    (0, 1, 2),  # 5
    (3,),  # 6
    (1, 2),  # 7, inside both 2 and 5: two direct parents
    (1,),  # 8, direct parents 3 and 7 (not 5: 3 lies between)
    (2,),  # 9, direct parents 7 and 4
]
# Direct containment (child, parent), excluding the precursor as a parent.
CHAIN_PARENTS = [(1, 3), (3, 5), (4, 2), (6, 4), (7, 2), (7, 5), (8, 3), (8, 7), (9, 7), (9, 4)]
CHAIN_COMPLEMENTS = [(1, 2), (3, 4), (5, 6)]
# The precursor's direct children: it reads them, they never read it.
CHAIN_PRECURSOR_CHILDREN = [2, 5]


def _memberships(candidates: list[tuple[int, ...]], atom_offset: int = 0) -> tuple:
    atoms = [a + atom_offset for c in candidates for a in c]
    components = [i for i, c in enumerate(candidates) for _ in c]
    return th.tensor(atoms), th.tensor(components)


def _expected_chain_edges(rows: list[int]) -> set[tuple[int, int]]:
    expected = set()
    for child, parent in CHAIN_PARENTS + CHAIN_COMPLEMENTS:
        expected |= {(rows[child], rows[parent]), (rows[parent], rows[child])}
    expected |= {(rows[0], rows[child]) for child in CHAIN_PRECURSOR_CHILDREN}
    return expected


def _edge_set(edges: th.Tensor) -> set[tuple[int, int]]:
    return {(int(q), int(k)) for q, k in edges.T}


def test_chain_edges_are_direct_containment_complements_and_a_receiving_precursor() -> None:
    atoms, components = _memberships(CHAIN)
    edges = candidate_graph_edges(
        atoms,
        components,
        th.zeros(len(CHAIN), dtype=th.long),
        th.tensor([0]),
        th.tensor([4]),
        num_components=len(CHAIN),
    )
    assert edges.dtype == th.long and edges.shape[0] == 2
    assert _edge_set(edges) == _expected_chain_edges(list(range(len(CHAIN))))
    assert all(k != 0 for _q, k in _edge_set(edges))  # the precursor never sends
    assert all(q != k for q, k in _edge_set(edges))  # self-loops are the head's job


def test_edges_stay_within_spectra_with_interleaved_rows_and_small_chunks() -> None:
    # Spectrum 0 is the chain (atoms 0-3); spectrum 1 a two-atom molecule (atoms 4-5) with its
    # precursor and both single-cut sides. Rows interleave the two spectra.
    pair = [(0, 1), (0,), (1,)]
    rows_chain = [0, 1, 2, 4, 5, 7, 8, 9, 11, 12]
    rows_pair = [3, 6, 10]
    atoms_c, comps_c = _memberships(CHAIN)
    atoms_p, comps_p = _memberships(pair, atom_offset=4)
    atoms = th.cat([atoms_c, atoms_p])
    components = th.cat([th.tensor(rows_chain)[comps_c], th.tensor(rows_pair)[comps_p]])
    batch = th.zeros(13, dtype=th.long)
    batch[rows_pair] = 1
    for chunk_size in (1, 64):
        edges = candidate_graph_edges(
            atoms,
            components,
            batch,
            th.tensor([0, 4]),
            th.tensor([4, 2]),
            num_components=13,
            chunk_size=chunk_size,
        )
        expected = _expected_chain_edges(rows_chain) | {
            (3, 6),  # precursor reads each side
            (3, 10),
            (6, 10),  # the two sides are complements
            (10, 6),
        }
        assert _edge_set(edges) == expected


def test_single_candidate_spectrum_has_no_edges() -> None:
    edges = candidate_graph_edges(
        th.tensor([0]),
        th.tensor([0]),
        th.tensor([0]),
        th.tensor([0]),
        th.tensor([1]),
        num_components=1,
    )
    assert edges.shape == (2, 0)


# 2. The head's mask.


def test_candidate_graph_is_off_by_default_and_validated() -> None:
    assert not _mz_head(attention_type="full").candidate_graph
    for attention_type in ("induced", "none"):
        with pytest.raises(ValueError, match="candidate_graph requires attention_type='full'"):
            _mz_head(attention_type=attention_type, candidate_graph=True)
    inputs = _head_inputs()
    edges = th.tensor([[0], [1]])
    with pytest.raises(ValueError, match="supplied exactly when candidate_graph is on"):
        _mz_head(attention_type="full")(
            *inputs, batch_size=1, num_masks=4, candidate_graph_edges=edges
        )
    head = _mz_head(attention_type="full", candidate_graph=True)
    with pytest.raises(ValueError, match="supplied exactly when candidate_graph is on"):
        head(*inputs, batch_size=1, num_masks=4)
    with pytest.raises(ValueError, match=r"\(2, E\) torch.long"):
        head(*inputs, batch_size=1, num_masks=4, candidate_graph_edges=edges.float())


def test_candidate_graph_limits_one_layer_reach_to_the_edges() -> None:
    """With one layer, a candidate reads only itself and the keys of its edges."""
    head = _mz_head(attention_type="full", candidate_graph=True)
    # 0 reads 1; 1 reads 0 and 2; nothing reads 3.
    edges = {"candidate_graph_edges": th.tensor([[0, 1, 1], [1, 0, 2]])}

    assert _moved_outputs(head, 0, **edges).tolist() == [True, True, False, False]
    assert _moved_outputs(head, 2, **edges).tolist() == [False, True, True, False]
    assert _moved_outputs(head, 3, **edges).tolist() == [False, False, False, True]


def test_complete_graph_equals_full_attention_with_the_same_state_dict() -> None:
    full = _mz_head(attention_type="full")
    graph = _mz_head(attention_type="full", candidate_graph=True)
    inputs = _head_inputs()
    everyone = th.tensor([(q, k) for q in range(4) for k in range(4) if q != k]).T

    assert list(graph.state_dict()) == list(full.state_dict())
    assert all(th.equal(graph.state_dict()[k], v) for k, v in full.state_dict().items())
    assert th.allclose(
        graph(*inputs, batch_size=1, num_masks=4, candidate_graph_edges=everyone)[2],
        full(*inputs, batch_size=1, num_masks=4)[2],
        atol=1e-6,
    )


def test_candidate_graph_mask_values_and_padded_rows() -> None:
    padding = th.tensor([[False, False, False, True]])
    # Rows 5, 6, 7 of a 9-row batch sit in slots 0, 1, 2 of this bucket's only molecule.
    bias = _mz_head(attention_type="full", candidate_graph=True)._candidate_graph_bias(
        th.tensor([[5, 7, 0], [6, 5, 1]]),  # 5 -> 6, 7 -> 5, and an edge of another bucket
        9,
        th.tensor([5, 6, 7]),
        th.tensor([0, 0, 0]),
        th.tensor([0, 1, 2]),
        padding,
        th.float32,
    )
    inf = float("-inf")
    expected = th.tensor(
        [
            [0.0, 0.0, inf, inf],
            [inf, 0.0, inf, inf],
            [0.0, inf, 0.0, inf],
            [0.0, 0.0, 0.0, 0.0],  # the padded row keeps every key
        ]
    )
    assert bias.shape == (1, 1, 4, 4)
    assert th.equal(bias[0, 0], expected)


def test_candidate_graph_keeps_padded_buckets_finite_through_backward() -> None:
    head = _mz_head(attention_type="full", candidate_graph=True, num_layers=2).train()
    inputs = list(_head_inputs())
    inputs[5] = th.tensor([0, 0, 0, 1])  # molecule 0 has three candidates, molecule 1 one
    edges = th.tensor([[0, 1], [1, 0]])

    outputs = head(*inputs, batch_size=2, num_masks=4, candidate_graph_edges=edges)
    total = sum(output.sum() for output in outputs)
    total.backward()
    assert th.isfinite(total)
    assert all(p.grad is None or th.isfinite(p.grad).all() for p in head.parameters())


# 3. Model wiring.


def _unique_full_model(batch, **kwargs):
    return _build_model(
        batch,
        unique_fragment_mode=True,
        proposal_selection_only=True,
        use_base_probability_feature=False,
        intensity_transformer_attention="full",
        max_masks=4,
        max_enumerated_masks=8,
        **kwargs,
    )


def _forward(model, batch) -> dict[str, th.Tensor]:
    return model(mol_pyg=batch, spec_prec_type=th.tensor([0, 0]), spec_nce=th.tensor([10.0, 40.0]))


def test_model_candidate_graph_requires_unique_fragment_mode() -> None:
    batch = _build_batch(("CCOC", "CCO"))
    with pytest.raises(ValueError, match="intensity_candidate_graph requires unique_fragment_mode"):
        # Induced attention reaches this check (full attention alone already needs unique mode).
        _build_model(batch, intensity_candidate_graph=True)


def test_model_candidate_graph_adds_no_parameters_and_trains() -> None:
    batch = _build_batch(("CCOC", "CCO"))
    full = _unique_full_model(batch)
    graph = _unique_full_model(batch, intensity_candidate_graph=True)
    assert graph.intensity_head.candidate_graph and not full.intensity_head.candidate_graph
    assert list(graph.state_dict()) == list(full.state_dict())
    graph.train()
    out = _forward(graph, batch)
    loss = out["pred_logprobs"].exp().sum()
    loss.backward()
    assert th.isfinite(out["pred_logprobs"]).all()
    assert all(p.grad is None or th.isfinite(p.grad).all() for p in graph.parameters())


def test_model_candidate_graph_changes_the_allocation_only_through_attention() -> None:
    """Same weights and candidates: the graph mask changes only the allocation."""
    batch = _build_batch(("CCOC", "CCO"))
    full = _unique_full_model(batch).eval()
    graph = _unique_full_model(batch, intensity_candidate_graph=True).eval()
    graph.load_state_dict(full.state_dict())
    captured = []
    original = graph.intensity_head.forward

    def spy(*args, **kwargs):
        captured.append(kwargs["candidate_graph_edges"])
        return original(*args, **kwargs)

    graph.intensity_head.forward = spy
    with th.no_grad():
        a, b = _forward(full, batch), _forward(graph, batch)
    edges = captured[0]
    assert edges.shape[0] == 2 and edges.shape[1] > 0
    assert th.equal(a["pred_mzs"], b["pred_mzs"])  # same candidates and channels
    assert not th.allclose(a["pred_logprobs"], b["pred_logprobs"])


def test_candidate_graph_edges_validates_its_inputs() -> None:
    atoms, components = _memberships([(0, 1), (0,), (1,)])
    args = (atoms, components, th.zeros(3, dtype=th.long), th.tensor([0]), th.tensor([2]))
    assert _edge_set(candidate_graph_edges(*args, num_components=3)) == {
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 1),
    }
    with pytest.raises(ValueError, match="1-D torch.long"):
        candidate_graph_edges(atoms.float(), *args[1:], num_components=3)
    with pytest.raises(ValueError, match="one spectrum per component"):
        candidate_graph_edges(*args, num_components=4)
    with pytest.raises(ValueError, match="missing component"):
        candidate_graph_edges(atoms, components + 5, *args[2:], num_components=3)
