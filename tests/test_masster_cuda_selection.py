"""Independent CPU-oracle tests for the optional exact CUDA mask selector."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch as th

from fragnnet.models.masster.path_selection.candidates import _merge_forced_masks
from fragnnet.models.masster.path_selection.core import (
    build_component_arrays_raw,
    top_mixture_bond_sets_raw,
)
from fragnnet.models.masster.path_selection.cuda_selection import (
    _extra_forced_words,
    _prepare_layout,
    rank_batch_cuda,
)

_CUDA = pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA device unavailable")


def _compare(
    logits: list[np.ndarray],
    limit: int,
    force_single: bool,
    extra_forced: list[list[tuple[int, ...]]] | None = None,
) -> None:
    values = [np.asarray(row, dtype=np.float32) for row in logits]
    device_logits = th.from_numpy(np.concatenate(values)).cuda().requires_grad_()
    original_logits = device_logits.clone()
    result = rank_batch_cuda(
        device_logits,
        [len(row) for row in values],
        ranked_mask_budget=limit,
        include_single_bond_cuts=force_single,
        extra_forced_bond_sets=extra_forced,
    )
    assert result.mask_ptr.dtype == th.int64 and result.mask_ptr.device.type == "cuda"
    assert result.broken_bonds.dtype == th.int32 and result.broken_bonds.device.type == "cuda"
    assert result.device_counts.shape == (len(values), 6)
    assert not result.mask_ptr.requires_grad and not result.broken_bonds.requires_grad
    assert th.equal(device_logits, original_logits)
    ptr_offset = bond_offset = 0
    for index, (row, counts) in enumerate(zip(values, result.counts, strict=True)):
        learned_ptr, learned_bonds = top_mixture_bond_sets_raw(
            row.astype(np.float64)[:, None], np.zeros(1), limit, 0.0, 0.0
        )
        forced = [()]
        if force_single:
            forced.extend((bond,) for bond in range(len(row)))
        base_forced_count = len(forced)
        if extra_forced is not None:
            forced.extend(extra_forced[index])
        expected_ptr, expected_bonds, forced_count = _merge_forced_masks(
            learned_ptr,
            learned_bonds,
            forced,
            np.diff(learned_ptr),
            len(learned_ptr) - 1,
            base_forced_count,
            True,
        )
        actual_ptr = result.mask_ptr[ptr_offset : ptr_offset + counts.num_masks + 1]
        actual_bonds = result.broken_bonds[bond_offset : bond_offset + counts.num_broken_bonds]
        np.testing.assert_array_equal(actual_ptr.cpu().numpy() - bond_offset, expected_ptr)
        np.testing.assert_array_equal(actual_bonds.cpu().numpy(), expected_bonds)
        assert counts.num_masks == len(expected_ptr) - 1
        assert counts.forced_mask_count == forced_count
        assert counts.learned_mask_count == len(learned_ptr) - 1
        assert counts.learned_single_cut_count == np.count_nonzero(np.diff(learned_ptr) == 1)
        assert counts.num_broken_bonds == len(expected_bonds)
        assert result.device_counts[index].cpu().tolist() == [
            0,
            counts.learned_mask_count,
            counts.learned_single_cut_count,
            counts.num_masks,
            counts.forced_mask_count,
            counts.num_broken_bonds,
        ]
        ptr_offset += counts.num_masks + 1
        bond_offset += counts.num_broken_bonds
    assert ptr_offset == result.mask_ptr.numel()
    assert bond_offset == result.broken_bonds.numel()


@pytest.mark.parametrize(
    ("counts", "budget", "scratch", "message"),
    [
        ([], 16, 1024**2, "at least one"),
        ([-1], 16, 1024**2, "integer bond counts"),
        ([2049], 16, 1024**2, "integer bond counts"),
        ([1.5], 16, 1024**2, "integer bond counts"),
        ([1], 0, 1024**2, "ranked_mask_budget"),
        ([1], 65537, 1024**2, "ranked_mask_budget"),
        ([1], 1.5, 1024**2, "ranked_mask_budget"),
        ([1], True, 1024**2, "ranked_mask_budget"),
        ([30], 4096, 1, "scratch needs"),
        ([1], 2, 0, "positive"),
    ],
)
def test_cuda_selection_layout_rejects_unsupported_bounds(
    counts: list[int], budget: int, scratch: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _prepare_layout(counts, budget, True, scratch)


def test_cuda_selection_reachable_state_and_memory_bounds() -> None:
    meta, _, scratch = _prepare_layout([0, 1, 30, 129], 4096, True, 512 * 1024**2)
    assert meta[:, 2].tolist() == [1, 2, 4096, 4096]
    assert meta[:, 4].tolist() == [1, 1, 1, 3]
    assert meta[:, 10].tolist() == [2, 4, 4127, 4226]
    assert 0 < scratch < 512 * 1024**2
    numpy_meta, _, numpy_scratch = _prepare_layout(
        np.asarray([0, 1, 30, 129], dtype=np.int64), np.int64(4096), True, 512 * 1024**2
    )
    np.testing.assert_array_equal(numpy_meta, meta)
    assert numpy_scratch == scratch
    with pytest.raises(ValueError, match="requires CUDA logits"):
        rank_batch_cuda(th.zeros(1), [1], ranked_mask_budget=2, include_single_bond_cuts=True)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ([[()]], "extra forced masks"),
        ([[(1,)]], "extra forced masks"),
        ([[(2, 1)]], "extra forced masks"),
        ([[(0, 5)]], "extra forced masks"),
        ([[(0, 1), (0, 1)]], "extra forced masks"),
        ([[], []], "one sequence per representative"),
    ],
)
def test_cuda_selection_rejects_invalid_extra_forced_masks(
    extra: list[list[tuple[int, ...]]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _extra_forced_words([5], extra, True)


def test_cuda_selection_packs_extra_forced_masks_into_bit_words() -> None:
    counts, words = _extra_forced_words([3, 70, 0], [[(0, 2)], [(1, 64), (0,)], []], False)
    assert counts == [1, 2, 0]
    assert words.view(np.uint64).tolist() == [0b101, 0b10, 1, 1, 0]
    meta, _, _ = _prepare_layout([3, 70, 0], 16, False, 512 * 1024**2, counts)
    assert meta[:, 13].tolist() == [0, 1, 5]
    assert meta[:, 14].tolist() == counts
    assert meta[:, 10].tolist() == [10, 19, 2]


@_CUDA
@pytest.mark.parametrize("force_single", [False, True])
@pytest.mark.parametrize("limit", [1, 17, 257])
def test_cuda_selection_extra_forced_masks_match_cpu_forced_merge(
    force_single: bool, limit: int
) -> None:
    rng = np.random.default_rng(907)
    rows, extra = [], []
    for size in [0, 1, 2, 3, 7, 30, 64, 65, 129]:
        row = rng.normal(size=size).astype(np.float32)
        rows.append(row)
        # Include learned-pool duplicates: the top masks by probability are the likeliest.
        learned_ptr, learned_bonds = top_mixture_bond_sets_raw(
            row.astype(np.float64)[:, None], np.zeros(1), 4, 0.0, 0.0
        )
        bond_sets = {
            tuple(learned_bonds[start:end].tolist())
            for start, end in itertools.pairwise(learned_ptr)
            if end - start >= 2
        }
        for _ in range(3):
            if size >= 2:
                cuts = int(rng.integers(2, min(size, 5) + 1))
                bond_sets.add(tuple(sorted(rng.choice(size, size=cuts, replace=False).tolist())))
        if not force_single and size >= 1:
            bond_sets.add((int(rng.integers(size)),))
        extra.append(sorted(bond_sets))
    _compare(rows, limit, force_single, extra)


@_CUDA
@pytest.mark.parametrize("force_single", [False, True])
@pytest.mark.parametrize("limit", [1, 17, 257])
def test_cuda_selection_exact_random_signed_zero_and_multiword_masks(
    force_single: bool, limit: int
) -> None:
    rng = np.random.default_rng(5711)
    rows = []
    for size in [0, 1, 2, 7, 30, 63, 64, 65, 127, 128, 129]:
        rows.append(rng.normal(size=size).astype(np.float32))
        rows.append(np.resize(np.asarray([0.0, -0.0, 0.5, -0.5], dtype=np.float32), size))
    _compare(rows, limit, force_single)


@_CUDA
@pytest.mark.parametrize("force_single", [False, True])
def test_cuda_selection_exhaustive_small_heap_ties(force_single: bool) -> None:
    # Enumerate every signed-zero/equal-magnitude case through five bonds. The
    # reference is the CPU heap, not another copy of the new bitset comparator.
    rows = [
        np.asarray(values, dtype=np.float32)
        for size in range(6)
        for values in itertools.product([-1.0, -0.0, 0.0, 1.0], repeat=size)
    ]
    _compare(rows, 64, force_single)


@_CUDA
def test_cuda_selection_extreme_finite_fp32_and_full_production_budget() -> None:
    rng = np.random.default_rng(20260917)
    values = np.asarray(
        [
            0.0,
            -0.0,
            np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
            np.finfo(np.float32).tiny,
            -np.finfo(np.float32).tiny,
            np.nextafter(np.float32(0), np.float32(1)),
            np.nextafter(np.float32(1), np.float32(2)),
            1.0,
            -1.0,
        ],
        dtype=np.float32,
    )
    rows = [rng.choice(values, size=size) for size in [2, 7, 30, 65, 129]]
    rows.extend([rng.normal(size=30).astype(np.float32), np.zeros(30, dtype=np.float32)])
    _compare(rows, 4096, True)


@_CUDA
@pytest.mark.parametrize("invalid", [float("inf"), -float("inf"), float("nan")])
def test_cuda_selection_rejects_nonfinite_logits(invalid: float) -> None:
    logits = th.tensor([1.0, invalid, 0.0], device="cuda")
    with pytest.raises(ValueError, match="finite FP32 logits"):
        rank_batch_cuda(logits, [1, 2], ranked_mask_budget=16, include_single_bond_cuts=True)


@_CUDA
def test_cuda_selection_rejects_float64_mixtures_floors_and_wrong_partition() -> None:
    with pytest.raises(ValueError, match="FP32 single-mixture"):
        rank_batch_cuda(
            th.zeros(2, dtype=th.float64, device="cuda"),
            [2],
            ranked_mask_budget=16,
            include_single_bond_cuts=True,
        )
    with pytest.raises(ValueError, match="FP32 single-mixture"):
        rank_batch_cuda(
            th.zeros((2, 2), device="cuda"),
            [2],
            ranked_mask_budget=16,
            include_single_bond_cuts=True,
        )
    for name in ["min_probability", "min_relative_probability"]:
        with pytest.raises(ValueError, match="disabled probability floors"):
            rank_batch_cuda(
                th.zeros(2, device="cuda"),
                [2],
                ranked_mask_budget=16,
                include_single_bond_cuts=True,
                **{name: 0.01},
            )
    with pytest.raises(ValueError, match="partition"):
        rank_batch_cuda(
            th.zeros(2, device="cuda"),
            [1],
            ranked_mask_budget=16,
            include_single_bond_cuts=True,
        )


@_CUDA
def test_cuda_selection_nondefault_stream_and_noncontiguous_logits() -> None:
    stream = th.cuda.Stream()
    with th.cuda.stream(stream):
        _compare([np.zeros(65, dtype=np.float32), np.asarray([0.5, -0.1])], 257, True)
        logits = th.tensor([0.5, 99.0, -0.1, 99.0], device="cuda")[::2]
        actual = rank_batch_cuda(logits, [2], ranked_mask_budget=16, include_single_bond_cuts=False)
        expected = rank_batch_cuda(
            logits.contiguous(), [2], ranked_mask_budget=16, include_single_bond_cuts=False
        )
        assert th.equal(actual.mask_ptr, expected.mask_ptr)
        assert th.equal(actual.broken_bonds, expected.broken_bonds)


@_CUDA
@pytest.mark.parametrize("target", [1, 6, 256])
def test_cuda_selection_connects_directly_to_gpu_topology(target: int) -> None:
    from fragnnet.models.masster.path_selection.cuda_topology import (
        CudaTopologyGraph,
        build_cuda_topology_device_masks,
    )

    graphs = [
        CudaTopologyGraph(70, (tuple(range(69)), tuple(range(1, 70))), target),
        CudaTopologyGraph(4, ((0, 1, 2), (1, 2, 0)), target),
        CudaTopologyGraph(1, ((), ()), target),
    ]
    rng = np.random.default_rng(124)
    values = [rng.normal(size=len(graph.endpoints[0])).astype(np.float32) for graph in graphs]
    ranked = rank_batch_cuda(
        th.from_numpy(np.concatenate(values)).cuda(),
        [len(row) for row in values],
        ranked_mask_budget=64,
        include_single_bond_cuts=True,
    )
    result = build_cuda_topology_device_masks(
        graphs,
        ranked.mask_ptr,
        ranked.broken_bonds,
        mask_counts=[count.num_masks for count in ranked.counts],
        forced_mask_counts=[count.forced_mask_count for count in ranked.counts],
        tile_masks=3,
    )
    offsets = [0] * 11
    for graph, row, counts in zip(graphs, values, result.counts, strict=True):
        learned_ptr, learned_bonds = top_mixture_bond_sets_raw(
            row.astype(np.float64)[:, None], np.zeros(1), 64, 0.0, 0.0
        )
        forced = [(), *((bond,) for bond in range(len(row)))]
        ptr, bonds, forced_count = _merge_forced_masks(
            learned_ptr,
            learned_bonds,
            forced,
            np.diff(learned_ptr),
            len(learned_ptr) - 1,
            len(forced),
            True,
        )
        expected = build_component_arrays_raw(
            graph.num_atoms,
            np.asarray(graph.endpoints, dtype=np.int64).T,
            (ptr, bonds),
            np.arange(len(row), dtype=np.int64),
            0,
            0,
            0,
            target,
            forced_count,
        )
        arrays = (*expected[:9], expected[11], expected[12])
        for field, (reference, length) in enumerate(zip(arrays, counts.field_lengths, strict=True)):
            actual = result.arrays[field][offsets[field] : offsets[field] + length]
            np.testing.assert_array_equal(actual.cpu().numpy(), reference)
            offsets[field] += length
        assert counts.scanned_mask_count == expected[13]
        assert counts.num_masks == len(expected[0])
        assert counts.num_components == expected[9]
        assert counts.num_unique_components == expected[10]
