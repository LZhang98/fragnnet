"""Exact CPU/CUDA differentials for the optional fixed-mask topology prototype."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch as th

from fragnnet.frag import masster_cuda_utils as cuda_utils
from fragnnet.frag.masster_candidates import _TopologyInput
from fragnnet.frag.masster_core import build_component_arrays_raw
from fragnnet.frag.masster_cuda_topology import (
    CudaTopologyGraph,
    _prepare_inputs,
    build_cuda_topology,
    build_cuda_topology_device_masks,
)

_CUDA = pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA device unavailable")


def _request(
    atoms: int,
    edges: list[tuple[int, int]],
    masks: list[tuple[int, ...]],
    target: int,
    forced: int = 0,
) -> _TopologyInput:
    ptr = np.r_[0, np.cumsum([len(mask) for mask in masks])].astype(np.int64)
    broken = np.asarray([bond for mask in masks for bond in mask], dtype=np.int64)
    endpoints = tuple(tuple(edge[row] for edge in edges) for row in range(2))
    return _TopologyInput(atoms, endpoints, ptr, broken, forced, target)


def _compare(
    requests: list[_TopologyInput], tile_masks: int, *, device_masks: bool = False
) -> None:
    if device_masks:
        mask_ptrs = []
        broken_offset = 0
        for request in requests:
            mask_ptrs.append(request.mask_ptr + broken_offset)
            broken_offset += request.broken_bonds.size
        ptr = th.from_numpy(np.concatenate(mask_ptrs)).cuda()
        broken = th.from_numpy(
            np.concatenate([request.broken_bonds for request in requests]).astype(np.int32)
        ).cuda()
        result = build_cuda_topology_device_masks(
            [
                CudaTopologyGraph(r.num_atoms, r.endpoints, r.max_unique_components)
                for r in requests
            ],
            ptr,
            broken,
            mask_counts=[len(r.mask_ptr) - 1 for r in requests],
            forced_mask_counts=[r.forced_mask_count for r in requests],
            tile_masks=tile_masks,
        )
    else:
        result = build_cuda_topology(requests, "cuda", tile_masks=tile_masks)
    field_offsets = [0] * 11
    for request, counts, physical in zip(
        requests, result.counts, result.physical_mask_counts, strict=True
    ):
        expected = build_component_arrays_raw(
            request.num_atoms,
            np.asarray(request.endpoints, dtype=np.int64).T,
            (request.mask_ptr, request.broken_bonds),
            np.arange(len(request.endpoints[0]), dtype=np.int64),
            0,
            0,
            0,
            request.max_unique_components,
            request.forced_mask_count,
        )
        expected_arrays = (*expected[:9], expected[11], expected[12])
        for field, (reference, length) in enumerate(
            zip(expected_arrays, counts.field_lengths, strict=True)
        ):
            actual = result.arrays[field][field_offsets[field] : field_offsets[field] + length]
            assert actual.device.type == "cuda"
            assert actual.dtype == th.long
            np.testing.assert_array_equal(actual.cpu().numpy(), reference)
            field_offsets[field] += length
        assert counts.num_masks == len(expected[0])
        assert counts.num_components == expected[9]
        assert counts.num_unique_components == expected[10]
        assert counts.scanned_mask_count == expected[13]
        assert (
            counts.scanned_mask_count
            <= physical
            <= min(
                len(request.mask_ptr) - 1,
                ((counts.scanned_mask_count + tile_masks - 1) // tile_masks) * tile_masks,
            )
        )
    assert field_offsets == [array.numel() for array in result.arrays]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"num_atoms": 0}, "num_atoms"),
        ({"num_atoms": 513}, "num_atoms"),
        ({"max_unique_components": None}, "positive unique"),
        ({"max_unique_components": 0}, "positive unique"),
        ({"forced_mask_count": 3}, "forced_mask_count"),
        ({"mask_ptr": np.asarray([0, 2, 1], dtype=np.int64)}, "CSR"),
        ({"broken_bonds": np.asarray([2], dtype=np.int64)}, "invalid local bond"),
        ({"endpoints": ((0,), (3,))}, "invalid local atom"),
        ({"mask_ptr": np.asarray([0, 0, 1], dtype=np.int32)}, "int64"),
        ({"max_unique_components": 1 << 63}, "int64"),
        ({"max_unique_components": 1.5}, "integer"),
        ({"num_atoms": 2.5}, "integer"),
    ],
)
def test_cuda_topology_rejects_invalid_inputs_before_device_allocation(
    change: dict[str, object], message: str
) -> None:
    request = replace(_request(2, [(0, 1)], [(), (0,)], 2), **change)
    with pytest.raises(ValueError, match=message):
        _prepare_inputs([request], 32, 1024**2)


def test_cuda_topology_scratch_bound_and_device_are_explicit() -> None:
    request = _request(2, [(0, 1)], [(), (0,)], 2)
    with pytest.raises(ValueError, match="scratch needs"):
        _prepare_inputs([request], 32, 1)
    with pytest.raises(ValueError, match="tile_masks"):
        _prepare_inputs([request], 0, 1024**2)
    with pytest.raises(ValueError, match="explicitly selected CUDA"):
        build_cuda_topology([request], "cpu")


@_CUDA
@pytest.mark.parametrize("target", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("tile_masks", [1, 3, 32, 128])
def test_cuda_topology_matches_exact_overshoot_and_equality(target: int, tile_masks: int) -> None:
    _compare([_request(4, [(0, 1), (1, 2), (2, 3)], [(), (0,), (0,), (1,)], target)], tile_masks)


@_CUDA
@pytest.mark.parametrize("tile_masks", [2, 32])
def test_cuda_topology_mixed_graphs_and_forced_prefix(tile_masks: int) -> None:
    requests = [
        _request(1, [], [(), (), ()], 1, 1),
        _request(4, [], [(), (), ()], 2, 2),
        _request(3, [(0, 1), (1, 2), (2, 0)], [(), (2,), (2, 0), (1, 0, 2)], 2, 3),
        _request(3, [(0, 1), (0, 1), (1, 2), (2, 0)], [(), (0,), (0, 1), (2, 3)], 1, 4),
        _request(4, [(0, 1), (2, 3)], [(), (0,), (0, 1)], 9, 0),
        _request(2, [(0, 1), (0, 0)], [(), (0, 0), (1, 0)], 5, 0),
    ]
    _compare(requests, tile_masks)


@_CUDA
def test_cuda_topology_multiword_signatures_and_max_atom_order() -> None:
    requests = [
        _request(71, [(0, 1), (0, 70), *((a, a + 1) for a in range(1, 69))], [(0,)], 256),
        _request(
            257,
            [(a, a + 1) for a in range(256)],
            [(), (255,), (255,), (128,), (64, 128, 255)],
            3,
            1,
        ),
        _request(257, [], [()], 256),
    ]
    _compare(requests, 2)


@_CUDA
@pytest.mark.parametrize("repeat", range(3))
def test_cuda_topology_multiwarp_convergence_and_reused_tiles(repeat: int) -> None:
    requests = [
        _request(
            atoms,
            [(a, a + 1) for a in range(atoms - 1)],
            [(), (atoms - 2,), (atoms // 2,)],
            4096,
        )
        for atoms in (257, 512)
    ]
    # Reach U exactly and then exhaust a long repeated pool: no overshoot means
    # every mask is retained, including suffixes in repeatedly reused tile buffers.
    requests.append(_request(3, [(0, 1), (1, 2)], [(), (0,)] + [()] * 255, 3))
    _compare(requests, 8 + repeat)


@_CUDA
def test_cuda_topology_matches_random_multigraphs_on_nondefault_stream() -> None:
    rng = np.random.default_rng(14079)
    requests = []
    for _ in range(24):
        atoms = int(rng.integers(1, 96))
        edges = [tuple(map(int, pair)) for pair in rng.integers(0, atoms, (atoms * 2, 2))]
        masks = [()]
        for _ in range(40):
            cut_count = int(rng.integers(0, len(edges) + 1))
            masks.append(tuple(map(int, rng.choice(len(edges), cut_count, replace=False))))
        requests.append(_request(atoms, edges, masks, int(rng.integers(1, atoms * 4)), 2))
    stream = th.cuda.Stream()
    with th.cuda.stream(stream):
        _compare(requests, 4)
        _compare(requests, 4, device_masks=True)


@_CUDA
@pytest.mark.parametrize(
    ("ptr", "broken"),
    [
        ([0, 0, 99], [0]),
        ([0, -1, 1], [0]),
        ([0, 1, 1], [1]),
        ([0, 1, 1], [-1]),
    ],
)
def test_device_mask_csr_errors_are_reported_without_device_fault(
    ptr: list[int], broken: list[int]
) -> None:
    graph = CudaTopologyGraph(2, ((0,), (1,)), 2)
    with pytest.raises(RuntimeError, match=r"status=\[3\]"):
        build_cuda_topology_device_masks(
            [graph],
            th.tensor(ptr, dtype=th.int64, device="cuda"),
            th.tensor(broken, dtype=th.int32, device="cuda"),
            mask_counts=[2],
            forced_mask_counts=[0],
        )
    # Validation rejects unsafe indices before connectivity dereferences them;
    # the same CUDA context must remain usable for a valid construction.
    _compare([_request(2, [(0, 1)], [(), (0,)], 2)], 2, device_masks=True)


@pytest.mark.parametrize(
    ("explicit", "environment", "expected"),
    [
        ("/tmp/explicit_cuda", "/tmp/env_cuda", "/tmp/explicit_cuda"),
        (None, "/tmp/env_cuda", "/tmp/env_cuda"),
        (None, None, "/tmp/path_cuda"),
    ],
)
def test_cuda_toolkit_discovery_respects_explicit_precedence(
    monkeypatch: pytest.MonkeyPatch,
    explicit: str | None,
    environment: str | None,
    expected: str,
) -> None:
    cuda_utils._discover_home.cache_clear()
    monkeypatch.setattr(cuda_utils, "_compiler_release", lambda home: (12, 1))
    assert (
        cuda_utils._discover_home(explicit, environment, "/tmp/path_cuda/bin/nvcc", "12.1")
        == expected
    )
    cuda_utils._discover_home.cache_clear()


def test_cuda_toolkit_discovery_rejects_explicit_mismatch_and_finds_installed_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda_utils._discover_home.cache_clear()

    def release(home: str) -> tuple[int, int]:
        return (11, 2) if home == "/tmp/old_cuda" else (12, 1)

    monkeypatch.setattr(cuda_utils, "_compiler_release", release)
    with pytest.raises(RuntimeError, match="does not match"):
        cuda_utils._discover_home("/tmp/old_cuda", "/tmp/new_cuda", None, "12.1")
    with pytest.raises(RuntimeError, match="does not match"):
        cuda_utils._discover_home(None, "/tmp/old_cuda", None, "12.1")
    result = cuda_utils._discover_home(None, None, "/tmp/old_cuda/bin/nvcc", "12.1")
    assert result.endswith("cuda-12.1")
    cuda_utils._discover_home.cache_clear()


def test_default_cuda_cache_is_stable_separate_by_user_and_stage(monkeypatch):
    monkeypatch.setattr(cuda_utils.Path, "home", lambda: cuda_utils.Path("/home/first_user"))
    first = cuda_utils.default_cuda_cache_dir("topology")
    assert first == cuda_utils.default_cuda_cache_dir("topology")
    assert first != cuda_utils.default_cuda_cache_dir("selection")
    assert cuda_utils.Path(first).parent == cuda_utils.Path("/tmp")
    assert "first_user" not in first
    monkeypatch.setattr(cuda_utils.Path, "home", lambda: cuda_utils.Path("/home/second_user"))
    assert first != cuda_utils.default_cuda_cache_dir("topology")
