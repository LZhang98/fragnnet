"""Explicitly loaded CUDA topology for CPU-selected or device-resident mask CSR.

It builds the same integer candidate arrays as the Cython reference using bounded mask tiles. The
native library is compiled lazily with an existing CUDA toolkit into a /tmp cache;
importing this module does not require a CUDA device, compiler, or binary.
"""

from __future__ import annotations

import ctypes
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import torch as th

from fragnnet.models.masster.candidates.cuda_build import (
    build_cuda_library,
    default_cuda_cache_dir,
    resolve_cuda_home,
)

if TYPE_CHECKING:
    from fragnnet.models.masster.candidates.builder import _TopologyInput

_BUILD_LOCK = Lock()
_META_WIDTH = 15
_STATE_WIDTH = 14
_MAX_ATOMS = 512
_MAX_BONDS = 2048
_MAX_MASKS = 65536


def _checked_integer(value: object, name: str) -> int:
    try:
        integer = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if integer < -(1 << 63) or integer >= 1 << 63:
        raise ValueError(f"{name} must fit signed int64")
    return integer


@dataclass(frozen=True)
class CudaTopologyGraph:
    """Static CPU graph metadata paired with device-resident selected mask CSR."""

    num_atoms: int
    endpoints: tuple[tuple[int, ...], tuple[int, ...]]
    max_unique_components: int


@dataclass(frozen=True)
class CudaTopologyCounts:
    """Host-known ragged lengths and reference logical stopping count for one graph."""

    num_masks: int
    num_components: int
    num_unique_components: int
    scanned_mask_count: int
    num_atom_memberships: int
    num_broken_bonds: int
    num_boundary_memberships: int

    @property
    def field_lengths(self) -> tuple[int, ...]:
        """Lengths of the eleven integer output fields, in raw-kernel order."""
        return (
            self.num_masks,
            self.num_components,
            self.num_components,
            self.num_atom_memberships,
            self.num_atom_memberships,
            self.num_broken_bonds,
            self.num_broken_bonds,
            self.num_boundary_memberships,
            self.num_boundary_memberships,
            self.num_components,
            self.num_unique_components,
        )


@dataclass(frozen=True)
class CudaTopologyBatch:
    """Eleven concatenated GPU fields whose values remain molecule-local.

    ``arrays`` contains raw arrays 0:9 followed by component-to-unique mappings and
    first representative components. Each field concatenates representatives in
    input order. ``counts`` provides the per-representative splitting lengths.
    Physical counts include first-pass speculative tile suffixes, excluding replay;
    logical scanned counts exactly match the CPU stopping contract.
    """

    arrays: tuple[th.Tensor, ...]
    counts: tuple[CudaTopologyCounts, ...]
    physical_mask_counts: tuple[int, ...]
    scratch_bytes: int
    native_load_ms: float


@lru_cache(maxsize=8)
def _load_native_library(
    cache_dir: str, cuda_home: str, capability: tuple[int, int]
) -> ctypes.CDLL:
    flags = [
        "-O3",
        "-std=c++17",
        "--shared",
        "-Xcompiler=-fPIC",
        f"-arch=sm_{capability[0]}{capability[1]}",
        "-Xlinker=-rpath",
        f"-Xlinker={cuda_home}/lib64",
    ]
    library = build_cuda_library(
        Path(__file__).with_suffix(".cu"),
        "topology",
        _BUILD_LOCK,
        cache_dir,
        cuda_home,
        flags,
    )
    library.masster_topology_launch.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        *([ctypes.c_int] * 6),
        ctypes.c_uint64,
    ]
    library.masster_topology_launch.restype = ctypes.c_int
    library.masster_topology_validate_mask_csr.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int,
        ctypes.c_int64,
        ctypes.c_uint64,
    ]
    library.masster_topology_validate_mask_csr.restype = ctypes.c_int
    library.masster_topology_error_string.argtypes = [ctypes.c_int]
    library.masster_topology_error_string.restype = ctypes.c_char_p
    return library


def _prepare_inputs(
    requests: Sequence[_TopologyInput | CudaTopologyGraph],
    tile_masks: int,
    max_scratch_bytes: int,
    *,
    device_mask_layout: Sequence[tuple[int, int]] | None = None,
    device_broken_count: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, ...], int]:
    """Validate CPU inputs and compute explicit bounds before allocating device buffers."""
    if not requests:
        raise ValueError("CUDA topology requires at least one representative")
    if tile_masks < 1 or tile_masks > 256:
        raise ValueError("tile_masks must be in [1, 256]")
    if max_scratch_bytes <= 0:
        raise ValueError("max_scratch_bytes must be positive")
    if device_mask_layout is not None and len(device_mask_layout) != len(requests):
        raise ValueError("device mask counts must provide one entry per representative")
    metadata: list[list[int]] = []
    all_endpoints: list[np.ndarray] = []
    all_mask_ptr: list[np.ndarray] = []
    all_broken: list[np.ndarray] = []
    edge_offset = mask_offset = broken_offset = label_offset = signature_offset = 0
    dictionary_offset = first_offset = hash_offset = 0
    for representative, request in enumerate(requests):
        atoms = _checked_integer(request.num_atoms, "num_atoms")
        if atoms < 1 or atoms > _MAX_ATOMS:
            raise ValueError(f"CUDA topology supports num_atoms in [1, {_MAX_ATOMS}]")
        if request.max_unique_components is None or request.max_unique_components <= 0:
            raise ValueError("CUDA topology prototype requires a positive unique-fragment target")
        target = _checked_integer(request.max_unique_components, "unique-fragment target")
        endpoints = np.asarray(request.endpoints, dtype=np.int64)
        if endpoints.ndim != 2 or endpoints.shape[0] != 2:
            raise ValueError("endpoints must contain two bond-index rows")
        bonds = endpoints.shape[1]
        if bonds > _MAX_BONDS:
            raise ValueError(f"CUDA topology supports at most {_MAX_BONDS} bonds")
        if np.any(endpoints < 0) or np.any(endpoints >= atoms):
            raise ValueError("endpoints reference an invalid local atom")
        if device_mask_layout is None:
            ptr = np.asarray(request.mask_ptr)
            broken = np.asarray(request.broken_bonds)
            if (
                ptr.dtype != np.int64
                or broken.dtype != np.int64
                or ptr.ndim != 1
                or broken.ndim != 1
            ):
                raise ValueError("mask_ptr and broken_bonds must be one-dimensional int64 arrays")
            masks = ptr.size - 1
            if ptr.size == 0 or ptr[0] != 0 or np.any(ptr[1:] < ptr[:-1]) or ptr[-1] != broken.size:
                raise ValueError("invalid mask CSR pointers")
            if np.any(broken < 0) or np.any(broken >= bonds):
                raise ValueError("broken_bonds references an invalid local bond")
            if np.any(np.diff(ptr) > (1 << 30) - 1):
                raise ValueError("mask cut count exceeds supported int32 boundary counters")
            forced = _checked_integer(request.forced_mask_count, "forced_mask_count")
            all_mask_ptr.append(ptr + broken_offset)
            all_broken.append(broken.astype(np.int32, copy=True))
            broken_offset += broken.size
        else:
            masks, forced = device_mask_layout[representative]
            masks = _checked_integer(masks, "mask_count")
            forced = _checked_integer(forced, "forced_mask_count")
        if masks < 1 or masks > _MAX_MASKS:
            raise ValueError(f"CUDA topology supports mask counts in [1, {_MAX_MASKS}]")
        if forced < 0 or forced > masks:
            raise ValueError("forced_mask_count must lie within the mask pool")
        words = (atoms + 63) // 64
        # Before the forced prefix ends, every scanned atom may be unique. After
        # it ends, at most one whole mask overshoots the target before stopping.
        capacity = min(masks * atoms, max(forced * atoms, target + atoms))
        hash_capacity = 1 << (max(2 * capacity, 2) - 1).bit_length()
        metadata.append(
            [
                atoms,
                bonds,
                masks,
                words,
                edge_offset,
                mask_offset,
                label_offset,
                signature_offset,
                dictionary_offset,
                first_offset,
                hash_offset,
                hash_capacity,
                capacity,
                target,
                forced,
            ]
        )
        all_endpoints.append(endpoints.T.astype(np.int32, copy=True).reshape(-1))
        edge_offset += bonds
        mask_offset += masks + 1
        label_offset += tile_masks * atoms
        signature_offset += tile_masks * atoms * words
        dictionary_offset += capacity * words
        first_offset += capacity
        hash_offset += hash_capacity
    meta = np.asarray(metadata, dtype=np.int64)
    endpoints_array = np.concatenate(all_endpoints)
    ptr_array = np.concatenate(all_mask_ptr) if all_mask_ptr else np.empty(0, dtype=np.int64)
    broken_array = np.concatenate(all_broken) if all_broken else np.empty(0, dtype=np.int32)
    sizes = (
        label_offset,
        signature_offset,
        dictionary_offset,
        first_offset,
        hash_offset,
        mask_offset,
    )
    scratch_bytes = (
        meta.nbytes
        + endpoints_array.nbytes
        + ptr_array.nbytes
        + broken_array.nbytes
        + 4 * label_offset
        + 8 * (signature_offset + dictionary_offset + first_offset)
        + 4 * hash_offset
        + 16 * mask_offset
        + 8 * (_STATE_WIDTH + 11) * len(requests)
    )
    if device_mask_layout is not None:
        scratch_bytes += 8 * mask_offset + 4 * device_broken_count
    if scratch_bytes > max_scratch_bytes:
        raise ValueError(
            f"CUDA topology scratch needs {scratch_bytes} bytes, exceeding explicit limit "
            f"{max_scratch_bytes}; reduce batch/tile size or use CPU topology"
        )
    return meta, endpoints_array, ptr_array, broken_array, sizes, scratch_bytes


def build_cuda_topology(
    requests: Sequence[_TopologyInput],
    device: th.device | str,
    *,
    tile_masks: int = 128,
    cache_dir: str | None = None,
    cuda_home: str | None = None,
    max_scratch_bytes: int = 512 * 1024**2,
) -> CudaTopologyBatch:
    """Construct exact GPU topology for preselected CPU mask CSR inputs.

    This prototype supports unique-fragment mode with explicit forced prefixes.
    Selection, grouping and SMARTS eligibility are caller responsibilities; no
    SMARTS mechanisms are added here. All computation uses the current CUDA stream.
    One batched host count read allocates ragged outputs; a final status read checks
    construction errors. These synchronization boundaries belong in benchmark time.

    Args:
        requests: Representative molecule-local graph and selected-mask descriptors.
        device: CUDA device that receives all eleven output index tensors.
        tile_masks: Bounded masks per graph in shared topology scratch, 1 through 256.
        cache_dir: Explicit /tmp directory, or the default per-user native cache.
        cuda_home: Existing CUDA toolkit root, or discover from environment/PATH.
        max_scratch_bytes: Explicit allocation ceiling, excluding final output tensors.

    Returns:
        Concatenated GPU arrays, host splitting/stopping counts and scratch size.

    Raises:
        ValueError: An input, unsupported bound, mode or scratch budget is invalid.
        RuntimeError: CUDA/compiler availability, compilation or kernel execution fails.
    """
    device = th.device(device)
    if device.type != "cuda":
        raise ValueError("CUDA topology requires an explicitly selected CUDA device")
    meta, endpoints, ptr, broken, sizes, scratch_bytes = _prepare_inputs(
        requests, tile_masks, max_scratch_bytes
    )
    return _run_cuda_topology(
        meta,
        endpoints,
        ptr,
        broken,
        sizes,
        scratch_bytes,
        device,
        tile_masks=tile_masks,
        cache_dir=cache_dir,
        cuda_home=cuda_home,
    )


def build_cuda_topology_device_masks(
    graphs: Sequence[CudaTopologyGraph],
    mask_ptr: th.Tensor,
    broken_bonds: th.Tensor,
    *,
    mask_counts: Sequence[int],
    forced_mask_counts: Sequence[int],
    tile_masks: int = 128,
    cache_dir: str | None = None,
    cuda_home: str | None = None,
    max_scratch_bytes: int = 512 * 1024**2,
) -> CudaTopologyBatch:
    """Build topology directly from device-resident ranked-mask CSR.

    Graph descriptors contain static host metadata only. No mask entries or logits
    are copied to CPU. Device CSR validation precedes connectivity kernels and reports
    malformed pointers/bond indices through the existing batched status read.

    Args:
        graphs: Representative graph metadata with positive unique targets.
        mask_ptr: CUDA int64 vector concatenating each representative's M+1 pointers,
            whose entries index the single concatenated broken-bond tensor.
        broken_bonds: CUDA int32 vector of molecule-local broken bond IDs.
        mask_counts: Host-known mask count for each representative.
        forced_mask_counts: Host-known required prefix length for each representative.
        tile_masks: Bounded masks per graph in scratch, 1 through 256.
        cache_dir: Explicit /tmp directory, or the default per-user native cache.
        cuda_home: Existing CUDA toolkit root, or discover from environment/PATH.
        max_scratch_bytes: Explicit global scratch limit, excluding final outputs.

    Returns:
        The same concatenated fields and host metadata as ``build_cuda_topology``.

    Raises:
        ValueError: Tensor layout, device, counts or graph bounds are unsupported.
        RuntimeError: CUDA is unavailable, compilation fails or device CSR is invalid.
    """
    if (
        mask_ptr.device.type != "cuda"
        or broken_bonds.device != mask_ptr.device
        or mask_ptr.dtype != th.int64
        or broken_bonds.dtype != th.int32
        or mask_ptr.ndim != 1
        or broken_bonds.ndim != 1
        or not mask_ptr.is_contiguous()
        or not broken_bonds.is_contiguous()
    ):
        raise ValueError("device mask CSR requires contiguous CUDA int64 pointers and int32 bonds")
    if len(mask_counts) != len(graphs) or len(forced_mask_counts) != len(graphs):
        raise ValueError("mask counts must provide one entry per representative")
    if mask_ptr.numel() != sum(count + 1 for count in mask_counts):
        raise ValueError("device mask pointer length disagrees with mask_counts")
    layout = list(zip(mask_counts, forced_mask_counts, strict=True))
    meta, endpoints, _, _, sizes, scratch_bytes = _prepare_inputs(
        graphs,
        tile_masks,
        max_scratch_bytes,
        device_mask_layout=layout,
        device_broken_count=broken_bonds.numel(),
    )
    return _run_cuda_topology(
        meta,
        endpoints,
        mask_ptr,
        broken_bonds,
        sizes,
        scratch_bytes,
        mask_ptr.device,
        tile_masks=tile_masks,
        cache_dir=cache_dir,
        cuda_home=cuda_home,
        validate_device_masks=True,
    )


def _run_cuda_topology(
    meta: np.ndarray,
    endpoints: np.ndarray,
    ptr: np.ndarray | th.Tensor,
    broken: np.ndarray | th.Tensor,
    sizes: tuple[int, ...],
    scratch_bytes: int,
    device: th.device,
    *,
    tile_masks: int,
    cache_dir: str | None,
    cuda_home: str | None,
    validate_device_masks: bool = False,
) -> CudaTopologyBatch:
    num_representatives = meta.shape[0]
    if not th.cuda.is_available():
        raise RuntimeError("CUDA topology requested without a usable CUDA device")
    with th.cuda.device(device):
        load_start = perf_counter()
        library = _load_native_library(
            default_cuda_cache_dir("topology") if cache_dir is None else cache_dir,
            resolve_cuda_home(cuda_home),
            th.cuda.get_device_capability(device),
        )
        native_load_ms = (perf_counter() - load_start) * 1000.0
        label_size, signature_size, dictionary_size, first_size, hash_size, prefix_size = sizes
        buffers = [
            th.from_numpy(meta).to(device),
            th.from_numpy(endpoints).to(device),
            ptr if isinstance(ptr, th.Tensor) else th.from_numpy(ptr).to(device),
            broken if isinstance(broken, th.Tensor) else th.from_numpy(broken).to(device),
            th.empty(label_size, dtype=th.int32, device=device),
            th.empty(signature_size, dtype=th.int64, device=device),
            th.empty(dictionary_size, dtype=th.int64, device=device),
            th.empty(first_size, dtype=th.int64, device=device),
            th.zeros(hash_size, dtype=th.int32, device=device),
            th.zeros(prefix_size, dtype=th.int64, device=device),
            th.zeros(prefix_size, dtype=th.int64, device=device),
            th.zeros((num_representatives, _STATE_WIDTH), dtype=th.int64, device=device),
            th.empty(0, dtype=th.int64, device=device),
            th.empty(0, dtype=th.int64, device=device),
        ]
        maximum_atoms = int(meta[:, 0].max())
        maximum_bonds = int(meta[:, 1].max())
        stream = th.cuda.current_stream(device).cuda_stream

        def launch(tile_start: int, replay: bool) -> None:
            pointers = (ctypes.c_uint64 * len(buffers))(*(value.data_ptr() for value in buffers))
            error = library.masster_topology_launch(
                pointers,
                num_representatives,
                tile_start,
                tile_masks,
                maximum_atoms,
                maximum_bonds,
                int(replay),
                stream,
            )
            if error:
                message = library.masster_topology_error_string(error).decode()
                raise RuntimeError(f"CUDA topology kernel launch failed: {message}")

        if validate_device_masks:
            pointers = (ctypes.c_uint64 * len(buffers))(*(value.data_ptr() for value in buffers))
            error = library.masster_topology_validate_mask_csr(
                pointers,
                num_representatives,
                buffers[3].numel(),
                stream,
            )
            if error:
                message = library.masster_topology_error_string(error).decode()
                raise RuntimeError(f"CUDA topology CSR validation launch failed: {message}")
        for tile_start in range(0, int(meta[:, 2].max()), tile_masks):
            launch(tile_start, False)
        state = buffers[11].cpu().numpy()
        if np.any(state[:, 12]) or not np.all(state[:, 0]):
            raise RuntimeError(f"CUDA topology first pass failed: status={state[:, 12].tolist()}")
        counts = tuple(
            CudaTopologyCounts(
                num_masks=int(row[6]),
                num_components=int(row[7]),
                num_unique_components=int(row[8]),
                scanned_mask_count=int(row[5]),
                num_atom_memberships=int(row[9]),
                num_broken_bonds=int(row[10]),
                num_boundary_memberships=int(row[11]),
            )
            for row in state
        )
        lengths = np.asarray([count.field_lengths for count in counts], dtype=np.int64)
        field_totals = lengths.sum(axis=0)
        field_starts = np.r_[0, np.cumsum(field_totals[:-1])]
        offsets = field_starts[None, :] + np.cumsum(lengths, axis=0) - lengths
        buffers[12] = th.empty(int(field_totals.sum()), dtype=th.int64, device=device)
        buffers[13] = th.from_numpy(offsets).to(device)
        for tile_start in range(0, max(count.num_masks for count in counts), tile_masks):
            launch(tile_start, True)
        errors = buffers[11][:, 12].cpu().tolist()
        if any(errors):
            raise RuntimeError(f"CUDA topology assembly failed: status={errors}")
        return CudaTopologyBatch(
            arrays=tuple(buffers[12].split(field_totals.tolist())),
            counts=counts,
            physical_mask_counts=tuple(int(row[13]) for row in state),
            scratch_bytes=scratch_bytes,
            native_load_ms=native_load_ms,
        )
