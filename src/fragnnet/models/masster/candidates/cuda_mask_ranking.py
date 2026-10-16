"""Exact CUDA selector for finite FP32 single-component logits.

Status: PRODUCTION, the mask-ranking stage of the CUDA backend (`cuda_candidates`). The CPU
selector remains the template default. Compilation is lazy and importing this module does
not require a CUDA device or toolkit.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np
import torch as th

from fragnnet.models.masster.candidates.cuda_build import (
    build_cuda_library,
    default_cuda_cache_dir,
    resolve_cuda_home,
)

_BUILD_LOCK = Lock()
_MAX_BONDS = 2048
_MAX_RANKED_MASKS = 65536
_STATE_WIDTH = 6


@dataclass(frozen=True)
class RankedMaskCounts:
    """Per-representative lengths, including full learned-pool diagnostics."""

    num_masks: int
    forced_mask_count: int
    learned_mask_count: int
    learned_single_cut_count: int
    num_broken_bonds: int


@dataclass(frozen=True)
class RankedMasksCuda:
    """Packed CUDA CSR masks in representative input order.

    Every representative contributes ``num_masks + 1`` int64 ``mask_ptr`` entries.
    Pointers index the concatenated int32 ``broken_bonds`` buffer, whose values are
    molecule-local bond IDs. ``device_counts`` columns are status, learned masks,
    learned single cuts, merged masks, forced prefix length, and broken-bond entries.
    Only these small counts synchronize to the host; logits never leave the device.
    """

    mask_ptr: th.Tensor
    broken_bonds: th.Tensor
    counts: tuple[RankedMaskCounts, ...]
    device_counts: th.Tensor
    scratch_bytes: int
    native_load_ms: float


@lru_cache(maxsize=8)
def _load_native_library(
    cache_dir: str, cuda_home: str, capability: tuple[int, int]
) -> ctypes.CDLL:
    flags = (
        "-O3",
        "-std=c++17",
        "--shared",
        "-Xcompiler=-fPIC",
        f"-arch=sm_{capability[0]}{capability[1]}",
        "--fmad=false",
        "--ftz=false",
        "-Xlinker=-rpath",
        f"-Xlinker={cuda_home}/lib64",
    )
    library = build_cuda_library(
        Path(__file__).with_suffix(".cu"),
        "selection",
        _BUILD_LOCK,
        cache_dir,
        cuda_home,
        flags,
    )
    library.masster_selection_launch.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
    ]
    library.masster_selection_launch.restype = ctypes.c_int
    library.masster_selection_error_string.argtypes = [ctypes.c_int]
    library.masster_selection_error_string.restype = ctypes.c_char_p
    return library


def _extra_forced_words(
    bond_counts: Sequence[int],
    extra_forced_bond_sets: Sequence[Sequence[Sequence[int]]] | None,
    include_single_bond_cuts: bool,
) -> tuple[list[int], np.ndarray]:
    """Pack extra forced masks as per-representative bit words, validating each set.

    Returns the extra mask count per representative and the concatenated uint64 words,
    stored as int64 for transfer. Each representative uses ``max(ceil(bonds / 64), 1)``
    words per mask, matching the kernel's mask layout.
    """
    if extra_forced_bond_sets is None:
        return [0] * len(bond_counts), np.zeros(0, dtype=np.int64)
    if len(extra_forced_bond_sets) != len(bond_counts):
        raise ValueError("extra_forced_bond_sets must contain one sequence per representative")
    counts: list[int] = []
    words: list[np.ndarray] = []
    for bonds, bond_sets in zip(bond_counts, extra_forced_bond_sets, strict=True):
        width = max((int(bonds) + 63) // 64, 1)
        packed = np.zeros((len(bond_sets), width), dtype=np.uint64)
        seen: set[tuple[int, ...]] = set()
        for row, bond_set in enumerate(bond_sets):
            key = tuple(int(bond) for bond in bond_set)
            if (
                not key
                or (include_single_bond_cuts and len(key) == 1)
                or any(left >= right for left, right in zip(key, key[1:]))
                or key[0] < 0
                or key[-1] >= bonds
                or key in seen
            ):
                raise ValueError(
                    "extra forced masks must be unique, strictly increasing, in-range bond "
                    "sets that are not already in the empty/single-cut prefix"
                )
            seen.add(key)
            for bond in key:
                packed[row, bond // 64] |= np.uint64(1) << np.uint64(bond % 64)
        counts.append(len(bond_sets))
        words.append(packed.reshape(-1))
    flat = np.concatenate(words) if words else np.zeros(0, dtype=np.uint64)
    return counts, flat.view(np.int64)


def _prepare_layout(
    bond_counts: Sequence[int],
    ranked_mask_budget: int,
    include_single_bond_cuts: bool,
    max_scratch_bytes: int,
    extra_forced_counts: Sequence[int] | None = None,
) -> tuple[np.ndarray, tuple[int, ...], int]:
    """Compute ragged workspace bounds using graph metadata, never GPU logits."""
    if len(bond_counts) == 0:
        raise ValueError("CUDA selection requires at least one representative")
    if extra_forced_counts is None:
        extra_forced_counts = [0] * len(bond_counts)
    if len(extra_forced_counts) != len(bond_counts):
        raise ValueError("extra_forced_counts must contain one count per representative")
    if (
        isinstance(ranked_mask_budget, bool)
        or not isinstance(ranked_mask_budget, Integral)
        or not 1 <= ranked_mask_budget <= _MAX_RANKED_MASKS
    ):
        raise ValueError(f"ranked_mask_budget must be in [1, {_MAX_RANKED_MASKS}]")
    ranked_mask_budget = int(ranked_mask_budget)
    if max_scratch_bytes <= 0:
        raise ValueError("max_scratch_bytes must be positive")
    rows = []
    input_offset = cost_offset = bits_offset = node_offset = heap_offset = 0
    mask_offset = prefix_offset = work_offset = extra_offset = 0
    for count, extra_count in zip(bond_counts, extra_forced_counts, strict=True):
        if (
            isinstance(count, bool)
            or not isinstance(count, Integral)
            or not 0 <= count <= _MAX_BONDS
        ):
            raise ValueError(f"CUDA selection supports integer bond counts in [0, {_MAX_BONDS}]")
        bonds = int(count)
        # Saturate before shifting, so large molecules never construct 2**bonds.
        limit = min(ranked_mask_budget, 1 << min(bonds, ranked_mask_budget.bit_length()))
        capacity = 2 * limit + 1
        words = max((bonds + 63) // 64, 1)
        mask_capacity = limit + 1 + (bonds if include_single_bond_cuts else 0) + extra_count
        rows.append(
            [
                bonds,
                input_offset,
                limit,
                capacity,
                words,
                cost_offset,
                bits_offset,
                node_offset,
                heap_offset,
                mask_offset,
                mask_capacity,
                prefix_offset,
                work_offset,
                extra_offset,
                extra_count,
            ]
        )
        input_offset += bonds
        cost_offset += max(bonds, 1)
        bits_offset += capacity * words
        node_offset += capacity
        heap_offset += capacity + 1
        mask_offset += mask_capacity * words
        prefix_offset += mask_capacity + 1
        work_offset += 2 * words
        extra_offset += extra_count * words
    meta = np.asarray(rows, dtype=np.int64)
    sizes = (
        cost_offset,
        bits_offset,
        node_offset,
        heap_offset,
        mask_offset,
        work_offset,
        prefix_offset,
    )
    scratch_bytes = (
        meta.nbytes
        + 12 * cost_offset
        + 8 * bits_offset
        + 8 * node_offset
        + 4 * heap_offset
        + 8 * (mask_offset + work_offset + prefix_offset)
        + 8 * _STATE_WIDTH * len(rows)
        + 16 * len(rows)  # Per-representative output pointer/bond offsets.
        + 8 * extra_offset  # Extra forced mask words.
    )
    if scratch_bytes > max_scratch_bytes:
        raise ValueError(
            f"CUDA selection scratch needs {scratch_bytes} bytes, exceeding explicit limit "
            f"{max_scratch_bytes}; reduce batch/budget or use CPU selection"
        )
    return meta, sizes, scratch_bytes


def rank_batch_cuda(
    bond_logits: th.Tensor,
    bond_counts: Sequence[int],
    *,
    ranked_mask_budget: int,
    include_single_bond_cuts: bool,
    extra_forced_bond_sets: Sequence[Sequence[Sequence[int]]] | None = None,
    min_probability: float = 0.0,
    min_relative_probability: float = 0.0,
    cache_dir: str | None = None,
    cuda_home: str | None = None,
    max_scratch_bytes: int = 512 * 1024**2,
) -> RankedMasksCuda:
    """Rank and force-merge exact single-component masks without copying logits to CPU.

    Scope is unique-fragment mode, zero probability floors, and finite FP32 logits.
    Each representative supplies one contiguous logit segment; grouping and its
    equality checks belong to the caller. Double heap arithmetic and flip-position tie
    order match the CPU kernel. Required empty/single-cut masks form the prefix,
    followed by any extra forced masks (SMARTS template cuts); learned duplicates of
    forced masks are dropped, while diagnostics count the complete original learned pool.

    Args:
        bond_logits: Flat FP32 CUDA logits of shape ``(sum(bond_counts),)``.
        bond_counts: Host-known bond counts, one per independent representative.
        ranked_mask_budget: Maximum learned masks per representative, 1 through 65536.
        include_single_bond_cuts: Force all single cuts after the required empty mask.
        extra_forced_bond_sets: Optional per-representative bond sets forced after the
            empty/single-cut prefix, in order. Each set must be unique, strictly
            increasing, nonempty and, when singles are forced, have two or more bonds.
        min_probability: Only zero is supported by this prototype.
        min_relative_probability: Only zero is supported by this prototype.
        cache_dir: Explicit /tmp directory, or the default per-user native cache.
        cuda_home: Existing toolkit root, or discover a compatible installed toolkit.
        max_scratch_bytes: Allocation limit excluding final CSR tensors and input logits.

    Returns:
        GPU CSR tensors, device diagnostics and host ragged lengths. One batched count
        read synchronizes ranking before exact output allocation; CSR filling remains
        on the current CUDA stream.

    Raises:
        ValueError: Unsupported device, dtype, shape, floors, bounds or nonfinite logits.
        RuntimeError: Native compilation, launch or workspace invariants fail.
    """
    if bond_logits.device.type != "cuda":
        raise ValueError("CUDA selection requires CUDA logits")
    if bond_logits.dtype != th.float32 or bond_logits.ndim != 1:
        raise ValueError("CUDA selection supports only one-dimensional FP32 single-mixture logits")
    if min_probability != 0.0 or min_relative_probability != 0.0:
        raise ValueError("CUDA selection supports only disabled probability floors")
    extra_counts, extra_words = _extra_forced_words(
        bond_counts, extra_forced_bond_sets, include_single_bond_cuts
    )
    meta, sizes, scratch_bytes = _prepare_layout(
        bond_counts, ranked_mask_budget, include_single_bond_cuts, max_scratch_bytes, extra_counts
    )
    if int(meta[:, 0].sum()) != bond_logits.numel():
        raise ValueError("bond_counts must partition the flat bond_logits tensor")
    device = bond_logits.device
    with th.cuda.device(device):
        native_start = perf_counter()
        toolkit = resolve_cuda_home(cuda_home)
        library = _load_native_library(
            default_cuda_cache_dir("selection") if cache_dir is None else cache_dir,
            toolkit,
            th.cuda.get_device_capability(device),
        )
        native_load_ms = 1000.0 * (perf_counter() - native_start)
        costs, bits, nodes, heap, masks, work, prefix = sizes
        buffers = [
            bond_logits.detach().contiguous(),
            th.from_numpy(meta).to(device),
            th.empty(costs, dtype=th.float64, device=device),
            th.empty(costs, dtype=th.int32, device=device),
            th.empty(nodes, dtype=th.float64, device=device),
            th.empty(bits, dtype=th.int64, device=device),
            th.empty(heap, dtype=th.int32, device=device),
            th.empty(masks, dtype=th.int64, device=device),
            th.empty(work, dtype=th.int64, device=device),
            th.empty((len(bond_counts), _STATE_WIDTH), dtype=th.int64, device=device),
            th.empty(prefix, dtype=th.int64, device=device),
            th.empty(0, dtype=th.int64, device=device),
            th.empty(0, dtype=th.int64, device=device),
            th.empty(0, dtype=th.int32, device=device),
            th.from_numpy(extra_words).to(device),
        ]
        stream = th.cuda.current_stream(device).cuda_stream

        def launch(csr_phase: bool) -> None:
            pointers = (ctypes.c_uint64 * len(buffers))(*(value.data_ptr() for value in buffers))
            error = library.masster_selection_launch(
                pointers, len(bond_counts), int(include_single_bond_cuts), int(csr_phase), stream
            )
            if error:
                message = library.masster_selection_error_string(error).decode()
                raise RuntimeError(f"CUDA selection kernel launch failed: {message}")

        launch(False)
        state = buffers[9].cpu().numpy()
        if np.any(state[:, 0] == 1):
            raise ValueError("CUDA selection requires finite FP32 logits")
        if np.any(state[:, 0]):
            raise RuntimeError(f"CUDA selection workspace invariant failed: {state[:, 0].tolist()}")
        counts = tuple(
            RankedMaskCounts(
                num_masks=int(row[3]),
                forced_mask_count=int(row[4]),
                learned_mask_count=int(row[1]),
                learned_single_cut_count=int(row[2]),
                num_broken_bonds=int(row[5]),
            )
            for row in state
        )
        lengths = np.asarray(
            [(count.num_masks + 1, count.num_broken_bonds) for count in counts], dtype=np.int64
        )
        offsets = np.cumsum(lengths, axis=0) - lengths
        buffers[11] = th.from_numpy(offsets).to(device)
        buffers[12] = th.empty(int(lengths[:, 0].sum()), dtype=th.int64, device=device)
        buffers[13] = th.empty(int(lengths[:, 1].sum()), dtype=th.int32, device=device)
        launch(True)
        return RankedMasksCuda(
            buffers[12], buffers[13], counts, buffers[9], scratch_bytes, native_load_ms
        )
