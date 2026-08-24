"""Direct fragment generation from independent bond-break probabilities.

This module translates one logit per precursor bond into ranked global cut
masks and connected-component fragments. It intentionally has no fragmentation
depth: the number of broken bonds in a mask is determined only by its learned
probability.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankedCutMask:
    """A complete bond-cut mask ranked under independent Bernoulli logits."""

    broken_bond_indices: tuple[int, ...]
    log_probability: float
    log_probability_from_map: float

    @property
    def cut_count(self) -> int:
        """Return the number of broken bonds in the mask."""
        return len(self.broken_bond_indices)


@dataclass(frozen=True)
class CutMaskComponents:
    """Connected components produced by one ranked cut mask."""

    cut_mask: RankedCutMask
    component_atom_masks: tuple[int, ...]


@dataclass(frozen=True)
class FragmentMarginal:
    """Truncated marginal probability for one precursor atom subset."""

    atom_mask: int
    log_probability: float
    contributing_mask_count: int

    @property
    def atom_indices(self) -> tuple[int, ...]:
        """Decode the Python-integer atom bitmask into sorted atom indices."""
        indices: list[int] = []
        remaining = self.atom_mask
        while remaining:
            lowest_bit = remaining & -remaining
            indices.append(lowest_bit.bit_length() - 1)
            remaining ^= lowest_bit
        return tuple(indices)


@dataclass(frozen=True)
class FragmentGenerationResult:
    """Ranked masks and their aggregated connected-component fragments."""

    cut_masks: tuple[CutMaskComponents, ...]
    fragments: tuple[FragmentMarginal, ...]


def _validate_bond_logits(bond_logits: Sequence[float]) -> tuple[float, ...]:
    logits = tuple(float(logit) for logit in bond_logits)
    if not all(math.isfinite(logit) for logit in logits):
        raise ValueError("bond_logits must contain only finite values")
    return logits


def _stable_logaddexp(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    larger = max(left, right)
    return larger + math.log1p(math.exp(min(left, right) - larger))


def _normalize_bond_endpoints(
    num_atoms: int,
    bond_endpoints: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    """Validate PyG-style ``(2, num_bonds)`` endpoints and return edge pairs."""
    endpoint_rows = tuple(tuple(int(atom) for atom in row) for row in bond_endpoints)
    if len(endpoint_rows) != 2 or len(endpoint_rows[0]) != len(endpoint_rows[1]):
        raise ValueError("bond_endpoints must have shape (2, num_bonds)")
    if any(atom < 0 or atom >= num_atoms for row in endpoint_rows for atom in row):
        raise ValueError("bond_endpoints contains an atom index outside [0, num_atoms)")
    return tuple(zip(endpoint_rows[0], endpoint_rows[1]))


def iter_ranked_cut_masks(
    bond_logits: Sequence[float],
    *,
    include_empty: bool = True,
) -> Iterator[RankedCutMask]:
    """Yield every cut mask in exact descending Bernoulli probability order.

    The MAP mask contains every bond with a positive logit. Any other mask is
    represented by flipping a subset of MAP decisions; flip ``b`` has a
    nonnegative log-probability cost ``abs(bond_logits[b])``. A heap enumerates
    the smallest subset sums without materializing all ``2**num_bonds`` masks.

    Args:
        bond_logits: One finite Bernoulli logit per precursor bond.
        include_empty: Whether to yield the mask containing no broken bonds.

    Yields:
        Ranked cut masks. Ties are deterministic but have no chemical ordering.

    Raises:
        ValueError: If any logit is non-finite.
    """
    logits = _validate_bond_logits(bond_logits)
    num_bonds = len(logits)

    map_broken = frozenset(index for index, logit in enumerate(logits) if logit > 0.0)
    map_log_probability = sum(
        -float(np.logaddexp(0.0, -logit)) if logit > 0.0 else -float(np.logaddexp(0.0, logit))
        for logit in logits
    )
    sorted_flips = sorted((abs(logit), index) for index, logit in enumerate(logits))

    def build_result(
        flip_positions: tuple[int, ...], total_penalty: float
    ) -> RankedCutMask:
        flipped_bonds = {sorted_flips[position][1] for position in flip_positions}
        broken_bonds = tuple(sorted(map_broken.symmetric_difference(flipped_bonds)))
        return RankedCutMask(
            broken_bond_indices=broken_bonds,
            log_probability=map_log_probability - total_penalty,
            log_probability_from_map=-total_penalty,
        )

    map_result = build_result((), 0.0)
    if include_empty or map_result.broken_bond_indices:
        yield map_result

    if num_bonds == 0:
        return

    # Each heap entry stores one subset of positions in ``sorted_flips``. From
    # a subset ending at i, its two children either append i+1 or replace i by
    # i+1. This enumerates every nonempty subset exactly once.
    heap: list[tuple[float, tuple[int, ...]]] = [(sorted_flips[0][0], (0,))]
    while heap:
        total_penalty, flip_positions = heapq.heappop(heap)
        result = build_result(flip_positions, total_penalty)
        if include_empty or result.broken_bond_indices:
            yield result

        last_position = flip_positions[-1]
        next_position = last_position + 1
        if next_position >= num_bonds:
            continue

        next_cost = sorted_flips[next_position][0]
        heapq.heappush(
            heap,
            (
                total_penalty + next_cost,
                flip_positions + (next_position,),
            ),
        )
        heapq.heappush(
            heap,
            (
                total_penalty - sorted_flips[last_position][0] + next_cost,
                flip_positions[:-1] + (next_position,),
            ),
        )


def top_k_cut_masks(
    bond_logits: Sequence[float],
    k: int,
    *,
    include_empty: bool = True,
    min_probability: float = 0.0,
    min_relative_probability: float = 0.0,
) -> tuple[RankedCutMask, ...]:
    """Return at most ``k`` masks meeting absolute and MAP-relative floors."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not math.isfinite(min_probability) or not 0.0 <= min_probability <= 1.0:
        raise ValueError(
            f"min_probability must be finite and in [0, 1], got {min_probability}"
        )
    if (
        not math.isfinite(min_relative_probability)
        or not 0.0 <= min_relative_probability <= 1.0
    ):
        raise ValueError(
            "min_relative_probability must be finite and in [0, 1], "
            f"got {min_relative_probability}"
        )

    from fragnnet.frag.mass_shatter_core import top_k_cut_masks_raw

    return tuple(
        RankedCutMask(
            broken_bond_indices=broken_bond_indices,
            log_probability=log_probability,
            log_probability_from_map=log_probability_from_map,
        )
        for broken_bond_indices, log_probability, log_probability_from_map in top_k_cut_masks_raw(
            bond_logits,
            k,
            include_empty,
            min_probability,
            min_relative_probability,
        )
    )


def _connected_component_atom_masks(
    num_atoms: int,
    endpoints: tuple[tuple[int, int], ...],
    broken_bond_indices: Iterable[int],
) -> tuple[int, ...]:
    return _connected_component_atom_masks_many(
        num_atoms,
        endpoints,
        [tuple(broken_bond_indices)],
    )[0]


def _connected_component_atom_masks_many(
    num_atoms: int,
    endpoints: tuple[tuple[int, int], ...],
    broken_bond_sets: list[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    """Translate many masks in one compiled call."""
    from fragnnet.frag.mass_shatter_core import connected_component_atom_masks_many

    endpoint_array = np.asarray(endpoints, dtype=np.int64).reshape(-1, 2)
    return connected_component_atom_masks_many(
        num_atoms,
        endpoint_array,
        broken_bond_sets,
    )


def connected_component_atom_masks(
    num_atoms: int,
    bond_endpoints: Sequence[Sequence[int]],
    broken_bond_indices: Iterable[int],
) -> tuple[int, ...]:
    """Return connected components after removing selected precursor bonds.

    Components are encoded as arbitrary-width Python integer bitmasks, where
    bit ``i`` denotes precursor atom ``i``. This representation is compact,
    hashable, and is not limited to 64 atoms. Bond endpoints use PyG's
    ``(2, num_bonds)`` layout.
    """
    if num_atoms <= 0:
        raise ValueError(f"num_atoms must be positive, got {num_atoms}")
    endpoints = _normalize_bond_endpoints(num_atoms, bond_endpoints)
    return _connected_component_atom_masks(num_atoms, endpoints, broken_bond_indices)


def translate_cut_masks(
    cut_masks: Iterable[RankedCutMask],
    *,
    num_atoms: int,
    bond_endpoints: Sequence[Sequence[int]],
) -> tuple[CutMaskComponents, ...]:
    """Translate ranked cut masks into connected-component atom masks."""
    if num_atoms <= 0:
        raise ValueError(f"num_atoms must be positive, got {num_atoms}")
    endpoints = _normalize_bond_endpoints(num_atoms, bond_endpoints)
    cut_masks = tuple(cut_masks)
    component_masks = _connected_component_atom_masks_many(
        num_atoms,
        endpoints,
        [cut_mask.broken_bond_indices for cut_mask in cut_masks],
    )
    return tuple(
        CutMaskComponents(
            cut_mask=cut_mask,
            component_atom_masks=masks,
        )
        for cut_mask, masks in zip(cut_masks, component_masks)
    )


def aggregate_fragment_marginals(
    cut_masks: Iterable[CutMaskComponents],
    *,
    include_unsplit_precursor: bool = False,
) -> tuple[FragmentMarginal, ...]:
    """Aggregate truncated fragment marginals across evaluated cut masks.

    A cut mask yielding one connected component contributes nothing unless
    ``include_unsplit_precursor`` is true. This excludes ring openings and the
    no-cut state from the default fragment collection.
    """
    log_probabilities: dict[int, float] = {}
    contributing_counts: dict[int, int] = {}

    for mask_components in cut_masks:
        components = mask_components.component_atom_masks
        if len(components) == 1 and not include_unsplit_precursor:
            continue
        for atom_mask in components:
            log_probabilities[atom_mask] = _stable_logaddexp(
                log_probabilities.get(atom_mask, -math.inf),
                mask_components.cut_mask.log_probability,
            )
            contributing_counts[atom_mask] = contributing_counts.get(atom_mask, 0) + 1

    fragments = [
        FragmentMarginal(
            atom_mask=atom_mask,
            log_probability=log_probability,
            contributing_mask_count=contributing_counts[atom_mask],
        )
        for atom_mask, log_probability in log_probabilities.items()
    ]
    fragments.sort(key=lambda fragment: (-fragment.log_probability, fragment.atom_mask))
    return tuple(fragments)


def generate_fragment_marginals(
    bond_logits: Sequence[float],
    *,
    num_atoms: int,
    bond_endpoints: Sequence[Sequence[int]],
    max_masks: int = 500,
    max_unique_fragments: int | None = None,
    max_enumerated_masks: int | None = None,
    include_empty_mask: bool = True,
    include_unsplit_precursor: bool = False,
) -> FragmentGenerationResult:
    """Generate marginals, continuing past masks that yield no new fragments.

    ``max_masks`` is the initial mask budget. If it does not produce
    ``max_unique_fragments`` usable atom subsets, enumeration continues up to
    ``max_enumerated_masks``. By default these limits are ``max_masks`` and ten
    times ``max_masks``, respectively.
    """
    if max_masks <= 0:
        raise ValueError(f"max_masks must be positive, got {max_masks}")
    if max_unique_fragments is None:
        max_unique_fragments = max_masks
    if max_unique_fragments <= 0:
        raise ValueError(
            f"max_unique_fragments must be positive, got {max_unique_fragments}"
        )
    if max_enumerated_masks is None:
        max_enumerated_masks = 10 * max_masks
    if max_enumerated_masks < max_masks:
        raise ValueError("max_enumerated_masks must be at least max_masks")

    if num_atoms <= 0:
        raise ValueError(f"num_atoms must be positive, got {num_atoms}")
    endpoints = _normalize_bond_endpoints(num_atoms, bond_endpoints)
    if len(bond_logits) != len(endpoints):
        raise ValueError(
            "bond_logits and bond_endpoints must describe the same number of bonds"
        )

    # 1. Enumerate exact masks until both useful-output targets or the safety cap.
    translated_masks: list[CutMaskComponents] = []
    unique_fragments: set[int] = set()
    for ranked_mask in iter_ranked_cut_masks(
        bond_logits,
        include_empty=include_empty_mask,
    ):
        components = _connected_component_atom_masks(
            num_atoms,
            endpoints,
            ranked_mask.broken_bond_indices,
        )
        translated_masks.append(
            CutMaskComponents(
                cut_mask=ranked_mask,
                component_atom_masks=components,
            )
        )
        if len(components) > 1 or include_unsplit_precursor:
            unique_fragments.update(components)

        evaluated_count = len(translated_masks)
        if evaluated_count >= max_masks and len(unique_fragments) >= max_unique_fragments:
            break
        if evaluated_count == max_enumerated_masks:
            break

    # 2. Marginalize duplicate atom subsets across all evaluated masks.
    fragments = aggregate_fragment_marginals(
        translated_masks,
        include_unsplit_precursor=include_unsplit_precursor,
    )
    return FragmentGenerationResult(cut_masks=tuple(translated_masks), fragments=fragments)
