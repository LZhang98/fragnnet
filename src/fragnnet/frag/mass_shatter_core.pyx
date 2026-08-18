# cython: boundscheck=False
# cython: cdivision=True
# cython: language_level=3
# cython: nonecheck=False
# cython: profile=False
# cython: wraparound=False
"""Compiled CPU kernels for MASShatter candidate generation.

Each kernel here is the hot-path equivalent of a pure-Python routine in
``bond_mask_generator.py`` / ``mass_shatter_candidates.py``. They exist because
candidate generation runs per molecule per training step, so the Python-level
per-mask overhead (Torch ops, dataclasses, set churn) dominates the actual work.
The kernels therefore take and return plain tuples/lists of ``int`` and let the
callers rewrap them into ``RankedCutMask`` objects.

Shared vocabulary:
    bond logit: Bernoulli logit for "this precursor bond is cut", one per bond.
    cut mask: the set of broken bond indices, stored as a sorted ``tuple[int]``.
    MAP mask: the most probable cut mask, i.e. every bond with a positive logit.
"""

import heapq

import numpy as np

cimport numpy as cnp
from libc.math cimport exp, fabs, isfinite, log, log1p
from libc.stdlib cimport free, malloc
from libc.string cimport memset


# SplitMix64: a self-contained, seedable PRNG so sampling stays reproducible
# without calling back into NumPy/Torch generators inside the attempt loop.
cdef inline unsigned long long _splitmix64(unsigned long long* state) noexcept nogil:
    cdef unsigned long long value
    state[0] += 0x9E3779B97F4A7C15ULL
    value = state[0]
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL
    return value ^ (value >> 31)


# Draw from the *open* interval (0, 1): the Gumbel trick takes log(-log(u)), so
# both u == 0.0 and u == 1.0 would produce infinities. Taking the top 53 bits and
# offsetting by half an ulp keeps the draw strictly inside the interval.
cdef inline double _uniform_open(unsigned long long* state) noexcept nogil:
    return ((<double>(_splitmix64(state) >> 11)) + 0.5) * (1.0 / 9007199254740992.0)


def bounded_additive_masks_raw(
    object proposal_logits,
    object cut_count_weights,
    set unavailable_bond_sets,
    Py_ssize_t target_count,
    Py_ssize_t max_attempts,
    unsigned long long seed,
):
    """Sample unique bounded-cardinality masks without per-attempt Torch ops.

    This is the ``bounded_additive`` exploration proposal. Each attempt draws a
    cut count from the categorical ``cut_count_weights``, then picks exactly that
    many bonds by Gumbel top-k over ``proposal_logits`` (adding Gumbel noise and
    taking the largest scores samples a size-k subset without replacement,
    proportional to the softmax weights). Bounding the cardinality is what keeps
    the proposal on chemically plausible low-cut masks instead of the
    overwhelmingly many high-cut masks that dominate an unconstrained Bernoulli
    draw.

    Attempts that land on an already-seen mask are rejected rather than retried
    in a targeted way, so the caller must budget ``max_attempts`` above
    ``target_count``; the loop returns short if it runs out of attempts.

    Args:
        proposal_logits: One finite per-bond logit, already temperature-scaled by
            the caller. Shape (num_bonds,), coerced to contiguous float64.
        cut_count_weights: Nonnegative unnormalized weights indexed by cut count,
            so entry ``c`` is the weight of drawing a mask with ``c`` broken
            bonds. Shape (num_cut_counts,) with ``num_cut_counts <= num_bonds+1``.
        unavailable_bond_sets: Masks already emitted elsewhere (as sorted index
            tuples). Copied, not mutated; drawn masks are added to the copy so
            results are unique against both the caller's set and each other.
        target_count: How many new unique masks to return.
        max_attempts: Upper bound on sampling attempts, including rejections.
        seed: Seed for the internal SplitMix64 stream.

    Returns:
        A list of at most ``target_count`` sorted ``tuple[int]`` cut masks, none
        of which appeared in ``unavailable_bond_sets``. No probabilities are
        returned: these are exploration proposals, not ranked masks.

    Raises:
        ValueError: If the counts are negative, the arrays are not 1-D, more cut
            counts than bonds are requested, a logit is non-finite, or a weight
            is non-finite or negative.
        MemoryError: If the work buffers cannot be allocated.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] logits_array
    cdef cnp.ndarray[cnp.float64_t, ndim=1] weights_array
    cdef double[::1] logits
    cdef double[::1] weights
    cdef Py_ssize_t num_bonds, num_weights, attempt, index, cut_count
    cdef Py_ssize_t selected_count, best_index
    cdef double total_weight = 0.0
    cdef double draw, cumulative_weight, score, best_score
    cdef unsigned char* selected = NULL
    cdef double* scores = NULL
    cdef unsigned long long state = seed
    cdef tuple broken_bond_indices
    cdef list broken_bonds, results = []
    cdef set unavailable = set(unavailable_bond_sets)

    if target_count < 0:
        raise ValueError(f"target_count must be nonnegative, got {target_count}")
    if max_attempts < 0:
        raise ValueError(f"max_attempts must be nonnegative, got {max_attempts}")
    logits_array = np.ascontiguousarray(proposal_logits, dtype=np.float64)
    weights_array = np.ascontiguousarray(cut_count_weights, dtype=np.float64)
    if logits_array.ndim != 1 or weights_array.ndim != 1:
        raise ValueError("proposal_logits and cut_count_weights must be one-dimensional")
    logits = logits_array
    weights = weights_array
    num_bonds = logits.shape[0]
    num_weights = weights.shape[0]
    if num_weights > num_bonds + 1:
        raise ValueError("cut_count_weights cannot request more cuts than bonds")
    for index in range(num_bonds):
        if not isfinite(logits[index]):
            raise ValueError("proposal_logits must contain only finite values")
    for index in range(num_weights):
        if not isfinite(weights[index]) or weights[index] < 0.0:
            raise ValueError("cut_count_weights must be finite and nonnegative")
        total_weight += weights[index]
    # Nothing to sample from: an empty budget, a bondless molecule, or a
    # degenerate all-zero cardinality distribution.
    if target_count == 0 or max_attempts == 0 or num_bonds == 0 or total_weight == 0.0:
        return results

    # Reused across attempts so the loop performs no allocation.
    selected = <unsigned char*>malloc(num_bonds * sizeof(unsigned char))
    scores = <double*>malloc(num_bonds * sizeof(double))
    if selected == NULL or scores == NULL:
        if selected != NULL:
            free(selected)
        if scores != NULL:
            free(scores)
        raise MemoryError()
    try:
        for attempt in range(max_attempts):
            if len(results) == target_count:
                break

            # Inverse-CDF draw of the cut count over the (unnormalized) weights.
            # The pre-loop default guards against the cumulative sum falling a
            # rounding error short of ``draw`` on the final entry.
            draw = _uniform_open(&state) * total_weight
            cumulative_weight = 0.0
            cut_count = num_weights - 1
            for index in range(num_weights):
                cumulative_weight += weights[index]
                if draw < cumulative_weight:
                    cut_count = index
                    break

            # Gumbel perturbation: -log(-log(u)) is a standard Gumbel sample, so
            # the top ``cut_count`` scores are a softmax sample without
            # replacement (the Gumbel top-k trick).
            memset(selected, 0, num_bonds * sizeof(unsigned char))
            for index in range(num_bonds):
                draw = _uniform_open(&state)
                scores[index] = logits[index] - log(-log(draw))
            # Selection sort rather than a heap: ``cut_count`` is small (a
            # handful of cuts), so k linear scans beat the heap bookkeeping.
            selected_count = 0
            while selected_count < cut_count:
                best_index = -1
                best_score = -float("inf")
                for index in range(num_bonds):
                    if selected[index]:
                        continue
                    score = scores[index]
                    if score > best_score:
                        best_score = score
                        best_index = index
                selected[best_index] = 1
                selected_count += 1

            # Ascending scan, so the tuple is sorted by construction and hashes
            # consistently against the caller's already-sorted mask tuples.
            broken_bonds = []
            for index in range(num_bonds):
                if selected[index]:
                    broken_bonds.append(index)
            broken_bond_indices = tuple(broken_bonds)
            if broken_bond_indices in unavailable:
                continue  # Duplicate draw: spend the attempt, keep nothing.
            unavailable.add(broken_bond_indices)
            results.append(broken_bond_indices)
    finally:
        free(selected)
        free(scores)
    return results


def top_k_cut_masks_raw(
    object bond_logits,
    Py_ssize_t k,
    bint include_empty=True,
    double min_probability=0.0,
):
    """Return raw exact top-k Bernoulli masks without Python dataclass overhead.

    Bonds are cut independently, so the most probable mask (the MAP mask) breaks
    every bond with a positive logit. Any other mask is the MAP mask with some
    subset of bond decisions flipped, and flipping bond ``b`` costs a fixed
    ``abs(bond_logits[b])`` of log-probability. Ranking masks by probability is
    therefore ranking flip subsets by their summed cost, and a heap walks the
    smallest subset sums in order without ever materializing all ``2**num_bonds``
    masks.

    The heap holds subsets of *positions* into the cost-sorted flip list. A
    subset ending at position ``i`` has exactly two children — append ``i+1``, or
    replace ``i`` with ``i+1`` — which reaches every nonempty subset exactly once.

    Args:
        bond_logits: One finite Bernoulli logit per precursor bond. Shape
            (num_bonds,), coerced to contiguous float64.
        k: Maximum number of masks to return. Must be positive.
        include_empty: Whether a mask that breaks no bonds may be returned.
            Defaults to True.
        min_probability: Stop once masks fall below this probability. Defaults to
            0.0, meaning no probability floor.

    Returns:
        Up to ``k`` tuples ``(broken_bond_indices, log_probability,
        log_probability_from_map)`` in exact descending probability order, where
        the third entry is the log-probability relative to the MAP mask (0.0 for
        the MAP mask itself, negative otherwise). Ties break deterministically
        but carry no chemical meaning. The caller wraps these into
        ``RankedCutMask``.

    Raises:
        ValueError: If ``k`` is not positive, ``min_probability`` is non-finite
            or outside [0, 1], ``bond_logits`` is not 1-D, or a logit is
            non-finite.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] logits_array
    cdef double[::1] logits
    cdef Py_ssize_t num_bonds, index, position, last_position, next_position
    cdef double logit, map_log_probability, total_penalty, next_cost
    cdef double min_log_probability
    cdef list sorted_flips, heap, results
    cdef tuple flip_positions, next_positions, broken_bonds
    cdef set map_broken, flipped_bonds

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not isfinite(min_probability) or min_probability < 0.0 or min_probability > 1.0:
        raise ValueError(
            f"min_probability must be finite and in [0, 1], got {min_probability}"
        )
    min_log_probability = log(min_probability) if min_probability > 0.0 else -float("inf")

    logits_array = np.ascontiguousarray(bond_logits, dtype=np.float64)
    if logits_array.ndim != 1:
        raise ValueError("bond_logits must be one-dimensional")
    logits = logits_array
    num_bonds = logits.shape[0]
    map_broken = set()
    sorted_flips = []
    map_log_probability = 0.0

    for index in range(num_bonds):
        logit = logits[index]
        if not isfinite(logit):
            raise ValueError("bond_logits must contain only finite values")
        if logit > 0.0:
            map_broken.add(index)
        # log(sigmoid(|logit|)) written as -log1p(exp(-|logit|)): the MAP mask
        # always takes the more probable side of each bond, and the ``fabs``
        # form is the numerically stable branch for either sign.
        map_log_probability -= log1p(exp(-fabs(logit)))
        # Flipping this bond off the MAP choice costs |logit| in log-probability.
        sorted_flips.append((fabs(logit), index))
    sorted_flips.sort()

    results = []
    broken_bonds = tuple(sorted(map_broken))
    # Nothing can beat the MAP mask, so if it misses the floor, nothing passes.
    if map_log_probability < min_log_probability:
        return results
    if include_empty or broken_bonds:
        results.append((broken_bonds, map_log_probability, 0.0))
        if len(results) == k:
            return results
    if num_bonds == 0:
        return results

    # Seed with the single cheapest flip; every other subset descends from it.
    heap = [(sorted_flips[0][0], (0,))]
    while heap and len(results) < k:
        total_penalty, flip_positions = heapq.heappop(heap)
        # Penalties pop in ascending order, so the first mask under the floor
        # means every remaining mask is too, and we can stop rather than skip.
        if map_log_probability - total_penalty < min_log_probability:
            break
        # Flipping a bond toggles its MAP decision, hence symmetric difference.
        flipped_bonds = set()
        for position in flip_positions:
            flipped_bonds.add(sorted_flips[position][1])
        broken_bonds = tuple(sorted(map_broken.symmetric_difference(flipped_bonds)))
        if include_empty or broken_bonds:
            results.append(
                (
                    broken_bonds,
                    map_log_probability - total_penalty,
                    -total_penalty,
                )
            )

        last_position = flip_positions[len(flip_positions) - 1]
        next_position = last_position + 1
        if next_position >= num_bonds:
            continue  # This branch is exhausted; neither child exists.
        next_cost = sorted_flips[next_position][0]
        # Child 1: add one more flip on top of this subset.
        heapq.heappush(
            heap,
            (
                total_penalty + next_cost,
                flip_positions + (next_position,),
            ),
        )
        # Child 2: swap the last flip for the next-cheapest one, keeping the
        # subset the same size. Both children cost at least as much as the
        # parent, which is what makes the pop order exact.
        next_positions = flip_positions[: len(flip_positions) - 1] + (next_position,)
        heapq.heappush(
            heap,
            (
                total_penalty - sorted_flips[last_position][0] + next_cost,
                next_positions,
            ),
        )
    return results


# Union-find lookup with path halving. No union-by-rank: molecules have tens of
# atoms, so path compression alone keeps this effectively flat.
cdef inline Py_ssize_t _find_root(Py_ssize_t* parents, Py_ssize_t atom) noexcept nogil:
    while parents[atom] != atom:
        parents[atom] = parents[parents[atom]]
        atom = parents[atom]
    return atom


def connected_component_atom_masks_many(
    Py_ssize_t num_atoms,
    object bond_endpoints,
    list broken_bond_sets,
):
    """Translate many cut masks for one molecule with reusable C work buffers.

    Removing the broken bonds splits the precursor into connected components —
    the fragments the mask produces. Each component is returned as a Python
    integer bitmask with bit ``i`` set when atom ``i`` belongs to it: compact,
    hashable, and (unlike a fixed-width integer) not capped at 64 atoms.

    All masks for a molecule are translated in one call because the union-find
    buffers depend only on the molecule, so they are allocated once and reset
    per mask instead of per call.

    Args:
        num_atoms: Number of precursor atoms. Must be positive.
        bond_endpoints: Atom index pairs of shape (num_bonds, 2), coerced to
            contiguous int64. Note this is the transpose of PyG's ``(2, E)``
            layout; the caller reshapes before passing.
        broken_bond_sets: One iterable of broken bond indices per mask.

    Returns:
        One entry per input mask, each a tuple of per-component atom bitmasks
        sorted ascending so equal component sets compare equal.

    Raises:
        ValueError: If ``num_atoms`` is not positive, ``bond_endpoints`` has the
            wrong shape, a broken bond index is out of range, or an endpoint
            falls outside ``[0, num_atoms)``.
        MemoryError: If the work buffers cannot be allocated.
    """
    cdef cnp.ndarray[cnp.int64_t, ndim=2] endpoints_array
    cdef cnp.int64_t[:, ::1] endpoints
    cdef Py_ssize_t num_bonds, mask_index, atom, bond, left, right
    cdef Py_ssize_t left_root, right_root, root
    cdef Py_ssize_t* parents = NULL
    cdef unsigned char* broken = NULL
    cdef object atom_bit, broken_bond_set, component_by_root
    cdef list results

    if num_atoms <= 0:
        raise ValueError(f"num_atoms must be positive, got {num_atoms}")
    endpoints_array = np.ascontiguousarray(bond_endpoints, dtype=np.int64)
    if endpoints_array.ndim != 2 or endpoints_array.shape[1] != 2:
        raise ValueError("bond_endpoints must have shape (num_bonds, 2)")
    endpoints = endpoints_array
    num_bonds = endpoints.shape[0]

    # Allocated once for the whole batch of masks; ``max(num_bonds, 1)`` avoids a
    # zero-size malloc for a single-atom precursor with no bonds.
    parents = <Py_ssize_t*>malloc(num_atoms * sizeof(Py_ssize_t))
    broken = <unsigned char*>malloc(max(num_bonds, 1) * sizeof(unsigned char))
    if parents == NULL or broken == NULL:
        if parents != NULL:
            free(parents)
        if broken != NULL:
            free(broken)
        raise MemoryError("unable to allocate MASShatter topology work buffers")

    results = []
    try:
        for broken_bond_set in broken_bond_sets:
            # Reset to the all-singletons forest for this mask.
            for atom in range(num_atoms):
                parents[atom] = atom
            memset(broken, 0, num_bonds * sizeof(unsigned char))
            for bond_value in broken_bond_set:
                bond = int(bond_value)
                if bond < 0 or bond >= num_bonds:
                    raise ValueError("broken_bond_indices contains an invalid bond index")
                broken[bond] = 1

            # Union across every surviving bond; what remains disconnected are
            # the fragments.
            for bond in range(num_bonds):
                left = endpoints[bond, 0]
                right = endpoints[bond, 1]
                if left < 0 or right < 0 or left >= num_atoms or right >= num_atoms:
                    raise ValueError(
                        "bond_endpoints contains an atom index outside [0, num_atoms)"
                    )
                if broken[bond]:
                    continue
                left_root = _find_root(parents, left)
                right_root = _find_root(parents, right)
                if left_root != right_root:
                    parents[right_root] = left_root

            # Accumulate each component's atoms into one arbitrary-width Python
            # int, keyed by the component's union-find root.
            component_by_root = {}
            for atom in range(num_atoms):
                root = _find_root(parents, atom)
                atom_bit = 1 << int(atom)
                component_by_root[root] = component_by_root.get(root, 0) | atom_bit
            results.append(tuple(sorted(component_by_root.values())))
    finally:
        free(parents)
        free(broken)
    return results
