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
Compatibility entrypoints return tuples/lists, while hot-path entrypoints keep
cut masks and candidate memberships in numeric CSR/NumPy arrays.

Shared vocabulary:
    bond logit: Bernoulli logit for "this precursor bond is cut", one per bond.
    cut mask: the set of broken bond indices, stored as a sorted ``tuple[int]``.
    MAP mask: the most probable cut mask, i.e. every bond with a positive logit.
"""

from time import perf_counter

import numpy as np

cimport numpy as cnp
from libc.math cimport exp, fabs, isfinite, log, log1p
from libc.stdint cimport uint64_t
from libc.stdlib cimport calloc, free, malloc
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


cdef inline bint _top_k_node_less(
    Py_ssize_t left_node,
    Py_ssize_t right_node,
    double[::1] node_penalties,
    cnp.int64_t[::1] node_lengths,
    cnp.int64_t[:, ::1] node_positions,
) noexcept nogil:
    """Compare two numeric nodes by the exact legacy Python heap key."""
    cdef Py_ssize_t position
    cdef Py_ssize_t left_length = node_lengths[left_node]
    cdef Py_ssize_t right_length = node_lengths[right_node]
    cdef Py_ssize_t shared_length = min(left_length, right_length)

    if node_penalties[left_node] < node_penalties[right_node]:
        return True
    if node_penalties[left_node] > node_penalties[right_node]:
        return False
    for position in range(shared_length):
        if node_positions[left_node, position] < node_positions[right_node, position]:
            return True
        if node_positions[left_node, position] > node_positions[right_node, position]:
            return False
    return left_length < right_length


cdef inline void _top_k_heap_push(
    Py_ssize_t node,
    cnp.int64_t[::1] heap,
    Py_ssize_t* heap_size,
    double[::1] node_penalties,
    cnp.int64_t[::1] node_lengths,
    cnp.int64_t[:, ::1] node_positions,
) noexcept nogil:
    cdef Py_ssize_t index = heap_size[0]
    cdef Py_ssize_t parent

    heap_size[0] += 1
    while index > 0:
        parent = (index - 1) // 2
        if not _top_k_node_less(
            node,
            heap[parent],
            node_penalties,
            node_lengths,
            node_positions,
        ):
            break
        heap[index] = heap[parent]
        index = parent
    heap[index] = node


cdef inline Py_ssize_t _top_k_heap_pop(
    cnp.int64_t[::1] heap,
    Py_ssize_t* heap_size,
    double[::1] node_penalties,
    cnp.int64_t[::1] node_lengths,
    cnp.int64_t[:, ::1] node_positions,
) noexcept nogil:
    cdef Py_ssize_t result = heap[0]
    cdef Py_ssize_t last, index, left, right, child

    heap_size[0] -= 1
    if heap_size[0] == 0:
        return result
    last = heap[heap_size[0]]
    index = 0
    while True:
        left = 2 * index + 1
        if left >= heap_size[0]:
            break
        right = left + 1
        child = left
        if right < heap_size[0] and _top_k_node_less(
            heap[right],
            heap[left],
            node_penalties,
            node_lengths,
            node_positions,
        ):
            child = right
        if not _top_k_node_less(
            heap[child],
            last,
            node_penalties,
            node_lengths,
            node_positions,
        ):
            break
        heap[index] = heap[child]
        index = child
    heap[index] = last
    return result


def top_k_cut_masks_csr_raw(
    object bond_logits,
    Py_ssize_t k,
    bint include_empty=True,
    double min_probability=0.0,
):
    """Return exact top-k Bernoulli masks as numeric CSR arrays.

    The subset-sum heap stays entirely in typed numeric buffers. Its comparison
    key is exactly the legacy Python ``(penalty, flip_positions)`` tuple: first
    the double penalty, then the lexicographic sequence of positions into the
    cost-sorted flip list.

    Returns:
        ``(mask_ptr, broken_bond_indices, log_probability,
        log_probability_from_map)``. The first two arrays are int64 CSR arrays;
        the probability arrays are float64 and retain legacy ordering and
        arithmetic.
    """
    cdef cnp.ndarray[cnp.float64_t, ndim=1] logits_array
    cdef double[::1] logits
    cdef cnp.ndarray[cnp.float64_t, ndim=1] flip_costs_array
    cdef double[::1] flip_costs
    cdef cnp.ndarray[cnp.int64_t, ndim=1] flip_bonds_array
    cdef cnp.int64_t[::1] flip_bonds
    cdef cnp.ndarray[cnp.uint64_t, ndim=1] map_words_array
    cdef cnp.uint64_t[::1] map_words
    cdef cnp.ndarray[cnp.float64_t, ndim=1] node_penalties_array
    cdef double[::1] node_penalties
    cdef cnp.ndarray[cnp.int64_t, ndim=1] node_lengths_array
    cdef cnp.int64_t[::1] node_lengths
    cdef cnp.ndarray[cnp.int64_t, ndim=2] node_positions_array
    cdef cnp.int64_t[:, ::1] node_positions
    cdef cnp.ndarray[cnp.int64_t, ndim=1] heap_array
    cdef cnp.int64_t[::1] heap
    cdef cnp.ndarray[cnp.uint64_t, ndim=2] output_words_array
    cdef cnp.uint64_t[:, ::1] output_words
    cdef cnp.ndarray[cnp.int64_t, ndim=1] output_cut_counts_array
    cdef cnp.int64_t[::1] output_cut_counts
    cdef cnp.ndarray[cnp.float64_t, ndim=1] output_log_probabilities_array
    cdef double[::1] output_log_probabilities
    cdef cnp.ndarray[cnp.float64_t, ndim=1] output_relative_probabilities_array
    cdef double[::1] output_relative_probabilities
    cdef cnp.ndarray[cnp.int64_t, ndim=1] mask_ptr_array
    cdef cnp.int64_t[::1] mask_ptr
    cdef cnp.ndarray[cnp.int64_t, ndim=1] broken_bonds_array
    cdef cnp.int64_t[::1] broken_bonds
    cdef cnp.ndarray[cnp.float64_t, ndim=1] log_probabilities_array
    cdef double[::1] log_probabilities
    cdef cnp.ndarray[cnp.float64_t, ndim=1] relative_probabilities_array
    cdef double[::1] relative_probabilities
    cdef Py_ssize_t num_bonds, word_count, map_cut_count = 0, search_capacity
    cdef Py_ssize_t index, insertion, position, word, bond
    cdef Py_ssize_t node, node_count, node_length, last_position, next_position
    cdef Py_ssize_t append_node, replacement_node, heap_size = 0
    cdef Py_ssize_t output_count = 0, cut_count, total_broken = 0, cursor = 0
    cdef double logit, cost, map_log_probability = 0.0
    cdef double total_penalty, next_cost, candidate_log_probability
    cdef double replacement_penalty, min_log_probability
    cdef object node_capacity_object

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
    word_count = (num_bonds + 63) // 64
    flip_costs_array = np.empty(num_bonds, dtype=np.float64)
    flip_bonds_array = np.empty(num_bonds, dtype=np.int64)
    map_words_array = np.zeros(word_count, dtype=np.uint64)
    flip_costs = flip_costs_array
    flip_bonds = flip_bonds_array
    map_words = map_words_array

    # 1. Build the MAP mask and flip costs in original bond order so floating
    # accumulation remains bit-for-bit compatible with the legacy kernel.
    for index in range(num_bonds):
        logit = logits[index]
        if not isfinite(logit):
            raise ValueError("bond_logits must contain only finite values")
        if logit > 0.0:
            map_words[index // 64] |= 1ULL << (index % 64)
            map_cut_count += 1
        map_log_probability -= log1p(exp(-fabs(logit)))
        flip_costs[index] = fabs(logit)
        flip_bonds[index] = index

    # 2. Sort by the exact legacy tuple key ``(cost, bond)``. Molecules have
    # only tens of bonds, so typed insertion sort avoids a C callback here.
    for index in range(1, num_bonds):
        cost = flip_costs[index]
        bond = flip_bonds[index]
        insertion = index
        while insertion > 0 and (
            cost < flip_costs[insertion - 1]
            or (
                cost == flip_costs[insertion - 1]
                and bond < flip_bonds[insertion - 1]
            )
        ):
            flip_costs[insertion] = flip_costs[insertion - 1]
            flip_bonds[insertion] = flip_bonds[insertion - 1]
            insertion -= 1
        flip_costs[insertion] = cost
        flip_bonds[insertion] = bond

    # Nothing can beat the MAP mask, so if it misses the floor, nothing passes.
    if map_log_probability < min_log_probability:
        return (
            np.zeros(1, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    if num_bonds == 0:
        if include_empty:
            return (
                np.zeros(2, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.asarray([map_log_probability], dtype=np.float64),
                np.asarray([0.0], dtype=np.float64),
            )
        return (
            np.zeros(1, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    # Cap work buffers by the mask state space without ever evaluating an
    # overflowing ``2**num_bonds``. This preserves cheap large-k calls on small
    # molecules, where the heap naturally exhausts before reaching k.
    search_capacity = 1
    for index in range(num_bonds):
        if search_capacity >= k:
            search_capacity = k
            break
        if search_capacity > k // 2:
            search_capacity = k
            break
        search_capacity *= 2
    output_words_array = np.zeros((search_capacity, word_count), dtype=np.uint64)
    output_cut_counts_array = np.empty(search_capacity, dtype=np.int64)
    output_log_probabilities_array = np.empty(search_capacity, dtype=np.float64)
    output_relative_probabilities_array = np.empty(search_capacity, dtype=np.float64)
    output_words = output_words_array
    output_cut_counts = output_cut_counts_array
    output_log_probabilities = output_log_probabilities_array
    output_relative_probabilities = output_relative_probabilities_array

    if include_empty or map_cut_count > 0:
        for word in range(word_count):
            output_words[0, word] = map_words[word]
        output_cut_counts[0] = map_cut_count
        output_log_probabilities[0] = map_log_probability
        output_relative_probabilities[0] = 0.0
        output_count = 1

    if output_count < k:
        # Every pop emits a mask except, at most, the single excluded empty mask.
        # Therefore no more than k nodes pop and no more than 2*k+1 are created.
        node_capacity_object = search_capacity
        node_capacity_object = 2 * node_capacity_object + 1
        node_penalties_array = np.empty(node_capacity_object, dtype=np.float64)
        node_lengths_array = np.empty(node_capacity_object, dtype=np.int64)
        node_positions_array = np.empty(
            (node_capacity_object, num_bonds),
            dtype=np.int64,
        )
        heap_array = np.empty(node_capacity_object // 2 + 1, dtype=np.int64)
        node_penalties = node_penalties_array
        node_lengths = node_lengths_array
        node_positions = node_positions_array
        heap = heap_array

        # 3. Seed with the cheapest flip and enumerate nonempty flip subsets.
        node_penalties[0] = flip_costs[0]
        node_lengths[0] = 1
        node_positions[0, 0] = 0
        node_count = 1
        _top_k_heap_push(
            0,
            heap,
            &heap_size,
            node_penalties,
            node_lengths,
            node_positions,
        )

        while heap_size > 0 and output_count < k:
            node = _top_k_heap_pop(
                heap,
                &heap_size,
                node_penalties,
                node_lengths,
                node_positions,
            )
            total_penalty = node_penalties[node]
            candidate_log_probability = map_log_probability - total_penalty
            if candidate_log_probability < min_log_probability:
                break

            node_length = node_lengths[node]
            cut_count = map_cut_count
            for word in range(word_count):
                output_words[output_count, word] = map_words[word]
            for position in range(node_length):
                bond = flip_bonds[node_positions[node, position]]
                output_words[output_count, bond // 64] ^= 1ULL << (bond % 64)
                if map_words[bond // 64] & (1ULL << (bond % 64)):
                    cut_count -= 1
                else:
                    cut_count += 1
            if include_empty or cut_count > 0:
                output_cut_counts[output_count] = cut_count
                output_log_probabilities[output_count] = candidate_log_probability
                output_relative_probabilities[output_count] = -total_penalty
                output_count += 1
                if output_count == k:
                    break

            last_position = node_positions[node, node_length - 1]
            next_position = last_position + 1
            if next_position >= num_bonds:
                continue
            next_cost = flip_costs[next_position]

            # Append child: ``positions + (next_position,)``.
            append_node = node_count
            node_count += 1
            node_penalties[append_node] = total_penalty + next_cost
            node_lengths[append_node] = node_length + 1
            for position in range(node_length):
                node_positions[append_node, position] = node_positions[node, position]
            node_positions[append_node, node_length] = next_position
            _top_k_heap_push(
                append_node,
                heap,
                &heap_size,
                node_penalties,
                node_lengths,
                node_positions,
            )

            # Replacement child: preserve ``(penalty - old_cost) + next_cost``
            # as two left-associated operations so rounding and ties stay exact.
            replacement_node = node_count
            node_count += 1
            replacement_penalty = total_penalty - flip_costs[last_position]
            replacement_penalty = replacement_penalty + next_cost
            node_penalties[replacement_node] = replacement_penalty
            node_lengths[replacement_node] = node_length
            for position in range(node_length - 1):
                node_positions[replacement_node, position] = node_positions[node, position]
            node_positions[replacement_node, node_length - 1] = next_position
            _top_k_heap_push(
                replacement_node,
                heap,
                &heap_size,
                node_penalties,
                node_lengths,
                node_positions,
            )

    # 4. Compact retained bitsets into exact-size numeric CSR arrays.
    for index in range(output_count):
        total_broken += output_cut_counts[index]
    mask_ptr_array = np.empty(output_count + 1, dtype=np.int64)
    broken_bonds_array = np.empty(total_broken, dtype=np.int64)
    log_probabilities_array = np.empty(output_count, dtype=np.float64)
    relative_probabilities_array = np.empty(output_count, dtype=np.float64)
    mask_ptr = mask_ptr_array
    broken_bonds = broken_bonds_array
    log_probabilities = log_probabilities_array
    relative_probabilities = relative_probabilities_array
    mask_ptr[0] = 0
    for index in range(output_count):
        for bond in range(num_bonds):
            if output_words[index, bond // 64] & (1ULL << (bond % 64)):
                broken_bonds[cursor] = bond
                cursor += 1
        mask_ptr[index + 1] = cursor
        log_probabilities[index] = output_log_probabilities[index]
        relative_probabilities[index] = output_relative_probabilities[index]
    return (
        mask_ptr_array,
        broken_bonds_array,
        log_probabilities_array,
        relative_probabilities_array,
    )


def top_k_cut_masks_raw(
    object bond_logits,
    Py_ssize_t k,
    bint include_empty=True,
    double min_probability=0.0,
):
    """Return legacy tuples materialized from the numeric top-k kernel."""
    cdef object mask_ptr, broken_bonds, log_probabilities, relative_probabilities
    cdef Py_ssize_t mask, start, end
    cdef list results = []

    (
        mask_ptr,
        broken_bonds,
        log_probabilities,
        relative_probabilities,
    ) = top_k_cut_masks_csr_raw(
        bond_logits,
        k,
        include_empty,
        min_probability,
    )
    for mask in range(log_probabilities.shape[0]):
        start = mask_ptr[mask]
        end = mask_ptr[mask + 1]
        results.append(
            (
                tuple(broken_bonds[start:end].tolist()),
                float(log_probabilities[mask]),
                float(relative_probabilities[mask]),
            )
        )
    return results


# Union-find lookup with path halving. No union-by-rank: molecules have tens of
# atoms, so path compression alone keeps this effectively flat.
cdef inline Py_ssize_t _find_root(Py_ssize_t* parents, Py_ssize_t atom) noexcept nogil:
    while parents[atom] != atom:
        parents[atom] = parents[parents[atom]]
        atom = parents[atom]
    return atom


def build_component_arrays_raw(
    Py_ssize_t num_atoms,
    object bond_endpoints,
    object broken_bond_sets,
    object global_bond_indices,
    Py_ssize_t atom_offset,
    Py_ssize_t mask_offset,
    Py_ssize_t component_offset,
):
    """Build topology and flat candidate arrays without Python component objects.

    Components retain the legacy arbitrary-width-integer ordering. Because the
    components of one mask are disjoint, ascending integer bitmask order is
    exactly ascending maximum-member-atom order. Cut masks may be the legacy
    list of index tuples or a numeric ``(mask_ptr, broken_bonds)`` CSR tuple.
    """
    cdef cnp.ndarray[cnp.int64_t, ndim=2] endpoints_array
    cdef cnp.int64_t[:, ::1] endpoints
    cdef cnp.ndarray[cnp.int64_t, ndim=1] global_bonds_array
    cdef cnp.int64_t[::1] global_bonds
    cdef cnp.ndarray[cnp.int64_t, ndim=1] mask_ptr_array
    cdef cnp.int64_t[::1] mask_ptr
    cdef cnp.ndarray[cnp.int64_t, ndim=1] local_broken_array
    cdef cnp.int64_t[::1] local_broken
    cdef cnp.ndarray[cnp.int64_t, ndim=1] cut_counts_array
    cdef cnp.int64_t[::1] cut_counts
    cdef cnp.ndarray[cnp.int64_t, ndim=1] labels_array
    cdef cnp.int64_t[::1] labels
    cdef cnp.ndarray[cnp.int64_t, ndim=1] component_offsets_array
    cdef cnp.int64_t[::1] component_offsets
    cdef cnp.ndarray[cnp.int64_t, ndim=1] component_sizes_array
    cdef cnp.int64_t[::1] component_sizes
    cdef cnp.ndarray[cnp.int64_t, ndim=1] component_mask_indices_array
    cdef cnp.int64_t[::1] component_mask_indices
    cdef cnp.ndarray[cnp.int64_t, ndim=1] component_cut_counts_array
    cdef cnp.int64_t[::1] component_cut_counts
    cdef cnp.ndarray[cnp.int64_t, ndim=1] atom_indices_array
    cdef cnp.int64_t[::1] atom_indices
    cdef cnp.ndarray[cnp.int64_t, ndim=1] atom_component_indices_array
    cdef cnp.int64_t[::1] atom_component_indices
    cdef cnp.ndarray[cnp.int64_t, ndim=1] mask_indices_array
    cdef cnp.int64_t[::1] mask_indices
    cdef cnp.ndarray[cnp.int64_t, ndim=1] broken_bond_indices_array
    cdef cnp.int64_t[::1] broken_bond_indices
    cdef cnp.ndarray[cnp.int64_t, ndim=1] boundary_bond_indices_array
    cdef cnp.int64_t[::1] boundary_bond_indices
    cdef cnp.ndarray[cnp.int64_t, ndim=1] boundary_component_indices_array
    cdef cnp.int64_t[::1] boundary_component_indices
    cdef Py_ssize_t num_masks, num_bonds, total_broken, max_components
    cdef Py_ssize_t mask, atom, bond, item, cursor, component_cursor
    cdef Py_ssize_t component, component_count, root, left, right
    cdef Py_ssize_t left_root, right_root, left_component, right_component
    cdef Py_ssize_t membership_cursor, boundary_cursor, table_size, slot
    cdef Py_ssize_t existing_component, unique_component_count, word, words
    cdef Py_ssize_t* parents = NULL
    cdef Py_ssize_t* broken_epoch = NULL
    cdef Py_ssize_t* atom_roots = NULL
    cdef Py_ssize_t* component_max = NULL
    cdef Py_ssize_t* root_by_max = NULL
    cdef Py_ssize_t* root_to_component = NULL
    cdef Py_ssize_t* membership_offsets = NULL
    cdef Py_ssize_t* membership_cursors = NULL
    cdef Py_ssize_t* hash_slots = NULL
    cdef uint64_t* signatures = NULL
    cdef uint64_t signature_hash
    cdef bint signatures_equal
    cdef object broken_bond_set, bond_value
    cdef double topology_start, topology_ms, assembly_start, assembly_ms

    topology_start = perf_counter()
    if num_atoms <= 0:
        raise ValueError(f"num_atoms must be positive, got {num_atoms}")
    endpoints_array = np.ascontiguousarray(bond_endpoints, dtype=np.int64)
    if endpoints_array.ndim != 2 or endpoints_array.shape[1] != 2:
        raise ValueError("bond_endpoints must have shape (num_bonds, 2)")
    endpoints = endpoints_array
    num_bonds = endpoints.shape[0]
    global_bonds_array = np.ascontiguousarray(global_bond_indices, dtype=np.int64)
    if global_bonds_array.ndim != 1 or global_bonds_array.shape[0] != num_bonds:
        raise ValueError("global_bond_indices must contain one entry per bond")
    global_bonds = global_bonds_array
    for bond in range(num_bonds):
        left = endpoints[bond, 0]
        right = endpoints[bond, 1]
        if left < 0 or right < 0 or left >= num_atoms or right >= num_atoms:
            raise ValueError("bond_endpoints contains an atom index outside [0, num_atoms)")

    if isinstance(broken_bond_sets, tuple):
        if len(broken_bond_sets) != 2:
            raise ValueError("CSR cut masks must be a (mask_ptr, broken_bonds) tuple")
        mask_ptr_array = np.ascontiguousarray(broken_bond_sets[0], dtype=np.int64)
        local_broken_array = np.ascontiguousarray(
            broken_bond_sets[1],
            dtype=np.int64,
        )
        if mask_ptr_array.ndim != 1 or local_broken_array.ndim != 1:
            raise ValueError("CSR mask_ptr and broken_bonds must be one-dimensional")
        mask_ptr = mask_ptr_array
        local_broken = local_broken_array
        if mask_ptr.shape[0] == 0 or mask_ptr[0] != 0:
            raise ValueError("CSR mask_ptr must start at zero")
        num_masks = mask_ptr.shape[0] - 1
        if num_masks <= 0:
            raise ValueError("broken_bond_sets must contain at least one mask")
        total_broken = local_broken.shape[0]
        for mask in range(num_masks):
            if mask_ptr[mask + 1] < mask_ptr[mask]:
                raise ValueError("CSR mask_ptr must be monotonically nondecreasing")
        if mask_ptr[num_masks] != total_broken:
            raise ValueError("CSR mask_ptr must end at the broken_bonds length")
        for item in range(total_broken):
            bond = local_broken[item]
            if bond < 0 or bond >= num_bonds:
                raise ValueError("broken_bond_indices contains an invalid bond index")
        cut_counts_array = np.empty(num_masks, dtype=np.int64)
        cut_counts = cut_counts_array
        for mask in range(num_masks):
            cut_counts[mask] = mask_ptr[mask + 1] - mask_ptr[mask]
    elif isinstance(broken_bond_sets, list):
        num_masks = len(broken_bond_sets)
        if num_masks <= 0:
            raise ValueError("broken_bond_sets must contain at least one mask")
        total_broken = 0
        for broken_bond_set in broken_bond_sets:
            total_broken += len(broken_bond_set)
        mask_ptr_array = np.empty(num_masks + 1, dtype=np.int64)
        cut_counts_array = np.empty(num_masks, dtype=np.int64)
        local_broken_array = np.empty(total_broken, dtype=np.int64)
        mask_ptr = mask_ptr_array
        cut_counts = cut_counts_array
        local_broken = local_broken_array
        cursor = 0
        mask_ptr[0] = 0
        for mask in range(num_masks):
            broken_bond_set = broken_bond_sets[mask]
            cut_counts[mask] = len(broken_bond_set)
            for bond_value in broken_bond_set:
                bond = int(bond_value)
                if bond < 0 or bond >= num_bonds:
                    raise ValueError("broken_bond_indices contains an invalid bond index")
                local_broken[cursor] = bond
                cursor += 1
            mask_ptr[mask + 1] = cursor
    else:
        raise TypeError("broken_bond_sets must be a list or CSR tuple")

    max_components = num_masks * num_atoms
    words = (num_atoms + 63) // 64
    labels_array = np.empty(max_components, dtype=np.int64)
    component_offsets_array = np.empty(num_masks + 1, dtype=np.int64)
    component_sizes_array = np.empty(max_components, dtype=np.int64)
    labels = labels_array
    component_offsets = component_offsets_array
    component_sizes = component_sizes_array

    parents = <Py_ssize_t*>malloc(num_atoms * sizeof(Py_ssize_t))
    broken_epoch = <Py_ssize_t*>malloc(max(num_bonds, 1) * sizeof(Py_ssize_t))
    atom_roots = <Py_ssize_t*>malloc(num_atoms * sizeof(Py_ssize_t))
    component_max = <Py_ssize_t*>malloc(num_atoms * sizeof(Py_ssize_t))
    root_by_max = <Py_ssize_t*>malloc(num_atoms * sizeof(Py_ssize_t))
    root_to_component = <Py_ssize_t*>malloc(num_atoms * sizeof(Py_ssize_t))
    signatures = <uint64_t*>calloc(max_components * words, sizeof(uint64_t))
    if (
        parents == NULL
        or broken_epoch == NULL
        or atom_roots == NULL
        or component_max == NULL
        or root_by_max == NULL
        or root_to_component == NULL
        or signatures == NULL
    ):
        if parents != NULL:
            free(parents)
        if broken_epoch != NULL:
            free(broken_epoch)
        if atom_roots != NULL:
            free(atom_roots)
        if component_max != NULL:
            free(component_max)
        if root_by_max != NULL:
            free(root_by_max)
        if root_to_component != NULL:
            free(root_to_component)
        if signatures != NULL:
            free(signatures)
        raise MemoryError("unable to allocate fused MASShatter topology buffers")

    memset(broken_epoch, 0xff, max(num_bonds, 1) * sizeof(Py_ssize_t))
    component_cursor = 0
    component_offsets[0] = 0
    try:
        for mask in range(num_masks):
            for atom in range(num_atoms):
                parents[atom] = atom
            memset(component_max, 0xff, num_atoms * sizeof(Py_ssize_t))
            memset(root_by_max, 0xff, num_atoms * sizeof(Py_ssize_t))
            for item in range(mask_ptr[mask], mask_ptr[mask + 1]):
                broken_epoch[local_broken[item]] = mask

            for bond in range(num_bonds):
                if broken_epoch[bond] == mask:
                    continue
                left_root = _find_root(parents, endpoints[bond, 0])
                right_root = _find_root(parents, endpoints[bond, 1])
                if left_root != right_root:
                    parents[right_root] = left_root

            for atom in range(num_atoms):
                root = _find_root(parents, atom)
                atom_roots[atom] = root
                component_max[root] = atom
            for root in range(num_atoms):
                if component_max[root] >= 0:
                    root_by_max[component_max[root]] = root
            component_count = 0
            for atom in range(num_atoms):
                root = root_by_max[atom]
                if root >= 0:
                    root_to_component[root] = component_cursor + component_count
                    component_sizes[component_cursor + component_count] = 0
                    component_count += 1
            for atom in range(num_atoms):
                component = root_to_component[atom_roots[atom]]
                labels[mask * num_atoms + atom] = component
                component_sizes[component] += 1
                signatures[component * words + atom // 64] |= 1ULL << (atom % 64)
            component_cursor += component_count
            component_offsets[mask + 1] = component_cursor
        topology_ms = (perf_counter() - topology_start) * 1000.0

        assembly_start = perf_counter()
        table_size = 1
        while table_size < 2 * component_cursor:
            table_size <<= 1
        hash_slots = <Py_ssize_t*>malloc(table_size * sizeof(Py_ssize_t))
        membership_offsets = <Py_ssize_t*>malloc(
            (component_cursor + 1) * sizeof(Py_ssize_t)
        )
        membership_cursors = <Py_ssize_t*>malloc(
            max(component_cursor, 1) * sizeof(Py_ssize_t)
        )
        if hash_slots == NULL or membership_offsets == NULL or membership_cursors == NULL:
            raise MemoryError("unable to allocate fused MASShatter assembly buffers")
        memset(hash_slots, 0xff, table_size * sizeof(Py_ssize_t))

        unique_component_count = 0
        for component in range(component_cursor):
            signature_hash = 1469598103934665603ULL
            for word in range(words):
                signature_hash ^= signatures[component * words + word]
                signature_hash *= 1099511628211ULL
            slot = <Py_ssize_t>(signature_hash & (table_size - 1))
            while hash_slots[slot] >= 0:
                existing_component = hash_slots[slot]
                signatures_equal = True
                for word in range(words):
                    if (
                        signatures[existing_component * words + word]
                        != signatures[component * words + word]
                    ):
                        signatures_equal = False
                        break
                if signatures_equal:
                    break
                slot = (slot + 1) & (table_size - 1)
            if hash_slots[slot] < 0:
                hash_slots[slot] = component
                unique_component_count += 1

        component_mask_indices_array = np.empty(component_cursor, dtype=np.int64)
        component_cut_counts_array = np.empty(component_cursor, dtype=np.int64)
        atom_indices_array = np.empty(max_components, dtype=np.int64)
        atom_component_indices_array = np.empty(max_components, dtype=np.int64)
        mask_indices_array = np.empty(total_broken, dtype=np.int64)
        broken_bond_indices_array = np.empty(total_broken, dtype=np.int64)
        boundary_bond_indices_array = np.empty(2 * total_broken, dtype=np.int64)
        boundary_component_indices_array = np.empty(2 * total_broken, dtype=np.int64)
        component_mask_indices = component_mask_indices_array
        component_cut_counts = component_cut_counts_array
        atom_indices = atom_indices_array
        atom_component_indices = atom_component_indices_array
        mask_indices = mask_indices_array
        broken_bond_indices = broken_bond_indices_array
        boundary_bond_indices = boundary_bond_indices_array
        boundary_component_indices = boundary_component_indices_array

        membership_offsets[0] = 0
        for component in range(component_cursor):
            membership_offsets[component + 1] = (
                membership_offsets[component] + component_sizes[component]
            )
            membership_cursors[component] = membership_offsets[component]
        if membership_offsets[component_cursor] != max_components:
            raise ValueError("component labels do not partition every atom exactly once")

        for mask in range(num_masks):
            for component in range(component_offsets[mask], component_offsets[mask + 1]):
                component_mask_indices[component] = mask_offset + mask
                component_cut_counts[component] = cut_counts[mask]
            for atom in range(num_atoms):
                component = labels[mask * num_atoms + atom]
                membership_cursor = membership_cursors[component]
                atom_indices[membership_cursor] = atom_offset + atom
                atom_component_indices[membership_cursor] = component_offset + component
                membership_cursors[component] += 1

        cursor = 0
        boundary_cursor = 0
        for mask in range(num_masks):
            for item in range(mask_ptr[mask], mask_ptr[mask + 1]):
                bond = local_broken[item]
                mask_indices[cursor] = mask_offset + mask
                broken_bond_indices[cursor] = global_bonds[bond]
                cursor += 1
                left_component = labels[mask * num_atoms + endpoints[bond, 0]]
                right_component = labels[mask * num_atoms + endpoints[bond, 1]]
                if left_component == right_component:
                    continue
                boundary_bond_indices[boundary_cursor] = global_bonds[bond]
                boundary_component_indices[boundary_cursor] = (
                    component_offset + left_component
                )
                boundary_cursor += 1
                boundary_bond_indices[boundary_cursor] = global_bonds[bond]
                boundary_component_indices[boundary_cursor] = (
                    component_offset + right_component
                )
                boundary_cursor += 1
        assembly_ms = (perf_counter() - assembly_start) * 1000.0
    finally:
        free(parents)
        free(broken_epoch)
        free(atom_roots)
        free(component_max)
        free(root_by_max)
        free(root_to_component)
        free(signatures)
        if hash_slots != NULL:
            free(hash_slots)
        if membership_offsets != NULL:
            free(membership_offsets)
        if membership_cursors != NULL:
            free(membership_cursors)

    return (
        cut_counts_array,
        component_mask_indices_array,
        component_cut_counts_array,
        atom_indices_array,
        atom_component_indices_array,
        mask_indices_array,
        broken_bond_indices_array,
        boundary_bond_indices_array[:boundary_cursor],
        boundary_component_indices_array[:boundary_cursor],
        component_cursor,
        unique_component_count,
        topology_ms,
        assembly_ms,
    )


def unpack_component_memberships_raw(
    list component_atom_masks,
    Py_ssize_t num_atoms,
    Py_ssize_t expected_memberships,
):
    """Decode up-to-four-word component masks into flat membership arrays.

    The caller groups components by cut mask, so the returned component indices
    preserve that input order and atoms remain ascending within each component.
    """
    cdef cnp.ndarray[cnp.int64_t, ndim=1] component_indices
    cdef cnp.ndarray[cnp.int64_t, ndim=1] atom_indices
    cdef Py_ssize_t component_index, atom_index, bit_index, word_index
    cdef Py_ssize_t cursor = 0
    cdef Py_ssize_t word_atom_count
    cdef uint64_t atom_mask
    cdef object atom_mask_object

    if num_atoms <= 0 or num_atoms > 256:
        raise ValueError("num_atoms must be in [1, 256]")
    if expected_memberships < 0:
        raise ValueError("expected_memberships must be nonnegative")
    component_indices = np.empty(expected_memberships, dtype=np.int64)
    atom_indices = np.empty(expected_memberships, dtype=np.int64)

    for component_index in range(len(component_atom_masks)):
        atom_mask_object = component_atom_masks[component_index]
        if atom_mask_object < 0 or atom_mask_object >> num_atoms:
            raise ValueError("component atom mask references an invalid atom")
        for word_index in range((num_atoms + 63) // 64):
            atom_mask = (
                atom_mask_object >> (word_index * 64)
            ) & 0xffffffffffffffff
            word_atom_count = num_atoms - word_index * 64
            if word_atom_count > 64:
                word_atom_count = 64
            for bit_index in range(word_atom_count):
                if atom_mask & (1ULL << bit_index):
                    if cursor == expected_memberships:
                        raise ValueError(
                            "component masks contain too many atom memberships"
                        )
                    atom_index = word_index * 64 + bit_index
                    component_indices[cursor] = component_index
                    atom_indices[cursor] = atom_index
                    cursor += 1
    if cursor != expected_memberships:
        raise ValueError("component masks do not partition every atom exactly once")
    return component_indices, atom_indices


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
