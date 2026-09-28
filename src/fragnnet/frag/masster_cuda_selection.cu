// Exact single-component ranked masks. CUDA is an optional, explicit prototype.
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {
constexpr int META_WIDTH = 15;
constexpr int STATE_WIDTH = 6;

__device__ bool node_less(
    int left, int right, const double* penalties, const uint64_t* flips, int words
) {
    if (penalties[left] < penalties[right]) return true;
    if (penalties[left] > penalties[right]) return false;
    // Lexicographic sorted flip-position tuples, including proper-prefix ordering.
    for (int word = 0; word < words; ++word) {
        const uint64_t left_word = flips[static_cast<int64_t>(left) * words + word];
        const uint64_t right_word = flips[static_cast<int64_t>(right) * words + word];
        const uint64_t difference = left_word ^ right_word;
        if (!difference) continue;
        const uint64_t lowest = difference & (~difference + 1);
        const bool owner_left = (left_word & lowest) != 0;
        const uint64_t other_word = owner_left ? right_word : left_word;
        const uint64_t above_mask = ~(lowest | (lowest - 1));
        bool other_has_higher = (other_word & above_mask) != 0;
        for (int tail = word + 1; tail < words && !other_has_higher; ++tail) {
            const int other = owner_left ? right : left;
            other_has_higher = flips[static_cast<int64_t>(other) * words + tail] != 0;
        }
        return owner_left == other_has_higher;
    }
    return false;
}

__device__ void heap_push(
    int node, int* heap, int& size, const double* penalties,
    const uint64_t* flips, int words
) {
    int index = size++;
    while (index > 0) {
        const int parent = (index - 1) / 2;
        if (!node_less(node, heap[parent], penalties, flips, words)) break;
        heap[index] = heap[parent];
        index = parent;
    }
    heap[index] = node;
}

__device__ int heap_pop(
    int* heap, int& size, const double* penalties, const uint64_t* flips, int words
) {
    const int result = heap[0];
    if (--size == 0) return result;
    const int last = heap[size];
    int index = 0;
    while (true) {
        const int left = 2 * index + 1;
        if (left >= size) break;
        const int right = left + 1;
        int child = left;
        if (right < size && node_less(heap[right], heap[left], penalties, flips, words)) {
            child = right;
        }
        if (!node_less(heap[child], last, penalties, flips, words)) break;
        heap[index] = heap[child];
        index = child;
    }
    heap[index] = last;
    return result;
}

__device__ bool is_extra_forced(
    const uint64_t* mask, const uint64_t* extra_forced, int extra_count, int words
) {
    for (int extra = 0; extra < extra_count; ++extra) {
        bool equal = true;
        for (int word = 0; word < words && equal; ++word) {
            equal = mask[word] == extra_forced[static_cast<int64_t>(extra) * words + word];
        }
        if (equal) return true;
    }
    return false;
}

__device__ void emit_learned(
    const uint64_t* mask, int words, bool force_singles, const uint64_t* extra_forced,
    int extra_count, uint64_t* output, int64_t* prefix, int64_t* state
) {
    int cuts = 0;
    for (int word = 0; word < words; ++word) cuts += __popcll(mask[word]);
    ++state[1];
    if (cuts == 1) ++state[2];
    if (cuts == 0 || (force_singles && cuts == 1)) return;
    if (is_extra_forced(mask, extra_forced, extra_count, words)) return;
    const int64_t row = state[3]++;
    for (int word = 0; word < words; ++word) output[row * words + word] = mask[word];
    state[5] += cuts;
    prefix[row + 1] = state[5];
}

__global__ void rank_masks(
    const float* logits, const int64_t* metadata, double* all_costs, int* all_bonds,
    double* all_penalties, uint64_t* all_flips, int* all_heaps, uint64_t* all_masks,
    uint64_t* all_work_words, int64_t* all_state, int64_t* all_prefix, bool force_singles,
    const uint64_t* all_extra_forced
) {
    // A literal serial heap per representative is the correctness baseline.
    if (threadIdx.x != 0) return;
    const int64_t* meta = metadata + blockIdx.x * META_WIDTH;
    int64_t* state = all_state + blockIdx.x * STATE_WIDTH;
    const int bonds = static_cast<int>(meta[0]);
    const int limit = static_cast<int>(meta[2]);
    const int capacity = static_cast<int>(meta[3]);
    const int words = static_cast<int>(meta[4]);
    const uint64_t* extra_forced = all_extra_forced + meta[13];
    const int extra_count = static_cast<int>(meta[14]);
    double* costs = all_costs + meta[5];
    int* sorted_bonds = all_bonds + meta[5];
    uint64_t* flips = all_flips + meta[6];
    double* penalties = all_penalties + meta[7];
    int* heap = all_heaps + meta[8];
    uint64_t* output = all_masks + meta[9];
    int64_t* prefix = all_prefix + meta[11];
    uint64_t* map = all_work_words + meta[12];
    uint64_t* current = map + words;
    for (int column = 0; column < STATE_WIDTH; ++column) state[column] = 0;
    for (int word = 0; word < words; ++word) map[word] = 0;

    // 1. FP32 input converts exactly to FP64. No log/exp values affect this domain's
    // ranking or disabled floors, and even extreme finite FP32 sums cannot overflow.
    for (int bond = 0; bond < bonds; ++bond) {
        const double value = static_cast<double>(logits[meta[1] + bond]);
        if (!isfinite(value)) {
            state[0] = 1;
            return;
        }
        if (value > 0.0) map[bond / 64] |= uint64_t(1) << (bond % 64);
        costs[bond] = fabs(value);
        sorted_bonds[bond] = bond;
    }
    for (int index = 1; index < bonds; ++index) {
        const double cost = costs[index];
        const int bond = sorted_bonds[index];
        int insertion = index;
        while (insertion > 0 && (
            cost < costs[insertion - 1] ||
            (cost == costs[insertion - 1] && bond < sorted_bonds[insertion - 1])
        )) {
            costs[insertion] = costs[insertion - 1];
            sorted_bonds[insertion] = sorted_bonds[insertion - 1];
            --insertion;
        }
        costs[insertion] = cost;
        sorted_bonds[insertion] = bond;
    }

    // 2. Forced prefix precedes ranked masks in unique-fragment mode. Extra forced masks
    // (SMARTS template cuts) follow the empty and single-cut masks; the host passes them
    // deduplicated and never empty or, with forced singles, single-bond.
    const int base_forced = 1 + (force_singles ? bonds : 0);
    for (int row = 0; row < base_forced; ++row) {
        for (int word = 0; word < words; ++word) {
            output[static_cast<int64_t>(row) * words + word] = 0;
        }
        if (row > 0) {
            const int bond = row - 1;
            output[static_cast<int64_t>(row) * words + bond / 64] = uint64_t(1) << (bond % 64);
        }
        prefix[row + 1] = row;
    }
    prefix[0] = 0;
    int64_t forced_bonds = base_forced - 1;
    for (int extra = 0; extra < extra_count; ++extra) {
        const int64_t row = base_forced + extra;
        for (int word = 0; word < words; ++word) {
            const uint64_t value = extra_forced[static_cast<int64_t>(extra) * words + word];
            output[row * words + word] = value;
            forced_bonds += __popcll(value);
        }
        prefix[row + 1] = forced_bonds;
    }
    state[3] = base_forced + extra_count;
    state[4] = base_forced + extra_count;
    state[5] = forced_bonds;
    emit_learned(map, words, force_singles, extra_forced, extra_count, output, prefix, state);
    if (bonds == 0 || state[1] == limit) return;

    // 3. Heap keys preserve the CPU (penalty, flip-position tuple) order exactly.
    penalties[0] = costs[0];
    for (int word = 0; word < words; ++word) flips[word] = word == 0 ? 1 : 0;
    int node_count = 1;
    int heap_size = 0;
    heap_push(0, heap, heap_size, penalties, flips, words);
    while (heap_size > 0 && state[1] < limit) {
        const int node = heap_pop(heap, heap_size, penalties, flips, words);
        const uint64_t* node_flips = flips + static_cast<int64_t>(node) * words;
        const double total = penalties[node];
        for (int word = 0; word < words; ++word) current[word] = map[word];
        int last = -1;
        for (int word = 0; word < words; ++word) {
            uint64_t remaining = node_flips[word];
            while (remaining) {
                const int position = word * 64 + __ffsll(remaining) - 1;
                const int bond = sorted_bonds[position];
                current[bond / 64] ^= uint64_t(1) << (bond % 64);
                last = position;
                remaining &= remaining - 1;
            }
        }
        emit_learned(
            current, words, force_singles, extra_forced, extra_count, output, prefix, state
        );
        if (state[1] == limit) break;
        const int next = last + 1;
        if (next >= bonds) continue;
        if (node_count + 2 > capacity) {
            state[0] = 2;
            return;
        }
        const int append = node_count++;
        const int replacement = node_count++;
        uint64_t* appended = flips + static_cast<int64_t>(append) * words;
        uint64_t* replaced = flips + static_cast<int64_t>(replacement) * words;
        for (int word = 0; word < words; ++word) {
            appended[word] = node_flips[word];
            replaced[word] = node_flips[word];
        }
        appended[next / 64] |= uint64_t(1) << (next % 64);
        replaced[last / 64] ^= uint64_t(1) << (last % 64);
        replaced[next / 64] |= uint64_t(1) << (next % 64);
        penalties[append] = __dadd_rn(total, costs[next]);
        penalties[replacement] = __dadd_rn(__dsub_rn(total, costs[last]), costs[next]);
        heap_push(append, heap, heap_size, penalties, flips, words);
        heap_push(replacement, heap, heap_size, penalties, flips, words);
    }
}

__global__ void write_csr(
    const int64_t* metadata, const uint64_t* all_masks, const int64_t* all_prefix,
    const int64_t* all_state, const int64_t* output_offsets,
    int64_t* output_ptr, int* output_bonds
) {
    const int64_t* meta = metadata + blockIdx.x * META_WIDTH;
    const int64_t* state = all_state + blockIdx.x * STATE_WIDTH;
    const int words = static_cast<int>(meta[4]);
    const uint64_t* masks = all_masks + meta[9];
    const int64_t* prefix = all_prefix + meta[11];
    const int64_t ptr_offset = output_offsets[2 * blockIdx.x];
    const int64_t bond_offset = output_offsets[2 * blockIdx.x + 1];
    for (int64_t row = threadIdx.x; row < state[3]; row += blockDim.x) {
        int64_t cursor = bond_offset + prefix[row];
        output_ptr[ptr_offset + row] = cursor;
        for (int word = 0; word < words; ++word) {
            uint64_t remaining = masks[row * words + word];
            while (remaining) {
                output_bonds[cursor++] = word * 64 + __ffsll(remaining) - 1;
                remaining &= remaining - 1;
            }
        }
    }
    if (threadIdx.x == 0) output_ptr[ptr_offset + state[3]] = bond_offset + prefix[state[3]];
}
}  // namespace

extern "C" int masster_selection_launch(
    const uint64_t* pointers, int representatives, int force_singles,
    int csr_phase, uint64_t stream_address
) {
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_address);
    if (!csr_phase) {
        rank_masks<<<representatives, 32, 0, stream>>>(
            reinterpret_cast<const float*>(pointers[0]),
            reinterpret_cast<const int64_t*>(pointers[1]),
            reinterpret_cast<double*>(pointers[2]), reinterpret_cast<int*>(pointers[3]),
            reinterpret_cast<double*>(pointers[4]), reinterpret_cast<uint64_t*>(pointers[5]),
            reinterpret_cast<int*>(pointers[6]), reinterpret_cast<uint64_t*>(pointers[7]),
            reinterpret_cast<uint64_t*>(pointers[8]), reinterpret_cast<int64_t*>(pointers[9]),
            reinterpret_cast<int64_t*>(pointers[10]), force_singles != 0,
            reinterpret_cast<const uint64_t*>(pointers[14])
        );
    } else {
        write_csr<<<representatives, 128, 0, stream>>>(
            reinterpret_cast<const int64_t*>(pointers[1]),
            reinterpret_cast<const uint64_t*>(pointers[7]),
            reinterpret_cast<const int64_t*>(pointers[10]),
            reinterpret_cast<const int64_t*>(pointers[9]),
            reinterpret_cast<const int64_t*>(pointers[11]),
            reinterpret_cast<int64_t*>(pointers[12]), reinterpret_cast<int*>(pointers[13])
        );
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" const char* masster_selection_error_string(int error) {
    return cudaGetErrorString(static_cast<cudaError_t>(error));
}
