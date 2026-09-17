// Optional, integer-only CUDA topology prototype. Loaded explicitly by the Python wrapper.
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
enum Meta {
    ATOMS, BONDS, MASKS, WORDS, EDGE_OFFSET, MASK_OFFSET, LABEL_OFFSET,
    SIGNATURE_OFFSET, DICTIONARY_OFFSET, FIRST_OFFSET, HASH_OFFSET,
    HASH_CAPACITY, UNIQUE_CAPACITY, TARGET, FORCED, META_WIDTH
};
enum State {
    DONE, UNIQUE_COUNT, COMPONENT_COUNT, BOUNDARY_COUNT, TARGET_PREFIX,
    SCANNED, RETAINED, RETAINED_COMPONENTS, RETAINED_UNIQUE, RETAINED_ATOMS,
    RETAINED_BROKEN, RETAINED_BOUNDARY, ERROR, PHYSICAL_MASKS, STATE_WIDTH
};
struct Buffers {
    const int64_t* meta;
    const int32_t* endpoints;
    const int64_t* mask_ptr;
    const int32_t* broken;
    int32_t* labels;
    uint64_t* signatures;
    uint64_t* dictionary;
    int64_t* first_components;
    int32_t* hash_slots;
    int64_t* component_prefix;
    int64_t* boundary_prefix;
    int64_t* state;
    int64_t* output;
    const int64_t* output_offsets;
};

__device__ uint64_t signature_hash(const uint64_t* words, int count) {
    uint64_t value = 1469598103934665603ULL;
    for (int word = 0; word < count; ++word) {
        value ^= words[word];
        value *= 1099511628211ULL;
    }
    return value;
}

__device__ bool signatures_equal(const uint64_t* left, const uint64_t* right, int count) {
    for (int word = 0; word < count; ++word) {
        if (left[word] != right[word]) return false;
    }
    return true;
}

__global__ void validate_mask_csr(
    const int64_t* metadata, const int64_t* ptr, const int32_t* broken,
    int64_t* states, int representatives, int64_t broken_count
) {
    const int representative = blockIdx.x;
    const int64_t* m = metadata + representative * META_WIDTH;
    int64_t* error = states + representative * STATE_WIDTH + ERROR;
    const int64_t begin = ptr[m[MASK_OFFSET]];
    const int64_t end = ptr[m[MASK_OFFSET] + m[MASKS]];
    if (begin < 0 || end < begin || end > broken_count) {
        if (threadIdx.x == 0) *error = 3;
        return;
    }
    if (threadIdx.x == 0) {
        const int64_t expected_begin = representative == 0 ? 0 : ptr[m[MASK_OFFSET] - 1];
        if (begin != expected_begin || (representative + 1 == representatives && end != broken_count)) {
            atomicExch(reinterpret_cast<unsigned long long*>(error), 3ULL);
        }
    }
    for (int64_t mask = threadIdx.x; mask < m[MASKS]; mask += blockDim.x) {
        const int64_t left = ptr[m[MASK_OFFSET] + mask];
        const int64_t right = ptr[m[MASK_OFFSET] + mask + 1];
        if (left < begin || right < left || right > end || right - left > 1073741823LL) {
            atomicExch(reinterpret_cast<unsigned long long*>(error), 3ULL);
        }
    }
    for (int64_t item = begin + threadIdx.x; item < end; item += blockDim.x) {
        if (broken[item] < 0 || broken[item] >= m[BONDS]) {
            atomicExch(reinterpret_cast<unsigned long long*>(error), 3ULL);
        }
    }
}

// Each CTA handles one small graph. Labels converge to the maximum atom ID in
// each component, which directly supplies the CPU component-order convention.
__global__ void connected_components(Buffers b, int tile_start, int tile_masks, bool replay) {
    const int representative = blockIdx.x;
    const int tile_mask = blockIdx.y;
    const int64_t* m = b.meta + representative * META_WIDTH;
    int64_t* s = b.state + representative * STATE_WIDTH;
    const int mask = tile_start + tile_mask;
    const int atom_count = m[ATOMS];
    const int bond_count = m[BONDS];
    const int words = m[WORDS];
    if (s[ERROR] || mask >= (replay ? s[RETAINED] : m[MASKS]) || (!replay && s[DONE])) return;

    extern __shared__ int32_t work[];
    int32_t* current = work;
    int32_t* next = work + atom_count;
    int32_t* component_of_root = work + 2 * atom_count;
    int32_t* broken_flags = work + 3 * atom_count;
    int32_t* changed = broken_flags + bond_count;
    int32_t* boundary_count = changed + 1;
    int32_t* component_count = changed + 2;
    const int64_t begin = b.mask_ptr[m[MASK_OFFSET] + mask];
    const int64_t end = b.mask_ptr[m[MASK_OFFSET] + mask + 1];
    const int32_t* endpoints = b.endpoints + 2 * m[EDGE_OFFSET];
    uint64_t* signatures = b.signatures + m[SIGNATURE_OFFSET] +
        static_cast<int64_t>(tile_mask) * atom_count * words;
    for (int atom = threadIdx.x; atom < atom_count; atom += blockDim.x) current[atom] = atom;
    for (int bond = threadIdx.x; bond < bond_count; bond += blockDim.x) broken_flags[bond] = 0;
    for (int index = threadIdx.x; index < atom_count * words; index += blockDim.x) signatures[index] = 0;
    if (threadIdx.x == 0) *boundary_count = 0;
    __syncthreads();
    for (int64_t item = begin + threadIdx.x; item < end; item += blockDim.x) {
        atomicExch(broken_flags + b.broken[item], 1);
    }
    __syncthreads();

    for (int iteration = 0; iteration < atom_count; ++iteration) {
        if (threadIdx.x == 0) *changed = 0;
        // Pointer jumping stays within the same component and never lowers a
        // label. Intact-edge updates remain, so the fixed point is still exactly
        // the maximum atom ID, with the original atom_count convergence bound.
        for (int atom = threadIdx.x; atom < atom_count; atom += blockDim.x) {
            next[atom] = max(current[atom], current[current[atom]]);
        }
        __syncthreads();
        for (int bond = threadIdx.x; bond < bond_count; bond += blockDim.x) {
            if (broken_flags[bond]) continue;
            const int left = endpoints[2 * bond];
            const int right = endpoints[2 * bond + 1];
            atomicMax(next + left, current[right]);
            atomicMax(next + right, current[left]);
        }
        __syncthreads();
        for (int atom = threadIdx.x; atom < atom_count; atom += blockDim.x) {
            if (next[atom] != current[atom]) atomicExch(changed, 1);
        }
        __syncthreads();
        // Every warp must capture this round's flag before any warp resets it
        // for the next round. Otherwise a late warp could exit a live CTA.
        const bool stable = *changed == 0;
        __syncthreads();
        int32_t* swap = current;
        current = next;
        next = swap;
        if (stable) break;
    }
    if (threadIdx.x == 0) {
        int count = 0;
        for (int atom = 0; atom < atom_count; ++atom) {
            if (current[atom] == atom) component_of_root[atom] = count++;
        }
        *component_count = count;
    }
    __syncthreads();
    for (int atom = threadIdx.x; atom < atom_count; atom += blockDim.x) {
        const int component = component_of_root[current[atom]];
        b.labels[m[LABEL_OFFSET] + static_cast<int64_t>(tile_mask) * atom_count + atom] = component;
        atomicOr(reinterpret_cast<unsigned long long*>(signatures + component * words + atom / 64),
                 1ULL << (atom % 64));
    }
    for (int64_t item = begin + threadIdx.x; item < end; item += blockDim.x) {
        const int bond = b.broken[item];
        if (current[endpoints[2 * bond]] != current[endpoints[2 * bond + 1]]) {
            atomicAdd(boundary_count, 2);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && !replay) {
        // Counts temporarily occupy the next-prefix slots. The ordered coordinator
        // below converts these entries to cumulative prefixes before the next tile.
        b.component_prefix[m[MASK_OFFSET] + mask + 1] = *component_count;
        b.boundary_prefix[m[MASK_OFFSET] + mask + 1] = *boundary_count;
        atomicAdd(reinterpret_cast<unsigned long long*>(s + PHYSICAL_MASKS), 1ULL);
    }
}

// One ordered GPU worker per representative defines first-occurrence identities.
// Connectivity is parallel over masks; hash insertion order is intentionally exact.
__global__ void deduplicate_tile(Buffers b, int tile_start, int tile_masks) {
    const int representative = blockIdx.x;
    if (threadIdx.x != 0) return;
    const int64_t* m = b.meta + representative * META_WIDTH;
    int64_t* s = b.state + representative * STATE_WIDTH;
    if (s[DONE] || s[ERROR]) return;
    const int words = m[WORDS];
    uint64_t* dictionary = b.dictionary + m[DICTIONARY_OFFSET];
    int64_t* first = b.first_components + m[FIRST_OFFSET];
    int32_t* slots = b.hash_slots + m[HASH_OFFSET];
    for (int tile_mask = 0; tile_mask < tile_masks; ++tile_mask) {
        const int mask = tile_start + tile_mask;
        if (mask >= m[MASKS]) break;
        const int64_t prefix_index = m[MASK_OFFSET] + mask + 1;
        const int component_count = b.component_prefix[prefix_index];
        const int boundary_count = b.boundary_prefix[prefix_index];
        const uint64_t* signatures = b.signatures + m[SIGNATURE_OFFSET] +
            static_cast<int64_t>(tile_mask) * m[ATOMS] * words;
        for (int component = 0; component < component_count; ++component) {
            const uint64_t* signature = signatures + component * words;
            int64_t slot = signature_hash(signature, words) & (m[HASH_CAPACITY] - 1);
            while (slots[slot] != 0 &&
                   !signatures_equal(signature, dictionary + (slots[slot] - 1) * words, words)) {
                slot = (slot + 1) & (m[HASH_CAPACITY] - 1);
            }
            if (slots[slot] == 0) {
                const int64_t unique = s[UNIQUE_COUNT];
                if (unique >= m[UNIQUE_CAPACITY]) { s[ERROR] = 1; return; }
                for (int word = 0; word < words; ++word) dictionary[unique * words + word] = signature[word];
                first[unique] = s[COMPONENT_COUNT] + component;
                slots[slot] = unique + 1;
                ++s[UNIQUE_COUNT];
            }
        }
        s[COMPONENT_COUNT] += component_count;
        s[BOUNDARY_COUNT] += boundary_count;
        b.component_prefix[prefix_index] = s[COMPONENT_COUNT];
        b.boundary_prefix[prefix_index] = s[BOUNDARY_COUNT];
        s[SCANNED] = mask + 1;
        if (s[TARGET_PREFIX] == 0 && s[UNIQUE_COUNT] >= m[TARGET]) s[TARGET_PREFIX] = mask + 1;
        const bool overshot = s[UNIQUE_COUNT] > m[TARGET];
        if ((overshot && mask + 1 >= m[FORCED]) || mask + 1 == m[MASKS]) {
            const int retained = overshot ? max(s[TARGET_PREFIX], m[FORCED]) : mask + 1;
            const int64_t retained_components = b.component_prefix[m[MASK_OFFSET] + retained];
            int retained_unique = 0;
            while (retained_unique < s[UNIQUE_COUNT] && first[retained_unique] < retained_components) {
                ++retained_unique;
            }
            s[RETAINED] = retained;
            s[RETAINED_COMPONENTS] = retained_components;
            s[RETAINED_UNIQUE] = retained_unique;
            s[RETAINED_ATOMS] = static_cast<int64_t>(retained) * m[ATOMS];
            s[RETAINED_BROKEN] = b.mask_ptr[m[MASK_OFFSET] + retained] - b.mask_ptr[m[MASK_OFFSET]];
            s[RETAINED_BOUNDARY] = b.boundary_prefix[m[MASK_OFFSET] + retained];
            s[DONE] = 1;
            break;
        }
    }
}

__global__ void assemble_tile(Buffers b, int tile_start) {
    const int representative = blockIdx.x;
    const int tile_mask = blockIdx.y;
    const int64_t* m = b.meta + representative * META_WIDTH;
    int64_t* s = b.state + representative * STATE_WIDTH;
    const int mask = tile_start + tile_mask;
    if (mask >= s[RETAINED]) return;
    const int atom_count = m[ATOMS];
    const int words = m[WORDS];
    const int64_t* offsets = b.output_offsets + representative * 11;
    const int64_t component_start = b.component_prefix[m[MASK_OFFSET] + mask];
    const int component_count = b.component_prefix[m[MASK_OFFSET] + mask + 1] - component_start;
    const int64_t begin = b.mask_ptr[m[MASK_OFFSET] + mask];
    const int64_t end = b.mask_ptr[m[MASK_OFFSET] + mask + 1];
    const int64_t cut_count = end - begin;
    const int32_t* labels = b.labels + m[LABEL_OFFSET] + static_cast<int64_t>(tile_mask) * atom_count;
    const uint64_t* signatures = b.signatures + m[SIGNATURE_OFFSET] +
        static_cast<int64_t>(tile_mask) * atom_count * words;
    const uint64_t* dictionary = b.dictionary + m[DICTIONARY_OFFSET];
    const int32_t* slots = b.hash_slots + m[HASH_OFFSET];

    for (int component = threadIdx.x; component < component_count; component += blockDim.x) {
        const int64_t index = component_start + component;
        b.output[offsets[1] + index] = mask;
        b.output[offsets[2] + index] = cut_count;
        const uint64_t* signature = signatures + component * words;
        int64_t slot = signature_hash(signature, words) & (m[HASH_CAPACITY] - 1);
        while (slots[slot] != 0 &&
               !signatures_equal(signature, dictionary + (slots[slot] - 1) * words, words)) {
            slot = (slot + 1) & (m[HASH_CAPACITY] - 1);
        }
        if (slots[slot] == 0 || slots[slot] - 1 >= s[RETAINED_UNIQUE]) {
            atomicExch(reinterpret_cast<unsigned long long*>(s + ERROR), 2ULL);
        } else {
            b.output[offsets[9] + index] = slots[slot] - 1;
        }
    }
    extern __shared__ int32_t cursors[];
    if (threadIdx.x == 0) {
        // Stable counting sort emits ascending atoms within each component without
        // quadratic atom-pair storage or nondeterministic atomic membership order.
        for (int component = 0; component < component_count; ++component) cursors[component] = 0;
        for (int atom = 0; atom < atom_count; ++atom) ++cursors[labels[atom]];
        int position = 0;
        for (int component = 0; component < component_count; ++component) {
            const int size = cursors[component];
            cursors[component] = position;
            position += size;
        }
        for (int atom = 0; atom < atom_count; ++atom) {
            const int component = labels[atom];
            const int64_t index = static_cast<int64_t>(mask) * atom_count + cursors[component]++;
            b.output[offsets[3] + index] = atom;
            b.output[offsets[4] + index] = component_start + component;
        }
        b.output[offsets[0] + mask] = cut_count;
        int64_t boundary = b.boundary_prefix[m[MASK_OFFSET] + mask];
        const int32_t* endpoints = b.endpoints + 2 * m[EDGE_OFFSET];
        for (int64_t item = begin; item < end; ++item) {
            const int bond = b.broken[item];
            const int64_t broken_index = item - b.mask_ptr[m[MASK_OFFSET]];
            b.output[offsets[5] + broken_index] = mask;
            b.output[offsets[6] + broken_index] = bond;
            const int left = labels[endpoints[2 * bond]];
            const int right = labels[endpoints[2 * bond + 1]];
            if (left == right) continue;
            b.output[offsets[7] + boundary] = bond;
            b.output[offsets[8] + boundary++] = component_start + left;
            b.output[offsets[7] + boundary] = bond;
            b.output[offsets[8] + boundary++] = component_start + right;
        }
    }
    if (mask == 0) {
        for (int64_t unique = threadIdx.x; unique < s[RETAINED_UNIQUE]; unique += blockDim.x) {
            b.output[offsets[10] + unique] = b.first_components[m[FIRST_OFFSET] + unique];
        }
    }
}
}  // namespace

extern "C" int mass_shatter_topology_validate_mask_csr(
    const uint64_t* pointers, int representatives, int64_t broken_count, uint64_t stream_pointer
) {
    validate_mask_csr<<<representatives, 128, 0, reinterpret_cast<cudaStream_t>(stream_pointer)>>>(
        reinterpret_cast<const int64_t*>(pointers[0]), reinterpret_cast<const int64_t*>(pointers[2]),
        reinterpret_cast<const int32_t*>(pointers[3]), reinterpret_cast<int64_t*>(pointers[11]),
        representatives, broken_count
    );
    return static_cast<int>(cudaGetLastError());
}

extern "C" int mass_shatter_topology_launch(
    const uint64_t* pointers, int representatives, int tile_start, int tile_masks,
    int maximum_atoms, int maximum_bonds, int replay, uint64_t stream_pointer
) {
    Buffers b{
        reinterpret_cast<const int64_t*>(pointers[0]), reinterpret_cast<const int32_t*>(pointers[1]),
        reinterpret_cast<const int64_t*>(pointers[2]), reinterpret_cast<const int32_t*>(pointers[3]),
        reinterpret_cast<int32_t*>(pointers[4]), reinterpret_cast<uint64_t*>(pointers[5]),
        reinterpret_cast<uint64_t*>(pointers[6]), reinterpret_cast<int64_t*>(pointers[7]),
        reinterpret_cast<int32_t*>(pointers[8]), reinterpret_cast<int64_t*>(pointers[9]),
        reinterpret_cast<int64_t*>(pointers[10]), reinterpret_cast<int64_t*>(pointers[11]),
        reinterpret_cast<int64_t*>(pointers[12]), reinterpret_cast<const int64_t*>(pointers[13])
    };
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_pointer);
    const dim3 grid(representatives, tile_masks);
    const size_t shared_bytes = (3 * maximum_atoms + maximum_bonds + 3) * sizeof(int32_t);
    connected_components<<<grid, 128, shared_bytes, stream>>>(b, tile_start, tile_masks, replay != 0);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) return static_cast<int>(error);
    if (replay) {
        assemble_tile<<<grid, 128, maximum_atoms * sizeof(int32_t), stream>>>(b, tile_start);
    } else {
        deduplicate_tile<<<representatives, 1, 0, stream>>>(b, tile_start, tile_masks);
    }
    return static_cast<int>(cudaGetLastError());
}

extern "C" const char* mass_shatter_topology_error_string(int error) {
    return cudaGetErrorString(static_cast<cudaError_t>(error));
}
