"""Weisfeiler-Lehman subgraph hashes of DAG fragments (for nb-isomorphism grouping).
"""

from collections import Counter
from hashlib import blake2b

import numpy as np


def _hash_label(label, digest_size=32):
    """
    Adapted from https://networkx.org/documentation/stable/_modules/networkx/algorithms/graph_hashing.html
    """
    return blake2b(label.encode("ascii"), digest_size=digest_size).hexdigest()


def wl_hash(
    elems: list,
    bond_type_idxs: list,
    node_mask: np.ndarray,
    edges: np.ndarray,
    node_to_edge_idx: np.ndarray,
    include_bond_type: bool = False,
    max_iterations: int = -1,
) -> int:
    """
    Adapted from https://networkx.org/documentation/stable/_modules/networkx/algorithms/graph_hashing.html
    """

    cur_hashes = []
    num_nodes = len(elems)
    for i in range(num_nodes):
        if node_mask[i]:
            cur_hashes.append(str(elems[i]))
        else:
            cur_hashes.append("")
    cur_counter = Counter(cur_hashes)
    cur_counter.pop("", None)
    graph_hash_counts = sorted(cur_counter.items(), key=lambda x: x[0])
    iterations = np.sum(node_mask)
    assert iterations <= num_nodes, (iterations, num_nodes)
    if max_iterations == -1:
        max_iterations = iterations
    else:
        assert max_iterations >= 0, max_iterations
    ct = 0
    while ct < iterations and ct < max_iterations:
        # print(cur_hashes)
        new_hashes = []
        temp_atoms = 0
        # Step 2: Update hashes with local neighborhoods
        for node_idx in range(num_nodes):
            cur_hash = cur_hashes[node_idx]
            if not node_mask[node_idx]:
                new_hashes.append(cur_hash)
                continue
            # Count num atoms in this loop
            temp_atoms += 1
            # Get local neighbors
            neighbor_labels = []
            for edge_idx in node_to_edge_idx[node_idx]:
                if edge_idx == -1:
                    break
                node_idx_1, node_idx_2 = edges[edge_idx]
                if node_idx_1 == node_idx:
                    targ_node_idx = node_idx_2
                else:
                    targ_node_idx = node_idx_1
                assert targ_node_idx != node_idx
                if not node_mask[targ_node_idx]:
                    continue
                targ_hash = cur_hashes[targ_node_idx]
                if include_bond_type:
                    bondtype = bond_type_idxs[edge_idx]
                    neighbor_label = f"_{bondtype}_{targ_hash}"
                else:
                    neighbor_label = f"_{targ_hash}"
                neighbor_labels.append(neighbor_label)
            new_hash = cur_hash + "".join(sorted(neighbor_labels))
            new_hash = _hash_label(new_hash)
            new_hashes.append(new_hash)
        assert temp_atoms == iterations, (temp_atoms, iterations)
        new_counter = Counter(new_hashes)
        new_counter.pop("", None)
        graph_hash_counts.extend(sorted(new_counter.items(), key=lambda x: x[0]))
        cur_hashes = new_hashes
        # print(f"> {ct}")
        # print(new_graph_hash)
        # print(cur_hashes)
        ct += 1
    graph_hash = _hash_label(str(tuple(graph_hash_counts)))
    return graph_hash


def get_subgraph_hashes(
    nodes_mask_matrix: np.ndarray,
    elems: list,
    bond_type_idxs: list,
    edges: np.ndarray,
    node_to_edge_idx: np.ndarray,
    include_bond_type: bool,
    max_iterations: int,
) -> list:
    subgraph_hashes = []
    num_subgraphs = nodes_mask_matrix.shape[0]
    for i in range(num_subgraphs):
        subgraph_mask = nodes_mask_matrix[i]
        subgraph_hash = wl_hash(
            elems,
            bond_type_idxs,
            subgraph_mask,
            edges,
            node_to_edge_idx,
            include_bond_type=include_bond_type,
            max_iterations=max_iterations,
        )
        subgraph_hashes.append(subgraph_hash)
    return subgraph_hashes
