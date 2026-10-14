"""Pool atom and bond embeddings over cut-mask components, complements and boundaries, plus the
sparse component-membership and boundary-orientation lookups they share."""

from __future__ import annotations

from typing import Literal

import torch as th

from fragnnet.utils.misc_utils import _validate_tensor_condition


def _pool_indexed_embeddings(
    item_embeddings: th.Tensor,
    item_indices: th.Tensor,
    component_indices: th.Tensor,
    *,
    num_components: int,
    reduction: Literal["sum", "mean"] = "mean",
    require_nonempty: bool,
) -> th.Tensor:
    if item_embeddings.ndim != 2:
        raise ValueError("item_embeddings must have shape (num_items, embedding_dim)")
    if num_components < 0:
        raise ValueError(f"num_components must be nonnegative, got {num_components}")
    if item_indices.ndim != 1 or component_indices.shape != item_indices.shape:
        raise ValueError("item_indices and component_indices must be matching vectors")
    if item_indices.dtype != th.long or component_indices.dtype != th.long:
        raise ValueError("item_indices and component_indices must have dtype torch.long")
    if reduction not in {"sum", "mean"}:
        raise ValueError(f"unknown component pooling reduction: {reduction}")
    if item_indices.numel() > 0:
        _validate_tensor_condition(
            ((item_indices >= 0) & (item_indices < item_embeddings.shape[0])).all(),
            "item_indices references a missing item embedding",
        )
        _validate_tensor_condition(
            ((component_indices >= 0) & (component_indices < num_components)).all(),
            "component_indices references a missing component",
        )

    pooled = item_embeddings.new_zeros((num_components, item_embeddings.shape[1])).index_add(
        0,
        component_indices,
        item_embeddings[item_indices],
    )
    if reduction == "mean" or require_nonempty:
        component_sizes = th.zeros(
            num_components,
            dtype=th.long,
            device=component_indices.device,
        ).index_add(0, component_indices, th.ones_like(component_indices))
        if require_nonempty:
            _validate_tensor_condition(
                (component_sizes > 0).all(),
                "every component must contain at least one pooled item",
            )
    if reduction == "mean":
        pooled = pooled / component_sizes.clamp(min=1).to(dtype=pooled.dtype).unsqueeze(1)
    return pooled


def pool_component_embeddings(
    item_embeddings: th.Tensor,
    item_indices: th.Tensor,
    component_indices: th.Tensor,
    *,
    num_components: int,
    reduction: Literal["sum", "mean"] = "mean",
) -> th.Tensor:
    """Pool sparse item/component membership pairs into component embeddings."""
    return _pool_indexed_embeddings(
        item_embeddings,
        item_indices,
        component_indices,
        num_components=num_components,
        reduction=reduction,
        require_nonempty=True,
    )


def pool_component_and_complement_embeddings(
    atom_embeddings: th.Tensor,
    atom_indices: th.Tensor,
    atom_component_indices: th.Tensor,
    component_batch_indices: th.Tensor,
    atom_batch_indices: th.Tensor,
    *,
    num_components: int,
    num_molecules: int,
    reduction: Literal["sum", "mean"] = "mean",
) -> tuple[th.Tensor, th.Tensor]:
    """Pool components and derive their complements from precursor totals.

    Every component belongs to one precursor, so its complement sum is the
    precursor sum minus the component sum. This avoids materializing the much
    larger complement-atom/component membership relation.
    """
    if atom_embeddings.ndim != 2:
        raise ValueError("atom_embeddings must be a matrix")
    if atom_indices.ndim != 1 or atom_component_indices.shape != atom_indices.shape:
        raise ValueError("atom_indices and atom_component_indices must be matching vectors")
    if atom_indices.dtype != th.long or atom_component_indices.dtype != th.long:
        raise ValueError("atom and component membership indices must have dtype torch.long")
    if component_batch_indices.shape != (num_components,):
        raise ValueError("component_batch_indices must contain one index per component")
    if atom_batch_indices.shape != (atom_embeddings.shape[0],):
        raise ValueError("atom_batch_indices must contain one index per atom")
    if component_batch_indices.dtype != th.long or atom_batch_indices.dtype != th.long:
        raise ValueError("batch indices must have dtype torch.long")
    if num_components < 0 or num_molecules < 0:
        raise ValueError("component and molecule counts must be nonnegative")
    if reduction not in {"sum", "mean"}:
        raise ValueError(f"unknown component pooling reduction: {reduction}")
    if atom_indices.numel() > 0:
        _validate_tensor_condition(
            ((atom_indices >= 0) & (atom_indices < atom_embeddings.shape[0])).all(),
            "atom_indices references a missing atom embedding",
        )
        _validate_tensor_condition(
            ((atom_component_indices >= 0) & (atom_component_indices < num_components)).all(),
            "atom_component_indices references a missing component",
        )
    if num_components > 0:
        _validate_tensor_condition(
            ((component_batch_indices >= 0) & (component_batch_indices < num_molecules)).all(),
            "component_batch_indices references a missing molecule",
        )

    component_sums = atom_embeddings.new_zeros(
        (num_components, atom_embeddings.shape[1])
    ).index_add(0, atom_component_indices, atom_embeddings[atom_indices])
    component_sizes = th.zeros(
        num_components,
        dtype=th.long,
        device=atom_embeddings.device,
    ).index_add(0, atom_component_indices, th.ones_like(atom_component_indices))
    _validate_tensor_condition(
        (component_sizes > 0).all(),
        "every component must contain at least one atom",
    )

    precursor_sums = atom_embeddings.new_zeros((num_molecules, atom_embeddings.shape[1])).index_add(
        0, atom_batch_indices, atom_embeddings
    )
    precursor_sizes = th.zeros(
        num_molecules,
        dtype=th.long,
        device=atom_embeddings.device,
    ).index_add(0, atom_batch_indices, th.ones_like(atom_batch_indices))
    complement_sizes = precursor_sizes[component_batch_indices] - component_sizes
    complement_sums = precursor_sums[component_batch_indices] - component_sums
    complement_sums = th.where(
        (complement_sizes > 0).unsqueeze(1),
        complement_sums,
        th.zeros_like(complement_sums),
    )

    if reduction == "sum":
        return component_sums, complement_sums
    component_means = component_sums / component_sizes.to(component_sums).unsqueeze(1)
    complement_means = complement_sums / complement_sizes.clamp(min=1).to(
        complement_sums
    ).unsqueeze(1)
    return component_means, complement_means


def pool_boundary_bond_embeddings(
    bond_embeddings: th.Tensor,
    boundary_bond_indices: th.Tensor,
    component_indices: th.Tensor,
    *,
    num_components: int,
    reduction: Literal["sum", "mean"] = "mean",
) -> th.Tensor:
    """Pool sparse boundary-bond/component pairs into component embeddings."""
    return _pool_indexed_embeddings(
        bond_embeddings,
        boundary_bond_indices,
        component_indices,
        num_components=num_components,
        reduction=reduction,
        require_nonempty=False,
    )


def boundary_bond_endpoint_orientation(
    bond_endpoints: th.Tensor,
    boundary_bond_indices: th.Tensor,
    boundary_component_indices: th.Tensor,
    atom_indices: th.Tensor,
    atom_component_indices: th.Tensor,
    *,
    num_atoms: int,
) -> tuple[th.Tensor, th.Tensor]:
    """Split every (boundary bond, component) pair into its retained and lost atom.

    A boundary bond has exactly one endpoint inside the component and one outside it, but the
    candidate builders emit boundary bonds without that orientation. Membership is recovered by
    looking each endpoint up in the sparse ``(component, atom)`` membership relation.

    Args:
        bond_endpoints: Undirected chemical bonds, shape ``(2, num_bonds)``.
        boundary_bond_indices: Bond index of each boundary pair, shape ``(num_pairs,)``.
        boundary_component_indices: Component index of each boundary pair, shape ``(num_pairs,)``.
        atom_indices: Atom index of each membership entry, shape ``(num_membership,)``.
        atom_component_indices: Component index of each membership entry, same shape.
        num_atoms: Total atoms in the batch, used to pack ``(component, atom)`` into one key.

    Returns:
        ``(inside_atom_indices, outside_atom_indices)``, each shape ``(num_pairs,)``.

    Raises:
        ValueError: If the inputs disagree in shape or dtype.
    """
    if bond_endpoints.ndim != 2 or bond_endpoints.shape[0] != 2:
        raise ValueError("bond_endpoints must have shape (2, num_bonds)")
    if boundary_bond_indices.shape != boundary_component_indices.shape:
        raise ValueError("boundary bond and component indices must be matching vectors")
    if atom_indices.shape != atom_component_indices.shape:
        raise ValueError("atom and component membership indices must be matching vectors")
    if any(
        tensor.dtype != th.long
        for tensor in (
            bond_endpoints,
            boundary_bond_indices,
            boundary_component_indices,
            atom_indices,
            atom_component_indices,
        )
    ):
        raise ValueError("boundary orientation indices must have dtype torch.long")
    if num_atoms <= 0:
        raise ValueError("num_atoms must be positive")

    left = bond_endpoints[0][boundary_bond_indices]
    right = bond_endpoints[1][boundary_bond_indices]
    if boundary_bond_indices.numel() == 0:
        return left, right

    membership_keys = th.sort(atom_component_indices * num_atoms + atom_indices).values

    def contains(atoms: th.Tensor) -> th.Tensor:
        keys = boundary_component_indices * num_atoms + atoms
        positions = th.searchsorted(membership_keys, keys).clamp(max=membership_keys.numel() - 1)
        return membership_keys[positions] == keys

    left_inside = contains(left)
    _validate_tensor_condition(
        (left_inside != contains(right)).all(),
        "a boundary bond must have exactly one endpoint inside its component",
    )
    inside = th.where(left_inside, left, right)
    outside = th.where(left_inside, right, left)
    return inside, outside


def component_membership_rows(
    atom_indices: th.Tensor,
    atom_component_indices: th.Tensor,
    query_atoms: th.Tensor,
    query_components: th.Tensor,
    *,
    num_atoms: int,
) -> th.Tensor:
    """Look up the membership row of each queried (component, atom) pair.

    Reuses ``boundary_bond_endpoint_orientation``'s sorted-key/``searchsorted`` idiom, packing
    ``(component, atom)`` into one integer key so a query becomes one sorted lookup.

    Args:
        atom_indices: Atom index of each membership entry, shape ``(M,)``.
        atom_component_indices: Component index of each membership entry, shape ``(M,)``.
        query_atoms: Atom index of each query, shape ``(Q,)``.
        query_components: Component index of each query, shape ``(Q,)``.
        num_atoms: Total atoms in the batch, used to pack ``(component, atom)`` into one key.

    Returns:
        Row index of each query into ``atom_indices``/``atom_component_indices``, shape ``(Q,)``.

    Raises:
        ValueError: If the inputs disagree in shape or dtype, or a queried pair is not a member.
    """
    if atom_indices.shape != atom_component_indices.shape:
        raise ValueError("atom_indices and atom_component_indices must be matching vectors")
    if query_atoms.shape != query_components.shape:
        raise ValueError("query_atoms and query_components must be matching vectors")
    if any(
        tensor.dtype != th.long
        for tensor in (atom_indices, atom_component_indices, query_atoms, query_components)
    ):
        raise ValueError("component_membership_rows indices must have dtype torch.long")
    if num_atoms <= 0:
        raise ValueError("num_atoms must be positive")
    if query_atoms.numel() == 0:
        return th.empty(0, dtype=th.long, device=atom_indices.device)

    membership_keys, order = th.sort(atom_component_indices * num_atoms + atom_indices)
    query_keys = query_components * num_atoms + query_atoms
    positions = th.searchsorted(membership_keys, query_keys).clamp(max=membership_keys.numel() - 1)
    rows = order[positions]
    _validate_tensor_condition(
        (membership_keys[positions] == query_keys).all(),
        "component_membership_rows received a pair that is not a member",
    )
    return rows


def fragment_internal_edges(
    chemical_bond_index: th.Tensor,
    atom_indices: th.Tensor,
    atom_component_indices: th.Tensor,
    *,
    num_atoms: int,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Find each fragment's own internal (both-endpoints-retained) chemical bonds.

    A fragment's internal bonds are the molecule's chemical bonds whose two endpoints are both
    members of that same fragment. A boundary bond -- exactly one endpoint inside -- is excluded
    automatically by the same sorted-key membership test ``boundary_bond_endpoint_orientation``
    uses to recover a boundary bond's retained side, so the two relations are provably disjoint.
    Built fresh from ``chemical_bond_index`` rather than any pre-doubled ``edge_index``, so the
    result depends only on its own arguments.

    Args:
        chemical_bond_index: Undirected chemical bonds, shape ``(2, num_bonds)``.
        atom_indices: Atom index of each (fragment, atom) membership entry, shape ``(M,)``.
        atom_component_indices: Fragment index of each membership entry, shape ``(M,)``.
        num_atoms: Total atoms in the batch, used to pack ``(fragment, atom)`` into one key.

    Returns:
        ``(row_u, row_v, bond_indices)``, each shape ``(E,)``: for every chemical bond retained
        by a fragment (both endpoints members), ``row_u``/``row_v`` are the membership-ROW
        indices (into ``atom_indices``/``atom_component_indices``, not atom ids) of its two
        endpoints, with ``row_u``'s atom id lower than ``row_v``'s, and ``bond_indices`` is the
        chemical-bond index it came from. Each undirected bond contributes exactly one row per
        fragment that retains both its endpoints.

    Raises:
        ValueError: If the inputs disagree in shape or dtype.
    """
    if chemical_bond_index.ndim != 2 or chemical_bond_index.shape[0] != 2:
        raise ValueError("chemical_bond_index must have shape (2, num_bonds)")
    if atom_indices.ndim != 1 or atom_component_indices.shape != atom_indices.shape:
        raise ValueError("atom_indices and atom_component_indices must be matching vectors")
    if any(
        tensor.dtype != th.long
        for tensor in (chemical_bond_index, atom_indices, atom_component_indices)
    ):
        raise ValueError("fragment_internal_edges indices must have dtype torch.long")
    if num_atoms <= 0:
        raise ValueError("num_atoms must be positive")

    device = chemical_bond_index.device
    empty = th.empty(0, dtype=th.long, device=device)
    num_bonds = chemical_bond_index.shape[1]
    if num_bonds == 0 or atom_indices.numel() == 0:
        return empty, empty, empty

    # A directed bucket of every atom's neighbours, sorted by source atom so each atom's
    # neighbours sit in one contiguous run (a CSR layout built fresh from chemical_bond_index).
    directed_src = th.cat([chemical_bond_index[0], chemical_bond_index[1]])
    directed_dst = th.cat([chemical_bond_index[1], chemical_bond_index[0]])
    directed_bond_id = th.cat([th.arange(num_bonds, device=device)] * 2)
    src_order = th.argsort(directed_src, stable=True)
    sorted_dst = directed_dst[src_order]
    sorted_bond_id = directed_bond_id[src_order]
    out_degree = th.bincount(directed_src, minlength=num_atoms)
    csr_offsets = th.zeros(num_atoms + 1, dtype=th.long, device=device)
    csr_offsets[1:] = out_degree.cumsum(0)

    # Expand every membership row by its own atom's out-degree: row m (atom a, fragment f)
    # produces one candidate neighbour per bond incident on a, the ragged-expansion arithmetic
    # `_contextualize_by_molecule` already uses for its own per-molecule buckets.
    num_memberships = atom_indices.shape[0]
    degree_per_row = out_degree[atom_indices]
    expanded_rows = th.repeat_interleave(
        th.arange(num_memberships, device=device),
        degree_per_row,
    )
    if expanded_rows.numel() == 0:
        return empty, empty, empty
    row_starts = degree_per_row.cumsum(0) - degree_per_row
    positions_in_row = th.arange(expanded_rows.shape[0], device=device) - row_starts[expanded_rows]
    csr_index = csr_offsets[atom_indices[expanded_rows]] + positions_in_row
    candidate_dst_atoms = sorted_dst[csr_index]
    candidate_bond_ids = sorted_bond_id[csr_index]
    candidate_src_atoms = atom_indices[expanded_rows]
    candidate_fragments = atom_component_indices[expanded_rows]

    # A candidate survives only if its neighbour is also a member of the SAME fragment (the
    # sorted-key membership test), and only in the canonical (lower atom id first) direction,
    # so each undirected internal bond contributes exactly one row per retaining fragment.
    membership_keys, order = th.sort(atom_component_indices * num_atoms + atom_indices)
    query_keys = candidate_fragments * num_atoms + candidate_dst_atoms
    query_positions = th.searchsorted(membership_keys, query_keys).clamp(
        max=membership_keys.numel() - 1
    )
    neighbour_is_member = membership_keys[query_positions] == query_keys
    canonical_direction = candidate_src_atoms < candidate_dst_atoms
    keep = neighbour_is_member & canonical_direction
    # One nonzero() sync instead of three separate boolean-mask syncs below; an all-False `keep`
    # already falls through to empty tensors here, so no separate early-return is needed.
    keep_idx = keep.nonzero(as_tuple=True)[0]

    row_u = expanded_rows[keep_idx]
    row_v = order[query_positions[keep_idx]]
    bond_indices = candidate_bond_ids[keep_idx]
    return row_u, row_v, bond_indices


def pool_complement_embeddings(
    atom_embeddings: th.Tensor,
    complement_atom_indices: th.Tensor,
    component_indices: th.Tensor,
    *,
    num_components: int,
    reduction: Literal["sum", "mean"] = "mean",
) -> th.Tensor:
    """Pool atoms outside each component, allowing an empty precursor complement."""
    return _pool_indexed_embeddings(
        atom_embeddings,
        complement_atom_indices,
        component_indices,
        num_components=num_components,
        reduction=reduction,
        require_nonempty=False,
    )
