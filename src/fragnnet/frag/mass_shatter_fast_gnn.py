"""FastGNN-style connected-fragment beam proposals on the model's tensor device.

Search is discrete and detached. Each fragment scores boundary bonds as cut and internal bonds as intact, leaving
outside bonds unrestricted. The model normalizes over retained fragments.
This is not MASShatter's marginalized cut-pathway probability. One adapter row per
fragment reuses the downstream sparse membership contract without topology enumeration.
"""

from __future__ import annotations

from time import perf_counter

import torch as th
import torch.nn.functional as F

from fragnnet.frag.mass_shatter_candidates import (
    MassShatterCandidateBatch,
    MassShatterCandidateTimings,
)


def _unique_rows_lex(key: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    n, C = key.shape
    device = key.device
    if n == 0:
        return key, th.zeros(0, dtype=th.long, device=device)
    idx = th.arange(n, device=device)
    for c in range(C - 1, -1, -1):
        order = key[idx, c].argsort(stable=True)
        idx = idx[order]
    sorted_key = key[idx]
    is_new = th.ones(n, dtype=th.bool, device=device)
    is_new[1:] = (sorted_key[1:] != sorted_key[:-1]).any(dim=1)
    rep_idx = idx[is_new]
    uniq_key = sorted_key[is_new]
    return uniq_key, rep_idx


def _group_rank(score: th.Tensor, group: th.Tensor, num_groups: int) -> th.Tensor:
    B = score.shape[0]
    val_order = score.argsort(descending=True, stable=True)
    grp_by_val = group[val_order]
    grp_order = grp_by_val.argsort(stable=True)
    final_order = val_order[grp_order]
    grp_final = group[final_order]
    counts = th.bincount(grp_final, minlength=num_groups)
    starts = th.cumsum(counts, 0) - counts
    positions_in_group = th.arange(B, device=score.device) - starts[grp_final]
    rank = th.empty(B, dtype=th.long, device=score.device)
    rank[final_order] = positions_in_group
    return rank


@th.no_grad()
def _search(
    canon_logit: th.Tensor,
    canon_u: th.Tensor,
    canon_v: th.Tensor,
    canon_bs: th.Tensor,
    bs: th.Tensor,
    num_mols: int,
    beam_width: int,
    num_rounds: int,
    max_order: int,
    top_k: int,
    overprovision: int = 4,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    device = canon_logit.device

    mol_atom_count = th.bincount(bs, minlength=num_mols)
    mol_atom_offset = th.zeros(num_mols + 1, dtype=th.long, device=device)
    mol_atom_offset[1:] = th.cumsum(mol_atom_count, 0)
    max_atoms = int(mol_atom_count.max().item()) if mol_atom_count.numel() > 0 else 0

    WORD_BITS = 63
    num_words = max(1, (max_atoms + WORD_BITS - 1) // WORD_BITS)

    local_u = canon_u - mol_atom_offset[canon_bs]
    local_v = canon_v - mol_atom_offset[canon_bs]

    n_edges_total = canon_logit.shape[0]
    mol_edge_count = th.bincount(canon_bs, minlength=num_mols)
    max_edges = int(mol_edge_count.max().item())
    mol_edge_offset = th.zeros(num_mols + 1, dtype=th.long, device=device)
    mol_edge_offset[1:] = th.cumsum(mol_edge_count, 0)
    edge_local_idx = th.arange(n_edges_total, device=device) - mol_edge_offset[canon_bs]
    edge_u_pad = th.zeros(num_mols, max(max_edges, 1), dtype=th.long, device=device)
    edge_v_pad = th.zeros(num_mols, max(max_edges, 1), dtype=th.long, device=device)
    edge_valid_pad = th.zeros(num_mols, max(max_edges, 1), dtype=th.bool, device=device)
    edge_global_pad = th.zeros(num_mols, max(max_edges, 1), dtype=th.long, device=device)
    edge_u_pad[canon_bs, edge_local_idx] = local_u
    edge_v_pad[canon_bs, edge_local_idx] = local_v
    edge_valid_pad[canon_bs, edge_local_idx] = True
    edge_global_pad[canon_bs, edge_local_idx] = th.arange(n_edges_total, device=device)

    adj_bool = th.zeros(num_mols, max_atoms, max_atoms, dtype=th.bool, device=device)
    adj_bool[canon_bs, local_u, local_v] = True
    adj_bool[canon_bs, local_v, local_u] = True
    log_cut = F.logsigmoid(canon_logit)
    log_intact = F.logsigmoid(-canon_logit)
    adj_cut = canon_logit.new_zeros(num_mols, max_atoms, max_atoms)
    adj_cut[canon_bs, local_u, local_v] = log_cut
    adj_cut[canon_bs, local_v, local_u] = log_cut
    adj_correction = th.zeros_like(adj_cut)
    adj_correction[canon_bs, local_u, local_v] = log_intact - 2 * log_cut
    adj_correction[canon_bs, local_v, local_u] = log_intact - 2 * log_cut
    adj_bool_f = adj_bool.float()
    deg_cut = adj_cut.sum(-1)  # (num_mols, max_atoms)
    deg_count = adj_bool_f.sum(-1)  # (num_mols, max_atoms)

    atom_valid = th.arange(max_atoms, device=device).unsqueeze(0) < mol_atom_count.unsqueeze(1)

    atom_idx = th.arange(max_atoms, device=device)
    atom_word = atom_idx // WORD_BITS  # (max_atoms,)
    atom_bit = atom_idx % WORD_BITS  # (max_atoms,)
    atom_bitval = (1 << atom_bit).long()  # (max_atoms,)

    atom_word_bitmask = th.zeros(max_atoms, num_words, dtype=th.long, device=device)
    atom_word_bitmask[atom_idx, atom_word] = atom_bitval

    # 1. Pack atom sets into 63-bit words; padding atoms never enter a set.
    contrib = atom_valid.long() * atom_bitval.unsqueeze(0)
    full_packed = th.zeros(num_mols, num_words, dtype=th.long, device=device)
    for w in range(num_words):
        lo = w * WORD_BITS
        hi = min((w + 1) * WORD_BITS, max_atoms)
        full_packed[:, w] = contrib[:, lo:hi].sum(-1)

    cur_mask = th.zeros(num_mols, 1, max_atoms, dtype=th.bool, device=device)
    cur_packed = th.zeros(num_mols, 1, num_words, dtype=th.long, device=device)
    cur_score = th.zeros(num_mols, 1, device=device)
    cur_bcount = th.zeros(num_mols, 1, device=device)
    cur_valid = th.ones(num_mols, 1, dtype=th.bool, device=device)

    all_packed = [full_packed]
    all_score = [canon_logit.new_zeros(num_mols)]
    all_mol = [th.arange(num_mols, device=device)]

    # 2. Grow only connected sets; an over-limit intermediate may become valid later.
    for r in range(num_rounds):
        cur_beam = cur_mask.shape[1]
        cur_mask_f = cur_mask.float()
        to_F = th.bmm(cur_mask_f, adj_correction)  # (num_mols, cur_beam, max_atoms)
        to_F_count = th.bmm(cur_mask_f, adj_bool_f)  # (num_mols, cur_beam, max_atoms)

        # New outside edges become boundary; edges to F change from cut to intact.
        new_score = cur_score.unsqueeze(-1) + deg_cut.unsqueeze(1) + to_F
        new_bcount = cur_bcount.unsqueeze(-1) + deg_count.unsqueeze(1) - 2 * to_F_count

        if r == 0:
            can_absorb = atom_valid.unsqueeze(1) & cur_valid.unsqueeze(-1)
        else:
            can_absorb = (
                (to_F_count > 0) & (~cur_mask) & atom_valid.unsqueeze(1) & cur_valid.unsqueeze(-1)
            )

        # Per-molecule growth depth makes proposals independent of batch padding.
        can_absorb &= (r < (mol_atom_count + 1) // 2)[:, None, None]
        new_score = th.where(can_absorb, new_score, th.full_like(new_score, float("-inf")))

        order_ok = new_bcount <= max_order
        recorded = can_absorb & order_ok
        if recorded.any():
            k_idx, b_idx = th.meshgrid(
                th.arange(cur_beam, device=device),
                th.arange(max_atoms, device=device),
                indexing="ij",
            )
            k_idx = k_idx.unsqueeze(0).expand(num_mols, -1, -1)
            b_idx = b_idx.unsqueeze(0).expand(num_mols, -1, -1)
            mol_idx = (
                th.arange(num_mols, device=device).view(-1, 1, 1).expand(-1, cur_beam, max_atoms)
            )

            rec_mol = mol_idx[recorded]
            rec_k = k_idx[recorded]
            rec_b = b_idx[recorded]
            rec_score = new_score[recorded]

            rec_packed = cur_packed[rec_mol, rec_k] | atom_word_bitmask[rec_b]
            rec_key, rec_rep = _unique_rows_lex(th.cat([rec_mol[:, None], rec_packed], -1))
            rec_score = rec_score[rec_rep]
            rec_keep = _group_rank(rec_score, rec_key[:, 0], num_mols) < top_k
            all_packed.append(rec_key[rec_keep, 1:])
            all_score.append(rec_score[rec_keep])
            all_mol.append(rec_key[rec_keep, 0])

        if r == num_rounds - 1:
            break

        flat_score = new_score.reshape(num_mols, -1)  # (num_mols, cur_beam*max_atoms)
        over_k = min(overprovision * beam_width, flat_score.shape[1])
        top_val, top_flat_idx = th.topk(flat_score, k=over_k, dim=1)
        top_k_idx = top_flat_idx // max_atoms
        top_b_idx = top_flat_idx % max_atoms

        mol_rep = th.arange(num_mols, device=device).unsqueeze(1).expand(-1, over_k)
        valid_top = th.isfinite(top_val)

        mol_rep = mol_rep[valid_top]
        top_k_idx = top_k_idx[valid_top]
        top_b_idx = top_b_idx[valid_top]
        top_val = top_val[valid_top]
        top_bcount = new_bcount[mol_rep, top_k_idx, top_b_idx]

        if mol_rep.numel() == 0:
            cur_mask = th.zeros(num_mols, 0, max_atoms, dtype=th.bool, device=device)
            cur_packed = th.zeros(num_mols, 0, num_words, dtype=th.long, device=device)
            cur_score = th.zeros(num_mols, 0, device=device)
            cur_bcount = th.zeros(num_mols, 0, device=device)
            cur_valid = th.zeros(num_mols, 0, dtype=th.bool, device=device)
            break

        new_mask_flat = cur_mask[mol_rep, top_k_idx].clone()
        new_mask_flat[th.arange(new_mask_flat.shape[0], device=device), top_b_idx] = True
        packed = cur_packed[mol_rep, top_k_idx] | atom_word_bitmask[top_b_idx]  # (n, num_words)

        dedup_key = th.cat([mol_rep.unsqueeze(-1), packed], dim=-1)

        g_key, rep_idx = _unique_rows_lex(dedup_key)

        g_mol = g_key[:, 0]
        g_packed = g_key[:, 1:]
        g_score = top_val[rep_idx]
        g_bcount = top_bcount[rep_idx]
        g_mask = new_mask_flat[rep_idx]

        rank = _group_rank(g_score, g_mol, num_mols)
        keep = rank < beam_width
        keep_mol = g_mol[keep]
        keep_score = g_score[keep]
        keep_bcount = g_bcount[keep]
        keep_mask = g_mask[keep]
        keep_packed = g_packed[keep]
        keep_rank = rank[keep]

        next_beam = min(beam_width, int(keep_rank.max().item()) + 1) if keep_rank.numel() > 0 else 0
        cur_mask = th.zeros(num_mols, next_beam, max_atoms, dtype=th.bool, device=device)
        cur_packed = th.zeros(num_mols, next_beam, num_words, dtype=th.long, device=device)
        cur_score = th.full((num_mols, next_beam), float("-inf"), device=device)
        cur_bcount = th.zeros(num_mols, next_beam, device=device)
        cur_valid = th.zeros(num_mols, next_beam, dtype=th.bool, device=device)
        if next_beam > 0:
            cur_mask[keep_mol, keep_rank] = keep_mask
            cur_packed[keep_mol, keep_rank] = keep_packed
            cur_score[keep_mol, keep_rank] = keep_score
            cur_bcount[keep_mol, keep_rank] = keep_bcount
            cur_valid[keep_mol, keep_rank] = True

    all_packed = th.cat(all_packed, dim=0)  # (n, num_words)
    all_score = th.cat(all_score)
    all_mol = th.cat(all_mol)

    dedup_key = th.cat([all_mol.unsqueeze(-1), all_packed], dim=-1)
    g_key, rep_idx = _unique_rows_lex(dedup_key)

    g_mol = g_key[:, 0]
    g_packed = g_key[:, 1:]  # (n_frag, num_words)
    g_score = all_score[rep_idx]

    frag_natoms = ((g_packed[:, atom_word] >> atom_bit.unsqueeze(0)) & 1).sum(-1)
    non_trivial = frag_natoms < mol_atom_count[g_mol]

    g_mol = g_mol[non_trivial]
    g_packed = g_packed[non_trivial]
    g_score = g_score[non_trivial]

    rank = _group_rank(g_score, g_mol, num_mols)
    keep = rank < top_k
    g_mol, g_packed, g_score = g_mol[keep], g_packed[keep], g_score[keep]

    # 3. Add complements only when they are connected.
    if g_mol.shape[0] > 0:
        n_cur = g_mol.shape[0]
        atom_col_cur = g_packed[:, atom_word]
        frag_mask_cur = ((atom_col_cur >> atom_bit.unsqueeze(0)) & 1).bool()  # (n_cur, max_atoms)
        complement_mask = atom_valid[g_mol] & (~frag_mask_cur)  # (n_cur, max_atoms)

        local_idx = _group_rank(th.arange(n_cur, device=device).float(), g_mol, num_mols)
        cap = int(local_idx.max().item()) + 1

        comp_padded = th.zeros(num_mols, cap, max_atoms, dtype=th.bool, device=device)
        comp_padded[g_mol, local_idx] = complement_mask

        seed_idx = complement_mask.long().argmax(dim=-1)
        reach = th.zeros(num_mols, cap, max_atoms, dtype=th.bool, device=device)
        reach[g_mol, local_idx, seed_idx] = True
        reach = reach & comp_padded

        for _r in range(max_atoms):
            step = th.bmm(reach.float(), adj_bool_f) > 0
            new_reach = (reach | step) & comp_padded
            if th.equal(new_reach, reach):
                break
            reach = new_reach

        reach_cur = reach[g_mol, local_idx]  # (n_cur, max_atoms)
        mirror_ok = reach_cur.sum(-1) == complement_mask.sum(-1)

        if mirror_ok.any():
            mirror_mol = g_mol[mirror_ok]
            mirror_packed = full_packed[mirror_mol] ^ g_packed[mirror_ok]
            mirror_mask = complement_mask[mirror_ok]
            mirror_u = th.gather(mirror_mask, 1, edge_u_pad[mirror_mol])
            mirror_v = th.gather(mirror_mask, 1, edge_v_pad[mirror_mol])
            mirror_valid = edge_valid_pad[mirror_mol]
            # Complements share boundary bonds, but have different internal bonds.
            padded_logits = canon_logit.new_zeros(edge_valid_pad.shape)
            padded_logits[canon_bs, edge_local_idx] = canon_logit
            mirror_logits = padded_logits[mirror_mol]
            mirror_score = (
                F.logsigmoid(mirror_logits) * ((mirror_u != mirror_v) & mirror_valid)
                + F.logsigmoid(-mirror_logits) * (mirror_u & mirror_v & mirror_valid)
            ).sum(-1)

            merged_mol = th.cat([g_mol, mirror_mol])
            merged_packed = th.cat([g_packed, mirror_packed], dim=0)
            merged_score = th.cat([g_score, mirror_score])

            merged_key = th.cat([merged_mol.unsqueeze(-1), merged_packed], dim=-1)
            mg_key, mrep_idx = _unique_rows_lex(merged_key)
            g_mol = mg_key[:, 0]
            g_packed = mg_key[:, 1:]
            g_score = merged_score[mrep_idx]

    # 4. Reserve one slot for the intact precursor, then enforce the final fragment budget.
    rank = _group_rank(g_score, g_mol, num_mols)
    keep = rank < top_k - 1
    g_mol = th.cat([g_mol[keep], th.arange(num_mols, device=device)])
    g_packed = th.cat([g_packed[keep], full_packed])
    order = g_mol.argsort(stable=True)
    g_mol, g_packed = g_mol[order], g_packed[order]
    n_frag = g_mol.shape[0]
    atom_col = g_packed[:, atom_word]  # (n_frag, max_atoms)
    frag_mask = ((atom_col >> atom_bit.unsqueeze(0)) & 1).bool()  # (n_frag, max_atoms)

    global_atom_base = mol_atom_offset[g_mol]  # (n_frag,)
    atom_local = atom_idx.unsqueeze(0).expand(n_frag, -1)
    sel = frag_mask

    flat_local = atom_local[sel]
    flat_owner = th.arange(n_frag, device=device).unsqueeze(-1).expand(-1, max_atoms)[sel]
    flat_global = global_atom_base[flat_owner] + flat_local

    frag_edge_u_local = edge_u_pad[g_mol]
    frag_edge_v_local = edge_v_pad[g_mol]
    u_in = th.gather(frag_mask, 1, frag_edge_u_local)
    v_in = th.gather(frag_mask, 1, frag_edge_v_local)
    is_boundary = (u_in != v_in) & edge_valid_pad[g_mol]
    is_internal = u_in & v_in & edge_valid_pad[g_mol]
    frag_edge_owner = th.arange(n_frag, device=device)[:, None].expand_as(is_boundary)[is_boundary]
    frag_edge_id = edge_global_pad[g_mol][is_boundary]
    internal_owner = th.arange(n_frag, device=device)[:, None].expand_as(is_internal)[is_internal]
    internal_id = edge_global_pad[g_mol][is_internal]
    return g_mol, flat_global, flat_owner, frag_edge_id, frag_edge_owner, internal_id, internal_owner


def build_fast_gnn_candidates(
    bond_logits: th.Tensor,
    chemical_bond_index: th.Tensor,
    atom_batch_indices: th.Tensor,
    atom_ptr: th.Tensor,
    *,
    max_fragments: int,
    beam_width: int,
    max_boundary_cuts: int,
) -> MassShatterCandidateBatch:
    """Search connected atom sets with boundary-cut and internal-intact scores.

    Args:
        bond_logits: One finite logit per chemical bond, shape ``(E,)``.
        chemical_bond_index: Unique undirected bonds, shape ``(2, E)``.
        atom_batch_indices: Contiguous molecule IDs, shape ``(N,)``.
        atom_ptr: Nonempty molecule offsets, shape ``(B + 1,)``.
        max_fragments: Final per-molecule cap, including the intact precursor.
        beam_width: Maximum unique states expanded per molecule and growth round.
        max_boundary_cuts: Maximum boundary bonds for emitted proper fragments.

    Returns:
        Sparse membership adapter with one row per unique fragment. Mask/mechanism
        indices are identity mappings, not latent fragmentation events. Scores are
        differentiable log probabilities of cutting boundary bonds and keeping internal
        bonds intact, with outside bonds unrestricted and no charge factors. Fragment
        events overlap, so their score sum must
        not be interpreted as retained probability mass. Timings measure host wall
        time and do not synchronize asynchronous CUDA kernels.

    Raises:
        ValueError: For invalid budgets, shapes, devices, or graph indices.
    """
    start = perf_counter()
    if any(
        isinstance(v, bool) or not isinstance(v, int)
        for v in (max_fragments, beam_width, max_boundary_cuts)
    ):
        raise ValueError("fragment/beam budgets and boundary cuts must be integers")
    if min(max_fragments, beam_width) < 1 or max_boundary_cuts < 0:
        raise ValueError("fragment/beam budgets must be positive and boundary cuts nonnegative")
    if bond_logits.ndim != 1 or not bond_logits.is_floating_point():
        raise ValueError("fast_gnn requires one floating-point logit per bond")
    if chemical_bond_index.shape != (2, bond_logits.numel()):
        raise ValueError("chemical_bond_index must have shape (2, num_bonds)")
    for value in (chemical_bond_index, atom_batch_indices, atom_ptr):
        if value.dtype != th.long or value.device != bond_logits.device:
            raise ValueError("graph indices must be int64 on the bond-logit device")
    if atom_ptr.ndim != 1 or atom_ptr.numel() < 2 or atom_batch_indices.ndim != 1:
        raise ValueError("invalid atom_ptr or atom_batch_indices shape")
    counts = atom_ptr[1:] - atom_ptr[:-1]
    batch_size = counts.numel()
    if (
        atom_ptr[0] != 0
        or atom_ptr[-1] != atom_batch_indices.numel()
        or (counts <= 0).any()
        or not th.isfinite(bond_logits).all()
    ):
        raise ValueError("fast_gnn requires nonempty molecules and finite bond logits")
    expected_batch = th.repeat_interleave(th.arange(batch_size, device=counts.device), counts)
    if not th.equal(atom_batch_indices, expected_batch):
        raise ValueError("atoms must be contiguous within each molecule")
    if chemical_bond_index.numel() and (
        (chemical_bond_index < 0).any() or (chemical_bond_index >= atom_batch_indices.numel()).any()
    ):
        raise ValueError("bond endpoint out of range")
    u, v = chemical_bond_index
    bond_batch = atom_batch_indices[u]
    if (u == v).any() or not th.equal(bond_batch, atom_batch_indices[v]):
        raise ValueError("bonds must join distinct atoms in the same molecule")
    canonical = chemical_bond_index.sort(dim=0).values.T
    if th.unique(canonical, dim=0).shape[0] != bond_logits.numel():
        raise ValueError("fast_gnn requires unique chemical bonds")
    # Search expects bond rows grouped by molecule; map memberships back afterwards.
    order = bond_batch.argsort(stable=True)
    max_atoms = int(counts.max().item())
    selection_start = perf_counter()
    with th.autocast(device_type=bond_logits.device.type, enabled=False):
        search_logits = bond_logits.detach().float()
        (
            frag_batch,
            atoms,
            owners,
            sorted_bonds,
            boundary_owners,
            sorted_internal,
            internal_owners,
        ) = _search(
            search_logits[order],
            u[order],
            v[order],
            bond_batch[order],
            atom_batch_indices,
            batch_size,
            beam_width,
            (max_atoms + 1) // 2,
            max_boundary_cuts,
            max_fragments,
        )
    boundary_bonds = order[sorted_bonds]
    scoring_start = perf_counter()
    num_fragments = frag_batch.numel()
    ids = th.arange(num_fragments, device=bond_logits.device)
    # Recompute from live logits: search decisions never retain an autograd graph.
    live_logits = (
        bond_logits.float() if bond_logits.dtype in (th.float16, th.bfloat16) else bond_logits
    )
    scores = live_logits.new_zeros(num_fragments).index_add(
        0, boundary_owners, F.logsigmoid(live_logits[boundary_bonds])
    ).index_add(
        0, internal_owners, F.logsigmoid(-live_logits[order[sorted_internal]])
    )
    cuts = th.bincount(boundary_owners, minlength=num_fragments)
    fragment_counts = th.bincount(frag_batch, minlength=batch_size)
    single_fraction = (
        scores.new_zeros(batch_size).index_add(0, frag_batch, (cuts == 1).to(scores))
        / fragment_counts
    )
    count_tuple = tuple(fragment_counts.tolist())
    finish = perf_counter()
    timings = MassShatterCandidateTimings(
        input_transfer_ms=0.0,
        mask_selection_ms=(scoring_start - selection_start) * 1000,
        topology_assembly_ms=0.0,
        forced_merge_ms=0.0,
        smarts_assembly_ms=0.0,
        batch_packing_ms=0.0,
        device_transfer_ms=0.0,
        probability_scoring_ms=(finish - scoring_start) * 1000,
        output_transfer_ms=(finish - scoring_start) * 1000,
        total_ms=(finish - start) * 1000,
    )
    return MassShatterCandidateBatch(
        mask_log_probabilities=scores,
        mask_batch_indices=frag_batch,
        mask_cut_counts=cuts,
        mechanism_mask_indices=ids,
        mechanism_rule_indices=th.zeros_like(ids),
        mechanism_boundary_bond_indices=boundary_bonds,
        boundary_mechanism_indices=boundary_owners,
        component_mask_indices=ids,
        component_mechanism_indices=ids,
        component_batch_indices=frag_batch,
        component_cut_counts=cuts,
        component_unique_indices=ids,
        unique_representative_component_indices=ids,
        atom_indices=atoms,
        atom_component_indices=owners,
        boundary_bond_indices=boundary_bonds,
        boundary_component_indices=boundary_owners,
        proposal_single_cut_fraction=single_fraction,
        unique_component_count=fragment_counts,
        component_counts=count_tuple,
        unique_component_counts=count_tuple,
        selected_mask_counts=count_tuple,
        retained_mask_counts=count_tuple,
        scanned_mask_counts=count_tuple,
        timings=timings,
    )
