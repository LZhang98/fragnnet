import torch
import torch.nn as nn
import torch.nn.functional as F
from ss_mol_gen.gen_model.transformer_base import CEemb, MolgraphEncoder
from ss_mol_gen.utils.chem_utils import ADDUCT_DICT, ADDUCT_MW, HMASS
from torch_geometric.utils import scatter

MHWP = ADDUCT_MW[ADDUCT_DICT["[M+H]+"]]
MHWN = ADDUCT_MW[ADDUCT_DICT["[M-H]-"]]


def scatter_log_softmax(src, index, dim=0, dim_size=None, eps=1e-16):
    N = dim_size if dim_size is not None else int(index.max()) + 1
    src_max = scatter(src.detach(), index, dim, dim_size=N, reduce="max")
    out = src - src_max.index_select(dim, index)
    out = torch.nan_to_num(out, nan=float("-inf"), posinf=float("inf"), neginf=float("-inf"))
    out_sum = scatter(out.exp(), index, dim, dim_size=N, reduce="sum") + eps
    log_out_sum = out_sum.log().index_select(dim, index)
    return out - log_out_sum


def ppm_dist(a, b):
    return (a - b).abs() / b.clamp(200) * 1e6


def scatter_logsumexp(src, index, dim=0, dim_size=None):
    N = dim_size if dim_size is not None else int(index.max()) + 1
    src_max = scatter(src.detach(), index, dim, dim_size=N, reduce="max")
    src_max = torch.nan_to_num(src_max, neginf=0.0)
    out = (src - src_max.index_select(dim, index)).exp()
    out_sum = out.new_zeros(N).scatter_add(dim, index, out)
    return src_max + out_sum.clamp_min(1e-38).log()


def compute_mol_unexplained(gt_bs, gt_intensity, matched_gt_idx, num_mols, eps=1e-8):
    total_gt = gt_intensity.new_zeros(num_mols).scatter_add(0, gt_bs, gt_intensity)
    explained_gt = gt_intensity.new_zeros(num_mols)
    if matched_gt_idx.numel() > 0:
        explained_gt = explained_gt.scatter_add(
            0, gt_bs[matched_gt_idx], gt_intensity[matched_gt_idx]
        )
    unexplained = (total_gt - explained_gt).clamp_min(0) / total_gt.clamp_min(eps)
    return unexplained.detach()


def edge_decouple_loss(
    canon_logit,
    canon_bs,
    num_mols,
    g_mol,
    frag_edge_owner,
    frag_edge_id,
    matched_frag_idx,
    matched_weight,
    mol_unexplained=None,
    explore_coef=0.1,
    eps=1e-8,
):
    device = canon_logit.device
    n_frag = g_mol.shape[0]
    n_edges_total = canon_logit.shape[0]

    frag_weight = torch.zeros(n_frag, device=device, dtype=matched_weight.dtype)
    if matched_frag_idx.numel() > 0:
        frag_weight[matched_frag_idx] = matched_weight.detach()

    pos_weight = torch.zeros(n_edges_total, device=device, dtype=frag_weight.dtype)
    if frag_edge_owner.numel() > 0:
        pos_weight.scatter_add_(0, frag_edge_id, frag_weight[frag_edge_owner])

    mol_total_weight = torch.zeros(num_mols, device=device, dtype=frag_weight.dtype)
    if matched_frag_idx.numel() > 0:
        mol_total_weight.scatter_add_(0, g_mol[matched_frag_idx], matched_weight.detach())
    edge_mol_total = mol_total_weight[canon_bs]

    has_evidence = edge_mol_total > eps
    target_cut = torch.where(
        has_evidence, pos_weight / edge_mol_total.clamp_min(eps), torch.zeros_like(pos_weight)
    )
    target_logit = torch.logit(target_cut.clamp(eps, 1 - eps))
    floor = F.binary_cross_entropy_with_logits(target_logit, target_cut, reduction="none")

    loss_per_edge = (
        F.binary_cross_entropy_with_logits(canon_logit, target_cut, reduction="none") - floor
    )
    denom = has_evidence.sum().clamp_min(1)
    loss = (loss_per_edge * has_evidence.float()).sum() / denom

    explore_loss = canon_logit.new_zeros(())
    if mol_unexplained is not None:
        no_evidence_edge = pos_weight <= eps
        explore_w = mol_unexplained[canon_bs].clamp_min(0) * no_evidence_edge.float()
        explore_target = torch.ones_like(canon_logit)
        explore_loss_per_edge = F.binary_cross_entropy_with_logits(
            canon_logit, explore_target, reduction="none"
        )
        explore_denom = explore_w.sum().clamp_min(eps)
        explore_loss = (explore_loss_per_edge * explore_w).sum() / explore_denom
        loss = loss + explore_coef * explore_loss

    return loss, target_cut, has_evidence, explore_loss


def distribute_matched_weight_to_frags(edge_aux, matched_pred, matched_gt_idx, gt_intensity):
    merge_group_id = edge_aux["merge_group_id"]
    raw_frag_idx = edge_aux["raw_frag_idx"]
    raw_log_intensity = edge_aux["raw_log_intensity"]
    merged_log_intensity_full = edge_aux["merged_log_intensity_full"]
    n_frag = edge_aux["n_frag"]
    valid_index = edge_aux["valid_index"]

    num_groups = merged_log_intensity_full.shape[0]
    group_weight_full = gt_intensity.new_zeros(num_groups)
    if matched_pred.any():
        matched_group_idx = valid_index[matched_pred]
        matched_group_weight = gt_intensity[matched_gt_idx]
        group_weight_full = group_weight_full.scatter_add(
            0, matched_group_idx, matched_group_weight
        )

    # share of each pre-merge row's predicted mass within its merged group
    within_group_w = (raw_log_intensity - merged_log_intensity_full[merge_group_id]).exp()
    raw_credit = within_group_w * group_weight_full[merge_group_id]

    is_real_frag = raw_frag_idx < n_frag
    raw_credit = raw_credit[is_real_frag]
    raw_frag_of_row = raw_frag_idx[is_real_frag]

    frag_credit = raw_credit.new_zeros(n_frag).scatter_add(0, raw_frag_of_row, raw_credit)
    matched_frag_idx = (frag_credit > 0).nonzero(as_tuple=True)[0]
    matched_weight = frag_credit[matched_frag_idx]
    return matched_frag_idx, matched_weight


def match_and_ce_loss_fast(
    pred_mz,
    pred_log_intensity,
    pred_bs,
    edge_aux,
    gt_mz,
    gt_intensity,
    gt_bs,
    ppm=10.0,
    offset=1e6,
    edge_explore_coef=0.1,
):
    assert gt_mz.numel() > 0

    gt_key = gt_bs.double() * offset + gt_mz.double()
    sort_idx = gt_key.argsort()
    sorted_gt_key = gt_key[sort_idx]
    sorted_gt_mz = gt_mz[sort_idx]
    sorted_gt_bs = gt_bs[sort_idx]
    n_gt = sorted_gt_key.shape[0]

    pred_key = pred_bs.double() * offset + pred_mz.double()
    pos = torch.searchsorted(sorted_gt_key, pred_key)
    pos_l = (pos - 1).clamp(min=0)
    pos_r = pos.clamp(max=n_gt - 1)

    dist_l = ppm_dist(pred_mz, sorted_gt_mz[pos_l]).masked_fill(
        pred_bs != sorted_gt_bs[pos_l], float("inf")
    )
    dist_r = ppm_dist(pred_mz, sorted_gt_mz[pos_r]).masked_fill(
        pred_bs != sorted_gt_bs[pos_r], float("inf")
    )

    use_r = dist_r < dist_l
    best_dist = torch.where(use_r, dist_r, dist_l)
    best_sorted_idx = torch.where(use_r, pos_r, pos_l)

    matched_pred = best_dist <= ppm
    matched_gt_idx = sort_idx[best_sorted_idx[matched_pred]]

    if matched_pred.any():
        cand_dist = best_dist[matched_pred]
        cand_pred_idx = matched_pred.nonzero(as_tuple=True)[0]

        min_dist_per_gt = cand_dist.new_full((n_gt,), float("inf"))
        min_dist_per_gt.scatter_reduce_(
            0, matched_gt_idx, cand_dist, reduce="amin", include_self=True
        )
        is_min_dist = cand_dist <= min_dist_per_gt[matched_gt_idx]

        INF_IDX = cand_pred_idx.max() + 1
        min_idx_per_gt = torch.full(
            (n_gt,), INF_IDX, dtype=cand_pred_idx.dtype, device=cand_pred_idx.device
        )
        min_idx_per_gt.scatter_reduce_(
            0,
            matched_gt_idx[is_min_dist],
            cand_pred_idx[is_min_dist],
            reduce="amin",
            include_self=True,
        )
        is_best = is_min_dist & (cand_pred_idx == min_idx_per_gt[matched_gt_idx])

        matched_pred = torch.zeros_like(matched_pred)
        matched_pred[cand_pred_idx[is_best]] = True
        matched_gt_idx = matched_gt_idx[is_best]
    ori_prob = pred_log_intensity

    pred_log_intensity = scatter_log_softmax(ori_prob, pred_bs, dim=0)
    gt_log_intensity = torch.log(gt_intensity.clamp(1e-6))
    pred_intensity = pred_log_intensity.exp()
    tloss = -(
        gt_intensity[matched_gt_idx] * pred_log_intensity[matched_pred]
    ).sum()  # +pfrag_total.exp()[~matched_pred].sum()*0.1

    edge_loss = None
    edge_explore_loss = None
    if edge_aux is not None:
        matched_frag_idx, matched_weight = distribute_matched_weight_to_frags(
            edge_aux, matched_pred, matched_gt_idx, gt_intensity
        )
        mol_unexplained = compute_mol_unexplained(
            gt_bs, gt_intensity, matched_gt_idx, edge_aux["num_mols"]
        )
        edge_loss, _, _, edge_explore_loss = edge_decouple_loss(
            edge_aux["canon_logit"],
            edge_aux["canon_bs"],
            edge_aux["num_mols"],
            edge_aux["frag_mol"],
            edge_aux["frag_edge_owner"],
            edge_aux["frag_edge_id"],
            matched_frag_idx,
            matched_weight,
            mol_unexplained=mol_unexplained,
            explore_coef=edge_explore_coef,
        )

    pred_intensity_sum = (
        pred_intensity.new_zeros(pred_bs.max() + 1).scatter_reduce(
            0, pred_bs, (pred_intensity**2), reduce="sum"
        )
        ** 0.5
    )
    pred_intensity = pred_intensity / pred_intensity_sum[pred_bs].clamp(1e-6)

    gt_intensity_sum = (
        gt_intensity.new_zeros(gt_bs.max() + 1).scatter_reduce(
            0, gt_bs, (gt_intensity**2), reduce="sum"
        )
        ** 0.5
    )
    gt_intensity = gt_intensity / gt_intensity_sum[gt_bs].clamp(1e-6)

    per_peak_prod = pred_intensity[matched_pred] * gt_intensity[matched_gt_idx]
    spec_cosine = torch.scatter_reduce(
        per_peak_prod.new_zeros(gt_bs.max() + 1),
        0,
        pred_bs[matched_pred],
        per_peak_prod,
        reduce="sum",
    )
    return (
        tloss / (gt_bs.max() + 1),
        ori_prob[matched_pred].exp().sum(),
        spec_cosine,
        matched_gt_idx.shape[0] / gt_bs.shape[0],
        edge_loss,
        edge_explore_loss,
    )


def unique_rows_lex(key):
    n, C = key.shape
    device = key.device
    if n == 0:
        return key, torch.zeros(0, dtype=torch.long, device=device)
    idx = torch.arange(n, device=device)
    for c in range(C - 1, -1, -1):
        order = key[idx, c].argsort(stable=True)
        idx = idx[order]
    sorted_key = key[idx]
    is_new = torch.ones(n, dtype=torch.bool, device=device)
    is_new[1:] = (sorted_key[1:] != sorted_key[:-1]).any(dim=1)
    rep_idx = idx[is_new]
    uniq_key = sorted_key[is_new]
    return uniq_key, rep_idx


def group_rank(score, group, num_groups):
    B = score.shape[0]
    val_order = score.argsort(descending=True)
    grp_by_val = group[val_order]
    grp_order = grp_by_val.argsort(stable=True)
    final_order = val_order[grp_order]
    grp_final = group[final_order]
    counts = torch.bincount(grp_final, minlength=num_groups)
    starts = torch.cumsum(counts, 0) - counts
    positions_in_group = torch.arange(B, device=score.device) - starts[grp_final]
    rank = torch.empty(B, dtype=torch.long, device=score.device)
    rank[final_order] = positions_in_group
    return rank


def enumerate_node_combos(
    canon_logit,
    canon_u,
    canon_v,
    canon_bs,
    bs,
    atom_weight,
    num_H,
    atom_embed,
    num_mols,
    beam_width,
    num_rounds,
    max_order,
    top_k,
    overprovision=4,
    mirror_symmetric=True,
    bond_order=None,
):
    device = canon_logit.device
    H = atom_embed.shape[-1]

    mol_atom_count = torch.bincount(bs, minlength=num_mols)
    mol_atom_offset = torch.zeros(num_mols + 1, dtype=torch.long, device=device)
    mol_atom_offset[1:] = torch.cumsum(mol_atom_count, 0)
    max_atoms = int(mol_atom_count.max().item()) if mol_atom_count.numel() > 0 else 0

    WORD_BITS = 63
    num_words = max(1, (max_atoms + WORD_BITS - 1) // WORD_BITS)

    C_mol = torch.zeros(num_mols, device=device).scatter_add(0, canon_bs, -F.softplus(canon_logit))

    if max_atoms == 0 or canon_logit.numel() == 0:
        empty = atom_weight.new_zeros(0)
        empty_long = mol_atom_count.new_zeros(0)
        return (
            empty,
            mol_atom_count.new_zeros(0),
            atom_embed.new_zeros(0, H),
            empty,
            C_mol,
            empty_long,
            empty_long,
            empty_long,
            empty,
        )

    local_u = canon_u - mol_atom_offset[canon_bs]
    local_v = canon_v - mol_atom_offset[canon_bs]

    n_edges_total = canon_logit.shape[0]
    mol_edge_count = torch.bincount(canon_bs, minlength=num_mols)
    max_edges = int(mol_edge_count.max().item())
    mol_edge_offset = torch.zeros(num_mols + 1, dtype=torch.long, device=device)
    mol_edge_offset[1:] = torch.cumsum(mol_edge_count, 0)
    edge_local_idx = torch.arange(n_edges_total, device=device) - mol_edge_offset[canon_bs]
    edge_u_pad = torch.zeros(num_mols, max(max_edges, 1), dtype=torch.long, device=device)
    edge_v_pad = torch.zeros(num_mols, max(max_edges, 1), dtype=torch.long, device=device)
    edge_valid_pad = torch.zeros(num_mols, max(max_edges, 1), dtype=torch.bool, device=device)
    edge_global_pad = torch.zeros(num_mols, max(max_edges, 1), dtype=torch.long, device=device)

    edge_logit_pad = torch.zeros(num_mols, max(max_edges, 1), device=device)
    edge_u_pad[canon_bs, edge_local_idx] = local_u
    edge_v_pad[canon_bs, edge_local_idx] = local_v
    edge_valid_pad[canon_bs, edge_local_idx] = True
    edge_global_pad[canon_bs, edge_local_idx] = torch.arange(n_edges_total, device=device)
    edge_logit_pad[canon_bs, edge_local_idx] = canon_logit

    adj_bool = torch.zeros(num_mols, max_atoms, max_atoms, dtype=torch.bool, device=device)
    adj_bool[canon_bs, local_u, local_v] = True
    adj_bool[canon_bs, local_v, local_u] = True
    adj_logit = canon_logit.new_zeros(num_mols, max_atoms, max_atoms)
    adj_logit[canon_bs, local_u, local_v] = canon_logit
    adj_logit[canon_bs, local_v, local_u] = canon_logit
    adj_bool_f = adj_bool.float()
    deg_logit = adj_logit.sum(-1)
    deg_count = adj_bool_f.sum(-1)

    atom_valid = torch.arange(max_atoms, device=device).unsqueeze(0) < mol_atom_count.unsqueeze(1)

    atom_idx = torch.arange(max_atoms, device=device)
    atom_word = atom_idx // WORD_BITS
    atom_bit = atom_idx % WORD_BITS
    atom_bitval = (1 << atom_bit).long()

    atom_word_bitmask = torch.zeros(max_atoms, num_words, dtype=torch.long, device=device)
    atom_word_bitmask[atom_idx, atom_word] = atom_bitval

    full_packed = None
    if mirror_symmetric:
        contrib = atom_valid.long() * atom_bitval.unsqueeze(0)
        full_packed = torch.zeros(num_mols, num_words, dtype=torch.long, device=device)
        for w in range(num_words):
            lo = w * WORD_BITS
            hi = min((w + 1) * WORD_BITS, max_atoms)
            full_packed[:, w] = contrib[:, lo:hi].sum(-1)

    cur_mask = torch.zeros(num_mols, 1, max_atoms, dtype=torch.bool, device=device)
    cur_packed = torch.zeros(num_mols, 1, num_words, dtype=torch.long, device=device)

    cur_score = torch.zeros(num_mols, 1, device=device)
    cur_bcount = torch.zeros(num_mols, 1, device=device)
    cur_valid = torch.ones(num_mols, 1, dtype=torch.bool, device=device)

    all_packed, all_score, all_mol = [], [], []

    for r in range(num_rounds):
        cur_beam = cur_mask.shape[1]
        cur_mask_f = cur_mask.float()
        to_F = torch.bmm(cur_mask_f, adj_logit)
        to_F_count = torch.bmm(cur_mask_f, adj_bool_f)

        new_score = cur_score.unsqueeze(-1) + deg_logit.unsqueeze(1) - 2 * to_F
        new_bcount = cur_bcount.unsqueeze(-1) + deg_count.unsqueeze(1) - 2 * to_F_count

        if r == 0:
            can_absorb = atom_valid.unsqueeze(1) & cur_valid.unsqueeze(-1)
        else:
            can_absorb = (
                (to_F_count > 0) & (~cur_mask) & atom_valid.unsqueeze(1) & cur_valid.unsqueeze(-1)
            )

        new_score = torch.where(can_absorb, new_score, torch.full_like(new_score, float("-inf")))

        order_ok = new_bcount <= max_order
        recorded = can_absorb & order_ok
        if recorded.any():
            k_idx, b_idx = torch.meshgrid(
                torch.arange(cur_beam, device=device),
                torch.arange(max_atoms, device=device),
                indexing="ij",
            )
            k_idx = k_idx.unsqueeze(0).expand(num_mols, -1, -1)
            b_idx = b_idx.unsqueeze(0).expand(num_mols, -1, -1)
            mol_idx = (
                torch.arange(num_mols, device=device).view(-1, 1, 1).expand(-1, cur_beam, max_atoms)
            )

            rec_mol = mol_idx[recorded]
            rec_k = k_idx[recorded]
            rec_b = b_idx[recorded]
            rec_score = new_score[recorded]

            rec_packed = cur_packed[rec_mol, rec_k] | atom_word_bitmask[rec_b]
            all_packed.append(rec_packed)
            all_score.append(rec_score)
            all_mol.append(rec_mol)

        if r == num_rounds - 1:
            break

        flat_score = new_score.reshape(num_mols, -1)
        over_k = min(overprovision * beam_width, flat_score.shape[1])
        # this should be argsort for break tie
        top_val, top_flat_idx = torch.topk(flat_score, k=over_k, dim=1)
        top_k_idx = top_flat_idx // max_atoms
        top_b_idx = top_flat_idx % max_atoms

        mol_rep = torch.arange(num_mols, device=device).unsqueeze(1).expand(-1, over_k)
        valid_top = torch.isfinite(top_val)

        mol_rep = mol_rep[valid_top]
        top_k_idx = top_k_idx[valid_top]
        top_b_idx = top_b_idx[valid_top]
        top_val = top_val[valid_top]
        top_bcount = new_bcount[mol_rep, top_k_idx, top_b_idx]

        if mol_rep.numel() == 0:
            cur_mask = torch.zeros(num_mols, 0, max_atoms, dtype=torch.bool, device=device)
            cur_packed = torch.zeros(num_mols, 0, num_words, dtype=torch.long, device=device)
            cur_score = torch.zeros(num_mols, 0, device=device)
            cur_bcount = torch.zeros(num_mols, 0, device=device)
            cur_valid = torch.zeros(num_mols, 0, dtype=torch.bool, device=device)
            break

        new_mask_flat = cur_mask[mol_rep, top_k_idx].clone()
        new_mask_flat[torch.arange(new_mask_flat.shape[0], device=device), top_b_idx] = True
        packed = cur_packed[mol_rep, top_k_idx] | atom_word_bitmask[top_b_idx]

        dedup_key = torch.cat([mol_rep.unsqueeze(-1), packed], dim=-1)
        g_key, rep_idx = unique_rows_lex(dedup_key)

        g_mol_step = g_key[:, 0]
        g_packed_step = g_key[:, 1:]
        g_score_step = top_val[rep_idx]
        g_bcount_step = top_bcount[rep_idx]
        g_mask_step = new_mask_flat[rep_idx]

        rank = group_rank(g_score_step, g_mol_step, num_mols)
        keep = rank < beam_width
        keep_mol = g_mol_step[keep]
        keep_score = g_score_step[keep]
        keep_bcount = g_bcount_step[keep]
        keep_mask = g_mask_step[keep]
        keep_packed = g_packed_step[keep]
        keep_rank = rank[keep]

        next_beam = min(beam_width, int(keep_rank.max().item()) + 1) if keep_rank.numel() > 0 else 0
        cur_mask = torch.zeros(num_mols, next_beam, max_atoms, dtype=torch.bool, device=device)
        cur_packed = torch.zeros(num_mols, next_beam, num_words, dtype=torch.long, device=device)
        cur_score = torch.full((num_mols, next_beam), float("-inf"), device=device)
        cur_bcount = torch.zeros(num_mols, next_beam, device=device)
        cur_valid = torch.zeros(num_mols, next_beam, dtype=torch.bool, device=device)
        if next_beam > 0:
            cur_mask[keep_mol, keep_rank] = keep_mask
            cur_packed[keep_mol, keep_rank] = keep_packed
            cur_score[keep_mol, keep_rank] = keep_score
            cur_bcount[keep_mol, keep_rank] = keep_bcount
            cur_valid[keep_mol, keep_rank] = True

    if not all_packed:
        empty = atom_weight.new_zeros(0)
        empty_long = mol_atom_count.new_zeros(0)
        return (
            empty,
            mol_atom_count.new_zeros(0),
            atom_embed.new_zeros(0, H),
            empty,
            C_mol,
            empty_long,
            empty_long,
            empty_long,
            empty,
        )

    all_packed = torch.cat(all_packed, dim=0)
    all_score = torch.cat(all_score)
    all_mol = torch.cat(all_mol)

    dedup_key = torch.cat([all_mol.unsqueeze(-1), all_packed], dim=-1)
    g_key, rep_idx = unique_rows_lex(dedup_key)

    g_mol = g_key[:, 0]
    g_packed = g_key[:, 1:]
    g_score = all_score[rep_idx]

    popcount = torch.zeros(g_packed.shape[0], dtype=torch.long, device=device)
    for w in range(num_words):
        tmp = g_packed[:, w].clone()
        for _ in range(WORD_BITS):
            popcount += tmp & 1
            tmp = tmp >> 1
    frag_natoms = popcount
    non_trivial = frag_natoms < mol_atom_count[g_mol]

    g_mol = g_mol[non_trivial]
    g_packed = g_packed[non_trivial]
    g_score = g_score[non_trivial]

    rank = group_rank(g_score, g_mol, num_mols)
    keep = rank < top_k
    g_mol, g_packed = g_mol[keep], g_packed[keep]

    if mirror_symmetric and g_mol.shape[0] > 0:
        n_cur = g_mol.shape[0]
        atom_col_cur = g_packed[:, atom_word]
        frag_mask_cur = ((atom_col_cur >> atom_bit.unsqueeze(0)) & 1).bool()
        complement_mask = atom_valid[g_mol] & (~frag_mask_cur)

        local_idx = group_rank(torch.arange(n_cur, device=device).float(), g_mol, num_mols)
        cap = int(local_idx.max().item()) + 1

        comp_padded = torch.zeros(num_mols, cap, max_atoms, dtype=torch.bool, device=device)
        comp_padded[g_mol, local_idx] = complement_mask

        seed_idx = complement_mask.long().argmax(dim=-1)
        reach = torch.zeros(num_mols, cap, max_atoms, dtype=torch.bool, device=device)
        reach[g_mol, local_idx, seed_idx] = True
        reach = reach & comp_padded

        for _r in range(max_atoms):
            step = torch.bmm(reach.float(), adj_bool_f) > 0
            new_reach = (reach | step) & comp_padded
            if torch.equal(new_reach, reach):
                break
            reach = new_reach

        reach_cur = reach[g_mol, local_idx]
        mirror_ok = reach_cur.sum(-1) == complement_mask.sum(-1)

        if mirror_ok.any():
            mirror_mol = g_mol[mirror_ok]
            mirror_packed = full_packed[mirror_mol] ^ g_packed[mirror_ok]
            g_mol = torch.cat([g_mol, mirror_mol])
            g_packed = torch.cat([g_packed, mirror_packed], dim=0)
            merged_key = torch.cat([g_mol.unsqueeze(-1), g_packed], dim=-1)
            mg_key, mrep_idx = unique_rows_lex(merged_key)
            g_mol = mg_key[:, 0]
            g_packed = mg_key[:, 1:]

    n_frag = g_mol.shape[0]
    atom_col = g_packed[:, atom_word]
    frag_mask = ((atom_col >> atom_bit.unsqueeze(0)) & 1).bool()

    global_atom_base = mol_atom_offset[g_mol]
    atom_local = atom_idx.unsqueeze(0).expand(n_frag, -1)
    sel = frag_mask

    flat_local = atom_local[sel]
    flat_owner = torch.arange(n_frag, device=device).unsqueeze(-1).expand(-1, max_atoms)[sel]
    flat_global = global_atom_base[flat_owner] + flat_local

    frag_mz = atom_weight.new_zeros(n_frag).scatter_add(0, flat_owner, atom_weight[flat_global])
    frag_num_h = num_H.new_zeros(n_frag).scatter_add(0, flat_owner, num_H[flat_global])
    frag_embed = atom_embed.new_zeros(n_frag, H).scatter_add(
        0, flat_owner.unsqueeze(-1).expand(-1, H), atom_embed[flat_global]
    )

    frag_edge_u_local = edge_u_pad[g_mol]
    frag_edge_v_local = edge_v_pad[g_mol]
    frag_edge_valid = edge_valid_pad[g_mol]
    frag_edge_global = edge_global_pad[g_mol]
    frag_edge_logit = edge_logit_pad[g_mol]

    u_in_frag = torch.gather(frag_mask, 1, frag_edge_u_local)
    v_in_frag = torch.gather(frag_mask, 1, frag_edge_v_local)
    is_boundary = (u_in_frag != v_in_frag) & frag_edge_valid
    is_internal = u_in_frag & v_in_frag & frag_edge_valid

    frag_edge_owner = (
        torch.arange(n_frag, device=device).unsqueeze(-1).expand(-1, max(max_edges, 1))[is_boundary]
    )
    frag_edge_id = frag_edge_global[is_boundary]

    boundary_term = ((frag_edge_logit - F.softplus(frag_edge_logit)) * is_boundary.float()).sum(-1)
    internal_term = (-F.softplus(frag_edge_logit) * is_internal.float()).sum(-1)
    frag_score = boundary_term + internal_term

    if bond_order is not None and frag_edge_owner.numel() > 0:
        frag_h_cap = atom_weight.new_zeros(n_frag).scatter_add(
            0, frag_edge_owner, bond_order[frag_edge_id]
        )
        frag_h_cap = torch.where(frag_h_cap > 0, frag_h_cap, torch.ones_like(frag_h_cap))
    else:
        frag_h_cap = atom_weight.new_full((n_frag,), 1e6)

    return (
        frag_mz,
        g_mol,
        frag_embed,
        frag_score,
        C_mol,
        frag_edge_owner,
        frag_edge_id,
        frag_num_h,
        frag_h_cap,
    )


def merge_close_mz(mz, log_intensity, bs_idx, ppm=1.0, eps=1e-9):
    device = mz.device
    n = mz.shape[0]
    if n == 0:
        empty_long = torch.zeros(0, dtype=torch.long, device=device)
        return mz, log_intensity, bs_idx, empty_long

    val_order = mz.argsort(descending=False)
    bs_by_val = bs_idx[val_order]
    group_order = bs_by_val.argsort(stable=True)
    final_order = val_order[group_order]

    sorted_mz = mz[final_order]
    sorted_bs = bs_idx[final_order]
    sorted_log_intensity = log_intensity[final_order]

    same_bs = sorted_bs[1:] == sorted_bs[:-1]
    ppm_diff = (sorted_mz[1:] - sorted_mz[:-1]).abs() / sorted_mz[:-1].clamp_min(eps) * 1e6
    connect_to_prev = same_bs & (ppm_diff <= ppm)

    new_group = torch.ones(n, dtype=torch.bool, device=device)
    new_group[1:] = ~connect_to_prev
    sorted_group_id = torch.cumsum(new_group.long(), dim=0) - 1
    num_groups = int(sorted_group_id[-1].item()) + 1

    group_max = sorted_log_intensity.new_full((num_groups,), float("-inf")).scatter_reduce(
        0, sorted_group_id, sorted_log_intensity, reduce="amax", include_self=True
    )
    group_max = torch.nan_to_num(group_max, neginf=0.0)
    shifted = sorted_log_intensity - group_max[sorted_group_id]
    sum_exp = shifted.new_zeros(num_groups).scatter_add(0, sorted_group_id, shifted.exp())
    merged_log_intensity = group_max + sum_exp.clamp_min(1e-38).log()

    w = (sorted_log_intensity - merged_log_intensity[sorted_group_id]).exp()
    merged_mz = sorted_mz.new_zeros(num_groups).scatter_add(0, sorted_group_id, w * sorted_mz)
    merged_bs = torch.zeros(num_groups, dtype=torch.long, device=device).scatter(
        0, sorted_group_id, sorted_bs
    )

    group_id = torch.zeros(n, dtype=torch.long, device=device)
    group_id[final_order] = sorted_group_id
    return merged_mz, merged_log_intensity, merged_bs, group_id


def gumbel_sigmoid_noise(logit, temperature, eps=1e-10):
    u = torch.rand_like(logit).clamp(eps, 1 - eps)
    logistic_noise = torch.log(u) - torch.log1p(-u)
    return (logit + logistic_noise) / temperature


class FastGNN(nn.Module):
    def __init__(
        self,
        layers,
        hidden_size,
        dropout,
        beam_width,
        max_charge,
        num_hs=4,
        max_order=3,
        top_k=2048,
        # gumbel_temperature=10,
        **wargs,
    ):
        super().__init__()
        self.top_k = top_k
        self.max_order = max_order
        self.num_hs = num_hs
        self.beam_width = beam_width

        self.graph_encoder = MolgraphEncoder(
            num_layers=layers,
            hidden_size=hidden_size,
            dropout=dropout,
            max_charge=max_charge,
        )

        self.polarity_emb = nn.Embedding(2, hidden_size)
        self.adduct_emb = nn.Embedding(len(ADDUCT_DICT), hidden_size)
        self.ce_emb = CEemb(hidden_size)
        self.add_hs_out_layer = nn.Sequential(
            nn.LayerNorm(hidden_size * 2 + 1),
            nn.Linear(hidden_size * 2 + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_hs * 2 + 2),
            nn.LogSoftmax(-1),
        )
        self.break_predictor = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.cmf_prob_pos = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            nn.LogSoftmax(-1),
        )
        self.cmf_emb_pos = nn.Embedding(1, hidden_size)

        self.cmf_prob_neg = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            nn.LogSoftmax(-1),
        )
        self.cmf_emb_neg = nn.Embedding(1, hidden_size)

    def add_h(self, emb, ce, mz, prob, bs_idx, frag_num_h, frag_h_cap):
        ret_mz = []
        ret_prob = []
        ret_bs_idx = []
        ori_prob_score = []
        # reshape emb
        emb = torch.cat([emb, ce, prob[..., None]], -1)
        binding_logit = self.add_hs_out_layer(emb)  # (num_graphs, num_hs)
        # end of emb
        valid_pos = (mz > 0).float()
        for h in range(self.num_hs * 2 + 1):
            delta = h - self.num_hs
            valid_h = ((frag_num_h + delta) >= 0).float() * (abs(delta) <= frag_h_cap).float()
            ret_mz.append((mz + delta * HMASS * valid_pos).clamp(0) * valid_h)
            ret_prob.append(prob + binding_logit[:, h])
            ret_bs_idx.append(bs_idx)
            ori_prob_score.append(prob)
        return ret_mz, ret_prob, ret_bs_idx, ori_prob_score

    def forward(self, inps):
        graph = inps["graph"]
        polarity_raw = inps["polarity"]
        polarity = self.polarity_emb(inps["polarity"].long())
        ce = self.ce_emb(inps["ce"].float())
        adduct = self.adduct_emb(inps["adduct"].long())
        atom_weight = graph.atom_weight.float()
        adduct_weight = inps["adduct_weight"].float()
        not_ign_cmf = (inps["ign_cmf"] < 1).float()

        bs = graph.batch
        assert bs.numel() > 0, "empty batch: no atoms in this batch"
        assert graph.num_graphs > 0, "empty batch: no graphs in this batch"
        assert not torch.isnan(atom_weight).any()
        num_H = graph.x[..., 2].long()
        bs_size = int(bs.max().item()) + 1
        _, bs_count = torch.unique(bs, return_counts=True)
        num_atoms = graph.x.shape[0]
        is_ring_edge = graph.is_ring_edge

        edges = graph.edge_index
        assert edges.shape[0] == 2
        graph_embed = self.graph_encoder(graph, (polarity + ce + adduct)[bs])
        mol_embed = graph_embed.new_zeros(bs_size, graph_embed.shape[-1]).scatter_add(
            0, bs.unsqueeze(-1).expand(-1, graph_embed.shape[-1]), graph_embed
        )

        # predict edge logit
        edge_feat = graph_embed[edges[0]] + graph_embed[edges[1]]
        edge_logit = self.break_predictor(edge_feat).squeeze(-1)
        # if self.training:
        #    edge_logit=gumbel_sigmoid_noise(edge_logit, self.gumbel_temperature)
        #    self.gumbel_temperature=max(self.gumbel_temperature*0.999, 1)

        # canonical edges
        cano_mask = edges[0] < edges[1]
        canon_u = edges[0][cano_mask]
        canon_v = edges[1][cano_mask]
        canon_bs = bs[canon_u]
        canon_logit = edge_logit[cano_mask]
        canon_is_ring_edge = is_ring_edge[cano_mask]
        canon_bond_order = graph.bond_order[cano_mask]

        max_atom = bs_count.max()

        (
            frag_mz,
            frag_bs,
            frag_emb,
            frag_score,
            C_mol,
            frag_edge_owner,
            frag_edge_id,
            frag_num_h,
            frag_h_cap,
        ) = enumerate_node_combos(
            canon_logit,
            canon_u,
            canon_v,
            canon_bs,
            bs,
            atom_weight,
            num_H,
            graph_embed,
            num_mols=bs_size,
            num_rounds=(max_atom + 1) // 2,
            beam_width=self.beam_width * self.top_k,
            max_order=self.max_order,
            top_k=self.top_k,
            mirror_symmetric=True,
            bond_order=canon_bond_order,
        )

        n_frag = frag_mz.shape[0]
        frag_mol = frag_bs  # raw fragment -> molecule id, aligned with frag_edge_owner/frag_edge_id

        # get unbreak prob
        unbreak_bs_idx = torch.arange(bs_size, device=atom_weight.device)

        pos_flag = (polarity_raw > 0).float() * not_ign_cmf  # (bs_size,)
        neg_flag = (polarity_raw == 0).float() * not_ign_cmf  # (bs_size,)
        change_flag = not_ign_cmf > 0  # (bs_size,) bool

        frag_change_flag = change_flag[frag_bs]  # (n_frag,)
        frag_pos_flag = pos_flag[frag_bs]
        frag_neg_flag = neg_flag[frag_bs]

        cmf_prob_pos = self.cmf_prob_pos(frag_emb)  # (n_frag, 2)
        cmf_prob_neg = self.cmf_prob_neg(frag_emb)  # (n_frag, 2)

        cmf_frag_mz = (
            frag_mz
            + MHWP * (frag_mz > 0).float() * frag_pos_flag
            + MHWN * (frag_mz > 0).float() * frag_neg_flag
        )

        frag_mz = frag_mz + adduct_weight[frag_bs] * (frag_mz > 0).float()
        unbreak_mz = (
            torch.zeros(bs_size, device=atom_weight.device).scatter_add(0, bs, atom_weight)
            + adduct_weight
        )
        unbreak_num_h = num_H.new_zeros(bs_size).scatter_add(0, bs, num_H)

        frag_score_crf = (
            frag_score + cmf_prob_pos[:, 0] * frag_pos_flag + cmf_prob_neg[:, 0] * frag_neg_flag
        )
        frag_score_cmf = (
            frag_score + cmf_prob_pos[:, 1] * frag_pos_flag + cmf_prob_neg[:, 1] * frag_neg_flag
        )[frag_change_flag]

        merge_frag = torch.cat(
            [
                frag_emb,
                mol_embed,
                (
                    frag_emb
                    + self.cmf_emb_pos.weight * frag_pos_flag[..., None]
                    + self.cmf_emb_neg.weight * frag_neg_flag[..., None]
                )[frag_change_flag],
            ],
            0,
        )
        merge_ce = torch.cat([ce[frag_bs], ce, ce[frag_bs][frag_change_flag]], 0)
        frag_mz = torch.cat([frag_mz, unbreak_mz, cmf_frag_mz[frag_change_flag]], 0)
        frag_num_h = torch.cat([frag_num_h, unbreak_num_h, frag_num_h[frag_change_flag]], 0)
        frag_score = torch.cat([frag_score_crf, C_mol, frag_score_cmf], 0)

        unbreak_h_cap = torch.full((bs_size,), float(self.num_hs), device=atom_weight.device)
        frag_h_cap_full = torch.cat([frag_h_cap, unbreak_h_cap, frag_h_cap[frag_change_flag]], 0)

        frag_bs = torch.cat([frag_bs, unbreak_bs_idx, frag_bs[frag_change_flag]], 0)

        n_total = frag_mz.shape[0]
        frag_mz_list, frag_prob_list, frag_bs_list, ori_prob_list = self.add_h(
            merge_frag, merge_ce, frag_mz, frag_score, frag_bs, frag_num_h, frag_h_cap_full
        )
        raw_mz = torch.cat(frag_mz_list, 0)
        raw_log_intensity = torch.cat(frag_prob_list, 0)
        raw_bs_idx = torch.cat(frag_bs_list, 0)

        num_h_variants = len(frag_mz_list)
        raw_frag_idx = torch.arange(n_total, device=raw_mz.device).repeat(num_h_variants)

        mz, log_intensity, mz_bs_idx, merge_group_id = merge_close_mz(
            raw_mz, raw_log_intensity, raw_bs_idx
        )
        valid_pos = mz > 0

        edge_aux = {
            "canon_logit": canon_logit,
            "canon_bs": canon_bs,
            "num_mols": bs_size,
            "frag_mol": frag_mol,
            "frag_edge_owner": frag_edge_owner,
            "frag_edge_id": frag_edge_id,
            "n_frag": n_frag,
            "raw_frag_idx": raw_frag_idx,
            "merge_group_id": merge_group_id,
            "raw_log_intensity": raw_log_intensity,
            "merged_log_intensity_full": log_intensity,
            "valid_index": valid_pos.nonzero(as_tuple=True)[0],
        }

        return mz[valid_pos], log_intensity[valid_pos], mz_bs_idx[valid_pos], edge_aux
