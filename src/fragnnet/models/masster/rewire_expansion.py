"""After-selection stage of the MASSter rewire rules: rule products of the kept candidate pool.

For every kept ordinary fragment F of a spectrum and every rule match whose mapped atoms all lie in F, the stage
applies the rule's edit (delete atoms, cut bonds, form bonds) and returns the connected pieces that hold a formed
bond ("products"). Products already in the kept pool are MERGED into their ordinary twin (indicators and a
valence adjustment, no new row); the others are EXTRAS, appended to the candidate set subject to a per-spectrum
relative cap. Pure torch on the device of the inputs, integer ops only, no autograd.

Reference semantics: ``scripts/debug/smarts/debug_masster_rewire_reach_ceiling.py`` (``edited_products`` for
depth 0, ``one_cut_pieces`` for depth 1; production depth 1 excludes rule-cut bonds, see the implementation spec,
correction C3). Dedup identity is the atom set alone.

Ordinary rows with identical atom sets inside one spectrum are tolerated (SMARTS-appended rows are added after
``fragment_keep`` and could in principle repeat a ranked mask): all such rows are treated as twins of the product.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch as th
import torch.nn.functional as F

from fragnnet.models.masster.candidates.rewire_rules import RewireMatchSet

BITS_PER_WORD = 62  # nonnegative int64 words
EDIT_FEATURE_DIM = 4  # n_formed, n_internal_cut, n_deleted_heavy, depth
_PARENT_KEY_SCALE = 2**16  # quantum of the parent log-probability sort key
_ROW_BITS = 2**31 - 1

ChannelFn = Callable[
    [th.Tensor, th.Tensor, th.Tensor, int, th.Tensor, th.Tensor],
    tuple[th.Tensor, th.Tensor],
]  # (atom_indices, atom_fragment_indices, fragment_batch_indices, num_fragments, cut_counts, valence_adjustment)
#    -> (ion_mzs (n, M, H), valid_channels (n, M, H) bool)

REWIRE_DIAGNOSTIC_NAMES: tuple[str, ...] = (
    "rewire_match_count",
    "rewire_truncated_match_count",
    "rewire_pair_count",
    "rewire_contained_pair_count",
    "rewire_precap_extra_count",
    "rewire_merged_count",
    "rewire_dropped_no_valid_count",
    "rewire_dropped_cap_count",
    "rewire_extra_count",
    "rewire_cap_tie_count",
)

# Largest rows-per-chunk and element count seen by the last call (read by the memory-bound test).
_LAST_STATS: dict[str, int] = {"max_chunk_rows": 0, "chunk_rows_budget": 0, "num_host_chunks": 0}


@dataclass(frozen=True)
class RewireExpansion:
    num_extra: int  # E
    extra_atom_indices: th.Tensor  # (Mx,) long, GLOBAL atom ids of every extra's membership
    extra_atom_extra_indices: th.Tensor  # (Mx,) long in [0, E)
    extra_boundary_bond_indices: th.Tensor  # (Bx,) long, columns of chemical_bond_index
    extra_boundary_extra_indices: th.Tensor  # (Bx,) long in [0, E)
    extra_batch_indices: th.Tensor  # (E,) long spectrum
    extra_cut_counts: th.Tensor  # (E,) long = |Boundary(P)|
    extra_valence_adjustment: th.Tensor  # (E,) long = -2 * (n_formed - n_internal_cut), merged max
    extra_ion_mzs: th.Tensor  # (E, M, H) the channel_fn dtype
    extra_valid_channels: th.Tensor  # (E, M, H) bool, every row has >= 1 valid channel
    extra_rule_indicators: th.Tensor  # (E, R) float32 0/1
    extra_edit_features: th.Tensor  # (E, 4) float32
    extra_parent: th.Tensor  # (E,) long, ordinary post-keep row of the best parent
    merged_rows: th.Tensor  # (K,) long, ordinary post-keep rows that have a product twin
    merged_valid_channels: th.Tensor  # (K, M, H) bool, OR into the ordinary row's valid_channels
    merged_rule_indicators: th.Tensor  # (K, R) float32
    merged_edit_features: th.Tensor  # (K, 4) float32
    diagnostics: dict[str, th.Tensor]  # name -> (batch_size,) float32


# ---- helpers ------------------------------------------------------------------------------------


def atom_set_words(member: th.Tensor, num_words: int) -> th.Tensor:
    """(n, A) bool -> (n, W) int64 bitset words of 62 bits."""
    n, a = member.shape
    m = F.pad(member.to(th.long), (0, num_words * BITS_PER_WORD - a)).view(n, num_words, BITS_PER_WORD)
    weights = th.bitwise_left_shift(
        th.ones(BITS_PER_WORD, dtype=th.long, device=member.device),
        th.arange(BITS_PER_WORD, device=member.device),
    )
    return (m * weights).sum(-1)


def words_to_member(words: th.Tensor, num_atoms: int) -> th.Tensor:
    """Inverse of ``atom_set_words``."""
    n, w = words.shape
    shifts = th.arange(BITS_PER_WORD, device=words.device)
    bits = (th.bitwise_right_shift(words[:, :, None], shifts) & 1).bool()
    return bits.reshape(n, w * BITS_PER_WORD)[:, :num_atoms]


def lexsort(keys: Sequence[th.Tensor]) -> th.Tensor:
    """Permutation sorting by keys[0] (most significant), then keys[1], ...; all ascending, stable."""
    order = th.arange(keys[0].shape[0], device=keys[0].device)
    for key in reversed(keys):
        order = order[th.argsort(key[order], stable=True)]
    return order


def label_components(
    member: th.Tensor,
    edge_row: th.Tensor,
    edge_a: th.Tensor,
    edge_b: th.Tensor,
    max_sweeps: int | None = None,
) -> th.Tensor:
    """(R, A) long: smallest atom id of each atom's connected component, -1 outside ``member``."""
    r, a = member.shape
    ids = th.arange(a, device=member.device).expand(r, a)
    labels = th.where(member, ids, th.full_like(ids, a)).contiguous()  # a = sentinel
    if edge_row.numel() == 0:
        return th.where(member, labels, th.full_like(labels, -1))
    src = th.cat([edge_row * a + edge_a, edge_row * a + edge_b])
    dst = th.cat([edge_row * a + edge_b, edge_row * a + edge_a])
    num_sweeps = a + 1 if max_sweeps is None else max_sweeps
    for sweep in range(num_sweeps):
        flat = labels.reshape(-1)
        new = flat.clone().scatter_reduce(0, dst, flat[src], reduce="amin", include_self=True).view(r, a)
        jumped = th.minimum(new, new.gather(1, new.clamp(max=a - 1)))  # pointer jumping
        jumped = th.where(new == a, new, jumped)
        if (sweep % 4 == 3 or sweep == num_sweeps - 1) and th.equal(jumped, labels):  # one sync every 4 sweeps
            break
        labels = jumped.contiguous()
    else:
        raise ValueError("component labelling did not converge within the atom count")
    return th.where(member, labels, th.full_like(labels, -1))


def extra_limits(kept_rows: np.ndarray, max_extra_fraction: float, max_extra: int) -> np.ndarray:
    """Per-spectrum extra budget ``min(max_extra, floor(fraction * kept ordinary rows))``, in int64 on the host.

    The epsilon keeps products like ``0.29 * 100 = 28.999999999999996`` from flooring one low.
    """
    return np.minimum(np.floor(max_extra_fraction * kept_rows + 1e-9).astype(np.int64), int(max_extra))


# ---- empty result -------------------------------------------------------------------------------


def _host_diagnostics(match_sets: Sequence[RewireMatchSet], device: th.device) -> dict[str, th.Tensor]:
    diag = {name: th.zeros(len(match_sets), dtype=th.float32, device=device) for name in REWIRE_DIAGNOSTIC_NAMES}
    diag["rewire_match_count"] = th.tensor([len(m.matches) for m in match_sets], dtype=th.float32, device=device)
    diag["rewire_truncated_match_count"] = th.tensor(
        [m.num_truncated for m in match_sets], dtype=th.float32, device=device
    )
    return diag


def _empty_expansion(
    device: th.device, num_rules: int, diagnostics: dict[str, th.Tensor], channel_shape: tuple[int, int] = (0, 0),
    ion_dtype: th.dtype = th.float32,
) -> RewireExpansion:
    long0 = th.zeros(0, dtype=th.long, device=device)
    m, h = channel_shape
    return RewireExpansion(
        num_extra=0,
        extra_atom_indices=long0,
        extra_atom_extra_indices=long0,
        extra_boundary_bond_indices=long0,
        extra_boundary_extra_indices=long0,
        extra_batch_indices=long0,
        extra_cut_counts=long0,
        extra_valence_adjustment=long0,
        extra_ion_mzs=th.zeros((0, m, h), dtype=ion_dtype, device=device),
        extra_valid_channels=th.zeros((0, m, h), dtype=th.bool, device=device),
        extra_rule_indicators=th.zeros((0, num_rules), dtype=th.float32, device=device),
        extra_edit_features=th.zeros((0, EDIT_FEATURE_DIM), dtype=th.float32, device=device),
        extra_parent=long0,
        merged_rows=long0,
        merged_valid_channels=th.zeros((0, m, h), dtype=th.bool, device=device),
        merged_rule_indicators=th.zeros((0, num_rules), dtype=th.float32, device=device),
        merged_edit_features=th.zeros((0, EDIT_FEATURE_DIM), dtype=th.float32, device=device),
        diagnostics=diagnostics,
    )


# ---- host checks and packing --------------------------------------------------------------------


def _validate_host(
    match_sets: Sequence[RewireMatchSet],
    active: list[int],
    atom_ptr_host: np.ndarray,
    atom_numbers_host: np.ndarray,
    host_bond_index: th.Tensor,
) -> None:
    """Every violation is a ValueError naming the spectrum (and rule)."""
    n_atoms = np.diff(atom_ptr_host)
    bu = host_bond_index[0].numpy()
    bv = host_bond_index[1].numpy()
    bond_batch = np.searchsorted(atom_ptr_host[1:], bu, side="right")
    if bu.size and (
        (np.diff(bond_batch) < 0).any() or (np.searchsorted(atom_ptr_host[1:], bv, side="right") != bond_batch).any()
    ):
        raise ValueError("chemical_bond_index must be sorted by spectrum with both endpoints in one spectrum")
    total = int(atom_ptr_host[-1]) + 1
    bond_keys = np.minimum(bu, bv).astype(np.int64) * total + np.maximum(bu, bv)
    for s in active:
        base = int(atom_ptr_host[s])
        n = int(n_atoms[s])
        for match in match_sets[s].matches:
            where = f"spectrum {s} rule {match.rule_index}"
            ids = list(match.motif_atoms) + list(match.deleted)
            ids += [a for p in match.cut_pairs for a in p] + [a for p in match.form_pairs for a in p]
            if any(not 0 <= a < n for a in ids):
                raise ValueError(f"{where}: a match atom id is outside the molecule ({n} atoms)")
            if len(match.motif_atoms) != len(match.motif_numbers):
                raise ValueError(f"{where}: motif_atoms and motif_numbers differ in length")
            graph_numbers = atom_numbers_host[base + np.asarray(match.motif_atoms, dtype=np.int64)]
            if not np.array_equal(graph_numbers, np.asarray(match.motif_numbers, dtype=np.int64)):
                raise ValueError(
                    f"{where}: matched elements differ from the graph atoms (RDKit and graph atom order disagree)"
                )
            if match.cut_pairs:
                keys = np.asarray(
                    [(base + min(p)) * total + base + max(p) for p in match.cut_pairs], dtype=np.int64
                )
                missing = ~np.isin(keys, bond_keys)
                if missing.any():
                    raise ValueError(f"{where}: cut pair {match.cut_pairs[int(np.argmax(missing))]} is not a bond")


def _pack_matches(match_sets: Sequence[RewireMatchSet], active: list[int], amax: int, device: th.device) -> SimpleNamespace:
    flat = [(s, m) for s in active for m in match_sets[s].matches]
    mt = len(flat)
    kmax = max(len(m.motif_atoms) for _, m in flat)
    cmax = max(1, max(len(m.cut_pairs) for _, m in flat))
    fmax = max(1, max(len(m.form_pairs) for _, m in flat))
    spec = np.zeros(mt, dtype=np.int64)
    rule = np.zeros(mt, dtype=np.int64)
    motif = np.full((mt, kmax), -1, dtype=np.int64)
    deleted = np.zeros((mt, amax), dtype=bool)
    cut_u = np.full((mt, cmax), -1, dtype=np.int64)
    cut_v = np.full((mt, cmax), -1, dtype=np.int64)
    form_u = np.full((mt, fmax), -1, dtype=np.int64)
    form_v = np.full((mt, fmax), -1, dtype=np.int64)
    ndel = np.zeros(mt, dtype=np.int64)
    for i, (s, m) in enumerate(flat):
        spec[i], rule[i], ndel[i] = s, m.rule_index, m.n_deleted_heavy
        motif[i, : len(m.motif_atoms)] = m.motif_atoms
        deleted[i, list(m.deleted)] = True
        for k, (a, b) in enumerate(m.cut_pairs):
            cut_u[i, k], cut_v[i, k] = a, b
        for k, (a, b) in enumerate(m.form_pairs):
            form_u[i, k], form_v[i, k] = a, b

    def to(x):
        return th.from_numpy(x).to(device)

    return SimpleNamespace(
        spec=to(spec), rule=to(rule), motif=to(motif), deleted=to(deleted), cut_u=to(cut_u), cut_v=to(cut_v),
        form_u=to(form_u), form_v=to(form_v), ndel=to(ndel),
    )


# ---- component labelling on edited graphs -------------------------------------------------------


def _labels(
    ctx: SimpleNamespace,
    p_mem: th.Tensor,
    spec: th.Tensor,
    cu: th.Tensor,
    cv: th.Tensor,
    fu: th.Tensor,
    fv: th.Tensor,
    form_valid: th.Tensor,
    removed_bond: th.Tensor | None = None,
) -> th.Tensor:
    """Component labels of P minus rule cuts (and one removed bond) plus the formed bonds, per row."""
    rows = p_mem.shape[0]
    cnt = ctx.bond_cnt[spec]
    er = th.repeat_interleave(th.arange(rows, device=p_mem.device), cnt)
    off = th.arange(er.numel(), device=p_mem.device) - (cnt.cumsum(0) - cnt)[er]
    eb = ctx.bond_ptr[spec][er] + off
    u, v = ctx.bond_u[eb], ctx.bond_v[eb]
    keep = p_mem[er, u] & p_mem[er, v]
    cue, cve = cu[er], cv[er]
    is_cut = (((cue == u[:, None]) & (cve == v[:, None])) | ((cue == v[:, None]) & (cve == u[:, None]))).any(1)
    keep &= ~is_cut
    if removed_bond is not None:
        keep &= eb != removed_bond[er]
    er, u, v = er[keep], u[keep], v[keep]
    fr, fk = th.nonzero(form_valid, as_tuple=True)
    er = th.cat([er, fr])
    u = th.cat([u, fu[fr, fk]])
    v = th.cat([v, fv[fr, fk]])
    return label_components(p_mem, er, u, v)


def _emit(
    p_mem: th.Tensor,
    lab: th.Tensor,
    spec: th.Tensor,
    match: th.Tensor,
    ctx: SimpleNamespace,
    depth: int,
    parent_key: th.Tensor,
    strict: bool,
) -> dict[str, th.Tensor]:
    """One piece per distinct component that holds a formed bond of the row's match."""
    dev = p_mem.device
    rows = p_mem.shape[0]
    fu, fv = ctx.form_u[match], ctx.form_v[match]
    cu, cv = ctx.cut_u[match], ctx.cut_v[match]
    rr = th.arange(rows, device=dev)[:, None].expand_as(fu)
    present = fu >= 0
    inside = p_mem[rr, fu.clamp_min(0)] & p_mem[rr, fv.clamp_min(0)]
    if strict and bool((present & ~inside).any()):
        raise ValueError("a formed-bond endpoint is outside the edited fragment (rule/graph mismatch)")
    form_valid = present & inside
    fl = th.where(form_valid, lab[rr, fu.clamp_min(0)], th.full_like(fu, -2))
    same = (fl[:, :, None] == fl[:, None, :]) & (fl[:, :, None] >= 0)
    slot = th.arange(fl.shape[1], device=dev)
    earlier = (same & (slot[None, None, :] < slot[None, :, None])).any(2)  # [0, i, j] is j < i
    pr, pk = th.nonzero((fl >= 0) & ~earlier, as_tuple=True)
    plabel = fl[pr, pk]
    pmem = lab[pr] == plabel[:, None]
    n_formed = (fl[pr] == plabel[:, None]).sum(1)
    pi = th.arange(pr.numel(), device=dev)[:, None]
    cu_p, cv_p = cu[pr], cv[pr]
    internal = (cu_p >= 0) & pmem[pi.expand_as(cu_p), cu_p.clamp_min(0)] & pmem[pi.expand_as(cv_p), cv_p.clamp_min(0)]
    n_cut = internal.sum(1)
    return {
        "member": pmem,
        "batch": spec[pr],
        "match": match[pr],
        "rule": ctx.rule[match[pr]],
        "n_formed": n_formed,
        "n_cut": n_cut,
        "ndel": ctx.ndel[match[pr]],
        "depth": th.full_like(n_formed, depth),
        "parent_key": parent_key[pr],
    }


def _reduce_pieces(rec: dict[str, th.Tensor], num_words: int, num_rules: int) -> dict[str, th.Tensor]:
    """Group records on (batch, atom-set words) and keep per-group statistics only."""
    dev = rec["batch"].device
    n = rec["batch"].shape[0]
    if n == 0:
        return rec
    order = lexsort([rec["batch"], *[rec["words"][:, w] for w in range(num_words)]])
    sb, sw = rec["batch"][order], rec["words"][order]
    new = th.ones(n, dtype=th.bool, device=dev)
    new[1:] = (sw[1:] != sw[:-1]).any(1) | (sb[1:] != sb[:-1])
    gid = new.long().cumsum(0) - 1
    g = int(gid[-1]) + 1

    def reduce(x, how, init):
        return th.full((g,), init, dtype=x.dtype, device=dev).scatter_reduce(0, gid, x[order], how, include_self=True)

    big = 2**62
    ind = th.zeros((g, num_rules), dtype=th.uint8, device=dev).scatter_reduce(
        0, gid[:, None].expand(-1, num_rules), rec["ind"][order], "amax", include_self=True
    )
    return {
        "batch": sb[new],
        "words": sw[new],
        "ind": ind,
        "n_formed": reduce(rec["n_formed"], "amax", 0),
        "n_cut": reduce(rec["n_cut"], "amax", 0),
        "ndel": reduce(rec["ndel"], "amax", 0),
        "depth": reduce(rec["depth"], "amin", big),
        "adj": reduce(rec["adj"], "amax", -big),
        "parent_key": reduce(rec["parent_key"], "amax", -big),
    }


def _concat_records(parts: list[dict[str, th.Tensor]]) -> dict[str, th.Tensor]:
    return {k: th.cat([p[k] for p in parts]) for k in parts[0]}


# ---- the stage ----------------------------------------------------------------------------------


def expand_rewire_candidates(
    *,
    match_sets: Sequence[RewireMatchSet],
    atom_ptr: th.Tensor,
    atom_numbers: th.Tensor,
    chemical_bond_index: th.Tensor,
    host_chemical_bond_index: th.Tensor,
    fragment_atom_indices: th.Tensor,
    atom_fragment_indices: th.Tensor,
    fragment_batch_indices: th.Tensor,
    parent_log_probabilities: th.Tensor,
    parent_blocked: th.Tensor | None,
    bond_logits: th.Tensor | None,
    channel_fn: ChannelFn,
    max_extra_fraction: float,
    max_extra: int,
    continuation_cuts: int,
    pair_chunk: int,
    num_rules: int,
) -> RewireExpansion:
    """Rule products of the kept pool, merged into ordinary twins or appended as capped extras (see module doc)."""
    device = atom_ptr.device
    batch_size = atom_ptr.numel() - 1
    n_ord = int(fragment_batch_indices.numel())
    if len(match_sets) != batch_size:
        raise ValueError(f"expected {batch_size} match sets (one per spectrum), got {len(match_sets)}")
    if continuation_cuts > 0 and bond_logits is None:
        raise ValueError("rewire continuation_cuts > 0 needs the detached bond logits")
    if pair_chunk < 1:
        raise ValueError("pair_chunk must be >= 1")
    if parent_log_probabilities.shape != (n_ord,):
        raise ValueError("parent_log_probabilities must have one value per ordinary row")
    diag = _host_diagnostics(match_sets, device)
    active = [s for s in range(batch_size) if match_sets[s].matches]
    if not active:
        return _empty_expansion(device, num_rules, diag)

    # 1. host checks
    atom_ptr_host = atom_ptr.cpu().numpy().astype(np.int64)
    n_atoms = th.diff(atom_ptr)
    _validate_host(match_sets, active, atom_ptr_host, atom_numbers.cpu().numpy(), host_chemical_bond_index)
    if host_chemical_bond_index.shape != chemical_bond_index.shape or not bool(
        th.equal(host_chemical_bond_index.cpu(), chemical_bond_index.cpu())
    ):
        raise ValueError("host_chemical_bond_index must equal chemical_bond_index")
    if n_ord and bool(
        ((fragment_atom_indices < atom_ptr[fragment_batch_indices[atom_fragment_indices]])
         | (fragment_atom_indices >= atom_ptr[fragment_batch_indices[atom_fragment_indices] + 1])).any()
    ):
        raise ValueError("an ordinary fragment membership atom lies outside its spectrum")

    # 2. pack matches; per-spectrum bond tables
    amax = int(max(atom_ptr_host[s + 1] - atom_ptr_host[s] for s in active))
    num_words = -(-amax // BITS_PER_WORD)
    pack = _pack_matches(match_sets, active, amax, device)
    bond_batch = th.bucketize(chemical_bond_index[0].contiguous(), atom_ptr[1:], right=True)
    bond_cnt = th.bincount(bond_batch, minlength=batch_size)
    bond_ptr = th.cat([bond_cnt.new_zeros(1), bond_cnt.cumsum(0)])[:-1]
    ctx = SimpleNamespace(
        bond_cnt=bond_cnt, bond_ptr=bond_ptr,
        bond_u=chemical_bond_index[0] - atom_ptr[bond_batch], bond_v=chemical_bond_index[1] - atom_ptr[bond_batch],
        rule=pack.rule, ndel=pack.ndel, cut_u=pack.cut_u, cut_v=pack.cut_v, form_u=pack.form_u, form_v=pack.form_v,
    )
    match_count = th.bincount(pack.spec, minlength=batch_size)
    match_start = th.cat([match_count.new_zeros(1), match_count.cumsum(0)])[:-1]

    # 3. dense membership of the ordinary rows of active spectra
    is_active = th.zeros(batch_size, dtype=th.bool, device=device)
    is_active[th.tensor(active, device=device)] = True
    active_frag = is_active[fragment_batch_indices]
    frag_row = active_frag.long().cumsum(0) - 1
    num_rows = int(active_frag.sum())  # host sync 1
    active_ids = th.nonzero(active_frag).flatten()
    member = th.zeros((num_rows, amax), dtype=th.bool, device=device)
    in_active = active_frag[atom_fragment_indices]
    rows_of_atoms = frag_row[atom_fragment_indices[in_active]]
    local_atoms = fragment_atom_indices[in_active] - atom_ptr[fragment_batch_indices[atom_fragment_indices[in_active]]]
    member[rows_of_atoms, local_atoms] = True
    ord_batch = fragment_batch_indices[active_ids]
    ord_words = atom_set_words(member, num_words)
    blocked_rows = None if parent_blocked is None else parent_blocked[active_ids]
    lp_rows = parent_log_probabilities[active_ids]

    # 4. pairs (ordinary row x match of its spectrum), chunked by an element budget
    reps = match_count[ord_batch]
    cum_reps = reps.cumsum(0)
    num_pairs = int(cum_reps[-1]) if num_rows else 0  # host sync 2
    diag["rewire_pair_count"] = th.zeros(batch_size, dtype=th.float32, device=device).index_add_(0, ord_batch, reps.float())
    max_bonds = int(bond_cnt.max()) if bond_cnt.numel() else 0
    unit = max(amax, max_bonds, 1)
    rows_per_chunk = max(1, pair_chunk // unit)
    _LAST_STATS.update(max_chunk_rows=0, chunk_rows_budget=rows_per_chunk, num_host_chunks=0)

    parent_key_rows = th.zeros(num_rows, dtype=th.long, device=device)
    if num_rows:
        parent_q = th.round(lp_rows.float().double() * _PARENT_KEY_SCALE)
        parent_q = th.where(th.isfinite(parent_q), parent_q, th.zeros_like(parent_q)).long()
        parent_key_rows = (parent_q << 32) + (_ROW_BITS - th.arange(num_rows, device=device))
    contained_count = th.zeros(batch_size, dtype=th.float32, device=device)
    running: dict[str, th.Tensor] | None = None

    def absorb(rec: dict[str, th.Tensor]) -> None:
        nonlocal running
        if rec["batch"].numel() == 0:
            return
        rec = dict(rec)
        rec["words"] = atom_set_words(rec.pop("member"), num_words)
        rec["ind"] = F.one_hot(rec.pop("rule"), num_rules).to(th.uint8)
        rec["adj"] = -2 * (rec["n_formed"] - rec["n_cut"])
        rec.pop("match")
        rec = _reduce_pieces(rec, num_words, num_rules)
        running = rec if running is None else _reduce_pieces(_concat_records([running, rec]), num_words, num_rules)

    for lo in range(0, num_pairs, rows_per_chunk):
        hi = min(num_pairs, lo + rows_per_chunk)
        ids = th.arange(lo, hi, device=device)
        pf = th.bucketize(ids, cum_reps, right=True)
        pm = match_start[ord_batch][pf] + (ids - (cum_reps - reps)[pf])
        mot = pack.motif[pm]
        inside = (member[pf[:, None].expand_as(mot), mot.clamp_min(0)] | (mot < 0)).all(1)
        if blocked_rows is not None:
            inside &= ~blocked_rows[pf]
        pf, pm = pf[inside], pm[inside]
        if pf.numel() == 0:
            continue
        if not bool(th.isfinite(lp_rows[pf]).all()):
            raise ValueError("a parent proposal log-probability is not finite")
        spec = pack.spec[pm]
        contained_count += th.bincount(spec, minlength=batch_size).float()
        _LAST_STATS["max_chunk_rows"] = max(_LAST_STATS["max_chunk_rows"], int(pf.numel()))
        _LAST_STATS["num_host_chunks"] += 1
        p_mem = member[pf] & ~pack.deleted[pm]
        cu, cv, fu, fv = pack.cut_u[pm], pack.cut_v[pm], pack.form_u[pm], pack.form_v[pm]
        lab = _labels(ctx, p_mem, spec, cu, cv, fu, fv, fu >= 0)
        d0 = _emit(p_mem, lab, spec, pm, ctx, 0, parent_key_rows[pf], strict=True)
        absorb({k: v for k, v in d0.items()})
        if continuation_cuts > 0 and d0["batch"].numel():
            _continue_pieces(d0, ctx, bond_logits, continuation_cuts, rows_per_chunk, unit, absorb)

    diag["rewire_contained_pair_count"] = contained_count
    if running is None:
        return _empty_expansion(device, num_rules, diag)

    # 5. exact dedup against the ordinary rows, atom set only
    run = running
    gp = run["batch"].numel()
    words = th.cat([ord_words, run["words"]])
    batch = th.cat([ord_batch, run["batch"]])
    flag = th.cat([th.zeros(num_rows, dtype=th.long, device=device), th.ones(gp, dtype=th.long, device=device)])
    order = lexsort([batch, *[words[:, w] for w in range(num_words)], flag])
    sw, sb = words[order], batch[order]
    new = th.ones(order.numel(), dtype=th.bool, device=device)
    new[1:] = (sw[1:] != sw[:-1]).any(1) | (sb[1:] != sb[:-1])
    gid_sorted = new.long().cumsum(0) - 1
    gc = int(gid_sorted[-1]) + 1
    gid = th.empty_like(gid_sorted)
    gid[order] = gid_sorted
    has_ord = th.zeros(gc, dtype=th.bool, device=device)
    has_ord[gid[:num_rows]] = True
    run_group = gid[num_rows:]
    run_of_group = th.full((gc,), -1, dtype=th.long, device=device)
    run_of_group[run_group] = th.arange(gp, device=device)
    is_extra = ~has_ord[run_group]
    diag["rewire_precap_extra_count"] = th.bincount(run["batch"][is_extra], minlength=batch_size).float()
    twin_run = run_of_group[gid[:num_rows]]  # run row of every ordinary row's group, -1 if none
    merged_local = th.nonzero(twin_run >= 0).flatten()
    diag["rewire_merged_count"] = th.bincount(
        run["batch"][~is_extra], minlength=batch_size
    ).float()

    # 6. boundary, cut counts, channels for every distinct product set (chunked by the same element budget)
    cut_counts = th.zeros(gp, dtype=th.long, device=device)
    g_rows, g_atoms, b_runs, b_cols, ion_parts, valid_parts = [], [], [], [], [], []
    prod_chunk = max(1, pair_chunk // unit)
    for lo in range(0, gp, prod_chunk):
        hi = min(gp, lo + prod_chunk)
        c_batch = run["batch"][lo:hi]
        c_member = words_to_member(run["words"][lo:hi], amax)
        cnt = bond_cnt[c_batch]
        er = th.repeat_interleave(th.arange(hi - lo, device=device), cnt)
        boff = th.arange(er.numel(), device=device) - (cnt.cumsum(0) - cnt)[er]
        ebond = bond_ptr[c_batch][er] + boff
        cross = c_member[er, ctx.bond_u[ebond]] ^ c_member[er, ctx.bond_v[ebond]]
        b_run, b_col = er[cross], ebond[cross]
        c_cuts = th.bincount(b_run, minlength=hi - lo)
        cut_counts[lo:hi] = c_cuts
        g_row, g_local = th.nonzero(c_member, as_tuple=True)
        c_atoms = g_local + atom_ptr[c_batch[g_row]]
        c_ion, c_valid = channel_fn(c_atoms, g_row, c_batch, hi - lo, c_cuts, run["adj"][lo:hi])
        g_rows.append(g_row + lo)
        g_atoms.append(c_atoms)
        b_runs.append(b_run + lo)
        b_cols.append(b_col)
        ion_parts.append(c_ion)
        valid_parts.append(c_valid)
    g_row, atom_idx, b_run, b_col = th.cat(g_rows), th.cat(g_atoms), th.cat(b_runs), th.cat(b_cols)
    ion_mzs, valid = th.cat(ion_parts), th.cat(valid_parts)
    any_valid = valid.flatten(1).any(1)
    diag["rewire_dropped_no_valid_count"] = th.bincount(run["batch"][is_extra & ~any_valid], minlength=batch_size).float()
    ext = th.nonzero(is_extra & any_valid).flatten()

    # 7. cap on the deduplicated valid extras
    q = run["parent_key"] >> 32
    ext_order = lexsort([run["batch"][ext], -q[ext], run["depth"][ext], *[run["words"][ext][:, w] for w in range(num_words)]])
    ext = ext[ext_order]
    eb = run["batch"][ext]
    kept_rows = th.bincount(fragment_batch_indices, minlength=batch_size).cpu().numpy().astype(np.int64)
    limit = extra_limits(kept_rows, max_extra_fraction, max_extra)
    limit_t = th.from_numpy(limit).to(device)
    if ext.numel():
        counts = th.bincount(eb, minlength=batch_size)
        rank = th.arange(ext.numel(), device=device) - (counts.cumsum(0) - counts)[eb]
        keep = rank < limit_t[eb]
        same_next = (eb[1:] == eb[:-1]) & (rank[:-1] == limit_t[eb[:-1]] - 1) & (rank[1:] == limit_t[eb[:-1]])
        tie = same_next & (q[ext][1:] == q[ext][:-1]) & (run["depth"][ext][1:] == run["depth"][ext][:-1])
        diag["rewire_cap_tie_count"] = th.bincount(eb[:-1][tie], minlength=batch_size).float()
        diag["rewire_dropped_cap_count"] = th.bincount(eb[~keep], minlength=batch_size).float()
        ext = ext[keep]
    fin = ext[lexsort([run["batch"][ext], *[run["words"][ext][:, w] for w in range(num_words)]])]
    diag["rewire_extra_count"] = th.bincount(run["batch"][fin], minlength=batch_size).float()

    # 8. emit
    inv = th.full((gp,), -1, dtype=th.long, device=device)
    inv[fin] = th.arange(fin.numel(), device=device)
    a_sel = inv[g_row] >= 0
    b_sel = inv[b_run] >= 0
    features = th.stack([run["n_formed"], run["n_cut"], run["ndel"], run["depth"]], dim=1).float()
    return RewireExpansion(
        num_extra=int(fin.numel()),
        extra_atom_indices=atom_idx[a_sel],
        extra_atom_extra_indices=inv[g_row[a_sel]],
        extra_boundary_bond_indices=b_col[b_sel],
        extra_boundary_extra_indices=inv[b_run[b_sel]],
        extra_batch_indices=run["batch"][fin],
        extra_cut_counts=cut_counts[fin],
        extra_valence_adjustment=run["adj"][fin],
        extra_ion_mzs=ion_mzs[fin],
        extra_valid_channels=valid[fin],
        extra_rule_indicators=run["ind"][fin].float(),
        extra_edit_features=features[fin],
        extra_parent=active_ids[_ROW_BITS - (run["parent_key"][fin] & 0xFFFFFFFF)],
        merged_rows=active_ids[merged_local],
        merged_valid_channels=valid[twin_run[merged_local]],
        merged_rule_indicators=run["ind"][twin_run[merged_local]].float(),
        merged_edit_features=features[twin_run[merged_local]],
        diagnostics=diag,
    )


def _continue_pieces(
    d0: dict[str, th.Tensor],
    ctx: SimpleNamespace,
    bond_logits: th.Tensor,
    continuation_cuts: int,
    rows_per_chunk: int,
    unit: int,
    absorb: Callable[[dict[str, th.Tensor]], None],
) -> None:
    """Depth 1: remove one of the top-k precursor bonds inside each depth-0 piece (rule cuts excluded)."""
    dev = d0["member"].device
    pm = d0["member"]
    q = pm.shape[0]
    cnt = ctx.bond_cnt[d0["batch"]]
    er = th.repeat_interleave(th.arange(q, device=dev), cnt)
    off = th.arange(er.numel(), device=dev) - (cnt.cumsum(0) - cnt)[er]
    eb = ctx.bond_ptr[d0["batch"]][er] + off
    u, v = ctx.bond_u[eb], ctx.bond_v[eb]
    cue, cve = ctx.cut_u[d0["match"][er]], ctx.cut_v[d0["match"][er]]
    is_cut = (((cue == u[:, None]) & (cve == v[:, None])) | ((cue == v[:, None]) & (cve == u[:, None]))).any(1)
    inb = pm[er, u] & pm[er, v] & ~is_cut
    er, eb = er[inb], eb[inb]
    key = -(bond_logits[eb].double() * 1e6).round().long()
    order = lexsort([er, key, eb])
    er, eb = er[order], eb[order]
    counts = th.bincount(er, minlength=q)
    rank = th.arange(er.numel(), device=dev) - (counts.cumsum(0) - counts)[er]
    sel = rank < continuation_cuts
    er, eb = er[sel], eb[sel]
    for lo in range(0, er.numel(), rows_per_chunk):
        e, b = er[lo : lo + rows_per_chunk], eb[lo : lo + rows_per_chunk]
        sub = pm[e]
        spec, match = d0["batch"][e], d0["match"][e]
        cu, cv, fu, fv = ctx.cut_u[match], ctx.cut_v[match], ctx.form_u[match], ctx.form_v[match]
        rr = th.arange(sub.shape[0], device=dev)[:, None].expand_as(fu)
        form_valid = (fu >= 0) & sub[rr, fu.clamp_min(0)] & sub[rr, fv.clamp_min(0)]
        lab = _labels(ctx, sub, spec, cu, cv, fu, fv, form_valid, removed_bond=b)
        absorb(_emit(sub, lab, spec, match, ctx, 1, d0["parent_key"][e], strict=False))

