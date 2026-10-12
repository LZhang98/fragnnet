"""MASSter's annotation export: formula, node and joint tables built from the pathway outputs.

``MassterPL.postprocess_export_peaks`` builds them per batch, under FraGNNet's index names, so the
shared exporter treats MASSter's annotation like FraGNNet's. Moved out of core ``script_utils`` in
the reorganization (docs/plans/repo_reorg_plan.md, Phase 4).
"""

from dataclasses import dataclass

import numpy as np
import torch as th
from pyteomics.mass import Composition

from fragnnet.models.masster.path_selection.smarts_templates import MASSTER_RULE_VOCABULARY
from fragnnet.utils.formula_utils import (
    ELEMENT_TO_IDX,
    MASK_SIZE,
    NUM_ELEMENTS,
    PERIODIC_TABLE,
    PREC_TYPE_TO_CMF_FORMULA_DIFF,
    PREC_TYPE_TO_FORMULA_DIFF,
    formula_arr_to_str,
    th_mask_to_long,
)


@dataclass
class MassterAnnotationTables:
    """MASSter's per-batch stand-ins for the tables FraGNNet's DAG dataset supplies.

    The export resolves FraGNNet annotation through a formula axis, a node axis and the joint
    (formula, node) grid between them. MASSter has no DAG, so these are built per batch from
    its output pathways, and the exporter then filters, rebases and writes them exactly as it
    does FraGNNet's. Formula and node indices are batch-global and contiguous per spectrum;
    the ``*_cumsizes`` offsets turn them into per-spectrum ones.

    Attributes:
        frag_formula_peak_idxs: Per-spectrum formula row of every aggregated peak, shape (P,).
        formula_batch_idxs: Spectrum of every formula row, shape (R,).
        formula_logprobs: log P(f), the formula's share of its spectrum's intensity, (R,).
        formula_cumsizes: Formula-row offsets, shape (B + 1,).
        formula_strs: Neutral fragment formula (H shift included) per row, shape (R,).
        formula_deltas: Precursor delta formula per row, e.g. ``"H1"``, shape (R,).
        joint_batch_idxs: Spectrum of every (formula, node) cell, shape (J,).
        joint_formula_idxs: Batch-global formula row of every cell, shape (J,).
        joint_node_idxs: Batch-global node of every cell, shape (J,).
        formula_node_logprobs: log P(n | f), shape (J,).
        node_formula_logprobs: log P(f | n), shape (J,).
        node_batch_idxs: Spectrum of every node, shape (N,).
        node_logprobs: log P(n), the node's share of its spectrum's intensity, shape (N,).
        node_cumsizes: Node offsets, shape (B + 1,).
        node_ccs: Atom bitmask per node in the FraGNNet ``cc`` encoding, shape (N, 2), or
            None when not requested.
        num_multi_formula_peaks: Peaks whose pathways carry more than one formula row.
        node_rules: SMARTS rule names behind each node, ``";"``-joined in vocabulary order
            and empty for a node no rule produces, shape (N,); None when not requested.
    """

    frag_formula_peak_idxs: th.Tensor
    formula_batch_idxs: th.Tensor
    formula_logprobs: th.Tensor
    formula_cumsizes: th.Tensor
    formula_strs: np.ndarray
    formula_deltas: np.ndarray
    joint_batch_idxs: th.Tensor
    joint_formula_idxs: th.Tensor
    joint_node_idxs: th.Tensor
    formula_node_logprobs: th.Tensor
    node_formula_logprobs: th.Tensor
    node_batch_idxs: th.Tensor
    node_logprobs: th.Tensor
    node_cumsizes: th.Tensor
    node_ccs: th.Tensor | None
    num_multi_formula_peaks: int
    node_rules: np.ndarray | None = None


def _unique_rows(rows: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
    """``th.unique(rows, dim=0, return_inverse=True)``, built from stable 1-D sorts.

    Same rows, same lexicographic order, same inverse. ``th.unique`` with ``dim`` took 86% of
    the MASSter annotation's CPU time (``ll04i00m``, batch 32); one stable argsort per
    non-constant column is much cheaper.

    Args:
        rows: Integer keys, shape (n, k).

    Returns:
        (unique rows, shape (u, k); inverse, shape (n,)).
    """
    order = th.arange(rows.shape[0], device=rows.device)
    if rows.shape[0] > 1:
        # Sorting by the last column first and the first column last leaves the rows in
        # lexicographic order. A constant column changes neither the grouping nor the order.
        varying = (rows.amax(dim=0) != rows.amin(dim=0)).tolist()
        for column in reversed(range(rows.shape[1])):
            if varying[column]:
                order = order[th.argsort(rows[order, column], stable=True)]
    sorted_rows = rows[order]
    starts = th.ones(rows.shape[0], dtype=th.bool, device=rows.device)
    starts[1:] = (sorted_rows[1:] != sorted_rows[:-1]).any(dim=1)
    inverse = th.empty_like(order)
    inverse[order] = th.cumsum(starts, 0) - 1
    return sorted_rows[starts], inverse


def _batch_offsets(batch_idxs: th.Tensor, batch_size: int) -> th.Tensor:
    """Offsets of a batch-sorted axis, shape (batch_size + 1,)."""
    counts = th.bincount(batch_idxs, minlength=batch_size)
    return th.cat([counts.new_zeros(1), th.cumsum(counts, 0)])


def masster_annotation_tables(
    pathway_peak_idxs: th.Tensor,
    peak_batch_idxs: th.Tensor,
    pathway_ints: th.Tensor,
    pathway_fragment_idxs: th.Tensor,
    pathway_ion_mode_idxs: th.Tensor,
    pathway_h_shifts: th.Tensor,
    fragment_batch_idxs: th.Tensor,
    fragment_atom_indices: th.Tensor,
    fragment_atom_fragment_idxs: th.Tensor,
    atom_atomic_numbers: th.Tensor,
    atom_hydrogen_counts: th.Tensor,
    atom_ptr: th.Tensor,
    precursor_types: list[str],
    return_node_ccs: bool,
    fragment_rule_fragment_idxs: th.Tensor | None = None,
    fragment_rule_idxs: th.Tensor | None = None,
) -> MassterAnnotationTables:
    """Group MASSter's output pathways into FraGNNet's formula / joint / node annotation.

    A pathway is one (fragment, ion mode, H shift) channel; the exported peak is every pathway
    at one 4-decimal m/z. Pathways are grouped three ways:

    - **formula** rows: one per (peak, neutral formula, precursor delta). The neutral formula is
      the fragment's own formula plus its H shift, as in FraGNNet, so neutral + delta is the
      ion. Where that neutral formula would need a negative count (a hydrogen-poor fragment
      ionized by the adduct's own hydrogen, e.g. ``[M+H]+`` at H shift -1), the row carries
      the ion formula with an empty delta instead, the way FraGNNet writes ``[M+H]+``
      charge retention.
    - **node** rows: one per distinct atom set in a spectrum, so duplicate latent fragments
      (the non-unique fragment mode) share a node.
    - **joint** cells: one per (formula row, node).

    With ``w`` the L1-normalized pathway intensity, ``P(f) = sum w`` over the row's pathways,
    ``P(n) = sum w`` over the node's, ``P(n|f) = w(f, n) / P(f)`` and ``P(f|n) = w(f, n) / P(n)``.
    A ratio whose denominator is 0 is reported as 0.

    Args:
        pathway_peak_idxs: Aggregated peak of every pathway, shape (Q,).
        peak_batch_idxs: Spectrum of every aggregated peak, batch-sorted, shape (P,).
        pathway_ints: Linear, per-spectrum L1-normalized intensity per pathway, shape (Q,).
        pathway_fragment_idxs: Fragment of every pathway, shape (Q,).
        pathway_ion_mode_idxs: 0 for CRF, 1 for CMF, shape (Q,).
        pathway_h_shifts: Signed hydrogen shift per pathway, shape (Q,).
        fragment_batch_idxs: Spectrum of every fragment, shape (Fr,).
        fragment_atom_indices: Batch-global atom of every (fragment, atom) membership, (M,).
        fragment_atom_fragment_idxs: Fragment of every membership, shape (M,).
        atom_atomic_numbers: Atomic number per atom of the batched graphs, shape (A,).
        atom_hydrogen_counts: Implicit hydrogen count per atom, shape (A,).
        atom_ptr: Atom offsets of the batched graphs, shape (B + 1,).
        precursor_types: Precursor type string per spectrum, length B.
        return_node_ccs: Build the (N, 2) ``cc`` bitmasks.
        fragment_rule_fragment_idxs: Fragment of every (fragment, SMARTS rule) pair, shape
            (R,), or None to skip the node rules.
        fragment_rule_idxs: Rule index of every pair, from 1 into
            ``MASSTER_RULE_VOCABULARY``, shape (R,). Given exactly with the fragments.

    Returns:
        The annotation tables, on the device of the pathway tensors.

    Raises:
        ValueError: If a fragment holds an element the formula vocabulary lacks, or
            ``return_node_ccs`` is set for a molecule larger than ``MASK_SIZE`` atoms.
    """
    device = pathway_peak_idxs.device
    batch_size = len(precursor_types)
    num_peaks = int(peak_batch_idxs.numel())
    num_fragments = int(fragment_batch_idxs.numel())
    h_column = ELEMENT_TO_IDX["H"]

    # 1. Element counts per fragment, in the column order `formula_arr_to_str` renders.
    column_of_atomic_number = th.full((119,), -1, dtype=th.long, device=device)
    for element, column in ELEMENT_TO_IDX.items():
        column_of_atomic_number[PERIODIC_TABLE.GetAtomicNumber(element)] = column
    membership_numbers = atom_atomic_numbers[fragment_atom_indices].long()
    membership_columns = column_of_atomic_number[membership_numbers]
    if (membership_columns < 0).any():
        unknown = sorted(set(membership_numbers[membership_columns < 0].tolist()))
        raise ValueError(f"fragment atoms with atomic numbers {unknown} have no formula column")
    fragment_counts = th.zeros((num_fragments, NUM_ELEMENTS), dtype=th.long, device=device)
    fragment_counts.index_put_(
        (fragment_atom_fragment_idxs, membership_columns),
        th.ones_like(membership_columns),
        accumulate=True,
    )
    fragment_counts[:, h_column].index_add_(
        0, fragment_atom_fragment_idxs, atom_hydrogen_counts[fragment_atom_indices].long()
    )

    # 2. One node per distinct atom set in a spectrum, keyed by a molecule-local bitmask.
    molecule_sizes = atom_ptr[1:] - atom_ptr[:-1]
    max_atoms = int(molecule_sizes.max().item()) if batch_size > 0 else 0
    if return_node_ccs and max_atoms > MASK_SIZE:
        raise ValueError(
            f"node cc bitmasks hold {MASK_SIZE} atoms; this batch has a molecule with "
            f"{max_atoms}. Export with return_cc_hash=False."
        )
    mask_bits = max(MASK_SIZE, -(-max_atoms // 64) * 64)
    local_atoms = fragment_atom_indices - atom_ptr[fragment_batch_idxs[fragment_atom_fragment_idxs]]
    fragment_masks = th.zeros((num_fragments, mask_bits), dtype=th.bool, device=device)
    fragment_masks[fragment_atom_fragment_idxs, local_atoms] = True
    fragment_keys = th.cat(
        [fragment_batch_idxs[:, None].long(), th_mask_to_long(fragment_masks.long())], dim=1
    )
    used_fragments = th.unique(pathway_fragment_idxs)
    node_keys, used_to_node = _unique_rows(fragment_keys[used_fragments])
    fragment_to_node = th.full((num_fragments,), -1, dtype=th.long, device=device)
    fragment_to_node[used_fragments] = used_to_node
    pathway_nodes = fragment_to_node[pathway_fragment_idxs]
    node_batch_idxs = node_keys[:, 0]
    if (fragment_rule_fragment_idxs is None) != (fragment_rule_idxs is None):
        raise ValueError("fragment_rule_fragment_idxs and fragment_rule_idxs go together")
    node_rules = None
    if fragment_rule_fragment_idxs is not None:
        # A node is an atom set, so it carries every rule behind any fragment it merges.
        rule_nodes = fragment_to_node[fragment_rule_fragment_idxs].tolist()
        rules_by_node: dict[int, set[int]] = {}
        for node, rule in zip(rule_nodes, fragment_rule_idxs.tolist(), strict=True):
            if node >= 0:
                rules_by_node.setdefault(node, set()).add(rule)
        node_rules = np.full(node_keys.shape[0], "", dtype=object)
        for node, rules in rules_by_node.items():
            node_rules[node] = ";".join(MASSTER_RULE_VOCABULARY[rule - 1] for rule in sorted(rules))

    # 3. Neutral formula and precursor delta per pathway; fold negative neutral formulas.
    deltas = sorted(
        {""}
        | {PREC_TYPE_TO_FORMULA_DIFF[p] for p in precursor_types}
        | {PREC_TYPE_TO_CMF_FORMULA_DIFF.get(p, "") for p in precursor_types}
    )
    delta_ids = {delta: i for i, delta in enumerate(deltas)}
    # Column 0 is the CRF delta, column 1 the CMF one, indexed by the pathway's ion mode.
    delta_by_spectrum_mode = th.tensor(
        [
            [
                delta_ids[PREC_TYPE_TO_FORMULA_DIFF[p]],
                delta_ids[PREC_TYPE_TO_CMF_FORMULA_DIFF.get(p, "")],
            ]
            for p in precursor_types
        ],
        dtype=th.long,
        device=device,
    ).reshape(batch_size, 2)
    delta_counts = th.zeros((len(deltas), NUM_ELEMENTS), dtype=th.long, device=device)
    for i, delta in enumerate(deltas):
        for element, count in Composition(formula=delta).items():
            if element in ELEMENT_TO_IDX:
                delta_counts[i, ELEMENT_TO_IDX[element]] = count
    pathway_batch_idxs = peak_batch_idxs[pathway_peak_idxs]
    pathway_counts = fragment_counts[pathway_fragment_idxs].clone()
    pathway_counts[:, h_column] += pathway_h_shifts.long()
    pathway_delta_ids = delta_by_spectrum_mode[pathway_batch_idxs, pathway_ion_mode_idxs.long()]
    folded = (pathway_counts < 0).any(dim=1)
    if folded.any():
        pathway_counts[folded] += delta_counts[pathway_delta_ids[folded]]
        pathway_delta_ids[folded] = delta_ids[""]
        if (pathway_counts[folded] < 0).any():
            raise ValueError("a pathway's ion formula has a negative element count")

    # 4. Formula rows. Keys lead with the peak, so rows come out peak- and batch-sorted.
    formula_keys, pathway_rows = _unique_rows(
        th.cat([pathway_peak_idxs[:, None], pathway_delta_ids[:, None], pathway_counts], dim=1)
    )
    num_rows = int(formula_keys.shape[0])
    row_peaks = formula_keys[:, 0]
    formula_batch_idxs = peak_batch_idxs[row_peaks]
    formula_probs = pathway_ints.new_zeros(num_rows).index_add(0, pathway_rows, pathway_ints)
    formula_cumsizes = _batch_offsets(formula_batch_idxs, batch_size)

    # Each peak points at its most intense formula row; there is one for every peak.
    best_probs = formula_probs.new_full((num_peaks,), -th.inf).scatter_reduce(
        0, row_peaks, formula_probs, reduce="amax"
    )
    row_ids = th.arange(num_rows, device=device)
    peak_rows = th.full((num_peaks,), num_rows, dtype=th.long, device=device).scatter_reduce(
        0,
        row_peaks,
        th.where(formula_probs == best_probs[row_peaks], row_ids, num_rows),
        reduce="amin",
    )
    frag_formula_peak_idxs = peak_rows - formula_cumsizes[peak_batch_idxs]
    rows_per_peak = th.bincount(row_peaks, minlength=num_peaks)

    # 5. Joint (formula, node) cells, sorted by formula row and so by spectrum.
    joint_keys, pathway_cells = _unique_rows(th.stack([pathway_rows, pathway_nodes], dim=1))
    joint_probs = pathway_ints.new_zeros(joint_keys.shape[0]).index_add(
        0, pathway_cells, pathway_ints
    )
    node_probs = pathway_ints.new_zeros(node_keys.shape[0]).index_add(
        0, pathway_nodes, pathway_ints
    )
    tiny = th.finfo(pathway_ints.dtype).tiny
    joint_rows, joint_nodes = joint_keys[:, 0], joint_keys[:, 1]
    formula_node_probs = joint_probs / formula_probs[joint_rows].clamp_min(tiny)
    node_formula_probs = joint_probs / node_probs[joint_nodes].clamp_min(tiny)

    # 6. Strings, rendered once per distinct formula.
    unique_formulas, formula_string_ids = _unique_rows(formula_keys[:, 2:])
    rendered = np.array(
        [formula_arr_to_str(counts) for counts in unique_formulas.cpu().numpy()], dtype=object
    )
    formula_strs = rendered[formula_string_ids.cpu().numpy()]
    formula_deltas = np.array(deltas, dtype=object)[formula_keys[:, 1].cpu().numpy()]

    return MassterAnnotationTables(
        frag_formula_peak_idxs=frag_formula_peak_idxs,
        formula_batch_idxs=formula_batch_idxs,
        formula_logprobs=th.log(formula_probs),
        formula_cumsizes=formula_cumsizes,
        formula_strs=formula_strs,
        formula_deltas=formula_deltas,
        joint_batch_idxs=formula_batch_idxs[joint_rows],
        joint_formula_idxs=joint_rows,
        joint_node_idxs=joint_nodes,
        formula_node_logprobs=th.log(formula_node_probs),
        node_formula_logprobs=th.log(node_formula_probs),
        node_batch_idxs=node_batch_idxs,
        node_logprobs=th.log(node_probs),
        node_cumsizes=_batch_offsets(node_batch_idxs, batch_size),
        node_ccs=node_keys[:, 1 : 1 + MASK_SIZE // 64] if return_node_ccs else None,
        num_multi_formula_peaks=int((rows_per_peak > 1).sum().item()),
        node_rules=node_rules,
    )
