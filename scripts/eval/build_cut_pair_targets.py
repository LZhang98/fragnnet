"""Build complementary-cut pair targets (``q_cut_evidence``) for fragment intensity ranking.

Phase 1 of ``docs/plans/fragnnet/complementary_cut_supervision_experiment_plan.md``. For every spectrum and
every DAG parent, this script finds child pairs ``(A, B)`` that exactly partition the parent's
atoms, scores the relative *observable evidence* for each side, and emits a ragged cache in the
same schema the sibling-ranking loss already consumes, plus a sidecar audit report.

What the target is
------------------
``q_cut_evidence`` estimates the relative observable evidence for the two sides of one cut. It is
NOT a bond-breaking probability, a fragment population, a charge-retention probability, or an
edge-``beta`` label.

Two uncertainty channels are kept separate, which is the central design point:

  - **Raw intensity sets target softness.** The unobserved side is floored at the spectrum's
    detection limit rather than assigned zero, because peak absence is not evidence of zero
    fragment probability.
  - **Responsibility sets group weight.** A peak matched by ``n_j`` DAG nodes contributes
    ``1 / n_j`` to the pair's training weight, so ambiguous attribution trains weakly without
    distorting the target itself.

Collapsing these (e.g. comparing responsibility-discounted evidence against the detection limit)
would make a 20%-intensity peak with confidence 0.1 look identical to a clean 2% peak. They are not
equally close to the noise floor.

Detection limit
---------------
``target_rel_ints_thresh`` defaults to 0.0, so there is no explicit preprocessing cutoff to reuse.
The floor is therefore estimated from the training split only, stratified by
``(inst_type, m/z bin)``, frozen, and then applied unchanged to validation. It must be finite and
strictly positive; the builder raises otherwise rather than substituting an epsilon.

Provenance
----------
The emitted cache is stamped ``metadata.builder == "cut_pair"``. It can be loaded through the
existing ``sibling_rank_*`` slot for a pilot, where it will be logged as ``sibling_rank_kl`` -- the
stamp plus the cache fingerprint are what keep that pilot attributable.

Example:
    python scripts/eval/build_cut_pair_targets.py \
        --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags \
        --splits train val \
        --out_fp data/targets/nist20_cut_pair_train_val.pt
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch as th
from oracle_dataset_utils import resolve_dsets_for_spec_fp

import fragnnet.utils.data_utils as data_utils
from fragnnet.models.fragnnet.frag_utils import MAX_H_TRANSFER, get_node_feats, th_long_to_mask
from fragnnet.utils.data_utils import CHARGE_FACTOR_MAP, get_charge
from fragnnet.utils.formula_utils import (
    PREC_TYPE_TO_CMF_MASS_DIFF,
    PREC_TYPE_TO_MASS_DIFF,
    cmf_duplicate_crf_h_shift,
    prec_type_removed_counts,
    tracked_element_counts,
)
from fragnnet.utils.spec_utils import calculate_match_mzs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dag_oracle_scores import (  # noqa: E402
    _allowed_formula_idxs,
    _detect_is_compressed,
    _tolerance_for_inst_type,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE_VERSION = 2
BUILDER_NAME = "cut_pair"
MZ_BIN_WIDTH = 100.0
# Confidence stratum both Phase 2 gates are sized on; fixed here so it cannot be tuned later.
HIGH_CONF_Q = 0.8
# Quantiles of the per-spectrum weakest retained peak, reported so the chosen LOD estimator can be
# judged against alternatives before a full build.
LOD_AUDIT_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
LOD_QUANTILE = 0.5

# Precursors that carry no CMF channel at all, matching the dataloader's include_cmf guard.
# Anything absent from PREC_TYPE_TO_CMF_MASS_DIFF is excluded on the same grounds.
CMF_EXCLUDED_PREC = ("[M]+", "[M-]")


def _duplicate_cmf_precursors(max_h_transfer: int | None) -> list[str]:
    """List CMF precursors whose ions are already represented by a CRF H-shift."""
    effective_h_transfer = MAX_H_TRANSFER if max_h_transfer is None else max_h_transfer
    return sorted(
        prec_type
        for prec_type in PREC_TYPE_TO_CMF_MASS_DIFF
        if cmf_duplicate_crf_h_shift(prec_type, effective_h_transfer) is not None
    )


def _mz_bin(mz: float | np.ndarray) -> np.ndarray:
    """Return the m/z stratum index for the detection-limit table."""
    return np.floor(np.asarray(mz, dtype=np.float64) / MZ_BIN_WIDTH).astype(np.int64)


def estimate_lod_table(
    spec_rows: list[dict], quantile: float = LOD_QUANTILE
) -> tuple[dict[tuple[str, int], float], dict[str, float], float]:
    """Estimate the per-spectrum detection limit from training spectra only.

    For each ``(inst_type, m/z bin)`` the floor is a quantile over spectra of the smallest
    sum-normalised intensity actually retained in that bin -- i.e. the weakest peak the pipeline
    was observed to keep there. Coarser fallbacks are returned for strata with no training support.

    This is a proxy, not a measured instrument threshold: the weakest retained peak depends on how
    many peaks a spectrum has, so the central quantile sits above the true floor and will discard
    some genuinely observed low-intensity evidence as sub-LOD. ``lod_quantile_audit`` reports the
    alternatives so the choice can be judged before a full build.

    Args:
        spec_rows: Training-split spectrum records, each with ``inst_type`` and ``peaks``.
        quantile: Quantile of the per-spectrum minima to use, in [0, 1].

    Returns:
        Tuple ``(by_inst_bin, by_inst, global_floor)`` of floors on the sum-normalised intensity
        scale.

    Raises:
        ValueError: If no strictly positive intensity exists anywhere in the training split.
    """
    per_stratum: dict[tuple[str, int], list[float]] = defaultdict(list)
    per_inst: dict[str, list[float]] = defaultdict(list)
    all_minima: list[float] = []

    for row in spec_rows:
        peaks = row["peaks"]
        if not peaks:
            continue
        mzs, ints = (np.asarray(v, dtype=np.float64) for v in zip(*peaks))
        total = float(ints.sum())
        if total <= 0:
            continue
        rel = ints / total
        keep = rel > 0
        if not keep.any():
            continue
        mzs, rel = mzs[keep], rel[keep]
        inst = str(row["inst_type"])
        bins = _mz_bin(mzs)
        for b in np.unique(bins):
            per_stratum[(inst, int(b))].append(float(rel[bins == b].min()))
        per_inst[inst].append(float(rel.min()))
        all_minima.append(float(rel.min()))

    if not all_minima:
        raise ValueError("No strictly positive training intensity; cannot estimate a LOD floor.")

    by_inst_bin = {k: float(np.quantile(v, quantile)) for k, v in per_stratum.items()}
    by_inst = {k: float(np.quantile(v, quantile)) for k, v in per_inst.items()}
    global_floor = float(np.quantile(all_minima, quantile))
    if not np.isfinite(global_floor) or global_floor <= 0:
        raise ValueError(f"Estimated global LOD floor is not positive: {global_floor}")
    for key, value in by_inst_bin.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Estimated LOD floor for stratum {key} is not positive: {value}")
    return by_inst_bin, by_inst, global_floor


def lod_quantile_audit(spec_rows: list[dict]) -> dict[str, float]:
    """Report the global floor each candidate quantile would produce.

    The spread between these is how much the sub-LOD filter depends on an arbitrary choice; a wide
    spread means the retained-pair count is an artefact of the estimator, not of the data.
    """
    return {f"q{q:g}": estimate_lod_table(spec_rows, quantile=q)[2] for q in LOD_AUDIT_QUANTILES}


def _lod_for(
    inst: str,
    mzs: np.ndarray,
    by_inst_bin: dict[tuple[str, int], float],
    by_inst: dict[str, float],
    global_floor: float,
) -> float:
    """Return the side-level floor: the minimum LOD over a side's candidate channels."""
    floors = []
    for b in np.unique(_mz_bin(mzs)):
        if (inst, int(b)) in by_inst_bin:
            floors.append(by_inst_bin[(inst, int(b))])
        elif inst in by_inst:
            floors.append(by_inst[inst])
        else:
            floors.append(global_floor)
    out = float(min(floors)) if floors else global_floor
    if not np.isfinite(out) or out <= 0:
        raise ValueError(f"Non-positive LOD floor for inst={inst}: {out}")
    return out


def _shuffle_seed(seed: int, spec_id: int, parent_idx: int, a: int, b: int) -> int:
    """Stable per-pair seed, independent of worker scheduling."""
    modulus = 2**63 - 1
    return (
        int(seed) * 6364136223846793005
        + int(spec_id) * 1442695040888963407
        + int(parent_idx) * 3202034522624059733
        + int(a) * 2718281828459045235
        + int(b) * 3141592653589793239
    ) % modulus


def _complementary_pairs(masks: np.ndarray, edge_index: np.ndarray) -> list[tuple[int, int, int]]:
    """Enumerate deduplicated ``(parent, a, b)`` triples whose children partition the parent.

    Args:
        masks: Boolean atom masks of shape (num_nodes, mask_bits).
        edge_index: DAG connectivity of shape (2, num_edges).

    Returns:
        Sorted, deduplicated list of ``(parent, a, b)`` with ``a < b``.
    """
    pairs: set[tuple[int, int, int]] = set()
    for parent in np.unique(edge_index[0]):
        children = np.unique(edge_index[1][edge_index[0] == parent])
        if children.size < 2:
            continue
        pm = masks[parent]
        psz = int(pm.sum())
        cm = masks[children]
        csz = cm.sum(axis=1)
        # A complement must have exactly the parent's remaining atoms, so sizes must sum.
        for i in range(children.size):
            for j in range(i + 1, children.size):
                if csz[i] + csz[j] != psz:
                    continue
                if (cm[i] & cm[j]).any():
                    continue
                if not np.array_equal(cm[i] | cm[j], pm):
                    continue
                a, b = int(children[i]), int(children[j])
                pairs.add((int(parent), min(a, b), max(a, b)))
    return sorted(pairs)


def load_spec_df_with_ce_fill(
    spec_fp: str, dsets: list[str] | None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load spectra using the training dataset's collision-energy contract, not the analysis one.

    ``branching_headroom.load_spec_df`` drops every row with a missing ``nce``. The training
    dataset instead derives it from ``ace`` (``nce = ace * 500 / (prec_mz * charge_factor)``), so
    that filter silently discards thousands of spectra the model will actually train on. Those
    spectra have peaks, supported adducts, and a usable ACE; excluding them would leave 5.8% of
    train/validation with zero cut supervision.

    Args:
        spec_fp: Path to the spectrum dataframe pickle.
        dsets: Datasets to keep, or None for all.

    Returns:
        Tuple of the filtered frame and a dict of per-filter drop counts.
    """
    log.info("Loading %s", spec_fp)
    spec_df = pd.read_pickle(spec_fp)
    counts = {"rows_in": len(spec_df)}
    if dsets:
        spec_df = spec_df[spec_df["dset"].isin(dsets)]
    counts["after_dset"] = len(spec_df)
    spec_df = spec_df[spec_df["prec_type"].isin(PREC_TYPE_TO_MASS_DIFF.keys())]
    counts["after_prec_type"] = len(spec_df)
    spec_df = spec_df[spec_df["peaks"].apply(len) > 0]
    counts["after_nonempty_peaks"] = len(spec_df)

    # Same fill as base_dataset: NCE from ACE when NCE is absent or sentinel -1.
    charge_factors = spec_df["prec_type"].map(
        lambda pt: CHARGE_FACTOR_MAP.get(np.abs(get_charge(pt)), CHARGE_FACTOR_MAP["large"])
    )
    ace = spec_df["ace"].where(spec_df["ace"].notna() & (spec_df["ace"] != -1))
    nce = spec_df["nce"].where(spec_df["nce"].notna() & (spec_df["nce"] != -1))
    computed = (ace * 500.0) / (spec_df["prec_mz"] * charge_factors)
    filled = pd.to_numeric(nce, errors="coerce").fillna(pd.to_numeric(computed, errors="coerce"))
    counts["nce_filled_from_ace"] = int(nce.isna().sum() - filled.isna().sum())
    spec_df = spec_df.assign(nce=filled)

    spec_df = spec_df[spec_df["nce"].notna()]
    counts["after_nce"] = len(spec_df)
    return (
        spec_df[
            [
                "spec_id",
                "mol_id",
                "peaks",
                "prec_mz",
                "prec_type",
                "inst_type",
                "frag_mode",
                "nce",
            ]
        ],
        counts,
    )


def read_generation_manifest(frag_dp: str) -> dict | None:
    """Load the DAG generation manifest, if the artifact carries one.

    Returns:
        The parsed manifest, or None when no manifest is present next to the DAGs.
    """
    for candidate in (
        os.path.join(frag_dp, "generation_manifest.json"),
        os.path.join(os.path.dirname(os.path.normpath(frag_dp)), "generation_manifest.json"),
    ):
        if os.path.isfile(candidate):
            with open(candidate) as fh:
                return json.load(fh)
    return None


def check_min_frag_atoms(manifest: dict | None, observed_min_size: int) -> str:
    """Fail closed unless the DAG demonstrably retains the smallest counter-fragments.

    A nonzero ``min_frag_atoms`` silently removes exactly the small complements for which charge
    retention is most asymmetric, so the pair target would be built on a biased subset. A manifest
    is authoritative.

    Without one, the observation is **one-sided falsification, not proof**. Generation with
    ``min_frag_atoms == k`` cannot emit any fragment below ``k`` atoms, so seeing a 1-atom fragment
    rules out every threshold above 1 -- and 0 versus 1 is operationally moot, since zero-atom
    children are rejected unconditionally by ``multi_cut_bfs.pyx`` at any setting. The converse
    does NOT hold: a corpus can lack 1-atom fragments simply because none of its molecules produced
    one, which is entirely plausible on a small or limited build. Absence is therefore inconclusive
    and the build stops for want of a manifest; it is not a finding that filtering occurred.

    Enumerating complementary pairs among surviving children proves nothing either way: a
    complement that generation already discarded leaves no trace among the children that remain.

    Args:
        manifest: Parsed generation manifest, or None if the artifact carries none.
        observed_min_size: Smallest fragment atom count seen anywhere in the scanned corpus.

    Returns:
        A short string recording which evidence was used, for the cache metadata.

    Raises:
        ValueError: If the manifest records a nonzero value, or no manifest exists and the corpus
            shows no 1-atom fragment with which to falsify a threshold.
    """
    if manifest is not None:
        if "min_frag_atoms" not in manifest:
            raise ValueError(
                "Generation manifest exists but records no min_frag_atoms; refusing to guess."
            )
        recorded = int(manifest["min_frag_atoms"])
        if recorded != 0:
            raise ValueError(
                f"DAG was generated with min_frag_atoms={recorded}, not 0. The small "
                "counter-fragments this target depends on were filtered out at generation."
            )
        return f"manifest: min_frag_atoms={recorded}"

    if observed_min_size > 1:
        raise ValueError(
            f"Inconclusive: no generation manifest, and the smallest fragment in this corpus has "
            f"{observed_min_size} atoms. That is consistent with min_frag_atoms >= "
            f"{observed_min_size}, but also with an unfiltered corpus that simply produced no "
            "smaller fragment -- common on a limited build. This is not proof of filtering; the "
            "build stops because the precondition cannot be established. Supply a generation "
            "manifest, or rerun over a corpus large enough to falsify a threshold."
        )
    return (
        f"falsification: no manifest; a {observed_min_size}-atom fragment is present, which rules "
        "out any min_frag_atoms > 1"
    )


def _cut_bond_type(mol, mask_a: np.ndarray, mask_b: np.ndarray) -> str:
    """Classify the bond(s) severed between two complementary sides.

    Returns a coarse ``(ring, element-pair)`` bucket, matching the descriptor the DAG pruning work
    used, or ``multi`` when the cut severed more than one bond (a ring opening).
    """
    if mol is None:
        return "unknown"
    atoms_a = set(np.nonzero(mask_a)[0].tolist())
    atoms_b = set(np.nonzero(mask_b)[0].tolist())
    severed = [
        bond
        for bond in mol.GetBonds()
        if (bond.GetBeginAtomIdx() in atoms_a and bond.GetEndAtomIdx() in atoms_b)
        or (bond.GetBeginAtomIdx() in atoms_b and bond.GetEndAtomIdx() in atoms_a)
    ]
    if not severed:
        return "none"
    if len(severed) > 1:
        return f"multi_{len(severed)}"
    bond = severed[0]
    sym = {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()}
    hetero = sum(1 for s in sym if s not in ("C", "H"))
    pair = "XX" if hetero == 2 else ("CX" if hetero == 1 else "CC")
    return f"{'ring' if bond.IsInRing() else 'chain'}_{pair}"


def _complement_incompleteness(masks: np.ndarray, edge_index: np.ndarray) -> int:
    """Count children whose set-complement within the parent is absent from that parent.

    Diagnostic only, never a gate: a multi-piece cut legitimately yields children whose complement
    is a disconnected union and therefore not itself a node. A sharp rise in this count across
    corpora is still a useful smell test for aggressive small-fragment filtering.
    """
    missing = 0
    for parent in np.unique(edge_index[0]):
        children = np.unique(edge_index[1][edge_index[0] == parent])
        if children.size < 2:
            continue
        pm = masks[parent]
        present = {mask.tobytes() for mask in masks[children]}
        for child in children:
            complement = pm & ~masks[child]
            if complement.any() and complement.tobytes() not in present:
                missing += 1
    return missing


def node_candidate_ions(
    h_formulae_idx: np.ndarray,
    formula_peak_mzs: np.ndarray,
    allowed_rows: set[int],
    prec_type: str,
    max_h_transfer: int | None = None,
    formula_element_counts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build each node's candidate ion m/z under the model's real CRF/CMF contract.

    The CRF channel uses every formula row the node carries within the H-transfer budget. Whether
    a CMF channel exists at all is precursor-dependent, and applying the CMF offset unconditionally
    -- as the diagnostic helpers do -- invents channels the model never scores, which inflates the
    match count ``n_j`` and makes unrelated sides look mass-aliased. A precursor has no CMF channel
    when it carries no CMF entry (``[M]+``/``[M-]`` and the in-source loss precursors), or when its
    CMF ion is already enumerated as a CRF hydrogen shift (``[M+H]+``/``[M-H]-``). Every remaining
    CMF-bearing precursor takes the full H-shift range, matching the dataloader.

    Args:
        h_formulae_idx: Per-node formula rows of shape (num_nodes, num_h_slots), column 0 being
            the node's own zero-H-shift row.
        formula_peak_mzs: Neutral formula m/z of shape (num_formulas, num_isotopes).
        allowed_rows: Formula rows permitted by the H-transfer budget.
        prec_type: Precursor adduct, selecting the ion-mass conventions.
        max_h_transfer: H-transfer budget in force, used to decide whether the CRF ladder already
            reaches the CMF ion. ``None`` means the DAG's own range, read off the slot count.
        formula_element_counts: Per-formula counts in the layout returned by
            ``tracked_element_counts``. When supplied, rows that cannot supply a precursor loss
            are excluded independently for CRF and CMF, matching the dataloader.

    Returns:
        Tuple ``(ion_mz, node_of_cand, is_cmf)`` of shape (M,) each: every candidate ion m/z, the
        node it belongs to, and whether it came from the CMF convention. Row 0 (NULL formula) and
        zero-mass isotope padding are dropped.
    """
    std_diff = PREC_TYPE_TO_MASS_DIFF[prec_type]
    # Mirrors src/fragnnet/models/fragnnet/spec_mol_frag_dataset.py: a precursor whose CMF ion is already
    # enumerated as a CRF hydrogen shift emits no CMF rows, and every other CMF-bearing precursor
    # emits the full H-shift range.
    effective_h_transfer = (
        (h_formulae_idx.shape[1] - 1) // 2 if max_h_transfer is None else max_h_transfer
    )
    include_cmf = (
        prec_type not in CMF_EXCLUDED_PREC
        and prec_type in PREC_TYPE_TO_CMF_MASS_DIFF
        and cmf_duplicate_crf_h_shift(prec_type, effective_h_transfer) is None
    )
    cmf_diff = PREC_TYPE_TO_CMF_MASS_DIFF[prec_type] if include_cmf else None

    def feasible_rows(rows: set[int], *, cmf: bool) -> set[int]:
        """Keep only rows whose formula survives this channel's precursor delta."""
        if formula_element_counts is None:
            return rows
        removed = prec_type_removed_counts(prec_type, cmf=cmf)
        if not removed:
            return rows
        return {
            row
            for row in rows
            if all(
                formula_element_counts[row, position] >= amount
                for position, amount in removed.items()
            )
        }

    ions: list[np.ndarray] = []
    node_of: list[np.ndarray] = []
    cmf_flag: list[np.ndarray] = []
    for n in range(h_formulae_idx.shape[0]):
        rows = {int(r) for r in np.unique(h_formulae_idx[n]) if r != 0}
        rows &= allowed_rows
        crf_rows = feasible_rows(rows, cmf=False)
        chans: list[np.ndarray] = []
        flags: list[np.ndarray] = []
        if crf_rows:
            crf = formula_peak_mzs[sorted(crf_rows)].reshape(-1)
            crf = crf[crf > 0.0] + std_diff
            chans.append(crf)
            flags.append(np.zeros(crf.shape[0], dtype=bool))
        if include_cmf:
            cmf_rows = feasible_rows(rows, cmf=True)
            if cmf_rows:
                cmf = formula_peak_mzs[sorted(cmf_rows)].reshape(-1)
                cmf = cmf[cmf > 0.0] + cmf_diff
                chans.append(cmf)
                flags.append(np.ones(cmf.shape[0], dtype=bool))
        if not chans:
            continue
        mzs = np.concatenate(chans)
        if mzs.size == 0:
            continue
        ions.append(mzs)
        node_of.append(np.full(mzs.shape[0], n, dtype=np.int64))
        cmf_flag.append(np.concatenate(flags))

    if not ions:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=bool),
        )
    return (
        np.concatenate(ions).astype(np.float64),
        np.concatenate(node_of),
        np.concatenate(cmf_flag),
    )


def _group_by_node(
    ion_mz: np.ndarray, node_of_cand: np.ndarray, is_cmf: np.ndarray, num_nodes: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group flattened candidate ion m/z and their CMF flags by owning node."""
    out: list[tuple[np.ndarray, np.ndarray]] = [
        (np.zeros(0, dtype=np.float64), np.zeros(0, dtype=bool))
    ] * num_nodes
    order = np.argsort(node_of_cand, kind="stable")
    nodes_sorted = node_of_cand[order]
    mz_sorted = ion_mz[order]
    cmf_sorted = is_cmf[order]
    edges = np.searchsorted(nodes_sorted, np.arange(num_nodes + 1))
    for n in range(num_nodes):
        out[n] = (mz_sorted[edges[n] : edges[n + 1]], cmf_sorted[edges[n] : edges[n + 1]])
    return out


def _match_peaks_to_nodes(
    true_mzs: np.ndarray,
    ion_mz: np.ndarray,
    node_of_cand: np.ndarray,
    num_nodes: int,
    tolerance: float,
    tolerance_min_mz: float,
) -> np.ndarray:
    """Boolean (num_true_peaks, num_nodes) matrix over already-converted candidate ion m/z."""
    match = calculate_match_mzs(
        th.tensor(true_mzs, dtype=th.float64),
        th.tensor(ion_mz, dtype=th.float64),
        tolerance=tolerance,
        relative=True,
        tolerance_min_mz=tolerance_min_mz,
    ).numpy()
    peak_node = np.zeros((true_mzs.shape[0], num_nodes), dtype=bool)
    hit_i, hit_j = np.nonzero(match)
    peak_node[hit_i, node_of_cand[hit_j]] = True
    return peak_node


def _envelopes_overlap(
    ions_a: np.ndarray,
    ions_b: np.ndarray,
    tolerance: float,
    tolerance_min_mz: float,
    cmf_a: np.ndarray | None = None,
    cmf_b: np.ndarray | None = None,
) -> tuple[bool, str]:
    """Whether A's and B's candidate ion envelopes collide under the matching contract.

    An observation in a shared channel cannot identify a side, so such pairs are excluded rather
    than treated as evidence. Inputs are already-converted ion m/z from ``node_candidate_ions``,
    so the precursor's CRF/CMF rule is applied exactly once, upstream.

    The window is **two** tolerances wide, not one. The matcher accepts a peak within one tolerance
    of a candidate, so a peak sitting between two candidates matches both whenever their centres
    are up to two tolerances apart: A = 100.0000 and B = 100.0030 at a 0.0020 Da tolerance are
    3 mDa apart, yet an observation at 100.0015 matches each of them. A one-tolerance gate misses
    exactly those pairs, and the shared peak is then silently dropped from both directional masks
    while the pair still trains on whatever evidence is left.
    """
    if ions_a.size == 0 or ions_b.size == 0:
        return False, "none"
    if cmf_a is None:
        cmf_a = np.zeros(ions_a.shape[0], dtype=bool)
    if cmf_b is None:
        cmf_b = np.zeros(ions_b.shape[0], dtype=bool)
    diff = np.abs(ions_a[:, None] - ions_b[None, :])
    # Divide by the smaller centre: the matcher's divisor is the observed peak, which lies between
    # the two candidates, so this is the conservative (widest) window.
    divisor = np.maximum(np.minimum(ions_a[:, None], ions_b[None, :]), tolerance_min_mz)
    hit = diff / divisor <= 2.0 * tolerance
    if not hit.any():
        return False, "none"

    # Attribute the collision so the audit can say which channel produced it.
    hit_i, hit_j = np.nonzero(hit)
    exact = bool((diff[hit_i, hit_j] == 0.0).any())
    a_cmf, b_cmf = cmf_a[hit_i], cmf_b[hit_j]
    if (~a_cmf & ~b_cmf).any():
        channel = "crf_crf"
    elif (a_cmf & b_cmf).any():
        channel = "cmf_cmf"
    else:
        channel = "crf_cmf"
    return True, f"{channel}_{'exact' if exact else 'tolerance'}"


def pair_evidence(
    rel_ints: np.ndarray,
    n_match: np.ndarray,
    dir_a: np.ndarray,
    dir_b: np.ndarray,
    l_a: float,
    l_b: float,
) -> tuple[float, float, float] | None:
    """Score one complementary pair from its directional peaks.

    The target and the weight deliberately read different channels: ``q`` is built from
    undiscounted intensity so that softness tracks distance from the detection limit, while
    ``w_pair`` is responsibility-discounted so that ambiguous attribution trains weakly. Mixing
    them would make a 20% peak at confidence 0.1 indistinguishable from a clean 2% peak.

    Args:
        rel_ints: Per-spectrum normalised peak intensities of shape (num_peaks,).
        n_match: Number of DAG nodes matching each peak, shape (num_peaks,).
        dir_a: Boolean mask of peaks matching side A only, shape (num_peaks,).
        dir_b: Boolean mask of peaks matching side B only, shape (num_peaks,).
        l_a: Side-level detection floor for A, on the same scale as ``rel_ints``.
        l_b: Side-level detection floor for B.

    Returns:
        Dict with ``q_a``/``q_b`` (the target), ``w_pair`` (responsibility-weighted training
        weight), ``w_strict`` (unique-match-only, audit-only), ``evidence_lod_ratio`` (how far the
        stronger side sits above its floor), ``dir_intensity`` (directional intensity explained),
        and ``n_dir_peaks``. None if neither side clears its floor.
    """
    i_a = float(rel_ints[dir_a].sum())
    i_b = float(rel_ints[dir_b].sum())
    if i_a <= 0 and i_b <= 0:
        return None
    ratio = max(i_a / l_a, i_b / l_b)
    if ratio <= 1.0:
        return None

    w_pair = float((rel_ints[dir_a] / n_match[dir_a]).sum()) + float(
        (rel_ints[dir_b] / n_match[dir_b]).sum()
    )
    unique_a = dir_a & (n_match == 1)
    unique_b = dir_b & (n_match == 1)
    w_strict = float(rel_ints[unique_a].sum()) + float(rel_ints[unique_b].sum())
    u_a, u_b = max(i_a, l_a), max(i_b, l_b)
    denom = u_a + u_b
    return {
        "q_a": u_a / denom,
        "q_b": u_b / denom,
        "w_pair": w_pair,
        "w_strict": w_strict,
        "evidence_lod_ratio": ratio,
        "dir_intensity": i_a + i_b,
        "n_dir_peaks": int(dir_a.sum() + dir_b.sum()),
    }


def _empty_mol_result(group_rows: list[dict]) -> dict:
    return {
        "spec_ids": np.asarray([row["spec_id"] for row in group_rows], dtype=np.int64),
        "spec_group_counts": np.zeros(len(group_rows), dtype=np.int32),
        "group_child_counts": np.zeros(0, dtype=np.int32),
        "child_nodes": np.zeros(0, dtype=np.int32),
        "q_obs": np.zeros(0, dtype=np.float32),
        "q_obs_shuffled": np.zeros(0, dtype=np.float32),
        "group_weights": np.zeros(0, dtype=np.float32),
        "audit": defaultdict(float),
        "audit_rows": [],
        "min_fragment_size": None,
    }


def _process_mol_group(
    mol_id: int,
    group_rows: list[dict],
    mol,
    frag_dp: str,
    is_compressed: bool,
    max_h_transfer: int | None,
    shuffle_seed: int,
    max_pairs_per_parent: int,
    lod_tables: tuple[dict, dict, float],
) -> dict:
    """Build cut-pair target arrays for every spectrum of one molecule."""
    import fragnnet.models.fragnnet.frag_utils as frag_utils

    by_inst_bin, by_inst, global_floor = lod_tables
    result = _empty_mol_result(group_rows)
    audit = result["audit"]

    dag_d = frag_utils.load_frag_d(mol_id, frag_dp, is_compressed=is_compressed)
    if dag_d is None:
        return result

    dag = dag_d["dag"]
    num_nodes = int(dag.x.shape[0])
    nfi = dag.node_feat_idxs[0]
    masks = th_long_to_mask(get_node_feats(dag.x, nfi, "cc").long()).numpy().astype(bool)
    edge_index = dag.edge_index.numpy()

    # 1. Structural pairs, deduplicated -- CE-independent, so computed once per molecule. This is
    #    a per-MOLECULE count; every later counter is per pair-spectrum. Keep them separate.
    pairs = _complementary_pairs(masks, edge_index)
    audit["structural_pairs"] += len(pairs)
    audit["complement_incomplete_children"] += _complement_incompleteness(masks, edge_index)
    audit["molecules"] += 1
    # Smallest fragment anywhere bounds the generation-time min_frag_atoms; checked in main().
    result["min_fragment_size"] = int(masks.sum(axis=1).min())
    if not pairs:
        return result

    allowed = set(_allowed_formula_idxs(dag_d["idx_by_h_delta"], max_h_transfer))
    h_formulae_idx = get_node_feats(dag.x, nfi, "h_formulae_idx").numpy()
    formula_peak_mzs = dag_d["formula_peak_mzs"].numpy()
    formula_element_counts = np.asarray(
        [tracked_element_counts(formula) for formula in dag_d["idx_to_formula"].values()],
        dtype=np.int16,
    )
    # The depth feature is a range indicator over depths, not a scalar: the first set column is
    # the node's minimum depth. argmax returns that first 1.
    depth = get_node_feats(dag.x, nfi, "depth").numpy().argmax(axis=1)

    spec_group_counts, group_child_counts = [], []
    child_nodes, q_obs, q_obs_shuffled, group_weights = [], [], [], []
    # Candidate ions are precursor-dependent (the CRF/CMF rule differs per adduct), so they are
    # built once per prec_type rather than once per molecule.
    cand_cache: dict[str, tuple[np.ndarray, np.ndarray, list[np.ndarray]]] = {}
    alias_cache: dict[tuple[str, str, int, int], bool] = {}

    for row in group_rows:
        groups_before = len(group_child_counts)
        peaks = row["peaks"]
        if not peaks:
            spec_group_counts.append(0)
            continue
        true_mzs, true_ints = (np.asarray(v, dtype=np.float64) for v in zip(*peaks))
        total_intensity = float(true_ints.sum())
        if total_intensity <= 0:
            spec_group_counts.append(0)
            continue
        rel_ints = true_ints / total_intensity

        inst = str(row["inst_type"])
        prec_type = str(row["prec_type"])
        if prec_type not in PREC_TYPE_TO_MASS_DIFF:
            audit["unsupported_prec_type_spectra"] += 1
            spec_group_counts.append(0)
            continue
        tolerance, tolerance_min_mz = _tolerance_for_inst_type(inst)

        if prec_type not in cand_cache:
            ion_mz, node_of_cand, is_cmf = node_candidate_ions(
                h_formulae_idx,
                formula_peak_mzs,
                allowed,
                prec_type,
                max_h_transfer,
                formula_element_counts,
            )
            cand_cache[prec_type] = (
                ion_mz,
                node_of_cand,
                _group_by_node(ion_mz, node_of_cand, is_cmf, num_nodes),
            )
        ion_mz, node_of_cand, cand_by_node = cand_cache[prec_type]
        if ion_mz.size == 0:
            spec_group_counts.append(0)
            continue

        peak_node = _match_peaks_to_nodes(
            true_mzs, ion_mz, node_of_cand, num_nodes, tolerance, tolerance_min_mz
        )
        # n_j: how many DAG nodes -- not just this pair -- can claim each peak.
        n_match = peak_node.sum(axis=1).astype(np.float64)
        directional_peaks: set[int] = set()

        # 2. Filters, in the order the plan fixes: alias -> zero evidence -> sub-LOD -> cap.
        retained_by_parent: dict[int, list[dict]] = defaultdict(list)
        for parent, a, b in pairs:
            audit["candidate_pair_spectra"] += 1
            mz_a, cmf_a = cand_by_node[a]
            mz_b, cmf_b = cand_by_node[b]
            key = (prec_type, inst, a, b)
            if key in alias_cache:
                aliased, channel = alias_cache[key]
            else:
                aliased, channel = _envelopes_overlap(
                    mz_a, mz_b, tolerance, tolerance_min_mz, cmf_a, cmf_b
                )
                alias_cache[key] = (aliased, channel)
            if aliased:
                audit["alias_dropped"] += 1
                audit[f"alias_channel__{channel}"] += 1
                result["audit_rows"].append(
                    {
                        "kind": "discarded",
                        "reason": "alias",
                        "size_min": int(min(masks[a].sum(), masks[b].sum())),
                    }
                )
                continue

            hit_a, hit_b = peak_node[:, a], peak_node[:, b]
            # Exact backstop to the envelope gate: if any observed peak actually matched both
            # sides, this spectrum cannot identify a side for this cut. Dropping the shared peak
            # from both directional masks and training on the remainder is precisely what the plan
            # forbids -- "unknown" must not become evidence.
            if (hit_a & hit_b).any():
                audit["shared_peak_dropped"] += 1
                continue

            dir_a = hit_a & ~hit_b
            dir_b = hit_b & ~hit_a
            size_min = int(min(masks[a].sum(), masks[b].sum()))
            if rel_ints[dir_a].sum() <= 0 and rel_ints[dir_b].sum() <= 0:
                audit["zero_evidence_dropped"] += 1
                result["audit_rows"].append(
                    {"kind": "discarded", "reason": "zero_evidence", "size_min": size_min}
                )
                continue

            l_a = _lod_for(inst, mz_a, by_inst_bin, by_inst, global_floor)
            l_b = _lod_for(inst, mz_b, by_inst_bin, by_inst, global_floor)
            scored = pair_evidence(rel_ints, n_match, dir_a, dir_b, l_a, l_b)
            if scored is None:
                audit["sub_lod_dropped"] += 1
                result["audit_rows"].append(
                    {"kind": "discarded", "reason": "sub_lod", "size_min": size_min}
                )
                continue

            directional_peaks.update(np.nonzero(dir_a | dir_b)[0].tolist())
            scored.update(
                {
                    "parent": parent,
                    "a": a,
                    "b": b,
                    "size_min": size_min,
                    "bond_type": _cut_bond_type(mol, masks[a], masks[b]),
                }
            )
            retained_by_parent[parent].append(scored)

        # 3. Cap AFTER the evidence filters, so sampling does not mostly select unobserved pairs.
        capped: list[dict] = []
        for parent, entries in retained_by_parent.items():
            if len(entries) > max_pairs_per_parent:
                rng = np.random.default_rng(
                    _shuffle_seed(shuffle_seed, int(row["spec_id"]), parent, 0, 0)
                )
                idx = rng.choice(len(entries), size=max_pairs_per_parent, replace=False)
                audit["capped_dropped"] += len(entries) - max_pairs_per_parent
                entries = [entries[i] for i in sorted(idx)]
            capped.extend(entries)
            result["audit_rows"].append({"kind": "pairs_per_parent", "value": len(entries)})

        # 4. Node multiplicity is only knowable once the final retained set is fixed.
        mult: dict[int, int] = defaultdict(int)
        for e in capped:
            mult[e["a"]] += 1
            mult[e["b"]] += 1

        for e in sorted(capped, key=lambda d: (d["parent"], d["a"], d["b"])):
            m_mean = (mult[e["a"]] + mult[e["b"]]) / 2.0
            weight = e["w_pair"] / m_mean if m_mean > 0 else 0.0
            if weight <= 0:
                audit["zero_weight_dropped"] += 1
                continue
            target = np.asarray([e["q_a"], e["q_b"]], dtype=np.float32)
            rng = np.random.default_rng(
                _shuffle_seed(shuffle_seed, int(row["spec_id"]), e["parent"], e["a"], e["b"])
            )
            shuffled = target[rng.permutation(2)]

            group_child_counts.append(2)
            child_nodes.extend([e["a"], e["b"]])
            q_obs.extend(target.tolist())
            q_obs_shuffled.extend(shuffled.tolist())
            group_weights.append(weight)
            result["audit_rows"].append(
                {
                    "kind": "pair",
                    # Identity, so the Phase 2 replicate gate can match the same cut across
                    # replicate spectra rather than only reading marginal distributions.
                    "spec_id": int(row["spec_id"]),
                    "mol_id": int(mol_id),
                    "parent": int(e["parent"]),
                    "a": int(e["a"]),
                    "b": int(e["b"]),
                    "q_a": float(target[0]),
                    "q_max": float(target.max()),
                    "w_pair": float(e["w_pair"]),
                    "w_strict": float(e["w_strict"]),
                    "weight": float(weight),
                    "mult": float(m_mean),
                    "inst_type": inst,
                    "prec_type": prec_type,
                    "frag_mode": str(row["frag_mode"]),
                    "nce": float(row["nce"]),
                    "depth": int(max(depth[e["a"]], depth[e["b"]])),
                    "bond_type": e["bond_type"],
                    "size_min": e["size_min"],
                    "evidence_lod_ratio": float(e["evidence_lod_ratio"]),
                    "n_dir_peaks": int(e["n_dir_peaks"]),
                }
            )

        # Directional explained-intensity coverage: the share of this spectrum's intensity carried
        # by peaks any retained pair attributes to exactly one side. This is the common definition
        # the cut-vs-sibling comparison must use on both target families.
        if directional_peaks:
            idx = np.fromiter(directional_peaks, dtype=np.int64)
            result["audit_rows"].append(
                {
                    "kind": "spectrum",
                    "spec_id": int(row["spec_id"]),
                    "directional_coverage": float(rel_ints[idx].sum()),
                    "n_directional_peaks": int(idx.size),
                    "n_peaks": int(rel_ints.size),
                }
            )

        spec_group_counts.append(len(group_child_counts) - groups_before)

    result.update(
        {
            "spec_group_counts": np.asarray(spec_group_counts, dtype=np.int32),
            "group_child_counts": np.asarray(group_child_counts, dtype=np.int32),
            "child_nodes": np.asarray(child_nodes, dtype=np.int32),
            "q_obs": np.asarray(q_obs, dtype=np.float32),
            "q_obs_shuffled": np.asarray(q_obs_shuffled, dtype=np.float32),
            "group_weights": np.asarray(group_weights, dtype=np.float32),
        }
    )
    return result


def _kish_ess(weights: np.ndarray) -> float:
    """Kish effective sample size ``(sum w)^2 / sum(w^2)``."""
    if weights.size == 0:
        return 0.0
    denom = float((weights.astype(np.float64) ** 2).sum())
    return float(weights.astype(np.float64).sum() ** 2 / denom) if denom > 0 else 0.0


def _concat_results(results: list[dict]) -> dict:
    """Combine molecule-local arrays into one ragged tensor cache."""
    names = [
        "spec_ids",
        "spec_group_counts",
        "group_child_counts",
        "child_nodes",
        "q_obs",
        "q_obs_shuffled",
        "group_weights",
    ]
    combined = {name: np.concatenate([r[name] for r in results]) for name in names}
    if np.unique(combined["spec_ids"]).size != combined["spec_ids"].size:
        raise ValueError("Input splits contain duplicate spec_id values.")

    spec_group_ptr = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(combined["spec_group_counts"], dtype=np.int64)]
    )
    group_child_ptr = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(combined["group_child_counts"], dtype=np.int64)]
    )
    weights = combined["group_weights"]
    stats = {
        "num_spectra": int(combined["spec_ids"].size),
        "num_groups": int(weights.size),
        "num_child_entries": int(combined["child_nodes"].size),
        "mean_group_weight": float(weights.mean()) if weights.size else 0.0,
        "kish_ess": _kish_ess(weights),
    }
    return {
        "version": CACHE_VERSION,
        "spec_ids": th.from_numpy(combined["spec_ids"]),
        "spec_group_ptr": th.from_numpy(spec_group_ptr),
        "group_child_ptr": th.from_numpy(group_child_ptr),
        "child_nodes": th.from_numpy(combined["child_nodes"]),
        "q_obs": th.from_numpy(combined["q_obs"]),
        "q_obs_shuffled": th.from_numpy(combined["q_obs_shuffled"]),
        "group_weights": th.from_numpy(weights),
        "stats": stats,
    }


def _build_audit(results: list[dict], cache: dict, split_of: dict[int, str]) -> dict:
    """Assemble the sidecar audit report required before any Phase 2 gate is run."""
    counters: dict[str, float] = defaultdict(float)
    for r in results:
        for k, v in r["audit"].items():
            counters[k] += v

    rows = [row for r in results for row in r["audit_rows"]]
    pairs = [row for row in rows if row["kind"] == "pair"]
    ppp = np.asarray([row["value"] for row in rows if row["kind"] == "pairs_per_parent"])

    spec_ids = cache["spec_ids"].numpy()
    ptr = cache["spec_group_ptr"].numpy()
    per_spec_groups = np.diff(ptr)
    split_arr = np.asarray([split_of[int(s)] for s in spec_ids])

    # Structural facts are per MOLECULE; every filter counter below is per PAIR-SPECTRUM. Mixing
    # the two makes "candidates" look smaller than "retained", which is how the earlier report
    # showed 3,394 candidates against 4,580 retained groups.
    retained = int(cache["group_weights"].numel())
    considered = int(counters["candidate_pair_spectra"])
    filtered = {
        k: int(counters[k])
        for k in (
            "alias_dropped",
            "shared_peak_dropped",
            "zero_evidence_dropped",
            "sub_lod_dropped",
            "capped_dropped",
            "zero_weight_dropped",
        )
    }
    report: dict = {
        "builder": BUILDER_NAME,
        "cache_version": CACHE_VERSION,
        "structural": {
            "molecules": int(counters["molecules"]),
            "structural_pairs": int(counters["structural_pairs"]),
            "complement_incomplete_children": int(counters["complement_incomplete_children"]),
            "min_fragment_size": min(
                (r["min_fragment_size"] for r in results if r["min_fragment_size"] is not None),
                default=None,
            ),
        },
        "pair_spectra": {
            "candidate_pair_spectra": considered,
            **filtered,
            "retained": retained,
            "unsupported_prec_type_spectra": int(counters["unsupported_prec_type_spectra"]),
            "balance_check_ok": considered == retained + sum(filtered.values()),
        },
        "retained_groups": retained,
        "kish_ess": cache["stats"]["kish_ess"],
    }

    for split in sorted(set(split_arr)):
        sel = split_arr == split
        g = per_spec_groups[sel]
        report[f"split_{split}"] = {
            "num_spectra": int(sel.sum()),
            "num_spectra_with_zero_groups": int((g == 0).sum()),
            "groups_per_spectrum_mean": float(g.mean()) if g.size else 0.0,
            "groups_per_spectrum_p50": float(np.percentile(g, 50)) if g.size else 0.0,
            "groups_per_spectrum_p95": float(np.percentile(g, 95)) if g.size else 0.0,
        }

    if pairs:
        pair_split = np.asarray([split_of[p["spec_id"]] for p in pairs])
        for split in sorted(set(pair_split)):
            report[f"split_{split}"].update(
                _pair_block([p for p in pairs if split_of[p["spec_id"]] == split])
            )
        report["targets_all_splits"] = _pair_block(pairs)
        report["strata"] = _strata_block(pairs)
        report["alias_channels"] = {
            k.removeprefix("alias_channel__"): int(v)
            for k, v in sorted(counters.items())
            if k.startswith("alias_channel__")
        }
        report["retained_vs_discarded"] = _retained_vs_discarded_block(rows, pairs)

    spectra_rows = [row for row in rows if row["kind"] == "spectrum"]
    if spectra_rows:
        cov = np.asarray([r["directional_coverage"] for r in spectra_rows])
        n_dir = np.asarray([r["n_directional_peaks"] for r in spectra_rows])
        n_pk = np.asarray([r["n_peaks"] for r in spectra_rows])
        report["directional_coverage"] = {
            "definition": (
                "share of a spectrum's total intensity carried by peaks that some retained pair "
                "attributes to exactly one side; use this same definition for sibling targets"
            ),
            "spectra_with_any_directional_peak": len(spectra_rows),
            "mean_explained_intensity_fraction": float(cov.mean()),
            "p50": float(np.percentile(cov, 50)),
            "p95": float(np.percentile(cov, 95)),
            "distinct_directional_peaks_total": int(n_dir.sum()),
            "mean_directional_peaks_per_spectrum": float(n_dir.mean()),
            "mean_peak_fraction_directional": float((n_dir / np.maximum(n_pk, 1)).mean()),
        }

    if ppp.size:
        report["pairs_per_parent"] = {
            "p50": float(np.percentile(ppp, 50)),
            "p95": float(np.percentile(ppp, 95)),
            "max": float(ppp.max()),
        }
    return report


def _pair_block(pairs: list[dict]) -> dict:
    """Target, weight, and structural-count summaries for one set of retained pairs."""
    if not pairs:
        return {"num_pairs": 0}
    q_max = np.asarray([p["q_max"] for p in pairs])
    w_pair = np.asarray([p["w_pair"] for p in pairs])
    w_strict = np.asarray([p["w_strict"] for p in pairs])
    weight = np.asarray([p["weight"] for p in pairs])
    mult = np.asarray([p["mult"] for p in pairs])
    high = q_max >= HIGH_CONF_Q
    return {
        "num_pairs": len(pairs),
        "num_molecules": len({p["mol_id"] for p in pairs}),
        "num_distinct_cut_facts": len({(p["spec_id"], p["parent"], p["a"], p["b"]) for p in pairs}),
        "num_distinct_cuts_across_spectra": len(
            {(p["mol_id"], p["parent"], p["a"], p["b"]) for p in pairs}
        ),
        "targets": {
            "q_max_mean": float(q_max.mean()),
            "q_max_p50": float(np.percentile(q_max, 50)),
            "q_max_hist_0.5_to_1.0": np.histogram(q_max, bins=10, range=(0.5, 1.0))[0].tolist(),
            "frac_q_max_gt_0.99": float((q_max > 0.99).mean()),
        },
        f"high_conf_subset_q>={HIGH_CONF_Q}": {
            "num_pairs": int(high.sum()),
            "frac": float(high.mean()),
            "num_molecules": len({p["mol_id"] for p, k in zip(pairs, high) if k}),
            "kish_ess": _kish_ess(weight[high]),
        },
        "weights": {
            "w_pair_mean": float(w_pair.mean()),
            "w_pair_p50": float(np.percentile(w_pair, 50)),
            "w_strict_mean": float(w_strict.mean()),
            "w_strict_nonzero_frac": float((w_strict > 0).mean()),
            "strict_over_responsibility_ratio": (
                float(w_strict.sum() / w_pair.sum()) if w_pair.sum() > 0 else 0.0
            ),
            "kish_ess_before_multiplicity": _kish_ess(w_pair),
            "kish_ess_after_multiplicity": _kish_ess(weight),
            "node_multiplicity_mean": float(mult.mean()),
            "node_multiplicity_p95": float(np.percentile(mult, 95)),
            "node_multiplicity_max": float(mult.max()),
        },
    }


def _retained_vs_discarded_block(rows: list[dict], pairs: list[dict]) -> dict:
    """Compare the size composition of retained pairs against each discard reason.

    A retained set biased toward large-vs-large splits would mean the target never sees the small
    counter-fragments the whole hypothesis is about.
    """
    discarded = [r for r in rows if r["kind"] == "discarded"]
    out: dict = {
        "retained": _size_summary([p["size_min"] for p in pairs]),
    }
    by_reason: dict[str, list[int]] = defaultdict(list)
    for r in discarded:
        by_reason[r["reason"]].append(r["size_min"])
    for reason, sizes in sorted(by_reason.items()):
        out[f"discarded_{reason}"] = _size_summary(sizes)
    return out


def _size_summary(sizes: list[int]) -> dict:
    """Distribution of the smaller side's heavy-atom count."""
    if not sizes:
        return {"n": 0}
    arr = np.asarray(sizes, dtype=np.float64)
    return {
        "n": int(arr.size),
        "smaller_side_atoms_mean": float(arr.mean()),
        "smaller_side_atoms_p50": float(np.percentile(arr, 50)),
        "smaller_side_atoms_p95": float(np.percentile(arr, 95)),
        "frac_smaller_side_le_2_atoms": float((arr <= 2).mean()),
    }


def _strata_block(pairs: list[dict]) -> dict:
    """Retained-pair counts and mean target strength by every stratum the gates report on."""
    nce = np.asarray([p["nce"] for p in pairs], dtype=np.float64)
    nce_bins = np.digitize(nce, [20.0, 40.0, 60.0, 80.0, 120.0])
    nce_labels = ["<20", "20-40", "40-60", "60-80", "80-120", ">=120"]

    # How far the stronger side sits above its detection floor. q is a deterministic function of
    # this ratio, so the joint distribution shows whether targets are driven by real evidence or
    # by pairs sitting just over the threshold.
    ratio = np.asarray([p["evidence_lod_ratio"] for p in pairs], dtype=np.float64)
    ratio_bins = np.digitize(ratio, [2.0, 4.0, 10.0, 50.0])
    ratio_labels = ["1-2x", "2-4x", "4-10x", "10-50x", ">=50x"]

    out: dict[str, dict] = {}
    for name, keys in (
        ("by_inst_type", [p["inst_type"] for p in pairs]),
        ("by_prec_type", [p["prec_type"] for p in pairs]),
        ("by_frag_mode", [p["frag_mode"] for p in pairs]),
        ("by_depth", [str(p["depth"]) for p in pairs]),
        ("by_bond_type", [p["bond_type"] for p in pairs]),
        ("by_nce", [nce_labels[b] for b in nce_bins]),
        ("by_evidence_lod_ratio", [ratio_labels[b] for b in ratio_bins]),
    ):
        counts: dict[str, int] = defaultdict(int)
        qsum: dict[str, float] = defaultdict(float)
        for key, pair in zip(keys, pairs):
            counts[key] += 1
            qsum[key] += pair["q_max"]
        out[name] = {
            k: {"num_pairs": counts[k], "q_max_mean": qsum[k] / counts[k]}
            for k in sorted(counts, key=lambda x: -counts[x])
        }
    return out


def cache_fingerprint(cache: dict) -> str:
    """Short content hash tying a run's provenance to this exact cache.

    Hashes tensor payloads and metadata rather than the saved file, because ``torch.save`` is not
    byte-reproducible -- a file-level hash changes on every rerun and cannot anchor provenance.

    ``metadata.cache_fingerprint`` is excluded from the hashed view so the value stays verifiable
    after it is stamped in; hashing it would make every saved cache disagree with itself.
    """
    h = hashlib.sha256()
    for key in sorted(k for k, v in cache.items() if isinstance(v, th.Tensor)):
        h.update(key.encode())
        h.update(np.ascontiguousarray(cache[key].numpy()).tobytes())
    canonical = {k: v for k, v in cache["metadata"].items() if k != "cache_fingerprint"}
    h.update(json.dumps(canonical, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def verify_cache_coverage(cache: dict, required_spec_ids: set[int]) -> None:
    """Fail unless the cache carries an entry for every spec_id the dataset will request.

    A spectrum with no usable pair groups still needs a zero-group entry: the training loader
    raises on a batch in which no spec_id is cached at all, and the builder's own spectrum filter
    is stricter than the dataset's (the dataset fills some missing collision energies that the
    builder drops). Reporting the gap is not enough -- an absent id is a crash at training time.

    Args:
        cache: The built cache, carrying ``spec_ids``.
        required_spec_ids: Exact spec_ids the instantiated dataset will ask for.

    Raises:
        ValueError: If any required spec_id has no entry.
    """
    present = {int(s) for s in cache["spec_ids"].tolist()}
    missing = sorted(required_spec_ids - present)
    if missing:
        raise ValueError(
            f"Cache is missing {len(missing)} of {len(required_spec_ids)} required spec_ids "
            f"(e.g. {missing[:10]}). Training would hit a wholly uncached batch."
        )


def _pad_missing_spec_ids(results: list[dict], required_spec_ids: set[int]) -> list[dict]:
    """Append zero-group entries so every required spec_id is represented in the cache."""
    present = {int(s) for r in results for s in r["spec_ids"].tolist()}
    missing = sorted(required_spec_ids - present)
    if not missing:
        return results
    pad = _empty_mol_result([{"spec_id": s} for s in missing])
    log.info("Padding cache with %d zero-group spectra so coverage is total.", len(missing))
    return [*results, pad]


def verify_cache_fingerprint(cache: dict) -> None:
    """Recompute a saved cache's fingerprint and confirm it matches the stored stamp.

    Raises:
        ValueError: If the cache carries no stamp, or the recomputed hash differs.
    """
    stored = cache["metadata"]["cache_fingerprint"]
    recomputed = cache_fingerprint(cache)
    if stored != recomputed:
        raise ValueError(
            f"Cache fingerprint mismatch: stored {stored}, recomputed {recomputed}. "
            "The cache was modified after stamping, or the hash is not self-consistent."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--frag_dp", required=True)
    parser.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    parser.add_argument("--split_dp", default="data/split/nist20_ma_mi_inchikey")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--dsets", "--dset", nargs="+", default=None)
    parser.add_argument("--max_h_transfer", type=int, default=None)
    parser.add_argument("--max_pairs_per_parent", type=int, default=32)
    parser.add_argument("--limit_mols", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle_seed", type=int, default=1729)
    parser.add_argument("--n_jobs", type=int, default=data_utils.JOBLIB_N_JOBS)
    parser.add_argument("--out_fp", required=True)
    parser.add_argument("--report_fp", default=None)
    parser.add_argument("--lod_quantile", type=float, default=LOD_QUANTILE)
    parser.add_argument("--mol_df_fp", default="data/proc/nist20/mol_df.pkl")
    parser.add_argument(
        "--require_ids_fp",
        default=None,
        help=(
            "CSV with a spec_id column listing the exact ids the instantiated train/val dataset "
            "will request. The build fails if the cache does not cover every one of them. Use "
            "this to reconcile against the real control dataset, whose spectrum filter differs "
            "from this script's."
        ),
    )
    args = parser.parse_args()

    # 1. Load the requested splits, keeping each spec_id's split label for the audit.
    dsets = resolve_dsets_for_spec_fp(args.spec_fp, args.dsets)
    spec_df, load_counts = load_spec_df_with_ce_fill(args.spec_fp, dsets)
    log.info("Spectrum load: %s", load_counts)
    split_of: dict[int, str] = {}
    split_ids: set[int] = set()
    for split in args.splits:
        ids = pd.read_csv(os.path.join(args.split_dp, f"{split}_ids.csv"))["spec_id"].tolist()
        split_ids.update(ids)
        split_of.update({int(i): split for i in ids})
    spec_df = spec_df[spec_df["spec_id"].isin(split_ids)]

    # Spectrum loading drops unsupported adducts, empty spectra, and rows with missing metadata.
    # Record exactly which split IDs did not survive: training raises on a wholly uncached batch,
    # so this gap is what a run must reconcile against its dataset before it starts.
    loaded_ids = {int(s) for s in spec_df["spec_id"]}
    dropped_by_loader = sorted(split_ids - loaded_ids)
    coverage = {
        "split_ids_requested": len(split_ids),
        "spectra_loaded": len(loaded_ids),
        "dropped_by_spec_loader": len(dropped_by_loader),
        "dropped_fraction": len(dropped_by_loader) / max(len(split_ids), 1),
        "dropped_id_sample": dropped_by_loader[:20],
    }
    if dropped_by_loader:
        log.warning(
            "%d/%d requested split spec_ids were dropped by the spectrum loader and will be "
            "absent from the cache. Reconcile against the training dataset before use.",
            len(dropped_by_loader),
            len(split_ids),
        )

    if args.limit_mols is not None:
        rng = np.random.default_rng(args.seed)
        mol_ids = spec_df["mol_id"].unique()
        keep = rng.choice(mol_ids, size=min(args.limit_mols, len(mol_ids)), replace=False)
        spec_df = spec_df[spec_df["mol_id"].isin(keep)]
        coverage["limit_mols_applied"] = int(args.limit_mols)

    # 2. Freeze the detection limit on train only, before it can see validation.
    train_rows = spec_df[spec_df["spec_id"].map(lambda s: split_of[int(s)] == "train")]
    if train_rows.empty:
        raise ValueError("No training spectra available; the LOD floor must be frozen on train.")
    train_records = train_rows.to_dict("records")
    lod_tables = estimate_lod_table(train_records, quantile=args.lod_quantile)
    lod_audit = lod_quantile_audit(train_records)
    log.info("LOD global floor by quantile: %s", {k: f"{v:.3e}" for k, v in lod_audit.items()})
    log.info(
        "LOD floor frozen on %d train spectra: %d (inst, mz-bin) strata, global=%.3e",
        len(train_rows),
        len(lod_tables[0]),
        lod_tables[2],
    )

    groups = spec_df.groupby("mol_id", sort=False)
    sample_mol_id = int(next(iter(groups.groups)))
    is_compressed = _detect_is_compressed(args.frag_dp, sample_mol_id)
    # Molecules are needed only to classify the severed bond for the bond-type stratum.
    mol_df = pd.read_pickle(args.mol_df_fp)
    mol_by_id = dict(zip(mol_df["mol_id"], mol_df["mol"], strict=True))
    mol_inputs = [
        (int(mol_id), group.to_dict("records"), mol_by_id[int(mol_id)]) for mol_id, group in groups
    ]

    # 3. Build molecule-wise; node indices stay local to each DAG.
    worker = functools.partial(
        _process_mol_group,
        frag_dp=args.frag_dp,
        is_compressed=is_compressed,
        max_h_transfer=args.max_h_transfer,
        shuffle_seed=args.shuffle_seed,
        max_pairs_per_parent=args.max_pairs_per_parent,
        lod_tables=lod_tables,
    )
    log.info(
        "Building cut-pair targets for %d spectra from %d molecules.",
        len(spec_df),
        len(mol_inputs),
    )
    per_mol = data_utils.par_apply(mol_inputs, worker, need_unpack=True, n_jobs=args.n_jobs)

    # 4. Gate on min_frag_atoms before anything is written: a filtered corpus makes the whole
    #    target biased, so this must fail the build rather than annotate it.
    observed_min_size = min(
        (r["min_fragment_size"] for r in per_mol if r["min_fragment_size"] is not None),
        default=None,
    )
    if observed_min_size is None:
        raise ValueError("No DAG was loaded; cannot verify min_frag_atoms or build targets.")
    manifest = read_generation_manifest(args.frag_dp)
    min_frag_evidence = check_min_frag_atoms(manifest, observed_min_size)
    log.info("min_frag_atoms gate passed -- %s", min_frag_evidence)

    # 5. Concatenate, stamp provenance, and save one portable CPU cache plus its audit sidecar.
    #    Pad first: every requested split id must have an entry even when the spectrum loader
    #    dropped it, otherwise a batch of those ids arrives wholly uncached and training raises.
    required_ids = split_ids if args.limit_mols is None else {int(s) for s in spec_df["spec_id"]}
    cache = _concat_results(_pad_missing_spec_ids(per_mol, required_ids))
    verify_cache_coverage(cache, required_ids)
    if args.require_ids_fp is not None:
        external = {int(i) for i in pd.read_csv(args.require_ids_fp)["spec_id"]}
        verify_cache_coverage(cache, external)
        log.info("Cache covers all %d spec_ids in %s.", len(external), args.require_ids_fp)
    cache["metadata"] = {
        "builder": BUILDER_NAME,
        "frag_dp": args.frag_dp,
        "spec_fp": args.spec_fp,
        "split_dp": args.split_dp,
        "splits": list(args.splits),
        "max_h_transfer": args.max_h_transfer,
        "max_pairs_per_parent": args.max_pairs_per_parent,
        "shuffle_seed": args.shuffle_seed,
        "high_conf_q": HIGH_CONF_Q,
        "min_frag_atoms_evidence": min_frag_evidence,
        "cmf_contract": {
            "excluded_prec": list(CMF_EXCLUDED_PREC),
            "cmf_bearing_prec": sorted(PREC_TYPE_TO_CMF_MASS_DIFF),
            "duplicate_of_crf_shift_prec": _duplicate_cmf_precursors(args.max_h_transfer),
        },
        "lod_contract": {
            "source": "estimated_train_only",
            "quantile": args.lod_quantile,
            "mz_bin_width": MZ_BIN_WIDTH,
            "global_floor": lod_tables[2],
            "by_inst": lod_tables[1],
            # Full selected table, so a rebuild can reproduce the exact filter that was applied.
            "by_inst_bin": {f"{inst}|{b}": v for (inst, b), v in sorted(lod_tables[0].items())},
        },
        "coverage": coverage,
    }
    # Fingerprint the content before stamping it in; the hash excludes the stamp itself.
    cache["metadata"]["cache_fingerprint"] = cache_fingerprint(cache)
    verify_cache_fingerprint(cache)
    out_dp = os.path.dirname(os.path.abspath(args.out_fp))
    os.makedirs(out_dp, exist_ok=True)
    th.save(cache, args.out_fp)
    verify_cache_fingerprint(th.load(args.out_fp, weights_only=False))

    report = _build_audit(per_mol, cache, split_of)
    report["cache_fingerprint"] = cache["metadata"]["cache_fingerprint"]
    report["cache_fp"] = args.out_fp
    report["coverage"] = coverage
    report["lod_quantile_audit"] = lod_audit
    report["min_frag_atoms_evidence"] = min_frag_evidence
    report_fp = args.report_fp or os.path.splitext(args.out_fp)[0] + "_audit.json"
    with open(report_fp, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    log.info(
        "Saved %s (%.1f MiB): %d spectra, %d pairs, Kish ESS %.1f.",
        args.out_fp,
        os.path.getsize(args.out_fp) / 2**20,
        cache["spec_ids"].numel(),
        cache["group_weights"].numel(),
        cache["stats"]["kish_ess"],
    )
    log.info("Audit written to %s", report_fp)
    log.info("Structural: %s", report["structural"])
    log.info("Pair-spectra: %s", report["pair_spectra"])


if __name__ == "__main__":
    main()
