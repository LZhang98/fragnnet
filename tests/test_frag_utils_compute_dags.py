"""Tests for compute_dags in frag_utils.py.

Tests cover:
1. Brominated amine (BrCCNc1ccccc1): previously caused KeyError: 255 due to
   uint8 underflow in diff_cc_mask and bitmask passed directly as atom indices.
2. Brominated piperidine (BrCCC1CCCCN1): same bug class.
3. DAG result structure: expected keys and tensor shapes.
4. cc_bit_mask_to_atom_idx correctly excludes uint8 underflow values (255).
5. Signed subtraction + clip yields correct diff atom ids.
"""

from __future__ import annotations

import numpy as np
import pytest

from fragnnet.frag.compute_frags import MASK_DTYPE
from fragnnet.utils.frag_utils import (
    cc_bit_mask_to_atom_idx,
    compute_dags,
    extract_mol_info,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_compute_dags(smiles: str, max_depth: int = 3) -> dict:
    """Run compute_dags with default parameters on a SMILES string."""
    mol_d = extract_mol_info(smiles)
    return compute_dags(
        mol_d,
        max_depth=max_depth,
        h_prior=True,
        max_h_transfer=2,
        frag_max_time=int(1e6),
        isotopes=False,
        nb_isomorphic=False,
        wl_max_iterations=-1,
        multi_cut_bfs=False,
    )


# ---------------------------------------------------------------------------
# 1 & 2. Regression: KeyError: 255 on brominated molecules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "smiles",
    [
        "BrCCNc1ccccc1",  # N-phenyl-2-bromoethylamine
        "BrCCC1CCCCN1",   # 1-(3-bromopropyl)piperidine
    ],
)
def test_compute_dags_brominated_molecule_no_keyerror(smiles: str) -> None:
    """Brominated molecules previously raised KeyError: 255 due to uint8
    underflow in diff_cc_mask (0 - 1 = 255 in uint8) and the bitmask being
    passed directly to compute_cc_h_floor as atom indices instead of being
    converted via cc_bit_mask_to_atom_idx first."""
    dag_d = _run_compute_dags(smiles)
    assert dag_d is not None


@pytest.mark.parametrize(
    "smiles",
    [
        "BrCCNc1ccccc1",
        "BrCCC1CCCCN1",
    ],
)
def test_compute_dags_brominated_molecule_has_edges(smiles: str) -> None:
    """Brominated molecules with multiple bonds must produce DAG edges."""
    dag_d = _run_compute_dags(smiles)
    assert dag_d["dag_num_edges"] > 0


# ---------------------------------------------------------------------------
# 3. DAG result structure
# ---------------------------------------------------------------------------


def test_compute_dags_result_keys() -> None:
    """Result dict must contain the core expected keys."""
    dag_d = _run_compute_dags("CCO")
    for key in ("dag", "dag_num_edges", "dag_num_nodes", "max_depth", "reached_depth"):
        assert key in dag_d, f"Missing key: {key}"


def test_compute_dags_edge_index_shape() -> None:
    """dag.edge_index has shape (2, num_edges)."""
    dag_d = _run_compute_dags("CCO")
    assert dag_d["dag"].edge_index.shape[0] == 2
    assert dag_d["dag"].edge_index.shape[1] == dag_d["dag_num_edges"]


def test_compute_dags_node_feat_shape() -> None:
    """dag.x has num_nodes rows."""
    dag_d = _run_compute_dags("CCO")
    assert dag_d["dag"].x.shape[0] == dag_d["dag_num_nodes"]


# ---------------------------------------------------------------------------
# 4. cc_bit_mask_to_atom_idx excludes uint8 underflow value 255
# ---------------------------------------------------------------------------


def test_cc_bit_mask_to_atom_idx_excludes_uint8_underflow() -> None:
    """uint8 subtraction 0 - 1 = 255; cc_bit_mask_to_atom_idx must not
    include those positions since it only selects positions where value == 1."""
    from_mask = np.array([1, 0, 1], dtype=MASK_DTYPE)
    to_mask = np.array([1, 1, 1], dtype=MASK_DTYPE)

    # Demonstrate the underflow: 0 - 1 = 255 in uint8
    raw_diff = from_mask - to_mask
    assert raw_diff[1] == 255, "Expected uint8 underflow to produce 255"

    # cc_bit_mask_to_atom_idx only picks positions with value exactly 1
    atom_ids = cc_bit_mask_to_atom_idx(raw_diff)
    assert 255 not in atom_ids
    # raw_diff is [0, 255, 0] — no position has value 1, so result is empty
    assert len(atom_ids) == 0


# ---------------------------------------------------------------------------
# 5. Signed subtraction + clip gives correct diff atom ids
# ---------------------------------------------------------------------------


def test_diff_cc_mask_signed_subtraction_correct() -> None:
    """With signed subtraction and clip, diff correctly identifies atoms in
    parent but not child, ignoring underflow."""
    from_mask = np.array([1, 1, 1, 0], dtype=MASK_DTYPE)
    to_mask = np.array([0, 1, 0, 0], dtype=MASK_DTYPE)

    diff = np.clip(
        from_mask.astype(np.int16) - to_mask.astype(np.int16), 0, 1
    ).astype(MASK_DTYPE)
    atom_ids = cc_bit_mask_to_atom_idx(diff)

    # Atoms 0 and 2 are in from but not to; atom 1 is in both; atom 3 in neither
    np.testing.assert_array_equal(np.sort(atom_ids), [0, 2])
