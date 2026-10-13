import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rdkit")

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "eval" / "bond_type_ce_allocation.py"
SPEC = importlib.util.spec_from_file_location("bond_type_ce_allocation", SCRIPT_PATH)
BT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BT)


def test_tv_decomposition_splits_between_and_within_type():
    types = {0: "C-O", 1: "C-O", 2: "C-C", BT.PRECURSOR: BT.PRECURSOR}
    # Right type mass, wrong C-O bond: all error is within type.
    within = BT.tv_decomposition({0: 1.0}, {1: 1.0}, types)
    assert within["tv_bond"] == pytest.approx(1.0)
    assert within["tv_type"] == pytest.approx(0.0)
    assert within["tv_within"] == pytest.approx(1.0)
    # Precursor predicted where a C-C bond fires: all error is between types.
    between = BT.tv_decomposition({BT.PRECURSOR: 1.0}, {2: 1.0}, types)
    assert between["tv_type"] == pytest.approx(1.0)
    assert between["tv_within"] == pytest.approx(0.0)
    # Identical distributions have no error.
    same = BT.tv_decomposition({0: 0.3, 2: 0.7}, {0: 0.3, 2: 0.7}, types)
    assert same["tv_bond"] == pytest.approx(0.0)


def test_bond_labels_types_and_rejects_bad_bond():
    labels = BT.bond_labels("CC(=O)Oc1ccccc1", 10, np.array([[0, 1], [1, 2], [1, 3], [3, 4], [4, 5]]))
    assert labels == ["C-C", "C=O", "C-O", "C-O", "C:C"]
    with pytest.raises(ValueError):
        BT.bond_labels("CCO", 3, np.array([[0, 2]]))
    with pytest.raises(ValueError):
        BT.bond_labels("CCO", 4, np.array([[0, 1]]))


def test_spectrum_records_splits_mass_over_cut_bonds_and_precursor():
    # Fragment 0 is the precursor (no cut bonds), fragment 1 is cut at bonds 7 and 9.
    chan_mz = [np.array([300.0]), np.array([150.0, 151.00783])]
    chan_w = [np.array([0.6]), np.array([0.3, 0.1])]
    frag_mass = np.array([0.6, 0.4])
    tm = np.array([300.0, 150.0])
    ti = np.array([0.2, 0.8])
    p, t, covered = BT.spectrum_records(
        frag_mass, chan_mz, chan_w, [np.array([], dtype=np.int64), np.array([7, 9])], tm, ti, False
    )
    assert covered == pytest.approx(1.0)
    assert p[BT.PRECURSOR] == pytest.approx(0.6)
    assert p[7] == pytest.approx(0.2) and p[9] == pytest.approx(0.2)
    assert t[BT.PRECURSOR] == pytest.approx(0.2)
    assert t[7] == pytest.approx(0.4) and t[9] == pytest.approx(0.4)
