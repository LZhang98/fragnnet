import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "debug" / "masster_memorization" / "intensity_displacement.py"
SPEC = importlib.util.spec_from_file_location("intensity_displacement", SCRIPT_PATH)
ID = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ID)


@pytest.fixture
def glucose_ion() -> np.ndarray:
    """[M+H]+ of C6H12O6."""
    return ID.formula_vec("C6H13O6")


def ion_mz(formula: str) -> float:
    return float(ID.formula_vec(formula) @ ID.MV - ID.M_E)


def test_assign_formulas_exact_and_unassignable(glucose_ion):
    mzs = np.array([ion_mz("C6H11O5"), ion_mz("C6H13O6"), ion_mz("C3H5O2"), 150.0])
    v, flag = ID.assign_formulas(glucose_ion, mzs)
    assert (v[0] == ID.formula_vec("C6H11O5")).all()
    assert (v[1] == glucose_ion).all()
    assert (v[2] == ID.formula_vec("C3H5O2")).all()
    assert flag.tolist() == [False, False, False, True]


def test_assign_formulas_flags_second_formula_within_tolerance(glucose_ion):
    # With a 0.1 Da tolerance, CO and C2H4 (0.036 Da apart) both match near 28.
    mz = np.array([ion_mz("C2H5O") - ID.MASS["O"] + 2 * ID.MASS["H"] + ID.MASS["C"] - 0.018])
    _, flag = ID.assign_formulas(glucose_ion, mz, tol_da=0.1)
    assert flag.tolist() == [True]


def test_pair_classes_cover_every_class(glucose_ion):
    formulas = ["C6H13O6", "C6H11O5", "C6H10O5", "C4H7O3", "C3H5O3", "C4H7O2"]
    v = np.stack([ID.formula_vec(f) for f in formulas])
    prec = np.array([True, False, False, False, False, False])
    amb = np.zeros(len(v), bool)
    cls = ID.pair_classes(v, v, prec, prec, amb, amb)
    names = {(a, b): ID.CLASSES[cls[i, j]] for i, a in enumerate(formulas) for j, b in enumerate(formulas)}
    assert names[("C6H13O6", "C6H11O5")] == "prec"
    assert names[("C6H11O5", "C6H10O5")] == "hshift"
    # C6H11O5 -> C4H7O3 is C2H4O2 (acetic-acid-like) lost: a vocabulary loss.
    assert names[("C6H11O5", "C4H7O3")] == "nl1"
    assert names[("C4H7O3", "C6H11O5")] == "nl1"
    # C6H11O5 -> C3H5O3: heavy difference C3O2, not in the vocabulary.
    assert names[("C6H11O5", "C3H5O3")] == "nested"
    # C3H5O3 and C4H7O2 (C3O3 vs C4O2): neither heavy-atom set contains the other.
    assert names[("C3H5O3", "C4H7O2")] == "sibling"
    amb[1] = True
    cls = ID.pair_classes(v, v, prec, prec, amb, amb)
    assert ID.CLASSES[cls[1, 3]] == "ambig"
    assert ID.CLASSES[cls[0, 1]] == "prec"


def test_transport_prefers_cheap_class_and_conserves_mass():
    x = np.array([0.3, 0.1])
    y = np.array([0.1, 0.3])
    cost = np.array([[5.0, 1.0], [1.0, 5.0]])
    flow = ID.transport(x, y, cost, units=400)
    assert flow.sum() == pytest.approx(0.4)
    assert flow[0, 1] == pytest.approx(0.3, abs=1e-3)
    assert flow[1, 0] == pytest.approx(0.1, abs=1e-3)
    assert flow[0, 0] == pytest.approx(0.0, abs=1e-3)


def test_quantize_sums_exactly():
    v = np.array([0.5, 0.3333, 0.1667, 1e-5])
    q = ID._quantize(v, 500)
    assert q.sum() == 500
    assert (np.abs(q - v / v.sum() * 500) < 1.0).all()


def test_bias_record_identities():
    rng = np.random.default_rng(0)
    tm = np.array([100.0, 150.0, 200.0, 300.0])
    ti = np.array([1.0, 0.5, 0.2, 0.1])
    # Run A covers every true peak but the last; run B adds a false candidate.
    preds = [(np.array([100.0, 150.0, 200.0]), rng.random(3)),
             (np.array([100.0, 150.0, 200.0, 250.0]), rng.random(4))]
    sp = ID.candidate_space(tm, ti, preds)
    r = ID.bias_record(sp)
    # Cosine gap = displacement + coverage, for each run.
    assert 1.0 - r["cos_a"] == pytest.approx(r["disp_a"] + r["coverage"])
    assert 1.0 - r["cos_b"] == pytest.approx(r["disp_b"] + r["coverage"])
    # Mean single-run displacement = |e_bar|^2 + |d|^2 (parallelogram law).
    assert (r["disp_a"] + r["disp_b"]) / 2.0 == pytest.approx(r["e_bar2"] + r["d2"])
    # Identical runs have no run-specific part.
    same = ID.bias_record(ID.candidate_space(tm, ti, [preds[0], preds[0]]))
    assert same["d2"] == pytest.approx(0.0)


def test_displacement_classes_single_h2o_loss(glucose_ion):
    cm = np.array([ion_mz("C6H11O5"), ion_mz("C6H9O4")])
    info = ID._info(cm, glucose_ion, prec_mz=ion_mz("C6H13O6"))
    q = np.array([0.8, 0.2])
    y = np.array([0.2, 0.8])
    out = ID.displacement_classes(q, y, cm, info)
    assert out["D"] == pytest.approx(0.6)
    assert out["nl1"] == pytest.approx(0.6, abs=2e-3)
    assert out["down"] == pytest.approx(0.6, abs=2e-3)
    assert out["unres"] == pytest.approx(0.0, abs=2e-3)
