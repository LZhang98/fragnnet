import pytest
import torch as th

from fragnnet.dataset.spec_mol_frag_dataset import SpecMolFragDataset
from fragnnet.utils.formula_utils import (
    ELECTRON_MASS,
    PREC_TYPE_TO_CMF_MASS_DIFF,
    PREC_TYPE_TO_MASS_DIFF,
)


class _FakePyG:
    def __init__(self, x: th.Tensor, node_feat_idxs: th.Tensor):
        # minimal attributes used by _apply_fragment_mz_change
        self.x = x
        # code expects node_feat_idxs[0] to be indexable
        self.node_feat_idxs = [node_feat_idxs]


def _make_obj(include_cmf: bool = True, with_cmf_feat: bool = True):
    obj = object.__new__(SpecMolFragDataset)
    feats = ["cmf_h_formulae_idx"] if with_cmf_feat else []
    obj.frag_params = {
        "include_cmf": include_cmf,
        "pyg_node_feats": feats,
        "formula_peak_mzs": True,
        "formula_peak_probs": False,
    }
    return obj


def _make_pyg(num_nodes: int, crf_values: th.Tensor):
    """Build a _FakePyG with crf_values in the h_formulae_idx slice (NODE_FEAT_TO_IDX=3)
    and zeros in cmf_h_formulae_idx (NODE_FEAT_TO_IDX=6).
    In production both CRF and CMF features have the same width (formula_node_feat_size).
    """
    crf_width = crf_values.shape[1]
    cmf_width = crf_width  # always equal to crf_width in production DAGs
    # idxs: depth=1, cc=3, base=3, crf=crf_width, h_counts=crf_width, nb=1, cmf=cmf_width
    node_feat_idxs = th.tensor(
        [
            0,
            1,
            4,
            7,
            7 + crf_width,
            7 + 2 * crf_width,
            7 + 2 * crf_width + 1,
            7 + 2 * crf_width + 1 + cmf_width,
        ]
    )
    total_width = int(node_feat_idxs[-1].item())
    x = th.zeros((num_nodes, total_width), dtype=th.int64)
    crf_start = int(node_feat_idxs[3].item())
    x[:, crf_start : crf_start + crf_width] = crf_values
    return _FakePyG(x, node_feat_idxs)


def test_cmf_excluded_precursor_crf_only():
    """[M]+ is still excluded; result should be CRF peaks only."""
    obj = _make_obj(include_cmf=True, with_cmf_feat=False)
    neutral = th.tensor([[10.0, 11.0], [20.0, 21.0]])
    prec = "[M]+"
    fake_pyg = _make_pyg(1, th.zeros((1, 2), dtype=th.int64))
    frag_data = {"frag_formula_peak_mzs": neutral.clone(), "frag_pyg": fake_pyg}

    res = SpecMolFragDataset._apply_fragment_mz_change(
        obj, frag_data, prec, neutral, clone_pyg=False
    )
    expected = neutral + PREC_TYPE_TO_MASS_DIFF[prec]

    assert th.allclose(res["frag_formula_peak_mzs"], expected)
    assert res["frag_formula_peak_mzs"].shape == expected.shape


def test_cmf_applied_updates_peaks_and_pyg():
    """[M+Na]+ (non-H adduct): CMF appends peaks with H mass_diff and fills all CRF cols."""
    obj = _make_obj()
    neutral = th.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    num_formulae = neutral.shape[0]
    probs = th.tensor([[0.5, 0.5], [0.6, 0.4], [0.4, 0.6]], dtype=th.float32)

    crf_values = th.tensor([[0, 1], [2, 0]], dtype=th.int64)
    fake_pyg = _make_pyg(2, crf_values)
    node_feat_idxs = fake_pyg.node_feat_idxs[0]

    frag_data = {
        "frag_formula_peak_mzs": neutral.clone(),
        "frag_formula_peak_probs": probs.clone(),
        "frag_pyg": fake_pyg,
    }
    prec = "[M+Na]+"

    res = SpecMolFragDataset._apply_fragment_mz_change(
        obj, frag_data, prec, neutral, clone_pyg=False
    )

    crf_expected = neutral + PREC_TYPE_TO_MASS_DIFF[prec]
    assert th.allclose(res["frag_formula_peak_mzs"][:num_formulae], crf_expected)

    cmf_expected = neutral[1:] + PREC_TYPE_TO_CMF_MASS_DIFF[prec]
    assert th.allclose(res["frag_formula_peak_mzs"][num_formulae:], cmf_expected)

    # For [M+Na]+ all CMF columns should be filled (full H-shift range)
    cmf_start = int(node_feat_idxs[6].item())
    cmf_end = int(node_feat_idxs[7].item())
    cmf_matrix = res["frag_pyg"].x[:, cmf_start:cmf_end]
    expected_cmf = th.where(crf_values != 0, crf_values + num_formulae - 1, crf_values)
    assert th.equal(cmf_matrix[:, : crf_values.shape[1]], expected_cmf)


class _FakePyGClone(_FakePyG):
    def clone(self):
        return _FakePyGClone(self.x.clone(), self.node_feat_idxs[0])


def test_cmf_mplus_h_direct_cation_peaks():
    """[M+H]+: CMF applies with mass_diff = -ELECTRON_MASS (direct cation mode)."""
    obj = _make_obj()
    neutral = th.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    num_formulae = neutral.shape[0]

    crf_values = th.tensor([[1, 0], [2, 1]], dtype=th.int64)
    fake_pyg = _make_pyg(2, crf_values)

    frag_data = {"frag_formula_peak_mzs": neutral.clone(), "frag_pyg": fake_pyg}
    prec = "[M+H]+"

    res = SpecMolFragDataset._apply_fragment_mz_change(
        obj, frag_data, prec, neutral, clone_pyg=False
    )

    crf_expected = neutral + PREC_TYPE_TO_MASS_DIFF[prec]
    assert th.allclose(res["frag_formula_peak_mzs"][:num_formulae], crf_expected)

    # CMF peaks: neutral[1:] + (-ELECTRON_MASS)  ≈ neutral mass directly
    cmf_expected = neutral[1:] + PREC_TYPE_TO_CMF_MASS_DIFF[prec]
    assert th.allclose(res["frag_formula_peak_mzs"][num_formulae:], cmf_expected)
    # CRF peak at delta_h=0: neutral[i] + H_MASS; CMF peak: neutral[i] - ELECTRON_MASS
    # They differ by ~1.008 Da, confirming no duplication.
    assert not th.allclose(
        res["frag_formula_peak_mzs"][:num_formulae],
        res["frag_formula_peak_mzs"][:num_formulae] * 0
        + res["frag_formula_peak_mzs"][num_formulae : num_formulae + 1].expand(num_formulae, -1),
    )


def test_cmf_mplus_h_col0_only():
    """[M+H]+: only column 0 of cmf_h_formulae_idx is filled; columns 1+ stay 0."""
    obj = _make_obj()
    neutral = th.tensor([[10.0], [20.0], [30.0]])
    num_formulae = neutral.shape[0]

    # crf col 0 = 2 (non-zero), col 1 = 1 (non-zero)
    crf_values = th.tensor([[2, 1]], dtype=th.int64)
    fake_pyg = _make_pyg(1, crf_values)
    node_feat_idxs = fake_pyg.node_feat_idxs[0]

    frag_data = {"frag_formula_peak_mzs": neutral.clone(), "frag_pyg": fake_pyg}
    res = SpecMolFragDataset._apply_fragment_mz_change(
        obj, frag_data, "[M+H]+", neutral, clone_pyg=False
    )

    cmf_start = int(node_feat_idxs[6].item())
    cmf_end = int(node_feat_idxs[7].item())
    cmf_matrix = res["frag_pyg"].x[:, cmf_start:cmf_end]

    # col 0: crf_col0=2 → cmf_col0 = 2 + num_formulae - 1 = 4
    assert cmf_matrix[0, 0].item() == 2 + num_formulae - 1
    # cols 1+: must be 0 regardless of crf col 1
    assert th.equal(cmf_matrix[:, 1:], th.zeros_like(cmf_matrix[:, 1:]))


def test_cmf_mminus_h_direct_anion_peaks():
    """[M-H]-: CMF applies with mass_diff = +ELECTRON_MASS (direct anion mode)."""
    obj = _make_obj()
    neutral = th.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    num_formulae = neutral.shape[0]

    crf_values = th.tensor([[1, 0], [2, 1]], dtype=th.int64)
    fake_pyg = _make_pyg(2, crf_values)

    frag_data = {"frag_formula_peak_mzs": neutral.clone(), "frag_pyg": fake_pyg}
    prec = "[M-H]-"

    res = SpecMolFragDataset._apply_fragment_mz_change(
        obj, frag_data, prec, neutral, clone_pyg=False
    )

    crf_expected = neutral + PREC_TYPE_TO_MASS_DIFF[prec]
    assert th.allclose(res["frag_formula_peak_mzs"][:num_formulae], crf_expected)

    cmf_expected = neutral[1:] + PREC_TYPE_TO_CMF_MASS_DIFF[prec]
    assert th.allclose(res["frag_formula_peak_mzs"][num_formulae:], cmf_expected)
    # CMF mass diff for [M-H]- is +ELECTRON_MASS ≈ +0.00055
    assert abs(PREC_TYPE_TO_CMF_MASS_DIFF[prec] - ELECTRON_MASS) < 1e-6


def test_cmf_mminus_h_col0_only():
    """[M-H]-: only column 0 of cmf_h_formulae_idx is filled; columns 1+ stay 0."""
    obj = _make_obj()
    neutral = th.tensor([[10.0], [20.0], [30.0]])
    num_formulae = neutral.shape[0]

    crf_values = th.tensor([[2, 1]], dtype=th.int64)
    fake_pyg = _make_pyg(1, crf_values)
    node_feat_idxs = fake_pyg.node_feat_idxs[0]

    frag_data = {"frag_formula_peak_mzs": neutral.clone(), "frag_pyg": fake_pyg}
    res = SpecMolFragDataset._apply_fragment_mz_change(
        obj, frag_data, "[M-H]-", neutral, clone_pyg=False
    )

    cmf_start = int(node_feat_idxs[6].item())
    cmf_end = int(node_feat_idxs[7].item())
    cmf_matrix = res["frag_pyg"].x[:, cmf_start:cmf_end]

    assert cmf_matrix[0, 0].item() == 2 + num_formulae - 1
    assert th.equal(cmf_matrix[:, 1:], th.zeros_like(cmf_matrix[:, 1:]))


@pytest.mark.parametrize(
    "prec, expect_col0_only",
    [
        ("[M+H]+", True),
        ("[M-H]-", True),
        ("[M-H-CO2]-", True),
        ("[M+Na]+", False),
        ("[M+NH4]+", False),
        ("[M+K]+", False),
        ("[M+Cl]-", False),
    ],
)
def test_cmf_fill_strategy_by_adduct(prec, expect_col0_only):
    """[M+H]+ and [M-H]- fill only column 0 (direct-charge); all other CMF adducts fill all columns."""
    obj = _make_obj()
    neutral = th.tensor([[10.0], [20.0], [30.0], [40.0]])
    num_formulae = neutral.shape[0]

    # set every CRF column to a non-zero index so we can detect which columns get written
    crf_width = 3
    crf_values = th.tensor([[1, 2, 3]], dtype=th.int64)  # all non-zero
    fake_pyg = _make_pyg(1, crf_values)
    node_feat_idxs = fake_pyg.node_feat_idxs[0]

    frag_data = {"frag_formula_peak_mzs": neutral.clone(), "frag_pyg": fake_pyg}
    SpecMolFragDataset._apply_fragment_mz_change(obj, frag_data, prec, neutral, clone_pyg=False)

    cmf_start = int(node_feat_idxs[6].item())
    cmf_end = int(node_feat_idxs[7].item())
    cmf_matrix = frag_data["frag_pyg"].x[:, cmf_start:cmf_end]

    if expect_col0_only:
        # column 0 written, columns 1+ must be 0
        assert cmf_matrix[0, 0].item() == 1 + num_formulae - 1, prec
        assert th.equal(cmf_matrix[:, 1:], th.zeros_like(cmf_matrix[:, 1:])), prec
    else:
        # all columns written
        expected = crf_values + num_formulae - 1  # all non-zero → all shifted
        assert th.equal(cmf_matrix, expected), prec


def test_cmf_mass_diffs_direct_charge():
    """[M+H]+ and [M-H]- CMF mass diffs are ±ELECTRON_MASS (direct charge, no H transfer)."""
    assert abs(PREC_TYPE_TO_CMF_MASS_DIFF["[M+H]+"] - (-ELECTRON_MASS)) < 1e-9
    assert abs(PREC_TYPE_TO_CMF_MASS_DIFF["[M-H]-"] - ELECTRON_MASS) < 1e-9


def test_cmf_mass_diff_m_h_co2_is_direct_anion():
    """[M-H-CO2]- shares the [frag-H]- carrier of [M-H]-, so its CMF diff is +ELECTRON_MASS."""
    assert abs(PREC_TYPE_TO_CMF_MASS_DIFF["[M-H-CO2]-"] - ELECTRON_MASS) < 1e-9


def test_no_adduct_has_degenerate_cmf_mass_diff():
    """No adduct may give CMF the same mass diff as CRF.

    CMF(F) == CRF(F) makes every CMF row an exact m/z duplicate of its own CRF row, so the
    CMF doubling adds no candidate m/z while still enlarging the formula softmax. This is a
    regression guard: [M-H-CO2]- previously reused the CRF value.
    """
    degenerate = [
        pt
        for pt, cmf in PREC_TYPE_TO_CMF_MASS_DIFF.items()
        if abs(cmf - PREC_TYPE_TO_MASS_DIFF[pt]) < 1e-9
    ]
    assert degenerate == [], f"CMF mass diff equals CRF mass diff for: {degenerate}"


def test_unknown_precursor_raises_keyerror():
    obj = _make_obj()
    neutral = th.tensor([[1.0, 2.0]])
    frag_data = {"frag_formula_peak_mzs": neutral.clone()}
    with pytest.raises(KeyError):
        SpecMolFragDataset._apply_fragment_mz_change(
            obj, frag_data, "UNKNOWN_PREC", neutral, clone_pyg=False
        )


def test_crf_matrix_zero_results_in_zero_cmf_slice():
    obj = _make_obj()
    neutral = th.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])

    crf_values = th.zeros((1, 2), dtype=th.int64)
    fake_pyg = _make_pyg(1, crf_values)
    node_feat_idxs = fake_pyg.node_feat_idxs[0]

    frag_data = {"frag_formula_peak_mzs": neutral.clone(), "frag_pyg": fake_pyg}
    res = SpecMolFragDataset._apply_fragment_mz_change(
        obj, frag_data, "[M+Na]+", neutral, clone_pyg=False
    )

    cmf_start = int(node_feat_idxs[6].item())
    cmf_end = int(node_feat_idxs[7].item())
    cmf_matrix = res["frag_pyg"].x[:, cmf_start:cmf_end]
    assert th.equal(cmf_matrix, th.zeros_like(cmf_matrix))


def test_clone_pyg_returns_new_object_and_original_unchanged():
    obj = _make_obj()
    neutral = th.tensor([[10.0, 11.0], [20.0, 21.0]])

    crf_values = th.tensor([[1, 0]], dtype=th.int64)
    fake_pyg = _FakePyGClone(
        _make_pyg(1, crf_values).x,
        _make_pyg(1, crf_values).node_feat_idxs[0],
    )
    original_x_before = fake_pyg.x.clone()
    frag_data = {"frag_formula_peak_mzs": neutral.clone(), "frag_pyg": fake_pyg}

    res = SpecMolFragDataset._apply_fragment_mz_change(
        obj, frag_data, "[M+Na]+", neutral, clone_pyg=True
    )

    assert res["frag_pyg"].x is not fake_pyg.x
    assert th.equal(fake_pyg.x, original_x_before)


def test_single_formula_no_cmf_appended():
    obj = _make_obj()
    neutral = th.tensor([[42.0, 43.0]])
    fake_pyg = _make_pyg(1, th.zeros((1, 2), dtype=th.int64))
    frag_data = {"frag_formula_peak_mzs": neutral.clone(), "frag_pyg": fake_pyg}
    prec = "[M+Na]+"

    res = SpecMolFragDataset._apply_fragment_mz_change(
        obj, frag_data, prec, neutral, clone_pyg=False
    )
    # neutral[1:] is empty, so CMF appends nothing
    expected = neutral + PREC_TYPE_TO_MASS_DIFF[prec]
    assert th.allclose(res["frag_formula_peak_mzs"], expected)


def test_multiple_precursors_shapes():
    precs = ["[M+H]+", "[M+Na]+", "[M+K]+", "[M-H]-", "[M+Cl]-", "[M]+"]
    neutral = th.tensor(
        [[10.0, 10.1, 10.2], [20.0, 20.1, 20.2], [30.0, 30.1, 30.2], [40.0, 40.1, 40.2]]
    )
    num_formulae = neutral.shape[0]

    for prec in precs:
        obj = _make_obj()
        crf_values = th.tensor([[0, 1], [1, 0], [2, 2]], dtype=th.int64)
        fake_pyg = _make_pyg(3, crf_values)
        frag_data = {"frag_formula_peak_mzs": neutral.clone(), "frag_pyg": fake_pyg}

        res = SpecMolFragDataset._apply_fragment_mz_change(
            obj, frag_data, prec, neutral, clone_pyg=False
        )

        excluded = prec in ["[M]+", "[M-]"]

        if excluded:
            expected = neutral + PREC_TYPE_TO_MASS_DIFF[prec]
            assert th.allclose(res["frag_formula_peak_mzs"], expected), prec
            assert res["frag_formula_peak_mzs"].shape == neutral.shape, prec
        else:
            crf_expected = neutral + PREC_TYPE_TO_MASS_DIFF[prec]
            cmf_expected = neutral[1:] + PREC_TYPE_TO_CMF_MASS_DIFF[prec]
            out = res["frag_formula_peak_mzs"]
            assert out.shape[0] == num_formulae + (num_formulae - 1), prec
            assert th.allclose(out[:num_formulae], crf_expected), prec
            assert th.allclose(out[num_formulae:], cmf_expected), prec
