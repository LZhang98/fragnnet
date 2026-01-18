import torch as th

from fragnnet.dataset.spec_mol_frag_dataset import SpecMolFragDataset
from fragnnet.utils.formula_utils import PREC_TYPE_TO_CMF_MASS_DIFF, PREC_TYPE_TO_MASS_DIFF


class _FakePyG:
    def __init__(self, x: th.Tensor, node_feat_idxs: th.Tensor):
        # minimal attributes used by _apply_fragment_mz_change
        self.x = x
        # code expects node_feat_idxs[0] to be indexable
        self.node_feat_idxs = [node_feat_idxs]


def test_cmf_excluded_precursor_crf_only():
    obj = object.__new__(SpecMolFragDataset)
    obj.frag_params = {"include_cmf": True, "pyg_node_feats": []}

    neutral = th.tensor([[10.0, 11.0], [20.0, 21.0]])
    frag_data = {"frag_formula_peak_mzs": neutral.clone()}
    prec = "[M-H]-"

    res = SpecMolFragDataset._apply_fragment_mz_change(
        obj, frag_data, prec, neutral, clone_pyg=False
    )
    expected = neutral + PREC_TYPE_TO_MASS_DIFF[prec]

    assert th.allclose(res["frag_formula_peak_mzs"], expected)
    assert res["frag_formula_peak_mzs"].shape == expected.shape


def test_cmf_applied_updates_peaks_and_pyg():
    obj = object.__new__(SpecMolFragDataset)
    # enable CMF and include cmf node feat name
    obj.frag_params = {
        "include_cmf": True,
        "pyg_node_feats": ["cmf_h_formulae_idx"],
        "formula_peak_mzs": True,
        "formula_peak_probs": True,
    }

    # neutral peaks: 3 formulas x 2 isotopes
    neutral = th.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    num_formulae = neutral.shape[0]

    # simple probs (normalized rows)
    probs = th.tensor([[0.5, 0.5], [0.6, 0.4], [0.4, 0.6]], dtype=th.float32)

    # build fake frag_pyg
    # construct node_feat_idxs such that h_formulae_idx at idx=3 has width 2 and cmf at idx=6 has width 2
    node_feat_idxs = th.tensor([0, 1, 2, 5, 7, 8, 11, 13])
    total_width = int(node_feat_idxs[-1].item())
    num_nodes = 2
    x = th.zeros((num_nodes, total_width), dtype=th.int64)

    # place crf formula matrix in the crf slice (node_feat_idxs[3]:node_feat_idxs[4]) width=2
    crf_start = int(node_feat_idxs[3].item())
    crf_end = int(node_feat_idxs[4].item())
    crf_width = crf_end - crf_start
    # set some non-zero indices to indicate possible formula matches
    x[:, crf_start:crf_end] = th.tensor([[0, 1], [2, 0]], dtype=th.int64)

    fake_pyg = _FakePyG(x, node_feat_idxs)

    frag_data = {
        "frag_formula_peak_mzs": neutral.clone(),
        "frag_formula_peak_probs": probs.clone(),
        "frag_pyg": fake_pyg,
    }

    prec = "[M+Na]+"  # this precursor should allow CMF (not in exclusion list)

    res = SpecMolFragDataset._apply_fragment_mz_change(
        obj, frag_data, prec, neutral, clone_pyg=False
    )

    # Check peaks: first block == neutral + CRF mass diff
    crf_expected = neutral + PREC_TYPE_TO_MASS_DIFF[prec]
    assert th.allclose(res["frag_formula_peak_mzs"][:num_formulae], crf_expected)

    # Check CMF peaks appended: neutral[1:] + CMF mass diff
    cmf_expected = neutral[1:] + PREC_TYPE_TO_CMF_MASS_DIFF[prec]
    assert th.allclose(res["frag_formula_peak_mzs"][num_formulae:], cmf_expected)

    # Check frag_pyg.x cmf slice updated as per logic
    cmf_start = int(node_feat_idxs[6].item())
    cmf_end = int(node_feat_idxs[7].item())
    crf_matrix = fake_pyg.x[:, crf_start:crf_end]
    cmf_matrix = res["frag_pyg"].x[:, cmf_start:cmf_end]

    # expected: where crf != 0 -> crf + num_formulae - 1, else crf
    expected_cmf = th.where(crf_matrix != 0, crf_matrix + num_formulae - 1, crf_matrix)
    assert th.equal(cmf_matrix, expected_cmf)
