"""masster_merge_same_mz_rows: exact-mass row merge required by intensity-transformed targets."""

import copy
from pathlib import Path

import pytest
import torch as th

import fragnnet.runner as runner
from fragnnet.models.masster.masster_pl import MassterPL
from fragnnet.utils.spec_utils import get_ints_untransform_func
from tests.test_masster_model import _build_batch, _build_model

INPUTS_SMILES = ("CCOC(=O)c1ccccc1", "CC(=O)Nc1ccc(O)cc1")


def _inputs(batch):
    return {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 0]),
        "spec_nce": th.tensor([20.0, 40.0]),
    }


def test_exact_rows_merge_by_spectrum_and_4_decimal_mz() -> None:
    batch = _build_batch(INPUTS_SMILES)
    model = _build_model(batch, include_cmf=False, max_h_transfer=2, merge_same_mz_rows=True).eval()
    merged = model(**_inputs(batch))
    rows = model(**_inputs(batch), return_annotation=True)

    expected = {}
    for mz, prob, b in zip(
        rows["pred_pathway_mzs"].tolist(),
        rows["pred_pathway_scores"].tolist(),
        rows["pred_pathway_batch_idxs"].tolist(),
    ):
        key = (b, round(mz, 4))
        expected[key] = expected.get(key, 0.0) + prob
    observed = {
        (int(b), round(float(mz), 4)): float(p.exp())
        for mz, p, b in zip(merged["pred_mzs"], merged["pred_logprobs"], merged["pred_batch_idxs"])
    }
    assert observed.keys() == expected.keys()
    # several pathways share an exact m/z (H-shift channels of different fragments)
    assert merged["pred_mzs"].numel() < rows["pred_pathway_mzs"].numel()
    # the annotation run emits the same merged spectrum, plus the pathway rows above
    assert th.equal(merged["pred_mzs"], rows["pred_mzs"])
    assert th.allclose(merged["pred_logprobs"], rows["pred_logprobs"])
    for key, value in expected.items():
        assert observed[key] == pytest.approx(value, rel=1e-5)


def test_default_keeps_one_row_per_pathway() -> None:
    batch = _build_batch(INPUTS_SMILES)
    model = _build_model(batch, include_cmf=False, max_h_transfer=2).eval()
    default = model(**_inputs(batch))
    rows = model(**_inputs(batch), return_annotation=True)
    assert th.equal(default["pred_mzs"], rows["pred_mzs"])
    assert th.allclose(default["pred_logprobs"], rows["pred_logprobs"])


def test_sqrt_untransform_of_merged_sum_differs_from_per_pathway() -> None:
    # Why the flag is required: squaring each pathway (sum p^2) under-predicts a shared m/z.
    untransform = get_ints_untransform_func("sqrt")
    rows = th.tensor([0.3, 0.3, 0.4])  # two pathways at one m/z, one alone
    per_pathway = untransform(rows, th.zeros(3, dtype=th.long))
    per_pathway = th.stack([per_pathway[:2].sum(), per_pathway[2]])
    merged = untransform(th.tensor([0.6, 0.4]), th.zeros(2, dtype=th.long))
    assert (merged / merged.sum())[0] == pytest.approx(0.6**2 / (0.6**2 + 0.4**2))
    assert (per_pathway / per_pathway.sum())[0] < (merged / merged.sum())[0]


@pytest.fixture(scope="module")
def small_config() -> dict:
    config = runner.load_config(
        str(Path(__file__).parents[1] / "config"), None, None, model_type="masster"
    )
    config.update(
        model_type="masster",
        compile=False,
        sparse_cosine_similarity=True,
        oos_tolerance_multiple=1,
        gaussian_renormalize=True,
        mol_hidden_size=16,
        mol_num_layers=1,
        masster_context_dim=8,
        masster_metadata_embedding_dim=4,
        masster_head_hidden_dim=16,
        masster_intensity_transformer_dim=16,
        masster_intensity_transformer_num_layers=1,
        masster_intensity_transformer_feedforward_dim=32,
        masster_intensity_transformer_num_inducing_points=4,
        masster_unique_fragment_mode=True,
        masster_selected_mask_limit=2,
        masster_unique_fragment_target=2,
        masster_ranked_mask_budget=2,
        masster_max_h_transfer=1,
    )
    return config


def test_pl_rejects_transformed_targets_without_merge(small_config: dict) -> None:
    assert small_config["masster_merge_same_mz_rows"] is False  # template default
    with pytest.raises(ValueError, match="requires masster_merge_same_mz_rows: true"):
        MassterPL(**{**copy.deepcopy(small_config), "ints_transform": "sqrt"})
    pl = MassterPL(
        **{
            **copy.deepcopy(small_config),
            "ints_transform": "sqrt",
            "masster_merge_same_mz_rows": True,
        }
    )
    assert pl.model.merge_same_mz_rows is True
    assert MassterPL(**copy.deepcopy(small_config)).model.merge_same_mz_rows is False
