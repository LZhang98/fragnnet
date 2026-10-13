"""RT/CCS label identity, masked supervision, shared gradients, and compatibility."""

import copy

import numpy as np
import pandas as pd
import pytest
import torch as th
from rdkit import Chem
from torch_geometric.data import Batch

from fragnnet.dataset.base_dataset import BaseDataset
from fragnnet.dataset.property_labels import join_rt_ccs_labels
from fragnnet.dataset.spec_mol_dataset import SpecMolDataset
from fragnnet.inference import batch_rows, export_columns, runners
from fragnnet.models.masster.masster_pl import MassterPL
from fragnnet.models.masster.rt_ccs_heads import property_regression_terms
from fragnnet.runner import load_config
from fragnnet.utils.feat_utils import get_mol_graph
from fragnnet.utils.migrations import get_pl_hparams


def _config(enabled=True):
    config = load_config("config", None, None, model_type="masster")
    config.update(
        model_type="masster",
        compile=False,
        sparse_cosine_similarity=True,
        gaussian_renormalize=True,
        loss_vectorized=True,
        oos_tolerance_multiple=1,
        masster_predict_rt=enabled,
        masster_predict_ccs=enabled,
        mol_gnn_type="GINE",
        mol_hidden_size=16,
        mol_num_layers=2,
        mol_num_heads=4,
        mol_normalization="layer",
        mol_dropout=0.0,
        masster_context_dim=8,
        masster_metadata_embedding_dim=4,
        masster_head_hidden_dim=12,
        masster_head_dropout=0.0,
        masster_intensity_transformer_dim=16,
        masster_intensity_transformer_num_layers=1,
        masster_intensity_transformer_num_heads=4,
        masster_intensity_transformer_feedforward_dim=32,
        masster_intensity_transformer_num_inducing_points=4,
        masster_selected_mask_limit=2,
        masster_unique_fragment_target=2,
        masster_ranked_mask_budget=2,
        masster_max_h_transfer=1,
        compute_train_auxiliary_metrics=False,
        track_datapoint_metrics=False,
        train_sample_weight="none",
        eval_sample_weight="none",
    )
    config["spec_params"].update(
        merge=False,
        merge_train_only=False,
        rt=enabled,
        ccs=enabled,
        prec_types=["[M+H]+", "[M-H]-"],
    )
    return config


def _inputs(config):
    p = config["mol_params"]
    graph = get_mol_graph(
        Chem.MolFromSmiles("CCO"),
        p["pyg_node_feats"],
        p["pyg_edge_feats"],
        p["pyg_pe_embed_k"],
        True,
    )
    return {
        "mol_pyg": Batch.from_data_list([graph, graph]),
        "spec_prec_type": th.tensor([0, 0]),
        "spec_inst_type": th.tensor([0, 0]),
        "spec_nce": th.tensor([10.0, 40.0]),
        "batch_size": th.tensor(2),
    }


def test_property_loss_masks_missing_and_nonpositive_labels():
    pred = th.tensor([120.0, 200.0, 300.0, 400.0, 500.0], requires_grad=True)
    target = th.tensor([60.0, -1.0, float("nan"), float("inf"), 0.0])
    loss, error, percent, valid = property_regression_terms(pred, target, 60.0)
    th.testing.assert_close(loss, th.tensor([0.5, 0.0, 0.0, 0.0, 0.0]))
    th.testing.assert_close(error, th.tensor([60.0, 0.0, 0.0, 0.0, 0.0]))
    th.testing.assert_close(percent, th.tensor([100.0, 0.0, 0.0, 0.0, 0.0]))
    assert valid.tolist() == [True, False, False, False, False]
    loss.sum().backward()
    th.testing.assert_close(pred.grad, th.tensor([1 / 60, 0.0, 0.0, 0.0, 0.0]))
    missing = property_regression_terms(pred, th.full_like(pred, float("nan")), 60.0)[0]
    assert missing.requires_grad and missing.sum() == 0


def test_label_join_preserves_identity_and_masks_sentinels(tmp_path):
    labels = pd.DataFrame(
        {"dset_spec_id": ["b", "a"], "rtinseconds": [120.0, 60.0], "ccs": [-1.0, 150.0]}
    )
    fp = tmp_path / "labels.csv"
    labels.to_csv(fp, index=False)
    original = pd.DataFrame(
        {"dset": ["enveda", "enveda"], "dset_spec_id": ["a", "b"], "spec_id": [9, 2]}, index=[7, 3]
    )
    result = join_rt_ccs_labels(original, str(fp))
    assert result.index.tolist() == [7, 3]
    assert result.spec_id.tolist() == [9, 2]
    assert result.rtinseconds.tolist() == [60.0, 120.0]
    assert result.ccs.iloc[0] == 150.0 and np.isnan(result.ccs.iloc[1])
    assert "ccs" not in original
    with pytest.raises(ValueError, match="every processed spectrum"):
        join_rt_ccs_labels(original.assign(dset_spec_id=["a", "missing"]), str(fp))
    with pytest.raises(ValueError, match="single source"):
        join_rt_ccs_labels(original.assign(dset=["enveda", "other"]), str(fp))
    pd.concat([labels, labels]).to_csv(fp, index=False)
    with pytest.raises(ValueError, match="unique"):
        join_rt_ccs_labels(original, str(fp))


def test_dataset_property_tensors_and_collation():
    dataset = object.__new__(BaseDataset)
    dataset.spec_params = {
        "rt": True,
        "ccs": True,
        "remove_isotope_peaks": False,
        "sparse": True,
        "prec_type": False,
        "prec_type_str": False,
        "inst_type": False,
        "prec_mass_diff": False,
        "nce": False,
        "ace": False,
        "prec_mz": False,
        "unique_id": False,
        "counts": False,
    }
    samples = [
        dataset._process_spec(pd.Series({"peaks": [(40.0, 1.0)], "rtinseconds": 60.0, "ccs": ccs}))
        for ccs in (150.0, -1.0)
    ]
    batch = SpecMolDataset.collate_fn(samples)
    th.testing.assert_close(batch["spec_rt"], th.tensor([60.0, 60.0]))
    assert batch["spec_ccs"].shape == (2,)
    assert batch["spec_ccs"][0] == 150.0 and th.isnan(batch["spec_ccs"][1])


def test_heads_ce_invariance_adduct_conditioning_and_shared_gradients():
    th.manual_seed(42)
    config = _config()
    pl = MassterPL(**config).eval()
    inputs = _inputs(config)
    output = pl(**inputs)
    for name in ("pred_rt", "pred_ccs"):
        assert output[name].shape == (2,)
        assert th.isfinite(output[name]).all() and (output[name] > 0).all()
        th.testing.assert_close(output[name][0], output[name][1])
    altered = pl(**{**inputs, "spec_prec_type": th.tensor([1, 1])})
    th.testing.assert_close(output["pred_rt"], altered["pred_rt"])
    assert not th.allclose(output["pred_ccs"], altered["pred_ccs"])
    (output["pred_rt"].sum() + output["pred_ccs"].sum()).backward()
    for module in (
        pl.model.precursor_encoder,
        pl.model.rt_head.regressor,
        pl.model.ccs_head.regressor,
    ):
        assert sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in pl.model.intensity_head.parameters())


def test_joint_common_step_and_valid_label_epoch_denominator():
    config = _config()
    pl = MassterPL(**config).eval()
    batch = _inputs(config)
    with th.no_grad():
        output = pl(**batch)
    batch.update(
        spec_mzs=output["pred_mzs"],
        spec_ints=output["pred_logprobs"].exp(),
        spec_batch_idxs=output["pred_batch_idxs"],
        spec_prec_mz=th.tensor([47.0, 47.0]),
        spec_unique_id=th.tensor([1, 2]),
        mol_smiles=np.array(["CCO", "CCO"]),
        spec_per_group=th.ones(2),
        spec_per_mol=th.ones(2),
        group_per_mol=th.ones(2),
        spec_rt=output["pred_rt"] + 60.0,
        spec_ccs=th.tensor([output["pred_ccs"][0] + 100.0, -1.0]),
    )
    result = pl._common_step(batch, split="train")
    th.testing.assert_close(result["rt_loss"], th.tensor([0.5, 0.5]))
    th.testing.assert_close(result["ccs_loss"], th.tensor([0.5, 0.0]))
    expected = (
        config["primary_loss_weight"] * result["primary_loss"]
        + 0.1 * result["rt_loss"]
        + 0.1 * result["ccs_loss"]
    )
    th.testing.assert_close(result["loss"], expected)
    result["mean_loss"].backward()
    assert pl.model.ccs_head.regressor[0].weight.grad.abs().sum() > 0
    pl._accumulate_running_metrics(result, "train")
    metrics, _ = pl._reduce_running_metrics("train")
    assert metrics["rt_mae"] == pytest.approx(60.0)
    assert metrics["ccs_mae"] == pytest.approx(100.0)
    assert metrics["ccs_label_fraction"] == 0.5
    # An entirely unlabelled batch cannot dilute the valid-label epoch metric.
    unlabelled = dict(result)
    for name in ("ccs_loss", "ccs_mae", "ccs_mape", "ccs_label_fraction"):
        unlabelled[name] = th.zeros(2)
    pl._accumulate_running_metrics(unlabelled, "train")
    assert pl._reduce_running_metrics("train")[0]["ccs_mae"] == pytest.approx(100.0)
    unlabeled_inputs = {k: v for k, v in batch.items() if k not in ("spec_rt", "spec_ccs")}
    inferred = pl.inference_step(unlabeled_inputs, metrics=None)
    assert "pred_rt" in inferred and "pred_ccs" in inferred
    with pytest.raises(ValueError, match="training requires"):
        pl._common_step(unlabeled_inputs, split="train")


def test_legacy_state_and_msms_outputs_unchanged(tmp_path):
    config = _config(False)
    th.manual_seed(19)
    old = MassterPL(**config).eval()
    th.manual_seed(19)
    enabled = MassterPL(**_config(True)).eval()
    for name, value in old.state_dict().items():
        th.testing.assert_close(value, enabled.state_dict()[name], rtol=0, atol=0)
    inputs = _inputs(config)
    a, b = old(**inputs), enabled(**inputs)
    for key in ("pred_mzs", "pred_logprobs", "pred_batch_idxs"):
        th.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert "pred_rt" not in a
    legacy = copy.deepcopy(config)
    for key in (
        "masster_predict_rt_ccs",
        "masster_predict_rt",
        "masster_predict_ccs",
        "masster_rt_loss_weight",
        "masster_ccs_loss_weight",
        "masster_rt_scale",
        "masster_ccs_scale",
    ):
        del legacy[key]
    del legacy["spec_params"]["rt_ccs"], legacy["spec_params"]["rt_ccs_labels_fp"]
    restored = MassterPL(**get_pl_hparams({"hyper_parameters": legacy}))
    restored.load_state_dict(old.state_dict(), strict=True)
    direct = MassterPL(**legacy, _legacy_candidate_backend=True)
    direct.load_state_dict(old.state_dict(), strict=True)
    checkpoint = tmp_path / "model.ckpt"
    th.save({"state_dict": enabled.state_dict(), "hyper_parameters": _config(True)}, checkpoint)
    saved = th.load(checkpoint, weights_only=False)
    restored = MassterPL(**get_pl_hparams(saved))
    restored.load_state_dict(saved["state_dict"], strict=True)
    th.testing.assert_close(restored.eval()(**inputs)["pred_rt"], b["pred_rt"])


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"masster_context_in_gnn": True}, "context-free"),
        ({"masster_encoder_context_token": True}, "context-free"),
        ({"masster_ccs_scale": 0.0}, "scales"),
        ({"masster_rt_loss_weight": -1.0}, "loss_weight"),
    ],
)
def test_invalid_property_config(overrides, match):
    config = _config()
    config.update(overrides)
    with pytest.raises(ValueError, match=match):
        MassterPL(**config)


def test_prepared_enveda_arm_and_export_tokens():
    config = load_config("config", "config/masster/enveda/enveda_180_rt_ccs.yml", None)
    assert config["masster_predict_rt"] and config["spec_params"]["rt"]
    assert config["masster_predict_ccs"] and config["spec_params"]["ccs"]
    assert not config["spec_params"]["merge"]
    assert config["spec_params"]["rt_labels_fp"] == "data/df/enveda_180_df.csv"
    assert {"pred_rt", "pred_ccs"} <= set(export_columns.list_output_subset_options())


def test_property_predictions_export_as_scalars():
    config = _config()
    pl = MassterPL(**config).eval()
    batch = _inputs(config)
    batch.update(spec_unique_id=th.tensor([8, 9]), mol_id=[4, 4])
    expected = pl.inference_step(batch, metrics=None)
    subset = {"pred_mzs", "pred_ints", "pred_rt", "pred_ccs"}
    exported = runners.run_inference_pkl(
        model=pl,
        dataloader=[batch],
        device=th.device("cpu"),
        batch_cutoff=-1,
        nb_iso=False,
        output_set=subset,
        evaluate=False,
    )
    for name in ("pred_rt", "pred_ccs"):
        th.testing.assert_close(exported[name], expected[name])
    rows = batch_rows.process_batch_to_tabular_row(
        batch_input=batch,
        model=pl,
        device=th.device("cpu"),
        evaluate=False,
        eval_score_types=[],
        BATCH_IDX_VARS=["pred_batch_idxs"],
        NODE_IDX_VARS=[],
        FORMULA_IDX_VARS=[],
        min_peak_int=0.0,
        rebase_indices=False,
        return_formula_str=False,
        return_cc_hash=False,
        output_set=subset,
    )
    columns = [column.name for column in export_columns.selected_column_specs(False, False)]
    for name in ("pred_rt", "pred_ccs"):
        assert rows[columns.index(name)] == pytest.approx(expected[name].detach().tolist())


@pytest.mark.parametrize("name", ["rt", "ccs"])
def test_independent_head_training(name):
    config = _config(False)
    config[f"masster_predict_{name}"] = True
    config["spec_params"][name] = True
    pl = MassterPL(**config).eval()
    other = "ccs" if name == "rt" else "rt"
    assert getattr(pl.model, f"{other}_head") is None
    batch = _inputs(config)
    output = pl(**batch)
    assert f"pred_{other}" not in output
    batch.update(
        spec_mzs=output["pred_mzs"].detach(),
        spec_ints=output["pred_logprobs"].detach().exp(),
        spec_batch_idxs=output["pred_batch_idxs"],
        spec_prec_mz=th.tensor([47.0, 47.0]),
        spec_unique_id=th.tensor([1, 2]),
        mol_smiles=np.array(["CCO", "CCO"]),
        spec_per_group=th.ones(2),
        spec_per_mol=th.ones(2),
        group_per_mol=th.ones(2),
    )
    scale = config[f"masster_{name}_scale"]
    batch[f"spec_{name}"] = output[f"pred_{name}"].detach() + scale
    batch[f"spec_{name}"][1] = float("nan")
    result = pl._common_step(batch, split="train")
    th.testing.assert_close(result[f"{name}_loss"], th.tensor([0.5, 0.0]))
    assert f"{other}_loss" not in result
    result["mean_loss"].backward()
    assert getattr(pl.model, f"{name}_head").regressor[0].weight.grad.abs().sum() > 0
    pl._accumulate_running_metrics(result, "train")
    assert pl._reduce_running_metrics("train")[0][f"{name}_mae"] == pytest.approx(scale)


def test_disjoint_property_label_sets(tmp_path):
    from fragnnet.dataset.property_labels import join_property_labels

    original = pd.DataFrame({"dset": ["enveda"] * 3, "dset_spec_id": ["a", "b", "c"]})
    rt = tmp_path / "rt.csv"
    ccs = tmp_path / "ccs.csv"
    pd.DataFrame({"dset_spec_id": ["b"], "rtinseconds": [120.0]}).to_csv(rt, index=False)
    pd.DataFrame({"dset_spec_id": ["a", "c"], "ccs": [150.0, -1.0]}).to_csv(ccs, index=False)
    result = join_property_labels(join_property_labels(original, str(rt), "rt"), str(ccs), "ccs")
    assert result.rtinseconds.notna().tolist() == [False, True, False]
    assert result.ccs.notna().tolist() == [True, False, False]
    assert result.rtinseconds.iloc[1] == 120.0 and result.ccs.iloc[0] == 150.0
    assert result.dset_spec_id.tolist() == original.dset_spec_id.tolist()


def test_original_combined_checkpoint_migrates_without_changing_predictions():
    config = _config()
    original = MassterPL(**config).eval()
    state = original.state_dict()
    renames = {
        "model.rt_head.regressor.": "model.rt_ccs_heads.rt_head.",
        "model.ccs_head.regressor.": "model.rt_ccs_heads.ccs_head.",
        "model.ccs_head.adduct_embedding.": "model.rt_ccs_heads.adduct_embedding.",
    }
    for key in list(state):
        for new, old in renames.items():
            if key.startswith(new):
                state[old + key[len(new) :]] = state.pop(key)
                break
    del config["masster_predict_rt"], config["masster_predict_ccs"]
    config["masster_predict_rt_ccs"] = True
    restored = MassterPL(**get_pl_hparams({"hyper_parameters": config})).eval()
    restored.load_state_dict(state, strict=True)
    inputs = _inputs(config)
    expected, actual = original(**inputs), restored(**inputs)
    for key in ("pred_rt", "pred_ccs", "pred_logprobs"):
        th.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)


def test_legacy_optimizer_slots_follow_renamed_parameters(monkeypatch):
    from fragnnet.pl_model.spectrum_pl import SpectrumPL

    pl = MassterPL(**_config())
    parameters = dict(pl.named_parameters())
    # Original wrapper registered its adduct embedding before both regressors.
    head_names = [
        "model.ccs_head.adduct_embedding.weight",
        *[name for name in parameters if name.startswith("model.rt_head.")],
        *[name for name in parameters if name.startswith("model.ccs_head.regressor.")],
    ]
    old_names = [name for name in parameters if name not in head_names] + head_names
    state = {}
    for name in old_names:
        old = name.replace("model.rt_head.regressor.", "model.rt_ccs_heads.rt_head.")
        old = old.replace("model.ccs_head.regressor.", "model.rt_ccs_heads.ccs_head.")
        old = old.replace(
            "model.ccs_head.adduct_embedding.", "model.rt_ccs_heads.adduct_embedding."
        )
        state[old] = parameters[name].detach().clone()
    optimizer = pl.configure_optimizers()["optimizer"]
    saved = optimizer.state_dict()
    for index, name in enumerate(old_names):
        saved["state"][index] = {
            "step": th.tensor(1.0),
            "exp_avg": th.full_like(parameters[name], float(index)),
            "exp_avg_sq": th.ones_like(parameters[name]),
        }
    checkpoint = {"state_dict": state, "optimizer_states": [saved]}
    monkeypatch.setattr(SpectrumPL, "on_load_checkpoint", lambda self, checkpoint: None)
    pl.on_load_checkpoint(checkpoint)
    optimizer.load_state_dict(saved)
    for name, parameter in parameters.items():
        th.testing.assert_close(
            optimizer.state[parameter]["exp_avg"],
            th.full_like(parameter, float(old_names.index(name))),
        )
