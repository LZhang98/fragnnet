"""Merged CE weights preserve target semantics and reach both conditioning models."""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch as th

from fragnnet.dataset.base_dataset import BaseDataset, _ce_list_to_stats
from fragnnet.model.mass_shatter_model import MassShatterContextEncoder
from fragnnet.runner import load_config
from fragnnet.utils.pl_utils import get_pl_hparams
from fragnnet.utils.proc_utils import merge_spec_df
from fragnnet.utils.spec_utils import weighted_mean_by_batch
from tests.test_ce_multi_location import _build


@pytest.fixture
def spectra() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "spec_id": [1, 2, 3, 4],
            "group_id": [1, 1, 1, 2],
            "mol_id": [1, 1, 1, 2],
            "inst_type": ["FT"] * 4,
            "frag_mode": ["HCD"] * 4,
            "prec_type": ["[M+H]+"] * 4,
            "prec_mz": [100.0] * 4,
            "nce": [[10.0], [10.0], [20.0, 30.0, 40.0], [80.0]],
            "ace": [[5.0], [15.0], [25.0, 35.0, 45.0], [160.0]],
            "peaks": [[(100.0, 1.0)], [(200.0, 1.0)], [(300.0, 1.0)], [(400.0, 1.0)]],
        }
    )


def merge(spectra: pd.DataFrame, mode: str = "weighted") -> pd.DataFrame:
    return merge_spec_df(
        spectra,
        keep_ces=True,
        method="tolerance_consensus",
        ce_key="nce",
        tolerance_by_inst={"FT": {"rel": 1e-5, "min_mz": 200.0}},
        normalize_each_spectrum=True,
        equal_ce_weight=True,
        ce_aggregation=mode,
    )


def test_weighted_context_is_order_invariant_and_preserves_targets(spectra) -> None:
    actual = merge(spectra).set_index("group_id")
    reversed_rows = merge(spectra.iloc[::-1]).set_index("group_id")
    legacy = merge(spectra, "legacy").set_index("group_id")
    for group in (1, 2):
        assert actual.loc[group, "peaks"] == legacy.loc[group, "peaks"]
        assert actual.loc[group, "peaks"] == reversed_rows.loc[group, "peaks"]
        for key in ("nce", "ace", "nce_weights", "ace_weights"):
            assert actual.loc[group, key] == reversed_rows.loc[group, key]
    assert actual.loc[1, "nce"] == [10.0, 20.0, 30.0, 40.0]
    assert actual.loc[1, "nce_weights"] == pytest.approx([0.5, 1 / 6, 1 / 6, 1 / 6])
    assert actual.loc[1, "ace"] == [5.0, 15.0, 25.0, 35.0, 45.0]
    assert actual.loc[1, "ace_weights"] == pytest.approx([0.25, 0.25, 1 / 6, 1 / 6, 1 / 6])
    assert actual.loc[2, "ace_weights"] == [1.0]
    assert legacy.loc[1, "ace"] == [5.0, 25.0, 35.0, 45.0]
    assert "ace_weights" not in legacy


def test_replicating_a_schedule_does_not_change_its_weight(spectra) -> None:
    repeated = pd.concat([spectra, spectra.iloc[:2]], ignore_index=True)
    actual = merge(repeated).iloc[0]
    expected = merge(spectra).iloc[0]
    for key in ("nce", "ace", "nce_weights", "ace_weights"):
        assert actual[key] == pytest.approx(expected[key])


def test_weighted_context_retains_missing_token_and_rejects_empty_schedule(spectra) -> None:
    spectra.at[0, "ace"] = [np.nan]
    actual = merge(spectra).iloc[0]
    assert actual.ace[0] == -1
    assert actual.ace_weights[0] == pytest.approx(0.25)
    spectra.at[0, "ace"] = []
    with pytest.raises(ValueError, match="nonempty"):
        merge(spectra)


def test_weighted_aggregation_rejects_incompatible_merge(spectra) -> None:
    with pytest.raises(ValueError, match="equal-CE consensus"):
        merge_spec_df(spectra, keep_ces=True, ce_aggregation="weighted")


def test_dataset_carries_weights_and_weighted_stats(tmp_path, spectra) -> None:
    # Exercise raw scalar preparation, merging, row conversion, and ragged collation.
    raw = spectra.copy()
    raw["nce"] = [10.0, 10.0, 20.0, 80.0]
    raw["ace"] = [5.0, 15.0, 40.0, 160.0]
    for key in ("nce_extra_1", "nce_extra_2", "ace_extra_1", "ace_extra_2"):
        raw[key] = -1.0
    raw[["spec_id", "mol_id", "group_id"]].to_csv(tmp_path / "train_ids.csv", index=False)
    config = load_config("config/template.yml", None, None)
    params = config["spec_params"]
    params.update(
        merge=True,
        merge_keep_ces=True,
        nce=True,
        ace=True,
        merge_method="tolerance_consensus",
        merge_equal_ce_weight=True,
        merge_normalize_each_spectrum=True,
        merge_ce_aggregation="weighted",
        drop_empty_spectra=False,
        unique_id=False,
        counts=False,
    )
    frames = BaseDataset._setup_dfs(
        raw,
        pd.DataFrame({"mol_id": [1, 2], "smiles": ["CCO", "CO"]}),
        str(tmp_path),
        ["train"],
        config["subsample_params"],
        params,
    )
    dataset = BaseDataset()
    dataset.spec_params = params
    dataset.ce_key = frames[-1]
    dataset._setup_prec_type_to_idx()
    dataset._setup_inst_type_to_idx()
    samples = [dataset._process_spec(row) for _, row in frames[0].iterrows()]
    batch_size, keys, batch = BaseDataset._setup_collate(samples)
    BaseDataset._special_collate(keys, batch)
    BaseDataset._standard_collate(batch_size, keys, batch)
    th.testing.assert_close(batch["spec_ace"], th.tensor([5.0, 15.0, 40.0, 160.0]))
    th.testing.assert_close(batch["spec_ace_weights"], th.tensor([0.25, 0.25, 0.5, 1.0]))
    th.testing.assert_close(batch["spec_ace_batch_idxs"], th.tensor([0, 0, 0, 1]))
    th.testing.assert_close(batch["spec_ce_weights"], batch["spec_nce_weights"])
    assert batch["spec_ace_stats"][0, 0].item() == pytest.approx(25.0)
    assert batch["spec_ace_stats"][1, 0].item() == pytest.approx(160.0)


@pytest.mark.parametrize("ce_type", ["embed", "id"])
def test_mass_shatter_uses_weighted_ace_and_preserves_gradients(ce_type) -> None:
    encoder = MassShatterContextEncoder(
        num_precursor_types=1,
        num_instrument_types=0,
        num_fragmentation_modes=0,
        use_nce=False,
        use_ace=True,
        int_embedder="abs-sines",
        ce_insert_type=ce_type,
        ce_insert_size=4,
        metadata_embedding_dim=4,
        context_dim=8,
        nce_max=200,
        nce_mean=60,
        nce_std=40,
        ace_max=300,
        ace_mean=20,
        ace_std=15,
    )
    values = th.tensor([5.0, 15.0, 40.0, 160.0])
    weights = th.tensor([0.25, 0.25, 0.5, 1.0])
    if ce_type == "embed":
        features = encoder.ace_embedding(values.long().unsqueeze(1))
    else:
        features = ((values - 20) / 15).unsqueeze(1)
    expected_ce = th.stack(
        (0.25 * features[0] + 0.25 * features[1] + 0.5 * features[2], features[3])
    )
    expected = encoder.projector(
        th.cat((encoder.precursor_embedding(th.tensor([0, 0])), expected_ce), dim=1)
    )
    actual = encoder(
        batch_size=2,
        precursor_types=th.tensor([0, 0]),
        instrument_types=None,
        fragmentation_modes=None,
        nce=None,
        nce_batch_indices=None,
        ace=values,
        ace_batch_indices=th.tensor([0, 0, 0, 1]),
        ace_weights=weights,
    )
    th.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert encoder.projector[0].weight.grad.abs().sum() > 0
    if ce_type == "embed":
        assert encoder.ace_embedding[1].weight.grad.abs().sum() > 0
        # The configured range distinguishes high readings instead of saturating at 100.
        high = encoder._embed_ce_values(
            th.tensor([150.0, 250.0]), None, 2, 300, encoder.ace_embedding
        )
        assert not th.allclose(high[0], high[1])


def test_fragnnet_ce_embedding_and_scaler_stats_use_same_weights() -> None:
    model = _build(use_ace=True, ace_max=300)
    nce = th.tensor([10.0, 20.0])
    ace = th.tensor([5.0, 15.0, 40.0])
    actual = model.embed_ce(
        nce,
        th.zeros(2, dtype=th.long),
        1,
        ace,
        th.zeros(3, dtype=th.long),
        th.tensor([0.5, 0.5]),
        th.tensor([0.25, 0.25, 0.5]),
    )
    n = model.ce_embedder(model.ce_transform(nce)).mean(0)
    a = model.ce_embedder_ace(model.ace_transform(ace))
    expected = th.cat((n, 0.25 * a[0] + 0.25 * a[1] + 0.5 * a[2])).unsqueeze(0)
    th.testing.assert_close(actual, expected)
    stats = _ce_list_to_stats(ace, th.tensor([0.25, 0.25, 0.5]))
    th.testing.assert_close(stats, th.tensor([[25.0, (237.5) ** 0.5, 1.0]]))
    actual.square().sum().backward()
    assert model.ce_embedder_ace[1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("weights", [[0.0, 0.0], [-1.0, 2.0], [float("nan"), 1.0]])
def test_invalid_reading_weights_fail(weights) -> None:
    with pytest.raises(ValueError):
        weighted_mean_by_batch(th.ones(2, 3), th.tensor(weights), th.zeros(2, dtype=th.long), 1)


def test_checkpoint_migration_and_merged_config_pair() -> None:
    old = {"hyper_parameters": {"model_type": "mass_shatter", "spec_params": {"merge": True}}}
    snapshot = copy.deepcopy(old)
    assert get_pl_hparams(old)["spec_params"]["merge_ce_aggregation"] == "legacy"
    assert old == snapshot
    configs = [
        load_config("config/template.yml", str(p), None)
        for p in sorted(Path("config/masshatter/merged/context_ablation").glob("*.yml"))
    ]
    assert len(configs) == 2
    assert {k for k in configs[0] if configs[0][k] != configs[1][k]} == {
        "wandb_name",
        "mass_shatter_context_in_gnn",
    }
    for config in configs:
        assert config["ace_max"] == 300
        assert config["spec_params"]["merge_ce_aggregation"] == "weighted"


def test_zero_weight_or_missing_ce_has_no_scaler_statistics() -> None:
    th.testing.assert_close(
        _ce_list_to_stats(th.tensor([10.0, -1.0]), th.tensor([0.0, 1.0])),
        th.zeros(1, 3),
    )
    with pytest.raises(ValueError, match="weights"):
        _ce_list_to_stats(th.tensor([10.0, 20.0]), th.tensor([1.0]))


def test_mass_shatter_wrapper_passes_reading_weights() -> None:
    from rdkit import Chem
    from torch_geometric.data import Batch

    from fragnnet.utils.feat_utils import get_mol_graph
    from tests.test_mass_shatter_pl import _build_pl

    config = load_config("config/template.yml", None, None)
    spec_params = config["spec_params"]
    spec_params.update(nce=True, ace=True, merge_keep_ces=True)
    pl, config = _build_pl(spec_params=spec_params, ace_max=300)
    pl.eval()
    mol_params = config["mol_params"]
    graph = Batch.from_data_list(
        [
            get_mol_graph(
                Chem.MolFromSmiles("CCO"),
                mol_params["pyg_node_feats"],
                mol_params["pyg_edge_feats"],
                mol_params["pyg_pe_embed_k"],
                True,
            )
        ]
    )
    captured = []
    handle = pl.model.context_encoder.register_forward_pre_hook(
        lambda module, args, kwargs: captured.append(kwargs),
        with_kwargs=True,
    )
    try:
        output = pl(
            mol_pyg=graph,
            spec_prec_type=th.tensor([0]),
            spec_inst_type=th.tensor([0]),
            spec_nce=th.tensor([10.0, 20.0]),
            spec_nce_batch_idxs=th.tensor([0, 0]),
            spec_nce_weights=th.tensor([0.5, 0.5]),
            spec_ace=th.tensor([5.0, 15.0, 40.0]),
            spec_ace_batch_idxs=th.tensor([0, 0, 0]),
            spec_ace_weights=th.tensor([0.25, 0.25, 0.5]),
        )
    finally:
        handle.remove()
    th.testing.assert_close(captured[0]["ace_weights"], th.tensor([0.25, 0.25, 0.5]))
    th.testing.assert_close(captured[0]["nce_weights"], th.tensor([0.5, 0.5]))
    assert th.isfinite(output["pred_logprobs"]).all()
