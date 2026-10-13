"""spec_params.merge_train_only merges the train split and leaves val/test per spectrum."""

import numpy as np
import pandas as pd
import pytest
import torch as th

from fragnnet.dataset.base_dataset import BaseDataset
from fragnnet.runner import load_config


@pytest.fixture
def raw_spectra() -> pd.DataFrame:
    # Group 1: three replicates of one molecule, one without CE (the -1 sentinel).
    # Group 2: a single spectrum of another molecule.
    raw = pd.DataFrame(
        {
            "spec_id": [1, 2, 3, 4],
            "group_id": [1, 1, 1, 2],
            "mol_id": [1, 1, 1, 2],
            "inst_type": ["ORB"] * 4,
            "frag_mode": ["HCD"] * 4,
            "prec_type": ["[M+H]+"] * 4,
            "prec_mz": [300.0] * 4,
            "nce": [np.nan] * 4,
            "ace": [10.0, 20.0, np.nan, 30.0],
            "peaks": [
                [(100.0, 1.0)],
                [(100.0, 1.0), (150.0, 1.0)],
                [(150.0, 1.0)],
                [(120.0, 1.0)],
            ],
        }
    )
    for key in ("nce_extra_1", "nce_extra_2", "ace_extra_1", "ace_extra_2"):
        raw[key] = np.nan
    return raw


@pytest.fixture
def spec_params() -> dict:
    params = load_config("config", None, None)["spec_params"]
    params.update(
        merge=False,
        merge_keep_ces=False,
        merge_train_only=True,
        nce=True,
        ace=True,
        merge_method="tolerance_consensus",
        merge_equal_ce_weight=True,
        merge_normalize_each_spectrum=True,
        drop_empty_spectra=False,
        unique_id=True,
        counts=True,
    )
    return params


def _dataset(tmp_path, raw: pd.DataFrame, split: str, spec_params: dict) -> BaseDataset:
    raw[["spec_id", "mol_id", "group_id"]].to_csv(tmp_path / f"{split}_ids.csv", index=False)
    dataset = BaseDataset()
    dataset._base_init(
        spec_fp=raw,
        mol_fp=pd.DataFrame({"mol_id": [1, 2], "smiles": ["CCO", "CO"]}),
        split_dp=str(tmp_path),
        split=split,
        subsample_params=load_config("config", None, None)["subsample_params"],
        spec_params=spec_params,
        enable_progress_bar=False,
    )
    return dataset


def test_train_split_is_merged_by_group(tmp_path, raw_spectra, spec_params) -> None:
    dataset = _dataset(tmp_path, raw_spectra, "train", spec_params)

    assert dataset.id_key == "group_id"
    assert dataset.spec_params["merge"] and dataset.spec_params["merge_keep_ces"]
    assert sorted(dataset.spec_df["group_id"]) == [1, 2]
    assert dataset.spec_per_group == {1: 1, 2: 1}
    group = dataset.spec_df.set_index("group_id").loc[1]
    # Equal weight per CE schedule (10, 20, missing): 100 Da is in 2 of 3, 150 Da in 2 of 3.
    mzs, ints = zip(*group["peaks"], strict=True)
    assert mzs == pytest.approx((100.0, 150.0))
    assert ints == pytest.approx((0.5 / 3 + 1.0 / 3, 0.5 / 3 + 1.0 / 3))
    # The merged row conditions on the whole ladder, missing-CE sentinel included.
    assert sorted(group["ace"]) == [-1, 10.0, 20.0]
    sample = dataset._process_spec(dataset.spec_df.iloc[0])
    assert sample["spec_unique_id"].item() == dataset.spec_df.iloc[0]["group_id"]


def test_val_split_stays_per_spectrum(tmp_path, raw_spectra, spec_params) -> None:
    dataset = _dataset(tmp_path, raw_spectra, "val", spec_params)

    assert dataset.id_key == "spec_id"
    assert not dataset.spec_params["merge"] and not dataset.spec_params["merge_keep_ces"]
    assert sorted(dataset.spec_df["spec_id"]) == [1, 2, 3, 4]
    assert dataset.spec_per_group == {1: 3, 2: 1}
    sample = dataset._process_spec(dataset.spec_df[dataset.spec_df["spec_id"] == 3].iloc[0])
    th.testing.assert_close(sample["spec_ce"], th.tensor([-1.0]))
    assert sample["spec_unique_id"].item() == 3


def test_caller_spec_params_are_not_mutated(tmp_path, raw_spectra, spec_params) -> None:
    _dataset(tmp_path, raw_spectra, "train", spec_params)

    assert spec_params["merge"] is False and spec_params["merge_keep_ces"] is False


def test_merge_train_only_refuses_merge(tmp_path, raw_spectra, spec_params) -> None:
    spec_params["merge"] = True

    with pytest.raises(ValueError, match="merge_train_only requires merge: False"):
        _dataset(tmp_path, raw_spectra, "val", spec_params)
