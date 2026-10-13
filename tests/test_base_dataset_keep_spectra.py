"""``BaseDataset.keep_spectra``: the one way model code drops spectra after ``_base_init``."""

import pandas as pd
import pytest

from fragnnet.dataset.base_dataset import BaseDataset


@pytest.fixture
def merged_dataset() -> BaseDataset:
    """A merged dataset: each ``spec_df`` row stands for two unmerged spectra."""
    dataset = BaseDataset.__new__(BaseDataset)
    dataset.spec_params = {"merge": True, "remove_isotope_peaks": False}
    dataset.spec_df = pd.DataFrame(
        {"spec_id": [100, 101], "group_id": [100, 101], "mol_id": [1, 2]}
    )
    dataset.mol_df = pd.DataFrame({"mol_id": [1, 2]}).set_index("mol_id", drop=False)
    dataset.um_spec_df = pd.DataFrame(
        {"spec_id": [10, 11, 12, 13], "group_id": [100, 100, 101, 101], "mol_id": [1, 1, 2, 2]}
    )
    dataset.split_df = dataset.um_spec_df.copy()
    return dataset


def test_keep_spectra_drops_dependents(merged_dataset: BaseDataset) -> None:
    removed = merged_dataset.keep_spectra(lambda frame: frame["mol_id"] != 1)

    assert removed == 1
    assert merged_dataset.spec_df["spec_id"].tolist() == [101]
    assert merged_dataset.spec_df.index.tolist() == [0]
    assert merged_dataset.mol_df["mol_id"].tolist() == [2]
    assert merged_dataset.um_spec_df["spec_id"].tolist() == [12, 13]
    assert merged_dataset.split_df["spec_id"].tolist() == [12, 13]
    assert merged_dataset.spec_per_mol == {2: 1}


def test_keep_spectra_keeping_everything_changes_nothing(merged_dataset: BaseDataset) -> None:
    um_spec_df = merged_dataset.um_spec_df

    assert merged_dataset.keep_spectra(lambda frame: frame["mol_id"] > 0) == 0
    assert merged_dataset.um_spec_df is um_spec_df
