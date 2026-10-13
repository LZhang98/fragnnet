"""Regression tests for predefined splits in preproc_scripts/04_prepare_split.py."""

import argparse
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "preproc_scripts" / "04_prepare_split.py"


def _load_split_module():
    spec = importlib.util.spec_from_file_location("prepare_split", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("id_type", ["spec_id", "mol_id", "group_id"])
def test_inline_predefined_numeric_ids_select_rows(tmp_path, monkeypatch, id_type):
    module = _load_split_module()
    # Bypass spectrum filters; this test covers only predefined id selection.
    monkeypatch.setattr(module, "filter_spec_mol", lambda spec_df, mol_df, **kwargs: (spec_df, mol_df))

    proc_dp = tmp_path / "proc"
    proc_dp.mkdir()
    pd.DataFrame({"mol_id": [0, 1, 2], "inchikey_s": ["A", "B", "C"]}).to_pickle(
        proc_dp / "mol_df.pkl"
    )
    pd.DataFrame(
        {"spec_id": [0, 1, 2], "mol_id": [0, 1, 2], "group_id": [0, 1, 2], "dset": ["d"] * 3}
    ).to_pickle(proc_dp / "spec_df.pkl")

    split_dp = tmp_path / "split"
    args = argparse.Namespace(
        proc_dp=str(proc_dp),
        split_dp=str(split_dp),
        primary_dsets=["d"],
        secondary_dsets=[],
        dag_filtering=False,
        split_key="inchikey_s",
        split_type="predefined",
        predefined_id_type=id_type,
        # inline ids arrive from argparse as strings
        predefined_train_ids=["0"],
        predefined_val_ids=["1"],
        predefined_test_ids=["2"],
        predefined_secondary_ids=None,
        predefined_train_id_fp=None,
        predefined_val_id_fp=None,
        predefined_test_id_fp=None,
        predefined_secondary_id_fp=None,
        **dict.fromkeys(
            [
                "prec_types", "frag_modes", "ion_modes", "inst_types", "max_peak_mz",
                "max_prec_mz", "min_prec_mz", "elements", "ces", "spec_type",
                "max_heavy_atom", "max_bond", "ce_types",
            ]
        ),
    )

    module.make_splits(args)

    for fn, expected in [("train_ids.csv", 0), ("val_ids.csv", 1), ("test_ids.csv", 2)]:
        assert pd.read_csv(split_dp / fn)[id_type].tolist() == [expected]
