"""Regression tests for preproc_scripts/dataset_migration/merge_datasets.py."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "preproc_scripts"
    / "dataset_migration"
    / "merge_datasets.py"
)


def _write_source(proc_dp, inchikeys, prec_types):
    """One source dataset whose molecules and groups use source-local ids starting at 0."""
    proc_dp.mkdir()
    mol_df = pd.DataFrame(
        {"mol_id": range(len(inchikeys)), "inchikey_s": inchikeys, "mol": [None] * len(inchikeys)}
    )
    spec_df = pd.DataFrame(
        {
            "spec_id": range(len(prec_types)),
            "mol_id": [0] * len(prec_types),
            "group_id": list(range(len(prec_types))),
            "prec_type": prec_types,
            "inst_type": ["QTOF"] * len(prec_types),
        }
    )
    mol_df.to_pickle(proc_dp / "mol_df.pkl")
    spec_df.to_pickle(proc_dp / "spec_df.pkl")


def test_merge_assigns_global_mol_and_group_ids(tmp_path):
    # Source A: molecule AAA with two adducts. Source B: molecule BBB (local id 0)
    # plus a shared molecule AAA, so local ids and group ids collide across sources.
    _write_source(tmp_path / "A", ["AAA"], ["[M+H]+", "[M-H]-"])
    _write_source(tmp_path / "B", ["BBB", "AAA"], ["[M+H]+"])
    out_dp = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--proc_dirs",
            str(tmp_path / "A"),
            str(tmp_path / "B"),
            "--output_dp",
            str(out_dp),
        ],
        check=True,
        capture_output=True,
    )
    mol_df = pd.read_pickle(out_dp / "mol_df.pkl")
    spec_df = pd.read_pickle(out_dp / "spec_df.pkl")

    # Molecule ids are unique, and every spectrum references the right molecule
    assert mol_df["mol_id"].is_unique
    assert set(spec_df["mol_id"]) <= set(mol_df["mol_id"])
    mol_key = dict(zip(mol_df["mol_id"], mol_df["inchikey_s"]))
    assert [mol_key[m] for m in spec_df["mol_id"]] == ["AAA", "AAA", "BBB"]

    # Groups follow (mol_id, prec_type, inst_type) and never span molecules
    assert (spec_df.groupby("group_id")["mol_id"].nunique() == 1).all()
    trio = spec_df.groupby(["mol_id", "prec_type", "inst_type"])["group_id"].nunique()
    assert (trio == 1).all()
    assert spec_df["group_id"].nunique() == 3
