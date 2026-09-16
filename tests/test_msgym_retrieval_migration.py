"""Tests for migrating legacy MSGYM retrieval chunks."""

from importlib import util
from pathlib import Path

import pandas as pd

SCRIPT_FP = (
    Path(__file__).resolve().parents[1]
    / "preproc_scripts/msgym_ms2c/01_prepare_msgym_retrieval.py"
)
SPEC = util.spec_from_file_location("prepare_msgym_retrieval", SCRIPT_FP)
assert SPEC is not None and SPEC.loader is not None
MSGYM = util.module_from_spec(SPEC)
SPEC.loader.exec_module(MSGYM)


def test_msgym_json_exports_selected_candidates(tmp_path: Path) -> None:
    raw_dp = tmp_path / "raw"
    export_dp = tmp_path / "export"
    raw_dp.mkdir()

    pd.DataFrame(
        {
            "identifier": ["query_a", "not_selected"],
            "smiles": ["C", "CC"],
            "precursor_mz": [17.0, 31.0],
            "adduct": ["[M+H]+", "[M+H]+"],
            "instrument_type": ["Orbitrap", "Orbitrap"],
            "collision_energy": [20.0, 30.0],
            "fold": ["test", "test"],
            "simulation_challenge": [True, False],
        }
    ).to_csv(raw_dp / "MassSpecGym1.5.tsv", sep="\t", index=False)
    candidates_fp = raw_dp / "candidates.json"
    candidates_fp.write_text('{"C": ["C", "CC"], "CC": ["CC"]}')

    manifest_fp, candidate_smiles_fp = MSGYM.export_msgym_json(
        raw_dp / "MassSpecGym1.5.tsv",
        candidates_fp,
        export_dp,
        "simulation_challenge",
    )
    assert pd.read_csv(candidate_smiles_fp)["smiles"].tolist() == ["C", "CC"]

    manifest = pd.read_pickle(manifest_fp)
    assert manifest["source_identifier"].tolist() == ["query_a", "query_a"]
    assert manifest["candidate_smiles"].tolist() == ["C", "CC"]


def test_build_proc_aligns_candidates_and_query_groups(tmp_path: Path) -> None:
    manifest_fp = tmp_path / "query_candidates.pkl.gz"
    output_dp = tmp_path / "proc"
    pd.DataFrame(
        {
            "source_identifier": ["query_a", "query_a", "query_a", "query_b"],
            "source_smiles": ["C", "C", "C", "CC"],
            "candidate_smiles": ["C", "CC", "CC", "CC"],
            "prec_type": ["[M+H]+", "[M+H]+", "[M+H]+", "[M-H]-"],
            "prec_mz": [17.0, 17.0, 17.0, 29.0],
            "inst_type": ["FT", "FT", "FT", "QTOF"],
            "frag_mode": ["HCD", "HCD", "HCD", "CID"],
            "ace": [20.0, 20.0, 20.0, 30.0],
            "nce": [float("nan")] * 4,
            "ion_mode": ["P", "P", "P", "N"],
            "spec_type": ["MS2"] * 4,
            "dset": ["msgym_1_5"] * 4,
            "peaks": [[], [], [], []],
        }
    ).to_pickle(manifest_fp)

    mol_fp, spec_fp, split_fp = MSGYM.build_proc_from_manifest(manifest_fp, output_dp)

    mol_df = pd.read_pickle(mol_fp)
    assert mol_df[["mol_id", "smiles"]].to_dict("records") == [
        {"mol_id": 0, "smiles": "C"},
        {"mol_id": 1, "smiles": "CC"},
    ]
    assert mol_df["formula"].tolist() == ["CH4", "C2H6"]

    spec_df = pd.read_pickle(spec_fp)
    assert spec_df["mol_id"].tolist() == [0, 1, 1]
    assert spec_df["spec_id"].tolist() == [0, 1, 2]
    assert spec_df["group_id"].tolist() == [0, 0, 1]
    assert spec_df["ace_extra_1"].tolist() == [-1.0, -1.0, -1.0]

    split_df = pd.read_csv(split_fp)
    assert split_df.to_dict("records") == [
        {"spec_id": 0, "mol_id": 0, "group_id": 0},
        {"spec_id": 1, "mol_id": 1, "group_id": 0},
        {"spec_id": 2, "mol_id": 1, "group_id": 1},
    ]


def test_msgym_json_selects_the_requested_fold(tmp_path: Path) -> None:
    raw_dp = tmp_path / "raw"
    raw_dp.mkdir()
    tsv_fp = raw_dp / "MassSpecGym1.5.tsv"
    pd.DataFrame(
        {
            "identifier": ["test_query", "val_query"],
            "smiles": ["C", "CC"],
            "precursor_mz": [17.0, 31.0],
            "adduct": ["[M+H]+", "[M+H]+"],
            "instrument_type": ["Orbitrap", "Orbitrap"],
            "collision_energy": [20.0, 30.0],
            "fold": ["test", "val"],
            "simulation_challenge": [True, True],
        }
    ).to_csv(tsv_fp, sep="\t", index=False)
    candidates_fp = raw_dp / "candidates.json"
    candidates_fp.write_text('{"C": ["C"], "CC": ["CC"]}')

    for fold, expected_query in (("test", "test_query"), ("val", "val_query")):
        manifest_fp, _ = MSGYM.export_msgym_json(
            tsv_fp, candidates_fp, tmp_path / fold, "simulation_challenge", fold
        )
        manifest = pd.read_pickle(manifest_fp)
        assert manifest["source_identifier"].tolist() == [expected_query]

    # The default stays on the test fold so existing invocations are unchanged.
    manifest_fp, _ = MSGYM.export_msgym_json(
        tsv_fp, candidates_fp, tmp_path / "default", "simulation_challenge"
    )
    assert pd.read_pickle(manifest_fp)["source_identifier"].tolist() == ["test_query"]
