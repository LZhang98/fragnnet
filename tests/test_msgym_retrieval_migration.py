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
