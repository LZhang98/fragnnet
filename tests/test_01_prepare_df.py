"""Unit tests for preproc_scripts/01_prepare_df.py module.

Tests validate:
- MSP parsing and column renaming
- NIST EI CSV processing
- Comment field parsing
- Data dictionary validation
"""

import gzip
import importlib.util
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import fragnnet.utils.data_utils as data_utils


def load_prepare_module():
    """Load the preprocessing script as a module via importlib."""
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "preproc_scripts" / "01_prepare_df.py"
    spec = importlib.util.spec_from_file_location("prepare_df", str(script_path))
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Unable to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_info_from_comments():
    """Test extraction of info from MSP comments field."""
    module = load_prepare_module()

    comments = 'computed SMILES="CC(C)O" MoNA Rating="10" fragmentation mode="CID"'
    # Function includes quotes in extraction; verify content is present
    extracted = module.extract_info_from_comments(comments, "computed SMILES")
    assert extracted is not None and "CC(C)O" in extracted
    assert module.extract_info_from_comments(comments, "nonexistent") is None


def test_preproc_msp_basic():
    """Test basic MSP parsing."""
    module = load_prepare_module()

    msp_content = """ID: 1
Name: Test Compound
PrecursorMZ: 100.5
Num peaks: 2
50 100
75 200

ID: 2
Name: Test Compound 2
PrecursorMZ: 150.5
Num peaks: 1
100 500

"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".msp", delete=False) as f:
        f.write(msp_content)
        msp_fp = f.name

    try:
        keys = {"ID", "Name", "PrecursorMZ", "Num peaks", "MS"}
        df = module.preproc_msp(msp_fp, keys, num_entries=-1)

        assert len(df) == 2, "Should have 2 spectra"
        assert df.iloc[0]["ID"] == "1"
        assert df.iloc[0]["Name"] == "Test Compound"
        assert df.iloc[0]["PrecursorMZ"] == "100.5"
        assert df.iloc[0]["Num peaks"] == 2
        assert "50 100" in df.iloc[0]["MS"]
        assert "75 200" in df.iloc[0]["MS"]
    finally:
        Path(msp_fp).unlink()


def test_preproc_msp_num_entries_limit():
    """Test MSP parsing with entry limit."""
    module = load_prepare_module()

    msp_content = """ID: 1
Name: Compound 1
Num peaks: 1
50 100

ID: 2
Name: Compound 2
Num peaks: 1
60 200

ID: 3
Name: Compound 3
Num peaks: 1
70 300

"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".msp", delete=False) as f:
        f.write(msp_content)
        msp_fp = f.name

    try:
        keys = {"ID", "Name", "Num peaks", "MS"}
        df = module.preproc_msp(msp_fp, keys, num_entries=2)
        assert len(df) == 2, "Should limit to 2 entries"
    finally:
        Path(msp_fp).unlink()


def test_merge_and_check_with_smiles():
    """Test merge_and_check when mol_df is None (SMILES in MSP)."""
    module = load_prepare_module()

    msp_df = pd.DataFrame(
        {
            "Name": ["Compound1", "Compound2"],
            "smiles": ["CC(C)O", "c1ccccc1"],
            "spec_id": [None, None],
            "extra_col": [1, 2],
        }
    )
    rename_dict = {"Name": "name", "smiles": "smiles", "spec_id": "spec_id"}

    spec_df = module.merge_and_check(msp_df, None, rename_dict)

    assert len(spec_df) == 2
    assert list(spec_df.columns) == ["name", "smiles", "spec_id", "dset_spec_id"]
    assert spec_df["spec_id"].tolist() == [0, 1]
    assert spec_df["dset_spec_id"].tolist() == [0, 1]


def test_nist_ei_constants():
    """Test NIST EI constants are set correctly."""
    module = load_prepare_module()

    assert module.NIST_EI_COLLISION_ENERGY == "70.0 eV"
    assert module.NIST_EI_PRECURSOR_TYPE == "[M]+"
    assert module.NIST_EI_SPEC_TYPE == "MS1"
    assert module.NIST_EI_INST_TYPE == "GC-MS"
    assert module.NIST_EI_ION_MODE == "P"
    assert module.NIST_EI_DATASET_NAME == "nist_ei"


def test_process_nist_ei_csv_basic():
    """Test NIST EI CSV processing."""
    module = load_prepare_module()

    csv_content = """name,peaks,exact_mass,formula,smiles,inchikey
Compound1,"50 100
75 200",100.0,C2H4O,CC(C)O,LFQSCWFLJHTTHZ-UHFFFAOYSA-N
Compound2,"60 300
80 400",150.0,C3H6O2,c1ccccc1,UHOVQNZJYSORNB-UHFFFAOYSA-N
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        csv_fp = f.name

    try:
        spec_df = module.process_nist_ei_csv(csv_fp)

        assert len(spec_df) == 2
        assert spec_df.iloc[0]["frag_mode"] == "EI"
        assert spec_df.iloc[0]["ion_type"] == "EI"
        assert spec_df.iloc[0]["col_energy"] == "70.0 eV"
        assert spec_df.iloc[0]["prec_type"] == "[M]+"
        assert spec_df.iloc[0]["spec_type"] == "MS1"
        assert spec_df.iloc[0]["inst_type"] == "GC-MS"
        assert spec_df.iloc[0]["ion_mode"] == "P"
        assert spec_df.iloc[0]["dset"] == "nist_ei"
        assert spec_df.iloc[0]["spec_id"] == 0
    finally:
        Path(csv_fp).unlink()


def test_process_ms_gym_infers_negative_ion_mode():
    """MassSpecGym ion polarity should follow the adduct charge."""
    module = load_prepare_module()

    tsv_content = (
        "simulation_challenge\tmzs\tintensities\tprecursor_mz\tadduct\tparent_mass\t"
        "instrument_type\tcollision_energy\tidentifier\tfold\tsmiles\n"
        "True\t100,150\t10,20\t99.0\t[M-H]-\t100.0\tQTOF\t20\tMSG0001\t0\tCCO\n"
        "False\t101\t11\t103.0\t[M+H]+\t102.0\tOrbitrap\t30 normalized=True\t"
        "MSG0002\t1\tCCN\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        f.write(tsv_content)
        tsv_fp = f.name

    try:
        sim_df, sim_split_df = module.process_ms_gym(tsv_fp, "simulation_challenge")
        extra_df, _ = module.process_ms_gym(tsv_fp, "extra")

        assert len(sim_df) == 1
        assert sim_df.iloc[0]["ion_mode"] == "N"
        assert sim_df.iloc[0]["peaks"] == "100 10\n150 20"
        assert sim_df.iloc[0]["prec_type"] == "[M-H]-"
        assert sim_split_df.to_dict("records") == [{"dset_spec_id": "MSG0001", "fold": 0}]

        assert len(extra_df) == 1
        assert extra_df.iloc[0]["ion_mode"] == "P"
        assert extra_df.iloc[0]["normalized"]
        assert extra_df.iloc[0]["col_energy"] == "NCE=30%"
    finally:
        Path(tsv_fp).unlink()


def test_msp_key_dict_has_required_keys():
    """Verify MSP key dictionary contains expected keys."""
    module = load_prepare_module()

    required_keys = ["Precursor_type", "PrecursorMZ", "Num peaks", "MS", "Name", "SMILES"]
    for key in required_keys:
        assert key in module.MSP_KEY_DICT, f"Missing key: {key}"


def test_pcdl_keys_dict_has_required_keys():
    """Verify PCDL key dictionary contains expected keys."""
    module = load_prepare_module()

    required_keys = [
        "ACCESSION:",
        "CH$SMILES:",
        "CH$FORMULA:",
        "MS$FOCUSED_ION: PRECURSOR_M/Z",
        "PK$NUM_PEAK:",
    ]
    for key in required_keys:
        assert key in module.PCDL_KEYS, f"Missing key: {key}"


def _enveda_entry(title: str, adduct: str, ce: str, ionmode: str) -> dict:
    return {
        "title": title,
        "pepmass": "326.198845",
        "charge": "1-" if ionmode == "ESI Negative" else "1+",
        "adduct": adduct,
        "collision_energies": ce,
        "instrument": "Bruker timsTOF Pro 2",
        "instrument_type": "Q-TOF",
        "ionmode": ionmode,
        "inchikey": "AAAAMGLVZZXVDT-UHFFFAOYSA-N",
        "formula": "C18H25N5O",
        "smiles": "CC(C)N1C=NN=C1C1CCCN(CC(=O)NC2=CC=CC=C2)C1",
        "iupac_name": "N-phenyl-2-acetamide",
        "ccs": "190.94",
        "num_peaks": 2,
        "mzs": ["92.050941", "149.071959"],
        "intensities": ["8751.00", "13456.00"],
        "num_isotope_peaks": 2,
        "isotope_mzs": ["326.198653", "327.202283"],
        "isotope_intensities": ["0.93", "0.07"],
    }


@pytest.fixture
def enveda_jsonl_fp(tmp_path: Path) -> Path:
    """Enveda-180 export with a stepped negative, a single positive and a 2-energy entry."""
    entries = [
        _enveda_entry("enveda_000001_01", "[M-H]-", "-60,-40,-20", "ESI Negative"),
        _enveda_entry("enveda_000001_02", "[M+H]+", "40", "ESI Positive"),
        _enveda_entry("enveda_000001_03", "[M+H]+", "20,60", "ESI Positive"),
    ]
    fp = tmp_path / "enveda-180-filtered.jsonl.gz"
    with gzip.open(fp, "wt") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return fp


def test_parse_enveda_ce_str_strips_polarity_and_sorts():
    """Negative-mode energies lose their sign; stepped lists come out lowest first."""
    module = load_prepare_module()

    assert module.parse_enveda_ce_str("-40") == ["40 eV"]
    assert module.parse_enveda_ce_str("-60,-40,-20") == ["20 eV", "40 eV", "60 eV"]
    assert module.parse_enveda_ce_str("20,40,60") == ["20 eV", "40 eV", "60 eV"]


def test_process_enveda_maps_columns(enveda_jsonl_fp: Path):
    """Enveda rows reach the unified schema, and their CE strings parse downstream."""
    module = load_prepare_module()

    spec_df = module.process_enveda(str(enveda_jsonl_fp))

    assert spec_df["spec_id"].tolist() == [0, 1, 2]
    assert spec_df["dset_spec_id"].tolist() == [
        "enveda_000001_01",
        "enveda_000001_02",
        "enveda_000001_03",
    ]
    assert spec_df["ion_mode"].tolist() == ["N", "P", "P"]
    assert spec_df["prec_type"].tolist() == ["[M-H]-", "[M+H]+", "[M+H]+"]
    assert spec_df["stepped"].tolist() == [True, False, True]
    neg = spec_df.iloc[0]
    assert neg["peaks"] == "92.050941 8751.00\n149.071959 13456.00"
    assert neg["prec_mz"] == "326.198845"
    assert neg["inst_type"] == "Q-TOF"
    assert neg["inst"] == "Bruker timsTOF Pro 2"
    assert neg["spec_type"] == "MS2"
    assert neg["frag_mode"] == "CID"
    assert [neg[col] for col in module.CE_COLS] == ["20 eV", "40 eV", "60 eV"]
    assert data_utils.parse_ace_str(neg["col_energy_extra_2"]) == pytest.approx(60.0)
    assert "isotope_mzs" not in spec_df.columns
    assert "mzs" not in spec_df.columns
    for value in module.MSP_KEY_DICT.values():
        assert value in spec_df.columns

    single = spec_df.iloc[1]
    assert single["col_energy"] == "40 eV"
    assert pd.isna(single["col_energy_extra_1"])
    assert pd.isna(single["col_energy_extra_2"])

    two_step = spec_df.iloc[2]
    assert [two_step["col_energy"], two_step["col_energy_extra_1"]] == ["20 eV", "60 eV"]
    assert pd.isna(two_step["col_energy_extra_2"])
