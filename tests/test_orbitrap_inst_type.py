"""Orbitrap instrument labels: OB (Elite, Velos, unnamed) and OB_F (Orbitrap Fusion); FT retired.

Tests validate:
- parse_inst_info maps NIST instrument models and unnamed Orbitrap strings to OB / OB_F
- the dataset migration labels FT rows from raw instrument names (data/df tables or raw MSPs,
  first source wins) and splits group_id by instrument without regrouping
- configs and spec_dfs that still name FT are refused
"""

import importlib.util
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fragnnet.dataset.base_dataset import BaseDataset
from fragnnet.runner import find_retired_inst_type_refs, load_config
from fragnnet.utils.data_utils import parse_inst_info


def _load_migration():
    path = Path(__file__).resolve().parent.parent / "preproc_scripts" / "dataset_migration"
    spec = importlib.util.spec_from_file_location("relabel", path / "relabel_orbitrap_inst_type.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("inst_type", "inst", "expected"),
    [
        ("HCD", "Thermo Finnigan Elite Orbitrap", "OB"),
        ("HCD", "Thermo Finnigan Velos Orbitrap", "OB"),
        ("HCD", "Orbitrap Fusion Lumos", "OB_F"),
        ("IT-FT/ion trap with FTMS", "Orbitrap Fusion Lumos", "OB_F"),
        ("ESI-QFT", np.nan, "OB"),
        ("FT", np.nan, "OB"),
        ("OB_F", np.nan, "OB_F"),
        (np.nan, "Orbitrap", "OB"),
        ("FTICR (LCMS)", np.nan, "Other"),
        ("Q-TOF", "Agilent QTOF 6530", "QTOF"),
    ],
)
def test_parse_inst_info_labels_orbitraps(inst_type, inst, expected):
    row = pd.Series(
        {"inst_type": inst_type, "inst": inst, "frag_mode": np.nan, "col_energy": "NCE=20%"}
    )
    assert parse_inst_info(row)[0] == expected


@pytest.fixture
def ft_spec_df():
    # Molecule 1: an Elite and a Lumos spectrum, split out of one old group. Molecule 2: named by
    # the second source only. Row 4: contributed, no name. Row 5: QTOF, untouched.
    return pd.DataFrame(
        {
            "spec_id": range(6),
            "dset_spec_id": ["10", "11", "20", "21", "MoNA1", "40"],
            "mol_id": [1, 1, 2, 2, 3, 4],
            "inst_type": ["FT"] * 5 + ["QTOF"],
            "group_id": [0, 0, 1, 1, 2, 3],
        }
    )


INST = pd.Series(
    {
        "10": "Thermo Finnigan Elite Orbitrap",
        "11": "Orbitrap Fusion Lumos",
        "20": "Thermo Finnigan Velos Orbitrap",
        "21": "Thermo Finnigan Velos Orbitrap",
    }
)


def test_relabel_uses_the_instrument_name(ft_spec_df):
    inst_type, _, source = _load_migration().relabel_inst_type(ft_spec_df, INST)
    assert inst_type.tolist() == ["OB", "OB_F", "OB", "OB", "OB", "QTOF"]
    assert source.tolist() == ["inst"] * 4 + ["default"]


def test_relabel_splits_groups_by_instrument(ft_spec_df):
    _, group_id, _ = _load_migration().relabel_inst_type(ft_spec_df, INST)
    # Old group 0 holds an Elite and a Lumos spectrum: OB keeps id 0, OB_F gets max + 1 = 4.
    assert group_id.tolist() == [0, 4, 1, 1, 2, 3]
    assert (group_id.index == ft_spec_df.index).all()
    assert (ft_spec_df["inst_type"] == "FT").sum() == 5  # the input is not modified


def test_relabel_keeps_groups_finer_than_the_current_key(ft_spec_df):
    # Stored groups may split one (mol, adduct, instrument) trio further; they must survive.
    df = ft_spec_df.assign(group_id=[0, 5, 1, 6, 2, 3])
    _, group_id, _ = _load_migration().relabel_inst_type(df, INST)
    assert group_id.tolist() == [0, 5, 1, 6, 2, 3]


def test_relabel_without_names_is_all_ob(ft_spec_df):
    inst_type, group_id, source = _load_migration().relabel_inst_type(ft_spec_df)
    assert inst_type.tolist() == ["OB"] * 5 + ["QTOF"]
    assert set(source) == {"default"}
    assert group_id.tolist() == ft_spec_df["group_id"].tolist()


MSP = """Name: A
Instrument_type: HCD
Instrument: Thermo Finnigan Elite Orbitrap
CAS#: 50-00-0;  NIST#: 10
Num Peaks: 1
50.0 100 "Instrument: not a field"

Name: B
Instrument: Orbitrap Fusion Lumos
CAS#: 50-00-1;  NIST#: 11
Num Peaks: 1
60.0 100

Name: B again
Instrument: Orbitrap Fusion Lumos
CAS#: 50-00-1;  NIST#: 11
Num Peaks: 0

Name: no NIST number
Instrument: Orbitrap Fusion Lumos
Num Peaks: 0
"""


@pytest.mark.parametrize("suffix", [".msp", ".zip"])
def test_read_msp_instruments(tmp_path, suffix):
    msp_fp = tmp_path / "lib.msp"
    msp_fp.write_text(MSP)
    if suffix == ".zip":
        with zipfile.ZipFile(tmp_path / "lib.zip", "w") as zf:
            zf.write(msp_fp, "lib.msp")
        msp_fp = tmp_path / "lib.zip"
    inst = _load_migration().read_instruments(str(msp_fp))
    assert inst.to_dict() == {"10": "Thermo Finnigan Elite Orbitrap", "11": "Orbitrap Fusion Lumos"}


def test_read_msp_instruments_refuses_conflicting_ids(tmp_path):
    msp_fp = tmp_path / "lib.msp"
    msp_fp.write_text(MSP.replace("Instrument: Orbitrap Fusion Lumos\nCAS#: 50-00-1;  NIST#: 11\nNum Peaks: 0",
                                  "Instrument: Thermo Finnigan Elite Orbitrap\nCAS#: 50-00-1;  NIST#: 11\nNum Peaks: 0"))
    with pytest.raises(ValueError, match="two instruments"):
        _load_migration().read_msp_instruments(str(msp_fp))


def test_main_takes_the_first_source_that_names_a_spectrum(tmp_path, ft_spec_df, monkeypatch):
    mig = _load_migration()
    spec_df_fp = tmp_path / "spec_df.pkl"
    ft_spec_df.to_pickle(spec_df_fp)
    msp_fp = tmp_path / "lib.msp"
    msp_fp.write_text(MSP)  # names 10 (Elite) and 11 (Lumos)
    table_fp = tmp_path / "later_release_df.csv"
    pd.DataFrame(
        {"dset_spec_id": ["11", "20"], "inst": ["Thermo Finnigan Elite Orbitrap", "Orbitrap Fusion Lumos"]}
    ).to_csv(table_fp, index=False)
    argv = ["relabel", "--spec_df_fp", str(spec_df_fp), "--inst_fp", str(msp_fp), str(table_fp)]
    monkeypatch.setattr("sys.argv", argv)
    mig.main()
    out = pd.read_pickle(spec_df_fp)
    # 11 keeps the MSP's Lumos over the table's Elite; 20 is named by the table; 21 and MoNA1 are unnamed.
    assert out["inst_type"].tolist() == ["OB", "OB_F", "OB_F", "OB", "OB", "QTOF"]
    backups = list(tmp_path.glob("spec_df_orig_*.pkl"))
    assert len(backups) == 1 and pd.read_pickle(backups[0]).equals(ft_spec_df)


def test_find_retired_inst_type_refs():
    cfg = {
        "spec_params": {
            "inst_types": ["OB", "FT"],
            "merge_tolerance_by_inst": {"FT": {"rel": 1e-5}},
        },
        "ce_context_ce_key_by_inst": {"OB": "nce"},
    }
    assert find_retired_inst_type_refs(cfg) == [
        "spec_params.inst_types[1]",
        "spec_params.merge_tolerance_by_inst.FT",
    ]
    assert find_retired_inst_type_refs({"spec_params": {"inst_types": ["OB", "OB_F"]}}) == []


def test_load_config_refuses_ft(tmp_path):
    custom = tmp_path / "custom.yml"
    custom.write_text('model_type: masster\nspec_params:\n  inst_types: ["FT"]\n')
    with pytest.raises(ValueError, match="retired instrument type 'FT'"):
        load_config("config", str(custom), [])


def test_template_resolves_without_ft():
    cfg = load_config("config", None, [], model_type="masster")
    assert find_retired_inst_type_refs(cfg) == []
    assert cfg["spec_params"]["inst_types"] == ["OB", "OB_F"]


def test_dataset_refuses_ft_spec_df(ft_spec_df):
    mol_df = pd.DataFrame({"mol_id": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="retired instrument type 'FT'"):
        BaseDataset._setup_dfs(ft_spec_df, mol_df, "unused", ["train"], {}, {})
