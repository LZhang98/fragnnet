"""Orbitrap instrument labels: OB (Elite, Velos, unnamed) and OB_F (Orbitrap Fusion); FT retired.

Tests validate:
- parse_inst_info maps NIST instrument models and unnamed Orbitrap strings to OB / OB_F
- the dataset migration labels FT rows from the instrument field, a reference dataset and the
  NIST eV-record rule, in that order, and recomputes group_id
- configs and spec_dfs that still name FT are refused
"""

import importlib.util
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
    # Molecule 1: an Elite and a Lumos spectrum named by the instrument field.
    # Molecule 2: matched to a reference. Molecule 3: NIST eV rule. Row 7: contributed, default.
    # m/z 500 makes the conversion equal to the NCE for [M+H]+.
    return pd.DataFrame(
        {
            "spec_id": range(9),
            "dset_spec_id": ["10", "11", "20", "21", "30", "31", "32", "MoNA1", "40"],
            "mol_id": [1, 1, 2, 2, 3, 3, 3, 3, 4],
            "prec_type": ["[M+H]+"] * 9,
            "inst_type": ["FT"] * 8 + ["QTOF"],
            "frag_mode": ["HCD"] * 6 + ["CID", "HCD", "CID"],
            "prec_mz": [500.0] * 9,
            "nce": [20.0, 20.0, 35.0, 44.0, 20.0, 20.0, 20.0, 20.0, np.nan],
            "ace": [20.0, 30.0, 35.0, np.nan, 20.0, np.nan, 20.0, np.nan, 15.0],
            "group_id": [0, 0, 1, 1, 2, 2, 2, 2, 3],
        }
    )


def test_relabel_uses_each_evidence_source(ft_spec_df):
    mig = _load_migration()
    inst = pd.Series({"10": "Thermo Finnigan Elite Orbitrap", "11": "Orbitrap Fusion Lumos"})
    keys = pd.Series(["a", "b", "k20", "k21", "c", "d", "e", "f", "g"], index=ft_spec_df.index)
    ref = pd.Series({"k20": "OB", "k21": "OB_F"})
    inst_type, _, source = mig.relabel_inst_type(ft_spec_df, inst, ref, keys, nist_ev_rule=True)
    assert inst_type.tolist() == [
        "OB",
        "OB_F",
        "OB",
        "OB_F",
        "OB",
        "OB_F",
        "OB",
        "OB",
        "QTOF",
    ]
    assert source.tolist() == [
        "inst",
        "inst",
        "reference",
        "reference",
        "ev_rule",
        "ev_rule",
        "default",
        "default",
    ]
    # The CID row (index 6) and the contributed row (index 7) are not eligible for the eV rule.
    assert source.loc[6] == "default" and source.loc[7] == "default"


def test_relabel_splits_groups_by_instrument(ft_spec_df):
    mig = _load_migration()
    inst = pd.Series({"10": "Thermo Finnigan Elite Orbitrap", "11": "Orbitrap Fusion Lumos"})
    _, group_id, _ = mig.relabel_inst_type(ft_spec_df, inst)
    # Old group 0 holds an Elite and a Lumos spectrum: OB keeps id 0, OB_F gets max + 1 = 4.
    assert group_id[0] == 0 and group_id[1] == 4
    # Groups that stay whole keep their ids, including the QTOF one.
    assert group_id[2:].tolist() == [1, 1, 2, 2, 2, 2, 3]
    assert (group_id.index == ft_spec_df.index).all()
    assert (ft_spec_df["inst_type"] == "FT").sum() == 8  # the input is not modified


def test_relabel_keeps_groups_finer_than_the_current_key(ft_spec_df):
    # Stored groups may split one (mol, adduct, instrument) trio further; they must survive.
    df = ft_spec_df.assign(group_id=[0, 0, 1, 5, 2, 2, 2, 2, 3])
    _, group_id, _ = _load_migration().relabel_inst_type(df)
    assert group_id.tolist() == [0, 0, 1, 5, 2, 2, 2, 2, 3]


def test_relabel_leaves_ev_rule_off_by_default(ft_spec_df):
    inst_type, _, source = _load_migration().relabel_inst_type(ft_spec_df)
    assert set(inst_type[source.index]) == {"OB"}
    assert set(source) == {"default"}


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


def test_match_keys_use_ev_when_recorded():
    mig = _load_migration()
    mol_df = pd.DataFrame({"mol_id": [1, 2], "inchikey": ["AAAAAAAAAAAAAA-X", "BBBBBBBBBBBBBB-Y"]})
    spec = pd.DataFrame(
        {
            "mol_id": [1, 1, 2],
            "prec_type": ["[M-H]-", "[M-H]-", "[M+H]+"],
            "prec_mz": [429.9693, 429.9693, 300.004],
            "nce": [15.0, 30.0, 44.0],
            "ace": [15.0, 15.0, np.nan],
        }
    )
    keys = mig.match_keys(spec, mol_df)
    # The same spectrum under NIST20 (NCE 15) and NIST23 (NCE doubled to 30) gets one key.
    assert keys[0] == keys[1] == "AAAAAAAAAAAAAA|[M-H]-|429.97|eV15.0"
    assert keys[2] == "BBBBBBBBBBBBBB|[M+H]+|300.0|NCE44.0"


def test_ev_rule_skips_negative_mode(ft_spec_df):
    df = ft_spec_df.assign(prec_type="[M-H]-")
    _, _, source = _load_migration().relabel_inst_type(df, nist_ev_rule=True)
    assert set(source) == {"default"}
