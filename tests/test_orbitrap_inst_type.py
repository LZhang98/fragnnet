"""Orbitrap instrument labels: ORB (Elite, Velos, unnamed) and ORB_F (Orbitrap Fusion).

FT was retired on 2026-09-28; OB / OB_F were renamed ORB / ORB_F on 2026-10-01.

Tests validate:
- parse_inst_info maps NIST instrument models and unnamed Orbitrap strings to ORB / ORB_F
- the dataset migration labels FT rows from raw instrument names (data/df tables or raw MSPs,
  first source wins) and splits group_id by instrument without regrouping
- the rename migration changes only OB / OB_F, in a spec_df's inst_type and a checkpoint's hparams
- configs, checkpoint hparams and spec_dfs that still name FT, OB or OB_F are refused
"""

import importlib.util
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch as th

from fragnnet.config import load_config
from fragnnet.dataset.base_dataset import BaseDataset
from fragnnet.utils.migrations import find_retired_inst_type_refs, get_pl_hparams
from fragnnet.utils.spec_parse_utils import parse_inst_info


def _load_script(name: str):
    path = Path(__file__).resolve().parent.parent / "preproc_scripts" / "dataset_migration"
    spec = importlib.util.spec_from_file_location(name, path / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_migration():
    return _load_script("relabel_orbitrap_inst_type")


@pytest.mark.parametrize(
    ("inst_type", "inst", "expected"),
    [
        ("HCD", "Thermo Finnigan Elite Orbitrap", "ORB"),
        ("HCD", "Thermo Finnigan Velos Orbitrap", "ORB"),
        ("HCD", "Orbitrap Fusion Lumos", "ORB_F"),
        ("IT-FT/ion trap with FTMS", "Orbitrap Fusion Lumos", "ORB_F"),
        ("ESI-QFT", np.nan, "ORB"),
        ("FT", np.nan, "ORB"),
        ("ORB_F", np.nan, "ORB_F"),
        (np.nan, "Orbitrap", "ORB"),
        ("FTICR (LCMS)", np.nan, "Other"),
        ("Q-TOF", "Agilent QTOF 6530", "QTOF"),
        # Spectraverse INSTRUMENT_TYPE values
        ("orbitrap", np.nan, "ORB"),
        ("qtof", np.nan, "QTOF"),
        ("iontrap", np.nan, "IT"),
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
    assert inst_type.tolist() == ["ORB", "ORB_F", "ORB", "ORB", "ORB", "QTOF"]
    assert source.tolist() == ["inst"] * 4 + ["default"]


def test_relabel_splits_groups_by_instrument(ft_spec_df):
    _, group_id, _ = _load_migration().relabel_inst_type(ft_spec_df, INST)
    # Old group 0 holds an Elite and a Lumos spectrum: ORB keeps id 0, ORB_F gets max + 1 = 4.
    assert group_id.tolist() == [0, 4, 1, 1, 2, 3]
    assert (group_id.index == ft_spec_df.index).all()
    assert (ft_spec_df["inst_type"] == "FT").sum() == 5  # the input is not modified


def test_relabel_keeps_groups_finer_than_the_current_key(ft_spec_df):
    # Stored groups may split one (mol, adduct, instrument) trio further; they must survive.
    df = ft_spec_df.assign(group_id=[0, 5, 1, 6, 2, 3])
    _, group_id, _ = _load_migration().relabel_inst_type(df, INST)
    assert group_id.tolist() == [0, 5, 1, 6, 2, 3]


def test_relabel_without_names_is_all_orb(ft_spec_df):
    inst_type, group_id, source = _load_migration().relabel_inst_type(ft_spec_df)
    assert inst_type.tolist() == ["ORB"] * 5 + ["QTOF"]
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
    assert out["inst_type"].tolist() == ["ORB", "ORB_F", "ORB_F", "ORB", "ORB", "QTOF"]
    backups = list(tmp_path.glob("spec_df_orig_*.pkl"))
    assert len(backups) == 1 and pd.read_pickle(backups[0]).equals(ft_spec_df)


@pytest.mark.parametrize("label", ["OB", "OB_F"])
def test_parse_inst_info_refuses_renamed_labels(label):
    row = pd.Series({"inst_type": label, "inst": np.nan, "frag_mode": np.nan, "col_energy": "NCE=20%"})
    with pytest.raises(ValueError, match=f"inst_type '{label}' is retired"):
        parse_inst_info(row)


def test_find_retired_inst_type_refs():
    cfg = {
        "spec_params": {
            "inst_types": ["ORB", "FT", "OB_F"],
            "merge_tolerance_by_inst": {"FT": {"rel": 1e-5}, "OB": {"rel": 1e-5}},
        },
        "ce_context_ce_key_by_inst": {"ORB": "nce"},
        "eval_inst_type": "OB",
    }
    assert find_retired_inst_type_refs(cfg) == [
        "spec_params.inst_types[1]",
        "spec_params.inst_types[2]",
        "spec_params.merge_tolerance_by_inst.FT",
        "spec_params.merge_tolerance_by_inst.OB",
        "eval_inst_type",
    ]
    assert find_retired_inst_type_refs({"spec_params": {"inst_types": ["ORB", "ORB_F"]}}) == []


@pytest.mark.parametrize("label", ["FT", "OB", "OB_F"])
def test_load_config_refuses_retired_labels(tmp_path, label):
    custom = tmp_path / "custom.yml"
    custom.write_text(f'model_type: masster\nspec_params:\n  inst_types: ["{label}"]\n')
    with pytest.raises(ValueError, match=r"retired instrument type at \['spec_params.inst_types\[0\]'\]"):
        load_config("config", str(custom), [])


def test_template_resolves_without_retired_labels():
    cfg = load_config("config", None, [], model_type="masster")
    assert find_retired_inst_type_refs(cfg) == []
    assert cfg["spec_params"]["inst_types"] == ["ORB", "ORB_F"]


def test_dataset_refuses_ft_spec_df(ft_spec_df):
    mol_df = pd.DataFrame({"mol_id": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match=r"retired instrument types \['FT'\]"):
        BaseDataset._setup_dfs(ft_spec_df, mol_df, "unused", ["train"], {}, {})


def test_dataset_refuses_ob_spec_df(ob_spec_df):
    mol_df = pd.DataFrame({"mol_id": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match=r"retired instrument types \['OB', 'OB_F'\]"):
        BaseDataset._setup_dfs(ob_spec_df, mol_df, "unused", ["train"], {}, {})


def test_dataset_refuses_retired_inst_types_in_spec_params():
    # A dataset built from an unmigrated checkpoint's hparams would filter out every ORB row.
    ds = SimpleNamespace(spec_params={"inst_types": ["OB", "OB_F", "QTOF"]})
    with pytest.raises(ValueError, match=r"retired types \['OB', 'OB_F'\]"):
        BaseDataset._setup_inst_type_to_idx(ds)


def test_inst_type_indices_survive_the_rename():
    # Embedding rows follow the sorted labels (_setup_inst_type_to_idx), so old weights stay valid
    # only if the rename keeps every label's sorted position.
    others = ["EI", "IT", "Other", "QQQ", "QTOF"]
    old = sorted(others + ["OB", "OB_F"])
    new = sorted(others + ["ORB", "ORB_F"])
    assert [{"OB": "ORB", "OB_F": "ORB_F"}.get(t, t) for t in old] == new


@pytest.fixture
def ob_spec_df():
    return pd.DataFrame(
        {
            "spec_id": range(4),
            "mol_id": [1, 2, 3, 4],
            "inst_type": ["OB", "OB_F", "QTOF", "OB"],
            "group_id": [0, 1, 2, 3],
            "peaks": [[(50.0, 1.0)]] * 4,
        }
    )


def test_rename_spec_df_changes_only_the_labels(tmp_path, ob_spec_df):
    spec_df_fp = tmp_path / "spec_df.pkl"
    ob_spec_df.to_pickle(spec_df_fp)
    _load_script("rename_orbitrap_inst_type").rename_spec_df(str(spec_df_fp))
    out = pd.read_pickle(spec_df_fp)
    assert out["inst_type"].tolist() == ["ORB", "ORB_F", "QTOF", "ORB"]
    assert out.drop(columns="inst_type").equals(ob_spec_df.drop(columns="inst_type"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["spec_df.pkl"]


def test_rename_spec_df_refuses_ft(tmp_path, ft_spec_df):
    spec_df_fp = tmp_path / "spec_df.pkl"
    ft_spec_df.to_pickle(spec_df_fp)
    with pytest.raises(ValueError, match="relabel_orbitrap_inst_type.py"):
        _load_script("rename_orbitrap_inst_type").rename_spec_df(str(spec_df_fp))
    assert pd.read_pickle(spec_df_fp).equals(ft_spec_df)


def _ob_hparams():
    return {
        "model_type": "masster",
        "spec_params": {
            "inst_types": ["OB", "OB_F"],
            "merge_tolerance_by_inst": {"OB_F": {"rel": 1e-5}, "QTOF": {"rel": 2e-5}, "OB": {"rel": 1e-5}},
        },
        "ce_context_ce_key_by_inst": {"OB": "nce", "OB_F": "nce", "QTOF": "ace"},
        "precursor_pair_eligible_inst_types": ["OB", "OB_F"],
        "job_name": "i1_inst_ob_obf",
    }


def test_rename_ckpt_renames_hparams_and_keeps_weights(tmp_path):
    ckpt_fp = tmp_path / "model.ckpt"
    weight = th.arange(6.0).reshape(2, 3)
    th.save({"hyper_parameters": _ob_hparams(), "state_dict": {"inst_emb.weight": weight}}, ckpt_fp)
    with pytest.raises(ValueError, match=r"checkpoint hparams name a retired instrument type"):
        get_pl_hparams(th.load(ckpt_fp, weights_only=False))

    _load_script("rename_orbitrap_inst_type").rename_ckpt(str(ckpt_fp))
    ckpt = th.load(ckpt_fp, weights_only=False)
    hparams = ckpt["hyper_parameters"]
    assert hparams["spec_params"]["inst_types"] == ["ORB", "ORB_F"]
    assert hparams["spec_params"]["merge_tolerance_by_inst"] == {
        "ORB_F": {"rel": 1e-5},
        "QTOF": {"rel": 2e-5},
        "ORB": {"rel": 1e-5},
    }
    assert hparams["ce_context_ce_key_by_inst"] == {"ORB": "nce", "ORB_F": "nce", "QTOF": "ace"}
    assert hparams["precursor_pair_eligible_inst_types"] == ["ORB", "ORB_F"]
    assert hparams["job_name"] == "i1_inst_ob_obf"
    assert th.equal(ckpt["state_dict"]["inst_emb.weight"], weight)
    assert get_pl_hparams(ckpt)["spec_params"]["inst_types"] == ["ORB", "ORB_F"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.ckpt"]


def test_rename_ckpt_refuses_ft(tmp_path):
    ckpt_fp = tmp_path / "model.ckpt"
    hparams = {"spec_params": {"inst_types": ["FT"]}, "inst_type_loss_tol": {"FT": {}, "OB": {}}}
    th.save({"hyper_parameters": hparams}, ckpt_fp)
    with pytest.raises(ValueError, match="trained on 'FT'"):
        _load_script("rename_orbitrap_inst_type").rename_ckpt(str(ckpt_fp))
    assert th.load(ckpt_fp, weights_only=False)["hyper_parameters"] == hparams
