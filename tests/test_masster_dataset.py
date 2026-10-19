"""Filter incompatible MASSter molecule/adduct pairs before batching."""

import pandas as pd
import pytest
from rdkit import Chem

from fragnnet.dataset import SpecMolDataset
from fragnnet.models.masster.masster_dataset_filter import (
    filter_masster_spectra,
    masster_has_valid_ion,
)


@pytest.mark.parametrize(
    ("smiles", "adduct", "include_cmf", "expected"),
    [
        ("FC(F)(F)F", "[M-H-H2O]-", True, False),
        ("CC", "[M-H2O+H]+", False, False),
        ("CC", "[M-H2O+H]+", True, False),
        ("CCO", "[M-H2O+H]+", False, True),
        ("C.[O]", "[M-H2O+H]+", False, False),
        ("[H+]", "[M-H]-", False, True),
    ],
)
def test_precursor_chemical_support_filter(
    smiles: str,
    adduct: str,
    include_cmf: bool,
    expected: bool,
) -> None:
    assert (
        masster_has_valid_ion(
            Chem.MolFromSmiles(smiles),
            adduct,
            0,
            include_cmf,
        )
        is expected
    )


@pytest.fixture
def spectrum_dataset() -> SpecMolDataset:
    dataset = SpecMolDataset.__new__(SpecMolDataset)
    dataset.spec_params = {"merge": False, "remove_isotope_peaks": False}
    dataset.spec_df = pd.DataFrame(
        {
            "spec_id": [10, 11, 12],
            "group_id": [10, 11, 12],
            "mol_id": [1, 1, 2],
            "prec_type": ["[M-H2O+H]+", "[M+H]+", "[M-H-H2O]-"],
        }
    )
    dataset.mol_df = pd.DataFrame(
        {
            "mol_id": [1, 2],
            "mol": [Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("FC(F)(F)F")],
        }
    ).set_index("mol_id", drop=False)
    dataset.um_spec_df = dataset.spec_df.copy()
    dataset.split_df = dataset.spec_df[["spec_id", "group_id", "mol_id"]].copy()
    return dataset


def test_dataset_filters_pairs_and_recomputes_counts(
    spectrum_dataset: SpecMolDataset,
    caplog: pytest.LogCaptureFixture,
) -> None:
    filter_masster_spectra(spectrum_dataset, max_h_transfer=0, include_cmf=False)

    assert spectrum_dataset.spec_df["spec_id"].tolist() == [11]
    assert spectrum_dataset.spec_df.index.tolist() == [0]
    assert spectrum_dataset.um_spec_df["spec_id"].tolist() == [11]
    assert spectrum_dataset.split_df["spec_id"].tolist() == [11]
    assert spectrum_dataset.mol_df["mol_id"].tolist() == [1]
    assert spectrum_dataset.spec_per_mol == {1: 1}
    assert spectrum_dataset.spec_per_group == {11: 1}
    assert "filtered 2/3 spectra" in caplog.text


def test_all_invalid_data_fails_during_dataset_setup(spectrum_dataset: SpecMolDataset) -> None:
    spectrum_dataset.spec_df = spectrum_dataset.spec_df.iloc[[0, 2]]
    with pytest.raises(ValueError, match="dataset has no chemically valid spectra"):
        filter_masster_spectra(spectrum_dataset, max_h_transfer=0, include_cmf=False)


def test_dataset_filter_runs_before_preprocessing(
    spectrum_dataset: SpecMolDataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def base_init(self, **kwargs):
        self.__dict__.update(spectrum_dataset.__dict__)

    def preprocess(self, cache):
        assert self.spec_df["spec_id"].tolist() == [11]

    monkeypatch.setattr(SpecMolDataset, "_base_init", base_init)
    monkeypatch.setattr(SpecMolDataset, "_preprocess_spec", preprocess)
    monkeypatch.setattr(SpecMolDataset, "_preprocess_mol", preprocess)
    SpecMolDataset(
        spec_fp="",
        mol_fp="",
        split_dp="",
        split="val",
        subsample_params={},
        spec_params={},
        mol_params={},
        model_type="masster",
        masster_max_h_transfer=0,
        masster_include_cmf=False,
        masster_h_transfer_bound="none",
        masster_smarts_templates=False,
        masster_rewire_rules=False,
    )


@pytest.mark.parametrize(
    ("smarts", "rewire", "expected"), [(False, False, False), (True, False, True), (False, True, True)]
)
def test_dataset_emits_the_graph_molecule_for_chemistry_matching(
    monkeypatch: pytest.MonkeyPatch, smarts: bool, rewire: bool, expected: bool
) -> None:
    monkeypatch.setattr(SpecMolDataset, "_base_init", lambda self, **kwargs: None)
    monkeypatch.setattr(SpecMolDataset, "_preprocess_spec", lambda self, cache: None)
    monkeypatch.setattr(SpecMolDataset, "_preprocess_mol", lambda self, cache: None)
    monkeypatch.setattr(
        "fragnnet.models.masster.masster_hooks.MassterHooks.filter_dataset",
        lambda self, dataset, config: None,
    )
    dataset = SpecMolDataset(
        spec_fp="",
        mol_fp="",
        split_dp="",
        split="val",
        subsample_params={},
        spec_params={},
        mol_params={"mf": False, "smiles": True, "fingerprint": False, "pyg": False},
        model_type="masster",
        masster_smarts_templates=smarts,
        masster_rewire_rules=rewire,
    )
    assert dataset.emit_graph_mol is expected
    # A molecule whose atoms are not in the order its SMILES parses to (a retrieval candidate).
    smiles = "CCO.c1ccccc1"
    parsed = Chem.MolFromSmiles(smiles)
    mol = Chem.RenumberAtoms(parsed, list(reversed(range(parsed.GetNumAtoms()))))
    mol_data = dataset._process_mol({"mol": mol, "smiles": smiles})
    assert mol_data["mol_smiles"] == [smiles]
    if not expected:
        assert "mol_graph_binary" not in mol_data
        return
    (binary,) = mol_data["mol_graph_binary"]
    restored = Chem.Mol(binary)
    assert [a.GetAtomicNum() for a in restored.GetAtoms()] == [a.GetAtomicNum() for a in mol.GetAtoms()]
    assert [a.GetAtomicNum() for a in restored.GetAtoms()] != [a.GetAtomicNum() for a in parsed.GetAtoms()]
