"""Tests for the rare chemistry A1/A3 molecule-acquisition sampler."""

import numpy as np
import pandas as pd
import pytest
import torch as th

from fragnnet.dataset.data_sampler import MoleculeAcquisitionSampler
from fragnnet.dataset.rarity_sampling import (
    CLASS_PANELS,
    RAW_CE_COLS,
    acquisition_keys,
    class_rarity_weights,
    match_class_panel,
    mixture_probabilities,
    molecule_keys,
)

KNOWN_MOLECULES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "testosterone": "CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O",
    "nitrobenzene": "O=[N+]([O-])c1ccccc1",
    "pyridine_n_oxide": "[O-][n+]1ccccc1",
    "aniline": "Nc1ccccc1",
    "acetanilide": "CC(=O)Nc1ccccc1",
    "triethylamine": "CCN(CC)CC",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "alliin_charge_separated": "C=CC[S+]([O-])CC(N)C(=O)O",
    "dmso": "CS(C)=O",
    "trimethyl_phosphate": "COP(=O)(OC)OC",
    "phosphine_oxide_charge_separated": "C[P+](C)(C)[O-]",
}


@pytest.fixture(scope="module")
def known_classes() -> pd.DataFrame:
    """Panel matches for a few molecules with unambiguous functional groups."""
    return match_class_panel(pd.Series(KNOWN_MOLECULES), "functional_groups_v1")


def _sampler(mol_codes, acq_codes, mol_probs, num_samples=20000, seed=0):
    return MoleculeAcquisitionSampler(
        mol_codes=np.asarray(mol_codes),
        acq_codes=np.asarray(acq_codes),
        mol_probs=np.asarray(mol_probs, dtype=np.float64),
        num_samples=num_samples,
        generator=th.Generator().manual_seed(seed),
    )


class TestClassPanel:
    def test_panel_names_unique(self):
        for panel in CLASS_PANELS.values():
            names = [name for name, _ in panel]
            assert len(names) == len(set(names))

    @pytest.mark.parametrize(
        ("molecule", "present", "absent"),
        [
            ("aspirin", ["carboxylic_acid", "carboxylic_ester"], ["ketone", "phenol"]),
            ("testosterone", ["steroid_core", "ketone", "aliphatic_alcohol"], ["phenol"]),
            ("nitrobenzene", ["nitro"], ["n_oxide", "aromatic_amine"]),
            ("pyridine_n_oxide", ["n_oxide"], ["nitro", "pyridine_like_n"]),
            ("aniline", ["aromatic_amine"], ["substituted_amide"]),
            ("acetanilide", ["substituted_amide"], ["aromatic_amine", "primary_amide"]),
            ("triethylamine", ["tertiary_aliphatic_amine"], ["secondary_aliphatic_amine"]),
            ("caffeine", ["purine", "imidazole"], ["indole"]),
            ("alliin_charge_separated", ["sulfoxide", "carboxylic_acid"], ["sulfone", "n_oxide"]),
            ("dmso", ["sulfoxide"], ["sulfone"]),
            ("trimethyl_phosphate", ["phosphoryl"], ["trivalent_phosphorus"]),
            ("phosphine_oxide_charge_separated", ["phosphoryl"], ["trivalent_phosphorus"]),
        ],
    )
    def test_known_molecules(self, known_classes, molecule, present, absent):
        row = known_classes.loc[molecule]
        assert row[present].all(), row[present]
        assert not row[absent].any(), row[absent]

    def test_unknown_panel_raises(self):
        with pytest.raises(ValueError, match="unknown class panel"):
            match_class_panel(pd.Series({"a": "CCO"}), "functional_groups_v0")

    def test_bad_smiles_raises(self):
        with pytest.raises(ValueError, match="cannot parse SMILES"):
            match_class_panel(pd.Series({"a": "C1CC"}), "functional_groups_v1")


class TestWeights:
    def test_rarest_class_sets_weight_without_summing(self):
        # Class sizes: rare1 = 1, rare5 = 5, common = 20 molecules.
        rows = {
            "m0": [True, False, True],  # rare1 + common -> cap 4
            "m1": [False, True, True],  # rare5 + common -> sqrt(20/5) = 2
            **{f"c{i}": [False, i < 4, True] for i in range(18)},
        }
        matrix = pd.DataFrame.from_dict(rows, orient="index", columns=["rare1", "rare5", "common"])
        matrix.loc["none"] = [False, False, False]
        assert matrix.sum().tolist() == [1, 5, 20]
        weights = class_rarity_weights(matrix, ref_count=20, cap=4.0)
        assert weights["m0"] == pytest.approx(4.0)
        assert weights["m1"] == pytest.approx(2.0)
        assert weights["c0"] == pytest.approx(2.0)
        assert weights["c10"] == pytest.approx(1.0)
        assert weights["none"] == pytest.approx(1.0)

    def test_mixture_probabilities(self):
        probs = mixture_probabilities(np.array([1.0, 1.0, 2.0]), uniform_mix=0.5)
        assert probs.sum() == pytest.approx(1.0)
        assert probs.tolist() == pytest.approx([0.5 / 3 + 0.125, 0.5 / 3 + 0.125, 0.5 / 3 + 0.25])
        assert mixture_probabilities(np.array([1.0, 4.0]), uniform_mix=1.0).tolist() == [0.5, 0.5]


class TestIdentity:
    def test_molecule_keys_and_missing_key(self):
        mol_df = pd.DataFrame({"inchikey_s": ["AAA", ""]}, index=[10, 11])
        spec_df = pd.DataFrame({"mol_id": [10, 10]})
        assert molecule_keys(spec_df, mol_df).tolist() == ["AAA", "AAA"]
        with pytest.raises(ValueError, match="no inchikey_s"):
            molecule_keys(pd.DataFrame({"mol_id": [10, 11]}), mol_df)

    def test_acquisition_keys_keep_recorded_ce(self):
        spec_df = pd.DataFrame(
            {
                "spec_id": [1, 2, 3, 4, 5],
                "prec_type": ["[M+H]+"] * 5,
                "inst_type": ["ORB", "ORB", "ORB", "ORB_F", "ORB"],
                "frag_mode": ["HCD"] * 5,
            }
        )
        raw = pd.DataFrame(
            {
                "ce_type": ["single"] * 5,
                "nce": [35.0, np.nan, 35.0, 35.0, 20.0],
                "nce_extra_1": [np.nan, np.nan, np.nan, np.nan, 30.0],
                "nce_extra_2": [np.nan, np.nan, np.nan, np.nan, 40.0],
                "ace": [np.nan, 35.0, 35.0, np.nan, np.nan],
                "ace_extra_1": [np.nan] * 5,
                "ace_extra_2": [np.nan] * 5,
            },
            index=pd.Index([1, 2, 3, 4, 5], name="spec_id"),
        )[list(RAW_CE_COLS)]
        keys = acquisition_keys(spec_df, raw)
        # NCE 35 alone, ACE 35 alone, both recorded, other instrument: all distinct.
        assert len(set(keys)) == 5
        assert keys[0] == "[M+H]+|ORB|HCD|single|nce=35.0,NA,NA|ace=NA,NA,NA"
        assert keys[4].endswith("nce=20.0,30.0,40.0|ace=NA,NA,NA")
        with pytest.raises(ValueError, match="no raw CE row"):
            acquisition_keys(spec_df, raw.drop(index=5))


class TestMoleculeAcquisitionSampler:
    def test_length_range_and_reproducibility(self):
        sampler = _sampler([0, 0, 1], [0, 1, 2], [0.5, 0.5], num_samples=37)
        first = list(sampler)
        assert len(sampler) == 37 and len(first) == 37
        assert set(first) <= {0, 1, 2}
        sampler.generator = th.Generator().manual_seed(0)
        assert list(sampler) == first
        sampler.generator = th.Generator().manual_seed(1)
        assert list(sampler) != first

    def test_uniform_molecules_ignore_spectrum_count(self):
        # Molecule 0 has 100 spectra under one key; molecule 1 has a single spectrum.
        mol_codes = [0] * 100 + [1]
        acq_codes = [0] * 100 + [1]
        draws = np.array(list(_sampler(mol_codes, acq_codes, [0.5, 0.5])))
        assert (draws == 100).mean() == pytest.approx(0.5, abs=0.02)

    def test_keys_uniform_within_molecule(self):
        # One molecule: key 0 has spectrum 0; key 1 has spectra 1..9.
        draws = np.array(list(_sampler([0] * 10, [0] + [1] * 9, [1.0])))
        assert (draws == 0).mean() == pytest.approx(0.5, abs=0.02)
        assert np.bincount(draws[draws > 0], minlength=10)[1:].min() > 0

    def test_molecule_probabilities_respected(self):
        draws = np.array(list(_sampler([0, 1, 2], [0, 1, 2], [0.1, 0.3, 0.6], 40000)))
        assert np.bincount(draws, minlength=3) / len(draws) == pytest.approx(
            [0.1, 0.3, 0.6], abs=0.01
        )

    def test_key_spanning_molecules_raises(self):
        with pytest.raises(ValueError, match="more than one molecule"):
            _sampler([0, 1], [0, 0], [0.5, 0.5])

    def test_bad_probabilities_raise(self):
        with pytest.raises(ValueError, match="sum to one"):
            _sampler([0, 1], [0, 1], [0.5, 0.6])
        with pytest.raises(ValueError, match="one positive probability"):
            _sampler([0, 1], [0, 1], [1.0])
