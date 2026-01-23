"""
Unit tests for inference preprocessing scripts.

Tests for:
- 01_prepare_mol_df.py (multiple input modes)
- 02_prepare_spec_df.py (spectrum dataframe creation)
- 03_run_inference_frag_gen.py (fragment generation)
"""

import importlib.util
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest


# Helper to import modules with numeric names
def _load_script_module(script_name):
    """Load a preprocessing script as a module."""
    script_path = Path(__file__).parent.parent / "preproc_scripts" / "inference" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPrepareMolDf:
    """Tests for 01_prepare_mol_df.py functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create and cleanup temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def test_smiles_list(self):
        """Test SMILES strings."""
        return ["CCO", "CC(=O)O", "CCN", "C", "CC"]

    @pytest.fixture
    def test_json_data(self):
        """Test JSON candidates data."""
        return {"CCO": ["CC", "C", "CCN"], "CC(=O)O": ["CCO", "CC", "C"]}

    def test_smiles_list_input(self, temp_dir, test_smiles_list):
        """Test SMILES list input mode."""
        mol_df_module = _load_script_module("01_prepare_mol_df.py")

        # Validate inputs
        mol_df_module.validate_inputs(test_smiles_list, max_heavy_atoms=128)

        # Create mol_df
        mol_df = mol_df_module.create_mol_df(test_smiles_list, ids_list=None, max_heavy_atoms=128)

        # Verify output
        assert len(mol_df) > 0
        assert "mol_id" in mol_df.columns
        assert "smiles" in mol_df.columns
        assert "mol" in mol_df.columns
        assert "formula" in mol_df.columns
        assert "inchikey_s" in mol_df.columns
        assert "exact_mw" in mol_df.columns
        assert mol_df["mol_id"].min() == 0
        assert all(mol_df["mol_id"] == range(len(mol_df)))

    def test_smiles_list_with_ids(self, temp_dir, test_smiles_list):
        """Test SMILES list with custom molecule IDs."""
        mol_df_module = _load_script_module("01_prepare_mol_df.py")

        ids = [f"mol_{i}" for i in range(len(test_smiles_list))]
        mol_df = mol_df_module.create_mol_df(test_smiles_list, ids_list=ids, max_heavy_atoms=128)

        assert len(mol_df) > 0
        assert "input_mol_id" in mol_df.columns
        assert mol_df["input_mol_id"].notna().all()


    def test_charged_molecules_filtered(self, temp_dir):
        """Test that charged molecules are filtered out."""
        mol_df_module = _load_script_module("01_prepare_mol_df.py")

        smiles_list = ["CCO", "[NH4+]", "CC(=O)O"]
        mol_df = mol_df_module.create_mol_df(smiles_list, ids_list=None, max_heavy_atoms=128)

        # Charged molecules should be filtered
        assert all(mol_df["charge"] == 0)

    def test_radical_molecules_filtered(self, temp_dir):
        """Test that molecules with radicals are filtered out."""
        mol_df_module = _load_script_module("01_prepare_mol_df.py")

        smiles_list = ["CCO", "[CH3]", "CC(=O)O"]
        mol_df = mol_df_module.create_mol_df(smiles_list, ids_list=None, max_heavy_atoms=128)

        # Radical molecules should be filtered
        assert all(mol_df["num_radicals"] == 0)

    def test_heavy_atom_limit(self, temp_dir):
        """Test that molecules exceeding heavy atom limit are filtered."""
        mol_df_module = _load_script_module("01_prepare_mol_df.py")

        # C60 has 60 heavy atoms
        smiles_list = ["CCO", "C1=CC=C2C=CC=CC2=C1"]  # naphthalene, 10 heavy atoms
        mol_df = mol_df_module.create_mol_df(smiles_list, ids_list=None, max_heavy_atoms=5)

        # Only CCO (3 heavy atoms) should remain
        assert len(mol_df) == 1
        assert mol_df.iloc[0]["formula"] == "C2H6O"

    def test_molecular_properties_computed(self, temp_dir):
        """Test that molecular properties are computed correctly."""
        mol_df_module = _load_script_module("01_prepare_mol_df.py")

        mol_df = mol_df_module.create_mol_df(["CCO"], ids_list=None, max_heavy_atoms=128)
        assert len(mol_df) == 1
        row = mol_df.iloc[0]
        assert row["formula"] == "C2H6O"
        assert 45 < row["exact_mw"] < 47
        assert row["num_heavy_atoms"] == 3
        assert row["charge"] == 0
        assert row["num_radicals"] == 0

    def test_json_input_mode(self, temp_dir, test_json_data):
        """Test JSON input mode with candidate molecules."""
        mol_df_module = _load_script_module("01_prepare_mol_df.py")

        # Extract unique SMILES from JSON
        unique_smiles = set()
        for query, candidates in test_json_data.items():
            unique_smiles.add(query)
            unique_smiles.update(candidates)

        smiles_list = list(unique_smiles)
        mol_df = mol_df_module.create_mol_df(smiles_list, ids_list=None, max_heavy_atoms=128)

        # Should have molecules created
        assert len(mol_df) > 0
        assert "mol_id" in mol_df.columns



class TestPrepareSpecDf:
    """Tests for 02_prepare_spec_df.py functionality."""

    @pytest.fixture
    def test_mol_df(self):
        """Create a test mol_df."""
        return pd.DataFrame(
            {
                "mol_id": [0, 1, 2],
                "smiles": ["CCO", "CC(=O)O", "CCN"],
                "formula": ["C2H6O", "C2H4O2", "C2H7N"],
                "inchikey_s": [
                    "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                    "QTBSBXVTEAMEQO-UHFFFAOYSA-N",
                    "XGZVUUAUJVXPNA-UHFFFAOYSA-N",
                ],
                "exact_mw": [46.04, 60.02, 45.06],
            }
        )

    def test_config_building(self, test_mol_df):
        """Test basic configuration building."""
        spec_df_module = _load_script_module("02_prepare_spec_df.py")

        config = spec_df_module.get_config(
            prec_types=["[M+H]+"],
            qtof_ces=[],
            ft_ces=["25.0NCE"],
            inst_types=["FT"],
            output_dir="/tmp",
            merged_spectra=False,
        )

        assert "combinations" in config
        assert len(config["combinations"]) == 1
        assert config["combinations"][0]["ce"] == 25.0
        assert config["combinations"][0]["ce_unit"] == "NCE"

    def test_config_multiple_conditions(self, test_mol_df):
        """Test config with multiple experimental combinations."""
        spec_df_module = _load_script_module("02_prepare_spec_df.py")

        config = spec_df_module.get_config(
            prec_types=["[M+H]+", "[M-H]-"],
            qtof_ces=[],
            ft_ces=["25.0NCE", "35.0NCE"],
            inst_types=["FT"],
            output_dir="/tmp",
            merged_spectra=False,
        )

        # Should have 2 prec_types * 2 CE values = 4 combinations
        assert len(config["combinations"]) == 4

    def test_spec_df_creation(self, test_mol_df):
        """Test basic spec_df creation."""
        spec_df_module = _load_script_module("02_prepare_spec_df.py")

        config = spec_df_module.get_config(
            prec_types=["[M+H]+"],
            qtof_ces=[],
            ft_ces=["25.0NCE"],
            inst_types=["FT"],
            output_dir="/tmp",
            merged_spectra=False,
        )

        spec_df = spec_df_module.create_spec_df(test_mol_df, config)

        # Should have molecules * combinations entries
        assert len(spec_df) == len(test_mol_df) * len(config["combinations"])

        required_cols = [
            "spec_id",
            "mol_id",
            "prec_type",
            "nce",
            "ace",
            "inst_type",
            "group_id",
            "dset",
            "prec_mz",
            "peaks",
            "frag_mode",
            "formula",
            "inchikey",
            "exact_mass",
        ]
        for col in required_cols:
            assert col in spec_df.columns

    def test_spec_df_with_nce_values(self, test_mol_df):
        """Test spec_df with NCE collision energies."""
        spec_df_module = _load_script_module("02_prepare_spec_df.py")

        config = spec_df_module.get_config(
            prec_types=["[M+H]+"],
            qtof_ces=[],
            ft_ces=["25.0NCE", "35.0NCE"],
            inst_types=["FT"],
            output_dir="/tmp",
            merged_spectra=False,
        )

        spec_df = spec_df_module.create_spec_df(test_mol_df, config)

        # Should have entries with NCE values (all populated for FT)
        assert spec_df["nce"].notna().all()  # All NCE values should be present for FT
        assert spec_df["ace"].isnull().all()  # All ACE should be NaN when using FT
        assert all(spec_df["nce"] > 0)

    def test_spec_df_with_ace_values(self, test_mol_df):
        """Test spec_df with ACE (eV) collision energies."""
        spec_df_module = _load_script_module("02_prepare_spec_df.py")

        config = spec_df_module.get_config(
            prec_types=["[M+H]+"],
            qtof_ces=["10.0eV", "20.0eV"],
            ft_ces=[],
            inst_types=["QTOF"],
            output_dir="/tmp",
            merged_spectra=False,
        )

        spec_df = spec_df_module.create_spec_df(test_mol_df, config)

        # Should have ACE values for QTOF
        assert spec_df["ace"].notna().all()
        assert spec_df["nce"].isnull().all()
        assert all(spec_df["ace"] > 0)

    def test_spec_df_precursor_mz(self, test_mol_df):
        """Test that precursor m/z is computed correctly."""
        spec_df_module = _load_script_module("02_prepare_spec_df.py")

        config = spec_df_module.get_config(
            prec_types=["[M+H]+"],
            qtof_ces=[],
            ft_ces=["25.0NCE"],
            inst_types=["FT"],
            output_dir="/tmp",
            merged_spectra=False,
        )

        spec_df = spec_df_module.create_spec_df(test_mol_df, config)

        # Precursor m/z should be computed and present
        assert spec_df["prec_mz"].notna().all()
        assert all(spec_df["prec_mz"] > 0)

    def test_spec_df_merged_spectra_mode(self, test_mol_df):
        """Test spec_df with merged spectra mode."""
        spec_df_module = _load_script_module("02_prepare_spec_df.py")

        config = spec_df_module.get_config(
            prec_types=["[M+H]+", "[M-H]-"],
            qtof_ces=[],
            ft_ces=["25.0NCE", "35.0NCE"],
            inst_types=["FT"],
            output_dir="/tmp",
            merged_spectra=True,
        )

        spec_df = spec_df_module.create_spec_df(test_mol_df, config)

        # Group IDs should be grouped by prec_type and inst_type
        groups = spec_df.groupby(["prec_type", "inst_type"])
        for (prec, inst), group in groups:
            unique_group_ids = group["group_id"].unique()
            # All entries with same (prec, inst) should have same group_id
            assert len(unique_group_ids) == 1

    def test_collision_energy_parsing(self):
        """Test parsing of various collision energy formats."""
        spec_df_module = _load_script_module("02_prepare_spec_df.py")

        # Test different NCE formats
        config = spec_df_module.get_config(
            prec_types=["[M+H]+"],
            qtof_ces=[],
            ft_ces=["25.0NCE", "25.0%", "30.0eV"],  # Different formats
            inst_types=["FT"],
            output_dir="/tmp",
            merged_spectra=False,
        )

        # Should parse all formats successfully
        assert len(config["combinations"]) == 3

    def test_validate_inputs_errors(self):
        """Test that validate_inputs raises errors for invalid inputs."""
        spec_df_module = _load_script_module("02_prepare_spec_df.py")

        # Empty prec_types should raise
        with pytest.raises(ValueError):
            spec_df_module.validate_inputs([], [], [], [])

        # Empty inst_types should raise
        with pytest.raises(ValueError):
            spec_df_module.validate_inputs(["[M+H]+"], [], [], [])


class TestIntegration:
    """Integration tests across multiple preprocessing stages."""

    def test_mol_df_to_spec_df_pipeline(self):
        """Test full mol_df to spec_df pipeline."""
        mol_df_module = _load_script_module("01_prepare_mol_df.py")
        spec_df_module = _load_script_module("02_prepare_spec_df.py")

        # Create mol_df
        smiles_list = ["CCO", "CC(=O)O", "CCN"]
        mol_df = mol_df_module.create_mol_df(smiles_list, ids_list=None, max_heavy_atoms=128)

        # Create spec_df from mol_df
        config = spec_df_module.get_config(
            prec_types=["[M+H]+"],
            qtof_ces=[],
            ft_ces=["25.0NCE"],
            inst_types=["FT"],
            output_dir="/tmp",
            merged_spectra=False,
        )

        spec_df = spec_df_module.create_spec_df(mol_df, config)

        # Verify integration
        assert len(mol_df) > 0
        assert len(spec_df) == len(mol_df) * len(config["combinations"])
        assert spec_df["mol_id"].isin(mol_df["mol_id"]).all()

    def test_config_json_generation(self):
        """Test that config JSON can be properly serialized."""
        spec_df_module = _load_script_module("02_prepare_spec_df.py")

        config = spec_df_module.get_config(
            prec_types=["[M+H]+", "[M-H]-"],
            qtof_ces=["10.0eV"],
            ft_ces=["25.0NCE"],
            inst_types=["FT", "QTOF"],
            output_dir="/tmp/test",
            merged_spectra=True,
        )

        # Should be JSON serializable
        config_json = json.dumps(config)
        config_reloaded = json.loads(config_json)

        assert len(config_reloaded["combinations"]) == len(config["combinations"])
        assert config_reloaded["input_parameters"]["merged_spectra"] == True

    def test_json_mode_end_to_end(self):
        """Test full JSON input mode from candidates to spectrum."""
        mol_df_module = _load_script_module("01_prepare_mol_df.py")
        spec_df_module = _load_script_module("02_prepare_spec_df.py")

        # Test data
        candidates = {
            "CCO": ["CC", "C", "CCN"],
            "CC(=O)O": ["CCO", "CC", "C"],
        }

        # Extract unique SMILES
        unique_smiles = set()
        for query, cands in candidates.items():
            unique_smiles.add(query)
            unique_smiles.update(cands)

        # Create mol_df
        mol_df = mol_df_module.create_mol_df(
            list(unique_smiles), ids_list=None, max_heavy_atoms=128
        )

        # Create spec_df
        config = spec_df_module.get_config(
            prec_types=["[M+H]+"],
            qtof_ces=[],
            ft_ces=["25.0NCE"],
            inst_types=["FT"],
            output_dir="/tmp",
            merged_spectra=False,
        )

        spec_df = spec_df_module.create_spec_df(mol_df, config)

        # Verify end-to-end
        assert len(mol_df) >= len(unique_smiles)
        assert len(spec_df) > 0
        assert "prec_mz" in spec_df.columns
