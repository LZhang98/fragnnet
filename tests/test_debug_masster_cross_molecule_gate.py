"""Correctness of the training-only cross-molecule feasibility census."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from scripts.debug.masster_memorization.debug_masster_cross_molecule_gate import (
    compare_profiles,
    fragment_profile,
    summarize,
    tolerance_cosine,
)


def test_precursor_removal_merges_duplicate_masses_and_rejects_invalid_peaks():
    peaks = [(100, 20), (98.6, 5), (98.5, 2), (30, 4), (30, 3), (40, 0), (np.nan, 1)]
    np.testing.assert_array_equal(fragment_profile(peaks, 100, 1.5), [[30, 7]])


def test_cosine_matches_once_and_respects_tolerance():
    left = np.array([[20.0, 3.0], [20.005, 4.0]])
    right = np.array([[20.0, 5.0]])
    assert tolerance_cosine(left, right, 0.01) == pytest.approx(0.8)
    assert tolerance_cosine(left, right + [1.0, 0.0], 0.01) == 0
    assert tolerance_cosine(np.empty((0, 2)), right, 0.01) == 0


def test_losses_can_agree_while_fragment_masses_differ():
    query = (np.array([[60.0, 1.0]]), np.array([[40.0, 1.0]]), np.empty((0, 2)))
    donor = (np.array([[70.0, 2.0]]), np.array([[40.0, 2.0]]), np.empty((0, 2)))
    scores = compare_profiles(query, donor, 0.01)
    assert scores == {
        "fragment_cos": 0.0,
        "loss_cos": 1.0,
        "large_loss_cos": 0.0,
        "top_loss_hit": 1.0,
    }


def test_summary_weights_molecules_equally():
    rows = []
    for mol, count, value in [(1, 1, 1.0), (2, 8, 0.0)]:
        for _ in range(count):
            rows.append(
                dict(
                    query_mol=mol,
                    donor_mol=10,
                    query_half=mol % 2,
                    tanimoto=0.3,
                    same_scaffold=False,
                    query_max_tanimoto=0.3,
                    query_fragment_fraction=0.8,
                    donor_fragment_fraction=0.8,
                    control_min_fragment_fraction=0.8,
                    neighbor_rank=1,
                    **dict.fromkeys(
                        ("fragment_cos", "loss_cos", "large_loss_cos", "top_loss_hit"), value
                    ),
                    **{
                        f"control_{name}": 0.0
                        for name in ("fragment_cos", "loss_cos", "large_loss_cos", "top_loss_hit")
                    },
                )
            )
    result = summarize(pd.DataFrame(rows), 1, 20)
    assert result.loc[result["slice"].eq("all"), "delta"].tolist() == [0.5] * 4


def test_end_to_end_uses_train_only_and_disjoint_identity_pools(tmp_path):
    molecules, spectra = [], []
    for i in range(18):
        mol = Chem.MolFromSmiles("C" * (i + 2) + "O")
        molecules.append(
            {
                "mol_id": i,
                "mol": mol,
                "formula": rdMolDescriptors.CalcMolFormula(mol),
                "scaffold": "",
            }
        )
        mass = Descriptors.ExactMolWt(mol) + 1.0073
        spectra.append(
            {
                "spec_id": i,
                "mol_id": i,
                "prec_type": "[M+H]+",
                "inst_type": "ORB",
                "frag_mode": "HCD",
                "nce": 30.0,
                "ce_type": "single",
                "prec_mz": mass,
                "peaks": [(mass - 18.0, 10.0), (mass, 100.0)],
            }
        )
    # Identical molecule under another ID must never leak across query/donor pools.
    molecules.append({**molecules[0], "mol_id": 99})
    spectra.append({**spectra[0], "spec_id": 99, "mol_id": 99})
    pd.DataFrame(molecules).to_pickle(tmp_path / "mol.pkl")
    pd.DataFrame(spectra).to_pickle(tmp_path / "spec.pkl")
    pd.DataFrame({"spec_id": list(range(16)) + [99]}).to_csv(tmp_path / "train.csv", index=False)
    config = {
        "spec_fp": str(tmp_path / "spec.pkl"),
        "mol_fp": str(tmp_path / "mol.pkl"),
        "train_ids_fp": str(tmp_path / "train.csv"),
        "seed": 42,
        "query_molecules": 3,
        "conditions_per_query": 1,
        "neighbors": 1,
        "controls": 3,
        "mass_window_da": 1000.0,
        "mass_window_fraction": 0.05,
        "control_mass_distance_tolerance_da": 1000.0,
        "peak_tolerance_da": 0.01,
        "precursor_exclusion_da": 1.5,
        "large_loss_min_da": 50.0,
        "bootstrap_repeats": 20,
    }
    (tmp_path / "config.yml").write_text(yaml.safe_dump(config))
    subprocess.run(
        [
            sys.executable,
            "scripts/debug/masster_memorization/debug_masster_cross_molecule_gate.py",
            "--config",
            str(tmp_path / "config.yml"),
            "--output",
            str(tmp_path / "out"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pairs = pd.read_pickle(tmp_path / "out/pairs.pkl")
    provenance = json.loads((tmp_path / "out/provenance.json").read_text())
    assert provenance["status"] == "complete"
    assert provenance["training_rows"] == 17
    assert len(pairs) == 3
    assert set(pairs.query_mol).isdisjoint(pairs.donor_mol)
    used = (
        set(pairs.query_spec) | set(pairs.donor_spec) | {s for row in pairs.controls for s in row}
    )
    assert used.isdisjoint({16, 17, 99})
    assert np.allclose(pairs.loss_cos, 1.0)
    assert pairs.same_scaffold.all()
    assert np.allclose(pairs.query_fragment_fraction, 10 / 110)
