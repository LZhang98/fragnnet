"""Tests for the tied static radius-2 representation probe."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "eval" / "fragment_static_r2_probe.py"
SPEC = importlib.util.spec_from_file_location("fragment_static_r2_probe", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def _chain_record(reverse_boundary: bool = False) -> dict:
    atom_x = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    directed_edges = np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    bond_x = np.asarray([[1.0], [1.0], [2.0], [2.0]], dtype=np.float32)
    retained, lost = (1, 0) if reverse_boundary else (0, 1)
    return {
        "atom_x": atom_x,
        "bond_index": directed_edges,
        "bond_x": bond_x,
        "bp_in": np.asarray([retained], dtype=np.int64),
        "bp_out": np.asarray([lost], dtype=np.int64),
    }


def test_static_r2_key_is_reproducible_across_atom_renumbering() -> None:
    original = _chain_record()
    permutation = np.asarray([2, 0, 1])
    inverse = np.argsort(permutation)
    renumbered = {
        "atom_x": original["atom_x"][permutation],
        "bond_index": inverse[original["bond_index"]],
        "bond_x": original["bond_x"].copy(),
        "bp_in": inverse[original["bp_in"]],
        "bp_out": inverse[original["bp_out"]],
    }

    assert PROBE.bond_environment_keys(original) == PROBE.bond_environment_keys(renumbered)


def test_static_r2_key_is_undirected_like_gate_6_3() -> None:
    forward = PROBE.bond_environment_keys(_chain_record())[0]
    reverse = PROBE.bond_environment_keys(_chain_record(reverse_boundary=True))[0]

    assert forward == reverse


def test_validation_only_environment_uses_unknown_bucket() -> None:
    train = [_chain_record()]
    validation_record = _chain_record()
    validation_record["atom_x"][2, 0] = 9.0
    validation = [validation_record]

    stats = PROBE.attach_static_r2_buckets(train, validation)

    assert train[0]["static_r2_bucket"].tolist() == [1]
    assert validation[0]["static_r2_bucket"].tolist() == [0]
    assert stats["val_unknown_fraction"] == pytest.approx(1.0)


def test_negative_radius_is_rejected() -> None:
    with pytest.raises(ValueError, match="radius must be non-negative"):
        PROBE.bond_environment_keys(_chain_record(), radius=-1)


def test_unselected_channels_are_excluded_from_bucket() -> None:
    original = _chain_record()
    changed_position = _chain_record()
    changed_position["atom_x"][:, 1] += 100.0

    original_key = PROBE.bond_environment_keys(original, atom_feature_indices=[0])
    changed_key = PROBE.bond_environment_keys(changed_position, atom_feature_indices=[0])

    assert original_key == changed_key
