"""Migration records for the repository reorganization (plan Phase 0 item 8).

Old checkpoints, W&B runs and config files reach the code with keys that were renamed
(``mass_shatter`` to ``masster``, ``masster_proposal_type`` to ``masster_proposal_method``,
``eval_target_apply_train_peak_filters`` to ``..._rank_filters``) or that did not exist yet.
Three boundaries backfill them, and this file records what each produces:

- ``utils.pl_utils.get_pl_hparams`` (checkpoint hparams), with a minimal stub per model family,
  several edge-case stubs, and the hparams of a real legacy ``mass_shatter`` checkpoint;
- ``runner.load_config`` (template, custom config and ``key=value`` CLI overrides, including
  dotted keys through ``migrate_legacy_masster_key``), and ``merge_wandb_config`` for a saved
  W&B run;
- the model constructors, which backfill the same keys into ``hparams`` before saving them.

The legacy checkpoint is ``saved_ckpts/c0_control_vcpho5qm.ckpt``. Its ``hyper_parameters`` were
extracted once into ``tests/golden/reorg/legacy_mass_shatter_hparams.json`` (437 keys, 92
``mass_shatter_*``), so these tests do not need the untracked checkpoint:

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    json.dump(ckpt["hyper_parameters"], fp, indent=1, sort_keys=True)

The checkpoint names the retired instrument type ``FT`` (retired 2026-09-28), which
``get_pl_hparams`` refuses at six paths. Its refusal is recorded as is. The migration records use
a mechanical ``FT`` to ``ORB`` relabel of those six paths, done in this file; no repository tool
relabels a checkpoint's hparams.

Records hold plain JSON values (keys, values, key sets), never class names or module paths, so
Phase 1 (ICEBERG retirement, registry) and Phase 6 (renames) leave them unchanged. ICEBERG
types are not recorded: they migrate the shared keys only and are retired in Phase 1.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import io
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import reorg_fixtures as F
import torch as th
import yaml
from reorg_golden import GOLDEN_DIR, ratchet_json
from reorg_layers import MODEL_TYPES

from fragnnet.runner import load_config, merge_wandb_config, validate_sampler_config
from fragnnet.utils.data_utils import find_retired_inst_type_refs
from fragnnet.utils.pl_utils import (
    get_pl_hparams,
    migrate_legacy_hparams,
    migrate_legacy_masster_key,
    migrate_legacy_masster_name,
    migrate_masster_proposal_method,
)
from fragnnet.utils.script_utils import get_config_from_ckpt
from fragnnet.utils.setup_utils import get_model_cls

GOLDEN = "migrations"
C0_FIXTURE_FP = GOLDEN_DIR / "legacy_mass_shatter_hparams.json"
ABSENT = "<absent>"

# Keys the migrations can supply, written out here so the records do not follow the code's
# private tables (pl_utils.py, `_LEGACY_*_DEFAULTS`).
SHARED_KEYS = (
    "mol_max_degree",
    "mol_max_spatial_distance",
    "mol_multi_hop_max_distance",
    "report_metric_std",
    "inst_type_loss_tol",
    "eval_target_apply_train_rank_filters",
    "use_nce",
    "use_ace",
)
FRAGNNET_KEYS = (
    "cc_channel_dropout",
    "candidate_ion_head",
    "candidate_ion_formula_features",
    "candidate_ion_neutral_loss_features",
    "candidate_ion_chunk_size",
    "candidate_ion_checkpoint",
    "ion_channel_factorized",
    "ion_channel_adduct_gated",
    "ion_channel_rank",
)
MASSTER_RT_CCS_KEYS = (
    "masster_predict_rt_ccs",
    "masster_predict_rt",
    "masster_predict_ccs",
    "masster_rt_loss_weight",
    "masster_ccs_loss_weight",
    "masster_rt_scale",
    "masster_ccs_scale",
)
MASSTER_CANDIDATE_KEYS = (
    "masster_candidate_backend",
    "masster_proposal_method",
    "masster_fast_gnn_beam_width",
    "masster_fast_gnn_max_boundary_cuts",
    "masster_fragment_selection",
    "masster_selected_fragment_target",
)
MASSTER_KEYS = (
    *MASSTER_RT_CCS_KEYS,
    "masster_context_in_gnn",
    "masster_intensity_transformer_attention",
    *MASSTER_CANDIDATE_KEYS,
)
SPEC_PARAMS_KEYS = (
    "rt",
    "rt_labels_fp",
    "ccs",
    "ccs_labels_fp",
    "rt_ccs",
    "rt_ccs_labels_fp",
    "merge_ce_aggregation",
)

# Families recorded below. ICEBERG types are left out (retired in Phase 1); `mass_shatter` is the
# legacy spelling of `masster` and only appears in checkpoints and custom configs.
FAMILIES = (
    "frag_gnn",
    "fragnnet",
    "masster",
    "mass_shatter",
    "neims",
    "massformer",
    "graff",
    "precursor",
    "gnn",
)
ALIAS_PAIRS = (
    ("frag_gnn", "fragnnet"),
    ("masster", "mass_shatter"),
)


def _family_keys(family: str) -> tuple[str, ...]:
    """Top-level keys the migrations can backfill for a model family."""
    extra: tuple[str, ...] = ()
    if family in ("frag_gnn", "fragnnet"):
        extra = FRAGNNET_KEYS
    elif family in ("masster", "mass_shatter"):
        extra = MASSTER_KEYS
    return (*SHARED_KEYS, *extra)


def _legacy_spelling(key: str) -> str:
    return "mass_shatter_" + key.removeprefix("masster_") if key.startswith("masster_") else key


def _quiet(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with stdout captured (load_config and the constructors print)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _error(fn: Callable[[], Any]) -> str:
    """The exception ``fn`` raises as ``"Type: message"``; fails when it does not raise."""
    try:
        _quiet(fn)
    except Exception as exc:  # noqa: BLE001 - the record is the exception itself
        return f"{type(exc).__name__}: {exc}"
    raise AssertionError("expected an exception")


def _stub(model_type: str, **extra: Any) -> dict:
    """Minimal checkpoint hparams: a model type, ``spec_params`` and ``extra`` keys."""
    return {"model_type": model_type, "spec_params": {"nce": True, "ace": False}, **extra}


def _pl_hparams(hparams: dict) -> dict:
    return get_pl_hparams({"hyper_parameters": hparams})


def _relabel_retired_ft(node: Any) -> Any:
    """Mechanical ``FT`` to ``ORB`` relabel of a checkpoint's hparams, keys and values alike."""
    if isinstance(node, str):
        return "ORB" if node == "FT" else node
    if isinstance(node, list):
        return [_relabel_retired_ft(item) for item in node]
    if isinstance(node, dict):
        relabelled = {}
        for key, value in node.items():
            new_key = "ORB" if key == "FT" else key
            if new_key != key and new_key in node:
                raise AssertionError(f"relabelling {key!r} collides with existing {new_key!r}")
            relabelled[new_key] = _relabel_retired_ft(value)
        return relabelled
    return node


def _write_yaml(directory: Path, name: str, content: Any) -> str:
    path = directory / name
    path.write_text(content if isinstance(content, str) else yaml.safe_dump(content))
    return str(path)


def _present(mapping: dict, keys: tuple[str, ...]) -> dict:
    return {key: mapping.get(key, ABSENT) for key in keys}


def _legacy_leftovers(config: dict) -> list[str]:
    """Keys that should not survive a migration: ``mass_shatter_*`` and the old proposal key."""
    return sorted(
        key for key in config if key.startswith("mass_shatter_") or key == "masster_proposal_type"
    )


def _contains(actual: Any, expected: Any) -> bool:
    """Whether ``actual`` holds ``expected``: dicts as subsets (a custom config is deep-merged
    into the template, which adds sibling keys), everything else by equality."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        return all(k in actual and _contains(actual[k], v) for k, v in expected.items())
    return actual == expected


def _validate_outcome(config: dict, **overrides: Any) -> str:
    """``validate_sampler_config`` on ``config`` plus ``overrides``: ``"ok"`` or the error."""
    candidate = {**copy.deepcopy(config), **overrides}
    try:
        validate_sampler_config(candidate)
    except ValueError as exc:
        return f"ValueError: {exc}"
    return "ok"


@pytest.fixture(scope="module")
def c0_raw() -> dict:
    """``hyper_parameters`` of ``c0_control_vcpho5qm.ckpt``, as stored (names retired ``FT``)."""
    return json.loads(C0_FIXTURE_FP.read_text())


@pytest.fixture
def c0_orb(c0_raw: dict) -> dict:
    """The c0 hparams with the six retired ``FT`` references relabelled ``ORB``."""
    return _relabel_retired_ft(c0_raw)


@pytest.fixture(scope="module")
def template_config() -> dict:
    """The template resolved for MASSter, the base the CLI-override records start from."""
    return _quiet(load_config, F.CONFIG_DIR, None, None, model_type="masster")


# ---------------------------------------------------------------------------------------------
# Coverage and fixture integrity
# ---------------------------------------------------------------------------------------------


def test_every_accepted_non_iceberg_model_type_has_a_family_record() -> None:
    uncovered = [t for t in MODEL_TYPES if t not in FAMILIES and not t.startswith("iceberg")]
    assert uncovered == []
    assert set(FAMILIES) - set(MODEL_TYPES) == {"mass_shatter"}


def test_c0_fixture_is_the_legacy_mass_shatter_checkpoint(c0_raw: dict) -> None:
    assert c0_raw["model_type"] == "mass_shatter"
    assert len(c0_raw) == 437
    assert sum(key.startswith("mass_shatter_") for key in c0_raw) == 92
    assert not any(key.startswith("masster_") for key in c0_raw)
    assert c0_raw["spec_params"]["inst_types"] == ["FT"]


# ---------------------------------------------------------------------------------------------
# get_pl_hparams
# ---------------------------------------------------------------------------------------------


def test_get_pl_hparams_minimal_stub_per_family() -> None:
    """A stub with only ``model_type`` and ``spec_params`` gets every family's legacy keys."""
    record = {}
    for family in FAMILIES:
        stub = _stub(family)
        migrated = _pl_hparams(stub)
        record[family] = migrated
        expected_model_type = "masster" if family == "mass_shatter" else family
        assert migrated["model_type"] == expected_model_type
        # Added: this family's top-level keys; spec_params gains the seven RT/CCS/CE keys.
        assert set(migrated) - set(stub) == set(_family_keys(family))
        assert set(migrated["spec_params"]) - set(stub["spec_params"]) == set(SPEC_PARAMS_KEYS)
        # use_nce/use_ace follow spec_params, not a constant.
        assert (migrated["use_nce"], migrated["use_ace"]) == (True, False)
        assert migrated["spec_params"]["merge_ce_aggregation"] == "legacy"
        assert not any(key.startswith("mass_shatter_") for key in migrated)
        assert stub == _stub(family), "get_pl_hparams must not mutate its input"
    assert record["frag_gnn"]["ion_channel_rank"] == 32
    assert record["frag_gnn"]["candidate_ion_chunk_size"] == 32768
    assert "ion_channel_rank" not in record["masster"]
    assert record["masster"]["masster_fast_gnn_max_boundary_cuts"] == 3
    assert record["masster"]["masster_proposal_method"] == "cut_masks"
    assert record["masster"]["masster_candidate_backend"] == "cpu"
    assert record["neims"]["mol_max_degree"] is None
    ratchet_json(GOLDEN, "get_pl_hparams:minimal_stub", record)


def test_get_pl_hparams_alias_pairs_migrate_identically() -> None:
    """Alias model types differ only by their own name; ``mass_shatter`` becomes ``masster``."""
    record = {}
    for first, second in ALIAS_PAIRS:
        migrated = [_pl_hparams(_stub(first)), _pl_hparams(_stub(second))]
        differing = sorted(
            k
            for k in set(migrated[0]) | set(migrated[1])
            if migrated[0].get(k) != migrated[1].get(k)
        )
        # The values of the keys that differ: the names themselves, as each pair spells them.
        record[f"{first}/{second}"] = {
            key: [migrated[0].get(key), migrated[1].get(key)] for key in differing
        }
    assert {pair: sorted(diff) for pair, diff in record.items()} == {
        "frag_gnn/fragnnet": ["model_type"],
        "masster/mass_shatter": [],
    }
    ratchet_json(GOLDEN, "get_pl_hparams:alias_pairs", record)


def test_get_pl_hparams_edge_case_stubs() -> None:
    """Missing ``spec_params``, explicit values, old spellings and both-spelling precedence."""
    explicit = {
        "model_type": "masster",
        "spec_params": {
            "nce": True,
            "ace": True,
            "rt": True,
            "rt_labels_fp": "rt.pkl",
            "ccs": True,
            "ccs_labels_fp": "ccs.pkl",
            "rt_ccs": True,
            "rt_ccs_labels_fp": "rt_ccs.pkl",
            "merge_ce_aggregation": "mean",
        },
        **dict.fromkeys(MASSTER_KEYS, 7),
        "masster_candidate_backend": "cuda",
        "masster_proposal_method": "connected_fragment_beam",
        "masster_fragment_selection": "marginal",
        "masster_intensity_transformer_attention": "full",
        "masster_context_in_gnn": True,
        "masster_predict_rt": True,
        "masster_selected_fragment_target": 128,
        "mol_max_degree": 64,
        "mol_max_spatial_distance": 128,
        "mol_multi_hop_max_distance": 5,
        "report_metric_std": True,
        "inst_type_loss_tol": {"ORB": {"rel": 1e-5, "min_mz": 200.0}},
        "eval_target_apply_train_rank_filters": True,
        "use_nce": False,
        "use_ace": False,
    }
    stubs = {
        "no_spec_params": {"model_type": "masster"},
        "explicit_values_win": explicit,
        "old_spelling_keys_and_proposal_type": _stub(
            "mass_shatter",
            mass_shatter_proposal_type="fast_gnn",
            mass_shatter_candidate_backend="cuda",
            mass_shatter_head_dropout=0.2,
        ),
        "both_spellings_current_wins": _stub(
            "mass_shatter", masster_head_dropout=0.5, mass_shatter_head_dropout=0.2
        ),
        "legacy_rank_filter_key": _stub("neims", eval_target_apply_train_peak_filters=True),
        "merge_ce_aggregation_explicit": {
            "model_type": "gnn",
            "spec_params": {"nce": False, "ace": True, "merge_ce_aggregation": "mean"},
        },
    }
    record = {name: _pl_hparams(stub) for name, stub in stubs.items()}

    assert record["no_spec_params"]["use_nce"] is False
    assert "spec_params" not in record["no_spec_params"]
    assert record["no_spec_params"]["masster_proposal_method"] == "cut_masks"
    assert record["explicit_values_win"] == stubs["explicit_values_win"]
    legacy = record["old_spelling_keys_and_proposal_type"]
    assert legacy["model_type"] == "masster"
    assert legacy["masster_head_dropout"] == 0.2
    assert legacy["masster_candidate_backend"] == "cuda"
    assert legacy["masster_proposal_method"] == "connected_fragment_beam"
    assert _legacy_leftovers(legacy) == []
    both = record["both_spellings_current_wins"]
    assert both["masster_head_dropout"] == 0.5 and _legacy_leftovers(both) == []
    rank = record["legacy_rank_filter_key"]
    assert rank["eval_target_apply_train_rank_filters"] is True
    assert "eval_target_apply_train_peak_filters" not in rank
    merge_ce = record["merge_ce_aggregation_explicit"]
    assert merge_ce["spec_params"]["merge_ce_aggregation"] == "mean"
    assert (merge_ce["use_nce"], merge_ce["use_ace"]) == (False, True)
    ratchet_json(GOLDEN, "get_pl_hparams:edge_case_stubs", record)


def test_c0_checkpoint_hparams_are_refused_as_stored(c0_raw: dict) -> None:
    """The saved checkpoint names the retired ``FT`` at six paths; ``get_pl_hparams`` refuses it."""
    with pytest.raises(ValueError, match="retired instrument type") as caught:
        _pl_hparams(c0_raw)
    paths = ast.literal_eval(re.search(r"at (\[.*?\])\.", str(caught.value)).group(1))
    assert sorted(paths) == sorted(find_retired_inst_type_refs(c0_raw))
    assert len(paths) == 6
    ratchet_json(
        GOLDEN, "get_pl_hparams:c0_refusal", {"error": "ValueError", "paths": sorted(paths)}
    )


def test_get_pl_hparams_on_relabelled_c0(c0_orb: dict) -> None:
    """The legacy checkpoint renames its 92 ``mass_shatter_*`` keys and gains seven RT/CCS keys."""
    assert find_retired_inst_type_refs(c0_orb) == []
    before = copy.deepcopy(c0_orb)
    migrated = _pl_hparams(c0_orb)
    assert c0_orb == before, "get_pl_hparams must not mutate its input"

    renamed = sorted(key for key in c0_orb if key.startswith("mass_shatter_"))
    assert len(renamed) == 92
    assert migrated["model_type"] == "masster"
    assert _legacy_leftovers(migrated) == []
    for old_key in renamed:
        assert migrated[migrate_legacy_masster_key(old_key)] == c0_orb[old_key], old_key

    carried = {key: value for key, value in c0_orb.items() if key not in renamed}
    unchanged = {key for key in carried if key not in ("model_type", "spec_params")}
    assert all(migrated[key] == carried[key] for key in unchanged)
    added = {
        key: migrated[key]
        for key in sorted(set(migrated) - set(carried))
        if key not in {migrate_legacy_masster_key(old) for old in renamed}
    }
    spec_added = {
        key: migrated["spec_params"][key]
        for key in sorted(set(migrated["spec_params"]) - set(c0_orb["spec_params"]))
    }
    spec_kept = {k: v for k, v in c0_orb["spec_params"].items() if k in migrated["spec_params"]}
    assert all(migrated["spec_params"][k] == v for k, v in spec_kept.items())
    assert len(migrated) == len(c0_orb) + 7 == 444
    assert set(added) == set(MASSTER_RT_CCS_KEYS)
    assert added["masster_rt_scale"] == 60.0 and added["masster_ccs_scale"] == 100.0
    assert added["masster_predict_rt_ccs"] is False
    assert spec_added == {
        "rt": False,
        "rt_labels_fp": None,
        "ccs": False,
        "ccs_labels_fp": None,
        "rt_ccs": False,
        "rt_ccs_labels_fp": None,
    }
    record = {
        "n_in": len(c0_orb),
        "n_out": len(migrated),
        "renamed": renamed,
        "model_type": [c0_orb["model_type"], migrated["model_type"]],
        "added": added,
        "spec_params_added": spec_added,
        # c0 already carries the later MASSter and shared keys, so those backfills are no-ops.
        "migration_keys_after": _present(migrated, (*SHARED_KEYS, *MASSTER_KEYS)),
    }
    ratchet_json(GOLDEN, "get_pl_hparams:c0_relabelled", record)


def test_get_pl_hparams_oldest_masster_stub(c0_orb: dict) -> None:
    """Take every migrated key out of c0, then see which values come back (the oldest stub)."""
    oldest = copy.deepcopy(c0_orb)
    removable = (*SHARED_KEYS, *MASSTER_KEYS, "masster_proposal_type")
    for key in removable:
        oldest.pop(key, None)
        oldest.pop(_legacy_spelling(key), None)
    for key in SPEC_PARAMS_KEYS:
        oldest["spec_params"].pop(key, None)
    assert not (set(removable) | {_legacy_spelling(k) for k in removable}) & set(oldest)

    migrated = _pl_hparams(oldest)
    backfilled = {
        key: migrated[key] for key in sorted((*SHARED_KEYS, *MASSTER_KEYS)) if key not in oldest
    }
    assert set(backfilled) == {*SHARED_KEYS, *MASSTER_KEYS}
    assert backfilled["masster_candidate_backend"] == "cpu"
    assert backfilled["masster_proposal_method"] == "cut_masks"
    assert backfilled["masster_fast_gnn_beam_width"] == 256
    assert backfilled["masster_fast_gnn_max_boundary_cuts"] == 3
    assert backfilled["masster_fragment_selection"] == "mask_order"
    assert backfilled["masster_selected_fragment_target"] is None
    assert backfilled["masster_context_in_gnn"] is False
    assert backfilled["masster_intensity_transformer_attention"] == "induced"
    assert backfilled["mol_max_degree"] is None
    assert backfilled["report_metric_std"] is False
    assert (backfilled["use_nce"], backfilled["use_ace"]) == (True, False)
    spec_backfilled = {key: migrated["spec_params"][key] for key in SPEC_PARAMS_KEYS}
    assert spec_backfilled["merge_ce_aggregation"] == "legacy"
    # Every other key is carried over, the ``mass_shatter_*`` ones under their new name.
    for key, value in oldest.items():
        if key.startswith("mass_shatter_"):
            assert migrated[migrate_legacy_masster_key(key)] == value
        elif key not in ("model_type", "spec_params"):
            assert migrated[key] == value
    record = {
        "n_in": len(oldest),
        "n_out": len(migrated),
        "backfilled": backfilled,
        "spec_params_backfilled": spec_backfilled,
    }
    ratchet_json(GOLDEN, "get_pl_hparams:oldest_masster_stub", record)


# ---------------------------------------------------------------------------------------------
# Key-level migration tables
# ---------------------------------------------------------------------------------------------


def test_migration_helper_tables() -> None:
    """Exact input/output pairs of the name, proposal and legacy-default migrations."""
    keys = [
        "mass_shatter_max_h_transfer",
        "masster_max_h_transfer",
        "mass_shatter_",
        "mass_shatter",
        "model_type",
        "spec_params.mass_shatter_x",
        "Mass_shatter_x",
        "mass_shatter_proposal_type",
    ]
    key_table = {key: migrate_legacy_masster_key(key) for key in keys}

    name_inputs = {
        "model_type_only": {"model_type": "mass_shatter"},
        "keys_under_current_model_type": {"model_type": "masster", "mass_shatter_a": 1},
        "keys_under_other_model_type": {"model_type": "frag_gnn", "mass_shatter_a": 1},
        "current_key_wins": {"model_type": "mass_shatter", "masster_a": 2, "mass_shatter_a": 1},
        "nested_keys_untouched": {
            "model_type": "mass_shatter",
            "spec_params": {"mass_shatter_x": 1},
        },
    }
    name_table = {}
    for name, config in name_inputs.items():
        migrated = copy.deepcopy(config)
        migrate_legacy_masster_name(migrated)
        name_table[name] = migrated

    proposal_inputs = {
        "old_key_fast_gnn": {"masster_proposal_type": "fast_gnn"},
        "old_key_cut_masks": {"masster_proposal_type": "cut_masks"},
        "canonical_key_wins_and_normalizes": {
            "masster_proposal_type": "cut_masks",
            "masster_proposal_method": "fast_gnn",
        },
        "canonical_unchanged": {"masster_proposal_method": "connected_fragment_beam"},
        "missing_stays_missing": {"other": 1},
    }
    proposal_table = {}
    for name, config in proposal_inputs.items():
        migrated = copy.deepcopy(config)
        migrate_masster_proposal_method(migrated)
        proposal_table[name] = migrated

    legacy_inputs = {
        "bare_masster": {"model_type": "masster"},
        "no_model_type": {},
        "old_rank_filter_key": {"eval_target_apply_train_peak_filters": True},
        "both_rank_filter_keys": {
            "eval_target_apply_train_peak_filters": True,
            "eval_target_apply_train_rank_filters": False,
        },
        "explicit_values_win": {
            "model_type": "frag_gnn",
            "ion_channel_rank": 8,
            "mol_max_degree": 64,
            "use_nce": False,
            "spec_params": {"nce": True},
        },
    }
    legacy_table = {}
    for name, config in legacy_inputs.items():
        migrated = copy.deepcopy(config)
        migrate_legacy_hparams(migrated)
        legacy_table[name] = migrated

    assert key_table["mass_shatter_max_h_transfer"] == "masster_max_h_transfer"
    assert key_table["mass_shatter_"] == "masster_"
    assert key_table["mass_shatter"] == "mass_shatter"
    assert key_table["spec_params.mass_shatter_x"] == "spec_params.mass_shatter_x"
    assert name_table["model_type_only"] == {"model_type": "masster"}
    assert name_table["keys_under_other_model_type"] == {"model_type": "frag_gnn", "masster_a": 1}
    assert name_table["current_key_wins"] == {"model_type": "masster", "masster_a": 2}
    assert name_table["nested_keys_untouched"]["spec_params"] == {"mass_shatter_x": 1}
    assert proposal_table["old_key_fast_gnn"] == {
        "masster_proposal_method": "connected_fragment_beam"
    }
    assert proposal_table["canonical_key_wins_and_normalizes"] == {
        "masster_proposal_method": "connected_fragment_beam"
    }
    assert proposal_table["missing_stays_missing"] == {"other": 1}
    assert legacy_table["old_rank_filter_key"]["eval_target_apply_train_rank_filters"] is True
    assert legacy_table["both_rank_filter_keys"]["eval_target_apply_train_rank_filters"] is False
    assert legacy_table["explicit_values_win"]["use_nce"] is False
    assert legacy_table["explicit_values_win"]["ion_channel_rank"] == 8
    assert "ion_channel_rank" not in legacy_table["bare_masster"]
    assert "use_nce" in legacy_table["no_model_type"]

    ratchet_json(
        GOLDEN,
        "migration_helper_tables",
        {
            "migrate_legacy_masster_key": key_table,
            "migrate_legacy_masster_name": name_table,
            "migrate_masster_proposal_method": proposal_table,
            "migrate_legacy_hparams": legacy_table,
        },
    )


# ---------------------------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------------------------


def test_load_config_template_per_family() -> None:
    """The template declares every migrated key, so no family is backfilled; record its values."""
    template = yaml.safe_load((Path(F.CONFIG_DIR) / "template.yml").read_text())
    record = {}
    for family in FAMILIES:
        if family == "mass_shatter":
            continue
        config = _quiet(load_config, F.CONFIG_DIR, None, None, model_type=family)
        assert config["model_type"] == family
        assert _legacy_leftovers(config) == []
        all_keys = {*SHARED_KEYS, *FRAGNNET_KEYS, *MASSTER_KEYS}
        assert all_keys <= set(config), sorted(all_keys - set(config))
        assert set(SPEC_PARAMS_KEYS) <= set(config["spec_params"])
        # The template decides: every migrated key resolves to the file's own value.
        assert all(config[key] == template[key] for key in all_keys), family
        assert all(
            config["spec_params"][key] == template["spec_params"][key] for key in SPEC_PARAMS_KEYS
        ), family
        record[family] = {
            "keys": _present(config, _family_keys(family)),
            "spec_params": _present(config["spec_params"], SPEC_PARAMS_KEYS),
        }
    # ... and the legacy fallback would have said something else, so a silent backfill over the
    # template would show up here (exact values live in the golden record only).
    for family in ("frag_gnn", "masster"):
        fallback = _pl_hparams(_stub(family))
        disagree = [key for key in _family_keys(family) if fallback[key] != template[key]]
        assert disagree, f"{family}: the template matches every legacy fallback"
    ratchet_json(GOLDEN, "load_config:template_per_family", record)


def test_load_config_alias_pairs_resolve_identically(tmp_path: Path) -> None:
    """Alias pairs resolve to the same config but their name, validated the same way."""
    record = {}
    for first, second in ALIAS_PAIRS:
        configs = [
            _quiet(
                load_config,
                F.CONFIG_DIR,
                _write_yaml(tmp_path, f"{name}.yml", {"model_type": name}),
                None,
            )
            for name in (first, second)
        ]
        differing = sorted(
            k for k in set(configs[0]) | set(configs[1]) if configs[0].get(k) != configs[1].get(k)
        )
        scenarios = {
            "default": {},
            "dynamic_batch_sampler": {"dynamic_batch_sampler": True},
            "masster_batch_sampler": {
                "masster_batch_sampler": True,
                "masster_batch_sampler_max": 8,
                "devices": 1,
            },
            "bad_simple_group_sampler": {
                "simple_group_sampler": True,
                "simple_group_sampler_type": "x",
            },
        }
        outcomes = {}
        for label, overrides in scenarios.items():
            first_out, second_out = (_validate_outcome(config, **overrides) for config in configs)
            # An error may quote the model type; everything else must be identical.
            normalized = [
                out.replace(f"'{config['model_type']}'", "'<model_type>'")
                for out, config in zip((first_out, second_out), configs)
            ]
            assert normalized[0] == normalized[1], (first, second, label)
            outcomes[label] = normalized[0]
        record[f"{first}/{second}"] = {
            "differing_keys": differing,
            "model_types": [config["model_type"] for config in configs],
            "validate_sampler_config": outcomes,
        }
    assert record["frag_gnn/fragnnet"]["differing_keys"] == ["model_type"]
    assert record["masster/mass_shatter"]["differing_keys"] == []
    assert record["masster/mass_shatter"]["model_types"] == ["masster", "masster"]
    assert (
        record["masster/mass_shatter"]["validate_sampler_config"]["masster_batch_sampler"] == "ok"
    )
    ratchet_json(GOLDEN, "load_config:alias_pairs", record)


def test_load_config_legacy_inputs(tmp_path: Path) -> None:
    """Pre-rename templates, bare templates, legacy custom configs and refused inputs."""
    masster_keys = (
        "masster_candidate_backend",
        "masster_proposal_method",
        "masster_fast_gnn_beam_width",
        "masster_fast_gnn_max_boundary_cuts",
    )
    bare = _write_yaml(tmp_path, "bare_template.yml", "model_type: frag_gnn\n")
    pre_rename = _write_yaml(
        tmp_path,
        "pre_rename_template.yml",
        {
            "model_type": "mass_shatter",
            "mass_shatter_candidate_backend": "cuda",
            "mass_shatter_proposal_type": "fast_gnn",
            "mass_shatter_head_dropout": 0.2,
        },
    )

    def custom(name: str, content: Any) -> str:
        return _write_yaml(tmp_path, f"{name}.yml", content)

    # A bare template declares no MASSter keys: the four-key backfill supplies them.
    from_bare = _quiet(load_config, bare, custom("bare_custom", {"model_type": "masster"}), None)
    assert from_bare == {
        "model_type": "masster",
        "masster_candidate_backend": "cpu",
        "masster_proposal_method": "cut_masks",
        "masster_fast_gnn_beam_width": 256,
        "masster_fast_gnn_max_boundary_cuts": 3,
    }
    # A bare template under another model type gets no MASSter keys.
    from_bare_other = _quiet(load_config, bare, None, None)
    assert from_bare_other == {"model_type": "frag_gnn"}
    # A template saved inside a pre-rename checkpoint is migrated before use.
    from_old_template = _quiet(load_config, pre_rename, None, None)
    assert from_old_template["model_type"] == "masster"
    assert from_old_template["masster_candidate_backend"] == "cuda"
    assert from_old_template["masster_proposal_method"] == "connected_fragment_beam"
    assert from_old_template["masster_head_dropout"] == 0.2
    assert _legacy_leftovers(from_old_template) == []
    # Legacy custom config on the current template: old keys renamed, CLI override renamed too.
    legacy_custom = custom(
        "legacy_custom",
        {
            "model_type": "mass_shatter",
            "mass_shatter_head_dropout": 0.37,
            "masster_proposal_type": "fast_gnn",
        },
    )
    legacy = _quiet(load_config, F.CONFIG_DIR, legacy_custom, ["mass_shatter_max_h_transfer=2"])
    assert legacy["model_type"] == "masster"
    assert legacy["masster_head_dropout"] == 0.37
    assert legacy["masster_max_h_transfer"] == 2
    assert legacy["masster_proposal_method"] == "connected_fragment_beam"
    assert _legacy_leftovers(legacy) == []
    # Both spellings in one custom config: the current key wins.
    both = _quiet(
        load_config,
        F.CONFIG_DIR,
        custom(
            "both",
            {
                "model_type": "masster",
                "masster_head_dropout": 0.5,
                "mass_shatter_head_dropout": 0.2,
                "masster_proposal_type": "cut_masks",
                "masster_proposal_method": "fast_gnn",
            },
        ),
        None,
    )
    assert both["masster_head_dropout"] == 0.5
    assert both["masster_proposal_method"] == "connected_fragment_beam"
    assert _legacy_leftovers(both) == []

    refusals = {}
    for label, inst_types in (("FT", ["FT"]), ("OB", ["OB"]), ("OB_F", ["OB_F"])):
        path = custom(f"retired_{label}", {"spec_params": {"inst_types": inst_types}})
        message = _error(lambda path=path: load_config(F.CONFIG_DIR, path, None))
        refusals[label] = message.split(". ")[0]
        assert refusals[label].startswith("ValueError: config names a retired instrument type")
    unknown_path = custom("unknown", {"model_type": "mass_shatter", "mass_shatter_not_a_key": 1})
    unknown = _error(lambda: load_config(F.CONFIG_DIR, unknown_path, None))
    assert unknown.endswith("declare: ['masster_not_a_key']")
    # `mass_shatter` is a legacy spelling migrated inside files; as the `model_type` argument (a
    # checkpoint's type) it is refused today. Recorded so a registry change cannot start
    # accepting it unnoticed.
    alias_argument = _error(
        lambda: load_config(F.CONFIG_DIR, None, None, model_type="mass_shatter")
    )
    assert alias_argument.startswith("ValueError: unknown model_type 'mass_shatter'")

    record = {
        "bare_template_custom_masster": from_bare,
        "bare_template_other_model": from_bare_other,
        "pre_rename_template": _present(
            from_old_template, ("model_type", "masster_head_dropout", *masster_keys)
        ),
        "legacy_custom_plus_override": {
            key: legacy[key]
            for key in (
                "model_type",
                "masster_head_dropout",
                "masster_max_h_transfer",
                "masster_proposal_method",
            )
        },
        "both_spellings_current_wins": {
            key: both[key] for key in ("masster_head_dropout", "masster_proposal_method")
        },
        "retired_instrument_refusals": refusals,
        "unknown_legacy_key": unknown.split("declare: ")[1],
        # The known-types list follows the registry (ICEBERG leaves in Phase 1), so cut it off.
        "legacy_name_as_model_type_argument": alias_argument.split(";")[0],
    }
    ratchet_json(GOLDEN, "load_config:legacy_inputs", record)


def test_load_config_cli_overrides(template_config: dict) -> None:
    """Dotted CLI overrides: ``migrate_legacy_masster_key`` runs on the dotted key first."""
    cases = {
        "legacy_key": ["mass_shatter_max_h_transfer=2"],
        "legacy_float_key": ["mass_shatter_head_dropout=0.3"],
        "current_key": ["masster_max_h_transfer=3"],
        "legacy_then_current_applied_in_order": [
            "mass_shatter_head_dropout=0.3",
            "masster_head_dropout=0.5",
        ],
        "proposal_type_fast_gnn": ["masster_proposal_type=fast_gnn"],
        "legacy_proposal_type_fast_gnn": ["mass_shatter_proposal_type=fast_gnn"],
        "legacy_proposal_method": ["mass_shatter_proposal_method=cut_masks"],
        "proposal_method_fast_gnn": ["masster_proposal_method=fast_gnn"],
        "nested_dotted_key": ["spec_params.nce=true"],
        "deeper_dotted_key": ["spec_params.merge_tolerance_by_inst.ORB.rel=0.5"],
        "plain_key": ["seed=5"],
    }
    watched = (
        "masster_max_h_transfer",
        "masster_head_dropout",
        "masster_proposal_method",
        "seed",
    )
    record = {}
    for label, override in cases.items():
        config = _quiet(load_config, F.CONFIG_DIR, None, override, model_type="masster")
        assert _legacy_leftovers(config) == [], label
        assert config["model_type"] == "masster"
        record[label] = {
            "override": override,
            "changed_top_level": {
                key: [template_config[key], config[key]]
                for key in sorted(template_config)
                if key != "spec_params" and template_config[key] != config[key]
            },
            "spec_params_nce": config["spec_params"]["nce"],
            "spec_params_orb_rel": config["spec_params"]["merge_tolerance_by_inst"]["ORB"]["rel"],
        }
        assert set(record[label]["changed_top_level"]) <= set(watched)
    assert record["legacy_key"]["changed_top_level"] == {
        "masster_max_h_transfer": [template_config["masster_max_h_transfer"], 2]
    }
    assert record["legacy_then_current_applied_in_order"]["changed_top_level"] == {
        "masster_head_dropout": [template_config["masster_head_dropout"], 0.5]
    }
    assert record["proposal_type_fast_gnn"]["changed_top_level"] == {
        "masster_proposal_method": [
            template_config["masster_proposal_method"],
            "connected_fragment_beam",
        ]
    }
    assert record["legacy_proposal_type_fast_gnn"] == record["proposal_type_fast_gnn"] | {
        "override": ["mass_shatter_proposal_type=fast_gnn"]
    }
    assert record["nested_dotted_key"]["spec_params_nce"] is True
    assert template_config["spec_params"]["nce"] is False
    assert record["deeper_dotted_key"]["spec_params_orb_rel"] == 0.5

    # The migration is model-type agnostic: under frag_gnn the legacy spelling is renamed too and
    # sets the (template-declared) MASSter key on a config that never reads it. Recorded as is.
    frag_gnn_base = _quiet(load_config, F.CONFIG_DIR, None, None, model_type="frag_gnn")
    other_model = {}
    for label, override in (
        ("legacy_key", ["mass_shatter_head_dropout=0.3"]),
        ("proposal_type_fast_gnn", ["masster_proposal_type=fast_gnn"]),
        ("plain_key", ["seed=5"]),
    ):
        config = _quiet(load_config, F.CONFIG_DIR, None, override, model_type="frag_gnn")
        assert config["model_type"] == "frag_gnn"
        assert _legacy_leftovers(config) == []
        other_model[label] = {
            "override": override,
            "changed": {
                key: [frag_gnn_base[key], config[key]]
                for key in sorted(frag_gnn_base)
                if frag_gnn_base[key] != config[key]
            },
        }
    assert set(other_model["legacy_key"]["changed"]) == {"masster_head_dropout"}
    assert other_model["legacy_key"]["changed"]["masster_head_dropout"][1] == 0.3
    assert other_model["proposal_type_fast_gnn"]["changed"] == {
        "masster_proposal_method": [
            frag_gnn_base["masster_proposal_method"],
            "connected_fragment_beam",
        ]
    }
    assert set(other_model["plain_key"]["changed"]) == {"seed"}
    record["frag_gnn_model_type"] = other_model

    errors = {
        "legacy_key_not_in_template": _error(
            lambda: load_config(
                F.CONFIG_DIR, None, ["mass_shatter_not_a_key=1"], model_type="masster"
            )
        ),
        "missing_section": _error(
            lambda: load_config(F.CONFIG_DIR, None, ["no_section.key=1"], model_type="masster")
        ),
        "retired_instrument_type": _error(
            lambda: load_config(
                F.CONFIG_DIR, None, ["spec_params.inst_types=[FT]"], model_type="masster"
            )
        ),
    }
    assert errors["legacy_key_not_in_template"] == "KeyError: 'masster_not_a_key'"
    assert errors["retired_instrument_type"].startswith("ValueError: config names a retired")
    errors["retired_instrument_type"] = errors["retired_instrument_type"].split(". ")[0]
    record["errors"] = errors
    ratchet_json(GOLDEN, "load_config:cli_overrides", record)


def test_load_config_of_c0_as_a_custom_config(tmp_path: Path, c0_orb: dict) -> None:
    """The legacy checkpoint hparams written as a custom config resolve against the template."""
    custom_fp = _write_yaml(tmp_path, "c0_custom.yml", c0_orb)
    config = _quiet(load_config, F.CONFIG_DIR, custom_fp, None)
    expected = copy.deepcopy(c0_orb)
    migrate_legacy_masster_name(expected)
    assert config["model_type"] == "masster"
    assert _legacy_leftovers(config) == []
    assert _contains(config, expected), "custom values win; nested dicts merge with the template"
    template = _quiet(load_config, F.CONFIG_DIR, None, None, model_type="masster")
    template_only = sorted(set(config) - set(expected))
    assert all(config[key] == template[key] for key in template_only)
    migrated_keys = (*SHARED_KEYS, *MASSTER_KEYS)
    # c0 predates the RT/CCS heads; load_config supplies the template's values for those.
    record = {
        "n_c0_keys": len(expected),
        "migration_keys_c0_lacks": sorted(key for key in migrated_keys if key not in expected),
        "migration_keys": _present(config, migrated_keys),
    }
    ratchet_json(GOLDEN, "load_config:c0_custom", record)


# ---------------------------------------------------------------------------------------------
# get_config_from_ckpt and W&B config
# ---------------------------------------------------------------------------------------------


def test_get_config_from_ckpt_of_legacy_c0(tmp_path: Path, c0_orb: dict) -> None:
    """Checkpoint hparams alone equal ``get_pl_hparams``; a custom config makes the template win."""
    ckpt_fp = str(tmp_path / "c0_stub.ckpt")
    th.save({"hyper_parameters": c0_orb}, ckpt_fp)
    migrated = _pl_hparams(c0_orb)

    assert _quiet(get_config_from_ckpt, ckpt_fp) == migrated

    custom_fp = _write_yaml(tmp_path, "custom.yml", {"masster_context_dim": 128})
    resolved = _quiet(get_config_from_ckpt, ckpt_fp, custom_fp, F.CONFIG_DIR)
    template = _quiet(load_config, F.CONFIG_DIR, None, None, model_type="masster")
    assert resolved == {**template, "masster_context_dim": 128}
    assert resolved["model_type"] == "masster"
    assert migrated["masster_head_dropout"] == 0.1
    record = {
        "masster_context_dim": resolved["masster_context_dim"],
        "model_type": resolved["model_type"],
        "head_dropout_checkpoint_vs_resolved": [
            migrated["masster_head_dropout"],
            resolved["masster_head_dropout"],
        ],
    }
    ratchet_json(GOLDEN, "get_config_from_ckpt:c0", record)


def test_legacy_wandb_run_config_merges_into_resolved_config() -> None:
    """A saved legacy W&B config is migrated, then merged into the resolved config.

    Mirrors ``runner.run_training`` (``migrate_legacy_masster_name``,
    ``migrate_masster_proposal_method``, ``merge_wandb_config`` on ``dict(wandb.config)``)
    without starting W&B.
    """
    config = _quiet(load_config, F.CONFIG_DIR, None, None, model_type="masster")
    before = copy.deepcopy(config)
    saved_run_config = {
        "model_type": "mass_shatter",
        "mass_shatter_head_dropout": 0.37,
        "masster_proposal_type": "fast_gnn",
        "mass_shatter_max_h_transfer": 2,
        "mass_shatter_key_only_in_wandb": 1,
        "spec_params": {"nce": False, "ace": True, "key_only_in_wandb": 1},
        "seed": 5,
    }
    migrate_legacy_masster_name(saved_run_config)
    migrate_masster_proposal_method(saved_run_config)
    _quiet(merge_wandb_config, config, saved_run_config)

    changed = {
        key: [before[key], config[key]]
        for key in sorted(before)
        if key != "spec_params" and before[key] != config[key]
    }
    spec_changed = {
        key: [before["spec_params"][key], config["spec_params"][key]]
        for key in sorted(before["spec_params"])
        if before["spec_params"][key] != config["spec_params"][key]
    }
    assert changed == {
        "masster_head_dropout": [before["masster_head_dropout"], 0.37],
        "masster_max_h_transfer": [before["masster_max_h_transfer"], 2],
        "masster_proposal_method": [before["masster_proposal_method"], "connected_fragment_beam"],
        "seed": [before["seed"], 5],
    }
    assert set(config) == set(before), "keys only W&B holds are not added"
    assert "key_only_in_wandb" not in config["spec_params"]
    assert _legacy_leftovers(config) == []
    ratchet_json(
        GOLDEN,
        "wandb_merge:legacy_run_config",
        {"changed_top_level": changed, "changed_spec_params": spec_changed},
    )


# ---------------------------------------------------------------------------------------------
# Model constructors
# ---------------------------------------------------------------------------------------------


def test_constructors_backfill_hparams_before_saving_them() -> None:
    """``SpectrumPL`` and ``MassterPL`` migrate kwargs before saving them.

    ``neims`` builds through the same ``SpectrumPL`` call as ``frag_gnn``, but its constructor
    needs more than the template (the NEIMS config trips an assertion), so only its
    ``get_pl_hparams`` record exists.
    """
    record = {}
    # FraGNNet: a current config with the migrated keys taken out, as an old checkpoint lacks them.
    fragnnet_kwargs = F.make_config("frag_gnn")
    for key in (*SHARED_KEYS, *FRAGNNET_KEYS):
        del fragnnet_kwargs[key]
    fragnnet_kwargs["spec_params"]["nce"] = True
    fragnnet_kwargs["spec_params"]["ace"] = False
    with F.deterministic():
        fragnnet = _quiet(get_model_cls("frag_gnn"), **fragnnet_kwargs)
    record["frag_gnn"] = _present(dict(fragnnet.hparams), (*SHARED_KEYS, *FRAGNNET_KEYS))
    assert ABSENT not in record["frag_gnn"].values()
    assert record["frag_gnn"]["ion_channel_rank"] == 32
    assert record["frag_gnn"]["mol_max_degree"] is None
    assert (record["frag_gnn"]["use_nce"], record["frag_gnn"]["use_ace"]) == (True, False)

    # MASSter: the pre-rename spelling of a key, of the model type and of the proposal method.
    # `fast_gnn` is the old name of `connected_fragment_beam`, which needs unique fragment mode.
    masster_kwargs = F.make_config("masster", None, {"masster_unique_fragment_mode": True})
    del masster_kwargs["masster_head_dropout"]
    del masster_kwargs["masster_proposal_method"]
    for key in SHARED_KEYS:
        del masster_kwargs[key]
    masster_kwargs["model_type"] = "mass_shatter"
    masster_kwargs["mass_shatter_head_dropout"] = 0.37
    masster_kwargs["masster_proposal_type"] = "fast_gnn"
    with F.deterministic():
        masster = _quiet(get_model_cls("masster"), **masster_kwargs)
    hparams = dict(masster.hparams)
    assert hparams["model_type"] == "masster"
    assert hparams["masster_head_dropout"] == 0.37
    assert hparams["masster_proposal_method"] == "connected_fragment_beam"
    assert _legacy_leftovers(hparams) == []
    record["masster"] = {
        "model_type": hparams["model_type"],
        "masster_head_dropout": hparams["masster_head_dropout"],
        "masster_proposal_method": hparams["masster_proposal_method"],
        **_present(hparams, SHARED_KEYS),
    }
    assert record["masster"]["report_metric_std"] is False

    ratchet_json(GOLDEN, "constructor_backfill", record)


def _write_old_masster_ckpt(
    path: Path,
    drop: tuple[str, ...] = (),
    drop_spec_params: tuple[str, ...] = (),
    set_keys: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    """Write a MASSter checkpoint the way an older release saved it.

    Builds a current module from ``config``, takes its hparams, removes ``drop`` (top level) and
    ``drop_spec_params`` (inside ``spec_params``), applies ``set_keys`` and saves the module's
    weights with those hparams. No real checkpoint is needed: ``load_from_checkpoint`` reads the
    weights and ``hyper_parameters`` only.

    Args:
        path: Where to write the checkpoint.
        drop: Top-level hparams an older release did not have.
        drop_spec_params: ``spec_params`` keys an older release did not have.
        set_keys: Hparams to set after dropping, e.g. old spellings or explicit values.
        config: Overrides of the MASSter test config the module is built from.

    Returns:
        The checkpoint path as a string.
    """
    with F.deterministic():
        module = F.build_module(F.make_config("masster", None, config or {}))
    hparams = copy.deepcopy(dict(module.hparams))
    for key in drop:
        del hparams[key]
    for key in drop_spec_params:
        del hparams["spec_params"][key]
    hparams.update(set_keys or {})
    ckpt = {
        "state_dict": module.state_dict(),
        "hyper_parameters": hparams,
        "pytorch-lightning_version": "2.0.0",
    }
    module.on_save_checkpoint(ckpt)
    th.save(ckpt, path)
    return str(path)


def test_masster_load_from_checkpoint_backfills_legacy_keys(tmp_path: Path) -> None:
    """``MassterPL.load_from_checkpoint`` supplies the keys an old checkpoint lacks.

    This is the real load path of a legacy checkpoint (a private marker makes the constructor
    backfill the candidate backend, proposal method, beam settings, fragment selection and RT/CCS
    keys), and its values differ from today's template (boundary cuts 3 against 5). Saved and
    explicitly passed values keep their precedence.
    """
    cls = get_model_cls("masster")
    old_spec_keys = tuple(key for key in SPEC_PARAMS_KEYS if key != "merge_ce_aggregation")
    load_keys = (*MASSTER_CANDIDATE_KEYS, *MASSTER_RT_CCS_KEYS)

    def load(path: str, **kwargs: Any) -> dict:
        with F.deterministic():
            module = _quiet(cls.load_from_checkpoint, path, map_location="cpu", **kwargs)
        hparams = dict(module.hparams)
        assert _legacy_leftovers(hparams) == []
        return {
            "model_type": hparams["model_type"],
            "keys": _present(hparams, load_keys),
            "spec_params": _present(hparams["spec_params"], old_spec_keys),
        }

    records = {}
    # 1. Everything the load path backfills is missing.
    path = _write_old_masster_ckpt(
        tmp_path / "old.ckpt", drop=load_keys, drop_spec_params=old_spec_keys
    )
    records["all_keys_missing"] = load(path)
    # 2. Explicit values saved in the checkpoint win over the backfill. Two checkpoints between
    # them hold a non-default value for each of the six keys (marginal selection needs its own
    # chain of settings; the beam proposal excludes it).
    unique = {"masster_unique_fragment_mode": True, "masster_fast_gnn_beam_width": 64}
    path = _write_old_masster_ckpt(
        tmp_path / "explicit_marginal.ckpt",
        drop=MASSTER_RT_CCS_KEYS,
        drop_spec_params=old_spec_keys,
        config={
            **unique,
            "masster_candidate_backend": "cuda",
            "masster_fast_gnn_max_boundary_cuts": 2,
            "masster_fragment_selection": "marginal",
            "masster_selected_fragment_target": 16,
            "masster_proposal_selection_only": True,
            "masster_proposal_anchored": True,
            "masster_proposal_loss_weight": 0.1,
        },
    )
    records["explicit_cuda_marginal_kept"] = load(path)
    path = _write_old_masster_ckpt(
        tmp_path / "explicit_beam.ckpt",
        drop=MASSTER_RT_CCS_KEYS,
        drop_spec_params=old_spec_keys,
        config={
            **unique,
            "masster_proposal_method": "connected_fragment_beam",
            "masster_fast_gnn_max_boundary_cuts": 2,
        },
    )
    records["explicit_beam_proposal_kept"] = load(path)
    # 3. A keyword passed to the loader outranks the saved and the backfilled value alike.
    records["keyword_override"] = load(
        _write_old_masster_ckpt(
            tmp_path / "kw.ckpt", drop=load_keys, drop_spec_params=old_spec_keys
        ),
        masster_fast_gnn_beam_width=128,
    )
    # 4. The pre-rename spellings of the model type, a key and the proposal method.
    path = _write_old_masster_ckpt(
        tmp_path / "spelling.ckpt",
        drop=(*load_keys, "masster_head_dropout"),
        drop_spec_params=old_spec_keys,
        set_keys={
            "model_type": "mass_shatter",
            "mass_shatter_head_dropout": 0.0,
            "mass_shatter_proposal_type": "cut_masks",
        },
    )
    records["legacy_spellings"] = load(path)

    missing = records["all_keys_missing"]
    assert missing["model_type"] == "masster"
    assert missing["keys"] == {
        "masster_candidate_backend": "cpu",
        "masster_proposal_method": "cut_masks",
        "masster_fast_gnn_beam_width": 256,
        "masster_fast_gnn_max_boundary_cuts": 3,
        "masster_fragment_selection": "mask_order",
        "masster_selected_fragment_target": None,
        "masster_predict_rt_ccs": False,
        "masster_predict_rt": False,
        "masster_predict_ccs": False,
        "masster_rt_loss_weight": 0.1,
        "masster_ccs_loss_weight": 0.1,
        "masster_rt_scale": 60.0,
        "masster_ccs_scale": 100.0,
    }
    assert missing["spec_params"] == {
        "rt": False,
        "rt_labels_fp": None,
        "ccs": False,
        "ccs_labels_fp": None,
        "rt_ccs": False,
        "rt_ccs_labels_fp": None,
    }
    marginal = records["explicit_cuda_marginal_kept"]["keys"]
    assert marginal["masster_candidate_backend"] == "cuda"
    assert marginal["masster_fragment_selection"] == "marginal"
    assert marginal["masster_selected_fragment_target"] == 16
    assert marginal["masster_fast_gnn_beam_width"] == 64
    assert marginal["masster_fast_gnn_max_boundary_cuts"] == 2
    assert marginal["masster_rt_scale"] == 60.0, "keys the checkpoint lacks are still backfilled"
    beam = records["explicit_beam_proposal_kept"]["keys"]
    assert beam["masster_proposal_method"] == "connected_fragment_beam"
    assert beam["masster_candidate_backend"] == "cpu"
    assert (beam["masster_fast_gnn_beam_width"], beam["masster_fast_gnn_max_boundary_cuts"]) == (
        64,
        2,
    )
    keyword = records["keyword_override"]["keys"]
    assert keyword["masster_fast_gnn_beam_width"] == 128
    assert keyword["masster_fast_gnn_max_boundary_cuts"] == 3
    spelling = records["legacy_spellings"]
    assert spelling["model_type"] == "masster"
    assert spelling["keys"]["masster_proposal_method"] == "cut_masks"
    assert spelling["keys"]["masster_candidate_backend"] == "cpu"
    ratchet_json(GOLDEN, "masster_load_from_checkpoint", records)
