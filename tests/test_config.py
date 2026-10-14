"""Tests for ``fragnnet.config``: CLI overrides through the shared migrations and checks."""

import pytest

from fragnnet.config import apply_override, load_config, merge_wandb_config, migrate_config_dict


@pytest.fixture(scope="module")
def masster_config() -> dict:
    return load_config("config", None, None, model_type="masster")


@pytest.fixture
def config(masster_config) -> dict:
    return {
        **masster_config,
        "spec_params": dict(masster_config["spec_params"]),
        "inst_type_loss_tol": dict(masster_config["inst_type_loss_tol"] or {}),
    }


def test_top_level_and_nested_values_parse_as_yaml(config) -> None:
    apply_override(config, "lr=0.0002")
    apply_override(config, "spec_params.nce=False")
    assert config["lr"] == pytest.approx(2e-4)
    assert config["spec_params"]["nce"] is False


def test_legacy_masster_spelling_still_applies(config) -> None:
    apply_override(config, "mass_shatter_max_h_transfer=3")
    assert config["masster_max_h_transfer"] == 3
    assert "mass_shatter_max_h_transfer" not in config


def test_original_proposal_key_and_value_are_renamed(config) -> None:
    apply_override(config, "masster_proposal_type=fast_gnn")
    assert config["masster_proposal_method"] == "connected_fragment_beam"
    assert "masster_proposal_type" not in config


def test_disabled_retired_objective_is_dropped(config) -> None:
    before = dict(config)
    apply_override(config, "ce_pair_loss_weight=0.0")
    assert config == before


def test_enabled_retired_objective_raises(config) -> None:
    with pytest.raises(ValueError, match="retired BCE objective"):
        apply_override(config, "ce_pair_loss_weight=0.5")


def test_open_mapping_accepts_a_new_data_key(config) -> None:
    apply_override(config, "inst_type_loss_tol.QTOF={rel: 1.0e-5, min_mz: 200.0}")
    assert config["inst_type_loss_tol"]["QTOF"] == {"rel": 1.0e-5, "min_mz": 200.0}


@pytest.mark.parametrize(
    "item,message",
    [
        ("lr_typo=1", "does not declare"),
        ("spec_params.typo=1", "does not declare"),
        ("no_such_section.key=1", "is not a config section"),
        ("lr.sub=1", "is not a config section"),
        ("lr", "is not key=value"),
    ],
)
def test_bad_overrides_raise(config, item: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        apply_override(config, item)


def test_load_config_applies_overrides_and_rejects_retired_instruments() -> None:
    config = load_config("config", None, ["seed=7"])
    assert config["seed"] == 7
    with pytest.raises(ValueError, match="retired instrument type"):
        load_config("config", None, ["spec_params.inst_types=[FT]"])


def test_migrate_config_dict_runs_every_migration() -> None:
    config = {
        "model_type": "mass_shatter",
        "mass_shatter_head_dropout": 0.2,
        "masster_proposal_type": "fast_gnn",
        "precursor_pair_loss_weight": 0.0,
    }
    migrate_config_dict(config)
    assert config == {
        "model_type": "masster",
        "masster_head_dropout": 0.2,
        "masster_proposal_method": "connected_fragment_beam",
    }


@pytest.mark.parametrize("item", ["spec_params={typo: true}", "frag_params={typo: true}"])
def test_section_overrides_reject_unknown_keys(item) -> None:
    with pytest.raises(ValueError, match="does not declare"):
        load_config("config", None, [item])


@pytest.mark.parametrize("value", ["false", "null", "[]", "1"])
def test_section_overrides_require_a_mapping(value) -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        load_config("config", None, [f"spec_params={value}"])


def test_section_overrides_merge_without_removing_required_settings(config) -> None:
    before = dict(config["spec_params"])
    apply_override(config, "spec_params={nce: false}")
    assert config["spec_params"] == {**before, "nce": False}


@pytest.mark.parametrize("model_type", ["not_a_model", "iceberg_gen", "mces_pretraining"])
def test_model_override_is_validated(model_type) -> None:
    with pytest.raises(ValueError, match="unknown model_type|retired"):
        load_config("config", None, [f"model_type={model_type}"])


def test_last_model_override_selects_migration_before_other_overrides(tmp_path, monkeypatch) -> None:
    from fragnnet import registry

    custom = tmp_path / "custom.yml"
    custom.write_text("model_type: frag_gnn\n")
    hooks = registry.get_model_hooks("masster")
    migrated_types = []
    original = hooks.migrate_template_config

    def migrate(config):
        migrated_types.append(config["model_type"])
        original(config)

    monkeypatch.setattr(hooks, "migrate_template_config", migrate)
    resolved = load_config(
        "config", str(custom), ["model_type=neims", "model_type=mass_shatter"]
    )
    assert resolved["model_type"] == "masster"
    assert migrated_types == ["masster"]


def test_wandb_merges_new_open_entries_and_preserves_other_settings() -> None:
    config = {
        "inst_type_loss_tol": {"ORB": {"rel": 1e-5, "min_mz": 200.0}},
        "ce_context_ce_key_by_inst": {"ORB": "nce"},
        "spec_params": {
            "merge_tolerance_by_inst": {"ORB": {"abs": 0.01}},
            "split_filters": {"train": {"filter_prec_types": False}},
            "nce": True,
        },
        "eval_mz_bin_res": [1.0],
    }
    saved = {
        "inst_type_loss_tol": {"ORB": {"abs": 0.02}, "QTOF": {"rel": 2e-5, "min_mz": 750}},
        "ce_context_ce_key_by_inst": {"QTOF": "ace"},
        "spec_params": {
            "merge_tolerance_by_inst": {"QTOF": {"abs": 0.03}},
            "split_filters": {"val": {"filter_prec_types": True}},
        },
        "eval_mz_bin_res": [1],
    }
    merge_wandb_config(config, saved)
    assert config["inst_type_loss_tol"] == {
        "ORB": {"rel": 1e-5, "min_mz": 200.0, "abs": 0.02},
        "QTOF": {"rel": 2e-5, "min_mz": 750},
    }
    assert config["ce_context_ce_key_by_inst"] == {"ORB": "nce", "QTOF": "ace"}
    assert config["spec_params"] == {
        "nce": True,
        "merge_tolerance_by_inst": {"ORB": {"abs": 0.01}, "QTOF": {"abs": 0.03}},
        "split_filters": {
            "train": {"filter_prec_types": False}, "val": {"filter_prec_types": True}
        },
    }
    assert isinstance(config["eval_mz_bin_res"][0], float)
    assert saved["inst_type_loss_tol"]["ORB"] == {"abs": 0.02}


def test_resumed_wandb_config_skips_nested_retired_keys() -> None:
    config = {"spec_params": {"nce": True, "ace": False}}
    merge_wandb_config(config, {"spec_params": {"nce": False, "removed": True}}, resumed=True)
    assert config == {"spec_params": {"nce": False, "ace": False}}
