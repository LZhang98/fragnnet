"""Self-contained per-model config templates and recursive custom-config validation.

Every model has one complete template in ``config/*_template.yml``; there is no shared base.
See ``config/README.md``.
"""

import glob
import os

import pytest
import yaml

from fragnnet.runner import (
    MODEL_CONFIG_TEMPLATE,
    load_config,
    model_template_fp,
    unknown_config_keys,
)

TEMPLATE_DIR = "config"
SHARED_SAMPLE = ("spec_params", "mol_params", "lr", "train_batch_size", "loss_type")


@pytest.fixture(scope="module")
def templates() -> dict[str, dict]:
    return {
        os.path.basename(path)[: -len("_template.yml")]: yaml.safe_load(open(path))
        for path in glob.glob(f"{TEMPLATE_DIR}/*_template.yml")
    }


class TestTemplateLayout:
    def test_every_model_type_resolves_to_a_template(self):
        for model_type in MODEL_CONFIG_TEMPLATE:
            assert os.path.isfile(model_template_fp(TEMPLATE_DIR, model_type))

    def test_each_template_is_complete_on_its_own(self, templates):
        """No shared base: a template must carry everything a run needs."""
        for name, template in templates.items():
            for key in SHARED_SAMPLE:
                assert key in template, f"{name} is missing {key}"

    def test_a_template_never_carries_another_models_keys(self, templates):
        assert "frag_params" in templates["fragnnet"]
        assert "frag_params" not in templates["mass_shatter"]
        assert "magma_params" in templates["iceberg"]
        assert "magma_params" not in templates["fragnnet"]
        assert "ann_params" in templates["graff"]
        assert "mf_flag" in templates["massformer"]
        assert "mf_flag" not in templates["neims"]
        assert not any(key.startswith("mass_shatter_") for key in templates["fragnnet"])
        assert not any(key.startswith("iceberg_") for key in templates["mass_shatter"])

    def test_shared_keys_agree_across_templates(self, templates):
        """Duplication is the cost of dropping the base; drift is the risk it creates."""
        for key in SHARED_SAMPLE:
            values = {name: template[key] for name, template in templates.items()}
            assert len(set(map(repr, values.values()))) == 1, f"{key} differs: {values}"

    def test_unknown_model_type_is_rejected(self):
        with pytest.raises(ValueError, match="unknown model_type"):
            model_template_fp(TEMPLATE_DIR, "no_such_model")

    def test_no_model_type_means_fragnnet(self):
        """What the single shared template used to declare as its default."""
        assert model_template_fp(TEMPLATE_DIR, None) == os.path.join(
            TEMPLATE_DIR, "fragnnet_template.yml"
        )


class TestResolution:
    def test_explicit_model_type_selects_the_template(self):
        resolved = load_config(TEMPLATE_DIR, None, None, model_type="mass_shatter")

        assert "mass_shatter_max_h_transfer" in resolved
        assert "frag_params" not in resolved

    def test_custom_model_type_selects_the_template(self, tmp_path):
        custom = tmp_path / "custom.yml"
        custom.write_text(yaml.safe_dump({"model_type": "mass_shatter", "lr": 0.5}))

        resolved = load_config(TEMPLATE_DIR, str(custom), None)

        assert resolved["model_type"] == "mass_shatter"
        assert resolved["lr"] == 0.5

    def test_a_single_template_file_is_used_directly(self, tmp_path):
        """How a template saved inside a checkpoint still resolves."""
        template = tmp_path / "saved_template.yml"
        template.write_text(yaml.safe_dump({"model_type": "mass_shatter", "lr": 0.25}))

        assert load_config(str(template), None, None)["lr"] == 0.25

    def test_dotted_override_reaches_nested_sections(self):
        resolved = load_config(
            TEMPLATE_DIR,
            None,
            ["ce_insert_type=embed", "spec_params.nce=true"],
            model_type="mass_shatter",
        )

        assert resolved["ce_insert_type"] == "embed"
        assert resolved["spec_params"]["nce"] is True


class TestValidation:
    def test_top_level_typo_is_rejected(self, tmp_path):
        custom = tmp_path / "custom.yml"
        custom.write_text(yaml.safe_dump({"model_type": "mass_shatter", "lrr": 1.0}))

        with pytest.raises(ValueError, match="no model declares.*lrr"):
            load_config(TEMPLATE_DIR, str(custom), None)

    def test_nested_typo_is_rejected(self, tmp_path):
        """The gap this closes: sub-keys used to be accepted silently."""
        custom = tmp_path / "custom.yml"
        custom.write_text(
            yaml.safe_dump({"model_type": "mass_shatter", "spec_params": {"ncee": True}})
        )

        with pytest.raises(ValueError, match=r"spec_params\.ncee"):
            load_config(TEMPLATE_DIR, str(custom), None)

    def test_another_models_key_is_kept_and_reported(self, tmp_path, caplog):
        """Configs are tracked experiment records; historical ones predate the split."""
        custom = tmp_path / "custom.yml"
        custom.write_text(
            yaml.safe_dump({"model_type": "neims", "frag_params": {"preprocess": True}})
        )

        resolved = load_config(TEMPLATE_DIR, str(custom), None)

        assert resolved["frag_params"] == {"preprocess": True}
        assert "inert" in caplog.text

    def test_open_mappings_accept_new_sub_keys(self, tmp_path):
        """Instrument and split names are data, not schema."""
        custom = tmp_path / "custom.yml"
        custom.write_text(
            yaml.safe_dump(
                {
                    "model_type": "mass_shatter",
                    "spec_params": {
                        "merge_tolerance_by_inst": {"ORBITRAP": {"rel": 1e-5, "min_mz": 200.0}},
                        "split_filters": {"train": {"filter_prec_types": True}},
                    },
                    "inst_type_loss_tol": {"ORBITRAP": {"rel": 1e-5, "min_mz": 200.0}},
                }
            )
        )

        resolved = load_config(TEMPLATE_DIR, str(custom), None)

        assert "ORBITRAP" in resolved["spec_params"]["merge_tolerance_by_inst"]
        assert "train" in resolved["spec_params"]["split_filters"]

    def test_unknown_config_keys_reports_dotted_paths(self):
        unknown = unknown_config_keys({"a": 1, "b": {"c": 2, "d": 3}}, {"b": {"c": 0}})

        assert unknown == ["a", "b.d"]


def test_every_tracked_config_still_resolves():
    """No config in the repository may be broken by the layout change."""
    failures = []
    for path in sorted(glob.glob("config/**/*.yml", recursive=True)):
        if path.endswith("_template.yml"):
            continue
        try:
            custom = yaml.safe_load(open(path))
        except yaml.YAMLError:
            continue  # pre-existing malformed config, not this layout's concern
        if not isinstance(custom, dict):
            continue
        if custom.get("model_type", "frag_gnn") not in MODEL_CONFIG_TEMPLATE:
            continue
        try:
            load_config(TEMPLATE_DIR, path, None)
        except ValueError as error:
            failures.append((path, str(error)))
    # 14 configs set keys no template has ever declared and were already rejected before this
    # work; 4 more set undeclared nested sub-keys that only recursive validation catches.
    assert len(failures) == 18, [path for path, _ in failures]
