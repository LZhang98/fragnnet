"""One config template for every model, plus recursive custom-config validation.

``config/template.yml`` declares every model's keys, so shared code can read any key without
checking ``model_type`` first and a key can never go missing for one model only. The layout
tests below are what keeps the file readable now that it is not split per model: each model's
keys have to sit in that model's section. See ``config/README.md``.
"""

import glob
import os
import re

import pytest
import yaml

from fragnnet.registry import RETIRED_MODEL_TYPES
from fragnnet.runner import (
    VALID_MODEL_TYPES,
    find_unknown_config_keys,
    load_config,
    resolve_template_path,
)

TEMPLATE_DIR = "config"
TEMPLATE_FP = os.path.join(TEMPLATE_DIR, "template.yml")

# Section title -> (the prefix its keys carry, the unprefixed keys that also belong there).
# Adding an unprefixed key to a model's section means declaring it here, which is the moment to
# ask whether it should be prefixed instead.
MODEL_SECTIONS = {
    "MASSter only": ("masster_", ()),
    "ICEBERG only": ("iceberg_", ("magma_params", "magma_dp")),
    "MassFormer only": ("mf_", ()),
    "NEIMS only": ("neims_", ()),
    "ChemLM only": ("chemlm_", ()),
    "GrAFF only": ("graff_", ("ann_params", "ann_fp")),
    "CLIP pretraining only": ("clip_", ("spectrum_encoder_params", "retrieval_eval_size")),
    "MCES pretraining only": ("mces_", ()),
}
# FraGNNet's keys are not consistently prefixed (frag_*, cc_*, nb_*, entropy weights, ...), so
# its section is checked only for keys that belong to another model.
FRAGNNET_SECTION = "FraGNNet only"
CORE_SECTION = "shared core"
# These run entirely on the shared core and have no section of their own.
CORE_ONLY_MODEL_TYPES = frozenset({"gnn", "precursor"})


def _sections() -> dict[str, list[str]]:
    """Map section title to the top-level keys declared under it, in file order."""
    sections: dict[str, list[str]] = {CORE_SECTION: []}
    current, after_banner = CORE_SECTION, False
    with open(TEMPLATE_FP) as template_file:
        lines = template_file.read().split("\n")
    for line in lines:
        if line.startswith("# ===="):
            after_banner = True
            continue
        if after_banner and line.startswith("# ") and line.endswith(" only"):
            current = line[2:]
            sections.setdefault(current, [])
        after_banner = False
        key = re.match(r"^([A-Za-z_][A-Za-z_0-9]*):", line)
        if key:
            sections[current].append(key.group(1))
    return sections


@pytest.fixture(scope="module")
def sections() -> dict[str, list[str]]:
    return _sections()


@pytest.fixture(scope="module")
def template() -> dict:
    with open(TEMPLATE_FP) as template_file:
        return yaml.safe_load(template_file)


class TestTemplateLayout:
    def test_the_template_declares_every_models_keys(self, template):
        """One schema: no model can lose a key another model's code reads."""
        for key in ("frag_params", "magma_params", "ann_params", "mf_flag", "spec_params"):
            assert key in template, key
        assert any(key.startswith("masster_") for key in template)
        assert any(key.startswith("iceberg_") for key in template)

    def test_no_duplicate_top_level_keys(self, sections):
        """YAML keeps the last of a repeated key, silently discarding the first."""
        declared = [key for keys in sections.values() for key in keys]
        duplicates = sorted({key for key in declared if declared.count(key) > 1})
        assert duplicates == []

    def test_every_model_section_appears_exactly_once(self, sections):
        expected = {CORE_SECTION, FRAGNNET_SECTION, *MODEL_SECTIONS}
        assert set(sections) == expected

    def test_a_models_keys_live_only_in_its_own_section(self, sections):
        """The readability rule, enforced: a prefixed key belongs to its model's section."""
        for title, (prefix, _) in MODEL_SECTIONS.items():
            for other_title, keys in sections.items():
                stray = [key for key in keys if key.startswith(prefix)]
                if other_title == title:
                    assert stray, f"{title} declares no {prefix}* key"
                else:
                    assert stray == [], f"{other_title} carries {title}'s keys: {stray}"

    def test_model_sections_declare_their_unprefixed_keys(self, sections):
        for title, (prefix, unprefixed) in MODEL_SECTIONS.items():
            found = tuple(key for key in sections[title] if not key.startswith(prefix))
            assert found == unprefixed, f"{title}: undeclared unprefixed keys"

    def test_the_fragnnet_section_holds_no_other_models_keys(self, sections):
        prefixes = tuple(prefix for prefix, _ in MODEL_SECTIONS.values())
        stray = [key for key in sections[FRAGNNET_SECTION] if key.startswith(prefixes)]
        assert stray == []

    def test_every_model_type_is_accounted_for(self, sections):
        """Every known model either owns a section or is declared core-only."""
        owned = {
            "masster",
            "massformer",
            "neims",
            "graff",
            "frag_gnn",
            "fragnnet",
        }
        assert owned | CORE_ONLY_MODEL_TYPES == set(VALID_MODEL_TYPES)

    def test_retired_model_types_are_not_valid(self):
        """Retired types keep their inert template section but cannot be selected."""
        assert {
            "iceberg_gen",
            "iceberg_inten",
            "mces_pretraining",
            "chemlm",
            "clip",
            "spec_mol_clip_pretraining",
        } == set(RETIRED_MODEL_TYPES)
        assert not set(RETIRED_MODEL_TYPES) & set(VALID_MODEL_TYPES)


class TestResolution:
    def test_a_directory_resolves_to_the_template(self):
        assert resolve_template_path(TEMPLATE_DIR) == TEMPLATE_FP

    def test_a_single_template_file_is_used_directly(self, tmp_path):
        """How a template saved inside a checkpoint still resolves."""
        template = tmp_path / "saved_template.yml"
        template.write_text(yaml.safe_dump({"model_type": "masster", "lr": 0.25}))

        assert load_config(str(template), None, None)["lr"] == 0.25

    def test_every_model_resolves_the_same_schema(self):
        """What the per-model split could not give: no model is missing another's key."""
        frag = load_config(TEMPLATE_DIR, None, None, model_type="frag_gnn")
        shatter = load_config(TEMPLATE_DIR, None, None, model_type="masster")

        assert set(frag) == set(shatter)
        assert "masster_max_h_transfer" in frag
        assert "frag_params" in shatter

    def test_custom_model_type_wins(self, tmp_path):
        custom = tmp_path / "custom.yml"
        custom.write_text(yaml.safe_dump({"model_type": "masster", "lr": 0.5}))

        resolved = load_config(TEMPLATE_DIR, str(custom), None)

        assert resolved["model_type"] == "masster"
        assert resolved["lr"] == 0.5

    def test_checkpoint_model_type_outranks_the_template_default(self, tmp_path):
        """A NEIMS checkpoint plus a config that stays silent must not become frag_gnn."""
        custom = tmp_path / "custom.yml"
        custom.write_text(yaml.safe_dump({"lr": 0.5}))

        resolved = load_config(TEMPLATE_DIR, str(custom), None, model_type="neims")

        assert resolved["model_type"] == "neims"

    def test_a_silent_config_means_fragnnet(self, tmp_path):
        custom = tmp_path / "custom.yml"
        custom.write_text(yaml.safe_dump({"lr": 0.5}))

        assert load_config(TEMPLATE_DIR, str(custom), None)["model_type"] == "frag_gnn"

    def test_unknown_model_type_is_rejected(self, tmp_path):
        custom = tmp_path / "custom.yml"
        custom.write_text(yaml.safe_dump({"model_type": "no_such_model"}))

        with pytest.raises(ValueError, match="unknown model_type"):
            load_config(TEMPLATE_DIR, str(custom), None)

    def test_dotted_override_reaches_nested_sections(self):
        resolved = load_config(
            TEMPLATE_DIR,
            None,
            ["ce_insert_type=embed", "spec_params.nce=true"],
            model_type="masster",
        )

        assert resolved["ce_insert_type"] == "embed"
        assert resolved["spec_params"]["nce"] is True


class TestValidation:
    def test_top_level_typo_is_rejected(self, tmp_path):
        custom = tmp_path / "custom.yml"
        custom.write_text(yaml.safe_dump({"model_type": "masster", "lrr": 1.0}))

        with pytest.raises(ValueError, match="does not declare.*lrr"):
            load_config(TEMPLATE_DIR, str(custom), None)

    def test_nested_typo_is_rejected(self, tmp_path):
        """A typo inside a section fails as loudly as one at the top level."""
        custom = tmp_path / "custom.yml"
        custom.write_text(yaml.safe_dump({"model_type": "masster", "spec_params": {"ncee": True}}))

        with pytest.raises(ValueError, match=r"spec_params\.ncee"):
            load_config(TEMPLATE_DIR, str(custom), None)

    def test_another_models_key_is_simply_inert(self, tmp_path):
        """One schema: a historical config carrying another model's key is valid, not a typo."""
        custom = tmp_path / "custom.yml"
        custom.write_text(yaml.safe_dump({"model_type": "neims", "masster_batch_sampler": False}))

        resolved = load_config(TEMPLATE_DIR, str(custom), None)

        assert resolved["model_type"] == "neims"
        assert resolved["masster_batch_sampler"] is False

    def test_a_staged_template_validates_the_same_way(self, tmp_path):
        """A cluster job stages one file; validation no longer depends on its neighbours."""
        staged = tmp_path / "config"
        staged.mkdir()
        with open(TEMPLATE_FP) as template_file:
            staged.joinpath("template.yml").write_text(template_file.read())
        custom = tmp_path / "custom.yml"
        custom.write_text(yaml.safe_dump({"model_type": "frag_gnn", "lrr": 1.0}))

        with pytest.raises(ValueError, match="does not declare.*lrr"):
            load_config(str(staged), str(custom), None)

    def test_open_mappings_accept_new_sub_keys(self, tmp_path):
        """Instrument and split names are data, not schema."""
        custom = tmp_path / "custom.yml"
        custom.write_text(
            yaml.safe_dump(
                {
                    "model_type": "masster",
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

    def test_find_unknown_config_keys_reports_dotted_paths(self):
        unknown = find_unknown_config_keys({"a": 1, "b": {"c": 2, "d": 3}}, {"b": {"c": 0}})

        assert unknown == ["a", "b.d"]


# Configs that set keys no template has ever declared and were already rejected before the
# template layout change, or undeclared nested sub-keys that only recursive validation catches.
# There were 18; 5 were MCES fine-tuning configs and 3 were CLIP configs, deleted with those models
# on 2026-10-12. Each must keep failing for exactly this reason.
KNOWN_REJECTED_CONFIGS = {
    "config/debug/debug_tiny_new_frag_mol_feats.yml",
    *(
        "config/masster/legacy/bond_size_curriculum/nist20_merged_" + name + "_bond_curriculum.yml"
        for name in (
            "mask256_gine_gsa384_bondcentric_a2_b3_explore256",
            "mask512_gine_gsa384_bondcentric_a2_b3_transformer_explore512",
            "mask512_gine_gsa384_bondcentric_a2_b3_transformer_logmeanexp_explore512",
        )
    ),
    *(
        "config/masster/legacy/explore/nist20_merged_" + name + ".yml"
        for name in (
            "mask256_gine_gsa384_bondcentric_a2_b3_explore256",
            "mask256_gine_gsa384_bondcentric_a2_b3_explore64",
            "mask512_gine_gsa384_bondcentric_a2_b3_transformer_explore512",
            "mask512_gine_gsa384_bondcentric_a2_b3_transformer_logmeanexp_explore256",
            "mask512_gine_gsa384_bondcentric_a2_b3_transformer_logmeanexp_explore512",
            "mask512_gine_gsa384_bondcentric_a2_b3_transformer_logmeanexp_explore512_layernorm",
        )
    ),
}

# Pre-existing configs that are not valid YAML, so never reach the template check.
KNOWN_MALFORMED_CONFIGS = {
    "config/ablations_merged/fraggnn_d3_edges/s1.yml",
    "config/ablations_merged/fraggnn_d3_edges/s2.yml",
    "config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_sqrt.yml",
}


def test_every_tracked_config_still_resolves():
    """No config in the repository may be broken by the layout change."""
    failures = {}
    malformed = set()
    for path in sorted(glob.glob("config/**/*.yml", recursive=True)):
        if os.path.abspath(path) == os.path.abspath(TEMPLATE_FP):
            continue
        try:
            with open(path) as config_file:
                custom = yaml.safe_load(config_file)
        except yaml.YAMLError:
            malformed.add(path)
            continue
        if not isinstance(custom, dict):
            continue
        if custom.get("model_type", "frag_gnn") not in VALID_MODEL_TYPES:
            continue
        try:
            load_config(TEMPLATE_DIR, path, None)
        except ValueError as error:
            failures[path] = str(error)
    assert malformed == KNOWN_MALFORMED_CONFIGS
    # Same set, not the same count: a newly rejected config cannot hide behind a fixed one.
    assert set(failures) == KNOWN_REJECTED_CONFIGS
    for path, message in failures.items():
        assert "sets keys the template does not declare" in message, (path, message)
