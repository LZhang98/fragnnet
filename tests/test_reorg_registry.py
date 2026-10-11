"""Registry coverage for the repository reorganization (plan Phase 0 item 1).

The expected mapping below is the contract every later phase must keep. It changes only
deliberately, in the same PR as the code:

- Phase 1 replaces the ICEBERG rows with retirement-error checks and maps ``massformer`` to
  ``SpecMolMFDataset`` (equivalence shown by the Phase 0 baseline records).
- Phase 6 changes module paths only; class names and behavior stay.

The resolved configs of every model type are also recorded, as one shared base plus per-type
differences, so a registry or migration refactor cannot silently change what ``load_config``
returns.
"""

from __future__ import annotations

import pytest
import yaml
from reorg_golden import ratchet_json
from reorg_layers import MODEL_TYPES

from fragnnet.runner import load_config
from fragnnet.utils.setup_utils import get_dataset_cls, get_model_cls

TEMPLATE_DIR = "config"

# model_type -> (Lightning module, dataset class), as "module:QualName".
EXPECTED_MAPPING = {
    "frag_gnn": (
        "fragnnet.pl_model.fragnnet_pl:FraGNNetPL",
        "fragnnet.dataset.spec_mol_frag_dataset:SpecMolFragDataset",
    ),
    "fragnnet": (
        "fragnnet.pl_model.fragnnet_pl:FraGNNetPL",
        "fragnnet.dataset.spec_mol_frag_dataset:SpecMolFragDataset",
    ),
    "masster": (
        "fragnnet.pl_model.masster_pl:MassterPL",
        "fragnnet.dataset.spec_mol_dataset:SpecMolDataset",
    ),
    "neims": (
        "fragnnet.pl_model.neims_pl:NeimsPL",
        "fragnnet.dataset.spec_mol_dataset:SpecMolDataset",
    ),
    "chemlm": (
        "fragnnet.pl_model.chemlm_pl:ChemLMPL",
        "fragnnet.dataset.spec_mol_dataset:SpecMolDataset",
    ),
    "massformer": (
        "fragnnet.massformer.pl_model:MassFormerPL",
        "fragnnet.dataset.spec_mol_dataset:SpecMolDataset",
    ),
    "iceberg_gen": (
        "fragnnet.iceberg.pl_model:IcebergGenPL",
        "fragnnet.iceberg.dataset:SpecMolMagmaGenDataset",
    ),
    "iceberg_inten": (
        "fragnnet.iceberg.pl_model:IcebergIntenPL",
        "fragnnet.iceberg.dataset:SpecMolMagmaIntenDataset",
    ),
    "graff": (
        "fragnnet.graff.pl_model:GrAFFPL",
        "fragnnet.graff.dataset:SpecMolAnnDataset",
    ),
    "precursor": (
        "fragnnet.pl_model.precursor_pl:PrecursorPL",
        "fragnnet.dataset.spec_mol_dataset:SpecMolDataset",
    ),
    "gnn": (
        "fragnnet.pl_model.gnn_pl:GNNPL",
        "fragnnet.dataset.spec_mol_dataset:SpecMolDataset",
    ),
    "mces_pretraining": (
        "fragnnet.pl_model.mces_pl:MCESPL",
        "fragnnet.dataset.mces_dataset:MCESDataset",
    ),
    "clip": (
        "fragnnet.pl_model.spectrum_mol_clip_pl:SpectrumMolClipPL",
        "fragnnet.dataset.spec_mol_dataset:SpecMolDataset",
    ),
    "spec_mol_clip_pretraining": (
        "fragnnet.pl_model.spectrum_mol_clip_pl:SpectrumMolClipPL",
        "fragnnet.dataset.spec_mol_dataset:SpecMolDataset",
    ),
}

ALIAS_PAIRS = (("frag_gnn", "fragnnet"), ("clip", "spec_mol_clip_pretraining"))


def _qualified(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def test_expected_mapping_covers_every_accepted_model_type() -> None:
    assert tuple(EXPECTED_MAPPING) == MODEL_TYPES


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_model_and_dataset_classes_match_expected_mapping(model_type: str) -> None:
    model_cls, dataset_cls = EXPECTED_MAPPING[model_type]
    assert _qualified(get_model_cls(model_type)) == model_cls
    assert _qualified(get_dataset_cls(model_type)) == dataset_cls


def test_unknown_model_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model_type"):
        load_config(TEMPLATE_DIR, None, None, model_type="not_a_model")
    with pytest.raises(ValueError, match="Unknown model type"):
        get_model_cls("not_a_model")
    with pytest.raises(ValueError, match="Unknown model type"):
        get_dataset_cls("not_a_model")


@pytest.mark.parametrize("pair", ALIAS_PAIRS, ids=lambda p: "/".join(p))
def test_alias_pairs_resolve_to_identical_configs(pair: tuple[str, str]) -> None:
    first, second = (load_config(TEMPLATE_DIR, None, None, model_type=t) for t in pair)
    differing = sorted(k for k in set(first) | set(second) if first.get(k) != second.get(k))
    assert differing == ["model_type"]
    assert get_model_cls(pair[0]) is get_model_cls(pair[1])
    assert get_dataset_cls(pair[0]) is get_dataset_cls(pair[1])


def test_legacy_mass_shatter_custom_config_resolves_to_masster(tmp_path) -> None:
    custom = tmp_path / "legacy.yml"
    custom.write_text(
        yaml.safe_dump({"model_type": "mass_shatter", "mass_shatter_max_h_transfer": 2})
    )
    config = load_config(TEMPLATE_DIR, str(custom), None)
    assert config["model_type"] == "masster"
    assert config["masster_max_h_transfer"] == 2
    assert not any(key.startswith("mass_shatter") for key in config)


def test_resolved_configs_match_golden() -> None:
    """Record ``load_config`` for every type as the frag_gnn base plus per-type differences."""
    configs = {t: load_config(TEMPLATE_DIR, None, None, model_type=t) for t in MODEL_TYPES}
    base = configs["frag_gnn"]
    record = {"base:frag_gnn": base}
    for model_type, config in configs.items():
        keys = sorted(set(base) | set(config))
        record[f"diff:{model_type}"] = {
            k: config.get(k, "<absent>") for k in keys if config.get(k) != base.get(k)
        }
    ratchet_json("registry", "resolved_configs", record)
