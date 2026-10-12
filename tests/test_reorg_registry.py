"""Registry coverage for the repository reorganization (plan Phase 0 item 1).

The expected mapping below is the contract every later phase must keep. It changes only
deliberately, in the same PR as the code:

- Phase 1 step 2 replaced the ICEBERG rows with retirement-error checks (below); Phase 1 step 3
  mapped ``massformer`` to ``SpecMolMFDataset`` (equivalence shown by the Phase 0 baseline
  records in ``test_reorg_baselines.py``, which did not change).
- Phase 6 changes module paths only; class names and behavior stay.

The resolved configs of every model type are also recorded, as one shared base plus per-type
differences, so a registry or migration refactor cannot silently change what ``load_config``
returns.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as th
import yaml
from reorg_golden import ratchet_json
from reorg_layers import MODEL_TYPES, RETIRED_MODEL_TYPES

from fragnnet.registry import CHEMLM_ARCHIVE_SHA, ICEBERG_ARCHIVE_SHA, MCES_ARCHIVE_SHA
from fragnnet.runner import load_config
from fragnnet.utils.script_utils import get_config_from_ckpt, init_model_from_ckpt
from fragnnet.utils.setup_utils import get_dataset_cls, get_model_cls

TEMPLATE_DIR = "config"
REPO_TEMPLATE = Path(__file__).resolve().parents[1] / "config" / "template.yml"

# model_type -> (Lightning module, dataset class), as "module:QualName".
EXPECTED_MAPPING = {
    "frag_gnn": (
        "fragnnet.models.fragnnet.fragnnet_pl:FraGNNetPL",
        "fragnnet.models.fragnnet.spec_mol_frag_dataset:SpecMolFragDataset",
    ),
    "fragnnet": (
        "fragnnet.models.fragnnet.fragnnet_pl:FraGNNetPL",
        "fragnnet.models.fragnnet.spec_mol_frag_dataset:SpecMolFragDataset",
    ),
    "masster": (
        "fragnnet.models.masster.masster_pl:MassterPL",
        "fragnnet.dataset.spec_mol_dataset:SpecMolDataset",
    ),
    "neims": (
        "fragnnet.pl_model.neims_pl:NeimsPL",
        "fragnnet.dataset.spec_mol_dataset:SpecMolDataset",
    ),
    "massformer": (
        "fragnnet.baselines.massformer.pl_model:MassFormerPL",
        "fragnnet.baselines.massformer.dataset:SpecMolMFDataset",
    ),
    "graff": (
        "fragnnet.baselines.graff.pl_model:GrAFFPL",
        "fragnnet.baselines.graff.dataset:SpecMolAnnDataset",
    ),
    "precursor": (
        "fragnnet.pl_model.precursor_pl:PrecursorPL",
        "fragnnet.dataset.spec_mol_dataset:SpecMolDataset",
    ),
    "gnn": (
        "fragnnet.pl_model.gnn_pl:GNNPL",
        "fragnnet.dataset.spec_mol_dataset:SpecMolDataset",
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


# What each retirement message must name besides the type: where to go instead, and the archive.
RETIREMENT_MARKERS = {
    "iceberg_gen": ("ms-pred", ICEBERG_ARCHIVE_SHA),
    "iceberg_inten": ("ms-pred", ICEBERG_ARCHIVE_SHA),
    "mces_pretraining": ("did not help", MCES_ARCHIVE_SHA),
    "chemlm": ("did not help", CHEMLM_ARCHIVE_SHA),
}


def _assert_retirement_error(excinfo: pytest.ExceptionInfo, model_type: str) -> None:
    message = str(excinfo.value)
    assert f"model_type {model_type!r} is retired" in message
    for marker in RETIREMENT_MARKERS[model_type]:
        assert marker in message


@pytest.mark.parametrize("model_type", RETIRED_MODEL_TYPES)
def test_retired_model_types_raise_the_retirement_error_on_every_path(
    model_type: str, tmp_path
) -> None:
    """Every way a retired type can be selected raises the actionable retirement message.

    Paths: a custom config, the ``model_type`` argument (a checkpoint's type), a template that
    declares it, the registry lookups, and the checkpoint loaders.
    """
    custom = tmp_path / "custom.yml"
    custom.write_text(yaml.safe_dump({"model_type": model_type}))
    with pytest.raises(ValueError) as excinfo:
        load_config(TEMPLATE_DIR, str(custom), None)
    _assert_retirement_error(excinfo, model_type)

    with pytest.raises(ValueError) as excinfo:
        load_config(TEMPLATE_DIR, None, None, model_type=model_type)
    _assert_retirement_error(excinfo, model_type)

    template = tmp_path / "template.yml"
    template_d = yaml.safe_load((REPO_TEMPLATE).read_text())
    template_d["model_type"] = model_type
    template.write_text(yaml.safe_dump(template_d))
    with pytest.raises(ValueError) as excinfo:
        load_config(str(template), None, None)
    _assert_retirement_error(excinfo, model_type)

    for lookup in (get_model_cls, get_dataset_cls):
        with pytest.raises(ValueError) as excinfo:
            lookup(model_type)
        _assert_retirement_error(excinfo, model_type)

    ckpt_fp = tmp_path / f"old_{model_type}.ckpt"
    th.save({"hyper_parameters": {"model_type": model_type}, "state_dict": {}}, ckpt_fp)
    with pytest.raises(ValueError) as excinfo:
        init_model_from_ckpt(str(ckpt_fp), {"model_type": model_type}, th.device("cpu"))
    _assert_retirement_error(excinfo, model_type)
    empty_custom = tmp_path / "empty.yml"
    empty_custom.write_text("")
    with pytest.raises(ValueError) as excinfo:
        get_config_from_ckpt(str(ckpt_fp), str(empty_custom), TEMPLATE_DIR)
    _assert_retirement_error(excinfo, model_type)


def test_no_module_hard_codes_the_shared_dataset_for_massformer() -> None:
    """MassFormer's Graphormer input is built only by ``SpecMolMFDataset`` (Phase 1 step 3).

    A script that maps ``massformer`` to the shared ``SpecMolDataset`` by hand would silently
    build batches without ``mol_mf_*`` keys.
    """
    import re
    import subprocess

    pattern = r"[\"']massformer[\"']\s*:\s*SpecMolDataset\b"
    tracked = subprocess.check_output(
        ["git", "ls-files", "*.py"], cwd=REPO_TEMPLATE.parents[1], text=True
    ).split()
    offenders = [
        path
        for path in tracked
        if re.search(pattern, (REPO_TEMPLATE.parents[1] / path).read_text(errors="ignore"))
    ]
    assert offenders == []


def test_shared_dataset_rejects_the_massformer_input_flag(tmp_path) -> None:
    """``mol_params.mf`` on a dataset that cannot build ``mol_mf`` fails at the first molecule.

    The check sits where the removed ``mol_mf`` code read the flag (``_process_mol``), so it
    fires during preprocessing or, without it, on the first item.
    """
    import reorg_fixtures as F

    data = F.write_synth_data(tmp_path, dag_backend=None)
    config = F.make_config("masster", data, {"mol_params": {"mf": True}})
    with pytest.raises(ValueError, match="mol_params.mf needs MassFormer's dataset"):
        dataset, _ = F.init_split(config, "train")
        dataset[0]
