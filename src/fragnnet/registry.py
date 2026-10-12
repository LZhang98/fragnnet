"""Model-type registry: which Lightning module and dataset class serve each ``model_type``.

Core code reaches the models only through this module. It holds import paths as strings and
imports a class only when that model type is requested, so importing core does not load any
model, baseline or optional dependency (see docs/plans/repo_reorg_plan.md, "Target layout").
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEntry:
    """Import paths (``"package.module:QualName"``) of one model type's classes.

    Attributes:
        model_cls: The Lightning module.
        dataset_cls: The dataset class that produces its batches.
    """

    model_cls: str
    dataset_cls: str


_SPEC_MOL_DATASET = "fragnnet.dataset.spec_mol_dataset:SpecMolDataset"
_FRAGNNET = ModelEntry(
    "fragnnet.pl_model.fragnnet_pl:FraGNNetPL",
    "fragnnet.dataset.spec_mol_frag_dataset:SpecMolFragDataset",
)
_CLIP = ModelEntry("fragnnet.pl_model.spectrum_mol_clip_pl:SpectrumMolClipPL", _SPEC_MOL_DATASET)

MODEL_REGISTRY: dict[str, ModelEntry] = {
    "frag_gnn": _FRAGNNET,
    "fragnnet": _FRAGNNET,
    "masster": ModelEntry("fragnnet.pl_model.masster_pl:MassterPL", _SPEC_MOL_DATASET),
    "neims": ModelEntry("fragnnet.pl_model.neims_pl:NeimsPL", _SPEC_MOL_DATASET),
    "chemlm": ModelEntry("fragnnet.pl_model.chemlm_pl:ChemLMPL", _SPEC_MOL_DATASET),
    "massformer": ModelEntry(
        "fragnnet.massformer.pl_model:MassFormerPL", "fragnnet.massformer.dataset:SpecMolMFDataset"
    ),
    "graff": ModelEntry(
        "fragnnet.graff.pl_model:GrAFFPL", "fragnnet.graff.dataset:SpecMolAnnDataset"
    ),
    "precursor": ModelEntry("fragnnet.pl_model.precursor_pl:PrecursorPL", _SPEC_MOL_DATASET),
    "gnn": ModelEntry("fragnnet.pl_model.gnn_pl:GNNPL", _SPEC_MOL_DATASET),
    "mces_pretraining": ModelEntry(
        "fragnnet.pl_model.mces_pl:MCESPL", "fragnnet.dataset.mces_dataset:MCESDataset"
    ),
    "clip": _CLIP,
    "spec_mol_clip_pretraining": _CLIP,
}

# Every model type a config may declare. Shared code reads any template key regardless of the
# model type, so this only guards against a typo in ``model_type`` itself.
VALID_MODEL_TYPES = frozenset(MODEL_REGISTRY)

# Last revision that contains the local ICEBERG fork (reachable from origin/masster).
ICEBERG_ARCHIVE_SHA = "13b1599d3cd85ef8f214799894e527adaf3ddd0a"
_ICEBERG_RETIRED = (
    "model_type {model_type!r} is retired: this repository's ICEBERG fork was removed on "
    "2026-10-11 (docs/plans/repo_reorg_plan.md). Run ICEBERG with external ms-pred "
    "(https://github.com/coleygroup/ms-pred) and score its prediction files with this "
    "repository's evaluation tools. The local fork, its configs and its checkpoints belong to "
    f"revision {ICEBERG_ARCHIVE_SHA}."
)

# Model types that were removed; selecting one raises an actionable error instead of the generic
# unknown-type error.
RETIRED_MODEL_TYPES: dict[str, str] = {
    "iceberg_gen": _ICEBERG_RETIRED,
    "iceberg_inten": _ICEBERG_RETIRED,
}


def raise_if_retired(model_type: str) -> None:
    """Raise the retirement message for a removed model type; do nothing otherwise.

    Args:
        model_type: Model type from a config, checkpoint or caller.

    Raises:
        ValueError: If ``model_type`` was retired.
    """
    if model_type in RETIRED_MODEL_TYPES:
        raise ValueError(RETIRED_MODEL_TYPES[model_type].format(model_type=model_type))


def _resolve(path: str) -> type:
    module_name, _, qualname = path.partition(":")
    obj = importlib.import_module(module_name)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


def _entry(model_type: str) -> ModelEntry:
    raise_if_retired(model_type)
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_REGISTRY[model_type]


def get_model_cls(model_type: str) -> type:
    """Return the Lightning module class for ``model_type``, importing it on first use.

    Args:
        model_type: A registered model type.

    Returns:
        The model class.

    Raises:
        ValueError: If the model type is unknown or retired.
    """
    return _resolve(_entry(model_type).model_cls)


def get_dataset_cls(model_type: str) -> type:
    """Return the dataset class for ``model_type``, importing it on first use.

    Args:
        model_type: A registered model type.

    Returns:
        The dataset class.

    Raises:
        ValueError: If the model type is unknown or retired.
    """
    return _resolve(_entry(model_type).dataset_cls)
