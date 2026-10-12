import importlib

from .base_dataset import BaseDataset
from .data_sampler import (
    BondCountCurriculumSampler,
    CEContextDynamicBatchSampler,
    DualGroupDynamicBatchSampler,
    FormulaPrecTypeGroupSampler,
    GroupSampler,
    MassterCEContextDynamicBatchSampler,
    SpecMolFragDynamicBatchSampler,
    get_group_sampler,
)
from .mces_dataset import MCESDataset
from .spec_mol_dataset import SpecMolDataset

__all__ = [
    "BaseDataset",
    "BondCountCurriculumSampler",
    "CEContextDynamicBatchSampler",
    "DualGroupDynamicBatchSampler",
    "FormulaPrecTypeGroupSampler",
    "GroupSampler",
    "MassterCEContextDynamicBatchSampler",
    "SpecMolFragDynamicBatchSampler",
    "get_group_sampler",
    "MCESDataset",
    "SpecMolDataset",
]

# FraGNNet's datasets load on first access (PEP 562), so importing a shared submodule of this
# package does not load them. They are not in ``__all__``. Phase 6 moves them out of this package.
_LAZY_EXPORTS = {
    "IterableSpecMolFragDataset": "iterable_spec_mol_frag_dataset",
    "SpecMolFragDataset": "spec_mol_frag_dataset",
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{_LAZY_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
