import importlib

from .base_dataset import BaseDataset
from .data_sampler import (
    BondCountCurriculumSampler,
    FormulaPrecTypeGroupSampler,
    GroupSampler,
    get_group_sampler,
)
from .mces_dataset import MCESDataset
from .spec_mol_dataset import SpecMolDataset

__all__ = [
    "BaseDataset",
    "BondCountCurriculumSampler",
    "FormulaPrecTypeGroupSampler",
    "GroupSampler",
    "get_group_sampler",
    "MCESDataset",
    "SpecMolDataset",
]

# FraGNNet's datasets and the FraGNNet and MASSter batch samplers load on first access (PEP 562),
# so importing a shared submodule of this package does not load them. They are not in
# ``__all__``. Phase 6 moves them out of this package.
_LAZY_EXPORTS = {
    "CEContextDynamicBatchSampler": "frag_data_sampler",
    "DualGroupDynamicBatchSampler": "frag_data_sampler",
    "IterableSpecMolFragDataset": "iterable_spec_mol_frag_dataset",
    "MassterCEContextDynamicBatchSampler": "masster_data_sampler",
    "SpecMolFragDataset": "spec_mol_frag_dataset",
    "SpecMolFragDynamicBatchSampler": "frag_data_sampler",
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{_LAZY_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
