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

# MASSter's batch sampler loads on first access (PEP 562), so importing a shared submodule of
# this package does not load it. It is not in ``__all__``. Phase 6 moves it out of this package.
_LAZY_EXPORTS = {
    "MassterCEContextDynamicBatchSampler": "masster_data_sampler",
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{_LAZY_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
