# ruff: noqa: I001
# isort: skip_file
# this is where all the PL models goes
# PL models control the training and evaluation of the models
# they are used by the trainer to run the training and evaluation loops
#
# Import order is deliberate: SpectrumPL is the base class every module below imports, so it
# must be bound first. Sorting this block alphabetically reintroduces a circular import.
import importlib

from .spectrum_pl import SpectrumPL
from .binned_pl import BinnedPL
from .neims_pl import NeimsPL
from .precursor_pl import PrecursorPL
from .gnn_pl import GNNPL
from .mces_pl import MCESPL
from .spectrum_mol_clip_pl import SpectrumMolClipPL
from .chemlm_pl import ChemLMPL

# FraGNNet and MASSter classes load on first access (PEP 562), so importing a shared submodule
# of this package loads neither model. Phase 6 moves them out of this package.
_LAZY_EXPORTS = {
    "FraGNNetPL": "fragnnet_pl",
    "MassterPL": "masster_pl",
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{_LAZY_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
