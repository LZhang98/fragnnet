# this is where all the torch models goes
# this controls model architecture and forward pass
import importlib

from .base_model import CEModel, InstModel, PrecModel
from .gnn_model import GNNModel
from .mol_encoder import MolEncoder
from .neims_model import NeimsModel
from .precursor_model import PrecursorModel
from .siamese_gnn_model import SiameseGNNModel
from .spectrum_encoder import SpectrumEncoder

# FraGNNet and MASSter classes load on first access (PEP 562), so importing a shared submodule
# of this package loads neither model. Phase 6 moves them out of this package.
_LAZY_EXPORTS = {
    "FraGNNetModel": "fragnnet_model",
    "MassterModel": "masster_model",
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{_LAZY_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
