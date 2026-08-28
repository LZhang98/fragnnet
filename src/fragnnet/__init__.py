__version__ = "0.0.0"

__all__ = ["FraGNNetInference", "__version__"]

def __getattr__(name):
	if name == "FraGNNetInference":
		from fragnnet.inference import FraGNNetInference
		return FraGNNetInference
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")