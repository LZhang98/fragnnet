"""
Unit tests for 03_inf_run_inference_frag_gen.py.
"""

import importlib.util
import sys
from pathlib import Path

import pytest


# Helper to import modules with numeric names
def _load_script_module(script_name):
    """Load a preprocessing script as a module."""
    script_path = Path(__file__).parent.parent / "preproc_scripts" / "inference" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_module():
    """Test that the module can be loaded."""
    module = _load_script_module("03_inf_run_inference_frag_gen.py")
    assert module is not None


def test_get_args_exists():
    """Test that get_args function exists."""
    module = _load_script_module("03_inf_run_inference_frag_gen.py")
    assert hasattr(module, "get_args")


def test_no_out_does_not_require_frag_dp(monkeypatch):
    """Compute-only runs do not need an output directory."""
    module = _load_script_module("03_inf_run_inference_frag_gen.py")
    monkeypatch.setattr(sys, "argv", ["frag_gen", "--proc_dp", "input", "--no-out"])

    args = module.get_args()

    assert args.no_out
    assert args.frag_dp is None


def test_frag_dp_required_when_writing_output(monkeypatch):
    """Normal runs must identify where generated DAGs should be written."""
    module = _load_script_module("03_inf_run_inference_frag_gen.py")
    monkeypatch.setattr(sys, "argv", ["frag_gen", "--proc_dp", "input"])

    with pytest.raises(SystemExit, match="2"):
        module.get_args()


def test_trim_process_memory(monkeypatch):
    """Garbage collection runs before glibc releases unused heap pages."""
    module = _load_script_module("03_inf_run_inference_frag_gen.py")
    calls = []

    class FakeMallocTrim:
        argtypes = None
        restype = None

        def __call__(self, pad):
            calls.append(("malloc_trim", pad))
            return 1

    class FakeLibc:
        malloc_trim = FakeMallocTrim()

    monkeypatch.setattr(module.gc, "collect", lambda: calls.append(("gc", None)))
    monkeypatch.setattr(module.ctypes, "CDLL", lambda name: FakeLibc())

    assert module.trim_process_memory()
    assert calls == [("gc", None), ("malloc_trim", 0)]


def test_trim_process_memory_without_malloc_trim(monkeypatch):
    """Platforms without glibc still complete Python garbage collection."""
    module = _load_script_module("03_inf_run_inference_frag_gen.py")
    calls = []

    monkeypatch.setattr(module.gc, "collect", lambda: calls.append(("gc", None)))
    monkeypatch.setattr(module.ctypes, "CDLL", lambda name: object())

    assert not module.trim_process_memory()
    assert calls == [("gc", None)]


def test_trim_process_memory_when_process_library_cannot_load(monkeypatch):
    """Failure to load the process library does not crash memory cleanup."""
    module = _load_script_module("03_inf_run_inference_frag_gen.py")

    def unavailable_library(name):
        raise OSError("library unavailable")

    monkeypatch.setattr(module.ctypes, "CDLL", unavailable_library)

    assert not module.trim_process_memory()
