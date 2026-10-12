"""Lazy package exports (reorganization plan Phase 1 step 5).

``fragnnet.model``, ``fragnnet.pl_model`` and ``fragnnet.dataset`` export only core classes
eagerly. FraGNNet's and MASSter's classes stay reachable through a PEP 562 ``__getattr__`` that
imports them on first access, so every existing import form keeps working while importing a
shared submodule no longer loads either model. ``reorg_layers.COMPAT_SHIMS`` lists these names
for the boundary check and must match the packages.

Import forms that depend on what is already loaded run in fresh interpreters.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor

import pytest
from reorg_layers import COMPAT_SHIMS, SRC_ROOT

LAZY = [
    (package, name, module.rsplit(".", 1)[1])
    for package, names in COMPAT_SHIMS.items()
    for name, module in names.items()
]

# Each form runs in a fresh interpreter with P (package), N (lazy name) and S (submodule) set.
FRESH_FORMS = {
    "attribute": """
        pkg = importlib.import_module(P)
        assert N not in vars(pkg), "lazy name bound eagerly"
        real = getattr(importlib.import_module(f"{P}.{S}"), N)
        assert getattr(pkg, N) is real
        assert vars(pkg)[N] is real, "first access is not cached"
    """,
    "from_package_import_class": """
        exec(f"from {P} import {N} as got")
        assert got is getattr(importlib.import_module(f"{P}.{S}"), N)
    """,
    "from_package_import_submodule": """
        pkg = importlib.import_module(P)
        print("prebound" if S in vars(pkg) else "fallback")
        exec(f"from {P} import {S} as got")
        assert got is sys.modules[f"{P}.{S}"]
    """,
    "import_package_submodule_as": """
        exec(f"import {P}.{S} as got")
        assert got is sys.modules[f"{P}.{S}"]
        assert getattr(got, N).__module__ == f"{P}.{S}"
    """,
}


def _run_fresh(form: str, package: str, name: str, sub: str) -> subprocess.CompletedProcess:
    code = f"import importlib, sys\nP, N, S = {package!r}, {name!r}, {sub!r}\n"
    code += textwrap.dedent(FRESH_FORMS[form])
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


@pytest.fixture(scope="module")
def fresh_results() -> dict[tuple[str, str, str], subprocess.CompletedProcess]:
    keys = [(form, package, name) for form in FRESH_FORMS for package, name, _ in LAZY]
    subs = {(package, name): sub for package, name, sub in LAZY}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = pool.map(lambda k: _run_fresh(k[0], k[1], k[2], subs[k[1], k[2]]), keys)
        return dict(zip(keys, results))


@pytest.mark.parametrize("form", FRESH_FORMS)
@pytest.mark.parametrize(("package", "name", "sub"), LAZY, ids=lambda v: str(v))
def test_import_form_resolves_lazy_name(fresh_results, form, package, name, sub) -> None:
    result = fresh_results[form, package, name]
    assert result.returncode == 0, result.stderr


def test_submodule_import_exercises_the_import_fallback(fresh_results) -> None:
    """``from pkg import submodule`` must work when the submodule is not loaded yet.

    Python then calls ``__getattr__`` first and imports the submodule only after an
    ``AttributeError``; any other exception breaks the import. ``fragnnet.dataset`` does not load
    FraGNNet's datasets, so its cases take that path.
    """
    paths = {
        (package, name): fresh_results[
            "from_package_import_submodule", package, name
        ].stdout.strip()
        for package, name, _ in LAZY
    }
    dataset_paths = {k: v for k, v in paths.items() if k[0] == "fragnnet.dataset"}
    assert dataset_paths and set(dataset_paths.values()) == {"fallback"}, paths


@pytest.mark.parametrize(("package", "name", "sub"), LAZY, ids=lambda v: str(v))
def test_monkeypatching_a_lazy_name_is_undone(monkeypatch, package, name, sub) -> None:
    pkg = importlib.import_module(package)
    real = getattr(importlib.import_module(f"{package}.{sub}"), name)
    fake = type(name, (), {})
    monkeypatch.setattr(pkg, name, fake)
    namespace: dict = {}
    exec(f"from {package} import {name} as got", namespace)
    assert namespace["got"] is fake
    monkeypatch.undo()
    exec(f"from {package} import {name} as got", namespace)
    assert namespace["got"] is real


@pytest.mark.parametrize("package", COMPAT_SHIMS)
def test_unknown_names_still_raise(package: str) -> None:
    pkg = importlib.import_module(package)
    assert not hasattr(pkg, "NotAnExport")
    with pytest.raises(AttributeError, match=f"module '{package}' has no attribute 'NotAnExport'"):
        pkg.NotAnExport  # noqa: B018
    with pytest.raises(ImportError):
        exec(f"from {package} import NotAnExport", {})


def test_lazy_dataset_names_are_not_star_exported() -> None:
    """``fragnnet.dataset`` is the only one of these packages with ``__all__``; the lazy names
    are not in it. (Without ``__all__``, a star import takes whatever public names are bound, so
    for ``model`` and ``pl_model`` it depends on what was accessed first; nothing star-imports
    them.)"""
    package = "fragnnet.dataset"
    pkg = importlib.import_module(package)
    assert not set(COMPAT_SHIMS[package]) & set(pkg.__all__)
    namespace: dict = {}
    exec(f"from {package} import *", namespace)
    assert not set(COMPAT_SHIMS[package]) & set(namespace)


def test_compat_shims_match_the_packages() -> None:
    """The boundary check's table equals each package's ``_LAZY_EXPORTS``, and no package
    defines an unregistered module-level ``__getattr__``."""
    for package, names in COMPAT_SHIMS.items():
        lazy = importlib.import_module(package)._LAZY_EXPORTS
        assert {n: f"{package}.{s}" for n, s in lazy.items()} == names
    with_getattr = set()
    for init in (SRC_ROOT / "fragnnet").rglob("__init__.py"):
        tree = ast.parse(init.read_text())
        if any(isinstance(n, ast.FunctionDef) and n.name == "__getattr__" for n in tree.body):
            with_getattr.add(".".join(init.parent.relative_to(SRC_ROOT).parts))
    assert with_getattr == set(COMPAT_SHIMS)
