"""Module-to-layer table and import-graph helpers for the repository reorganization.

See docs/plans/repo_reorg_plan.md ("Target layout", Phase 0 item 10). Until Phase 6 the owner
modules still live in core directories, so the dependency rule is checked against this explicit
table rather than against directory names. Each module is assigned the layer it belongs to after
the reorganization:

- ``core``: model-agnostic code (dataset/, model/, pl_model/, utils/, runner, registry).
- ``fragmentation``: bond-cut chemistry shared by both fragment models.
- ``fragnnet``, ``masster``: the two owner models.
- ``baseline:<name>``: retained (graff, massformer) and retired (iceberg) baselines.

Allowed edges: a layer may import itself and core; owner models and baselines may also import
``fragmentation``. Everything else is a violation. Tests are exempt and are not scanned.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

FRAGMENTATION = {
    "fragnnet.fragmentation",
    "fragnnet.fragmentation.candidate_support",
    "fragnnet.fragmentation.cut_bond_encoder",
    "fragnnet.fragmentation.hydrogen_cap",
    "fragnnet.fragmentation.smarts_prepass",
}
FRAGNNET = {
    "fragnnet.models.fragnnet.frag_data_sampler",
    "fragnnet.models.fragnnet.iterable_spec_mol_frag_dataset",
    "fragnnet.models.fragnnet.spec_mol_frag_dataset",
    "fragnnet.models.fragnnet.compute_frags",
    "fragnnet.models.fragnnet.fragnnet_hooks",
    "fragnnet.models.fragnnet.multi_cut_bfs",
    "fragnnet.models.fragnnet.fragment_competition",
    "fragnnet.models.fragnnet.fragnnet_model",
    "fragnnet.models.fragnnet.fragnnet_pl",
    "fragnnet.models.fragnnet.frag_feat_utils",
    # The remnant after the Phase 3 split; the element tables and DAG format are in core.
    "fragnnet.models.fragnnet.frag_utils",
}
MASSTER = {
    "fragnnet.models.masster.masster_data_sampler",
    "fragnnet.models.masster.bond_mask_generator",
    "fragnnet.models.masster.bond_fragment_heads",
    "fragnnet.models.masster.candidate_features",
    "fragnnet.models.masster.masster_model",
    "fragnnet.models.masster.rt_ccs",
    "fragnnet.models.masster.masster_pl",
}
MASSTER_PREFIXES = ("fragnnet.models.masster.",)
BASELINE_PACKAGES = {
    "fragnnet.baselines.graff": "baseline:graff",
    "fragnnet.baselines.massformer": "baseline:massformer",
    "fragnnet.iceberg": "baseline:iceberg",
}

OWNER_LAYERS = ("fragnnet", "masster")

# Every model_type the runner accepts (runner.VALID_MODEL_TYPES), in plan order. ICEBERG's two
# types were accepted until Phase 1 step 2 and are now retired.
MODEL_TYPES = (
    "frag_gnn",
    "fragnnet",
    "masster",
    "neims",
    "chemlm",
    "massformer",
    "graff",
    "precursor",
    "gnn",
    "mces_pretraining",
    "clip",
    "spec_mol_clip_pretraining",
)

# Model types that were removed and must raise the retirement error (registry.RETIRED_MODEL_TYPES).
RETIRED_MODEL_TYPES = ("iceberg_gen", "iceberg_inten")

# Core entry modules checked by the isolation gate (plan, Phase 1 gate table).
ENTRY_MODULES = (
    "fragnnet.runner",
    "fragnnet.utils.script_utils",
    "fragnnet.pl_model.spectrum_pl",
    "fragnnet.dataset.spec_mol_dataset",
    "fragnnet.dataset.base_dataset",
    "fragnnet.utils.feat_utils",
    "fragnnet.utils.proc_utils",
    "fragnnet.dataset.data_sampler",
    "fragnnet.utils.pl_utils",
)


def layer_of(module: str) -> str:
    """Return the post-reorganization layer of a ``fragnnet`` module."""
    if module in FRAGMENTATION:
        return "fragmentation"
    if module in FRAGNNET:
        return "fragnnet"
    if module in MASSTER or module.startswith(MASSTER_PREFIXES):
        return "masster"
    for package, layer in BASELINE_PACKAGES.items():
        if module == package or module.startswith(package + "."):
            return layer
    return "core"


def edge_allowed(src_layer: str, dst_layer: str) -> bool:
    """Whether a module in ``src_layer`` may import a module in ``dst_layer``."""
    if dst_layer in (src_layer, "core"):
        return True
    if dst_layer == "fragmentation":
        return src_layer != "core"
    return False


def _module_files() -> dict[str, Path]:
    files = {}
    for path in sorted((SRC_ROOT / "fragnnet").rglob("*.py")):
        rel = path.relative_to(SRC_ROOT).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        files[".".join(parts)] = path
    return files


def _compiled_modules() -> set[str]:
    names = set()
    for path in (SRC_ROOT / "fragnnet").rglob("*.pyx"):
        names.add(".".join(path.relative_to(SRC_ROOT).with_suffix("").parts))
    return names


def _static_reexports(files: dict[str, Path]) -> dict[str, dict[str, str]]:
    """{package: {name: defining module}} for eager ``from .x import Name`` in ``__init__``."""
    out: dict[str, dict[str, str]] = {}
    for module, path in files.items():
        if path.name != "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        names = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                base = _resolve_from(module, True, node)
                if base is None:
                    continue
                for alias in node.names:
                    names[alias.asname or alias.name] = base
        out[module] = names
    return out


def _resolve_from(module: str, is_package: bool, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    parts = module.split(".")
    keep = len(parts) - node.level + (1 if is_package else 0)
    base = ".".join(parts[:keep])
    return f"{base}.{node.module}" if node.module else base


def import_edges() -> dict[tuple[str, str], set[str]]:
    """All ``fragnnet`` import edges in src, module- and function-level.

    Returns {(importer, imported module): {names}}. Names imported from a package are resolved to
    the submodule that defines them when the package re-exports them.
    """
    files = _module_files()
    known = set(files) | _compiled_modules()
    reexports = _static_reexports(files)
    edges: dict[tuple[str, str], set[str]] = {}

    def add(src: str, dst: str, name: str) -> None:
        if dst == src or not dst.startswith("fragnnet"):
            return
        edges.setdefault((src, dst), set()).add(name)

    for module, path in files.items():
        tree = ast.parse(path.read_text())
        is_package = path.name == "__init__.py"
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = _resolve_from(module, is_package, node)
                if base is None or not base.startswith("fragnnet"):
                    continue
                for alias in node.names:
                    sub = f"{base}.{alias.name}"
                    if sub in known:
                        add(module, sub, alias.name)
                    elif alias.name in reexports.get(base, {}):
                        add(module, reexports[base][alias.name], alias.name)
                    else:
                        add(module, base, alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("fragnnet"):
                        add(module, alias.name, "*")
    return edges


def boundary_violations() -> dict[str, list[str]]:
    """{"importer -> imported": sorted names} for every edge the dependency rule forbids."""
    out = {}
    for (src, dst), names in import_edges().items():
        if not edge_allowed(layer_of(src), layer_of(dst)):
            out[f"{src} -> {dst}"] = sorted(names)
    return dict(sorted(out.items()))


_PROBE = r"""
import importlib, importlib.abc, json, sys
attempted = []
class _BlockDGL(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "dgl" or name.startswith("dgl."):
            attempted.append(name)
            raise ImportError("dgl blocked by the reorganization isolation gate")
        return None
sys.meta_path.insert(0, _BlockDGL())
target = sys.argv[1]
if target.startswith("model_type:"):
    from fragnnet.utils.setup_utils import get_dataset_cls, get_model_cls
    model_type = target.split(":", 1)[1]
    get_model_cls(model_type)
    get_dataset_cls(model_type)
else:
    importlib.import_module(target)
loaded = sorted(k for k in sys.modules if k == "fragnnet" or k.startswith("fragnnet."))
print(json.dumps({"loaded": loaded, "dgl_attempted": sorted(set(attempted))}))
"""


def probe_loaded_modules(target: str) -> dict[str, list[str]]:
    """Import ``target`` (a module, or ``model_type:<type>``) in a fresh CPU-only subprocess.

    Returns the ``fragnnet`` modules loaded and any attempted ``dgl`` imports (DGL is blocked).
    """
    env = {"CUDA_VISIBLE_DEVICES": "", "PYTHONHASHSEED": "0"}
    import os

    env = {**os.environ, **env}
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, target],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        check=False,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"probe of {target!r} failed:\n{result.stderr[-4000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def non_core_loaded(target: str) -> dict[str, list[str]]:
    """For an entry point, the non-core modules it loads, grouped by layer, plus dgl attempts."""
    probe = probe_loaded_modules(target)
    grouped: dict[str, list[str]] = {}
    for module in probe["loaded"]:
        layer = layer_of(module)
        if layer != "core":
            grouped.setdefault(layer, []).append(module)
    if probe["dgl_attempted"]:
        grouped["dgl"] = probe["dgl_attempted"]
    return dict(sorted(grouped.items()))
