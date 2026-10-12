"""The ``frag_utils`` split (reorganization plan Phase 3 step 5).

The element tables, formula and mask helpers moved to core ``formula_utils`` and the DAG storage
format to core ``dag_format_utils``. The FraGNNet-owned ``frag_utils`` remnant re-exported them
until Phase 6 (step 6), which retargeted every importer to core: the remnant now imports only the
core names it uses itself, and nothing imports a moved name through it. Phase 9 split the remnant
into ``models/fragnnet/dag/`` (builder, layout, formula, hashing, stats; graph was
``frag_feat_utils``), so these checks cover every module there.
"""

from __future__ import annotations

import ast
import importlib
import subprocess
from pathlib import Path

import pytest

from fragnnet.utils import dag_format_utils, formula_utils

MOVED = {
    formula_utils: (
        "CANONICAL_ELEMENT_ORDER",
        "CANONICAL_H_IDX",
        "ELEMENT_TO_IDX",
        "ELEMENT_TO_VE",
        "ELEMENTS",
        "HEAVY_ELEMENTS",
        "IDX_TO_ELEMENT",
        "MASK_SIZE",
        "MAX_NUM_EDGES",
        "MAX_NUM_NODES",
        "NUM_ELEMENTS",
        "formula_arr_to_str",
        "th_long_to_mask",
        "th_mask_to_long",
    ),
    dag_format_utils: (
        "PACKED_DAG_BYTES_CHUNK_SIZE",
        "PACKED_DAG_BYTES_DATASET",
        "PACKED_DAG_FORMAT",
        "PACKED_DAG_FORMAT_ATTR",
        "PACKED_DAG_INDEX_CHUNK_SIZE",
        "PACKED_DAG_INDEX_DATASET",
        "PACKED_DAG_INDEX_DTYPE",
        "PACKED_DAG_ZLIB_LEVEL",
        "_dump_pickle_file",
        "_legacy_frag_names",
        "_load_dag_hdf5_packed",
        "_load_pickle_file",
        "dump_dag_hdf5",
        "dump_dag_hdf5_packed",
        "dump_dag_pickle",
        "get_dag_output_path",
        "get_frag_fp",
        "get_frag_name",
        "initialize_packed_dag_hdf5",
        "load_dag_hdf5",
        "load_dag_hdf5_from_group",
        "load_frag_d",
        "save_frag_d",
    ),
}
PAIRS = [(module, name) for module, names in MOVED.items() for name in names]


REPO = Path(__file__).resolve().parents[1]
DAG_PACKAGE = "fragnnet.models.fragnnet.dag"
REMNANTS = tuple(
    f"{DAG_PACKAGE}.{name}" for name in ("builder", "layout", "formula", "hashing", "stats", "graph")
)
MOVED_NAMES = {name for _, name in PAIRS}


def _used_names(tree: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


@pytest.mark.parametrize("remnant", REMNANTS)
@pytest.mark.parametrize(("module", "name"), PAIRS, ids=lambda v: getattr(v, "__name__", v))
def test_remnant_binds_a_core_name_only_to_use_it(remnant: str, module, name: str) -> None:
    dag_module = importlib.import_module(remnant)
    if not hasattr(dag_module, name):
        return
    assert getattr(dag_module, name) is getattr(module, name)
    assert name in _used_names(ast.parse(Path(dag_module.__file__).read_text())), name


def test_nothing_imports_a_moved_name_through_the_remnant() -> None:
    files = subprocess.check_output(["git", "ls-files", "*.py"], cwd=REPO, text=True).split()
    found = []
    for f in files:
        text = (REPO / f).read_text()
        if DAG_PACKAGE not in text:
            continue
        tree = ast.parse(text)
        aliases = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in REMNANTS:
                found += [f"{f}: from-import {a.name}" for a in node.names if a.name in MOVED_NAMES]
            elif isinstance(node, ast.ImportFrom) and node.module == DAG_PACKAGE:
                aliases |= {a.asname or a.name for a in node.names}
            elif isinstance(node, ast.Import):
                aliases |= {a.asname for a in node.names if a.name in REMNANTS and a.asname}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in aliases
                and node.attr in MOVED_NAMES
            ):
                found.append(f"{f}:{node.lineno}: {node.value.id}.{node.attr}")
    assert not found


def test_functions_are_defined_in_core() -> None:
    for module, names in MOVED.items():
        for name in names:
            obj = getattr(module, name)
            if callable(obj) and hasattr(obj, "__module__"):
                assert obj.__module__ == module.__name__, name


@pytest.mark.parametrize(
    "name",
    [
        "update_bonds",
        "convert_cc_mask_to_int",
        "convert_cc_int_to_mask",
        "convert_cc_int_to_np_mask",
        "compute_approximate_cchs",
    ],
)
def test_dead_functions_are_deleted(name: str) -> None:
    """No callers; the Cython ``update_bonds`` is the production and test version."""
    for remnant in REMNANTS:
        assert not hasattr(importlib.import_module(remnant), name), remnant
