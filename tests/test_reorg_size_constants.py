"""Core fragment-DAG size constants (reorganization plan Phase 3 step 3).

``MAX_NUM_NODES``, ``MAX_NUM_EDGES`` and ``MASK_SIZE`` live in core ``formula_utils``. The Cython
builders need compile-time sizes, so ``compute_frags.pyx`` and ``multi_cut_bfs.pyx`` keep literal
copies, both as Python-level constants and as stack-array lengths. These tests fail when a core
limit changes without the Cython copies.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import fragnnet.models.fragnnet.dag.builder as dag_builder
import fragnnet.models.fragnnet.dag.compute_frags as compute_frags
import fragnnet.models.fragnnet.dag.multi_cut_bfs as multi_cut_bfs
import fragnnet.utils.proc_utils as proc_utils
from fragnnet.utils import formula_utils

FRAG_DIR = Path(compute_frags.__file__).parent
PYX_FILES = ("compute_frags.pyx", "multi_cut_bfs.pyx")


def test_core_values() -> None:
    assert formula_utils.MAX_NUM_NODES == 128
    assert formula_utils.MAX_NUM_EDGES == 4 * formula_utils.MAX_NUM_NODES
    assert formula_utils.MASK_SIZE == 128
    assert formula_utils.MASK_SIZE >= formula_utils.MAX_NUM_NODES
    assert formula_utils.MASK_SIZE % 64 == 0


@pytest.mark.parametrize("module", [compute_frags, multi_cut_bfs], ids=lambda m: m.__name__)
def test_cython_constants_match_core(module) -> None:
    assert module.MAX_NUM_NODES == formula_utils.MAX_NUM_NODES
    assert module.MAX_NUM_EDGES == formula_utils.MAX_NUM_EDGES


@pytest.mark.parametrize("pyx", PYX_FILES)
def test_cython_stack_arrays_use_core_sizes(pyx: str) -> None:
    """Every ``cdef`` stack array is sized from the node or edge limit: per node, per node plus
    one (a CSR prefix sum), per edge, or twice per edge (a CSR adjacency list)."""
    allowed = {
        formula_utils.MAX_NUM_NODES,
        formula_utils.MAX_NUM_NODES + 1,
        formula_utils.MAX_NUM_EDGES,
        2 * formula_utils.MAX_NUM_EDGES,
    }
    source = (FRAG_DIR / pyx).read_text()
    sizes = [int(n) for n in re.findall(r"^\s*cdef\s+[\w ]+?\s+\w+\[(\d+)\]", source, re.M)]
    assert sizes, f"no stack arrays found in {pyx}; the pattern no longer matches"
    assert set(sizes) <= allowed


def test_python_constants_come_from_core() -> None:
    """``proc_utils`` (core) and the DAG builder (was the ``frag_utils`` remnant) read the core
    values."""
    for name in ("MAX_NUM_NODES", "MAX_NUM_EDGES"):
        assert getattr(proc_utils, name) == getattr(formula_utils, name)
    for name in ("MAX_NUM_NODES", "MAX_NUM_EDGES", "MASK_SIZE"):
        assert getattr(dag_builder, name) == getattr(formula_utils, name)
