"""The ``frag_utils`` split (reorganization plan Phase 3 step 5).

The element tables, formula and mask helpers moved to core ``formula_utils`` and the DAG storage
format to core ``dag_format_utils``. The FraGNNet-owned ``frag_utils`` remnant re-exports them until
Phase 6, as the same objects, so removing the re-exports later is a pure import retarget.
"""

from __future__ import annotations

import pytest

import fragnnet.models.fragnnet.frag_utils as frag_utils
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
        "NUM_HEAVY_ELEMENTS",
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


@pytest.mark.parametrize(("module", "name"), PAIRS, ids=lambda v: getattr(v, "__name__", v))
def test_remnant_reexports_the_core_object(module, name: str) -> None:
    assert getattr(frag_utils, name) is getattr(module, name)


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
    assert not hasattr(frag_utils, name)
