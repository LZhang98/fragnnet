"""Unit tests for fragnnet.utils.misc_utils module.

Tests for general utility functions and helpers.
"""

import math

import numpy as np
import pandas as pd
import pytest
import torch as th

from fragnnet.utils import misc_utils


def test_none_or_nan_variants() -> None:
    assert misc_utils.none_or_nan(None)
    assert misc_utils.none_or_nan("")
    assert misc_utils.none_or_nan(float("nan"))
    assert misc_utils.none_or_nan(np.array([np.nan]))
    assert misc_utils.none_or_nan(pd.Series([np.nan]))
    assert misc_utils.none_or_nan(pd.DataFrame({"a": [np.nan]}))
    assert misc_utils.none_or_nan([])
    assert not misc_utils.none_or_nan([1])
    assert not misc_utils.none_or_nan(0)
    assert not misc_utils.none_or_nan("value")


def test_flatten_lol() -> None:
    assert misc_utils.flatten_lol([[1, 2], [3]]) == [1, 2, 3]


def test_deep_update_merges_nested_dicts() -> None:
    base = {"a": 1, "b": {"c": 2}}
    updated = misc_utils.deep_update(base, {"b": {"d": 3}}, {"e": 4})
    assert updated == {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}


def test_scatter_logsumexp_matches_manual() -> None:
    logits = th.tensor([0.0, 1.0, 0.0])
    subset_idxs = th.tensor([0, 0, 1])
    out = misc_utils.scatter_logsumexp(logits, subset_idxs)
    expected_group0 = math.log(1.0 + math.e)  # logsumexp([0, 1])
    expected_group1 = 0.0  # logsumexp([0])
    assert th.isclose(out[0], th.tensor(expected_group0), atol=1e-6)
    assert th.isclose(out[1], th.tensor(expected_group1), atol=1e-6)


def test_scatter_argtopk_basic_ordering() -> None:
    # group 0: values [5, 1, 3] at positions [10, 11, 12] -> top-2 by value: 10, 12
    # group 1: values [2, 4] at positions [20, 21] -> top-2 by value: 21, 20
    src = th.tensor([5.0, 1.0, 3.0, 2.0, 4.0])
    index = th.tensor([0, 0, 0, 1, 1])
    other_index = th.tensor([10, 11, 12, 20, 21])
    out = misc_utils.scatter_argtopk(src, index, other_index, k=2)
    assert th.equal(out, th.tensor([[10, 12], [21, 20]]))


def test_scatter_argtopk_pads_short_groups_with_minus_one() -> None:
    # group 0 has 3 elements (k=2 fits); group 1 has only 1 element (k=2 doesn't fit)
    src = th.tensor([5.0, 1.0, 3.0, 9.0])
    index = th.tensor([0, 0, 0, 1])
    other_index = th.tensor([10, 11, 12, 99])
    out = misc_utils.scatter_argtopk(src, index, other_index, k=2)
    assert th.equal(out, th.tensor([[10, 12], [99, -1]]))


def test_scatter_argtopk_handles_group_with_no_members() -> None:
    # index skips group 1 entirely (only groups 0 and 2 have elements); dim_size
    # left unset so it's inferred from index rather than passed explicitly.
    src = th.tensor([1.0, 2.0, 3.0, 4.0])
    index = th.tensor([0, 0, 2, 2])
    other_index = th.tensor([100, 101, 102, 103])
    out = misc_utils.scatter_argtopk(src, index, other_index, k=1)
    assert th.equal(out, th.tensor([[101], [-1], [103]]))


def test_scatter_argtopk_explicit_oversized_dim_size_raises() -> None:
    # Passing a dim_size larger than th.max(index)+1 hits a real shape mismatch in
    # scatter_argtopk's internal counts array (always sized th.max(index)+1). Callers
    # with ragged/incomplete index ranges must leave dim_size unset.
    src = th.tensor([1.0, 2.0, 3.0, 4.0])
    index = th.tensor([0, 0, 2, 2])
    other_index = th.tensor([100, 101, 102, 103])
    with pytest.raises(IndexError):
        misc_utils.scatter_argtopk(src, index, other_index, k=1, dim_size=5)


def test_to_device_and_to_cpu_roundtrip() -> None:
    data = {"t": th.tensor([1, 2, 3])}
    on_device = misc_utils.to_device(data.copy(), device="cpu")
    assert on_device["t"].device.type == "cpu"

    back_on_cpu = misc_utils.to_cpu(on_device.copy())
    assert back_on_cpu["t"].device.type == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
