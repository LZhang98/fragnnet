"""Unit tests for the three standalone helpers in spectrum_pl.py.

Covers the functions touched or added by the feiw_v2 ↔ ccx_run merge:
  - get_split_fn   : merged HEAD assertion + ccx_run th.bincount
  - list_to_batch_tensor : kept from HEAD
  - ragged_to_batch      : new from ccx_run

The import chain for spectrum_pl requires fragnnet.frag.multi_cut_bfs (a
Cython extension that may not be compiled). We stub it at module level so
the rest of the package can be imported normally.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Stub the uncompiled Cython extension before any fragnnet import
# ---------------------------------------------------------------------------
if "fragnnet.frag.multi_cut_bfs" not in sys.modules:
    _m = types.ModuleType("fragnnet.frag.multi_cut_bfs")
    _m.compute_ccs_multi_cut = MagicMock()
    _m.get_ring_edge_mask = MagicMock()
    sys.modules["fragnnet.frag.multi_cut_bfs"] = _m

from fragnnet.pl_model.spectrum_pl import (  # noqa: E402
    SpectrumPL,
    get_split_fn,
    list_to_batch_tensor,
    primary_ce_to_batch,
    ragged_to_batch,
    select_context_ce_values,
)


def test_select_context_ce_values_uses_per_instrument_axis():
    batch = {
        "spec_nce": torch.tensor([20.0, 30.0]),
        "spec_nce_batch_idxs": torch.tensor([0, 1]),
        "spec_ace": torch.tensor([15.0, 45.0]),
        "spec_ace_batch_idxs": torch.tensor([0, 1]),
        # sorted instrument order is FT, QTOF
        "spec_inst_type": torch.tensor([0, 1]),
    }

    result = select_context_ce_values(
        batch,
        batch_size=2,
        inst_types=["QTOF", "FT"],
        ce_key_by_inst={"FT": "nce", "QTOF": "ace"},
    )

    assert torch.equal(result, torch.tensor([20.0, 45.0]))


def test_select_context_ce_values_matches_sampler_valid_value_filtering():
    batch = {
        "spec_nce": torch.tensor([float("nan"), 35.0, -1.0, 45.0]),
        "spec_nce_batch_idxs": torch.tensor([0, 0, 1, 1]),
        "spec_inst_type": torch.tensor([0, 0]),
    }

    result = select_context_ce_values(
        batch,
        batch_size=2,
        inst_types=["FT"],
        ce_key_by_inst={"FT": "nce"},
    )

    assert torch.equal(result, torch.tensor([35.0, 45.0]))


class RecordingBatchSampler:
    """Minimal batch sampler recording seeded recomputations."""

    def __init__(self):
        self.generator = None
        self.precompute_calls = 0
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)
        self._pre_compute_batches()

    def _pre_compute_batches(self):
        self.precompute_calls += 1


class RecordingDataLoader:
    """Minimal dataloader with a configurable batch count."""

    def __init__(self, batch_sampler, num_batches):
        self.batch_sampler = batch_sampler
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches


class GrowingCurriculumBatchSampler(RecordingBatchSampler):
    """Batch sampler that expands from an epoch-zero subset to the full data."""

    def __init__(self):
        super().__init__()
        self.num_batches = 25

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if epoch >= 10:
            self.num_batches = 100


class DynamicLengthDataLoader:
    """Dataloader whose length follows its batch sampler."""

    def __init__(self, batch_sampler):
        self.batch_sampler = batch_sampler

    def __len__(self):
        return self.batch_sampler.num_batches


def test_seed_dataloader_refreshes_lightning_cached_batch_count():
    batch_sampler = RecordingBatchSampler()
    train_dataloader = RecordingDataLoader(batch_sampler, num_batches=17)
    limits = [4]
    combined_loader = types.SimpleNamespace(
        limits=limits, _iterator=types.SimpleNamespace(limits=limits)
    )
    data_fetcher = types.SimpleNamespace(reset=MagicMock())
    module = types.SimpleNamespace(
        current_epoch=2,
        hparams=types.SimpleNamespace(
            dag_size_curriculum=True,
            bond_size_curriculum=False,
            batch_sampler_type="spec_mol_frag",
            masster_batch_sampler=False,
        ),
        train_dl_seeds=torch.tensor([10, 11, 12, 13]),
        trainer=types.SimpleNamespace(
            train_dataloader=train_dataloader,
            fit_loop=types.SimpleNamespace(
                _combined_loader=combined_loader,
                _data_fetcher=data_fetcher,
                max_batches=4,
            ),
            val_check_interval=1.0,
            val_check_batch=4,
        ),
    )

    SpectrumPL._seed_dataloader(module, 2)

    assert batch_sampler.precompute_calls == 1
    assert batch_sampler.epochs == [2]
    assert batch_sampler.generator.initial_seed() == 12
    assert module.trainer.fit_loop._combined_loader.limits == [17]
    assert module.trainer.fit_loop._combined_loader._iterator.limits == [17]
    data_fetcher.reset.assert_called_once_with()
    assert module.trainer.fit_loop.max_batches == 17
    assert module.trainer.val_check_batch == 17


def test_seed_dataloader_refreshes_cached_batch_count_after_resume():
    batch_sampler = RecordingBatchSampler()
    train_dataloader = RecordingDataLoader(batch_sampler, num_batches=40)
    data_fetcher = types.SimpleNamespace(reset=MagicMock())
    module = types.SimpleNamespace(
        current_epoch=30,
        hparams=types.SimpleNamespace(
            dag_size_curriculum=True,
            bond_size_curriculum=False,
            batch_sampler_type="spec_mol_frag",
            masster_batch_sampler=False,
        ),
        train_dl_seeds=torch.arange(31),
        trainer=types.SimpleNamespace(
            train_dataloader=train_dataloader,
            fit_loop=types.SimpleNamespace(
                _combined_loader=types.SimpleNamespace(limits=[10]),
                _data_fetcher=data_fetcher,
                max_batches=10,
            ),
            val_check_interval=0.5,
            val_check_batch=5,
        ),
    )

    SpectrumPL._seed_dataloader(module, 30)

    assert batch_sampler.precompute_calls == 1
    assert batch_sampler.epochs == [30]
    assert batch_sampler.generator.initial_seed() == 30
    assert module.trainer.fit_loop._combined_loader.limits == [40]
    data_fetcher.reset.assert_called_once_with()
    assert module.trainer.fit_loop.max_batches == 40
    assert module.trainer.val_check_batch == 20


def test_seed_dataloader_advances_outer_curriculum_sampler_at_epoch_99():
    batch_sampler = GrowingCurriculumBatchSampler()
    train_dataloader = DynamicLengthDataLoader(batch_sampler)
    data_fetcher = types.SimpleNamespace(reset=MagicMock())
    module = types.SimpleNamespace(
        current_epoch=99,
        hparams=types.SimpleNamespace(
            dag_size_curriculum=True,
            bond_size_curriculum=False,
            batch_sampler_type="spec_mol_frag",
            masster_batch_sampler=False,
        ),
        train_dl_seeds=torch.arange(100),
        trainer=types.SimpleNamespace(
            train_dataloader=train_dataloader,
            fit_loop=types.SimpleNamespace(
                _combined_loader=types.SimpleNamespace(limits=[25]),
                _data_fetcher=data_fetcher,
                max_batches=25,
            ),
            val_check_interval=1.0,
            val_check_batch=25,
        ),
    )

    SpectrumPL._seed_dataloader(module, 99)

    assert batch_sampler.epochs == [99]
    assert batch_sampler.num_batches == 100
    assert module.trainer.fit_loop._combined_loader.limits == [100]
    data_fetcher.reset.assert_called_once_with()
    assert module.trainer.fit_loop.max_batches == 100
    assert module.trainer.val_check_batch == 100


def test_seed_dataloader_does_not_change_batch_counts_without_curriculum():
    batch_sampler = RecordingBatchSampler()
    train_dataloader = RecordingDataLoader(batch_sampler, num_batches=40)
    module = types.SimpleNamespace(
        current_epoch=2,
        hparams=types.SimpleNamespace(
            dag_size_curriculum=False,
            bond_size_curriculum=False,
            batch_sampler_type="spec_mol_frag",
            masster_batch_sampler=False,
        ),
        train_dl_seeds=torch.tensor([10, 11, 12]),
        trainer=types.SimpleNamespace(
            train_dataloader=train_dataloader,
            fit_loop=types.SimpleNamespace(
                _combined_loader=types.SimpleNamespace(limits=[10]),
                _data_fetcher=types.SimpleNamespace(reset=MagicMock()),
                max_batches=10,
            ),
            val_check_interval=1.0,
            val_check_batch=10,
        ),
    )

    SpectrumPL._seed_dataloader(module, 2)

    assert batch_sampler.precompute_calls == 1
    assert batch_sampler.epochs == []
    assert module.trainer.fit_loop._combined_loader.limits == [10]
    module.trainer.fit_loop._data_fetcher.reset.assert_not_called()
    assert module.trainer.fit_loop.max_batches == 10
    assert module.trainer.val_check_batch == 10


@pytest.mark.parametrize(
    ("batch_sampler_type", "masster_batch_sampler"),
    [("ce_context", False), ("spec_mol_frag", True)],
)
def test_seed_dataloader_advances_ce_context_sampler(
    batch_sampler_type,
    masster_batch_sampler,
):
    batch_sampler = RecordingBatchSampler()
    train_dataloader = RecordingDataLoader(batch_sampler, num_batches=17)
    data_fetcher = types.SimpleNamespace(reset=MagicMock())
    module = types.SimpleNamespace(
        current_epoch=2,
        hparams=types.SimpleNamespace(
            dag_size_curriculum=False,
            bond_size_curriculum=False,
            batch_sampler_type=batch_sampler_type,
            masster_batch_sampler=masster_batch_sampler,
        ),
        train_dl_seeds=torch.tensor([10, 11, 12]),
        trainer=types.SimpleNamespace(
            train_dataloader=train_dataloader,
            fit_loop=types.SimpleNamespace(
                _combined_loader=types.SimpleNamespace(limits=[4]),
                _data_fetcher=data_fetcher,
                max_batches=4,
            ),
            val_check_interval=1.0,
            val_check_batch=4,
        ),
    )

    SpectrumPL._seed_dataloader(module, 2)

    assert batch_sampler.epochs == [2]
    assert module.trainer.fit_loop._combined_loader.limits == [17]
    data_fetcher.reset.assert_called_once_with()


# ===========================================================================
# get_split_fn
# ===========================================================================


class TestGetSplitFn:
    """get_split_fn(bs_idx) -> split_fn that splits tensors by batch counts."""

    def _idx(self, *counts):
        """Build a sorted batch-index tensor from per-batch counts."""
        return torch.cat([torch.full((c,), i, dtype=torch.long) for i, c in enumerate(counts)])

    # --- basic correctness --------------------------------------------------

    def test_splits_1d_tensor_two_batches(self):
        bs_idx = self._idx(3, 2)
        split_fn = get_split_fn(bs_idx)
        t = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        parts = split_fn(t)
        assert len(parts) == 2
        torch.testing.assert_close(parts[0], torch.tensor([10.0, 20.0, 30.0]))
        torch.testing.assert_close(parts[1], torch.tensor([40.0, 50.0]))

    def test_splits_equal_sized_batches(self):
        bs_idx = self._idx(2, 2, 2)
        split_fn = get_split_fn(bs_idx)
        t = torch.arange(6, dtype=torch.float)
        parts = split_fn(t)
        assert len(parts) == 3
        for i, part in enumerate(parts):
            torch.testing.assert_close(part, torch.arange(i * 2, i * 2 + 2, dtype=torch.float))

    def test_single_batch(self):
        bs_idx = self._idx(4)
        split_fn = get_split_fn(bs_idx)
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        (part,) = split_fn(t)
        torch.testing.assert_close(part, t)

    def test_splits_2d_tensor(self):
        """th.split works on any dimension; split_fn should pass through."""
        bs_idx = self._idx(2, 3)
        split_fn = get_split_fn(bs_idx)
        t = torch.ones(5, 8)
        parts = split_fn(t)
        assert parts[0].shape == (2, 8)
        assert parts[1].shape == (3, 8)

    def test_unequal_batches_lengths_correct(self):
        bs_idx = self._idx(1, 4, 2)
        split_fn = get_split_fn(bs_idx)
        t = torch.arange(7, dtype=torch.float)
        parts = split_fn(t)
        assert [len(p) for p in parts] == [1, 4, 2]

    # --- assertion / edge cases ---------------------------------------------

    def test_empty_idx_no_assertion_error(self):
        """Empty batch_idx: diffs is empty → len(diffs)==0 branch → no assert."""
        bs_idx = torch.tensor([], dtype=torch.long)
        # Should not raise
        split_fn = get_split_fn(bs_idx)
        parts = split_fn(torch.tensor([]))
        assert len(parts) == 0

    def test_non_monotone_idx_raises(self):
        """Out-of-order indices (e.g. [0,1,0]) must trigger the assertion."""
        bs_idx = torch.tensor([0, 1, 0], dtype=torch.long)
        with pytest.raises(AssertionError):
            get_split_fn(bs_idx)

    def test_single_element_batches(self):
        bs_idx = self._idx(1, 1, 1)
        split_fn = get_split_fn(bs_idx)
        t = torch.tensor([7.0, 8.0, 9.0])
        parts = split_fn(t)
        assert [p.item() for p in parts] == [7.0, 8.0, 9.0]

    # --- consistency with ragged_to_batch -----------------------------------

    def test_consistent_with_ragged_to_batch(self):
        """Splitting then stacking should match ragged_to_batch output (padded rows)."""
        bs_idx = self._idx(3, 2)
        feat = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        split_fn = get_split_fn(bs_idx)
        parts = split_fn(feat)

        batched = ragged_to_batch(feat, bs_idx, pad_value=0.0)
        # Row 0: [1,2,3], row 1: [4,5,0]
        torch.testing.assert_close(batched[0, :3], parts[0])
        torch.testing.assert_close(batched[1, :2], parts[1])
        assert batched[1, 2].item() == 0.0  # padding


# ===========================================================================
# primary_ce_to_batch
# ===========================================================================


class TestPrimaryCeToBatch:
    """primary_ce_to_batch(spec_ce, spec_ce_batch_idxs, batch_size) -> CE per sample."""

    def test_flattened_ce_uses_first_value_per_sample(self):
        spec_ce = torch.tensor([10.0, 20.0, 30.0, 31.0, 40.0])
        spec_ce_batch_idxs = torch.tensor([0, 1, 2, 2, 3], dtype=torch.long)

        out = primary_ce_to_batch(spec_ce, spec_ce_batch_idxs, batch_size=4)

        torch.testing.assert_close(out, torch.tensor([10.0, 20.0, 30.0, 40.0]))

    def test_2d_ce_uses_first_column(self):
        spec_ce = torch.tensor([[11.0, 12.0], [21.0, 22.0], [31.0, 32.0]])

        out = primary_ce_to_batch(spec_ce, None, batch_size=3)

        torch.testing.assert_close(out, torch.tensor([11.0, 21.0, 31.0]))


# ===========================================================================
# list_to_batch_tensor
# ===========================================================================


class TestListToBatchTensor:
    """list_to_batch_tensor(inp, pad) pads a list of 1D tensors to a 2D batch."""

    def test_all_same_length_no_padding(self):
        inp = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        out = list_to_batch_tensor(inp, pad=0.0)
        assert out.shape == (2, 2)
        torch.testing.assert_close(out[0], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(out[1], torch.tensor([3.0, 4.0]))

    def test_shorter_row_padded_at_end(self):
        inp = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])]
        out = list_to_batch_tensor(inp, pad=0.0)
        assert out.shape == (2, 3)
        torch.testing.assert_close(out[0], torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(out[1], torch.tensor([4.0, 5.0, 0.0]))

    def test_pad_value_inf(self):
        inp = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
        out = list_to_batch_tensor(inp, pad=float("inf"))
        assert out.shape == (2, 2)
        assert out[1, 0].item() == 3.0
        assert out[1, 1].item() == float("inf")

    def test_single_element_list(self):
        inp = [torch.tensor([5.0, 6.0, 7.0])]
        out = list_to_batch_tensor(inp, pad=0.0)
        assert out.shape == (1, 3)
        torch.testing.assert_close(out[0], torch.tensor([5.0, 6.0, 7.0]))

    def test_dtype_preserved(self):
        inp = [torch.tensor([1, 2], dtype=torch.long), torch.tensor([3], dtype=torch.long)]
        out = list_to_batch_tensor(inp, pad=0)
        assert out.dtype == torch.long

    def test_multiple_rows_different_lengths(self):
        inp = [torch.ones(5), torch.ones(3), torch.ones(1)]
        out = list_to_batch_tensor(inp, pad=-1.0)
        assert out.shape == (3, 5)
        assert (out[1, 3:] == -1.0).all()
        assert (out[2, 1:] == -1.0).all()

    def test_output_is_2d(self):
        inp = [torch.arange(4, dtype=torch.float), torch.arange(2, dtype=torch.float)]
        out = list_to_batch_tensor(inp, pad=0.0)
        assert out.ndim == 2

    def test_values_not_mutated(self):
        """Original tensors are not modified by padding."""
        t = torch.tensor([1.0, 2.0])
        list_to_batch_tensor([t, torch.tensor([3.0])], pad=99.0)
        torch.testing.assert_close(t, torch.tensor([1.0, 2.0]))


# ===========================================================================
# ragged_to_batch
# ===========================================================================


class TestRaggedToBatch:
    """ragged_to_batch(feat, batch_idx, pad_value) → (B, max_len) tensor."""

    def _make(self, *groups):
        """groups: list of lists → (feat, batch_idx)."""
        feat = torch.cat([torch.tensor(g, dtype=torch.float) for g in groups])
        batch_idx = torch.cat(
            [torch.full((len(g),), i, dtype=torch.long) for i, g in enumerate(groups)]
        )
        return feat, batch_idx

    # --- basic correctness --------------------------------------------------

    def test_two_equal_groups(self):
        feat, batch_idx = self._make([1.0, 2.0], [3.0, 4.0])
        out = ragged_to_batch(feat, batch_idx, pad_value=0.0)
        assert out.shape == (2, 2)
        torch.testing.assert_close(out[0], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(out[1], torch.tensor([3.0, 4.0]))

    def test_unequal_groups_shorter_padded(self):
        feat, batch_idx = self._make([1.0, 2.0, 3.0], [4.0, 5.0])
        out = ragged_to_batch(feat, batch_idx, pad_value=-1.0)
        assert out.shape == (2, 3)
        torch.testing.assert_close(out[0], torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(out[1], torch.tensor([4.0, 5.0, -1.0]))

    def test_single_group(self):
        feat, batch_idx = self._make([10.0, 20.0, 30.0])
        out = ragged_to_batch(feat, batch_idx, pad_value=0.0)
        assert out.shape == (1, 3)
        torch.testing.assert_close(out[0], torch.tensor([10.0, 20.0, 30.0]))

    def test_three_groups_mixed_lengths(self):
        feat, batch_idx = self._make([1.0], [2.0, 3.0], [4.0, 5.0, 6.0])
        out = ragged_to_batch(feat, batch_idx, pad_value=0.0)
        assert out.shape == (3, 3)
        assert out[0, 0].item() == 1.0
        assert (out[0, 1:] == 0.0).all()
        torch.testing.assert_close(out[1, :2], torch.tensor([2.0, 3.0]))
        assert out[1, 2].item() == 0.0
        torch.testing.assert_close(out[2], torch.tensor([4.0, 5.0, 6.0]))

    def test_pad_value_inf(self):
        feat, batch_idx = self._make([1.0, 2.0], [3.0])
        out = ragged_to_batch(feat, batch_idx, pad_value=float("inf"))
        assert out[1, 1].item() == float("inf")

    def test_default_pad_value_is_zero(self):
        feat, batch_idx = self._make([1.0, 2.0], [3.0])
        out = ragged_to_batch(feat, batch_idx)
        assert out[1, 1].item() == 0.0

    def test_dtype_preserved_float32(self):
        feat = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        batch_idx = torch.tensor([0, 0, 1], dtype=torch.long)
        out = ragged_to_batch(feat, batch_idx, pad_value=0.0)
        assert out.dtype == torch.float32

    def test_dtype_preserved_int64(self):
        feat = torch.tensor([10, 20, 30], dtype=torch.long)
        batch_idx = torch.tensor([0, 1, 1], dtype=torch.long)
        out = ragged_to_batch(feat, batch_idx, pad_value=0)
        assert out.dtype == torch.long

    # --- equivalence with list_to_batch_tensor ------------------------------

    def test_equivalent_to_list_to_batch_tensor(self):
        """ragged_to_batch and list_to_batch_tensor must agree on padded values."""
        groups = [[1.0, 2.0, 3.0], [4.0, 5.0], [6.0]]
        inp_list = [torch.tensor(g, dtype=torch.float) for g in groups]
        feat, batch_idx = self._make(*groups)

        out_ragged = ragged_to_batch(feat, batch_idx, pad_value=0.0)
        out_list = list_to_batch_tensor(inp_list, pad=0.0)

        torch.testing.assert_close(out_ragged, out_list)

    def test_equivalent_to_list_to_batch_tensor_inf_pad(self):
        groups = [[1.0, 2.0], [3.0, 4.0, 5.0]]
        inp_list = [torch.tensor(g, dtype=torch.float) for g in groups]
        feat, batch_idx = self._make(*groups)

        out_ragged = ragged_to_batch(feat, batch_idx, pad_value=float("inf"))
        out_list = list_to_batch_tensor(inp_list, pad=float("inf"))

        torch.testing.assert_close(out_ragged, out_list)

    # --- assertions ---------------------------------------------------------

    def test_2d_feat_raises(self):
        feat = torch.ones(4, 2)  # ndim == 2
        batch_idx = torch.tensor([0, 0, 1, 1])
        with pytest.raises(AssertionError):
            ragged_to_batch(feat, batch_idx)

    def test_mismatched_sizes_raises(self):
        feat = torch.ones(5)
        batch_idx = torch.tensor([0, 0, 1])  # length 3 ≠ 5
        with pytest.raises(AssertionError):
            ragged_to_batch(feat, batch_idx)

    def test_2d_batch_idx_raises(self):
        feat = torch.ones(4)
        batch_idx = torch.ones(4, 1, dtype=torch.long)  # ndim == 2
        with pytest.raises(AssertionError):
            ragged_to_batch(feat, batch_idx)

    # --- round-trip with get_split_fn ---------------------------------------

    def test_roundtrip_split_then_ragged(self):
        """Split a flat tensor then re-batch it: must recover the original."""
        feat = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        batch_idx = torch.tensor([0, 0, 0, 1, 1])
        split_fn = get_split_fn(batch_idx)
        parts = split_fn(feat)
        # Re-concatenate and re-batch
        feat2 = torch.cat(parts)
        out = ragged_to_batch(feat2, batch_idx, pad_value=0.0)
        assert out.shape == (2, 3)
        torch.testing.assert_close(out[0], torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(out[1, :2], torch.tensor([4.0, 5.0]))
        assert out[1, 2].item() == 0.0
