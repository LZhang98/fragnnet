"""Unit tests for fragnnet.dataset.data_sampler module.

Tests for batch samplers, group samplers, and dynamic batch sampling strategies.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as th
from torch.utils.data import DataLoader, RandomSampler

# Load data_sampler directly to avoid package-level import side-effects
repo_root = Path(__file__).resolve().parents[1]
data_sampler_path = repo_root / "src" / "fragnnet" / "dataset" / "data_sampler.py"
spec = importlib.util.spec_from_file_location("data_sampler", str(data_sampler_path))
data_sampler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_sampler)

SpecMolFragDynamicBatchSampler = data_sampler.SpecMolFragDynamicBatchSampler
DualGroupDynamicBatchSampler = data_sampler.DualGroupDynamicBatchSampler
GroupSampler = data_sampler.GroupSampler
FormulaPrecTypeGroupSampler = data_sampler.FormulaPrecTypeGroupSampler


# ============================================================================
# Mock and Helper Classes
# ============================================================================


class MockFrag:
    """Mock fragment for testing."""

    def __init__(self, nodes: int, edges: int = 0):
        self.num_nodes = nodes
        self.num_edges = edges


class FakeFrag:
    """Alternative mock fragment."""

    def __init__(self, nodes, edges):
        self.num_nodes = nodes
        self.num_edges = edges


class SyntheticDataset:
    """Synthetic dataset for testing samplers."""

    def __init__(self, items):
        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


class FakeDataset:
    """Lightweight dataset for dynamic batch sampler tests."""

    def __init__(self, sizes):
        self.sizes = sizes

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, idx):
        nodes, edges = self.sizes[idx]
        return {"frag_pyg": SimpleNamespace(num_nodes=nodes, num_edges=edges)}


class DummyDataset:
    """Dummy dataset for dual group sampler tests."""

    def __init__(self, samples):
        self._s = samples

    def __len__(self):
        return len(self._s)

    def __getitem__(self, idx):
        return self._s[idx]


class DS2:
    """Another dataset variant."""

    def __init__(self, sizes):
        self.sizes = sizes

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, idx):
        nodes, edges = self.sizes[idx]
        return {"frag_pyg": SimpleNamespace(num_nodes=nodes, num_edges=edges)}


# ============================================================================
# Tests for SpecMolFragDynamicBatchSampler
# ============================================================================


class TestSpecMolFragDynamicBatchSampler:
    """Tests for dynamic batch sampling with node/edge constraints."""

    def test_dynamic_batch_sampler(self):
        """Test basic dynamic batch sampler functionality."""
        sizes = [(1, 2), (3, 4), (2, 1), (8, 2), (5, 6), (2, 2), (4, 1), (6, 6)]
        ds = FakeDataset(sizes)
        sampler = SpecMolFragDynamicBatchSampler(data_source=ds, max_num=10, limited_by="frag_node")
        dl = DataLoader(ds, batch_sampler=sampler, collate_fn=lambda x: x)
        batches = list(dl)
        assert len(batches) > 0

    def test_dynamic_batch_sampler_basic(self):
        """Test batches respect max_num constraint."""
        data_list = [
            {"idx": i, "frag_pyg": MockFrag(100 - i * 5, 12000 - i * 1000)} for i in range(12)
        ]
        ds = SyntheticDataset(data_list)

        sampler = SpecMolFragDynamicBatchSampler(
            data_source=ds, max_num=50000, limited_by="frag_edge", skip_too_big=False, sampler=None
        )

        batches = list(sampler)

        # Verify batches respect max_num constraint
        batch_edges = [sum(ds[idx]["frag_pyg"].num_edges for idx in batch) for batch in batches]
        assert all(e <= 50000 for e in batch_edges), "Some batches exceed max_num limit"

        # Verify no samples are duplicated or missing
        all_indices = [idx for batch in batches for idx in batch]
        expected_indices = set(range(len(ds)))
        actual_indices = set(all_indices)
        assert actual_indices == expected_indices

    def test_dynamic_batch_sampler_epoch_aware(self):
        """Test epoch setting affects batching order."""
        data_list = [
            {"idx": i, "frag_pyg": MockFrag(100 - i * 5, 12000 - i * 1000)} for i in range(12)
        ]
        ds = SyntheticDataset(data_list)

        generator = th.Generator()
        base_sampler = RandomSampler(ds, generator=generator)

        sampler = SpecMolFragDynamicBatchSampler(
            data_source=ds,
            max_num=50000,
            limited_by="frag_edge",
            skip_too_big=False,
            sampler=base_sampler,
        )

        epoch0_batches = list(sampler)
        sampler.set_epoch(1)
        epoch1_batches = list(sampler)

        assert len(epoch0_batches) > 0
        assert len(epoch1_batches) > 0

        # Setting same epoch should be deterministic
        sampler.set_epoch(0)
        epoch0_again = list(sampler)
        assert epoch0_batches == epoch0_again

    def test_dynamic_batch_sampler_no_base_sampler(self):
        """Test sampler works without base sampler."""
        data_list = [
            {"idx": i, "frag_pyg": MockFrag(100 - i * 5, 12000 - i * 1000)} for i in range(12)
        ]
        ds = SyntheticDataset(data_list)

        sampler = SpecMolFragDynamicBatchSampler(
            data_source=ds, max_num=50000, limited_by="frag_edge", skip_too_big=False, sampler=None
        )

        batches1 = list(sampler)
        batches2 = list(sampler)
        assert batches1 == batches2

    def test_dynamic_batch_sampler_skip_too_big(self):
        """Test skip_too_big flag."""
        data_list = [
            {"idx": 0, "frag_pyg": MockFrag(100, 12000)},
            {"idx": 1, "frag_pyg": MockFrag(100, 60000)},  # Too big
            {"idx": 2, "frag_pyg": MockFrag(110, 13000)},
        ]
        ds = SyntheticDataset(data_list)

        # With skip_too_big=True
        sampler = SpecMolFragDynamicBatchSampler(
            data_source=ds, max_num=50000, limited_by="frag_edge", skip_too_big=True
        )
        batches = list(sampler)
        all_indices = [idx for batch in batches for idx in batch]
        assert 1 not in all_indices

        # With skip_too_big=False
        sampler = SpecMolFragDynamicBatchSampler(
            data_source=ds, max_num=50000, limited_by="frag_edge", skip_too_big=False
        )
        batches = list(sampler)
        all_indices = [idx for batch in batches for idx in batch]
        assert 1 in all_indices

    def test_specmolfrag_dynamic_batch_nodes_limit_skip_true(self):
        """Test nodes limit with skip_too_big=True."""
        sizes = [(3, 1), (4, 2), (12, 4), (2, 1)]  # index 2 is too large
        ds = DS2(sizes)
        sampler = SpecMolFragDynamicBatchSampler(
            ds, max_num=10, limited_by="frag_node", skip_too_big=True
        )
        batches = list(sampler)
        all_indices = [i for b in batches for i in b]
        assert 2 not in all_indices
        assert all(sum(ds[i]["frag_pyg"].num_nodes for i in b) <= 10 for b in batches)

    def test_specmolfrag_dynamic_batch_nodes_limit_skip_false(self):
        """Test nodes limit with skip_too_big=False allows overflow."""
        sizes = [(3, 1), (4, 2), (12, 4), (2, 1)]
        ds = DS2(sizes)
        sampler = SpecMolFragDynamicBatchSampler(
            ds, max_num=10, limited_by="frag_node", skip_too_big=False
        )
        batches = list(sampler)
        all_indices = [i for b in batches for i in b]
        # Index 2 should be included despite exceeding budget
        assert 2 in all_indices


# ============================================================================
# Tests for DualGroupDynamicBatchSampler
# ============================================================================


class TestDualGroupDynamicBatchSampler:
    """Tests for dual-group dynamic batch sampling."""

    def test_dual_group_dynamic_batch_sampler_nodes_limit(self):
        """Test node budget constraint."""
        samples = [
            {"frag_pyg": FakeFrag(3, 2), "adduct": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(2, 1), "adduct": "A", "mol_id": 1, "formula": "Y"},
            {"frag_pyg": FakeFrag(4, 3), "adduct": "A", "mol_id": 2, "formula": "X"},
            {"frag_pyg": FakeFrag(1, 1), "adduct": "B", "mol_id": 3, "formula": "Z"},
            {"frag_pyg": FakeFrag(5, 4), "adduct": "B", "mol_id": 3, "formula": "Z"},
            {"frag_pyg": FakeFrag(2, 1), "adduct": "C", "mol_id": 4, "formula": "W"},
        ]

        ds = DummyDataset(samples)

        gen = th.Generator()
        gen.manual_seed(42)

        sampler = DualGroupDynamicBatchSampler(
            ds,
            max_num=6,
            limited_by="frag_node",
            skip_too_big=True,
            generator=gen,
        )

        batches = list(iter(sampler))

        # Expect at least one batch and all indices included
        assert len(batches) > 0
        included = set(i for b in batches for i in b)
        assert included == set(range(len(ds)))

        # Each batch must respect node budget
        for b in batches:
            total_nodes = sum(samples[i]["frag_pyg"].num_nodes for i in b)
            assert total_nodes <= 6

    def test_dual_group_dynamic_batch_sampler_edges_limit(self):
        """Test edge budget constraint."""
        samples = [
            {"frag_pyg": FakeFrag(3, 5), "adduct": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(2, 3), "adduct": "A", "mol_id": 1, "formula": "Y"},
            {"frag_pyg": FakeFrag(4, 7), "adduct": "A", "mol_id": 2, "formula": "X"},
            {"frag_pyg": FakeFrag(1, 2), "adduct": "B", "mol_id": 3, "formula": "Z"},
        ]

        ds = DummyDataset(samples)
        gen = th.Generator()
        gen.manual_seed(42)

        sampler = DualGroupDynamicBatchSampler(
            ds,
            max_num=10,
            limited_by="frag_edge",
            skip_too_big=False,
            generator=gen,
        )

        batches = list(iter(sampler))

        # Each batch must respect edge budget
        for b in batches:
            total_edges = sum(samples[i]["frag_pyg"].num_edges for i in b)
            assert total_edges <= 10

    def test_dual_group_overlap_removal(self):
        """Test overlapping samples removed from group2."""
        samples = [
            {"frag_pyg": FakeFrag(3, 2), "adduct": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(2, 1), "adduct": "A", "mol_id": 2, "formula": "X"},
            {"frag_pyg": FakeFrag(4, 3), "adduct": "B", "mol_id": 5, "formula": "Y"},
            {"frag_pyg": FakeFrag(1, 1), "adduct": "C", "mol_id": 3, "formula": "Z"},
        ]

        ds = DummyDataset(samples)
        gen = th.Generator()
        gen.manual_seed(42)

        sampler = DualGroupDynamicBatchSampler(
            ds,
            max_num=10,
            limited_by="frag_node",
            generator=gen,
        )

        # Check overlap removal
        assert len(sampler._group2) == 0

    def test_dual_group_sampler_skip_too_big(self):
        """Test skip_too_big flag."""
        samples = [
            {"frag_pyg": FakeFrag(3, 2), "adduct": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(100, 1), "adduct": "B", "mol_id": 2, "formula": "Y"},
        ]

        ds = DummyDataset(samples)
        gen = th.Generator()
        gen.manual_seed(42)

        sampler = DualGroupDynamicBatchSampler(
            ds,
            max_num=10,
            limited_by="frag_node",
            skip_too_big=True,
            generator=gen,
        )

        batches = list(iter(sampler))
        all_indices = [i for b in batches for i in b]
        assert 1 not in all_indices

    def test_dual_group_sampler_deterministic_with_generator(self):
        """Test determinism with manual seed."""
        samples = [
            {"frag_pyg": FakeFrag(3, 2), "adduct": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(2, 1), "adduct": "A", "mol_id": 1, "formula": "Y"},
            {"frag_pyg": FakeFrag(4, 3), "adduct": "B", "mol_id": 2, "formula": "X"},
        ]

        ds = DummyDataset(samples)

        g1 = th.Generator()
        g1.manual_seed(123)
        s1 = DualGroupDynamicBatchSampler(ds, max_num=10, generator=g1)
        out1 = list(iter(s1))

        g2 = th.Generator()
        g2.manual_seed(123)
        s2 = DualGroupDynamicBatchSampler(ds, max_num=10, generator=g2)
        out2 = list(iter(s2))

        assert out1 == out2

    def test_dual_group_sampler_set_epoch(self):
        """Test set_epoch changes order."""
        samples = [
            {"frag_pyg": FakeFrag(3, 2), "adduct": "A", "mol_id": 1, "formula": "X"},
            {"frag_pyg": FakeFrag(2, 1), "adduct": "A", "mol_id": 1, "formula": "Y"},
            {"frag_pyg": FakeFrag(4, 3), "adduct": "B", "mol_id": 2, "formula": "X"},
        ]

        ds = DummyDataset(samples)
        gen = th.Generator()
        gen.manual_seed(99)
        sampler = DualGroupDynamicBatchSampler(ds, max_num=10, generator=gen)

        sampler.set_epoch(0)
        out0 = list(iter(sampler))

        sampler.set_epoch(1)
        out1 = list(iter(sampler))

        # Different epochs should produce different orders
        assert out0 != out1


# ============================================================================
# Tests for FormulaPrecTypeGroupSampler
# ============================================================================


class TestFormulaPrecTypeGroupSampler:
    """Tests for formula and precursor type grouping."""

    def test_grouping_basic(self):
        """Test basic grouping by formula and prec_type."""
        items = []
        # cluster A: prec_type 'A', formula 'F1'
        for i in range(3):
            items.append(
                {
                    "frag_pyg": MockFrag(nodes=1 + i, edges=0),
                    "formula": "F1",
                    "prec_type": "A",
                    "mol_id": f"m{i}",
                    "group_id": th.tensor([0]),
                }
            )
        # cluster B: prec_type 'B', formula 'F2'
        for i in range(2):
            items.append(
                {
                    "frag_pyg": MockFrag(nodes=2 + i, edges=0),
                    "formula": "F2",
                    "prec_type": "B",
                    "mol_id": f"m{3 + i}",
                    "group_id": th.tensor([1]),
                }
            )
        # single cluster C
        items.append(
            {
                "frag_pyg": MockFrag(nodes=4, edges=0),
                "formula": "F3",
                "prec_type": "C",
                "mol_id": "m5",
                "group_id": th.tensor([2]),
            }
        )

        ds = SyntheticDataset(items)
        gen = th.Generator()
        gen.manual_seed(42)
        sampler = FormulaPrecTypeGroupSampler(ds, sample_k=1, generator=gen)

        counts = {}
        for idx in sampler:
            item = ds[idx]
            key = f"{item['prec_type']}::{item['formula']}"
            counts[key] = counts.get(key, 0) + 1

        # Each key yields at most sample_k
        assert all(c <= 1 for c in counts.values())

    def test_integration_with_dynamic_batch_sampler(self):
        """Test integration with SpecMolFragDynamicBatchSampler."""
        items = []
        for i in range(3):
            items.append(
                {
                    "frag_pyg": MockFrag(nodes=1 + i, edges=0),
                    "formula": "F1",
                    "prec_type": "A",
                    "mol_id": f"m{i}",
                    "group_id": th.tensor([0]),
                }
            )
        for i in range(2):
            items.append(
                {
                    "frag_pyg": MockFrag(nodes=2 + i, edges=0),
                    "formula": "F2",
                    "prec_type": "B",
                    "mol_id": f"m{3 + i}",
                    "group_id": th.tensor([1]),
                }
            )

        ds = SyntheticDataset(items)
        gen = th.Generator()
        gen.manual_seed(123)
        sampler = FormulaPrecTypeGroupSampler(ds, sample_k=2, generator=gen)

        batch_sampler = SpecMolFragDynamicBatchSampler(
            data_source=ds, max_num=6, limited_by="frag_node", sampler=sampler
        )
        batch_sampler.set_epoch(0)

        underlying = set(list(sampler))

        for batch in batch_sampler:
            total_nodes = sum(ds[idx]["frag_pyg"].num_nodes for idx in batch)
            assert total_nodes <= 6
            for idx in batch:
                assert idx in underlying

    def test_set_epoch_and_determinism(self):
        """Test determinism with set_epoch."""
        items = []
        for i in range(3):
            items.append(
                {
                    "frag_pyg": MockFrag(nodes=1 + i, edges=0),
                    "formula": "F1",
                    "prec_type": "A",
                    "mol_id": f"m{i}",
                    "group_id": th.tensor([0]),
                }
            )

        ds = SyntheticDataset(items)

        g1 = th.Generator()
        g1.manual_seed(999)
        s1 = FormulaPrecTypeGroupSampler(ds, sample_k=2, generator=g1)
        b1 = SpecMolFragDynamicBatchSampler(ds, max_num=9999, sampler=s1)
        b1.set_epoch(2)
        out1 = list(b1)

        g2 = th.Generator()
        g2.manual_seed(999)
        s2 = FormulaPrecTypeGroupSampler(ds, sample_k=2, generator=g2)
        b2 = SpecMolFragDynamicBatchSampler(ds, max_num=9999, sampler=s2)
        b2.set_epoch(2)
        out2 = list(b2)

        assert out1 == out2

        # Changing epoch should change ordering
        b2.set_epoch(3)
        out3 = list(b2)
        assert out1 != out3

    def test_missing_keys_raises(self):
        """Test that missing keys raise KeyError."""
        items = [
            {
                "frag_pyg": MockFrag(1),
                "formula": "F1",
                "mol_id": "m0",
                "group_id": th.tensor([0]),
            }  # missing 'prec_type'
        ]
        ds = SyntheticDataset(items)
        with pytest.raises(KeyError):
            FormulaPrecTypeGroupSampler(ds)


# ============================================================================
# Tests for GroupSampler
# ============================================================================


class TestGroupSampler:
    """Tests for basic group sampling."""

    def test_group_sampler_basic_and_epoch(self):
        """Test basic GroupSampler functionality."""
        items = []
        sizes = [4, 2, 1]
        for g, size in enumerate(sizes):
            for i in range(size):
                items.append(
                    {
                        "frag_pyg": MockFrag(1 + i),
                        "group_id": th.tensor([g]),
                    }
                )

        ds = SyntheticDataset(items)
        gen = th.Generator()
        gen.manual_seed(2024)
        sampler = GroupSampler(ds, sample_k=2, generator=gen)

        # Count samples per group
        counts = {}
        for idx in sampler:
            gid = ds[idx]["group_id"].item()
            counts[gid] = counts.get(gid, 0) + 1

        # Each group yields at most sample_k
        assert all(counts.get(g, 0) <= 2 for g in range(len(sizes)))

        expected_total = sum(min(s, 2) for s in sizes)
        assert len(sampler) == expected_total

    def test_group_sampler_determinism(self):
        """Test determinism across same seed + epoch."""
        items = []
        sizes = [3, 2]
        for g, size in enumerate(sizes):
            for i in range(size):
                items.append(
                    {
                        "frag_pyg": MockFrag(1 + i),
                        "group_id": th.tensor([g]),
                    }
                )

        ds = SyntheticDataset(items)

        g1 = th.Generator()
        g1.manual_seed(7)
        s1 = GroupSampler(ds, sample_k=2, generator=g1)
        s1.set_epoch(1)
        out1 = list(s1)

        g2 = th.Generator()
        g2.manual_seed(7)
        s2 = GroupSampler(ds, sample_k=2, generator=g2)
        s2.set_epoch(1)
        out2 = list(s2)

        assert out1 == out2


# ============================================================================
# Smoke Tests
# ============================================================================


class TestDataloaderSmoke:
    """Smoke tests for dataloader functionality."""

    def test_dynamic_batch_sampler_smoke(self):
        """Smoke test for dynamic batch sampler with dataloader."""
        sizes = [(1, 2), (3, 4), (2, 1), (8, 2), (5, 6), (2, 2), (4, 1), (6, 6)]
        ds = FakeDataset(sizes)
        sampler = SpecMolFragDynamicBatchSampler(data_source=ds, max_num=10, limited_by="frag_node")
        dl = DataLoader(ds, batch_sampler=sampler, collate_fn=lambda x: x)
        batches = list(dl)
        assert len(batches) > 0

    def test_dual_group_sampler_smoke(self):
        """Smoke test for dual group sampler."""
        samples = [
            {
                "frag_pyg": SimpleNamespace(num_nodes=3, num_edges=2),
                "adduct": "A",
                "mol_id": 1,
                "formula": "F1",
            },
            {
                "frag_pyg": SimpleNamespace(num_nodes=2, num_edges=1),
                "adduct": "A",
                "mol_id": 1,
                "formula": "F2",
            },
            {
                "frag_pyg": SimpleNamespace(num_nodes=4, num_edges=3),
                "adduct": "A",
                "mol_id": 2,
                "formula": "F1",
            },
            {
                "frag_pyg": SimpleNamespace(num_nodes=1, num_edges=1),
                "adduct": "B",
                "mol_id": 3,
                "formula": "F3",
            },
            {
                "frag_pyg": SimpleNamespace(num_nodes=5, num_edges=4),
                "adduct": "B",
                "mol_id": 3,
                "formula": "F3",
            },
            {
                "frag_pyg": SimpleNamespace(num_nodes=2, num_edges=1),
                "adduct": "C",
                "mol_id": 4,
                "formula": "F4",
            },
        ]

        ds = DummyDataset(samples)
        gen = th.Generator()
        gen.manual_seed(123)

        sampler = DualGroupDynamicBatchSampler(
            ds, max_num=6, limited_by="frag_node", skip_too_big=True, generator=gen
        )
        batches = list(iter(sampler))
        assert len(batches) > 0
        included = set(i for b in batches for i in b)
        assert included == set(range(len(ds)))
        for b in batches:
            total_nodes = sum(samples[i]["frag_pyg"].num_nodes for i in b)
            assert total_nodes <= 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
