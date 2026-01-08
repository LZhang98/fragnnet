import importlib.util
import pathlib
import torch as th

# Load data_sampler directly from source file to avoid package-level imports
data_sampler_path = pathlib.Path(__file__).resolve().parents[1] / "src" / "fragnnet" / "dataset" / "data_sampler.py"
spec = importlib.util.spec_from_file_location("data_sampler", str(data_sampler_path))
data_sampler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_sampler)
DualGroupDynamicBatchSampler = data_sampler.DualGroupDynamicBatchSampler


class DummyFrag:
    def __init__(self, nodes, edges):
        self.num_nodes = nodes
        self.num_edges = edges


class DummyDataset:
    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


def test_dual_group_dynamic_batch_sampler_nodes_limit():
    # Build small controlled dataset of 6 samples
    samples = [
        {"frag_pyg": DummyFrag(3, 2), "adduct": "A", "mol_id": 1, "formula": "X"},
        {"frag_pyg": DummyFrag(2, 1), "adduct": "A", "mol_id": 1, "formula": "Y"},
        {"frag_pyg": DummyFrag(4, 3), "adduct": "A", "mol_id": 2, "formula": "X"},
        {"frag_pyg": DummyFrag(1, 1), "adduct": "B", "mol_id": 3, "formula": "Z"},
        {"frag_pyg": DummyFrag(5, 4), "adduct": "B", "mol_id": 3, "formula": "Z"},
        {"frag_pyg": DummyFrag(2, 1), "adduct": "C", "mol_id": 4, "formula": "W"},
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

    # Expect at least one batch and that all indices are included
    assert len(batches) > 0
    included = set(i for b in batches for i in b)
    assert included == set(range(len(ds)))

    # Each batch must respect node budget
    for b in batches:
        total_nodes = sum(samples[i]["frag_pyg"].num_nodes for i in b)
        assert total_nodes <= 6


def test_dual_group_dynamic_batch_sampler_edges_limit():
    """Test that sampler respects edge budget."""
    samples = [
        {"frag_pyg": DummyFrag(3, 5), "adduct": "A", "mol_id": 1, "formula": "X"},
        {"frag_pyg": DummyFrag(2, 3), "adduct": "A", "mol_id": 1, "formula": "Y"},
        {"frag_pyg": DummyFrag(4, 7), "adduct": "A", "mol_id": 2, "formula": "X"},
        {"frag_pyg": DummyFrag(1, 2), "adduct": "B", "mol_id": 3, "formula": "Z"},
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


def test_dual_group_overlap_removal():
    """Test that overlapping samples are removed from group2."""
    samples = [
        # Sample 0: in both groups (A::1 and A::X)
        {"frag_pyg": DummyFrag(3, 2), "adduct": "A", "mol_id": 1, "formula": "X"},
        # Sample 1: in group1 only (A::2 and A::X) - overlaps with 0 via formula
        {"frag_pyg": DummyFrag(2, 1), "adduct": "A", "mol_id": 2, "formula": "X"},
        # Sample 2: also has mol_id (B::Y)
        {"frag_pyg": DummyFrag(4, 3), "adduct": "B", "mol_id": 5, "formula": "Y"},
        # Sample 3: in both groups (C::3 and C::Z)
        {"frag_pyg": DummyFrag(1, 1), "adduct": "C", "mol_id": 3, "formula": "Z"},
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

    # Check internal state: samples 0 and 3 should NOT be in group2
    # because they're in group1 (more specific)
    # With all samples having mol_id, overlap removal should empty group2
    assert len(sampler._group2) == 0


def test_dual_group_sampler_diversity():
    """Test that batches include samples from both group types when possible."""
    samples = [
        # Group1: adduct+mol_id
        {"frag_pyg": DummyFrag(2, 1), "adduct": "A", "mol_id": 1, "formula": "X"},
        {"frag_pyg": DummyFrag(2, 1), "adduct": "A", "mol_id": 2, "formula": "Y"},
        # Additional group1 members
        {"frag_pyg": DummyFrag(2, 1), "adduct": "B", "mol_id": 3, "formula": "Z"},
        {"frag_pyg": DummyFrag(2, 1), "adduct": "C", "mol_id": 4, "formula": "W"},
    ]

    ds = DummyDataset(samples)
    gen = th.Generator()
    gen.manual_seed(42)

    sampler = DualGroupDynamicBatchSampler(
        ds,
        max_num=8,
        limited_by="frag_node",
        sample_k1=1,
        sample_k2=1,
        generator=gen,
    )

    batches = list(iter(sampler))
    
    # All samples should be included
    included = set(i for b in batches for i in b)
    assert len(included) == 4


def test_dual_group_sampler_skip_too_big():
    """Test that skip_too_big parameter works correctly."""
    samples = [
        {"frag_pyg": DummyFrag(2, 1), "adduct": "A", "mol_id": 1, "formula": "X"},
        {"frag_pyg": DummyFrag(20, 15), "adduct": "A", "mol_id": 2, "formula": "Y"},  # Too big
        {"frag_pyg": DummyFrag(3, 2), "adduct": "B", "mol_id": 3, "formula": "Z"},
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
    
    # Sample 1 should be skipped
    included = set(i for b in batches for i in b)
    assert 1 not in included
    assert 0 in included
    assert 2 in included


def test_dual_group_sampler_deterministic_with_generator():
    """Test that providing a generator gives deterministic results."""
    samples = [
        {"frag_pyg": DummyFrag(2, 1), "adduct": "A", "mol_id": i, "formula": f"F{i}"}
        for i in range(10)
    ]

    ds = DummyDataset(samples)

    gen1 = th.Generator()
    gen1.manual_seed(123)
    sampler1 = DualGroupDynamicBatchSampler(ds, max_num=8, limited_by="frag_node", generator=gen1)
    batches1 = list(iter(sampler1))

    gen2 = th.Generator()
    gen2.manual_seed(123)
    sampler2 = DualGroupDynamicBatchSampler(ds, max_num=8, limited_by="frag_node", generator=gen2)
    batches2 = list(iter(sampler2))

    # Should produce identical batches
    assert len(batches1) == len(batches2)
    for b1, b2 in zip(batches1, batches2):
        assert b1 == b2


def test_dual_group_sampler_set_epoch():
    """Test that set_epoch produces different batches."""
    samples = [
        {"frag_pyg": DummyFrag(2, 1), "adduct": "A", "mol_id": i, "formula": f"F{i}"}
        for i in range(10)
    ]

    ds = DummyDataset(samples)
    gen = th.Generator()
    gen.manual_seed(42)

    sampler = DualGroupDynamicBatchSampler(ds, max_num=8, limited_by="frag_node", generator=gen)
    batches_epoch0 = list(iter(sampler))

    sampler.set_epoch(1)
    batches_epoch1 = list(iter(sampler))

    # Different epochs should produce different batch orderings
    # (unless we get extremely unlucky)
    assert batches_epoch0 != batches_epoch1


def test_dual_group_sampler_sample_k1():
    """Test that sample_k1 controls how many samples are taken from group1 per batch."""
    # Create dataset with multiple samples per group1 key
    samples = [
        # Group A::1 has 3 samples
        {"frag_pyg": DummyFrag(1, 1), "adduct": "A", "mol_id": 1, "formula": "X"},
        {"frag_pyg": DummyFrag(1, 1), "adduct": "A", "mol_id": 1, "formula": "X"},
        {"frag_pyg": DummyFrag(1, 1), "adduct": "A", "mol_id": 1, "formula": "X"},
        # Group B::2 has 2 samples
        {"frag_pyg": DummyFrag(1, 1), "adduct": "B", "mol_id": 2, "formula": "Y"},
        {"frag_pyg": DummyFrag(1, 1), "adduct": "B", "mol_id": 2, "formula": "Y"},
    ]

    ds = DummyDataset(samples)
    gen = th.Generator()
    gen.manual_seed(42)

    # With sample_k1=2, should try to take up to 2 samples from each group1 key per batch
    sampler = DualGroupDynamicBatchSampler(
        ds,
        max_num=10,
        limited_by="frag_node",
        sample_k1=2,
        sample_k2=0,  # Don't sample from group2
        generator=gen,
    )

    batches = list(iter(sampler))
    
    # First batch should have samples from both group1 keys
    # Should have up to 2 from A::1 and up to 2 from B::2
    first_batch = batches[0] if batches else []
    assert len(first_batch) > 0
    
    # All samples should eventually be included
    included = set(i for b in batches for i in b)
    assert len(included) == 5


def test_dual_group_sampler_sample_k2():
    """Test that sample_k2 controls how many samples are taken from group2 per batch."""
    # With mol_id required, group2 will be empty after overlap removal.
    samples = [
        {"frag_pyg": DummyFrag(1, 1), "adduct": "A", "mol_id": 1, "formula": "X"},
        {"frag_pyg": DummyFrag(1, 1), "adduct": "A", "mol_id": 2, "formula": "X"},
        {"frag_pyg": DummyFrag(1, 1), "adduct": "A", "mol_id": 3, "formula": "X"},
        {"frag_pyg": DummyFrag(1, 1), "adduct": "B", "mol_id": 4, "formula": "Y"},
        {"frag_pyg": DummyFrag(1, 1), "adduct": "B", "mol_id": 5, "formula": "Y"},
    ]

    ds = DummyDataset(samples)
    gen = th.Generator()
    gen.manual_seed(42)

    # With sample_k2=2, should try to take up to 2 samples from each group2 key per batch
    sampler = DualGroupDynamicBatchSampler(
        ds,
        max_num=10,
        limited_by="frag_node",
        sample_k1=0,  # Don't sample from group1
        sample_k2=2,
        generator=gen,
    )

    batches = list(iter(sampler))
    
    # Should create batches (group2 contributes none)
    assert len(batches) > 0
    
    # All samples should eventually be included via group1
    included = set(i for b in batches for i in b)
    assert len(included) == 5


def test_dual_group_sampler_k1_and_k2_together():
    """Test that sample_k1 and sample_k2 work together correctly."""
    samples = [
        # Group1: A::1 (2 samples, will be removed from group2)
        {"frag_pyg": DummyFrag(1, 1), "adduct": "A", "mol_id": 1, "formula": "X"},
        {"frag_pyg": DummyFrag(1, 1), "adduct": "A", "mol_id": 1, "formula": "X"},
        # Group1: B::2 (2 samples, will be removed from group2)
        {"frag_pyg": DummyFrag(1, 1), "adduct": "B", "mol_id": 2, "formula": "Y"},
        {"frag_pyg": DummyFrag(1, 1), "adduct": "B", "mol_id": 2, "formula": "Y"},
        # Additional group1 samples: C::Z
        {"frag_pyg": DummyFrag(1, 1), "adduct": "C", "mol_id": 3, "formula": "Z"},
        {"frag_pyg": DummyFrag(1, 1), "adduct": "C", "mol_id": 4, "formula": "Z"},
    ]

    ds = DummyDataset(samples)
    gen = th.Generator()
    gen.manual_seed(42)

    sampler = DualGroupDynamicBatchSampler(
        ds,
        max_num=10,
        limited_by="frag_node",
        sample_k1=1,  # Take 1 from each group1 key
        sample_k2=1,  # Take 1 from each group2 key
        generator=gen,
    )

    batches = list(iter(sampler))
    
    # First batch should have samples from group1 only
    first_batch = batches[0] if batches else []
    assert len(first_batch) >= 2
    
    # All samples should eventually be included
    included = set(i for b in batches for i in b)
    assert len(included) == 6


def test_dual_group_sampler_k_values_exceed_group_size():
    """Test behavior when k values exceed available samples in groups."""
    samples = [
        # Only 1 sample in group A::1
        {"frag_pyg": DummyFrag(2, 1), "adduct": "A", "mol_id": 1, "formula": "X"},
        # Only 1 sample in group B
        {"frag_pyg": DummyFrag(2, 1), "adduct": "B", "mol_id": 2, "formula": "Y"},
    ]

    ds = DummyDataset(samples)
    gen = th.Generator()
    gen.manual_seed(42)

    # Request more samples than available in each group
    sampler = DualGroupDynamicBatchSampler(
        ds,
        max_num=10,
        limited_by="frag_node",
        sample_k1=5,  # Request 5 but only 1 available
        sample_k2=5,  # Request 5 but only 1 available
        generator=gen,
    )

    batches = list(iter(sampler))
    
    # Should still work, just take what's available
    included = set(i for b in batches for i in b)
    assert len(included) == 2  # Both samples should be included
