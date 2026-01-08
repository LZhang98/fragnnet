import pytest
import torch as th

from fragnnet.dataset.data_sampler import (
    FormulaAdductGroupSampler,
    SpecMolFragDynamicBatchSampler,
    GroupSampler,
)


class MockFrag:
    def __init__(self, nodes: int, edges: int = 0):
        self.num_nodes = nodes
        self.num_edges = edges


class SyntheticDataset:
    def __init__(self, items):
        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def make_items():
    items = []
    # cluster A: adduct 'A', formula 'F1' -> indices 0,1,2
    for i in range(3):
        items.append(
            {
                'frag_pyg': MockFrag(nodes=1 + i, edges=0),
                'formula': 'F1',
                'adduct': 'A',
                'mol_id': f'm{i}',
                'group_id': th.tensor([0]),
            }
        )
    # cluster B: adduct 'B', formula 'F2' -> indices 3,4
    for i in range(2):
        items.append(
            {
                'frag_pyg': MockFrag(nodes=2 + i, edges=0),
                'formula': 'F2',
                'adduct': 'B',
                'mol_id': f'm{3 + i}',
                'group_id': th.tensor([1]),
            }
        )
    # single cluster C: index 5
    items.append(
        {
            'frag_pyg': MockFrag(nodes=4, edges=0),
            'formula': 'F3',
            'adduct': 'C',
            'mol_id': 'm5',
            'group_id': th.tensor([2]),
        }
    )
    return items


def test_grouping_basic():
    ds = SyntheticDataset(make_items())
    gen = th.Generator()
    gen.manual_seed(42)
    sampler = FormulaAdductGroupSampler(ds, sample_k=1, generator=gen)

    counts = {}
    for idx in sampler:
        item = ds[idx]
        key = f"{item['adduct']}::{item['formula']}"
        counts[key] = counts.get(key, 0) + 1

    # Each composite key yields at most sample_k (1)
    assert all(c <= 1 for c in counts.values())
    assert len(list(sampler)) == sum(counts.values())


def test_integration_with_dynamic_batch_sampler():
    ds = SyntheticDataset(make_items())
    gen = th.Generator()
    gen.manual_seed(123)
    sampler = FormulaAdductGroupSampler(ds, sample_k=2, generator=gen)

    batch_sampler = SpecMolFragDynamicBatchSampler(
        data_source=ds, max_num=6, limited_by='frag_node', sampler=sampler
    )
    batch_sampler.set_epoch(0)

    underlying = set(list(sampler))

    for batch in batch_sampler:
        total_nodes = sum(ds[idx]['frag_pyg'].num_nodes for idx in batch)
        assert total_nodes <= 6
        for idx in batch:
            assert idx in underlying


def test_set_epoch_and_determinism():
    ds = SyntheticDataset(make_items())

    g1 = th.Generator()
    g1.manual_seed(999)
    s1 = FormulaAdductGroupSampler(ds, sample_k=2, generator=g1)
    b1 = SpecMolFragDynamicBatchSampler(ds, max_num=9999, sampler=s1)
    b1.set_epoch(2)
    out1 = list(b1)

    g2 = th.Generator()
    g2.manual_seed(999)
    s2 = FormulaAdductGroupSampler(ds, sample_k=2, generator=g2)
    b2 = SpecMolFragDynamicBatchSampler(ds, max_num=9999, sampler=s2)
    b2.set_epoch(2)
    out2 = list(b2)

    assert out1 == out2

    # Changing epoch should change ordering
    b2.set_epoch(3)
    out3 = list(b2)
    assert out1 != out3


def test_missing_keys_raises():
    bad_items = make_items()
    # remove 'adduct' from first item
    bad_items[0] = {
        'frag_pyg': MockFrag(1),
        'formula': 'F1',
        'mol_id': 'm0',
        'group_id': th.tensor([0]),
    }
    ds = SyntheticDataset(bad_items)
    with pytest.raises(KeyError):
        FormulaAdductGroupSampler(ds)


def test_group_sampler_basic_and_epoch():
    # Build dataset with groups of sizes [4,2,1]
    items = []
    sizes = [4, 2, 1]
    for g, size in enumerate(sizes):
        for i in range(size):
            items.append(
                {
                    'frag_pyg': MockFrag(nodes=1),
                    'formula': f'F{g}',
                    'adduct': f'A{g}',
                    'mol_id': f'm{g}_{i}',
                    'group_id': th.tensor([g]),
                }
            )

    ds = SyntheticDataset(items)
    gen = th.Generator()
    gen.manual_seed(2024)
    sampler = GroupSampler(ds, sample_k=2, generator=gen)

    # Count samples per group
    counts = {}
    for idx in sampler:
        gid = ds[idx]['group_id'].item()
        counts[gid] = counts.get(gid, 0) + 1

    # Each group should yield at most sample_k
    assert all(counts.get(g, 0) <= 2 for g in range(len(sizes)))

    expected_total = sum(min(s, 2) for s in sizes)
    assert len(sampler) == expected_total

    # Determinism across same seed + epoch
    g1 = th.Generator(); g1.manual_seed(7)
    s1 = GroupSampler(ds, sample_k=2, generator=g1)
    s1.set_epoch(1)
    out1 = list(s1)

    g2 = th.Generator(); g2.manual_seed(7)
    s2 = GroupSampler(ds, sample_k=2, generator=g2)
    s2.set_epoch(1)
    out2 = list(s2)

    assert out1 == out2

