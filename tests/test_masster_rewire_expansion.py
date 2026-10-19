"""Tests for the MASSter rewire expansion stage (build step 2): parity with the G0 reference, dedup, cap."""

import importlib.util
import random
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch as th
from rdkit import Chem

from fragnnet.models.masster import rewire_expansion as rx
from fragnnet.models.masster.candidates import rewire_rules as rr

ROOT = Path(__file__).resolve().parents[1]
G0_SCRIPT = ROOT / "scripts/debug/smarts/debug_masster_rewire_reach_ceiling.py"

HAND = [
    "Cc1ccc(S(=O)(=O)Nc2ccccn2)cc1",  # sulfonamide
    "O=[N+]([O-])c1ccccc1SCC",  # nitro sulfide
    "O=S(=O)(c1ccccc1)c1ccc(Oc2ccccc2)cc1",  # diaryl ether sulfone
    "CCSCC(=O)Nc1ccc(OC)cc1",  # thioether amide
    "COP(=O)(OC)SCCC(=O)OC",  # thiophosphate
    "CNC(=O)Oc1ccccc1",  # carbamate
    "Cc1nc(SCC)sc1",  # thiazole thioether
    "c1ccccc1N=Nc1ccccc1",  # azo
]
RING_CASES = ["COP(=O)(OC)SCCC(=O)OC", "O=S1(=O)c2ccccc2-c2ccccc12"]
M_PLUS_H = "[M+H]+"


def _stub_channel_fn(atom_idx, atom_group, group_batch, num_groups, cut_counts, valence_adjustment):
    return th.zeros((num_groups, 2, 3)), th.ones((num_groups, 2, 3), dtype=th.bool)


def _load_g0():
    spec = importlib.util.spec_from_file_location("g0_reach_ceiling", G0_SCRIPT)
    sys.path.insert(0, str(G0_SCRIPT.parent))
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


try:
    G0 = _load_g0()
except Exception:  # noqa: BLE001  (a missing gate script or its imports only skips the parity tests)
    G0 = None
needs_g0 = pytest.mark.skipif(G0 is None, reason="G0 reference script not importable")


# ---- batch builder ------------------------------------------------------------------------------


def _lp(smiles, atoms):
    """Deterministic parent log-probability, a function of (molecule, atom set) only."""
    return -(zlib.crc32(f"{smiles}|{sorted(atoms)}".encode()) % 100000) / 10000.0 - 0.01


def _components(mol, cut):
    graph = nx.Graph()
    graph.add_nodes_from(range(mol.GetNumAtoms()))
    graph.add_edges_from(
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds() if frozenset(
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx())) not in cut
    )
    return [frozenset(c) for c in nx.connected_components(graph)]


def random_pool(mol, seed, draws=200):
    rng = random.Random(seed)
    bonds = [frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()]
    pool = {frozenset(range(mol.GetNumAtoms()))}
    for _ in range(draws):
        cut = set(rng.sample(bonds, min(rng.randint(1, 3), len(bonds))))
        pool |= set(_components(mol, cut))
    return sorted(pool, key=sorted)


def make_batch(smiles_list, pools=None, precursor_types=None, seed=0, **overrides):
    """kwargs for ``expand_rewire_candidates`` on a batch of molecules (CPU)."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    if pools is None:
        pools = [random_pool(m, seed + i) for i, m in enumerate(mols)]
    ptr, numbers, bond_cols = [0], [], []
    for m in mols:
        base = ptr[-1]
        numbers += [a.GetAtomicNum() for a in m.GetAtoms()]
        bond_cols += [(b.GetBeginAtomIdx() + base, b.GetEndAtomIdx() + base) for b in m.GetBonds()]
        ptr.append(base + m.GetNumAtoms())
    match_sets = [
        rr.filter_rewire_matches(
            rr.build_rewire_matches(m, max_matches_per_rule=1000),
            precursor_type=(precursor_types[i] if precursor_types else M_PLUS_H),
            num_known_rules=rr.MASSTER_NUM_REWIRE_RULES,
        )
        for i, m in enumerate(mols)
    ]
    frag_atoms, frag_of_atom, frag_batch, lps = [], [], [], []
    for s, pool in enumerate(pools):
        for atoms in pool:
            frag_atoms += [a + ptr[s] for a in sorted(atoms)]
            frag_of_atom += [len(frag_batch)] * len(atoms)
            frag_batch.append(s)
            lps.append(_lp(smiles_list[s], atoms))
    bond_index = th.tensor(bond_cols, dtype=th.long).T
    kw = dict(
        match_sets=match_sets,
        atom_ptr=th.tensor(ptr),
        atom_numbers=th.tensor(numbers),
        chemical_bond_index=bond_index,
        host_chemical_bond_index=bond_index.clone(),
        fragment_atom_indices=th.tensor(frag_atoms, dtype=th.long),
        atom_fragment_indices=th.tensor(frag_of_atom, dtype=th.long),
        fragment_batch_indices=th.tensor(frag_batch, dtype=th.long),
        parent_log_probabilities=th.tensor(lps, dtype=th.float32),
        parent_blocked=None,
        bond_logits=None,
        channel_fn=_stub_channel_fn,
        max_extra_fraction=100.0,
        max_extra=100000,
        continuation_cuts=0,
        pair_chunk=65536,
        num_rules=rr.MASSTER_NUM_REWIRE_RULES,
    )
    kw.update(overrides)
    kw["_mols"], kw["_pools"] = mols, pools  # test-side bookkeeping, popped by ``run``
    return kw


def run(kw):
    kw = {k: v for k, v in kw.items() if not k.startswith("_")}
    return rx.expand_rewire_candidates(**kw)


def extra_sets(ex, kw):
    """Per-spectrum list of local atom sets of the extras, in output order."""
    ptr = kw["atom_ptr"]
    out = defaultdict(list)
    members = defaultdict(set)
    for a, e in zip(ex.extra_atom_indices.tolist(), ex.extra_atom_extra_indices.tolist()):
        members[e].add(a)
    for e in range(ex.num_extra):
        s = int(ex.extra_batch_indices[e])
        out[s].append(frozenset(a - int(ptr[s]) for a in members[e]))
    return out


def ordinary_set(kw, row):
    atoms = kw["fragment_atom_indices"][kw["atom_fragment_indices"] == row].tolist()
    s = int(kw["fragment_batch_indices"][row])
    return frozenset(a - int(kw["atom_ptr"][s]) for a in atoms)


def merged_sets(ex, kw):
    out = defaultdict(set)
    for row in ex.merged_rows.tolist():
        out[int(kw["fragment_batch_indices"][row])].add(ordinary_set(kw, row))
    return out


def ref_products(mol, pool, depth1, corrected=True):
    """G0 reference over every raw match of the rules that list [M+H]+ (depth 0, optionally 1)."""
    bonds = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}
    out = set()
    for i, rule in enumerate(rr.REWIRE_RULES):
        if M_PLUS_H not in rule.precursors:
            continue
        rd = {"delete": list(rule.delete), "cut": [list(c) for c in rule.cut], "form": [list(f) for f in rule.form]}
        for match in rr.raw_rule_matches(mol, rr._PATTERNS[i]):
            cutset = {frozenset((match[a], match[b])) for a, b in rule.cut}
            for frag in pool:
                for piece, formed in G0.edited_products(frozenset(frag), match, rd, bonds):
                    if not formed:
                        continue
                    out.add(piece)
                    if depth1:
                        wrapped = bonds - cutset if corrected else bonds
                        for p in G0.one_cut_pieces(piece, formed, wrapped):
                            if any(f <= p for f in formed):
                                out.add(p)
    return out


# ---- helpers ------------------------------------------------------------------------------------


@pytest.mark.parametrize("amax", [1, 61, 62, 63, 123, 124, 125])
def test_words_round_trip(amax):
    rng = np.random.default_rng(amax)
    member = th.from_numpy(rng.random((9, amax)) < 0.5)
    member[0] = True
    words = rx.atom_set_words(member, -(-amax // rx.BITS_PER_WORD))
    assert (words >= 0).all()
    assert th.equal(rx.words_to_member(words, amax), member)


def test_lexsort_matches_sorted():
    rng = np.random.default_rng(0)
    keys = [th.from_numpy(rng.integers(0, 3, 200)) for _ in range(3)]
    order = rx.lexsort(keys).tolist()
    expected = sorted(range(200), key=lambda i: tuple(int(k[i]) for k in keys))
    assert order == expected


def _check_labels(num_atoms, edges, member=None):
    member = th.ones((1, num_atoms), dtype=th.bool) if member is None else member
    er = th.zeros(len(edges), dtype=th.long)
    a = th.tensor([e[0] for e in edges], dtype=th.long)
    b = th.tensor([e[1] for e in edges], dtype=th.long)
    labels = rx.label_components(member, er, a, b)[0]
    graph = nx.Graph()
    graph.add_nodes_from(th.nonzero(member[0]).flatten().tolist())
    graph.add_edges_from(edges)
    for comp in nx.connected_components(graph):
        assert {int(labels[i]) for i in comp} == {min(comp)}
    assert (labels[~member[0]] == -1).all()


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 30, 63])
def test_label_components_path(n):
    _check_labels(n, [(i, i + 1) for i in range(n - 1)])


def test_label_components_ring_two_components_isolated():
    _check_labels(6, [(i, (i + 1) % 6) for i in range(6)])
    _check_labels(7, [(0, 1), (1, 2), (3, 4), (4, 5)])  # atom 6 isolated, two components
    member = th.ones((1, 5), dtype=th.bool)
    member[0, 2] = False
    _check_labels(5, [(0, 1), (3, 4)], member)


def test_label_components_non_convergence_raises():
    n = 40
    edges = [(i, i + 1) for i in range(n - 1)]
    er = th.zeros(len(edges), dtype=th.long)
    a = th.tensor([e[0] for e in edges])
    b = th.tensor([e[1] for e in edges])
    with pytest.raises(ValueError, match="did not converge"):
        rx.label_components(th.ones((1, n), dtype=th.bool), er, a, b, max_sweeps=1)


def test_extra_limits_floor_and_ceiling():
    kept = np.array([3, 100, 100, 1000])
    assert rx.extra_limits(kept, 0.25, 256).tolist() == [0, 25, 25, 250]
    assert rx.extra_limits(kept, 0.29, 256).tolist() == [0, 29, 29, 256]  # 0.29 * 100 floors to 29, not 28
    assert rx.extra_limits(kept, 0.7, 256).tolist() == [2, 70, 70, 256]
    assert rx.extra_limits(kept, 1.0, 5).tolist() == [3, 5, 5, 5]


def test_reduce_pieces_merges_parents_and_stats():
    words = th.tensor([[5], [5], [9]])
    rec = {
        "batch": th.tensor([0, 0, 0]),
        "words": words,
        "ind": th.tensor([[1, 0], [0, 1], [1, 0]], dtype=th.uint8),
        "n_formed": th.tensor([1, 2, 1]),
        "n_cut": th.tensor([0, 1, 0]),
        "ndel": th.tensor([3, 2, 1]),
        "depth": th.tensor([1, 0, 0]),
        "adj": th.tensor([-2, -2, -2]),
        "parent_key": th.tensor([(-100 << 32) + 7, (-50 << 32) + 3, 0]),
    }
    out = rx._reduce_pieces(rec, 1, 2)
    assert out["words"].flatten().tolist() == [5, 9]
    assert out["ind"].tolist() == [[1, 1], [1, 0]]
    assert out["n_formed"].tolist() == [2, 1]
    assert out["depth"].tolist() == [0, 0]
    assert out["parent_key"][0].item() == (-50 << 32) + 3  # the better-scoring parent wins


# ---- parity with the G0 reference ---------------------------------------------------------------


@needs_g0
@pytest.mark.parametrize("seed", [0, 1])
def test_depth0_parity_hand_molecules(seed):
    kw = make_batch(HAND, seed=seed)
    ex = run(kw)
    extras, merged = extra_sets(ex, kw), merged_sets(ex, kw)
    total_extra = total_merged = 0
    for s, mol in enumerate(kw["_mols"]):
        expected = ref_products(mol, kw["_pools"][s], depth1=False)
        got = set(extras[s]) | merged[s]
        assert got == expected, (HAND[s], got ^ expected)
        assert len(extras[s]) == len(set(extras[s]))
        assert not set(extras[s]) & merged[s]
        assert ex.diagnostics["rewire_precap_extra_count"][s] + ex.diagnostics["rewire_merged_count"][s] == len(expected)
        assert ex.diagnostics["rewire_extra_count"][s] == len(extras[s])
        total_extra += len(extras[s])
        total_merged += len(merged[s])
    assert total_extra > 0
    assert ex.num_extra == total_extra


@needs_g0
def test_depth0_parity_on_a_random_pool_of_kept_rows_only():
    """The pool is a random subset of the cut components (a stand-in for the kept pool)."""
    pools = []
    for i, smi in enumerate(HAND):
        pool = random_pool(Chem.MolFromSmiles(smi), 100 + i)
        random.Random(i).shuffle(pool)
        pools.append(pool[: max(3, len(pool) // 3)])
    kw = make_batch(HAND, pools=pools)
    ex = run(kw)
    extras, merged = extra_sets(ex, kw), merged_sets(ex, kw)
    for s, mol in enumerate(kw["_mols"]):
        assert set(extras[s]) | merged[s] == ref_products(mol, pools[s], depth1=False)


@needs_g0
def test_depth1_parity_corrected_reference():
    smiles = HAND + [s for s in RING_CASES if s not in HAND]
    kw = make_batch(smiles, continuation_cuts=1000, bond_logits=None)
    kw["bond_logits"] = th.randn(kw["chemical_bond_index"].shape[1])
    ex = run(kw)
    extras, merged = extra_sets(ex, kw), merged_sets(ex, kw)
    for s, mol in enumerate(kw["_mols"]):
        expected = ref_products(mol, kw["_pools"][s], depth1=True, corrected=True)
        assert set(extras[s]) | merged[s] == expected, (smiles[s], len(expected))
    # the extras at depth 1 are a superset of depth 0
    kw0 = make_batch(smiles, continuation_cuts=0)
    assert run(kw0).num_extra < ex.num_extra


@needs_g0
def test_depth1_uncorrected_reference_differs_on_ring_type_rule():
    """C3: G0's one_cut_pieces restores rule-cut bonds, which under-splits ring-type cut rules."""
    smi = "COP(=O)(OC)SCCC(=O)OC"
    kw = make_batch([smi], continuation_cuts=1000)
    kw["bond_logits"] = th.randn(kw["chemical_bond_index"].shape[1])
    ex = run(kw)
    got = set(extra_sets(ex, kw)[0]) | merged_sets(ex, kw)[0]
    mol = kw["_mols"][0]
    assert got == ref_products(mol, kw["_pools"][0], depth1=True, corrected=True)
    assert got != ref_products(mol, kw["_pools"][0], depth1=True, corrected=False)


def test_continuation_selects_top_k_bonds_by_logit():
    smi = "CCSCC(=O)Nc1ccc(OC)cc1"
    kw = make_batch([smi], pools=[[frozenset(range(Chem.MolFromSmiles(smi).GetNumAtoms()))]])
    nb = kw["chemical_bond_index"].shape[1]
    base = run(dict(kw, continuation_cuts=0)).num_extra
    kw["bond_logits"] = th.randn(nb)
    sizes = [run(dict(kw, continuation_cuts=k)).num_extra for k in (0, 1, 3, 1000)]
    assert sizes[0] == base and sizes == sorted(sizes) and sizes[-1] > sizes[0]


# ---- mixed batches ------------------------------------------------------------------------------


def _signature(ex, kw, s):
    ptr = kw["atom_ptr"]
    members = defaultdict(set)
    for a, e in zip(ex.extra_atom_indices.tolist(), ex.extra_atom_extra_indices.tolist()):
        members[e].add(a - int(ptr[int(ex.extra_batch_indices[e])]))
    out = []
    for e in range(ex.num_extra):
        if int(ex.extra_batch_indices[e]) != s:
            continue
        out.append(
            (
                tuple(sorted(members[e])),
                tuple(sorted(ordinary_set(kw, int(ex.extra_parent[e])))),
                int(ex.extra_valence_adjustment[e]),
                int(ex.extra_cut_counts[e]),
                tuple(ex.extra_rule_indicators[e].tolist()),
                tuple(ex.extra_edit_features[e].tolist()),
            )
        )
    return out


def test_mixed_batch_order_and_singletons():
    smiles = ["CCSCC(=O)Nc1ccc(OC)cc1", "CCCCCC", "O=S(=O)(c1ccccc1)c1ccccc1", "O=[N+]([O-])c1ccccc1SCC"]
    pools = [random_pool(Chem.MolFromSmiles(s), 7 + i) for i, s in enumerate(smiles)]
    kw = make_batch(smiles, pools=pools)
    ex = run(kw)
    assert ex.diagnostics["rewire_match_count"][1] == 0  # no-match spectrum between two matching ones
    assert (ex.extra_batch_indices != 1).all()
    reference = {s: _signature(ex, kw, i) for i, s in enumerate(smiles)}
    assert all(reference[s] for s in smiles if s != "CCCCCC")
    for perm in ([3, 2, 1, 0], [1, 3, 0, 2]):
        sm = [smiles[i] for i in perm]
        k2 = make_batch(sm, pools=[pools[i] for i in perm])
        ex2 = run(k2)
        for j, s in enumerate(sm):
            assert _signature(ex2, k2, j) == reference[s]
    for i, s in enumerate(smiles):
        k3 = make_batch([s], pools=[pools[i]])
        assert _signature(run(k3), k3, 0) == reference[s]


# ---- hand values --------------------------------------------------------------------------------


def _full(smiles):
    return [frozenset(range(Chem.MolFromSmiles(smiles).GetNumAtoms()))]


def test_diphenyl_sulfone_so2_extrusion():
    # Rule 0 gives the 12-atom biphenyl-type product; rule 1 (s_to_o_aromatic_C) also gives four 7-atom
    # "aryl + O" products (two aryl ends x two sulfonyl O), which the G0 parity tests cover. Pin rule 0's extra.
    smi = "O=S(=O)(c1ccccc1)c1ccccc1"
    kw = make_batch([smi], pools=[_full(smi)])
    ex = run(kw)
    assert ex.num_extra == 5
    (e,) = (i for i in range(ex.num_extra) if ex.extra_rule_indicators[i, 0] == 1)
    assert len(extra_sets(ex, kw)[0][e]) == 12
    assert int(ex.extra_cut_counts[e]) == 2
    assert int(ex.extra_valence_adjustment[e]) == -2
    assert ex.extra_edit_features[e].tolist() == [1.0, 0.0, 3.0, 0.0]
    assert ex.extra_rule_indicators[e].nonzero().flatten().tolist() == [0]
    assert int(ex.extra_parent[e]) == 0
    assert (ex.extra_boundary_extra_indices == e).sum() == 2
    assert ex.extra_ion_mzs.shape == (5, 2, 3) and ex.extra_valid_channels.all()
    assert ex.merged_rows.numel() == 0
    rule1 = [i for i in range(ex.num_extra) if ex.extra_rule_indicators[i, 1] == 1]
    assert len(rule1) == 4 and all(len(extra_sets(ex, kw)[0][i]) == 7 for i in rule1)


def test_dibenzothiophene_dioxide_rule1_merges_into_precursor():
    smi = "O=S1(=O)c2ccccc2-c2ccccc12"
    kw = make_batch([smi], pools=[_full(smi)])
    ex = run(kw)
    assert ex.merged_rows.tolist() == [0]
    assert ex.merged_rule_indicators[0, 1] == 1
    assert ex.merged_edit_features[0].tolist() == [1.0, 2.0, 0.0, 0.0]
    # the product set equals the precursor: no extra of rule 1; rule 0 gives its own extras only
    assert all(len(a) < 15 for a in extra_sets(ex, kw)[0])
    assert (ex.extra_rule_indicators[:, 1] == 0).all()


def test_nitro_nitrite_loss_on_nitrobenzene():
    smi = "O=[N+]([O-])c1ccccc1"
    kw = make_batch([smi], pools=[_full(smi)])
    ex = run(kw)
    assert ex.num_extra == 1
    (atoms,) = extra_sets(ex, kw)[0]
    assert len(atoms) == 7  # six carbons and the nitrite O
    assert int(ex.extra_edit_features[0, 2]) == 2 and int(ex.extra_valence_adjustment[0]) == -2


def test_symmetric_sulfonyl_matches_make_one_extra():
    smi = "CS(=O)(=O)c1ccccc1"
    ex = run(make_batch([smi], pools=[_full(smi)]))
    assert (ex.extra_rule_indicators[:, 0] == 1).sum() == 1  # rule 0: one product, not one per O/end swap


def test_two_rules_one_atom_set_or_the_indicators():
    smi = "c1ccccc1Sc1ccccc1"  # rule 5 (acyclic thioether) and rule 7 (diaryl sulfide) give one product
    ex = run(make_batch([smi], pools=[_full(smi)]))
    assert ex.num_extra == 1
    assert ex.extra_rule_indicators[0].nonzero().flatten().tolist() == [5, 7]


def test_extra_equal_to_ordinary_row_merges():
    smi = "C1CCSC1"
    mol = Chem.MolFromSmiles(smi)
    carbons = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C")
    kw = make_batch([smi], pools=[[frozenset(range(5)), carbons]])
    ex = run(kw)
    assert ex.num_extra == 0
    assert ex.merged_rows.tolist() == [1] or ex.merged_rows.tolist() == [0]
    row = int(ex.merged_rows[0])
    assert ordinary_set(kw, row) == carbons
    assert ex.merged_rule_indicators[0].nonzero().flatten().tolist() == [6]
    assert ex.merged_edit_features[0].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert ex.diagnostics["rewire_merged_count"][0] == 1


def test_duplicate_ordinary_rows_are_both_merged_not_an_error():
    smi = "C1CCSC1"
    mol = Chem.MolFromSmiles(smi)
    carbons = frozenset(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C")
    kw = make_batch([smi], pools=[[frozenset(range(5)), carbons, carbons]])
    ex = run(kw)
    assert sorted(ex.merged_rows.tolist()) == [1, 2] and ex.num_extra == 0


# ---- cap ----------------------------------------------------------------------------------------


def _thioether_pool():
    smi = "CCSCC(=O)Nc1ccc(OC)cc1"
    mol = Chem.MolFromSmiles(smi)
    n = mol.GetNumAtoms()
    all_atoms = frozenset(range(n))
    methoxy_c = max(i for i in range(n) if mol.GetAtomWithIdx(i).GetSymbol() == "C"
                    and mol.GetAtomWithIdx(i).GetDegree() == 1 and i > 8)
    ethyl_c = 0
    return smi, [all_atoms, all_atoms - {methoxy_c}, all_atoms - {ethyl_c}]


def test_cap_orders_by_parent_log_probability():
    smi, pool = _thioether_pool()
    kw = make_batch([smi], pools=[pool], max_extra_fraction=0.7)  # K = 3 -> limit 2
    kw["parent_log_probabilities"] = th.tensor([-1.0, -3.0, -2.0])
    ex = run(kw)
    assert ex.num_extra == 2
    assert sorted(ex.extra_parent.tolist()) == [0, 2]
    d = ex.diagnostics
    assert d["rewire_precap_extra_count"][0] == 3 and d["rewire_dropped_cap_count"][0] == 1
    assert d["rewire_extra_count"][0] == 2 and d["rewire_cap_tie_count"][0] == 0
    kw["parent_log_probabilities"] = th.tensor([-3.0, -1.0, -2.0])
    assert sorted(run(kw).extra_parent.tolist()) == [1, 2]


def test_cap_ties_broken_by_atom_set_words():
    smi, pool = _thioether_pool()
    kw = make_batch([smi], pools=[pool], max_extra_fraction=0.7)
    kw["parent_log_probabilities"] = th.full((3,), -1.0)
    ex = run(kw)
    assert ex.diagnostics["rewire_cap_tie_count"][0] == 1
    full = run(dict(kw, max_extra_fraction=100.0))
    ordered = sorted(extra_sets(full, kw)[0], key=lambda a: sum(1 << i for i in a))
    assert set(extra_sets(ex, kw)[0]) == set(ordered[:2])


def test_cap_absolute_ceiling_and_floor():
    smi, pool = _thioether_pool()
    assert run(make_batch([smi], pools=[pool], max_extra_fraction=100.0, max_extra=1)).num_extra == 1
    # K = 3 and fraction 0.25 floors to zero extras
    assert run(make_batch([smi], pools=[pool], max_extra_fraction=0.25)).num_extra == 0
    assert run(make_batch([smi], pools=[pool], max_extra_fraction=0.34)).num_extra == 1


def test_invalid_channels_dropped_before_cap_and_merged_twin_gets_channels():
    smi, pool = _thioether_pool()

    def odd_only(atom_idx, atom_group, group_batch, num_groups, cut_counts, adj):
        sizes = th.bincount(atom_group, minlength=num_groups)
        valid = (sizes % 2 == 1)[:, None, None].expand(num_groups, 2, 3).clone()
        return th.zeros((num_groups, 2, 3)), valid

    kw = make_batch([smi], pools=[pool], channel_fn=odd_only)
    ex = run(kw)
    sizes = [len(a) for a in extra_sets(ex, kw)[0]]
    assert all(z % 2 == 1 for z in sizes)
    d = ex.diagnostics
    assert d["rewire_dropped_no_valid_count"][0] + ex.num_extra == d["rewire_precap_extra_count"][0]
    assert ex.extra_valid_channels.flatten(1).any(1).all()


def test_blocked_parents_make_no_extras():
    smi, pool = _thioether_pool()
    kw = make_batch([smi], pools=[pool])
    kw["parent_blocked"] = th.tensor([True, False, False])
    ex = run(kw)
    assert 0 not in ex.extra_parent.tolist() and ex.num_extra == 2
    kw["parent_blocked"] = th.ones(3, dtype=th.bool)
    assert run(kw).num_extra == 0
    assert (run(kw).diagnostics["rewire_contained_pair_count"] == 0).all()


# ---- hard errors --------------------------------------------------------------------------------


def _single(smi="O=S(=O)(c1ccccc1)c1ccccc1"):
    return make_batch([smi], pools=[_full(smi)])


def test_error_match_set_count_mismatch():
    kw = _single()
    kw["match_sets"] = kw["match_sets"] * 2
    with pytest.raises(ValueError, match="match sets"):
        run(kw)


def _with_match(kw, **changes):
    from dataclasses import replace

    ms = kw["match_sets"][0]
    kw["match_sets"] = [rr.RewireMatchSet((replace(ms.matches[0], **changes),) + ms.matches[1:], ms.num_truncated)]
    return kw


def test_error_motif_atom_out_of_range():
    kw = _with_match(_single(), motif_atoms=(0, 1, 99))
    with pytest.raises(ValueError, match="outside the molecule"):
        run(kw)


def test_error_motif_element_differs():
    kw = _single()
    n = len(kw["match_sets"][0].matches[0].motif_numbers)
    kw = _with_match(kw, motif_numbers=tuple([99] * n))
    with pytest.raises(ValueError, match="elements differ"):
        run(kw)


def test_error_unresolved_cut_pair():
    kw = make_batch(["O=S1(=O)c2ccccc2-c2ccccc12"], pools=[_full("O=S1(=O)c2ccccc2-c2ccccc12")])
    ms = kw["match_sets"][0]
    idx = next(i for i, m in enumerate(ms.matches) if m.cut_pairs)
    bad = ms.matches[idx]
    from dataclasses import replace

    nonbond = next(
        (a, b) for a in range(10) for b in range(a + 1, 10)
        if not (kw["chemical_bond_index"] == th.tensor([[a], [b]])).all(0).any()
        and not (kw["chemical_bond_index"] == th.tensor([[b], [a]])).all(0).any()
    )
    kw["match_sets"] = [rr.RewireMatchSet((replace(bad, cut_pairs=(nonbond,)),), 0)]
    with pytest.raises(ValueError, match="is not a bond"):
        run(kw)


def test_error_bond_index_not_sorted_by_spectrum():
    smiles = ["O=S(=O)(c1ccccc1)c1ccccc1", "CCSCC"]
    kw = make_batch(smiles)
    kw["chemical_bond_index"] = kw["chemical_bond_index"].flip(1)
    kw["host_chemical_bond_index"] = kw["chemical_bond_index"].clone()
    with pytest.raises(ValueError, match="sorted by spectrum"):
        run(kw)


def test_error_membership_outside_spectrum():
    smiles = ["O=S(=O)(c1ccccc1)c1ccccc1", "CCSCC"]
    kw = make_batch(smiles)
    atoms = kw["fragment_atom_indices"].clone()
    atoms[0] = int(kw["atom_ptr"][-1]) - 1  # a spectrum-1 atom inside a spectrum-0 fragment
    kw["fragment_atom_indices"] = atoms
    with pytest.raises(ValueError, match="outside its spectrum"):
        run(kw)


def test_error_continuation_without_bond_logits():
    with pytest.raises(ValueError, match="bond logits"):
        run(dict(_single(), continuation_cuts=1, bond_logits=None))


def test_error_non_finite_parent_probability():
    kw = _single()
    kw["parent_log_probabilities"] = th.full_like(kw["parent_log_probabilities"], -th.inf)
    with pytest.raises(ValueError, match="not finite"):
        run(kw)


# ---- chunking, word boundaries, zero extras -----------------------------------------------------


@needs_g0
def test_chunk_size_does_not_change_the_output():
    kw = make_batch(HAND, seed=3)
    outputs = []
    for chunk in (1, 7, 65536):
        ex = run(dict(kw, pair_chunk=chunk, continuation_cuts=2, bond_logits=th.linspace(-1, 1, kw["chemical_bond_index"].shape[1])))
        outputs.append(ex)
    base = outputs[0]
    for ex in outputs[1:]:
        assert ex.num_extra == base.num_extra
        for name in ("extra_atom_indices", "extra_batch_indices", "extra_parent", "extra_cut_counts",
                     "extra_valence_adjustment", "merged_rows", "extra_boundary_bond_indices"):
            assert th.equal(getattr(ex, name), getattr(base, name)), name
        for name in ("extra_rule_indicators", "extra_edit_features", "merged_rule_indicators", "merged_edit_features"):
            assert th.equal(getattr(ex, name), getattr(base, name)), name
        for name, value in base.diagnostics.items():
            if name != "rewire_cap_tie_count":
                assert th.equal(ex.diagnostics[name], value), name


@pytest.mark.parametrize("num_atoms", [61, 62, 63, 124, 125])
def test_word_boundary_molecules(num_atoms):
    k = num_atoms - 7
    smi = "C" * k + "Sc1ccccc1"
    mol = Chem.MolFromSmiles(smi)
    assert mol.GetNumAtoms() == num_atoms
    kw = make_batch([smi], pools=[_full(smi)])
    ex = run(kw)
    assert ex.num_extra == 1
    (atoms,) = extra_sets(ex, kw)[0]
    assert atoms == frozenset(range(num_atoms)) - {k}


def test_extra_memory_is_bounded_by_the_element_budget():
    smi = "C" * 113 + "Sc1ccccc1"  # 120 atoms, one thioether match
    mol = Chem.MolFromSmiles(smi)
    pools = [random_pool(mol, 0, draws=30)]
    kw = make_batch([smi], pools=pools, pair_chunk=4000)
    run(kw)
    unit = max(mol.GetNumAtoms(), mol.GetNumBonds())
    assert rx._LAST_STATS["max_chunk_rows"] <= max(1, 4000 // unit)
    assert rx._LAST_STATS["num_host_chunks"] > 1


def test_channel_scoring_is_chunked_by_the_element_budget():
    smi = "C" * 113 + "Sc1ccccc1"
    mol = Chem.MolFromSmiles(smi)
    kw = make_batch([smi], pools=[random_pool(mol, 0, draws=30)], pair_chunk=4000)
    sizes = []

    def recording(atom_idx, atom_group, group_batch, num_groups, cut_counts, adj):
        sizes.append(num_groups)
        return _stub_channel_fn(atom_idx, atom_group, group_batch, num_groups, cut_counts, adj)

    kw["channel_fn"] = recording
    ex = run(kw)
    unit = max(mol.GetNumAtoms(), mol.GetNumBonds())
    assert sizes and max(sizes) <= max(1, 4000 // unit)
    assert ex.diagnostics["rewire_precap_extra_count"].sum() >= 1


def test_host_bond_index_must_equal_the_device_bond_index():
    kw = make_batch(["CCSCC"], pools=[_full("CCSCC")])
    kw["host_chemical_bond_index"] = kw["chemical_bond_index"].flip(1).clone()
    with pytest.raises(ValueError, match="host_chemical_bond_index"):
        run(kw)


def test_zero_extra_batch_has_empty_outputs_and_finite_diagnostics():
    kw = make_batch(["CCCCCC", "CCO"])
    ex = run(kw)
    assert ex.num_extra == 0 and ex.merged_rows.numel() == 0
    assert ex.extra_ion_mzs.shape[0] == 0 and ex.extra_rule_indicators.shape == (0, rr.MASSTER_NUM_REWIRE_RULES)
    assert set(ex.diagnostics) == set(rx.REWIRE_DIAGNOSTIC_NAMES)
    assert all(th.isfinite(v).all() and v.shape == (2,) for v in ex.diagnostics.values())
    # a matching spectrum whose extras all vanish keeps the same shape contract
    smi = "O=S(=O)(c1ccccc1)c1ccccc1"
    kw = make_batch([smi], pools=[_full(smi)], max_extra_fraction=0.5)
    ex = run(kw)
    assert ex.num_extra == 0 and ex.diagnostics["rewire_dropped_cap_count"][0] == 5
    assert ex.extra_ion_mzs.shape == (0, 2, 3)
