"""Tests for the MASSter rewire rule registry and matcher (build step 1)."""

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from rdkit import Chem

from fragnnet.models.masster.candidates import rewire_rules as rr

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "output/rewire_rule_gate_2026_10_15/rule_catalog.json"
GATE = ROOT / "scripts/debug/smarts/debug_masster_rewire_rule_gate.py"

EXPECTED_NAMES = (
    "sulfonyl_so2_extrusion",
    "s_to_o_aromatic_C",
    "thiono_thiolo_aryl",
    "PS_O_transfer_6ring",
    "nitro_nitrite_NO_loss",
    "acyclic_thioether_s_extrusion",
    "cyclic_thioether_s_extrusion",
    "diaryl_sulfide_s_extrusion",
    "hetaryl_thio_alkyl_s_extrusion",
    "thiadiazole_cs2_extrusion",
    "diaryl_ether_o_excision",
    "carbamate_co2_extrusion",
    "co2_extrusion_aryl_Oside",
    "azo_N2_extrusion",
    "dinitrophenol_24_2NO",
    "acid_acyl_transfer_5",
    "haloalkyl_sulfonyl_X_to_S",
)


def _mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return mol


def _matches_of(smiles, rule_name):
    idx = rr.REWIRE_RULE_VOCABULARY.index(rule_name)
    ms = rr.build_rewire_matches(_mol(smiles), max_matches_per_rule=1000)
    return [m for m in ms.matches if m.rule_index == idx]


def _rule(name):
    return rr.REWIRE_RULES[rr.REWIRE_RULE_VOCABULARY.index(name)]


# ---- registry ----


def test_registry():
    assert len(rr.REWIRE_RULES) == 17
    assert rr.MASSTER_NUM_REWIRE_RULES == 17
    assert rr.REWIRE_RULE_VOCABULARY == EXPECTED_NAMES
    for i, rule in enumerate(rr.REWIRE_RULES):
        assert len(rule.precursors) == (1 if i in (13, 14, 15, 16) else 3)
    assert [r.null_smarts is not None for r in rr.REWIRE_RULES] == [i in (13, 14, 15) for i in range(17)]


@pytest.mark.skipif(not CATALOG.exists(), reason="gitignored rule catalog not present")
def test_rules_equal_catalog():
    admitted = [r for r in json.loads(CATALOG.read_text())["rules"] if r["status"] == "admitted"]
    assert len(admitted) == len(rr.REWIRE_RULES)
    for entry, rule in zip(admitted, rr.REWIRE_RULES):
        assert rule.name == entry["name"]
        assert rule.smarts == entry["smarts"]
        assert rule.delete == tuple(entry["delete"])
        assert rule.cut == tuple(tuple(p) for p in entry["cut"])
        assert rule.form == tuple(tuple(p) for p in entry["form"])
        assert rule.precursors == tuple(entry["precursors"])
        assert rule.null_smarts == entry.get("null_smarts")


def test_hashes_and_append_only():
    assert len(set(rr.REWIRE_RULE_SPEC_HASHES)) == 17
    swapped = list(rr.REWIRE_RULES)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError):
        rr.validate_rewire_rules(tuple(swapped), rr.REWIRE_RULE_VOCABULARY)
    with pytest.raises(ValueError):
        rr.validate_rewire_rules(rr.REWIRE_RULES[:-1], rr.REWIRE_RULE_VOCABULARY)
    rr.validate_rewire_rules(rr.REWIRE_RULES, rr.REWIRE_RULE_VOCABULARY)


def test_hash_sensitivity():
    base = _rule("s_to_o_aromatic_C")
    h = rr.zlib.crc32(rr.rewire_rule_spec(base).encode())
    for changed in (
        replace(base, smarts=base.smarts.replace("[!#1:5]", "[#6:5]")),
        replace(base, delete=(4,)),
        replace(base, cut=((1, 2),)),
        replace(base, form=((1, 3), (1, 4))),
        replace(base, precursors=("[M+H]+",)),
    ):
        assert rr.zlib.crc32(rr.rewire_rule_spec(changed).encode()) != h
    assert rr.zlib.crc32(rr.rewire_rule_spec(replace(base, null_smarts="[#6:1]~[#6:2]")).encode()) == h
    # pair orientation and ordering do not matter
    flipped = replace(base, cut=((2, 3), (2, 1)))
    assert rr.rewire_rule_spec(flipped) == rr.rewire_rule_spec(base)


@pytest.mark.parametrize(
    "change",
    [
        {"cut": ((1, 3),)},  # not a pattern bond
        {"delete": (2, 3, 4), "form": ((1, 2),)},  # form endpoint deleted
        {"delete": (9,)},  # unknown map number
        {"form": ()},  # empty form
        {"precursors": ("[M+K]+",)},  # unknown precursor
        {"cut": ((1, 1),)},  # equal endpoints
        {"smarts": "[#6:1(("},  # unparsable
        {"null_smarts": "[#6:9]~[#6:1]"},  # null map not subset
    ],
)
def test_import_validation(change):
    base = _rule("sulfonyl_so2_extrusion")
    bad = replace(base, **change)
    with pytest.raises(ValueError):
        rr.validate_rewire_rules((bad,), (base.name,))


# ---- golden instances ----


def test_golden_sulfonyl_so2():
    ms = _matches_of("O=S(=O)(c1ccccc1)c1ccccc1", "sulfonyl_so2_extrusion")
    assert len(ms) == 1  # raw hits: O3/O4 swap times the swapped :1/:5 ends, one edit
    m = ms[0]
    assert m.deleted == (0, 1, 2)
    assert m.form_pairs == ((3, 9),)
    assert m.cut_pairs == ()
    assert m.n_deleted_heavy == 3
    assert m.motif_atoms == (0, 1, 2, 3, 9)
    assert m.motif_numbers == (8, 16, 8, 6, 6)


def test_golden_nitro():
    ms = _matches_of("O=[N+]([O-])c1ccccc1", "nitro_nitrite_NO_loss")
    assert len(ms) == 1
    m = ms[0]
    assert m.deleted == (0, 1)
    assert m.form_pairs == ((2, 3),)
    assert m.n_deleted_heavy == 2


def test_golden_thioether_acyclic_vs_cyclic():
    acyc = "acyclic_thioether_s_extrusion"
    cyc = "cyclic_thioether_s_extrusion"
    open_chain = "CCSCC"
    assert len(_matches_of(open_chain, acyc)) == 1
    assert _matches_of(open_chain, cyc) == []
    ms = _matches_of(open_chain, acyc)
    assert ms[0].deleted == (2,)
    assert ms[0].form_pairs == ((1, 3),)
    assert _matches_of("C1CCSCC1", acyc) == []
    ring = _matches_of("C1CCSCC1", cyc)
    assert len(ring) == 1
    assert ring[0].deleted == (3,)
    assert ring[0].form_pairs == ((2, 4),)


def test_golden_dibenzothiophene_dioxide():
    # four distinct edits: two O choices times two aryl ends
    ms = _matches_of("O=S1(=O)c2ccccc2-c2ccccc12", "s_to_o_aromatic_C")
    assert len(ms) == 4
    keys = {(m.cut_pairs, m.form_pairs) for m in ms}
    assert len(keys) == len(ms)
    for m in ms:
        assert m.deleted == ()
        assert len(m.cut_pairs) == 2 and len(m.form_pairs) == 1


def test_golden_rule_with_dedup_counts_dibenzothiophene():
    # Document the dedup outcome precisely: edits that differ only in which sulfonyl O is chosen
    # have different cut pairs/form pairs (O index differs), so both O choices survive.
    ms = _matches_of("O=S1(=O)c2ccccc2-c2ccccc12", "s_to_o_aromatic_C")
    assert len(ms) == 4


def test_golden_other_rules():
    assert len(_matches_of("c1ccccc1Oc1ccccc1", "diaryl_ether_o_excision")) == 1
    assert len(_matches_of("c1ccccc1Sc1ccccc1", "diaryl_sulfide_s_extrusion")) == 1
    assert len(_matches_of("CSc1nccs1", "hetaryl_thio_alkyl_s_extrusion")) == 1
    assert len(_matches_of("c1ccccc1OP(=S)(OC)OC", "thiono_thiolo_aryl")) == 1
    assert len(_matches_of("CNC(=O)OC", "carbamate_co2_extrusion")) == 1
    assert len(_matches_of("CC(=O)Oc1ccccc1", "co2_extrusion_aryl_Oside")) == 1
    assert len(_matches_of("CS(=O)(=O)CCl", "haloalkyl_sulfonyl_X_to_S")) == 1
    assert len(_matches_of("c1ccccc1N=Nc1ccccc1", "azo_N2_extrusion")) == 1  # the two ends are one edit
    assert len(_matches_of("CCOP(=S)(OC)SCCC(=O)C", "PS_O_transfer_6ring")) == 1
    assert _matches_of("CCOP(=S)(OC)SCC(=O)C", "PS_O_transfer_6ring") == []  # ring too small
    assert len(_matches_of("Oc1ccc([N+](=O)[O-])cc1[N+](=O)[O-]", "dinitrophenol_24_2NO")) >= 1


def test_golden_acid_acyl_transfer():
    # N-acetyl glycine: HOOC-CH2-NH-C(=O)CH3, the 5-membered transfer pattern
    ms = _matches_of("CC(=O)NCC(=O)O", "acid_acyl_transfer_5")
    assert len(ms) == 1
    m = ms[0]
    assert len(m.cut_pairs) == 2 and len(m.form_pairs) == 1
    assert m.deleted == ()


def test_golden_thiadiazole():
    ms = _matches_of("CSc1nnc(C)s1", "thiadiazole_cs2_extrusion")
    assert len(ms) >= 1
    assert all(len(m.deleted) == 3 and m.n_deleted_heavy == 3 for m in ms)


# ---- determinism, truncation, filter ----


def test_truncation_and_determinism():
    mol = _mol("CCSCCSCC")
    full = rr.build_rewire_matches(mol, max_matches_per_rule=100)
    acyc = [m for m in full.matches if m.rule_index == 5]
    assert len(acyc) == 2
    cut = rr.build_rewire_matches(mol, max_matches_per_rule=1)
    kept = [m for m in cut.matches if m.rule_index == 5]
    assert kept == [min(acyc, key=lambda m: (m.motif_atoms, m.deleted, m.cut_pairs, m.form_pairs))]
    assert cut.num_truncated == len(full.matches) - len(cut.matches) == 1
    assert rr.build_rewire_matches(mol, max_matches_per_rule=1) == cut


def test_empty():
    ms = rr.build_rewire_matches(_mol("CCCC"), max_matches_per_rule=10)
    assert ms.matches == () and ms.num_truncated == 0


def test_filter():
    mol = _mol("c1ccccc1N=Nc1ccccc1")
    full = rr.build_rewire_matches(mol, max_matches_per_rule=10)
    assert any(m.rule_index == 13 for m in full.matches)
    pos = rr.filter_rewire_matches(full, precursor_type="[M+H]+", num_known_rules=17)
    neg = rr.filter_rewire_matches(full, precursor_type="[M-H]-", num_known_rules=17)
    assert all(m.rule_index != 13 for m in pos.matches)
    assert any(m.rule_index == 13 for m in neg.matches)
    few = rr.filter_rewire_matches(neg, precursor_type="[M-H]-", num_known_rules=5)
    assert all(m.rule_index < 5 for m in few.matches)
    big = rr.build_rewire_matches(_mol("CCSCCSCC"), max_matches_per_rule=1)
    assert rr.filter_rewire_matches(big, precursor_type="[M+H]+", num_known_rules=17).num_truncated == big.num_truncated


# ---- parity with the gate ----


@pytest.mark.skipif(not (CATALOG.exists() and GATE.exists()), reason="gate script or catalog missing")
def test_matches_equal_gate():
    sys.path.insert(0, str(GATE.parent))
    spec = importlib.util.spec_from_file_location("_rewire_gate_for_test", GATE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    catalog = {r["name"]: r for r in json.loads(CATALOG.read_text())["rules"] if r["status"] == "admitted"}
    smiles = [
        "O=S(=O)(c1ccccc1)c1ccccc1",
        "O=S1(=O)c2ccccc2-c2ccccc12",
        "O=[N+]([O-])c1ccccc1",
        "CCSCCSCC",
        "C1CCSCC1",
        "c1ccccc1Oc1ccccc1",
        "c1ccccc1Sc1ccccc1",
        "CSc1nnc(C)s1",
        "CNC(=O)OC",
        "CC(=O)Oc1ccccc1",
        "c1ccccc1N=Nc1ccccc1",
        "Oc1ccc([N+](=O)[O-])cc1[N+](=O)[O-]",
        "CC(=O)NCC(=O)O",
        "CS(=O)(=O)CCl",
        "c1ccccc1OP(=S)(OC)OC",
        "CSc1nccs1",
        "CCOP(=S)(OC)SCC(=O)CC",
    ]
    total = 0
    for smi in smiles:
        mol = _mol(smi)
        ms = rr.build_rewire_matches(mol, max_matches_per_rule=10_000)
        for i, rule in enumerate(rr.REWIRE_RULES):
            entry = catalog[rule.name]
            hits = gate._matches(mol, Chem.MolFromSmarts(entry["smarts"]))
            # the gate's raw matches collapse to our deduped edits
            expected = set()
            for h in hits:
                expected.add(
                    (
                        tuple(sorted(h.values())),
                        tuple(sorted(h[m] for m in entry["delete"])),
                        tuple(sorted(tuple(sorted((h[a], h[b]))) for a, b in entry["cut"])),
                        tuple(sorted(tuple(sorted((h[a], h[b]))) for a, b in entry["form"])),
                    )
                )
            got = {(m.motif_atoms, m.deleted, m.cut_pairs, m.form_pairs) for m in ms.matches if m.rule_index == i}
            assert got == expected, (smi, rule.name)
            total += len(got)
    assert total > 20
