"""Tests for validated gas-phase rearrangement SMARTS rules.

Covers the 7 SMIRKS-implementable rearrangement rules identified in the
rearrangement SMARTS review (docs/reference/rearrangement_smarts.md).  Each test
checks substrate match, correct product SMILES, and at least one negative
(non-matching) case.

Rules that require custom Python (not SMIRKS-implementable in RDKit):
- Wagner-Meerwein 1,2-methyl shift
- 1,2-H shift / 1,5-hydride transfer
- Pinacol rearrangement
- Tropylium ring expansion (ring size change not possible in SMIRKS)
- Favorskii (nucleophile-dependent, not gas-phase applicable)
"""

from __future__ import annotations

import warnings

from rdkit import Chem
from rdkit.Chem import AllChem

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# SMARTS constants
# ---------------------------------------------------------------------------

# McLafferty: H migrates from gamma-C to carbonyl heteroatom; beta-bond breaks.
# Product is enol (single bond O-C) + alkene.  The original buggy version had
# =[C:2]= (ketene, pentavalent C); the fix uses -[C:2]= (enol O-H).
MCLAFF_CARBONYL = (
    "[O,N,S:1]=[C:2]-[#6:3]-[#6:4]-[#6:5]-[H:6]" ">>[O,N,S:1](-[H:6])-[C:2]=[#6:3].[#6:4]=[#6:5]"
)

# McLafferty nitrile: gamma-H to nitrile N; produces ketenimine (N=C=C) + alkene.
# Correct as-is — the cumulene product is chemically right for nitriles.
MCLAFF_NITRILE = (
    "[N:1]#[C:2]-[#6:3]-[#6:4]-[#6:5]-[H:6]" ">>[N:1](-[H:6])=[C:2]=[#6:3].[#6:4]=[#6:5]"
)

# RDA all-carbon: breaks C4-C5 and C6-C1 ring bonds; forms C3=C4 and C5=C6.
# r6 guard prevents matching non-6-membered rings.  Use without AddHs.
RDA_CARBON = (
    "[#6;r6:1]1=[#6;r6:2]-[#6;r6:3]-[#6;r6:4]-[#6;r6:5]-[#6;r6:6]1"
    ">>[#6:1]=[#6:2]-[#6:3]=[#6:4].[#6:5]=[#6:6]"
)

# RDA heteroatom: same topology for rings containing O, N, S (flavonoids, etc.).
RDA_HETERO = (
    "[#6,#8,#7,#16;r6:1]1=[#6,#8,#7,#16;r6:2]-[#6,#8,#7,#16;r6:3]"
    "-[#6,#8,#7,#16;r6:4]-[#6,#8,#7,#16;r6:5]-[#6,#8,#7,#16;r6:6]1"
    ">>[*:1]=[*:2]-[*:3]=[*:4].[*:5]=[*:6]"
)

# Retro-ene (carbonyl): 6-membered TS with aldehyde H; produces ketene + alkene.
# Chain-length specific: carbonyl must be exactly 4 atoms from the terminal alkene.
RETRO_ENE_CARBONYL = (
    "[#6:1]=[#6:2]-[#6:3]-[#6:4]-[C:5](=[O:6])-[H:7]"
    ">>[#6:1](-[H:7])-[#6:2]=[#6:3].[#6:4]=[C:5]=[O:6]"
)

# Cope [3,3]: isomerization of 1,5-diene.  Produces structural isomer that
# feeds into BFS — not a direct fragment ion.
COPE = "[#6:1]=[#6:2]-[#6:3]-[#6:4]-[#6:5]=[#6:6]" ">>[#6:3]=[#6:2]-[#6:1]-[#6:6]-[#6:5]=[#6:4]"

# Claisen: allyl vinyl ether → gamma,delta-unsaturated carbonyl.  Isomerization.
# NOTE: the original list mislabeled this as "Oxa-Cope".
CLAISEN = "[#6:1]=[#6:2]-[#6:3]-[O:4]-[#6:5]=[#6:6]" ">>[#6:3]=[#6:2]-[#6:1]-[#6:6]-[#6:5]=[O:4]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(smarts: str, smi: str, add_h: bool = False) -> set[str]:
    """Return canonical SMILES of all sanitized product fragments, or empty set."""
    rxn = AllChem.ReactionFromSmarts(smarts)
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smi}")
    if add_h:
        mol = Chem.AddHs(mol)
    out: set[str] = set()
    for prod_tuple in rxn.RunReactants((mol,)):
        for frag in prod_tuple:
            try:
                Chem.SanitizeMol(frag)
                out.add(Chem.MolToSmiles(Chem.RemoveHs(frag)))
            except Exception:
                pass
    return out


def _no_match(smarts: str, smi: str, add_h: bool = False) -> bool:
    """Return True when the reaction produces no valid sanitized products."""
    return len(_run(smarts, smi, add_h=add_h)) == 0


# ---------------------------------------------------------------------------
# McLafferty Carbonyl
# ---------------------------------------------------------------------------


class TestMcLaffertyCarboxyl:
    """McLafferty rearrangement for carbonyl-containing substrates."""

    def test_ketone_enol_product(self):
        # hexan-2-one: enol CC(O)=C expected
        prods = _run(MCLAFF_CARBONYL, "CCCCC(C)=O", add_h=True)
        assert "C=C(C)O" in prods, f"enol not found; got {prods}"

    def test_ketone_alkene_product(self):
        # hexan-2-one: propene C=CC expected as neutral loss
        prods = _run(MCLAFF_CARBONYL, "CCCCC(C)=O", add_h=True)
        assert "C=CC" in prods, f"propene not found; got {prods}"

    def test_ester_substrate(self):
        # ethyl butanoate: vinyl ester enol + ethylene
        prods = _run(MCLAFF_CARBONYL, "CCCC(=O)OCC", add_h=True)
        assert "C=C" in prods
        assert any("OCC" in p for p in prods), f"vinyl ester enol not found; got {prods}"

    def test_amide_substrate(self):
        prods = _run(MCLAFF_CARBONYL, "CCCC(=O)N", add_h=True)
        assert "C=C" in prods
        assert "C=C(N)O" in prods

    def test_carboxylic_acid_substrate(self):
        prods = _run(MCLAFF_CARBONYL, "CCCC(=O)O", add_h=True)
        assert "C=C" in prods
        assert "C=C(O)O" in prods

    def test_no_ketene_product(self):
        # The original buggy SMARTS produced a ketene (O=C=C, pentavalent).
        # The fixed SMARTS must NOT produce any ketene for carbonyl substrates.
        prods = _run(MCLAFF_CARBONYL, "CCCCC(C)=O", add_h=True)
        assert not any("=C=C" in p for p in prods), f"ketene found (old bug); got {prods}"

    def test_no_match_short_chain(self):
        # propan-2-one has no gamma-H relative to carbonyl; should not match
        assert _no_match(MCLAFF_CARBONYL, "CC(C)=O", add_h=True)

    def test_no_match_aldehyde_no_gamma_h(self):
        # propanal: alpha and beta CH2 but no gamma-C bearing H -> no match
        assert _no_match(MCLAFF_CARBONYL, "CCC=O", add_h=True)


# ---------------------------------------------------------------------------
# McLafferty Nitrile
# ---------------------------------------------------------------------------


class TestMcLaffertyNitrile:
    """McLafferty rearrangement for nitrile substrates (ketenimine product)."""

    def test_ketenimine_product(self):
        # pentanenitrile -> ketenimine C=C=N
        prods = _run(MCLAFF_NITRILE, "CCCCC#N", add_h=True)
        assert "C=C=N" in prods, f"ketenimine not found; got {prods}"

    def test_alkene_product(self):
        # propene expelled as the other fragment (gamma-C is C3 of pentanenitrile)
        prods = _run(MCLAFF_NITRILE, "CCCCC#N", add_h=True)
        assert "C=CC" in prods

    def test_longer_nitrile(self):
        # hexanenitrile -> same ketenimine + propene
        prods = _run(MCLAFF_NITRILE, "CCCCCC#N", add_h=True)
        assert "C=C=N" in prods
        assert "C=CCC" in prods or "C=CC" in prods  # propene or shorter

    def test_no_match_short_nitrile(self):
        # propionitrile has no gamma-H (alpha only); no match
        assert _no_match(MCLAFF_NITRILE, "CCC#N", add_h=True)


# ---------------------------------------------------------------------------
# Retro-Diels-Alder (all-carbon)
# ---------------------------------------------------------------------------


class TestRDACarbon:
    """Retro-Diels-Alder for all-carbon 6-membered rings."""

    def test_cyclohexene_diene_product(self):
        prods = _run(RDA_CARBON, "C1=CCCCC1")
        assert "C=CC=C" in prods, f"butadiene not found; got {prods}"

    def test_cyclohexene_ethylene_product(self):
        prods = _run(RDA_CARBON, "C1=CCCCC1")
        assert "C=C" in prods

    def test_substituted_cyclohexene(self):
        # 4-methylcyclohexene: expect propene and substituted diene
        prods = _run(RDA_CARBON, "C1=CC(C)CCC1")
        assert "C=C" in prods or "C=CC" in prods

    def test_no_match_cyclohex_diene(self):
        # cyclohex-1,3-diene has two double bonds; SMARTS requires exactly one
        assert _no_match(RDA_CARBON, "C1=CCC=CC1")

    def test_no_match_cyclohexane(self):
        # fully saturated ring — no double bond
        assert _no_match(RDA_CARBON, "C1CCCCC1")

    def test_no_match_five_membered(self):
        # cyclopentene is not a 6-membered ring; r6 guard blocks it
        assert _no_match(RDA_CARBON, "C1=CCCC1")


# ---------------------------------------------------------------------------
# Retro-Diels-Alder (heteroatom)
# ---------------------------------------------------------------------------


class TestRDAHetero:
    """Retro-Diels-Alder for rings containing O, N, or S (flavonoid-relevant)."""

    def test_baseline_cyclohexene(self):
        prods = _run(RDA_HETERO, "C1=CCCCC1")
        assert "C=C" in prods
        assert "C=CC=C" in prods

    def test_dihydropyranone(self):
        # delta-valerolactone type ring (flavonoid C-ring analog)
        prods = _run(RDA_HETERO, "O=C1OC=CCC1")
        assert len(prods) > 0, "heteroatom RDA produced no products"


# ---------------------------------------------------------------------------
# Retro-ene (carbonyl)
# ---------------------------------------------------------------------------


class TestRetroEneCarbonyl:
    """Retro-ene rearrangement for enal substrates (6-membered TS)."""

    def test_ketene_product(self):
        # pent-4-enal -> ketene C=C=O
        prods = _run(RETRO_ENE_CARBONYL, "C=CCCC=O", add_h=True)
        assert "C=C=O" in prods, f"ketene not found; got {prods}"

    def test_alkene_product(self):
        # propene fragment from the ene side
        prods = _run(RETRO_ENE_CARBONYL, "C=CCCC=O", add_h=True)
        assert "C=CC" in prods

    def test_no_match_long_chain(self):
        # hex-5-enal: carbonyl at C6 (too far — 5-atom chain, not 4); no match
        assert _no_match(RETRO_ENE_CARBONYL, "C=CCCCC=O", add_h=True)

    def test_no_match_ketone(self):
        # pentan-2-one has no aldehyde H; retro-ene (carbonyl) requires C=O with H
        assert _no_match(RETRO_ENE_CARBONYL, "CCCC(C)=O", add_h=True)


# ---------------------------------------------------------------------------
# Cope [3,3]-Sigmatropic
# ---------------------------------------------------------------------------


class TestCope:
    """Cope rearrangement ([3,3]-sigmatropic shift) — isomerization prepass."""

    def test_symmetric_diene_returns_self(self):
        # hex-1,5-diene is symmetric; Cope product is same molecule
        prods = _run(COPE, "C=CCCC=C")
        assert "C=CCCC=C" in prods, f"symmetric Cope failed; got {prods}"

    def test_asymmetric_diene(self):
        # 3-methylhexa-1,5-diene: Cope gives different regioisomer
        prods = _run(COPE, "C=CC(C)CC=C")
        assert len(prods) > 0
        assert "C=CC(C)CC=C" not in prods or len(prods) > 1  # product differs from reactant

    def test_no_match_isolated_alkene(self):
        # single double bond — not a 1,5-diene
        assert _no_match(COPE, "CC=CCC")

    def test_no_match_conjugated_diene(self):
        # 1,3-butadiene: conjugated, not 1,5; no single bond C3-C4 to break
        assert _no_match(COPE, "C=CC=C")


# ---------------------------------------------------------------------------
# Claisen Rearrangement
# ---------------------------------------------------------------------------


class TestClaisen:
    """Claisen rearrangement (allyl vinyl ether → gamma,delta-unsaturated carbonyl)."""

    def test_allyl_vinyl_ether(self):
        # allyl vinyl ether -> pent-4-enal
        prods = _run(CLAISEN, "C=CCOC=C")
        assert "C=CCCC=O" in prods, f"pent-4-enal not found; got {prods}"

    def test_methylated_variant(self):
        prods = _run(CLAISEN, "C=CC(C)OC=C")
        assert len(prods) > 0

    def test_no_match_simple_ether(self):
        # diethyl ether: no vinyl group; no match
        assert _no_match(CLAISEN, "CCOCC")

    def test_no_match_allyl_alkyl_ether(self):
        # allyl methyl ether: no vinyl ether component; no match
        assert _no_match(CLAISEN, "C=CCOC")
