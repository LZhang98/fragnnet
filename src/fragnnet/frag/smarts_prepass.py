"""SMARTS-based rearrangement prepass for multi-bond BFS fragmentation.

Contains :class:`SmartsFragRule`, :data:`FRAG_RULES`, :func:`_apply_smarts_prepass`,
:class:`NeutralLossRule`, :data:`NL_RULES`, and :func:`_apply_nl_prepass` — the
pure-Python prepass for reactions that require new bond formation (1,3-elimination,
acyclic ester CO₂ loss) and neutral-loss rearrangements (SO₂ extrusion, SO₃ loss,
N₂O extrusion).

Kept in a separate module so the Cython ``.so`` can shadow ``multi_cut_bfs``
for the BFS entry-point while this prepass remains importable without any
``importlib`` hacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem

MASK_DTYPE = np.uint8


@dataclass
class SmartsFragRule:
    """SMARTS-based fragmentation rule for the rearrangement prepass.

    Encodes a 2-fragment reaction that cannot be described as a simple bond cut
    (e.g., 1,3-elimination, ester CO₂ loss) by specifying which bonds to cut and
    how to reassemble 3-component results into 2 fragments.

    All atom map numbers must be **1-based** (never 0).  RDKit treats map number 0
    as "no map", so :0 atoms cannot be retrieved by ``GetAtomMapNum()``.

    Attributes:
        name: Rule identifier for logging and debugging.
        reactant_smarts: SMARTS pattern for the reactant substructure only
            (no ">>" product part).  All mapped atoms use 1-based map numbers.
        cut_bond_map_pairs: Pairs of 1-based atom map numbers defining bonds to
            sever.  Each pair ``(a, b)`` identifies the bond between the matched
            atoms carrying map numbers *a* and *b*.
        merge_outer_map_nums: When cutting produces 3 CCs (n_ccs==3), the map
            numbers of the two "outer" atoms whose CCs are merged into one
            fragment.  Empty list discards n_ccs==3 results.
        filter_bond_map_nums: Optional pair of map numbers.  If the matched atoms
            are directly bonded in the original graph the rule is skipped — used
            to exclude cyclic cases already covered by BFS cut=2.
        min_num_atoms: Minimum number of atoms required for this rule to apply.
            Rules are skipped early if the molecule has fewer atoms, avoiding
            unnecessary SMARTS matching on small molecules.

    Example:
        >>> rule = SmartsFragRule(
        ...     name="crf12_0",
        ...     reactant_smarts="[*+0:1]-[*+0:2]-[*+0:3]-[*+0:4]",
        ...     cut_bond_map_pairs=[(1, 2), (3, 4)],
        ...     merge_outer_map_nums=[1, 4],
        ...     filter_bond_map_nums=(1, 4),
        ...     min_num_atoms=4,
        ... )
    """

    name: str
    reactant_smarts: str
    cut_bond_map_pairs: list[tuple[int, int]]
    merge_outer_map_nums: list[int] = field(default_factory=list)
    filter_bond_map_nums: tuple[int, int] | None = None
    min_num_atoms: int = 0


FRAG_RULES: list[SmartsFragRule] = [
    # crf12_0: acyclic 1,3-elimination (single bonds on both sides)
    # A–B–C–D  →  A·D (new bond) + B=C
    # Cut bonds A-B and C-D; outer atoms A,D merge; inner B,C stay together.
    # Skip if A and D are already directly bonded (4-membered ring → BFS cut=2).
    SmartsFragRule(
        name="crf12_0",
        reactant_smarts="[*+0:1]-[*+0:2]-[*+0:3]-[*+0:4]",
        cut_bond_map_pairs=[(1, 2), (3, 4)],
        merge_outer_map_nums=[1, 4],
        filter_bond_map_nums=(1, 4),
        min_num_atoms=4,
    ),
    # crf12_1: acyclic 1,3-elimination (double bond between inner atoms)
    # A–B=C–D  →  A·D (new bond) + B≡C
    SmartsFragRule(
        name="crf12_1",
        reactant_smarts="[*+0:1]-[*+0:2]=[*+0:3]-[*+0:4]",
        cut_bond_map_pairs=[(1, 2), (3, 4)],
        merge_outer_map_nums=[1, 4],
        filter_bond_map_nums=(1, 4),
        min_num_atoms=4,
    ),
    # crf13_2: CO2 loss from acyclic ester
    # R–C(=O)–O–R'  →  CO2  +  R·R' (new R-R' bond)
    # Cut bonds R-C and O-R'; outer R,R' merge; inner C,=O,O stay together (CO2).
    # Skip if R and R' are already directly bonded (lactone → BFS cut=2).
    SmartsFragRule(
        name="crf13_2",
        reactant_smarts="[*+0:1][C+0:2](=[O+0:3])[O+0:4][*+0:5]",
        cut_bond_map_pairs=[(1, 2), (4, 5)],
        merge_outer_map_nums=[1, 5],
        filter_bond_map_nums=(1, 5),
        min_num_atoms=5,
    ),
    # linear_tricyclic_bis_c4_loss: a 6-membered central ring ortho-fused to two
    # *separate* outer 6-rings on opposite edges (anthraquinone/anthracene-type
    # topology) loses both outer rings' non-fusion C4 units in one step, keeping
    # only the central ring (e.g. anthraquinone -> benzoquinone core).
    #
    # NOTE: a single-ring-release rule (cut just one ring's 2 outer bonds) was
    # tried first and empirically added ZERO new DAG formulas -- baseline's
    # native ring-open (n_ccs==1) + ring-cut (n_ccs==2) sequence already reaches
    # that 2-bond-cut atom mask within its normal depth budget. mbfs's measured
    # gain requires a *compounded* 4-bond cut (two ring releases at once) that
    # baseline cannot reach within max_depth=3 (each release costs 2 depth
    # steps sequentially -> 4 total, one over budget). This rule injects that
    # 4-bond result directly as a single depth-2 prepass node.
    #
    # Atom map: 1=central-ring atom (e.g. carbonyl C), 2/7=fusion pair of ring A
    # (2-7 is the shared fusion bond), 3-6=ring A's non-fusion atoms, 8=other
    # central-ring atom, 9/14=fusion pair of ring B (9-14 fusion bond),
    # 10-13=ring B's non-fusion atoms. Cut (2,3),(6,7),(9,10),(13,14) to release
    # rings A and B; merge their non-fusion atoms (3 and 10) into one outer
    # fragment, leaving the central ring {1,2,7,8,9,14} as the inner fragment.
    SmartsFragRule(
        name="linear_tricyclic_bis_c4_loss",
        reactant_smarts="[*+0:1]1[c+0:2]2[c+0:3][c+0:4][c+0:5][c+0:6][c+0:7]2[*+0:8][c+0:9]3[c+0:10][c+0:11][c+0:12][c+0:13][c+0:14]31",
        cut_bond_map_pairs=[(2, 3), (6, 7), (9, 10), (13, 14)],
        merge_outer_map_nums=[3, 10],
        min_num_atoms=14,
    ),
    # anthraquinone_bis_co_loss: same linearly-fused tricyclic, but the two central-ring
    # non-fusion atoms are explicit carbonyls (anthraquinone's 9,10-dione). Loses BOTH
    # carbonyl oxygens AND one flanking ring's non-fusion C4 unit in one compound cut,
    # keeping the other flanking ring fully intact (matches the empirically dominant
    # mbfs-confirmed formula family for anthraquinones: [M-2O-C4H4]-type ions, i.e. loss
    # of both C=O oxygens plus one ring, equivalent to bis-CO loss with ring contraction).
    # Atom map: 1/10=carbonyl O's, 2/9=carbonyl C's, 3/8=ring-A fusion pair, 4-7=ring A's
    # non-fusion atoms (released), 11/16=ring-B fusion pair, 12-15=ring B (kept intact).
    # Cuts: O=C double bonds (1,2)/(9,10) plus ring A's outer bonds (3,4)/(7,8); the 3
    # discarded CCs ({O1},{O10},{ring A C4}) merge into one "outer" byproduct mask, the
    # remaining connected piece (both carbonyl C's + intact ring B) is the "inner" kept
    # fragment. GetSubstructMatches naturally also returns the mirrored embedding
    # (releasing ring B, keeping ring A), so both directions are covered by one rule.
    # NOTE: map numbers must be 1-based -- RDKit treats map 0 as "unmapped" and silently
    # drops it from q_idx_to_map, which makes cut_bond_map_pairs lookups fail (the bug
    # in the first version of this rule, caught by empirical validation returning 0 hits).
    SmartsFragRule(
        name="anthraquinone_bis_co_loss",
        reactant_smarts=(
            "[O+0:1]=[C+0:2]1[c+0:3]2[c+0:4][c+0:5][c+0:6][c+0:7][c+0:8]2"
            "[C+0:9](=[O+0:10])[c+0:11]3[c+0:12][c+0:13][c+0:14][c+0:15][c+0:16]31"
        ),
        cut_bond_map_pairs=[(1, 2), (9, 10), (3, 4), (7, 8)],
        merge_outer_map_nums=[1, 10, 4],
        min_num_atoms=16,
    ),
    # xanthone_ring_cleavage: dibenzo-γ-pyranone (xanthone / xanthen-9-one) scaffold.
    # Two benzene rings fused on either side of a central ring containing both a
    # carbonyl (=O on :1) and a bridge ether O. Releases both flanking benzene C4
    # units simultaneously with 4 ring-bond cuts: (2,3) and (6,7) bracket the left
    # benzene's non-fusion atoms; (9,10) and (13,14) bracket the right benzene's
    # non-fusion atoms. Produces 3 CCs: left C4H4 outer, right C4H4 outer, and the
    # central pyranone ring (C5O2) as the inner kept fragment.
    # Baseline BFS cannot reach this within max_depth=3: releasing one benzene costs
    # 2 sequential cuts (depth +2), both take 4 → depth 4, one over budget. This
    # rule injects it as a single depth-2 prepass leaf.
    # merge_outer_map_nums=[3, 10]: one atom from each outer C4 piece.
    # min_num_atoms=15 (bare xanthone C13H8O2 has 15 heavy atoms).
    SmartsFragRule(
        name="xanthone_ring_cleavage",
        reactant_smarts=(
            "[O+0]=[c+0:1]1[c+0:2]2[c+0:3][c+0][c+0][c+0:6][c+0:7]2"
            "[o+0][c+0:9]3[c+0:10][c+0][c+0][c+0:13][c+0:14]31"
        ),
        cut_bond_map_pairs=[(2, 3), (6, 7), (9, 10), (13, 14)],
        merge_outer_map_nums=[3, 10],
        min_num_atoms=15,
    ),
    # phenoxazine_ring_cleavage: dibenz[b,e][1,4]oxazine (phenoxazine) scaffold.
    # Two benzene rings fused on either side of a central 6-membered ring containing
    # one N and one O bridge. Releases both flanking benzene C4 units simultaneously
    # with 4 ring-bond cuts: (2,3) and (6,7) bracket the right benzene's non-fusion
    # atoms; (9,10) and (13,14) bracket the left benzene's non-fusion atoms.
    # Produces 3 CCs: right C4 outer, left C4 outer, and the central C4NO ring as
    # the inner kept fragment.
    # Baseline BFS cannot reach this within max_depth=3: releasing one benzene costs
    # 2 sequential cuts (depth +2), both take 4 → depth 4, one over budget. This
    # rule injects it as a single depth-2 prepass leaf.
    # merge_outer_map_nums=[3, 10]: one atom from each outer C4 piece.
    # min_num_atoms=13 (bare phenoxazine C12H9NO has 13 heavy atoms).
    # Note: [N+0:1] matches both NH (phenoxazine) and N-alkyl (e.g. N-phenylphenoxazine)
    # since the N charge constraint (+0) excludes N+ quaternary but allows both variants.
    # [O+0] (not [o+0]) is required because the bridge O in phenoxazine is non-aromatic.
    SmartsFragRule(
        name="phenoxazine_ring_cleavage",
        reactant_smarts=(
            "[N+0:1]1[c+0:2]2[c+0:3][c+0][c+0][c+0:6][c+0:7]2[O+0][c+0:9]3"
            "[c+0:10][c+0][c+0][c+0:13][c+0:14]31"
        ),
        cut_bond_map_pairs=[(2, 3), (6, 7), (9, 10), (13, 14)],
        merge_outer_map_nums=[3, 10],
        min_num_atoms=13,
    ),
    # furanocoumarin_ring_cleavage: furan–benzene–pyranone (psoralen/linear furanocoumarin)
    # scaffold. The three fused rings share two fusion bonds: benzene∩furan (atoms :5–:3
    # shared edge) and benzene∩pyranone (atoms :8–:1 shared edge). This rule cuts 4
    # non-fusion ring bonds that bracket the central benzene's furan-facing atoms: (5,6)
    # and (3,4) release the furan's non-fusion atoms {furan CH, CH, O} as one outer CC,
    # while (1,2) and (7,8) release the entire pyranone ring as the other outer CC. The
    # inner kept fragment is the 4-atom open chain {benzene C's flanking the furan fusion
    # bond} plus any substituents on those atoms.
    # Baseline BFS cannot reach this within max_depth=3: releasing the furan non-fusion
    # atoms + pyranone simultaneously requires 4 sequential depth steps (two ring
    # openings × 2 each), one over budget. This rule injects it as a single depth-2
    # prepass leaf.
    # merge_outer_map_nums=[4, 8]: atom :4 is in the furan outer CC {7,8,9}; atom :8 is
    # in the pyranone outer CC {0,1,2,3,4,12,13} (psoralen numbering).
    # min_num_atoms=14 (bare psoralen C11H6O3 has 14 heavy atoms).
    SmartsFragRule(
        name="furanocoumarin_ring_cleavage",
        reactant_smarts=(
            "[O+0]=[c+0]1[c+0][c+0][c+0:8]2[c+0:7][c+0:5]3[c+0:6][c+0][o+0:4][c+0:3]3"
            "[c+0:2][c+0:1]2[o+0]1"
        ),
        cut_bond_map_pairs=[(1, 2), (3, 4), (5, 6), (7, 8)],
        merge_outer_map_nums=[4, 8],
        min_num_atoms=14,
    ),
    # morphinan_ring_cleavage: 5-ring morphinan alkaloid scaffold (morphine/codeine class).
    # Piperidine ring (ring 0) fused to a benzene ring (ring 1) via a dihydrofuran bridge ring
    # (ring 2, 5-membered, ether O) and a cyclohexene ring (ring 3), with a 6-membered bridge
    # ring (ring 4) spanning the whole system. 4 cuts simultaneously: N-C (piperidine, maps 1-2),
    # C-C (cyclohexene C15-C16, maps 3-4), C(bridgehead)-C (C6-C20, maps 5-6), c-C (C8-C22
    # aromatic-bridge, maps 7-8). Produces n_ccs=2: inner kept fragment is the benzene+dihydrofuran
    # bicyclic C10HxO2 piece (atoms from piperidine CH2-CH2-bridgehead + benzene + ether O chain);
    # outer is the large N-containing piece (allyl-N + ring 3 + ring 4 remainder).
    # GT-confirmed formula C10H8O2 in 20+ NIST20 morphinan molecules; base BFS misses it because
    # 4 cuts from root exceed max_cut_size=2 × max_depth=3 within the normal sequential budget.
    # min_num_atoms=18 (the ring system alone has 18 heavy atoms: 1N + 1O + 16C).
    # Note: ~ bonds in ring 3 allow single (naloxone) or double (morphine) C-C bond variants.
    SmartsFragRule(
        name="morphinan_ring_cleavage",
        reactant_smarts=(
            "[N+0:1]1[C:2][C][C:5]23[c]4[c:7]5[c][c][c][c]4O[C:3]2[C:4]~[C;r]~[C][C:6]3C1[C:8]5"
        ),
        cut_bond_map_pairs=[(1, 2), (3, 4), (5, 6), (7, 8)],
        merge_outer_map_nums=[],
        min_num_atoms=18,
    ),
    # benzopyranone_ring_cleavage: benzene fused to a γ-lactone (pyranone) ring,
    # i.e. the coumarin / chromen-2-one scaffold (and its substituted variants).
    # Requires 4 simultaneous bond cuts: two adjacent benzene C-C bonds on either
    # side of one ring carbon (:1) isolate that carbon from the benzene, while the
    # two pyranone bonds C8a-O (:4-:5) and C2-C3 (:6-:7) open the lactone ring.
    # Produces 3 CCs: {atom :1 + any exo-substituents} (small outer piece),
    # {atoms :5 + :6 + exo=O} (the O=C-O lactone fragment), and the remaining
    # large carbocyclic fragment. Outer CCs (containing :1 and :5) are merged into
    # one byproduct mask; the large carbon fragment is kept as the inner result.
    #
    # Baseline BFS cannot reach this fragment within max_depth=3: the two ring
    # openings (benzene + pyranone) each cost 2 sequential depth steps → 4 total,
    # one over budget. This rule injects the 4-bond compound cut directly as a
    # single depth-2 prepass leaf, matching the dominant mbfs-confirmed fragment
    # family for 3-aryl-7-methoxycoumarin scaffolds (empirically, 52 improved
    # molecules in NIST20 with wrecall_h4 gap > 0.01).
    #
    # Atom map:
    #   Benzene ring: :1-:2-:3-:4-:9-:10 (ring1; :1 is the isolated C)
    #   Pyranone ring: :4-:5-:6-:7-:8-:9 (ring2; :4/:9 are ring-fusion atoms)
    #   Exo =O on :6 is unmapped (included in match tuple but not in map dict).
    # Cuts: (1,2) and (10,1) to bracket :1; (4,5) for C8a-O; (6,7) for C2-C3.
    # NOTE: [OC] (used in an earlier draft) is invalid SMARTS — it means "atom that
    # is simultaneously O and C" (impossible), so always returns 0 matches.
    SmartsFragRule(
        name="benzopyranone_ring_cleavage",
        reactant_smarts=(
            "[c+0:1]1[c+0:2][c+0:3][c+0:4]2[o+0:5][c+0:6](=O)[c+0:7][c+0:8][c+0:9]2[c+0:10]1"
        ),
        cut_bond_map_pairs=[(1, 2), (10, 1), (4, 5), (6, 7)],
        merge_outer_map_nums=[1, 5],
        min_num_atoms=11,
    ),
]


@dataclass
class NeutralLossRule:
    """Rule for rearrangements that expel a neutral fragment and form a new bond.

    Unlike :class:`SmartsFragRule`, these reactions cannot be represented as
    simple bond cuts because the product requires bond formation (e.g. SO₂
    extrusion creates a new Ar–N bond).  The product is represented as an atom
    mask covering all precursor atoms *except* the neutral-loss atoms.  Because
    the new bond is not in the original graph, the product atoms may be
    disconnected in the original connectivity — H-count estimation will add
    excess H's at the free valences; the correct formula lands at a negative
    H-shift in the formula graph (covered by max_h_transfer).

    Attributes:
        name: Rule identifier for logging.
        reactant_smarts: SMARTS pattern.  All mapped atoms use 1-based map numbers.
        nl_map_nums: Map numbers of the atoms that leave as the neutral loss.
            The product mask = all atoms minus these.
        nl_mass: Exact monoisotopic neutral-loss mass (Da), for documentation.
        min_num_atoms: Minimum molecule heavy-atom count for early exit.
    """

    name: str
    reactant_smarts: str
    nl_map_nums: list[int]
    nl_mass: float
    min_num_atoms: int = 0


NL_RULES: list[NeutralLossRule] = [
    # Aromatic sulfonamide SO₂ extrusion (HIGHEST PRIORITY).
    # Ar–SO₂–NH–Het → [Ar–NH–Het]⁻ + SO₂  (NL = 63.962 Da).
    # A new Ar–N bond forms, so the product atoms are disconnected in the original
    # graph.  H-cap adds +2H (one at Ar–C, one at N); correct formula at H_delta=-2.
    # Confirmed in spec_id=610784 and 856552; affects thousands of NIST23 spectra.
    NeutralLossRule(
        name="aryl_sulfonamide_so2",
        reactant_smarts="[c:1]-[SX4:2](=[O:3])(=[O:4])-[NH:5]",
        nl_map_nums=[2, 3, 4],
        nl_mass=63.9619,
        min_num_atoms=8,
    ),
    # Aromatic sulfonate SO₂ loss.
    # Ar–SO₃⁻ → [Ar–O]⁻ + SO₂  (NL = 63.962 Da).
    # A new Ar–O bond forms; +2H excess at Ar–C and O-; correct formula at H_delta=-2.
    # Root cause of miss in pyrenes (82.8%) and phenanthrenes (50.9%) in [M-H]-.
    NeutralLossRule(
        name="aryl_sulfonate_so2",
        reactant_smarts="[c:1]-[SX4:2](=[O:3])(=[O:4])[O-,OH:5]",
        nl_map_nums=[2, 3, 4],
        nl_mass=63.9619,
        min_num_atoms=7,
    ),
    # Aromatic sulfonate SO₃ loss.
    # Ar–SO₃⁻ → [Ar]⁻ + SO₃  (NL = 79.957 Da).
    # Direct S–C bond cleavage; product is a connected Ar anion.
    # +1H excess at Ar–C; correct formula at H_delta=-1.
    NeutralLossRule(
        name="aryl_sulfonate_so3",
        reactant_smarts="[c:1]-[SX4:2](=[O:3])(=[O:4])[O-,OH:5]",
        nl_map_nums=[2, 3, 4, 5],
        nl_mass=79.9568,
        min_num_atoms=7,
    ),
    # N-nitramine N₂O extrusion.
    # R–N(NO₂)–NH–R' → [R–N=R'] + N₂O  (NL = 44.001 Da).
    # Confirmed in spec_id=550113 (0.1 mDa).  The O⁻ from NO₂ (:3) stays in the
    # product; +2H excess at O⁻ and at R' atom; correct formula at H_delta=-2.
    NeutralLossRule(
        name="n_nitramine_n2o",
        reactant_smarts="[N+:1](=[O:2])([O-:3])-[NH:4]-[#6,#7]",
        nl_map_nums=[1, 2, 4],
        nl_mass=44.0010,
        min_num_atoms=5,
    ),
    # Sulfoxide SO loss.
    # R–S(=O)–R' → [R–R'] + SO  (NL = 47.967 Da).
    # The C–S bonds break and a new C–C bond forms in the product; retained atoms
    # may be disconnected in the original graph (+2H excess; correct formula at H_delta=-2).
    # SX3 matches only sulfoxides (3-connected S), excluding sulfones (SX4) and thioethers.
    NeutralLossRule(
        name="sulfoxide_so_loss",
        reactant_smarts="[#6:1]-[SX3:2](=[O:3])-[#6:4]",
        nl_map_nums=[2, 3],
        nl_mass=47.9669,
        min_num_atoms=3,
    ),
    # Diarylethylene (stilbene-type) C₂H₂ extrusion.
    # Ar–CH=CH–Ar' → [Ar–Ar'] + HC≡CH  (NL = 26.016 Da).
    # The two vinyl CH atoms (:2, :3), each bearing one H, leave as acetylene (C₂H₂).
    # A new Ar–Ar' bond forms in the product (not present in the precursor), making
    # this a rearrangement outside BFS DAG coverage. Product atoms are disconnected
    # in the original graph; H-cap adds +1H at each free Ar valence, so the correct
    # biaryl formula lands at H_delta=-2 (two excess H from the two cut free valences).
    # `!R` on :2/:3 excludes bridged vinyl units in fused rings (indene, acenaphthylene)
    # where the vinyl CH=CH is part of a ring and BFS already covers the 2-bond cut.
    # min_num_atoms=8: minimum diarylethylene is two 5-membered aromatics + 2 vinyl C.
    NeutralLossRule(
        name="diarylvinyl_c2h2_loss",
        reactant_smarts="[c:1]-[CH;!R:2]=[CH;!R:3]-[c:4]",
        nl_map_nums=[2, 3],
        nl_mass=26.0157,
        min_num_atoms=8,
    ),
]

# Pre-compiled SMARTS patterns (keyed by rule name) — avoid recompiling per molecule.
_COMPILED_PATTERNS: dict[str, Chem.Mol] = {}
_COMPILED_NL_PATTERNS: dict[str, Chem.Mol] = {}


def _get_compiled_pattern(rule: SmartsFragRule) -> Chem.Mol | None:
    """Return a cached compiled SMARTS pattern for *rule*, compiling on first call."""
    if rule.name not in _COMPILED_PATTERNS:
        _COMPILED_PATTERNS[rule.name] = Chem.MolFromSmarts(rule.reactant_smarts)
    return _COMPILED_PATTERNS[rule.name]


def _union_find_ccs_all(
    num_nodes: int,
    node_mask: np.ndarray,
    edges: np.ndarray,
    edge_mask: np.ndarray,
    num_edges: int,
) -> list[np.ndarray]:
    """Compute all connected components without a cap on their count.

    Used by the SMARTS prepass where reactions may produce 3 components
    (e.g., 1,3-elimination splits A-B-C-D into {A}, {B,C}, {D}).

    Args:
        num_nodes: Total atom count.
        node_mask: Shape ``(num_nodes,)`` uint8 — active atoms.
        edges: Shape ``(MAX_NUM_EDGES, 2)`` int32 — bond endpoints.
        edge_mask: Shape ``(MAX_NUM_EDGES,)`` uint8 — active bonds.
        num_edges: Number of real bonds (un-padded prefix length).

    Returns:
        List of node-mask arrays (shape ``(num_nodes,)`` uint8), one per CC.
        Empty list if no active atoms.
    """
    # Use a Python list for parent to avoid numpy indexing overhead in the hot loop.
    parent = list(range(num_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    live_atoms = [i for i in range(num_nodes) if node_mask[i] == 1]
    if not live_atoms:
        return []

    for idx in range(num_edges):
        if edge_mask[idx] == 0:
            continue
        u, v = int(edges[idx, 0]), int(edges[idx, 1])
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv

    roots: dict[int, int] = {}
    for a in live_atoms:
        r = find(a)
        if r not in roots:
            roots[r] = len(roots)

    cc_masks = [np.zeros(num_nodes, dtype=MASK_DTYPE) for _ in range(len(roots))]
    for a in live_atoms:
        cc_masks[roots[find(a)]][a] = 1

    return cc_masks


def _build_edge_dict(edges: np.ndarray, num_edges: int) -> dict[tuple[int, int], int]:
    """Build a reverse lookup from (atom_u, atom_v) → edge index.

    Both orderings (u, v) and (v, u) are stored so lookups are O(1) regardless
    of the direction the edge was stored in the array.

    Args:
        edges: Shape ``(MAX_NUM_EDGES, 2)`` int32 — bond endpoints.
        num_edges: Number of real bonds (un-padded prefix length).

    Returns:
        Dict mapping ``(atom_a, atom_b)`` to its 0-based edge index.
    """
    d: dict[tuple[int, int], int] = {}
    for i in range(num_edges):
        u, v = int(edges[i, 0]), int(edges[i, 1])
        d[(u, v)] = i
        d[(v, u)] = i
    return d


def _apply_smarts_prepass(
    rules: list[SmartsFragRule],
    mol: Chem.Mol,
    base_edge_mask: np.ndarray,
    edges: np.ndarray,
    num_nodes: int,
    num_edges: int,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Apply SMARTS fragmentation rules to the root molecule.

    For each rule, matches the reactant SMARTS against *mol*, cuts the specified
    bonds, and decomposes the result into exactly two fragment node-masks.
    When a cut produces 3 components (e.g., 1,3-elimination), the two "outer"
    components (containing ``merge_outer_map_nums`` atoms) are merged into one
    fragment.

    Only the root (un-fragmented) molecule is processed; the resulting fragment
    pairs are injected as depth-1 children in the DAG.

    Args:
        rules: List of :class:`SmartsFragRule` to apply.
        mol: RDKit molecule (same atoms/indices used to build the edge arrays).
        base_edge_mask: Shape ``(MAX_NUM_EDGES,)`` uint8 — root edge mask (all
            real bonds active).
        edges: Shape ``(MAX_NUM_EDGES, 2)`` int32 — bond endpoints.
        num_nodes: Number of atoms.
        num_edges: Number of real bonds.

    Returns:
        List of ``(node_mask_a, node_mask_b, rule_idx)`` tuples, one per valid
        match.  ``rule_idx`` is the 0-based index into *rules* that produced the
        pair.  Duplicate fragment pairs are **not** deduplicated here; the BFS
        ``css_to_id_dict`` handles deduplication by node-mask key.
    """
    root_node_mask = np.ones(num_nodes, dtype=MASK_DTYPE)
    results: list[tuple[np.ndarray, np.ndarray, int]] = []

    # Fix 1: build (atom_a, atom_b) → edge_idx lookup once per molecule call.
    edge_dict = _build_edge_dict(edges, num_edges)

    for rule_idx, rule in enumerate(rules):
        # Fix 3: skip rules that can't match given the molecule size.
        if rule.min_num_atoms > 0 and num_nodes < rule.min_num_atoms:
            continue

        # Use pre-compiled SMARTS pattern (avoids recompiling on every molecule).
        patt = _get_compiled_pattern(rule)
        if patt is None:
            continue

        # Build: query atom index → atom map number (skip unmapped atoms, i.e. map==0)
        q_idx_to_map: dict[int, int] = {}
        for q_idx in range(patt.GetNumAtoms()):
            map_num = patt.GetAtomWithIdx(q_idx).GetAtomMapNum()
            if map_num != 0:
                q_idx_to_map[q_idx] = map_num

        matches = mol.GetSubstructMatches(patt)
        for match in matches:
            # match[q_idx] = mol atom index for query atom q_idx
            map_num_to_mol: dict[int, int] = {
                map_num: match[q_idx] for q_idx, map_num in q_idx_to_map.items()
            }

            # Skip cyclic cases already covered by BFS cut=2.
            # Fix 1: O(1) dict lookup instead of O(E) np.where scan.
            if rule.filter_bond_map_nums is not None:
                fa, fb = rule.filter_bond_map_nums
                mol_a = map_num_to_mol.get(fa)
                mol_b = map_num_to_mol.get(fb)
                if mol_a is not None and mol_b is not None:
                    bond_idx = edge_dict.get((mol_a, mol_b), -1)
                    if bond_idx != -1 and base_edge_mask[bond_idx] == 1:
                        continue  # atoms directly bonded → cyclic; covered by cut=2

            # Cut the specified bonds in a temporary mask.
            work_mask = base_edge_mask.copy()
            valid = True
            for map_a, map_b in rule.cut_bond_map_pairs:
                mol_a = map_num_to_mol.get(map_a)
                mol_b = map_num_to_mol.get(map_b)
                if mol_a is None or mol_b is None:
                    valid = False
                    break
                bond_idx = edge_dict.get((mol_a, mol_b), -1)  # Fix 1: O(1) lookup
                if bond_idx == -1:
                    valid = False
                    break
                work_mask[bond_idx] = 0
            if not valid:
                continue

            # Fix 2: _union_find_ccs_all now uses Python list for parent array
            # (avoids numpy element-wise indexing overhead in the union-find loop).
            cc_masks = _union_find_ccs_all(num_nodes, root_node_mask, edges, work_mask, num_edges)
            n_ccs = len(cc_masks)

            if n_ccs == 2:
                results.append((cc_masks[0], cc_masks[1], rule_idx))
            elif n_ccs >= 3 and rule.merge_outer_map_nums:
                # Identify which CCs contain the "outer" atoms. Merge ALL outer CCs into
                # one fragment and ALL remaining ("inner") CCs into the other -- generalizes
                # the original 2-outer/1-inner n_ccs==3 case to any CC count/split (e.g. one
                # CC per discarded atom/group, as in a multi-bond compound cut).
                outer_cc_set: set[int] = set()
                for map_num in rule.merge_outer_map_nums:
                    mol_idx = map_num_to_mol.get(map_num)
                    if mol_idx is not None:
                        for cc_idx, cc_mask in enumerate(cc_masks):
                            if cc_mask[mol_idx] == 1:
                                outer_cc_set.add(cc_idx)
                                break
                inner_cc_set = set(range(n_ccs)) - outer_cc_set
                if outer_cc_set and inner_cc_set:
                    outer_mask = np.zeros(num_nodes, dtype=MASK_DTYPE)
                    for cc_idx in outer_cc_set:
                        outer_mask |= cc_masks[cc_idx]
                    inner_mask = np.zeros(num_nodes, dtype=MASK_DTYPE)
                    for cc_idx in inner_cc_set:
                        inner_mask |= cc_masks[cc_idx]
                    results.append((outer_mask, inner_mask, rule_idx))

    return results


def _get_compiled_nl_pattern(rule: NeutralLossRule) -> Chem.Mol | None:
    """Return a cached compiled SMARTS pattern for *rule*, compiling on first call."""
    if rule.name not in _COMPILED_NL_PATTERNS:
        _COMPILED_NL_PATTERNS[rule.name] = Chem.MolFromSmarts(rule.reactant_smarts)
    return _COMPILED_NL_PATTERNS[rule.name]


def _apply_nl_prepass(
    rules: list[NeutralLossRule],
    mol: Chem.Mol,
    num_nodes: int,
) -> list[tuple[np.ndarray, int]]:
    """Apply neutral-loss rules to the root molecule.

    For each rule, matches the reactant SMARTS against *mol* and returns a
    product atom mask that excludes the neutral-loss atoms.  Because the
    product requires new bond formation, the retained atoms may be disconnected
    in the original graph; H-count estimation adds excess H's at the free
    valences.  The correct formula therefore lands at a negative H-shift
    (see :class:`NeutralLossRule` docstring).

    Args:
        rules: List of :class:`NeutralLossRule` to apply.
        mol: RDKit molecule.
        num_nodes: Number of heavy atoms.

    Returns:
        List of ``(product_mask, rule_idx)`` tuples.  *product_mask* is a
        ``(num_nodes,)`` uint8 array with 1 for retained atoms and 0 for
        neutral-loss atoms.  Duplicate NL atom sets (same atoms lost via
        different SMARTS embeddings) are deduplicated.
    """
    results: list[tuple[np.ndarray, int]] = []
    for rule_idx, rule in enumerate(rules):
        if rule.min_num_atoms > 0 and num_nodes < rule.min_num_atoms:
            continue
        patt = _get_compiled_nl_pattern(rule)
        if patt is None:
            continue
        q_idx_to_map: dict[int, int] = {}
        for q_idx in range(patt.GetNumAtoms()):
            map_num = patt.GetAtomWithIdx(q_idx).GetAtomMapNum()
            if map_num != 0:
                q_idx_to_map[q_idx] = map_num
        seen_nl_sets: set[frozenset] = set()
        for match in mol.GetSubstructMatches(patt):
            map_to_mol: dict[int, int] = {m: match[q] for q, m in q_idx_to_map.items()}
            nl_atoms: set[int] = set()
            valid = True
            for nl_map in rule.nl_map_nums:
                mol_idx = map_to_mol.get(nl_map)
                if mol_idx is None:
                    valid = False
                    break
                nl_atoms.add(mol_idx)
            if not valid:
                continue
            nl_key = frozenset(nl_atoms)
            if nl_key in seen_nl_sets:
                continue
            seen_nl_sets.add(nl_key)
            product_mask = np.ones(num_nodes, dtype=MASK_DTYPE)
            for idx in nl_atoms:
                product_mask[idx] = 0
            results.append((product_mask, rule_idx))
    return results
