"""mol_graph.py (adapted from SCARF)

Classes to featurize molecules into a graph with onehot concat feats on atoms
and bonds. Inspired by the dgllife library.

"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy
import torch as th
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from fragnnet.utils.formula_utils import CANONICAL_ELEMENT_ORDER
from fragnnet.utils.mol_utils import rdkit_import

# TODO: remove this registry

atom_feat_registry: dict[str, dict[str, Any]] = {}
bond_feat_registry: dict[str, dict[str, Any]] = {}


def register_bond_feat(cls: type) -> type:
    """Register a bond featurizer class in the global registry."""

    bond_feat_registry[cls.name] = {"fn": cls.featurize, "feat_size": cls.feat_size}
    return cls


def register_atom_feat(cls: type) -> type:
    """Register an atom featurizer class in the global registry."""

    atom_feat_registry[cls.name] = {"fn": cls.featurize, "feat_size": cls.feat_size}
    return cls


def get_mol_feats_sizes(
    atom_feats: Sequence[str] | None, bond_feats: Sequence[str] | None, pe_embed_k: int
) -> tuple[int, int]:
    """Return the dimensions of atom and bond feature vectors for a MolGraph."""

    mg = MolGraph(atom_feats, bond_feats, pe_embed_k)
    return mg.num_atom_feats, mg.num_bond_feats


def _atom_free_valence_slots(atom: Chem.rdchem.Atom) -> int:
    """Return how many more hydrogens an atom could carry in the intact molecule.

    Equal to ``ve - sbond - h``, the quantities FraGNNet builds in
    ``dag.builder.extract_mol_info``: ``ve`` is the atom's total valence (less a positive
    formal charge that funds a heavy-atom bond rather than an H slot), ``sbond`` its
    heavy-atom degree, and ``h`` its current hydrogen count. It is exactly the atom's
    share of unsaturation -- the sum of ``bond order - 1`` over its bonds -- so a
    saturated atom scores 0 and each double bond contributes 1 to both endpoints.

    Args:
        atom: RDKit atom from a molecule with implicit hydrogens.

    Returns:
        Spare hydrogen slots, before any bond to a neighbouring fragment is cut.
    """
    hydrogens = atom.GetTotalNumHs()
    valence = atom.GetTotalValence()
    # A cationic atom carrying no hydrogens spends its extra valence on a heavy-atom bond,
    # not on an H slot; protonated amines ([NH3+]) genuinely use it for H, hence the guard.
    if atom.GetFormalCharge() > 0 and hydrogens == 0:
        valence -= atom.GetFormalCharge()
    heavy_degree = atom.GetTotalDegree() - hydrogens
    return valence - heavy_degree - hydrogens


class MolGraph:
    def __init__(
        self,
        atom_feats: Sequence[str] | None = None,
        bond_feats: Sequence[str] | None = None,
        pe_embed_k: int = 0,
    ):
        """Create a molecular graph featurizer with atom/bond feature sets."""
        if atom_feats is None:
            atom_feats = [
                "a_onehot",
                "a_degree",
                "a_hybrid",
                "a_formal",
                "a_radical",
                "a_ring",
                "a_mass",
                "a_chiral",
            ]
        if bond_feats is None:
            bond_feats = ["b_degree"]

        self.pe_embed_k = pe_embed_k
        self.atom_feats = atom_feats
        self.bond_feats = bond_feats
        self.a_featurizers = []
        self.b_featurizers = []

        self.num_atom_feats = 0
        self.num_bond_feats = 0

        for i in self.atom_feats:
            if i not in atom_feat_registry:
                raise ValueError(f"Feat {i} not recognized")
            feat_obj = atom_feat_registry[i]
            self.num_atom_feats += feat_obj["feat_size"]
            self.a_featurizers.append(feat_obj["fn"])

        for i in self.bond_feats:
            if i not in bond_feat_registry:
                raise ValueError(f"Feat {i} not recognized")
            feat_obj = bond_feat_registry[i]
            self.num_bond_feats += feat_obj["feat_size"]
            self.b_featurizers.append(feat_obj["fn"])

        self.num_atom_feats += self.pe_embed_k
        self.needs_gasteiger = "a_charge" in self.atom_feats

    def get_mol_graph(
        self,
        mol: Chem.rdchem.Mol,
        bigraph: bool = True,
    ) -> dict[str, np.ndarray]:
        """Build raw numpy feature tensors for a molecule.

        Args:
            mol: RDKit molecule.
            bigraph: If ``True``, create bidirectional edges.

        Returns:
            Mapping containing atom features (|N| x d_n), bond features (|E| x d_e),
            and bond index tuples (|E| x 2).
        """
        if self.needs_gasteiger:
            AllChem.ComputeGasteigerCharges(mol)
        all_atoms = mol.GetAtoms()
        all_bonds = mol.GetBonds()
        bond_feats = []
        bond_tuples = []
        atom_feats = []
        for bond in all_bonds:
            strt = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            bond_tuples.append((strt, end))
            bond_feat = []
            for fn in self.b_featurizers:
                bond_feat.extend(fn(bond))
            bond_feats.append(bond_feat)

        for atom in all_atoms:
            atom_feat = []
            for fn in self.a_featurizers:
                atom_feat.extend(fn(atom))
            atom_feats.append(atom_feat)

        atom_feats = np.asarray(atom_feats).reshape(
            len(atom_feats), self.num_atom_feats - self.pe_embed_k
        )
        bond_feats = np.asarray(bond_feats).reshape(len(bond_feats), self.num_bond_feats)
        bond_tuples = np.asarray(bond_tuples, dtype=np.int64).reshape(-1, 2)

        # Add doubles
        if bigraph:
            rev_bonds = np.vstack([bond_tuples[:, 1], bond_tuples[:, 0]]).transpose(1, 0)
            bond_tuples = np.vstack([bond_tuples, rev_bonds])
            bond_feats = np.vstack([bond_feats, bond_feats])
        return {
            "atom_feats": atom_feats,
            "bond_feats": bond_feats,
            "bond_tuples": bond_tuples,
        }

    def get_pyg_graph(self, mol: Chem.rdchem.Mol, bigraph: bool = True) -> Data:
        """Convert molecule to PyTorch Geometric Data object with features.

        Args:
            mol: RDKit molecule object.
            bigraph: Whether to create bidirectional edges.

        Returns:
            PyTorch Geometric Data object with node and edge features.
        """
        mol_graph = self.get_mol_graph(mol, bigraph=bigraph)
        bond_inds = mol_graph["bond_tuples"]
        bond_feats = mol_graph["bond_feats"]
        atom_feats = mol_graph["atom_feats"]

        g = Data(
            x=th.from_numpy(atom_feats).float(),
            edge_index=th.from_numpy(bond_inds).long().transpose(1, 0),
            edge_attr=th.from_numpy(bond_feats).float(),
        )

        # Keep one entry per chemical bond alongside the possibly bidirectional
        # message-passing edges. PyG increments ``*_index`` attributes by the
        # atom offset when batching, so these endpoints remain valid globally.
        num_chemical_bonds = mol.GetNumBonds()
        g.chemical_bond_index = g.edge_index[:, :num_chemical_bonds].clone()
        g.chemical_bond_attr = g.edge_attr[:num_chemical_bonds].clone()
        g.atom_atomic_numbers = th.tensor(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()],
            dtype=th.long,
        )
        g.atom_hydrogen_counts = th.tensor(
            [atom.GetTotalNumHs() for atom in mol.GetAtoms()],
            dtype=th.long,
        )
        g.atom_formal_charges = th.tensor(
            [atom.GetFormalCharge() for atom in mol.GetAtoms()],
            dtype=th.long,
        )
        # Spare hydrogen capacity per atom, in the intact molecule: how many further
        # hydrogens the atom could carry if every one of its bonds were reduced to a single
        # bond. Mirrors FraGNNet's `ve_arr`/`sbond_arr` arithmetic in
        # `dag.builder.extract_mol_info` (`use_default_valence=False`), including its formal
        # charge correction, so both models bound hydrogen transfer the same way. Cutting a
        # bond frees one more slot per severed end, which the consumer adds per fragment.
        g.atom_free_valence_slots = th.tensor(
            [_atom_free_valence_slots(atom) for atom in mol.GetAtoms()],
            dtype=th.long,
        )
        # Always exposed as its own field, so a consumer can pool charge per fragment without
        # knowing the atom feature layout or requiring `a_charge` in the feature list. Measured
        # at 0.03 ms/molecule, 1.1 s for the whole NIST20 molecule set.
        if not self.needs_gasteiger:
            AllChem.ComputeGasteigerCharges(mol)
        g.atom_gasteiger_charges = th.tensor(
            [
                0.0 if math.isnan(v) or math.isinf(v) else v
                for v in (atom.GetDoubleProp("_GasteigerCharge") for atom in mol.GetAtoms())
            ],
            dtype=th.float32,
        )
        g.atom_monoisotopic_masses = th.tensor(
            [
                Chem.GetPeriodicTable().GetMostCommonIsotopeMass(atom.GetAtomicNum())
                for atom in mol.GetAtoms()
            ],
            dtype=th.float64,
        )

        if self.pe_embed_k > 0:
            pe_embeds = random_walk_pe(
                g,
                k=self.pe_embed_k,
            )
            if g.x is None:
                raise ValueError("Node features (g.x) cannot be None when pe_embed_k > 0")
            g.x = th.cat([g.x, pe_embeds], dim=-1)

        return g


class FeatBase:
    """Base class for atom and bond featurizers.

    Extend this class to implement custom atom and bond featurization functions.
    """

    feat_size = 0
    name = "base"

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom | Chem.rdchem.Bond) -> list[int]:
        """Featurize an atom or bond.

        Args:
            x: RDKit atom or bond object to featurize.

        Returns:
            List of integer feature values.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError()


@register_atom_feat
class AtomOneHot(FeatBase):
    """Atom element type features are one-hot encoded."""

    name = "a_onehot"
    # A copy: ``one_hot_encoding`` appends the unknown bucket to whatever list it is given, and
    # ``CANONICAL_ELEMENT_ORDER`` is shared with the formula code in ``dag.formula``.
    allowable_set = list(CANONICAL_ELEMENT_ORDER)
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom) -> list[int]:
        """One-hot encode atom element type.

        Args:
            x: RDKit atom object.

        Returns:
            One-hot encoded element type vector.
        """
        return one_hot_encoding(x.GetSymbol(), cls.allowable_set, True)


@register_atom_feat
class AtomDegree(FeatBase):
    """Atom degree as two raw values plus a heavy-atom-degree one-hot.

    The one-hot encodes ``GetDegree()``, the heavy-atom degree, because that is what bounds how
    many bonds an atom can lose. Total degree stays available as raw column 1, and as
    ``degree + a_num_hs`` wherever that feature is enabled.

    Note:
        Until 2026-09-20 ``featurize`` passed the pair ``[degree, total_degree]`` into
        ``one_hot_encoding``. A list never equals an ``int``, so all 12 one-hot columns were
        pinned to the unknown bucket and carried no information -- see §7.1 of
        ``docs/reference/masster_feature_set_audit.md``. The fix keeps the 14-column width,
        so a checkpoint trained before that date still loads but reads different inputs; such
        checkpoints must be retrained.
    """

    name = "a_degree"
    allowable_set = list(range(11))
    feat_size = len(allowable_set) + 1 + 2

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom) -> list[int]:
        """Encode atom degree as two raw values plus a heavy-atom-degree one-hot.

        Args:
            x: RDKit atom object.

        Returns:
            ``[degree, total_degree]`` followed by the one-hot of ``degree``.
        """
        degree = x.GetDegree()
        return [degree, x.GetTotalDegree()] + one_hot_encoding(degree, cls.allowable_set, True)


@register_atom_feat
class AtomHybrid(FeatBase):
    """Atom HybridizationType Charge features are one-hot encoded."""

    name = "a_hybrid"
    allowable_set = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom) -> list[int]:
        """One-hot encode atom hybridization type.

        Args:
            x: RDKit atom object.

        Returns:
            One-hot encoded hybridization type vector.
        """
        onehot = one_hot_encoding(x.GetHybridization(), cls.allowable_set, True)
        return onehot


@register_atom_feat
class AtomFormal(FeatBase):
    """Atom Formal Charge features are one-hot encoded."""

    name = "a_formal"
    allowable_set = list(range(-2, 3))
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom) -> list[int]:
        """One-hot encode atom formal charge.

        Args:
            x: RDKit atom object.

        Returns:
            One-hot encoded formal charge vector.
        """
        onehot = one_hot_encoding(x.GetFormalCharge(), cls.allowable_set, True)
        return onehot


@register_atom_feat
class AtomRadical(FeatBase):
    """AtomRadical features are one-hot encoded."""

    name = "a_radical"
    allowable_set = list(range(5))
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom) -> list[int]:
        """One-hot encode atom radical electron count.

        Args:
            x: RDKit atom object.

        Returns:
            One-hot encoded radical electron count vector.
        """
        onehot = one_hot_encoding(x.GetNumRadicalElectrons(), cls.allowable_set, True)
        return onehot


@register_atom_feat
class AtomRing(FeatBase):
    """Atom Ring and Atom Aromatic features are one-hot encoded.

    TODO: when used together with a_ring_size, the is_in_ring bits here are
    redundant (a_ring_size all-zero ↔ not in ring). The is_aromatic bits are
    NOT redundant (ring size does not imply aromaticity). Consider splitting into
    a standalone a_aromatic feature and dropping a_ring to eliminate the overlap.
    """

    name = "a_ring"
    allowable_set = [True, False]
    feat_size = len(allowable_set) * 2

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom) -> list[int]:
        """One-hot encode atom ring and aromaticity status.

        Args:
            x: RDKit atom object.

        Returns:
            List containing one-hot encoded ring status and aromaticity vectors.
        """
        onehot_ring = one_hot_encoding(x.IsInRing(), cls.allowable_set, False)
        onehot_aromatic = one_hot_encoding(x.GetIsAromatic(), cls.allowable_set, False)
        return onehot_ring + onehot_aromatic


@register_atom_feat
class AtomChiral(FeatBase):
    """Atom Chiral features are one-hot encoded. The last feature is for unspecified chirality."""

    name = "a_chiral"
    allowable_set = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ]
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom) -> list[int]:
        """One-hot encode atom chiral type.

        Args:
            x: RDKit atom object.

        Returns:
            One-hot encoded chiral type vector.
        """
        chiral_onehot = one_hot_encoding(x.GetChiralTag(), cls.allowable_set, True)
        return chiral_onehot


@register_atom_feat
class AtomMass(FeatBase):
    """Atom Mass features normalized by a coefficient."""

    name = "a_mass"
    coef = 0.01
    feat_size = 1

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom) -> list[float]:
        """Compute normalized atomic mass.

        Args:
            x: RDKit atom object.

        Returns:
            List containing normalized atomic mass value.
        """
        return [x.GetMass() * cls.coef]


# Pauling electronegativity (CRC Handbook) by element symbol. Noble gases He/Ne/Ar and Pm have no
# Pauling value and are absent; ``AtomPeriodic`` raises on them.
PAULING_ELECTRONEGATIVITY = {
    "H": 2.20, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
    "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19, "S": 2.58, "Cl": 3.16,
    "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83,
    "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01, "As": 2.18,
    "Se": 2.55, "Br": 2.96, "Kr": 3.00,
    "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.60, "Mo": 2.16, "Tc": 1.90,
    "Ru": 2.20, "Rh": 2.28, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96,
    "Sb": 2.05, "Te": 2.10, "I": 2.66, "Xe": 2.60,
    "Cs": 0.79, "Ba": 0.89, "La": 1.10, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14, "Sm": 1.17,
    "Eu": 1.20, "Gd": 1.20, "Tb": 1.20, "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25,
    "Yb": 1.10, "Lu": 1.27, "Hf": 1.30, "Ta": 1.50, "W": 2.36, "Re": 1.90, "Os": 2.20,
    "Ir": 2.20, "Pt": 2.28, "Au": 2.54, "Hg": 2.00, "Tl": 1.62, "Pb": 2.33, "Bi": 2.02,
    "Po": 2.00, "At": 2.20,
}  # fmt: skip
# Atomic number of the last element of each period 1..7.
_PERIOD_ENDS = (2, 10, 18, 36, 54, 86, 118)


def periodic_position(atomic_num: int) -> tuple[int, int]:
    """Return the (period, IUPAC group) of an element; lanthanides and actinides get group 3.

    Args:
        atomic_num: Atomic number, 1 to 118.

    Returns:
        Period 1-7 and group 1-18.

    Raises:
        ValueError: If ``atomic_num`` is outside 1-118.
    """
    if not 1 <= atomic_num <= _PERIOD_ENDS[-1]:
        raise ValueError(f"no periodic position for atomic number {atomic_num}")
    period = next(p for p, end in enumerate(_PERIOD_ENDS, start=1) if atomic_num <= end)
    pos = atomic_num - (_PERIOD_ENDS[period - 2] if period > 1 else 0) - 1
    if period == 1:
        return period, 1 if pos == 0 else 18
    if period <= 3:
        # s block, then p block from group 13.
        return period, pos + 1 if pos < 2 else pos + 11
    if period <= 5:
        return period, pos + 1
    # s block, 15 lanthanides/actinides in group 3, then groups 4-18.
    if pos < 2:
        return period, pos + 1
    return period, 3 if pos <= 16 else pos - 13


@register_atom_feat
class AtomPeriodic(FeatBase):
    """Continuous periodic-table properties, so a rare element can share what its group learned.

    Four values: period / 7, group / 18, Pauling electronegativity / 4 and valence (outer-shell)
    electrons / 8. Se then sits next to S, As next to P, and Br between Cl and I, where the
    element one-hot keeps every element orthogonal. A dummy atom (atomic number 0) gets zeros.
    """

    name = "a_periodic"
    feat_size = 4

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom) -> list[float]:
        """Compute the scaled periodic properties of an atom.

        Args:
            x: RDKit atom object.

        Returns:
            ``[period / 7, group / 18, electronegativity / 4, valence electrons / 8]``.

        Raises:
            ValueError: If the element has no Pauling electronegativity in the table.
        """
        atomic_num = x.GetAtomicNum()
        if atomic_num == 0:
            return [0.0] * cls.feat_size
        symbol = x.GetSymbol()
        if symbol not in PAULING_ELECTRONEGATIVITY:
            raise ValueError(f"a_periodic has no Pauling electronegativity for {symbol}")
        period, group = periodic_position(atomic_num)
        outer_electrons = Chem.GetPeriodicTable().GetNOuterElecs(atomic_num)
        return [
            period / 7,
            group / 18,
            PAULING_ELECTRONEGATIVITY[symbol] / 4,
            outer_electrons / 8,
        ]


@register_atom_feat
class AtomRingSize(FeatBase):
    """Ring size membership features for atoms.

    Five binary bits: IsInRingOfSize(3), IsInRingOfSize(4), IsInRingOfSize(5),
    IsInRingOfSize(6), and any ring of size ≥7. All zero for non-ring atoms.
    Complements a_ring (which only encodes is_in_ring and is_aromatic).
    """

    name = "a_ring_size"
    feat_size = 5

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom) -> list[int]:
        """Encode atom ring size membership.

        Args:
            x: RDKit atom object.

        Returns:
            Binary list of length 5: [in3, in4, in5, in6, inGe7].
        """
        ring_info = x.GetOwningMol().GetRingInfo()
        idx = x.GetIdx()
        in_small = [ring_info.IsAtomInRingOfSize(idx, s) for s in (3, 4, 5, 6)]
        in_ge7 = x.IsInRing() and not any(in_small)
        return [int(v) for v in in_small] + [int(in_ge7)]


@register_atom_feat
class AtomFusedRing(FeatBase):
    """Whether the atom belongs to two or more SSSR rings.

    A fused (ring-junction) atom cannot be separated from its neighbours by a single bond
    cleavage, so it behaves differently from an ordinary ring atom under fragmentation:
    breaking one ring bond at a junction opens a macrocycle instead of releasing a fragment.
    Neither ``a_ring`` nor ``a_ring_size`` encodes this -- both are satisfied by a single
    ring. Measured on 5,000 NIST20 molecules: 6.5% of atoms are fused and 40.3% of molecules
    contain at least one.
    """

    name = "a_fused_ring"
    feat_size = 2

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom) -> list[int]:
        """One-hot encode whether the atom sits at a ring fusion.

        Args:
            x: RDKit atom object.

        Returns:
            One-hot vector of length 2: ``[not fused, fused]``.
        """
        num_rings = x.GetOwningMol().GetRingInfo().NumAtomRings(x.GetIdx())
        return one_hot_encoding(num_rings >= 2, [False, True], False)


@register_atom_feat
class AtomCharge(FeatBase):
    """Gasteiger partial charge (continuous scalar).

    Requires AllChem.ComputeGasteigerCharges to have been called on the molecule
    before featurization. MolGraph handles this automatically when 'a_charge' is
    in the atom feature list.
    """

    name = "a_charge"
    feat_size = 1

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom) -> list[float]:
        """Return Gasteiger partial charge of the atom.

        Args:
            x: RDKit atom object.

        Returns:
            List containing the Gasteiger partial charge value, or 0.0 on failure.
        """
        v = x.GetDoubleProp("_GasteigerCharge")
        return [0.0 if math.isnan(v) or math.isinf(v) else v]


@register_atom_feat
class AtomNumHs(FeatBase):
    """Total hydrogen count (implicit + explicit) per atom, one-hot encoded.

    Encodes the mobile proton pool at each atom site, directly relevant to
    H-rearrangement fragmentation pathways (McLafferty, retro-Diels-Alder).
    Complements a_degree which only captures heavy-atom connectivity.
    """

    name = "a_num_hs"
    allowable_set = [0, 1, 2, 3]
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x: Chem.rdchem.Atom) -> list[int]:
        """One-hot encode total hydrogen count on the atom.

        Args:
            x: RDKit atom object.

        Returns:
            One-hot encoded H count vector of length 5: [0, 1, 2, 3, ≥4].
        """
        return one_hot_encoding(x.GetTotalNumHs(), cls.allowable_set, True)


@register_bond_feat
class BondDegree(FeatBase):
    """Bond degree/type features are one-hot encoded."""

    name = "b_degree"
    allowable_set = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x: Chem.rdchem.Bond) -> list[int]:
        """One-hot encode bond type.

        Args:
            x: RDKit bond object.

        Returns:
            One-hot encoded bond type vector.
        """
        return one_hot_encoding(x.GetBondType(), cls.allowable_set, True)


@register_bond_feat
class BondStereo(FeatBase):
    """Bond Stereo features are one-hot encoded. The last feature is for unspecified stereochemistry."""

    name = "b_stereo"
    allowable_set = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS,
    ]
    feat_size = len(allowable_set) + 1

    @classmethod
    def featurize(cls, x: Chem.rdchem.Bond) -> list[int]:
        """One-hot encode bond stereochemistry.

        Args:
            x: RDKit bond object.

        Returns:
            One-hot encoded bond stereochemistry vector.
        """
        return one_hot_encoding(x.GetStereo(), cls.allowable_set, True)


@register_bond_feat
class BondRing(FeatBase):
    """Bond Ring features are one-hot encoded. The last feature is for unspecified ring status."""

    name = "b_ring"
    feat_size = 2

    @classmethod
    def featurize(cls, x: Chem.rdchem.Bond) -> list[int]:
        """One-hot encode bond ring membership.

        Args:
            x: RDKit bond object.

        Returns:
            One-hot encoded bond ring status vector.
        """
        return one_hot_encoding(x.IsInRing(), [False, True], False)


@register_bond_feat
class BondConj(FeatBase):
    """Bond Conjugation features are one-hot encoded. The last feature is for unspecified conjugation."""

    name = "b_conj"
    feat_size = 2

    @classmethod
    def featurize(cls, x: Chem.rdchem.Bond) -> list[int]:
        """One-hot encode bond conjugation status.

        Args:
            x: RDKit bond object.

        Returns:
            One-hot encoded bond conjugation vector.
        """
        return one_hot_encoding(x.GetIsConjugated(), [False, True], False)


@register_bond_feat
class BondRingSize(FeatBase):
    """Ring size membership features for bonds.

    The bond analogue of ``a_ring_size``: five binary bits for membership in a ring of size
    3, 4, 5, 6, or >=7, all zero for acyclic bonds. Ring size governs cleavage propensity --
    a bond in a strained three- or four-membered ring breaks on a different footing than one
    in a benzene ring -- and ``a_ring_size`` only reaches it indirectly, through the two
    endpoints, which cannot distinguish a bond shared by two rings of different size.
    Measured on 5,000 NIST20 molecules: 50.3% of bonds are in a ring, distributed
    0.6 / 0.2 / 17.7 / 80.0 / 3.9% over [3, 4, 5, 6, >=7].
    """

    name = "b_ring_size"
    feat_size = 5

    @classmethod
    def featurize(cls, x: Chem.rdchem.Bond) -> list[int]:
        """Encode bond ring size membership.

        Args:
            x: RDKit bond object.

        Returns:
            Binary list of length 5: ``[in3, in4, in5, in6, inGe7]``.
        """
        ring_info = x.GetOwningMol().GetRingInfo()
        idx = x.GetIdx()
        in_small = [ring_info.IsBondInRingOfSize(idx, s) for s in (3, 4, 5, 6)]
        in_ge7 = x.IsInRing() and not any(in_small)
        return [int(v) for v in in_small] + [int(in_ge7)]


@register_bond_feat
class BondFusedRing(FeatBase):
    """Whether the bond belongs to two or more SSSR rings.

    The bond analogue of ``a_fused_ring``, and the sharper of the two for fragmentation:
    cleaving a bond shared by two rings cannot disconnect the molecule at all, so such bonds
    never yield a fragment on their own. Measured on 5,000 NIST20 molecules: 3.6% of bonds.
    """

    name = "b_fused_ring"
    feat_size = 2

    @classmethod
    def featurize(cls, x: Chem.rdchem.Bond) -> list[int]:
        """One-hot encode whether the bond is shared by two or more rings.

        Args:
            x: RDKit bond object.

        Returns:
            One-hot vector of length 2: ``[not fused, fused]``.
        """
        num_rings = x.GetOwningMol().GetRingInfo().NumBondRings(x.GetIdx())
        return one_hot_encoding(num_rings >= 2, [False, True], False)


def one_hot_encoding(x: Any, allowable_set: list[Any], encode_unknown: bool = False) -> list[int]:
    """One-hot encode ``x`` against an allowable set.

    Code adapted from the DGL LifeSci featurizers.

    Args:
        x: Value to encode.
        allowable_set: Enumerated set of allowable values.
        encode_unknown: If ``True``, append an unknown bucket and route missing values there.

    Returns:
        Binary indicator vector marking the matching allowable value.
    """

    if encode_unknown and (allowable_set[-1] is not None):
        allowable_set.append(None)

    if encode_unknown and (x not in allowable_set):
        x = None

    return [int(x == s) for s in allowable_set]


def get_mol_fp(
    mol: Chem.rdchem.Mol,
    morgan: bool,
    rdkit: bool,
    maccs: bool,
    morgan_radius: int = 3,
    morgan_nbits: int = 2048,
    rdkit_nbits: int = 2048,
) -> th.Tensor:
    """Compute concatenated molecular fingerprints as a float tensor.

    Args:
        mol: RDKit molecule object.
        morgan: Whether to include Morgan fingerprints.
        rdkit: Whether to include RDKit fingerprints.
        maccs: Whether to include MACCS fingerprints.
        morgan_radius: Radius for Morgan fingerprint (default: 3).
        morgan_nbits: Number of bits for Morgan fingerprint (default: 2048).
        rdkit_nbits: Number of bits for RDKit fingerprint (default: 2048).

    Returns:
        Float tensor containing concatenated fingerprints.

    Raises:
        AssertionError: If all fingerprint flags are False.
    """
    assert morgan or rdkit or maccs
    fps = []
    if morgan:
        fp = make_morgan_fingerprint(mol, radius=morgan_radius, nbits=morgan_nbits)
        fps.append(fp)
    if rdkit:
        fp = make_rdkit_fingerprint(mol, nbits=rdkit_nbits)
        fps.append(fp)
    if maccs:
        fp = make_maccs_fingerprint(mol)
        fps.append(fp)
    fp = th.as_tensor(np.concatenate(fps, axis=0), dtype=th.float)
    return fp


def get_mol_graph(
    mol: Chem.rdchem.Mol,
    atom_feats: Sequence[str],
    bond_feats: Sequence[str],
    pe_embed_k: int,
    bigraph: bool,
) -> Data:
    """Create a PyG molecule graph with selected features and optional RWPE.

    Args:
        mol: RDKit molecule object.
        atom_feats: Sequence of atom feature names to include.
        bond_feats: Sequence of bond feature names to include.
        pe_embed_k: Number of random walk positional encoding steps (0 to disable).
        bigraph: Whether to create bidirectional edges.

    Returns:
        PyTorch Geometric Data object representing the molecule graph.
    """

    mg = MolGraph(atom_feats, bond_feats, pe_embed_k)
    mol_pyg = mg.get_pyg_graph(mol, bigraph=bigraph)
    return mol_pyg


def random_walk_pe(g: Data, k: int) -> th.Tensor:
    """Random Walk Positional Encoding, as introduced in
    `Graph Neural Networks with Learnable Structural and Positional Representations
    <https://arxiv.org/abs/2110.07875>`__

    This function computes the random walk positional encodings as landing probabilities
    from 1-step to k-step, starting from each node to itself.

    Args:
        g: Homogeneous PyTorch Geometric Data object (graph).
        k: Number of random walk steps to include.

    Returns:
        Random walk positional encodings of shape (N, k) where N is the number of nodes.

    Example:
        >>> g = Data(edge_index=th.tensor([[0, 1, 1], [1, 1, 0]]), num_nodes=2)
        >>> random_walk_pe(g, 2)
        tensor([[0.0000, 0.5000],
                [0.5000, 0.7500]])
    """
    # sparse adjacency matrix
    A = scipy.sparse.csr_matrix(
        (
            th.ones_like(g.edge_index[0]).numpy(),
            (g.edge_index[0].numpy(), g.edge_index[1].numpy()),
        )
    )
    RW = A / (A.sum(1).reshape(-1, 1) + 1e-30)  # 1-step transition probability

    # Iterate for k steps
    PE = [RW.diagonal()]
    RW_power = RW.copy()
    for _ in range(k - 1):
        RW_power = RW_power @ RW
        PE.append(RW_power.diagonal())
    PE = np.stack(PE, axis=-1)
    PE = th.as_tensor(PE, dtype=th.float32)
    return PE


def make_morgan_fingerprint(
    mol: "Chem.rdchem.Mol",
    radius: int = 3,
    nbits: int = 2048,
    covert_to_np: bool = True,
) -> np.ndarray:
    """Make Morgan fingerprint from mol object.

    Args:
        mol: RDKit mol object
        radius: Fingerprint radius (default: 3). Must be > 0.
        nbits: Number of bits in fingerprint (default: 2048). Must be > 0.
        covert_to_np: Whether to convert to numpy array (default: True)

    Returns:
        Fingerprint as numpy array or RDKit fingerprint object

    Raises:
        ValueError: If radius <= 0 or nbits <= 0
    """
    if radius <= 0:
        raise ValueError(f"morgan_radius must be positive integer, got {radius}")
    if nbits <= 0:
        raise ValueError(f"morgan_nbits must be positive integer, got {nbits}")

    modules = rdkit_import("rdkit.Chem.AllChem", "rdkit.DataStructs")
    AllChem = modules[0]
    ds = modules[1]
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    if covert_to_np:
        fp_arr = np.zeros(1)
        ds.ConvertToNumpyArray(fp, fp_arr)
    else:
        fp_arr = fp
    return fp_arr


def make_rdkit_fingerprint(mol: "Chem.rdchem.Mol", nbits: int = 2048) -> np.array:
    """Make RDKit fingerprint from mol object.

    Args:
        mol: RDKit mol object
        nbits: Number of bits in fingerprint (default: 2048). Must be > 0.

    Returns:
        Fingerprint as numpy array

    Raises:
        ValueError: If nbits <= 0
    """
    if nbits <= 0:
        raise ValueError(f"rdkit_nbits must be positive integer, got {nbits}")

    chem, ds = rdkit_import("rdkit.Chem", "rdkit.DataStructs")
    fp = chem.RDKFingerprint(mol, fpSize=nbits)
    fp_arr = np.zeros(1)
    ds.ConvertToNumpyArray(fp, fp_arr)
    return fp_arr


def make_maccs_fingerprint(mol: "Chem.rdchem.Mol") -> np.ndarray:
    """make maccs fingerpirnt from mol object

    Args:
            mol (rdkit mol): mol object

    Returns:
            fp_arr:  np.array
    """

    maccs, ds = rdkit_import("rdkit.Chem.MACCSkeys", "rdkit.DataStructs")
    fp = maccs.GenMACCSKeys(mol)
    fp_arr = np.zeros(1)
    ds.ConvertToNumpyArray(fp, fp_arr)
    return fp_arr
