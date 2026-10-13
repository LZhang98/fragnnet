"""RDKit molecule helpers: parsing and standardizing SMILES, identifiers (InChI, InChIKey),
formula and simple properties, scaffolds, a molecule table's property columns (split out of
``utils/data_utils.py``), and molecule drawing (from ``utils/plot_utils.py``).

RDKit is imported lazily (``rdkit_import`` and function-level imports), so importing this
module does not load it.
"""

import importlib
import io
import warnings
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageOps

from fragnnet.utils.misc_utils import none_or_nan
from fragnnet.utils.parallel_utils import par_apply_series

if TYPE_CHECKING:
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import Mol


def rdkit_import(*module_strs: str) -> tuple:
    """until function for import rdkit modules

    Returns:
            tuple: tuple of loaded modules
    """
    RDLogger = importlib.import_module("rdkit.RDLogger")
    RDLogger.DisableLog("rdApp.*")
    modules = []
    for module_str in module_strs:
        modules.append(importlib.import_module(module_str))
    return tuple(modules)


def rdkit_ml_standardize(
    mol: "Chem.rdchem.Mol", canonicalize_tautomers: bool = True
) -> "Chem.rdchem.Mol":
    """Standardize a molecule for consistent representation for spectra machine learning tasks.
    Do not directly copy this to other chemical machine leanring code base, check list of operations
    you want to perform first.
    This function performs several important standardization steps:
    1. Cleans up the molecule (removing Hs, disconnecting metals, etc.)
    2. Extracts the parent fragment (largest/most relevant piece)
    3. Canonicalizes tautomers to a standard form (if enabled)
    Adapted from https://bitsilla.com/blog/2021/06/standardizing-a-molecule-using-rdkit/

    Args:
            mol (rdkit.Chem.rdchem.Mol): Input molecule object
            canonicalize_tautomers (bool, optional): Whether to canonicalize tautomers.
                    Set to False to preserve the input tautomeric form. Defaults to True.

    Returns:
            rdkit.Chem.rdchem.Mol: Standardized molecule
    """
    # Import necessary RDKit modules
    modules = rdkit_import(
        "rdkit.Chem", "rdkit.Chem.MolStandardize", "rdkit.Chem.MolStandardize.rdMolStandardize"
    )
    rdMolStandardize = modules[-1]

    # Step 1: Normalize and reionize — skip MetalDisconnector (would strip covalent C–metal bonds)
    mol = rdMolStandardize.Normalize(mol)
    phosphorus_oxide_normalizer = rdMolStandardize.NormalizerFromData(
        "Phosphorus oxide\t[P+:1]-[O-:2]>>[P+0:1]=[O+0:2]\n",
        rdMolStandardize.CleanupParameters(),
    )
    mol = phosphorus_oxide_normalizer.normalize(mol)
    mol = rdMolStandardize.Reionize(mol)

    # Step 2: Get the parent fragment
    # - If the molecule contains multiple fragments (e.g., salt forms)
    # - Identifies and returns only the "parent" fragment
    # - Removes counterions, solvents, and other small fragments
    # mol = rdMolStandardize.FragmentParent(mol)

    # Step 3: Neutralize the molecule
    # - Attempts to remove or neutralize charges where chemically reasonable
    # - Important for consistent representation across different sources
    # - Helps with similarity calculations and model predictions
    # note that no attempt is made at reionization at this step  nor at ionization at some pH (rdkit has no pKa caculator)
    # the main aim to to represent all molecules from different sources  in a (single) standard way, for use in ML, catalogue, etc.
    # NOTE: This is NOT a very safe operation for MS/MS prediction
    # uncharger = rdMolStandardize.Uncharger() # Create uncharger object (no convenience method available)
    # mol = uncharger.uncharge(mol)

    # Step 4: Canonicalize tautomers
    # - Converts between different tautomeric forms to ensure consistent representation
    # - Selects a single canonical tautomer when multiple forms are possible
    # - Critical for ensuring the same compound is always represented identically
    if canonicalize_tautomers:
        te = rdMolStandardize.TautomerEnumerator()
        mol = te.Canonicalize(mol)

    return mol


def mol_from_smiles(
    smiles: str, ml_standardize: bool = True, canonicalize_tautomers: bool = True
) -> "Chem.rdchem.Mol":
    """Convert a SMILES string to an RDKit molecule object

    This function creates an RDKit molecule from a SMILES string and optionally
    standardizes it using machine learning-specific standardization steps.

    Args:
            smiles (str): SMILES string representing the molecule
            ml_standardize (bool, optional): Whether to apply ML-specific standardization
                    that includes cleanup, parent fragment extraction, charge neutralization,
                    and tautomer canonicalization. Defaults to True.
            canonicalize_tautomers (bool, optional): Whether to canonicalize tautomers
                    during standardization. Only used when ml_standardize=True.
                    Set to False to preserve the input tautomeric form. Defaults to True.

    Returns:
            rdkit.Chem.rdchem.Mol: RDKit molecule object, or np.nan if conversion fails
    """
    modules = rdkit_import("rdkit.Chem")
    Chem = modules[0]
    try:
        mol = Chem.MolFromSmiles(smiles)
        if ml_standardize:
            mol = rdkit_ml_standardize(mol, canonicalize_tautomers=canonicalize_tautomers)
    except Exception:
        mol = np.nan
    if none_or_nan(mol):
        mol = np.nan
    return mol


def mol_to_smiles(
    mol: "Chem.rdchem.Mol",
    canonical: bool = True,
    isomericSmiles: bool = False,
    kekuleSmiles: bool = False,
) -> str:
    """get smiles string from give mol object

    Args:
            mol (rdkit.Chem.rdchem.Mol): mol object
            canonical (bool, optional):flag to get canonical smiles. Defaults to True.
            isomericSmiles (bool, optional): flag to get isomeric smiles. Defaults to False.
            kekuleSmiles (bool, optional): flag to get kekule version of smiles. Defaults to False.

    Returns:
            str: smile string, np.nan if fails
    """
    modules = rdkit_import("rdkit.Chem")
    Chem = modules[0]
    try:
        smiles = Chem.MolToSmiles(
            mol, canonical=canonical, isomericSmiles=isomericSmiles, kekuleSmiles=kekuleSmiles
        )
    except Exception:
        smiles = np.nan
    if none_or_nan(smiles):
        smiles = np.nan
    return smiles


def mol_to_formula(mol: "Chem.rdchem.Mol") -> str:
    """get formula string from give mol object

    Args:
            mol (rdkit.Chem.rdchem.Mol): mol object

    Returns:
            str: formula, np.nan if fails
    """
    modules = rdkit_import("rdkit.Chem.AllChem")
    AllChem = modules[0]
    try:
        formula = AllChem.CalcMolFormula(mol)
    except Exception:
        formula = np.nan
    return formula


def mol_to_inchikey(mol: "Chem.rdchem.Mol") -> str:
    """get inchikey string from give mol object

    Args:
            mol (rdkit.Chem.rdchem.Mol): mol object

    Returns:
            str: inchikey, np.nan if fails
    """
    modules = rdkit_import("rdkit.Chem.inchi")
    inchi = modules[0]
    try:
        inchikey = inchi.MolToInchiKey(mol)
    except Exception:
        inchikey = np.nan
    return inchikey


def mol_to_inchikey_s(mol: "Chem.rdchem.Mol") -> str:
    """get inchikey string from give mol object

    Args:
            mol (rdkit.Chem.rdchem.Mol): mol object

    Returns:
            str: inchikey, np.nan if fails
    """
    modules = rdkit_import("rdkit.Chem.inchi")
    inchi = modules[0]
    try:
        inchikey_s = inchi.MolToInchiKey(mol)[:14]
    except Exception:
        inchikey_s = np.nan
    return inchikey_s


def mol_to_inchi(mol: "Chem.rdchem.Mol") -> str:
    """get mol weight from mol object

    Args:
            mol (rdkit.Chem.rdchem.Mol): mol object

    Returns:
            str: inchi
    """
    modules = rdkit_import("rdkit.Chem.rdinchi")
    rdinchi = modules[0]
    try:
        inchi = rdinchi.MolToInchi(mol, options="-SNon")[0]
    except Exception:
        inchi = np.nan
    return inchi


def mol_to_mol_weight(mol: "Chem.rdchem.Mol", exact: bool = True) -> float:
    """get mol weight from mol object
    Args:
            mol (_type_): mol object
            exact (bool, optional): flag to get extact mass. Defaults to True.

    Returns:
            float: mw
    """
    modules = rdkit_import("rdkit.Chem.Descriptors")
    Desc = modules[0]
    try:
        if exact:
            mol_weight = float(Desc.ExactMolWt(mol))
        else:
            mol_weight = float(Desc.MolWt(mol))
    except Exception:
        mol_weight = np.nan
    return mol_weight


def mol_to_charge(mol: "Chem.rdchem.Mol") -> int:
    """get number of charges from mol object

    Returns:
            int: total number of charges, np.nan if fails
    """
    modules = rdkit_import("rdkit.Chem.rdmolops")
    rdmolops = modules[0]
    try:
        charge = rdmolops.GetFormalCharge(mol)
    except Exception:
        charge = np.nan
    return charge


def mol_to_num_atoms(mol: "Chem.rdchem.Mol", heavy: bool = True) -> int | float:
    """get total number of atoms in get mol object

    Args:
            mol (rdkit.Chem.rdchem.Mol): mol object
            heavy (bool, optional): count heavy atom only. Defaults to True.

    Returns:
            int: total number of atoms, np.nan if fails
    """
    try:
        num_atoms = mol.GetNumAtoms(onlyHeavy=heavy)
    except Exception:
        num_atoms = np.nan
    return num_atoms


def mol_to_num_bonds(mol: "Chem.rdchem.Mol", heavy: bool = True) -> int | float:
    """get total number of bondss in get mol object

    Args:
            mol (rdkit.Chem.rdchem.Mol): mol object
            heavy (bool, optional): count bond between heavy atom only. Defaults to True.

    Returns:
            int: total number of bonds, np.nan if fails
    """
    try:
        num_bonds = mol.GetNumBonds(onlyHeavy=heavy)
    except Exception:
        num_bonds = np.nan
    return num_bonds


def mol_to_num_radicals(mol: "Chem.rdchem.Mol") -> int | float:
    """get total number of radicals in the mol object

    Args:
            mol (rdkit.Chem.rdchem.Mol): mol object

    Returns:
            int: number of radicals, np.nan if fails
    """
    try:
        num_radicals = 0
        for atom in mol.GetAtoms():
            num_radicals += atom.GetNumRadicalElectrons()
    except Exception:
        num_radicals = np.nan
    return num_radicals


def check_single_mol(mol: "Chem.rdchem.Mol") -> bool:
    """check if mol object is a single mol (connected graph)
    Args:
            mol (rdkit.Chem.rdchem.Mol): mol object

    Returns:
            bool: True if is single mol, else false
    """
    modules = rdkit_import("rdkit.Chem.rdmolops")
    rdmolops = modules[0]
    try:
        num_frags = len(rdmolops.GetMolFrags(mol))
    except Exception:
        num_frags = np.nan
    valid = num_frags == 1
    return valid


def get_murcko_scaffold(
    mol: "Chem.rdchem.Mol", output_type: str = "smiles", include_chirality: bool = False
) -> str | float:
    """get murcko scaffold from mol object
    Args:
            mol (rdkit.Chem.rdchem.Mol): mol object
            output_type (str, optional): output type. Defaults to "smiles".
            include_chirality (bool, optional): include chirality. Defaults to False.
    Returns:
            str: murcko scaffold
    """

    if none_or_nan(mol):
        return np.nan
    MurckoScaffold = importlib.import_module("rdkit.Chem.Scaffolds.MurckoScaffold")
    if output_type == "smiles":
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=include_chirality
            )
        except Exception as e:
            warnings.warn(f"failed to get murcko scaffold: {e}", stacklevel=2)
            scaffold = np.nan
    else:
        raise NotImplementedError
    return scaffold


def compute_mol_df_props(
    mol_df: "pd.DataFrame",
    scaffold: bool = True,
    inchi: bool = True,
    use_tqdm: bool = True,
) -> "pd.DataFrame":
    """Compute standard molecular properties and add them as columns to ``mol_df``.

    Requires a ``mol`` column containing RDKit mol objects. All new columns are
    written in-place via ``loc`` and the modified dataframe is also returned.

    The following columns are always added:
    ``inchikey_s``, ``formula``, ``mw``, ``exact_mw``, ``num_atoms``,
    ``num_bonds``, ``charge``, ``single_mol``, ``num_radicals``.

    Args:
        mol_df: DataFrame with a ``mol`` column of RDKit mol objects.
        scaffold: If ``True`` (default), also compute the ``scaffold`` column
            (Murcko scaffold SMILES).
        inchi: If ``True`` (default), also compute the ``inchi`` column.
        use_tqdm: Passed through to :func:`par_apply_series`; set ``False`` to
            suppress progress bars (e.g. in tight loops).

    Returns:
        The same ``mol_df`` with new property columns added in-place.
    """
    mol_col = mol_df["mol"]
    mol_df.loc[:, "inchikey_s"] = par_apply_series(mol_col, mol_to_inchikey_s, use_tqdm=use_tqdm)
    if scaffold:
        mol_df.loc[:, "scaffold"] = par_apply_series(
            mol_col, get_murcko_scaffold, use_tqdm=use_tqdm
        )
    mol_df.loc[:, "formula"] = par_apply_series(mol_col, mol_to_formula, use_tqdm=use_tqdm)
    if inchi:
        mol_df.loc[:, "inchi"] = par_apply_series(mol_col, mol_to_inchi, use_tqdm=use_tqdm)
    mol_df.loc[:, "mw"] = par_apply_series(
        mol_col, lambda mol: mol_to_mol_weight(mol, exact=False), use_tqdm=use_tqdm
    )
    mol_df.loc[:, "exact_mw"] = par_apply_series(
        mol_col, lambda mol: mol_to_mol_weight(mol, exact=True), use_tqdm=use_tqdm
    )
    mol_df.loc[:, "num_atoms"] = par_apply_series(mol_col, mol_to_num_atoms, use_tqdm=use_tqdm)
    mol_df.loc[:, "num_bonds"] = par_apply_series(mol_col, mol_to_num_bonds, use_tqdm=use_tqdm)
    mol_df.loc[:, "charge"] = par_apply_series(mol_col, mol_to_charge, use_tqdm=use_tqdm)
    mol_df.loc[:, "single_mol"] = par_apply_series(mol_col, check_single_mol, use_tqdm=use_tqdm)
    mol_df.loc[:, "num_radicals"] = par_apply_series(
        mol_col, mol_to_num_radicals, use_tqdm=use_tqdm
    )
    return mol_df


def trim_img_by_white(img: Image.Image, padding: int = 0) -> Image.Image:
    """Crop a PIL image to the minimum rectangle based on whiteness/transparency.

    This function removes white and transparent pixels from the edges of an image,
    with automatic 5-pixel padding applied. Adapted from:
    https://github.com/connorcoley/retrosim/blob/master/retrosim/utils/draw.py

    Args:
        img: PIL Image object to crop.
        padding: Border padding in pixels to add after cropping. Defaults to 0.

    Returns:
        Cropped PIL Image object with optional padding.
    """

    # Convert to array
    as_array = np.array(img)  # N x N x (r,g,b,a)
    assert as_array.ndim == 3 and as_array.shape[2] == 3, as_array.shape
    # Content defined as non-white and non-transparent pixel
    has_content = np.sum(as_array, axis=2, dtype=np.uint32) != 255 * 3
    xs, ys = np.nonzero(has_content)
    # Crop down
    x_range = max([min(xs) - 5, 0]), min([max(xs) + 5, as_array.shape[0]])
    y_range = max([min(ys) - 5, 0]), min([max(ys) + 5, as_array.shape[1]])
    as_array_cropped = as_array[x_range[0] : x_range[1], y_range[0] : y_range[1], 0:3]
    img = Image.fromarray(as_array_cropped, mode="RGB")
    return ImageOps.expand(img, border=padding, fill=(255, 255, 255, 0))


def get_mol_im(smiles: str, width: int = 1000, height: int = 1000) -> Image.Image:
    """Generate a PIL image of a molecule from its SMILES string.

    Attempts to use cairosvg for high-quality SVG rendering, with fallback to
    RDKit's Cairo drawer if cairosvg is not available.

    Args:
        smiles: SMILES string representation of the molecule.
        width: Width of the output image in pixels. Defaults to 1000.
        height: Height of the output image in pixels. Defaults to 1000.

    Returns:
        PIL Image object containing the rendered molecule structure.
    """
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = mol_from_smiles(smiles)

    try:
        import cairosvg

        d = rdMolDraw2D.MolDraw2DSVG(width, height)
        d.DrawMolecule(mol)
        d.FinishDrawing()
        svg_buf = d.GetDrawingText()
        png_buf = cairosvg.svg2png(svg_buf)
        im = Image.open(io.BytesIO(png_buf))
    except ImportError:
        # Fallback to PNG drawer if cairosvg is not available
        d = rdMolDraw2D.MolDraw2DCairo(width, height)
        d.DrawMolecule(mol)
        d.FinishDrawing()
        png_buf = d.GetDrawingText()
        im = Image.open(io.BytesIO(png_buf))

    im = trim_img_by_white(im, padding=15)
    return im


def get_highlighted_bonds(mol: "Mol", node_atom_indices: list[int] | set) -> list[int]:
    """Extract bonds connecting atoms within a specified set of atom indices.

    Args:
        mol: RDKit Mol object.
        node_atom_indices: Set or list of atom indices to include.

    Returns:
        List of bond indices connecting atoms in the specified set.
    """
    atoms = []
    for a in mol.GetAtoms():
        atoms.append(a.GetIdx())
    bonds = []
    for bond in mol.GetBonds():
        aid1 = atoms[bond.GetBeginAtomIdx()]
        aid2 = atoms[bond.GetEndAtomIdx()]
        if aid1 in node_atom_indices and aid2 in node_atom_indices:
            bonds.append(mol.GetBondBetweenAtoms(aid1, aid2).GetIdx())
    return bonds
