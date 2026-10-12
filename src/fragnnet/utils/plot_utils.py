"""Plotting utilities for FraGNNet visualization.

This module provides functions for creating visualizations including:
- Converting matplotlib figures to PIL images
- Plotting histograms
- Image trimming and processing
- Molecular structure visualization
- Spectrum comparison plots
- Dual-axis plots for metric visualization
- Spectrum binning and matching
"""

import io

import numpy as np
from PIL import Image, ImageOps
from rdkit.Chem import Mol
from rdkit.Chem.Draw import rdMolDraw2D

from fragnnet.utils.data_utils import mol_from_smiles


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


def get_highlighted_bonds(mol: Mol, node_atom_indices: list[int] | set) -> list[int]:
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
