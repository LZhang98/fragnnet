"""Formulas of DAG fragments: atom indices of a connected-component bitmask and the
fragment formula arrays.
"""

import numpy as np

import fragnnet.models.fragnnet.dag.compute_frags as compute_frags
from fragnnet.fragmentation.hydrogen_cap import compute_cc_h_cap
from fragnnet.models.fragnnet.dag.compute_frags import MASK_DTYPE
from fragnnet.utils.formula_utils import ELEMENT_TO_IDX, formula_arr_to_str


def cc_bit_mask_to_atom_idx(cc: list | np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        cc (list | np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    cc_np = np.array(cc) if isinstance(cc, list) else cc
    atom_ids = np.where(cc_np == 1)[0].astype(np.int32)
    return atom_ids


def compute_approximate_formula(
    cc: list | np.ndarray,
    mol_d: dict,
    max_h_transfer: int,
    formula_strs: bool = False,
    bitmask: bool = True,
    base_formula: np.ndarray | None = None,
) -> tuple[dict[int, str] | dict[int, np.ndarray], dict[int, int]]:
    """
    Given a connected component (fragment) of a molecule, compute all possible formulas
    within a specified hydrogen transfer range (delta H), and return either formula strings
    or hydrogen counts for each possible transfer.

    Args:
        cc (list | np.ndarray): Connected component, either as a bitmask or atom indices.
        mol_d (dict): Molecule dictionary containing atom/bond info and arrays.
        max_h_transfer (int): Maximum allowed hydrogen transfer (delta H) for enumeration.
        formula_strs (bool, optional): If True, return formula strings; if False, return formula arrays. Defaults to False.
        bitmask (bool, optional): If True, interpret cc as bitmask; if False, as atom indices. Defaults to True.
        base_formula (np.ndarray, optional): Precomputed base formula array for the fragment. If None, computed internally. Defaults to None.

    Returns:
        tuple[dict[int, str] | dict[int, np.ndarray], dict[int, int]]:
            - delta_h_to_formula: Maps delta H (hydrogen transfer) to formula string (if formula_strs=True)
              or formula array (if formula_strs=False) for each possible configuration.
            - delta_h_to_h_count: Maps delta H to the corresponding hydrogen count, or -1 for invalid configurations.
    """

    # Extract molecule arrays and info
    bonds = mol_d["bonds"]  # Bond definitions (atom id pairs)
    atoms_to_bonds = mol_d["atoms_to_bonds"]  # Atom to bond mapping
    ve_arr = np.copy(mol_d["ve_arr"])
    sbond_arr = np.copy(mol_d["sbond_arr"])
    num_hs = mol_d["num_hs"]  # Total number of hydrogens in molecule
    hs_arr = mol_d["hs_arr"]  # Hydrogens per atom
    elem_idxs = mol_d["elem_idxs"]  # Element indices per atom
    bond_mask_arr = mol_d["bond_mask_arr"]

    # Compute base formula for fragment if not provided
    if base_formula is None and bitmask:
        base_formula = cc_bitmask_to_formula_arr(cc, elem_idxs)
    elif base_formula is None and not bitmask:
        base_formula = cc_to_formula_arr(cc, elem_idxs, bitmask=False)

    # Get atom indices for fragment
    atom_ids = cc_bit_mask_to_atom_idx(cc) if bitmask else cc

    # Update single bond and bond mask arrays for fragment
    sbond_arr, bond_mask_arr = compute_frags.update_bonds(
        atom_ids, sbond_arr, bond_mask_arr, bonds, atoms_to_bonds
    )
    # Compute hydrogen cap and floor for fragment
    cap = compute_cc_h_cap(atom_ids, ve_arr, sbond_arr, 0)
    floor = compute_frags.compute_cc_h_floor(
        atom_ids, ve_arr, sbond_arr, 0, bonds, atoms_to_bonds, bond_mask_arr
    )

    # Defensive checks on hydrogen bounds
    assert floor <= cap, (floor, cap, atom_ids)
    assert floor >= 0, floor

    # Count hydrogens present in fragment before transfer
    num_hs_prior = sum(hs_arr[atom] for atom in atom_ids)

    # Restrict hydrogen range to allowed transfer window
    floor = max(floor, num_hs_prior - max_h_transfer)
    cap = min(cap, num_hs_prior + max_h_transfer, num_hs)
    # Guard: the actual H count in the molecule is always a valid configuration.
    # compute_cc_h_floor can overcount for fragments containing nitro groups because
    # update_bonds treats all bonds as single bonds — the double bond O= in [N+](=O)[O-]
    # appears to have 1 free valence slot (ve=2, sbond=1), but its only neighbor N+ has
    # diff=0 (after formal-charge correction), so the floor can't be reduced and ends up
    # > num_hs_prior, making delta_h=0 invalid and assigning "" to the node formula.
    floor = min(floor, num_hs_prior)

    delta_h_to_formula = {}  # Maps delta H to formula string/array
    delta_h_to_h_count = {}  # Maps delta H to hydrogen count (or -1 if invalid)

    # Prepare formula string templates if requested
    if formula_strs:
        formula_template, formual_no_h = formula_arr_to_str(base_formula, get_h_template=True)

    # Enumerate all possible hydrogen transfers
    for delta_h in range(-max_h_transfer, max_h_transfer + 1):
        h = num_hs_prior + delta_h  # Proposed hydrogen count for fragment
        if h < floor or h > cap:
            # Invalid configuration: assign zero formula and -1 count
            formula = np.zeros_like(base_formula)
            formula_str = ""
            delta_h_to_h_count[delta_h] = -1
        else:
            # Valid configuration: update formula with new hydrogen count
            formula = np.copy(base_formula)
            formula[ELEMENT_TO_IDX["H"]] = h
            formula = tuple(formula)
            if h == 0:
                formula_str = formual_no_h
            else:
                formula_str = formula_template.format(h)
            delta_h_to_h_count[delta_h] = h
        # Store formula string if requested, else array/tuple
        if formula_strs:
            formula = formula_str
        delta_h_to_formula[delta_h] = formula
    return delta_h_to_formula, delta_h_to_h_count


def cc_to_formula_arr(cc: list | np.ndarray, elems: list, bitmask: bool = False) -> np.ndarray:
    """
    Convert a fragment (connected component) to a formula array (element counts, excluding hydrogens).

    Args:
        cc (list | np.ndarray): Fragment as bitmask or atom indices.
        elems (list): List of element symbols for each atom in molecule.
        bitmask (bool, optional): If True, interpret cc as bitmask; else as atom indices. Defaults to False.

    Returns:
        np.ndarray: Array of element counts (excluding hydrogens).
    """
    formula_arr = np.zeros([len(ELEMENT_TO_IDX)], dtype=int)
    if bitmask:
        for i in range(len(cc)):
            if cc[i]:
                elem = elems[i]
                formula_arr[ELEMENT_TO_IDX[elem]] += 1
    else:
        for atom in cc:
            elem = elems[atom]
            formula_arr[ELEMENT_TO_IDX[elem]] += 1
    return formula_arr


def cc_bitmask_to_formula_arr(cc: np.ndarray, elem_idxs: np.ndarray) -> np.ndarray:
    """
    Fast conversion of a fragment bitmask to a formula array (element counts, excluding hydrogens).
    Uses element indices for efficient counting.

    Args:
        cc (list | np.ndarray): Fragment bitmask (0/1 per atom).
        elem_idxs (list | np.ndarray): Element indices for each atom in molecule.

    Returns:
        np.ndarray: Array of element counts (excluding hydrogens).
    """
    formula_arr = np.zeros([len(ELEMENT_TO_IDX)], dtype=MASK_DTYPE)
    elem_idxs_np = np.array(elem_idxs) + 1  # Offset by 1 for masking
    elem_idx_np = np.multiply(cc, elem_idxs_np)
    unique, counts = np.unique(elem_idx_np, return_counts=True)
    for elem_idx, count in zip(unique, counts):
        if elem_idx == 0:
            continue
        else:
            formula_arr[elem_idx - 1] = count
    return formula_arr
