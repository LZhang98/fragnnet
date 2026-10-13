"""Hydrogen cap of a fragment (connected component).

FraGNNet's DAG builder uses it, and MASSter's valence-bound tests use it as their reference.
"""

import numpy as np


def compute_cc_h_cap(
    cc_atom_ids: np.ndarray, ve_arr: np.ndarray, sbond_arr: np.ndarray, num_radicals: int
) -> int:
    """compute max amount of Hs a cc can have.
        For any ccs the max amount of Hs it can have is the congifcation where all the bond are single
        And all the atom has max amount of Hs
    Args:
        cc (list|np.ndarray): cc mask in list form
        ve_arr (list|np.ndarray): max velance each atom can have
        sbond_arr (_type_): _description_
        num_radicals (_type_): _description_

    Returns:
        _type_: _description_
    """

    assert num_radicals == 0
    if not isinstance(cc_atom_ids, np.ndarray):
        cc_atom_ids = np.array(list(cc_atom_ids))
    cap_sbond_arr = sbond_arr[cc_atom_ids]
    cap_ve_arr = ve_arr[cc_atom_ids]
    cap_ve_mask = cap_ve_arr < cap_sbond_arr
    cap_ve_arr[cap_ve_mask] = cap_sbond_arr[cap_ve_mask]
    return np.sum(cap_ve_arr) - np.sum(cap_sbond_arr) - num_radicals
