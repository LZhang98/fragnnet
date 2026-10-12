"""Collision energy conversions: precursor charge from the precursor type, NCE <-> ACE, and
filling a missing NCE or ACE from the other (split out of ``utils/data_utils.py``).
"""

import warnings

import numpy as np
import pandas as pd

from fragnnet.utils.misc_utils import none_or_nan

CHARGE_FACTOR_MAP = {1: 1.00, 2: 0.90, 3: 0.85, 4: 0.80, 5: 0.75, "large": 0.75}


def get_charge(prec_type_str: str) -> int:
    """
    Parse the charge from the precursor type string.
    Args:
            prec_type_str (str): Precursor type string.
    Returns:
            int: Charge of the precursor.
    """
    if prec_type_str == "EI":
        return 1
    end_brac_idx = prec_type_str.index("]")
    charge_str = prec_type_str[end_brac_idx + 1 :]
    if charge_str == "-":
        charge_str = "1-"
    elif charge_str == "+":
        charge_str = "1+"
    assert len(charge_str) >= 2
    sign = charge_str[-1]
    assert sign in ["+", "-"], prec_type_str
    magnitude = int(charge_str[:-1])
    if sign == "+":
        charge = magnitude
    else:
        charge = -magnitude
    return charge


def nce_to_ace_helper(nce: float, charge: int, prec_mz: float) -> float:
    """Convert NCE to ACE.

    Args:
            nce (float): The normalized collision energy.
            charge (int): The charge state of the ion.
            prec_mz (float): The precursor mass-to-charge ratio.

    Returns:
            float: The calculated absolute collision energy.
    """
    if charge in CHARGE_FACTOR_MAP:
        charge_factor = CHARGE_FACTOR_MAP[charge]
    else:
        charge_factor = CHARGE_FACTOR_MAP["large"]
    ace = (nce * prec_mz * charge_factor) / 500.0
    return ace


def ace_to_nce_helper(ace: float, charge: int, prec_mz: float) -> float:
    """Convert ACE to NCE.

    Args:
            ace (float): The absolute collision energy.
            charge (int): The charge state of the ion.
            prec_mz (float): The precursor mass-to-charge ratio.

    Returns:
            float: The normalized collision energy.
    """
    if charge in CHARGE_FACTOR_MAP:
        charge_factor = CHARGE_FACTOR_MAP[charge]
    else:
        charge_factor = CHARGE_FACTOR_MAP["large"]
    nce = (ace * 500.0) / (prec_mz * charge_factor)

    return nce


def nce_to_ace(row: pd.Series, nce_colname: str) -> float:
    """convert nce to ace
    Args:
            row (pd.Series): row of the dataframe
            nce_colname (str): column name of nce
    Returns:
            float: ace
    """
    prec_mz = row["prec_mz"]
    nce = row[nce_colname]
    prec_type = row["prec_type"]
    charge = np.abs(get_charge(prec_type))
    ace = nce_to_ace_helper(nce, charge, prec_mz)
    return ace


def ace_to_nce(row: pd.Series, ace_colname: str) -> float:
    """convert ace to nce
    Args:
            row (pd.Series): row of the dataframe
            ace_colname (str): column name of ace
    Returns:
            float: nce
    """
    prec_mz = row["prec_mz"]
    ace = row[ace_colname]
    prec_type = row["prec_type"]
    charge = np.abs(get_charge(prec_type))
    nce = ace_to_nce_helper(ace, charge, prec_mz)
    return nce


def fill_missing_nce(row: pd.Series, ace_colname: str, nce_colname: str) -> float:
    """fill missing nce with ace
    Args:
            row (pd.Series): row of the dataframe
    Returns:
            float: nce
    """
    nce = row[nce_colname]
    ace = row[ace_colname]
    nce_missing = none_or_nan(nce) or nce == -1
    ace_missing = none_or_nan(ace) or ace == -1
    if nce_missing and not ace_missing:
        return ace_to_nce(row, ace_colname)
    elif not nce_missing:
        return nce
    else:
        warnings.warn(
            f"Warning: can not compute nce from ace. nce: {nce}, ace: {ace} "
            f"from columns {nce_colname}, {ace_colname}",
            stacklevel=2,
        )
        return np.nan


def fill_missing_ace(row: pd.Series, ace_colname: str, nce_colname: str) -> float:
    """fill missing ace with nce
    Args:
            row (pd.Series): row of the dataframe
    Returns:
            float: ace
    """
    nce = row[nce_colname]
    ace = row[ace_colname]
    ace_missing = none_or_nan(ace) or ace == -1
    nce_missing = none_or_nan(nce) or nce == -1
    if ace_missing and not nce_missing:
        return nce_to_ace(row, nce_colname)
    elif not ace_missing:
        return ace
    else:
        warnings.warn(
            f"Warning: can not compute ace from nce. ace: {ace}, nce: {nce} "
            f"from columns {ace_colname}, {nce_colname}",
            stacklevel=2,
        )
        return np.nan
