import importlib
import re

import numpy as np
import pandas as pd
import pyteomics
from pyteomics.mass import Composition

from fragnnet.utils.formula_utils import (
    ELECTRON_MASS,
    MASS,
    composition_to_mass,
    composition_to_string,
)
from fragnnet.utils.migrations import RETIRED_INST_TYPE_HELP
from fragnnet.utils.misc_utils import none_or_nan


def parse_ace_str(ce_str: str) -> float:
    if none_or_nan(ce_str):
        return np.nan
    matches = {
        # nist ones
        # this case is ambiguous (float(x) >= 2. or float(x) == 0.)
        r"^[\d]+[.]?[\d]*$": lambda x: float(x),
        r"^[\d]+[.]?[\d]*[ ]?eV$": lambda x: float(x.rstrip(" eV")),
        r"^NCE=[\d]+[.]?[\d]*% [\d]+[.]?[\d]*eV$": lambda x: float(x.split()[1].rstrip("eV")),
        # other ones
        r"^[\d]+[.]?[\d]*HCD$": lambda x: float(x.rstrip("HCD")),
        r"^CE [\d]+[.]?[\d]*$": lambda x: float(x.lstrip("CE ")),
    }
    for k, v in matches.items():
        # try:
        if re.match(k, ce_str):
            return v(ce_str)
        # except:
        #   import pdb; pdb.set_trace()
    return np.nan


def parse_nce_str(ce_str: str) -> float:
    if none_or_nan(ce_str):
        return np.nan
    matches = {
        # nist ones
        r"^NCE=[\d]+[.]?[\d]*% [\d]+[.]?[\d]*eV$": lambda x: float(
            x.split()[0].lstrip("NCE=").rstrip("%")
        ),
        r"^NCE=[\d]+[.]?[\d]*%$": lambda x: float(x.lstrip("NCE=").rstrip("%")),
        r"^[\d]+% resonant relative/normalized$": lambda x: float(x.split("%")[0]),
        # other ones
        # r"^[\d]+[.]?[\d]*$": lambda x: 100.*float(x) if float(x) < 2. else np.nan, # this case is ambiguous
        r"^[\d]+[.]?[\d]*[ ]?[%]? \(nominal\)$": lambda x: float(
            x.removesuffix(" (nominal)").rstrip(" %")
        ),
        r"^HCD [\d]+[.]?[\d]*%$": lambda x: float(x.lstrip("HCD ").rstrip("%")),
        r"^[\d]+[.]?[\d]* NCE$": lambda x: float(x.rstrip("NCE")),
        r"^[\d]+[.]?[\d]*\(NCE\)$": lambda x: float(x.rstrip("(NCE)")),
        r"^[\d]+[.]?[\d]*[ ]?%$": lambda x: float(x.rstrip(" %")),
        r"^HCD \(NCE [\d]+[.]?[\d]*%\)$": lambda x: float(
            x.removeprefix("HCD (NCE ").removesuffix("%)")
        ),
    }
    for k, v in matches.items():
        if re.match(k, ce_str):
            return v(ce_str)
    return np.nan


def parse_inst_info(df: pd.Series) -> tuple[str, str]:
    # Orbitraps are labelled by instrument model: the Orbitrap Fusion family (NIST's "Orbitrap
    # Fusion Lumos") is "ORB_F"; every other Orbitrap, named (Elite, Velos) or not, is "ORB". The
    # two fragment differently at the same NCE, see
    # docs/findings/nist_orbitrap_instrument_ce_scales_2026_09_28.md. "FT" is retired.

    inst_type_str = df["inst_type"]
    if inst_type_str in ("OB", "OB_F"):
        raise ValueError(f"inst_type {inst_type_str!r} is retired. {RETIRED_INST_TYPE_HELP}")
    inst_str = df.get("inst", np.nan)
    frag_mode_str = df.get("frag_mode", np.nan)
    col_energy_str = df.get("col_energy", np.nan)
    # instrument type
    if inst_type_str == "EI":
        assert inst_str == "EI"
        assert frag_mode_str == "EI"
        assert col_energy_str == "100"
        return "EI", "EI"
    if none_or_nan(inst_type_str):
        # resort to instrument
        inst_map = {"Maxis II HD Q-TOF Bruker": "QTOF", "qToF": "QTOF", "Orbitrap": "ORB"}
        if none_or_nan(inst_str):
            inst_type = np.nan
        elif inst_str in inst_map:
            inst_type = inst_map[inst_str]
        else:
            inst_type = "Other"
    else:
        inst_type_map = {
            "QTOF": "QTOF",
            "FT": "ORB",
            "ORB": "ORB",
            "ORB_F": "ORB_F",
            "Q-TOF": "QTOF",
            "HCD": "ORB",
            "QqQ": "QQQ",
            "QqQ/triple quadrupole": "QQQ",
            "IT/ion trap": "IT",
            "IT-FT/ion trap with FTMS": "ORB",
            "Q-ToF (LCMS)": "QTOF",
            "Bruker Q-ToF (LCMS)": "QTOF",
            "ESI-QTOF": "QTOF",
            "ESI-QFT": "ORB",
            "ESI-ITFT": "ORB",
            "Linear Ion Trap": "IT",
            "LC-ESI-QTOF": "QTOF",
            "LC-ESI-QFT": "ORB",
            "LC-ESI-QQQ": "QQQ",
            "LC-Q-TOF/MS": "QTOF",
            "LC-ESI-ITFT": "ORB",
            "LC-ESI-IT": "IT",
            "LC-QTOF": "QTOF",
            "qToF": "QTOF",
            # NPLLIB stuff
            # "Q-ToF (LCMS)": "QTOF",
            # "Bruker Q-ToF (LCMS)": "QTOF",
            "Unknown (LCMS)": "Other",
            "Ion Trap (LCMS)": "IT",
            "Orbitrap (LCMS)": "ORB",
            "FTICR (LCMS)": "Other",
            "ITFT": "ORB",
            "Orbitrap": "ORB",
            "QFT": "ORB",
            "orbitrap": "ORB",
            "qtof": "QTOF",
        }
        inst_type = inst_type_map.get(inst_type_str, "Other")
    if inst_type == "ORB" and isinstance(inst_str, str) and "Fusion" in inst_str:
        inst_type = "ORB_F"
    # fragmentation mode
    if inst_type_str == "HCD" or (isinstance(col_energy_str, str) and "HCD" in col_energy_str):
        frag_mode = "HCD"
    elif none_or_nan(frag_mode_str) or frag_mode_str == "CID":
        # default is CID if we don't know
        frag_mode = "CID"
    elif frag_mode_str == "HCD":
        frag_mode = "HCD"
    elif frag_mode_str == "EAD":
        # electron-activated dissociation (SCIEX ZenoTOF); its ace is the electron kinetic energy
        frag_mode = "EAD"
    else:
        frag_mode = np.nan
    return inst_type, frag_mode


def parse_ion_mode_str(ion_mode_str: str) -> str | float:
    if none_or_nan(ion_mode_str):
        return np.nan
    if ion_mode_str in ["P", "N", "E", "EI"]:
        return ion_mode_str
    elif ion_mode_str == "POSITIVE":
        return "P"
    elif ion_mode_str == "NEGATIVE":
        return "N"
    else:
        return np.nan


def parse_ri_str(ri_str: str) -> float:
    if none_or_nan(ri_str):
        return np.nan
    else:
        return float(ri_str)


def parse_prec_type_str(prec_type_str: str) -> str | float:
    if none_or_nan(prec_type_str):
        return np.nan
    if prec_type_str == "EI":
        return "EI"
    if prec_type_str.endswith("1+"):
        prec_type_str = prec_type_str.replace("1+", "+")
    elif prec_type_str.endswith("1-"):
        prec_type_str = prec_type_str.replace("1-", "-")
    # manually replace some
    prec_type_dict = {
        # "M+H": "[M+H]+",
        # "M-H": "[M-H]-",
        "[M+FA-H]-": "[M+CH2O2-H]-"
    }
    if prec_type_str in prec_type_dict:
        prec_type_str = prec_type_dict[prec_type_str]
    return prec_type_str


def infer_prec_mz(df: pd.Series) -> float:
    # these were observed to be missing prec_mz in MoNA
    prec_type_to_mass_diff = {
        "[M+HCOO]-": MASS("H") + MASS("C") + 2 * MASS("O") + ELECTRON_MASS,
        "[M]+": -ELECTRON_MASS,
        "[M+HOO]-": MASS("H") + 2 * MASS("O") + ELECTRON_MASS,
        "[M-OH]+": -MASS("H") - MASS("O") - ELECTRON_MASS,
        "[M+H-C12H20O9]+": -19 * MASS("H") - 12 * MASS("C") - 9 * MASS("O") - ELECTRON_MASS,
        "[M-H]-": -MASS("H") + ELECTRON_MASS,
        "[M+H]+": MASS("H") - ELECTRON_MASS,
        "[M+Na]+": MASS("Na") - ELECTRON_MASS,
    }

    prec_mz = df["prec_mz"]
    spec_type = df["spec_type"]
    if none_or_nan(prec_mz) and spec_type == "MS2":
        dset = df["dset"]
        assert "nist" not in dset, dset
        prec_type = df["prec_type"]
        exact_mw = df["exact_mw"]
        if not none_or_nan(prec_type):
            prec_mz = exact_mw + prec_type_to_mass_diff[prec_type]
    return prec_mz


def parse_peaks_str(peaks_str: str) -> list | float:
    # peaks still represented as string
    if none_or_nan(peaks_str):
        return np.nan
    lines = peaks_str.split("\n")
    peaks = []
    for line in lines:
        if len(line) == 0:
            continue
        line = line.split(" ")
        mz = line[0]
        ints = line[1]
        peaks.append((mz, ints))
    return peaks


def parse_product_isotope(annot: str, precursor: str) -> tuple:
    formula = Composition(formula="")
    precursor = Composition(formula=precursor)
    adducts = []
    losses = []
    isotope = 0
    charge = 1
    chunks = [""] + re.split(r"([\+-])", annot)
    for prefix, chunk in zip(chunks[::2], chunks[1::2]):
        if len(chunk) == 0:
            continue
        elif prefix == "^":
            charge = int(chunk)
        else:
            if chunk[-1] == "i" and chunk[-2:] != "Si":
                chunk = chunk[:-1]
                isotope = int(chunk) if chunk else 1
                if prefix == "-":
                    isotope = -isotope
            else:
                if chunk[0].isdigit():
                    n = int(chunk[0])
                    chunk = chunk[1:]
                else:
                    n = 1
                if chunk == "p":
                    chunk = precursor
                else:
                    chunk = Composition(formula=chunk) * n
                if prefix == "":
                    formula += chunk
                elif prefix == "+":
                    adducts.append(chunk)
                    formula += chunk
                elif prefix == "-":
                    losses.append(chunk)
                    formula -= chunk
    return formula, adducts, losses, isotope, charge


def is_pep_annot(annot: str) -> bool:
    pattern = r"(?:a|b|y)\d+|pi"
    return re.match(pattern, annot) is not None


def parse_annotations(row: pd.Series) -> tuple:
    RDLogger = importlib.import_module("rdkit.RDLogger")
    RDLogger.DisableLog("rdApp.*")
    peaks_str = row["peaks"]
    prec_type = row["prec_type"]
    dset = row["dset"]
    formula = row["formula"]
    notes = row["notes"]
    error = None
    current_supported_dsets = ["nist20_hr"]
    if dset not in current_supported_dsets:
        error = ValueError(
            f"dataset not supported, supported {current_supported_dsets}, given {dset}"
        )
    elif "glycan" in notes.lower():
        error = ValueError("glycans not supported (1)")
    elif "peptide" in notes.lower():
        error = ValueError("peptide not supported (1)")
    elif prec_type not in ["[M+H]+", "[M-H]-"]:
        error = ValueError("prec_type not supported")
    elif none_or_nan(peaks_str):
        error = ValueError("peaks_str is invalid")
    if error is not None:
        error = repr(error)
        return np.nan, np.nan, np.nan, np.nan, np.nan, error
    # `formula=` for the same reason as in `combine_formulae`: this is a molecular formula
    # in Hill notation, which drops the "1" on count-one elements, so an all-single-atom
    # formula like "CHN" or "CO" is a valid peptide sequence and parses as one positionally.
    precursor = Composition(formula=formula)
    if prec_type == "[M+H]+":
        precursor["H"] += 1
    elif prec_type == "[M-H]-":
        precursor["H"] -= 1
    precursor = composition_to_string(precursor)
    lines = peaks_str.split("\n")
    peak_mzs, products, losses, isotopes, exact_mzs = [], [], [], [], []
    all_good = True
    for line in lines:
        if not all_good:
            break
        if len(line) == 0:
            continue
        line = line.split(" ")
        if len(line) < 3:
            continue
        peak_mz = line[0]
        annots = " ".join(line[2:]).strip('"')
        # glycan parsing not implemented
        if "|" in annots:
            error = ValueError("glycans not supported (2)")
            all_good = False
            break
        # peptide parsing not implemented
        if is_pep_annot(annots):
            error = ValueError("peptides not supported (2)")
            all_good = False
            break
        annots = annots.split(";")
        for annot in annots:
            # strip the peak count
            annot = re.sub(r" ?\d+/\d+$", "", annot)
            if len(annot) == 0:
                error = ValueError("empty annotation")
                all_good = False
                break
            if annot in ("?", "more"):
                continue
            if "/" in annot:
                annot, ppm = annot.split("/")
            if "=" in annot:
                annot, _ = annot.split("=")
            try:
                product, _, _, isotope, _ = parse_product_isotope(annot, precursor)
            except pyteomics.auxiliary.structures.PyteomicsError as e:
                error = e
                all_good = False
                break
            except ValueError as e:
                error = e
                all_good = False
                break
            try:
                exact_mz = composition_to_mass(product, isotope)
            except RuntimeError as e:
                # something wrong with the formula
                error = e
                all_good = False
                break
            loss = Composition(formula=precursor) - product
            product = composition_to_string(product)
            loss = composition_to_string(loss)
            peak_mzs.append(peak_mz)
            products.append(product)
            losses.append(loss)
            isotopes.append(isotope)
            exact_mzs.append(exact_mz)
    error = repr(error)
    if all_good:
        return peak_mzs, products, losses, isotopes, exact_mzs, error
    else:
        return np.nan, np.nan, np.nan, np.nan, np.nan, error


def convert_peaks_to_float(peaks: list) -> list:
    # assumes no nan
    float_peaks = []
    for peak in peaks:
        float_peaks.append((float(peak[0]), float(peak[1])))
    return float_peaks


def get_res(peaks: list) -> int:
    # assumes no nan
    ress = []
    for mz, _ints in peaks:
        dec_idx = mz.find(".")
        if dec_idx == -1:
            res = 0
        else:
            res = len(mz) - (dec_idx + 1)
        ress.append(res)
    highest_res = max(ress)
    return highest_res


def parse_mass_gym_ce_str(ce_str: str) -> tuple[float, bool, bool]:
    """make it comparable to NIST so we can parse it

    Args:
            ce_str (_type_): _description_

    Returns:
            _type_: _description_
    """
    if pd.isna(ce_str):
        return "", False, False

    ce_str = str(ce_str)
    normalized = "normalized=True" in ce_str
    ramped = "ramped=True" in ce_str

    ce = ce_str.split(" ")[0]
    if normalized:
        ce = f"NCE={ce}%"
    else:
        ce = f"{ce} eV"

    return ce, normalized, ramped


