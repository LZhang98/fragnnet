"""Relabel the retired "FT" instrument type of an existing spec_df as "OB" / "OB_F".

Orbitraps are "OB_F" for the Orbitrap Fusion family and "OB" otherwise (Elite, Velos, unnamed);
they fragment differently at the same NCE, see
docs/findings/nist_orbitrap_instrument_ce_scales_2026_09_28.md. A row becomes OB_F on the first
of these that applies, and OB otherwise:

1. ``--inst_df_fp``: a ``data/df`` table (csv or pkl) with ``dset_spec_id`` and ``inst`` (the raw
   MSP ``Instrument`` field); "Fusion" in ``inst`` means OB_F. NIST23: ``data/df/nist23_hr_df.csv``.
2. ``--ref_spec_df_fp``/``--ref_mol_df_fp`` with ``--mol_df_fp``: an already-relabelled dataset.
   A spectrum takes the reference label when every reference spectrum with the same
   :func:`match_keys` key (InChIKey block 1, adduct, precursor m/z to 2 dp, and the recorded eV,
   or the NCE when there is no eV) agrees. NIST20 against the relabelled NIST23.
3. ``--nist_ev_rule``: remaining NIST-numbered positive-mode HCD spectra with an NCE but no
   recorded eV, or an eV more than 10% above ``NCE * m/z * charge factor / 500``. The Elite
   records ``floor(conversion)``; the Lumos records nothing or more (99.7-100% Lumos on NIST23
   [M+H]+). Negative-mode Elite eV conventions vary in NIST20, so the rule skips them.

Existing groups are split by the new label and never regrouped: a group that stays whole keeps
its ``group_id``; each extra instrument part gets a new id after the current maximum. (Stored
group_ids can come from an older, finer key than 02_prepare_proc.py's.) Split files select
spectra by ``spec_id``, so their ``group_id`` column is not read. The original file is copied
byte for byte next to it before it is replaced.

Usage:
    python preproc_scripts/dataset_migration/relabel_orbitrap_inst_type.py \\
        --spec_df_fp data/proc/nist23/spec_df.pkl --inst_df_fp data/df/nist23_hr_df.csv
    python preproc_scripts/dataset_migration/relabel_orbitrap_inst_type.py \\
        --spec_df_fp data/proc/nist20/spec_df.pkl --mol_df_fp data/proc/nist20/mol_df.pkl \\
        --ref_spec_df_fp data/proc/nist23/spec_df.pkl --ref_mol_df_fp data/proc/nist23/mol_df.pkl \\
        --nist_ev_rule
"""

import argparse
import gc
import shutil
from datetime import datetime

import numpy as np
import pandas as pd

from fragnnet.utils.data_utils import CHARGE_FACTOR_MAP, get_charge

ORBITRAP_TYPES = ("OB", "OB_F")


def _charge_factor(prec_type: str) -> float:
    try:
        charge = abs(get_charge(prec_type))
    except (AssertionError, AttributeError, ValueError):
        charge = 1  # malformed or unbracketed contributed-library labels; all singly charged
    return CHARGE_FACTOR_MAP.get(charge, CHARGE_FACTOR_MAP["large"])


def match_keys(spec_df: pd.DataFrame, mol_df: pd.DataFrame) -> pd.Series:
    """Cross-release spectrum key: InChIKey block 1 | adduct | m/z (2 dp) | energy.

    The energy is the recorded eV when there is one, else the NCE. The eV of a NIST spectrum is
    the same in NIST20 and NIST23, while NIST23 doubled the NCE of Elite/Velos negative-mode
    spectra, so an NCE key would miss those. Lumos spectra without an eV keep their NCE.
    """
    ik_col = next(c for c in mol_df.columns if "inchikey" in c.lower())
    ik = spec_df["mol_id"].map(mol_df.set_index("mol_id")[ik_col]).str[:14]
    ace = pd.to_numeric(spec_df["ace"], errors="coerce")
    nce = pd.to_numeric(spec_df["nce"], errors="coerce")
    energy = ("eV" + ace.astype(str)).where(ace.notna(), "NCE" + nce.astype(str))
    mz = pd.to_numeric(spec_df["prec_mz"], errors="coerce").round(2).astype(str)
    return ik + "|" + spec_df["prec_type"].astype(str) + "|" + mz + "|" + energy


def relabel_inst_type(
    spec_df: pd.DataFrame,
    inst_by_dset_spec_id: pd.Series | None = None,
    ref_label_by_key: pd.Series | None = None,
    keys: pd.Series | None = None,
    nist_ev_rule: bool = False,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Label ``inst_type == "FT"`` rows OB / OB_F and split ``group_id`` by instrument.

    Args:
        spec_df: Processed spectra with ``inst_type``, ``dset_spec_id``, ``mol_id``,
            ``prec_type``, ``prec_mz``, ``nce``, ``ace`` and ``frag_mode``.
        inst_by_dset_spec_id: Raw instrument name indexed by ``dset_spec_id`` (as str).
        ref_label_by_key: OB / OB_F indexed by :func:`match_keys` of a reference dataset.
        keys: :func:`match_keys` of ``spec_df``, required with ``ref_label_by_key``.
        nist_ev_rule: Apply the eV-record rule to the rows the other sources leave open.

    Returns:
        The new ``inst_type`` and ``group_id`` columns (aligned to ``spec_df.index``), and for
        every former FT row the evidence it was labelled by (``inst``, ``reference``,
        ``ev_rule`` or ``default``).
    """
    ft = spec_df["inst_type"] == "FT"
    label = pd.Series(np.nan, index=spec_df.index, dtype=object)
    source = pd.Series(np.nan, index=spec_df.index, dtype=object)

    # 1. Raw instrument field.
    if inst_by_dset_spec_id is not None:
        inst = spec_df["dset_spec_id"].astype(str).map(inst_by_dset_spec_id)
        named = ft & inst.notna()
        label[named] = np.where(inst[named].str.contains("Fusion"), "OB_F", "OB")
        source[named] = "inst"

    # 2. Unanimous reference label.
    if ref_label_by_key is not None:
        todo = ft & label.isna()
        ref = keys[todo].map(ref_label_by_key)
        hit = ref.index[ref.notna()]
        label[hit] = ref[hit]
        source[hit] = "reference"

    # 3. eV-record rule for the remaining NIST-numbered HCD spectra.
    if nist_ev_rule:
        todo = ft & label.isna()
        todo &= spec_df["dset_spec_id"].astype(str).str.fullmatch(r"\d+")
        todo &= spec_df["frag_mode"] == "HCD"
        todo &= spec_df["prec_type"].astype(str).str.endswith("+")
        nce = pd.to_numeric(spec_df["nce"], errors="coerce")
        ace = pd.to_numeric(spec_df["ace"], errors="coerce")
        cf = spec_df["prec_type"].map({p: _charge_factor(p) for p in spec_df["prec_type"].unique()})
        conv = nce * pd.to_numeric(spec_df["prec_mz"], errors="coerce") * cf / 500
        fusion = nce.notna() & (ace.isna() | (ace > 1.1 * conv))
        label[todo] = np.where(fusion[todo], "OB_F", "OB")
        source[todo] = "ev_rule"

    # 4. Every other FT row is an unnamed Orbitrap.
    rest = ft & label.isna()
    label[rest] = "OB"
    source[rest] = "default"
    inst_type = spec_df["inst_type"].where(~ft, label)

    # 5. Split existing groups by the new label; never regroup. Stored group_ids may come from an
    # older, finer key than 02_prepare_proc.py's, so a group keeps its id when it stays whole and
    # each extra instrument part gets a new id after the current maximum.
    old = spec_df["group_id"]
    part = (
        pd.DataFrame({"old": old, "inst_type": inst_type})
        .groupby(["old", "inst_type"], sort=True, dropna=False)
        .ngroup()
    )
    first_part = part.groupby(old).transform("min")
    extra = part != first_part
    extra_ids = pd.Series(part[extra].unique())
    new_id = pd.Series(np.arange(len(extra_ids)) + old.max() + 1, index=extra_ids.values)
    group_id = old.copy()
    group_id[extra] = part[extra].map(new_id).astype(old.dtype)
    return inst_type, group_id, source[ft]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--spec_df_fp", required=True)
    parser.add_argument("--inst_df_fp", default=None)
    parser.add_argument("--mol_df_fp", default=None)
    parser.add_argument("--ref_spec_df_fp", default=None)
    parser.add_argument("--ref_mol_df_fp", default=None)
    parser.add_argument("--nist_ev_rule", action="store_true")
    args = parser.parse_args()

    # 1. Evidence first, so a large reference is freed before the target loads.
    inst_by_id = None
    if args.inst_df_fp:
        inst_df = (
            pd.read_csv(args.inst_df_fp, usecols=["dset_spec_id", "inst"], dtype=str)
            if args.inst_df_fp.endswith(".csv")
            else pd.read_pickle(args.inst_df_fp)[["dset_spec_id", "inst"]].astype(str)
        )
        inst_df = inst_df[inst_df["inst"].str.contains("Orbitrap", na=False)]
        inst_by_id = inst_df.drop_duplicates("dset_spec_id").set_index("dset_spec_id")["inst"]
    ref_by_key = None
    if args.ref_spec_df_fp:
        ref = pd.read_pickle(args.ref_spec_df_fp)
        ref = ref.loc[
            ref["inst_type"].isin(ORBITRAP_TYPES),
            ["mol_id", "prec_type", "nce", "ace", "prec_mz", "inst_type"],
        ]
        ref_keys = match_keys(ref, pd.read_pickle(args.ref_mol_df_fp))
        n_labels = ref.groupby(ref_keys)["inst_type"].nunique()
        ref_by_key = ref.groupby(ref_keys)["inst_type"].first()[n_labels == 1]
        del ref, ref_keys
        gc.collect()

    # 2. Load the target and relabel.
    spec_df = pd.read_pickle(args.spec_df_fp)
    ft = spec_df["inst_type"] == "FT"
    print(f"> {args.spec_df_fp}: {len(spec_df):,} spectra, {int(ft.sum()):,} with inst_type FT")
    if not ft.any():
        print("> No FT rows. Skipping backup and save.")
        return
    keys = match_keys(spec_df, pd.read_pickle(args.mol_df_fp)) if ref_by_key is not None else None
    inst_type, group_id, source = relabel_inst_type(
        spec_df, inst_by_id, ref_by_key, keys, args.nist_ev_rule
    )
    print("> former FT rows by label and evidence:")
    print(pd.crosstab(inst_type[ft], source).to_string())
    print(f"> groups: {spec_df['group_id'].nunique():,} -> {group_id.nunique():,}")

    # 3. Copy the original byte for byte, then replace the two columns in place.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_fp = args.spec_df_fp.replace("spec_df.pkl", f"spec_df_orig_{timestamp}.pkl")
    if backup_fp == args.spec_df_fp:
        raise ValueError(f"expected a file named spec_df.pkl, got {args.spec_df_fp}")
    shutil.copy2(args.spec_df_fp, backup_fp)
    spec_df["inst_type"] = inst_type
    spec_df["group_id"] = group_id
    spec_df.to_pickle(args.spec_df_fp)
    print(f"> backup: {backup_fp}\n> updated: {args.spec_df_fp}")


if __name__ == "__main__":
    main()
