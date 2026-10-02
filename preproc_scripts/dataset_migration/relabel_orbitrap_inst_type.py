"""Relabel the retired "FT" instrument type of an existing spec_df as "ORB" / "ORB_F".

Orbitraps are "ORB_F" for the Orbitrap Fusion family and "ORB" otherwise (Elite, Velos, unnamed);
they fragment differently at the same NCE, see
docs/findings/nist_orbitrap_instrument_ce_scales_2026_09_28.md. The label comes from the raw
instrument name, joined on ``dset_spec_id``: "Fusion" in the name means ORB_F, as in
``parse_inst_info``. FT rows without a name are unnamed Orbitraps, ORB.

``--inst_fp`` takes the name sources in priority order; for each ``dset_spec_id`` the first
source that names it wins. A source is either
- a ``data/df`` table (.csv or .pkl) with ``dset_spec_id`` and ``inst`` columns, or
- a raw MSP (.msp, .msp.gz, or a .zip holding one .msp), read for ``--msp_id_key`` (default
  ``NIST#``, the NIST ``dset_spec_id``) and ``Instrument``.
NIST# is stable across releases, so a later release can name spectra an older export lacks:
1,011,493 NIST20 records share a NIST# with NIST23, none with a different instrument, and the
5,537 NIST20 spec_df spectra missing from the NIST20 export are all in NIST23.

Existing groups are split by the new label and never regrouped: a group that stays whole keeps
its ``group_id``; each extra instrument part gets a new id after the current maximum. (Stored
group_ids can come from an older, finer key than 02_prepare_proc.py's.) Split files select
spectra by ``spec_id``, so their ``group_id`` column is not read. The original file is copied
byte for byte next to it before it is replaced.

Usage:
    python preproc_scripts/dataset_migration/relabel_orbitrap_inst_type.py \\
        --spec_df_fp data/proc/nist23/spec_df.pkl --inst_fp data/df/nist23_hr_df.csv
    python preproc_scripts/dataset_migration/relabel_orbitrap_inst_type.py \\
        --spec_df_fp data/proc/nist20/spec_df.pkl \\
        --inst_fp data/raw/nist_20/nist_20_hr_msms.zip data/df/nist23_hr_df.csv
"""

import argparse
import importlib.util
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

MSP_SUFFIXES = (".msp", ".msp.gz", ".zip")


def _iter_msp_lines(msp_fp: str):
    path = Path(__file__).resolve().parent.parent / "01_prepare_df.py"
    spec = importlib.util.spec_from_file_location("prepare_df", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.iter_msp_lines(msp_fp)


def read_msp_instruments(msp_fp: str, id_key: str = "NIST#") -> pd.Series:
    """Raw ``Instrument`` of every MSP record, indexed by its ``id_key`` value (as str).

    Records without either field are skipped; a repeated id must name the same instrument.
    """
    ids, insts = [], []
    rec, in_peaks = {}, False
    for line in _iter_msp_lines(msp_fp):
        line = line.rstrip("\n")
        if not line.strip():
            if id_key in rec and "Instrument" in rec:
                ids.append(rec[id_key])
                insts.append(rec["Instrument"])
            rec, in_peaks = {}, False
        elif not in_peaks:
            key, _, value = line.partition(":")
            key = key.strip()
            if key.lower() == "num peaks":
                in_peaks = True
            elif key in (id_key, "Instrument") and key not in rec:
                rec[key] = value.strip()
    if id_key in rec and "Instrument" in rec:
        ids.append(rec[id_key])
        insts.append(rec["Instrument"])
    inst = pd.Series(insts, index=pd.Index(ids, name="dset_spec_id"), name="inst")
    conflicts = inst.groupby(level=0).nunique() > 1
    if conflicts.any():
        raise ValueError(f"{msp_fp}: ids naming two instruments: {conflicts.index[conflicts][:5].tolist()}")
    return inst[~inst.index.duplicated()]


def read_instruments(fp: str, msp_id_key: str = "NIST#") -> pd.Series:
    """Instrument names from a data/df table or a raw MSP, indexed by ``dset_spec_id`` (str)."""
    if fp.endswith(MSP_SUFFIXES):
        return read_msp_instruments(fp, msp_id_key)
    if fp.endswith(".csv"):
        df = pd.read_csv(fp, usecols=["dset_spec_id", "inst"], dtype=str)
    else:
        df = pd.read_pickle(fp)[["dset_spec_id", "inst"]]
        df = df.assign(dset_spec_id=df["dset_spec_id"].astype(str))
    df = df[df["inst"].notna()].drop_duplicates("dset_spec_id")
    return df.set_index("dset_spec_id")["inst"].astype(str)


def relabel_inst_type(
    spec_df: pd.DataFrame, inst_by_dset_spec_id: pd.Series | None = None
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Label ``inst_type == "FT"`` rows ORB / ORB_F and split ``group_id`` by instrument.

    Args:
        spec_df: Processed spectra with ``inst_type``, ``dset_spec_id`` and ``group_id``.
        inst_by_dset_spec_id: Raw instrument name indexed by ``dset_spec_id`` (as str).

    Returns:
        The new ``inst_type`` and ``group_id`` columns (aligned to ``spec_df.index``), and for
        every former FT row the evidence it was labelled by (``inst`` or ``default``).
    """
    ft = spec_df["inst_type"] == "FT"
    label = pd.Series(np.nan, index=spec_df.index, dtype=object)
    source = pd.Series(np.nan, index=spec_df.index, dtype=object)

    # 1. Raw instrument name.
    if inst_by_dset_spec_id is not None:
        inst = spec_df["dset_spec_id"].astype(str).map(inst_by_dset_spec_id)
        named = ft & inst.notna()
        label[named] = np.where(inst[named].str.contains("Fusion"), "ORB_F", "ORB")
        source[named] = "inst"

    # 2. Every other FT row is an unnamed Orbitrap.
    rest = ft & label.isna()
    label[rest] = "ORB"
    source[rest] = "default"
    inst_type = spec_df["inst_type"].where(~ft, label)

    # 3. Split existing groups by the new label; never regroup. Stored group_ids may come from an
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
    parser.add_argument(
        "--inst_fp",
        nargs="+",
        default=[],
        help="Instrument-name sources in priority order: data/df tables or raw MSPs.",
    )
    parser.add_argument("--msp_id_key", default="NIST#", help="MSP field holding dset_spec_id.")
    args = parser.parse_args()

    # 1. Instrument names, first source wins.
    inst_by_id, origin = None, None
    if args.inst_fp:
        parts = []
        for fp in args.inst_fp:
            inst = read_instruments(fp, args.msp_id_key)
            print(f"> {fp}: {len(inst):,} named spectra")
            parts.append(inst.to_frame().assign(origin=os.path.basename(fp)))
        named = pd.concat(parts)
        named = named[~named.index.duplicated()]
        inst_by_id, origin = named["inst"], named["origin"]

    # 2. Load the target and relabel.
    spec_df = pd.read_pickle(args.spec_df_fp)
    ft = spec_df["inst_type"] == "FT"
    print(f"> {args.spec_df_fp}: {len(spec_df):,} spectra, {int(ft.sum()):,} with inst_type FT")
    if not ft.any():
        print("> No FT rows. Skipping backup and save.")
        return
    inst_type, group_id, source = relabel_inst_type(spec_df, inst_by_id)
    if origin is not None:
        named_from = spec_df.loc[ft, "dset_spec_id"].astype(str).map(origin)
        source = source.where(source != "inst", "inst: " + named_from)
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
