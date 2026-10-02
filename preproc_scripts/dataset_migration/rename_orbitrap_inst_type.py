"""Rename the Orbitrap instrument types "OB" / "OB_F" to "ORB" / "ORB_F" (2026-10-01).

The labels mean the same instruments as before: ORB_F is the Orbitrap Fusion family (NIST's
Lumos), ORB every other Orbitrap (Elite, Velos, unnamed); see
docs/findings/nist_orbitrap_instrument_ce_scales_2026_09_28.md. Code and configs refuse the old
names, so existing data and checkpoints are migrated here.

- ``--spec_df_fp``: renames the ``inst_type`` values. No other column changes, group ids
  included. A spec_df that still has "FT" rows is refused; relabel it with
  relabel_orbitrap_inst_type.py first, which now writes ORB / ORB_F directly.
- ``--ckpt_fp``: renames every dict key, list item and string value "OB" / "OB_F" inside the
  checkpoint's ``hyper_parameters`` (instrument lists and per-instrument tables). Weights are
  untouched: embedding rows follow the sorted label order, and ORB < ORB_F sorts before every
  other label exactly where OB < OB_F did. A checkpoint trained on "FT" is refused, since its
  one-row instrument embedding needs expanding, not renaming.

Each file is written next to itself and then moved over the original, so an interrupted run
leaves the original intact. No backup is kept: the rename is one-to-one, and running it on a
migrated file changes nothing.

Usage:
    python preproc_scripts/dataset_migration/rename_orbitrap_inst_type.py \\
        --spec_df_fp data/proc/nist20/spec_df.pkl data/proc/nist23/spec_df.pkl
    python preproc_scripts/dataset_migration/rename_orbitrap_inst_type.py \\
        --ckpt_fp saved_ckpts/i1_inst_ob_obf_i50ig7rf/ckpt/model-epoch=059.ckpt
"""

import argparse
import os

import pandas as pd
import torch as th

from fragnnet.utils.data_utils import find_retired_inst_type_refs

RENAMES = {"OB": "ORB", "OB_F": "ORB_F"}


def rename_inst_types(node: object) -> object:
    """Return ``node`` with every dict key, list item and string equal to OB / OB_F renamed."""
    if isinstance(node, str):
        return RENAMES.get(node, node)
    if isinstance(node, dict):
        return type(node)(
            {RENAMES.get(key, key): rename_inst_types(value) for key, value in node.items()}
        )
    if isinstance(node, list):
        return [rename_inst_types(value) for value in node]
    return node


def rename_spec_df(spec_df_fp: str) -> None:
    """Rename OB / OB_F rows of a spec_df.pkl in place."""
    spec_df = pd.read_pickle(spec_df_fp)
    before = spec_df["inst_type"].value_counts(dropna=False)
    print(f"> {spec_df_fp}: {len(spec_df):,} spectra, {before.to_dict()}")
    if "FT" in before.index:
        raise ValueError(
            f"{spec_df_fp} still has FT rows: run relabel_orbitrap_inst_type.py on it first"
        )
    if not before.index.isin(list(RENAMES)).any():
        print("> No OB / OB_F rows. Skipping.")
        return
    spec_df["inst_type"] = spec_df["inst_type"].replace(RENAMES)
    expected = before.rename(index=RENAMES)

    # Free the frame before reading the copy back: two NIST23 frames do not fit in 31 GB.
    tmp_fp = f"{spec_df_fp}.tmp"
    spec_df.to_pickle(tmp_fp)
    del spec_df
    written = pd.read_pickle(tmp_fp)["inst_type"].value_counts(dropna=False)
    if not written.sort_index().equals(expected.sort_index()):
        raise RuntimeError(f"{tmp_fp}: written labels {written.to_dict()} != {expected.to_dict()}")
    os.replace(tmp_fp, spec_df_fp)
    print(f"> updated: {spec_df_fp}, {written.to_dict()}")


def rename_ckpt(ckpt_fp: str) -> None:
    """Rename OB / OB_F in a checkpoint's hyper_parameters in place."""
    ckpt = th.load(ckpt_fp, map_location="cpu", weights_only=False)
    old_refs = find_retired_inst_type_refs(ckpt["hyper_parameters"], "hyper_parameters")
    if not old_refs:
        print(f"> {ckpt_fp}: no retired instrument types. Skipping.")
        return
    renamed = rename_inst_types(ckpt["hyper_parameters"])
    if find_retired_inst_type_refs(renamed):
        raise ValueError(
            f"{ckpt_fp} was trained on 'FT': it needs a two-row instrument embedding, not a rename"
        )
    ckpt["hyper_parameters"] = renamed

    tmp_fp = f"{ckpt_fp}.tmp"
    th.save(ckpt, tmp_fp)
    written = th.load(tmp_fp, map_location="cpu", weights_only=False)["hyper_parameters"]
    if find_retired_inst_type_refs(written):
        raise RuntimeError(f"{tmp_fp}: retired instrument types left after the rename")
    os.replace(tmp_fp, ckpt_fp)
    print(f"> updated: {ckpt_fp}, {len(old_refs)} renamed: {old_refs}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--spec_df_fp", nargs="+", default=[])
    parser.add_argument("--ckpt_fp", nargs="+", default=[])
    args = parser.parse_args()
    if not args.spec_df_fp and not args.ckpt_fp:
        parser.error("pass --spec_df_fp and/or --ckpt_fp")
    for spec_df_fp in args.spec_df_fp:
        rename_spec_df(spec_df_fp)
    for ckpt_fp in args.ckpt_fp:
        rename_ckpt(ckpt_fp)


if __name__ == "__main__":
    main()
