"""Build the Spectraverse benchmark CV splits from the MGF ``FOLD=`` field.

Every Spectraverse record carries one ``FOLD=`` value in 1..10; folds are disjoint by 2D
InChIKey. For ``cv{k}`` the test split is FOLD ``k + 1``, the validation split is FOLD ``k``
(FOLD 0 wraps to 10, so cv0 validates on FOLD 10), and train is the other eight folds.
Record ``i`` (1-based) of the MGF is the spectrum whose ``dset_spec_id`` is ``str(i)``.
``mol_id``, ``spec_id`` and ``group_id`` come from the CURRENT ``spec_df.pkl``, so the ids
always agree with the processed data.

Fold source (exactly one of)::

    --mgf-fp   stream the MGF (only TITLE= and FOLD= lines are parsed, md5 is recorded)
    --fold-fp  a '<name>_fold.csv' with columns spec_id,dset_spec_id,fold written by
               preprocessing step 1; rows are joined to spec_df on ``dset_spec_id`` (the key
               that survives step 2, which relabels ``spec_id``)

``--check-against`` runs AFTER the splits are written; its exit code reflects spec_id-set
equality only (mol_id/group_id differences are logged). ``--dst-split-dp`` must not be, or lie
inside, the ``--check-against`` directory.

Example::

    python preproc_scripts/spectraverse/build_benchmark_split.py \
        --dst-split-dp /tmp/spv_split --check-against data/split/spectraverse_benchmark
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

NUM_FOLDS = 10
SPLITS = ("train", "val", "test")
COLUMNS = ["mol_id", "spec_id", "group_id"]
TITLE_RE = re.compile(rb"^TITLE=SPECTRAVERSE(\d+)\s*$")
DEFAULT_SPEC_FP = "data/proc/spectraverse/spec_df.pkl"
DEFAULT_MGF_FP = "data/raw/spectraverse/spectraverse-1.0.2/spectraverse-1.0.2.mgf"


def read_mgf_folds(mgf_fp: Path) -> tuple[list[int], str]:
    """Stream an MGF and return the FOLD of every record plus the file md5.

    Only ``TITLE=`` and ``FOLD=`` lines are parsed. Each record must have exactly one of each,
    and the TITLE number must equal the 1-based record index.

    Args:
        mgf_fp: Path to the Spectraverse MGF.

    Returns:
        ``(folds, md5)``: ``folds[i - 1]`` is the FOLD of record ``i``.

    Raises:
        ValueError: If a record lacks TITLE or FOLD (legacy INDEX/FOLD_* MGFs are not
            supported), repeats one, has a mismatched TITLE or a non-integer FOLD.
    """
    md5 = hashlib.md5()
    folds: list[int] = []
    title: int | None = None
    fold: int | None = None
    in_record = False
    with open(mgf_fp, "rb") as f:
        for line in f:
            md5.update(line)
            if line.startswith(b"BEGIN IONS"):
                if in_record:
                    raise ValueError(f"BEGIN IONS inside record {len(folds) + 1}")
                in_record, title, fold = True, None, None
            elif line.startswith(b"END IONS"):
                idx = len(folds) + 1
                if not in_record or title is None or fold is None:
                    raise ValueError(
                        f"record {idx} lacks TITLE or FOLD; a v1.0.2 FOLD= field is required "
                        "(legacy INDEX/FOLD_* MGFs are not supported)"
                    )
                if title != idx:
                    raise ValueError(f"record {idx} has TITLE number {title}")
                folds.append(fold)
                in_record = False
            elif in_record and line.startswith(b"TITLE="):
                match = TITLE_RE.match(line)
                if match is None or title is not None:
                    raise ValueError(f"record {len(folds) + 1}: bad or repeated {line!r}")
                title = int(match.group(1))
            elif in_record and line.startswith(b"FOLD="):
                if fold is not None:
                    raise ValueError(f"record {len(folds) + 1}: repeated FOLD")
                try:
                    fold = int(line[5:].strip())
                except ValueError as err:
                    raise ValueError(
                        f"record {len(folds) + 1}: FOLD is not an integer: {line!r}"
                    ) from err
    if in_record:
        raise ValueError("MGF ends inside a record")
    return folds, md5.hexdigest()


def assign_folds_from_mgf(spec_df: pd.DataFrame, folds: list[int]) -> pd.Series:
    """Map MGF record folds onto ``spec_df`` rows through ``dset_spec_id`` (1-based index).

    Args:
        spec_df: Processed spectra with ``dset_spec_id``.
        folds: ``folds[i - 1]`` is the FOLD of MGF record ``i``.

    Returns:
        Integer FOLD per ``spec_df`` row, aligned with its index.

    Raises:
        ValueError: If the record count differs from ``len(spec_df)`` or ids do not cover
            ``1..len(folds)`` exactly once.
    """
    if len(folds) != len(spec_df):
        raise ValueError(f"MGF has {len(folds)} records but spec_df has {len(spec_df)} rows")
    rec = spec_df["dset_spec_id"].astype(str).astype(int)
    if rec.nunique() != len(rec) or rec.min() != 1 or rec.max() != len(folds):
        raise ValueError("dset_spec_id is not a permutation of 1..n_records")
    return pd.Series([folds[i - 1] for i in rec], index=spec_df.index, dtype="int64")


def assign_folds_from_csv(spec_df: pd.DataFrame, fold_fp: Path) -> pd.Series:
    """Look up FOLD per ``spec_df`` row in a ``spec_id,dset_spec_id,fold`` helper CSV.

    Rows are joined on ``dset_spec_id`` (the MGF record index), which is stable through
    preprocessing step 2; ``spec_id`` is relabeled there and is not used for the join.

    Args:
        spec_df: Processed spectra with ``dset_spec_id``.
        fold_fp: CSV with columns ``spec_id``, ``dset_spec_id`` and ``fold``.

    Returns:
        Integer FOLD per ``spec_df`` row, aligned with its index.

    Raises:
        ValueError: If columns are wrong, dset_spec_ids repeat, or the dset_spec_id sets differ.
    """
    fold_df = pd.read_csv(fold_fp)
    if set(fold_df.columns) != {"spec_id", "dset_spec_id", "fold"}:
        raise ValueError(
            f"{fold_fp} must have columns spec_id,dset_spec_id,fold; got {list(fold_df.columns)}"
        )
    if fold_df["dset_spec_id"].duplicated().any():
        raise ValueError(f"{fold_fp} repeats dset_spec_id")
    rec = spec_df["dset_spec_id"].astype(str).astype(int)
    if rec.duplicated().any():
        raise ValueError("spec_df repeats dset_spec_id")
    if set(rec) != set(fold_df["dset_spec_id"]):
        raise ValueError(f"{fold_fp} dset_spec_id set differs from spec_df")
    lookup = fold_df.set_index("dset_spec_id")["fold"]
    return pd.Series(lookup.loc[rec].to_numpy(), index=spec_df.index, dtype="int64")


def inchikey_by_mol(mol_df: pd.DataFrame) -> pd.Series:
    """``inchikey_s`` (2D InChIKey) indexed by ``mol_id``."""
    return mol_df.drop_duplicates("mol_id").set_index("mol_id")["inchikey_s"]


def check_fold_disjointness(fold: pd.Series, mol_ids: pd.Series, mol_df: pd.DataFrame) -> int:
    """Require every 2D InChIKey (``inchikey_s``) to occur in exactly one fold.

    Args:
        fold: FOLD per spectrum.
        mol_ids: ``mol_id`` per spectrum, aligned with ``fold``.
        mol_df: Molecules with ``mol_id`` and ``inchikey_s``.

    Returns:
        Number of distinct 2D InChIKeys.

    Raises:
        ValueError: If a mol_id is missing from ``mol_df`` or an InChIKey spans folds.
    """
    key = inchikey_by_mol(mol_df)
    missing = ~mol_ids.isin(key.index)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} spectra have a mol_id absent from mol_df")
    keys = mol_ids.map(key)
    n_folds = fold.groupby(keys.to_numpy()).nunique()
    bad = n_folds[n_folds > 1]
    if len(bad):
        raise ValueError(f"{len(bad)} 2D InChIKeys span several folds (e.g. {list(bad.index[:3])})")
    return int(keys.nunique())


def val_fold_for_cv(cv: int) -> int:
    """Validation FOLD of ``cv`` (FOLD 0 wraps to 10)."""
    return cv if cv > 0 else NUM_FOLDS


def test_fold_for_cv(cv: int) -> int:
    """Test FOLD of ``cv``."""
    return cv + 1


def build_cv(spec_df: pd.DataFrame, fold: pd.Series, cv: int) -> dict[str, pd.DataFrame]:
    """Split ``spec_df`` into train/val/test id tables for one cv.

    Args:
        spec_df: Processed spectra with ``mol_id``, ``spec_id``, ``group_id``.
        fold: FOLD per spectrum, aligned with ``spec_df``.
        cv: Fold index in 0..9.

    Returns:
        ``{split: DataFrame[mol_id, spec_id, group_id]}`` sorted by integer spec_id.

    Raises:
        ValueError: If the splits do not cover every spectrum exactly once.
    """
    if not 0 <= cv < NUM_FOLDS:
        raise ValueError(f"cv must be in 0..{NUM_FOLDS - 1}, got {cv}")
    ids = spec_df[COLUMNS].copy()
    ids["spec_id"] = ids["spec_id"].astype(int)
    ids = ids.astype("int64")
    masks = {
        "val": fold == val_fold_for_cv(cv),
        "test": fold == test_fold_for_cv(cv),
    }
    masks["train"] = ~(masks["val"] | masks["test"])
    out = {name: ids[masks[name]].sort_values("spec_id").reset_index(drop=True) for name in SPLITS}
    all_ids = pd.concat([out[name]["spec_id"] for name in SPLITS])
    if len(all_ids) != len(ids) or all_ids.nunique() != len(ids):
        raise ValueError(f"cv{cv}: train+val+test does not cover each spectrum once")
    return out


def count_molecules(
    splits: dict[str, pd.DataFrame], key_by_mol: pd.Series
) -> tuple[dict[str, int], dict[str, int]]:
    """Count distinct 2D InChIKeys (the fold unit) and distinct mol_ids per split."""
    n_keys = {n: int(df["mol_id"].map(key_by_mol).nunique()) for n, df in splits.items()}
    n_mol_ids = {n: int(df["mol_id"].nunique()) for n, df in splits.items()}
    return n_keys, n_mol_ids


def write_cv(
    dst_dp: Path,
    cv: int,
    splits: dict[str, pd.DataFrame],
    counts: tuple[dict[str, int], dict[str, int]],
    md5: str | None,
) -> None:
    """Write ``{train,val,test}_ids.csv`` and ``stats.json`` into ``dst_dp/cv{cv}``.

    ``counts`` is ``(num_molecules, num_mol_ids)`` from :func:`count_molecules`; ``stats.json``
    reports ``num_molecules`` as distinct 2D InChIKeys (the unit the folds are disjoint over).
    """
    cv_dp = dst_dp / f"cv{cv}"
    cv_dp.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_csv(cv_dp / f"{name}_ids.csv", index=False)
    stats = {
        "cv": cv,
        "folds": {
            "train": [
                f
                for f in range(1, NUM_FOLDS + 1)
                if f not in (val_fold_for_cv(cv), test_fold_for_cv(cv))
            ],
            "val": val_fold_for_cv(cv),
            "test": test_fold_for_cv(cv),
        },
        "num_spectra": {name: len(df) for name, df in splits.items()},
        "num_molecules": counts[0],
        "num_mol_ids": counts[1],
        "mgf_md5": md5,
    }
    with open(cv_dp / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)


def check_against(
    built: dict[int, dict[str, pd.DataFrame]], ref_dp: Path
) -> tuple[bool, list[str]]:
    """Compare built splits with an existing split directory.

    Args:
        built: ``{cv: {split: ids}}`` from :func:`build_cv`.
        ref_dp: Directory holding ``cv{k}/{split}_ids.csv``.

    Returns:
        ``(all_sets_equal, report_lines)``. spec_id sets decide equality; mol_id/group_id
        row mismatches (rows aligned by spec_id) are only reported.
    """
    ok = True
    lines = []
    for cv, splits in sorted(built.items()):
        for name, new in splits.items():
            ref = pd.read_csv(ref_dp / f"cv{cv}" / f"{name}_ids.csv")
            same = set(new["spec_id"]) == set(ref["spec_id"]) and len(new) == len(ref)
            ok &= same
            merged = new.merge(ref, on="spec_id", suffixes=("", "_ref"))
            n_mol = int((merged["mol_id"] != merged["mol_id_ref"]).sum())
            n_grp = int((merged["group_id"] != merged["group_id_ref"]).sum())
            lines.append(
                f"cv{cv}/{name}: spec_id sets {'IDENTICAL' if same else 'DIFFER'} "
                f"(new {len(new)}, ref {len(ref)}); rows with differing mol_id {n_mol}, "
                f"group_id {n_grp}, of {len(merged)} shared"
            )
    return ok, lines


def check_dst(dst: Path, cv_folds: list[int], overwrite: bool, ref_dp: Path | None) -> None:
    """Refuse destinations that would mix builds or overwrite the reference split.

    Args:
        dst: Output split directory.
        cv_folds: The cv indices this run writes.
        overwrite: Whether ``--overwrite`` was given.
        ref_dp: ``--check-against`` directory, if any.

    Raises:
        NotADirectoryError: If ``dst`` exists and is not a directory.
        ValueError: If ``dst`` is, or lies inside or above, ``ref_dp``.
        FileExistsError: If ``dst`` is non-empty without ``--overwrite``, or holds anything
            other than files of the cvs this run rewrites (stale cvs from another build would
            otherwise sit next to the new ones).
    """
    if ref_dp is not None:
        d, r = dst.resolve(), ref_dp.resolve()
        if d.is_relative_to(r) or r.is_relative_to(d):
            raise ValueError(f"--dst-split-dp {dst} overlaps --check-against {ref_dp}")
    if not dst.exists():
        return
    if not dst.is_dir():
        raise NotADirectoryError(f"--dst-split-dp {dst} exists and is not a directory")
    existing = {p.name for p in dst.iterdir()}
    if not existing:
        return
    if not overwrite:
        raise FileExistsError(f"{dst} is non-empty; pass --overwrite to write into it")
    targeted = {f"cv{k}" for k in cv_folds}
    if existing - targeted:
        raise FileExistsError(
            f"{dst} holds {sorted(existing - targeted)} which this run would not rewrite; "
            "use an empty directory"
        )
    expected = {f"{n}_ids.csv" for n in SPLITS} | {"stats.json"}
    for name in existing:
        extra = {p.name for p in (dst / name).iterdir()} - expected
        if extra:
            raise FileExistsError(f"{dst / name} holds unexpected files {sorted(extra)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--spec-fp", type=Path, default=Path(DEFAULT_SPEC_FP))
    p.add_argument(
        "--mol-fp",
        type=Path,
        default=None,
        help="mol_df.pkl for the disjointness check (default: sibling of --spec-fp)",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--mgf-fp",
        type=Path,
        nargs="?",
        const=Path(DEFAULT_MGF_FP),
        help=f"MGF to read FOLD= from (bare flag: {DEFAULT_MGF_FP})",
    )
    src.add_argument(
        "--fold-fp", type=Path, help="'<name>_fold.csv' with spec_id,dset_spec_id,fold"
    )
    p.add_argument(
        "--expect-md5",
        default=None,
        help="raise unless the MGF md5 equals this (only with --mgf-fp)",
    )
    p.add_argument(
        "--dst-split-dp",
        type=Path,
        required=True,
        help="output split dir (required; never defaulted)",
    )
    p.add_argument("--folds", type=int, nargs="+", default=list(range(NUM_FOLDS)))
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="rewrite the cv dirs of this run in an existing --dst-split-dp (nothing else may be "
        "in it)",
    )
    p.add_argument(
        "--check-against",
        type=Path,
        default=None,
        help="existing split dir to compare spec_id sets with",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the splits; return 1 if ``--check-against`` finds differing spec_id sets."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dst = args.dst_split_dp
    for cv in args.folds:
        if not 0 <= cv < NUM_FOLDS:
            raise ValueError(f"--folds must be in 0..{NUM_FOLDS - 1}, got {cv}")
    if args.expect_md5 is not None and args.fold_fp is not None:
        raise ValueError("--expect-md5 needs --mgf-fp")
    check_dst(dst, args.folds, args.overwrite, args.check_against)

    spec_df = pd.read_pickle(args.spec_fp)
    if spec_df["spec_id"].astype(int).duplicated().any():
        raise ValueError("spec_df has repeated spec_id")
    md5 = None
    if args.fold_fp is not None:
        fold = assign_folds_from_csv(spec_df, args.fold_fp)
    else:
        mgf_fp = args.mgf_fp if args.mgf_fp is not None else Path(DEFAULT_MGF_FP)
        folds, md5 = read_mgf_folds(mgf_fp)
        LOGGER.info("read %d MGF records, md5 %s", len(folds), md5)
        if args.expect_md5 is not None and md5 != args.expect_md5:
            raise ValueError(f"MGF md5 {md5} differs from --expect-md5 {args.expect_md5}")
        fold = assign_folds_from_mgf(spec_df, folds)
    if set(fold.unique()) != set(range(1, NUM_FOLDS + 1)):
        raise ValueError(f"FOLD values {sorted(fold.unique())} are not exactly 1..{NUM_FOLDS}")

    mol_fp = args.mol_fp if args.mol_fp is not None else args.spec_fp.parent / "mol_df.pkl"
    mol_df = pd.read_pickle(mol_fp)
    n_keys = check_fold_disjointness(fold, spec_df["mol_id"], mol_df)
    LOGGER.info("folds disjoint over %d 2D InChIKeys", n_keys)

    key_by_mol = inchikey_by_mol(mol_df)
    built = {}
    for cv in args.folds:
        built[cv] = build_cv(spec_df, fold, cv)
        write_cv(dst, cv, built[cv], count_molecules(built[cv], key_by_mol), md5)
        LOGGER.info("cv%d: %s", cv, {n: len(df) for n, df in built[cv].items()})

    if args.check_against is not None:
        ok, lines = check_against(built, args.check_against)
        for line in lines:
            LOGGER.info(line)
        LOGGER.info("spec_id sets %s", "ALL IDENTICAL" if ok else "DIFFER")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
