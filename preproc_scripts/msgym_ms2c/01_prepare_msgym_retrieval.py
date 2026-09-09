"""Migrate legacy MSGYM retrieval chunks to the current MS2C input format.

This script reads the MSGYM retrieval-candidate JSON and spectrum TSV, exports
a candidate manifest and a de-duplicated candidate list for fresh downstream
preprocessing.

Usage:
    # 1. Export candidates for Spectraverse-style DAG generation.
    python preproc_scripts/msgym_ms2c/01_prepare_msgym_retrieval.py export \
        --tsv_fp data/raw/msgym_1_5/MassSpecGym1.5.tsv \
        --candidates_fp data/raw/msgym_1_5/MassSpecGym1.5_retrieval_candidates_mass.json \
        --output_dp data/ms2c/msgym/current/retrieval_candidates_mass

"""

from __future__ import annotations

import argparse
import json
import logging
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from fragnnet.utils import data_utils

logger = logging.getLogger(__name__)

REQUIRED_MANIFEST_COLUMNS = {
    "source_identifier",
    "source_smiles",
    "candidate_smiles",
    "prec_type",
    "prec_mz",
    "inst_type",
    "frag_mode",
    "ace",
    "nce",
    "ion_mode",
    "spec_type",
    "dset",
    "peaks",
}


def canonicalize_smiles(smiles: str) -> str:
    """Return the same canonical SMILES representation used by DAG preparation."""
    mol = data_utils.mol_from_smiles(smiles, ml_standardize=False)
    canonical_smiles = data_utils.mol_to_smiles(mol)
    if not canonical_smiles:
        raise ValueError(f"Could not canonicalize candidate SMILES: {smiles}")
    return canonical_smiles


def _ion_mode_from_prec_type(prec_type: str) -> str:
    """Infer the dataset ion-mode token from a MSGYM adduct string."""
    if prec_type.endswith("+"):
        return "P"
    if prec_type.endswith("-"):
        return "N"
    raise ValueError(f"Could not infer ion mode from precursor type: {prec_type}")


def export_msgym_json(
    tsv_fp: Path,
    candidates_fp: Path,
    output_dp: Path,
    subset: str,
    fold: str = "test",
) -> tuple[Path, Path]:
    """Export MSGYM JSON candidates and TSV metadata to a canonical manifest."""
    # 1. Select the retrieval spectra and load their candidate lists.
    spec_df = pd.read_csv(tsv_fp, sep="\t")
    required_columns = {
        "identifier",
        "smiles",
        "precursor_mz",
        "adduct",
        "instrument_type",
        "collision_energy",
        "fold",
        "simulation_challenge",
    }
    missing = required_columns - set(spec_df.columns)
    if missing:
        raise ValueError(f"Missing required MSGYM columns in {tsv_fp}: {sorted(missing)}")
    spec_df = spec_df[spec_df["fold"] == fold].copy()
    if spec_df.empty:
        raise ValueError(f"No MSGYM spectra found for fold {fold!r} in {tsv_fp}")
    if subset == "simulation_challenge":
        spec_df = spec_df[spec_df["simulation_challenge"]].copy()

    with candidates_fp.open() as f:
        candidates_by_smiles = json.load(f)
    if not isinstance(candidates_by_smiles, dict):
        raise ValueError(f"Candidate JSON must be a SMILES-to-list mapping: {candidates_fp}")

    # 2. Flatten each query spectrum over its candidate set. Candidate SMILES
    # are canonicalized once so they match the IDs emitted by DAG preparation.
    canonical_smiles: dict[str, str] = {}
    records: list[dict] = []
    missing_queries: list[str] = []
    for _, spectrum in spec_df.iterrows():
        query_smiles = spectrum["smiles"]
        candidates = candidates_by_smiles.get(query_smiles)
        if candidates is None:
            missing_queries.append(query_smiles)
            continue
        inst_type, frag_mode = data_utils.parse_inst_info(
            pd.Series(
                {"inst_type": spectrum["instrument_type"], "col_energy": spectrum["collision_energy"]}
            )
        )
        for smiles in candidates:
            if smiles not in canonical_smiles:
                canonical_smiles[smiles] = canonicalize_smiles(smiles)
            records.append(
                {
                    "source_identifier": spectrum["identifier"],
                    "source_smiles": query_smiles,
                    "candidate_smiles": canonical_smiles[smiles],
                    "prec_type": spectrum["adduct"],
                    "prec_mz": spectrum["precursor_mz"],
                    "inst_type": inst_type,
                    "frag_mode": frag_mode,
                    "ace": spectrum["collision_energy"],
                    "nce": float("nan"),
                    "ion_mode": _ion_mode_from_prec_type(spectrum["adduct"]),
                    "spec_type": "MS2",
                    "dset": "msgym_1_5",
                    "peaks": [],
                }
            )
    if missing_queries:
        raise ValueError(f"{len(set(missing_queries))} selected MSGYM queries are absent from JSON")

    # 3. Write the full manifest plus its de-duplicated candidate universe.
    # The latter is the input to Spectraverse-style global DAG generation.
    output_dp.mkdir(parents=True, exist_ok=True)
    manifest_fp = output_dp / "query_candidates.pkl.gz"
    manifest = pd.DataFrame(records)
    manifest.to_pickle(manifest_fp)

    candidates_fp = output_dp / "candidate_smiles.csv"
    pd.DataFrame({"smiles": sorted(manifest["candidate_smiles"].unique())}).to_csv(
        candidates_fp, index=False
    )
    logger.info(
        "Exported %d query-candidate records and %d unique candidates",
        len(manifest),
        len(manifest["candidate_smiles"].unique()),
    )
    return manifest_fp, candidates_fp


def build_proc_from_manifest(
    manifest_fp: Path,
    output_dp: Path,
    split_name: str = "test",
) -> tuple[Path, Path, Path]:
    """Convert an exported MSGYM manifest to global MS2C dataframes and a split file."""
    # 1. Validate the manifest and build the global candidate molecule table.
    manifest = pd.read_pickle(manifest_fp)
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Missing required manifest columns in {manifest_fp}: {sorted(missing)}")
    if manifest.empty:
        raise ValueError(f"MSGYM manifest is empty: {manifest_fp}")
    original_size = len(manifest)
    manifest = manifest.drop_duplicates(
        subset=["source_identifier", "candidate_smiles"],
        keep="first",
    ).reset_index(drop=True)
    if len(manifest) != original_size:
        logger.info(
            "Removed %d duplicate query-candidate records",
            original_size - len(manifest),
        )

    candidate_smiles = sorted(manifest["candidate_smiles"].unique())
    mol_df = pd.DataFrame(
        {
            "smiles": candidate_smiles,
            "mol_id": range(len(candidate_smiles)),
        }
    )
    mol_df["mol"] = data_utils.par_apply_series(
        mol_df["smiles"],
        partial(data_utils.mol_from_smiles, canonicalize_tautomers=False),
    )
    invalid_smiles = mol_df.loc[mol_df["mol"].isna(), "smiles"].tolist()
    if invalid_smiles:
        raise ValueError(
            f"Could not build molecules for {len(invalid_smiles)} candidate SMILES: "
            f"{invalid_smiles[:5]}"
        )
    data_utils.compute_mol_df_props(mol_df)

    # 2. Expand query metadata over candidates and align every row to mol_df.
    smiles_to_mol_id = mol_df.set_index("smiles")["mol_id"]
    spec_df = manifest.copy()
    spec_df["mol_id"] = spec_df["candidate_smiles"].map(smiles_to_mol_id)
    if spec_df["mol_id"].isna().any():
        raise ValueError("Some manifest candidates could not be mapped to mol_df")
    spec_df["mol_id"] = spec_df["mol_id"].astype(int)
    spec_df["spec_id"] = range(len(spec_df))
    spec_df["dset_spec_id"] = spec_df["spec_id"]
    spec_df["group_id"] = pd.factorize(spec_df["source_identifier"], sort=False)[0]
    for ce_type in ("ace", "nce"):
        spec_df[f"{ce_type}_extra_1"] = -1.0
        spec_df[f"{ce_type}_extra_2"] = -1.0

    # 3. Save the dataframes and the all-candidate test split used by inference.
    output_dp.mkdir(parents=True, exist_ok=True)
    mol_fp = output_dp / "mol_df.pkl"
    spec_fp = output_dp / "spec_df.pkl"
    split_fp = output_dp / f"{split_name}_ids.csv"
    mol_df.to_pickle(mol_fp)
    spec_df.to_pickle(spec_fp)
    spec_df[["spec_id", "mol_id", "group_id"]].to_csv(split_fp, index=False)
    logger.info(
        "Saved %d molecules and %d query-candidate spectra to %s",
        len(mol_df),
        len(spec_df),
        output_dp,
    )
    return mol_fp, spec_fp, split_fp


_CATEGORY_COLUMNS = (
    "source_identifier",
    "source_smiles",
    "prec_type",
    "inst_type",
    "frag_mode",
    "ion_mode",
    "spec_type",
    "dset",
)


def export_and_build_proc_chunked(
    tsv_fp: Path,
    candidates_fp: Path,
    output_dp: Path,
    subset: str,
    fold: str,
    chunk_size: int = 2000,
    mol_batch_size: int = 200_000,
) -> tuple[Path, Path, Path]:
    """Memory-bounded export + build-proc for folds too large to explode at once.

    The plain ``export`` + ``build-proc`` pair builds one Python dict per
    query-candidate pair before ever touching pandas, which is fine for msgym's
    test fold (~4.5M pairs) but OOMs on the train fold (~194k queries x up to
    256 mass-candidates = ~49M pairs, over 5.4M of them distinct molecules).

    Two independent memory blowups have to be avoided here, not one:

    1. The query x candidate expansion (~49M rows) is streamed `chunk_size`
       queries at a time using vectorized pandas ops, casting repeated string
       columns to ``category`` dtype so the concatenated `spec_df` stays a
       small multiple of its row count instead of one Python string object
       per cell. `mol_id` is assigned as each candidate's position in the
       sorted canonical-SMILES list, so this needs no RDKit mol objects at all.
    2. `mol_df`'s RDKit property columns (including a live ``mol`` object per
       row) are NOT safe to build for the whole candidate universe at once:
       pandas' own `memory_usage(deep=True)` undercounts RDKit's native heap
       allocations, and holding ~5.4M live Mol objects in one Series is what
       actually OOMs (observed here around ~3.3M concurrent objects on a 31GB
       box, well before the "deep" memory estimate would predict trouble).
       So `mol_df` is built and pickled in `mol_batch_size`-molecule shards
       under `proc/mol_df_shards/`, discarding each batch before starting the
       next; concatenate the shards yourself once the whole candidate
       universe needs to be resident (e.g. at inference time, on a bigger box).
    """
    # 1. Select the retrieval spectra for this fold.
    spec_df = pd.read_csv(tsv_fp, sep="\t")
    required_columns = {
        "identifier",
        "smiles",
        "precursor_mz",
        "adduct",
        "instrument_type",
        "collision_energy",
        "fold",
        "simulation_challenge",
    }
    missing = required_columns - set(spec_df.columns)
    if missing:
        raise ValueError(f"Missing required MSGYM columns in {tsv_fp}: {sorted(missing)}")
    spec_df = spec_df[spec_df["fold"] == fold].copy()
    if spec_df.empty:
        raise ValueError(f"No MSGYM spectra found for fold {fold!r} in {tsv_fp}")
    if subset == "simulation_challenge":
        spec_df = spec_df[spec_df["simulation_challenge"]].copy()
    spec_df = spec_df.reset_index(drop=True)

    with candidates_fp.open() as f:
        candidates_by_smiles = json.load(f)
    if not isinstance(candidates_by_smiles, dict):
        raise ValueError(f"Candidate JSON must be a SMILES-to-list mapping: {candidates_fp}")

    query_smiles = spec_df["smiles"].unique()
    missing_queries = [s for s in query_smiles if s not in candidates_by_smiles]
    if missing_queries:
        raise ValueError(f"{len(missing_queries)} selected MSGYM queries are absent from JSON")

    # 2. Canonicalize the deduplicated candidate universe once, in parallel,
    # then build mol_df exactly as `build_proc_from_manifest` would.
    logger.info("Collecting unique candidate SMILES for %d queries", len(query_smiles))
    raw_candidates: set[str] = set()
    for s in query_smiles:
        raw_candidates.update(candidates_by_smiles[s])
    raw_candidates = sorted(raw_candidates)
    logger.info("Canonicalizing %d unique candidate SMILES", len(raw_candidates))
    canonical = data_utils.par_apply_series(pd.Series(raw_candidates), canonicalize_smiles)
    # `canonicalize_smiles` can silently return np.nan for a handful of raw
    # SMILES RDKit fails to parse (np.nan is truthy, so its own `if not
    # canonical_smiles` guard doesn't catch it). Drop those rather than let a
    # stray float crash the sort/dedup below or corrupt a multi-million-row run.
    invalid_mask = canonical.isna()
    n_invalid = int(invalid_mask.sum())
    if n_invalid:
        invalid_examples = [raw_candidates[i] for i in canonical.index[invalid_mask][:5]]
        logger.warning(
            "Dropping %d/%d candidate SMILES that failed to canonicalize (e.g. %s)",
            n_invalid,
            len(raw_candidates),
            invalid_examples,
        )
    raw_to_canonical = {
        raw: canon
        for raw, canon, invalid in zip(raw_candidates, canonical, invalid_mask)
        if not invalid
    }

    output_dp.mkdir(parents=True, exist_ok=True)
    proc_dp = output_dp / "proc"
    proc_dp.mkdir(parents=True, exist_ok=True)

    # mol_id is just the candidate's position in the sorted canonical list, so
    # it can be assigned without ever touching RDKit -- the expensive, memory-
    # heavy mol_df property computation (below) is deferred and decoupled from
    # the spec_df streaming that follows.
    candidate_smiles_sorted = sorted(set(raw_to_canonical.values()))
    smiles_to_mol_id = {smiles: i for i, smiles in enumerate(candidate_smiles_sorted)}
    candidate_smiles_fp = output_dp / "candidate_smiles.csv"
    pd.DataFrame({"smiles": candidate_smiles_sorted}).to_csv(candidate_smiles_fp, index=False)
    logger.info("Deduplicated to %d unique canonical candidate molecules", len(candidate_smiles_sorted))

    raw_to_mol_id = {raw: smiles_to_mol_id[canon] for raw, canon in raw_to_canonical.items()}
    known_invalid_raw = set(raw_candidates) - raw_to_mol_id.keys()

    # 3. Stream the query x candidate expansion. Each chunk covers whole
    # queries, so duplicates and grouping never span chunk boundaries.
    shared_empty_peaks: list = []
    chunk_frames: list[pd.DataFrame] = []
    n_chunks = (len(spec_df) + chunk_size - 1) // chunk_size
    n_dropped_dupes = 0
    n_dropped_invalid_candidates = 0
    for chunk_idx, start in enumerate(range(0, len(spec_df), chunk_size)):
        chunk = spec_df.iloc[start : start + chunk_size]
        inst_frag = chunk.apply(
            lambda row: data_utils.parse_inst_info(
                pd.Series(
                    {"inst_type": row["instrument_type"], "col_energy": row["collision_energy"]}
                )
            ),
            axis=1,
        )
        chunk = chunk.assign(
            inst_type=[t[0] for t in inst_frag],
            frag_mode=[t[1] for t in inst_frag],
            ion_mode=chunk["adduct"].map(_ion_mode_from_prec_type),
            candidates=chunk["smiles"].map(candidates_by_smiles),
        )
        exploded = chunk.explode("candidates", ignore_index=True)
        mol_id = exploded["candidates"].map(raw_to_mol_id)
        valid = mol_id.notna()
        if not valid.all():
            # These are exactly the raw candidates dropped above for failing
            # canonicalization; any other cause of a missing mapping is a bug.
            unmapped = set(exploded.loc[~valid, "candidates"]) - known_invalid_raw
            if unmapped:
                raise ValueError(f"Some manifest candidates could not be mapped to mol_df: {sorted(unmapped)[:5]}")
            n_dropped_invalid_candidates += int((~valid).sum())
            exploded = exploded.loc[valid].reset_index(drop=True)
            mol_id = mol_id.loc[valid].reset_index(drop=True)

        record = pd.DataFrame(
            {
                "source_identifier": exploded["identifier"].astype("category"),
                "source_smiles": exploded["smiles"].astype("category"),
                "prec_type": exploded["adduct"].astype("category"),
                "prec_mz": exploded["precursor_mz"].astype("float32"),
                "inst_type": exploded["inst_type"].astype("category"),
                "frag_mode": exploded["frag_mode"].astype("category"),
                "ace": exploded["collision_energy"].astype("float32"),
                "nce": np.float32("nan"),
                "ion_mode": exploded["ion_mode"].astype("category"),
                "spec_type": pd.Categorical(["MS2"] * len(exploded)),
                "dset": pd.Categorical(["msgym_1_5"] * len(exploded)),
                "peaks": [shared_empty_peaks] * len(exploded),
                "mol_id": mol_id.astype("int32"),
            }
        )
        before = len(record)
        record = record.drop_duplicates(subset=["source_identifier", "mol_id"], keep="first")
        n_dropped_dupes += before - len(record)
        chunk_frames.append(record)
        logger.info(
            "Processed chunk %d/%d: %d source queries -> %d query-candidate rows",
            chunk_idx + 1,
            n_chunks,
            len(chunk),
            len(record),
        )

    spec_df_full = pd.concat(chunk_frames, ignore_index=True, copy=False)
    del chunk_frames
    if n_dropped_dupes:
        logger.info("Removed %d duplicate query-candidate records", n_dropped_dupes)
    if n_dropped_invalid_candidates:
        logger.info(
            "Removed %d query-candidate rows pointing at uncanonicalizable candidates",
            n_dropped_invalid_candidates,
        )
    for column in _CATEGORY_COLUMNS:
        spec_df_full[column] = spec_df_full[column].astype("category")
    spec_df_full["spec_id"] = np.arange(len(spec_df_full), dtype="int32")
    spec_df_full["dset_spec_id"] = spec_df_full["spec_id"]
    spec_df_full["group_id"] = pd.factorize(spec_df_full["source_identifier"], sort=False)[
        0
    ].astype("int32")
    for ce_type in ("ace", "nce"):
        spec_df_full[f"{ce_type}_extra_1"] = np.float32(-1.0)
        spec_df_full[f"{ce_type}_extra_2"] = np.float32(-1.0)

    spec_fp = proc_dp / "spec_df.pkl"
    spec_df_full.to_pickle(spec_fp)
    split_fp = proc_dp / f"{fold}_ids.csv"
    spec_df_full[["spec_id", "mol_id", "group_id"]].to_csv(split_fp, index=False)
    logger.info(
        "Saved %d query-candidate spectra to %s",
        len(spec_df_full),
        spec_fp,
    )
    del spec_df_full

    # 4. Build mol_df's RDKit properties (including a live `mol` object per
    # row) in disk-backed batches, so peak memory is bounded by
    # `mol_batch_size` rather than the full candidate universe.
    mol_shard_dp = proc_dp / "mol_df_shards"
    mol_shard_dp.mkdir(parents=True, exist_ok=True)
    n_mol_batches = (len(candidate_smiles_sorted) + mol_batch_size - 1) // mol_batch_size
    for batch_idx, start in enumerate(range(0, len(candidate_smiles_sorted), mol_batch_size)):
        shard_fp = mol_shard_dp / f"part-{batch_idx:05d}.pkl"
        if shard_fp.exists():
            logger.info("Skipping existing mol_df shard %d/%d: %s", batch_idx + 1, n_mol_batches, shard_fp)
            continue
        batch_smiles = candidate_smiles_sorted[start : start + mol_batch_size]
        batch_df = pd.DataFrame(
            {"smiles": batch_smiles, "mol_id": range(start, start + len(batch_smiles))}
        )
        batch_df["mol"] = data_utils.par_apply_series(
            batch_df["smiles"],
            partial(data_utils.mol_from_smiles, canonicalize_tautomers=False),
            use_tqdm=False,
        )
        invalid_smiles = batch_df.loc[batch_df["mol"].isna(), "smiles"].tolist()
        if invalid_smiles:
            raise ValueError(
                f"Could not build molecules for {len(invalid_smiles)} candidate SMILES "
                f"that canonicalized cleanly: {invalid_smiles[:5]}"
            )
        data_utils.compute_mol_df_props(batch_df, use_tqdm=False)
        batch_df.to_pickle(shard_fp)
        logger.info(
            "Built mol_df shard %d/%d (%d molecules) -> %s",
            batch_idx + 1,
            n_mol_batches,
            len(batch_df),
            shard_fp,
        )
        del batch_df

    logger.info(
        "Saved %d unique candidate molecules across %d shards under %s",
        len(candidate_smiles_sorted),
        n_mol_batches,
        mol_shard_dp,
    )
    return mol_shard_dp, spec_fp, split_fp


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--tsv_fp", type=Path, required=True)
    export_parser.add_argument("--candidates_fp", type=Path, required=True)
    export_parser.add_argument("--output_dp", type=Path, required=True)
    export_parser.add_argument(
        "--subset",
        choices=["simulation_challenge", "all"],
        default="simulation_challenge",
        help="Use simulation-challenge spectra (default) or all spectra in --fold.",
    )
    export_parser.add_argument(
        "--fold",
        choices=["train", "val", "test"],
        default="test",
        help="MSGYM fold to export spectra from (default: test).",
    )

    proc_parser = subparsers.add_parser(
        "build-proc",
        help="Convert query_candidates.pkl.gz to mol_df.pkl and spec_df.pkl.",
    )
    proc_parser.add_argument("--manifest_fp", type=Path, required=True)
    proc_parser.add_argument("--output_dp", type=Path, required=True)
    proc_parser.add_argument(
        "--split_name",
        default="test",
        help="Name used for the emitted split file, e.g. '{split_name}_ids.csv'.",
    )

    large_parser = subparsers.add_parser(
        "build-proc-large",
        help="Fused, chunked export + build-proc for folds too large to explode in memory "
        "(e.g. the msgym train fold).",
    )
    large_parser.add_argument("--tsv_fp", type=Path, required=True)
    large_parser.add_argument("--candidates_fp", type=Path, required=True)
    large_parser.add_argument("--output_dp", type=Path, required=True)
    large_parser.add_argument(
        "--subset",
        choices=["simulation_challenge", "all"],
        default="all",
        help="Use simulation-challenge spectra or all spectra in --fold (default: all).",
    )
    large_parser.add_argument(
        "--fold",
        choices=["train", "val", "test"],
        default="train",
        help="MSGYM fold to export spectra from (default: train).",
    )
    large_parser.add_argument(
        "--chunk_size",
        type=int,
        default=2000,
        help="Number of source queries to expand into candidate rows per chunk (default: 2000).",
    )
    large_parser.add_argument(
        "--mol_batch_size",
        type=int,
        default=200_000,
        help="Number of candidate molecules to build RDKit props for per mol_df shard "
        "(default: 200000). Lower this if a shard batch still OOMs.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = get_args()
    if args.command == "export":
        export_msgym_json(args.tsv_fp, args.candidates_fp, args.output_dp, args.subset, args.fold)
    elif args.command == "build-proc":
        build_proc_from_manifest(args.manifest_fp, args.output_dp, args.split_name)
    elif args.command == "build-proc-large":
        export_and_build_proc_chunked(
            args.tsv_fp,
            args.candidates_fp,
            args.output_dp,
            args.subset,
            args.fold,
            args.chunk_size,
            args.mol_batch_size,
        )


if __name__ == "__main__":
    main()
