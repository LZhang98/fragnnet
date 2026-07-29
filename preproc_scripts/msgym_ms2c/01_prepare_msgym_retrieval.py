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
from pathlib import Path

import pandas as pd

from fragnnet.utils import data_utils

logger = logging.getLogger(__name__)


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
    spec_df = spec_df[spec_df["fold"] == "test"].copy()
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
        help="Use test-fold simulation-challenge spectra (default) or all test spectra.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = get_args()
    export_msgym_json(args.tsv_fp, args.candidates_fp, args.output_dp, args.subset)


if __name__ == "__main__":
    main()
