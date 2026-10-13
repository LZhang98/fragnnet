"""Build MSGYM MS2C retrieval-candidate exports for every dataset fold.

Runs the export + build-proc pipeline from 01_prepare_msgym_retrieval.py once
per (candidate type, fold) pair, so train/val/test retrieval-candidate sets
exist side by side under a common output directory:

    <output_dp>/retrieval_candidates_<type>_<fold>/
        query_candidates.pkl.gz
        candidate_smiles.csv
        proc/
            mol_df.pkl
            spec_df.pkl
            <fold>_ids.csv

The MSGYM candidate JSONs are keyed by query SMILES across all folds, so this
works unmodified for train/val/test.

Usage:
    python preproc_scripts/msgym_ms2c/02_split_msgym_ms2c.py \
        --tsv_fp data/raw/msgym_1_5/MassSpecGym1.5.tsv \
        --mass_candidates_fp data/raw/msgym_1_5/MassSpecGym1.5_retrieval_candidates_mass.json \
        --formula_candidates_fp data/raw/msgym_1_5/MassSpecGym1.5_retrieval_candidates_formula.json \
        --output_dp data/ms2c/msgym/v1_5
"""

from __future__ import annotations

import argparse
import logging
from importlib import util
from pathlib import Path

logger = logging.getLogger(__name__)

_SCRIPT_FP = Path(__file__).resolve().parent / "01_prepare_msgym_retrieval.py"
_SPEC = util.spec_from_file_location("prepare_msgym_retrieval", _SCRIPT_FP)
assert _SPEC is not None and _SPEC.loader is not None
_PREPARE = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PREPARE)

FOLDS = ("train", "val", "test")


def split_msgym_ms2c(
    tsv_fp: Path,
    mass_candidates_fp: Path,
    formula_candidates_fp: Path,
    output_dp: Path,
    subset: str = "all",
    folds: tuple[str, ...] = FOLDS,
) -> None:
    """Export + build-proc retrieval candidates for the given folds and candidate types."""
    candidates_by_type = {
        "mass": mass_candidates_fp,
        "formula": formula_candidates_fp,
    }
    for cand_type, candidates_fp in candidates_by_type.items():
        for fold in folds:
            fold_dp = output_dp / f"retrieval_candidates_{cand_type}_{fold}"
            logger.info("Exporting %s candidates for fold=%s to %s", cand_type, fold, fold_dp)
            manifest_fp, _ = _PREPARE.export_msgym_json(
                tsv_fp, candidates_fp, fold_dp, subset, fold
            )
            _PREPARE.build_proc_from_manifest(manifest_fp, fold_dp / "proc", fold)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv_fp", type=Path, required=True)
    parser.add_argument("--mass_candidates_fp", type=Path, required=True)
    parser.add_argument("--formula_candidates_fp", type=Path, required=True)
    parser.add_argument("--output_dp", type=Path, required=True)
    parser.add_argument(
        "--subset",
        choices=["simulation_challenge", "all"],
        default="all",
        help="Use simulation-challenge spectra or all spectra in each fold (default: all).",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        choices=list(FOLDS),
        default=list(FOLDS),
        help="Folds to export (default: train val test). The train fold is large "
        "(~49M query-candidate rows for mass candidates) and may need chunking "
        "to avoid OOM; pass e.g. --folds test to skip it.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = get_args()
    split_msgym_ms2c(
        args.tsv_fp,
        args.mass_candidates_fp,
        args.formula_candidates_fp,
        args.output_dp,
        args.subset,
        tuple(args.folds),
    )


if __name__ == "__main__":
    main()
