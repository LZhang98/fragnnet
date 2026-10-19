"""Dataset-filter helpers for oracle/eval diagnostics."""

from __future__ import annotations

import argparse
import os

import pandas as pd

NIST20_DSETS = ["nist20_hr"]
NIST23_DSETS = ["nist23_hr1", "nist23_hr2"]
# Oracle CSV tier: above-precursor peaks dropped, no UWPR denoise.
DEFAULT_TIER = "1_above_prec"


def add_dsets_arg(parser: argparse.ArgumentParser, help_text: str) -> None:
    parser.add_argument(
        "--dsets",
        "--dset",
        nargs="+",
        default=None,
        help=f"{help_text} Default: infer from --spec_fp (NIST23 -> nist23_hr1 nist23_hr2).",
    )


def resolve_dsets_for_spec_fp(spec_fp: str, dsets: list[str] | None) -> list[str]:
    if dsets is not None:
        return dsets
    norm_spec_fp = os.path.normpath(spec_fp).lower()
    if "nist23" in norm_spec_fp:
        return NIST23_DSETS.copy()
    return NIST20_DSETS.copy()


def resolve_metric_col(df: pd.DataFrame, base: str, tier: str = DEFAULT_TIER) -> str:
    """Resolve a metric column from current tier-suffixed or legacy unsuffixed oracle CSVs."""
    suffixed = f"{base}__{tier}"
    if suffixed in df.columns:
        return suffixed
    if base in df.columns:
        return base
    raise KeyError(f"Missing column {base!r} or {suffixed!r}")
