"""Dataset-filter helpers for oracle/eval diagnostics."""

from __future__ import annotations

import argparse
import os

NIST20_DSETS = ["nist20_hr"]
NIST23_DSETS = ["nist23_hr1", "nist23_hr2"]


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
