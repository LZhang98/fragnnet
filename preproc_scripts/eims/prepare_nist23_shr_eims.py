from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

SUBSCRIPT_DIGITS = "₀₁₂₃₄₅₆₇₈₉"
SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
TRANSLATION_TABLE = str.maketrans(
    {c: str(i) for i, c in enumerate(SUBSCRIPT_DIGITS)}
    | {c: str(i) for i, c in enumerate(SUPERSCRIPT_DIGITS)}
    | {"⁺": "+", "⁻": "-"}
)


def normalize_chem_text(text: str) -> str:
    """Convert Unicode subscripts/superscripts to ASCII digits/signs."""

    return text.translate(TRANSLATION_TABLE)


def parse_peak_line(line: str) -> tuple[float, float]:
    """Parse a single peak line of the format: nominal_mz intensity annotation exact_mz."""

    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Peak line malformed: {line}")

    intensity = float(parts[1])

    mz_value: float
    try:
        mz_value = float(parts[-1])
    except ValueError:
        mz_value = float(parts[0])

    return mz_value, intensity


def parse_entry(dset_spec_id: str, lines: Iterable[str]) -> dict[str, object]:
    rows = [ln.strip() for ln in lines if ln.strip()]
    if len(rows) < 5:
        raise ValueError(f"Entry {dset_spec_id} has insufficient rows: {len(rows)}")

    name = rows[0]
    smiles = rows[1]
    formula = normalize_chem_text(rows[2])
    inchikey = rows[3]

    exact_mass_line = rows[4]
    if not exact_mass_line.lower().startswith("exact mass"):
        raise ValueError(f"Entry {dset_spec_id} missing exact mass line: {exact_mass_line}")
    exact_mass = float(exact_mass_line.split(":", 1)[1])

    peaks_raw = rows[5:]
    peaks_parsed = [parse_peak_line(p) for p in peaks_raw]
    peaks = "\n".join(f"{mz} {inten}" for mz, inten in peaks_parsed)

    return {
        "dset": "nist23_shr_ei",
        "dset_spec_id": dset_spec_id,
        "name": name,
        "title": name,
        "smiles": smiles,
        "formula": formula,
        "inchikey": inchikey,
        "exact_mass": exact_mass,
        "prec_mz": exact_mass,
        "prec_type": "[M]+",
        "spec_type": "MS1",
        "ion_mode": "P",
        "ion_type": "EI",
        "frag_mode": "EI",
        "col_energy": "70 eV",
        "inst_type": "GC-MS",
        "peaks": peaks,
        "num_peaks": len(peaks_parsed),
    }


def build_dataframe(zip_path: Path, max_files: int | None = None) -> pd.DataFrame:
    entries: list[dict[str, object]] = []
    with ZipFile(zip_path) as zf:
        names = sorted(zf.namelist())
        if max_files is not None:
            names = names[:max_files]

        for name in names:
            with zf.open(name) as handle:
                lines = handle.read().decode("utf-8", errors="replace").splitlines()
                try:
                    entry = parse_entry(dset_spec_id=name.removesuffix(".txt"), lines=lines)
                except ValueError as exc:
                    logging.warning("Skipping %s in %s: %s", name, zip_path, exc)
                    continue
                entries.append(entry)

    spec_df = pd.DataFrame(entries)
    # Note: `build_dataframes` will overwrite `spec_id` when combining multiple archives;
    # we still set it here so `build_dataframe` can be used standalone.
    spec_df.loc[:, "spec_id"] = range(len(spec_df))

    optional_cols = [
        "mw",
        "cas_num",
        "rating",
        "ri",
        "inst",
        "notes",
        "col_energy_extra_1",
        "col_energy_extra_2",
        "ramped",
        "stepped",
        "normalized",
    ]
    for col in optional_cols:
        if col not in spec_df.columns:
            spec_df.loc[:, col] = pd.NA

    spec_df.loc[:, ["ramped", "stepped", "normalized"]] = False
    return spec_df


def build_dataframes(zip_paths: list[Path], max_files: int | None = None) -> pd.DataFrame:
    """Parse multiple zip archives and concatenate into one dataframe."""

    df_list: list[pd.DataFrame] = []
    for zip_path in zip_paths:
        logging.info("Reading zip from %s", zip_path)
        df_list.append(build_dataframe(zip_path, max_files=max_files))

    if not df_list:
        raise ValueError("No zip files provided")

    combined = pd.concat(df_list, ignore_index=True)
    combined.loc[:, "spec_id"] = range(len(combined))
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NIST23 SHR EI TXT zip to stage-1 dataframe."
    )
    parser.add_argument("--zip_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dp", type=Path, default=Path("data/df"))
    parser.add_argument("--output_name", type=str, default="nist23_shr_ei")
    parser.add_argument("--max_files", type=int, default=None, help="Limit files for debugging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    spec_df = build_dataframes(list(args.zip_paths), max_files=args.max_files)

    args.output_dp.mkdir(parents=True, exist_ok=True)
    output_fp = args.output_dp / f"{args.output_name}_df.csv"
    logging.info("Saving %d spectra to %s", len(spec_df), output_fp)
    spec_df.to_csv(output_fp, index=False)


if __name__ == "__main__":
    main()
