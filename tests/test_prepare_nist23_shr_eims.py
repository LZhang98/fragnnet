"""Regression tests for EI peak formats with and without exact mass annotations."""

import pytest

from preproc_scripts.eims.prepare_nist23_shr_eims import parse_entry, parse_peak_line


@pytest.mark.parametrize(
    "line,expected",
    [
        ("43 999", (43.0, 999.0)),
        ("43 999 C2H3O", (43.0, 999.0)),
        ("43 999 C2H3O 43.0184", (43.0184, 999.0)),
        ("43 999 C2H3O N/A", (43.0, 999.0)),
    ],
)
def test_peak_mass_uses_exact_field_only_when_present(line: str, expected: tuple) -> None:
    assert parse_peak_line(line) == expected


def test_entry_preserves_unannotated_nominal_peaks() -> None:
    entry = parse_entry("example", [
        "ethanol", "CCO", "C2H6O", "example-key", "Exact Mass: 46.0419",
        "43 999", "46 100 C2H6O 46.0419",
    ])
    assert entry["peaks"] == "43.0 999.0\n46.0419 100.0"
    assert entry["num_peaks"] == 2
