"""Check connectivity-level HR-EI holdout filtering preserves the original partition rows."""

import pandas as pd
import pytest

from scripts.debug.ei.prepare_hr_ei_holdout_split import exclude_rows


def test_all_records_and_stereoisomers_with_excluded_connectivity_are_removed():
    molecules = pd.DataFrame({"mol_id": [1, 2, 3], "inchikey_s": ["SHARED", "SHARED", "RETAINED"]})
    original = pd.DataFrame(
        {"spec_id": [10, 11, 12, 13], "mol_id": [1, 2, 3, 1], "group_id": [4, 5, 6, 7]}
    )
    kept, removed = exclude_rows(original, molecules, {"SHARED"})
    assert kept.spec_id.tolist() == [12]
    assert kept.group_id.tolist() == [6]
    assert removed.spec_id.tolist() == [10, 11, 13]
    assert len(original) == 4


def test_split_with_unknown_molecule_fails():
    molecules = pd.DataFrame({"mol_id": [1], "inchikey_s": ["KNOWN"]})
    with pytest.raises(ValueError, match="unmapped"):
        exclude_rows(pd.DataFrame({"mol_id": [2]}), molecules, {"KNOWN"})
