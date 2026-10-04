"""Audit fragment-pair exports against independently saved i1 spectral predictions.

Checks all spectrum identities, base/winner m/z, probability shares, outcome membership,
and JSONL group completeness. Does not fit any correction or infer a true fragment label.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def audit(output: Path, reference: Path) -> dict:
    """Compare the entire export to saved predictions and its own candidate-group rows."""
    pairs = pd.read_csv(output / "pairs.csv").set_index("spec_id")
    saved = pd.read_pickle(reference)["spectra"].set_index("spec_id")
    if not pairs.index.is_unique or not saved.index.is_unique:
        raise ValueError("duplicate spectrum IDs")
    if len(pairs.index.difference(saved.index)):
        raise ValueError("export contains unknown spectra")
    max_share_error = 0.0
    differences = []
    for sid, row in pairs.iterrows():
        ref = saved.loc[sid]
        if row.mol_id != ref.mol_id:
            raise ValueError(f"molecule mismatch: {sid}")
        pm, pi = np.asarray(ref.pred_mz, float), np.asarray(ref.pred_int, float)
        tm, ti = np.asarray(ref.true_mz, float), np.asarray(ref.true_int, float)
        pi, ti = pi / pi.sum(), ti / ti.sum()
        mass, inv = np.unique(np.round(pm, 4), return_inverse=True)
        intensity = np.bincount(inv, weights=pi)
        winner = int(np.argmax(intensity))
        base = int(np.argmax(ti))
        tol = row.tolerance_rel * max(tm[base], row.tolerance_min_mz)
        compatible = np.abs(pm - tm[base]) <= tol
        outcome = (
            "uncovered"
            if not compatible.any()
            else "hit"
            if abs(mass[winner] - tm[base]) <= tol
            else "under_ranked"
        )
        # Targets and identities must agree exactly; predictions are freshly rerun and
        # historical candidate-set differences are reported, never silently tolerated.
        np.testing.assert_allclose(row.target_mz, tm[base], rtol=0, atol=1e-7)
        np.testing.assert_allclose(row.target_measured_share, ti[base], rtol=1e-4, atol=3e-6)
        at_new_winner = np.abs(tm - row.winner_mz) <= row.tolerance_rel * np.maximum(
            tm, row.tolerance_min_mz
        )
        np.testing.assert_allclose(
            [row.winner_measured_max_share, row.winner_measured_sum_share],
            [ti[at_new_winner].max() if at_new_winner.any() else 0, ti[at_new_winner].sum()],
            rtol=1e-4,
            atol=3e-6,
        )
        actual = np.array([row.target_predicted_group_share, row.winner_predicted_share])
        expected = np.array([pi[compatible].sum(), intensity[winner]])
        error = float(np.max(np.abs(actual - expected)))
        max_share_error = max(max_share_error, error)
        differences.append(
            {
                "spec_id": sid,
                "max_share_error": error,
                "shares_within_tolerance": bool(
                    np.allclose(actual, expected, rtol=1e-4, atol=3e-6)
                ),
                "winner_diff": bool(abs(row.winner_mz - mass[winner]) > 1e-7),
                "outcome_diff": outcome != row.outcome,
                "target_channel_count_diff": int(row.target_channel_count - compatible.sum()),
                "winner_channel_count_diff": int(row.winner_channel_count - (inv == winner).sum()),
                "reference_winner_mz": float(mass[winner]),
                "reference_outcome": outcome,
            }
        )

    # Group audits use every emitted JSONL row, independent of the inference summary counters.
    counts, shares, structures, identities = defaultdict(int), defaultdict(float), {}, set()
    channels_read = 0
    with (output / "channels.jsonl").open() as handle:
        for line in handle:
            ch = json.loads(line)
            sid = ch["spec_id"]
            row = pairs.loc[sid]
            identity = (sid, ch["channel_index"])
            if identity in identities or ch["mol_id"] != row.mol_id:
                raise ValueError("duplicate channel or wrong molecule")
            identities.add(identity)
            channels_read += 1
            is_target = abs(ch["mz"] - row.target_mz) <= row.tolerance_rel * max(
                row.target_mz, row.tolerance_min_mz
            )
            is_winner = np.round(ch["mz"], 4) == row.winner_mz
            if is_target != ch["target_compatible"] or is_winner != ch["winner_contributor"]:
                raise ValueError(f"incorrect group membership: {identity}")
            if not (is_target or is_winner):
                raise ValueError("irrelevant channel in export")
            for group, member in (("target", is_target), ("winner", is_winner)):
                if member:
                    key = (sid, group)
                    counts[key] += 1
                    shares[key] += ch["predicted_share"]
                    structures.setdefault(key, set()).add(tuple(ch["atom_indices"]))
    for sid, row in pairs.iterrows():
        winner_matches_base = abs(row.winner_mz - row.target_mz) <= row.tolerance_rel * max(
            row.target_mz, row.tolerance_min_mz
        )
        expected_outcome = (
            "uncovered"
            if not counts[(sid, "target")]
            else "hit"
            if winner_matches_base
            else "under_ranked"
        )
        if row.outcome != expected_outcome:
            raise ValueError(f"incorrect internal outcome: {sid}")
        for group, share_field in (
            ("target", "target_predicted_group_share"),
            ("winner", "winner_predicted_share"),
        ):
            key = (sid, group)
            if counts[key] != row[f"{group}_channel_count"]:
                raise ValueError(f"incomplete group: {key}")
            if len(structures.get(key, set())) != row[f"{group}_structure_count"]:
                raise ValueError(f"structure count mismatch: {key}")
            np.testing.assert_allclose(shares[key], row[share_field], rtol=1e-12, atol=1e-14)
    difference_table = pd.DataFrame(differences)
    difference_table.to_csv(output / "reference_differences.csv", index=False)
    mismatches = {
        "share_tolerance_failures": int((~difference_table.shares_within_tolerance).sum()),
        "changed_winners": int(difference_table.winner_diff.sum()),
        "changed_outcomes": int(difference_table.outcome_diff.sum()),
        "changed_target_channel_counts": int(
            (difference_table.target_channel_count_diff != 0).sum()
        ),
        "changed_winner_channel_counts": int(
            (difference_table.winner_channel_count_diff != 0).sum()
        ),
    }
    return {
        "status": "passed",
        "status_scope": "internal export consistency and measured-target reproduction",
        "reference_status": "differences_detected" if any(mismatches.values()) else "matched",
        "reference_differences": mismatches,
        "reference_share_quantiles": difference_table.max_share_error.quantile(
            [0.5, 0.99, 1]
        ).to_dict(),
        "spectra": len(pairs),
        "channel_rows": channels_read,
        "reference": str(reference),
        "max_reference_share_error": max_share_error,
        "reference_spectra": len(saved),
        "complete_reference_split": len(pairs) == len(saved),
    }


def main() -> None:
    """Write a machine-readable audit only after every assertion passes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dp", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output_dp, args.reference)
    (args.output_dp / "audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
