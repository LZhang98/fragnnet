"""Matched, molecule-replicated success/failure contrasts for ambiguity-preserving pairs.

Requires --include_rival exports. This is an observational diagnostic on reused validation
molecules, not a ranking correction or an untouched estimate of model improvement. A descriptor
comparison is definite only when every compatible structure agrees on its direction. Each
molecule contributes its mean error once per stratum and direction. Opposing directions are
compared only on common support, with molecule-clustered uncertainty and disjoint molecule halves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

LOGGER = logging.getLogger(__name__)
ATTRIBUTES = (
    "cut_count",
    "atom_fraction",
    "abs_h_shift",
    "ring_cut",
    "breaks_aromatic",
    "loses_aromatic",
    "keeps_base",
    "is_precursor",
)
CUT_CLASSES = (
    "amide_CN",
    "ester_acylO",
    "alkyl_N",
    "aryl_N",
    "alkyl_O",
    "aryl_O",
    "alkyl_CC",
    "benzylic_CC",
    "acyl_CC",
    "N_N",
    "N_O",
    "sulfonyl_SN",
)
MATCH_COLUMNS = ["instrument", "energy_bin", "similarity_bin", "mass_bin", "margin_bin", "relation"]


def definite_direction(left: list[float], right: list[float]) -> float:
    """Return +1/-1 only for strictly separated ranges; equal constants give 0, else NaN."""
    if not left or not right:
        return np.nan
    if min(left) > max(right):
        return 1.0
    if max(left) < min(right):
        return -1.0
    if min(left) == max(left) == min(right) == max(right):
        return 0.0
    return np.nan


def atom_relation(left: frozenset, right: frozenset, n_atoms: int) -> str:
    """Compare retained atom sets; relationships are descriptive, not proven pathways."""
    if left == right:
        return "same_atoms"
    if left < right or right < left:
        return "nested"
    if not left & right:
        return "complementary" if len(left | right) == n_atoms else "disjoint"
    return "overlapping"


def summarize_groups(target: list[dict], rival: list[dict]) -> dict:
    """Keep consensus directions across all alternatives; never weight by model probability."""
    if not target or not rival:
        return {}
    t_sets = {frozenset(c["atom_indices"]) for c in target}
    r_sets = {frozenset(c["atom_indices"]) for c in rival}
    n_atoms = round(target[0]["atom_count"] / target[0]["atom_fraction"])
    relations = {atom_relation(t, r, n_atoms) for t in t_sets for r in r_sets}
    record = {
        "relation": next(iter(relations)) if len(relations) == 1 else "ambiguous",
        "single_structure_pair": len(t_sets) == len(r_sets) == 1,
        "fragment_only_pair": max(c["atom_fraction"] for c in target + rival) < 1,
    }
    signatures = []
    for group in (target, rival):
        sig = {tuple(sorted(c["cut_classes"])) for c in group}
        signatures.append("|".join(next(iter(sig))) or "precursor" if len(sig) == 1 else None)
    record["choice_key"] = " vs ".join(sorted(signatures)) if None not in signatures else None
    record["directed_choice_key"] = " > ".join(signatures) if None not in signatures else None
    for attr in ATTRIBUTES:
        values = []
        for group in (target, rival):
            if attr == "abs_h_shift":
                x = [abs(c["h_shift"]) for c in group]
            elif attr == "is_precursor":
                x = [int(c["atom_fraction"] == 1 and c["h_shift"] == 0) for c in group]
            else:
                x = [float(c[attr]) for c in group]
            values.append(x)
        record[attr] = definite_direction(*values)
    for name in CUT_CLASSES:
        values = [
            [float(any(cl.removeprefix("ring:") == name for cl in c["cut_classes"])) for c in group]
            for group in (target, rival)
        ]
        record["cut_" + name] = definite_direction(*values)
    return record


def matched_effect(table: pd.DataFrame, feature: str, columns: list[str]) -> dict:
    """Common-support error-rate difference (+ direction minus -), clustered by molecule.

    Average repeated spectra within molecule/stratum/direction. Exclude a molecule from a
    stratum if it occurs in both directions, so each contrast compares different molecules.
    Keep >=2 molecules per direction per stratum; overlap weights are min(n_plus, n_minus).
    Confidence intervals use a cluster sandwich with fixed common-support strata/weights.
    """
    selected = table[table[feature].isin([-1, 1])].dropna(subset=columns).copy()
    empty = {
        "difference": np.nan,
        "se": np.nan,
        "lo": np.nan,
        "hi": np.nan,
        "p": np.nan,
        "strata": 0,
        "molecules": 0,
        "rows": 0,
        "error_plus": np.nan,
        "error_minus": np.nan,
    }
    if selected.empty:
        return empty
    selected["stratum"] = selected.groupby(columns, observed=True).ngroup()
    cell = (
        selected.groupby(["stratum", "mol_id", feature], observed=True).failure.mean().reset_index()
    )
    both = cell.groupby(["stratum", "mol_id"], observed=True).size()
    both = set(both[both > 1].index)
    cell = cell[[((s, m) not in both) for s, m in zip(cell.stratum, cell.mol_id, strict=True)]]
    stats = cell.groupby(["stratum", feature], observed=True).failure.agg(["mean", "size"])
    sizes = stats["size"].unstack(feature).reindex(columns=[-1.0, 1.0]).fillna(0)
    valid = sizes.index[(sizes.min(axis=1) >= 2)]
    if not len(valid):
        return empty
    weights = sizes.loc[valid].min(axis=1)
    weights /= weights.sum()
    means = stats["mean"].unstack(feature).loc[valid]
    plus, minus = float(weights @ means[1.0]), float(weights @ means[-1.0])
    delta = plus - minus
    cell = cell[cell.stratum.isin(valid)].copy()
    joined = cell.join(stats, on=["stratum", feature])
    influence = (joined.failure - joined["mean"]) * joined[feature] / joined["size"]
    influence *= joined.stratum.map(weights)
    by_mol = influence.groupby(joined.mol_id).sum()
    n = len(by_mol)
    se = float(np.sqrt(np.square(by_mol).sum() * n / max(n - 1, 1)))
    p = float(2 * norm.sf(abs(delta) / se)) if se > 0 else (0.0 if delta else 1.0)
    return {
        "difference": delta,
        "se": se,
        "lo": delta - 1.96 * se,
        "hi": delta + 1.96 * se,
        "p": p,
        "strata": len(valid),
        "molecules": n,
        "rows": len(
            selected.merge(
                cell[["stratum", "mol_id", feature]], on=["stratum", "mol_id", feature], how="inner"
            )
        ),
        "error_plus": plus,
        "error_minus": minus,
    }


def bh_adjust(values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg over the full prespecified family; untestable entries count as p=1."""
    result = pd.Series(np.nan, index=values.index)
    finite = values.dropna().sort_values()
    if len(finite):
        q = finite.to_numpy() * len(values) / np.arange(1, len(finite) + 1)
        result.loc[finite.index] = np.minimum(1, np.minimum.accumulate(q[::-1])[::-1])
    return result


def build_table(source: Path, metadata: Path) -> pd.DataFrame:
    """Load the same-forward groups, verify counts/shares, and attach conditions by spec_id."""
    pairs = pd.read_pickle(source / "pairs.pkl").set_index("spec_id")
    meta = pd.read_pickle(metadata)
    if not pairs.index.is_unique or not meta.index.is_unique:
        raise ValueError("duplicate spectrum IDs")
    if not np.array_equal(pairs.mol_id.to_numpy(), meta.loc[pairs.index].mol_id.to_numpy()):
        raise ValueError("molecule metadata mismatch")
    pairs = pairs.join(meta[["novelty", "heavy_atoms"]], validate="one_to_one")
    if pairs[["novelty", "heavy_atoms"]].isna().any().any():
        raise ValueError("missing context metadata")
    rows = []
    current, target, rival = None, [], []
    with (source / "channels.jsonl").open() as handle:
        for line in handle:
            channel = json.loads(line)
            sid = channel["spec_id"]
            if sid != current and current is not None:
                rows.append(check_group_row(current, target, rival, pairs))
                target, rival = [], []
            current = sid
            if channel["target_compatible"]:
                target.append(channel)
            if channel["rival_contributor"]:
                rival.append(channel)
        if current is not None:
            rows.append(check_group_row(current, target, rival, pairs))
    features = pd.DataFrame(rows).set_index("spec_id")
    if not features.index.is_unique or set(features.index) != set(pairs.index):
        raise ValueError("channel groups do not cover exactly the spectrum table")
    table = pairs.join(features, validate="one_to_one")
    table["failure"] = (table.outcome == "under_ranked").astype(float)
    table["rival_target_ratio"] = table.rival_measured_max_share / table.target_measured_share
    table["energy_bin"] = np.digitize(table.nce, [20, 40, 60, 100], right=True)
    table["similarity_bin"] = np.digitize(table.novelty, [0.4, 0.6, 0.8])
    table["mass_bin"] = np.digitize(table.precursor_mz, [250, 400, 600])
    table["margin_bin"] = np.digitize(table.rival_target_ratio, [0.25, 0.75])
    table["mass_orientation"] = np.sign(table.target_mz - table.rival_mz)
    table["more_familiar"] = np.where(table.novelty >= 0.6, 1.0, -1.0)
    table["higher_energy"] = np.where(table.nce > 40, 1.0, -1.0)
    table["eligible"] = (
        (table.outcome != "uncovered")
        & table.rival_available
        & (table.target_tied_base_count == 1)
        & (table.rival_target_ratio < 1 - 1e-6)
    )
    # The split is by molecule only, selected without consulting outcomes or descriptors.
    ids = np.sort(table.mol_id.unique())
    shuffled = np.random.default_rng(20260930).permutation(ids)
    halves = {mid: int(i >= len(ids) // 2) for i, mid in enumerate(shuffled)}
    table["half"] = table.mol_id.map(halves)
    return table


def check_group_row(sid: int, target: list[dict], rival: list[dict], pairs: pd.DataFrame) -> dict:
    """Audit every target/rival group before constructing consensus descriptors."""
    row = pairs.loc[sid]
    for group, channels in (("target", target), ("rival", rival)):
        if len(channels) != row[f"{group}_channel_count"]:
            raise ValueError(f"incomplete {group} group: {sid}")
        key = "target_predicted_group_share" if group == "target" else "rival_predicted_share"
        np.testing.assert_allclose(
            sum(c["predicted_share"] for c in channels), row[key], rtol=1e-12, atol=1e-14
        )
    if {c["channel_index"] for c in target} & {c["channel_index"] for c in rival}:
        raise ValueError(f"target overlaps rival: {sid}")
    if any(c["mol_id"] != row.mol_id for c in target + rival):
        raise ValueError("wrong molecule in channel group")
    return {"spec_id": sid, **summarize_groups(target, rival)}


def analyze(table: pd.DataFrame, output: Path) -> None:
    """Report fixed descriptor contrasts, molecule-half replication, and sensitivity subsets."""
    eligible = table[table.eligible].copy()
    features = (
        list(ATTRIBUTES) + ["cut_" + c for c in CUT_CLASSES] + ["more_familiar", "higher_energy"]
    )
    records = []
    for mode in (
        "conditions",
        "same_choice",
        "single_structure",
        "stable_export",
        "fragment_only_conditions",
        "fragment_only_same_choice",
    ):
        for half in ("all", "0", "1"):
            subset = eligible if half == "all" else eligible[eligible.half == int(half)]
            if mode.startswith("fragment_only"):
                subset = subset[subset.fragment_only_pair == True]  # noqa: E712
            if mode == "stable_export":
                subset = subset[subset.stable_export]
            if mode == "single_structure":
                subset = subset[subset.single_structure_pair == True]  # noqa: E712
            for feature in features:
                columns = MATCH_COLUMNS.copy()
                if mode in (
                    "same_choice",
                    "single_structure",
                    "stable_export",
                    "fragment_only_same_choice",
                ):
                    columns.append(
                        "directed_choice_key"
                        if feature in ("more_familiar", "higher_energy")
                        else "choice_key"
                    )
                if feature in ("more_familiar", "higher_energy"):
                    columns.append("mass_orientation")
                if feature == "more_familiar":
                    columns.remove("similarity_bin")
                if feature == "higher_energy":
                    columns.remove("energy_bin")
                records.append(
                    {
                        "mode": mode,
                        "half": half,
                        "feature": feature,
                        **matched_effect(subset, feature, columns),
                    }
                )
    result = pd.DataFrame(records)
    result["q"] = result.groupby(["mode", "half"]).p.transform(bh_adjust)
    result.to_csv(output / "matched_contrasts.csv", index=False)
    strata = eligible.groupby(["relation", "similarity_bin", "energy_bin"], observed=True).agg(
        spectra=("failure", "size"), molecules=("mol_id", "nunique"), error_rate=("failure", "mean")
    )
    strata.to_csv(output / "descriptive_strata.csv")
    molecule_rates = eligible.groupby(["relation", "mol_id"]).failure.mean().reset_index()
    molecule_rates.groupby("relation").failure.agg(["count", "mean"]).to_csv(
        output / "relation_rates.csv"
    )
    selected = result[
        (result["mode"] == "same_choice")
        & (result.half == "0")
        & (result.q < 0.05)
        & (result.molecules >= 50)
    ]
    replication = selected[["feature", "difference", "q", "molecules"]].merge(
        result[(result["mode"] == "same_choice") & (result.half == "1")],
        on="feature",
        suffixes=("_discovery", "_replication"),
    )
    replication["replicated"] = (
        (np.sign(replication.difference_discovery) == np.sign(replication.difference_replication))
        & (replication.q_replication < 0.05)
        & (replication.molecules_replication >= 50)
    )
    replication.to_csv(output / "replication.csv", index=False)
    summary = {
        "spectra": len(table),
        "eligible": len(eligible),
        "eligible_molecules": eligible.mol_id.nunique(),
        "definite_choice_spectra": int(eligible.choice_key.notna().sum()),
        "single_structure_spectra": int(eligible.single_structure_pair.sum()),
        "replicated_features": replication.loc[replication.replicated, "feature"].tolist(),
        "scope": "observational associations on reused validation, not a model improvement",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    LOGGER.info("%s", summary)


def main() -> None:
    """Build the complete feature table, then run the prespecified matched contrasts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference_pairs", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    table = build_table(args.source, args.metadata)
    prior = pd.read_pickle(args.reference_pairs).set_index("spec_id").reindex(table.index)
    if prior.mol_id.isna().any() or not np.array_equal(prior.mol_id, table.mol_id):
        raise ValueError("prior export IDs differ")
    stable = np.ones(len(table), dtype=bool)
    for column in ("outcome", "target_channel_count", "winner_channel_count", "winner_mz"):
        stable &= (table[column] == prior[column]).to_numpy()
    for column in ("target_predicted_group_share", "winner_predicted_share"):
        stable &= np.isclose(table[column], prior[column], rtol=1e-4, atol=3e-6)
    table["stable_export"] = stable
    table.to_pickle(args.output / "features.pkl")
    table.to_csv(args.output / "features.csv")
    analyze(table, args.output)
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "source": str(args.source),
                "reference_pairs": str(args.reference_pairs),
                "metadata": str(args.metadata),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "seed": 20260930,
                "match_columns": MATCH_COLUMNS,
                "molecule_split": "random halves, all spectra of a molecule stay together",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
