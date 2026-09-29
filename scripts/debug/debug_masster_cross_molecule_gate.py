"""Train-only census of structure-selected spectrum and neutral-loss neighbors.

Select donors without reading query targets. Compare each chemical neighbor against
mass/acquisition-matched random donors, excluding identity and formula matches.
This is an offline feasibility diagnostic, not an evaluation of model improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

LOGGER = logging.getLogger(__name__)
METRICS = ("fragment_cos", "loss_cos", "large_loss_cos", "top_loss_hit")


def sha256(path: str | Path) -> str:
    """Hash an input or source file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fragment_profile(peaks: list, precursor: float, exclusion: float) -> np.ndarray:
    """Return sorted (m/z, intensity) rows below the precursor exclusion region."""
    values = np.asarray(peaks, dtype=float).reshape(-1, 2)
    keep = np.isfinite(values).all(axis=1)
    keep &= (values[:, 0] > 0) & (values[:, 0] < precursor - exclusion)
    keep &= values[:, 1] > 0
    values = values[keep]
    if not len(values):
        return values
    masses, inverse = np.unique(values[:, 0], return_inverse=True)
    return np.column_stack([masses, np.bincount(inverse, weights=values[:, 1])])


def tolerance_cosine(left: np.ndarray, right: np.ndarray, tolerance: float) -> float:
    """Greedily match peak products one-to-one within an absolute Da tolerance.

    Profiles have shape (peaks, 2), ordered by mass. Empty profiles score zero.
    No intensity transform is applied. This is not the production binned cosine.
    """
    if not len(left) or not len(right):
        return 0.0
    low = np.searchsorted(right[:, 0], left[:, 0] - tolerance, side="left")
    high = np.searchsorted(right[:, 0], left[:, 0] + tolerance, side="right")
    matches = sorted(
        (
            (left[i, 1] * right[j, 1], i, j)
            for i in range(len(left))
            for j in range(low[i], high[i])
        ),
        reverse=True,
    )
    used_left, used_right = set(), set()
    dot = 0.0
    for product, i, j in matches:
        if i not in used_left and j not in used_right:
            dot += product
            used_left.add(i)
            used_right.add(j)
    return float(dot / (np.linalg.norm(left[:, 1]) * np.linalg.norm(right[:, 1])))


def compare_profiles(query: tuple, donor: tuple, tolerance: float) -> dict:
    """Compare absolute fragment masses, losses, larger losses, and the strongest loss."""
    q_frag, q_loss, q_large = query
    d_frag, d_loss, d_large = donor
    top_hit = float(
        len(q_loss) > 0
        and len(d_loss) > 0
        and abs(q_loss[np.argmax(q_loss[:, 1]), 0] - d_loss[np.argmax(d_loss[:, 1]), 0])
        <= tolerance
    )
    return {
        "fragment_cos": tolerance_cosine(q_frag, d_frag, tolerance),
        "loss_cos": tolerance_cosine(q_loss, d_loss, tolerance),
        "large_loss_cos": tolerance_cosine(q_large, d_large, tolerance),
        "top_loss_hit": top_hit,
    }


def summarize(pairs: pd.DataFrame, seed: int, repeats: int) -> pd.DataFrame:
    """Bootstrap paired gains by query molecule within prespecified diagnostic slices."""
    slices = {
        "all": np.ones(len(pairs), dtype=bool),
        "half_0": pairs.query_half == 0,
        "half_1": pairs.query_half == 1,
        "tanimoto_le_0.4": pairs.tanimoto <= 0.4,
        "tanimoto_0.4_0.6": (pairs.tanimoto > 0.4) & (pairs.tanimoto <= 0.6),
        "tanimoto_0.6_0.8": (pairs.tanimoto > 0.6) & (pairs.tanimoto <= 0.8),
        "tanimoto_gt_0.8": pairs.tanimoto > 0.8,
        "different_scaffold": ~pairs.same_scaffold,
        "globally_novel_le_0.4": pairs.query_max_tanimoto <= 0.4,
        "substantial_fragmentation": (pairs.query_fragment_fraction >= 0.2)
        & (pairs.donor_fragment_fraction >= 0.2)
        & (pairs.control_min_fragment_fraction >= 0.2),
        "top_neighbor": pairs.neighbor_rank == 1,
        "top_neighbor_le_0.4": (pairs.neighbor_rank == 1) & (pairs.tanimoto <= 0.4),
    }
    for half in (0, 1):
        slices[f"half_{half}_le_0.4"] = (pairs.query_half == half) & (pairs.tanimoto <= 0.4)
    for half in (0, 1):
        slices[f"half_{half}_globally_novel"] = (pairs.query_half == half) & (
            pairs.query_max_tanimoto <= 0.4
        )
    slices["globally_novel_substantial"] = (
        slices["globally_novel_le_0.4"] & slices["substantial_fragmentation"]
    )
    rng = np.random.default_rng(seed)
    rows = []
    for label, mask in slices.items():
        subset = pairs.loc[mask]
        if subset.empty:
            continue
        for metric in METRICS:
            by_molecule = subset.groupby("query_mol")[[metric, f"control_{metric}"]].mean()
            delta = (by_molecule[metric] - by_molecule[f"control_{metric}"]).to_numpy()
            boot = np.array(
                [rng.choice(delta, len(delta), replace=True).mean() for _ in range(repeats)]
            )
            rows.append(
                {
                    "slice": label,
                    "metric": metric,
                    "molecules": len(delta),
                    "pairs": len(subset),
                    "donors": subset.donor_mol.nunique(),
                    "neighbor": by_molecule[metric].mean(),
                    "control": by_molecule[f"control_{metric}"].mean(),
                    "delta": delta.mean(),
                    "ci_low": np.quantile(boot, 0.025),
                    "ci_high": np.quantile(boot, 0.975),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the frozen census and write per-pair scores and provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/analysis/masster_cross_molecule_gate.yaml")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    started = time.time()
    rng = np.random.default_rng(config["seed"])

    # 1. Freeze source identity before reading any spectra. Only train IDs are loaded.
    provenance = {
        "status": "running",
        "config": config,
        "command": sys.argv,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"])
        ).hexdigest(),
        "source_sha256": sha256(__file__),
        "config_sha256": sha256(args.config),
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": "cpu",
        "started_unix": started,
        "inputs": {
            key: {"path": config[key], "sha256": sha256(config[key])}
            for key in ("spec_fp", "mol_fp", "train_ids_fp")
        },
    }
    provenance_path = args.output / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2))
    train_ids = pd.read_csv(config["train_ids_fp"])
    spectra = pd.read_pickle(config["spec_fp"])
    spectra = spectra[spectra.spec_id.astype(int).isin(train_ids.spec_id)].copy()
    n_training_spectra = len(spectra)
    spectra["spec_id"] = spectra.spec_id.astype(int)
    spectra["prec_mz"] = pd.to_numeric(spectra.prec_mz)
    measured_single = spectra.ce_type.eq("single") & spectra.nce.notna() & (spectra.nce >= 0)
    spectra = spectra[
        measured_single & spectra.prec_type.eq("[M+H]+") & spectra.inst_type.isin(["OB", "OB_F"])
    ].copy()
    if not len(spectra):
        raise ValueError("no eligible training spectra")
    molecules = pd.read_pickle(config["mol_fp"])
    molecules = molecules[molecules.mol_id.isin(spectra.mol_id)].set_index("mol_id")
    molecules["identity"] = [Chem.MolToInchiKey(mol).split("-")[0] for mol in molecules.mol]
    if not molecules.identity.str.len().eq(14).all():
        raise ValueError("missing molecular identities")
    spectra["identity"] = spectra.mol_id.map(molecules.identity)
    spectra["formula"] = spectra.mol_id.map(molecules.formula)
    if spectra.frag_mode.isna().any():
        raise ValueError("missing fragmentation mode")
    spectra = (
        spectra.sort_values("spec_id")
        .drop_duplicates(["identity", "inst_type", "frag_mode", "nce"])
        .reset_index(drop=True)
    )
    identities = np.array(sorted(spectra.identity.unique()))
    rng.shuffle(identities)
    if len(identities) <= config["query_molecules"]:
        raise ValueError("not enough training identities for a separate donor pool")
    queries = set(identities[: config["query_molecules"]])
    half = {identity: i % 2 for i, identity in enumerate(identities[: config["query_molecules"]])}
    donor_frame = spectra[~spectra.identity.isin(queries)]
    strata = {
        key: rows.index.to_numpy()
        for key, rows in donor_frame.groupby(["inst_type", "frag_mode", "nce"], sort=True)
    }
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = {mid: generator.GetFingerprint(mol) for mid, mol in molecules.mol.items()}
    donor_fingerprints = [
        fingerprints[mid] for mid in donor_frame.drop_duplicates("identity").mol_id
    ]
    max_similarity = {
        mid: max(DataStructs.BulkTanimotoSimilarity(fingerprints[mid], donor_fingerprints))
        for mid in spectra[spectra.identity.isin(queries)].mol_id.unique()
    }
    selected_queries = []
    for _, rows in spectra[spectra.identity.isin(queries)].groupby("identity", sort=True):
        rows = rows.sort_values(["inst_type", "frag_mode", "nce"])
        positions = np.linspace(
            0, len(rows) - 1, min(len(rows), config["conditions_per_query"]), dtype=int
        )
        selected_queries.extend(rows.index[positions].tolist())
    LOGGER.info(
        "Training rows %d; eligible distinct conditions %d; query identities %d; donor identities %d",
        n_training_spectra,
        len(spectra),
        len(queries),
        donor_frame.identity.nunique(),
    )

    # 2. Build profiles lazily. Neighbor selection never reads these target arrays.
    cache = {}
    fragment_fractions = {}

    def profiles(index: int) -> tuple:
        if index not in cache:
            row = spectra.loc[index]
            fragment = fragment_profile(row.peaks, row.prec_mz, config["precursor_exclusion_da"])
            raw = np.asarray(row.peaks, dtype=float).reshape(-1, 2)
            valid = np.isfinite(raw).all(axis=1) & (raw[:, 1] > 0)
            total = raw[valid, 1].sum()
            fragment_fractions[index] = float(fragment[:, 1].sum() / total) if total > 0 else 0.0
            loss = fragment.copy()[::-1]
            loss[:, 0] = row.prec_mz - loss[:, 0]
            cache[index] = (fragment, loss, loss[loss[:, 0] >= config["large_loss_min_da"]])
        return cache[index]

    records, coverage = [], []
    for number, index in enumerate(selected_queries):
        query = spectra.loc[index]
        candidates = strata.get(
            (query.inst_type, query.frag_mode, query.nce), np.array([], dtype=int)
        )
        pool = spectra.loc[candidates]
        window = max(config["mass_window_da"], config["mass_window_fraction"] * query.prec_mz)
        distances = (pool.prec_mz - query.prec_mz).abs()
        pool = pool[(distances <= window) & pool.formula.ne(query.formula)]
        support = {
            "query_spec": int(query.spec_id),
            "query_mol": int(query.mol_id),
            "eligible_donors": len(pool),
            "scored_neighbors": 0,
        }
        if len(pool) >= config["controls"] + config["neighbors"]:
            similarity = np.array(
                DataStructs.BulkTanimotoSimilarity(
                    fingerprints[query.mol_id], [fingerprints[mid] for mid in pool.mol_id]
                )
            )
            ordering = np.argsort(-similarity, kind="stable")
            neighbor_indices = pool.index.to_numpy()[ordering[: config["neighbors"]]]
            for rank, position in enumerate(ordering[: config["neighbors"]], start=1):
                donor_index = int(pool.index[position])
                donor = spectra.loc[donor_index]
                donor_distance = abs(donor.prec_mz - query.prec_mz)
                control_pool = pool[~pool.index.isin(neighbor_indices)]
                mass_difference = (
                    (control_pool.prec_mz - query.prec_mz).abs() - donor_distance
                ).abs()
                control_pool = control_pool[
                    mass_difference <= config["control_mass_distance_tolerance_da"]
                ]
                if len(control_pool) < config["controls"]:
                    continue
                controls = rng.choice(
                    control_pool.index.to_numpy(), config["controls"], replace=False
                )
                scores = compare_profiles(
                    profiles(index), profiles(donor_index), config["peak_tolerance_da"]
                )
                control_scores = [
                    compare_profiles(profiles(index), profiles(int(c)), config["peak_tolerance_da"])
                    for c in controls
                ]
                q_scaffold = molecules.loc[query.mol_id, "scaffold"]
                d_scaffold = molecules.loc[donor.mol_id, "scaffold"]
                records.append(
                    {
                        "query_spec": int(query.spec_id),
                        "query_mol": int(query.mol_id),
                        "query_half": half[query.identity],
                        "donor_spec": int(donor.spec_id),
                        "donor_mol": int(donor.mol_id),
                        "inst_type": query.inst_type,
                        "frag_mode": query.frag_mode,
                        "nce": query.nce,
                        "neighbor_rank": rank,
                        "tanimoto": similarity[position],
                        "same_scaffold": q_scaffold == d_scaffold,
                        "query_max_tanimoto": max_similarity[query.mol_id],
                        "query_fragment_fraction": fragment_fractions[index],
                        "donor_fragment_fraction": fragment_fractions[donor_index],
                        "control_min_fragment_fraction": min(
                            fragment_fractions[int(c)] for c in controls
                        ),
                        "mass_distance": donor_distance,
                        "control_mass_distance": float(
                            (spectra.loc[controls].prec_mz - query.prec_mz).abs().mean()
                        ),
                        "controls": [int(c) for c in spectra.loc[controls].spec_id],
                        "control_tanimoto": float(
                            np.mean(
                                DataStructs.BulkTanimotoSimilarity(
                                    fingerprints[query.mol_id],
                                    [fingerprints[mid] for mid in spectra.loc[controls].mol_id],
                                )
                            )
                        ),
                        "query_empty": len(profiles(index)[0]) == 0,
                        "query_large_empty": len(profiles(index)[2]) == 0,
                        **scores,
                        **{
                            f"control_{metric}": np.mean([s[metric] for s in control_scores])
                            for metric in METRICS
                        },
                    }
                )
                support["scored_neighbors"] += 1
        coverage.append(support)
        if (number + 1) % 500 == 0:
            LOGGER.info(
                "Queries %d/%d; scored neighbor pairs %d",
                number + 1,
                len(selected_queries),
                len(records),
            )
    if not records:
        raise ValueError("no query has enough matched controls")

    # 3. Molecule-clustered uncertainty and full paired audit trail.
    pairs = pd.DataFrame(records)
    pairs.to_pickle(args.output / "pairs.pkl")
    pd.DataFrame(coverage).to_csv(args.output / "coverage.csv", index=False)
    summary = summarize(pairs, config["seed"], config["bootstrap_repeats"])
    summary.to_csv(args.output / "summary.csv", index=False)
    provenance.update(
        {
            "status": "complete",
            "elapsed_seconds": time.time() - started,
            "training_rows": n_training_spectra,
            "eligible_conditions": len(spectra),
            "query_identities": len(queries),
            "query_conditions": len(selected_queries),
            "donor_identities": int(donor_frame.identity.nunique()),
            "scored_pairs": len(pairs),
            "scored_query_molecules": int(pairs.query_mol.nunique()),
        }
    )
    provenance_path.write_text(json.dumps(provenance, indent=2))
    LOGGER.info(
        "Complete in %.1fs\n%s",
        time.time() - started,
        summary[summary["slice"].isin(["all", "tanimoto_le_0.4"])].to_string(index=False),
    )


if __name__ == "__main__":
    main()
