"""Audit exact structural and cut-context refinements of fixed formula correspondences."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
import yaml
from rdkit import Chem, rdBase

from scripts.debug.debug_masster_candidate_correspondence import match_groups, sha256

LOGGER = logging.getLogger(__name__)
TIERS = ("formula", "connectivity", "cut_context")


def canonical_subgraph(mol: Chem.Mol, atoms: set[int], roles: dict[int, int] | None = None) -> str:
    """Canonical parent-graph descriptor, not a sanitized or optimized product ion.

    Structure omits H/stereo/isotopes. Context retains parent H counts and encodes atom roles.
    Parent aromaticity is preserved even when the subgraph opens an aromatic ring.
    """
    graph = Chem.RWMol()
    mapping = {}
    for idx in sorted(atoms):
        original = mol.GetAtomWithIdx(idx)
        atom = Chem.Atom(original.GetAtomicNum())
        atom.SetFormalCharge(original.GetFormalCharge())
        atom.SetIsAromatic(original.GetIsAromatic())
        atom.SetNoImplicit(True)
        if roles is not None:
            atom.SetNumExplicitHs(original.GetTotalNumHs())
            atom.SetAtomMapNum(roles[idx])
        mapping[idx] = graph.AddAtom(atom)
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in atoms and b in atoms:
            graph.AddBond(mapping[a], mapping[b], bond.GetBondType())
            graph.GetBondBetweenAtoms(mapping[a], mapping[b]).SetIsAromatic(bond.GetIsAromatic())
    return Chem.MolToSmiles(graph.GetMol(), canonical=True, isomericSmiles=False)


def structural_signatures(mol: Chem.Mol, atoms: tuple[int, ...], radius: int) -> tuple[str, str]:
    """Return exact retained connectivity and oriented boundary-neighborhood descriptors."""
    kept = set(atoms)
    structure = canonical_subgraph(mol, kept)
    boundary, anchors = [], set()
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (a in kept) != (b in kept):
            anchors.update((a, b))
            boundary.append((str(bond.GetBondType()), bond.IsInRing(), bond.GetIsConjugated()))
    context_atoms = set(anchors)
    frontier = set(anchors)
    for _ in range(radius):
        frontier = {
            neighbor.GetIdx()
            for idx in frontier
            for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors()
        } - context_atoms
        context_atoms.update(frontier)
    # Keep the full retained graph so attachment positions remain anchored within it.
    context_atoms.update(kept)
    roles = {idx: (1 if idx in kept else 2) + (2 if idx in anchors else 0) for idx in context_atoms}
    context = canonical_subgraph(mol, context_atoms, roles)
    return structure, json.dumps([context, sorted(boundary)], separators=(",", ":"))


def annotate_groups(
    frame: pd.DataFrame, mol: Chem.Mol, radius: int, memo: dict, catalog: dict
) -> pd.DataFrame:
    """Preserve all structural explanations; only singleton signature sets are unambiguous."""
    attrs = frame.attrs
    numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
    hydrogens = [a.GetTotalNumHs() for a in mol.GetAtoms()]
    if not np.array_equal(numbers, attrs["atom_numbers"]) or not np.array_equal(
        hydrogens, attrs["atom_hydrogens"]
    ):
        raise ValueError("Source molecule and annotated graph atom order disagree")
    bonds = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()}
    if bonds != {tuple(sorted(edge)) for edge in attrs["chemical_bonds"].T}:
        raise ValueError("Source molecule and annotated chemical bonds disagree")
    structures, contexts = [set() for _ in range(len(frame))], [set() for _ in range(len(frame))]
    for atoms, mode, shift, group in zip(
        attrs["path_atoms"],
        attrs["path_modes"],
        attrs["path_h_shifts"],
        attrs["path_group"],
        strict=True,
    ):
        atoms = tuple(sorted(atoms))
        if atoms not in memo:
            memo[atoms] = structural_signatures(mol, atoms, radius)
        structure, context = memo[atoms]
        channel_context = json.dumps([context, int(mode), int(shift)], separators=(",", ":"))
        skey = hashlib.sha256(structure.encode()).hexdigest()
        ckey = hashlib.sha256(channel_context.encode()).hexdigest()
        catalog[skey], catalog[ckey] = structure, channel_context
        structures[int(group)].add(skey)
        contexts[int(group)].add(ckey)
    result = frame.copy()
    # Detailed pathway metadata remain in the pinned extraction cache; avoid copying
    # large ragged arrays on every pandas row filter during the pair audit.
    result.attrs = {}
    result["structure_keys"] = [tuple(sorted(keys)) for keys in structures]
    result["context_keys"] = [tuple(sorted(keys)) for keys in contexts]
    return result


def strict_edges(left: pd.DataFrame, right: pd.DataFrame, tier: str) -> list[tuple[int, int]]:
    """Refine ion-formula edges without using measured intensity or predicted probabilities."""
    edges = match_groups(left, right, "ion")
    if tier == "formula":
        return edges
    if tier not in TIERS:
        raise ValueError("Unknown matching tier")
    structure_l, structure_r = left.structure_keys.tolist(), right.structure_keys.tolist()
    edges = [
        (a, b)
        for a, b in edges
        if len(structure_l[a]) == len(structure_r[b]) == 1 and structure_l[a] == structure_r[b]
    ]
    if tier == "connectivity":
        return edges
    context_l, context_r = left.context_keys.tolist(), right.context_keys.tolist()
    return [
        (a, b)
        for a, b in edges
        if len(context_l[a]) == len(context_r[b]) == 1 and context_l[a] == context_r[b]
    ]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine is undefined for zero norms; avoid counting absent peaks as agreement."""
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-12 else float("nan")


def edge_scores(left: pd.DataFrame, right: pd.DataFrame, edges: list) -> tuple:
    """Measured agreement and signed i1 allocation-residual agreement, retaining weak peaks."""
    if len(edges) < 2:
        return float("nan"), float("nan")
    a, b = np.asarray(edges).T
    yq, yd = left.target.to_numpy()[a], right.target.to_numpy()[b]
    rq, rd = yq - left.pred.to_numpy()[a], yd - right.pred.to_numpy()[b]
    return cosine(yq, yd), cosine(rq, rd)


def finite_mean(values: list) -> float:
    """Average defined controls only; keep unavailable comparisons missing."""
    values = np.asarray(values)
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else float("nan")


def pair_audit(left: pd.DataFrame, right: pd.DataFrame, cfg: dict, seed: int) -> list[dict]:
    """Compare all tiers with same-pair shuffles and edge-count-matched formula subsets."""
    records = []
    formula = strict_edges(left, right, "formula")
    for tier in TIERS:
        rng = np.random.default_rng(seed)
        edges = strict_edges(left, right, tier)
        a, b = np.asarray(edges, dtype=int).reshape(-1, 2).T
        yq, yd = left.target.to_numpy()[a], right.target.to_numpy()[b]
        rq, rd = yq - left.pred.to_numpy()[a], yd - right.pred.to_numpy()[b]
        measured, residual = edge_scores(left, right, edges)
        shuffles, subsets = [], []
        if len(edges) >= 2:
            for _ in range(cfg["control_repeats"]):
                order = rng.permutation(len(edges))
                perm = np.arange(len(edges))
                perm[order] = np.roll(order, 1)
                shuffles.append(
                    (
                        cosine(yq, yd[perm]),
                        cosine(rq, rd[perm]),
                        cosine(rq, yd[perm] - right.pred.to_numpy()[b]),
                    )
                )
                chosen = rng.choice(len(formula), len(edges), replace=False)
                subsets.append(edge_scores(left, right, [formula[k] for k in chosen]))
        record = {
            "tier": tier,
            "matches": len(edges),
            "query_intensity": float(yq.sum()),
            "two_strong": int(
                (
                    (yq >= cfg["reliable_intensity_fraction"])
                    & (yd >= cfg["reliable_intensity_fraction"])
                ).sum()
                >= 2
            ),
            "measured_cosine": measured,
            "residual_cosine": residual,
        }
        for k, name in enumerate(("measured", "residual")):
            record[name + "_shuffle"] = finite_mean([x[k] for x in shuffles])
            record[name + "_count_control"] = finite_mean([x[k] for x in subsets])
            record[name + "_minus_shuffle"] = record[name + "_cosine"] - record[name + "_shuffle"]
            record[name + "_minus_count"] = (
                record[name + "_cosine"] - record[name + "_count_control"]
            )
        # Post-hoc sanity controls: the original residual shuffle also moves the
        # donor prediction, so it can reward matched model errors without label information.
        zero_reference = cosine(rq, -right.pred.to_numpy()[b]) if len(edges) >= 2 else float("nan")
        record["residual_zero_reference"] = zero_reference
        record["residual_minus_zero_reference"] = residual - zero_reference
        record["residual_minus_label_only_shuffle"] = residual - finite_mean(
            [x[2] for x in shuffles]
        )
        records.append(record)
    return records


def summarize(records: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Average acquisitions within each query identity, then bootstrap identities equally."""
    metrics = [
        "matches",
        "query_intensity",
        "two_strong",
        "measured_cosine",
        "residual_cosine",
        "measured_minus_shuffle",
        "residual_minus_shuffle",
        "measured_minus_count",
        "residual_minus_count",
        "residual_zero_reference",
        "residual_minus_zero_reference",
        "residual_minus_label_only_shuffle",
    ]
    rows = []
    for keys, group in records.groupby(["novel", "kind", "tier"]):
        molecular = group.groupby("query_mol")[metrics].mean()
        for metric in metrics:
            values = molecular[metric].dropna().to_numpy()
            if len(values):
                rng = np.random.default_rng(cfg["seed"])
                boot = values[
                    rng.integers(len(values), size=(cfg["bootstrap_repeats"], len(values)))
                ].mean(1)
                lower, upper = np.quantile(boot, [0.025, 0.975])
            else:
                lower = upper = float("nan")
            rows.append(
                {
                    "novel": keys[0],
                    "kind": keys[1],
                    "tier": keys[2],
                    "metric": metric,
                    "query_identities": len(molecular),
                    "defined_identities": len(values),
                    "pairs": len(group),
                    "mean": finite_mean(values),
                    "ci_low": lower,
                    "ci_high": upper,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Run a training-only structural alignment screen with pinned inference artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])  # noqa: NPY002
    th.manual_seed(cfg["seed"])
    started = time.time()
    root = args.input
    extraction = json.loads((root / "extraction/provenance.json").read_text())
    if (
        extraction["status"] != "complete"
        or extraction["query_split"] != "train"
        or not extraction["retain_pathways"]
    ):
        raise ValueError("Requires complete training-only pathway extraction")
    if extraction["external_pairs_sha256"] != sha256(root / "pairs.csv"):
        raise ValueError("Pair manifest changed")
    run = {
        "status": "running",
        "config": cfg,
        "command": sys.argv,
        "device": "cpu",
        "started_unix": started,
        "source_sha256": sha256(__file__),
        "label_sanity_post_hoc": True,
        "config_sha256": sha256(args.config),
        "rdkit": rdBase.rdkitVersion,
        "numpy": np.__version__,
        "torch": th.__version__,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
        "extraction_sha256": sha256(root / "extraction/provenance.json"),
        "cache_sha256": sha256(root / "extraction/candidate_groups.pkl"),
        "pairs_sha256": sha256(root / "pairs.csv"),
    }
    (root / "audit_provenance.json").write_text(json.dumps(run, indent=2))
    (root / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    # 1. Check source identity and preserve every structural alternative at each peak.
    mol_input = extraction["inputs"]["mol_fp"]
    if sha256(mol_input["path"]) != mol_input["sha256"]:
        raise ValueError("Molecule source changed")
    molecules = pd.read_pickle(mol_input["path"]).set_index("mol_id").mol
    cache = pd.read_pickle(root / "extraction/candidate_groups.pkl")
    pairs = pd.read_csv(root / "pairs.csv")
    if not (pairs.nce.ge(cfg["nce_min"]) & pairs.nce.lt(cfg["nce_max"])).all():
        raise ValueError("Unexpected NCE")
    memo, catalog, annotated, ambiguity = {}, {}, {}, []
    for number, (sid, frame) in enumerate(cache.items()):
        mid = frame.attrs["mol_id"]
        annotated[sid] = annotate_groups(
            frame, molecules.loc[mid], cfg["boundary_radius"], memo.setdefault(mid, {}), catalog
        )
        group = annotated[sid]
        eligible = group[group.eligible]
        ambiguity.append(
            {
                "spec_id": sid,
                "mol_id": mid,
                "eligible": len(eligible),
                "unique_structure": int(eligible.structure_keys.map(len).eq(1).sum()),
                "unique_context": int(eligible.context_keys.map(len).eq(1).sum()),
            }
        )
        if number % 50 == 0:
            LOGGER.info("Structural signatures %d/%d spectra", number + 1, len(cache))
    pd.to_pickle(annotated, root / "structural_groups.pkl")
    pd.to_pickle(catalog, root / "signature_catalog.pkl")
    pd.DataFrame(ambiguity).to_csv(root / "ambiguity.csv", index=False)
    # 2. Fixed pairs, nested matching rules, coverage and label permutation controls.
    records, matches = [], []
    for number, row in enumerate(pairs.itertuples()):
        left, right = annotated[row.query_spec], annotated[row.donor_spec]
        for result in pair_audit(
            left, right, cfg, cfg["seed"] + int(row.query_spec) + int(row.donor_spec)
        ):
            records.append({**row._asdict(), **result})
        for tier in TIERS:
            for a, b in strict_edges(left, right, tier):
                matches.append(
                    {
                        "query_spec": row.query_spec,
                        "donor_spec": row.donor_spec,
                        "kind": row.kind,
                        "tier": tier,
                        "query_group": a,
                        "donor_group": b,
                    }
                )
        if number % 50 == 0:
            LOGGER.info("Pair controls %d/%d", number + 1, len(pairs))
    results = pd.DataFrame(records)
    results.to_csv(root / "pair_results.csv", index=False)
    pd.DataFrame(matches).to_csv(root / "structural_matches.csv", index=False)
    summary = summarize(results, cfg)
    summary.to_csv(root / "summary.csv", index=False)
    # 3. Apply the predeclared unfamiliar-neighbor gate; no model gain is claimed.
    decisions = {}
    for tier in TIERS[1:]:
        rows = summary[(summary.novel) & summary.kind.eq("neighbor") & summary.tier.eq(tier)]
        tests = rows[rows.metric.isin(["residual_minus_shuffle", "residual_minus_count"])]
        decisions[tier] = bool(
            len(tests) == 2
            and tests.defined_identities.ge(cfg["minimum_novel_identities"]).all()
            and tests.ci_low.gt(0).all()
        )
    run.update(
        status="complete",
        elapsed_seconds=time.time() - started,
        spectra=len(cache),
        query_identities=int(pairs.query_mol.nunique()),
        residual_screen_pass=decisions,
        summary_sha256=sha256(root / "summary.csv"),
        pair_results_sha256=sha256(root / "pair_results.csv"),
    )
    (root / "audit_provenance.json").write_text(json.dumps(run, indent=2))
    LOGGER.info("Complete: %s", decisions)


if __name__ == "__main__":
    main()
