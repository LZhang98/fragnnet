"""Select validation queries and training references without inspecting query spectra."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from scripts.debug.debug_masster_candidate_correspondence import sha256

LOGGER = logging.getLogger(__name__)


def retrieve_condition(
    query: pd.Series, pool: pd.DataFrame, fingerprints: dict, cfg: dict, rng: np.random.Generator
) -> tuple | None:
    """Select closest eligible chemistry and a mass-distance-matched random control."""
    window = max(cfg["mass_window_da"], cfg["mass_window_fraction"] * query.prec_mz)
    distance = (pool.prec_mz - query.prec_mz).abs()
    pool = pool[
        (distance <= window) & pool.formula.ne(query.formula) & pool.identity.ne(query.identity)
    ]
    if len(pool) < 6:
        return None
    similarities = np.array(
        DataStructs.BulkTanimotoSimilarity(
            fingerprints[query.mol_id], [fingerprints[mid] for mid in pool.mol_id]
        )
    )
    order = np.argsort(-similarities, kind="stable")
    neighbor = pool.iloc[order[0]]
    others = pool.iloc[order[3:]]
    others = others[
        ((others.prec_mz - query.prec_mz).abs() - abs(neighbor.prec_mz - query.prec_mz)).abs()
        <= cfg["control_mass_distance_tolerance_da"]
    ]
    if len(others) < 3:
        return None
    control = others.iloc[int(rng.integers(len(others)))]
    return neighbor, control, float(similarities[order[0]])


def main() -> None:
    """Freeze a fresh validation cohort, donor identities, and molecule-disjoint folds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((Path(cfg["census_dir"]) / "provenance.json").read_text())
    for r in provenance["inputs"].values():
        if sha256(r["path"]) != r["sha256"]:
            raise ValueError(f"Changed input: {r['path']}")
    # 1. Split before any metadata selection; measured intensities are never consulted.
    train_fp = Path(provenance["inputs"]["train_ids_fp"]["path"])
    val_fp = train_fp.with_name("val_ids.csv")
    train_ids, val_ids = set(pd.read_csv(train_fp).spec_id), set(pd.read_csv(val_fp).spec_id)
    if train_ids & val_ids:
        raise ValueError("Overlapping spectrum splits")
    spec = pd.read_pickle(provenance["inputs"]["spec_fp"]["path"]).drop(columns="peaks")
    spec = spec[spec.spec_id.isin(train_ids | val_ids)].copy()
    spec["prec_mz"] = pd.to_numeric(spec.prec_mz)
    spec = spec[
        spec.prec_type.eq("[M+H]+")
        & spec.inst_type.isin(["ORB", "ORB_F"])
        & spec.ce_type.eq("single")
        & spec.nce.notna()
        & spec.nce.ge(0)
    ].copy()
    mol = pd.read_pickle(provenance["inputs"]["mol_fp"]["path"]).set_index("mol_id")
    mol = mol.loc[sorted(spec.mol_id.unique())].copy()
    mol["identity"] = [Chem.MolToInchiKey(m).split("-")[0] for m in mol.mol]
    spec["identity"] = spec.mol_id.map(mol.identity)
    spec["formula"] = spec.mol_id.map(mol.formula)
    keys = ["identity", "inst_type", "frag_mode", "nce"]
    train = spec[spec.spec_id.isin(train_ids)].sort_values("spec_id").drop_duplicates(keys)
    val = spec[spec.spec_id.isin(val_ids)].sort_values("spec_id").drop_duplicates(keys)
    if set(train.identity) & set(val.identity):
        raise ValueError("Training and validation molecular connectivity identities overlap")
    val = val[val.nce.ge(cfg["nce_min"]) & val.nce.lt(cfg["nce_max"])].copy()
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = {mid: generator.GetFingerprint(m) for mid, m in mol.mol.items()}
    refs = [fingerprints[mid] for mid in train.drop_duplicates("identity").mol_id]
    representative = val.sort_values("mol_id").drop_duplicates("identity")
    maximum = {
        row.identity: max(DataStructs.BulkTanimotoSimilarity(fingerprints[row.mol_id], refs))
        for row in representative.itertuples()
    }
    representative = representative.assign(novel=representative.identity.map(maximum).le(0.4))
    strata = dict(tuple(train.groupby(["inst_type", "frag_mode", "nce"], sort=True)))
    rng = np.random.default_rng(cfg["seed"])
    pairs, cohort = [], []
    # 2. Sample identities first, then choose a feasible condition nearest NCE 60.
    for novel in [True, False]:
        pool = representative[representative.novel == novel].sort_values("identity")
        sample = pool.iloc[rng.permutation(len(pool))[: cfg["queries_per_novelty_group"]]]
        for number, row in enumerate(sample.itertuples()):
            conditions = val[val.identity.eq(row.identity)].copy()
            conditions["distance"] = abs(conditions.nce - cfg["preferred_nce"])
            conditions = conditions.sort_values(["distance", "nce", "inst_type", "spec_id"])
            record = {
                "identity": row.identity,
                "mol_id": int(row.mol_id),
                "novel": bool(novel),
                "max_train_tanimoto": maximum[row.identity],
                "fold": number % 2,
                "selected": False,
            }
            for _, q in conditions.iterrows():
                donors = strata.get((q.inst_type, q.frag_mode, q.nce))
                if donors is None:
                    continue
                result = retrieve_condition(q, donors, fingerprints, cfg, rng)
                if result is None:
                    continue
                neighbor, control, _similarity = result
                record.update(
                    selected=True,
                    spec_id=int(q.spec_id),
                    nce=float(q.nce),
                    inst_type=q.inst_type,
                    prec_mz=float(q.prec_mz),
                    mol_id=int(q.mol_id),
                )
                for kind, d in [("neighbor", neighbor), ("control", control)]:
                    sim = DataStructs.TanimotoSimilarity(
                        fingerprints[q.mol_id], fingerprints[d.mol_id]
                    )
                    pairs.append(
                        {
                            "query_spec": int(q.spec_id),
                            "donor_spec": int(d.spec_id),
                            "query_mol": int(q.mol_id),
                            "donor_mol": int(d.mol_id),
                            "kind": kind,
                            "novel": bool(novel),
                            "query_max_tanimoto": maximum[row.identity],
                            "tanimoto": sim,
                            "nce": float(q.nce),
                            "inst_type": q.inst_type,
                            "query_prec_mz": float(q.prec_mz),
                            "donor_prec_mz": float(d.prec_mz),
                            "fold": number % 2,
                            "identity": row.identity,
                        }
                    )
                break
            cohort.append(record)
        LOGGER.info("Selected stratum novel=%s", novel)
    pd.DataFrame(pairs).to_csv(args.output / "gate_pairs.csv", index=False)
    pd.DataFrame(cohort).to_csv(args.output / "cohort.csv", index=False)
    inputs = dict(provenance["inputs"])
    inputs["val_ids_fp"] = {"path": str(val_fp), "sha256": sha256(val_fp)}
    (args.output / "selection_provenance.json").write_text(
        json.dumps(
            {
                "config": cfg,
                "source_sha256": sha256(__file__),
                "inputs": inputs,
                "pairs_sha256": sha256(args.output / "gate_pairs.csv"),
                "cohort_sha256": sha256(args.output / "cohort.csv"),
                "eligible_train_identities": train.identity.nunique(),
                "eligible_val_identities": val.identity.nunique(),
                "available_novel_queries": int(representative.novel.sum()),
            },
            indent=2,
        )
    )
    (args.output / "selection_source.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "config.yaml").write_bytes(Path(args.config).read_bytes())
    LOGGER.info(
        "Cohort: %s", pd.DataFrame(cohort).groupby("novel").selected.agg(["sum", "size"]).to_dict()
    )


if __name__ == "__main__":
    main()
