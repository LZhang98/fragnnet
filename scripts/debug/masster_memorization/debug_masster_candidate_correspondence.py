"""Audit fixed formula correspondences between training molecules; no fitting.

Select one acquisition/neighbor/control triple per query without using intensities.
Reconstruct complete ion formulas from annotated MASSter pathways, verify their masses,
then match resolvable candidate groups by ion or nonnegative neutral-loss composition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
import yaml
from rdkit import Chem
from torch.utils.data import ConcatDataset, DataLoader, Subset

from fragnnet.inference.checkpoint import apply_inference_config, init_model_from_ckpt
from fragnnet.runner import load_config
from fragnnet.utils.formula_utils import PREC_TYPE_TO_CMF_COMP_DIFF, PREC_TYPE_TO_COMP_DIFF
from fragnnet.utils.setup_utils import get_dataset_cls

LOGGER = logging.getLogger(__name__)
TABLE = Chem.GetPeriodicTable()
ELEMENTS = 119
ELECTRON_MASS = 0.000548579909065


def sha256(path: str | Path) -> str:
    """Return a streaming file identity."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def formula_text(counts: np.ndarray) -> str:
    """Serialize the full elemental vector unambiguously, in atomic-number order."""
    return "".join(f"{TABLE.GetElementSymbol(z)}{int(n)}" for z, n in enumerate(counts) if n)


def reconstruct_formulas(batch: dict, output: dict, precursor_types: list[str]) -> tuple:
    """Return pathway ion formulas (C,119), precursor formulas (B,119), and mass error.

    This bounded audit supports [M+H]+ only. Formula alternatives retain every atom set;
    H shifts and CRF/CMF adduct composition follow the production model's constants.
    """
    graph = batch["mol_pyg"]
    numbers = graph.atom_atomic_numbers.cpu().numpy().astype(int)
    hydrogens = graph.atom_hydrogen_counts.cpu().numpy().astype(int)
    atoms = np.zeros((len(numbers), ELEMENTS), dtype=np.int64)
    atoms[np.arange(len(numbers)), numbers] = 1
    atoms[:, 1] += hydrogens
    group = graph.batch.cpu().numpy()
    precursor = np.zeros((int(batch["batch_size"]), ELEMENTS), dtype=np.int64)
    np.add.at(precursor, group, atoms)
    fragments = np.zeros((len(output["ion_h_fragment_batch_idxs"]), ELEMENTS), dtype=np.int64)
    np.add.at(
        fragments,
        output["fragment_atom_fragment_idxs"].cpu().numpy(),
        atoms[output["fragment_atom_indices"].cpu().numpy()],
    )
    ptypes = batch["spec_prec_type"].cpu().numpy().reshape(-1)
    if any(precursor_types[int(i)] != "[M+H]+" for i in ptypes):
        raise ValueError("This audit currently supports singly charged [M+H]+ only")
    precursor[:, 1] += 1
    path_frag = output["pred_pathway_fragment_idxs"].cpu().numpy()
    path_modes = output["pred_pathway_ion_mode_idxs"].cpu().numpy()
    if not np.isin(path_modes, [0, 1]).all():
        raise ValueError("Unknown ion channel")
    formulas = fragments[path_frag].copy()
    for mode, mapping in enumerate((PREC_TYPE_TO_COMP_DIFF, PREC_TYPE_TO_CMF_COMP_DIFF)):
        for element, delta in mapping["[M+H]+"].items():
            formulas[path_modes == mode, TABLE.GetAtomicNumber(element)] += delta
    formulas[:, 1] += output["pred_pathway_h_shifts"].cpu().numpy()
    if (formulas < 0).any():
        raise ValueError("Negative ion composition")
    masses = np.array([0.0] + [TABLE.GetMostCommonIsotopeMass(z) for z in range(1, ELEMENTS)])
    calculated = formulas @ masses - ELECTRON_MASS
    error = np.abs(calculated - output["pred_mzs"].cpu().numpy())
    if len(error) and error.max() > 2e-5:
        raise ValueError(f"Formula/mass disagreement: {error.max():.6g} Da")
    return formulas, precursor, float(error.max(initial=0.0))


def candidate_groups(
    mzs: np.ndarray,
    probs: np.ndarray,
    formulas: np.ndarray,
    precursor: np.ndarray,
    atom_sets: list[tuple],
    true_mzs: np.ndarray,
    true_ints: np.ndarray,
    precursor_mz: float,
    rel: float,
    min_mz: float,
    exclusion: float,
) -> pd.DataFrame:
    """Group identical candidate masses and conservatively reject unresolved alternatives.

    Candidate eligibility uses predictions/chemistry only. Overlapping instrument-tolerance
    windows are excluded. A measured peak must fit exactly one candidate window; intensity is
    never copied to multiple groups. Intensities are fractions of the full processed target.
    """
    masses, inverse = np.unique(np.round(mzs, 6), return_inverse=True)
    pred = np.bincount(inverse, weights=probs, minlength=len(masses))
    if pred.sum() > 0:
        pred /= pred.sum()
    widths = rel * np.maximum(masses, min_mz)
    resolvable = np.ones(len(masses), dtype=bool)
    close = np.diff(masses) <= widths[:-1] + widths[1:]
    resolvable[:-1] &= ~close
    resolvable[1:] &= ~close
    target = np.zeros(len(masses))
    for mass, intensity in zip(true_mzs, true_ints):
        eligible = np.flatnonzero(np.abs(masses - mass) <= rel * max(mass, min_mz))
        if len(eligible) == 1:
            target[eligible[0]] += intensity
    if true_ints.sum() > 0:
        target /= true_ints.sum()
    records = []
    for i, mass in enumerate(masses):
        paths = np.flatnonzero(inverse == i)
        alternatives = np.unique(formulas[paths], axis=0)
        unique_formula = len(alternatives) == 1
        eligible = bool(resolvable[i] and unique_formula and mass < precursor_mz - exclusion)
        ion = alternatives[0]
        loss = precursor - ion
        loss_valid = bool((loss >= 0).all() and loss.sum() > 0)
        records.append(
            {
                "group": i,
                "mz": mass,
                "native_mz": mzs[paths[0]],
                "pred": float(pred[i]),
                "target": float(target[i]),
                "eligible": eligible,
                "ion": formula_text(ion) if unique_formula else "",
                "loss": formula_text(loss) if unique_formula and loss_valid else "",
                "formula_alternatives": len(alternatives),
                "structural_alternatives": len({atom_sets[k] for k in paths}),
                "unresolved": not bool(resolvable[i]),
            }
        )
    frame = pd.DataFrame(records)
    frame.attrs.update(
        true_mzs=true_mzs.copy(), true_ints=true_ints.copy(), precursor_mz=precursor_mz
    )
    return frame


def match_groups(left: pd.DataFrame, right: pd.DataFrame, kind: str) -> list[tuple[int, int]]:
    """Match unique eligible formula keys, or the one-to-one union of both formula views.

    All mappings are selected without inspecting target intensities or predicted shares.
    Conflicts in the union are dropped instead of arbitrarily assigning a partner.
    """
    if kind == "union":
        edges = set(match_groups(left, right, "ion") + match_groups(left, right, "loss"))
        left_degree = Counter(a for a, _ in edges)
        right_degree = Counter(b for _, b in edges)
        return sorted((a, b) for a, b in edges if left_degree[a] == right_degree[b] == 1)
    if kind not in ("ion", "loss"):
        raise ValueError("Expected ion, loss or union")
    lhs = left[left.eligible & left[kind].ne("")]
    rhs = right[right.eligible & right[kind].ne("")]
    lhs = lhs[~lhs[kind].duplicated(keep=False)]
    rhs = rhs[~rhs[kind].duplicated(keep=False)]
    other = dict(zip(rhs[kind], rhs.group))
    return [
        (int(row.group), int(other[getattr(row, kind)]))
        for row in lhs.itertuples()
        if getattr(row, kind) in other
    ]


def select_pairs(
    census: pd.DataFrame, count: int, seed: int, condition: str = "lowest_spec_id"
) -> pd.DataFrame:
    """Select balanced novelty strata, one spectrum and nearest eligible donor per query."""
    rng = np.random.default_rng(seed)
    if condition == "lowest_spec_id":
        ordered = census.sort_values(["query_mol", "query_spec", "neighbor_rank"])
    elif condition == "highest_nce":
        ordered = census.sort_values(
            ["query_mol", "nce", "query_spec", "neighbor_rank"],
            ascending=[True, False, True, True],
        )
    else:
        raise ValueError("Unknown condition selection")
    representatives = ordered.drop_duplicates("query_mol")
    records = []
    for novel in (True, False):
        pool = representatives[(representatives.query_max_tanimoto <= 0.4) == novel]
        chosen = pool.iloc[rng.permutation(len(pool))[:count]]
        for row in chosen.itertuples():
            for kind, sid in [("neighbor", row.donor_spec), ("control", row.controls[0])]:
                records.append(
                    {
                        "query_spec": int(row.query_spec),
                        "donor_spec": int(sid),
                        "query_mol": int(row.query_mol),
                        "kind": kind,
                        "novel": novel,
                        "query_max_tanimoto": float(row.query_max_tanimoto),
                    }
                )
    return pd.DataFrame(records)


def summarize_pairs(pairs: pd.DataFrame, cache: dict, floor: float) -> tuple:
    """Summarize coverage per pair and save individual correspondences for follow-up."""
    summaries, matches = [], []
    for row in pairs.itertuples():
        left, right = cache[row.query_spec], cache[row.donor_spec]
        left_rows = list(left.itertuples(index=False))
        right_rows = list(right.itertuples(index=False))
        for view in ("ion", "loss", "union"):
            edges = match_groups(left, right, view)
            both_positive = reliable = either_reliable = 0
            for a, b in edges:
                q, d = left_rows[a], right_rows[b]
                both_positive += int(q.target > 0 and d.target > 0)
                reliable += int(q.target >= floor and d.target >= floor)
                either_reliable += int(q.target >= floor or d.target >= floor)
                matches.append(
                    {
                        "query_spec": row.query_spec,
                        "donor_spec": row.donor_spec,
                        "kind": row.kind,
                        "novel": row.novel,
                        "view": view,
                        "query_group": a,
                        "donor_group": b,
                        "query_ion": q.ion,
                        "donor_ion": d.ion,
                        "query_loss": q.loss,
                        "donor_loss": d.loss,
                        "query_target": q.target,
                        "donor_target": d.target,
                        "query_pred": q.pred,
                        "donor_pred": d.pred,
                    }
                )
            summaries.append(
                {
                    "query_spec": row.query_spec,
                    "donor_spec": row.donor_spec,
                    "query_mol": row.query_mol,
                    "kind": row.kind,
                    "novel": row.novel,
                    "view": view,
                    "matches": len(edges),
                    "both_positive": both_positive,
                    "both_reliable": reliable,
                    "either_reliable": either_reliable,
                    "two_reliable": reliable >= 2,
                    "query_covered_intensity": sum(left_rows[a].target for a, _ in edges),
                    "donor_covered_intensity": sum(right_rows[b].target for _, b in edges),
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(matches)


def main() -> None:
    """Run bounded, reproducible annotated inference and the deterministic matching audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/analysis/masster_candidate_correspondence.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs-csv", type=Path, help="Preselected query/training-reference pairs")
    parser.add_argument("--query-split", choices=("train", "val"), default="train")
    parser.add_argument(
        "--retain-pathways",
        action="store_true",
        help="Save all local atom sets and channels for structural audits",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())
    started = time.time()
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])  # noqa: NPY002
    th.manual_seed(cfg["seed"])
    th.set_num_threads(cfg["threads"])
    args.output.mkdir(parents=True, exist_ok=True)
    # 1. Pin sources, then select pairs without examining spectral intensities.
    census_dir = Path(cfg["census_dir"])
    provenance = json.loads((census_dir / "provenance.json").read_text())
    for record in provenance["inputs"].values():
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"Input changed: {record['path']}")
    if args.pairs_csv is None:
        pairs = select_pairs(
            pd.read_pickle(census_dir / "pairs.pkl"),
            cfg["queries_per_novelty_group"],
            cfg["seed"],
            cfg["condition_selection"],
        )
    else:
        pairs = pd.read_csv(args.pairs_csv)
        if pairs.empty or pairs[["query_spec", "donor_spec"]].duplicated().any():
            raise ValueError("External pair manifest is empty or duplicated")
    pairs.to_csv(args.output / "selected_pairs.csv", index=False)
    wanted = set(pairs.query_spec) | set(pairs.donor_spec)
    train_ids = set(pd.read_csv(provenance["inputs"]["train_ids_fp"]["path"]).spec_id)
    query_ids_path = Path(provenance["inputs"]["train_ids_fp"]["path"]).with_name(
        f"{args.query_split}_ids.csv"
    )
    query_ids = set(pd.read_csv(query_ids_path).spec_id)
    if not set(pairs.query_spec).issubset(query_ids):
        raise ValueError("Query spectrum outside requested split")
    if not set(pairs.donor_spec).issubset(train_ids):
        raise ValueError("Nontraining reference selected")
    if args.query_split == "val" and query_ids & train_ids:
        raise ValueError("Train/validation spectra overlap")
    run = {
        "status": "running",
        "query_split": args.query_split,
        "retain_pathways": args.retain_pathways,
        "query_ids_sha256": sha256(query_ids_path),
        "command": sys.argv,
        "external_pairs_sha256": sha256(args.pairs_csv) if args.pairs_csv else None,
        "config": cfg,
        "started_unix": started,
        "device": "cpu",
        "torch": th.__version__,
        "numpy": np.__version__,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
        "source_sha256": sha256(__file__),
        "config_sha256": sha256(args.config),
        "checkpoint_sha256": sha256(cfg["checkpoint"]),
        "inputs": provenance["inputs"],
        "census_pairs_sha256": sha256(census_dir / "pairs.pkl"),
        "selected_pairs_sha256": sha256(args.output / "selected_pairs.csv"),
    }
    (args.output / "provenance.json").write_text(json.dumps(run, indent=2))
    (args.output / "source.py").write_text(Path(__file__).read_text())
    # 2. Use the actual i1 training dataset/candidates in deterministic eval mode on CPU.
    config = apply_inference_config(
        load_config("config", cfg["model_config"], None),
        disable_preproc=True,
        num_workers=0,
        eval_batch_size=cfg["batch_size"],
    )
    config["masster_candidate_backend"] = "cpu"
    config["masster_fragment_connectivity_layers"] = 0
    model = init_model_from_ckpt(cfg["checkpoint"], config, th.device("cpu")).eval()
    dataset = get_dataset_cls(config["model_type"])(split="train", **config)
    collate_fn = dataset.get_collate_fn()
    specs = dataset.spec_df.reset_index(drop=True)
    if args.query_split == "val":
        queries = get_dataset_cls(config["model_type"])(split="val", **config)
        specs = pd.concat([specs, queries.spec_df], ignore_index=True)
        dataset = ConcatDataset([dataset, queries])
    chosen = specs.index[specs.spec_id.isin(wanted)].tolist()
    rows = specs.iloc[chosen].reset_index(drop=True)
    if set(rows.spec_id) != wanted or rows.spec_id.duplicated().any():
        raise ValueError("Missing or duplicate selected spectrum IDs")
    by_id = rows.set_index("spec_id")
    for pair in pairs.itertuples():
        q, d = by_id.loc[pair.query_spec], by_id.loc[pair.donor_spec]
        if q.mol_id == d.mol_id or q.prec_type != "[M+H]+":
            raise ValueError("Invalid identity or adduct")
        if any(q[k] != d[k] for k in ("inst_type", "prec_type", "frag_mode", "nce")):
            raise ValueError("Acquisition mismatch")
    loader = DataLoader(
        Subset(dataset, chosen),
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    cache, cursor, mass_error = {}, 0, 0.0
    with th.inference_mode():
        for batch in loader:
            size = int(batch["batch_size"])
            batch_rows = rows.iloc[cursor : cursor + size]
            if not np.array_equal(batch["spec_unique_id"].numpy(), batch_rows.spec_id.to_numpy()):
                raise ValueError("Unexpected batch order")
            batch = model.transfer_batch_to_device(batch, th.device("cpu"), 0)
            out = model(**batch, return_explainability_tensors=True)
            formulas, precursors, error = reconstruct_formulas(
                batch, out, model.model.precursor_types
            )
            mass_error = max(mass_error, error)
            tm, ti, tb = model.preproc_spec(
                batch["spec_mzs"],
                batch["spec_ints"],
                batch["spec_batch_idxs"],
                train=False,
                pred=False,
                batch_size=size,
            )
            membership = {}
            for atom, fragment in zip(
                out["fragment_atom_indices"].numpy(), out["fragment_atom_fragment_idxs"].numpy()
            ):
                membership.setdefault(int(fragment), []).append(int(atom))
            atom_sets = [
                tuple(membership[int(f)]) for f in out["pred_pathway_fragment_idxs"].numpy()
            ]
            for view, row in enumerate(batch_rows.itertuples()):
                indices = np.flatnonzero(out["pred_batch_idxs"].numpy() == view)
                peaks = tb.numpy() == view
                tol = config["inst_type_loss_tol"][row.inst_type]
                cache[int(row.spec_id)] = candidate_groups(
                    out["pred_mzs"].numpy()[indices],
                    out["pred_logprobs"].exp().numpy()[indices],
                    formulas[indices],
                    precursors[view],
                    [atom_sets[k] for k in indices],
                    tm.numpy()[peaks],
                    ti.numpy()[peaks],
                    float(row.prec_mz),
                    tol["rel"],
                    tol["min_mz"],
                    cfg["precursor_exclusion_da"],
                )
                if args.retain_pathways:
                    graph = batch["mol_pyg"]
                    start, end = int(graph.ptr[view]), int(graph.ptr[view + 1])
                    bonds = graph.chemical_bond_index.numpy()
                    bond_mask = (bonds[0] >= start) & (bonds[0] < end)
                    frame = cache[int(row.spec_id)]
                    frame.attrs.update(
                        mol_id=int(row.mol_id),
                        path_atoms=[tuple(a - start for a in atom_sets[k]) for k in indices],
                        path_modes=out["pred_pathway_ion_mode_idxs"].numpy()[indices],
                        path_h_shifts=out["pred_pathway_h_shifts"].numpy()[indices],
                        path_group=np.unique(
                            np.round(out["pred_mzs"].numpy()[indices], 6), return_inverse=True
                        )[1],
                        atom_numbers=graph.atom_atomic_numbers.numpy()[start:end],
                        atom_hydrogens=graph.atom_hydrogen_counts.numpy()[start:end],
                        chemical_bonds=bonds[:, bond_mask] - start,
                    )
            cursor += size
            LOGGER.info("Annotated %d/%d spectra", cursor, len(rows))
    # 3. Save candidate groups, correspondences, pair coverage and stratum summaries.
    pd.to_pickle(cache, args.output / "candidate_groups.pkl")
    stats, matches = summarize_pairs(pairs, cache, cfg["reliable_intensity_fraction"])
    stats.to_csv(args.output / "pair_coverage.csv", index=False)
    matches.to_csv(args.output / "correspondences.csv", index=False)
    summary = (
        stats.groupby(["novel", "kind", "view"])
        .agg(
            pairs=("query_spec", "size"),
            matches_mean=("matches", "mean"),
            two_reliable_pairs=("two_reliable", "sum"),
            two_reliable_fraction=("two_reliable", "mean"),
            both_positive_mean=("both_positive", "mean"),
            both_reliable_mean=("both_reliable", "mean"),
            query_covered_intensity=("query_covered_intensity", "mean"),
            donor_covered_intensity=("donor_covered_intensity", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(args.output / "summary.csv", index=False)
    # Array-valued spectrum metadata are not comparable during pandas concatenation.
    frames = []
    for frame in cache.values():
        copy = frame.copy()
        copy.attrs = {}
        frames.append(copy)
    all_groups = pd.concat(frames, ignore_index=True)
    run.update(
        status="complete",
        elapsed_seconds=time.time() - started,
        spectra=len(cache),
        query_molecules=int(pairs.query_mol.nunique()),
        max_formula_mass_error_da=mass_error,
        candidate_groups=len(all_groups),
        eligible_groups=int(all_groups.eligible.sum()),
        unresolved_groups=int(all_groups.unresolved.sum()),
        multiple_formula_groups=int((all_groups.formula_alternatives > 1).sum()),
        multiple_structure_groups=int((all_groups.structural_alternatives > 1).sum()),
        resolved_model_config=config,
    )
    (args.output / "provenance.json").write_text(json.dumps(run, indent=2, default=str))
    LOGGER.info("Complete: %s", summary.to_dict("records"))


if __name__ == "__main__":
    main()
