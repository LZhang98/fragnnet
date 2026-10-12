"""Export ambiguity-preserving measured-base-peak versus predicted-winner comparisons.

One summary row per spectrum and one JSONL row per channel in either comparison group.
Target membership uses every channel within the instrument tolerance of the measured base
peak, with no probability cutoff. Winner membership uses the channels contributing to the
largest predicted peak after rounding m/z to four decimals (the instrument-cosine convention).
No measured intensity is assigned to an individual structure. Atom indices and boundary bonds
describe candidate subgraphs, not experimentally identified cuts or complete mechanisms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from rdkit import Chem
from torch.utils.data import DataLoader, Subset

from fragnnet.inference.checkpoint import apply_inference_config, init_model_from_ckpt
from fragnnet.runner import load_config
from fragnnet.utils.setup_utils import get_dataset_cls

try:
    from .debug_masster_base_peak_tables import MoleculeChemistry
except ImportError:
    from debug_masster_base_peak_tables import MoleculeChemistry

LOGGER = logging.getLogger(__name__)


def compare_peaks(
    channel_mz: np.ndarray,
    channel_intensity: np.ndarray,
    target_mz: np.ndarray,
    target_intensity: np.ndarray,
    rel: float,
    min_mz: float,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Compare the base peak to the summed predicted winner, retaining all mass matches.

    Intensities are independently normalized to sum one. Tied measured base peaks choose
    the first input peak and are flagged; tied predicted winners choose the lowest rounded
    m/z. Measured support at the winner reports both max and sum, since multiple experimental
    peaks can lie in its tolerance window. ``strict_measured_preference`` uses their maximum.
    Empty targets or predictions raise rather than silently dropping a spectrum.
    """
    if not len(channel_mz) or not len(target_mz):
        raise ValueError("comparison requires nonempty predictions and targets")
    for mz, intensity in ((channel_mz, channel_intensity), (target_mz, target_intensity)):
        if mz.shape != intensity.shape or mz.ndim != 1:
            raise ValueError("m/z and intensity must be matching vectors")
        if not np.isfinite(mz).all() or not np.isfinite(intensity).all():
            raise ValueError("nonfinite peak data")
        if (intensity < 0).any() or intensity.sum() <= 0:
            raise ValueError("intensities must be nonnegative with positive sum")
    pred = channel_intensity / channel_intensity.sum()
    measured = target_intensity / target_intensity.sum()
    base = int(np.argmax(measured))
    base_mz = float(target_mz[base])
    target_mask = np.abs(channel_mz - base_mz) <= rel * max(base_mz, min_mz)
    peak_mz, inverse = np.unique(np.round(channel_mz, 4), return_inverse=True)
    peak_intensity = np.bincount(inverse, weights=pred)
    top = int(np.argmax(peak_intensity))
    winner_mz = float(peak_mz[top])
    winner_mask = inverse == top
    measured_match = np.abs(target_mz - winner_mz) <= rel * np.maximum(target_mz, min_mz)
    winner_measured = measured[measured_match]
    winner_max = float(winner_measured.max()) if len(winner_measured) else 0.0
    hit = abs(winner_mz - base_mz) <= rel * max(base_mz, min_mz)
    outcome = "uncovered" if not target_mask.any() else ("hit" if hit else "under_ranked")
    return (
        {
            "outcome": outcome,
            "target_mz": base_mz,
            "target_measured_share": float(measured[base]),
            "target_predicted_group_share": float(pred[target_mask].sum()),
            "target_channel_count": int(target_mask.sum()),
            "target_exact_peak_count": int(np.unique(inverse[target_mask]).size),
            "target_tied_base_count": int(np.count_nonzero(measured == measured[base])),
            "winner_mz": winner_mz,
            "winner_predicted_share": float(peak_intensity[top]),
            "winner_measured_max_share": winner_max,
            "winner_measured_sum_share": float(winner_measured.sum()),
            "winner_measured_peak_count": int(measured_match.sum()),
            "winner_channel_count": int(winner_mask.sum()),
            "winner_tied_peak_count": int(np.count_nonzero(peak_intensity == peak_intensity[top])),
            "strict_measured_preference": bool(measured[base] > winner_max),
            "groups_overlap": bool((target_mask & winner_mask).any()),
            "predicted_peak_count": len(peak_mz),
            "channel_count": len(channel_mz),
        },
        target_mask,
        winner_mask,
    )


def strongest_rival(
    channel_mz: np.ndarray,
    channel_intensity: np.ndarray,
    target_mz: np.ndarray,
    target_intensity: np.ndarray,
    rel: float,
    min_mz: float,
) -> tuple[dict, np.ndarray]:
    """Select the strongest rounded predicted peak wholly outside the base-peak group.

    Selection is identical for hits and misses. Exclude a rounded peak if its center or any
    contributing channel matches the measured base peak. No rival is represented explicitly.
    """
    measured = target_intensity / target_intensity.sum()
    base = int(np.argmax(measured))
    base_mz = float(target_mz[base])
    tol = rel * max(base_mz, min_mz)
    mz, inverse = np.unique(np.round(channel_mz, 4), return_inverse=True)
    pred = channel_intensity / channel_intensity.sum()
    intensity = np.bincount(inverse, weights=pred)
    excluded = np.abs(mz - base_mz) <= tol
    excluded[np.unique(inverse[np.abs(channel_mz - base_mz) <= tol])] = True
    available = np.flatnonzero(~excluded)
    if not len(available):
        return {
            "rival_available": False,
            "rival_mz": np.nan,
            "rival_predicted_share": 0.0,
            "rival_measured_max_share": np.nan,
            "rival_channel_count": 0,
        }, np.zeros(len(channel_mz), dtype=bool)
    rival = int(available[np.argmax(intensity[available])])
    matched = np.abs(target_mz - mz[rival]) <= rel * np.maximum(target_mz, min_mz)
    return {
        "rival_available": True,
        "rival_mz": float(mz[rival]),
        "rival_predicted_share": float(intensity[rival]),
        "rival_measured_max_share": float(measured[matched].max()) if matched.any() else 0.0,
        "rival_channel_count": int((inverse == rival).sum()),
    }, inverse == rival


def describe_fragment(chem: MoleculeChemistry, atoms: tuple[int, ...]) -> dict:
    """Describe original-molecule membership and its boundary without assigning ion chemistry."""
    inside = set(atoms)
    if not inside or len(inside) != len(atoms) or min(inside) < 0 or max(inside) >= chem.n_atoms:
        raise ValueError("invalid fragment membership")
    cuts = [i for i, (u, v) in enumerate(chem.bonds) if (u in inside) != (v in inside)]
    return {
        "atom_indices": list(atoms),
        "retained_subgraph_smiles": Chem.MolFragmentToSmiles(chem.mol, atomsToUse=list(atoms)),
        "boundary_bond_indices": cuts,
        "boundary_atom_pairs": [list(chem.bonds[i]) for i in cuts],
        "cut_classes": [chem.bond_class[i] for i in cuts],
        "cut_count": len(cuts),
        "atom_count": len(atoms),
        "atom_fraction": len(atoms) / chem.n_atoms,
        "ring_cut": any(chem.bond_ring[i] for i in cuts),
        "base_class": chem.base_class,
        "keeps_base": any(m.issubset(inside) for m in chem.base_matches),
        "loses_aromatic": any(not (r & inside) for r in chem.aromatic_rings),
        "breaks_aromatic": any(0 < len(r & inside) < len(r) for r in chem.aromatic_rings),
    }


def file_identity(path: str | Path) -> dict:
    """Record a streaming SHA256 so large inputs need not be read into RAM twice."""
    path = Path(path)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def main() -> None:
    """Run annotated inference and export summaries, group members, and provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--custom_fp", required=True)
    parser.add_argument("--spec_fp", required=True)
    parser.add_argument("--output_dp", required=True)
    parser.add_argument(
        "--include_rival",
        action="store_true",
        help="also export strongest alternative outside the measured base group",
    )
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="first N spectra for smoke checks")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    started = time.time()
    random.seed(0)
    np.random.seed(0)  # noqa: NPY002
    th.manual_seed(0)
    output = Path(args.output_dp)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "channels.jsonl").exists():
        raise FileExistsError(f"refusing to overwrite {output / 'channels.jsonl'}")

    # 1. Resolve the checkpoint's config and use the normal inference preprocessing.
    config = load_config("config", args.custom_fp, [f"spec_fp={args.spec_fp}"])
    if (
        config["model_type"] != "masster"
        or config["masster_smarts_templates"]
        or config["spec_params"]["merge"]
    ):
        raise ValueError("this diagnostic requires unmerged MASSter without templates")
    config = apply_inference_config(
        config, disable_preproc=True, num_workers=0, eval_batch_size=args.batch_size
    )
    device = th.device(args.device)
    if device.type == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select --device cpu")
    if device.type == "cpu":
        config["masster_candidate_backend"] = "cpu"
    model = init_model_from_ckpt(args.checkpoint, config, device)
    model.eval()
    dataset = get_dataset_cls(config["model_type"])(split=args.split, **config)
    specs = dataset.spec_df.reset_index(drop=True)
    keep = np.arange(min(args.limit, len(specs)) if args.limit else len(specs))
    if not len(keep):
        raise ValueError("no spectra selected")
    molecules = pd.read_pickle(config["mol_fp"]).set_index("mol_id")["mol"]
    inst_types = config["spec_params"]["inst_types"]
    loader = DataLoader(
        Subset(dataset, keep.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.get_collate_fn(),
        num_workers=0,
    )
    provenance = {
        "status": "running",
        "args": vars(args),
        "seed": 0,
        "started_unix": started,
        "torch": th.__version__,
        "device": th.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"])
        ).hexdigest(),
        "inputs": [
            file_identity(p)
            for p in (
                __file__,
                args.checkpoint,
                args.custom_fp,
                args.spec_fp,
                config["mol_fp"],
                Path(config["split_dp"]) / f"{args.split}_ids.csv",
            )
        ],
        "scope": "exploratory; no fitting or untouched evaluation",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2, default=str))

    # 2. Stream every channel in either group, preserving zero-share compatible alternatives.
    summary, molecule_rows = [], {}
    chem_cache, fragment_cache = {}, {}
    cursor, channel_rows = 0, 0
    with (output / "channels.jsonl").open("w") as handle, th.inference_mode():
        for batch in loader:
            size = int(batch["batch_size"])
            batch_specs = specs.iloc[keep[cursor : cursor + size]]
            if not np.array_equal(batch["spec_unique_id"].numpy(), batch_specs.spec_id.to_numpy()):
                raise RuntimeError("batch spectrum order mismatch")
            batch = model.transfer_batch_to_device(batch, device, dataloader_idx=0)
            out = model.inference_step(
                {**batch, "return_explainability_tensors": True}, metrics=None
            )
            true = model.preproc_spec(
                batch["spec_mzs"],
                batch["spec_ints"],
                batch["spec_batch_idxs"],
                train=False,
                pred=False,
                batch_size=size,
            )
            tm, ti, tb = [x.cpu().numpy() for x in true]
            arrays = {
                k: out[k].cpu().numpy()
                for k in (
                    "pred_mzs",
                    "pred_ints",
                    "pred_batch_idxs",
                    "pred_pathway_fragment_idxs",
                    "pred_pathway_h_shifts",
                    "pred_pathway_ion_mode_idxs",
                    "fragment_atom_indices",
                    "fragment_atom_fragment_idxs",
                    "ion_h_fragment_batch_idxs",
                )
            }
            atom_ptr = batch["mol_pyg"].ptr.cpu().numpy()
            inst_idx = batch["spec_inst_type"].reshape(-1).cpu().numpy()
            member_order = np.argsort(arrays["fragment_atom_fragment_idxs"], kind="stable")
            member_frag = arrays["fragment_atom_fragment_idxs"][member_order]
            member_atom = arrays["fragment_atom_indices"][member_order]
            for view, row in enumerate(batch_specs.itertuples()):
                mid, sid = int(row.mol_id), int(row.spec_id)
                if mid not in chem_cache:
                    mol = molecules.loc[mid]
                    chem_cache[mid] = MoleculeChemistry(mol)
                    mapped = Chem.Mol(mol)
                    for atom in mapped.GetAtoms():
                        atom.SetAtomMapNum(atom.GetIdx() + 1)
                    molecule_rows[mid] = {
                        "mol_id": mid,
                        "smiles": Chem.MolToSmiles(mol),
                        "indexed_smiles": Chem.MolToSmiles(mapped),
                    }
                chem = chem_cache[mid]
                if atom_ptr[view + 1] - atom_ptr[view] != chem.n_atoms:
                    raise RuntimeError("molecule and graph atom counts differ")
                channels = np.flatnonzero(
                    (arrays["pred_batch_idxs"] == view) & (arrays["pred_mzs"] < config["mz_max"])
                )
                cm = arrays["pred_mzs"][channels].astype(float)
                cp = arrays["pred_ints"][channels].astype(float)
                peak_mask = tb == view
                inst = inst_types[int(inst_idx[view])]
                tol = config["inst_type_loss_tol"][inst]
                record, target_mask, winner_mask = compare_peaks(
                    cm,
                    cp,
                    tm[peak_mask].astype(float),
                    ti[peak_mask].astype(float),
                    tol["rel"],
                    tol["min_mz"],
                )
                nce = np.asarray(row.nce, dtype=float).reshape(-1)
                if nce.size != 1:
                    raise ValueError("expected one NCE per unmerged spectrum")
                record.update(
                    {
                        "spec_id": sid,
                        "mol_id": mid,
                        "split": args.split,
                        "nce": float(nce[0]),
                        "instrument": inst,
                        "adduct": row.prec_type,
                        "precursor_mz": float(row.prec_mz),
                        "tolerance_rel": tol["rel"],
                        "tolerance_min_mz": tol["min_mz"],
                    }
                )
                masks = {"target": target_mask, "winner": winner_mask}
                if args.include_rival:
                    rival_record, rival_mask = strongest_rival(
                        cm,
                        cp,
                        tm[peak_mask].astype(float),
                        ti[peak_mask].astype(float),
                        tol["rel"],
                        tol["min_mz"],
                    )
                    record.update(rival_record)
                    masks["rival"] = rival_mask
                groups = {name: set() for name in masks}
                channel_keys = {name: set() for name in masks}
                for local in np.flatnonzero(np.logical_or.reduce(list(masks.values()))):
                    channel = channels[local]
                    frag = int(arrays["pred_pathway_fragment_idxs"][channel])
                    if arrays["ion_h_fragment_batch_idxs"][frag] != view:
                        raise RuntimeError("channel fragment belongs to another spectrum")
                    lo, hi = (
                        np.searchsorted(member_frag, frag, side="left"),
                        np.searchsorted(member_frag, frag, side="right"),
                    )
                    atoms = tuple(sorted((member_atom[lo:hi] - atom_ptr[view]).tolist()))
                    key = (mid, atoms)
                    if key not in fragment_cache:
                        fragment_cache[key] = describe_fragment(chem, atoms)
                    h = int(arrays["pred_pathway_h_shifts"][channel])
                    mode = int(arrays["pred_pathway_ion_mode_idxs"][channel])
                    for group, mask in masks.items():
                        if mask[local]:
                            groups[group].add(atoms)
                            channel_keys[group].add((atoms, h, mode))
                    handle.write(
                        json.dumps(
                            {
                                "spec_id": sid,
                                "mol_id": mid,
                                "channel_index": int(local),
                                "target_compatible": bool(target_mask[local]),
                                "winner_contributor": bool(winner_mask[local]),
                                **(
                                    {"rival_contributor": bool(masks["rival"][local])}
                                    if args.include_rival
                                    else {}
                                ),
                                "mz": float(cm[local]),
                                "predicted_share": float(cp[local] / cp.sum()),
                                "h_shift": h,
                                "ion_mode_index": mode,
                                **fragment_cache[key],
                            }
                        )
                        + "\n"
                    )
                    channel_rows += 1
                for group in groups:
                    record[f"{group}_structure_count"] = len(groups[group])
                    record[f"{group}_distinct_channel_count"] = len(channel_keys[group])
                    record[f"{group}_structurally_ambiguous"] = len(groups[group]) > 1
                summary.append(record)
            cursor += size
            if cursor % (args.batch_size * 50) == 0 or cursor == len(keep):
                LOGGER.info("%d/%d spectra, %.1f s", cursor, len(keep), time.time() - started)

    # 3. Export complete, disjoint outcome tables and reproducibility metadata.
    table = pd.DataFrame(summary)
    if len(table) != len(keep) or table.spec_id.duplicated().any():
        raise RuntimeError("missing or duplicate spectrum records")
    table.to_pickle(output / "pairs.pkl")
    table.to_csv(output / "pairs.csv", index=False)
    for outcome, name in (
        ("under_ranked", "ranking_pairs"),
        ("hit", "controls"),
        ("uncovered", "coverage_failures"),
    ):
        table[table.outcome == outcome].to_csv(output / f"{name}.csv", index=False)
    pd.DataFrame(molecule_rows.values()).to_csv(output / "molecules.csv", index=False)
    provenance.update(
        {
            "status": "complete",
            "elapsed_seconds": time.time() - started,
            "spectra": len(table),
            "molecules": table.mol_id.nunique(),
            "channel_rows": channel_rows,
            "outcomes": table.outcome.value_counts().to_dict(),
        }
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    LOGGER.info("Complete: %s", provenance["outcomes"])


if __name__ == "__main__":
    main()
