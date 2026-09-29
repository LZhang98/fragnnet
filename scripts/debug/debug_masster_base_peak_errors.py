"""Base-peak errors of a MASSter checkpoint: which candidate the model puts on top, and which
the measured spectrum does.

Stage 1 of the wrong-peak analysis (2026-09-28). One annotated forward per batch, eval mode.
Per spectrum it records:

- the candidate-space split of ``1 - cos`` into coverage, ranking and calibration, as in the
  [ranking decomposition](docs/findings/masster/masster_ranking_decomposition_2026_09_23.md).
  Channels sharing an m/z are merged into one candidate peak; each target peak maps to its
  nearest candidate within the instrument tolerance;
- the base-peak outcome: ``hit`` (the model's top candidate is the measured base peak),
  ``under_ranked`` (the base peak is a candidate but not the model's top) or ``uncovered``;
- for the model's top-k and the target's top-k candidate peaks: m/z, predicted and target
  share, both ranks, and every candidate structure at that m/z (fragment atoms, H shift, ion
  mode, predicted probability). A peak several structures can explain keeps all of them;
- for every candidate, structure attributes (cut count, H shift, precursor, ring cut, aromatic
  ring lost or broken, basic site kept). Per spectrum it writes their predicted mean
  ``ep_*`` and target mean ``et_*`` (over covered target intensity; a peak with several
  structures splits its target intensity by the model's shares), and the target share on
  peaks whose structures disagree, ``amb_*``. That is calibration in the large, with no
  selection by target rank. Candidates with prediction >= 1e-5 or target intensity are
  also saved as arrays (``.cands.npz``) for the cross-fitted reweighting in the tables script.

The chemistry of those structures and the grouped tables come from
``debug_masster_base_peak_tables.py``, offline, from this script's output.

Run from the repository root, e.g.:

    python scripts/debug/debug_masster_base_peak_errors.py \
        --checkpoint "saved_ckpts/i1_inst_ob_obf_i50ig7rf/ckpt/model-epoch=059.ckpt" \
        --custom_fp config/masster/mh_ft_memorization_route_2026_09_24/i1_inst_ob_obf.yml \
        --split val --output output/base_peak_errors/i1_e059_val.pkl
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
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader, Subset

from fragnnet.runner import load_config
from fragnnet.utils.script_utils import apply_inference_config, init_model_from_ckpt
from fragnnet.utils.setup_utils import get_dataset_cls

# Support both the script entry point and imports from pytest.
try:
    from .debug_masster_base_peak_tables import MoleculeChemistry
except ImportError:
    from debug_masster_base_peak_tables import MoleculeChemistry

LOGGER = logging.getLogger(__name__)

# Per-candidate structure attributes, in the column order of the saved arrays.
CANDIDATE_ATTRIBUTES = (
    "n_cuts",
    "h_shift_abs",
    "is_precursor",
    "frac_atoms",
    "ring_cut",
    "loses_aromatic",
    "breaks_aromatic",
    "keeps_base",
)
# Candidates below this predicted share and without target intensity are not saved.
SAVE_FLOOR = 1e-5

# Channels whose m/z agree to this many decimals are one candidate peak (same ion formula).
MZ_DECIMALS = 6


def merge_channels(mzs: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge one spectrum's output channels by m/z.

    Args:
        mzs: Channel m/z, shape ``(C,)``.
        probs: Channel probabilities, shape ``(C,)``.

    Returns:
        Sorted unique m/z ``(K,)``, summed probability per unique m/z ``(K,)`` and each
        channel's candidate index ``(C,)``.
    """
    unique, inverse = np.unique(np.round(mzs, MZ_DECIMALS), return_inverse=True)
    return unique, np.bincount(inverse, weights=probs, minlength=unique.size), inverse


def map_targets(
    peak_mzs: np.ndarray,
    peak_ints: np.ndarray,
    cand_mzs: np.ndarray,
    rel: float,
    min_mz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each target peak to its nearest candidate within ``rel * max(m/z, min_mz)``.

    Args:
        peak_mzs: Target m/z, shape ``(T,)``.
        peak_ints: Target intensities, shape ``(T,)``.
        cand_mzs: Sorted candidate m/z, shape ``(K,)``.
        rel: Relative tolerance (1e-5 is 10 ppm).
        min_mz: m/z below which the tolerance is held at ``rel * min_mz``.

    Returns:
        Target intensity summed on each candidate ``(K,)``, and each peak's candidate index,
        -1 when no candidate is within tolerance ``(T,)``.
    """
    if cand_mzs.size == 0:
        return np.zeros(0), np.full(peak_mzs.size, -1)
    upper = np.clip(np.searchsorted(cand_mzs, peak_mzs), 0, cand_mzs.size - 1)
    lower = np.maximum(upper - 1, 0)
    use_lower = np.abs(peak_mzs - cand_mzs[lower]) < np.abs(peak_mzs - cand_mzs[upper])
    nearest = np.where(use_lower, lower, upper)
    within = np.abs(peak_mzs - cand_mzs[nearest]) <= rel * np.maximum(peak_mzs, min_mz)
    peak_to_cand = np.where(within, nearest, -1)
    on_cand = np.bincount(
        nearest[within], weights=peak_ints[within], minlength=cand_mzs.size
    ).astype(float)
    return on_cand, peak_to_cand


def cosine_decomposition(pred: np.ndarray, on_cand: np.ndarray, target_norm: float) -> dict:
    """Split ``1 - cos`` into coverage, ranking and calibration.

    ``oracle = |T_covered| / |T|`` bounds any prediction on these candidates. ``iso`` is the
    cosine of the isotonic (non-decreasing in ``pred``) least-squares fit of ``T_covered``:
    the projection onto that monotone cone, which maximizes cosine over every vector ordered
    like ``pred``. Then coverage ``= 1 - oracle``, ranking ``= oracle - iso`` and calibration
    ``= iso - cos``.

    Args:
        pred: Predicted intensity per candidate, shape ``(K,)``.
        on_cand: Target intensity per candidate, shape ``(K,)``.
        target_norm: L2 norm of the whole target, uncovered peaks included.

    Returns:
        Dict with ``cos``, ``oracle``, ``iso``, ``coverage``, ``ranking``, ``calibration``.
    """
    cos = float(pred @ on_cand / (np.linalg.norm(pred) * target_norm))
    oracle = float(np.linalg.norm(on_cand) / target_norm)
    fit = IsotonicRegression(increasing=True).fit_transform(pred, on_cand)
    iso = float(np.linalg.norm(fit) / target_norm)
    return {
        "cos": cos,
        "oracle": oracle,
        "iso": iso,
        "coverage": 1.0 - oracle,
        "ranking": oracle - iso,
        "calibration": iso - cos,
    }


def base_peak_outcome(
    peak_mzs: np.ndarray,
    peak_ints: np.ndarray,
    peak_to_cand: np.ndarray,
    cand_mzs: np.ndarray,
    pred: np.ndarray,
    rel: float,
    min_mz: float,
) -> str:
    """Classify the measured base peak as ``hit``, ``under_ranked`` or ``uncovered``.

    ``hit`` means the model's highest candidate lies within tolerance of the base peak.
    """
    base = int(np.argmax(peak_ints))
    if peak_to_cand[base] < 0:
        return "uncovered"
    top = cand_mzs[int(np.argmax(pred))]
    if abs(top - peak_mzs[base]) <= rel * max(peak_mzs[base], min_mz):
        return "hit"
    return "under_ranked"


def fragment_attributes(inside: np.ndarray, chem: MoleculeChemistry) -> dict[str, np.ndarray]:
    """Structure attributes of each fragment, vectorized over a membership matrix.

    Matches ``MoleculeChemistry.annotate`` for the attributes that do not depend on the H shift.

    Args:
        inside: Boolean membership, shape ``(F, n_atoms)``, row f true on fragment f's atoms.
        chem: The molecule's lookups (bonds, ring bonds, aromatic rings, basic-site matches).

    Returns:
        Name -> float array of shape ``(F,)``: ``n_atoms``, ``frac_atoms``, ``n_cuts``,
        ``ring_cut``, ``loses_aromatic``, ``breaks_aromatic`` and ``keeps_base``.
    """
    fragments = inside.shape[0]
    bonds = np.asarray(chem.bonds, dtype=np.int64).reshape(-1, 2)
    crossing = inside[:, bonds[:, 0]] != inside[:, bonds[:, 1]]
    ring = np.asarray(chem.bond_ring, dtype=bool)
    loses = np.zeros(fragments, dtype=bool)
    breaks = np.zeros(fragments, dtype=bool)
    for members in chem.aromatic_rings:
        count = inside[:, sorted(members)].sum(axis=1)
        loses |= count == 0
        breaks |= (count > 0) & (count < len(members))
    keeps_base = np.zeros(fragments, dtype=bool)
    for match in chem.base_matches:
        keeps_base |= inside[:, sorted(match)].all(axis=1)
    n_atoms = inside.sum(axis=1)
    return {
        "n_atoms": n_atoms.astype(float),
        "frac_atoms": n_atoms / chem.n_atoms,
        "n_cuts": crossing.sum(axis=1).astype(float),
        "ring_cut": crossing[:, ring].any(axis=1).astype(float),
        "loses_aromatic": loses.astype(float),
        "breaks_aromatic": breaks.astype(float),
        "keeps_base": keeps_base.astype(float),
    }


def channel_attributes(
    fragment: dict[str, np.ndarray], local: np.ndarray, h_shifts: np.ndarray, n_atoms: int
) -> np.ndarray:
    """Attribute matrix of output channels, columns in ``CANDIDATE_ATTRIBUTES`` order.

    Args:
        fragment: ``fragment_attributes`` of the spectrum's fragments.
        local: Each channel's fragment row, shape ``(C,)``.
        h_shifts: Each channel's H shift, shape ``(C,)``.
        n_atoms: Heavy atoms of the molecule.

    Returns:
        Float array of shape ``(C, len(CANDIDATE_ATTRIBUTES))``.
    """
    columns = {
        "h_shift_abs": np.abs(h_shifts).astype(float),
        "is_precursor": ((fragment["n_atoms"][local] == n_atoms) & (h_shifts == 0)).astype(float),
    }
    return np.column_stack(
        [columns[a] if a in columns else fragment[a][local] for a in CANDIDATE_ATTRIBUTES]
    )


def descending_ranks(values: np.ndarray) -> np.ndarray:
    """1-based rank of each entry under a stable descending sort (1 = largest)."""
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(values.size, dtype=np.int64)
    ranks[order] = np.arange(1, values.size + 1)
    return ranks


def _identity(path: str) -> dict:
    data = Path(path).read_bytes()
    return {"path": path, "bytes": len(data), "md5": hashlib.md5(data).hexdigest()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--custom_fp", required=True, help="the checkpoint's run config")
    parser.add_argument("--template_fp", default="config")
    parser.add_argument("--override", nargs="*", default=[], help="config overrides, key=value")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--molecules",
        type=int,
        default=0,
        help="random molecules to keep, all spectra each; 0 = all",
    )
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--output", required=True, help="pickle; .peaks.pkl and .provenance.json beside it"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the annotated forward over one split and write the spectrum and peak tables."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    th.manual_seed(args.seed)
    np.random.seed(args.seed)  # noqa: NPY002 - seed legacy dependencies as well
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # 1. Model and dataset with the checkpoint's own config.
    config = load_config(args.template_fp, args.custom_fp, list(args.override))
    if config["model_type"] != "masster":
        raise ValueError(f"expected model_type masster, got {config['model_type']}")
    config = apply_inference_config(
        config, disable_preproc=True, num_workers=0, eval_batch_size=args.batch_size
    )
    device = th.device(args.device)
    if device.type == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu explicitly")
    if device.type == "cpu":
        config["masster_candidate_backend"] = "cpu"
    model = init_model_from_ckpt(args.checkpoint, config, device)
    model.eval()
    dataset = get_dataset_cls(config["model_type"])(split=args.split, **config)
    specs = dataset.spec_df.reset_index(drop=True)
    molecules = pd.read_pickle(config["mol_fp"]).set_index("mol_id")["mol"]
    chem_cache: dict[int, MoleculeChemistry] = {}
    inst_types = list(config["spec_params"]["inst_types"])
    tolerance = {
        i: (config["inst_type_loss_tol"][name]["rel"], config["inst_type_loss_tol"][name]["min_mz"])
        for i, name in enumerate(inst_types)
    }

    # 2. Molecule-level sample, drawn before any target is read.
    keep = np.arange(len(specs))
    if args.molecules:
        mol_ids = np.sort(specs.mol_id.unique())
        chosen = rng.choice(mol_ids, min(args.molecules, mol_ids.size), replace=False)
        keep = np.flatnonzero(specs.mol_id.isin(chosen).to_numpy())
    loader = DataLoader(
        Subset(dataset, keep.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.get_collate_fn(),
        num_workers=0,
    )
    provenance = {
        "args": vars(args),
        "started_unix": started,
        "torch": th.__version__,
        "gpu": th.cuda.get_device_name(device) if device.type == "cuda" else None,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"])
        ).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": [_identity(p) for p in (args.checkpoint, config["spec_fp"], config["mol_fp"])],
        "spectra": int(keep.size),
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    LOGGER.info(
        "%s: %d spectra, %d molecules", args.split, keep.size, specs.iloc[keep].mol_id.nunique()
    )

    spectrum_rows, peak_rows = [], []
    saved = {"spec_id": [], "count": [], "pred": [], "tgt": [], "x": [], "target_norm": []}
    cursor = 0
    with th.inference_mode():
        for batch in loader:
            size = int(batch["batch_size"])
            rows = specs.iloc[keep[cursor : cursor + size]]
            if not np.array_equal(batch["spec_unique_id"].numpy(), rows.spec_id.to_numpy()):
                raise RuntimeError("batch order does not match the spectrum table")
            batch = model.transfer_batch_to_device(batch, device, dataloader_idx=0)
            out = model(**batch, return_explainability_tensors=True)
            # 3. Eval-convention target: relative filter and normalization, no transform.
            true_mzs, true_ints, true_idx = model.preproc_spec(
                batch["spec_mzs"],
                batch["spec_ints"],
                batch["spec_batch_idxs"],
                train=False,
                pred=False,
                batch_size=size,
            )
            true_mzs = true_mzs.double().cpu().numpy()
            true_ints = true_ints.double().cpu().numpy()
            true_idx = true_idx.cpu().numpy()
            pred_mzs = out["pred_mzs"].double().cpu().numpy()
            pred_probs = out["pred_logprobs"].double().exp().cpu().numpy()
            pred_idx = out["pred_batch_idxs"].cpu().numpy()
            pred_frag = out["pred_pathway_fragment_idxs"].cpu().numpy()
            pred_h = out["pred_pathway_h_shifts"].cpu().numpy()
            pred_mode = out["pred_pathway_ion_mode_idxs"].cpu().numpy()
            member_atoms = out["fragment_atom_indices"].cpu().numpy()
            member_frag = out["fragment_atom_fragment_idxs"].cpu().numpy()
            frag_view = out["ion_h_fragment_batch_idxs"].cpu().numpy()
            atom_ptr = batch["mol_pyg"].ptr.cpu().numpy()
            inst_idx = batch["spec_inst_type"].reshape(-1).cpu().numpy()
            order = np.argsort(member_frag, kind="stable")
            member_frag, member_atoms = member_frag[order], member_atoms[order]
            frag_start = np.searchsorted(member_frag, np.arange(frag_view.size))
            frag_stop = np.searchsorted(member_frag, np.arange(frag_view.size), side="right")
            if np.any(np.diff(frag_view) < 0):
                raise RuntimeError("fragments are not grouped by spectrum")
            view_start = np.searchsorted(frag_view, np.arange(size))
            view_stop = np.searchsorted(frag_view, np.arange(size), side="right")

            for view, row in enumerate(rows.itertuples()):
                # 4. Merge this spectrum's channels into candidate peaks and map the target.
                channels = np.flatnonzero(pred_idx == view)
                probs = pred_probs[channels] / pred_probs[channels].sum()
                cand_mzs, pred, inverse = merge_channels(pred_mzs[channels], probs)
                peaks = true_idx == view
                peak_mzs, peak_ints = true_mzs[peaks], true_ints[peaks]
                rel, min_mz = tolerance[int(inst_idx[view])]
                on_cand, peak_to_cand = map_targets(peak_mzs, peak_ints, cand_mzs, rel, min_mz)
                parts = cosine_decomposition(pred, on_cand, float(np.linalg.norm(peak_ints)))
                outcome = base_peak_outcome(
                    peak_mzs, peak_ints, peak_to_cand, cand_mzs, pred, rel, min_mz
                )
                pred_rank = descending_ranks(pred)
                tgt_rank = descending_ranks(on_cand)
                base = int(np.argmax(peak_ints))
                # 5. Attributes of every candidate: per fragment from its membership row, per
                # channel with its H shift, then the model-weighted mean per candidate peak.
                if row.mol_id not in chem_cache:
                    chem_cache[row.mol_id] = MoleculeChemistry(molecules.loc[row.mol_id])
                chem = chem_cache[row.mol_id]
                if atom_ptr[view + 1] - atom_ptr[view] != chem.n_atoms:
                    raise RuntimeError(f"spectrum {row.spec_id}: graph and molecule atoms differ")
                first, last = view_start[view], view_stop[view]
                members = slice(frag_start[first], frag_stop[last - 1])
                inside = np.zeros((last - first, chem.n_atoms), dtype=bool)
                inside[member_frag[members] - first, member_atoms[members] - atom_ptr[view]] = True
                x = channel_attributes(
                    fragment_attributes(inside, chem),
                    pred_frag[channels] - first,
                    pred_h[channels],
                    chem.n_atoms,
                )
                weight = np.maximum(pred, 1e-300)[:, None]
                cand_x = (
                    np.column_stack(
                        [
                            np.bincount(inverse, probs * x[:, a], cand_mzs.size)
                            for a in range(x.shape[1])
                        ]
                    )
                    / weight
                )
                cand_x2 = (
                    np.column_stack(
                        [
                            np.bincount(inverse, probs * x[:, a] ** 2, cand_mzs.size)
                            for a in range(x.shape[1])
                        ]
                    )
                    / weight
                )
                ambiguous = (cand_x2 - cand_x**2) > 1e-9
                covered_mass = on_cand.sum()
                calibration = {}
                for a, name in enumerate(CANDIDATE_ATTRIBUTES):
                    calibration[f"ep_{name}"] = float(probs @ x[:, a])
                    calibration[f"et_{name}"] = (
                        float(on_cand @ cand_x[:, a] / covered_mass) if covered_mass > 0 else np.nan
                    )
                    calibration[f"amb_{name}"] = (
                        float(on_cand @ ambiguous[:, a] / covered_mass)
                        if covered_mass > 0
                        else np.nan
                    )
                keep_cands = (pred >= SAVE_FLOOR) | (on_cand > 0)
                saved["spec_id"].append(int(row.spec_id))
                saved["count"].append(int(keep_cands.sum()))
                saved["pred"].append(pred[keep_cands].astype(np.float32))
                saved["tgt"].append(on_cand[keep_cands].astype(np.float32))
                saved["x"].append(cand_x[keep_cands].astype(np.float32))
                saved["target_norm"].append(float(np.linalg.norm(peak_ints)))
                spectrum_rows.append(
                    {
                        "spec_id": int(row.spec_id),
                        "mol_id": int(row.mol_id),
                        "outcome": outcome,
                        "base_mz": float(peak_mzs[base]),
                        "base_cand": int(peak_to_cand[base]),
                        "base_pred_rank": (
                            int(pred_rank[peak_to_cand[base]]) if peak_to_cand[base] >= 0 else -1
                        ),
                        "pred_top_mz": float(cand_mzs[int(np.argmax(pred))]),
                        "pred_top_share": float(pred.max()),
                        "n_candidates": int(cand_mzs.size),
                        "n_peaks": int(peak_mzs.size),
                        **parts,
                        **calibration,
                    }
                )
                # 6. Structures behind the model's and the target's top-k candidate peaks.
                selected = set(np.flatnonzero(pred_rank <= args.top_k).tolist())
                selected |= set(np.flatnonzero((tgt_rank <= args.top_k) & (on_cand > 0)).tolist())
                for cand in sorted(selected):
                    structures = []
                    for channel in np.flatnonzero(inverse == cand):
                        fragment = int(pred_frag[channels[channel]])
                        atoms = member_atoms[frag_start[fragment] : frag_stop[fragment]]
                        structures.append(
                            (
                                tuple(sorted((atoms - atom_ptr[view]).tolist())),
                                int(pred_h[channels[channel]]),
                                int(pred_mode[channels[channel]]),
                                float(probs[channel]),
                            )
                        )
                    peak_rows.append(
                        {
                            "spec_id": int(row.spec_id),
                            "cand": int(cand),
                            "mz": float(cand_mzs[cand]),
                            "pred": float(pred[cand]),
                            "tgt": float(on_cand[cand]),
                            "pred_rank": int(pred_rank[cand]),
                            "tgt_rank": int(tgt_rank[cand]) if on_cand[cand] > 0 else -1,
                            "structures": structures,
                        }
                    )
            cursor += size
            if cursor % (args.batch_size * 20) == 0 or cursor == keep.size:
                LOGGER.info("%d/%d spectra, %.0f s", cursor, keep.size, time.time() - started)

    # 7. Write the tables and candidate arrays, and close the provenance record.
    spectra = pd.DataFrame(spectrum_rows)
    spectra.to_pickle(output)
    pd.DataFrame(peak_rows).to_pickle(output.with_suffix(".peaks.pkl"))
    np.savez_compressed(
        output.with_suffix(".cands.npz"),
        spec_id=np.asarray(saved["spec_id"]),
        offsets=np.concatenate([[0], np.cumsum(saved["count"])]),
        pred=np.concatenate(saved["pred"]),
        tgt=np.concatenate(saved["tgt"]),
        x=np.concatenate(saved["x"]),
        target_norm=np.asarray(saved["target_norm"]),
        attributes=np.asarray(CANDIDATE_ATTRIBUTES),
    )
    provenance["elapsed_seconds"] = time.time() - started
    provenance["status"] = "complete"
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    LOGGER.info(
        "cos %.4f | coverage %.4f ranking %.4f calibration %.4f | %s",
        spectra.cos.mean(),
        spectra.coverage.mean(),
        spectra.ranking.mean(),
        spectra.calibration.mean(),
        spectra.outcome.value_counts(normalize=True).round(4).to_dict(),
    )


if __name__ == "__main__":
    main()
