"""Export ``scripts/run_inference.py`` predictions as MGF with per-peak fragment SMILES.

Each kept peak is annotated with its most probable fragment: the argmax of
``pred_formula_node_probs`` over the joint rows of the peak's formula row, as described in
``docs/guides/run_inference.md#fragment-annotation``. The fragment is written as the
precursor's own atoms with their parent hydrogen counts and formal charges, so atoms at a cut
bond appear as radicals. The ion is that fragment plus a signed element delta (``+H``,
``-H2``, ``+Na``, empty for none), with the precursor's charge::

    137.0597 1000.0 "[CH2]c1ccc(O)c(OC)c1"
    144.0478 174.7 "CC1=C(CCO)SC=[N+]1 +H"

The delta is the ion formula (neutral formula plus ``pred_formula_adducts``) minus the
fragment's atoms, so it folds the model's hydrogen shift and the precursor adduct together.

Several prediction directories (for example one per fold checkpoint) are averaged: each
spectrum is sum-normalized per directory, peaks are matched by 4-decimal m/z, and fragment
shares are averaged with the same weights.

Three files are written: the annotated MGF, a ``*_plain.mgf`` copy with two-column peak
lines (pyteomics and matchms read a third peak column as a charge and reject the annotated
file), and a ``*_annotations.csv`` table with every contributing fragment of every kept peak.

Example::

    python scripts/misc/export_annotated_mgf.py \\
        --pred_dp output/run/pred_cv0 output/run/pred_cv1 \\
        --mol_fp output/run/mol_df.pkl.gz --spec_fp output/run/spec_df.pkl.gz \\
        --output_fp output/run/predictions.mgf
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re

import numpy as np
import pandas as pd
from pyteomics.mass import Composition
from rdkit import Chem

from fragnnet.models.fragnnet.frag_utils import th_long_to_mask
from fragnnet.utils.data_utils import ace_to_nce_helper, get_charge, nce_to_ace_helper
from fragnnet.utils.formula_utils import (
    PREC_TYPE_TO_CMF_COMP_DIFF,
    PREC_TYPE_TO_COMP_DIFF,
    formula_delta_to_signed_str,
    get_formulae_hill_notation,
)

logger = logging.getLogger(__name__)

# Fragmentation mode written per instrument: Orbitrap collision cells are HCD, QTOF is CID.
FRAG_MODE_BY_INST = {"ORB": "HCD", "ORB_F": "HCD", "QTOF": "CID"}
ORBITRAP_INST_TYPES = {"ORB", "ORB_F"}
REQUIRED_COLUMNS = (
    "spec_id",
    "pred_mzs",
    "pred_ints",
    "frag_formula_peak_idxs",
    "pred_formula_strs",
    "pred_formula_adducts",
    "pred_joint_formula_idxs",
    "pred_joint_node_idxs",
    "pred_formula_node_probs",
    "pred_node_ccs",
)


def fragment_smiles(mol: Chem.Mol, atoms: tuple[int, ...]) -> str:
    """Write an atom subset of a molecule as SMILES.

    Every kept atom keeps its parent hydrogen count and formal charge; a bond to a removed
    atom becomes a radical. Rings that lose an atom are written in Kekulé form.

    Args:
        mol: Parent molecule; atom indices follow its atom order.
        atoms: Indices of the fragment's atoms.

    Returns:
        Canonical SMILES of the fragment.
    """
    fragment = Chem.RWMol(mol)
    Chem.Kekulize(fragment, clearAromaticFlags=True)
    for atom in fragment.GetAtoms():
        atom.SetNumExplicitHs(atom.GetTotalNumHs())
        atom.SetNoImplicit(True)
    for idx in sorted(set(range(fragment.GetNumAtoms())) - set(atoms), reverse=True):
        fragment.RemoveAtom(idx)
    fragment = fragment.GetMol()
    Chem.SanitizeMol(fragment)
    return Chem.MolToSmiles(fragment)


def parse_signed_delta(delta: str) -> Composition:
    """Parse a signed adduct string such as ``"+H"``, ``"-H+Na"`` or ``"-CHO2"``.

    Args:
        delta: Signed delta as written by ``formula_delta_to_signed_str``; ``""`` is zero.

    Returns:
        Element counts of the delta, negative for removed atoms.

    Raises:
        ValueError: If the string is not a sequence of signed formula groups.
    """
    groups = re.findall(r"([+-])([^+-]+)", delta)
    if "".join(sign + group for sign, group in groups) != delta:
        raise ValueError(f"not a signed delta: {delta!r}")
    counts = Composition()
    for sign, group in groups:
        for element, count in Composition(formula=group).items():
            counts[element] += count if sign == "+" else -count
    return counts


def _counts_str(counts: dict[str, int]) -> str:
    return "".join(f"{element}{count}" for element, count in counts.items() if count)


def _heavy(counts: dict[str, int]) -> dict[str, int]:
    return {element: count for element, count in counts.items() if element != "H" and count}


def load_predictions(pred_dp: str) -> pd.DataFrame:
    """Concatenate the ``predictions_chunk_*`` pickles of one ``run_inference`` directory."""
    files = sorted(glob.glob(os.path.join(pred_dp, "predictions_chunk_*.pkl*")))
    if not files:
        raise FileNotFoundError(f"no predictions_chunk_*.pkl[.gz] files in {pred_dp}")
    pred_df = pd.concat([pd.read_pickle(fp) for fp in files], ignore_index=True)
    missing = [c for c in REQUIRED_COLUMNS if c not in pred_df.columns]
    if missing:
        raise ValueError(
            f"{pred_dp} lacks {missing}. Run run_inference.py with --return_formula_str True "
            "--return_cc_hash True and pred_formula_node_probs in --output_subset "
            "(or --output_everything True)."
        )
    if pred_df["spec_id"].duplicated().any():
        raise ValueError(f"{pred_dp} holds duplicate spec_ids")
    return pred_df.set_index("spec_id")


def annotate_spectrum(
    pred_rows: list[pd.Series],
    mol: Chem.Mol,
    prec_type: str,
    min_rel_intensity: float,
) -> list[dict]:
    """Average one spectrum's predictions and annotate every kept peak.

    With ``w_d(k)`` the sum-normalized intensity of peak ``k`` in prediction ``d`` of ``D``,
    the peak's intensity is ``mean_d w_d(k)`` and a fragment's share of the peak is
    ``sum_d w_d(k) P_d(n | f(k)) / sum_d w_d(k)``.

    Args:
        pred_rows: The spectrum's prediction row from each prediction directory.
        mol: Parent molecule the fragment bitmasks index.
        prec_type: Precursor type of the spectrum.
        min_rel_intensity: Peaks under this fraction of the base peak are dropped.

    Returns:
        One dict per kept peak, sorted by m/z: ``mz``, ``intensity`` (base peak 1000),
        ``formula`` (ion, with charge sign), and ``fragments``, a list of ``(smiles, delta, share)`` sorted
        by decreasing share.

    Raises:
        ValueError: If a kept peak has no candidate fragment, or a fragment's atoms are
            inconsistent with its peak's ion formula.
    """
    # 1. Accumulate intensity, ion formula and fragment weights per 4-decimal m/z.
    allowed_heavy = [
        _heavy(PREC_TYPE_TO_COMP_DIFF[prec_type]),
        _heavy(PREC_TYPE_TO_CMF_COMP_DIFF.get(prec_type, {})),
    ]
    peaks: dict[float, dict] = {}
    for row in pred_rows:
        ints = np.asarray(row["pred_ints"], dtype=np.float64)
        weights = ints / ints.sum()
        joint_formulas = np.asarray(row["pred_joint_formula_idxs"])
        joint_nodes = np.asarray(row["pred_joint_node_idxs"])
        joint_probs = np.asarray(row["pred_formula_node_probs"], dtype=np.float64)
        node_atoms = [
            tuple(np.flatnonzero(m).tolist()) for m in th_long_to_mask(row["pred_node_ccs"]).numpy()
        ]
        for k, (mz, weight) in enumerate(zip(np.asarray(row["pred_mzs"]), weights)):
            f = int(row["frag_formula_peak_idxs"][k])
            ion = Composition(formula=row["pred_formula_strs"][f]) + parse_signed_delta(
                row["pred_formula_adducts"][f]
            )
            peak = peaks.setdefault(round(float(mz), 4), {"weight": 0.0, "ions": {}, "nodes": {}})
            peak["weight"] += weight / len(pred_rows)
            ion_key = _counts_str(ion)
            peak["ions"][ion_key] = peak["ions"].get(ion_key, 0.0) + weight
            for cell in np.flatnonzero(joint_formulas == f):
                atoms = node_atoms[joint_nodes[cell]]
                peak["nodes"][atoms] = peak["nodes"].get(atoms, 0.0) + weight * joint_probs[cell]

    # 2. Keep peaks above the threshold and resolve their fragments.
    base = max(peak["weight"] for peak in peaks.values())
    annotated = []
    for mz in sorted(peaks):
        peak = peaks[mz]
        if peak["weight"] < min_rel_intensity * base:
            continue
        if not peak["nodes"]:
            raise ValueError(f"peak {mz} has no candidate fragment")
        ion = Composition(formula=max(peak["ions"], key=peak["ions"].get))
        total = sum(peak["nodes"].values())
        fragments = []
        for atoms, node_weight in sorted(peak["nodes"].items(), key=lambda kv: -kv[1]):
            counts = Composition()
            for idx in atoms:
                atom = mol.GetAtomWithIdx(idx)
                counts[atom.GetSymbol()] += 1
                counts["H"] += atom.GetTotalNumHs()
            delta = {el: ion.get(el, 0) - counts.get(el, 0) for el in set(ion) | set(counts)}
            if _heavy(delta) not in allowed_heavy:
                raise ValueError(
                    f"peak {mz}: atoms {atoms} do not build ion {_counts_str(ion)} "
                    f"for {prec_type}"
                )
            fragments.append(
                (
                    fragment_smiles(mol, atoms),
                    formula_delta_to_signed_str(_counts_str(delta)),
                    node_weight / total,
                )
            )
        annotated.append(
            {
                "mz": mz,
                "intensity": peak["weight"] / base * 1000.0,
                "formula": get_formulae_hill_notation({el: n for el, n in ion.items() if n})
                + ("+" if get_charge(prec_type) > 0 else "-"),
                "fragments": fragments,
            }
        )
    return annotated


def spectrum_header(spec: pd.Series, mol_entry: pd.Series, source: str) -> list[str]:
    """MGF header lines for one spectrum.

    ``COLLISION_ENERGY`` carries its unit: ``% NCE`` on Orbitraps, ``eV`` elsewhere. A
    missing value is converted from the other with the dataset's own NCE <-> eV rule, and
    the line is omitted when the spectrum has neither.
    """
    charge = get_charge(spec["prec_type"])
    inst = spec["inst_type"]
    nce, ace = spec["nce"], spec["ace"]
    collision_energy, ce_tag = None, "noCE"
    if inst in ORBITRAP_INST_TYPES:
        if pd.isna(nce) and pd.notna(ace):
            nce = ace_to_nce_helper(ace, abs(charge), spec["prec_mz"])
        if pd.notna(nce):
            collision_energy, ce_tag = f"{nce:.1f}% NCE", f"NCE{nce:.1f}"
    else:
        if pd.isna(ace) and pd.notna(nce):
            ace = nce_to_ace_helper(nce, abs(charge), spec["prec_mz"])
        if pd.notna(ace):
            collision_energy, ce_tag = f"{ace:g} eV", f"{ace:g}eV"
    name = mol_entry["input_mol_id"] if "input_mol_id" in mol_entry else mol_entry.name
    return [
        f"TITLE={name}_{spec['prec_type']}_{inst}_{ce_tag}",
        f"SPECTRUM_ID={spec.name}",
        f"PEPMASS={spec['prec_mz']:.4f}",
        f"CHARGE={abs(charge)}{'+' if charge > 0 else '-'}",
        f"ION={spec['prec_type']}",
        f"IONMODE={'positive' if charge > 0 else 'negative'}",
        f"SMILES={mol_entry['smiles']}",
        f"FORMULA={mol_entry['formula']}",
        f"INSTRUMENT_TYPE={inst}",
        f"FRAGMENTATION_MODE={FRAG_MODE_BY_INST.get(inst, spec['frag_mode'])}",
        *([f"COLLISION_ENERGY={collision_energy}"] if collision_energy else []),
        f"SOURCE={source}",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--pred_dp",
        nargs="+",
        required=True,
        help="run_inference.py --save_dp directories; several are averaged per spectrum",
    )
    parser.add_argument("--mol_fp", required=True, help="mol_df given to run_inference.py")
    parser.add_argument("--spec_fp", required=True, help="spec_df given to run_inference.py")
    parser.add_argument("--output_fp", required=True, help="annotated .mgf path")
    parser.add_argument(
        "--min_rel_intensity",
        type=float,
        default=0.01,
        help="drop peaks under this fraction of the base peak (default 0.01 = 1%%)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 1. Load inputs; every directory must cover the same spectra.
    mol_df = pd.read_pickle(args.mol_fp).set_index("mol_id")
    spec_df = pd.read_pickle(args.spec_fp).set_index("spec_id")
    preds = [load_predictions(dp) for dp in args.pred_dp]
    spec_ids = sorted(preds[0].index)
    for dp, pred in zip(args.pred_dp, preds):
        if sorted(pred.index) != spec_ids:
            raise ValueError(f"{dp} covers different spec_ids than {args.pred_dp[0]}")
    source = "mean prediction of " + ", ".join(
        os.path.basename(os.path.normpath(dp)) for dp in args.pred_dp
    )

    # 2. Annotate every spectrum and render its MGF block and table rows.
    annotated_lines, plain_lines, rows = [], [], []
    for spec_id in spec_ids:
        spec = spec_df.loc[spec_id]
        mol_entry = mol_df.loc[spec["mol_id"]]
        peaks = annotate_spectrum(
            [pred.loc[spec_id] for pred in preds],
            mol_entry["mol"],
            spec["prec_type"],
            args.min_rel_intensity,
        )
        header = ["BEGIN IONS", *spectrum_header(spec, mol_entry, source)]
        header.append(f"NUM_PEAKS={len(peaks)}")
        annotated_lines += header
        plain_lines += header
        for peak in peaks:
            smiles, delta, _share = peak["fragments"][0]
            label = f"{smiles} {delta}" if delta else smiles
            annotated_lines.append(f'{peak["mz"]:.4f} {peak["intensity"]:.1f} "{label}"')
            plain_lines.append(f'{peak["mz"]:.4f} {peak["intensity"]:.1f}')
            for rank, (frag_smiles, frag_delta, share) in enumerate(peak["fragments"]):
                rows.append(
                    {
                        "spec_id": spec_id,
                        "title": header[1].removeprefix("TITLE="),
                        "mz": peak["mz"],
                        "intensity": round(peak["intensity"], 1),
                        "ion_formula": peak["formula"],
                        "fragment_rank": rank,
                        "frag_smiles": frag_smiles,
                        "delta": frag_delta,
                        "share": round(share, 4),
                    }
                )
        annotated_lines += ["END IONS", ""]
        plain_lines += ["END IONS", ""]

    # 3. Write the annotated MGF, its plain copy and the fragment table.
    stem = args.output_fp.removesuffix(".mgf")
    with open(args.output_fp, "w") as f:
        f.write("\n".join(annotated_lines))
    with open(stem + "_plain.mgf", "w") as f:
        f.write("\n".join(plain_lines))
    pd.DataFrame(rows).to_csv(stem + "_annotations.csv", index=False)
    logger.info(
        "wrote %d spectra to %s (+ _plain.mgf, _annotations.csv)", len(spec_ids), args.output_fp
    )


if __name__ == "__main__":
    main()
