"""G0 of the rewire-rules training plan: rule reach on the candidate pool i1 actually keeps.

The rule gate (``debug_masster_rewire_rule_gate.py``) measures R_all: a rule applied to every
connected fragment that contains its motif, then ordinary cuts. Training applies a rule only to
the fragments the model KEEPS (``masster_unique_fragment_target`` unique sets under
``mask_order``), so this script reads i1's kept pool from an annotated forward pass on the
validation split and measures, per mined rearrangement peak:

- ``after0``: some kept fragment F contains a full rule match, and a connected piece of the
  edited F (F minus deleted atoms, minus rule cuts, plus formed bonds) has the peak's heavy
  composition with own H within +-4 (the stage with ``masster_rewire_continuation_cuts: 0``);
- ``after1``: the same, or a piece after one more cut of a precursor bond inside a depth-0
  product (an upper bound of ``continuation_cuts`` > 0, which cuts only the top-k bonds);
- ``null0`` / ``null1``: the same with random edits of the same shape (the gate's topology null,
  same number of edits, three seeded draws, real sites excluded);
- ``all``: R_all for the same peak, read from the gate's rows.

It also counts extras per spectrum (distinct product atom sets not already in the kept pool,
before the stage's per-precursor cap) for the node-growth estimate.

Run from the repository root (GPU optional):
    python scripts/debug/smarts/debug_masster_rewire_reach_ceiling.py --out output/rewire_g0_2026_10_15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))

import debug_masster_rewire_rule_gate as G  # noqa: E402

from fragnnet.inference.checkpoint import (  # noqa: E402
    apply_inference_config,
    get_config_from_ckpt,
    init_model_from_ckpt,
)
from fragnnet.runner import load_config  # noqa: E402
from fragnnet.utils.misc_utils import deep_update  # noqa: E402
from fragnnet.utils.setup_utils import get_dataset_cls  # noqa: E402

CHECKPOINT = "saved_ckpts/i1_inst_ob_obf_i50ig7rf/ckpt/model-epoch=059.ckpt"
I1_CONFIG = "config/masster/mh_ft_memorization_route_2026_09_24/i1_inst_ob_obf.yml"
RULES = "output/rewire_rule_gate_2026_10_15/survivors.json"
MINING = Path("output/rearrangement_mining_2026_10_14")
GATE_ROWS = "output/rewire_rule_gate_2026_10_15/survivors/rows.pkl"
NULL_DRAWS = 3


# 1. Edits inside a kept fragment -----------------------------------------------------------


def _components(atoms: frozenset[int], adjacency: dict[int, set[int]]) -> list[frozenset[int]]:
    seen, pieces = set(), []
    for start in atoms:
        if start in seen:
            continue
        stack, piece = [start], set()
        while stack:
            atom = stack.pop()
            if atom in piece:
                continue
            piece.add(atom)
            stack.extend(n for n in adjacency[atom] if n in atoms and n not in piece)
        seen |= piece
        pieces.append(frozenset(piece))
    return pieces


def edited_products(
    fragment: frozenset[int], match: dict[int, int], rule: dict, bonds: set[frozenset[int]]
) -> list[tuple[frozenset[int], frozenset[frozenset[int]]]]:
    """Connected pieces of F after the rule's edit, each with the formed bonds it holds.

    Returns nothing unless F contains every mapped atom of the match. Deleted atoms leave in the
    neutral and are never part of a product.
    """
    if not set(match.values()) <= fragment:
        return []
    deleted = {match[m] for m in rule.get("delete", []) if m in match}
    cut = {frozenset((match[a], match[b])) for a, b in rule.get("cut", [])}
    formed = {frozenset((match[a], match[b])) for a, b in rule.get("form", [])}
    kept = frozenset(fragment - deleted)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for bond in (bonds - cut) | formed:
        a, b = tuple(bond)
        if a in kept and b in kept:
            adjacency[a].add(b)
            adjacency[b].add(a)
    out = []
    for piece in _components(kept, adjacency):
        out.append((piece, frozenset(f for f in formed if f <= piece)))
    return out


def one_cut_pieces(
    piece: frozenset[int], formed: frozenset[frozenset[int]], bonds: set[frozenset[int]]
) -> set[frozenset[int]]:
    """Pieces after cutting one precursor bond inside a product (formed bonds are never cut)."""
    inside = [b for b in bonds if b <= piece]
    out = set()
    for removed in inside:
        adjacency: dict[int, set[int]] = defaultdict(set)
        for bond in set(inside) - {removed} | set(formed):
            a, b = tuple(bond)
            adjacency[a].add(b)
            adjacency[b].add(a)
        out.update(_components(piece, adjacency))
    return out


# 2. Per-spectrum evaluation ------------------------------------------------------------------


def _composition_index(mol, sets) -> dict[tuple, list[int]]:
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    hydrogens = [a.GetTotalNumHs() for a in mol.GetAtoms()]
    index: dict[tuple, list[int]] = defaultdict(list)
    for atoms in sets:
        key = tuple(sorted(Counter(symbols[i] for i in atoms).items()))
        index[key].append(sum(hydrogens[i] for i in atoms))
    return index


def _hit(index: dict[tuple, list[int]], ion: str) -> bool:
    comp = G.parse_formula(ion)
    h = comp.pop("H", 0)
    key = tuple(sorted(comp.items()))
    return any(abs(h - own) <= 4 for own in index.get(key, ()))


def products_for(pool, matches, rules_by_name, bonds, depth1: bool):
    """Distinct product atom sets of every (kept F, match) pair, depth 0 and optionally 1."""
    depth0: dict[frozenset[int], frozenset] = {}
    for rule_name, match in matches:
        rule = rules_by_name[rule_name]
        for fragment in pool:
            for piece, formed in edited_products(fragment, match, rule, bonds):
                if formed:  # a piece without the new bond is an ordinary connected set
                    depth0.setdefault(piece, formed)
    with_cut = set(depth0)
    if depth1:
        for piece, formed in depth0.items():
            # Keep only pieces that still hold a formed bond; the rest are ordinary sets.
            with_cut |= {p for p in one_cut_pieces(piece, formed, bonds) if any(f <= p for f in formed)}
    return set(depth0), with_cut


def null_matches(mol, rule: dict, real: list[dict[int, int]], key: str) -> list[list[dict[int, int]]]:
    """The gate's null: same edit shape at random sites, as many edits as real matches."""
    rng = np.random.default_rng(zlib.crc32(f"{key}|{rule['name']}".encode()))
    null_maps = [a.GetAtomMapNum() for a in rule["_null_pattern"].GetAtoms()]
    real_atoms = {frozenset(m[k] for k in null_maps if k in m) for m in real}
    pool = [m for m in G._matches(mol, rule["_null_pattern"]) if frozenset(m.values()) not in real_atoms]
    if not pool:
        return []
    return [
        [pool[i] for i in rng.choice(len(pool), min(len(real), len(pool)), replace=False)]
        for _ in range(NULL_DRAWS)
    ]


# 3. Main -----------------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--rules", default=RULES)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="matching molecules, for a smoke run")
    parser.add_argument("--no-depth1", action="store_true")
    args = parser.parse_args(argv)
    RDLogger.DisableLog("rdApp.*")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # 3.1 i1 from its checkpoint. Checkpoint hparams are the config; the template and i1's YAML
    # only fill keys the checkpoint predates (deep merge, so nested dicts keep new keys).
    saved = get_config_from_ckpt(args.checkpoint)
    config = deep_update(load_config("config", I1_CONFIG, []), saved)
    config = apply_inference_config(
        config, disable_preproc=True, num_workers=0,
        train_batch_size=args.batch_size, eval_batch_size=args.batch_size,
    )
    for key, expected in (
        ("masster_smarts_templates", False),
        ("masster_unique_fragment_mode", True),
        ("masster_fragment_selection", "mask_order"),
    ):
        if config.get(key) != expected:
            raise ValueError(f"expected {key}={expected!r}, checkpoint has {config.get(key)!r}")
    device = th.device(args.device if th.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cpu":
        config["masster_candidate_backend"] = "cpu"
    model = init_model_from_ckpt(args.checkpoint, config, device)
    model.eval()
    dataset = get_dataset_cls(config["model_type"])(split="val", **config)
    specs = dataset.spec_df.reset_index(drop=True)
    molecules = pd.read_pickle(config["mol_fp"]).set_index("mol_id")["mol"]

    # 3.2 Rule matches per validation molecule; only matching molecules need a forward pass.
    rules = json.loads(Path(args.rules).read_text())
    G._init(rules, "[M+H]+")
    rules_by_name = {r["name"]: r for r in G._RULES}
    real_by_mol, null_by_mol = {}, {}
    for mol_id in specs.mol_id.unique():
        mol = molecules.loc[mol_id]
        real = [(r["name"], m) for r in G._RULES for m in G._matches(mol, r["_pattern"])]
        if not real:
            continue
        real_by_mol[mol_id] = real
        draws: list[list[tuple[str, dict]]] = [[] for _ in range(NULL_DRAWS)]
        for r in G._RULES:
            mine = [m for name, m in real if name == r["name"]]
            if mine:
                for d, picked in enumerate(null_matches(mol, r, mine, f"nist20:{mol_id}")):
                    draws[d] += [(r["name"], m) for m in picked]
        null_by_mol[mol_id] = draws
    matching = sorted(real_by_mol)
    if args.limit:
        matching = matching[: args.limit]
    print(
        f"val: {specs.mol_id.nunique()} molecules, {len(specs)} spectra; "
        f"{len(real_by_mol)} molecules match a rule; scoring {len(matching)}",
        flush=True,
    )

    # 3.3 Mined rearrangement peaks of these spectra and the gate's R_all per (molecule, ion).
    rearr = pd.read_pickle(MINING / "rearrangement.pkl")
    rearr = rearr[(rearr.dataset == "nist20") & (rearr.split == "val")]
    share = rearr.groupby("spec_id").weight.sum()
    gate = pd.read_pickle(GATE_ROWS).dropna(subset=["ion"])
    gate_hit = gate.assign(h=gate.real == "hit").groupby(["key", "ion"]).h.any().to_dict()
    peaks_by_spec = {sid: g for sid, g in rearr.groupby("spec_id")}

    keep = np.flatnonzero(specs.mol_id.isin(set(matching)).to_numpy())
    loader = DataLoader(
        Subset(dataset, keep), batch_size=args.batch_size, shuffle=False,
        collate_fn=dataset.get_collate_fn(), num_workers=0,
    )
    peak_rows, spec_rows, cursor = [], [], 0
    bond_cache, index_cache = {}, {}
    with th.inference_mode():
        for batch in loader:
            rows = specs.iloc[keep[cursor : cursor + int(batch["batch_size"])]]
            batch = model.transfer_batch_to_device(batch, device, dataloader_idx=0)
            out_t = model(**batch, return_explainability_tensors=True)
            atom_ptr = batch["mol_pyg"].ptr.cpu().numpy()
            atoms = out_t["fragment_atom_indices"].cpu().numpy()
            members = out_t["fragment_atom_fragment_idxs"].cpu().numpy()
            views = out_t["ion_h_fragment_batch_idxs"].cpu().numpy()
            pools: dict[int, set[frozenset[int]]] = defaultdict(set)
            order = np.argsort(members, kind="stable")
            ids, starts = np.unique(members[order], return_index=True)
            for fragment, group in zip(ids, np.split(atoms[order], starts[1:]), strict=True):
                view = int(views[fragment])
                pools[view].add(frozenset((group - atom_ptr[view]).tolist()))
            for view, row in enumerate(rows.itertuples()):
                mol = molecules.loc[row.mol_id]
                n_atoms = int(atom_ptr[view + 1] - atom_ptr[view])
                if n_atoms != mol.GetNumAtoms():
                    raise ValueError(f"graph/mol atom count mismatch for mol {row.mol_id}")
                if row.mol_id not in bond_cache:
                    bond_cache[row.mol_id] = {
                        frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()
                    }
                bonds = bond_cache[row.mol_id]
                pool = pools[view]
                real0, real1 = products_for(pool, real_by_mol[row.mol_id], rules_by_name, bonds, not args.no_depth1)
                index0, index1 = _composition_index(mol, real0), _composition_index(mol, real1)
                nulls = [
                    products_for(pool, draw, rules_by_name, bonds, not args.no_depth1)
                    for draw in null_by_mol[row.mol_id]
                ]
                null_index = [(_composition_index(mol, a), _composition_index(mol, b)) for a, b in nulls]
                spec_rows.append(
                    {
                        "spec_id": row.spec_id, "mol_id": row.mol_id, "pool": len(pool),
                        "extras0": len(real0 - pool), "extras1": len(real1 - pool),
                        "null_extras0": float(np.mean([len(a - pool) for a, _ in nulls])) if nulls else np.nan,
                        "share": float(share.get(row.spec_id, 0.0)),
                    }
                )
                for peak in peaks_by_spec.get(row.spec_id, pd.DataFrame()).itertuples():
                    peak_rows.append(
                        {
                            "spec_id": row.spec_id, "mol_id": row.mol_id, "mz": peak.mz,
                            "weight": peak.weight, "ion": peak.ion,
                            "share": float(share.get(row.spec_id, 0.0)),
                            "all": bool(gate_hit.get((f"nist20:{row.mol_id}", peak.ion), False)),
                            "after0": _hit(index0, peak.ion), "after1": _hit(index1, peak.ion),
                            "null0": np.mean([_hit(a, peak.ion) for a, _ in null_index]) if null_index else np.nan,
                            "null1": np.mean([_hit(b, peak.ion) for _, b in null_index]) if null_index else np.nan,
                        }
                    )
            cursor += len(rows)
            if (cursor // len(rows)) % 50 == 0:
                print(f"{cursor}/{len(keep)} spectra, {time.time() - started:.0f}s", flush=True)
    peaks = pd.DataFrame(peak_rows)
    spectra = pd.DataFrame(spec_rows)
    peaks.to_pickle(out / "peaks.pkl")
    spectra.to_pickle(out / "spectra.pkl")

    # 3.4 Summary: % of ALL validation rearrangement intensity (whole and tail).
    lines = []
    for label, sel in (("all val", rearr), ("tail >=25%", rearr[rearr.spec_id.map(share) >= 0.25])):
        total = sel.weight.sum()
        p = peaks[peaks.spec_id.isin(sel.spec_id)]
        lines.append(
            {
                "subset": label,
                "R_all_pct": 100 * p.weight[p["all"]].sum() / total,
                "R_after0_pct": 100 * p.weight[p.after0].sum() / total,
                "R_after1_pct": 100 * p.weight[p.after1].sum() / total,
                "null0_pct": 100 * (p.weight * p.null0.fillna(0)).sum() / total,
                "null1_pct": 100 * (p.weight * p.null1.fillna(0)).sum() / total,
                "after1_over_all": p.weight[p.after1].sum() / max(p.weight[p["all"]].sum(), 1e-12),
                "survival_of_all_hits": p.weight[p["all"] & p.after1].sum() / max(p.weight[p["all"]].sum(), 1e-12),
            }
        )
    n_val_spectra = len(specs)
    growth = {
        "val_spectra": n_val_spectra,
        "scored_spectra": len(spectra),
        "mean_pool": spectra.pool.mean(),
        "extras0_mean_matching": spectra.extras0.mean(),
        "extras0_p95_matching": spectra.extras0.quantile(0.95),
        "extras0_max": spectra.extras0.max(),
        "extras1_mean_matching": spectra.extras1.mean(),
        "extras1_p95_matching": spectra.extras1.quantile(0.95),
        "extras0_capped64_mean_all_val": spectra.extras0.clip(upper=64).sum() / n_val_spectra,
        "node_growth_capped64_mean_pct": 100 * spectra.extras0.clip(upper=64).sum() / n_val_spectra / spectra.pool.mean(),
        "elapsed_s": time.time() - started,
    }
    summary = pd.DataFrame(lines)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "growth.json").write_text(json.dumps({k: float(v) for k, v in growth.items()}, indent=1))
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    print(json.dumps({k: round(float(v), 2) for k, v in growth.items()}, indent=1))


if __name__ == "__main__":
    main()
