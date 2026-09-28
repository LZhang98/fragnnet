"""Coverage gate for small-bridge excision followed by cleavage of c0 candidates.

Invoked by debug_masster_smarts_rule_gate.py continuations. This is an atom-set
coverage diagnostic, not a chemical mechanism simulator or a trained loss channel.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import random
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from rdkit import Chem
from torch.utils.data import DataLoader, Subset

from fragnnet.frag.masster_templates import (
    _is_connected,
    build_masster_templates,
    filter_masster_templates,
)
from fragnnet.runner import load_config
from fragnnet.utils.script_utils import (
    apply_inference_config,
    get_config_from_ckpt,
    init_model_from_ckpt,
)
from fragnnet.utils.setup_utils import get_dataset_cls

# Support both the script entry point and imports from pytest.
try:
    from .debug_masster_smarts_rule_gate import TOLERANCE, ChannelScorer, edge_tuple, reachable_h
except ImportError:
    from debug_masster_smarts_rule_gate import TOLERANCE, ChannelScorer, edge_tuple, reachable_h

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Bridge:
    """Connected removed atoms with two acyclic attachment bonds."""

    removed: frozenset[int]
    endpoints: frozenset[int]
    environment: str


def enumerate_bridges(mol: Chem.Mol, max_atoms: int = 3) -> list[Bridge]:
    """Enumerate connected X with two non-ring external bonds and distinct endpoints.

    Group keys are canonical local SMILES with atom maps 1 for X and 2 for retained
    endpoints. Original hydrogen counts are retained; keys do not encode remote chemistry.
    """
    adjacency = [{neighbor.GetIdx() for neighbor in a.GetNeighbors()} for a in mol.GetAtoms()]
    current = {frozenset((a,)) for a in range(mol.GetNumAtoms())}
    bridges = []
    for _ in range(max_atoms):
        following = set()
        for removed in sorted(current, key=lambda x: tuple(sorted(x))):
            boundary = [(a, b) for a in sorted(removed) for b in sorted(adjacency[a] - removed)]
            endpoints = frozenset(b for _, b in boundary)
            if (
                len(boundary) == 2
                and len(endpoints) == 2
                and not any(mol.GetAtomWithIdx(a).IsInRing() for a in removed)
                and not any(mol.GetBondBetweenAtoms(a, b).IsInRing() for a, b in boundary)
            ):
                local = Chem.RWMol(mol)
                for atom in local.GetAtoms():
                    atom.SetAtomMapNum(1 if atom.GetIdx() in removed else 2)
                    atom.SetNumExplicitHs(mol.GetAtomWithIdx(atom.GetIdx()).GetTotalNumHs())
                    atom.SetNoImplicit(True)
                for atom in reversed(range(mol.GetNumAtoms())):
                    if atom not in removed | endpoints:
                        local.RemoveAtom(atom)
                environment = Chem.MolToSmiles(local, canonical=True, isomericSmiles=False)
                bridges.append(Bridge(removed, endpoints, environment))
            for atom in set().union(*(adjacency[a] for a in removed)) - removed:
                following.add(removed | {atom})
        current = following
    return bridges


def continuation_sets(pool: set[tuple[int, ...]], bridge: Bridge) -> set[tuple[int, ...]]:
    """Lift connected S containing X,a,b to the product S minus X."""
    required = bridge.removed | bridge.endpoints
    return {
        tuple(a for a in atoms if a not in bridge.removed)
        for atoms in pool
        if required.issubset(atoms)
    }


def covered(peaks: np.ndarray, channels: np.ndarray, tolerance: tuple[float, float]) -> np.ndarray:
    """Match peaks to sorted channels with the instrument's relative/floor tolerance."""
    if not channels.size:
        return np.zeros(peaks.size, dtype=bool)
    channels = np.sort(channels)
    relative, floor = tolerance
    upper = np.clip(np.searchsorted(channels, peaks), 0, channels.size - 1)
    lower = np.maximum(upper - 1, 0)
    return (abs(peaks - channels[upper]) <= np.maximum(channels[upper], floor) * relative) | (
        abs(peaks - channels[lower]) <= np.maximum(channels[lower], floor) * relative
    )


class MoleculeCoverage:
    """Cache the channel and exhaustive same-composition queries for one molecule."""

    def __init__(self, mol: Chem.Mol, scorer: ChannelScorer, cap: int):
        self.mol, self.scorer, self.cap = mol, scorer, cap
        self.edges = edge_tuple(mol)
        self.symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        self.hydrogens = [a.GetTotalNumHs() for a in mol.GetAtoms()]
        self.channels = {}
        self.reaches = {}

    def prepare(self, sets: set[tuple[int, ...]], reconnect: bool) -> None:
        missing = sorted(s for s in sets if (s, reconnect) not in self.channels)
        if missing:
            mzs, valid = self.scorer(self.mol, missing, 0, reconnect=reconnect)
            for atoms, mass, allowed in zip(missing, mzs, valid, strict=True):
                own_h = sum(self.hydrogens[a] for a in atoms)
                self.channels[atoms, reconnect] = (mass[allowed], np.arange(-4, 5)[allowed] + own_h)

    def hit(
        self,
        sets: set[tuple[int, ...]],
        peaks: np.ndarray,
        tolerance: tuple,
        *,
        reconnect: bool = False,
        unique: bool = False,
    ) -> tuple[np.ndarray, int]:
        self.prepare(sets, reconnect)
        hit = np.zeros(peaks.size, bool)
        capped = 0
        for atoms in sorted(sets):
            masses, counts_h = self.channels[atoms, reconnect]
            matches = covered(peaks, masses, tolerance)
            if not matches.any():
                continue
            if unique:
                composition = Counter(self.symbols[a] for a in atoms)
                matched_channels = np.array(
                    [covered(peaks, np.array([mass]), tolerance).any() for mass in masses]
                )
                masses, counts_h = masses[matched_channels], counts_h[matched_channels]
                targets_h = frozenset(int(h) for h in counts_h)
                key = (tuple(sorted(composition.items())), targets_h)
                if key not in self.reaches:
                    self.reaches[key] = reachable_h(
                        self.mol,
                        self.edges,
                        composition,
                        len(atoms),
                        cap=self.cap,
                        required_h=targets_h,
                    )
                reach = self.reaches[key]
                if reach is None:
                    capped += 1
                    continue
                # Conservative: any connected twin within +-4 H excludes the channel,
                # even if its own valence filter would reject that particular H shift.
                keep = [not any(abs(h - r) <= 4 for r in reach) for h in counts_h]
                matches = covered(peaks, masses[np.asarray(keep, bool)], tolerance)
            hit |= matches
        return hit, capped


def random_disconnected(
    mol: Chem.Mol, products: set[tuple[int, ...]], rng: np.random.Generator
) -> tuple[set[tuple[int, ...]], int]:
    """Match product count and size with distinct random disconnected atom sets."""
    edges = edge_tuple(mol)
    nulls = set()
    for product in sorted(products):
        for _ in range(1000):
            draw = tuple(
                sorted(rng.choice(mol.GetNumAtoms(), len(product), replace=False).tolist())
            )
            if draw not in nulls and not _is_connected(draw, edges):
                nulls.add(draw)
                break
    return nulls, len(products) - len(nulls)


def evaluate_spectrum(
    mol: Chem.Mol,
    pool: set[tuple[int, ...]],
    pool_channels: np.ndarray,
    bridges: list[Bridge],
    templates: set[tuple[int, ...]],
    peaks: np.ndarray,
    weights: np.ndarray,
    tolerance: tuple,
    cache: MoleculeCoverage,
    rng: np.random.Generator,
    families: set[str] | None = None,
) -> list[dict]:
    """Compare continuation products to the pool, templates, whole products, and null."""
    ordinary = covered(peaks, pool_channels, tolerance)
    template_hit, _ = cache.hit(templates, peaks, tolerance)
    baseline = ordinary | template_hit
    whole = {tuple(a for a in range(mol.GetNumAtoms()) if a not in b.removed) for b in bridges}
    whole_hit, _ = cache.hit(whole, peaks, tolerance)
    # All generic whole products join the baseline to isolate post-excision cleavage.
    continuation_baseline = baseline | whole_hit
    by_environment = defaultdict(set)
    for bridge in bridges:
        if families is not None and bridge.environment not in families:
            continue
        by_environment[bridge.environment] |= continuation_sets(pool, bridge) - whole
    by_environment["__all__"] = set().union(*by_environment.values())
    records = []
    for environment, products in sorted(by_environment.items()):
        nulls, missing = random_disconnected(mol, products, rng)
        raw, _ = cache.hit(products, peaks, tolerance)
        remaining_peaks = peaks[~continuation_baseline]
        remaining_weights = weights[~continuation_baseline]
        unique, capped = cache.hit(products, remaining_peaks, tolerance, unique=True)
        strict, _ = cache.hit(products, remaining_peaks, tolerance, reconnect=True, unique=True)
        null, null_capped = cache.hit(nulls, remaining_peaks, tolerance, unique=True)
        strict_null, _ = cache.hit(nulls, remaining_peaks, tolerance, reconnect=True, unique=True)
        records.append(
            {
                "environment": environment,
                "pool_count": len(pool),
                "bridges": len(bridges),
                "products": len(products),
                "null_count": len(nulls),
                "null_missing": missing,
                "pool_covered": float(weights[ordinary].sum()),
                "template_added": float(weights[template_hit & ~ordinary].sum()),
                "whole_added": float(weights[whole_hit & ~baseline].sum()),
                "continuation_added": float(weights[raw & ~continuation_baseline].sum()),
                "unique": float(remaining_weights[unique].sum()),
                "strict_unique": float(remaining_weights[strict].sum()),
                "null": float(remaining_weights[null].sum()) if not missing else np.nan,
                "strict_null": float(remaining_weights[strict_null].sum())
                if not missing
                else np.nan,
                "capped": capped,
                "null_capped": null_capped,
            }
        )
    return records


def summarize(frame: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Bootstrap paired differences over molecule means, keeping incomplete cases visible."""
    rows = []
    rng = np.random.default_rng(seed)
    for environment, group in frame.groupby("environment"):
        means = group.groupby("mol_id")[["unique", "null", "strict_unique", "strict_null"]].mean()
        paired = means.dropna()
        row = {
            "environment": environment,
            "molecules": len(means),
            "spectra": len(group),
            "products_mean": group.products.mean(),
            "products_max": group.products.max(),
            "capped": int(group.capped.sum()),
            "null_capped": int(group.null_capped.sum()),
            "null_missing": int(group.null_missing.sum()),
        }
        for column in (
            "pool_covered",
            "template_added",
            "whole_added",
            "continuation_added",
            "unique",
            "null",
            "strict_unique",
            "strict_null",
        ):
            row[column + "_pct"] = 100 * group.groupby("mol_id")[column].mean().mean()
        for prefix, real, null in (
            ("", "unique", "null"),
            ("strict_", "strict_unique", "strict_null"),
        ):
            diff = (paired[real] - paired[null]).to_numpy() * 100
            row[prefix + "diff_pct"] = diff.mean() if diff.size else np.nan
            if diff.size >= 3:
                draws = rng.choice(diff, (2000, diff.size), replace=True).mean(axis=1)
                row[prefix + "lo"], row[prefix + "hi"] = np.quantile(draws, [0.025, 0.975])
            else:
                row[prefix + "lo"] = row[prefix + "hi"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _alias_ft(value):
    """Rename the historical single label without changing the learned embedding width."""
    if isinstance(value, dict):
        return {("OB" if k == "FT" else k): _alias_ft(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_alias_ft(v) for v in value]
    return "OB" if isinstance(value, str) and value == "FT" else value


def _identity(path: str) -> dict:
    item = Path(path)
    stat = item.stat()
    return {"path": str(item.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def main(argv: list[str] | None = None) -> None:
    """Run a reproducible bounded c0 coverage gate, with optional frozen val families."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="saved_ckpts/c0_control_vcpho5qm.ckpt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--families", help="train-selected JSON; required for val")
    parser.add_argument("--molecules", type=int, default=256)
    parser.add_argument("--spectra-per-molecule", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reach-cap", type=int, default=200000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if min(args.molecules, args.spectra_per_molecule, args.batch_size, args.reach_cap) < 1:
        parser.error("sample sizes and reach cap must be positive")
    if args.split == "val" and not args.families:
        parser.error("validation requires a frozen train-selected --families file")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    th.set_num_threads(2)
    th.manual_seed(args.seed)
    np.random.seed(args.seed)  # noqa: NPY002 - seed legacy dependencies as well
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # 1. Restore checkpoint semantics, aliasing only its obsolete instrument label.
    saved = get_config_from_ckpt(args.checkpoint)
    if saved["spec_params"]["inst_types"] != ["FT"] or saved["spec_params"]["prec_types"] != [
        "[M+H]+"
    ]:
        raise ValueError("this diagnostic is scoped to c0's historical [M+H]+/FT checkpoint")
    config = load_config(
        "config", "config/masster/mh_ft_memorization_route_2026_09_24/c0_control.yml", []
    )
    config.update(_alias_ft(saved))
    config = apply_inference_config(
        config,
        disable_preproc=True,
        num_workers=0,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
    )
    for key, expected in (
        ("masster_smarts_templates", False),
        ("masster_include_cmf", False),
        ("masster_max_h_transfer", 4),
        ("masster_h_transfer_bound", "valence"),
        ("masster_unique_fragment_mode", True),
    ):
        if config[key] != expected:
            raise ValueError(f"expected {key}={expected}")
    device = th.device(args.device)
    if device.type == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu explicitly")
    if device.type == "cpu":
        config["masster_candidate_backend"] = "cpu"
    model = init_model_from_ckpt(args.checkpoint, config, device)
    dataset_config = copy.deepcopy(config)
    dataset_config["spec_params"]["inst_types"] = ["OB", "OB_F"]
    dataset = get_dataset_cls(config["model_type"])(split=args.split, **dataset_config)
    specs = dataset.spec_df.reset_index(drop=True)
    molecules = pd.read_pickle(config["mol_fp"]).set_index("mol_id")["mol"]
    allowed = None if not args.families else set(json.loads(Path(args.families).read_text()))

    # 2. Random molecule sampling precedes target inspection; retain zero-bridge molecules.
    ids = np.sort(specs.mol_id.unique())
    selected = rng.choice(ids, min(args.molecules, len(ids)), replace=False)
    keep, bridges_by_mol, templates_by_mol = [], {}, {}
    for mol_id in selected:
        positions = np.flatnonzero(specs.mol_id.to_numpy() == mol_id)
        keep.extend(
            rng.choice(positions, min(args.spectra_per_molecule, len(positions)), replace=False)
        )
        mol = molecules.loc[mol_id]
        bridges = enumerate_bridges(mol)
        bridges_by_mol[mol_id] = bridges
        templates = filter_masster_templates(
            build_masster_templates(mol), precursor_type="[M+H]+", fragmentation_mode="HCD"
        )
        edges = edge_tuple(mol)
        templates_by_mol[mol_id] = {
            tuple(s) for t in templates for s in t.components if not _is_connected(s, edges)
        }
    keep = sorted(keep)
    specs.iloc[keep][["spec_id", "mol_id", "inst_type", "prec_type"]].to_csv(
        output.with_suffix(".sample.csv"), index=False
    )
    provenance = {
        "args": vars(args),
        "started_unix": started,
        "torch": th.__version__,
        "device": str(device),
        "gpu": th.cuda.get_device_name(device) if device.type == "cuda" else None,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"])
        ).hexdigest(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).with_name("debug_masster_smarts_rule_gate.py"))
        },
        "inputs": [_identity(p) for p in [args.checkpoint, config["spec_fp"], config["mol_fp"]]],
        "split_files": [
            _identity(str(p)) for p in sorted(Path(config["split_dp"]).rglob("*")) if p.is_file()
        ],
        "config": config,
        "label_alias": "FT -> OB; both observed OB/OB_F input indices set to zero",
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    LOGGER.info(
        "Selected %d molecules, %d spectra; %d molecules have eligible bridges",
        len(selected),
        len(keep),
        sum(bool(v) for v in bridges_by_mol.values()),
    )
    loader = DataLoader(
        Subset(dataset, keep),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.get_collate_fn(),
        num_workers=0,
    )
    scorer = ChannelScorer()
    caches, records = {}, []
    cursor = 0

    # 3. Read final retained memberships and chemical channels from annotation outputs.
    with th.inference_mode():
        for batch in loader:
            rows = specs.iloc[keep[cursor : cursor + int(batch["batch_size"])]]
            batch = model.transfer_batch_to_device(batch, device, dataloader_idx=0)
            batch["spec_inst_type"].zero_()
            out = model(**batch, return_explainability_tensors=True)
            atom_ptr = batch["mol_pyg"].ptr.cpu().numpy()
            atoms = out["fragment_atom_indices"].cpu().numpy()
            memberships = out["fragment_atom_fragment_idxs"].cpu().numpy()
            views = out["ion_h_fragment_batch_idxs"].cpu().numpy()
            masses = out["ion_h_dense_mzs"].double().cpu().numpy()
            valid = out["ion_h_chemical_valid_channels"].cpu().numpy()
            pools = defaultdict(set)
            order = np.argsort(memberships, kind="stable")
            fragment_ids, starts = np.unique(memberships[order], return_index=True)
            for fragment, members in zip(
                fragment_ids, np.split(atoms[order], starts[1:]), strict=True
            ):
                view = views[fragment]
                pools[view].add(tuple(sorted((members - atom_ptr[view]).tolist())))
            for view, row in enumerate(rows.itertuples()):
                mol = molecules.loc[row.mol_id]
                if row.mol_id not in caches:
                    caches[row.mol_id] = MoleculeCoverage(mol, scorer, args.reach_cap)
                # Raw stored intensities: deliberately no new target filtering or sqrt scaling.
                peaks = np.asarray(row.peaks, float).reshape(-1, 2)
                weights = peaks[:, 1] / peaks[:, 1].sum()
                result = evaluate_spectrum(
                    mol,
                    pools[view],
                    masses[views == view][valid[views == view]],
                    bridges_by_mol[row.mol_id],
                    templates_by_mol[row.mol_id],
                    peaks[:, 0],
                    weights,
                    TOLERANCE[row.inst_type],
                    caches[row.mol_id],
                    rng,
                    families=allowed,
                )
                for record in result:
                    record.update(
                        spec_id=row.spec_id,
                        mol_id=row.mol_id,
                        split=args.split,
                        inst_type=row.inst_type,
                    )
                records.extend(result)
            cursor += len(rows)
            if cursor % (args.batch_size * 8) == 0 or cursor == len(keep):
                pd.DataFrame(records).to_pickle(output)
                LOGGER.info(
                    "Scored %d/%d spectra in %.1fs", cursor, len(keep), time.time() - started
                )

    # 4. Freeze exploratory train selections before any validation targets are inspected.
    frame = pd.DataFrame(records)
    frame.to_pickle(output)
    summary = summarize(frame, args.seed)
    summary.to_csv(output.with_suffix(".summary.csv"), index=False)
    if args.split == "train":
        passing = summary[
            (summary.environment != "__all__")
            & (summary.molecules >= 10)
            & (summary.strict_lo > 0)
            & (summary.capped == 0)
            & (summary.null_capped == 0)
            & (summary.null_missing == 0)
        ]
        output.with_suffix(".families.json").write_text(
            json.dumps(passing.environment.tolist(), indent=2)
        )
    provenance["elapsed_seconds"] = time.time() - started
    provenance["status"] = "complete"
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    LOGGER.info(
        "Summary:\n%s",
        summary.sort_values("strict_diff_pct", ascending=False).head(15).to_string(index=False),
    )


if __name__ == "__main__":
    main()
