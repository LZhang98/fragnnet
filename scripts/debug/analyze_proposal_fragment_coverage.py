"""Compare `connected_fragment_beam` and `cut_masks` candidate selection on the same logits.

Reproduces the measurements recorded in
`docs/findings/connected_fragment_beam_reuse_and_equivalence_2026_09_17.md`. Both selectors
receive identical bond logits and a matched *unique-fragment* budget, so only the selection
objective differs. Three scans are available:

- ``distributions``: emitted fragment size, ring and complement-asymmetry statistics across a
  sweep of bond-break probabilities.
- ``controls``: whether the size bias is explained by ``max_boundary_cuts`` or by the candidate
  budget rather than by the score.
- ``coverage``: captured mask-probability mass and top-K recall per budget against a reference
  universe, plus a decomposition of unreached mass by fragment size.
- ``implementations``: exact emitted-set comparison against an alternative beam module supplied
  with ``--beam-implementation``, for example a historical revision extracted with
  ``git show <rev>:src/fragnnet/frag/mass_shatter_fast_gnn.py``.

Bond logits are synthetic by default, so the sweep tests whether *any* learned logits could
remove the effect. Pass ``--checkpoint`` to capture **trained** bond logits from a MASShatter
checkpoint instead; only the encoder and bond-break head are required, so a checkpoint whose
intensity head has since changed shape still works. This is candidate construction only: no
optimizer step is taken and no spectrum is predicted.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import random
import sys
import time
import types
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch as th
import torch.nn.functional as F

from fragnnet.frag.mass_shatter_candidates import (
    MassShatterCandidateBatch,
    build_mass_shatter_candidates,
)
from fragnnet.frag.mass_shatter_fast_gnn import build_fast_gnn_candidates
from fragnnet.model.mass_shatter_model import MassShatterModel
from fragnnet.pl_model import MassShatterPL
from fragnnet.runner import init_dataloader, init_dataset, load_config

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG = (
    "config/massshatter/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_cmf_dual_ce_fast_gnn_proposal.yml"
)
# Production ratio: mass_shatter_ranked_mask_budget / mass_shatter_unique_fragment_target.
ENUMERATION_SLACK = 16
# (mu, sigma) for the synthetic bond-logit sweep, from near-stable to near-even cutting.
LOGIT_SCENARIOS: tuple[tuple[str, float, float], ...] = (
    ("p_vlow", -3.0, 1.0),
    ("p_low", -2.0, 1.0),
    ("p_mid", -1.0, 1.5),
    ("p_high", 0.0, 2.0),
)


class MolecularBatch:
    """One captured batch of precursor graphs plus derived per-bond and per-molecule tensors."""

    def __init__(self, mol_pyg: Any) -> None:
        self.chemical_bond_index: th.Tensor = mol_pyg.chemical_bond_index
        self.atom_batch_indices: th.Tensor = mol_pyg.batch
        self.atom_ptr: th.Tensor = mol_pyg.ptr
        self.num_molecules = int(self.atom_ptr.numel() - 1)
        self.num_atoms = int(self.atom_batch_indices.numel())
        self.num_bonds = int(self.chemical_bond_index.shape[1])
        if self.num_bonds == 0:
            raise ValueError("captured batch contains no chemical bonds")
        self.molecule_sizes: th.Tensor = self.atom_ptr[1:] - self.atom_ptr[:-1]
        self.bond_molecule: th.Tensor = self.atom_batch_indices[self.chemical_bond_index[0]]

    def intact_baseline(self, bond_logits: th.Tensor) -> th.Tensor:
        """Return per-molecule ``sum_all log(1 - p_e)``, shape ``(num_molecules,)``."""
        return th.zeros(self.num_molecules).index_add(
            0, self.bond_molecule, F.logsigmoid(-bond_logits)
        )


def _deep_fill(existing: dict, defaults: dict) -> dict:
    """Fill keys absent from a saved config with template defaults, recursing into nested dicts."""
    merged = dict(existing)
    for key, value in defaults.items():
        if key not in merged:
            merged[key] = value
        elif isinstance(value, dict) and isinstance(merged[key], dict):
            merged[key] = _deep_fill(merged[key], value)
    return merged


def capture_trained_logits(
    checkpoint_path: Path, batch_size: int, subsample_size: int
) -> tuple[MolecularBatch, th.Tensor]:
    """Return one batch and the trained bond logits its proposal path produces.

    The checkpoint's saved hyperparameters are filled from `config/models_template` where keys are
    absent, then loaded non-strictly: only encoder, bond-break-head and charge-head tensors must
    match, so a checkpoint predating the current intensity head remains usable. A forward hook
    captures the bond logits, and the rest of the forward is allowed to fail because the
    intensity head may be randomly initialized and is never used here.

    Raises:
        ValueError: The proposal path did not load completely, or no logits were captured.
    """
    checkpoint = th.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = _deep_fill(dict(checkpoint["hyper_parameters"]), load_config("config/models_template", None, None))
    config.update(num_workers=0, share_memory=False, pl_enable_progress_bar=False)
    config["train_batch_size"] = batch_size
    config["subsample_params"]["subsample_size"] = subsample_size

    model = MassShatterPL(**config)
    result = model.load_state_dict(checkpoint["state_dict"], strict=False)
    unloaded = [key for key in result.missing_keys if "intensity_head" not in key]
    if unloaded:
        raise ValueError(f"proposal path did not load from the checkpoint: {unloaded[:5]}")
    model.eval()
    LOGGER.info(
        "loaded proposal path from %s (epoch %s); %d intensity-head tensors left random",
        checkpoint_path,
        checkpoint.get("epoch"),
        len(result.missing_keys),
    )

    L.seed_everything(config["seed"], workers=True)
    (dataset,) = init_dataset(config, splits=("train",))
    raw_batch = next(iter(init_dataloader(dataset, config)))
    captured: dict[str, th.Tensor] = {}
    handle = model.model.bond_break_head.register_forward_hook(
        lambda _module, _inputs, output: captured.setdefault("logits", output.detach())
    )
    try:
        with th.no_grad():
            model._common_step(raw_batch, "train", log=False)
    except Exception as error:  # noqa: BLE001 - the hook has already fired
        LOGGER.info("forward stopped after the hook (%s); bond logits were captured",
                    type(error).__name__)
    finally:
        handle.remove()
    if "logits" not in captured:
        raise ValueError("the bond-break head never ran; no trained logits were captured")
    batch = MolecularBatch(raw_batch["mol_pyg"])
    logits = captured["logits"].float().reshape(-1)
    if logits.numel() != batch.num_bonds:
        raise ValueError(f"captured {logits.numel()} logits for {batch.num_bonds} bonds")
    probability = th.sigmoid(logits)
    LOGGER.info(
        "trained p(cut): mean %.4f | p10 %.4f p25 %.4f p50 %.4f p75 %.4f p90 %.4f",
        float(probability.mean()),
        *np.percentile(probability.numpy(), [10, 25, 50, 75, 90]),
    )
    return batch, logits


def load_batch(config_path: str, batch_size: int, subsample_size: int) -> MolecularBatch:
    """Capture the first training batch for the given experiment configuration."""
    config = load_config("config/models_template", config_path, None)
    config.update(num_workers=0, share_memory=False, pl_enable_progress_bar=False)
    config["train_batch_size"] = batch_size
    config["subsample_params"]["subsample_size"] = subsample_size
    L.seed_everything(config["seed"], workers=True)
    (dataset,) = init_dataset(config, splits=("train",))
    batch = next(iter(init_dataloader(dataset, config)))
    return MolecularBatch(batch["mol_pyg"])


def synthetic_bond_logits(batch: MolecularBatch, mu: float, sigma: float, seed: int) -> th.Tensor:
    """Draw one normal bond logit per chemical bond, shape ``(num_bonds,)``."""
    generator = th.Generator().manual_seed(seed)
    return (th.randn(batch.num_bonds, generator=generator) * sigma + mu).float()


def select_beam(batch: MolecularBatch, logits: th.Tensor, budget: int, cuts: int):
    """Run the connected-fragment beam at a matched unique-fragment budget."""
    return build_fast_gnn_candidates(
        logits,
        batch.chemical_bond_index,
        batch.atom_batch_indices,
        batch.atom_ptr,
        max_fragments=budget,
        beam_width=budget,
        max_boundary_cuts=cuts,
    )


def select_cut_masks(batch: MolecularBatch, logits: th.Tensor, budget: int):
    """Run `cut_masks` at the same unique-fragment budget with the production enumeration slack."""
    return build_mass_shatter_candidates(
        logits,
        batch.chemical_bond_index,
        batch.atom_batch_indices,
        batch.atom_ptr,
        max_masks=budget,
        include_single_bond_cuts=True,
        max_unique_components=budget,
        max_enumerated_masks=ENUMERATION_SLACK * budget,
        candidate_backend="cpu",
    )


def fragment_table(
    candidates: MassShatterCandidateBatch,
    batch: MolecularBatch,
    intact_baseline: th.Tensor,
    logits: th.Tensor,
) -> dict[str, Any]:
    """Describe every emitted unique fragment.

    Args:
        candidates: Candidate adapter from either selector.
        batch: The captured precursor graphs.
        intact_baseline: Per-molecule ``sum_all log(1 - p_e)``, shape ``(num_molecules,)``.
        logits: Per-bond cut logits, shape ``(num_bonds,)``.

    Returns:
        Mapping with per-fragment ``size``, ``cycle_rank``, ``size_fraction``, ``molecule``,
        ``mask_log_probability`` (``log P(m = boundary(f))``) and canonical ``keys`` usable to
        compare fragments across selectors.
    """
    (
        fragment_batch,
        fragment_atom_indices,
        atom_fragment_indices,
        _boundary_bond_indices,
        boundary_fragment_indices,
        _cut_counts,
    ) = MassShatterModel._unique_component_memberships(candidates)
    num_fragments = candidates.num_unique_components
    membership = th.zeros(num_fragments, batch.num_atoms, dtype=th.bool)
    membership[atom_fragment_indices, fragment_atom_indices] = True
    return table_from_membership(
        membership,
        fragment_batch,
        batch,
        intact_baseline,
        logits,
        th.bincount(boundary_fragment_indices, minlength=num_fragments),
    )


def table_from_membership(
    membership: th.Tensor,
    fragment_batch: th.Tensor,
    batch: MolecularBatch,
    intact_baseline: th.Tensor,
    logits: th.Tensor,
    boundary_counts: th.Tensor | None = None,
) -> dict[str, Any]:
    """Derive per-fragment statistics from a boolean atom-membership matrix.

    Args:
        membership: Boolean matrix of shape ``(num_fragments, num_atoms)``.
        fragment_batch: Molecule index per fragment, shape ``(num_fragments,)``.
        batch: The captured precursor graphs.
        intact_baseline: Per-molecule ``sum_all log(1 - p_e)``, shape ``(num_molecules,)``.
        logits: Per-bond cut logits, shape ``(num_bonds,)``.
        boundary_counts: Optional precomputed boundary size per fragment.
    """
    num_fragments = int(membership.shape[0])
    left, right = batch.chemical_bond_index
    is_boundary = membership[:, left] ^ membership[:, right]
    is_internal = membership[:, left] & membership[:, right]
    size = membership.sum(dim=1)
    # Connected fragment: cycle rank = internal bonds - atoms + 1, zero for a tree.
    cycle_rank = is_internal.sum(dim=1) - size + 1
    # log P(m = boundary(f)) = sum_all log(1-p) + sum_boundary logit(p).
    mask_log_probability = intact_baseline[fragment_batch] + is_boundary.float() @ logits
    return {
        "size": size.numpy(),
        "cycle_rank": cycle_rank.numpy(),
        "boundary": (
            is_boundary.sum(dim=1) if boundary_counts is None else boundary_counts
        ).numpy(),
        "size_fraction": (
            size.double() / batch.molecule_sizes[fragment_batch].double()
        ).numpy(),
        "molecule": fragment_batch.numpy(),
        "mask_log_probability": mask_log_probability.numpy(),
        "keys": [
            (int(molecule), membership[index].nonzero().flatten().numpy().tobytes())
            for index, molecule in enumerate(fragment_batch)
        ],
        "count": int(num_fragments),
    }


def _per_molecule_mass(
    scores: np.ndarray, molecules: np.ndarray, denominators: dict[int, float], num_molecules: int
) -> float:
    """Mean over molecules of the probability mass captured, relative to a reference total."""
    captured = []
    for molecule in range(num_molecules):
        selected = scores[molecules == molecule]
        if selected.size == 0:
            captured.append(0.0)
            continue
        total = float(th.logsumexp(th.from_numpy(selected), 0))
        captured.append(float(np.exp(total - denominators[molecule])))
    return float(np.mean(captured))


def run_distributions(
    batch: MolecularBatch,
    budget: int,
    cuts: int,
    seed: int,
    trained_logits: th.Tensor | None = None,
) -> list[dict]:
    """Compare emitted size, ring and complement-asymmetry statistics across the logit sweep.

    With ``trained_logits`` the sweep collapses to that single real operating point.
    """
    scenarios = (
        (("trained", 0.0, 0.0),) if trained_logits is not None else LOGIT_SCENARIOS
    )
    rows: list[dict] = []
    for label, mu, sigma in scenarios:
        logits = (
            trained_logits
            if trained_logits is not None
            else synthetic_bond_logits(batch, mu, sigma, seed)
        )
        baseline = batch.intact_baseline(logits)
        probability = float(th.sigmoid(logits).mean())
        for name, candidates in (
            ("connected_fragment_beam", select_beam(batch, logits, budget, cuts)),
            ("cut_masks", select_cut_masks(batch, logits, budget)),
        ):
            table = fragment_table(candidates, batch, baseline, logits)
            size, fraction, cycle = table["size"], table["size_fraction"], table["cycle_rank"]
            row = {
                "scenario": label,
                "mean_cut_probability": probability,
                "selector": name,
                "fragments": table["count"],
                "size_mean": float(size.mean()),
                "size_median": float(np.median(size)),
                "singleton_fraction": float((size == 1).mean()),
                "acyclic_fraction": float((cycle <= 0).mean()),
                "mean_size_fraction": float(fraction.mean()),
                "above_half_fraction": float((fraction > 0.5).mean()),
                "at_least_three_quarters_fraction": float((fraction >= 0.75).mean()),
            }
            rows.append(row)
            LOGGER.info(
                "%-8s p=%.3f %-24s n=%6d size %6.2f | acyclic %.3f | >half %.3f | >=.75 %.3f",
                label,
                probability,
                name,
                row["fragments"],
                row["size_mean"],
                row["acyclic_fraction"],
                row["above_half_fraction"],
                row["at_least_three_quarters_fraction"],
            )
    return rows


def run_controls(
    batch: MolecularBatch,
    cut_limits: tuple[int, ...],
    budgets: tuple[int, ...],
    seed: int,
    trained_logits: th.Tensor | None = None,
) -> list[dict]:
    """Test whether the boundary-cut cap or the candidate budget explains the size bias."""
    logits = (
        trained_logits if trained_logits is not None
        else synthetic_bond_logits(batch, -2.0, 1.0, seed)
    )
    baseline = batch.intact_baseline(logits)
    rows: list[dict] = []
    for cuts in cut_limits:
        table = fragment_table(select_beam(batch, logits, 256, cuts), batch, baseline, logits)
        fraction = table["size_fraction"]
        rows.append(
            {
                "control": "max_boundary_cuts",
                "value": cuts,
                "fragments": table["count"],
                "size_mean": float(table["size"].mean()),
                "above_half_fraction": float((fraction > 0.5).mean()),
            }
        )
        LOGGER.info(
            "max_boundary_cuts=%-3d n=%6d size %5.2f | >half %.3f",
            cuts,
            table["count"],
            rows[-1]["size_mean"],
            rows[-1]["above_half_fraction"],
        )
    for budget in budgets:
        table = fragment_table(select_beam(batch, logits, budget, 8), batch, baseline, logits)
        fraction = table["size_fraction"]
        rows.append(
            {
                "control": "budget",
                "value": budget,
                "fragments": table["count"],
                "size_mean": float(table["size"].mean()),
                "above_half_fraction": float((fraction > 0.5).mean()),
            }
        )
        LOGGER.info(
            "target=%-5d      n=%6d size %5.2f | >half %.3f",
            budget,
            table["count"],
            rows[-1]["size_mean"],
            rows[-1]["above_half_fraction"],
        )
    return rows


def run_coverage(
    batch: MolecularBatch,
    budgets: tuple[int, ...],
    reference_budget: int,
    cuts: int,
    seed: int,
    trained_logits: th.Tensor | None = None,
) -> dict[str, Any]:
    """Measure captured mask-probability mass per budget and decompose what stays unreached."""
    logits = (
        trained_logits if trained_logits is not None
        else synthetic_bond_logits(batch, -2.0, 1.0, seed)
    )
    baseline = batch.intact_baseline(logits)

    reference: dict[tuple[int, bytes], tuple[float, int, float]] = {}
    beam_keys: set[tuple[int, bytes]] = set()
    for name, candidates in (
        ("beam", select_beam(batch, logits, reference_budget, cuts)),
        ("cut_masks", select_cut_masks(batch, logits, reference_budget // 4)),
    ):
        table = fragment_table(candidates, batch, baseline, logits)
        for key, score, molecule, fraction in zip(
            table["keys"],
            table["mask_log_probability"],
            table["molecule"],
            table["size_fraction"],
            strict=True,
        ):
            reference[key] = (float(score), int(molecule), float(fraction))
        if name == "beam":
            beam_keys = set(table["keys"])
        LOGGER.info("reference universe after %s: %d fragments", name, len(reference))

    keys = list(reference)
    scores = np.array([reference[key][0] for key in keys])
    molecules = np.array([reference[key][1] for key in keys])
    fractions = np.array([reference[key][2] for key in keys])
    denominators = {
        molecule: float(th.logsumexp(th.from_numpy(scores[molecules == molecule]), 0))
        for molecule in range(batch.num_molecules)
    }
    ranks: dict[tuple[int, bytes], int] = {}
    position, current = 0, None
    for index in np.lexsort((-scores, molecules)):
        position = 0 if molecules[index] != current else position + 1
        current = molecules[index]
        ranks[keys[index]] = position

    rows: list[dict] = []
    for budget in budgets:
        for name in ("connected_fragment_beam", "cut_masks"):
            start = time.perf_counter()
            candidates = (
                select_beam(batch, logits, budget, cuts)
                if name == "connected_fragment_beam"
                else select_cut_masks(batch, logits, budget)
            )
            elapsed_ms = 1000.0 * (time.perf_counter() - start)
            table = fragment_table(candidates, batch, baseline, logits)
            known = [key in reference for key in table["keys"]]
            mass = _per_molecule_mass(
                table["mask_log_probability"][known],
                table["molecule"][known],
                denominators,
                batch.num_molecules,
            )
            hits = sum(ranks[key] < budget for key, seen in zip(table["keys"], known) if seen)
            wanted = sum(
                min(budget, int((molecules == molecule).sum()))
                for molecule in range(batch.num_molecules)
            )
            rows.append(
                {
                    "selector": name,
                    "budget": budget,
                    "fragments": table["count"],
                    "captured_mass": mass,
                    "recall_at_budget": hits / wanted,
                    "elapsed_ms": elapsed_ms,
                    "ms_per_fragment": elapsed_ms / max(table["count"], 1),
                }
            )
            LOGGER.info(
                "K=%-5d %-24s n=%6d mass %.4f recall %.3f %8.1f ms (%.4f ms/frag)",
                budget,
                name,
                table["count"],
                mass,
                rows[-1]["recall_at_budget"],
                elapsed_ms,
                rows[-1]["ms_per_fragment"],
            )

    bands: list[dict] = []
    for low, high in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
        inside = (fractions >= low) & (fractions < high)
        reached = np.array([key in beam_keys for key in keys]) & inside
        band = {
            "low": low,
            "high": high,
            "reference_fragments": int(inside.sum()),
            "share_of_reference_mass": _per_molecule_mass(
                scores[inside], molecules[inside], denominators, batch.num_molecules
            ),
            "beam_reach_by_count": float(reached.sum() / max(int(inside.sum()), 1)),
            "beam_reached_mass": _per_molecule_mass(
                scores[reached], molecules[reached], denominators, batch.num_molecules
            ),
        }
        bands.append(band)
        LOGGER.info(
            "band [%.2f,%.2f) n=%6d mass share %.4f | beam reaches %.3f | beam mass %.4f",
            low,
            high,
            band["reference_fragments"],
            band["share_of_reference_mass"],
            band["beam_reach_by_count"],
            band["beam_reached_mass"],
        )
    return {
        "reference_universe": len(reference),
        "beam_reference_pool": len(beam_keys),
        "budgets": rows,
        "size_bands": bands,
    }


class _Absorbing:
    """Stands in for an unavailable import: absorbs attribute access, indexing and calls."""

    def __getattr__(self, _name: str) -> _Absorbing:
        return self

    def __getitem__(self, _key: object) -> _Absorbing:
        return self

    def __call__(self, *_args: object, **_kwargs: object) -> _Absorbing:
        return self

    def __float__(self) -> float:
        return 0.0


def load_beam_implementation(module_path: Path, max_stubs: int = 8):
    """Import an alternative beam module and return it.

    Accepts either a revision of `mass_shatter_fast_gnn` exposing ``build_fast_gnn_candidates``,
    or the standalone research prototype exposing ``enumerate_node_combos``. Research prototypes
    often import packages that are not installed here while their search function is pure tensor
    code, so absent top-level modules are stubbed and the import retried.

    Args:
        module_path: A Python file exposing one of the two entry points.
        max_stubs: Maximum number of absent modules to stub before giving up.

    Raises:
        ValueError: The file cannot be imported or exposes neither entry point.
    """
    stubbed: list[str] = []
    for _ in range(max_stubs + 1):
        spec = importlib.util.spec_from_file_location("alternative_beam", module_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot import a beam implementation from {module_path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as error:
            name = error.name
            if name is None or name in stubbed:
                raise ValueError(f"cannot satisfy imports of {module_path}: {error}") from error
            # A permissive stub. Prototypes often index or call these names at import time,
            # so every attribute resolves to an object that absorbs attribute access,
            # subscripting and calls. Nothing this analysis runs depends on their values.
            stub = types.ModuleType(name)
            stub.__getattr__ = lambda _attribute: _Absorbing()  # type: ignore[attr-defined]
            stub.__path__ = []  # type: ignore[attr-defined]
            for suffix in ("", ".gen_model", ".gen_model.transformer_base", ".utils",
                           ".utils.chem_utils"):
                sys.modules.setdefault(f"{name}{suffix}", stub)
            stubbed.append(name)
            continue
        if stubbed:
            LOGGER.info("stubbed absent imports to load the prototype: %s", ", ".join(stubbed))
        if not hasattr(module, "build_fast_gnn_candidates") and not hasattr(
            module, "enumerate_node_combos"
        ):
            raise ValueError(
                f"{module_path} exposes neither build_fast_gnn_candidates nor "
                "enumerate_node_combos"
            )
        return module
    raise ValueError(f"cannot import {module_path} after stubbing {max_stubs} modules")


def prototype_fragment_table(
    module: Any,
    batch: MolecularBatch,
    logits: th.Tensor,
    intact_baseline: th.Tensor,
    budget: int,
    cuts: int,
) -> dict[str, Any]:
    """Run the standalone prototype's search and describe its emitted fragments.

    The prototype returns pooled atom embeddings rather than memberships, so it is called with
    an identity ``atom_embed``; the pooled result is then exactly the atom-membership matrix.
    Unlike the installed beam it excludes the intact precursor, and it ranks by the mask log
    probability rather than by boundary-plus-internal score.
    """
    left, right = batch.chemical_bond_index
    bond_batch = batch.atom_batch_indices[left]
    order = bond_batch.argsort(stable=True)
    max_atoms = int(batch.molecule_sizes.max())
    _mz, fragment_batch, pooled, _score, _baseline, *_rest = module.enumerate_node_combos(
        logits[order],
        left[order],
        right[order],
        bond_batch[order],
        batch.atom_batch_indices,
        th.ones(batch.num_atoms),
        th.zeros(batch.num_atoms),
        th.eye(batch.num_atoms),
        batch.num_molecules,
        budget,
        (max_atoms + 1) // 2,
        cuts,
        budget,
    )
    return table_from_membership(
        pooled > 0.5, fragment_batch, batch, intact_baseline, logits
    )


def run_implementations(
    batch: MolecularBatch,
    alternative_module: Any,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    trained_logits: th.Tensor | None = None,
) -> list[dict]:
    """Compare emitted fragments between the installed beam and an alternative implementation.

    Two kinds of alternative are supported. A module exposing ``build_fast_gnn_candidates`` is a
    revision of the installed search, so the comparison is an exact set difference. The research
    prototype's ``enumerate_node_combos`` instead ranks by the mask log probability, so it is a
    different objective and the interesting quantity is how much probability mass each reaches.

    A fixed batch is used, so this does **not** measure invariance to batch composition, which is
    the property the stable-tie ordering exists to provide.
    """
    is_prototype = not hasattr(alternative_module, "build_fast_gnn_candidates")
    scenarios = (
        ((0, 0.0),) if trained_logits is not None else [(s, mu) for s in seeds for mu in (-2.0, -1.0)]
    )
    rows: list[dict] = []
    for seed, mu in scenarios:
        logits = (
            trained_logits if trained_logits is not None
            else synthetic_bond_logits(batch, mu, 1.0, seed)
        )
        baseline = batch.intact_baseline(logits)
        for budget in budgets:
            for cuts in (5, 8):
                installed_table = fragment_table(
                    select_beam(batch, logits, budget, cuts), batch, baseline, logits
                )
                other_table = (
                    prototype_fragment_table(
                        alternative_module, batch, logits, baseline, budget, cuts
                    )
                    if is_prototype
                    else fragment_table(
                        alternative_module.build_fast_gnn_candidates(
                            logits,
                            batch.chemical_bond_index,
                            batch.atom_batch_indices,
                            batch.atom_ptr,
                            max_fragments=budget,
                            beam_width=budget,
                            max_boundary_cuts=cuts,
                        ),
                        batch,
                        baseline,
                        logits,
                    )
                )
                # The prototype drops the intact precursor; compare proper fragments only.
                def proper(table: dict[str, Any]) -> set:
                    keep = table["size_fraction"] < 1.0
                    return {key for key, ok in zip(table["keys"], keep, strict=True) if ok}

                installed = proper(installed_table) if is_prototype else set(
                    installed_table["keys"]
                )
                other = proper(other_table) if is_prototype else set(other_table["keys"])
                row = {
                    "logit_seed": seed,
                    "logit_mu": mu,
                    "logit_source": "checkpoint" if trained_logits is not None else "synthetic",
                    "alternative": "prototype" if is_prototype else "beam_revision",
                    "budget": budget,
                    "max_boundary_cuts": cuts,
                    "installed_fragments": len(installed),
                    "alternative_fragments": len(other),
                    "identical": installed == other,
                    "only_installed": len(installed - other),
                    "only_alternative": len(other - installed),
                    "jaccard": (
                        len(installed & other) / max(len(installed | other), 1)
                    ),
                }
                if is_prototype:
                    row["installed_mean_size"] = float(installed_table["size"].mean())
                    row["alternative_mean_size"] = float(other_table["size"].mean())
                    row["installed_above_half"] = float(
                        (installed_table["size_fraction"] > 0.5).mean()
                    )
                    row["alternative_above_half"] = float(
                        (other_table["size_fraction"] > 0.5).mean()
                    )
                rows.append(row)
                if is_prototype:
                    LOGGER.info(
                        "K=%-5d cuts=%-2d installed n=%5d size %5.2f >half %.3f | "
                        "prototype n=%5d size %5.2f >half %.3f | jaccard %.3f",
                        budget, cuts, len(installed), row["installed_mean_size"],
                        row["installed_above_half"], len(other),
                        row["alternative_mean_size"], row["alternative_above_half"],
                        row["jaccard"],
                    )
                else:
                    LOGGER.info(
                        "seed=%-5d mu=%-5.1f K=%-5d cuts=%-2d n=%6d identical=%-5s "
                        "only-installed=%d only-alternative=%d",
                        seed, mu, budget, cuts, len(installed), row["identical"],
                        row["only_installed"], row["only_alternative"],
                    )
    return rows


def main() -> None:
    """Run the requested scans and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--mode",
        choices=("all", "distributions", "controls", "coverage", "implementations"),
        default="all",
    )
    parser.add_argument(
        "--beam-implementation",
        type=Path,
        default=None,
        help="alternative beam module to compare against; required for --mode implementations",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--subsample-size", type=int, default=128)
    parser.add_argument("--budget", type=int, default=256, help="matched unique-fragment budget")
    parser.add_argument("--max-boundary-cuts", type=int, default=5)
    parser.add_argument("--coverage-budgets", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--reference-budget", type=int, default=4096)
    parser.add_argument("--logit-seed", type=int, default=17)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="capture trained bond logits from this MASShatter checkpoint instead of sampling",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_size, args.subsample_size, args.budget, args.reference_budget) <= 0:
        parser.error("batch, subsample, budget and reference budget must be positive")
    if args.max_boundary_cuts < 0 or min(args.coverage_budgets) <= 0:
        parser.error("boundary cuts must be nonnegative and coverage budgets positive")
    if args.mode == "implementations" and args.beam_implementation is None:
        parser.error("--mode implementations requires --beam-implementation")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    random.seed(args.logit_seed)
    th.manual_seed(args.logit_seed)
    th.use_deterministic_algorithms(True, warn_only=True)

    trained_logits: th.Tensor | None = None
    if args.checkpoint is None:
        batch = load_batch(args.config, args.batch_size, args.subsample_size)
    else:
        batch, trained_logits = capture_trained_logits(
            args.checkpoint, args.batch_size, args.subsample_size
        )
    LOGGER.info(
        "batch: %d molecules, atoms %d/%d/%d (min/median/max), %d bonds",
        batch.num_molecules,
        int(batch.molecule_sizes.min()),
        int(batch.molecule_sizes.median()),
        int(batch.molecule_sizes.max()),
        batch.num_bonds,
    )

    report: dict[str, Any] = {
        "config": args.config,
        "batch_size": args.batch_size,
        "subsample_size": args.subsample_size,
        "logit_seed": args.logit_seed,
        "logit_source": "checkpoint" if trained_logits is not None else "synthetic",
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "enumeration_slack": ENUMERATION_SLACK,
        "molecule_atom_counts": batch.molecule_sizes.tolist(),
    }
    if args.mode in ("all", "distributions"):
        LOGGER.info("--- emitted fragment distributions ---")
        report["distributions"] = run_distributions(
            batch, args.budget, args.max_boundary_cuts, args.logit_seed, trained_logits
        )
    if args.mode in ("all", "controls"):
        LOGGER.info("--- controls: boundary-cut cap and candidate budget ---")
        report["controls"] = run_controls(
            batch, (3, 5, 8, 12, 20), (256, 512, 1024), args.logit_seed, trained_logits
        )
    if args.mode in ("all", "coverage"):
        LOGGER.info("--- coverage per unit budget ---")
        report["coverage"] = run_coverage(
            batch,
            tuple(args.coverage_budgets),
            args.reference_budget,
            max(args.max_boundary_cuts, 8),
            args.logit_seed,
            trained_logits,
        )
    if args.mode == "implementations":
        LOGGER.info("--- emitted-set comparison against %s ---", args.beam_implementation)
        report["implementations"] = run_implementations(
            batch,
            load_beam_implementation(args.beam_implementation),
            (args.logit_seed, 99, 2024),
            (64, args.budget),
            trained_logits,
        )
    args.output.write_text(json.dumps(report, indent=2))
    LOGGER.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
