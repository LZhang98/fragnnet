"""Benchmark MASShatter candidate generation on a synthetic molecular batch."""

from __future__ import annotations

import argparse
import logging
import math
import statistics
import time

import torch as th

from fragnnet.frag.mass_shatter_candidates import (
    MassShatterStaticGraph,
    build_mass_shatter_candidates,
)
from fragnnet.frag.mass_shatter_templates import MassShatterTemplate

LOGGER = logging.getLogger(__name__)


def build_chain_batch(
    batch_size: int,
    num_atoms: int,
    device: th.device,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """Build a batch of equal-size chain graphs with deterministic logits."""
    atom_ptr = th.arange(
        0,
        (batch_size + 1) * num_atoms,
        num_atoms,
        dtype=th.long,
    )
    atom_batch_indices = th.arange(batch_size, dtype=th.long).repeat_interleave(num_atoms)
    left_endpoints: list[int] = []
    right_endpoints: list[int] = []
    for molecule_index in range(batch_size):
        atom_start = molecule_index * num_atoms
        left_endpoints.extend(range(atom_start, atom_start + num_atoms - 1))
        right_endpoints.extend(range(atom_start + 1, atom_start + num_atoms))
    chemical_bond_index = th.tensor(
        [left_endpoints, right_endpoints],
        dtype=th.long,
    )
    bond_logits = th.linspace(-2.0, 2.0, num_atoms - 1).repeat(batch_size)
    bond_logits.requires_grad_(True)
    return tuple(
        tensor.to(device)
        for tensor in (
            bond_logits,
            chemical_bond_index,
            atom_batch_indices,
            atom_ptr,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_atoms", type=int, default=20)
    parser.add_argument("--max_masks", type=int, default=1024)
    parser.add_argument("--max_unique_components", type=int)
    parser.add_argument("--max_enumerated_masks", type=int)
    parser.add_argument("--num_mixtures", type=int, default=1)
    parser.add_argument("--mask_selection_threads", type=int, default=1)
    parser.add_argument("--candidate_group_multiplicity", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--static_graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smarts", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.device == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.num_mixtures < 1:
        raise ValueError("num_mixtures must be positive")
    if args.mask_selection_threads < 1:
        raise ValueError("mask_selection_threads must be positive")
    if args.candidate_group_multiplicity < 1:
        raise ValueError("candidate_group_multiplicity must be positive")
    if args.batch_size % args.candidate_group_multiplicity != 0:
        raise ValueError("batch_size must be divisible by candidate_group_multiplicity")
    if args.max_unique_components is not None and args.max_enumerated_masks is None:
        raise ValueError("max_enumerated_masks is required when max_unique_components is enabled")
    device = th.device(args.device)

    # 1. Build one deterministic workload matching the L40S mask budget.
    bond_logits, chemical_bond_index, atom_batch_indices, atom_ptr = build_chain_batch(
        args.batch_size,
        args.num_atoms,
        device,
    )
    mixture_log_weights = None
    if args.num_mixtures > 1:
        molecule_logits = bond_logits.reshape(args.batch_size, args.num_atoms - 1)
        bond_logits = th.stack(
            [molecule_logits.roll(shifts=mixture, dims=1) for mixture in range(args.num_mixtures)],
            dim=2,
        ).reshape(-1, args.num_mixtures)
        bond_logits.requires_grad_(True)
        mixture_log_weights = th.full(
            (args.batch_size, args.num_mixtures),
            -math.log(args.num_mixtures),
            dtype=th.float64,
            device=device,
        )

    candidate_group_ids = None
    if args.candidate_group_multiplicity > 1:
        candidate_group_ids = th.arange(args.batch_size, device=device).div(
            args.candidate_group_multiplicity,
            rounding_mode="floor",
        )
    static_graph = None
    if args.static_graph:
        static_graph = MassShatterStaticGraph(
            chemical_bond_index=chemical_bond_index.cpu(),
            atom_batch_indices=atom_batch_indices.cpu(),
            atom_ptr=atom_ptr.cpu(),
        )
    smarts_templates = None
    if args.smarts:
        cut_bond = max(args.num_atoms // 2 - 1, 0)
        left_component = tuple(range(cut_bond + 1))
        right_component = tuple(range(cut_bond + 1, args.num_atoms))
        template = MassShatterTemplate(
            broken_bond_indices=(cut_bond,),
            components=(left_component, right_component),
            rule_index=1,
            rule_name="synthetic_chain_split",
        )
        smarts_templates = tuple((template,) for _ in range(args.batch_size))

    candidate_kwargs = {
        "max_masks": args.max_masks,
        "max_unique_components": args.max_unique_components,
        "max_enumerated_masks": args.max_enumerated_masks,
        "include_single_bond_cuts": True,
        "mask_selection_threads": args.mask_selection_threads,
        "mixture_log_weights": mixture_log_weights,
        "candidate_group_ids": candidate_group_ids,
        "static_graph": static_graph,
        "smarts_templates": smarts_templates,
    }

    # 2. Warm caches before collecting uncached candidate-generation medians.
    for _ in range(args.warmup):
        build_mass_shatter_candidates(
            bond_logits,
            chemical_bond_index,
            atom_batch_indices,
            atom_ptr,
            **candidate_kwargs,
        )

    # 3. Run repeated candidate-generation passes.
    durations_ms: list[float] = []
    candidates = None
    for _ in range(args.repeats):
        if device.type == "cuda":
            th.cuda.synchronize()
        start = time.perf_counter()
        candidates = build_mass_shatter_candidates(
            bond_logits,
            chemical_bond_index,
            atom_batch_indices,
            atom_ptr,
            **candidate_kwargs,
        )
        if device.type == "cuda":
            th.cuda.synchronize()
        durations_ms.append((time.perf_counter() - start) * 1000.0)

    # 4. Report end-to-end candidate throughput and the final phase profile.
    assert candidates is not None
    LOGGER.info(
        "device=%s mixtures=%d threads=%d group_multiplicity=%d durations_ms=%s",
        device,
        args.num_mixtures,
        args.mask_selection_threads,
        args.candidate_group_multiplicity,
        durations_ms,
    )
    LOGGER.info("median_ms=%.3f", statistics.median(durations_ms))
    LOGGER.info(
        "masks=%d components=%d representatives=%d memberships=%d boundaries=%d",
        candidates.mask_batch_indices.numel(),
        candidates.component_mask_indices.numel(),
        candidates.unique_representative_component_indices.numel(),
        candidates.atom_indices.numel(),
        candidates.boundary_bond_indices.numel(),
    )
    LOGGER.info("selected_mask_counts=%s", candidates.selected_mask_counts)
    LOGGER.info("retained_mask_counts=%s", candidates.retained_mask_counts)
    LOGGER.info("scanned_mask_counts=%s", candidates.scanned_mask_counts)
    LOGGER.info(
        "retained_selected_ratios=%s",
        tuple(
            retained / selected
            for retained, selected in zip(
                candidates.retained_mask_counts,
                candidates.selected_mask_counts,
                strict=True,
            )
        ),
    )
    index_tensor_names = (
        "mask_batch_indices",
        "mask_cut_counts",
        "mechanism_mask_indices",
        "mechanism_rule_indices",
        "mechanism_boundary_bond_indices",
        "boundary_mechanism_indices",
        "component_mask_indices",
        "component_mechanism_indices",
        "component_batch_indices",
        "component_cut_counts",
        "component_unique_indices",
        "unique_representative_component_indices",
        "atom_indices",
        "atom_component_indices",
        "boundary_bond_indices",
        "boundary_component_indices",
    )
    transferred_bytes = sum(
        getattr(candidates, name).numel() * getattr(candidates, name).element_size()
        for name in index_tensor_names
    )
    transferred_bytes += 8 * (
        bond_logits.shape[0] + 2 * int(candidates.mask_cut_counts.sum().item())
    )
    LOGGER.info("candidate_index_bytes=%d", transferred_bytes)
    LOGGER.info("phase_profile=%s", candidates.timings)


if __name__ == "__main__":
    main()
