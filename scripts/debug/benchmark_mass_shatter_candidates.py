"""Benchmark MASShatter candidate generation on a synthetic molecular batch."""

from __future__ import annotations

import argparse
import logging
import statistics
import time

import torch as th

from fragnnet.frag.mass_shatter_candidates import build_mass_shatter_candidates

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
    bond_logits = th.linspace(
        -2.0,
        2.0,
        chemical_bond_index.shape[1],
        requires_grad=True,
    )
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
    parser.add_argument("--max_masks", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.device == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = th.device(args.device)

    # 1. Build one deterministic workload matching the L40S mask budget.
    inputs = build_chain_batch(args.batch_size, args.num_atoms, device)

    # 2. Run repeated uncached candidate-generation passes.
    durations_ms: list[float] = []
    candidates = None
    for _ in range(args.repeats):
        if device.type == "cuda":
            th.cuda.synchronize()
        start = time.perf_counter()
        candidates = build_mass_shatter_candidates(
            *inputs,
            max_masks=args.max_masks,
            explore_masks=0,
            exploration_mode="bounded_additive",
            exploration_temperature=1.0,
            exploration_cut_count_weights=(0.0, 0.0, 0.35, 0.30, 0.20, 0.10, 0.05),
            include_single_bond_cuts=True,
        )
        if device.type == "cuda":
            th.cuda.synchronize()
        durations_ms.append((time.perf_counter() - start) * 1000.0)

    # 3. Report end-to-end throughput and the final pass's phase profile.
    assert candidates is not None
    LOGGER.info("device=%s durations_ms=%s", device, durations_ms)
    LOGGER.info("median_ms=%.3f", statistics.median(durations_ms))
    LOGGER.info(
        "masks=%d components=%d memberships=%d",
        candidates.mask_batch_indices.numel(),
        candidates.component_mask_indices.numel(),
        candidates.atom_indices.numel(),
    )
    LOGGER.info("phase_profile=%s", candidates.timings)


if __name__ == "__main__":
    main()
