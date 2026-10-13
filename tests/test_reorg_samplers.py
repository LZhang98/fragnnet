"""Sampler batch-order records for the repository reorganization (plan Phase 0 item 6).

Every sampler the runner can select is built through ``runner.init_dataloader`` on a real
synthetic dataset from ``reorg_fixtures`` (``SpecMolFragDataset`` with packed DAGs for FraGNNet,
``SpecMolDataset`` for MASSter), by the config switches that select it:

- FraGNNet: ``SpecMolFragDynamicBatchSampler`` over a random, group, formula+adduct or weighted
  sampler, the DAG-size curriculum, ``DualGroupDynamicBatchSampler`` and
  ``CEContextDynamicBatchSampler``;
- MASSter: ``MassterCEContextDynamicBatchSampler``, ``BondCountCurriculumSampler`` and the
  generic group samplers behind a plain ``BatchSampler`` (``GroupSampler`` and the seven
  ``get_group_sampler`` types).

The selected sampler class names are asserted, so a wrong switch fails loudly. For each case the
batches (``spec_unique_id`` of every batch of one pass over the real dataloader) are recorded for
six epochs by driving the real ``SpectrumPL._seed_dataloader(epoch)`` through a stub trainer that
exposes exactly what the hook touches; the records never call ``set_epoch`` themselves. A resumed
run (a fresh dataset, dataloader and module, then ``_seed_dataloader(N)``) must reproduce epochs
``N`` and ``N + 1`` of the uninterrupted run: samplers hold no checkpointed state, so this guards
the ``(hparams.seed, epoch)`` derivation. Each case also asserts invariants (coverage, per-batch
budget, per-batch spectrum cap) next to the golden comparison in
``tests/golden/reorg/samplers.json``.

Behavior pinned as it is today, including accidents a refactor must not change silently:

- ``meta_info.json`` beside the DAGs gives the dynamic samplers raw DAG sizes; without it they
  load each item, whose bigraph has doubled edges. The ``*_no_meta_info`` cases pin that.
- ``FormulaPrecTypeGroupSampler`` is never re-sampled by ``_seed_dataloader`` (the hook only
  recomputes samplers that define ``_pre_compute_batches``), so every epoch repeats one order;
  under a DAG curriculum the hook calls its ``set_epoch``, which ignores the run seed.
- ``DualGroupDynamicBatchSampler`` iterates sets of ``adduct::mol_id`` strings, so its order
  changes with ``PYTHONHASHSEED`` (even ``num_batches`` does). Two records cover it. One is
  taken in this process with the key sets handed to the sampler in sorted order
  (``sorted_dual_group_keys``): it depends only on torch's generator, so it holds on every
  Python version and hash seed and guards the batching logic. The other is produced by a fresh
  interpreter with ``PYTHONHASHSEED=0`` and keyed by the Python minor version (the string hash
  differs between versions): it guards the real hash order, on ``PINNED_PYTHON`` only. There a
  missing record fails; on another minor version that comparison is skipped.
- ``DualGroupDynamicBatchSampler`` yields numpy integers and ``SpecMolFragDataset.__getitem__``
  asserts ``isinstance(idx, int)``, so a real dual-group train batch cannot be loaded. Its case
  therefore reads the batch sampler's indices instead of the dataloader's batches, and
  ``test_dual_group_batches_load_through_the_dataset`` is a strict ``xfail`` that documents the bug.
- ``DualGroupDynamicBatchSampler.set_epoch`` (epoch-only seed, ignores the run seed) is never
  called by the hook for any config ``validate_sampler_config`` accepts;
  ``test_dual_group_set_epoch_reseeds_from_the_epoch_alone`` pins it directly.

The two size curricula run on their own table (``TABLES["curriculum"]``) whose sizes make the
quantile rank ``ceil(N * fraction)`` land in a different run of equal sizes than ``floor`` or
``round`` would. The number of eligible spectra of every epoch is a literal in the case, not
recomputed from the sampler's formula, so a rounding change is a count mismatch;
``test_curriculum_table_separates_rounding_rules`` keeps the table honest.

The index sequences depend on torch's CPU random streams (``randperm``, ``multinomial``); the
torch version of the golden is stored under ``env`` and named when a comparison fails. Regenerate
after a torch upgrade, never to make a refactor pass:

    REORG_UPDATE_GOLDEN=1 pytest tests/test_reorg_samplers.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import types
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import reorg_fixtures as F
import torch as th
from reorg_golden import UPDATE, load_json, ratchet_json, write_json
from torch.utils.data import DataLoader

from fragnnet.models.fragnnet.fragnnet_data_sampler import DualGroupDynamicBatchSampler
from fragnnet.pl_model.spectrum_pl import SpectrumPL

GOLDEN = "samplers"
ENV_KEY = "env"
PINNED_PYTHON = (3, 11)  # the interpreter the dual-group golden is recorded with
SEED = 42
EPOCHS = tuple(range(6))
RESUME_AT = (1, 4)  # each resume check replays epochs N and N + 1
# Everything SpectrumPL._setup_sampler and SpectrumPL._seed_dataloader read from hparams. The
# stub carries only these, so a hook that starts reading another hyperparameter fails the test.
# model_type joined in Phase 4: _seed_dataloader asks the model's hooks whether to advance.
HOOK_HPARAMS = (
    "model_type",
    "seed",
    "max_epochs",
    "train_batch_size",
    "accumulate_grad_batches",
    "automatic_optimization",
    "dag_size_curriculum",
    "bond_size_curriculum",
    "batch_sampler_type",
    "masster_batch_sampler",
)
BASE_OVERRIDES: dict[str, Any] = {
    "max_epochs": 8,
    "train_batch_size": 3,
    "eval_batch_size": 4,
    "accumulate_grad_batches": 2,
    "spec_params": {"prec_types": ["[M+H]+", "[M+Na]+"]},
}
EVAL_CAP = BASE_OVERRIDES["eval_batch_size"]

Row = tuple[int, int, int, str, str, str, str]


@dataclass(frozen=True)
class Table:
    """Molecules and spectra of one synthetic dataset.

    Attributes:
        smiles: Molecules; ``mol_id`` is the position.
        rows: ``(mol_id, group_id, nce, split, prec_type, inst_type, frag_mode)`` per spectrum in
            ``spec_id`` order; group = molecule x adduct x instrument, as in the real tables.
        mixed_mode_groups: Groups whose spectra mix fragmentation modes.
    """

    smiles: tuple[str, ...]
    rows: tuple[Row, ...]
    mixed_mode_groups: frozenset[int] = frozenset()

    @property
    def spectra(self) -> tuple[F.Spectrum, ...]:
        """The rows as the fixture's ``Spectrum`` records."""
        return tuple(
            F.Spectrum(i, mol, group, float(nce), split, prec_type=prec, inst_type=inst)
            for i, (mol, group, nce, split, prec, inst, _mode) in enumerate(self.rows)
        )


# Main table. Covered structure: three-level CE ladders, a CE replicate (group 2), a single-CE
# group (4), a second adduct (5), a second instrument (3), isomer pairs (0/1, 6/7), a group with
# mixed fragmentation modes (6) and oversized molecules (4 in train, 10 in val).
MAIN = Table(
    smiles=(
        "CCO",  # 0: C2H6O, isomer of 1
        "COC",  # 1
        "CC(=O)O",  # 2
        "CNC(C)=O",  # 3
        "Oc1ccccc1",  # 4: 212 DAG edges, over every budget below
        "NCC(=O)O",  # 5
        "CC(C)O",  # 6: C3H8O, isomer of 7
        "CCCO",  # 7
        "COC(C)=O",  # 8
        "CCC(=O)O",  # 9
        "Nc1ccccc1",  # 10: 212 DAG edges
    ),
    rows=(
        (0, 0, 20, "train", "[M+H]+", "ORB", "HCD"),
        (0, 0, 30, "train", "[M+H]+", "ORB", "HCD"),
        (0, 0, 40, "train", "[M+H]+", "ORB", "HCD"),
        (1, 1, 20, "train", "[M+H]+", "ORB", "HCD"),
        (1, 1, 40, "train", "[M+H]+", "ORB", "HCD"),
        (2, 2, 30, "train", "[M+H]+", "ORB", "HCD"),
        (2, 2, 30, "train", "[M+H]+", "ORB", "HCD"),
        (2, 2, 50, "train", "[M+H]+", "ORB", "HCD"),
        (3, 3, 20, "train", "[M+H]+", "ORB_F", "HCD"),
        (3, 3, 40, "train", "[M+H]+", "ORB_F", "HCD"),
        (3, 3, 60, "train", "[M+H]+", "ORB_F", "HCD"),
        (5, 4, 25, "train", "[M+H]+", "ORB", "HCD"),
        (5, 5, 20, "train", "[M+Na]+", "ORB", "HCD"),
        (5, 5, 40, "train", "[M+Na]+", "ORB", "HCD"),
        (6, 6, 20, "train", "[M+H]+", "ORB", "HCD"),
        (6, 6, 40, "train", "[M+H]+", "ORB", "CID"),
        (7, 7, 30, "train", "[M+H]+", "ORB", "CID"),
        (7, 7, 50, "train", "[M+H]+", "ORB", "CID"),
        (4, 8, 20, "train", "[M+H]+", "ORB", "HCD"),
        (4, 8, 40, "train", "[M+H]+", "ORB", "HCD"),
        (8, 9, 20, "val", "[M+H]+", "ORB", "HCD"),
        (8, 9, 40, "val", "[M+H]+", "ORB", "HCD"),
        (9, 10, 30, "val", "[M+H]+", "ORB", "HCD"),
        (9, 10, 50, "val", "[M+H]+", "ORB", "HCD"),
        (10, 11, 20, "val", "[M+H]+", "ORB", "HCD"),
        (10, 11, 40, "val", "[M+H]+", "ORB", "HCD"),
    ),
    mixed_mode_groups=frozenset({6}),
)

# Curriculum table: 20 train spectra whose sorted heavy-bond counts and fragment-DAG node counts
# form runs of equal values ending at the cumulative ranks below, so the quantile rank
# ceil(20 * fraction) of the schedule (start 0.5, 3 epochs: 10, 14, 17, 20) lands in another run
# than floor (10, 13, 16, 20) or round (10, 13, 17, 20) would:
#   bonds  2   3   4   5   6   7        (mol 0/1, 2/3, 4, 5, 6, 7)
#   spectra 5  5   3   3   2   2    -> run ends at ranks 5, 10, 13, 16, 18, 20
#   nodes  6   10/11 17 24  31  48     -> the same ends, plus 8 inside the bond-3 run
# Isomer pairs 0/1 (C2H6O) and 2/3 (C3H8O); molecule 7 (phenol, 48 nodes, 212 edges) is over the
# node budget of 40 and molecule 9 (aniline) is the oversized validation molecule.
CURRICULUM = Table(
    smiles=(
        "CCO",  # 0: 2 bonds, 6 DAG nodes
        "COC",  # 1: 2 bonds, 6 nodes
        "CC(C)O",  # 2: 3 bonds, 11 nodes
        "CCCO",  # 3: 3 bonds, 10 nodes
        "NCC(=O)O",  # 4: 4 bonds, 17 nodes
        "CCCC(=O)O",  # 5: 5 bonds, 24 nodes
        "C1CCCCC1",  # 6: 6 bonds, 31 nodes
        "Oc1ccccc1",  # 7: 7 bonds, 48 nodes
        "CC(=O)O",  # 8: validation
        "Nc1ccccc1",  # 9: validation, 48 nodes
    ),
    rows=(
        (0, 0, 20, "train", "[M+H]+", "ORB", "HCD"),
        (0, 0, 30, "train", "[M+H]+", "ORB", "HCD"),
        (0, 0, 40, "train", "[M+H]+", "ORB", "HCD"),
        (1, 1, 20, "train", "[M+H]+", "ORB", "HCD"),
        (1, 1, 40, "train", "[M+H]+", "ORB", "HCD"),
        (2, 2, 20, "train", "[M+H]+", "ORB", "HCD"),
        (2, 2, 40, "train", "[M+H]+", "ORB", "HCD"),
        (3, 3, 20, "train", "[M+H]+", "ORB", "HCD"),
        (3, 3, 30, "train", "[M+H]+", "ORB", "HCD"),
        (3, 3, 50, "train", "[M+H]+", "ORB", "HCD"),
        (4, 4, 20, "train", "[M+H]+", "ORB", "HCD"),
        (4, 4, 40, "train", "[M+H]+", "ORB", "HCD"),
        (4, 4, 60, "train", "[M+H]+", "ORB", "HCD"),
        (5, 5, 20, "train", "[M+H]+", "ORB", "HCD"),
        (5, 5, 40, "train", "[M+H]+", "ORB", "HCD"),
        (5, 5, 60, "train", "[M+H]+", "ORB", "HCD"),
        (6, 6, 30, "train", "[M+H]+", "ORB", "HCD"),
        (6, 6, 50, "train", "[M+H]+", "ORB", "HCD"),
        (7, 7, 20, "train", "[M+H]+", "ORB", "HCD"),
        (7, 7, 40, "train", "[M+H]+", "ORB", "HCD"),
        (8, 8, 20, "val", "[M+H]+", "ORB", "HCD"),
        (8, 8, 40, "val", "[M+H]+", "ORB", "HCD"),
        (9, 9, 20, "val", "[M+H]+", "ORB", "HCD"),
        (9, 9, 40, "val", "[M+H]+", "ORB", "HCD"),
    ),
)
TABLES: dict[str, Table] = {"main": MAIN, "curriculum": CURRICULUM}


@dataclass(frozen=True)
class Curriculum:
    """A size curriculum: eligibility grows linearly from ``start`` to every spectrum.

    Attributes:
        key: Size the curriculum ranks by, ``frag_node`` (DAG nodes) or ``bond`` (heavy bonds).
        start: Fraction of the smallest spectra eligible at epoch 0.
        epochs: Epoch at which every spectrum is eligible.
        eligible: Number of train spectra eligible in each of ``EPOCHS``, written out by hand
            from the table's size runs (not computed with the sampler's rounding rule).
    """

    key: str
    start: float
    epochs: int
    eligible: tuple[int, ...]


@dataclass(frozen=True)
class Case:
    """One sampler configuration and what it must produce.

    Attributes:
        name: Golden key.
        model_type: ``frag_gnn`` (SpecMolFragDataset) or ``masster`` (SpecMolDataset).
        table: Key of ``TABLES``, the synthetic table the case runs on.
        overrides: Config switches that select the sampler.
        train: Expected train ``batch_sampler/sampler`` class names (``sampler`` omitted when
            the batch sampler has none).
        val: The same for the validation loader.
        kind: Coverage rule checked each epoch (see ``_check_train_epoch``).
        budget: Per-spectrum size the sampler packs by, if any.
        max_num: Budget per batch.
        cap: Maximum spectra per train batch.
        hook: Whether ``_seed_dataloader`` takes the set_epoch/Lightning-counter branch.
        meta_info: Whether ``meta_info.json`` sits beside the DAGs.
        frozen: Whether every epoch repeats one order (``FormulaPrecTypeGroupSampler``).
        seed_independent: Whether the order ignores the run seed (true of a frozen order, and of
            ``FormulaPrecTypeGroupSampler`` under a curriculum, whose ``set_epoch`` reseeds it
            from the construction-time generator).
        aligned: Whether batch boundaries include every multiple of ``cap`` spectra.
        k: Per-group cap (group samplers) or average per group (weighted samplers).
        curriculum: The size curriculum, if the case runs one.
        fixed: Fixed-size batching through a plain ``BatchSampler``.
        drop_last: ``drop_last`` of the fixed batching.
        via_loader: Whether batches are read through the real dataloader (collate included).
            False reads the batch sampler's indices, for a sampler whose indices the dataset
            rejects (see ``test_dual_group_batches_load_through_the_dataset``).
    """

    name: str
    model_type: str
    overrides: dict[str, Any]
    train: str
    val: str
    kind: str
    budget: str | None = None
    max_num: int | None = None
    cap: int = 3
    hook: bool = False
    meta_info: bool = True
    frozen: bool = False
    seed_independent: bool = False
    aligned: bool = False
    k: int = 0
    curriculum: Curriculum | None = None
    fixed: bool = False
    drop_last: bool = False
    via_loader: bool = True
    table: str = "main"


FRAG_CAP = BASE_OVERRIDES["train_batch_size"] * BASE_OVERRIDES["accumulate_grad_batches"]
DYN = {
    "automatic_optimization": False,
    "dynamic_batch_sampler": True,
    "dynamic_batch_sampler_mode": "frag_edge",
    "dynamic_batch_sampler_max": 100,
}
DYN_VAL = "SpecMolFragDynamicBatchSampler/SequentialSampler"
PLAIN_VAL = "BatchSampler/SequentialSampler"
MASSTER = {"automatic_optimization": True}
MASSTER_CE_MAX = 12  # directed molecular edges; phenol has 14, aniline 14
SIMPLE_TYPES = {
    "group": "weighted_unique",
    "mol": "weighted_unique",
    "group_mol": "weighted_unique",
    "adduct_inst_type": "weighted_replace",
    "adduct_inst_type_group": "weighted_replace",
    "adduct_inst_type_frag_mode": "weighted_replace",
    "adduct_inst_type_frag_mode_group": "weighted_replace",
}


def _frag_case(name: str, overrides: dict[str, Any], train: str, kind: str, **kw: Any) -> Case:
    kw = {"budget": "frag_edge", "max_num": 100, "cap": FRAG_CAP, "val": DYN_VAL, **kw}
    return Case(name, "frag_gnn", {**DYN, **overrides}, train, kind=kind, **kw)


def _fixed_case(name: str, overrides: dict[str, Any], inner: str, kind: str, **kw: Any) -> Case:
    train = f"BatchSampler/{inner}"
    return Case(name, "masster", {**MASSTER, **overrides}, train, PLAIN_VAL, kind, fixed=True, **kw)


# Eligible train spectra per epoch for both curricula on ``CURRICULUM``: the smallest 10, 16
# (bond counts <= 5, nodes <= 24), 18 (<= 6, <= 31), then all 20. Floor would give 10, 13, 16, 20.
CURRICULUM_ELIGIBLE = (10, 16, 18, 20, 20, 20)
CE_CONTEXT = {"batch_sampler_type": "ce_context", "ce_context_spectra_per_context": 2}
DUAL_GROUP = {
    "dynamic_batch_sampler": False,
    "dual_group_dynamic_batch_sampler": True,
    "dual_group_dynamic_batch_sampler_mode": "frag_edge",
    "dual_group_dynamic_batch_sampler_max": 100,
    "dual_group_sampler_k1": 1,
    "dual_group_sampler_k2": 1,
    "dual_group_sampler_k1_per_key": 2,
    "dual_group_sampler_k2_per_key": 1,
}
DYN_BATCHER = "SpecMolFragDynamicBatchSampler"
CASES: dict[str, Case] = {
    case.name: case
    for case in (
        _frag_case("frag_dynamic_random", {}, f"{DYN_BATCHER}/RandomSampler", "all", aligned=True),
        _frag_case(
            "frag_dynamic_group",
            {"group_sampler": True, "group_sampler_max_per_group": 2},
            f"{DYN_BATCHER}/GroupSampler",
            "group_cap",
            aligned=True,
            k=2,
        ),
        _frag_case(
            "frag_dynamic_simple_group",
            {
                "simple_group_sampler": True,
                "simple_group_sampler_type": "adduct_inst_type",
                "simple_group_sampler_avg_per_group": 2,
            },
            f"{DYN_BATCHER}/WeightedRandomSampler",
            "weighted_replace",
            aligned=True,
            k=2,
        ),
        _frag_case(
            "frag_dynamic_formula_group",
            {
                "formula_prec_type_group_sampler": True,
                "formula_prec_type_group_sampler_max_per_group": 2,
            },
            f"{DYN_BATCHER}/FormulaPrecTypeGroupSampler",
            "formula_cap",
            frozen=True,
            seed_independent=True,
            aligned=True,
            k=2,
        ),
        _frag_case(
            "frag_dag_curriculum",
            {
                "dynamic_batch_sampler_mode": "frag_node",
                "dynamic_batch_sampler_max": 40,
                "dag_size_curriculum": True,
                "dag_size_curriculum_start_fraction": 0.5,
                "dag_size_curriculum_epochs": 3,
            },
            f"{DYN_BATCHER}/RandomSampler",
            "curriculum",
            budget="frag_node",
            max_num=40,
            hook=True,
            aligned=True,
            curriculum=Curriculum("frag_node", 0.5, 3, CURRICULUM_ELIGIBLE),
            table="curriculum",
        ),
        _frag_case(
            "frag_dag_curriculum_formula_group",
            {
                "dynamic_batch_sampler_mode": "frag_node",
                "dynamic_batch_sampler_max": 40,
                "dag_size_curriculum": True,
                "dag_size_curriculum_start_fraction": 0.5,
                "dag_size_curriculum_epochs": 3,
                "formula_prec_type_group_sampler": True,
                "formula_prec_type_group_sampler_max_per_group": 2,
            },
            f"{DYN_BATCHER}/FormulaPrecTypeGroupSampler",
            "curriculum_cap",
            budget="frag_node",
            max_num=40,
            hook=True,
            aligned=True,
            seed_independent=True,
            k=2,
            curriculum=Curriculum("frag_node", 0.5, 3, CURRICULUM_ELIGIBLE),
            table="curriculum",
        ),
        _frag_case(
            "frag_dual_group",
            DUAL_GROUP,
            "DualGroupDynamicBatchSampler",
            "dual_group",
            via_loader=False,
        ),
        _frag_case(
            "frag_ce_context",
            CE_CONTEXT,
            "CEContextDynamicBatchSampler",
            "ce_context",
            hook=True,
            val="CEContextDynamicBatchSampler",
        ),
        _frag_case(
            "frag_dynamic_random_no_meta_info",
            {},
            f"{DYN_BATCHER}/RandomSampler",
            "all",
            aligned=True,
            meta_info=False,
        ),
        _frag_case(
            "frag_ce_context_no_meta_info",
            CE_CONTEXT,
            "CEContextDynamicBatchSampler",
            "ce_context",
            hook=True,
            meta_info=False,
            val="CEContextDynamicBatchSampler",
        ),
        Case(
            "masster_ce_context",
            "masster",
            {
                **MASSTER,
                "masster_batch_sampler": True,
                "masster_batch_sampler_max": MASSTER_CE_MAX,
                "ce_context_spectra_per_context": 2,
            },
            "MassterCEContextDynamicBatchSampler",
            "MassterCEContextDynamicBatchSampler",
            "ce_context",
            budget="mol_edge",
            max_num=MASSTER_CE_MAX,
            cap=BASE_OVERRIDES["train_batch_size"],
            hook=True,
        ),
        _fixed_case(
            "masster_bond_curriculum",
            {
                "bond_size_curriculum": True,
                "bond_size_curriculum_start_fraction": 0.5,
                "bond_size_curriculum_epochs": 3,
            },
            "BondCountCurriculumSampler",
            "curriculum",
            hook=True,
            curriculum=Curriculum("bond", 0.5, 3, CURRICULUM_ELIGIBLE),
            table="curriculum",
        ),
        _fixed_case(
            "masster_group_fixed",
            {"group_sampler": True, "group_sampler_max_per_group": 2},
            "GroupSampler",
            "group_cap",
            k=2,
        ),
        _fixed_case("masster_random_fixed", {}, "RandomSampler", "all"),
        _fixed_case(
            "masster_random_fixed_drop_last",
            {"drop_last": True},
            "RandomSampler",
            "drop_last",
            drop_last=True,
        ),
        *(
            _fixed_case(
                f"masster_simple_{sampler_type}",
                {
                    "simple_group_sampler": True,
                    "simple_group_sampler_type": sampler_type,
                    "simple_group_sampler_avg_per_group": 2,
                },
                "WeightedRandomSampler",
                kind,
                k=2,
            )
            for sampler_type, kind in SIMPLE_TYPES.items()
        ),
    )
}
DUAL_GROUP_CASE = "frag_dual_group"


# --------------------------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------------------------


def write_data(root: Path, table: Table, meta_info: bool) -> F.SynthData:
    """Write a sampler table and add the fragmentation-mode column the fixture omits.

    Args:
        root: Empty directory to write into.
        table: Molecules and spectra to write.
        meta_info: Whether ``meta_info.json`` is written beside the DAGs.

    Returns:
        The written dataset; ``spec_df.pkl`` on disk carries ``frag_mode``.
    """
    data = F.write_synth_data(root, spectra=table.spectra, smiles=table.smiles, meta_info=meta_info)
    spec_df = pd.read_pickle(data.spec_fp)
    spec_df["frag_mode"] = [table.rows[int(spec_id)][6] for spec_id in spec_df["spec_id"]]
    spec_df.to_pickle(data.spec_fp)
    return data


class DataStore:
    """Synthetic datasets, each written on first use for a ``(table, meta_info)`` pair."""

    def __init__(self, factory: pytest.TempPathFactory) -> None:
        self._factory = factory
        self._written: dict[tuple[str, bool], F.SynthData] = {}

    def get(self, table: str, meta_info: bool = True) -> F.SynthData:
        """The dataset of ``TABLES[table]``, with or without ``meta_info.json``."""
        key = (table, meta_info)
        if key not in self._written:
            root = self._factory.mktemp(f"samplers_{table}_{'meta' if meta_info else 'no_meta'}")
            self._written[key] = write_data(root, TABLES[table], meta_info)
        return self._written[key]

    def for_case(self, case: Case) -> F.SynthData:
        """The dataset ``case`` runs on."""
        return self.get(case.table, case.meta_info)


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> DataStore:
    """Lazily written synthetic datasets, shared by the tests of this module."""
    return DataStore(tmp_path_factory)


@pytest.fixture(scope="module", autouse=True)
def record_environment() -> None:
    """When regenerating, store the torch build the sequences were recorded on."""
    if UPDATE:
        write_json(GOLDEN, ENV_KEY, environment())


class RecordCache(dict):
    """``{case name: record}`` computed on first access, so tests can share one run per case."""

    def __init__(self, store: DataStore) -> None:
        super().__init__()
        self._store = store

    def __missing__(self, name: str) -> dict[str, Any]:
        case = CASES[name]
        self[name] = record_case(case, self._store.for_case(case))
        return self[name]


@pytest.fixture(scope="module")
def records(store: DataStore) -> RecordCache:
    """Lazily filled cache of ``record_case`` results, shared by the tests of this module."""
    return RecordCache(store)


def environment() -> dict[str, str]:
    """The torch build whose CPU random streams (``randperm``, ``multinomial``) the golden holds."""
    return {"torch": str(th.__version__)}


def ratchet(key: str, record: dict[str, Any]) -> None:
    """``ratchet_json`` on the sampler golden; a mismatch names the torch builds involved.

    Args:
        key: Record key inside the golden file.
        record: JSON-serializable record to compare.

    Raises:
        AssertionError: If the record differs; the message adds the recorded and the running
            torch version when they differ, since a changed random stream looks like a sampler
            change.
    """
    try:
        ratchet_json(GOLDEN, key, record)
    except AssertionError as error:
        recorded = load_json(GOLDEN).get(ENV_KEY)
        if recorded is not None and recorded != environment():
            raise AssertionError(
                f"{error}\nthe golden was recorded with {recorded} but this is {environment()}: "
                "a changed torch random stream moves every sampler order; confirm the cause "
                "before regenerating"
            ) from None
        raise


@dataclass
class SplitInfo:
    """Independent facts about the spectra of one split, from the frames and the DAGs."""

    ids: list[int]
    group: dict[int, int]
    mol: dict[int, int]
    formula_key: dict[int, str]
    ce: dict[int, float]
    nodes: dict[int, int]
    edges: dict[int, int]
    bonds: dict[int, int]
    mol_edges: dict[int, int]

    def size(self, budget: str) -> dict[int, int]:
        """Per-spectrum size under a sampler budget (``frag_node``/``frag_edge``/``mol_edge``)."""
        return {"frag_node": self.nodes, "frag_edge": self.edges, "mol_edge": self.mol_edges}[
            budget
        ]


def split_info(ds: Any, data: F.SynthData, case: Case) -> SplitInfo:
    """Derive sizes and group keys independently of the samplers, and cross-check the dataset."""
    ids = [int(i) for i in ds.spec_df["spec_id"]]
    mol = {i: int(m) for i, m in zip(ids, ds.spec_df["mol_id"])}
    group = {i: int(g) for i, g in zip(ids, ds.spec_df["group_id"])}
    prec = dict(zip(ids, ds.spec_df["prec_type"]))
    formula_key = {i: f"{prec[i]}::{ds.mol_df.loc[mol[i], 'formula']}" for i in ids}
    ce = {i: float(np.mean(v)) for i, v in zip(ids, ds.spec_df["nce"])}
    bonds = {i: int(ds.mol_df.loc[mol[i], "num_bonds"]) for i in ids}
    # raw DAG counts; the loaded bigraph doubles the edges, which only matters without
    # meta_info.json because that file holds the raw counts
    edge_scale = 1 if case.meta_info else 2
    nodes = {i: int(data.dags[mol[i]]["dag"].num_nodes) for i in ids}
    edges = {i: edge_scale * int(data.dags[mol[i]]["dag"].num_edges) for i in ids}
    mol_edges = {i: 2 * bonds[i] for i in ids}
    if case.model_type == "frag_gnn":
        if case.meta_info:
            assert [list(x) for x in ds.precom_meta_info] == [[nodes[i], edges[i]] for i in ids]
        else:
            assert not hasattr(ds, "precom_meta_info")
            loaded = [ds[row]["frag_pyg"] for row in range(len(ids))]
            assert [(g.num_nodes, g.num_edges) for g in loaded] == [
                (nodes[i], edges[i]) for i in ids
            ]
    else:
        assert [ds[row]["mol_pyg"].num_edges for row in range(len(ids))] == [
            mol_edges[i] for i in ids
        ]
    assert ds.ce_key == "nce"
    return SplitInfo(ids, group, mol, formula_key, ce, nodes, edges, bonds, mol_edges)


# --------------------------------------------------------------------------------------------
# Building, driving and recording
# --------------------------------------------------------------------------------------------


def make_case_config(case: Case, data: F.SynthData, seed: int) -> dict[str, Any]:
    """Resolved config that selects ``case``'s sampler on ``data``."""
    return F.make_config(case.model_type, data, BASE_OVERRIDES, case.overrides, {"seed": seed})


@contextmanager
def open_split(config: dict[str, Any], split: str) -> Iterator[tuple[Any, DataLoader]]:
    """The runner's dataset and dataloader for ``split``; the DAG file is closed on exit."""
    dataset, loader = F.init_split(config, split)
    try:
        yield dataset, loader
    finally:
        F.close_dataset(dataset)


def sampler_names(loader: DataLoader) -> str:
    """``BatchSampler/Sampler`` class names of a loader (no ``/`` part without a sampler)."""
    batch_sampler = loader.batch_sampler
    inner = getattr(batch_sampler, "sampler", None)
    name = type(batch_sampler).__name__
    return name if inner is None else f"{name}/{type(inner).__name__}"


class SortedKeys:
    """Read-only view of a set of strings that iterates in sorted order, not hash order.

    ``DualGroupDynamicBatchSampler._sample_from_group_dict`` only truth-tests its active-key set
    and calls ``list()`` on it, so this stands in for the set there.
    """

    def __init__(self, keys: set[str]) -> None:
        self._keys = keys

    def __bool__(self) -> bool:
        return bool(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._keys))


@contextmanager
def sorted_dual_group_keys() -> Iterator[None]:
    """Make ``DualGroupDynamicBatchSampler`` visit its group keys in sorted order.

    The sampler permutes ``list(set of "adduct::mol_id" strings)``, so its batches depend on
    ``PYTHONHASHSEED``. Handing it a sorted view of each key set at the one place it reads them
    leaves all of its logic untouched and makes its order a pure function of the torch generator,
    so it can be compared across processes and Python versions.

    Raises:
        AssertionError: If the sampler no longer has ``_sample_from_group_dict``; the pin has to
            be adapted to wherever the sampler now reads its key sets.
    """
    original = getattr(DualGroupDynamicBatchSampler, "_sample_from_group_dict", None)
    assert original is not None, "DualGroupDynamicBatchSampler._sample_from_group_dict moved"

    def sorted_view(self: Any, group_dict: Any, active_keys: set[str], *args: Any, **kw: Any):
        return original(self, group_dict, SortedKeys(active_keys), *args, **kw)

    DualGroupDynamicBatchSampler._sample_from_group_dict = sorted_view
    try:
        yield
    finally:
        DualGroupDynamicBatchSampler._sample_from_group_dict = original


def stub_module(config: dict[str, Any], loader: DataLoader, hook: bool) -> types.SimpleNamespace:
    """A module and trainer exposing exactly what ``SpectrumPL._seed_dataloader`` touches.

    ``train_dl_seeds`` comes from the real ``SpectrumPL._setup_sampler``. ``hparams`` carries
    only ``HOOK_HPARAMS``, so a hook that reads any other hyperparameter raises
    ``AttributeError``. The non-curriculum branch only reads ``trainer.train_dataloader``; any
    other access raises too. The curriculum, CE-context and MASSter branch also resets
    Lightning's cached batch counts.

    Args:
        config: Resolved config; ``HOOK_HPARAMS`` of it become ``hparams``.
        loader: The train dataloader.
        hook: Whether the hook takes the curriculum/CE-context/MASSter branch.

    Returns:
        The stub module; ``fit_loop_resets`` counts ``_data_fetcher.reset`` calls.
    """
    hparams = types.SimpleNamespace(**{key: config[key] for key in HOOK_HPARAMS})
    module = types.SimpleNamespace(hparams=hparams, fit_loop_resets=0)
    SpectrumPL._setup_sampler(module)
    trainer = types.SimpleNamespace(train_dataloader=loader)
    if hook:
        num_batches = len(loader)

        def reset() -> None:
            module.fit_loop_resets += 1

        trainer.val_check_interval = 0.5
        trainer.val_check_batch = max(1, int(num_batches * 0.5))
        trainer.fit_loop = types.SimpleNamespace(
            _combined_loader=types.SimpleNamespace(limits=[num_batches]),
            _data_fetcher=types.SimpleNamespace(reset=reset),
            max_batches=num_batches,
        )
    module.trainer = trainer
    return module


def record_epochs(
    module: types.SimpleNamespace,
    epochs: Sequence[int],
    hook: bool,
    row_ids: Sequence[int] | None = None,
) -> dict[int, dict[str, Any]]:
    """Drive the real hook for each epoch, then take exactly one pass over the batches.

    One pass only: a plain ``RandomSampler`` and ``BondCountCurriculumSampler`` draw their order
    lazily from the generator, so a second pass would differ.

    Args:
        module: Stub module from ``stub_module``.
        epochs: Epochs to run, in order.
        hook: Whether the hook also resets Lightning's cached batch counts.
        row_ids: ``spec_id`` of every dataset row. When given, the pass reads the batch
            sampler's row indices and maps them through this table instead of loading the
            batches through the dataloader.

    Returns:
        ``{epoch: {"batches": [[spec_unique_id, ...], ...], "num_batches": len(loader)}}``.
    """
    out = {}
    resets_before = module.fit_loop_resets
    for done, epoch in enumerate(epochs, start=1):
        SpectrumPL._seed_dataloader(module, epoch)
        loader = module.trainer.train_dataloader
        num_batches = len(loader)
        if row_ids is None:
            batches = [[int(i) for i in batch["spec_unique_id"]] for batch in loader]
        else:
            batches = [[row_ids[int(row)] for row in batch] for batch in loader.batch_sampler]
        assert len(batches) == num_batches, (epoch, len(batches), num_batches)
        out[epoch] = {"batches": batches, "num_batches": num_batches}
        if hook:
            fit_loop = module.trainer.fit_loop
            assert fit_loop._combined_loader.limits == [num_batches]
            assert fit_loop.max_batches == num_batches
            assert module.trainer.val_check_batch == max(1, int(num_batches * 0.5))
            assert module.fit_loop_resets == resets_before + done
    return out


def encode(batches: Sequence[Sequence[int]]) -> str:
    """One line per epoch: batches separated by ``|``, spectrum ids by spaces."""
    return " | ".join(" ".join(str(i) for i in batch) for batch in batches)


def _quantile_limit(values: Sequence[int], start: float, epochs: int, epoch: int) -> int:
    """Size limit of a curriculum epoch: a quantile of the whole split that grows linearly."""
    fraction = start + (1.0 - start) * min(epoch / epochs, 1.0)
    target = min(len(values), max(1, math.ceil(len(values) * fraction)))
    return sorted(values)[target - 1]


def _check_batch_shape(case: Case, info: SplitInfo, epoch: int, batches: list[list[int]]) -> None:
    where = f"{case.name} epoch {epoch}"
    assert all(batches), f"{where}: empty batch"
    assert max(len(b) for b in batches) <= case.cap, f"{where}: batch over {case.cap} spectra"
    if case.fixed:
        assert all(len(b) == case.cap for b in batches[:-1]), where
        assert (len(batches[-1]) == case.cap) if case.drop_last else len(batches[-1]) <= case.cap
    if case.budget is not None:
        size = info.size(case.budget)
        for batch in batches:
            total = sum(size[i] for i in batch)
            assert total <= case.max_num, f"{where}: {batch} costs {total} > {case.max_num}"
    if case.aligned:
        # gradient accumulation: an optimizer step every `cap` spectra needs a batch boundary there
        ends = set(np.cumsum([len(b) for b in batches]).tolist())
        total_spectra = sum(len(b) for b in batches)
        assert all(m in ends for m in range(case.cap, total_spectra, case.cap)), where


def curriculum_sizes(case: Case, info: SplitInfo) -> dict[int, int]:
    """Per-spectrum sizes the curriculum of ``case`` ranks by."""
    assert case.curriculum is not None
    return {"frag_node": info.nodes, "bond": info.bonds}[case.curriculum.key]


def _check_train_epoch(
    case: Case, info: SplitInfo, loader: DataLoader, epoch: int, batches: list[list[int]]
) -> int | None:
    """Invariants of one train epoch: coverage by ``case.kind``, budget and spectrum cap.

    Returns:
        For a curriculum case, the number of train spectra eligible in this epoch (before the
        budget drops oversized ones), checked against the hand-written ``Curriculum.eligible``;
        otherwise None.
    """
    where = f"{case.name} epoch {epoch}"
    _check_batch_shape(case, info, epoch, batches)
    flat = [i for batch in batches for i in batch]
    size = info.size(case.budget) if case.budget else None
    fit = [i for i in info.ids if size is None or size[i] <= case.max_num]
    members: dict[int, list[int]] = {}
    for i in info.ids:
        members.setdefault(info.group[i], []).append(i)
    inner = getattr(loader.batch_sampler, "sampler", None)
    kind = case.kind
    if kind == "all":
        assert sorted(flat) == sorted(fit), f"{where}: every fitting spectrum exactly once"
    elif kind == "drop_last":
        assert len(set(flat)) == len(flat) == len(info.ids) // case.cap * case.cap, where
    elif kind in ("group_cap", "formula_cap"):
        key = info.group if kind == "group_cap" else info.formula_key
        by_key: dict[Any, list[int]] = {}
        for i in info.ids:
            by_key.setdefault(key[i], []).append(i)
        assert len(set(flat)) == len(flat), f"{where}: no spectrum twice"
        counts = Counter(key[i] for i in flat)
        for name, ids in by_key.items():
            fits = [i in fit for i in ids]
            assert all(fits) or not any(fits), "fixture: one key must not mix fitting sizes"
            assert counts[name] == (min(case.k, len(ids)) if fits[0] else 0), (where, name)
    elif kind == "weighted_unique":
        expected = sum(min(case.k, len(ids)) for ids in members.values())
        assert inner.num_samples == len(flat) == len(set(flat)) == expected, where
    elif kind == "weighted_replace":
        assert inner.num_samples == len(info.ids), where
        assert set(flat) <= set(fit) and len(flat) <= len(info.ids), where
        if case.fixed:
            assert len(flat) == len(info.ids), where
            assert len(set(flat)) < len(flat), f"{where}: drawn with replacement"
    elif kind in ("curriculum", "curriculum_cap"):
        curriculum = case.curriculum
        values = curriculum_sizes(case, info)
        limit = _quantile_limit(
            [values[i] for i in info.ids], curriculum.start, curriculum.epochs, epoch
        )
        eligible_all = [i for i in info.ids if values[i] <= limit]
        assert len(eligible_all) == curriculum.eligible[epoch], (
            f"{where}: {len(eligible_all)} spectra eligible at size <= {limit}, "
            f"expected {curriculum.eligible[epoch]}"
        )
        eligible = [i for i in fit if values[i] <= limit]
        if kind == "curriculum":
            assert sorted(flat) == sorted(eligible), f"{where}: eligible at <= {limit}"
            assert epoch > 0 or len(eligible) < len(fit), f"{where}: epoch 0 is restricted"
            past = epoch >= curriculum.epochs
            assert not past or eligible == fit, f"{where}: past the curriculum, all eligible"
        else:
            assert len(set(flat)) == len(flat) and set(flat) <= set(
                eligible
            ), f"{where}: <= {limit}"
            counts = Counter(info.formula_key[i] for i in flat)
            assert max(counts.values()) <= case.k, f"{where}: <= {case.k} per formula and adduct"
        return len(eligible_all)
    elif kind == "dual_group":
        assert sorted(flat) == sorted(fit), f"{where}: every fitting spectrum exactly once"
        paired = [
            batch for batch in batches if max(Counter(info.mol[i] for i in batch).values()) > 1
        ]
        assert paired, f"{where}: k1_per_key=2 puts same-molecule spectra in a batch"
    elif kind == "ce_context":
        assert sorted(flat) == sorted(fit), f"{where}: every fitting spectrum exactly once"
        for batch in batches:
            counts = Counter(info.group[i] for i in batch)
            assert max(counts.values()) <= 2, f"{where}: {batch} holds >2 spectra of a context"
            for group in counts:
                ces = [info.ce[i] for i in batch if info.group[i] == group]
                assert len(set(ces)) == len(ces), f"{where}: replicate CE in one batch {batch}"
            for group in TABLES[case.table].mixed_mode_groups:
                assert counts[group] <= 1, f"{where}: mixed-mode group is primary-loss only"
        assert any(
            max(Counter(info.group[i] for i in batch).values()) == 2 for batch in batches
        ), f"{where}: no adjacent-CE pair was batched"
    else:
        raise AssertionError(f"unknown kind {kind}")
    return None


def _check_val(case: Case, info: SplitInfo, batches: list[list[int]]) -> None:
    """Validation loaders cover every spectrum; oversized ones become singleton batches."""
    flat = [i for batch in batches for i in batch]
    assert sorted(flat) == sorted(info.ids), f"{case.name} val: every spectrum exactly once"
    assert max(len(b) for b in batches) <= EVAL_CAP, f"{case.name} val: over {EVAL_CAP} spectra"
    if case.kind != "ce_context":  # CE-context batching reorders, the others keep dataset order
        assert flat == info.ids, f"{case.name} val: sequential order"
    if case.budget is not None:
        size = info.size(case.budget)
        oversized = [i for i in info.ids if size[i] > case.max_num]
        assert oversized, "fixture: validation must contain an oversized spectrum"
        for batch in batches:
            assert sum(size[i] for i in batch) <= case.max_num or (
                len(batch) == 1 and batch[0] in oversized
            ), f"{case.name} val: {batch} over budget and not a singleton"


def record_case(case: Case, data: F.SynthData, seed: int = SEED) -> dict[str, Any]:
    """Build, drive and check one case; return its golden record.

    Besides the per-epoch invariants this asserts that the six train epochs are not all the
    same order (unless ``case.frozen``), that a different run seed changes epochs 4 and 5, and
    that a resumed run (fresh dataset, dataloader and module) reproduces epochs N and N + 1.

    Args:
        case: The sampler configuration.
        data: Synthetic dataset to build it on.
        seed: Run seed (``hparams.seed``).

    Returns:
        A JSON-serializable record: sampler class names, per-spectrum sizes, one line per epoch,
        the validation batches and, for a curriculum, the eligible spectra of each epoch.
    """
    config = make_case_config(case, data, seed)
    other_seed = make_case_config(case, data, seed + 1)
    if case.curriculum is not None:
        prefix = {"frag_node": "dag", "bond": "bond"}[case.curriculum.key]
        assert config[f"{prefix}_size_curriculum"] is True
        assert config[f"{prefix}_size_curriculum_start_fraction"] == case.curriculum.start
        assert config[f"{prefix}_size_curriculum_epochs"] == case.curriculum.epochs
        assert len(case.curriculum.eligible) == len(EPOCHS)
    record: dict[str, Any] = {"samplers": f"train={case.train} val={case.val}"}
    with F.deterministic():
        with open_split(config, "train") as (train_ds, loader):
            assert sampler_names(loader) == case.train, "wrong train sampler selected"
            info = split_info(train_ds, data, case)
            row_ids = info.ids if not case.via_loader else None
            full = record_epochs(stub_module(config, loader, case.hook), EPOCHS, case.hook, row_ids)
            stats = getattr(loader.batch_sampler, "stats", None)
        eligible = [
            _check_train_epoch(case, info, loader, epoch, full[epoch]["batches"])
            for epoch in EPOCHS
        ]
        orders = {encode(full[e]["batches"]) for e in EPOCHS}
        if case.frozen:
            assert len(orders) == 1, f"{case.name}: expected the same order every epoch"
        else:
            assert len(orders) > 1, f"{case.name}: every epoch repeats one order"

        for start in RESUME_AT:
            with open_split(config, "train") as (_, resumed_loader):
                module = stub_module(config, resumed_loader, case.hook)
                resumed = record_epochs(module, (start, start + 1), case.hook, row_ids)
            for epoch in (start, start + 1):
                assert resumed[epoch] == full[epoch], f"{case.name}: resume at {start}"

        with open_split(other_seed, "train") as (_, loader_b):
            module = stub_module(other_seed, loader_b, case.hook)
            changed = record_epochs(module, (4, 5), case.hook, row_ids)
        if case.frozen or case.seed_independent:
            assert all(
                changed[e]["batches"] == full[e]["batches"] for e in (4, 5)
            ), f"{case.name}: the run seed unexpectedly reaches the sampler"
        else:
            assert any(
                changed[e]["batches"] != full[e]["batches"] for e in (4, 5)
            ), f"{case.name}: the run seed does not reach the sampler"

        with open_split(config, "val") as (val_ds, val_loader):
            assert sampler_names(val_loader) == case.val, "wrong val sampler selected"
            val_info = split_info(val_ds, data, case)
            val_batches = [[int(i) for i in batch["spec_unique_id"]] for batch in val_loader]
        _check_val(case, val_info, val_batches)

    if case.budget is not None:
        record[f"size_{case.budget}"] = " ".join(str(info.size(case.budget)[i]) for i in info.ids)
    record["num_batches"] = [full[e]["num_batches"] for e in EPOCHS]
    for epoch in EPOCHS:
        record[f"epoch_{epoch}"] = encode(full[epoch]["batches"])
    record["val"] = encode(val_batches)
    if case.curriculum is not None:
        record["eligible"] = eligible
    if case.kind == "ce_context":
        record["ce_stats_epoch_5"] = dict(stats)
    return record


# --------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------


def test_train_dl_seeds_follow_the_run_seed() -> None:
    """The hook's per-epoch generator seeds are draws seeded by ``hparams.seed``.

    ``_setup_sampler`` draws ``max_epochs + 1`` seeds; ``_seed_dataloader(epoch)`` indexes them.
    """
    seeds = {}
    for seed in (SEED, SEED + 1):
        module = types.SimpleNamespace(
            hparams=types.SimpleNamespace(
                seed=seed,
                max_epochs=8,
                train_batch_size=3,
                accumulate_grad_batches=2,
                automatic_optimization=False,
            )
        )
        SpectrumPL._setup_sampler(module)
        assert module._max_batch_size == 6
        assert tuple(module.train_dl_seeds.shape) == (9,)
        seeds[f"seed_{seed}"] = module.train_dl_seeds.tolist()
    assert len(set(seeds[f"seed_{SEED}"])) == 9
    assert set(seeds[f"seed_{SEED}"]).isdisjoint(seeds[f"seed_{SEED + 1}"])
    ratchet("train_dl_seeds", seeds)


@pytest.mark.parametrize("name", [n for n in CASES if n != DUAL_GROUP_CASE])
def test_sampler_batch_sequences_match_golden(name: str, records: RecordCache) -> None:
    """Batch sequences of six epochs, resume equality and invariants of one sampler."""
    ratchet(name, records[name])


def test_meta_info_json_changes_dynamic_batches(records: RecordCache) -> None:
    """Without ``meta_info.json`` the samplers see the loaded bigraph, with doubled edges."""
    for with_meta, without in (
        ("frag_dynamic_random", "frag_dynamic_random_no_meta_info"),
        ("frag_ce_context", "frag_ce_context_no_meta_info"),
    ):
        raw = [int(x) for x in records[with_meta]["size_frag_edge"].split()]
        loaded = [int(x) for x in records[without]["size_frag_edge"].split()]
        assert loaded == [2 * x for x in raw]
        assert [records[without][f"epoch_{e}"] for e in EPOCHS] != [
            records[with_meta][f"epoch_{e}"] for e in EPOCHS
        ]
        assert sum(records[without]["num_batches"]) > sum(records[with_meta]["num_batches"])


def test_frag_ce_context_coverage_buckets(records: RecordCache) -> None:
    """Coverage buckets of the CE-context sampler on the fixture's group structure."""
    stats = records["frag_ce_context"]["ce_stats_epoch_5"]
    assert stats["total_spectra"] == 20
    assert stats["mixed_mode_spectra"] == 2  # group 6 mixes HCD and CID
    assert stats["missing_ce_spectra"] == 0
    assert stats["single_level_spectra"] == 1  # group 4 has one CE
    assert stats["eligible_spectra"] == 17
    assert stats["oversized_spectra"] == 2  # both phenol spectra exceed the budget
    assert stats["selected_spectra"] == 18


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "known bug at the time of recording: DualGroupDynamicBatchSampler yields numpy ints (its "
        "group lists are arrays) and SpecMolFragDataset.__getitem__ asserts isinstance(idx, int), "
        "so dual_group_dynamic_batch_sampler cannot load a train batch. When the indices are "
        "cast to int this turns XPASS (strict): drop the marker, set via_loader=True on the case "
        "and regenerate the golden record deliberately"
    ),
)
def test_dual_group_batches_load_through_the_dataset(store: DataStore) -> None:
    """The first dual-group train batch loads through the real dataset and collate function."""
    case = CASES[DUAL_GROUP_CASE]
    config = make_case_config(case, store.for_case(case), SEED)
    with F.deterministic(), open_split(config, "train") as (_, loader):
        SpectrumPL._seed_dataloader(stub_module(config, loader, case.hook), 0)
        batch = next(iter(loader))
    assert batch["spec_unique_id"].numel() > 0


def test_dual_group_set_epoch_reseeds_from_the_epoch_alone(store: DataStore) -> None:
    """``DualGroupDynamicBatchSampler.set_epoch`` seeds its generator with the epoch number.

    No config that ``validate_sampler_config`` accepts makes the hook call it (the hook only
    recomputes the batches of this sampler), so the recorded sequences cannot see it. The source
    documents that the run seed does not reach it; both facts are pinned here, in one process
    (so one string-hash order), by comparing run seeds and epochs.
    """
    case = CASES[DUAL_GROUP_CASE]
    data = store.for_case(case)
    batches: dict[tuple[int, int], list[list[int]]] = {}
    with F.deterministic():
        for seed in (SEED, SEED + 1):
            with open_split(make_case_config(case, data, seed), "train") as (_, loader):
                sampler = loader.batch_sampler
                for epoch in (3, 4):
                    sampler.set_epoch(epoch)
                    assert sampler.generator.initial_seed() == epoch
                    batches[seed, epoch] = [[int(i) for i in batch] for batch in sampler]
    assert batches[SEED, 3] == batches[SEED + 1, 3], "the run seed must not reach set_epoch"
    assert batches[SEED, 4] == batches[SEED + 1, 4]
    assert batches[SEED, 3] != batches[SEED, 4], "the epoch must reach the generator"


def test_dual_group_record_with_sorted_keys_matches_golden(store: DataStore) -> None:
    """Dual-group batches with the group keys visited in sorted order, in this process.

    ``sorted_dual_group_keys`` removes the string-hash order, so this record is the same on every
    Python version and hash seed and guards the sampler's batching logic (key permutation, per-key
    and per-batch takes, budget, fill phase, ``return_batch_at``) wherever the pinned-hash-seed
    record below is skipped. It also checks every invariant and the resume equality.
    """
    case = CASES[DUAL_GROUP_CASE]
    with sorted_dual_group_keys():
        record = record_case(case, store.for_case(case))
    assert sum(record["num_batches"]) > 0
    ratchet(f"{DUAL_GROUP_CASE}:sorted_keys", record)


def test_dual_group_record_with_pinned_hash_seed(
    store: DataStore, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Dual-group batches, compared in an interpreter with ``PYTHONHASHSEED=0``.

    The sampler permutes a ``list`` of a ``set`` of strings, so its order changes with the hash
    seed. The in-process run (whatever hash seed pytest has) checks every invariant and the
    resume equality, so over many runs it covers many hash orders; only the golden comparison of
    the production order needs the pinned interpreter. On ``PINNED_PYTHON`` that golden is
    mandatory: a missing record fails instead of skipping. The hash-independent logic guard is
    ``test_dual_group_record_with_sorted_keys_matches_golden``.
    """
    case = CASES[DUAL_GROUP_CASE]
    in_process = record_case(case, store.for_case(case))
    assert in_process["samplers"] == f"train={case.train} val={case.val}"

    work = tmp_path_factory.mktemp("samplers_dual_child")
    out = work / "record.json"
    command = [sys.executable, str(Path(__file__).resolve()), str(work / "data"), str(out)]
    env = {**os.environ, "PYTHONHASHSEED": "0", "CUDA_VISIBLE_DEVICES": ""}
    done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    pinned = json.loads(out.read_text())
    # only the validation loader is hash independent; the train batches and their number are not
    assert pinned["num_batches"][0] > 0
    assert pinned["val"] == in_process["val"]

    version = f"py{sys.version_info.major}.{sys.version_info.minor}"
    key = f"{DUAL_GROUP_CASE}:{version}"
    pinned_python = (sys.version_info.major, sys.version_info.minor) == PINNED_PYTHON
    if not pinned_python and not UPDATE and key not in load_json(GOLDEN):
        pytest.skip(
            f"no dual-group record for Python {version}; the string hash differs by version "
            f"and only Python {PINNED_PYTHON[0]}.{PINNED_PYTHON[1]} is pinned, so the "
            "dual-group batch order is unguarded here"
        )
    ratchet(key, pinned)


@pytest.mark.parametrize("name", ["frag_dag_curriculum", "masster_bond_curriculum"])
def test_curriculum_table_separates_rounding_rules(name: str, store: DataStore) -> None:
    """The curriculum table makes floor, round and an off-by-one rank change the eligible set.

    The sampler takes the size at rank ``ceil(N * fraction)``; every spectrum at most that size
    is eligible. With ties at the rank, neighbouring rounding rules give the same set and the
    sequences cannot tell them apart. This recomputes the eligible counts of each alternative
    rule from the table's sizes and requires them to differ from the pinned counts.
    """
    case = CASES[name]
    curriculum = case.curriculum
    data = store.for_case(case)
    mols = [row[0] for row in TABLES[case.table].rows if row[3] == "train"]
    if curriculum.key == "frag_node":
        sizes = [int(data.dags[mol]["dag"].num_nodes) for mol in mols]
    else:
        sizes = [int(data.mol_df.loc[mol, "num_bonds"]) for mol in mols]
    ordered = sorted(sizes)
    num = len(ordered)

    def eligible_at(rank: int) -> int:
        limit = ordered[min(num, max(1, rank)) - 1]
        return sum(size <= limit for size in ordered)

    targets = [
        num * (curriculum.start + (1.0 - curriculum.start) * min(e / curriculum.epochs, 1.0))
        for e in EPOCHS
    ]
    assert [eligible_at(math.ceil(target)) for target in targets] == list(curriculum.eligible)
    # alternative rank rules and how many epochs each must change
    alternatives = {
        "floor": (math.floor, 2),
        "round": (round, 1),
        "ceil + 1": (lambda target: math.ceil(target) + 1, 1),
    }
    for rule, (rank_of, needed) in alternatives.items():
        counts = [eligible_at(rank_of(target)) for target in targets]
        differing = sum(a != b for a, b in zip(counts, curriculum.eligible))
        assert differing >= needed, f"{rule} gives {counts}, same as {curriculum.eligible}"
    assert len(set(curriculum.eligible)) >= 3, "the curriculum should grow over several epochs"


def _child_main(argv: Sequence[str]) -> None:
    """Record the dual-group case in this (hash-seed-pinned) interpreter and dump it as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    case = CASES[DUAL_GROUP_CASE]
    data = write_data(args.data_root, TABLES[case.table], meta_info=case.meta_info)
    args.out.write_text(json.dumps(record_case(case, data), sort_keys=True))


if __name__ == "__main__":
    _child_main(sys.argv[1:])
