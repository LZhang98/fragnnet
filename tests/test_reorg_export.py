"""Inference-export records for the repository reorganization (plan Phase 0 item 7).

For FraGNNet (``frag_gnn``) and MASSter (``masster``) this runs the three exporters of
``fragnnet.utils.script_utils`` on the shared synthetic fixture, with a freshly seeded tiny model
on CPU, and records what each one writes:

- ``run_inference_pkl``: the flat concatenated dict (keys, dtypes, shapes, digested values);
- ``run_inference_tabular`` / ``run_inference_tabular_iter``: the DataFrame (columns, dtypes,
  per-row shapes, digested values) and the column-major batches the iterator yields;
- ``run_inference_hdf5``: the dataset names, dtypes, shapes, chunking, compression and digested
  values, plus the returned spectrum count and the chunked-equals-single-flush property.

Two switches are recorded on top: the explainability switch (which ``return_explainability_tensors``
value each exporter hands the model, and which model outputs it adds) and rule provenance (MASSter
names the SMARTS rule behind each node; FraGNNet raises). The fixture carries one extra molecule,
N-phenylbenzenesulfonamide, because the rule record is vacuous unless a rule matches it.

Records are exact (see ``reorg_golden.ratchet_records``) and regenerated deliberately. The golden
holds behavior only: column names, dtypes, shapes and values, never class names or module paths,
which Phase 1 and Phase 6 change. The model outputs MASSter's pkl export carries (about 70 raw
keys) are recorded as they are: that dict is what the export hands out.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
import reorg_fixtures as F
import torch as th
from reorg_golden import ratchet_records

from fragnnet.utils import script_utils as su

MODEL_TYPES = ("frag_gnn", "masster")

# mol_id 6 matches MASSter rules aryl_sulfonamide_so2 and substituted_aryl_sulfonamide_o_transfer.
SULFONAMIDE = "O=S(=O)(Nc1ccccc1)c1ccccc1"
SULFONAMIDE_MOL_ID = 6
SMILES = (*F.SMILES, SULFONAMIDE)
SPECTRA = (*F.DEFAULT_SPECTRA, F.Spectrum(10, SULFONAMIDE_MOL_ID, 6, 30.0, "test"))
N_SPECTRA = len(SPECTRA)
EVAL_BATCH_SIZE = 4  # 11 spectra -> batches of 4, 4 and 3, so index rebasing is exercised
EXPECTED_BATCH_SIZES = [4, 4, 3]
SULFONAMIDE_RULES = {"aryl_sulfonamide_so2", "substituted_aryl_sulfonamide_o_transfer"}

PEAKS_ONLY = frozenset({"pred_mzs", "pred_ints"})
SCORE_COLUMNS = ["cos_sim_0.1", "cos_sim_0.01", "cos_hun"]
# Emitted column order ahead of the annotation and score columns (script_utils.COLUMN_SPECS).
BASE_COLUMNS = [
    "spec_id",
    "mol_id",
    "pred_mzs",
    "pred_ints",
    "pred_oos_prob",
    "pred_rt",
    "pred_ccs",
    "frag_formula_peak_idxs",
    "frag_formula_peak_mzs",
    "pred_formula_formula_idxs",
    "pred_formula_probs",
    "pred_joint_formula_idxs",
    "pred_formula_node_probs",
    "pred_joint_node_idxs",
    "pred_node_formula_probs",
    "pred_node_probs",
    "pred_node_node_idxs",
    "index_rebased",
]
# Columns that hold one value per spectrum in the HDF5 file; every other dataset is a flat
# ragged array sliced by a <prefix>start_points / <prefix>end_points pair.
SCALAR_DATASETS = {
    "spec_id",
    "mol_id",
    "pred_oos_prob",
    "pred_rt",
    "pred_ccs",
    "index_rebased",
    *SCORE_COLUMNS,
}
INDEX_DATASETS = {
    f"{prefix}{kind}_points"
    for prefix in ("", "node_", "formula_", "joint_")
    for kind in ("start", "end")
}
# Ragged datasets of the everything export, with the index-pair prefix that slices them.
RAGGED_PREFIX = {
    "pred_mzs": "",
    "pred_ints": "",
    "frag_formula_peak_idxs": "",
    "frag_formula_peak_mzs": "",
    "pred_node_probs": "node_",
    "pred_node_node_idxs": "node_",
    "pred_node_ccs": "node_",
    "pred_node_rules": "node_",
    "pred_formula_formula_idxs": "formula_",
    "pred_formula_probs": "formula_",
    "pred_formula_strs": "formula_",
    "pred_formula_adducts": "formula_",
    "pred_joint_formula_idxs": "joint_",
    "pred_formula_node_probs": "joint_",
    "pred_joint_node_idxs": "joint_",
    "pred_node_formula_probs": "joint_",
}
# Never su.setup_device: on CPU it sets the thread count and strict determinism process-wide.
DEVICE = th.device("cpu")


@dataclass
class ExportRun:
    """Everything one model type's exports produced: the golden record and the raw outputs."""

    model_type: str
    record: dict[str, Any]
    pkl: dict[str, dict[str, Any]] = field(default_factory=dict)
    tabular: dict[str, pd.DataFrame] = field(default_factory=dict)
    iter_batches: list[list[list]] = field(default_factory=list)
    h5_paths: dict[str, Path] = field(default_factory=dict)
    h5_counts: dict[str, int] = field(default_factory=dict)
    flags: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: dict[str, dict[str, Any]] = field(default_factory=dict)
    switch_effect: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------- digests
def _kind(value: Any) -> str:
    if th.is_tensor(value):
        return f"tensor:{value.dtype}"
    if isinstance(value, np.ndarray):
        return f"ndarray:{value.dtype}"
    return type(value).__name__


def _is_block(value: Any) -> bool:
    return th.is_tensor(value) or isinstance(value, (np.ndarray, list, tuple))


def _jsonable(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else str(value)


def _cell(value: Any) -> dict[str, Any]:
    """Kind, shape and a sha256 over the exact bytes of one exported value."""
    if th.is_tensor(value) or isinstance(value, np.ndarray):
        array = value.detach().cpu().numpy() if th.is_tensor(value) else value
        shape, raw = list(array.shape), np.ascontiguousarray(array).tobytes()
    elif isinstance(value, (list, tuple)):
        shape, raw = [len(value)], json.dumps(value, default=_jsonable).encode()
    else:
        shape, raw = [], json.dumps(value, default=_jsonable).encode()
    return {"kind": _kind(value), "shape": shape, "sha256": hashlib.sha256(raw).hexdigest()[:16]}


def _scalar(value: Any) -> Any:
    """A JSON-safe, exact spelling of one scalar (floats as hex, NaN as the string ``nan``)."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return "nan" if math.isnan(value) else float(value).hex()
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _column_record(cells: list[Any]) -> dict[str, Any]:
    """One column over all spectra: scalar columns keep their values, blocks keep shape + hash."""
    kinds = sorted({_kind(cell) for cell in cells})
    if not any(_is_block(cell) for cell in cells):
        return {"kinds": kinds, "values": [_scalar(cell) for cell in cells]}
    digests = [_cell(cell) for cell in cells]
    joined = "".join(f"{d['kind']}|{d['sha256']}" for d in digests).encode()
    return {
        "kinds": kinds,
        "shapes": [d["shape"] for d in digests],
        "sha256": hashlib.sha256(joined).hexdigest()[:16],
    }


def _frame_record(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_rows": len(frame),
        "columns": list(frame.columns),
        "df_dtypes": {col: str(frame[col].dtype) for col in frame.columns},
        "cells": {col: _column_record(frame[col].tolist()) for col in frame.columns},
    }


def _pkl_record(vals: dict[str, Any]) -> dict[str, Any]:
    """Sorted by key: the dict's key order follows a ``set`` of metric names (PYTHONHASHSEED).

    MASSter always emits ``candidate_*_ms`` wall-clock timings (masster_model.py). They are not
    reproducible, so only their names are recorded.
    """
    timing = sorted(key for key in vals if key.endswith("_ms"))
    return {
        "n_keys": len(vals) - len(timing),
        "timing_keys": timing,
        "values": {key: _cell(vals[key]) for key in sorted(vals) if key not in timing},
    }


def _h5_record(path: Path) -> dict[str, Any]:
    """Dataset names, dtypes, shapes, chunking, compression and digested values of one file."""
    datasets = {}
    with h5py.File(path, "r") as handle:
        root_attrs = {key: str(value) for key, value in sorted(handle.attrs.items())}
        for name in sorted(handle):
            dataset = handle[name]
            is_text = h5py.check_string_dtype(dataset.dtype) is not None
            values = dataset.asstr()[...] if is_text else dataset[...]
            raw = (
                json.dumps(values.tolist()).encode()
                if is_text
                else np.ascontiguousarray(values).tobytes()
            )
            record: dict[str, Any] = {
                "dtype": "vlen-str" if is_text else str(dataset.dtype),
                "shape": list(dataset.shape),
                "maxshape": list(dataset.maxshape),
                "compression": dataset.compression,
                "compression_opts": dataset.compression_opts,
                "shuffle": dataset.shuffle,
                # h5py picks uncompressed chunk shapes itself, which is its business.
                "chunks": (
                    list(dataset.chunks) if dataset.compression else dataset.chunks is not None
                ),
                "sha256": hashlib.sha256(raw).hexdigest()[:16],
            }
            if name in SCALAR_DATASETS or name in INDEX_DATASETS:
                record["values"] = [_scalar(v) for v in values.tolist()]
            datasets[name] = record
    return {"root_attrs": root_attrs, "datasets": datasets}


# ----------------------------------------------------------------------------- setup
def _inference_config(model_type: str, data: F.SynthData) -> dict:
    """The config ``scripts/run_inference.py`` ends up with, on the synthetic dataset."""
    overrides = {"masster_smarts_templates": True} if model_type == "masster" else {}
    config = F.make_config(model_type, data, overrides)
    su.apply_inference_config(
        config,
        auxiliary_scores=["cos_sim", "cos_hun"],
        disable_preproc=True,
        eval_batch_size=EVAL_BATCH_SIZE,
        drop_empty_spectra=False,
    )
    return config


@contextlib.contextmanager
def _inference_session(config: dict) -> Iterator[th.nn.Module]:
    """A fresh seeded model in eval mode under the plan's determinism settings, stdout silenced."""
    with F.deterministic(), contextlib.redirect_stdout(io.StringIO()):
        yield F.build_module(config).eval()


def _annotation(model_type: str) -> dict[str, bool]:
    """Annotation flags: formula strings and cc hashes, plus rule provenance for MASSter."""
    flags = {"return_formula_str": True, "return_cc_hash": True}
    if model_type == "masster":
        flags["return_rule_provenance"] = True
    return flags


def _annotation_columns(formula_str: bool, cc_hash: bool, rules: bool = False) -> list[str]:
    """Expected annotation columns, in emitted order (``pred_formula_adducts`` rides on either)."""
    columns = []
    if formula_str:
        columns.append("pred_formula_strs")
    if formula_str or cc_hash:
        columns.append("pred_formula_adducts")
    if cc_hash:
        columns.append("pred_node_ccs")
    if rules:
        columns.append("pred_node_rules")
    return columns


def _raises(call: Callable[[], Any], keep_message: bool = True) -> dict[str, Any]:
    try:
        call()
    except (KeyError, ValueError, RuntimeError, NotImplementedError) as err:
        record = {"raises": type(err).__name__}
        if keep_message:
            record["message"] = str(err)
        return record
    return {"raises": None}


# ----------------------------------------------------------------------------- records
def _idx_vars_record(model: th.nn.Module) -> dict[str, Any]:
    """Which index arrays each (output_set, annotation flags) pair asks the model for."""
    subsets: dict[str, set[str] | None] = {
        "all": None,
        "empty": set(),
        "peaks": set(PEAKS_ONLY),
        "node_probs": {"pred_node_probs"},
        "formula_probs": {"pred_formula_probs"},
        "joint_node_idxs": {"pred_joint_node_idxs"},
        "oos": {"pred_oos_prob"},
    }
    record = {}
    for name, subset in subsets.items():
        for formula_str in (False, True):
            for cc_hash in (False, True):
                batch, node, formula, _ = su.get_idx_vars(
                    subset, model, False, formula_str, cc_hash
                )
                record[f"{name}|formula_str={int(formula_str)}|cc_hash={int(cc_hash)}"] = {
                    "batch": batch,
                    "node": node,
                    "formula": formula,
                    # The expression run_inference_pkl and process_batch_to_tabular_row duplicate.
                    "needs_explainability": bool(
                        {"pred_formula_formula_idxs"} & set(formula)
                        or {"pred_node_node_idxs"} & set(node)
                    ),
                }
    return record


def _columns_record(model: th.nn.Module) -> dict[str, list[str]]:
    """The tabular header for every valid annotation-flag combination."""
    record = {}
    for formula_str in (False, True):
        for cc_hash in (False, True):
            for rules in (False, True):
                if rules and not cc_hash:
                    continue  # rejected, see the error records
                key = f"formula_str={int(formula_str)}|cc_hash={int(cc_hash)}|rules={int(rules)}"
                record[key] = su.get_tabular_columns(
                    model.hparams, formula_str, cc_hash, return_rule_provenance=rules
                )
    return record


def _spy_flags(model: th.nn.Module, calls: dict[str, Callable[[], Any]]) -> dict[str, Any]:
    """The ``return_explainability_tensors`` value each call hands ``inference_step``."""
    seen: list[str] = []
    real = model.inference_step

    def spy(batch: dict, **kwargs: Any) -> Any:
        seen.append(str(batch.get("return_explainability_tensors", "ABSENT")))
        return real(batch, **kwargs)

    record = {}
    model.inference_step = spy
    try:
        for label, call in calls.items():
            seen.clear()
            outcome = _raises(call, keep_message=False)
            record[label] = {"flag_values": sorted(set(seen)), "raises": outcome["raises"]}
    finally:
        del model.inference_step
    return record


def _switch_effect(model: th.nn.Module, loader: Any) -> dict[str, Any]:
    """Which model outputs ``return_explainability_tensors`` adds, on the first batch."""
    batch = next(iter(loader))
    keys = {}
    with th.inference_mode():
        for flag in (False, True):
            step_batch = {**copy.deepcopy(batch), "return_explainability_tensors": flag}
            output = model.inference_step(step_batch, metrics=None)
            keys[flag] = {k for k, v in output.items() if v is not None and not k.endswith("_ms")}
    return {
        "n_outputs_without": len(keys[False]),
        "n_outputs_with": len(keys[True]),
        "added_by_flag": sorted(keys[True] - keys[False]),
        "removed_by_flag": sorted(keys[False] - keys[True]),
    }


def _tabular_cases(model_type: str) -> dict[str, dict[str, Any]]:
    annotated = _annotation(model_type)
    return {
        "everything": {"output_set": None, "evaluate": True, **annotated},
        "peaks_only": {"output_set": set(PEAKS_ONLY)},
        "peaks_formula_str": {"output_set": set(PEAKS_ONLY), "return_formula_str": True},
        "peaks_cc_hash": {"output_set": set(PEAKS_ONLY), "return_cc_hash": True},
        "node_probs_only": {"output_set": {"pred_node_probs"}},
        "everything_min_peak_int_50": {"output_set": None, "min_peak_int": 50.0, **annotated},
        "everything_not_rebased": {"output_set": None, "rebase_indices": False, **annotated},
        "first_batch_only": {"output_set": set(PEAKS_ONLY), "batch_cutoff": 1},
    }


def _h5_cases(model_type: str) -> dict[str, dict[str, Any]]:
    annotated = _annotation(model_type)
    return {
        "everything": {"output_set": None, "evaluate": True, **annotated},
        "peaks_only": {"output_set": set(PEAKS_ONLY)},
        "peaks_only_keep_none_cols": {"output_set": set(PEAKS_ONLY), "drop_none_cols": False},
        "peaks_only_uncompressed": {
            "output_set": set(PEAKS_ONLY),
            "compression": None,
            "compression_opts": None,
        },
    }


def _pkl_cases(model_type: str) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {
        "everything": {"output_set": None, "evaluate": True},
        "everything_no_scores": {"output_set": None, "evaluate": False},
        "peaks": {"output_set": set(PEAKS_ONLY), "evaluate": True},
    }
    if model_type == "frag_gnn":
        # MASSter's pkl rejects probability tokens (see the error records).
        cases["node_probs"] = {"output_set": {"pred_node_probs"}, "evaluate": False}
    return cases


def _run_pkl(config: dict, loader: Any, **kwargs: Any) -> dict[str, Any]:
    with _inference_session(config) as model:
        return su.run_inference_pkl(model, loader, DEVICE, **kwargs)


def _run_tabular(config: dict, loader: Any, **kwargs: Any) -> pd.DataFrame:
    with _inference_session(config) as model:
        return su.run_inference_tabular(model, loader, DEVICE, **kwargs)


def _run_hdf5(config: dict, loader: Any, path: Path, **kwargs: Any) -> int:
    with _inference_session(config) as model:
        return su.run_inference_hdf5(model, loader, DEVICE, save_path=str(path), **kwargs)


def _record_exports(model_type: str, config: dict, loader: Any, h5_dir: Path) -> ExportRun:
    """Run every export once, each on a fresh seeded model, and collect the golden record."""
    h5_dir.mkdir(parents=True, exist_ok=True)
    run = ExportRun(model_type, record={"n_spectra": N_SPECTRA})
    record = run.record

    with _inference_session(config) as model:
        for key, columns in _columns_record(model).items():
            record[f"columns/{key}"] = columns
        for key, demand in _idx_vars_record(model).items():
            record[f"idx_vars/{key}"] = demand
        run.switch_effect = _switch_effect(model, loader)
    record["explainability_outputs"] = run.switch_effect

    for label, kwargs in _pkl_cases(model_type).items():
        run.pkl[label] = _run_pkl(config, loader, **kwargs)
        record[f"pkl/{label}"] = _pkl_record(run.pkl[label])

    for label, kwargs in _tabular_cases(model_type).items():
        run.tabular[label] = _run_tabular(config, loader, **kwargs)
        record[f"tabular/{label}"] = _frame_record(run.tabular[label])
    # A second, independent model and run must reproduce the first one exactly.
    again = _run_tabular(config, loader, **_tabular_cases(model_type)["everything"])
    record["tabular_repeatable"] = _frame_record(again) == record["tabular/everything"]

    with _inference_session(config) as model:
        kwargs = _tabular_cases(model_type)["everything"]
        run.iter_batches = list(su.run_inference_tabular_iter(model, loader, DEVICE, **kwargs))
    columns = su.get_tabular_columns(
        config,
        kwargs["return_formula_str"],
        kwargs["return_cc_hash"],
        None,
        kwargs.get("return_rule_provenance", False),
    )
    rebuilt: dict[str, list] = {col: [] for col in columns}
    for batch_rows in run.iter_batches:
        for col, values in zip(columns, batch_rows):
            rebuilt[col].extend(values)
    record["tabular_iter"] = {
        "n_batches": len(run.iter_batches),
        "n_columns_per_batch": [len(rows) for rows in run.iter_batches],
        "n_rows_per_batch": [len(rows[0]) for rows in run.iter_batches],
        "equals_tabular": _frame_record(pd.DataFrame(rebuilt)) == record["tabular/everything"],
    }

    for label, kwargs in _h5_cases(model_type).items():
        path = h5_dir / f"{label}.hdf5"
        run.h5_counts[label] = _run_hdf5(config, loader, path, **kwargs)
        run.h5_paths[label] = path
        record[f"hdf5/{label}"] = {
            "return_value": run.h5_counts[label],
            "return_type": type(run.h5_counts[label]).__name__,
            **_h5_record(path),
        }
    # chunk_size=3 flushes after every batch: three flushes must equal the single one.
    chunked = h5_dir / "everything_chunk3.hdf5"
    kwargs = {**_h5_cases(model_type)["everything"], "chunk_size": 3}
    run.h5_counts["everything_chunk3"] = _run_hdf5(config, loader, chunked, **kwargs)
    run.h5_paths["everything_chunk3"] = chunked
    record["hdf5_chunked_equals_single_flush"] = (
        _h5_record(chunked)["datasets"] == record["hdf5/everything"]["datasets"]
    )

    if model_type == "masster":
        # Natural isotope rows (nominal mass mode) export as peaks, but cannot be annotated.
        run.tabular["natural_isotope_peaks_only"] = _run_tabular(
            _natural_isotope_config(config), loader, output_set=set(PEAKS_ONLY)
        )
        record["tabular/natural_isotope_peaks_only"] = _frame_record(
            run.tabular["natural_isotope_peaks_only"]
        )

    run.flags = _record_flags(config, loader, h5_dir)
    for label, value in run.flags.items():
        record[f"explainability_flag/{label}"] = value
    run.errors = _record_errors(model_type, config, loader, h5_dir)
    for label, value in run.errors.items():
        record[f"errors/{label}"] = value
    return run


def _record_flags(config: dict, loader: Any, h5_dir: Path) -> dict[str, Any]:
    """The explainability switch as each exporter sets it, by output set and annotation flags."""
    peaks = set(PEAKS_ONLY)

    def pkl(model: th.nn.Module, **kw: Any) -> Any:
        return su.run_inference_pkl(model, loader, DEVICE, evaluate=False, **kw)

    def tab(model: th.nn.Module, **kw: Any) -> Any:
        return su.run_inference_tabular(model, loader, DEVICE, **kw)

    def h5(model: th.nn.Module, **kw: Any) -> Any:
        return su.run_inference_hdf5(
            model, loader, DEVICE, save_path=str(h5_dir / "flag.hdf5"), **kw
        )

    with _inference_session(config) as model:
        calls: dict[str, Callable[[], Any]] = {
            "pkl|peaks": lambda: pkl(model, output_set=peaks),
            "pkl|node_probs": lambda: pkl(model, output_set={"pred_node_probs"}),
            "pkl|formula_probs": lambda: pkl(model, output_set={"pred_formula_probs"}),
            "pkl|all": lambda: pkl(model, output_set=None),
            "tabular|peaks": lambda: tab(model, output_set=peaks),
            "tabular|peaks+formula_str": lambda: tab(
                model, output_set=peaks, return_formula_str=True
            ),
            "tabular|peaks+cc_hash": lambda: tab(model, output_set=peaks, return_cc_hash=True),
            "tabular|node_probs": lambda: tab(model, output_set={"pred_node_probs"}),
            "tabular|all": lambda: tab(model, output_set=None),
            "hdf5|peaks": lambda: h5(model, output_set=peaks),
            "hdf5|all": lambda: h5(model, output_set=None),
        }
        return _spy_flags(model, calls)


def _natural_isotope_config(config: dict) -> dict:
    """MASSter with an isotope envelope, which needs the nominal mass mode."""
    return {**config, "masster_mass_mode": "nominal", "masster_isotope_envelope": "natural"}


def _record_errors(model_type: str, config: dict, loader: Any, h5_dir: Path) -> dict[str, Any]:
    """The documented refusals: rule provenance on FraGNNet, tokens MASSter's pkl cannot serve."""
    rules_without_hash = {"return_formula_str": True, "return_rule_provenance": True}
    rules_with_hash = {"return_cc_hash": True, "return_rule_provenance": True}
    cases: dict[str, tuple[Callable[..., Any], dict[str, Any]]] = {
        "tabular_rules_without_cc_hash": (su.run_inference_tabular, rules_without_hash),
        "tabular_rules_with_cc_hash": (su.run_inference_tabular, rules_with_hash),
        "hdf5_rules_without_cc_hash": (su.run_inference_hdf5, rules_without_hash),
        "hdf5_rules_with_cc_hash": (su.run_inference_hdf5, rules_with_hash),
        "pkl_node_probs": (su.run_inference_pkl, {"output_set": {"pred_node_probs"}}),
        "pkl_formula_probs": (su.run_inference_pkl, {"output_set": {"pred_formula_probs"}}),
        "pkl_oos_prob": (su.run_inference_pkl, {"output_set": {"pred_oos_prob"}}),
    }
    record = {}
    for label, (exporter, kwargs) in cases.items():
        if exporter is su.run_inference_hdf5:
            kwargs = {**kwargs, "save_path": str(h5_dir / "error.hdf5")}
        with _inference_session(config) as model:
            record[label] = _raises(lambda: exporter(model, loader, DEVICE, **kwargs))
    for label, exporter in {
        "tabular_model_in_training_mode": su.run_inference_tabular,
        "pkl_model_in_training_mode": su.run_inference_pkl,
    }.items():
        with _inference_session(config) as model:
            model.train()
            # The message embeds the module's repr, so only the exception type is recorded.
            record[label] = _raises(lambda: exporter(model, loader, DEVICE), keep_message=False)
    if model_type == "masster":
        isotope_cases = {
            "natural_isotope_with_annotation": _annotation(model_type),
            "natural_isotope_with_formula_str": {"return_formula_str": True},
        }
        for label, kwargs in isotope_cases.items():
            with _inference_session(_natural_isotope_config(config)) as model:
                record[label] = _raises(
                    lambda: su.run_inference_tabular(model, loader, DEVICE, **kwargs)
                )
    return record


def _generate(model_type: str, workdir: Path) -> ExportRun:
    dag_backend = "h5" if model_type == "frag_gnn" else None
    data = F.write_synth_data(
        workdir / "data", spectra=SPECTRA, smiles=SMILES, dag_backend=dag_backend
    )
    config = _inference_config(model_type, data)
    with F.deterministic():
        dataset, loader = F.init_split(config, "predict_only")
    try:
        assert len(dataset) == N_SPECTRA
        return _record_exports(model_type, config, loader, workdir / "h5")
    finally:
        F.close_dataset(dataset)


@pytest.fixture(scope="module", params=MODEL_TYPES)
def export_run(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> ExportRun:
    """All exports of one model type, built once per module."""
    return _generate(request.param, tmp_path_factory.mktemp(f"export_{request.param}"))


# ----------------------------------------------------------------------------- helpers
def _rows(frame: pd.DataFrame, column: str) -> list[Any]:
    return frame[column].tolist()


def _peak_counts(frame: pd.DataFrame) -> list[int]:
    return [len(mzs) for mzs in _rows(frame, "pred_mzs")]


def _h5_slices(path: Path, name: str, prefix: str) -> list[np.ndarray]:
    """Per-spectrum slices of a ragged HDF5 dataset, through its index pair."""
    with h5py.File(path, "r") as handle:
        starts, ends = handle[f"{prefix}start_points"][...], handle[f"{prefix}end_points"][...]
        dataset = handle[name]
        values = dataset.asstr()[...] if h5py.check_string_dtype(dataset.dtype) else dataset[...]
    return [values[start:end] for start, end in zip(starts, ends)]


def _expected_h5_names(model_type: str, everything: bool = True) -> set[str]:
    """Dataset names the everything export writes (peaks-only export: ``everything=False``)."""
    if not everything:
        names = {"spec_id", "mol_id", "index_rebased", "start_points", "end_points"}
        names |= {"pred_mzs", "pred_ints"}
        if model_type == "frag_gnn":  # written whatever the output set (see script_utils)
            names |= {"frag_formula_peak_idxs", "frag_formula_peak_mzs"}
        return names
    # pred_rt / pred_ccs are all-NaN placeholders that drop_none_cols removes.
    names = (set(SCALAR_DATASETS) | INDEX_DATASETS | set(RAGGED_PREFIX)) - {"pred_rt", "pred_ccs"}
    if model_type == "frag_gnn":
        return names - {"pred_node_rules"}
    # MASSter writes no P(out of spectrum) and has no per-peak formula m/z column.
    return names - {"pred_oos_prob", "frag_formula_peak_mzs"}


# ----------------------------------------------------------------------------- tests
def test_export_records_match_golden(export_run: ExportRun) -> None:
    """Every export's schema and digested values equal the stored record, bit for bit."""
    ratchet_records("export", export_run.model_type, export_run.record)


def test_tabular_header_rows_and_dtypes(export_run: ExportRun) -> None:
    """One row per spectrum, in the COLUMN_SPECS order, with the documented dtypes."""
    masster = export_run.model_type == "masster"
    annotated = _annotation_columns(True, True, rules=masster)
    expected = {
        "everything": BASE_COLUMNS + annotated + SCORE_COLUMNS,
        "peaks_only": BASE_COLUMNS + SCORE_COLUMNS,
        "peaks_formula_str": BASE_COLUMNS + _annotation_columns(True, False) + SCORE_COLUMNS,
        "peaks_cc_hash": BASE_COLUMNS + _annotation_columns(False, True) + SCORE_COLUMNS,
        "node_probs_only": BASE_COLUMNS + SCORE_COLUMNS,
        "everything_min_peak_int_50": BASE_COLUMNS + annotated + SCORE_COLUMNS,
        "everything_not_rebased": BASE_COLUMNS + annotated + SCORE_COLUMNS,
        "first_batch_only": BASE_COLUMNS + SCORE_COLUMNS,
        "natural_isotope_peaks_only": BASE_COLUMNS + SCORE_COLUMNS,
    }
    spec_ids = [spec.spec_id for spec in SPECTRA]
    mol_ids = [spec.mol_id for spec in SPECTRA]
    for label, frame in export_run.tabular.items():
        assert list(frame.columns) == expected[label], label
        n_rows = EVAL_BATCH_SIZE if label == "first_batch_only" else N_SPECTRA
        assert len(frame) == n_rows, label
        assert _rows(frame, "spec_id") == spec_ids[:n_rows], label
        assert _rows(frame, "mol_id") == mol_ids[:n_rows], label
        assert frame["spec_id"].dtype == np.int64 and frame["mol_id"].dtype == np.int64
        assert frame["index_rebased"].dtype == bool
        assert frame["pred_rt"].dtype == np.float64 and frame["pred_ccs"].dtype == np.float64
        assert frame["pred_rt"].isna().all() and frame["pred_ccs"].isna().all()
        assert (frame["index_rebased"] == (label != "everything_not_rebased")).all(), label
    # The header is the same one get_tabular_columns reports, and the header table is recorded.
    record = export_run.record
    assert record["columns/formula_str=1|cc_hash=1|rules=0"] == (
        BASE_COLUMNS + _annotation_columns(True, True) + SCORE_COLUMNS
    )
    assert record["columns/formula_str=0|cc_hash=0|rules=0"] == BASE_COLUMNS + SCORE_COLUMNS
    assert record["columns/formula_str=0|cc_hash=1|rules=1"] == (
        BASE_COLUMNS + _annotation_columns(False, True, rules=True) + SCORE_COLUMNS
    )


def test_tabular_peak_and_score_values(export_run: ExportRun) -> None:
    """Peaks are per-spectrum tensors on the 1000 base-peak scale; scores exist only when scored."""
    masster = export_run.model_type == "masster"
    frame = export_run.tabular["everything"]
    mz_dtype = th.float64 if masster else th.float32
    for mzs, ints in zip(_rows(frame, "pred_mzs"), _rows(frame, "pred_ints")):
        assert mzs.dtype == mz_dtype and ints.dtype == th.float32
        assert mzs.ndim == 1 and mzs.shape == ints.shape and mzs.numel() > 0
        assert bool((mzs > 0).all()) and bool(th.isfinite(ints).all()) and bool((ints > 0).all())
        assert ints.max().item() == pytest.approx(1000.0)
    if masster:  # MASSter has no P(out of spectrum): a NaN placeholder per spectrum
        assert frame["pred_oos_prob"].dtype == np.float64 and frame["pred_oos_prob"].isna().all()
    else:
        assert frame["pred_oos_prob"].dtype == np.float32 and frame["pred_oos_prob"].notna().all()
    for column in SCORE_COLUMNS:
        scores = frame[column]
        assert scores.dtype == np.float64 and scores.between(0.0, 1.0).all(), column
    # Without evaluate there is nothing to score against: the score columns are NaN placeholders.
    unscored = export_run.tabular["peaks_only"]
    assert all(unscored[column].isna().all() for column in SCORE_COLUMNS)

    # min_peak_int filters peaks, not spectra: every row survives, intensities stay above it.
    filtered = export_run.tabular["everything_min_peak_int_50"]
    assert len(filtered) == N_SPECTRA
    assert all(ints.min().item() >= 50.0 for ints in _rows(filtered, "pred_ints"))
    full, kept = _peak_counts(frame), _peak_counts(filtered)
    assert all(k <= f for k, f in zip(kept, full)) and sum(kept) < sum(full)

    first = export_run.tabular["first_batch_only"]
    assert _peak_counts(first) == full[:EVAL_BATCH_SIZE]


def test_tabular_annotation_columns(export_run: ExportRun) -> None:
    """Formula strings, adducts, node cc hashes and rules are aligned with their index arrays."""
    masster = export_run.model_type == "masster"
    frame = export_run.tabular["everything"]
    for row in range(N_SPECTRA):
        n_formulas = len(frame["pred_formula_formula_idxs"][row])
        n_nodes = len(frame["pred_node_node_idxs"][row])
        strs, adducts = frame["pred_formula_strs"][row], frame["pred_formula_adducts"][row]
        assert isinstance(strs, list) and all(isinstance(s, str) for s in strs)
        assert isinstance(adducts, list) and all(isinstance(a, str) for a in adducts)
        assert len(strs) == len(adducts) == n_formulas == len(frame["pred_formula_probs"][row])
        ccs = frame["pred_node_ccs"][row]
        assert ccs.dtype == th.int64 and tuple(ccs.shape) == (n_nodes, 2)
        assert len(frame["pred_node_probs"][row]) == n_nodes
        joint = len(frame["pred_joint_node_idxs"][row])
        assert joint == len(frame["pred_joint_formula_idxs"][row])
        assert joint == len(frame["pred_formula_node_probs"][row])
        if masster:
            rules = frame["pred_node_rules"][row]
            assert isinstance(rules, list) and len(rules) == n_nodes
    # Probabilities of one spectrum's nodes are positive and individually at most one.
    assert all(bool((p > 0).all() and (p <= 1).all()) for p in _rows(frame, "pred_node_probs"))
    # Not rebased: only the flag column differs for the first spectrum of the first batch.
    unrebased = export_run.tabular["everything_not_rebased"]
    assert th.equal(unrebased["pred_node_node_idxs"][0], frame["pred_node_node_idxs"][0])


def test_explainability_off_leaves_marginal_columns_empty(export_run: ExportRun) -> None:
    """Without annotation or probability tokens the marginal columns are NaN placeholders."""
    marginals = [
        "pred_formula_formula_idxs",
        "pred_formula_probs",
        "pred_joint_formula_idxs",
        "pred_formula_node_probs",
        "pred_joint_node_idxs",
        "pred_node_formula_probs",
        "pred_node_probs",
        "pred_node_node_idxs",
    ]
    peaks_only = export_run.tabular["peaks_only"]
    for column in marginals:
        assert peaks_only[column].dtype == np.float64 and peaks_only[column].isna().all(), column
    annotation = {"pred_formula_strs", "pred_formula_adducts", "pred_node_ccs", "pred_node_rules"}
    assert annotation.isdisjoint(peaks_only.columns)

    # A P(n) token turns the node marginals on and leaves the formula marginals off.
    node_probs = export_run.tabular["node_probs_only"]
    assert all(th.is_tensor(p) for p in _rows(node_probs, "pred_node_probs"))
    assert all(th.is_tensor(i) for i in _rows(node_probs, "pred_node_node_idxs"))
    assert node_probs["pred_formula_probs"].isna().all()

    # An annotation flag alone also needs the marginals: node cc hashes come with node indices.
    with_hash = export_run.tabular["peaks_cc_hash"]
    assert all(th.is_tensor(c) and c.shape[1] == 2 for c in _rows(with_hash, "pred_node_ccs"))
    with_strs = export_run.tabular["peaks_formula_str"]
    assert all(isinstance(s, list) for s in _rows(with_strs, "pred_formula_strs"))


def test_explainability_switch_per_exporter(export_run: ExportRun) -> None:
    """Which return_explainability_tensors value each exporter hands the model."""
    flags = {label: (v["flag_values"], v["raises"]) for label, v in export_run.flags.items()}
    masster = export_run.model_type == "masster"
    tabular_and_hdf5 = {
        "tabular|peaks": (["False"], None),
        "tabular|peaks+formula_str": (["True"], None),
        "tabular|peaks+cc_hash": (["True"], None),
        "tabular|node_probs": (["True"], None),
        "tabular|all": (["True"], None),
        "hdf5|peaks": (["False"], None),
        "hdf5|all": (["True"], None),
    }
    if masster:
        # run_inference_pkl only sets the key for FraGNNet; MASSter's model reads it as False.
        pkl = {
            "pkl|peaks": (["ABSENT"], None),
            "pkl|node_probs": (["ABSENT"], "KeyError"),
            "pkl|formula_probs": (["ABSENT"], "KeyError"),
            "pkl|all": (["ABSENT"], None),
        }
    else:
        pkl = {
            "pkl|peaks": (["False"], None),
            "pkl|node_probs": (["True"], None),
            "pkl|formula_probs": (["True"], None),
            "pkl|all": (["True"], None),
        }
    assert flags == {**pkl, **tabular_and_hdf5}


def test_explainability_flag_adds_model_outputs(export_run: ExportRun) -> None:
    """The switch only adds outputs; FraGNNet adds the marginals, MASSter the pathway tables."""
    effect = export_run.switch_effect
    assert effect["removed_by_flag"] == []
    assert effect["n_outputs_with"] == effect["n_outputs_without"] + len(effect["added_by_flag"])
    if export_run.model_type == "frag_gnn":
        assert (effect["n_outputs_without"], effect["n_outputs_with"]) == (8, 23)
        assert {"pred_node_logprobs", "pred_formula_logprobs", "pred_node_node_idxs"} <= set(
            effect["added_by_flag"]
        )
    else:
        assert len(effect["added_by_flag"]) == 7
        assert {"fragment_atom_indices", "pred_pathway_fragment_idxs"} <= set(
            effect["added_by_flag"]
        )


def test_pkl_schema(export_run: ExportRun) -> None:
    """The pkl export is one flat dict, spectra identified by unique_id, indices rebatched."""
    masster = export_run.model_type == "masster"
    everything, no_scores = export_run.pkl["everything"], export_run.pkl["everything_no_scores"]
    peaks = export_run.pkl["peaks"]
    assert set(peaks) == {"pred_mzs", "pred_ints", "pred_batch_idxs"}
    scored_only = {"unique_id", "smiles", "true_mzs", "true_logprobs", "true_batch_idxs"}
    assert scored_only | {"true_prec_mzs", *SCORE_COLUMNS} <= set(everything)
    assert not ((scored_only | set(SCORE_COLUMNS)) & set(no_scores))
    assert {"pred_mzs", "pred_ints", "pred_batch_idxs"} <= set(no_scores)
    assert ("pred_oos_logprobs" in no_scores) == (not masster)  # MASSter has no P(out of spectrum)
    # Flat per-peak arrays share one length; spectrum indices run 0..n-1 across the 3 batches.
    n_peaks = peaks["pred_mzs"].shape[0]
    assert peaks["pred_ints"].shape == peaks["pred_batch_idxs"].shape == (n_peaks,)
    assert peaks["pred_batch_idxs"].dtype == th.int64
    assert sorted(set(peaks["pred_batch_idxs"].tolist())) == list(range(N_SPECTRA))
    assert peaks["pred_mzs"].dtype == (th.float64 if masster else th.float32)
    assert everything["unique_id"].tolist() == [spec.spec_id for spec in SPECTRA]
    canonical = F.build_mol_df(SMILES)["smiles"].tolist()
    assert everything["smiles"] == [canonical[spec.mol_id] for spec in SPECTRA]
    for column in SCORE_COLUMNS:
        assert tuple(everything[column].shape) == (N_SPECTRA,)
        tabular_scores = th.tensor(export_run.tabular["everything"][column].to_numpy())
        assert th.equal(everything[column].double(), tabular_scores), column
    assert everything["true_batch_idxs"].max().item() == N_SPECTRA - 1

    tabular_counts = _peak_counts(export_run.tabular["everything"])
    if masster:
        # MASSter's pkl is candidate-level: one row per fragment candidate, many per peak.
        assert n_peaks > sum(tabular_counts)
        assert export_run.pkl["everything"]["pred_mzs"].shape[0] == n_peaks
        assert export_run.record["pkl/everything"]["timing_keys"], "wall-clock keys are emitted"
    else:
        # FraGNNet's pkl and tabular export carry the same peaks, spectrum by spectrum.
        assert n_peaks == sum(tabular_counts)
        tabular = export_run.tabular["everything"]
        for index in range(N_SPECTRA):
            selected = peaks["pred_batch_idxs"] == index
            assert th.equal(peaks["pred_mzs"][selected], tabular["pred_mzs"][index])
            assert th.equal(peaks["pred_ints"][selected], tabular["pred_ints"][index])
        node_probs = export_run.pkl["node_probs"]
        assert set(node_probs) == {"pred_node_probs", "pred_node_node_idxs", "pred_node_batch_idxs"}
        n_nodes = sum(len(nodes) for nodes in _rows(tabular, "pred_node_probs"))
        assert node_probs["pred_node_probs"].shape == (n_nodes,)
        assert node_probs["pred_node_node_idxs"].shape == (n_nodes,)
        batch_idxs = node_probs["pred_node_batch_idxs"]
        assert batch_idxs.max().item() == N_SPECTRA - 1 and bool((batch_idxs.diff() >= 0).all())


def test_pkl_rejects_probability_tokens_for_masster(export_run: ExportRun) -> None:
    """MASSter's model output has no P(f)/P(n)/P(oos) log-probabilities, so those tokens fail."""
    for label in ("pkl_node_probs", "pkl_formula_probs", "pkl_oos_prob"):
        error = export_run.errors[label]
        if export_run.model_type == "masster":
            assert error["raises"] == "KeyError", label
            assert "which is absent/None" in error["message"], label
        else:
            assert error["raises"] is None, label


def test_hdf5_schema(export_run: ExportRun) -> None:
    """Dataset names, dtypes, shapes and filters of the HDF5 export; the return value is a count."""
    masster = export_run.model_type == "masster"
    everything = export_run.record["hdf5/everything"]
    assert everything["return_type"] == "int" and everything["return_value"] == N_SPECTRA
    assert set(everything["datasets"]) == _expected_h5_names(export_run.model_type)
    datasets = everything["datasets"]
    n_peaks = sum(_peak_counts(export_run.tabular["everything"]))
    for name, record in datasets.items():
        assert record["maxshape"][0] is None, name
        if name in SCALAR_DATASETS or name in INDEX_DATASETS:
            assert record["shape"] == [N_SPECTRA], name
        else:
            assert record["shape"][0] > 0, name
    assert datasets["pred_mzs"]["shape"] == [n_peaks]
    assert datasets["pred_mzs"]["dtype"] == ("float64" if masster else "float32")
    assert datasets["pred_ints"]["dtype"] == "float32"
    assert datasets["spec_id"]["dtype"] == datasets["mol_id"]["dtype"] == "int64"
    assert datasets["index_rebased"]["dtype"] == "bool"
    assert datasets["pred_node_ccs"]["dtype"] == "int64"
    assert datasets["pred_node_ccs"]["shape"][1:] == [2]
    assert datasets["pred_node_ccs"]["chunks"] == [65536, 2]
    for name in ("pred_formula_strs", "pred_formula_adducts"):
        assert datasets[name]["dtype"] == "vlen-str" and datasets[name]["shuffle"] is False
    if masster:
        assert datasets["pred_node_rules"]["dtype"] == "vlen-str"
        assert datasets["pred_node_rules"]["shape"] == datasets["pred_node_probs"]["shape"]
    assert datasets["pred_mzs"]["shuffle"] is True
    for name, record in datasets.items():  # gzip level 4, 65536-row chunks, on every dataset
        assert record["compression"] == "gzip" and record["compression_opts"] == 4, name
        assert record["chunks"][0] == 65536, name

    # start/end points tile the flat arrays without gaps.
    for prefix in ("", "node_", "formula_", "joint_"):
        starts = datasets[f"{prefix}start_points"]["values"]
        ends = datasets[f"{prefix}end_points"]["values"]
        assert starts[0] == 0 and starts[1:] == ends[:-1], prefix
        family = [n for n, p in RAGGED_PREFIX.items() if p == prefix and n in datasets]
        assert family and all(datasets[n]["shape"][0] == ends[-1] for n in family), prefix


def test_hdf5_matches_tabular(export_run: ExportRun) -> None:
    """The HDF5 slices reproduce the tabular rows exactly, spectrum by spectrum."""
    frame = export_run.tabular["everything"]
    path = export_run.h5_paths["everything"]
    for name in ("pred_mzs", "pred_ints", "pred_node_probs", "pred_formula_probs"):
        slices = _h5_slices(path, name, RAGGED_PREFIX[name])
        for row, values in enumerate(slices):
            assert np.array_equal(values, frame[name][row].numpy()), (name, row)
    for name in ("pred_formula_strs", "pred_formula_adducts"):
        slices = _h5_slices(path, name, "formula_")
        assert [list(values) for values in slices] == _rows(frame, name), name
    node_ccs = _h5_slices(path, "pred_node_ccs", "node_")
    assert all(np.array_equal(v, frame["pred_node_ccs"][i].numpy()) for i, v in enumerate(node_ccs))
    with h5py.File(path, "r") as handle:
        assert handle["spec_id"][...].tolist() == _rows(frame, "spec_id")
        assert handle["mol_id"][...].tolist() == _rows(frame, "mol_id")
        for column in SCORE_COLUMNS:
            assert np.array_equal(handle[column][...], frame[column].to_numpy()), column
    # chunk_size=3 flushes after every batch; one flush at the end must give the same file.
    assert export_run.record["hdf5_chunked_equals_single_flush"] is True
    assert export_run.h5_counts["everything_chunk3"] == N_SPECTRA


def test_hdf5_drops_placeholder_columns(export_run: ExportRun) -> None:
    """drop_none_cols removes all-NaN scalar columns; keeping them stores NaN float64 columns."""
    record = export_run.record
    peaks_only = record["hdf5/peaks_only"]["datasets"]
    assert set(peaks_only) == _expected_h5_names(export_run.model_type, everything=False)
    kept = record["hdf5/peaks_only_keep_none_cols"]["datasets"]
    placeholders = {"pred_rt", "pred_ccs", "pred_oos_prob", *SCORE_COLUMNS}
    assert placeholders <= set(kept)
    for name in placeholders:
        assert kept[name]["dtype"] == "float64" and kept[name]["shape"] == [N_SPECTRA]
        assert set(kept[name]["values"]) == {"nan"}
    # Everything the peaks-only export wrote is also in the keep-none file, with equal values.
    assert all(kept[name] == peaks_only[name] for name in peaks_only)
    uncompressed = record["hdf5/peaks_only_uncompressed"]["datasets"]
    assert set(uncompressed) == set(peaks_only)
    assert all(d["compression"] is None and d["chunks"] is True for d in uncompressed.values())
    for name in peaks_only:
        assert uncompressed[name]["sha256"] == peaks_only[name]["sha256"], name
    for label in ("peaks_only", "peaks_only_keep_none_cols", "peaks_only_uncompressed"):
        assert record[f"hdf5/{label}"]["return_value"] == N_SPECTRA


def test_tabular_iter_yields_column_major_batches(export_run: ExportRun) -> None:
    """The iterator yields one column-major batch per DataLoader batch."""
    rules = int(export_run.model_type == "masster")
    columns = export_run.record[f"columns/formula_str=1|cc_hash=1|rules={rules}"]
    assert len(export_run.iter_batches) == len(EXPECTED_BATCH_SIZES)
    for batch_rows, size in zip(export_run.iter_batches, EXPECTED_BATCH_SIZES):
        assert len(batch_rows) == len(columns)
        assert all(len(values) == size for values in batch_rows)
    assert export_run.record["tabular_iter"] == {
        "n_batches": 3,
        "n_columns_per_batch": [len(columns)] * 3,
        "n_rows_per_batch": EXPECTED_BATCH_SIZES,
        "equals_tabular": True,
    }


def test_rule_provenance(export_run: ExportRun) -> None:
    """MASSter names the SMARTS rule behind each node; FraGNNet and a missing cc hash raise."""
    errors = export_run.errors
    for exporter in ("tabular", "hdf5"):
        error = errors[f"{exporter}_rules_without_cc_hash"]
        assert error["raises"] == "ValueError" and "set return_cc_hash" in error["message"]
    if export_run.model_type == "frag_gnn":
        for exporter in ("tabular", "hdf5"):
            error = errors[f"{exporter}_rules_with_cc_hash"]
            assert error["raises"] == "ValueError"
            assert error["message"] == "return_rule_provenance is a MASSter SMARTS annotation"
        assert "pred_node_rules" not in export_run.tabular["everything"].columns
        return
    assert errors["tabular_rules_with_cc_hash"]["raises"] is None
    assert errors["hdf5_rules_with_cc_hash"]["raises"] is None
    frame = export_run.tabular["everything"]
    position = list(frame.columns).index("pred_node_rules")
    assert position == list(frame.columns).index("pred_node_ccs") + 1
    sulfonamide_row = [spec.mol_id for spec in SPECTRA].index(SULFONAMIDE_MOL_ID)
    for row, rules in enumerate(_rows(frame, "pred_node_rules")):
        names = {name for name in rules if name}
        if row == sulfonamide_row:
            assert names == SULFONAMIDE_RULES
        else:
            assert names == set(), row  # no other fixture molecule matches a MASSter rule
    # The same strings reach the HDF5 file, one per node.
    slices = _h5_slices(export_run.h5_paths["everything"], "pred_node_rules", "node_")
    assert [list(values) for values in slices] == _rows(frame, "pred_node_rules")
    assert {name for values in slices for name in values} >= SULFONAMIDE_RULES
    # Without the flag the column is absent from every export.
    assert "pred_node_rules" not in export_run.tabular["peaks_cc_hash"].columns


def test_masster_isotope_rows_cannot_be_annotated(export_run: ExportRun) -> None:
    """Isotope rows export as peaks, but annotating them is refused (they have no formula)."""
    if export_run.model_type != "masster":
        assert not [key for key in export_run.errors if key.startswith("natural_isotope")]
        return
    for label in ("natural_isotope_with_annotation", "natural_isotope_with_formula_str"):
        error = export_run.errors[label]
        assert error["raises"] == "NotImplementedError", label
        assert "isotope" in error["message"] and "export without explanations" in error["message"]
    isotope = export_run.tabular["natural_isotope_peaks_only"]
    monoisotopic = export_run.tabular["peaks_only"]
    assert len(isotope) == N_SPECTRA
    assert list(isotope.columns) == list(monoisotopic.columns)
    # The nominal mass mode puts every peak on an integer m/z; the exact mode does not.
    assert all(bool((mzs == mzs.round()).all()) for mzs in _rows(isotope, "pred_mzs"))
    assert not any(bool((mzs == mzs.round()).all()) for mzs in _rows(monoisotopic, "pred_mzs"))


def test_exports_refuse_training_mode(export_run: ExportRun) -> None:
    """Exports raise instead of silently running a model with dropout on."""
    assert export_run.errors["tabular_model_in_training_mode"] == {"raises": "RuntimeError"}
    assert export_run.errors["pkl_model_in_training_mode"] == {"raises": "RuntimeError"}


def test_exports_are_repeatable_in_process(export_run: ExportRun) -> None:
    """A second fresh model gives the same tabular export, bit for bit."""
    assert export_run.record["tabular_repeatable"] is True
