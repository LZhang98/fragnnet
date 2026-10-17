"""Frozen-token allocation probe: a local cut scorer plus a penalized context correction.

Stage 1 of docs/plans/masster/masster_local_context_allocation_plan.md, on fixed MASSter
representations. For one checkpoint (a "producer") the eval-mode forward is run once per
manifest split and cached: every chemically valid candidate fragment with its final
contextualized intensity-transformer token, the checkpoint's allocation logit, its valid ion/H
channels and their fixed log-probabilities, retained/lost boundary bonds, and the targets.
Offline arms then refit the scalar allocation head and/or a new local scorer on those caches:

    s_f = l_f + c_f,  c_f = w . t_f + b  (the producer's mask_residual_head, refit)
    l_f = sum over boundary bonds b of g(radius-2 environment of b, acquisition)
    P(f | x) = softmax over valid f of s_f; channel log-probabilities stay fixed
    loss = production sparse Gaussian CE + lambda * mean_x mean_f (c_f - mean_f c_f)^2

Subcommands:

- ``manifests``: fit / tune / confirm (and untouched) split directories and novelty.
- ``extract``: one cached eval-mode forward per producer and split, with alignment checks.
- ``fit``: every fitted arm in lockstep on identical batches; stopping, penalty strength,
  calibration and the control choice are selected on tuning molecules only, then locked.
- ``report``: the single locked read on confirmation molecules.

Selection never reads the confirmation split; ``report`` refuses to run without the lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch as th
import torch.nn.functional as F
from rdkit import DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.nn import GINEConv

from fragnnet.inference.checkpoint import (
    apply_inference_config,
    get_config_from_ckpt,
    init_model_from_ckpt,
)
from fragnnet.loss.cross_entropy import get_sparse_cross_entropy_fn
from fragnnet.models.masster.masster_component_pooling import boundary_bond_endpoint_orientation
from fragnnet.models.masster.masster_pathway_probabilities import masked_ion_hydrogen_log_probabilities
from fragnnet.runner import load_config
from fragnnet.utils.setup_utils import get_dataset_cls
from fragnnet.utils.spec_similarity import cos_sim_helper
from fragnnet.utils.spec_utils import batched_bin_func

try:
    from .debug_masster_base_peak_errors import (
        _identity,
        descending_ranks,
        map_targets,
        merge_channels,
    )
    from .debug_masster_base_peak_tables import clustered_bootstrap, pair_relation
except ImportError:  # executed as a script
    from debug_masster_base_peak_errors import (
        _identity,
        descending_ranks,
        map_targets,
        merge_channels,
    )
    from debug_masster_base_peak_tables import clustered_bootstrap, pair_relation

LOGGER = logging.getLogger(__name__)

# Raw-feature whitelist of the local encoder: column slices of mol_pyg.x / chemical_bond_attr
# for pyg_node_feats [a_onehot, a_degree, a_hybrid, a_formal, a_radical, a_ring, a_ring_size,
# a_num_hs, a_mass, a_chiral, a_charge] and pyg_edge_feats [b_degree, b_ring, b_conj].
# a_ring (in ring, aromatic) and b_ring are INHERITED precursor-wide RDKit perception.
# Excluded: a_degree, a_hybrid, a_radical, a_ring_size, a_mass, a_chiral, a_charge (Gasteiger,
# whole molecule) and b_conj (conjugated system).
EXPECTED_NODE_FEATS = [
    "a_onehot",
    "a_degree",
    "a_hybrid",
    "a_formal",
    "a_radical",
    "a_ring",
    "a_ring_size",
    "a_num_hs",
    "a_mass",
    "a_chiral",
    "a_charge",
]
EXPECTED_EDGE_FEATS = ["b_degree", "b_ring", "b_conj"]
ATOM_COLUMNS = np.r_[0:18, 38:44, 59:64, 50:54]  # element, formal charge, H count, ring/arom
BOND_COLUMNS = np.r_[0:5, 5:7]  # bond type (incl. aromatic), bond in ring
NODE_WIDTH, EDGE_WIDTH = 71, 9

NCE_EDGES = (20.0, 40.0, 60.0)  # bins <=20, (20,40], (40,60], >60, as bond_local_learner
NUM_NCE_BINS = 4
NOVELTY_EDGES = (0.4, 0.6)
MAX_CUT_ONEHOT = 6
MIN_SECOND = 0.05  # measured #2 >= 0.05 x #1, as the base-peak tables
NUM_H_SHIFTS = 9
SPLITS = ("fit", "tune", "confirm", "untouched")


# ---------------------------------------------------------------------------------------------
# Manifests and novelty
# ---------------------------------------------------------------------------------------------


def build_manifests(local_dp: Path, canonical_dp: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """Fit / tune / confirm / untouched manifests for the local and full-data producers.

    The local producers were trained on ``local_dp/train_ids.csv`` and selected on
    ``local_dp/val_ids.csv``; the full-data producer on the canonical train split, selected on
    the canonical val split. Every CE row of a molecule stays in one manifest.

    Args:
        local_dp: The local-screen split directory (train/val/test_ids.csv).
        canonical_dp: The canonical ``nist20_inchikey`` split directory.

    Returns:
        ``{"local": {...}, "full": {...}}``, each mapping split name to its id table.

    Raises:
        ValueError: If a disjointness or containment check fails.
    """
    local = {s: pd.read_csv(local_dp / f"{s}_ids.csv") for s in ("train", "val")}
    canon = {s: pd.read_csv(canonical_dp / f"{s}_ids.csv") for s in ("train", "val", "test")}
    mols = {
        k: set(v.mol_id)
        for k, v in {
            **{f"l_{s}": d for s, d in local.items()},
            **{f"c_{s}": d for s, d in canon.items()},
        }.items()
    }
    if not mols["l_train"] <= mols["c_train"]:
        raise ValueError("local train molecules are not all canonical train molecules")
    if not mols["l_val"] <= mols["c_val"]:
        raise ValueError("local val molecules are not all canonical val molecules")
    if len({*mols["c_train"]} & mols["c_val"]) or len(mols["c_train"] & mols["c_test"]):
        raise ValueError("canonical splits share molecules")
    manifests = {
        "local": {
            "fit": local["train"],
            "tune": canon["train"][~canon["train"].mol_id.isin(mols["l_train"])],
            "confirm": local["val"],
            "untouched": canon["test"],
        },
        "full": {
            "fit": local["train"],
            "tune": canon["val"][~canon["val"].mol_id.isin(mols["l_val"])],
            "confirm": local["val"],
            "untouched": canon["test"],
        },
    }
    for name, splits in manifests.items():
        sets = {s: set(d.mol_id) for s, d in splits.items()}
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1 :]:
                if sets[a] & sets[b]:
                    raise ValueError(f"{name}: {a} and {b} share molecules")
    return {n: {s: d.reset_index(drop=True) for s, d in m.items()} for n, m in manifests.items()}


def novelty_table(
    mol_df: pd.DataFrame, query: set[int], references: dict[str, set[int]]
) -> pd.DataFrame:
    """Nearest-reference Morgan r2 (2048-bit) Tanimoto for every query molecule.

    A query molecule that is itself in a reference set is compared against the others only.

    Args:
        mol_df: ``mol_id`` and RDKit ``mol`` columns.
        query: Molecules to score.
        references: Name -> reference molecule ids.

    Returns:
        One row per query molecule: ``mol_id`` and ``nov_<name>`` per reference.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = mol_df.set_index("mol_id")["mol"]
    needed = sorted(query | set().union(*references.values()))
    fps = {m: generator.GetFingerprint(mols.loc[m]) for m in needed}
    out = pd.DataFrame({"mol_id": sorted(query)})
    for name, ref in references.items():
        ref_ids = sorted(ref)
        ref_fps = [fps[m] for m in ref_ids]
        position = {m: i for i, m in enumerate(ref_ids)}
        best = []
        for m in out.mol_id:
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[m], ref_fps))
            if m in position:
                sims[position[m]] = -1.0
            best.append(float(sims.max()))
        out[f"nov_{name}"] = best
    return out


def run_manifests(args: argparse.Namespace) -> None:
    """Write the manifest split directories and the novelty table."""
    out_dp = Path(args.output_dp)
    manifests = build_manifests(Path(args.local_split_dp), Path(args.canonical_split_dp))
    summary = {}
    for name, splits in manifests.items():
        (out_dp / "splits" / name).mkdir(parents=True, exist_ok=True)
        for split, frame in splits.items():
            frame.to_csv(out_dp / "splits" / name / f"{split}_ids.csv", index=False)
            summary[f"{name}/{split}"] = {
                "molecules": int(frame.mol_id.nunique()),
                "spectra": len(frame),
            }
    local_train = set(manifests["local"]["fit"].mol_id)
    canon_train = set(pd.read_csv(Path(args.canonical_split_dp) / "train_ids.csv").mol_id)
    query = set().union(*(set(d.mol_id) for m in manifests.values() for d in m.values()))
    mol_df = pd.read_pickle(args.mol_fp)
    table = novelty_table(mol_df, query, {"local": local_train, "full": canon_train})
    table.to_csv(out_dp / "novelty.csv", index=False)
    (out_dp / "manifests.json").write_text(json.dumps(summary, indent=2))
    LOGGER.info("manifests: %s", summary)


# ---------------------------------------------------------------------------------------------
# Segment helpers
# ---------------------------------------------------------------------------------------------


def segment_logsumexp(values: th.Tensor, segments: th.Tensor, num: int) -> th.Tensor:
    """Per-segment log-sum-exp.

    Args:
        values: Shape ``(N,)``.
        segments: Segment id of each value, shape ``(N,)``, in ``[0, num)``.
        num: Number of segments.

    Returns:
        Shape ``(num,)``; ``-inf`` for empty segments.
    """
    maxima = values.new_full((num,), -th.inf).scatter_reduce(
        0, segments, values, reduce="amax", include_self=True
    )
    safe = th.where(th.isfinite(maxima), maxima, th.zeros_like(maxima))
    sums = values.new_zeros(num).index_add(0, segments, (values - safe[segments]).exp())
    return safe + sums.log()


def segment_mean(values: th.Tensor, segments: th.Tensor, num: int) -> th.Tensor:
    """Per-segment mean, shape ``(num,)``; zero for empty segments."""
    sums = values.new_zeros(num).index_add(0, segments, values)
    counts = values.new_zeros(num).index_add(0, segments, th.ones_like(values))
    return sums / counts.clamp_min(1.0)


def context_penalty(context: th.Tensor, segments: th.Tensor, num: int) -> th.Tensor:
    """``R(x) = mean_f (c_f - mean_f c_f)^2`` per spectrum, shape ``(num,)``.

    Invariant to a per-spectrum offset of ``context``, which cancels in the softmax.
    """
    centered = context - segment_mean(context, segments, num)[segments]
    return segment_mean(centered.square(), segments, num)


def binned_cosine(
    channel_logprobs: th.Tensor,
    channel_bins: th.Tensor,
    target_bins: th.Tensor,
    bin_segments: th.Tensor,
    num: int,
) -> th.Tensor:
    """Production ``cos_sim_0.01`` from per-channel compact bin ids (differentiable).

    Args:
        channel_logprobs: Log-probability of each output channel, shape ``(C,)``.
        channel_bins: Global compact bin id of each channel, shape ``(C,)``; ``-1`` for
            channels at or above ``mz_max`` (dropped, as the production metric does).
        target_bins: Binned target intensity per compact bin, shape ``(Q,)``.
        bin_segments: Spectrum of each compact bin, shape ``(Q,)``.
        num: Number of spectra.

    Returns:
        Cosine per spectrum, shape ``(num,)``; 0 where a side has no binned mass.
    """
    keep = channel_bins >= 0
    pred = target_bins.new_zeros(target_bins.shape[0]).index_add(
        0, channel_bins[keep], channel_logprobs[keep].exp().to(target_bins.dtype)
    )
    dot = pred.new_zeros(num).index_add(0, bin_segments, pred * target_bins)
    pred_norm = pred.new_zeros(num).index_add(0, bin_segments, pred.square()).sqrt()
    tgt_norm = pred.new_zeros(num).index_add(0, bin_segments, target_bins.square()).sqrt()
    denom = pred_norm * tgt_norm
    return th.where(denom > 0, dot / denom.clamp_min(1e-30), th.zeros_like(dot))


# ---------------------------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------------------------


class ForwardCapture:
    """Capture the pre-filter unique-fragment tensors of one MassterModel forward.

    Installs instance-attribute wrappers on ``_unique_component_memberships`` and
    ``_fragment_masses_and_valid_channels`` and hooks on ``mask_residual_head`` (input token and
    output logit) and ``ion_hydrogen_head``. No source is edited; ``close`` restores everything.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.records: dict[str, list] = {}
        self._shadowed = {
            name: name in vars(model)
            for name in ("_unique_component_memberships", "_fragment_masses_and_valid_channels")
        }
        self._originals = {
            name: getattr(model, name)
            for name in ("_unique_component_memberships", "_fragment_masses_and_valid_channels")
        }
        for name, original in self._originals.items():
            setattr(model, name, self._wrap(name, original))
        head = model.intensity_head
        self._handles = [
            head.mask_residual_head.register_forward_hook(self._residual_hook),
            head.ion_hydrogen_head.register_forward_hook(self._ion_h_hook),
        ]

    def _wrap(self, name: str, original: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            self.records.setdefault(name, []).append(tuple(t.detach() for t in result))
            return result

        return wrapped

    def _residual_hook(self, _module: nn.Module, inputs: tuple, output: th.Tensor) -> None:
        self.records.setdefault("tokens", []).append(inputs[0].detach())
        self.records.setdefault("residuals", []).append(output.detach().squeeze(1))

    def _ion_h_hook(self, _module: nn.Module, _inputs: tuple, output: th.Tensor) -> None:
        self.records.setdefault("ion_h", []).append(output.detach())

    def reset(self) -> None:
        self.records = {}

    def take(self) -> dict[str, Any]:
        """Return the single captured call of each record; raise unless each fired once."""
        names = (
            "_unique_component_memberships",
            "_fragment_masses_and_valid_channels",
            "tokens",
            "residuals",
            "ion_h",
        )
        for name in names:
            if len(self.records.get(name, [])) != 1:
                raise RuntimeError(f"{name} fired {len(self.records.get(name, []))} times")
        return {name: self.records[name][0] for name in names}

    def close(self) -> None:
        for name, original in self._originals.items():
            if self._shadowed[name]:
                setattr(self.model, name, original)
            else:
                delattr(self.model, name)
        for handle in self._handles:
            handle.remove()


def rebuild_prediction(got: dict[str, Any], size: int, mz_dtype: th.dtype) -> dict[str, th.Tensor]:
    """Rebuild the model's flat prediction from one :class:`ForwardCapture` record.

    In unique-fragment mode with ``proposal_selection_only``, every fragment's allocation logit
    is ``mask_residual_head(token)``; the model keeps the fragments with at least one valid
    ion/H channel, normalizes the logits over them per spectrum and adds each fragment's
    masked ion/H log-softmax. Channels are flattened row-major over ``(fragment, h_shift)``.

    Args:
        got: The captured tensors (pre-validity-filter rows).
        size: Number of spectra in the batch.
        mz_dtype: The dtype of the model's ``pred_mzs``.

    Returns:
        ``valid`` (F,) mask; per valid fragment ``v_spec``, ``v_s``, ``v_channels`` (Fv, H)
        and ``chan_lp_dense`` (Fv, H); per output channel ``path_lp``, ``path_mz``,
        ``path_frag`` (valid-fragment index) and ``path_h`` (h-shift column).

    Raises:
        RuntimeError: If the captured tensors disagree in rows or grouping.
    """
    frag_spec, _atom_idx, _atom_frag, _bnd_bond, _bnd_frag, cut_counts = got[
        "_unique_component_memberships"
    ]
    ion_mzs, valid_channels = got["_fragment_masses_and_valid_channels"]
    tokens, residuals = got["tokens"], got["residuals"]
    num_frag = tokens.shape[0]
    for name, t in (
        ("frag_spec", frag_spec),
        ("cut_counts", cut_counts),
        ("ion_mzs", ion_mzs),
        ("residuals", residuals),
        ("valid_channels", valid_channels),
    ):
        if t.shape[0] != num_frag:
            raise RuntimeError(f"{name} has {t.shape[0]} rows, tokens {num_frag}")
    if valid_channels.shape[1] != 1:
        raise RuntimeError("expected a single ion mode (no CMF)")
    if th.any(frag_spec[1:] < frag_spec[:-1]):
        raise RuntimeError("fragments are not grouped by spectrum")
    num_h = valid_channels.shape[2]
    ion_h = got["ion_h"].reshape(num_frag, 1, num_h)
    valid = valid_channels.flatten(1).any(1)
    v_spec = frag_spec[valid]
    v_s = residuals[valid]
    frag_lp = v_s - segment_logsumexp(v_s, v_spec, size)[v_spec]
    v_channels = valid_channels[valid].flatten(1)
    chan_lp_dense = masked_ion_hydrogen_log_probabilities(
        ion_h[valid], valid_channels[valid]
    ).flatten(1)
    device = v_s.device
    path_frag = th.arange(v_s.numel(), device=device)[:, None].expand_as(v_channels)
    path_h = th.arange(num_h, device=device)[None, :].expand_as(v_channels)
    return {
        "valid": valid,
        "v_spec": v_spec,
        "v_s": v_s,
        "v_channels": v_channels,
        "chan_lp_dense": chan_lp_dense,
        "path_lp": (frag_lp[:, None] + chan_lp_dense)[v_channels],
        "path_mz": ion_mzs[valid].flatten(1)[v_channels].to(mz_dtype),
        "path_frag": path_frag[v_channels],
        "path_h": path_h[v_channels],
    }


class Appender:
    """Append arrays to raw binary files; ``close`` writes their dtype/shape metadata."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.files: dict[str, Any] = {}
        self.meta: dict[str, dict] = {}

    def add(self, name: str, array: np.ndarray) -> None:
        array = np.ascontiguousarray(array)
        if name not in self.files:
            self.files[name] = open(self.directory / f"{name}.bin", "wb")  # noqa: SIM115
            self.meta[name] = {"dtype": array.dtype.str, "tail": list(array.shape[1:]), "rows": 0}
        elif list(array.shape[1:]) != self.meta[name]["tail"]:
            raise ValueError(f"{name}: inconsistent trailing shape")
        self.files[name].write(array.tobytes())
        self.meta[name]["rows"] += int(array.shape[0])

    def close(self) -> None:
        for handle in self.files.values():
            handle.close()
        (self.directory / "arrays.json").write_text(json.dumps(self.meta, indent=2))


def load_arrays(directory: Path, names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Memory-map the arrays written by :class:`Appender`."""
    meta = json.loads((directory / "arrays.json").read_text())
    out = {}
    for name, m in meta.items():
        if names is not None and name not in names:
            continue
        shape = (m["rows"], *m["tail"])
        if m["rows"] == 0:
            out[name] = np.zeros(shape, dtype=np.dtype(m["dtype"]))
        else:
            out[name] = np.memmap(
                directory / f"{name}.bin", dtype=np.dtype(m["dtype"]), mode="r", shape=shape
            )
    return out


def merge_config(base: dict, own: dict) -> tuple[dict, dict]:
    """Overlay ``own`` on ``base`` recursively; report the keys only ``base`` holds.

    Args:
        base: Template-derived config (every current schema key).
        own: The checkpoint's hparams, which win wherever they hold a key.

    Returns:
        The merged config and ``{dotted key: value}`` for keys filled from ``base``.
    """
    filled: dict[str, Any] = {}

    def merge(b: dict, o: dict, prefix: str) -> dict:
        out = dict(b)
        for k, v in o.items():
            nested = isinstance(v, dict) and isinstance(b.get(k), dict)
            out[k] = merge(b[k], v, f"{prefix}{k}.") if nested else v
        for k in b:
            if k not in o:
                filled[f"{prefix}{k}"] = b[k]
        return out

    return merge(base, own, ""), filled


def producer_config(checkpoint: str, template_fp: str, custom_fp: str) -> tuple[dict, dict]:
    """The checkpoint's own hparams, completed by the template for keys it predates.

    Checkpoint hparams are the configuration (they win for every key they hold). Keys added
    to the schema after the checkpoint was trained are absent from its hparams; those come
    from ``template_fp`` + ``custom_fp`` (default-off by the schema's convention) and are
    returned so the caller can record them.

    Returns:
        The merged config and ``{key: filled value}`` for keys the checkpoint lacked.
    """
    return merge_config(load_config(template_fp, custom_fp, []), get_config_from_ckpt(checkpoint))


def _check_config(config: dict, model: nn.Module) -> dict:
    """Assert the settings the frozen-token reconstruction relies on; return a summary."""
    required = {
        "model_type": "masster",
        "masster_unique_fragment_mode": True,
        "masster_proposal_selection_only": True,
        "masster_proposal_anchored": True,
        "masster_min_fragment_relative_probability": 0.0,
        "masster_fragment_selection": "mask_order",
        "masster_min_mask_relative_probability": 0.0,
        "masster_intensity_output": "probability",
        "masster_include_cmf": False,
        "masster_min_ion_mode_relative_probability": 0.0,
        "masster_min_ion_h_relative_probability": 0.0,
        "masster_min_ion_h_absolute_probability": 0.0,
        "masster_share_proposal_across_ev": False,
        "loss_type": "cross_entropy",
        "eval_target_apply_train_rank_filters": False,
        "train_target_top_k_peaks": -1,
        "sum_ints": True,
        "ints_transform": "none",
    }
    for key, value in required.items():
        if config[key] != value:
            raise ValueError(f"{key} = {config[key]!r}, expected {value!r}")
    if config["mol_params"]["pyg_node_feats"] != EXPECTED_NODE_FEATS:
        raise ValueError(f"unexpected pyg_node_feats {config['mol_params']['pyg_node_feats']}")
    if config["mol_params"]["pyg_edge_feats"] != EXPECTED_EDGE_FEATS:
        raise ValueError(f"unexpected pyg_edge_feats {config['mol_params']['pyg_edge_feats']}")
    inner = model.model
    if inner.evaluation_ion_h_top_k != 0 or inner.evaluation_joint_channel_relative_threshold:
        raise ValueError("evaluation channel selection is active")
    if model.binned_loss:
        raise ValueError("binned loss is on; the pred train preprocessing is not the identity")
    if inner.intensity_head.mask_residual_head.in_features != 256:
        raise ValueError("unexpected intensity token width")
    return {k: config[k] for k in required}


def _compact_bins(
    keys_pred: th.Tensor, keys_true: th.Tensor, num_bins: int
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """Union the occupied (spectrum, bin) keys of both sides into compact ids.

    Returns:
        Compact id of each pred key and each true key, the spectrum of each compact id and
        the 0.01-Da bin index of each compact id.
    """
    union, inverse = th.unique(th.cat([keys_pred, keys_true]), return_inverse=True)
    return (
        inverse[: keys_pred.numel()],
        inverse[keys_pred.numel() :],
        union // num_bins,
        union % num_bins,
    )


def extract(args: argparse.Namespace) -> None:
    """Run the eval-mode forward once over one manifest split and cache it."""
    th.manual_seed(args.seed)
    np.random.seed(args.seed)  # noqa: NPY002 - seed legacy dependencies as well
    random.seed(args.seed)
    out_dp = Path(args.output_dp)
    out_dp.mkdir(parents=True, exist_ok=True)
    if (out_dp / "arrays.json").exists():
        raise FileExistsError(f"{out_dp} already holds a cache")
    started = time.time()

    # 1. Model and dataset with the checkpoint's own configuration.
    ckpt = th.load(args.checkpoint, map_location="cpu", weights_only=False)
    config, filled = producer_config(args.checkpoint, args.template_fp, args.custom_fp)
    config["split_dp"] = args.split_dp
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
    checked = _check_config(config, model)
    dataset_cls = get_dataset_cls(config["model_type"])
    dataset = dataset_cls(split=args.split, **config)
    specs = dataset.spec_df.reset_index(drop=True)
    if args.max_spectra:
        dataset = th.utils.data.Subset(dataset, list(range(min(args.max_spectra, len(specs)))))
        specs = specs.iloc[: len(dataset)]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset_cls.get_collate_fn(),
        num_workers=0,
    )
    # The exact kwargs of MassterPL._setup_loss_fn's primary CE, stored for the offline fits.
    sce_kwargs = {
        "dist": config["output_distribution"],
        "vectorized": config["loss_vectorized"],
        "tolerance": model.tolerance,
        "relative": model.relative,
        "tolerance_min_mz": model.tolerance_min_mz,
        "oos_tolerance_multiple": 1.0,
        "gaussian_renormalize": config["gaussian_renormalize"],
        "pm_tolerance_multiple": config["pm_tolerance_multiple"],
        "loss_batch_size": config["loss_batch_size"],
    }
    sce_fn = get_sparse_cross_entropy_fn(**sce_kwargs)
    mz_max, bin_res = float(config["mz_max"]), 0.01
    num_bins = int(round(mz_max / bin_res))
    # spec_inst_type indexes the sorted instrument list (base_dataset, inst_tol_lookup).
    inst_types = sorted(config["spec_params"]["inst_types"])
    if len(inst_types) != 2:
        raise ValueError("the acquisition encoder expects exactly two instrument types")
    tolerance = {
        i: (config["inst_type_loss_tol"][n]["rel"], config["inst_type_loss_tol"][n]["min_mz"])
        for i, n in enumerate(inst_types)
    }
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
        "inputs": [_identity(p) for p in (args.checkpoint, config["spec_fp"], config["mol_fp"])]
        + [_identity(str(Path(args.split_dp) / f"{args.split}_ids.csv"))],
        "checkpoint_epoch": int(ckpt["epoch"]),
        "checkpoint_global_step": int(ckpt["global_step"]),
        "checked_config": checked,
        "filled_from_template": filled,
        "sce_kwargs": sce_kwargs,
        "mz_max": mz_max,
        "spectra_in_manifest": int(
            pd.read_csv(Path(args.split_dp) / f"{args.split}_ids.csv").shape[0]
        ),
        "spectra_after_dataset_filters": len(specs),
        "tolerance": tolerance,
        "status": "running",
    }
    (out_dp / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    del ckpt

    capture = ForwardCapture(model.model)
    arrays = Appender(out_dp)
    spectrum_rows: list[dict] = []
    molecules: dict[int, dict] = {}
    worst = {"logprob": 0.0, "residual": 0.0, "token_fp16_logit": 0.0, "cos": 0.0, "ce": 0.0}
    cursor = frag_total = chan_total = bin_total = cpeak_total = 0
    try:
        with th.inference_mode():
            for batch in loader:
                size = int(batch["batch_size"])
                rows = specs.iloc[cursor : cursor + size]
                if not np.array_equal(batch["spec_unique_id"].numpy(), rows.spec_id.to_numpy()):
                    raise RuntimeError("batch order does not match the spectrum table")
                batch = model.transfer_batch_to_device(batch, device, dataloader_idx=0)
                capture.reset()
                out = model(**batch, return_explainability_tensors=True)
                if "pred_pathway_scores" in out:  # merged rows: use the per-pathway rows
                    out = {
                        **out,
                        "pred_mzs": out["pred_pathway_mzs"],
                        "pred_ints": out["pred_pathway_scores"],
                        "pred_logprobs": out["pred_pathway_scores"].log(),
                        "pred_batch_idxs": out["pred_pathway_batch_idxs"],
                    }
                got = capture.take()

                # 2-3. Pre-filter fragments, the validity filter and the rebuilt prediction.
                weight = model.model.intensity_head.mask_residual_head.weight[0]
                bias = model.model.intensity_head.mask_residual_head.bias[0]
                rebuilt = rebuild_prediction(got, size, out["pred_mzs"].dtype)
                (frag_spec, atom_idx, atom_frag, bnd_bond, bnd_frag, cut_counts) = got[
                    "_unique_component_memberships"
                ]
                tokens, residuals = got["tokens"], got["residuals"]
                valid, v_spec, v_s = rebuilt["valid"], rebuilt["v_spec"], rebuilt["v_s"]
                v_channels, chan_lp_dense = rebuilt["v_channels"], rebuilt["chan_lp_dense"]
                path_lp, path_mz = rebuilt["path_lp"], rebuilt["path_mz"]
                path_frag = rebuilt["path_frag"]
                if int(valid.sum()) != out["ion_h_fragment_batch_idxs"].numel():
                    raise RuntimeError("validity filter does not match the model output")
                worst["residual"] = max(
                    worst["residual"],
                    float((tokens @ weight + bias - residuals).abs().max()),
                )
                worst["token_fp16_logit"] = max(
                    worst["token_fp16_logit"],
                    float((tokens.half().float() @ weight + bias - residuals).abs().max()),
                )
                if not th.equal(path_mz, out["pred_mzs"]):
                    raise RuntimeError("rebuilt channel m/z differ from the model output")
                if not th.equal(v_spec[path_frag], out["pred_batch_idxs"]):
                    raise RuntimeError("rebuilt channel spectra differ from the model output")
                worst["logprob"] = max(
                    worst["logprob"], float((path_lp - out["pred_logprobs"]).abs().max())
                )
                if worst["logprob"] > args.logprob_atol:
                    raise RuntimeError(f"rebuilt log-probabilities differ by {worst['logprob']}")

                # 4. Targets: the training and eval conventions must agree for these configs.
                train_mz, train_lp, train_idx = model.preproc_spec(
                    batch["spec_mzs"],
                    batch["spec_ints"],
                    batch["spec_batch_idxs"],
                    train=True,
                    pred=False,
                    log_in=False,
                    log_out=True,
                    batch_size=size,
                )
                eval_mz, eval_prob, eval_idx = model.preproc_spec(
                    batch["spec_mzs"],
                    batch["spec_ints"],
                    batch["spec_batch_idxs"],
                    train=False,
                    pred=False,
                    batch_size=size,
                )
                if not (th.equal(train_mz, eval_mz) and th.equal(train_idx, eval_idx)):
                    raise RuntimeError("training and eval target peaks differ")
                if not th.allclose(train_lp.exp(), eval_prob, rtol=1e-5, atol=1e-8):
                    raise RuntimeError("training and eval target intensities differ")
                inst_idx = batch["spec_inst_type"].reshape(-1)
                tol_s = model.inst_tol_lookup[inst_idx].to(train_lp.dtype)
                min_mz_s = model.inst_min_mz_lookup[inst_idx].to(train_lp.dtype)

                # 5. The model's own CE and cos_sim_0.01, and our compact-bin cosine.
                ce_model = sce_fn(
                    train_mz,
                    train_lp,
                    train_idx,
                    out["pred_mzs"],
                    out["pred_logprobs"],
                    out["pred_batch_idxs"],
                    out["pred_logprobs"].new_zeros(size),
                    tol_per_sample=tol_s,
                    min_mz_per_sample=min_mz_s,
                )[0]
                ce_rebuilt = sce_fn(
                    train_mz,
                    train_lp,
                    train_idx,
                    path_mz,
                    path_lp,
                    v_spec[path_frag],
                    path_lp.new_zeros(size),
                    tol_per_sample=tol_s,
                    min_mz_per_sample=min_mz_s,
                )[0]
                worst["ce"] = max(worst["ce"], float((ce_model - ce_rebuilt).abs().max()))
                pm, pi, pidx = model.preproc_spec(
                    out["pred_mzs"],
                    out["pred_logprobs"],
                    out["pred_batch_idxs"],
                    train=False,
                    pred=True,
                    log_in=True,
                    log_out=False,
                )
                in_range = pm < mz_max
                t_bins = batched_bin_func(
                    eval_mz, eval_prob, eval_idx, mz_max, bin_res, "sum", sparse=True
                )
                p_bins = batched_bin_func(
                    pm[in_range],
                    pi[in_range],
                    pidx[in_range],
                    mz_max,
                    bin_res,
                    "sum",
                    sparse=True,
                )
                cos_model = cos_sim_helper(*t_bins, *p_bins)
                grid = th.arange(
                    bin_res, mz_max + bin_res, step=bin_res, device=device, dtype=path_mz.dtype
                )
                chan_bin = th.searchsorted(grid, path_mz, right=True)
                true_grid = th.arange(
                    bin_res, mz_max + bin_res, step=bin_res, device=device, dtype=eval_mz.dtype
                )
                true_bin = th.searchsorted(true_grid, eval_mz, right=True)
                chan_in = path_mz < mz_max
                chan_key = v_spec[path_frag] * num_bins + chan_bin
                true_key = eval_idx * num_bins + true_bin
                pred_c, true_c, bin_spec, _bin_idx = _compact_bins(
                    chan_key[chan_in], true_key, num_bins
                )
                tbin = eval_prob.new_zeros(bin_spec.numel()).index_add(0, true_c, eval_prob)
                chan_cbin = th.full_like(chan_bin, -1)
                chan_cbin[chan_in] = pred_c
                if cos_model.shape[0] != size:
                    raise RuntimeError("a spectrum has no in-range predicted channel")
                cos_ours = binned_cosine(path_lp, chan_cbin, tbin, bin_spec, size)
                worst["cos"] = max(worst["cos"], float((cos_ours - cos_model).abs().max()))
                if worst["cos"] > args.cos_atol:
                    raise RuntimeError(f"compact-bin cosine differs by {worst['cos']}")
                # fp16 token storage: the cosine of the checkpoint head applied to fp16 tokens.
                s16 = tokens[valid].half().float() @ weight + bias
                lp16 = (s16 - segment_logsumexp(s16, v_spec, size)[v_spec])[path_frag]
                lp16 = lp16 + chan_lp_dense[v_channels]
                cos16 = binned_cosine(lp16, chan_cbin, tbin, bin_spec, size)
                worst["cos_fp16"] = max(
                    worst.get("cos_fp16", 0.0), float((cos16 - cos_model).abs().max())
                )

                # 6. Per-spectrum CPU work: orientation, memberships, candidate peaks, pairs.
                num_atoms = int(batch["mol_pyg"].num_nodes)
                inside, outside = boundary_bond_endpoint_orientation(
                    batch["mol_pyg"].chemical_bond_index,
                    bnd_bond,
                    bnd_frag,
                    atom_idx,
                    atom_frag,
                    num_atoms=num_atoms,
                )
                new_index = th.cumsum(valid.long(), 0) - 1
                keep_atom = valid[atom_frag]
                keep_bnd = valid[bnd_frag]
                ptr = batch["mol_pyg"].ptr
                bond_graph = batch["mol_pyg"].batch[batch["mol_pyg"].chemical_bond_index[0]]
                bond_ptr = th.zeros(size + 1, dtype=th.long, device=device)
                bond_ptr[1:] = th.cumsum(th.bincount(bond_graph, minlength=size), 0)
                cpu = {
                    "v_spec": v_spec,
                    "v_s": v_s,
                    "cut": cut_counts[valid],
                    "tokens": tokens[valid],
                    "chan_frag": path_frag,
                    "chan_lp": chan_lp_dense[v_channels],
                    "chan_mz": path_mz,
                    "chan_cbin": chan_cbin,
                    "path_lp": path_lp,
                    "mem_frag": new_index[atom_frag[keep_atom]],
                    "mem_atom": atom_idx[keep_atom],
                    "bnd_frag": new_index[bnd_frag[keep_bnd]],
                    "bnd_bond": bnd_bond[keep_bnd],
                    "bnd_in": inside[keep_bnd],
                    "bnd_out": outside[keep_bnd],
                    "bin_spec": bin_spec,
                    "tbin": tbin,
                    "true_c": true_c,
                    "eval_mz": eval_mz,
                    "eval_prob": eval_prob,
                    "eval_idx": eval_idx,
                    "train_lp": train_lp,
                    "ptr": ptr,
                    "bond_ptr": bond_ptr,
                    "x": batch["mol_pyg"].x,
                    "cbi": batch["mol_pyg"].chemical_bond_index,
                    "cba": batch["mol_pyg"].chemical_bond_attr,
                    "ce_model": ce_model,
                    "cos_model": cos_model,
                    "nce": batch["spec_nce"],
                    "nce_idx": batch["spec_nce_batch_idxs"],
                    "inst": inst_idx,
                    "prec_mz": batch["spec_prec_mz"],
                }
                cpu = {k: v.detach().cpu().numpy() for k, v in cpu.items()}
                frag_start = np.searchsorted(cpu["v_spec"], np.arange(size + 1))
                chan_spec = cpu["v_spec"][cpu["chan_frag"]]
                chan_start = np.searchsorted(chan_spec, np.arange(size + 1))
                bin_start = np.searchsorted(cpu["bin_spec"], np.arange(size + 1))
                true_start = np.searchsorted(cpu["eval_idx"], np.arange(size + 1))
                true_base = arrays.meta["true_mz"]["rows"] if "true_mz" in arrays.meta else 0
                # Compact bin ids are stored per spectrum (0 .. n_bins - 1), -1 out of range.
                chan_cbin_local = np.where(
                    cpu["chan_cbin"] >= 0, cpu["chan_cbin"] - bin_start[chan_spec], -1
                )
                mem_order = np.argsort(cpu["mem_frag"], kind="stable")
                bnd_order = np.argsort(cpu["bnd_frag"], kind="stable")
                for key in ("mem_frag", "mem_atom"):
                    cpu[key] = cpu[key][mem_order]
                for key in ("bnd_frag", "bnd_bond", "bnd_in", "bnd_out"):
                    cpu[key] = cpu[key][bnd_order]
                nce_mean = np.bincount(cpu["nce_idx"], weights=cpu["nce"], minlength=size)
                nce_count = np.bincount(cpu["nce_idx"], minlength=size)
                nce_mean = nce_mean / np.maximum(nce_count, 1)

                batch_arrays: dict[str, list] = {
                    k: []
                    for k in (
                        "chan_cpeak",
                        "cpeak_tgt",
                        "cpeak_nstruct",
                        "cpeak_near_base",
                        "cpeak_bdom_frag",
                        "tbin_np",
                    )
                }
                for view, row in enumerate(rows.itertuples()):
                    f0, f1 = frag_start[view], frag_start[view + 1]
                    c0, c1 = chan_start[view], chan_start[view + 1]
                    b0, b1 = bin_start[view], bin_start[view + 1]
                    t0, t1 = true_start[view], true_start[view + 1]
                    a0, a1 = cpu["ptr"][view], cpu["ptr"][view + 1]
                    k0, k1 = cpu["bond_ptr"][view], cpu["bond_ptr"][view + 1]
                    cuts = cpu["cut"][f0:f1]
                    if np.count_nonzero(cuts == 0) != 1:
                        raise RuntimeError(f"spectrum {row.spec_id}: not exactly one precursor")
                    prec_local = int(np.flatnonzero(cuts == 0)[0])
                    rel, min_mz = tolerance[int(cpu["inst"][view])]
                    # Candidate peaks under the checkpoint's own shares (B).
                    probs = np.exp(cpu["path_lp"][c0:c1].astype(np.float64))
                    probs = probs / probs.sum()
                    mzs = cpu["chan_mz"][c0:c1].astype(np.float64)
                    cand_mzs, pred, inverse = merge_channels(mzs, probs)
                    peak_mzs = cpu["eval_mz"][t0:t1].astype(np.float64)
                    peak_ints = cpu["eval_prob"][t0:t1].astype(np.float64)
                    on_cand, peak_to_cand = map_targets(peak_mzs, peak_ints, cand_mzs, rel, min_mz)
                    base = int(np.argmax(peak_ints))
                    base_tol = rel * max(peak_mzs[base], min_mz)
                    near_base = np.abs(cand_mzs - peak_mzs[base]) <= base_tol
                    nstruct = np.bincount(inverse, minlength=cand_mzs.size)
                    local_frag = cpu["chan_frag"][c0:c1] - f0
                    # B-dominant fragment of every candidate peak (argmax share within it).
                    order = np.lexsort((-probs, inverse))
                    first = np.ones(order.size, dtype=bool)
                    first[1:] = inverse[order][1:] != inverse[order][:-1]
                    bdom = np.full(cand_mzs.size, -1, dtype=np.int64)
                    bdom[inverse[order][first]] = local_frag[order][first]
                    # Nonprecursor target: drop peaks within tolerance of a precursor channel.
                    prec_ch = local_frag == prec_local
                    prec_mzs_ch = mzs[prec_ch]
                    tol_peaks = rel * np.maximum(peak_mzs, min_mz)
                    near_prec = (
                        (
                            np.abs(peak_mzs[:, None] - prec_mzs_ch[None, :]) <= tol_peaks[:, None]
                        ).any(1)
                        if prec_mzs_ch.size
                        else np.zeros(peak_mzs.size, bool)
                    )
                    tc = cpu["true_c"][t0:t1] - b0
                    tbin_np = np.zeros(b1 - b0)
                    np.add.at(tbin_np, tc[~near_prec], peak_ints[~near_prec])
                    # Measured #1 / #2 candidate peaks and their relation (B structures).
                    tgt_rank = descending_ranks(on_cand)
                    c_1 = int(np.flatnonzero(tgt_rank == 1)[0]) if cand_mzs.size else -1
                    c_2 = int(np.flatnonzero(tgt_rank == 2)[0]) if cand_mzs.size > 1 else -1
                    relation, pair_ok, unamb = "none", False, False
                    if c_2 >= 0 and on_cand[c_1] > 0 and on_cand[c_2] >= MIN_SECOND * on_cand[c_1]:
                        members = (
                            cpu["mem_atom"][
                                np.searchsorted(cpu["mem_frag"], f0 + bdom[c_1]) : np.searchsorted(
                                    cpu["mem_frag"], f0 + bdom[c_1], side="right"
                                )
                            ]
                            - a0
                        )
                        members_2 = (
                            cpu["mem_atom"][
                                np.searchsorted(cpu["mem_frag"], f0 + bdom[c_2]) : np.searchsorted(
                                    cpu["mem_frag"], f0 + bdom[c_2], side="right"
                                )
                            ]
                            - a0
                        )
                        relation = pair_relation(
                            tuple(sorted(members.tolist())),
                            tuple(sorted(members_2.tolist())),
                            int(a1 - a0),
                        )
                        pair_ok = True
                        unamb = bool(nstruct[c_1] == 1 and nstruct[c_2] == 1)
                    base_frag = bdom[peak_to_cand[base]] if peak_to_cand[base] >= 0 else -1
                    batch_arrays["chan_cpeak"].append(inverse.astype(np.int32))
                    batch_arrays["cpeak_tgt"].append(on_cand.astype(np.float32))
                    batch_arrays["cpeak_nstruct"].append(nstruct.astype(np.int16))
                    batch_arrays["cpeak_near_base"].append(near_base)
                    batch_arrays["cpeak_bdom_frag"].append(bdom.astype(np.int32))
                    batch_arrays["tbin_np"].append(tbin_np.astype(np.float32))
                    if row.mol_id not in molecules:
                        molecules[int(row.mol_id)] = {
                            "x": cpu["x"][a0:a1].astype(np.float32),
                            "bond_index": (cpu["cbi"][:, k0:k1] - a0).astype(np.int32),
                            "bond_attr": cpu["cba"][k0:k1].astype(np.float32),
                        }
                    elif molecules[int(row.mol_id)]["x"].shape[0] != a1 - a0:
                        raise RuntimeError(f"molecule {row.mol_id}: atom counts differ")
                    spectrum_rows.append(
                        {
                            "spec_id": int(row.spec_id),
                            "mol_id": int(row.mol_id),
                            "nce": float(nce_mean[view]),
                            "nce_count": int(nce_count[view]),
                            "inst_idx": int(cpu["inst"][view]),
                            "prec_mz": float(cpu["prec_mz"][view]),
                            "n_frag": int(f1 - f0),
                            "n_chan": int(c1 - c0),
                            "n_bins": int(b1 - b0),
                            "n_true": int(t1 - t0),
                            "n_cpeak": int(cand_mzs.size),
                            "n_atoms": int(a1 - a0),
                            "frag_offset": frag_total + int(f0),
                            "chan_offset": chan_total + int(c0),
                            "bin_offset": bin_total + int(b0),
                            "cpeak_offset": cpeak_total,
                            "true_offset": true_base + int(t0),
                            "prec_frag": prec_local,
                            "tol_rel": rel,
                            "tol_min_mz": min_mz,
                            "ce_model": float(cpu["ce_model"][view]),
                            "cos_model": float(cpu["cos_model"][view]),
                            "base_cpeak": int(peak_to_cand[base]),
                            "base_cut": int(cuts[base_frag]) if base_frag >= 0 else -1,
                            "pair_c1": c_1,
                            "pair_c2": c_2,
                            "pair_ok": pair_ok,
                            "pair_relation": relation,
                            "pair_unambiguous": unamb,
                            "tgt_np_mass": float(peak_ints[~near_prec].sum()),
                        }
                    )
                    cpeak_total += int(cand_mzs.size)

                # 7. Append this batch.
                num_valid = int(cpu["v_s"].size)
                arrays.add("tokens", cpu["tokens"].astype(np.float16))
                arrays.add("s_ckpt", cpu["v_s"].astype(np.float32))
                arrays.add("cut", cpu["cut"].astype(np.int16))
                arrays.add("frag_spec", (cpu["v_spec"] + cursor).astype(np.int32))
                arrays.add("chan_frag", (cpu["chan_frag"] + frag_total).astype(np.int64))
                arrays.add("chan_lp", cpu["chan_lp"].astype(np.float32))
                arrays.add("chan_mz", cpu["chan_mz"])
                arrays.add("chan_cbin", chan_cbin_local.astype(np.int32))
                arrays.add("chan_cpeak", np.concatenate(batch_arrays["chan_cpeak"]))
                arrays.add("tbin", cpu["tbin"].astype(np.float32))
                arrays.add("tbin_np", np.concatenate(batch_arrays["tbin_np"]))
                arrays.add("cpeak_tgt", np.concatenate(batch_arrays["cpeak_tgt"]))
                arrays.add("cpeak_nstruct", np.concatenate(batch_arrays["cpeak_nstruct"]))
                arrays.add("cpeak_near_base", np.concatenate(batch_arrays["cpeak_near_base"]))
                arrays.add("cpeak_bdom_frag", np.concatenate(batch_arrays["cpeak_bdom_frag"]))
                arrays.add("true_mz", cpu["eval_mz"])
                arrays.add("true_lp", cpu["train_lp"])
                arrays.add("true_prob", cpu["eval_prob"])
                bnd_spec = cpu["v_spec"][cpu["bnd_frag"]]
                arrays.add("bnd_frag", (cpu["bnd_frag"] + frag_total).astype(np.int64))
                arrays.add(
                    "bnd_bond", (cpu["bnd_bond"] - cpu["bond_ptr"][bnd_spec]).astype(np.int16)
                )
                arrays.add("bnd_in", (cpu["bnd_in"] - cpu["ptr"][bnd_spec]).astype(np.int16))
                arrays.add("bnd_out", (cpu["bnd_out"] - cpu["ptr"][bnd_spec]).astype(np.int16))
                frag_total += num_valid
                chan_total += int(cpu["chan_frag"].size)
                bin_total += int(cpu["bin_spec"].size)
                cursor += size
                if cursor % (args.batch_size * 20) == 0:
                    LOGGER.info("%d / %d spectra, %d fragments", cursor, len(specs), frag_total)
    finally:
        capture.close()
        arrays.close()

    spectra = pd.DataFrame(spectrum_rows)
    spectra.to_pickle(out_dp / "spectra.pkl")
    mol_ids = sorted(molecules)
    atom_offsets = np.concatenate([[0], np.cumsum([molecules[m]["x"].shape[0] for m in mol_ids])])
    bond_offsets = np.concatenate(
        [[0], np.cumsum([molecules[m]["bond_attr"].shape[0] for m in mol_ids])]
    )
    np.savez(
        out_dp / "molecules.npz",
        mol_id=np.asarray(mol_ids, dtype=np.int64),
        atom_offsets=atom_offsets,
        bond_offsets=bond_offsets,
        x=np.concatenate([molecules[m]["x"] for m in mol_ids]),
        bond_index=np.concatenate([molecules[m]["bond_index"] for m in mol_ids], axis=1),
        bond_attr=np.concatenate([molecules[m]["bond_attr"] for m in mol_ids]),
    )
    provenance.update(
        {
            "status": "complete",
            "elapsed_seconds": time.time() - started,
            "spectra": len(spectra),
            "fragments": frag_total,
            "channels": chan_total,
            "reconstruction_max_abs": worst,
            "mean_cos_model": float(spectra.cos_model.mean()),
            "mean_ce_model": float(spectra.ce_model.mean()),
            "pairs": int(spectra.pair_ok.sum()),
        }
    )
    (out_dp / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    LOGGER.info(
        "extracted %d spectra, %d fragments in %.0f s; mean cos %.4f; worst %s",
        len(spectra),
        frag_total,
        time.time() - started,
        spectra.cos_model.mean(),
        worst,
    )


# ---------------------------------------------------------------------------------------------
# Cached splits
# ---------------------------------------------------------------------------------------------


@dataclass
class Batch:
    """One batch of cached spectra on the device; indices are batch-local."""

    size: int
    frag_spec: th.Tensor
    tokens: th.Tensor | None
    s_ckpt: th.Tensor
    cut: th.Tensor
    prec: th.Tensor
    chan_frag: th.Tensor
    chan_lp: th.Tensor
    chan_mz: th.Tensor
    chan_cbin: th.Tensor
    tbin: th.Tensor
    tbin_np: th.Tensor
    bin_spec: th.Tensor
    true_mz: th.Tensor
    true_lp: th.Tensor
    true_spec: th.Tensor
    tol: th.Tensor
    min_mz: th.Tensor
    acq: th.Tensor
    nce_bin: th.Tensor
    graph: dict[str, th.Tensor] | None
    bnd_frag: th.Tensor | None
    bnd_atoms: th.Tensor | None
    bnd_edge: th.Tensor | None
    rows: np.ndarray


def _ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenate ``arange(s, s + c)`` over rows, vectorized."""
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offsets = np.repeat(starts - np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    return np.arange(total, dtype=np.int64) + offsets


def nce_bins(nce: np.ndarray) -> np.ndarray:
    """NCE bin per spectrum: 0 (<=20), 1 (20-40], 2 (40-60], 3 (>60)."""
    return np.digitize(nce, NCE_EDGES, right=True)


class CachedSplit:
    """A cached producer split: spectrum table, memory-mapped arrays and molecules."""

    def __init__(self, directory: Path, load_tokens: bool = True) -> None:
        self.directory = directory
        self.spectra = pd.read_pickle(directory / "spectra.pkl")
        names = (
            None
            if load_tokens
            else [n for n in json.loads((directory / "arrays.json").read_text()) if n != "tokens"]
        )
        # Arrays stay memory-mapped (the tuning split's channel arrays alone are ~5 GB); the
        # page cache keeps the hot ones resident. Only the fragment index is loaded.
        self.arrays = load_arrays(directory, names)
        self.arrays["frag_spec"] = np.asarray(self.arrays["frag_spec"])
        mol = np.load(directory / "molecules.npz")
        self.mol = {k: mol[k] for k in mol.files}
        self.mol_row = {int(m): i for i, m in enumerate(self.mol["mol_id"])}
        sp = self.spectra
        self.frag_offset = sp.frag_offset.to_numpy(np.int64)
        self.n_frag = sp.n_frag.to_numpy(np.int64)
        self.chan_offset = sp.chan_offset.to_numpy(np.int64)
        self.n_chan = sp.n_chan.to_numpy(np.int64)
        self.bin_offset = sp.bin_offset.to_numpy(np.int64)
        self.n_bins = sp.n_bins.to_numpy(np.int64)
        self.true_offset = sp.true_offset.to_numpy(np.int64)
        self.n_true = sp.n_true.to_numpy(np.int64)
        self.cpeak_offset = sp.cpeak_offset.to_numpy(np.int64)
        self.n_cpeak = sp.n_cpeak.to_numpy(np.int64)
        self.nce_bin = nce_bins(sp.nce.to_numpy())
        self.bnd_start = np.searchsorted(
            np.asarray(self.arrays["bnd_frag"]), np.arange(len(self.arrays["s_ckpt"]) + 1)
        )

    def __len__(self) -> int:
        return len(self.spectra)

    def batch(
        self, rows: np.ndarray, device: th.device, need_tokens: bool, need_graph: bool
    ) -> Batch:
        """Gather spectra ``rows`` into a device batch."""
        rows = np.asarray(rows, dtype=np.int64)
        size = rows.size
        a = self.arrays
        frag = _ranges(self.frag_offset[rows], self.n_frag[rows])
        frag_spec = np.repeat(np.arange(size), self.n_frag[rows])
        frag_base = np.concatenate([[0], np.cumsum(self.n_frag[rows])[:-1]])
        chan = _ranges(self.chan_offset[rows], self.n_chan[rows])
        chan_spec = np.repeat(np.arange(size), self.n_chan[rows])
        chan_frag = a["chan_frag"][chan] - self.frag_offset[rows][chan_spec] + frag_base[chan_spec]
        bins = _ranges(self.bin_offset[rows], self.n_bins[rows])
        bin_spec = np.repeat(np.arange(size), self.n_bins[rows])
        bin_base = np.concatenate([[0], np.cumsum(self.n_bins[rows])[:-1]])
        cbin = a["chan_cbin"][chan].astype(np.int64)
        cbin = np.where(cbin >= 0, cbin + bin_base[chan_spec], -1)
        true = _ranges(self.true_offset[rows], self.n_true[rows])
        true_spec = np.repeat(np.arange(size), self.n_true[rows])
        sp = self.spectra.iloc[rows]
        prec = np.zeros(frag.size, dtype=bool)
        prec[frag_base + sp.prec_frag.to_numpy()] = True
        inst = sp.inst_idx.to_numpy()
        nce = sp.nce.to_numpy()
        acq = np.column_stack([nce / 100.0, inst == 0, inst == 1]).astype(np.float32)

        def dev(x: np.ndarray, dtype: th.dtype | None = None) -> th.Tensor:
            t = th.from_numpy(np.ascontiguousarray(x))
            return t.to(device=device, dtype=dtype) if dtype is not None else t.to(device)

        graph = bnd_frag_t = bnd_atoms = bnd_edge = None
        if need_graph:
            mol_rows = np.asarray([self.mol_row[int(m)] for m in sp.mol_id])
            unique_mols, mol_inv = np.unique(mol_rows, return_inverse=True)
            n_at = self.mol["atom_offsets"][unique_mols + 1] - self.mol["atom_offsets"][unique_mols]
            n_bd = self.mol["bond_offsets"][unique_mols + 1] - self.mol["bond_offsets"][unique_mols]
            at_base = np.concatenate([[0], np.cumsum(n_at)[:-1]])
            bd_base = np.concatenate([[0], np.cumsum(n_bd)[:-1]])
            atoms = _ranges(self.mol["atom_offsets"][unique_mols], n_at)
            bonds = _ranges(self.mol["bond_offsets"][unique_mols], n_bd)
            bond_mol = np.repeat(np.arange(unique_mols.size), n_bd)
            bidx = self.mol["bond_index"][:, bonds] + at_base[bond_mol]
            x = self.mol["x"][atoms][:, ATOM_COLUMNS]
            battr = self.mol["bond_attr"][bonds][:, BOND_COLUMNS]
            graph = {
                "x": dev(x, th.float32),
                "edge_index": dev(np.concatenate([bidx, bidx[::-1]], axis=1), th.long),
                "edge_attr": dev(np.concatenate([battr, battr]), th.float32),
                "bond_attr": dev(battr, th.float32),
            }
            b0, b1 = self.bnd_start[frag], self.bnd_start[frag + 1]
            brows = _ranges(b0, b1 - b0)
            b_frag_local = np.repeat(np.arange(frag.size), b1 - b0)
            b_spec = frag_spec[b_frag_local]
            b_mol = mol_inv[b_spec]
            bnd_frag_t = dev(b_frag_local, th.long)
            bnd_atoms = dev(
                np.stack(
                    [
                        a["bnd_in"][brows].astype(np.int64) + at_base[b_mol],
                        a["bnd_out"][brows].astype(np.int64) + at_base[b_mol],
                    ]
                ),
                th.long,
            )
            bnd_edge = dev(a["bnd_bond"][brows].astype(np.int64) + bd_base[b_mol], th.long)
        return Batch(
            size=size,
            frag_spec=dev(frag_spec, th.long),
            tokens=dev(np.asarray(a["tokens"][frag]), th.float32) if need_tokens else None,
            s_ckpt=dev(a["s_ckpt"][frag], th.float32),
            cut=dev(a["cut"][frag], th.long),
            prec=dev(prec),
            chan_frag=dev(chan_frag, th.long),
            chan_lp=dev(a["chan_lp"][chan], th.float32),
            chan_mz=dev(a["chan_mz"][chan]),
            chan_cbin=dev(cbin, th.long),
            tbin=dev(a["tbin"][bins], th.float32),
            tbin_np=dev(a["tbin_np"][bins], th.float32),
            bin_spec=dev(bin_spec, th.long),
            true_mz=dev(a["true_mz"][true]),
            true_lp=dev(a["true_lp"][true]),
            true_spec=dev(true_spec, th.long),
            tol=dev(sp.tol_rel.to_numpy()),
            min_mz=dev(sp.tol_min_mz.to_numpy()),
            acq=dev(acq, th.float32),
            nce_bin=dev(self.nce_bin[rows], th.long),
            graph=graph,
            bnd_frag=bnd_frag_t,
            bnd_atoms=bnd_atoms,
            bnd_edge=bnd_edge,
            rows=rows,
        )


# ---------------------------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------------------------


class AcquisitionEncoder(nn.Module):
    """NCE and instrument (the adduct is constant in this pilot) -> a small code."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, width))

    def forward(self, acq: th.Tensor) -> th.Tensor:
        return self.net(acq)


class LocalCutScorer(nn.Module):
    """``l_f = sum_b g(h_in, h_out, e_b, a(x))``, a radius-2 GINE over the intact precursor.

    Two GINE layers give each atom a radius-2 receptive field; a boundary bond reads its
    retained (inside) and lost (outside) endpoints in that order, so complementary fragments
    of one cut can score differently. The zero-cut precursor gets an acquisition-only survival
    score. Both output layers start at zero, so the scorer starts at ``l = 0`` everywhere.
    """

    def __init__(self, width: int = 32, acq_width: int = 8) -> None:
        super().__init__()
        atom_in, bond_in = len(ATOM_COLUMNS), len(BOND_COLUMNS)
        self.atom_in = nn.Linear(atom_in, width)
        self.convs = nn.ModuleList(
            GINEConv(
                nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width)),
                edge_dim=bond_in,
            )
            for _ in range(2)
        )
        self.acq = AcquisitionEncoder(acq_width)
        self.bond = nn.Sequential(
            nn.Linear(2 * width + bond_in + acq_width, width), nn.ReLU(), nn.Linear(width, 1)
        )
        self.survival = nn.Sequential(nn.Linear(acq_width, 16), nn.ReLU(), nn.Linear(16, 1))
        for head in (self.bond[-1], self.survival[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def atom_states(self, graph: dict[str, th.Tensor]) -> th.Tensor:
        h = self.atom_in(graph["x"])
        for conv in self.convs:
            h = F.relu(conv(h, graph["edge_index"], graph["edge_attr"]))
        return h

    def forward(self, b: Batch) -> th.Tensor:
        h = self.atom_states(b.graph)
        a = self.acq(b.acq)
        edge = b.graph["bond_attr"][b.bnd_edge]
        rows = th.cat(
            [h[b.bnd_atoms[0]], h[b.bnd_atoms[1]], edge, a[b.frag_spec[b.bnd_frag]]], dim=1
        )
        bond_scores = self.bond(rows).squeeze(1)
        local = h.new_zeros(b.frag_spec.numel()).index_add(0, b.bnd_frag, bond_scores)
        survival = self.survival(a).squeeze(1)[b.frag_spec]
        return th.where(b.prec, survival, local)


class CutCountScorer(nn.Module):
    """Attribution control: ``l_f = MLP(onehot(min(k_f, 6)), a(x))``; zero at start."""

    def __init__(self, width: int = 32, acq_width: int = 8) -> None:
        super().__init__()
        self.acq = AcquisitionEncoder(acq_width)
        self.net = nn.Sequential(
            nn.Linear(MAX_CUT_ONEHOT + 1 + acq_width, width), nn.ReLU(), nn.Linear(width, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, b: Batch) -> th.Tensor:
        cuts = F.one_hot(b.cut.clamp_max(MAX_CUT_ONEHOT), MAX_CUT_ONEHOT + 1).float()
        a = self.acq(b.acq)[b.frag_spec]
        return self.net(th.cat([cuts, a], dim=1)).squeeze(1)


@dataclass(frozen=True)
class ArmSpec:
    """One fitted arm: which terms enter the score and the penalty strength name."""

    name: str
    context: bool
    local: str  # "none", "gine" or "cut"
    penalty: str  # "zero", "weak" or "strong"
    family: str


FITTED_ARMS = (
    ArmSpec("C", True, "none", "zero", "context"),
    ArmSpec("G1", True, "none", "weak", "context"),
    ArmSpec("G2", True, "none", "strong", "context"),
    ArmSpec("U", True, "gine", "zero", "joint"),
    ArmSpec("R1", True, "gine", "weak", "joint"),
    ArmSpec("R2", True, "gine", "strong", "joint"),
    ArmSpec("L", False, "gine", "zero", "local"),
)
CUT_ARMS = (
    ArmSpec("K0", True, "cut", "zero", "cutcount"),
    ArmSpec("K1", True, "cut", "weak", "cutcount"),
    ArmSpec("K2", True, "cut", "strong", "cutcount"),
)


class ArmModel(nn.Module):
    """The score of one arm: optional refit context head plus optional local term."""

    def __init__(self, spec: ArmSpec, head_weight: th.Tensor, head_bias: th.Tensor) -> None:
        super().__init__()
        self.spec = spec
        # Built for every arm (and dropped when unused) so the local encoder that follows
        # draws the same seeded weights in every arm of that structure, L included.
        self.head = nn.Linear(head_weight.shape[1], 1)
        with th.no_grad():
            self.head.weight.copy_(head_weight)
            self.head.bias.copy_(head_bias)
        if not spec.context:
            self.head = None
        self.local = {"none": None, "gine": LocalCutScorer, "cut": CutCountScorer}[spec.local]
        if self.local is not None:
            self.local = self.local()

    def forward(self, b: Batch) -> tuple[th.Tensor, th.Tensor | None, th.Tensor | None]:
        """Return the score ``s``, the context term ``c`` and the local term ``l``."""
        context = self.head(b.tokens).squeeze(1) if self.head is not None else None
        local = self.local(b) if self.local is not None else None
        if context is None:
            return local, None, local
        if local is None:
            return context, context, None
        return context + local, context, local


def fragment_logprobs(scores: th.Tensor, b: Batch) -> th.Tensor:
    """Log-softmax of fragment scores within each spectrum."""
    return scores - segment_logsumexp(scores, b.frag_spec, b.size)[b.frag_spec]


def channel_logprobs(scores: th.Tensor, b: Batch) -> th.Tensor:
    """Output-channel log-probabilities: fragment allocation plus fixed channel shares."""
    return fragment_logprobs(scores, b)[b.chan_frag] + b.chan_lp


def spectral_ce(scores: th.Tensor, b: Batch, sce_fn: Any) -> th.Tensor:
    """The production sparse Gaussian CE per spectrum for these fragment scores."""
    lp = channel_logprobs(scores, b).to(b.true_lp.dtype)
    return sce_fn(
        b.true_mz,
        b.true_lp,
        b.true_spec,
        b.chan_mz,
        lp,
        b.frag_spec[b.chan_frag],
        lp.new_zeros(b.size),
        tol_per_sample=b.tol.to(b.true_lp.dtype),
        min_mz_per_sample=b.min_mz.to(b.true_lp.dtype),
    )[0]


def spectrum_cosine(scores: th.Tensor, b: Batch, nonprecursor: bool = False) -> th.Tensor:
    """Production ``cos_sim_0.01`` per spectrum (optionally with the precursor removed)."""
    lp = channel_logprobs(scores, b)
    if not nonprecursor:
        return binned_cosine(lp, b.chan_cbin, b.tbin, b.bin_spec, b.size)
    bins = th.where(b.prec[b.chan_frag], th.full_like(b.chan_cbin, -1), b.chan_cbin)
    return binned_cosine(lp, bins, b.tbin_np, b.bin_spec, b.size)


def calibrate_scores(
    family: str, params: th.Tensor, scores: th.Tensor, b: Batch, extra: th.Tensor | None = None
) -> th.Tensor:
    """Apply a calibration family to fragment scores.

    ``gamma``: ``g * s``. ``nce``: ``g[bin] * s + d[bin] * [precursor]`` per NCE bin.
    ``stack``: the ``nce`` family plus ``beta * extra`` (a frozen local score).
    """
    if family == "gamma":
        return params[0] * scores
    fbin = b.nce_bin[b.frag_spec]
    out = params[fbin] * scores + params[NUM_NCE_BINS + fbin] * b.prec.to(scores.dtype)
    if family == "stack":
        out = out + params[2 * NUM_NCE_BINS] * extra
    return out


def initial_params(family: str, device: th.device) -> th.Tensor:
    size = {"gamma": 1, "nce": 2 * NUM_NCE_BINS, "stack": 2 * NUM_NCE_BINS + 1}[family]
    params = th.zeros(size, device=device, dtype=th.float64)
    params[: 1 if family == "gamma" else NUM_NCE_BINS] = 1.0
    return params


# ---------------------------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------------------------


def _batches(n: int, size: int, rng: np.random.Generator | None) -> list[np.ndarray]:
    order = rng.permutation(n) if rng is not None else np.arange(n)
    return [order[i : i + size] for i in range(0, n, size)]


def _mol_parity(split: CachedSplit) -> np.ndarray:
    return split.spectra.mol_id.to_numpy() % 2


def _calibration_subset(
    split: CachedSplit, rows: np.ndarray, max_mols: int, seed: int
) -> np.ndarray:
    """A fixed molecule subsample of ``rows`` for fitting low-dimensional calibrations."""
    mols = np.unique(split.spectra.mol_id.to_numpy()[rows])
    if mols.size > max_mols:
        mols = np.random.default_rng(seed).choice(mols, max_mols, replace=False)
    return rows[np.isin(split.spectra.mol_id.to_numpy()[rows], mols)]


def compute_scores(
    split: CachedSplit,
    arms: dict[str, ArmModel],
    device: th.device,
    batch_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Fragment scores ``{"s": (num_fragments,)}`` of each arm over a whole split (eval)."""
    n_frag = len(split.arrays["s_ckpt"])
    out = {name: {"s": np.zeros(n_frag, np.float32)} for name in arms}
    need_graph = any(m.spec.local == "gine" for m in arms.values())
    for m in arms.values():
        m.eval()
    with th.no_grad():
        for rows in _batches(len(split), batch_size, None):
            b = split.batch(rows, device, need_tokens=True, need_graph=need_graph)
            frag = _ranges(split.frag_offset[rows], split.n_frag[rows])
            for name, model in arms.items():
                out[name]["s"][frag] = model(b)[0].float().cpu().numpy()
    return out


def _score_batch(split: CachedSplit, rows: np.ndarray, device: th.device) -> Batch:
    return split.batch(rows, device, need_tokens=False, need_graph=False)


def cosine_of_scores(
    split: CachedSplit,
    scores: np.ndarray,
    rows: np.ndarray,
    device: th.device,
    batch_size: int,
    nonprecursor: bool = False,
) -> np.ndarray:
    """Per-spectrum production cosine of fixed fragment scores over ``rows``."""
    out = np.zeros(rows.size)
    with th.no_grad():
        for i in range(0, rows.size, batch_size):
            part = rows[i : i + batch_size]
            b = _score_batch(split, part, device)
            frag = _ranges(split.frag_offset[part], split.n_frag[part])
            s = th.from_numpy(scores[frag]).to(device)
            out[i : i + part.size] = spectrum_cosine(s, b, nonprecursor).cpu().numpy()
    return out


def fit_calibration(
    split: CachedSplit,
    family: str,
    scores: np.ndarray,
    rows: np.ndarray,
    device: th.device,
    extra: np.ndarray | None = None,
    chunk: int = 4096,
) -> np.ndarray:
    """Fit a calibration family by maximizing mean production cosine over ``rows``.

    The batches are built once and held on the device; L-BFGS over at most nine parameters.
    """
    batches = []
    for i in range(0, rows.size, chunk):
        part = rows[i : i + chunk]
        frag = _ranges(split.frag_offset[part], split.n_frag[part])
        b = _score_batch(split, part, device)
        s = th.from_numpy(scores[frag]).to(device, th.float64)
        e = th.from_numpy(extra[frag]).to(device, th.float64) if extra is not None else None
        batches.append((b, s, e))
    params = initial_params(family, device).requires_grad_(True)
    optimizer = th.optim.LBFGS([params], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")

    def closure() -> th.Tensor:
        optimizer.zero_grad()
        total = params.new_zeros(())
        for b, s, e in batches:
            cal = calibrate_scores(family, params, s, b, e)
            lp = fragment_logprobs(cal, b)[b.chan_frag] + b.chan_lp.to(th.float64)
            cos = binned_cosine(lp, b.chan_cbin, b.tbin.to(th.float64), b.bin_spec, b.size)
            loss = -cos.sum() / rows.size
            loss.backward()
            total = total + loss.detach()
        return total

    optimizer.step(closure)
    return params.detach().cpu().numpy()


def apply_calibration(
    split: CachedSplit,
    family: str,
    params: np.ndarray,
    scores: np.ndarray,
    extra: np.ndarray | None = None,
) -> np.ndarray:
    """Calibrated fragment scores over the whole split (numpy)."""
    fbin = split.nce_bin[split.arrays["frag_spec"]]
    prec = np.zeros(scores.size)
    prec[split.frag_offset + split.spectra.prec_frag.to_numpy()] = 1.0
    if family == "gamma":
        return (params[0] * scores).astype(np.float32)
    out = params[fbin] * scores + params[NUM_NCE_BINS + fbin] * prec
    if family == "stack":
        out = out + params[2 * NUM_NCE_BINS] * extra
    return out.astype(np.float32)


def oof_calibrated_cosine(
    split: CachedSplit,
    family: str,
    scores: np.ndarray,
    device: th.device,
    max_mols: int,
    seed: int,
    batch_size: int,
    extra: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Two-fold (molecule parity) out-of-fold calibrated cosine over the whole split."""
    parity = _mol_parity(split)
    oof = np.zeros(len(split))
    fitted = []
    for fold in (0, 1):
        fit_rows = _calibration_subset(split, np.flatnonzero(parity == fold), max_mols, seed + fold)
        params = fit_calibration(split, family, scores, fit_rows, device, extra)
        fitted.append(params)
        test_rows = np.flatnonzero(parity != fold)
        cal = apply_calibration(split, family, params, scores, extra)
        oof[test_rows] = cosine_of_scores(split, cal, test_rows, device, batch_size)
    return oof, fitted


def _penalty_values(
    split: CachedSplit, head: ArmModel, sce_fn: Any, device: th.device, rows: np.ndarray, bs: int
) -> tuple[float, float]:
    """Mean spectral CE of the unfitted head, of a uniform allocation, and mean penalty.

    Returns:
        ``(L0, L_uniform, R0)``: the checkpoint head's CE, the CE with every fragment score set
        to 0 (uniform over valid fragments, channel shares unchanged) and the head's R(x).
    """
    ce_sum = uniform_sum = pen_sum = 0.0
    with th.no_grad():
        for i in range(0, rows.size, bs):
            b = split.batch(rows[i : i + bs], device, need_tokens=True, need_graph=False)
            s, c, _ = head(b)
            ce_sum += float(spectral_ce(s, b, sce_fn).sum())
            uniform_sum += float(spectral_ce(th.zeros_like(s), b, sce_fn).sum())
            pen_sum += float(context_penalty(c, b.frag_spec, b.size).sum())
    return ce_sum / rows.size, uniform_sum / rows.size, pen_sum / rows.size


def _load_head(checkpoint: str) -> tuple[th.Tensor, th.Tensor]:
    state = th.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    return (
        state["model.intensity_head.mask_residual_head.weight"].float(),
        state["model.intensity_head.mask_residual_head.bias"].float(),
    )


def _sce_from_provenance(split: CachedSplit) -> Any:
    """Rebuild the producer's exact CE closure from the kwargs stored at extraction."""
    provenance = json.loads((split.directory / "provenance.json").read_text())
    return get_sparse_cross_entropy_fn(**provenance["sce_kwargs"])


def train_lockstep(
    specs: tuple[ArmSpec, ...],
    fit: CachedSplit,
    tune: CachedSplit,
    lambdas: dict[str, float],
    head: tuple[th.Tensor, th.Tensor],
    sce_fn: Any,
    args: argparse.Namespace,
    out_dp: Path,
    device: th.device,
) -> tuple[dict[str, ArmModel], pd.DataFrame]:
    """Fit ``specs`` together on identical batches; early-stop each on tuning CE.

    Returns:
        The best-tuning-CE state of each arm and the per-epoch history.
    """
    arms = {}
    for s in specs:
        # Same seed per arm: arms of one structure start from identical local weights.
        th.manual_seed(args.seed)
        arms[s.name] = ArmModel(s, *head).to(device)
    optimizers = {n: th.optim.Adam(m.parameters(), lr=args.lr) for n, m in arms.items()}
    best = dict.fromkeys(arms, (np.inf, -1, None))
    stopped = dict.fromkeys(arms, False)
    rng = np.random.default_rng(args.seed)
    history = []
    need_graph = any(s.local == "gine" for s in specs)

    def evaluate(
        split: CachedSplit, epoch: int, label: str, names: list[str]
    ) -> dict[str, dict[str, float]]:
        sums = {
            n: dict.fromkeys(("ce", "cos", "pen", "lvar", "cl", "cc", "ll"), 0.0) for n in names
        }
        for m in arms.values():
            m.eval()
        with th.no_grad():
            for rows in _batches(len(split), args.eval_batch_size, None):
                b = split.batch(rows, device, need_tokens=True, need_graph=need_graph)
                for name in names:
                    model = arms[name]
                    s, c, l_ = model(b)
                    acc = sums[name]
                    acc["ce"] += float(spectral_ce(s, b, sce_fn).sum())
                    acc["cos"] += float(spectrum_cosine(s, b).sum())
                    if c is not None:
                        acc["pen"] += float(context_penalty(c, b.frag_spec, b.size).sum())
                    if l_ is not None:
                        acc["lvar"] += float(context_penalty(l_, b.frag_spec, b.size).sum())
                    if c is not None and l_ is not None:
                        cc = c - segment_mean(c, b.frag_spec, b.size)[b.frag_spec]
                        lc = l_ - segment_mean(l_, b.frag_spec, b.size)[b.frag_spec]
                        acc["cl"] += float((cc * lc).sum())
                        acc["cc"] += float(cc.square().sum())
                        acc["ll"] += float(lc.square().sum())
        res = {}
        for name, acc in sums.items():
            res[name] = {
                "split": label,
                "epoch": epoch,
                "arm": name,
                "ce": acc["ce"] / len(split),
                "cos": acc["cos"] / len(split),
                "context_rms": float(np.sqrt(acc["pen"] / len(split))),
                "local_rms": float(np.sqrt(acc["lvar"] / len(split))),
                "corr_cl": (
                    acc["cl"] / np.sqrt(acc["cc"] * acc["ll"])
                    if acc["cc"] * acc["ll"] > 0
                    else np.nan
                ),
            }
        return res

    for epoch in range(args.max_epochs + 1):
        if epoch > 0:
            train_sums = {n: [0.0, 0.0] for n in arms}
            for m in arms.values():
                m.train()
            for rows in _batches(len(fit), args.batch_size, rng):
                b = fit.batch(rows, device, need_tokens=True, need_graph=need_graph)
                for name, model in arms.items():
                    if stopped[name]:
                        continue
                    s, c, _ = model(b)
                    ce = spectral_ce(s, b, sce_fn).mean()
                    loss = ce
                    lam = lambdas[model.spec.penalty]
                    if lam > 0.0:
                        loss = loss + lam * context_penalty(c, b.frag_spec, b.size).mean()
                    optimizers[name].zero_grad()
                    loss.backward()
                    optimizers[name].step()
                    train_sums[name][0] += float(ce) * b.size
                    train_sums[name][1] += float(loss) * b.size
            for name in (n for n in arms if not stopped[n]):
                history.append(
                    {
                        "split": "fit_train_mode",
                        "epoch": epoch,
                        "arm": name,
                        "ce": train_sums[name][0] / len(fit),
                        "loss": train_sums[name][1] / len(fit),
                    }
                )
        tuned = evaluate(tune, epoch, "tune", [n for n in arms if not stopped[n]])
        for name, row in tuned.items():
            history.append(row)
            ce = row["ce"]
            if ce < best[name][0] - args.min_delta:
                state = {k: v.detach().cpu().clone() for k, v in arms[name].state_dict().items()}
                best[name] = (ce, epoch, state)
            elif epoch - best[name][1] >= args.patience:
                stopped[name] = True
        LOGGER.info(
            "epoch %d tune CE %s",
            epoch,
            {n: round(r["ce"], 4) for n, r in tuned.items()},
        )
        pd.DataFrame(history).to_csv(out_dp / f"history_{specs[0].family}.csv", index=False)
        if all(stopped.values()):
            break
    for name, model in arms.items():
        model.load_state_dict(best[name][2])
        th.save(
            {
                "state_dict": best[name][2],
                "best_epoch": best[name][1],
                "best_tune_ce": best[name][0],
                "spec": model.spec.__dict__,
                "converged": stopped[name],
            },
            out_dp / f"arm_{name}.pt",
        )
    # Eval-mode fit-split read of the selected states, for the capacity comparison.
    fit_eval = evaluate(fit, -1, "fit_eval", list(arms))
    for name, row in fit_eval.items():
        row["epoch"] = best[name][1]
        history.append(row)
    pd.DataFrame(history).to_csv(out_dp / f"history_{specs[0].family}.csv", index=False)
    return arms, pd.DataFrame(history)


def run_fit(args: argparse.Namespace) -> None:
    """Fit and select every arm of one producer on tuning molecules; write the lock."""
    th.manual_seed(args.seed)
    np.random.seed(args.seed)  # noqa: NPY002 - seed legacy dependencies as well
    random.seed(args.seed)
    started = time.time()
    cache_dp, out_dp = Path(args.cache_dp), Path(args.output_dp)
    out_dp.mkdir(parents=True, exist_ok=True)
    if (out_dp / "lock.json").exists():
        raise FileExistsError(f"{out_dp} is already locked")
    device = th.device(args.device)
    fit = CachedSplit(cache_dp / "fit")
    tune = CachedSplit(cache_dp / "tune")
    checkpoint = json.loads((cache_dp / "fit" / "provenance.json").read_text())["args"][
        "checkpoint"
    ]
    head = _load_head(checkpoint)
    sce_fn = _sce_from_provenance(fit)

    # 1. Penalty strengths from the unfitted head on a fixed training calibration sample.
    calib_rows = np.sort(
        np.random.default_rng(args.seed).permutation(len(fit))[: args.lambda_sample]
    )
    probe = ArmModel(FITTED_ARMS[0], *head).to(device).eval()
    l0, l_uniform, r0 = _penalty_values(
        fit, probe, sce_fn, device, calib_rows, args.eval_batch_size
    )
    if not r0 > 0:
        raise RuntimeError("initial context penalty is zero; inspect the score extraction")
    # The production CE is a Gaussian-density cross-entropy with an arbitrary offset (it is
    # negative here), so "1% / 10% of the initial loss" is read offset-free: of the loss the
    # checkpoint head removes relative to a uniform allocation.
    excess = l_uniform - l0
    if not excess > 0:
        raise RuntimeError("the checkpoint head does not beat a uniform allocation")
    lambdas = {"zero": 0.0, "weak": 0.01 * excess / r0, "strong": 0.1 * excess / r0}
    LOGGER.info("L0 %.5f, L_uniform %.5f, R0 %.5f, lambdas %s", l0, l_uniform, r0, lambdas)

    # 2. Fit the main arms in lockstep, early-stopping each on tuning CE.
    arms, _history = train_lockstep(
        FITTED_ARMS, fit, tune, lambdas, head, sce_fn, args, out_dp, device
    )

    # 3. Tuning scores, calibrations and out-of-fold selection (tuning only).
    tune_scores = compute_scores(tune, arms, device, args.eval_batch_size)
    tune_scores["B"] = {"s": tune.arrays["s_ckpt"].astype(np.float32)}
    selection = {
        "lambdas": lambdas,
        "L0": l0,
        "L_uniform": l_uniform,
        "R0": r0,
        "calibration_max_molecules": args.calib_max_mols,
        "arms": {},
    }

    def calibrate_arm(name: str, scores: np.ndarray, extra: np.ndarray | None = None) -> dict:
        record = {}
        families = ("stack",) if extra is not None else ("gamma", "nce")
        for family in families:
            oof, _ = oof_calibrated_cosine(
                tune,
                family,
                scores,
                device,
                args.calib_max_mols,
                args.seed,
                args.eval_batch_size,
                extra,
            )
            rows = _calibration_subset(
                tune, np.arange(len(tune)), 2 * args.calib_max_mols, args.seed + 7
            )
            params = fit_calibration(tune, family, scores, rows, device, extra)
            record[family] = {"oof_cos": float(oof.mean()), "params": params.tolist()}
        record["raw_tune_cos"] = float(
            cosine_of_scores(
                tune, scores, np.arange(len(tune)), device, args.eval_batch_size
            ).mean()
        )
        best_family = max(families, key=lambda f: record[f]["oof_cos"])
        record["family"] = best_family
        record["oof_cos"] = record[best_family]["oof_cos"]
        # Penalty strengths within a family are compared under one fixed calibration family.
        record["select_oof_cos"] = record["stack" if extra is not None else "nce"]["oof_cos"]
        return record

    for name in ["B", *arms]:
        selection["arms"][name] = calibrate_arm(name, tune_scores[name]["s"])
        if name in arms:
            ckpt = th.load(out_dp / f"arm_{name}.pt", weights_only=False)
            selection["arms"][name].update(
                {
                    "best_epoch": ckpt["best_epoch"],
                    "best_tune_ce": ckpt["best_tune_ce"],
                    "converged": ckpt["converged"],
                }
            )
        LOGGER.info(
            "%s: %s",
            name,
            {
                k: v
                for k, v in selection["arms"][name].items()
                if k in ("family", "oof_cos", "raw_tune_cos")
            },
        )
    selection["arms"]["S"] = calibrate_arm("S", tune_scores["B"]["s"], tune_scores["L"]["s"])

    def pick(family_names: list[str]) -> str:
        return max(family_names, key=lambda n: selection["arms"][n]["select_oof_cos"])

    selection["selected"] = {
        "context": pick(["C", "G1", "G2"]),
        "joint": pick(["U", "R1", "R2"]),
    }
    # 4. The cut-count attribution control, scheduled from tuning results only.
    selection["cut_count_scheduled"] = selection["selected"]["joint"] in ("R1", "R2")
    if selection["cut_count_scheduled"]:
        k_arms, _ = train_lockstep(CUT_ARMS, fit, tune, lambdas, head, sce_fn, args, out_dp, device)
        k_scores = compute_scores(tune, k_arms, device, args.eval_batch_size)
        for name in k_arms:
            selection["arms"][name] = calibrate_arm(name, k_scores[name]["s"])
            ckpt = th.load(out_dp / f"arm_{name}.pt", weights_only=False)
            selection["arms"][name].update(
                {
                    "best_epoch": ckpt["best_epoch"],
                    "best_tune_ce": ckpt["best_tune_ce"],
                    "converged": ckpt["converged"],
                }
            )
        selection["selected"]["cutcount"] = pick(["K0", "K1", "K2"])
    selection["B_control_family"] = selection["arms"]["B"]["family"]
    selection["elapsed_seconds"] = time.time() - started
    selection["checkpoint"] = checkpoint
    selection["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    selection["git_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    (out_dp / "lock.json").write_text(json.dumps(selection, indent=2, default=float))
    LOGGER.info("locked selection: %s", selection["selected"])


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------


def candidate_reads(split: CachedSplit, scores: np.ndarray) -> pd.DataFrame:
    """Base-peak hits, pair correctness and the dominant fragments of the measured pair.

    ``dom_1`` / ``dom_2`` are the fragments (spectrum-local) carrying most of these scores'
    share at the measured #1 / #2 candidate peaks; the pair's relation label itself stays the
    one fixed at extraction from the checkpoint's own dominant structures.
    """
    a = split.arrays
    frag_spec = a["frag_spec"]
    s = scores.astype(np.float64)
    maxima = np.full(len(split), -np.inf)
    np.maximum.at(maxima, frag_spec, s)
    exp = np.exp(s - maxima[frag_spec])
    denom = np.bincount(frag_spec, weights=exp, minlength=len(split))
    frag_p = exp / denom[frag_spec]
    chan_p = frag_p[a["chan_frag"]] * np.exp(a["chan_lp"].astype(np.float64))
    chan_spec = frag_spec[a["chan_frag"]]
    cpeak_global = split.cpeak_offset[chan_spec] + a["chan_cpeak"]
    total_cpeaks = int(split.cpeak_offset[-1] + split.n_cpeak[-1])
    cp_pred = np.bincount(cpeak_global, weights=chan_p, minlength=total_cpeaks)
    # Dominant fragment of each candidate peak under these scores.
    order = np.lexsort((-chan_p, cpeak_global))
    first = np.ones(order.size, dtype=bool)
    first[1:] = cpeak_global[order][1:] != cpeak_global[order][:-1]
    dom = np.full(total_cpeaks, -1, dtype=np.int64)
    dom[cpeak_global[order][first]] = (
        a["chan_frag"][order][first] - split.frag_offset[chan_spec[order][first]]
    )
    sp = split.spectra
    starts = split.cpeak_offset
    # Top candidate peak per spectrum; ties go to the lowest index, as np.argmax.
    cp_spec = np.repeat(np.arange(len(split)), split.n_cpeak)
    top_order = np.lexsort((-cp_pred, cp_spec))
    top = top_order[np.searchsorted(cp_spec[top_order], np.arange(len(split)))]
    hit = a["cpeak_near_base"][top].astype(bool) & (sp.base_cpeak.to_numpy() >= 0)
    c1 = starts + sp.pair_c1.to_numpy().clip(0)
    c2 = starts + sp.pair_c2.to_numpy().clip(0)
    correct = cp_pred[c1] > cp_pred[c2]
    return pd.DataFrame(
        {"base_hit": hit, "pair_correct": correct, "dom_1": dom[c1], "dom_2": dom[c2]},
        index=sp.index,
    )


def pair_summary(frame: pd.DataFrame) -> dict[str, float]:
    """Balanced nested accuracy and relation accuracies over valid pairs of ``frame``."""
    pairs = frame[frame.pair_ok]
    out = {"pairs": len(pairs)}
    nested = pairs[pairs.pair_relation.isin(["first_inside_second", "second_inside_first"])]
    deeper = nested[nested.pair_relation == "first_inside_second"].pair_correct.mean()
    shallower = nested[nested.pair_relation == "second_inside_first"].pair_correct.mean()
    out["nested_balanced"] = float(np.nanmean([deeper, shallower]))
    out["nested_winner_deeper"] = float(deeper)
    out["nested_winner_shallower"] = float(shallower)
    for relation in ("disjoint", "overlapping", "complementary", "same_atoms"):
        out[f"acc_{relation}"] = float(pairs[pairs.pair_relation == relation].pair_correct.mean())
    return out


def _bootstrap_delta(mol_ids: np.ndarray, delta: np.ndarray, reps: int, seed: int) -> dict:
    sums = clustered_bootstrap(mol_ids, {"d": delta, "n": np.ones_like(delta)}, reps, seed)
    means = sums[:, 0] / sums[:, 1]
    return {
        "mean": float(delta.mean()),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
    }


def run_report(args: argparse.Namespace) -> None:
    """The single locked read on the confirmation (and, if asked, untouched) molecules."""
    cache_dp, fit_dp = Path(args.cache_dp), Path(args.fit_dp)
    lock = json.loads((fit_dp / "lock.json").read_text())
    device = th.device(args.device)
    head = _load_head(lock["checkpoint"])
    novelty = pd.read_csv(Path(args.manifest_dp) / "novelty.csv").set_index("mol_id")[
        f"nov_{args.novelty_ref}"
    ]
    names = [
        n
        for n in ("C", "G1", "G2", "U", "R1", "R2", "L", "K0", "K1", "K2")
        if (fit_dp / f"arm_{n}.pt").exists()
    ]
    arms = {}
    for name in names:
        saved = th.load(fit_dp / f"arm_{name}.pt", weights_only=False)
        model = ArmModel(ArmSpec(**saved["spec"]), *head).to(device)
        model.load_state_dict(saved["state_dict"])
        arms[name] = model
    results = {}
    for split_name in ("confirm", *(("untouched",) if args.read_untouched else ())):
        split = CachedSplit(cache_dp / split_name)
        scores = compute_scores(split, arms, device, args.eval_batch_size)
        scores["B"] = {"s": split.arrays["s_ckpt"].astype(np.float32)}
        scores["S"] = {"s": scores["B"]["s"], "extra": scores["L"]["s"]}
        sp = split.spectra.copy()
        sp["novelty"] = sp.mol_id.map(novelty).to_numpy()
        sp["novelty_bin"] = np.digitize(sp.novelty, NOVELTY_EDGES)
        sp["nce_bin"] = split.nce_bin
        all_rows = np.arange(len(split))
        per_spec = {}
        for name, rec in lock["arms"].items():
            if name not in scores:
                continue
            raw = scores[name]["s"]
            family = rec["family"]
            params = np.asarray(rec[family]["params"])
            cal = apply_calibration(split, family, params, raw, scores[name].get("extra"))
            if name == "S":
                raw = cal
            frame = pd.DataFrame(index=sp.index)
            frame["cos_raw"] = cosine_of_scores(split, raw, all_rows, device, args.eval_batch_size)
            frame["cos_cal"] = cosine_of_scores(split, cal, all_rows, device, args.eval_batch_size)
            frame["cos_np_cal"] = cosine_of_scores(
                split, cal, all_rows, device, args.eval_batch_size, nonprecursor=True
            )
            reads = candidate_reads(split, raw)
            frame = frame.join(reads)
            per_spec[name] = frame
        summary = {}
        base = per_spec["B"]
        np_ok = (sp.tgt_np_mass > 0).to_numpy()
        for name, frame in per_spec.items():
            joined = pd.concat([sp, frame], axis=1)
            entry = {
                "cos_raw": float(frame.cos_raw.mean()),
                "cos_cal": float(frame.cos_cal.mean()),
                "cos_np_cal": float(frame.cos_np_cal[np_ok].mean()),
                "base_hit": float(frame.base_hit.mean()),
                "pairs_all": pair_summary(joined),
                "pairs_unambiguous": pair_summary(joined[joined.pair_unambiguous]),
                "pairs_unambiguous_novel": pair_summary(
                    joined[joined.pair_unambiguous & (joined.novelty < NOVELTY_EDGES[0])]
                ),
                # Share of measured pairs whose two dominant fragments equal B's; the rest
                # are ambiguous peaks where the fixed B-structure relation label may not
                # describe this arm's own attribution.
                "dominant_agrees_B": float(
                    ((frame.dom_1 == base.dom_1) & (frame.dom_2 == base.dom_2))[sp.pair_ok].mean()
                ),
                "cos_cal_by_novelty": frame.cos_cal.groupby(sp.novelty_bin).mean().to_dict(),
                "cos_cal_by_nce": frame.cos_cal.groupby(sp.nce_bin).mean().to_dict(),
                "cos_cal_by_inst": frame.cos_cal.groupby(sp.inst_idx).mean().to_dict(),
                "base_hit_by_cut": frame.base_hit.groupby(sp.base_cut.clip(-1, 3)).mean().to_dict(),
            }
            if name != "B":
                d = (frame.cos_cal - base.cos_cal).to_numpy()
                entry["delta_cal_vs_B"] = _bootstrap_delta(
                    sp.mol_id.to_numpy(), d, args.reps, args.seed
                )
                entry["delta_raw_vs_B"] = _bootstrap_delta(
                    sp.mol_id.to_numpy(),
                    (frame.cos_raw - base.cos_raw).to_numpy(),
                    args.reps,
                    args.seed,
                )
                entry["delta_cal_vs_B_by_novelty"] = (
                    pd.Series(d, index=sp.index).groupby(sp.novelty_bin).mean().to_dict()
                )
            summary[name] = entry
        selected = lock["selected"]["joint"]
        others = ("C", "U", "G1", "G2", lock["selected"]["context"], "S")
        for other in (*others, lock["selected"].get("cutcount")):
            if other in per_spec and other != selected:
                d = (per_spec[selected].cos_cal - per_spec[other].cos_cal).to_numpy()
                summary[selected][f"delta_cal_vs_{other}"] = _bootstrap_delta(
                    sp.mol_id.to_numpy(), d, args.reps, args.seed
                )
        summary["counts"] = {
            "spectra": len(sp),
            "molecules": int(sp.mol_id.nunique()),
            "pairs": int(sp.pair_ok.sum()),
            "pairs_unambiguous": int((sp.pair_ok & sp.pair_unambiguous).sum()),
            "nonprecursor_spectra": int(np_ok.sum()),
            "by_novelty": sp.novelty_bin.value_counts().sort_index().to_dict(),
        }
        summary["selected"] = lock["selected"]
        summary["mean_cos_model"] = float(sp.cos_model.mean())
        results[split_name] = summary
        pd.concat(dict(per_spec), axis=1).to_pickle(
            fit_dp / f"report_{split_name}_per_spectrum.pkl"
        )
    (fit_dp / "report.json").write_text(json.dumps(results, indent=2, default=float))
    LOGGER.info("report written to %s", fit_dp / "report.json")


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    RDLogger.DisableLog("rdApp.*")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("manifests")
    p.add_argument("--local_split_dp", default="output/local_encoder_noise_2026_09_27/split")
    p.add_argument("--canonical_split_dp", default="data/split/nist20_inchikey")
    p.add_argument("--mol_fp", default="data/proc/nist20/mol_df.pkl")
    p.add_argument("--output_dp", default="output/local_context_allocation")

    p = sub.add_parser("extract")
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--custom_fp",
        default="config/masster/mh_ft_memorization_route_2026_09_24/i1_inst_ob_obf.yml",
        help="fills only the keys the checkpoint hparams predate",
    )
    p.add_argument("--template_fp", default="config")
    p.add_argument("--split_dp", required=True)
    p.add_argument("--split", required=True, choices=SPLITS)
    p.add_argument("--output_dp", required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_spectra", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logprob_atol", type=float, default=1e-4)
    p.add_argument("--cos_atol", type=float, default=1e-5)

    p = sub.add_parser("fit")
    p.add_argument("--cache_dp", required=True)
    p.add_argument("--output_dp", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--eval_batch_size", type=int, default=512)
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--min_delta", type=float, default=1e-5)
    p.add_argument("--lambda_sample", type=int, default=4096)
    p.add_argument("--calib_max_mols", type=int, default=1500)

    p = sub.add_parser("report")
    p.add_argument("--cache_dp", required=True)
    p.add_argument("--fit_dp", required=True)
    p.add_argument("--manifest_dp", default="output/local_context_allocation")
    p.add_argument("--novelty_ref", required=True, choices=("local", "full"))
    p.add_argument("--read_untouched", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_batch_size", type=int, default=512)
    p.add_argument("--reps", type=int, default=1000)

    args = parser.parse_args()
    {"manifests": run_manifests, "extract": extract, "fit": run_fit, "report": run_report}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
