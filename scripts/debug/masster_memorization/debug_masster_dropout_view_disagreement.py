"""How far does MASSter's own training dropout move its predicted peaks?

This prices the R-Drop proposal of ``docs/notes/masster_generalization_methods_2026_09_27.md``
(section 1, and item 2 of its cheap checks) before any training code is written. R-Drop adds
``lambda * [KL(P_1 || P_2) + KL(P_2 || P_1)] / 2`` between two dropout views of one spectrum. If
the checkpoint's own dropout barely moves the predicted peaks, that penalty has almost nothing
to act on at any sensible ``lambda``, and raising dropout to create disagreement is what ``h1``
already did (allocator dropout 0.1 -> 0.2, -0.0067). The same pass records the logit spread that
the note's centered-logit penalty (section 2) would act on.

Conditions, all on the same batches in the same order, so every contrast is paired:

  eval      every module in eval mode: the production prediction and its logged cosine.
  head_v    the proposal modules (context encoder, precursor encoder, bond, charge, mixture and
            mechanism heads) stay in eval mode, so every view gets the eval candidate set, and the
            cut-bond encoder and candidate allocator run in train mode with their own dropout
            draw. This is the note's head-only R-Drop view: one candidate set, two head passes.
  full_v    every dropout layer in train mode, BatchNorm left in eval mode, so the checkpoint's
            running statistics are used and never updated. Candidates can change between views;
            this prices candidate churn, the whole-model form the note defers.

The model's own top-level training flag stays off in every condition, so each condition's
logged cosine uses the production eval-time channel selection. The two views are compared on
the DENSE distribution instead: every chemically valid (fragment, ion mode, H shift) pathway,
which is what the training loss sees. Pathways are merged into peak groups by m/z within each
spectrum (a new group starts at a gap above ``GROUP_GAP_DA``), so isomeric fragments and
equivalent H-shift channels are summed before comparison: the spectrum constrains their total,
not their split. Groups present in only one view (possible under ``full`` only) enter with zero
probability there, so the symmetric KL is reported for ``head`` alone.

Per spectrum it writes: view-pair symmetric KL, total variation (the share of predicted
intensity moved), Jensen-Shannon, cosine and top-peak agreement, at exact-group and 0.01-Da
resolution; each condition's logged cosine and loss; each view's distance from ``eval``; the eval
prediction's 0.01-Da total variation to the target, the model error in the view-pair's units;
and, for the logit penalty, the spread of the centered valid-fragment log probabilities and of
the peak-group log probabilities, with predicted and target entropies at 0.01 Da.

With ``--head_views N`` (a power of two above 2) the head scope draws N views, and each spectrum
also records the cosine of the average of the first n = 4, 8, ... N views and of the N-view
average flattened. Averaging n views leaves 1/n of one view's noise, so the curve over n
separates the MC-dropout mean from the noise a two-view average still carries.

Spectra are a seeded random sample of each split (a ``spec_id`` prefix reads ~0.011 high on val;
see the memorization attempts audit). Validation molecules are scored for novelty as their
maximum Morgan-2 Tanimoto to any training molecule, and training molecules to any other one.

Usage:
    python scripts/debug/masster_memorization/debug_masster_dropout_view_disagreement.py \
        --model_ckpt saved_ckpts/i1_inst_ob_obf_i50ig7rf/ckpt/model-epoch=059.ckpt \
        --custom_fp config/masster/mh_ft_memorization_route_2026_09_24/i1_inst_ob_obf.yml \
        --num_spectra 1024 --output_fp output/dropout_views/i1.pkl
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch as th
from lightning import seed_everything
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from fragnnet.runner import load_config
from fragnnet.utils.script_utils import apply_inference_config, init_model_from_ckpt, setup_device
from fragnnet.utils.setup_utils import get_dataset_cls

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Pathways closer than this in m/z (Da) are one peak group. Same-formula pathways agree to
# float rounding; distinct formulas this close are not separable at FT's 10 ppm anyway.
GROUP_GAP_DA = 5e-4
BIN_DA = 0.01
# The configs' mz_max; main() refuses a config with another value.
MZ_MAX = 1500.0
VIEWS = 2
FLATTENING_POWERS = (0.5, 0.625, 0.75, 0.875)
MC_FLATTENING_POWERS = (0.75, 0.875)
CAPTURE_KEYS = (
    "ion_h_dense_mzs",
    "ion_h_dense_log_weights",
    "ion_h_chemical_valid_channels",
    "ion_h_fragment_batch_idxs",
    "cross_ev_fragment_logprobs",
    "cross_ev_fragment_batch_idxs",
)
COSINE = "cos_sim_0.01"


class OutputCapture:
    """Keep the dense-output tensors of every MassterModel forward pass.

    ``MassterPL.forward`` pops keys from the model's output dict, so the tensors are copied out
    by a forward hook before that happens.
    """

    def __init__(self, module: nn.Module) -> None:
        self.output: dict[str, th.Tensor] | None = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, args: Any, output: dict[str, th.Tensor]) -> None:
        missing = [key for key in CAPTURE_KEYS if key not in output]
        if missing:
            raise KeyError(f"model output lacks {missing}; unique-fragment mode is required")
        self.output = {key: output[key].detach() for key in CAPTURE_KEYS}


def set_noise_scope(pl_module: nn.Module, scope: str) -> list[str]:
    """Put the modules of one noise scope in train mode and return the active dropout sites.

    Args:
        pl_module: The loaded ``MassterPL``.
        scope: ``eval``, ``head`` or ``full``.

    Returns:
        Names of the modules whose dropout is live in this scope, for the run log.
    """
    pl_module.eval()
    model = pl_module.model
    if scope == "head":
        for module in (model.intensity_head, model.boundary_bond_encoder):
            if module is not None:
                module.train()
    elif scope == "full":
        pl_module.train()
        for module in pl_module.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        # Top-level flags only: keep the eval-time channel selection and training-only
        # candidate perturbations of MassterModel off, as in the other two scopes.
        pl_module.training = False
        model.training = False
    elif scope != "eval":
        raise ValueError(f"unknown scope {scope}")
    return [
        name
        for name, module in pl_module.named_modules()
        if module.training and _dropout_rate(module) > 0.0
    ]


def _dropout_rate(module: nn.Module) -> float:
    if isinstance(module, (nn.Dropout, nn.MultiheadAttention)):
        return float(module.p if isinstance(module, nn.Dropout) else module.dropout)
    # Modules that call F.dropout themselves keep their rate as a float attribute.
    rates = [getattr(module, name, 0.0) for name in ("dropout", "input_dropout")]
    return max([0.0] + [float(rate) for rate in rates if isinstance(rate, float)])


def dense_pathways(capture: dict[str, th.Tensor]) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Flatten the dense output into (spectrum, m/z, log probability) per valid pathway."""
    valid = capture["ion_h_chemical_valid_channels"]
    batch = capture["ion_h_fragment_batch_idxs"][:, None, None].expand_as(valid)[valid]
    mzs = capture["ion_h_dense_mzs"][valid].to(th.float64)
    log_weights = capture["ion_h_dense_log_weights"][valid].to(th.float64)
    return batch, mzs, log_weights


def support_mismatch(
    first: tuple[th.Tensor, th.Tensor, th.Tensor],
    second: tuple[th.Tensor, th.Tensor, th.Tensor],
    batch_size: int,
) -> np.ndarray:
    """Per spectrum, True when two dense outputs do not hold the same m/z pathway multiset.

    The CUDA builder returns the same fragments in a call-dependent order, which is harmless
    here. It can also resolve an exact proposal-score tie at the fragment budget differently
    between calls, swapping one isomer for another, so the check is per spectrum.
    """
    mismatch = np.zeros(batch_size, dtype=bool)
    for spectrum in range(batch_size):
        a = first[1][first[0] == spectrum].sort().values
        b = second[1][second[0] == spectrum].sort().values
        mismatch[spectrum] = a.shape != b.shape or bool(((a - b).abs() > 1e-9).any())
    return mismatch


def _segment_logsumexp(values: th.Tensor, segments: th.Tensor, num: int) -> th.Tensor:
    maxima = values.new_full((num,), -th.inf).scatter_reduce(
        0, segments, values, reduce="amax", include_self=True
    )
    safe = th.where(th.isfinite(maxima), maxima, th.zeros_like(maxima))
    sums = values.new_zeros(num).index_add(0, segments, (values - safe[segments]).exp())
    return sums.log() + safe


def group_ids(batch: th.Tensor, keys: th.Tensor, gap: float | None) -> tuple[th.Tensor, int]:
    """Number peak groups within each spectrum, by m/z gap or by exact integer key."""
    order = th.argsort(keys, stable=True)
    order = order[th.argsort(batch[order], stable=True)]
    sorted_batch, sorted_keys = batch[order], keys[order]
    new = th.ones_like(sorted_batch, dtype=th.bool)
    if order.numel() > 1:
        step = sorted_keys[1:] - sorted_keys[:-1]
        split = step > gap if gap is not None else step != 0
        new[1:] = (sorted_batch[1:] != sorted_batch[:-1]) | split
    ids = th.empty_like(order)
    ids[order] = th.cumsum(new.to(th.long), 0) - 1
    return ids, int(new.sum())


def metric_bins(mzs: th.Tensor) -> th.Tensor:
    """The logged 0.01-Da cosine's bin of each m/z, as ``batched_bin_func`` assigns it.

    Edges are built in the tensor's own dtype and searched with ``right=True``. A target peak
    recorded at exactly 141.01 then shares a bin with a 141.0102 prediction, which
    ``floor(mz / 0.01)`` in float64 would split; matching the metric exactly avoids that.
    """
    edges = th.arange(BIN_DA, MZ_MAX + BIN_DA, BIN_DA, device=mzs.device, dtype=mzs.dtype)
    return th.searchsorted(edges, mzs, right=True)


def in_metric_range(
    pathways: tuple[th.Tensor, th.Tensor, th.Tensor],
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Drop pathways at or above ``MZ_MAX``, as the logged metric does before binning."""
    keep = pathways[1] < MZ_MAX
    return pathways[0][keep], pathways[1][keep], pathways[2][keep]


def pair_distances(
    first: tuple[th.Tensor, th.Tensor, th.Tensor],
    second: tuple[th.Tensor, th.Tensor, th.Tensor],
    batch_size: int,
    binned: bool,
) -> dict[str, np.ndarray]:
    """Compare two dense predictions of the same spectra on their union of peak groups.

    Args:
        first: ``dense_pathways`` of one condition.
        second: ``dense_pathways`` of the other.
        batch_size: Spectra in the batch.
        binned: Group at 0.01-Da bins instead of by m/z gap.

    Returns:
        Per-spectrum arrays: ``tv`` = 0.5 * sum |P - Q|, ``js`` (nats), ``cos``, ``top1_same``,
        ``churn`` (mass on groups the other view lacks, averaged over the two views) and
        ``symkl`` = 0.5 * sum (P - Q)(log P - log Q), which is inf where a group is one-sided.
    """
    if binned:
        first, second = in_metric_range(first), in_metric_range(second)
    batch = th.cat((first[0], second[0]))
    log_weights = th.cat((first[2], second[2]))
    source = th.cat((th.zeros_like(first[0]), th.ones_like(second[0])))
    if binned:
        keys = th.cat((metric_bins(first[1]), metric_bins(second[1])))
    else:
        keys = th.cat((first[1].to(th.float64), second[1].to(th.float64)))
    ids, num = group_ids(batch, keys, None if binned else GROUP_GAP_DA)
    log_p = _segment_logsumexp(log_weights, ids * 2 + source, 2 * num).reshape(num, 2)
    group_batch = th.zeros(num, dtype=th.long, device=batch.device).scatter_(0, ids, batch)
    p = log_p.exp()
    one_sided = th.isinf(log_p).any(dim=1)
    log_ratio = th.where(one_sided, th.zeros_like(p[:, 0]), log_p[:, 0] - log_p[:, 1])
    mixture = p.mean(dim=1, keepdim=True)
    js_terms = (p * (log_p - mixture.log()).nan_to_num(nan=0.0, neginf=0.0)).sum(dim=1) / 2.0

    def per_spectrum(values: th.Tensor) -> th.Tensor:
        return values.new_zeros(batch_size).index_add(0, group_batch, values)

    tv = per_spectrum((p[:, 0] - p[:, 1]).abs()) / 2.0
    symkl = per_spectrum((p[:, 0] - p[:, 1]) * log_ratio) / 2.0
    symkl = th.where(per_spectrum(one_sided.to(p.dtype)) > 0, th.full_like(symkl, th.inf), symkl)
    dot = per_spectrum(p[:, 0] * p[:, 1])
    norms = per_spectrum(p[:, 0] ** 2).sqrt() * per_spectrum(p[:, 1] ** 2).sqrt()
    churn = per_spectrum(th.where(one_sided[:, None], p, th.zeros_like(p)).sum(dim=1)) / 2.0
    top = []
    for column in range(2):
        best = p.new_full((batch_size,), -1.0).scatter_reduce(
            0, group_batch, p[:, column], reduce="amax", include_self=True
        )
        is_best = p[:, column] == best[group_batch]
        # The smallest group id attaining the maximum, so ties resolve identically in both.
        candidates = th.where(is_best, th.arange(num, device=p.device), num)
        top.append(
            th.full((batch_size,), num, device=p.device).scatter_reduce(
                0, group_batch, candidates, reduce="amin", include_self=True
            )
        )
    return {
        "tv": tv.cpu().numpy(),
        "js": per_spectrum(js_terms).cpu().numpy(),
        "cos": (dot / norms.clamp_min(1e-300)).cpu().numpy(),
        "top1_same": (top[0] == top[1]).cpu().numpy(),
        "churn": churn.cpu().numpy(),
        "symkl": symkl.cpu().numpy(),
    }


def spread_statistics(
    capture: dict[str, th.Tensor],
    results: dict[str, Any],
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Logit spread and entropies of the eval prediction, and the target's entropy.

    The valid-fragment log probabilities are the allocation logits ``z_f`` less a per-spectrum
    constant, so their within-spectrum standard deviation is the centered spread the proposed
    penalty acts on. The peak-group log probabilities are its channel-aware variant.
    """
    fragment_log_p = capture["cross_ev_fragment_logprobs"].to(th.float64)
    fragment_batch = capture["cross_ev_fragment_batch_idxs"]

    def per_spectrum_std(values: th.Tensor, segments: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        counts = th.bincount(segments, minlength=batch_size).to(values)
        means = values.new_zeros(batch_size).index_add(0, segments, values) / counts
        squares = values.new_zeros(batch_size).index_add(
            0, segments, (values - means[segments]) ** 2
        )
        return (squares / counts).sqrt(), counts

    fragment_std, fragment_counts = per_spectrum_std(fragment_log_p, fragment_batch)
    # Which fragments carry the penalty mean_f (z_f - mean z)^2: the smallest set holding 99% of
    # the allocation (the core), against the rest. Var_P(z) weights each fragment by its mass.
    fragment_p = fragment_log_p.exp()
    order = th.argsort(-fragment_log_p, stable=True)
    order = order[th.argsort(fragment_batch[order], stable=True)]
    totals = fragment_p.new_zeros(batch_size).index_add(0, fragment_batch, fragment_p)
    starts = th.cumsum(totals, 0) - totals
    sorted_p = fragment_p[order]
    mass_before = th.cumsum(sorted_p, 0) - starts[fragment_batch[order]] - sorted_p
    core = th.empty_like(fragment_batch, dtype=th.bool)
    core[order] = mass_before < 0.99
    means = fragment_p.new_zeros(batch_size).index_add(0, fragment_batch, fragment_log_p)
    squares = (fragment_log_p - means[fragment_batch] / fragment_counts[fragment_batch]) ** 2
    tail_squares = fragment_p.new_zeros(batch_size).index_add(
        0, fragment_batch, th.where(core, th.zeros_like(squares), squares)
    )
    all_squares = fragment_p.new_zeros(batch_size).index_add(0, fragment_batch, squares)
    expected = fragment_p.new_zeros(batch_size).index_add(
        0, fragment_batch, fragment_p * fragment_log_p
    )
    mass_weighted_variance = fragment_p.new_zeros(batch_size).index_add(
        0, fragment_batch, fragment_p * (fragment_log_p - expected[fragment_batch]) ** 2
    )
    core_counts = th.bincount(fragment_batch[core], minlength=batch_size).to(fragment_p)
    batch, mzs, log_weights = dense_pathways(capture)
    ids, num = group_ids(batch, mzs, GROUP_GAP_DA)
    group_log_p = _segment_logsumexp(log_weights, ids, num)
    group_batch = th.zeros(num, dtype=th.long, device=batch.device).scatter_(0, ids, batch)
    group_std, group_counts = per_spectrum_std(group_log_p, group_batch)
    group_p = group_log_p.exp()
    top1 = group_p.new_zeros(batch_size).scatter_reduce(
        0, group_batch, group_p, reduce="amax", include_self=True
    )
    mass = group_p.new_zeros(batch_size).index_add(0, group_batch, group_p)
    return {
        "n_fragments": fragment_counts.cpu().numpy(),
        "n_groups": group_counts.cpu().numpy(),
        "dense_mass": mass.cpu().numpy(),
        "logp_fragment_std": fragment_std.cpu().numpy(),
        "logp_fragment_var_p": mass_weighted_variance.cpu().numpy(),
        "core99_fraction": (core_counts / fragment_counts).cpu().numpy(),
        "penalty_tail_share": (tail_squares / all_squares).cpu().numpy(),
        "logp_group_std": group_std.cpu().numpy(),
        "top1_group_mass": top1.cpu().numpy(),
        "n_target_peaks": th.bincount(results["true_batch_idxs"].reshape(-1), minlength=batch_size)
        .cpu()
        .numpy(),
    }


def binned_entropy(pathways: tuple[th.Tensor, th.Tensor, th.Tensor], batch_size: int) -> np.ndarray:
    """Per-spectrum entropy (nats) of a distribution after summing it into 0.01-Da bins."""
    batch, mzs, log_weights = in_metric_range(pathways)
    bin_ids, bins = group_ids(batch, metric_bins(mzs), None)
    bin_batch = th.zeros(bins, dtype=th.long, device=batch.device).scatter_(0, bin_ids, batch)
    bin_p = log_weights.new_zeros(bins).index_add(0, bin_ids, log_weights.exp())
    bin_p = bin_p / bin_p.new_zeros(batch_size).index_add(0, bin_batch, bin_p)[bin_batch]
    terms = th.where(bin_p > 0, -bin_p * bin_p.log(), th.zeros_like(bin_p))
    return terms.new_zeros(batch_size).index_add(0, bin_batch, terms).cpu().numpy()


def target_pathways(
    results: dict[str, Any], batch_size: int
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """The evaluation target as (spectrum, m/z, log probability), normalized per spectrum."""
    true_batch = results["true_batch_idxs"].reshape(-1)
    true_log_p = results["true_logprobs"].reshape(-1).to(th.float64)
    true_log_p = true_log_p - _segment_logsumexp(true_log_p, true_batch, batch_size)[true_batch]
    # The target keeps its own m/z dtype: the logged metric bins each side in its own precision.
    return true_batch, results["true_mzs"].reshape(-1), true_log_p


def mixture(*parts: tuple[th.Tensor, th.Tensor, th.Tensor]) -> tuple[th.Tensor, ...]:
    """Equal-weight average of several dense predictions of the same spectra."""
    offset = float(np.log(len(parts)))
    return (
        th.cat([part[0] for part in parts]),
        th.cat([part[1] for part in parts]),
        th.cat([part[2] for part in parts]) - offset,
    )


def flattened(
    pathways: tuple[th.Tensor, th.Tensor, th.Tensor], power: float, batch_size: int
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """``P(g)^power`` over exact peak groups, renormalized per spectrum; one row per group."""
    batch, mzs, log_weights = pathways
    ids, num = group_ids(batch, mzs, GROUP_GAP_DA)
    group_log_p = power * _segment_logsumexp(log_weights, ids, num)
    group_batch = th.zeros(num, dtype=th.long, device=batch.device).scatter_(0, ids, batch)
    group_mz = mzs.new_zeros(num).scatter_(0, ids, mzs)
    normalizers = _segment_logsumexp(group_log_p, group_batch, batch_size)
    return group_batch, group_mz, group_log_p - normalizers[group_batch]


def mc_head_sizes(head_views: int) -> list[int]:
    """Numbers of head views averaged in the MC curve: 1, 2, 4, ... up to ``head_views``."""
    if head_views < VIEWS or head_views > 16 or head_views & (head_views - 1):
        raise ValueError(f"--head_views must be a power of two from {VIEWS} to 16")
    return [2**power for power in range(head_views.bit_length())]


def mc_limit_weights(sizes: list[int]) -> np.ndarray:
    """Least-squares intercept weights of ``y_n = S_inf - k / n`` over the given view counts.

    For a score that is smooth in the prediction, averaging ``n`` independent dropout views
    leaves ``1 / n`` of one view's noise variance, so to second order the expected score of the
    ``n``-view average falls on a line in ``1 / n``. Its intercept ``S_inf = sum_n w_n y_n`` is
    the score of the MC-dropout mean, estimated per spectrum with these weights.

    Args:
        sizes: View counts ``n``, at least two distinct.

    Returns:
        Weights ``w_n``, shape ``(len(sizes),)``; they sum to 1 and annihilate ``1 / n``.
    """
    x = 1.0 / np.asarray(sizes, dtype=float)
    centered = x - x.mean()
    return 1.0 / x.size - x.mean() * centered / (centered**2).sum()


def novelty(frame: pd.DataFrame, mol_df: pd.DataFrame, train_mol_ids: np.ndarray) -> pd.Series:
    """Max Morgan-2 Tanimoto of each sampled molecule to any other training molecule."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    smiles = mol_df.set_index("mol_id")["smiles"] if "mol_id" in mol_df else mol_df["smiles"]
    train_ids = np.unique(train_mol_ids)
    train_fps = [generator.GetFingerprint(Chem.MolFromSmiles(smiles[m])) for m in train_ids]
    scores = {}
    for mol_id in frame["mol_id"].unique():
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                generator.GetFingerprint(Chem.MolFromSmiles(smiles[mol_id])), train_fps
            )
        )
        similarities[train_ids == mol_id] = -1.0
        scores[mol_id] = float(similarities.max())
    return frame["mol_id"].map(scores)


def evaluate_split(
    pl_module: nn.Module,
    dataset: Any,
    positions: np.ndarray,
    args: argparse.Namespace,
    device: th.device,
    capture: OutputCapture,
    seed: int,
) -> pd.DataFrame:
    """Run every condition on the sampled spectra of one split; one row per spectrum."""
    loader = DataLoader(
        Subset(dataset, positions.tolist()),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.get_collate_fn(),
    )
    scopes = ("eval", "head", "full")
    head_labels = [f"head{view}" for view in range(args.head_views)]
    conditions = (
        [("eval", 0)]
        + [("head", view) for view in range(args.head_views)]
        + [("full", view) for view in range(VIEWS)]
    )
    frames = []
    with th.inference_mode():
        for batch_index, batch in enumerate(tqdm(loader, desc=dataset.split, unit="batch")):
            batch = pl_module.transfer_batch_to_device(batch, device, dataloader_idx=0)
            batch_size = int(batch["batch_size"])
            outputs = {}
            for scope, view in conditions:
                set_noise_scope(pl_module, scope)
                # One dropout draw per (batch, scope, view), reproducible across reruns.
                th.manual_seed(
                    seed * 1_000_003 + batch_index * 64 + scopes.index(scope) * 16 + view
                )
                results = pl_module._common_step(batch, split=dataset.split, log=False)
                outputs[f"{scope}{view}"] = (results, capture.output)
            eval_results, eval_capture = outputs["eval0"]
            nce = th.full((batch_size,), th.nan, dtype=th.float64)
            nce[batch["spec_nce_batch_idxs"].reshape(-1).cpu()] = (
                batch["spec_nce"].reshape(-1).cpu().to(th.float64)
            )
            row: dict[str, Any] = {
                "split": dataset.split,
                "spec_id": eval_results["unique_id"].reshape(-1).cpu().numpy(),
                "nce": nce.numpy(),
            }
            for label, (results, _) in outputs.items():
                row[f"cos_{label}"] = results[COSINE].reshape(-1).cpu().numpy()
                row[f"loss_{label}"] = results["loss"].reshape(-1).cpu().numpy()
            dense = {label: dense_pathways(output) for label, (_, output) in outputs.items()}
            # The head views should score the eval candidate set. Record where a budget tie gave
            # a view another isomer; the analysis drops those spectra from the head contrasts.
            row["head_support_mismatch"] = np.logical_or.reduce(
                [
                    support_mismatch(dense[label], dense["eval0"], batch_size)
                    for label in head_labels
                ]
            )
            for name, (first, second) in {
                "head": ("head0", "head1"),
                "full": ("full0", "full1"),
                "head_vs_eval": ("head0", "eval0"),
                "full_vs_eval": ("full0", "eval0"),
            }.items():
                for binned in (False, True):
                    distances = pair_distances(dense[first], dense[second], batch_size, binned)
                    suffix = "_bin" if binned else ""
                    for key, values in distances.items():
                        row[f"{name}_{key}{suffix}"] = values
            # Dense 0.01-Da cosine to the target, the logged metric's binning on every valid
            # pathway: each view, the average of the two views of a scope (MC dropout), and the
            # eval prediction and head average flattened as P^power. Averaging adds something
            # beyond flattening only if the flattened average beats the flattened eval.
            target = target_pathways(eval_results, batch_size)
            mc_head = mixture(dense["head0"], dense["head1"])
            # With more than two head views, also the average of the first n views for n = 4,
            # 8, ...: E[cos] ~ S_inf - k / n tests whether two views understate the MC mean.
            mc_sizes = [n for n in mc_head_sizes(args.head_views) if n > VIEWS]
            mc_many = {n: mixture(*(dense[label] for label in head_labels[:n])) for n in mc_sizes}
            predictions = {
                "eval": dense["eval0"],
                **{label: dense[label] for label in (*head_labels, "full0", "full1")},
                "mc_head": mc_head,
                "mc_full": mixture(dense["full0"], dense["full1"]),
                **{
                    f"eval_pow{power}": flattened(dense["eval0"], power, batch_size)
                    for power in FLATTENING_POWERS
                },
                **{
                    f"mc_head_pow{power}": flattened(mc_head, power, batch_size)
                    for power in MC_FLATTENING_POWERS
                },
                **{f"mc_head_n{n}": mc_many[n] for n in mc_sizes},
                **{
                    f"mc_head_n{mc_sizes[-1]}_pow{power}": flattened(
                        mc_many[mc_sizes[-1]], power, batch_size
                    )
                    for power in MC_FLATTENING_POWERS
                    if mc_sizes
                },
            }
            for label, prediction in predictions.items():
                distances = pair_distances(prediction, target, batch_size, binned=True)
                row[f"dcos_{label}"] = distances["cos"]
                if label in ("eval", "head0", "mc_head", "full0"):
                    row[f"entropy_{label}_bin"] = binned_entropy(prediction, batch_size)
                if label == "eval":
                    row["eval_target_tv_bin"] = distances["tv"]
            row["entropy_target_bin"] = binned_entropy(target, batch_size)
            row.update(spread_statistics(eval_capture, eval_results, batch_size))
            frames.append(pd.DataFrame(row))
    frame = pd.concat(frames, ignore_index=True)
    sampled = dataset.spec_df.iloc[positions]
    if not np.array_equal(sampled[dataset.id_key].to_numpy(), frame["spec_id"].to_numpy()):
        raise RuntimeError("batch spectra do not match the sampled spec_df rows")
    frame["mol_id"] = sampled["mol_id"].to_numpy()
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template_fp", default="config")
    parser.add_argument("--custom_fp", help="the run's config; required unless --summarize")
    parser.add_argument("--model_ckpt", help="required unless --summarize")
    parser.add_argument(
        "--weights",
        choices=("state_dict", "ema_training"),
        default="state_dict",
        help="ema_training loads an EMA run's training weights instead of its average",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--num_spectra", type=int, default=1024, help="sampled per split")
    parser.add_argument(
        "--head_views",
        type=int,
        default=VIEWS,
        help="head dropout views per spectrum (power of two up to 16); above 2 also records "
        "the MC-dropout curve over 1, 2, 4, ... views",
    )
    parser.add_argument("--sample_seed", type=int, default=20260927)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_fp", help="per-spectrum pickle; required unless --summarize")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument(
        "--summarize",
        nargs="+",
        default=None,
        help="per-spectrum pickles to summarize instead of running inference",
    )
    parser.add_argument("--bootstrap_draws", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summarize:
        summarize(args.summarize, args.bootstrap_draws)
        return
    missing = [name for name in ("custom_fp", "model_ckpt", "output_fp") if not getattr(args, name)]
    if missing:
        raise ValueError(f"inference needs --{', --'.join(missing)}")
    mc_head_sizes(args.head_views)
    config_d = load_config(args.template_fp, args.custom_fp, args.override)
    if config_d["model_type"] != "masster" or not config_d["masster_unique_fragment_mode"]:
        raise ValueError("this probe needs a unique-fragment MASSter config")
    config_d["auxiliary_scores"] = list(dict.fromkeys([*config_d["auxiliary_scores"], "cos_sim"]))
    if 0.01 not in config_d["eval_mz_bin_res"]:
        raise ValueError(f"{COSINE} needs 0.01 in eval_mz_bin_res")
    if config_d["mz_max"] != MZ_MAX or not config_d["sum_ints"]:
        raise ValueError(f"the dense cosine replicates mz_max {MZ_MAX} with summed bins")
    # The train split's auxiliary metrics are off by default; the probe reads its cosine.
    config_d["compute_train_auxiliary_metrics"] = True
    config_d = apply_inference_config(
        config_d,
        disable_preproc=True,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    device = setup_device(config_d, device=args.device)
    seed_everything(config_d["seed"], workers=True)
    pl_module = init_model_from_ckpt(args.model_ckpt, config_d, device)
    if args.weights == "ema_training":
        checkpoint = th.load(args.model_ckpt, map_location=str(device), weights_only=False)
        state = checkpoint["ema_training_state_dict"]
        missing, unexpected = pl_module.load_state_dict(state, strict=False)
        non_float = {
            name for name, value in pl_module.state_dict().items() if not value.is_floating_point()
        }
        if unexpected or set(missing) - non_float:
            raise RuntimeError(f"training weights do not match: {missing} {unexpected}")
        logger.info("Loaded %d EMA training tensors", len(state))
    for scope in ("eval", "head", "full"):
        logger.info("%s scope dropout sites: %s", scope, set_noise_scope(pl_module, scope))
    capture = OutputCapture(pl_module.model)

    dataset_cls = get_dataset_cls(config_d["model_type"])
    datasets = {split: dataset_cls(split=split, **config_d) for split in {"train", *args.splits}}
    frames = []
    for split in args.splits:
        dataset = datasets[split]
        rng = np.random.default_rng(args.sample_seed + (0 if split == "train" else 1))
        positions = np.sort(rng.choice(len(dataset), size=args.num_spectra, replace=False))
        frame = evaluate_split(
            pl_module, dataset, positions, args, device, capture, args.sample_seed
        )
        frames.append(frame)
        pd.concat(frames, ignore_index=True).to_pickle(args.output_fp)
    frame = pd.concat(frames, ignore_index=True)
    # Each split's dataset keeps only its own molecules, so look SMILES up across all of them.
    mol_df = pd.concat([dataset.mol_df for dataset in datasets.values()])
    mol_df = mol_df[~mol_df.index.duplicated()]
    frame["max_tanimoto"] = novelty(frame, mol_df, datasets["train"].spec_df["mol_id"].to_numpy())
    frame["checkpoint"] = os.path.abspath(args.model_ckpt)
    frame["weights"] = args.weights
    frame.to_pickle(args.output_fp)
    report(frame)
    logger.info("Saved %d rows to %s", len(frame), args.output_fp)


def report(frame: pd.DataFrame) -> None:
    """Log split means of the quantities the R-Drop and logit-penalty gates read."""
    columns = [
        "cos_eval0",
        "cos_head0",
        "cos_full0",
        "eval_target_tv_bin",
        "head_tv",
        "head_tv_bin",
        "head_symkl",
        "head_cos",
        "head_top1_same",
        "head_vs_eval_tv",
        "full_tv_bin",
        "full_churn",
        "full_top1_same",
        "logp_fragment_std",
        "logp_group_std",
        "entropy_eval_bin",
        "entropy_target_bin",
    ]
    logger.info(
        "Spectra whose head views drew another tied isomer (dropped below): %s",
        frame.groupby("split")["head_support_mismatch"].mean().to_dict(),
    )
    kept = frame[~frame["head_support_mismatch"]]
    logger.info("\n%s", kept.groupby("split")[columns].mean().T.to_string(float_format="%.4f"))


SUMMARY_COLUMNS = (
    "cos_eval0",
    "eval_target_tv_bin",
    "head_tv",
    "head_symkl",
    "head_cos",
    "head_top1_same",
    "head_vs_eval_tv",
    "head_cos_cost",
    "full_tv_bin",
    "full_churn",
    "full_top1_same",
    "full_cos_cost",
    "logp_fragment_std",
    "logp_fragment_var_p",
    "core99_fraction",
    "penalty_tail_share",
    "logp_group_std",
    "entropy_eval_bin",
    "entropy_target_bin",
    "n_fragments",
)


def molecule_bootstrap(
    values: np.ndarray, molecules: np.ndarray, draws: int, rng: np.random.Generator
) -> np.ndarray:
    """Spectrum-weighted means of ``values`` over molecule resamples, shape ``(draws,)``."""
    codes, inverse = np.unique(molecules, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=codes.size)
    counts = np.bincount(inverse, minlength=codes.size).astype(float)
    picks = rng.integers(0, codes.size, size=(draws, codes.size))
    return sums[picks].sum(axis=1) / counts[picks].sum(axis=1)


def summarize(paths: list[str], draws: int) -> None:
    """Log the gate tables from saved per-spectrum pickles; no inference.

    Means are spectrum-weighted with 95% molecule-bootstrap intervals. Spectra whose head views
    drew another tied isomer are dropped. ``*_cos_cost`` is the eval cosine minus one dropout
    view's: what a single training-noise draw costs on the logged metric.
    """
    rng = np.random.default_rng(0)
    frames = {}
    for path in paths:
        frame = pd.read_pickle(path)
        frame = frame[~frame["head_support_mismatch"]].copy()
        frame["head_cos_cost"] = frame["cos_eval0"] - frame["cos_head0"]
        frame["full_cos_cost"] = frame["cos_eval0"] - frame["cos_full0"]
        frame["error"] = 1.0 - frame["cos_eval0"]
        frames[os.path.splitext(os.path.basename(path))[0]] = frame

    # 1. Split means with molecule-clustered intervals, and the train - val difference.
    rows = []
    for name, frame in frames.items():
        for column in SUMMARY_COLUMNS:
            record = {"checkpoint": name, "metric": column}
            boots = {}
            for split, part in frame.groupby("split"):
                values = part[column].to_numpy(dtype=float)
                boots[split] = molecule_bootstrap(values, part["mol_id"].to_numpy(), draws, rng)
                low, high = np.percentile(boots[split], [2.5, 97.5])
                record[split] = f"{values.mean():.4f} [{low:.4f}, {high:.4f}]"
            if {"train", "val"} <= set(boots):
                difference = boots["train"] - boots["val"]
                low, high = np.percentile(difference, [2.5, 97.5])
                record["train - val"] = f"{difference.mean():+.4f} [{low:+.4f}, {high:+.4f}]"
            rows.append(record)
    table = pd.DataFrame(rows).set_index(["checkpoint", "metric"])
    logger.info("Split means [95%% molecule bootstrap]\n%s", table.to_string())

    # 2. Does the dropout disagreement track the error, within each split?
    rows = []
    for name, frame in frames.items():
        for split, part in frame.groupby("split"):
            rows.append(
                {
                    "checkpoint": name,
                    "split": split,
                    "rho(head_tv, 1 - cos)": part["head_tv"].corr(part["error"], method="spearman"),
                    "rho(head_tv, target tv)": part["head_tv"].corr(
                        part["eval_target_tv_bin"], method="spearman"
                    ),
                    "rho(head_tv, n_fragments)": part["head_tv"].corr(
                        part["n_fragments"], method="spearman"
                    ),
                }
            )
    logger.info("Spearman correlations\n%s", pd.DataFrame(rows).to_string(float_format="%.3f"))

    # 3. Strata: novelty (max Tanimoto to another training molecule) and NCE.
    for name, frame in frames.items():
        strata = frame.assign(
            novelty=pd.cut(frame["max_tanimoto"], [-1.0, 0.4, 0.6, 0.8, 1.01]),
            nce_bin=pd.cut(frame["nce"], [0, 30, 60, 100, 1000], right=False),
        )
        for key in ("novelty", "nce_bin"):
            grouped = strata.groupby(["split", key], observed=True)
            logger.info(
                "%s by %s\n%s",
                name,
                key,
                grouped[["head_tv", "head_symkl", "full_tv_bin", "cos_eval0", "logp_group_std"]]
                .mean()
                .join(grouped.size().rename("n"))
                .to_string(float_format="%.4f"),
            )

    # 4. Dense 0.01-Da cosine of each prediction variant, paired against the dense eval one.
    variants = [
        "head0",
        "head1",
        "mc_head",
        "full0",
        "full1",
        "mc_full",
        *(f"eval_pow{power}" for power in FLATTENING_POWERS),
        *(f"mc_head_pow{power}" for power in MC_FLATTENING_POWERS),
    ]
    rows = []
    for name, frame in frames.items():
        for split, part in frame.groupby("split"):
            record = {
                "checkpoint": name,
                "split": split,
                "dcos_eval": f"{part['dcos_eval'].mean():.4f}",
            }
            for variant in variants:
                difference = (part[f"dcos_{variant}"] - part["dcos_eval"]).to_numpy()
                boots = molecule_bootstrap(difference, part["mol_id"].to_numpy(), draws, rng)
                low, high = np.percentile(boots, [2.5, 97.5])
                record[variant] = f"{difference.mean():+.4f} [{low:+.4f}, {high:+.4f}]"
            rows.append(record)
    logger.info(
        "Dense cosine to target, variant - eval [95%% molecule bootstrap]\n%s",
        pd.DataFrame(rows).set_index(["checkpoint", "split"]).T.to_string(),
    )

    # 4b. MC-dropout curve, for pickles recorded with more than two head views. Single views are
    # averaged over all views; n views are the first n. S_inf is the 1/n-line intercept over every
    # n, and "last two" is 2 y(N) - y(N/2): the two agree only if the curve is linear in 1/n.
    rows = []
    for name, frame in frames.items():
        head_views = sum(column.startswith("dcos_head") for column in frame.columns)
        if head_views <= VIEWS:
            continue
        sizes = mc_head_sizes(head_views)
        weights = mc_limit_weights(sizes)
        val_means = frame.loc[frame["split"] == "val"].mean(numeric_only=True)
        best_power = max(
            (f"dcos_eval_pow{power}" for power in FLATTENING_POWERS), key=val_means.__getitem__
        )
        best_mc_power = max(
            (f"dcos_mc_head_n{sizes[-1]}_pow{power}" for power in MC_FLATTENING_POWERS),
            key=val_means.__getitem__,
        )
        for split, part in frame.groupby("split"):
            curve = {
                1: part[[f"dcos_head{view}" for view in range(head_views)]].mean(axis=1),
                2: part["dcos_mc_head"],
                **{n: part[f"dcos_mc_head_n{n}"] for n in sizes if n > VIEWS},
            }
            limit = sum(weight * curve[n] for weight, n in zip(weights, sizes))
            cells = {
                **{f"n={n}": curve[n] - part["dcos_eval"] for n in sizes},
                "S_inf": limit - part["dcos_eval"],
                "S_inf (last two)": 2 * curve[sizes[-1]] - curve[sizes[-2]] - part["dcos_eval"],
                f"S_inf - {best_power[5:]}": limit - part[best_power],
                f"{best_mc_power[5:]} - {best_power[5:]}": part[best_mc_power] - part[best_power],
            }
            record = {"checkpoint": name, "split": split}
            for label, values in cells.items():
                boots = molecule_bootstrap(
                    values.to_numpy(dtype=float), part["mol_id"].to_numpy(), draws, rng
                )
                low, high = np.percentile(boots, [2.5, 97.5])
                record[label] = f"{values.mean():+.4f} [{low:+.4f}, {high:+.4f}]"
            rows.append(record)
    if rows:
        logger.info(
            "MC-dropout head curve, cosine - eval [95%% molecule bootstrap]; best eval flattening "
            "chosen on val\n%s",
            pd.DataFrame(rows).set_index(["checkpoint", "split"]).T.to_string(),
        )

    # 5. Paired checkpoint contrasts on the same spectra.
    names = list(frames)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            merged = frames[first].merge(
                frames[second], on=["split", "spec_id", "mol_id"], suffixes=("_a", "_b")
            )
            for split, part in merged.groupby("split"):
                cells = []
                for column in ("head_tv", "full_tv_bin", "cos_eval0", "logp_group_std"):
                    difference = (part[f"{column}_a"] - part[f"{column}_b"]).to_numpy()
                    boots = molecule_bootstrap(difference, part["mol_id"].to_numpy(), draws, rng)
                    low, high = np.percentile(boots, [2.5, 97.5])
                    cells.append(f"{column} {difference.mean():+.4f} [{low:+.4f}, {high:+.4f}]")
                logger.info(
                    "%s - %s, %s (n=%d): %s", first, second, split, len(part), "; ".join(cells)
                )


if __name__ == "__main__":
    main()
