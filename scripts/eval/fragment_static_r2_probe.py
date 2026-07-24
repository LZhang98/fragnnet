"""Compare a tied static radius-2 cut representation with a k=1 bond GNN.

This is the follow-up experiment proposed in
``docs/fragment_bond_set_representation.md`` §7.4. Both arms use the same
alias-free sibling-ranking cache and receive only their structural readout plus
collision energy; fragment composition is deliberately omitted.

``static_r2`` assigns each cut bond the exact undirected categorical key used
by gate §6.3: ring membership of the central bond and the sorted pair of
endpoint environments out to two bonds. Identical keys share one learned
embedding across molecules. Validation-only keys map to one unknown embedding.

``bond_k1`` is the measured fallback: a one-layer molecule GNN followed by the
non-degenerate directed cut-bond encoder.

Usage:
    python scripts/eval/fragment_static_r2_probe.py \
        --train cache_train.pt \
        --val cache_val.pt
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import torch as th
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FeatureKey: TypeAlias = tuple[float, ...]
AtomEnvKey: TypeAlias = tuple[FeatureKey, tuple]
BondEnvKey: TypeAlias = tuple[FeatureKey, AtomEnvKey, AtomEnvKey]

# The cache builder uses, in order:
# a_onehot(18), a_degree(14), a_hybrid(6), a_formal(6), a_radical(6),
# a_ring(4), a_mass(1), a_chiral(5), a_ring_size(5), random-walk PE(16).
# Gate §6.3 defined an atom environment by element, formal charge and aromaticity.
R2_ATOM_FEATURE_INDICES = tuple(range(0, 18)) + tuple(range(38, 44)) + (52, 53)
# Gate §6.3 uses bond type on environment edges, but only ring membership on
# the central cut bond.
R2_ENV_BOND_FEATURE_INDICES = tuple(range(0, 5))
R2_CENTRAL_BOND_FEATURE_INDICES = (5, 6)


def _feature_key(
    row: np.ndarray,
    feature_indices: Sequence[int] | None = None,
) -> FeatureKey:
    """Convert one static feature row to a stable, hashable key."""
    if feature_indices is not None:
        row = row[np.asarray(feature_indices)]
    return tuple(float(value) for value in row)


def _molecule_adjacency(
    record: dict,
    environment_bond_feature_indices: Sequence[int] | None,
) -> list[list[tuple[int, FeatureKey]]]:
    """Build a directed adjacency list with static bond-feature keys."""
    num_atoms = record["atom_x"].shape[0]
    adjacency: list[list[tuple[int, FeatureKey]]] = [[] for _ in range(num_atoms)]
    for edge_idx, (source, target) in enumerate(record["bond_index"].T):
        adjacency[int(source)].append(
            (
                int(target),
                _feature_key(
                    record["bond_x"][edge_idx],
                    environment_bond_feature_indices,
                ),
            )
        )
    return adjacency


def _rooted_atom_env(
    atom_idx: int,
    blocked_idx: int,
    radius: int,
    atom_keys: Sequence[FeatureKey],
    adjacency: Sequence[Sequence[tuple[int, FeatureKey]]],
    cache: dict[tuple[int, int, int], AtomEnvKey],
) -> AtomEnvKey:
    """Return a rooted static atom environment, excluding the cut direction."""
    cache_key = (atom_idx, blocked_idx, radius)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    children = []
    if radius > 0:
        for neighbor_idx, bond_key in adjacency[atom_idx]:
            if neighbor_idx == blocked_idx:
                continue
            child_env = _rooted_atom_env(
                neighbor_idx,
                atom_idx,
                radius - 1,
                atom_keys,
                adjacency,
                cache,
            )
            children.append((bond_key, child_env))
        children.sort()

    result: AtomEnvKey = (atom_keys[atom_idx], tuple(children))
    cache[cache_key] = result
    return result


def bond_environment_keys(
    record: dict,
    radius: int = 2,
    atom_feature_indices: Sequence[int] | None = None,
    environment_bond_feature_indices: Sequence[int] | None = None,
    central_bond_feature_indices: Sequence[int] | None = None,
) -> list[BondEnvKey]:
    """Compute one undirected, static radius-N key per cached boundary bond.

    The returned list is aligned with ``record["bp_in"]`` / ``record["bp_out"]``.
    """
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")

    atom_keys = [
        _feature_key(row, atom_feature_indices) for row in record["atom_x"]
    ]
    adjacency = _molecule_adjacency(
        record, environment_bond_feature_indices
    )
    edge_features = {
        (int(source), int(target)): _feature_key(
            record["bond_x"][edge_idx], central_bond_feature_indices
        )
        for edge_idx, (source, target) in enumerate(record["bond_index"].T)
    }
    cache: dict[tuple[int, int, int], AtomEnvKey] = {}
    keys = []
    for retained_idx, lost_idx in zip(
        record["bp_in"], record["bp_out"], strict=True
    ):
        retained = int(retained_idx)
        lost = int(lost_idx)
        try:
            bond_key = edge_features[(retained, lost)]
        except KeyError as exc:
            raise ValueError(
                f"boundary pair ({retained}, {lost}) is absent from the molecule graph"
            ) from exc
        retained_env = _rooted_atom_env(
            retained, lost, radius, atom_keys, adjacency, cache
        )
        lost_env = _rooted_atom_env(
            lost, retained, radius, atom_keys, adjacency, cache
        )
        endpoint_envs = tuple(sorted((retained_env, lost_env)))
        keys.append((bond_key, endpoint_envs[0], endpoint_envs[1]))
    return keys


def attach_static_r2_buckets(
    train_records: list[dict],
    val_records: list[dict],
    radius: int = 2,
    atom_feature_indices: Sequence[int] | None = None,
    environment_bond_feature_indices: Sequence[int] | None = None,
    central_bond_feature_indices: Sequence[int] | None = None,
) -> dict[str, float]:
    """Fit a train-only environment vocabulary and attach bucket IDs in place.

    Bucket zero is reserved for validation environments absent from training.
    """
    train_keys = [
        bond_environment_keys(
            record,
            radius,
            atom_feature_indices,
            environment_bond_feature_indices,
            central_bond_feature_indices,
        )
        for record in train_records
    ]
    support = Counter(key for record_keys in train_keys for key in record_keys)
    vocabulary = {
        key: bucket_id
        for bucket_id, key in enumerate(sorted(support), start=1)
    }

    for record, record_keys in zip(train_records, train_keys, strict=True):
        record["static_r2_bucket"] = np.asarray(
            [vocabulary[key] for key in record_keys], dtype=np.int64
        )

    num_val_bonds = 0
    num_unknown = 0
    for record in val_records:
        record_keys = bond_environment_keys(
            record,
            radius,
            atom_feature_indices,
            environment_bond_feature_indices,
            central_bond_feature_indices,
        )
        bucket_ids = np.asarray(
            [vocabulary.get(key, 0) for key in record_keys], dtype=np.int64
        )
        record["static_r2_bucket"] = bucket_ids
        num_val_bonds += len(bucket_ids)
        num_unknown += int((bucket_ids == 0).sum())

    supports = np.asarray(list(support.values()), dtype=np.int64)
    return {
        "num_buckets": float(len(vocabulary)),
        "mean_train_support": float(supports.mean()) if len(supports) else 0.0,
        "median_train_support": float(np.median(supports)) if len(supports) else 0.0,
        "singleton_fraction": (
            float((supports == 1).mean()) if len(supports) else 0.0
        ),
        "val_unknown_fraction": num_unknown / max(num_val_bonds, 1),
    }


def _segment_mean(
    source: th.Tensor,
    segment_idx: th.Tensor,
    num_segments: int,
) -> th.Tensor:
    """Mean-pool rows into segments."""
    output = th.zeros(
        num_segments,
        source.shape[1],
        device=source.device,
        dtype=source.dtype,
    ).index_add_(0, segment_idx, source)
    counts = th.zeros(
        num_segments, device=source.device, dtype=source.dtype
    ).index_add_(
        0,
        segment_idx,
        th.ones(segment_idx.shape[0], device=source.device, dtype=source.dtype),
    )
    return output / counts.clamp(min=1.0).unsqueeze(1)


class MolGNN(nn.Module):
    """One-layer molecule GNN used by the measured learned fallback."""

    def __init__(self, atom_dim: int, bond_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.atom_projection = nn.Linear(atom_dim, hidden_dim)
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + bond_dim, hidden_dim),
            nn.SiLU(),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        atom_features: th.Tensor,
        bond_index: th.Tensor,
        bond_features: th.Tensor,
    ) -> th.Tensor:
        atom_embeddings = self.atom_projection(atom_features)
        source, target = bond_index
        messages = self.message(
            th.cat(
                [
                    atom_embeddings[source],
                    atom_embeddings[target],
                    bond_features,
                ],
                dim=-1,
            )
        )
        aggregate = th.zeros_like(atom_embeddings).index_add_(0, target, messages)
        return self.update(th.cat([atom_embeddings, aggregate], dim=-1))


class CutBondEncoder(nn.Module):
    """Non-degenerate encoding of a directed pair of endpoint embeddings."""

    def __init__(self, atom_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3 * atom_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        retained: th.Tensor,
        lost: th.Tensor,
    ) -> th.Tensor:
        return self.network(th.cat([retained, lost, retained * lost], dim=-1))


class FragmentScorer(nn.Module):
    """Score sibling fragments using one structural representation and NCE."""

    def __init__(
        self,
        arm: str,
        atom_dim: int,
        bond_dim: int,
        num_static_buckets: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if arm not in {"static_r2", "bond_k1"}:
            raise ValueError(f"unknown arm {arm!r}")
        self.arm = arm
        if arm == "static_r2":
            self.static_embedding = nn.Embedding(
                num_static_buckets + 1, hidden_dim
            )
        else:
            self.gnn = MolGNN(atom_dim, bond_dim, hidden_dim)
            self.cut_encoder = CutBondEncoder(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict) -> th.Tensor:
        num_fragments = int(batch["nce"].shape[0])
        if self.arm == "static_r2":
            bond_embeddings = self.static_embedding(batch["static_r2_bucket"])
        else:
            atom_embeddings = self.gnn(
                batch["atom_x"], batch["bond_index"], batch["bond_x"]
            )
            bond_embeddings = self.cut_encoder(
                atom_embeddings[batch["bp_in"]],
                atom_embeddings[batch["bp_out"]],
            )
        fragment_embeddings = _segment_mean(
            bond_embeddings, batch["bp_frag"], num_fragments
        )
        inputs = th.cat([fragment_embeddings, batch["nce"].unsqueeze(1)], dim=-1)
        return self.head(inputs).squeeze(-1)


def _concatenate(
    arrays: Sequence[np.ndarray],
    dtype: th.dtype,
    device: str,
) -> th.Tensor:
    return th.as_tensor(np.concatenate(arrays), dtype=dtype, device=device)


def collate(
    records: list[dict],
    group_indices: list[tuple[int, int]],
    device: str,
) -> dict:
    """Collate selected sibling groups, retaining only probe inputs and targets."""
    atom_features = []
    bond_indices = []
    bond_features = []
    bp_in = []
    bp_out = []
    bp_frag = []
    static_buckets = []
    nce = []
    observed = []
    group_ids = []
    weights = []
    atom_offset = 0
    fragment_offset = 0
    groups_by_record: dict[int, list[int]] = {}
    for record_idx, group_idx in group_indices:
        groups_by_record.setdefault(record_idx, []).append(group_idx)

    num_groups = 0
    for record_idx, selected_groups in groups_by_record.items():
        record = records[record_idx]
        atom_features.append(record["atom_x"])
        bond_indices.append(record["bond_index"] + atom_offset)
        bond_features.append(record["bond_x"])
        for group_idx in selected_groups:
            group = record["groups"][group_idx]
            children = group["children"]
            num_children = len(children)
            for child_offset, node_idx in enumerate(children):
                mask = record["bp_frag"] == node_idx
                bp_in.append(record["bp_in"][mask] + atom_offset)
                bp_out.append(record["bp_out"][mask] + atom_offset)
                bp_frag.append(
                    np.full(
                        int(mask.sum()),
                        fragment_offset + child_offset,
                        dtype=np.int64,
                    )
                )
                static_buckets.append(record["static_r2_bucket"][mask])
            nce.append(
                np.full(num_children, group["nce"], dtype=np.float32)
            )
            observed.append(group["q_obs"])
            group_ids.append(
                np.full(num_children, num_groups, dtype=np.int64)
            )
            weights.append(np.float32(group["weight"]))
            fragment_offset += num_children
            num_groups += 1
        atom_offset += record["atom_x"].shape[0]

    return {
        "atom_x": _concatenate(atom_features, th.float32, device),
        "bond_index": th.as_tensor(
            np.concatenate(bond_indices, axis=1),
            dtype=th.long,
            device=device,
        ),
        "bond_x": _concatenate(bond_features, th.float32, device),
        "bp_in": _concatenate(bp_in, th.long, device),
        "bp_out": _concatenate(bp_out, th.long, device),
        "bp_frag": _concatenate(bp_frag, th.long, device),
        "static_r2_bucket": _concatenate(static_buckets, th.long, device),
        "nce": _concatenate(nce, th.float32, device),
        "q_obs": _concatenate(observed, th.float32, device),
        "group_id": _concatenate(group_ids, th.long, device),
        "weight": th.as_tensor(
            np.asarray(weights), dtype=th.float32, device=device
        ),
        "num_groups": num_groups,
    }


def _group_log_softmax(
    scores: th.Tensor,
    group_ids: th.Tensor,
    num_groups: int,
) -> th.Tensor:
    maxima = th.full(
        (num_groups,), -1e30, device=scores.device
    ).index_reduce_(0, group_ids, scores, "amax", include_self=True)
    exponentials = (scores - maxima[group_ids]).exp()
    denominators = th.zeros(
        num_groups, device=scores.device
    ).index_add_(0, group_ids, exponentials)
    return scores - maxima[group_ids] - denominators.log()[group_ids]


def evaluate(
    model: FragmentScorer,
    batches: list[dict],
) -> dict[str, float]:
    """Return intensity-weighted KL, TV, and dominant-sibling mismatch."""
    model.eval()
    kl_sum = 0.0
    tv_sum = 0.0
    dominant_match_sum = 0.0
    weight_sum = 0.0
    with th.no_grad():
        for batch in batches:
            log_prediction = _group_log_softmax(
                model(batch), batch["group_id"], batch["num_groups"]
            )
            prediction = log_prediction.exp()
            group_weights = batch["weight"]
            row_weights = group_weights[batch["group_id"]]
            observed = batch["q_obs"]
            kl_sum += float(
                (
                    row_weights
                    * observed
                    * (observed.clamp(min=1e-9).log() - log_prediction)
                ).sum()
            )
            group_tv = th.zeros(
                batch["num_groups"], device=prediction.device
            ).index_add_(
                0, batch["group_id"], (prediction - observed).abs()
            )
            tv_sum += float((0.5 * group_tv * group_weights).sum())

            predicted_top = th.full(
                (batch["num_groups"],), -1e30, device=prediction.device
            ).index_reduce_(
                0,
                batch["group_id"],
                prediction,
                "amax",
                include_self=True,
            )
            observed_top = th.full(
                (batch["num_groups"],), -1e30, device=prediction.device
            ).index_reduce_(
                0,
                batch["group_id"],
                observed,
                "amax",
                include_self=True,
            )
            matches = th.zeros(
                batch["num_groups"], device=prediction.device
            ).index_add_(
                0,
                batch["group_id"],
                (
                    (prediction >= predicted_top[batch["group_id"]] - 1e-9)
                    & (observed >= observed_top[batch["group_id"]] - 1e-9)
                ).float(),
            )
            dominant_match_sum += float(
                (matches.clamp(max=1.0) * group_weights).sum()
            )
            weight_sum += float(group_weights.sum())

    return {
        "kl": kl_sum / weight_sum,
        "tv": tv_sum / weight_sum,
        "dom_mismatch": 1.0 - dominant_match_sum / weight_sum,
    }


def _make_batches(
    records: list[dict],
    group_indices: list[tuple[int, int]],
    batch_groups: int,
    device: str,
) -> list[dict]:
    return [
        collate(records, group_indices[start : start + batch_groups], device)
        for start in range(0, len(group_indices), batch_groups)
    ]


def run_arm(
    arm: str,
    train_records: list[dict],
    val_batches: list[dict],
    num_static_buckets: int,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, float]:
    """Train one arm at the original gate-4 budget and return its metrics."""
    th.manual_seed(seed)
    np.random.seed(seed)
    model = FragmentScorer(
        arm=arm,
        atom_dim=train_records[0]["atom_x"].shape[1],
        bond_dim=train_records[0]["bond_x"].shape[1],
        num_static_buckets=num_static_buckets,
        hidden_dim=args.hidden,
    ).to(args.device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
    all_groups = [
        (record_idx, group_idx)
        for record_idx, record in enumerate(train_records)
        for group_idx in range(len(record["groups"]))
    ]
    random = np.random.default_rng(seed)
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        random.shuffle(all_groups)
        loss_sum = 0.0
        num_batches = 0
        for start in range(0, len(all_groups), args.batch_groups):
            batch = collate(
                train_records,
                all_groups[start : start + args.batch_groups],
                args.device,
            )
            log_prediction = _group_log_softmax(
                model(batch), batch["group_id"], batch["num_groups"]
            )
            loss = -(
                batch["weight"][batch["group_id"]]
                * batch["q_obs"]
                * log_prediction
            ).sum() / batch["weight"].sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss)
            num_batches += 1
        logger.info(
            "[%s seed=%d] epoch %d train_loss=%.4f",
            arm,
            seed,
            epoch,
            loss_sum / max(num_batches, 1),
        )

    val_metrics = evaluate(model, val_batches)
    train_sample = all_groups[: len(val_batches) * args.batch_groups]
    train_batches = _make_batches(
        train_records, train_sample, args.batch_groups, args.device
    )
    train_metrics = evaluate(model, train_batches)
    return {
        "train_tv": train_metrics["tv"],
        "tv": val_metrics["tv"],
        "gap_tv": val_metrics["tv"] - train_metrics["tv"],
        "train_kl": train_metrics["kl"],
        "kl": val_metrics["kl"],
        "gap_kl": val_metrics["kl"] - train_metrics["kl"],
        "dom_mismatch": val_metrics["dom_mismatch"],
        "params": float(sum(parameter.numel() for parameter in model.parameters())),
        "train_minutes": (time.time() - start_time) / 60.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument(
        "--arms", nargs="+", choices=["static_r2", "bond_k1"],
        default=["static_r2", "bond_k1"]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_groups", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device", default="cuda" if th.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--audit_only",
        action="store_true",
        help="Build and report the train-only bucket vocabulary without training.",
    )
    args = parser.parse_args()

    train_records = th.load(args.train, weights_only=False)["records"]
    val_records = th.load(args.val, weights_only=False)["records"]
    atom_dim = train_records[0]["atom_x"].shape[1]
    bond_dim = train_records[0]["bond_x"].shape[1]
    if atom_dim <= max(R2_ATOM_FEATURE_INDICES):
        raise ValueError(
            f"cache atom_dim={atom_dim} does not match the gate-4 feature layout"
        )
    required_bond_index = max(
        R2_ENV_BOND_FEATURE_INDICES + R2_CENTRAL_BOND_FEATURE_INDICES
    )
    if bond_dim <= required_bond_index:
        raise ValueError(
            f"cache bond_dim={bond_dim} does not match the gate-4 feature layout"
        )
    vocabulary_stats = attach_static_r2_buckets(
        train_records,
        val_records,
        atom_feature_indices=R2_ATOM_FEATURE_INDICES,
        environment_bond_feature_indices=R2_ENV_BOND_FEATURE_INDICES,
        central_bond_feature_indices=R2_CENTRAL_BOND_FEATURE_INDICES,
    )
    num_static_buckets = int(vocabulary_stats["num_buckets"])
    logger.info(
        "static-r2 vocabulary: %d buckets, mean support %.2f, median support %.1f, "
        "%.1f%% singletons, %.1f%% unknown validation bonds",
        num_static_buckets,
        vocabulary_stats["mean_train_support"],
        vocabulary_stats["median_train_support"],
        100 * vocabulary_stats["singleton_fraction"],
        100 * vocabulary_stats["val_unknown_fraction"],
    )
    if args.audit_only:
        return

    val_groups = [
        (record_idx, group_idx)
        for record_idx, record in enumerate(val_records)
        for group_idx in range(len(record["groups"]))
    ]
    val_batches = _make_batches(
        val_records, val_groups, args.batch_groups, args.device
    )

    results = {
        arm: [
            run_arm(
                arm,
                train_records,
                val_batches,
                num_static_buckets,
                args,
                seed,
            )
            for seed in args.seeds
        ]
        for arm in args.arms
    }

    print(
        f"\n{'arm':<12s} {'trainTV':>9s} {'valTV':>9s} {'gapTV':>9s} "
        f"{'trainKL':>9s} {'valKL':>9s} {'gapKL':>9s} {'dom_mis':>15s}"
    )
    for arm, metrics in results.items():
        mean = {
            key: float(np.mean([run[key] for run in metrics]))
            for key in metrics[0]
        }
        dom_std = float(np.std([run["dom_mismatch"] for run in metrics]))
        print(
            f"{arm:<12s} {mean['train_tv']:>9.4f} {mean['tv']:>9.4f} "
            f"{mean['gap_tv']:>9.4f} {mean['train_kl']:>9.4f} "
            f"{mean['kl']:>9.4f} {mean['gap_kl']:>9.4f} "
            f"{mean['dom_mismatch']:>7.4f} +/-{dom_std:.4f}"
        )


if __name__ == "__main__":
    main()
