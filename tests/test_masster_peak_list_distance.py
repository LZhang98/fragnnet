"""Tests for direct, non-Gaussian MASSter peak-list objectives."""

import math

import pytest
import torch as th

from fragnnet.loss.padding import build_loss_chunks
from fragnnet.loss.peak_list import sparse_peak_list_distance
from fragnnet.utils.misc_utils import TOLERANCE_MIN_MZ
from fragnnet.utils.spec_utils import (
    calculate_match_mzs,
    linear_sum_assignment,
    round_aggregate_peaks,
)


@pytest.mark.parametrize(
    "objective",
    ["cosine_distance", "hellinger_distance", "jss_distance", "kl_distance"],
)
@pytest.mark.parametrize("matching", ["closest", "hungarian"])
def test_identical_spectra_have_zero_loss(objective: str, matching: str) -> None:
    mzs = th.tensor([100.0, 150.0])
    intensities = th.tensor([1.0, 3.0])
    batch_indices = th.zeros(2, dtype=th.long)

    primary, entropy = sparse_peak_list_distance(
        mzs,
        intensities,
        batch_indices,
        mzs,
        intensities,
        batch_indices,
        objective=objective,
        matching=matching,
        tolerance=0.01,
        relative=False,
    )

    assert th.allclose(primary, th.zeros(1), atol=1e-7)
    assert th.allclose(entropy, th.zeros(1), atol=1e-7)


def test_closest_matching_sums_many_predictions_at_one_observed_peak() -> None:
    true_mzs = th.tensor([100.0, 150.0])
    true_intensities = th.tensor([3.0, 1.0])
    pred_mzs = th.tensor([99.996, 100.004, 150.0])
    pred_intensities = th.tensor([1.0, 2.0, 1.0], requires_grad=True)
    true_batch_indices = th.zeros(2, dtype=th.long)
    pred_batch_indices = th.zeros(3, dtype=th.long)

    primary, _ = sparse_peak_list_distance(
        true_mzs,
        true_intensities,
        true_batch_indices,
        pred_mzs,
        pred_intensities,
        pred_batch_indices,
        objective="cosine_distance",
        matching="closest",
        tolerance=0.01,
        relative=False,
    )
    primary.sum().backward()

    assert th.allclose(primary, th.zeros(1), atol=1e-7)
    assert pred_intensities.grad is not None
    assert th.isfinite(pred_intensities.grad).all()


def test_hellinger_has_zero_safe_structural_support_and_recovery_gradient() -> None:
    logits = th.tensor([-10.0, 0.0], dtype=th.float64, requires_grad=True)
    pred_intensities = th.nn.functional.softplus(logits)

    primary, _ = sparse_peak_list_distance(
        th.tensor([100.0], dtype=th.float64),
        th.tensor([1.0], dtype=th.float64),
        th.zeros(1, dtype=th.long),
        th.tensor([100.0, 150.0], dtype=th.float64),
        pred_intensities,
        th.zeros(2, dtype=th.long),
        objective="hellinger_distance",
        matching="closest",
        tolerance=0.01,
        relative=False,
    )
    primary.sum().backward()

    th.testing.assert_close(primary, th.tensor([0.9919072577443437], dtype=th.float64))
    assert logits.grad is not None
    assert th.isfinite(logits.grad).all()
    assert logits.grad[0] == pytest.approx(-0.004046014277967066)


def test_hellinger_zero_predicted_coordinate_has_finite_backward() -> None:
    pred_intensities = th.tensor([0.0, 1.0], requires_grad=True)
    primary, _ = sparse_peak_list_distance(
        th.tensor([100.0]),
        th.tensor([1.0]),
        th.zeros(1, dtype=th.long),
        th.tensor([100.0, 150.0]),
        pred_intensities,
        th.zeros(2, dtype=th.long),
        objective="hellinger_distance",
        matching="closest",
        tolerance=0.01,
        relative=False,
    )
    primary.sum().backward()

    th.testing.assert_close(primary, th.ones(1))
    assert pred_intensities.grad is not None
    assert th.isfinite(pred_intensities.grad).all()


def test_hungarian_is_one_to_one_and_mz_only() -> None:
    true_mzs = th.tensor([100.0, 150.0])
    true_intensities = th.tensor([1.0, 1.0])
    pred_mzs = th.tensor([99.999, 100.002, 150.0])
    pred_batch_indices = th.zeros(3, dtype=th.long)
    true_batch_indices = th.zeros(2, dtype=th.long)

    low_first, _ = sparse_peak_list_distance(
        true_mzs,
        true_intensities,
        true_batch_indices,
        pred_mzs,
        th.tensor([0.1, 100.0, 1.0]),
        pred_batch_indices,
        objective="cosine_distance",
        matching="hungarian",
        tolerance=0.01,
        relative=False,
    )
    high_first, _ = sparse_peak_list_distance(
        true_mzs,
        true_intensities,
        true_batch_indices,
        pred_mzs,
        th.tensor([100.0, 0.1, 1.0]),
        pred_batch_indices,
        objective="cosine_distance",
        matching="hungarian",
        tolerance=0.01,
        relative=False,
    )

    assert low_first > high_first


def test_four_decimal_clustering_sums_intensity_and_keeps_gradient() -> None:
    true_mzs = th.tensor([100.0, 150.0])
    true_intensities = th.tensor([3.0, 1.0])
    pred_mzs = th.tensor([100.00001, 100.00002, 150.0])
    pred_intensities = th.tensor([1.0, 2.0, 1.0], requires_grad=True)
    true_batch_indices = th.zeros(2, dtype=th.long)
    pred_batch_indices = th.zeros(3, dtype=th.long)

    primary, _ = sparse_peak_list_distance(
        true_mzs,
        true_intensities,
        true_batch_indices,
        pred_mzs,
        pred_intensities,
        pred_batch_indices,
        objective="jss_distance",
        matching="closest",
        tolerance=0.01,
        relative=False,
        decimals=4,
    )
    primary.sum().backward()

    assert th.allclose(primary, th.zeros(1), atol=1e-7)
    assert pred_intensities.grad is not None
    assert th.isfinite(pred_intensities.grad).all()


@pytest.mark.parametrize("matching", ["closest", "hungarian"])
def test_zero_entries_are_valid_and_negative_entries_are_rejected(matching: str) -> None:
    true_mzs = th.tensor([100.0, 150.0])
    true_intensities = th.tensor([1.0, 1.0])
    pred_mzs = th.tensor([100.0, 150.0])
    batch_indices = th.zeros(2, dtype=th.long)

    primary, entropy = sparse_peak_list_distance(
        true_mzs,
        true_intensities,
        batch_indices,
        pred_mzs,
        th.tensor([1.0, 0.0]),
        batch_indices,
        objective="jss_distance",
        matching=matching,
        tolerance=0.01,
        relative=False,
    )
    assert th.isfinite(primary).all()
    assert th.isfinite(entropy).all()

    with pytest.raises(ValueError, match="nonnegative"):
        sparse_peak_list_distance(
            true_mzs,
            true_intensities,
            batch_indices,
            pred_mzs,
            th.tensor([1.0, -0.1]),
            batch_indices,
            objective="cosine_distance",
            matching=matching,
            tolerance=0.01,
            relative=False,
        )


def test_unmatched_prediction_is_retained_and_suppressed() -> None:
    pred_intensities = th.tensor([1.0, 1.0], requires_grad=True)
    primary, _ = sparse_peak_list_distance(
        th.tensor([100.0]),
        th.tensor([1.0]),
        th.tensor([0]),
        th.tensor([100.0, 200.0]),
        pred_intensities,
        th.tensor([0, 0]),
        objective="cosine_distance",
        matching="closest",
        tolerance=0.01,
        relative=False,
    )
    primary.sum().backward()

    assert primary.item() > 0
    assert pred_intensities.grad is not None
    assert pred_intensities.grad[1].item() > 0


def test_kl_smoothing_keeps_missing_predicted_support_finite() -> None:
    pred_intensities = th.tensor([1.0, 0.0], requires_grad=True)
    primary, _ = sparse_peak_list_distance(
        th.tensor([100.0, 150.0]),
        th.tensor([0.5, 0.5]),
        th.tensor([0, 0]),
        th.tensor([100.0, 150.0]),
        pred_intensities,
        th.tensor([0, 0]),
        objective="kl_distance",
        matching="hungarian",
        tolerance=0.01,
        relative=False,
        kl_epsilon=1e-4,
    )

    assert th.isfinite(primary).all()
    assert primary.item() > 0.0
    primary.sum().backward()
    assert pred_intensities.grad is not None
    assert th.isfinite(pred_intensities.grad).all()


def test_kl_smoothing_does_not_reward_splitting_unmatched_intensity() -> None:
    losses = []
    for unmatched_peak_count in (1, 1_000):
        pred_mzs = th.cat(
            [
                th.tensor([100.0]),
                th.arange(unmatched_peak_count, dtype=th.float32) + 200.0,
            ]
        )
        pred_intensities = th.cat(
            [
                th.tensor([0.5]),
                th.full((unmatched_peak_count,), 0.5 / unmatched_peak_count),
            ]
        )
        primary, _ = sparse_peak_list_distance(
            th.tensor([100.0]),
            th.tensor([1.0]),
            th.tensor([0]),
            pred_mzs,
            pred_intensities,
            th.zeros(unmatched_peak_count + 1, dtype=th.long),
            objective="kl_distance",
            matching="hungarian",
            tolerance=0.01,
            relative=False,
            kl_epsilon=1e-4,
        )
        losses.append(primary)

    assert th.allclose(losses[0], losses[1], atol=1e-6)


@pytest.mark.parametrize("pred_dtype", [th.float16, th.bfloat16])
def test_kl_smoothing_is_stable_for_mixed_precision_predictions(
    pred_dtype: th.dtype,
) -> None:
    pred_intensities = th.tensor(
        [1.0, 0.0],
        dtype=pred_dtype,
        requires_grad=True,
    )
    primary, _ = sparse_peak_list_distance(
        th.tensor([100.0, 150.0]),
        th.tensor([0.5, 0.5]),
        th.tensor([0, 0]),
        th.tensor([100.0, 150.0]),
        pred_intensities,
        th.tensor([0, 0]),
        objective="kl_distance",
        matching="hungarian",
        tolerance=0.01,
        relative=False,
        kl_epsilon=1e-4,
    )
    primary.sum().backward()

    assert th.isfinite(primary).all()
    assert pred_intensities.grad is not None
    assert th.isfinite(pred_intensities.grad).all()


def test_hungarian_drops_candidates_outside_all_tolerance_windows() -> None:
    """Predictions outside every tolerance window match nothing, wherever they sit.

    Only their intensity enters the loss, through the unmatched tail, so moving them to
    other out-of-window m/z values must leave the loss unchanged. With a single feasible
    pair, Hungarian and closest matching must also agree.
    """

    def loss_for(pred_mzs: list[float], matching: str) -> float:
        primary, _ = sparse_peak_list_distance(
            th.tensor([100.0]),
            th.tensor([1.0]),
            th.tensor([0]),
            th.tensor(pred_mzs),
            th.ones(len(pred_mzs)),
            th.zeros(len(pred_mzs), dtype=th.long),
            objective="jss_distance",
            matching=matching,
            tolerance=0.01,
            relative=False,
        )
        return float(primary[0])

    baseline = loss_for([100.0, 200.0, 300.0], "hungarian")
    assert baseline == pytest.approx(loss_for([100.0, 700.0, 900.0], "hungarian"))
    assert baseline == pytest.approx(loss_for([100.0, 200.0, 300.0], "closest"))
    # One matched peak carrying a third of the predicted mass against a single observed
    # peak: JS divergence of [1/3, 1/3, 1/3] against [1, 0, 0], in bits.
    expected = (0.5 * math.log(3.0) - 0.5 * (2.0 / 3.0) * math.log(2.0)) / math.log(2.0)
    assert baseline == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize(
    "objective", ["cosine_distance", "hellinger_distance", "jss_distance", "kl_distance"]
)
def test_instrument_tolerance_is_passed_to_peak_matching(objective: str) -> None:
    primary, _ = sparse_peak_list_distance(
        th.tensor([100.0, 100.0]),
        th.ones(2),
        th.tensor([0, 1]),
        th.tensor([100.003, 100.003]),
        th.ones(2),
        th.tensor([0, 1]),
        objective=objective,
        matching="hungarian",
        tol_per_sample=th.tensor([1.0e-5, 2.0e-5]),
        min_mz_per_sample=th.tensor([200.0, 200.0]),
    )

    assert primary[0] > 0.0
    assert primary[1] == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("pred_dtype", [th.float16, th.bfloat16])
def test_cosine_accepts_mixed_precision_predictions(pred_dtype: th.dtype) -> None:
    pred_intensities = th.tensor([1.0, 2.0], dtype=pred_dtype, requires_grad=True)
    primary, _ = sparse_peak_list_distance(
        th.tensor([100.0, 150.0]),
        th.tensor([1.0, 2.0]),
        th.tensor([0, 0]),
        th.tensor([100.0, 150.0]),
        pred_intensities,
        th.tensor([0, 0]),
        objective="cosine_distance",
        matching="closest",
        tolerance=0.01,
        relative=False,
    )
    primary.sum().backward()

    assert th.allclose(primary, th.zeros(1), atol=1e-7)
    assert pred_intensities.grad is not None
    assert th.isfinite(pred_intensities.grad).all()


def _sparse_peak_list_distance_loop_reference(
    true_mzs: th.Tensor,
    true_intensities: th.Tensor,
    true_batch_idxs: th.Tensor,
    pred_mzs: th.Tensor,
    pred_intensities: th.Tensor,
    pred_batch_idxs: th.Tensor,
    *,
    objective: str,
    matching: str,
    tolerance: float = 1e-5,
    relative: bool = True,
    tolerance_min_mz: float = TOLERANCE_MIN_MZ,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
    decimals: int = 4,
    kl_epsilon: float = 1e-4,
) -> tuple[th.Tensor, th.Tensor]:
    """Per-spectrum reference for sparse_peak_list_distance, kept for equivalence testing.

    This is the implementation the chunked, padded version replaced. It skips the
    argument validation, which the production function still performs, and exists only
    so the batched objectives can be checked against it. Production code must not call it.
    """
    true_mzs, true_intensities, true_batch_idxs = round_aggregate_peaks(
        true_mzs, true_intensities, true_batch_idxs, decimals=decimals, agg="sum"
    )
    pred_mzs, pred_intensities, pred_batch_idxs = round_aggregate_peaks(
        pred_mzs, pred_intensities, pred_batch_idxs, decimals=decimals, agg="sum"
    )
    batch_size = int(true_batch_idxs.max().item()) + 1
    if pred_batch_idxs.min().item() < 0 or pred_batch_idxs.max().item() >= batch_size:
        raise ValueError("predicted batch indices reference a missing observed spectrum")

    primary_losses = true_intensities.new_empty(batch_size)
    entropy_losses = true_intensities.new_empty(batch_size)
    log_two = math.log(2.0)

    for batch_index in range(batch_size):
        true_mask = true_batch_idxs == batch_index
        pred_mask = pred_batch_idxs == batch_index
        batch_true_mzs = true_mzs[true_mask]
        batch_true_intensities = true_intensities[true_mask]
        batch_pred_mzs = pred_mzs[pred_mask]
        batch_pred_intensities = pred_intensities[pred_mask]
        if batch_true_mzs.numel() == 0 or batch_pred_mzs.numel() == 0:
            raise ValueError(f"spectrum {batch_index} must contain observed and predicted peaks")
        if batch_true_intensities.sum().item() <= 0:
            raise ValueError(f"observed spectrum {batch_index} has zero total intensity")
        if batch_pred_intensities.sum().item() <= 0:
            raise ValueError(f"predicted spectrum {batch_index} has zero total intensity")

        if tol_per_sample is None:
            match_mask = calculate_match_mzs(
                batch_true_mzs,
                batch_pred_mzs,
                tolerance=tolerance,
                relative=relative,
                tolerance_min_mz=tolerance_min_mz,
            )
        else:
            match_mask = calculate_match_mzs(
                batch_true_mzs,
                batch_pred_mzs,
                relative=True,
                tol_per_true=tol_per_sample[batch_index].expand_as(batch_true_mzs),
                min_mz_per_true=min_mz_per_sample[batch_index].expand_as(batch_true_mzs),
            )
        match_mask = th.as_tensor(match_mask, dtype=th.bool, device=batch_true_mzs.device)
        mass_distance = (batch_true_mzs[:, None] - batch_pred_mzs[None, :]).abs()

        matched_true_indices = th.empty(
            0,
            dtype=th.long,
            device=batch_true_mzs.device,
        )
        matched_pred_indices = th.empty(
            0,
            dtype=th.long,
            device=batch_true_mzs.device,
        )
        if matching == "closest":
            feasible_predictions = match_mask.any(dim=0)
            if feasible_predictions.any():
                feasible_distances = mass_distance[:, feasible_predictions].masked_fill(
                    ~match_mask[:, feasible_predictions],
                    th.inf,
                )
                matched_true_indices = feasible_distances.argmin(dim=0)
                matched_pred_indices = th.where(feasible_predictions)[0]
        elif match_mask.any():
            feasible_true = match_mask.any(dim=1)
            feasible_pred = match_mask.any(dim=0)
            feasible_true_indices = th.where(feasible_true)[0]
            feasible_pred_indices = th.where(feasible_pred)[0]
            feasible_match_mask = match_mask[feasible_true][:, feasible_pred]
            feasible_mass_distance = mass_distance[feasible_true][:, feasible_pred]
            # Invalid pairs receive a cost larger than every possible sum of valid
            # distances, so assignment first maximizes valid cardinality and then
            # minimizes total m/z distance. Drop isolated rows and columns before
            # solving because MASSter usually has many candidates outside every
            # observed peak's tolerance window.
            invalid_cost = (feasible_mass_distance.max().detach() + 1.0) * (
                min(feasible_match_mask.shape) + 1
            )
            assignment_cost = feasible_mass_distance.masked_fill(
                ~feasible_match_mask,
                invalid_cost,
            )
            row_indices, column_indices = linear_sum_assignment(
                assignment_cost.detach(),
                maximize=False,
            )
            valid_assignments = feasible_match_mask[row_indices, column_indices]
            matched_true_indices = feasible_true_indices[row_indices[valid_assignments]]
            matched_pred_indices = feasible_pred_indices[column_indices[valid_assignments]]

        pred_at_true = batch_pred_intensities.new_zeros(batch_true_intensities.shape)
        if matched_pred_indices.numel() > 0:
            pred_at_true = pred_at_true.index_add(
                0,
                matched_true_indices,
                batch_pred_intensities[matched_pred_indices],
            )
        unmatched_predictions = th.ones(
            batch_pred_intensities.shape,
            dtype=th.bool,
            device=batch_pred_intensities.device,
        )
        unmatched_predictions[matched_pred_indices] = False
        aligned_pred = th.cat(
            [pred_at_true, batch_pred_intensities[unmatched_predictions]],
            dim=0,
        )
        aligned_true = th.cat(
            [
                batch_true_intensities,
                batch_true_intensities.new_zeros(int(unmatched_predictions.sum().item())),
            ],
            dim=0,
        )

        distance_dtype = th.promote_types(aligned_pred.dtype, aligned_true.dtype)
        if distance_dtype in {th.float16, th.bfloat16}:
            distance_dtype = th.float32
        distance_pred = aligned_pred.to(dtype=distance_dtype)
        distance_true = aligned_true.to(dtype=distance_dtype)
        if objective == "cosine_distance":
            similarity = th.dot(distance_pred, distance_true) / (
                th.linalg.vector_norm(distance_pred) * th.linalg.vector_norm(distance_true)
            )
            primary_losses[batch_index] = 1.0 - similarity
        elif objective == "hellinger_distance":
            pred_distribution = distance_pred / distance_pred.sum()
            true_distribution = distance_true / distance_true.sum()
            # Plain sqrt of the product, with no zero-safe clamping. The production
            # path clamps inside each factor to keep the backward finite at structural
            # zeros; the two must still agree on value.
            coefficient = (pred_distribution * true_distribution).sqrt().sum()
            primary_losses[batch_index] = 1.0 - coefficient
        elif objective == "jss_distance":
            pred_distribution = distance_pred / distance_pred.sum()
            true_distribution = distance_true / distance_true.sum()
            mixture = 0.5 * (pred_distribution + true_distribution)

            def x_log_x(values: th.Tensor) -> th.Tensor:
                return values * values.clamp_min(th.finfo(values.dtype).tiny).log()

            divergence = (
                0.5 * x_log_x(pred_distribution).sum()
                + 0.5 * x_log_x(true_distribution).sum()
                - x_log_x(mixture).sum()
            )
            primary_losses[batch_index] = divergence / log_two
        else:
            pred_distribution = distance_pred / distance_pred.sum()
            true_distribution = distance_true / distance_true.sum()
            # Reserve a fixed total epsilon mass for observed support. Unlike adding
            # epsilon per aligned entry, this remains invariant when the same unmatched
            # intensity is split across more predicted peaks.
            smoothed_pred_distribution = (
                1.0 - kl_epsilon
            ) * pred_distribution + kl_epsilon * true_distribution
            positive_true = true_distribution > 0.0
            primary_losses[batch_index] = (
                true_distribution[positive_true]
                * (
                    true_distribution[positive_true].log()
                    - smoothed_pred_distribution[positive_true].log()
                )
            ).sum()

        pred_distribution = distance_pred / distance_pred.sum()
        true_distribution = distance_true / distance_true.sum()
        pred_entropy = -(
            pred_distribution
            * pred_distribution.clamp_min(th.finfo(pred_distribution.dtype).tiny).log()
        ).sum()
        true_entropy = -(
            true_distribution
            * true_distribution.clamp_min(th.finfo(true_distribution.dtype).tiny).log()
        ).sum()
        entropy_losses[batch_index] = (pred_entropy - true_entropy).square()

    return primary_losses, entropy_losses


def _random_peak_lists(
    seed: int,
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """Build a ragged batch with m/z collisions and spectra that match nothing."""
    generator = th.Generator().manual_seed(seed)
    batch_size = int(th.randint(1, 7, (1,), generator=generator).item())
    true_mzs, true_ints, true_idxs = [], [], []
    pred_mzs, pred_ints, pred_idxs = [], [], []
    for batch_index in range(batch_size):
        num_true = int(th.randint(1, 12, (1,), generator=generator).item())
        num_pred = int(th.randint(1, 25, (1,), generator=generator).item())
        observed = th.round(th.rand(num_true, generator=generator) * 400.0 + 100.0, decimals=4)
        true_mzs.append(observed)
        true_ints.append(th.rand(num_true, generator=generator) + 1e-3)
        true_idxs.append(th.full((num_true,), batch_index, dtype=th.long))
        # Half the predictions sit inside tolerance of an observed peak, half are far
        # away, and every third spectrum has no in-tolerance prediction at all.
        picks = th.randint(0, num_true, (num_pred,), generator=generator)
        # One-sided jitter. Symmetric offsets round to exactly equal m/z distances in
        # float32, and a tied Hungarian cost makes the matched pair — and therefore the
        # loss — ambiguous. That ambiguity is covered by its own test below.
        jitter = th.rand(num_pred, generator=generator) * 2e-4 + 1e-6
        near = observed[picks] + jitter
        far = th.rand(num_pred, generator=generator) * 400.0 + 600.0
        use_near = th.rand(num_pred, generator=generator) < 0.5
        if batch_index % 3 == 2:
            use_near = th.zeros(num_pred, dtype=th.bool)
        pred_mzs.append(th.where(use_near, near, far))
        pred_ints.append(th.rand(num_pred, generator=generator) + 1e-3)
        pred_idxs.append(th.full((num_pred,), batch_index, dtype=th.long))
    return (
        th.cat(true_mzs),
        th.cat(true_ints),
        th.cat(true_idxs),
        th.cat(pred_mzs),
        th.cat(pred_ints),
        th.cat(pred_idxs),
    )


class TestPeakListDistanceMatchesLoopReference:
    """The chunked, padded implementation must agree with the per-spectrum loop."""

    @pytest.mark.parametrize(
        "objective",
        ["cosine_distance", "hellinger_distance", "jss_distance", "kl_distance"],
    )
    @pytest.mark.parametrize("matching", ["closest", "hungarian"])
    @pytest.mark.parametrize("loss_batch_size", [1, 3, 32])
    @pytest.mark.parametrize("seed", range(8))
    def test_matches_reference(
        self, objective: str, matching: str, loss_batch_size: int, seed: int
    ) -> None:
        """Losses agree for every objective, matching rule, and chunk width."""
        true_mzs, true_ints, true_idxs, pred_mzs, pred_ints, pred_idxs = _random_peak_lists(seed)
        primary, entropy = sparse_peak_list_distance(
            true_mzs,
            true_ints,
            true_idxs,
            pred_mzs,
            pred_ints,
            pred_idxs,
            objective=objective,
            matching=matching,
            tolerance=1e-5,
            relative=True,
            loss_batch_size=loss_batch_size,
        )
        ref_primary, ref_entropy = _sparse_peak_list_distance_loop_reference(
            true_mzs,
            true_ints,
            true_idxs,
            pred_mzs,
            pred_ints,
            pred_idxs,
            objective=objective,
            matching=matching,
            tolerance=1e-5,
            relative=True,
        )
        assert th.allclose(primary, ref_primary, atol=1e-6, rtol=1e-5)
        assert th.allclose(entropy, ref_entropy, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("matching", ["closest", "hungarian"])
    def test_gradients_match_reference(self, matching: str) -> None:
        """Gradients to predicted intensities agree with the per-spectrum loop."""
        true_mzs, true_ints, true_idxs, pred_mzs, pred_ints, pred_idxs = _random_peak_lists(3)

        batched_ints = pred_ints.clone().requires_grad_(True)
        primary, _ = sparse_peak_list_distance(
            true_mzs,
            true_ints,
            true_idxs,
            pred_mzs,
            batched_ints,
            pred_idxs,
            objective="jss_distance",
            matching=matching,
        )
        primary.sum().backward()

        reference_ints = pred_ints.clone().requires_grad_(True)
        ref_primary, _ = _sparse_peak_list_distance_loop_reference(
            true_mzs,
            true_ints,
            true_idxs,
            pred_mzs,
            reference_ints,
            pred_idxs,
            objective="jss_distance",
            matching=matching,
        )
        ref_primary.sum().backward()

        assert th.allclose(batched_ints.grad, reference_ints.grad, atol=1e-6, rtol=1e-5)

    def test_per_sample_tolerance_matches_reference(self) -> None:
        """The instrument-tolerance path agrees with the per-spectrum loop."""
        true_mzs, true_ints, true_idxs, pred_mzs, pred_ints, pred_idxs = _random_peak_lists(5)
        batch_size = int(true_idxs.max()) + 1
        tol_per_sample = th.full((batch_size,), 2e-5)
        min_mz_per_sample = th.full((batch_size,), TOLERANCE_MIN_MZ)
        kwargs = {
            "objective": "jss_distance",
            "matching": "hungarian",
            "tol_per_sample": tol_per_sample,
            "min_mz_per_sample": min_mz_per_sample,
        }
        primary, entropy = sparse_peak_list_distance(
            true_mzs, true_ints, true_idxs, pred_mzs, pred_ints, pred_idxs, **kwargs
        )
        ref_primary, ref_entropy = _sparse_peak_list_distance_loop_reference(
            true_mzs, true_ints, true_idxs, pred_mzs, pred_ints, pred_idxs, **kwargs
        )
        assert th.allclose(primary, ref_primary, atol=1e-6, rtol=1e-5)
        assert th.allclose(entropy, ref_entropy, atol=1e-6, rtol=1e-5)


def test_hungarian_tie_matches_exactly_one_of_the_tied_candidates() -> None:
    """Two candidates equidistant from one observed peak give an ambiguous loss.

    Hungarian matching is one-to-one, so only one of an exactly-tied pair is matched and
    the other stays in the unmatched tail. Both assignments have the same m/z cost, and
    the objective is not a function of that cost, so either outcome is a correct solve
    and the loss is whichever branch the solver happened to take. This is inherent to
    one-to-one matching, not a property of any particular implementation.
    """
    observed_mz = th.tensor([445.5466])
    candidate_mzs = th.tensor([445.54651, 445.54669])
    candidate_ints = th.tensor([0.9376779, 0.3350758])
    distances = (observed_mz[:, None] - candidate_mzs[None, :]).abs()
    # Self-check: without an exact tie this test would assert nothing interesting.
    assert distances[0, 0] == distances[0, 1]

    def jensen_shannon_bits(pred: th.Tensor, true: th.Tensor) -> float:
        pred = pred / pred.sum()
        true = true / true.sum()
        mixture = 0.5 * (pred + true)

        def x_log_x(values: th.Tensor) -> th.Tensor:
            return (values * values.clamp_min(th.finfo(values.dtype).tiny).log()).sum()

        divergence = 0.5 * x_log_x(pred) + 0.5 * x_log_x(true) - x_log_x(mixture)
        return float(divergence / math.log(2.0))

    observed_int = th.tensor([1.0])
    first_matched = jensen_shannon_bits(
        th.tensor([candidate_ints[0], candidate_ints[1]]), th.tensor([observed_int[0], 0.0])
    )
    second_matched = jensen_shannon_bits(
        th.tensor([candidate_ints[1], candidate_ints[0]]), th.tensor([observed_int[0], 0.0])
    )
    assert first_matched != pytest.approx(second_matched)

    primary, _ = sparse_peak_list_distance(
        observed_mz,
        observed_int,
        th.zeros(1, dtype=th.long),
        candidate_mzs,
        candidate_ints,
        th.zeros(2, dtype=th.long),
        objective="jss_distance",
        matching="hungarian",
    )
    assert float(primary[0]) == pytest.approx(first_matched, abs=1e-6) or float(
        primary[0]
    ) == pytest.approx(second_matched, abs=1e-6)

    # Closest matching has no such ambiguity: both candidates join the observed peak.
    closest_primary, _ = sparse_peak_list_distance(
        observed_mz,
        observed_int,
        th.zeros(1, dtype=th.long),
        candidate_mzs,
        candidate_ints,
        th.zeros(2, dtype=th.long),
        objective="jss_distance",
        matching="closest",
    )
    assert float(closest_primary[0]) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.skipif(not th.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "objective", ["cosine_distance", "hellinger_distance", "jss_distance", "kl_distance"]
)
@pytest.mark.parametrize("matching", ["closest", "hungarian"])
def test_matches_reference_on_cuda(objective: str, matching: str) -> None:
    """CUDA agrees with the loop reference.

    The device matters here: peak matching runs under bfloat16 autocast on CUDA, and
    Hungarian matching dispatches to a different assignment backend, so CPU agreement
    does not imply CUDA agreement.
    """
    for seed in range(4):
        tensors = [tensor.cuda() for tensor in _random_peak_lists(seed)]
        primary, entropy = sparse_peak_list_distance(
            *tensors, objective=objective, matching=matching, loss_batch_size=3
        )
        ref_primary, ref_entropy = _sparse_peak_list_distance_loop_reference(
            *tensors, objective=objective, matching=matching
        )
        assert th.allclose(primary, ref_primary, atol=1e-5, rtol=1e-4)
        assert th.allclose(entropy, ref_entropy, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "objective", ["cosine_distance", "hellinger_distance", "jss_distance", "kl_distance"]
)
@pytest.mark.parametrize("matching", ["closest", "hungarian"])
def test_wildly_uneven_spectra_match_reference(objective: str, matching: str) -> None:
    """Chunks narrowed by the padding guard still agree with the loop reference.

    Peak counts spanning 1 to 1500 force ``build_loss_chunks`` to split below the
    ``loss_batch_size`` ceiling, so large spectra are padded alone instead of against
    small ones.
    """
    generator = th.Generator().manual_seed(11)
    sizes = [(2, 3), (900, 1200), (1, 2), (3, 4), (700, 1500), (2, 2)]
    true_mzs, true_ints, true_idxs = [], [], []
    pred_mzs, pred_ints, pred_idxs = [], [], []
    for batch_index, (num_true, num_pred) in enumerate(sizes):
        observed = th.round(th.rand(num_true, generator=generator) * 400.0 + 100.0, decimals=4)
        true_mzs.append(observed)
        true_ints.append(th.rand(num_true, generator=generator) + 1e-3)
        true_idxs.append(th.full((num_true,), batch_index, dtype=th.long))
        picks = th.randint(0, num_true, (num_pred,), generator=generator)
        pred_mzs.append(observed[picks] + th.rand(num_pred, generator=generator) * 2e-4 + 1e-6)
        pred_ints.append(th.rand(num_pred, generator=generator) + 1e-3)
        pred_idxs.append(th.full((num_pred,), batch_index, dtype=th.long))
    tensors = (
        th.cat(true_mzs),
        th.cat(true_ints),
        th.cat(true_idxs),
        th.cat(pred_mzs),
        th.cat(pred_ints),
        th.cat(pred_idxs),
    )

    chunks = build_loss_chunks(
        [0] + th.bincount(tensors[2]).cumsum(0).tolist(),
        [0] + th.bincount(tensors[5]).cumsum(0).tolist(),
        32,
    )
    # Self-check: without narrowing this would be a single chunk and test nothing new.
    assert len(chunks) > 1

    primary, entropy = sparse_peak_list_distance(
        *tensors, objective=objective, matching=matching, loss_batch_size=32
    )
    ref_primary, ref_entropy = _sparse_peak_list_distance_loop_reference(
        *tensors, objective=objective, matching=matching
    )
    assert th.allclose(primary, ref_primary, atol=1e-5, rtol=1e-4)
    assert th.allclose(entropy, ref_entropy, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("pred_dtype", [th.float16, th.bfloat16])
def test_half_precision_predictions_match_reference(pred_dtype: th.dtype) -> None:
    """Half-precision predictions are promoted to float32 and match the loop reference."""
    tensors = list(_random_peak_lists(2))
    tensors[4] = tensors[4].to(pred_dtype)
    primary, _ = sparse_peak_list_distance(*tensors, objective="jss_distance", matching="hungarian")
    ref_primary, _ = _sparse_peak_list_distance_loop_reference(
        *tensors, objective="jss_distance", matching="hungarian"
    )
    assert primary.dtype == th.float32
    assert th.allclose(primary, ref_primary, atol=1e-6, rtol=1e-5)


def test_spectrum_with_no_peaks_raises_like_reference() -> None:
    """A batch index carrying no peaks is rejected with the same error as the loop."""
    tensors = (
        th.tensor([100.0, 200.0]),
        th.tensor([1.0, 1.0]),
        th.tensor([0, 2]),
        th.tensor([100.0, 200.0]),
        th.tensor([1.0, 1.0]),
        th.tensor([0, 2]),
    )
    message = "spectrum 1 must contain observed and predicted peaks"
    with pytest.raises(ValueError, match=message):
        sparse_peak_list_distance(*tensors, objective="jss_distance", matching="hungarian")
    with pytest.raises(ValueError, match=message):
        _sparse_peak_list_distance_loop_reference(
            *tensors, objective="jss_distance", matching="hungarian"
        )


@pytest.mark.parametrize("matching", ["closest", "hungarian"])
def test_hellinger_value_and_gradient_are_chunk_invariant(matching: str) -> None:
    """Chunking is a padding choice, so it must not change values or gradients.

    The Hellinger branch reduces over the padded chunk, and its zero-safe clamping
    fires on exactly the padded coordinates that chunk width controls.
    """
    true_mzs, true_ints, true_idxs, pred_mzs, pred_ints, pred_idxs = _random_peak_lists(11)

    values = []
    gradients = []
    for loss_batch_size in (1, 2, 5, 64):
        chunked_ints = pred_ints.clone().requires_grad_(True)
        primary, entropy = sparse_peak_list_distance(
            true_mzs,
            true_ints,
            true_idxs,
            pred_mzs,
            chunked_ints,
            pred_idxs,
            objective="hellinger_distance",
            matching=matching,
            tolerance=1e-5,
            relative=True,
            loss_batch_size=loss_batch_size,
        )
        primary.sum().backward()
        assert th.isfinite(primary).all()
        assert th.isfinite(chunked_ints.grad).all()
        values.append((primary, entropy))
        gradients.append(chunked_ints.grad)

    for primary, entropy in values[1:]:
        th.testing.assert_close(primary, values[0][0])
        th.testing.assert_close(entropy, values[0][1])
    for gradient in gradients[1:]:
        th.testing.assert_close(gradient, gradients[0])


@pytest.mark.parametrize("matching", ["closest", "hungarian"])
def test_hellinger_scores_each_spectrum_independently_of_the_batch(matching: str) -> None:
    """A spectrum's loss must not depend on which spectra share its batch."""
    true_mzs, true_ints, true_idxs, pred_mzs, pred_ints, pred_idxs = _random_peak_lists(13)
    batch_size = int(true_idxs.max()) + 1

    batched, _ = sparse_peak_list_distance(
        true_mzs,
        true_ints,
        true_idxs,
        pred_mzs,
        pred_ints,
        pred_idxs,
        objective="hellinger_distance",
        matching=matching,
        tolerance=1e-5,
        relative=True,
        loss_batch_size=32,
    )

    alone = []
    for batch_index in range(batch_size):
        true_mask = true_idxs == batch_index
        pred_mask = pred_idxs == batch_index
        single, _ = sparse_peak_list_distance(
            true_mzs[true_mask],
            true_ints[true_mask],
            th.zeros(int(true_mask.sum()), dtype=th.long),
            pred_mzs[pred_mask],
            pred_ints[pred_mask],
            th.zeros(int(pred_mask.sum()), dtype=th.long),
            objective="hellinger_distance",
            matching=matching,
            tolerance=1e-5,
            relative=True,
        )
        alone.append(single)

    th.testing.assert_close(batched, th.cat(alone))


def _coverage_gap_batch():
    """One spectrum whose second observed peak no candidate can reach.

    Observed 100 (intensity 6) and 200 (intensity 4); the only candidate sits at 100, so
    40% of the observed intensity is unreachable and `wrecall` is 0.6.
    """
    true_mzs = th.tensor([100.0, 200.0], dtype=th.float64)
    true_ints = th.tensor([6.0, 4.0], dtype=th.float64)
    true_batch = th.tensor([0, 0], dtype=th.long)
    pred_mzs = th.tensor([100.0], dtype=th.float64)
    pred_ints = th.tensor([1.0], dtype=th.float64)
    pred_batch = th.tensor([0], dtype=th.long)
    return true_mzs, true_ints, true_batch, pred_mzs, pred_ints, pred_batch


def test_escape_absent_leaves_every_objective_bit_identical():
    args = _coverage_gap_batch()
    for objective in ("cosine_distance", "hellinger_distance", "jss_distance", "kl_distance"):
        baseline = sparse_peak_list_distance(
            *args, objective=objective, matching="closest", tolerance=1e-5
        )
        explicit_none = sparse_peak_list_distance(
            *args,
            objective=objective,
            matching="closest",
            pred_dump_intensities=None,
            tolerance=1e-5,
        )
        assert th.equal(baseline[0], explicit_none[0])
        assert th.equal(baseline[1], explicit_none[1])


def test_escape_absorbs_the_unreachable_intensity_and_clears_the_loss_floor():
    """With no escape the loss cannot reach zero; with a correctly sized one it can."""
    args = _coverage_gap_batch()

    without, _ = sparse_peak_list_distance(
        *args, objective="hellinger_distance", matching="closest", tolerance=1e-5
    )
    # The predicted candidate mass is 1.0, so a dump of 4/6 puts 40% of the predicted
    # distribution on the escape, exactly the unreachable observed fraction.
    with_escape, _ = sparse_peak_list_distance(
        *args,
        objective="hellinger_distance",
        matching="closest",
        pred_dump_intensities=th.tensor([4.0 / 6.0], dtype=th.float64),
        tolerance=1e-5,
    )

    assert without.item() == pytest.approx(1.0 - math.sqrt(0.6), abs=1e-9)
    assert with_escape.item() == pytest.approx(0.0, abs=1e-9)


def test_escape_is_optimal_exactly_at_the_unreachable_fraction():
    args = _coverage_gap_batch()

    def hellinger(dump):
        value, _ = sparse_peak_list_distance(
            *args,
            objective="hellinger_distance",
            matching="closest",
            pred_dump_intensities=th.tensor([dump], dtype=th.float64),
            tolerance=1e-5,
        )
        return value.item()

    optimal = hellinger(4.0 / 6.0)
    assert optimal < hellinger(0.1)
    assert optimal < hellinger(0.4)
    assert optimal < hellinger(1.0)
    assert optimal < hellinger(3.0)


def test_escape_does_not_rescue_a_reachable_peak_the_model_scored_zero():
    """Reachability is structural: a candidate exists at 200, so its miss still costs."""
    true_mzs = th.tensor([100.0, 200.0], dtype=th.float64)
    true_ints = th.tensor([6.0, 4.0], dtype=th.float64)
    true_batch = th.tensor([0, 0], dtype=th.long)
    pred_mzs = th.tensor([100.0, 200.0], dtype=th.float64)
    pred_ints = th.tensor([1.0, 0.0], dtype=th.float64)
    pred_batch = th.tensor([0, 0], dtype=th.long)

    value, _ = sparse_peak_list_distance(
        true_mzs,
        true_ints,
        true_batch,
        pred_mzs,
        pred_ints,
        pred_batch,
        objective="hellinger_distance",
        matching="closest",
        pred_dump_intensities=th.tensor([4.0 / 6.0], dtype=th.float64),
        tolerance=1e-5,
    )

    # The 200 peak is inside a candidate's tolerance window, so it stays in the support and
    # the escape cannot absorb it; the loss must stay well away from zero.
    assert value.item() > 0.1


def test_escape_keeps_gradients_flowing_to_the_dump():
    args = _coverage_gap_batch()
    dump = th.tensor([0.2], dtype=th.float64, requires_grad=True)

    value, _ = sparse_peak_list_distance(
        *args,
        objective="hellinger_distance",
        matching="closest",
        pred_dump_intensities=dump,
        tolerance=1e-5,
    )
    value.sum().backward()

    assert dump.grad is not None
    assert th.isfinite(dump.grad).all()
    assert dump.grad.item() < 0.0  # under-dumping, so increasing the dump reduces the loss


def test_escape_makes_kl_finite_on_a_coverage_gap():
    args = _coverage_gap_batch()

    without, _ = sparse_peak_list_distance(
        *args, objective="kl_distance", matching="closest", tolerance=1e-5
    )
    with_escape, _ = sparse_peak_list_distance(
        *args,
        objective="kl_distance",
        matching="closest",
        pred_dump_intensities=th.tensor([4.0 / 6.0], dtype=th.float64),
        tolerance=1e-5,
    )

    assert th.isfinite(without).all()
    assert with_escape.item() == pytest.approx(0.0, abs=1e-9)
    assert with_escape.item() < without.item()


@pytest.mark.parametrize(
    "dump, match",
    [
        (th.tensor([[0.5]]), "one-dimensional"),
        (th.tensor([float("nan")]), "finite"),
        (th.tensor([-1.0]), "nonnegative"),
        (th.tensor([0.5, 0.5]), "one value per spectrum"),
    ],
)
def test_escape_rejects_malformed_input(dump, match):
    args = _coverage_gap_batch()
    with pytest.raises(ValueError, match=match):
        sparse_peak_list_distance(
            *args,
            objective="hellinger_distance",
            matching="closest",
            pred_dump_intensities=dump.to(th.float64),
            tolerance=1e-5,
        )
