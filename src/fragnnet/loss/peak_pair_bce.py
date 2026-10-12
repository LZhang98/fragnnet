"""Binary cross-entropy terms over peak pairs: CE-switch pairs and precursor contenders."""

import math

import torch as th
import torch.nn.functional as F


def ce_switch_pair_bce(
    true_mzs: th.Tensor,
    true_ints: th.Tensor,
    true_batch_idxs: th.Tensor,
    pred_mzs: th.Tensor,
    pred_logprobs: th.Tensor,
    pred_batch_idxs: th.Tensor,
    group_ids: th.Tensor,
    ce_values: th.Tensor,
    eligible_samples: th.Tensor,
    batch_size: int,
    min_endpoint_confidence: float,
    min_contender_relative_intensity: float,
    tol_per_sample: th.Tensor,
    min_mz_per_sample: th.Tensor,
) -> tuple[th.Tensor, th.Tensor]:
    """Compute online soft BCE supervision for adjacent CE winner switches.

    Each retained transition compares the lower-CE observed winner ``A`` with
    the higher-CE observed winner ``B`` at both endpoints. Predicted intensity
    aliases within the instrument tolerance are aggregated with log-sum-exp.
    The returned vector can be added directly to the existing per-spectrum
    loss before its fixed-denominator batch reduction.
    """
    losses = pred_logprobs.new_zeros(batch_size)
    eligible_endpoints = pred_logprobs.new_zeros(batch_size)
    if batch_size < 2:
        return losses, eligible_endpoints

    # Target discovery is deliberately CPU-only. One detached transfer avoids
    # data-dependent GPU synchronization inside the context/transition loops;
    # gradients still flow through the gathered prediction log-probabilities.
    true_mzs_cpu = true_mzs.detach().cpu()
    true_ints_cpu = true_ints.detach().cpu()
    true_batch_idxs_cpu = true_batch_idxs.detach().cpu()
    pred_mzs_cpu = pred_mzs.detach().cpu()
    pred_batch_idxs_cpu = pred_batch_idxs.detach().cpu()
    group_ids_cpu = group_ids.detach().cpu()
    ce_values_cpu = ce_values.detach().cpu()
    eligible_samples_cpu = eligible_samples.detach().cpu()
    tol_cpu = tol_per_sample.detach().cpu()
    min_mz_cpu = min_mz_per_sample.detach().cpu()

    true_index_lists: list[list[int]] = [[] for _ in range(batch_size)]
    for peak_idx, sample_idx in enumerate(true_batch_idxs_cpu.tolist()):
        true_index_lists[sample_idx].append(peak_idx)
    true_indices_by_sample = [th.tensor(indices, dtype=th.long) for indices in true_index_lists]
    pred_index_lists: list[list[int]] = [[] for _ in range(batch_size)]
    for peak_idx, sample_idx in enumerate(pred_batch_idxs_cpu.tolist()):
        pred_index_lists[sample_idx].append(peak_idx)
    pred_indices_by_sample = [th.tensor(indices, dtype=th.long) for indices in pred_index_lists]

    def _matches(mzs: th.Tensor, mass: float, sample_idx: int) -> th.Tensor:
        tolerance = float(tol_cpu[sample_idx]) * max(mass, float(min_mz_cpu[sample_idx]))
        return th.abs(mzs - mass) < tolerance

    def _observed_groups(sample_idx: int) -> list[tuple[float, float]]:
        """Merge raw observed aliases before ranking their summed intensities."""
        sample_indices = true_indices_by_sample[sample_idx]
        mzs = true_mzs_cpu[sample_indices]
        ints = true_ints_cpu[sample_indices]
        order = th.argsort(mzs)
        groups: list[list[tuple[float, float]]] = []
        group_upper_bounds: list[tuple[float, float]] = []
        for mz, intensity in zip(mzs[order].tolist(), ints[order].tolist()):
            tolerance = float(tol_cpu[sample_idx]) * max(mz, float(min_mz_cpu[sample_idx]))
            upper_bound = mz + tolerance
            if groups and mz < group_upper_bounds[-1][1]:
                if len(groups) > 1 and mz < group_upper_bounds[-2][1]:
                    return []
                if mz >= group_upper_bounds[-1][0]:
                    return []
                groups[-1].append((mz, intensity))
                min_upper, max_upper = group_upper_bounds[-1]
                group_upper_bounds[-1] = (
                    min(min_upper, upper_bound),
                    max(max_upper, upper_bound),
                )
            else:
                groups.append([(mz, intensity)])
                group_upper_bounds.append((upper_bound, upper_bound))
        return [
            (
                sum(mz * intensity for mz, intensity in group)
                / sum(intensity for _, intensity in group),
                sum(intensity for _, intensity in group),
            )
            for group in groups
            if sum(intensity for _, intensity in group) > 0
        ]

    context_transitions: dict[int, list[tuple]] = {}
    with th.no_grad():
        for group_id in th.unique(group_ids_cpu).tolist():
            sample_idxs = th.nonzero(
                (group_ids_cpu == group_id) & eligible_samples_cpu, as_tuple=False
            ).flatten()
            if sample_idxs.numel() < 2:
                continue
            observed_by_sample = {
                int(sample_idx): _observed_groups(int(sample_idx))
                for sample_idx in sample_idxs.tolist()
            }
            order = th.argsort(ce_values_cpu[sample_idxs])
            sample_idxs = sample_idxs[order]
            transitions = []
            for low_t, high_t in zip(sample_idxs[:-1], sample_idxs[1:]):
                low = int(low_t.item())
                high = int(high_t.item())
                low_groups = observed_by_sample[low]
                high_groups = observed_by_sample[high]
                if not low_groups or not high_groups:
                    continue
                mass_a = max(low_groups, key=lambda group: group[1])[0]
                mass_b = max(high_groups, key=lambda group: group[1])[0]
                if _matches(th.tensor([mass_b]), mass_a, low).item():
                    continue

                endpoint_data = []
                valid = True
                for endpoint in (low, high):
                    groups = observed_by_sample[endpoint]
                    matches_a = [
                        group
                        for group in groups
                        if _matches(th.tensor([group[0]]), mass_a, endpoint).item()
                    ]
                    matches_b = [
                        group
                        for group in groups
                        if _matches(th.tensor([group[0]]), mass_b, endpoint).item()
                    ]
                    if len(matches_a) != 1 or len(matches_b) != 1:
                        valid = False
                        break
                    int_a = matches_a[0][1]
                    int_b = matches_b[0][1]
                    # The detection-limit floor and singleton reliability table are
                    # separate audit outputs. Until they exist, keep v1 targets soft
                    # by training only endpoints where both contenders are observed.
                    if int_a <= 0 or int_b <= 0:
                        valid = False
                        break
                    max_intensity = max(group[1] for group in groups)
                    if min(int_a, int_b) < (min_contender_relative_intensity * max_intensity):
                        valid = False
                        break
                    total = int_a + int_b
                    confidence = abs(int_a - int_b) / max(int_a, int_b)
                    if confidence < min_endpoint_confidence:
                        valid = False
                        break
                    endpoint_data.append((endpoint, int_a / total, confidence))
                if valid:
                    for endpoint in (low, high):
                        endpoint_pred_idxs = pred_indices_by_sample[endpoint]
                        mzs = pred_mzs_cpu[endpoint_pred_idxs]
                        match_a = endpoint_pred_idxs[_matches(mzs, mass_a, endpoint)]
                        match_b = endpoint_pred_idxs[_matches(mzs, mass_b, endpoint)]
                        if not match_a.numel() or not match_b.numel():
                            valid = False
                            break
                        endpoint_data[endpoint == high] += (match_a, match_b)
                if valid:
                    transition_weight = min(endpoint_data[0][2], endpoint_data[1][2])
                    transitions.append((transition_weight, endpoint_data))
            if transitions:
                context_transitions[int(group_id)] = transitions

    match_indices = [
        match
        for transitions in context_transitions.values()
        for _, endpoint_data in transitions
        for _, _, _, match_a, match_b in endpoint_data
        for match in (match_a, match_b)
    ]
    flat_match_indices = (
        th.cat(match_indices).to(pred_logprobs.device)
        if match_indices
        else th.empty(0, dtype=th.long, device=pred_logprobs.device)
    )
    match_offset = 0
    for transitions in context_transitions.values():
        weight_sum = sum(transition[0] for transition in transitions)
        for weight, endpoint_data in transitions:
            for endpoint, target_a, _, match_a, match_b in endpoint_data:
                match_a_end = match_offset + match_a.numel()
                device_match_a = flat_match_indices[match_offset:match_a_end]
                match_b_end = match_a_end + match_b.numel()
                device_match_b = flat_match_indices[match_a_end:match_b_end]
                match_offset = match_b_end
                log_prob_a = th.logsumexp(pred_logprobs[device_match_a], dim=0)
                log_prob_b = th.logsumexp(pred_logprobs[device_match_b], dim=0)
                endpoint_weight = weight / (2.0 * weight_sum)
                losses[endpoint] = losses[endpoint] + endpoint_weight * (
                    F.binary_cross_entropy_with_logits(
                        log_prob_a - log_prob_b,
                        pred_logprobs.new_tensor(target_a),
                    )
                )
                eligible_endpoints[endpoint] += 1
    return losses, eligible_endpoints


def precursor_contender_bce(
    true_mzs: th.Tensor,
    true_ints: th.Tensor,
    true_batch_idxs: th.Tensor,
    pred_mzs: th.Tensor,
    pred_logprobs: th.Tensor,
    pred_batch_idxs: th.Tensor,
    precursor_mzs: th.Tensor,
    nce_values: th.Tensor,
    eligible_samples: th.Tensor,
    batch_size: int,
    nce_min: float,
    nce_max: float,
    absent_intensity_floor: float,
    tol_per_sample: th.Tensor,
    min_mz_per_sample: th.Tensor,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Supervise precursor versus the strongest supported non-precursor peak.

    For each eligible low-CE spectrum, the target is the observed soft intensity
    ratio between the precursor and the strongest observed non-precursor mass
    present in prediction support. Predicted aliases within instrument tolerance
    are aggregated before applying binary cross entropy to the log-intensity
    difference.
    """
    losses = pred_logprobs.new_zeros(batch_size)
    eligible_pairs = pred_logprobs.new_zeros(batch_size)
    precursor_wins = pred_logprobs.new_zeros(batch_size)
    if batch_size == 0:
        return losses, eligible_pairs, precursor_wins

    # 1. Move target discovery to CPU while retaining GPU prediction indices for gradients.
    true_mzs_cpu = true_mzs.detach().cpu()
    true_ints_cpu = true_ints.detach().cpu()
    true_batch_idxs_cpu = true_batch_idxs.detach().cpu()
    pred_mzs_cpu = pred_mzs.detach().cpu()
    pred_batch_idxs_cpu = pred_batch_idxs.detach().cpu()
    precursor_mzs_cpu = precursor_mzs.detach().cpu().reshape(-1)
    nce_values_cpu = nce_values.detach().cpu().reshape(-1)
    eligible_samples_cpu = eligible_samples.detach().cpu().reshape(-1)
    tol_cpu = tol_per_sample.detach().cpu().reshape(-1)
    min_mz_cpu = min_mz_per_sample.detach().cpu().reshape(-1)

    true_index_lists: list[list[int]] = [[] for _ in range(batch_size)]
    for peak_idx, sample_idx in enumerate(true_batch_idxs_cpu.tolist()):
        true_index_lists[sample_idx].append(peak_idx)
    true_indices_by_sample = [th.tensor(indices, dtype=th.long) for indices in true_index_lists]
    pred_index_lists: list[list[int]] = [[] for _ in range(batch_size)]
    for peak_idx, sample_idx in enumerate(pred_batch_idxs_cpu.tolist()):
        pred_index_lists[sample_idx].append(peak_idx)
    pred_indices_by_sample = [th.tensor(indices, dtype=th.long) for indices in pred_index_lists]

    def _matches(mzs: th.Tensor, mass: float, sample_idx: int) -> th.Tensor:
        tolerance = float(tol_cpu[sample_idx]) * max(mass, float(min_mz_cpu[sample_idx]))
        return th.abs(mzs - mass) < tolerance

    def _observed_groups(sample_idx: int) -> list[tuple[float, float]]:
        """Merge mutually unambiguous observed aliases at instrument tolerance."""
        indices = true_indices_by_sample[sample_idx]
        mzs = true_mzs_cpu[indices]
        ints = true_ints_cpu[indices]
        order = th.argsort(mzs)
        groups: list[list[tuple[float, float]]] = []
        upper_bounds: list[tuple[float, float]] = []
        for mz, intensity in zip(mzs[order].tolist(), ints[order].tolist()):
            tolerance = float(tol_cpu[sample_idx]) * max(mz, float(min_mz_cpu[sample_idx]))
            upper_bound = mz + tolerance
            if groups and mz < upper_bounds[-1][1]:
                if len(groups) > 1 and mz < upper_bounds[-2][1]:
                    return []
                if mz >= upper_bounds[-1][0]:
                    return []
                groups[-1].append((mz, intensity))
                low, high = upper_bounds[-1]
                upper_bounds[-1] = (min(low, upper_bound), max(high, upper_bound))
            else:
                groups.append([(mz, intensity)])
                upper_bounds.append((upper_bound, upper_bound))
        return [
            (
                sum(mz * intensity for mz, intensity in group)
                / sum(intensity for _, intensity in group),
                sum(intensity for _, intensity in group),
            )
            for group in groups
            if sum(intensity for _, intensity in group) > 0
        ]

    # 2. Select one observed precursor/fragment comparison per eligible spectrum.
    pair_data: list[tuple[int, float, th.Tensor, th.Tensor, bool]] = []
    for sample_idx in range(batch_size):
        nce = float(nce_values_cpu[sample_idx])
        if (
            not bool(eligible_samples_cpu[sample_idx])
            or not math.isfinite(nce)
            or nce < nce_min
            or nce >= nce_max
        ):
            continue
        groups = _observed_groups(sample_idx)
        pred_indices = pred_indices_by_sample[sample_idx]
        if not groups or pred_indices.numel() == 0:
            continue
        pred_sample_mzs = pred_mzs_cpu[pred_indices]
        precursor_mz = float(precursor_mzs_cpu[sample_idx])
        pred_precursor = pred_indices[_matches(pred_sample_mzs, precursor_mz, sample_idx)]
        if pred_precursor.numel() == 0:
            continue

        precursor_groups = [
            group
            for group in groups
            if bool(_matches(th.tensor([group[0]]), precursor_mz, sample_idx).item())
        ]
        if len(precursor_groups) > 1:
            continue
        base_intensity = max(intensity for _, intensity in groups)
        precursor_intensity = (
            precursor_groups[0][1] if precursor_groups else absent_intensity_floor * base_intensity
        )

        contender = None
        pred_contender = None
        for mass, intensity in sorted(groups, key=lambda group: group[1], reverse=True):
            if bool(_matches(th.tensor([mass]), precursor_mz, sample_idx).item()):
                continue
            matches = pred_indices[_matches(pred_sample_mzs, mass, sample_idx)]
            if matches.numel() > 0 and not th.isin(matches, pred_precursor).any():
                contender = (mass, intensity)
                pred_contender = matches
                break
        if contender is None or pred_contender is None:
            continue

        contender_intensity = contender[1]
        target_precursor = precursor_intensity / (precursor_intensity + contender_intensity)
        pair_data.append(
            (
                sample_idx,
                target_precursor,
                pred_precursor,
                pred_contender,
                precursor_intensity > contender_intensity,
            )
        )

    # 3. Gather selected aliases on-device and compute the differentiable pair loss.
    for sample_idx, target, precursor_idxs, contender_idxs, precursor_win in pair_data:
        precursor_idxs = precursor_idxs.to(pred_logprobs.device)
        contender_idxs = contender_idxs.to(pred_logprobs.device)
        pred_delta = th.logsumexp(pred_logprobs[precursor_idxs], dim=0) - th.logsumexp(
            pred_logprobs[contender_idxs], dim=0
        )
        losses[sample_idx] = F.binary_cross_entropy_with_logits(
            pred_delta, pred_logprobs.new_tensor(target)
        )
        eligible_pairs[sample_idx] = 1.0
        precursor_wins[sample_idx] = float(precursor_win)
    return losses, eligible_pairs, precursor_wins
