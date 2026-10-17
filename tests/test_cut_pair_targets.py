"""Tests for the complementary-cut pair target builder.

Covers the builder-verification list in
``docs/plans/fragnnet/complementary_cut_supervision_experiment_plan.md``: the target
uses raw intensity while responsibility only moves the group weight, sub-LOD and mass-aliased
pairs produce no gradient, a one-sided supra-LOD peak yields a soft target rather than ``(1, 0)``,
and stronger evidence moves the target monotonically away from ``0.5``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from fragnnet.models.fragnnet.fragnnet_adducts import tracked_element_counts
from fragnnet.utils.formula_utils import PREC_TYPE_TO_CMF_MASS_DIFF, PREC_TYPE_TO_MASS_DIFF

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "debug", "fragnnet")
)

from build_cut_pair_targets import (  # noqa: E402
    _complement_incompleteness,
    _complementary_pairs,
    _concat_results,
    _duplicate_cmf_precursors,
    _empty_mol_result,
    _envelopes_overlap,
    _kish_ess,
    _lod_for,
    _mz_bin,
    _pad_missing_spec_ids,
    _shuffle_seed,
    cache_fingerprint,
    check_min_frag_atoms,
    estimate_lod_table,
    lod_quantile_audit,
    node_candidate_ions,
    pair_evidence,
    verify_cache_coverage,
    verify_cache_fingerprint,
)


def _mask(bits: list[int], width: int = 8) -> np.ndarray:
    out = np.zeros(width, dtype=bool)
    out[bits] = True
    return out


# --- complementary pair enumeration -------------------------------------------------------


def test_exact_partition_is_found():
    """Caffeine-like case: parent {0..3} split into {0,1} and {2,3} is a complementary pair."""
    masks = np.stack([_mask([0, 1, 2, 3]), _mask([0, 1]), _mask([2, 3])])
    edge_index = np.array([[0, 0], [1, 2]])
    assert _complementary_pairs(masks, edge_index) == [(0, 1, 2)]


def test_overlapping_children_are_not_a_pair():
    """Children sharing an atom cannot be complementary even if their union is the parent."""
    masks = np.stack([_mask([0, 1, 2]), _mask([0, 1]), _mask([1, 2])])
    edge_index = np.array([[0, 0], [1, 2]])
    assert _complementary_pairs(masks, edge_index) == []


def test_incomplete_union_is_not_a_pair():
    """Disjoint children that do not cover the parent are not complementary."""
    masks = np.stack([_mask([0, 1, 2, 3]), _mask([0]), _mask([1])])
    edge_index = np.array([[0, 0], [1, 2]])
    assert _complementary_pairs(masks, edge_index) == []


def test_pairs_are_deduplicated_across_redundant_edges():
    """Repeated parent->child edges must not create duplicate facts."""
    masks = np.stack([_mask([0, 1, 2, 3]), _mask([0, 1]), _mask([2, 3])])
    edge_index = np.array([[0, 0, 0, 0], [1, 2, 1, 2]])
    assert _complementary_pairs(masks, edge_index) == [(0, 1, 2)]


def test_single_child_parent_yields_nothing():
    masks = np.stack([_mask([0, 1]), _mask([0])])
    edge_index = np.array([[0], [1]])
    assert _complementary_pairs(masks, edge_index) == []


# --- mass-alias exclusion -----------------------------------------------------------------


def test_same_mass_sides_are_flagged_as_overlapping():
    """A symmetric cut giving two equal-mass sides cannot identify a side; it must be excluded."""
    mz = np.array([100.0])
    assert _overlap(mz, mz.copy(), tolerance=1e-5, tolerance_min_mz=200.0)


def test_well_separated_sides_do_not_overlap():
    assert not _overlap(
        np.array([100.0]), np.array([180.0]), tolerance=1e-5, tolerance_min_mz=200.0
    )


def test_overlap_is_detected_at_the_tolerance_boundary():
    """Collisions introduced purely by instrument tolerance still count as aliasing."""
    tol, min_mz = 1e-5, 200.0
    # Divisor is floored at min_mz, so the absolute window is tol * min_mz = 2 mDa.
    just_inside = 100.0 + tol * min_mz * 0.5
    just_outside = 100.0 + tol * min_mz * 2.0
    assert _overlap(np.array([100.0]), np.array([just_inside]), tol, min_mz)
    assert not _overlap(np.array([100.0]), np.array([just_outside]), tol, min_mz)


def test_candidates_1_5_tolerances_apart_are_aliased():
    """A peak between two candidates matches both, so the gate must be two tolerances wide.

    The one-tolerance form missed exactly this: A=100.0000 and B=100.0030 sit 1.5 tolerances apart
    at a 2 mDa window, yet an observation at 100.0015 is within tolerance of each. The shared peak
    would then be stripped from both directional masks while the pair still trained on the rest.
    """
    tol, min_mz = 2e-5, 100.0  # absolute window = 2e-5 * 100 = 2 mDa
    a, b, observed = 100.0000, 100.0030, 100.0015
    abs_tol = tol * min_mz

    # Precondition: the matcher really does accept the observed peak for both candidates.
    assert abs(observed - a) <= abs_tol
    assert abs(observed - b) <= abs_tol
    # And their centres are more than one tolerance apart, which is what fooled the old gate.
    assert abs(b - a) > abs_tol
    assert _overlap(np.array([a]), np.array([b]), tol, min_mz)


def test_candidates_beyond_two_tolerances_are_not_aliased():
    """The widened gate must not swallow genuinely resolvable pairs."""
    tol, min_mz = 2e-5, 100.0
    abs_tol = tol * min_mz
    assert not _overlap(np.array([100.0]), np.array([100.0 + 2.5 * abs_tol]), tol, min_mz)


def test_empty_envelope_does_not_overlap():
    assert not _overlap(np.zeros(0), np.array([100.0]), tolerance=1e-5, tolerance_min_mz=200.0)


# --- CRF/CMF candidate contract -------------------------------------------------------------


# One node carrying formula rows 1 (its delta_h = 0, column 0) and 2 (delta_h = +1). Row 0 is NULL.
_H_IDX = np.array([[1, 2]], dtype=np.int64)
_FPM = np.array([[0.0], [100.0], [101.0]], dtype=np.float64)
_ALLOWED = {1, 2}


def test_direct_charge_adduct_gets_no_cmf_channel():
    """[M+H]+ emits CRF rows only: its CMF ion is the CRF row at delta_h = -1.

    Adding a CMF channel here would put a second candidate on an m/z the CRF ladder already
    produces, inflating the match count against a peak the model scores once.
    """
    ion_mz, node_of = _ions(_H_IDX, _FPM, _ALLOWED, "[M+H]+", 1)
    std = PREC_TYPE_TO_MASS_DIFF["[M+H]+"]
    assert sorted(np.round(ion_mz, 6)) == sorted(np.round([100.0 + std, 101.0 + std], 6))
    assert set(node_of.tolist()) == {0}


def test_direct_charge_adduct_gets_no_cmf_for_any_node_layout():
    """Formula rows are shared across nodes with different H-shifts.

    Whichever row is a given node's own delta_h = 0, no node gains a CMF candidate, so the two
    nodes below differ only in which CRF rows they carry.
    """
    h_idx = np.array([[1, 2], [2, 1]], dtype=np.int64)
    ion_mz, node_of = _ions(h_idx, _FPM, _ALLOWED, "[M+H]+", 1)
    std = PREC_TYPE_TO_MASS_DIFF["[M+H]+"]
    expected = sorted(np.round([100.0 + std, 101.0 + std], 6).tolist())

    assert sorted(np.round(ion_mz[node_of == 0], 6).tolist()) == expected
    assert sorted(np.round(ion_mz[node_of == 1], 6).tolist()) == expected


def test_direct_charge_adduct_keeps_cmf_without_hydrogen_transfer():
    """At max_h_transfer = 0 the CRF ladder cannot reach delta_h = -1, so CMF is load-bearing."""
    ion_mz, _ = _ions(_H_IDX, _FPM, _ALLOWED, "[M+H]+", 0)
    std = PREC_TYPE_TO_MASS_DIFF["[M+H]+"]
    cmf = PREC_TYPE_TO_CMF_MASS_DIFF["[M+H]+"]
    assert sorted(np.round(ion_mz, 6)) == sorted(
        np.round([100.0 + std, 101.0 + std, 100.0 + cmf, 101.0 + cmf], 6)
    )


def test_cmf_contract_preserves_an_explicit_zero_h_transfer_limit():
    """The cache audit must describe the same support the target builder uses."""
    assert _duplicate_cmf_precursors(max_h_transfer=0) == []
    assert set(_duplicate_cmf_precursors(max_h_transfer=None)) == {"[M+H]+", "[M-H]-"}


def test_migrating_charge_adduct_gets_cmf_at_every_h_shift():
    """[M+Na]+ migrates the charge to H+, so the full H-shift range is valid in CMF."""
    ion_mz, _ = _ions(_H_IDX, _FPM, _ALLOWED, "[M+Na]+")
    std = PREC_TYPE_TO_MASS_DIFF["[M+Na]+"]
    cmf = PREC_TYPE_TO_CMF_MASS_DIFF["[M+Na]+"]
    assert sorted(np.round(ion_mz, 6)) == sorted(
        np.round([100.0 + std, 101.0 + std, 100.0 + cmf, 101.0 + cmf], 6)
    )


def test_bare_radical_cation_has_no_cmf_channel():
    """[M]+ is excluded from CMF entirely, matching the dataloader's include_cmf guard."""
    ion_mz, _ = _ions(_H_IDX, _FPM, _ALLOWED, "[M]+")
    std = PREC_TYPE_TO_MASS_DIFF["[M]+"]
    assert sorted(np.round(ion_mz, 6)) == sorted(np.round([100.0 + std, 101.0 + std], 6))


def test_h_transfer_budget_excludes_disallowed_rows():
    ion_mz, _ = _ions(_H_IDX, _FPM, {1}, "[M+Na]+")
    std = PREC_TYPE_TO_MASS_DIFF["[M+Na]+"]
    cmf = PREC_TYPE_TO_CMF_MASS_DIFF["[M+Na]+"]
    assert sorted(np.round(ion_mz, 6)) == sorted(np.round([100.0 + std, 100.0 + cmf], 6))


def test_rows_outside_the_h_transfer_budget_are_dropped():
    """Only formula rows the budget allows become candidates."""
    h_idx = np.array([[1, 2]], dtype=np.int64)
    ion_mz, _ = _ions(h_idx, _FPM, {2}, "[M+H]+", 1)
    std = PREC_TYPE_TO_MASS_DIFF["[M+H]+"]
    assert sorted(np.round(ion_mz, 6)) == sorted(np.round([101.0 + std], 6))


def test_precursor_loss_candidates_require_the_lost_group():
    """Target candidates must not retain rows the dataloader nulls for a CO2 loss."""
    h_idx = np.array([[1, 2]], dtype=np.int64)
    formula_counts = np.asarray(
        [tracked_element_counts(formula) for formula in ("", "C2H1O", "C2H1O2")]
    )

    ion_mz, _ = _ions(
        h_idx,
        _FPM,
        _ALLOWED,
        "[M-H-CO2]-",
        formula_element_counts=formula_counts,
    )

    expected = _FPM[2, 0] + PREC_TYPE_TO_MASS_DIFF["[M-H-CO2]-"]
    assert np.allclose(ion_mz, [expected])


def test_null_formula_row_and_zero_mass_padding_are_dropped():
    h_idx = np.array([[0, 0]], dtype=np.int64)
    ion_mz, node_of = _ions(h_idx, _FPM, _ALLOWED, "[M+H]+")
    assert ion_mz.size == 0 and node_of.size == 0


# --- min_frag_atoms gate --------------------------------------------------------------------


def test_manifest_recording_zero_passes():
    assert "manifest" in check_min_frag_atoms({"min_frag_atoms": 0}, observed_min_size=1)


def test_manifest_recording_nonzero_fails_closed():
    with pytest.raises(ValueError, match="min_frag_atoms=3"):
        check_min_frag_atoms({"min_frag_atoms": 3}, observed_min_size=3)


def test_manifest_without_the_key_refuses_to_guess():
    with pytest.raises(ValueError, match="records no min_frag_atoms"):
        check_min_frag_atoms({"max_depth": 3}, observed_min_size=1)


def test_manifest_wins_over_contradicting_observation():
    """A manifest is authoritative: a filtered corpus can still happen to show small fragments."""
    with pytest.raises(ValueError, match="min_frag_atoms=2"):
        check_min_frag_atoms({"min_frag_atoms": 2}, observed_min_size=1)


def test_without_manifest_a_one_atom_fragment_falsifies_a_threshold():
    assert "falsification" in check_min_frag_atoms(None, observed_min_size=1)


def test_absent_small_fragment_is_reported_as_inconclusive_not_as_filtering():
    """A corpus can lack 1-atom fragments without any threshold; the message must not claim one."""
    with pytest.raises(ValueError) as excinfo:
        check_min_frag_atoms(None, observed_min_size=3)
    message = str(excinfo.value)
    assert "Inconclusive" in message
    assert "not proof of filtering" in message
    assert "manifest" in message


# --- detection limit ----------------------------------------------------------------------


def _spec(inst: str, peaks: list[tuple[float, float]]) -> dict:
    return {"inst_type": inst, "peaks": peaks}


def test_lod_is_estimated_per_inst_and_mz_bin():
    rows = [
        _spec("HCD", [(50.0, 90.0), (50.5, 10.0)]),
        _spec("HCD", [(50.0, 90.0), (50.5, 10.0)]),
    ]
    by_inst_bin, by_inst, global_floor = estimate_lod_table(rows)
    # Both peaks fall in bin 0; the weakest retained share there is 10/100.
    assert by_inst_bin[("HCD", 0)] == pytest.approx(0.1)
    assert by_inst["HCD"] == pytest.approx(0.1)
    assert global_floor == pytest.approx(0.1)


def test_lod_falls_back_from_stratum_to_inst_to_global():
    by_inst_bin = {("HCD", 0): 0.01}
    by_inst = {"HCD": 0.02}
    # Known stratum wins.
    assert _lod_for("HCD", np.array([50.0]), by_inst_bin, by_inst, 0.05) == pytest.approx(0.01)
    # Unknown bin falls back to the instrument floor.
    assert _lod_for("HCD", np.array([850.0]), by_inst_bin, by_inst, 0.05) == pytest.approx(0.02)
    # Unknown instrument falls back to the global floor.
    assert _lod_for("QTOF", np.array([50.0]), by_inst_bin, by_inst, 0.05) == pytest.approx(0.05)


def test_side_level_floor_is_the_minimum_over_candidate_channels():
    """L_A is the smallest intensity retainable across any of A's candidate m/z."""
    by_inst_bin = {("HCD", 0): 0.03, ("HCD", 1): 0.005}
    got = _lod_for("HCD", np.array([50.0, 150.0]), by_inst_bin, {"HCD": 0.02}, 0.05)
    assert got == pytest.approx(0.005)


def test_empty_training_intensity_raises():
    with pytest.raises(ValueError, match="cannot estimate a LOD floor"):
        estimate_lod_table([_spec("HCD", [])])


def test_zero_total_intensity_spectrum_is_skipped_not_crashed():
    rows = [_spec("HCD", [(50.0, 0.0)]), _spec("HCD", [(50.0, 100.0), (60.0, 25.0)])]
    _, _, global_floor = estimate_lod_table(rows)
    assert global_floor == pytest.approx(0.2)


def test_mz_bin_edges():
    assert _mz_bin(0.0) == 0
    assert _mz_bin(99.9) == 0
    assert _mz_bin(100.0) == 1
    assert np.array_equal(_mz_bin(np.array([50.0, 250.0])), np.array([0, 2]))


# --- target construction (the formula the plan fixes) ---------------------------------------


def _q_cut(
    i_a: float, i_b: float, l_a: float, l_b: float, n_a: float = 1.0, n_b: float = 1.0
) -> tuple[float, float]:
    """Drive the builder's ``pair_evidence`` with one directional peak per side."""
    got = _scored(i_a, i_b, l_a, l_b, n_a, n_b)
    if got is None:
        raise AssertionError("pair_evidence rejected a pair the test expected to retain")
    return got[0], got[1]


def _scored(
    i_a: float, i_b: float, l_a: float, l_b: float, n_a: float = 1.0, n_b: float = 1.0
) -> tuple[float, float, float, float] | None:
    """One A-peak and one B-peak, each matched by ``n_a``/``n_b`` DAG nodes.

    Flattens ``pair_evidence``'s dict to ``(q_a, q_b, w_pair, w_strict)`` for compact assertions.
    """
    rel_ints = np.array([i_a, i_b], dtype=np.float64)
    n_match = np.array([n_a, n_b], dtype=np.float64)
    dir_a = np.array([True, False])
    dir_b = np.array([False, True])
    got = pair_evidence(rel_ints, n_match, dir_a, dir_b, l_a, l_b)
    if got is None:
        return None
    return got["q_a"], got["q_b"], got["w_pair"], got["w_strict"]


def _overlap(*args, **kwargs) -> bool:
    """``_envelopes_overlap`` returns (aliased, channel); tests assert on the verdict."""
    return _envelopes_overlap(*args, **kwargs)[0]


def _ions(*args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """``node_candidate_ions`` returns (ion_mz, node_of, is_cmf); drop the channel flags."""
    ion_mz, node_of, _ = node_candidate_ions(*args, **kwargs)
    return ion_mz, node_of


def test_one_sided_supra_lod_peak_gives_a_soft_target_not_one_hot():
    """The plan's worked example: 2% one-sided evidence at a 1% floor -> [0.67, 0.33]."""
    q_a, q_b = _q_cut(0.02, 0.0, 0.01, 0.01)
    assert q_a == pytest.approx(2 / 3, abs=1e-6)
    assert q_b == pytest.approx(1 / 3, abs=1e-6)
    assert q_b > 0.0  # a (1, 0) cache would fail the loader's strictly-positive check


def test_stronger_evidence_moves_the_target_monotonically_away_from_half():
    floors = (0.01, 0.01)
    qs = [_q_cut(i, 0.0, *floors)[0] for i in (0.02, 0.05, 0.20, 0.50)]
    assert all(later > earlier for earlier, later in zip(qs, qs[1:]))
    assert qs[0] > 0.5
    assert qs[-1] < 1.0


def test_twenty_percent_one_sided_peak_matches_the_plans_worked_example():
    q_a, _ = _q_cut(0.20, 0.0, 0.01, 0.01)
    assert q_a == pytest.approx(0.9524, abs=1e-4)


def test_target_is_independent_of_attribution_confidence():
    """q reads raw intensity only; responsibility must move the weight and nothing else."""
    clean = _scored(0.20, 0.0, 0.01, 0.01, n_a=1.0)
    ambiguous = _scored(0.20, 0.0, 0.01, 0.01, n_a=10.0)
    assert clean is not None and ambiguous is not None
    assert clean[0] == pytest.approx(ambiguous[0])  # identical target
    assert clean[1] == pytest.approx(ambiguous[1])
    assert ambiguous[2] == pytest.approx(clean[2] / 10.0)  # weight alone absorbs the ambiguity


def test_equal_weighted_evidence_still_gives_different_target_softness():
    """The plan's counterexample: 20% at confidence 0.1 and a clean 2% peak carry the same
    weighted evidence, but the first is far further above the noise floor and must be harder."""
    loud_ambiguous = _scored(0.20, 0.0, 0.01, 0.01, n_a=10.0)
    quiet_clean = _scored(0.02, 0.0, 0.01, 0.01, n_a=1.0)
    assert loud_ambiguous is not None and quiet_clean is not None
    assert loud_ambiguous[2] == pytest.approx(quiet_clean[2])  # same weighted evidence
    assert loud_ambiguous[0] > quiet_clean[0]  # but not the same target


def test_equal_above_lod_evidence_gives_an_exact_tie():
    q_a, q_b = _q_cut(0.1, 0.1, 0.01, 0.01)
    assert q_a == pytest.approx(0.5)
    assert q_b == pytest.approx(0.5)


def test_both_sides_sub_lod_is_dropped_by_the_gate():
    """max(I_A/L_A, I_B/L_B) > 1 is the retention gate; sub-LOD must not become a fake near-tie."""
    assert _scored(0.004, 0.002, 0.01, 0.01) is None


def test_zero_evidence_pair_is_rejected():
    assert _scored(0.0, 0.0, 0.01, 0.01) is None


def test_pair_exactly_at_the_lod_is_rejected_but_just_above_is_kept():
    assert _scored(0.01, 0.0, 0.01, 0.01) is None
    assert _scored(0.010001, 0.0, 0.01, 0.01) is not None


def test_targets_always_sum_to_one():
    for i_a, i_b in [(0.3, 0.0), (0.0, 0.3), (0.05, 0.02), (0.5, 0.5)]:
        q_a, q_b = _q_cut(i_a, i_b, 0.01, 0.01)
        assert q_a + q_b == pytest.approx(1.0)
        assert q_a > 0 and q_b > 0


# --- weights and determinism ----------------------------------------------------------------


def test_kish_ess_is_n_for_uniform_weights_and_one_for_a_spike():
    assert _kish_ess(np.ones(8)) == pytest.approx(8.0)
    spike = np.array([1.0, 1e-12, 1e-12, 1e-12])
    assert _kish_ess(spike) == pytest.approx(1.0, abs=1e-6)
    assert _kish_ess(np.zeros(0)) == 0.0


def test_multiplicity_correction_shrinks_weights_of_reused_nodes():
    """group_weights = w_pair / ((m_A + m_B) / 2), so a node in many pairs trains each one less."""
    w_pair = 0.4
    assert w_pair / ((1 + 1) / 2) == pytest.approx(0.4)
    assert w_pair / ((4 + 6) / 2) == pytest.approx(0.08)


def test_shuffle_seed_is_stable_and_pair_specific():
    base = _shuffle_seed(1729, 42, 7, 3, 9)
    assert base == _shuffle_seed(1729, 42, 7, 3, 9)  # scheduling-independent
    assert base != _shuffle_seed(1729, 42, 7, 3, 10)
    assert base != _shuffle_seed(1729, 43, 7, 3, 9)
    assert base != _shuffle_seed(1730, 42, 7, 3, 9)


def _fake_cache(q: list[float], builder: str = "cut_pair") -> dict:
    import torch as th

    return {
        "version": 2,
        "q_obs": th.tensor(q, dtype=th.float32),
        "child_nodes": th.tensor([0, 1], dtype=th.int32),
        "metadata": {"builder": builder},
        "stats": {"num_groups": 1},  # non-tensor, non-metadata: must not affect the hash
    }


def test_cache_fingerprint_is_content_based_and_stable():
    """torch.save is not byte-reproducible, so provenance must hash content, not the file."""
    base = cache_fingerprint(_fake_cache([0.6, 0.4]))
    assert base == cache_fingerprint(_fake_cache([0.6, 0.4]))  # stable across identical content
    assert base != cache_fingerprint(_fake_cache([0.7, 0.3]))  # sensitive to targets
    assert base != cache_fingerprint(_fake_cache([0.6, 0.4], builder="sibling_rank"))


def test_fingerprint_stays_verifiable_after_being_stamped_in():
    """Hashing the stamp itself would make every saved cache disagree with its own metadata."""
    cache = _fake_cache([0.6, 0.4])
    cache["metadata"]["cache_fingerprint"] = cache_fingerprint(cache)
    verify_cache_fingerprint(cache)  # must not raise
    assert cache_fingerprint(cache) == cache["metadata"]["cache_fingerprint"]


def test_fingerprint_verification_detects_tampering():
    import torch as th

    cache = _fake_cache([0.6, 0.4])
    cache["metadata"]["cache_fingerprint"] = cache_fingerprint(cache)
    cache["q_obs"] = th.tensor([0.9, 0.1], dtype=th.float32)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_cache_fingerprint(cache)


def test_fingerprint_survives_a_torch_save_round_trip(tmp_path):
    """The review's failure mode: stored and recomputed hashes diverged on the saved cache."""
    import torch as th

    cache = _fake_cache([0.6, 0.4])
    cache["metadata"]["cache_fingerprint"] = cache_fingerprint(cache)
    fp = tmp_path / "cache.pt"
    th.save(cache, fp)
    verify_cache_fingerprint(th.load(fp, weights_only=False))


# --- cache coverage ---------------------------------------------------------------------------


def test_coverage_check_passes_when_every_required_id_is_present():
    import torch as th

    cache = {"spec_ids": th.tensor([1, 2, 3], dtype=th.int64)}
    verify_cache_coverage(cache, {1, 2, 3})  # must not raise
    verify_cache_coverage(cache, {1, 3})  # a subset is fine


def test_coverage_check_fails_on_a_missing_id():
    import torch as th

    cache = {"spec_ids": th.tensor([1, 2], dtype=th.int64)}
    with pytest.raises(ValueError, match="missing 1 of 3 required spec_ids"):
        verify_cache_coverage(cache, {1, 2, 99})


def test_padding_adds_zero_group_entries_for_dropped_ids():
    """The spectrum loader is stricter than the dataset, so dropped ids still need entries."""
    built = _empty_mol_result([{"spec_id": 1}, {"spec_id": 2}])
    padded = _pad_missing_spec_ids([built], {1, 2, 7, 8})
    all_ids = sorted(int(s) for r in padded for s in r["spec_ids"].tolist())
    assert all_ids == [1, 2, 7, 8]
    # Padded spectra contribute no groups, hence no gradient.
    pad_block = padded[-1]
    assert pad_block["spec_group_counts"].tolist() == [0, 0]
    assert pad_block["group_weights"].size == 0


def test_padding_is_a_no_op_when_coverage_is_already_total():
    built = _empty_mol_result([{"spec_id": 1}, {"spec_id": 2}])
    assert _pad_missing_spec_ids([built], {1, 2}) == [built]


def test_padded_cache_concatenates_into_a_valid_ragged_structure():
    """End-to-end of the guarantee: pad -> concat -> coverage holds and pointers stay consistent."""
    real = _empty_mol_result([{"spec_id": 5}])
    real["spec_group_counts"] = np.asarray([1], dtype=np.int32)
    real["group_child_counts"] = np.asarray([2], dtype=np.int32)
    real["child_nodes"] = np.asarray([3, 4], dtype=np.int32)
    real["q_obs"] = np.asarray([0.7, 0.3], dtype=np.float32)
    real["q_obs_shuffled"] = np.asarray([0.3, 0.7], dtype=np.float32)
    real["group_weights"] = np.asarray([0.5], dtype=np.float32)

    required = {5, 11, 12}
    cache = _concat_results(_pad_missing_spec_ids([real], required))
    verify_cache_coverage(cache, required)

    assert cache["spec_ids"].tolist() == [5, 11, 12]
    # Only spectrum 5 owns a group; the padded pair contribute none.
    assert cache["spec_group_ptr"].tolist() == [0, 1, 1, 1]
    assert cache["group_child_ptr"].tolist() == [0, 2]
    assert cache["child_nodes"].tolist() == [3, 4]
    assert cache["stats"]["num_spectra"] == 3
    assert cache["stats"]["num_groups"] == 1


def test_duplicate_spec_ids_are_rejected():
    a = _empty_mol_result([{"spec_id": 1}])
    b = _empty_mol_result([{"spec_id": 1}])
    with pytest.raises(ValueError, match="duplicate spec_id"):
        _concat_results([a, b])


# --- strict-versus-responsibility weighting ---------------------------------------------------


def test_strict_weight_counts_only_uniquely_matched_peaks():
    """Responsibility discounts an ambiguous peak; strict drops it outright (audit-only)."""
    clean = _scored(0.20, 0.0, 0.01, 0.01, n_a=1.0)
    ambiguous = _scored(0.20, 0.0, 0.01, 0.01, n_a=4.0)
    assert clean is not None and ambiguous is not None
    assert clean[3] == pytest.approx(0.20)  # w_strict keeps the full intensity
    assert ambiguous[3] == pytest.approx(0.0)  # n_j > 1 contributes nothing to strict
    assert ambiguous[2] == pytest.approx(0.05)  # but responsibility keeps 1/4 of it


# --- LOD quantile audit -----------------------------------------------------------------------


def test_lod_quantile_audit_is_monotone_across_quantiles():
    """A wide spread here means the sub-LOD filter is an artefact of the estimator choice."""
    rows = [_spec("HCD", [(50.0, 100.0), (60.0, float(k))]) for k in range(1, 40)]
    audit = lod_quantile_audit(rows)
    values = [audit[f"q{q:g}"] for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
    assert values == sorted(values)
    assert values[0] < values[-1]


def test_lod_quantile_changes_the_estimated_floor():
    rows = [_spec("HCD", [(50.0, 100.0), (60.0, float(k))]) for k in range(1, 40)]
    low = estimate_lod_table(rows, quantile=0.05)[2]
    high = estimate_lod_table(rows, quantile=0.95)[2]
    assert high > low


# --- complement-completeness diagnostic -------------------------------------------------------


def test_complement_incompleteness_flags_a_missing_counter_fragment():
    """Parent {0,1,2,3} has children {0,1} and {2}; {2}'s complement {0,1,3} is absent."""
    masks = np.stack([_mask([0, 1, 2, 3]), _mask([0, 1]), _mask([2])])
    edge_index = np.array([[0, 0], [1, 2]])
    assert _complement_incompleteness(masks, edge_index) == 2


def test_complement_incompleteness_is_zero_for_a_clean_partition():
    masks = np.stack([_mask([0, 1, 2, 3]), _mask([0, 1]), _mask([2, 3])])
    edge_index = np.array([[0, 0], [1, 2]])
    assert _complement_incompleteness(masks, edge_index) == 0


def test_shuffled_target_is_a_permutation_of_the_true_target():
    target = np.asarray([0.67, 0.33], dtype=np.float32)
    for spec_id in range(50):
        rng = np.random.default_rng(_shuffle_seed(1729, spec_id, 0, 1, 2))
        shuffled = target[rng.permutation(2)]
        assert sorted(shuffled.tolist()) == sorted(target.tolist())
