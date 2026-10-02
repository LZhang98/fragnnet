"""Scientific edge cases for the diagnostic peak preference objective."""

import pytest
import torch as th

from scripts.debug.debug_masster_peak_ranking_pilot import peak_ranking_loss, sample_specs


def test_mass_aggregation_and_gradient():
    # Two structures share the correct mass; their sum already beats the competitor.
    logp = th.tensor([0.3, 0.3, 0.4]).log().requires_grad_()
    loss, valid = peak_ranking_loss(
        th.tensor([100.0, 100.0, 150.0]),
        logp,
        th.tensor([100.0, 150.0]),
        th.tensor([1.0, 0.2]),
        1e-5,
        200.0,
        0.1,
    )
    assert valid
    assert float(loss.detach()) == pytest.approx(0.8 * th.log(th.tensor(1 + 0.4 / 0.6)).item())
    loss.backward()
    assert (logp.grad[:2] < 0).all()
    assert logp.grad[2] > 0


@pytest.mark.parametrize("target_mz,target_int", [([200.0], [1.0]), ([100.0, 150.0], [1.0, 0.95])])
def test_uncovered_and_near_tied_targets_skip(target_mz, target_int):
    logp = th.tensor([0.4, 0.6]).log().requires_grad_()
    loss, valid = peak_ranking_loss(
        th.tensor([100.0, 150.0]),
        logp,
        th.tensor(target_mz),
        th.tensor(target_int),
        1e-5,
        200.0,
        0.1,
    )
    assert not valid
    assert loss.item() == 0
    loss.backward()
    assert th.equal(logp.grad, th.zeros(2))


def test_multiple_compatible_masses_are_positive_not_negative():
    logp = th.tensor([0.2, 0.3, 0.5]).log().requires_grad_()
    loss, valid = peak_ranking_loss(
        th.tensor([100.0, 100.001, 150.0]),
        logp,
        th.tensor([100.0]),
        th.tensor([1.0]),
        1e-5,
        200.0,
        0.1,
    )
    assert valid
    assert loss.item() == pytest.approx(th.log(th.tensor(2.0)).item())
    loss.backward()
    assert (logp.grad[:2] < 0).all()


def test_wrong_order_costs_more_than_correct_order():
    args = (th.tensor([100.0, 150.0]), th.tensor([1.0, 0.2]), 1e-5, 200.0, 0.1)
    wrong, _ = peak_ranking_loss(th.tensor([100.0, 150.0]), th.tensor([0.2, 0.8]).log(), *args)
    correct, _ = peak_ranking_loss(th.tensor([100.0, 150.0]), th.tensor([0.8, 0.2]).log(), *args)
    assert wrong > correct


def test_sampling_is_identity_based_and_target_independent():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "inchikey_s": ["a", "a", "a", "b", "b"],
            "mol_id": [1, 1, 2, 3, 3],
            "spec_id": range(5),
            "intensity": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    selected = sample_specs(frame, 2, 2, 42)
    changed = sample_specs(frame.assign(intensity=0.0), 2, 2, 42)
    assert selected.spec_id.tolist() == changed.spec_id.tolist()
    assert selected.groupby("inchikey_s").size().to_dict() == {"a": 2, "b": 2}
