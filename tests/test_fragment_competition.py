"""Tests for top-k fragment competition in FraGNNet."""

import pytest
import torch as th

from fragnnet.model.fragment_competition import TopKFragmentTransformer


def _build(
    top_k: int = 1,
    *,
    context_dim: int = 2,
    detach_inputs: bool = True,
    max_delta: float = 5.0,
) -> TopKFragmentTransformer:
    return TopKFragmentTransformer(
        fragment_dim=6,
        context_dim=context_dim,
        top_k=top_k,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        max_delta=max_delta,
        detach_inputs=detach_inputs,
    )


def _inputs() -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    channel_logits = th.tensor(
        [
            [4.0, 3.0, 100.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [5.0, 2.0, -1.0],
        ]
    )
    fragment_embeddings = th.randn(4, 6)
    fragment_batch_idxs = th.tensor([0, 0, 1, 1])
    real_channel_mask = th.tensor(
        [
            [True, True, False],
            [True, True, True],
            [True, True, False],
            [True, True, True],
        ]
    )
    context = th.randn(2, 2)
    return (
        channel_logits,
        fragment_embeddings,
        fragment_batch_idxs,
        real_channel_mask,
        context,
    )


def test_selects_top_k_fragments_per_spectrum() -> None:
    module = _build(top_k=2)
    scores = th.tensor([5.0, 1.0, 4.0, 2.0, 9.0, 8.0])
    batch_idxs = th.tensor([0, 0, 0, 1, 1, 1])
    real_mask = th.tensor([True, True, False, True, True, True])

    selected = module.select_fragment_indices(
        scores,
        batch_idxs,
        real_mask,
        batch_size=2,
    )

    assert th.equal(selected, th.tensor([0, 1, 4, 5]))


def test_zero_initialized_residual_preserves_base_logits() -> None:
    module = _build()
    module.eval()
    inputs = _inputs()

    actual = module(*inputs[:-1], inputs[-1], batch_size=2)

    assert th.equal(actual, inputs[0])


def test_selected_fragment_gets_one_shared_valid_channel_residual() -> None:
    module = _build(top_k=1)
    module.eval()
    with th.no_grad():
        module.residual_head.bias.fill_(1.0)
    inputs = _inputs()

    actual = module(*inputs[:-1], inputs[-1], batch_size=2)
    difference = actual - inputs[0]

    assert difference[0, 0] != 0
    assert th.allclose(difference[0, :2], difference[0, :1])
    assert difference[0, 2] == 0
    assert th.equal(difference[1], th.zeros(3))
    assert th.equal(difference[2], th.zeros(3))
    assert th.allclose(difference[3], difference[3, :1])


def test_context_changes_learned_fragment_residual() -> None:
    th.manual_seed(13)
    module = _build(top_k=1)
    module.eval()
    with th.no_grad():
        module.residual_head.weight.fill_(0.25)
    inputs = _inputs()
    zero_context = th.zeros_like(inputs[-1])
    shifted_context = th.ones_like(inputs[-1])

    zero_output = module(*inputs[:-1], zero_context, batch_size=2)
    shifted_output = module(*inputs[:-1], shifted_context, batch_size=2)

    assert not th.allclose(zero_output, shifted_output)


def test_residual_is_smoothly_bounded() -> None:
    module = _build(top_k=1, max_delta=0.25)
    module.eval()
    with th.no_grad():
        module.residual_head.bias.fill_(100.0)
    inputs = _inputs()

    difference = module(*inputs[:-1], inputs[-1], batch_size=2) - inputs[0]

    assert difference.abs().max() <= 0.25
    assert difference.abs().max() > 0.24


def test_detached_inputs_train_only_the_residual_branch() -> None:
    module = _build(top_k=1, detach_inputs=True)
    with th.no_grad():
        module.residual_head.weight.fill_(0.25)
    inputs = list(_inputs())
    inputs[0] = inputs[0].requires_grad_()
    inputs[1] = inputs[1].requires_grad_()
    inputs[4] = inputs[4].requires_grad_()

    output = module(*inputs[:-1], inputs[-1], batch_size=2)
    output.sum().backward()

    assert inputs[0].grad is not None
    assert th.equal(inputs[0].grad, th.ones_like(inputs[0]))
    assert inputs[1].grad is None
    assert inputs[4].grad is None
    assert module.residual_head.weight.grad is not None
    assert th.isfinite(module.residual_head.weight.grad).all()
    assert th.any(module.residual_head.weight.grad != 0)


def test_fewer_fragments_than_top_k_remains_finite() -> None:
    module = _build(top_k=8)
    inputs = _inputs()

    actual = module(*inputs[:-1], inputs[-1], batch_size=2)

    assert th.isfinite(actual).all()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"top_k": 0}, "positive"),
        ({"hidden_dim": 7}, "divisible"),
        ({"dropout": 1.1}, "dropout"),
        ({"max_delta": 0.0}, "max_delta"),
    ],
)
def test_invalid_configuration_raises(override: dict, message: str) -> None:
    kwargs = {
        "fragment_dim": 6,
        "context_dim": 2,
        "top_k": 2,
        "hidden_dim": 8,
        "num_layers": 1,
        "num_heads": 2,
        "dropout": 0.0,
        "max_delta": 5.0,
        "detach_inputs": True,
    }
    kwargs.update(override)
    with pytest.raises(ValueError, match=message):
        TopKFragmentTransformer(**kwargs)
