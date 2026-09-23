"""Tests for the intensity-head memorization probe's corruption scopes."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch as th

from fragnnet.model.bond_fragment_heads import CandidateIntensityTransformerHead

SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "debug" / "debug_pooled_embedding_memorization.py"
)
SPEC = importlib.util.spec_from_file_location("debug_pooled_embedding_memorization", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
# The probe defines a dataclass, which resolves its own module through sys.modules.
sys.modules[SPEC.name] = PROBE
SPEC.loader.exec_module(PROBE)

EMBEDDING_DIM = 6
CONTEXT_DIM = 3
FEATURE_DIM = 2
# Two molecules with five and three candidates, interleaved as the model may deliver them.
BATCH_INDICES = th.tensor([0, 1, 0, 0, 1, 0, 1, 0])


def _head(attention_type: str) -> CandidateIntensityTransformerHead:
    th.manual_seed(0)
    head = CandidateIntensityTransformerHead(
        fragment_embedding_dim=EMBEDDING_DIM,
        context_dim=CONTEXT_DIM,
        model_dim=12,
        num_layers=2,
        num_heads=3,
        feedforward_dim=24,
        num_inducing_points=4,
        num_ion_modes=2,
        num_h_shifts=3,
        dropout=0.0,
        use_base_probability_feature=False,
        attention_type=attention_type,
        candidate_feature_dim=FEATURE_DIM,
        unique_fragment_tokens=True,
    )
    # The residual head is zero-initialized, which would make every output identical.
    th.nn.init.normal_(head.component_residual_head.weight)
    return head.eval()


def _inputs() -> tuple[tuple[th.Tensor, ...], dict]:
    generator = th.Generator().manual_seed(1)
    num_components = BATCH_INDICES.shape[0]
    blocks = tuple(th.randn(num_components, EMBEDDING_DIM, generator=generator) for _ in range(3))
    args = (
        *blocks,
        th.zeros(num_components),
        th.zeros(num_components),
        BATCH_INDICES,
        th.arange(num_components),
    )
    kwargs = {
        "batch_size": 2,
        "num_masks": num_components,
        "context": th.randn(num_components, CONTEXT_DIM, generator=generator),
        "candidate_features": th.randn(num_components, FEATURE_DIM, generator=generator),
    }
    return args, kwargs


def _residuals(
    head: CandidateIntensityTransformerHead,
    condition_label: str,
    set_drop_fraction: float = 0.5,
) -> th.Tensor:
    conditions = {
        condition.label: condition
        for condition in PROBE.build_conditions(["cc", "features", "set"], ["zero"])
    }
    args, kwargs = _inputs()
    with th.no_grad(), PROBE.PooledEmbeddingCorruptor(head, 7, set_drop_fraction) as corruptor:
        corruptor.condition = conditions[condition_label]
        return corruptor.head.forward(*args, **kwargs)[0]


def test_build_conditions_gives_set_scope_one_condition() -> None:
    labels = [
        condition.label
        for condition in PROBE.build_conditions(["features", "set"], ["molmean", "zero"])
    ]

    assert labels == ["none", "features_molmean", "features_zero", "set_drop"]
    with pytest.raises(ValueError, match="Unknown corruption scopes"):
        PROBE.build_conditions(["context"], ["zero"])


def test_drop_from_keys_keeps_padding_and_one_real_candidate_per_molecule() -> None:
    padding_mask = th.tensor(
        [
            [False, False, False, True],
            [True, False, False, True],
        ]
    )

    masked = PROBE.drop_from_keys(padding_mask, 0.999, th.Generator().manual_seed(0))

    assert masked.shape == padding_mask.shape
    assert masked[padding_mask].all()
    # Near-certain hiding empties both rows, so each keeps exactly its first real slot.
    assert th.equal(masked, th.tensor([[False, True, True, True], [True, False, True, True]]))


@pytest.mark.parametrize("attention_type", ["induced", "full"])
def test_set_drop_at_zero_fraction_is_the_identity(attention_type: str) -> None:
    head = _head(attention_type)

    reference = _residuals(head, "none")
    unmasked = _residuals(head, "set_drop", set_drop_fraction=0.0)

    assert th.equal(unmasked, reference)


@pytest.mark.parametrize("attention_type", ["induced", "full"])
def test_set_drop_changes_context_but_still_scores_every_candidate(attention_type: str) -> None:
    head = _head(attention_type)

    reference = _residuals(head, "none")
    dropped = _residuals(head, "set_drop", set_drop_fraction=0.5)

    assert dropped.shape == reference.shape == (BATCH_INDICES.shape[0],)
    assert th.isfinite(dropped).all()
    assert not th.allclose(dropped, reference)


def test_features_scope_corrupts_only_the_descriptor_block() -> None:
    head = _head("induced")
    args, kwargs = _inputs()
    with th.no_grad():
        expected = head(*args, **{**kwargs, "candidate_features": th.zeros(8, FEATURE_DIM)})[0]
        cc_zeroed = head(*(th.zeros_like(args[0]), th.zeros_like(args[1]), *args[2:]), **kwargs)[0]

    corrupted = _residuals(head, "features_zero")

    assert th.allclose(corrupted, expected)
    assert th.allclose(_residuals(head, "cc_zero"), cc_zeroed)


def test_restore_returns_the_uncorrupted_head() -> None:
    head = _head("induced")
    args, kwargs = _inputs()
    with th.no_grad():
        reference = head(*args, **kwargs)[0]
        with PROBE.PooledEmbeddingCorruptor(head, 7, 0.5) as corruptor:
            corruptor.condition = PROBE.build_conditions(["set"], [])[1]
            corruptor.head.forward(*args, **kwargs)
        restored = head(*args, **kwargs)[0]

    assert th.equal(restored, reference)


def test_corruptor_rejects_an_out_of_range_drop_fraction() -> None:
    with pytest.raises(ValueError, match="set_drop_fraction"):
        PROBE.PooledEmbeddingCorruptor(_head("induced"), 7, 1.0)
