"""`configure_optimizers` must honour the configured LR schedule, not silently decay.

`lr_schedule_type`, `lr_total_steps` and `lr_min_ratio` used to be dropped between the config
and `build_lr_scheduler`, so a run asking for cosine annealing trained on exponential decay
instead. Six MASShatter runs in 2026-09 carried `lr_schedule_type: cosine` and were inert.
"""

import math
from types import SimpleNamespace

import pytest
import torch as th

from fragnnet.pl_model.spectrum_pl import SpectrumPL
from fragnnet.runner import load_config


def _pl(**overrides):
    config = load_config("config/models_template", None, None)
    config.update(optimizer="adamw", lr=1e-3, weight_decay=0.01, lr_schedule=True)
    config.update(overrides)
    parameter = th.nn.Parameter(th.zeros(1))
    return SimpleNamespace(
        hparams=SimpleNamespace(**config),
        parameters=lambda: iter([parameter]),
    )


def _lr_at(scheduler, optimizer, step: int) -> float:
    """Advance a fresh scheduler to ``step`` and return the resulting learning rate."""
    for _ in range(step):
        optimizer.step()
        scheduler.step()
    return optimizer.param_groups[0]["lr"]


def test_cosine_schedule_reaches_its_floor():
    """Cosine must anneal to ``lr_min_ratio`` of peak by ``lr_total_steps``."""
    total, warmup, floor = 1000, 100, 0.01
    pl = _pl(
        lr_schedule_type="cosine",
        lr_total_steps=total,
        lr_min_ratio=floor,
        lr_warmup_steps=warmup,
    )
    ret = SpectrumPL.configure_optimizers(pl)
    optimizer = ret["optimizer"]
    scheduler = ret["lr_scheduler"]["scheduler"]

    assert _lr_at(scheduler, optimizer, warmup) == pytest.approx(1e-3, rel=1e-6)
    midpoint = _lr_at(scheduler, optimizer, (total - warmup) // 2)
    expected_mid = 1e-3 * (floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * 0.5)))
    assert midpoint == pytest.approx(expected_mid, rel=1e-6)
    end = _lr_at(scheduler, optimizer, total - warmup - (total - warmup) // 2)
    assert end == pytest.approx(1e-3 * floor, rel=1e-6)


def test_cosine_is_not_silently_exponential():
    """The regression itself: exponential decay at the same step is far above cosine's floor."""
    total, warmup = 1000, 100
    cosine = SpectrumPL.configure_optimizers(
        _pl(lr_schedule_type="cosine", lr_total_steps=total, lr_min_ratio=0.0,
            lr_warmup_steps=warmup)
    )
    exponential = SpectrumPL.configure_optimizers(
        _pl(lr_schedule_type="exponential", lr_warmup_steps=warmup)
    )
    cosine_lr = _lr_at(cosine["lr_scheduler"]["scheduler"], cosine["optimizer"], total)
    exp_lr = _lr_at(exponential["lr_scheduler"]["scheduler"], exponential["optimizer"], total)
    assert cosine_lr == pytest.approx(0.0, abs=1e-12)
    assert exp_lr > 1e-4


def test_cosine_without_total_steps_raises():
    """A cosine config missing `lr_total_steps` must fail loudly, not fall back."""
    with pytest.raises(ValueError, match="total_steps"):
        SpectrumPL.configure_optimizers(
            _pl(lr_schedule_type="cosine", lr_total_steps=None, lr_min_ratio=0.0)
        )


def test_schedule_off_configures_no_scheduler():
    ret = SpectrumPL.configure_optimizers(_pl(lr_schedule=False))
    assert "lr_scheduler" not in ret
