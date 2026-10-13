"""Training records for spectral auxiliaries and MASSter training stages.

BCE objectives were retired on 2026-10-12. Records now cover spectral leak,
spectral CE auxiliary, proposal-only stages and RT/CCS heads. Golden totals,
gradients and optimizer steps intentionally reflect removal of CE-switch BCE.
Regenerate deliberately: REORG_UPDATE_GOLDEN=1 pytest tests/test_reorg_losses.py
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import reorg_fixtures as F
import torch as th
from reorg_golden import ratchet_json, ratchet_records

GOLDEN = "losses"
MODELS = ("frag_gnn", "masster")
TRAIN_SPEC_IDS = [s.spec_id for s in F.DEFAULT_SPECTRA if s.split == "train"]
N_TRAIN = len(TRAIN_SPEC_IDS)
# The batch the main records train on: the whole train split, in the order the CE-context sampler
# draws for epoch 0 (the sampler is seeded by (seed, epoch); the epoch is pinned in train_batch).
PLAIN_BATCH_ORDER = [0, 1, 3, 4, 5, 6, 2]
# BatchNorm statistics change in every training forward pass, with or without a gradient.
BATCH_NORM_BUFFERS = ("running_mean", "running_var", "num_batches_tracked")


def batch_overrides(model_type: str, n_train: int) -> dict[str, Any]:
    """Overrides that make one batch hold the whole train split and keep the step deterministic.

    ``train_batch_size`` equal to the spectrum count makes FraGNNet's manual optimization step
    inside ``training_step`` (``record_one_step`` asserts ``global_step == 1``). The template
    default ``oos_tolerance_multiple`` of -1 selects untruncated Gaussian CE. Keep the
    recorded fixture's finite window of 1, matching its original tracked configuration.

    Args:
        model_type: ``"frag_gnn"`` or ``"masster"``.
        n_train: Number of train spectra in the dataset.

    Returns:
        The override mapping for ``F.make_config``.
    """
    overrides: dict[str, Any] = {"train_batch_size": n_train}
    if model_type == "frag_gnn":
        overrides["oos_tolerance_multiple"] = 1.0
    return overrides


PLAIN_STEP = {model_type: batch_overrides(model_type, N_TRAIN) for model_type in MODELS}
FRAG_GNN_OVERRIDES = PLAIN_STEP["frag_gnn"]
MASSTER_OVERRIDES = PLAIN_STEP["masster"]

# Proposal-only warmup as the stage recipe sets it up (masster_pl.py:2227-2235).
PROPOSAL_STAGE: dict[str, Any] = {
    "masster_unique_fragment_mode": True,
    "masster_proposal_selection_only": True,
    "masster_proposal_anchored": True,
    "masster_proposal_loss_weight": 0.1,
}
PROPOSAL_ONLY: dict[str, Any] = {**PROPOSAL_STAGE, "masster_proposal_only_until_epoch": 0}
SPECTRAL_OFF: dict[str, Any] = {
    "spectral_leak_loss_weight": 0.0,
    "spectral_ce_auxiliary_weight": 0.0,
}

# The ion-mode proposal term needs a non-degenerate CMF adduct, so two spectra become [M+Na]+.
# The head, its binary-logit semantics and a positive weight must be enabled together.
SODIUM_SPEC_IDS = (5, 6)
ION_MODE_HEAD: dict[str, Any] = {
    "masster_separate_ion_mode_proposal_head": True,
    "masster_ion_mode_proposal_binary_logits": True,
    "masster_ion_mode_proposal_loss_weight": 0.05,
    "spec_params": {"prec_types": ["[M+H]+", "[M+Na]+"]},
}

# RT (seconds) and CCS labels per spec_id. Spectrum 2 has no RT, spectrum 4 has no CCS and
# spectrum 5 has a non-positive CCS; both count as missing
# (rt_ccs_heads.py, property_regression_terms).
RT_LABELS = {0: 120.5, 1: 133.25, 2: np.nan, 3: 301.0, 4: 310.5, 5: 45.75, 6: 52.0}
CCS_LABELS = {0: 131.2, 1: 133.9, 2: 150.4, 3: 160.0, 4: np.nan, 5: -1.0, 6: 171.3}
PROPERTY_HEADS: dict[str, Any] = {
    "masster_predict_rt": True,
    "masster_predict_ccs": True,
    "spec_params": {"rt": True, "ccs": True},
}
# The two deprecated aliases (config/template.yml) that each enable both heads and both label sets.
PROPERTY_ALIASES: dict[str, Any] = {
    "masster_predict_rt_ccs": True,
    "spec_params": {"rt_ccs": True},
}
PROPERTY_LOSS_TERMS = (
    "rt_loss",
    "rt_mae",
    "rt_label_fraction",
    "ccs_loss",
    "ccs_mae",
    "ccs_label_fraction",
    "ccs_mape",
)
SPECTRAL_TERMS = ("spectral_leak_loss", "spectral_ce_auxiliary")
# masster_predict_rt_ccs is the deprecated alias that enables both heads.
HEAD_FLAG_PROPERTIES = {
    "masster_predict_rt": ("rt",),
    "masster_predict_ccs": ("ccs",),
    "masster_predict_rt_ccs": ("rt", "ccs"),
}

# Wider match windows than the 1e-5 relative one of the main steps: 1e-2 gives a 2 Da window below
# m/z 200 and 1e-1 a 20 Da one, which covers every predicted peak of the fixture molecules.
WIDER_TOLERANCES = (1.0e-2, 1.0e-1)

@dataclass
class Step:
    """One recorded optimizer step.

    Attributes:
        config: The resolved config the step ran with.
        batch: The collated train batch (never mutated; the step trains on a deep copy).
        record: The flat ``{key: tensor}`` record of ``reorg_fixtures.record_one_step``.
        params_before: Every state-dict tensor of the freshly built module, before the step.
    """

    config: dict
    batch: dict
    record: dict[str, th.Tensor]
    params_before: dict[str, th.Tensor]

    def loss(self, term: str) -> th.Tensor:
        """Per-spectrum loss term ``term``, shape ``(batch_size,)``."""
        return self.record[f"loss/{term}"]

    def grad_names(self) -> list[str]:
        """Sorted names of the parameters that received a gradient."""
        return sorted(k.removeprefix("grad/") for k in self.record if k.startswith("grad/"))


def add_property_labels(data: F.SynthData) -> None:
    """Add the ``rtinseconds`` and ``ccs`` columns that ``spec_params.rt``/``ccs`` read.

    Args:
        data: A dataset written by ``write_synth_data``; its spec_df pickle is rewritten in place.
    """
    spec_df = pd.read_pickle(data.spec_fp)
    spec_df["rtinseconds"] = spec_df["spec_id"].map(RT_LABELS).astype("float64")
    spec_df["ccs"] = spec_df["spec_id"].map(CCS_LABELS).astype("float64")
    spec_df.to_pickle(data.spec_fp)
    data.spec_df = spec_df


def train_batch(config: dict) -> dict:
    """The train batch the runner's dataloader yields for epoch 0.

    Args:
        config: A config from ``make_config`` with data paths filled in.

    Returns:
        The single collated batch holding the whole train split.
    """
    train_ids = pd.read_csv(Path(config["split_dp"]) / "train_ids.csv")["spec_id"].tolist()
    dataset, loader = F.init_split(config, "train")
    try:
        # CE-context samplers draw their groups from (seed, epoch); pin epoch 0.
        loader.batch_sampler.set_epoch(0)
        batches = list(loader)
    finally:
        F.close_dataset(dataset)
    assert len(batches) == 1, [int(b["batch_size"]) for b in batches]
    assert sorted(batches[0]["spec_unique_id"].tolist()) == sorted(train_ids)
    return batches[0]


def run_step(model_type: str, data: F.SynthData, *overrides: dict[str, Any]) -> Step:
    """Build a fresh module, take the train batch and record one optimizer step.

    Args:
        model_type: ``"frag_gnn"`` or ``"masster"``.
        data: The synthetic dataset.
        *overrides: Override mappings applied after the enabled-loss ones.

    Returns:
        The step with its config, batch, record and the parameters before the step.
    """
    config = F.make_config(model_type, data, *overrides, enabled_losses=True)
    with F.deterministic():
        batch = train_batch(config)
        module = F.build_module(config)
        before = {k: v.detach().clone() for k, v in module.state_dict().items()}
        record = F.record_one_step(module, batch)
    return Step(config, batch, record, before)


def batch_order(step: Step) -> dict[str, list[int]]:
    """Which spectra the recorded batch holds, in batch order (an environment-free record)."""
    return {
        "spec_unique_id": step.batch["spec_unique_id"].tolist(),
        "group_id": step.batch["group_id"].tolist(),
    }


def ratchet_step(key: str, step: Step) -> None:
    """Compare a step's batch order and bit-exact record with the golden file."""
    ratchet_json(GOLDEN, f"{key}:batch", batch_order(step))
    ratchet_records(GOLDEN, key, F.digest_record(step.record))


def spec_ids(step: Step) -> list[int]:
    """``spec_id`` of each batch position."""
    return step.batch["spec_unique_id"].reshape(-1).tolist()


def assert_plain_batch(step: Step) -> None:
    """The step trained on the whole default train split, in the sampler's epoch-0 order.

    The order is asserted literally so a sampler change fails here with a readable message; the
    golden ``<key>:batch`` record repeats it for every step.
    """
    groups = {s.spec_id: s.group_id for s in F.DEFAULT_SPECTRA}
    assert spec_ids(step) == PLAIN_BATCH_ORDER
    assert step.batch["group_id"].reshape(-1).tolist() == [groups[i] for i in PLAIN_BATCH_ORDER]


def assert_spectral_terms(step: Step) -> None:
    """The two auxiliary terms exist, are finite per spectrum and behave as their definitions say."""
    n = int(step.batch["batch_size"])
    for term in (*SPECTRAL_TERMS, "loss"):
        value = step.loss(term)
        assert value.shape == (n,), (term, value.shape)
        assert th.isfinite(value).all(), (term, value)
    # The leak is a fraction of predicted intensity outside the match windows.
    leak = step.loss("spectral_leak_loss")
    assert ((leak >= 0) & (leak <= 1)).all(), leak
    assert (leak > 0).any(), leak
    # An all-zero auxiliary would leave its code path unexercised by the record.
    assert (step.loss("spectral_ce_auxiliary") != 0).any(), step.loss("spectral_ce_auxiliary")


def assert_reductions(step: Step) -> None:
    """Total loss, its sum over the batch and the mean (unit sample weights) agree."""
    loss = step.loss("loss")
    total, weight = step.record["reduce/total_loss"], step.record["reduce/total_weight"]
    assert weight.item() == float(len(loss))
    th.testing.assert_close(total, loss.sum(), rtol=1e-5, atol=1e-6)
    th.testing.assert_close(step.record["reduce/mean_loss"], total / weight, rtol=1e-5, atol=1e-6)


def assert_joint_total(step: Step, extra: th.Tensor | None = None) -> None:
    """``loss`` is the primary loss plus each weighted auxiliary term (plus ``extra``)."""
    config = step.config
    expected = config["primary_loss_weight"] * step.loss("primary_loss")
    for term in SPECTRAL_TERMS:
        expected = expected + config[f"{term}_weight"] * step.loss(term)
    if extra is not None:
        expected = expected + extra
    th.testing.assert_close(step.loss("loss"), expected, rtol=1e-5, atol=1e-6)


def assert_step_updated_weights(step: Step) -> None:
    """Gradients are finite and non-trivial; the step moved the parameters that had a gradient.

    A parameter with a nonzero gradient must move. One with an all-zero gradient (for example a
    bias a softmax is invariant to) need not, and one without a gradient may only change if it is
    a BatchNorm buffer.
    """
    grads = {k.removeprefix("grad/"): v for k, v in step.record.items() if k.startswith("grad/")}
    assert grads
    assert all(th.isfinite(g).all() for g in grads.values())
    assert any(g.abs().sum() > 0 for g in grads.values())
    for name, before in step.params_before.items():
        moved = not th.equal(step.record[f"param_after/{name}"], before)
        if name not in grads:
            assert not moved or name.endswith(
                BATCH_NORM_BUFFERS
            ), f"{name} moved without a gradient"
        elif grads[name].abs().sum() > 0:
            assert moved, f"{name} has a nonzero gradient but did not move"


def assert_proposal_only_total(step: Step) -> th.Tensor:
    """In the warmup the total is exactly the weighted proposal loss (plus the ion-mode term).

    Args:
        step: A MASSter step taken with ``masster_proposal_only_until_epoch`` set.

    Returns:
        The expected per-spectrum total, which equals ``loss`` bit for bit.
    """
    config = step.config
    expected = config["masster_proposal_loss_weight"] * step.loss("proposal_loss")
    if config["masster_ion_mode_proposal_loss_weight"] > 0.0:
        ion_mode = step.loss("ion_mode_proposal_loss")
        expected = expected + config["masster_ion_mode_proposal_loss_weight"] * ion_mode
    assert (expected > 0).all()
    assert th.equal(step.loss("loss"), expected), (step.loss("loss"), expected)
    assert_reductions(step)
    return expected


def assert_same_update(first: Step, second: Step) -> None:
    """Both steps have the same gradient names and bit-identical gradients and parameters."""
    assert first.grad_names() == second.grad_names()
    for key, value in first.record.items():
        if key.startswith(("grad/", "param_after/")):
            assert th.equal(value, second.record[key]), key


def assert_no_intensity_head_update(only: Step) -> None:
    """The intensity head got no gradient and kept its initial parameters in the warmup."""
    assert not [n for n in only.grad_names() if "intensity_head" in n]
    for name, before in only.params_before.items():
        if "intensity_head" in name and not name.endswith(BATCH_NORM_BUFFERS):
            assert th.equal(only.record[f"param_after/{name}"], before), name


@pytest.fixture(scope="module")
def plain_data(tmp_path_factory: pytest.TempPathFactory) -> F.SynthData:
    """The default fixture: 6 molecules, 10 spectra and FraGNNet DAGs."""
    return F.write_synth_data(tmp_path_factory.mktemp("losses_plain"))


@pytest.fixture(scope="module")
def labeled_data(tmp_path_factory: pytest.TempPathFactory) -> F.SynthData:
    """The default fixture plus synthetic RT and CCS labels."""
    data = F.write_synth_data(tmp_path_factory.mktemp("losses_labeled"))
    add_property_labels(data)
    return data


@pytest.fixture(scope="module")
def sodium_data(tmp_path_factory: pytest.TempPathFactory) -> F.SynthData:
    """The default fixture with spectra 5 and 6 recorded as [M+Na]+ (peaks from the DAG)."""
    spectra = tuple(
        replace(s, prec_type="[M+Na]+") if s.spec_id in SODIUM_SPEC_IDS else s
        for s in F.DEFAULT_SPECTRA
    )
    return F.write_synth_data(tmp_path_factory.mktemp("losses_sodium"), spectra=spectra)


@pytest.fixture(scope="module")
def plain_steps(plain_data: F.SynthData) -> dict[str, Step]:
    """The item-4 step of each model on the default train batch."""
    return {m: run_step(m, plain_data, PLAIN_STEP[m]) for m in MODELS}


@pytest.fixture(scope="module")
def proposal_runs(plain_data: F.SynthData) -> dict[str, Step]:
    """Proposal-only steps and their controls, with identical weights and batch.

    ``only`` is the stage under test. ``joint`` is the same config without the warmup. ``bare`` is
    ``only`` with the remaining spectral weights at zero, so its gradient is the proposal loss alone.
    """
    return {
        "only": run_step("masster", plain_data, MASSTER_OVERRIDES, PROPOSAL_ONLY),
        "joint": run_step("masster", plain_data, MASSTER_OVERRIDES, PROPOSAL_STAGE),
        "bare": run_step("masster", plain_data, MASSTER_OVERRIDES, PROPOSAL_ONLY, SPECTRAL_OFF),
    }


# --------------------------------------------------------------------------------------------
# Item 4: enabled-loss records for FraGNNet and MASSter
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", MODELS)
def test_enabled_loss_record(model_type: str, plain_steps: dict[str, Step]) -> None:
    """Leak and CE auxiliary are active in one step; the record matches golden."""
    step = plain_steps[model_type]
    assert step.config["automatic_optimization"] is (model_type == "masster")
    assert_plain_batch(step)
    assert_spectral_terms(step)
    assert_reductions(step)
    assert_joint_total(step)
    assert_step_updated_weights(step)
    for term in SPECTRAL_TERMS:
        assert f"loss/{term}" in step.record
    ratchet_step(f"item4:{model_type}", step)


def test_alias_model_type_gives_identical_frag_gnn_record(
    plain_data: F.SynthData, plain_steps: dict[str, Step]
) -> None:
    """``fragnnet`` is an alias of ``frag_gnn``: the same step gives the same bits."""
    alias = run_step("fragnnet", plain_data, FRAG_GNN_OVERRIDES)
    base = plain_steps["frag_gnn"]
    assert sorted(alias.record) == sorted(base.record)
    differing = [k for k in base.record if not th.equal(alias.record[k], base.record[k])]
    assert not differing, differing


@pytest.mark.parametrize("model_type", MODELS)
def test_wider_match_windows_shrink_the_spectral_leak(
    model_type: str, plain_data: F.SynthData, plain_steps: dict[str, Step]
) -> None:
    """The leak is intensity outside the match windows, so wider windows leave less of it.

    The main steps use a 1e-5 relative window on an untrained model, which leaves 0.75 to 0.94 of
    the predicted intensity outside it. A 2 Da window lowers the leak of every spectrum, and a
    20 Da window covers every predicted peak, so nothing leaks. This exercises the in-window side
    of the term, which the near-1 values of the main record barely touch.
    """
    leaks = [plain_steps[model_type].loss("spectral_leak_loss")]
    for rel in WIDER_TOLERANCES:
        tolerance = {"inst_type_loss_tol": {"ORB": {"rel": rel, "min_mz": 200.0}}}
        step = run_step(model_type, plain_data, PLAIN_STEP[model_type], tolerance)
        assert spec_ids(step) == PLAIN_BATCH_ORDER
        leaks.append(step.loss("spectral_leak_loss"))
    tight, wide, widest = leaks
    assert (tight > 0.7).all() and (tight <= 1.0).all(), tight
    assert (wide > 0).all() and (wide < tight).all(), (wide, tight)
    assert (widest >= 0).all() and (widest < 1e-6).all(), widest


# --------------------------------------------------------------------------------------------
# Item 5a: MASSter proposal-only training, property heads off
# --------------------------------------------------------------------------------------------


def test_proposal_only_record(proposal_runs: dict[str, Step]) -> None:
    """The proposal-only step matches golden and still computes every spectral term."""
    step = proposal_runs["only"]
    assert step.config["masster_proposal_only_until_epoch"] == 0
    assert not step.config["masster_predict_rt"] and not step.config["masster_predict_ccs"]
    assert_plain_batch(step)
    for term in (*SPECTRAL_TERMS, "proposal_loss"):
        assert th.isfinite(step.loss(term)).all(), term
    assert_step_updated_weights(step)
    ratchet_step("item5a:proposal_only", step)


def test_proposal_only_loss_is_the_weighted_proposal_loss(proposal_runs: dict[str, Step]) -> None:
    """In the warmup the total is exactly ``masster_proposal_loss_weight * proposal_loss``."""
    step = proposal_runs["only"]
    assert step.config["masster_ion_mode_proposal_loss_weight"] == 0.0
    expected = assert_proposal_only_total(step)
    # The joint step of the same config is not that: the spectral terms enter it.
    joint = proposal_runs["joint"]
    assert not th.allclose(joint.loss("loss"), expected)
    assert_joint_total(
        joint,
        extra=step.config["masster_proposal_loss_weight"] * joint.loss("proposal_loss"),
    )


def test_proposal_only_spectral_terms_carry_no_gradient(proposal_runs: dict[str, Step]) -> None:
    """The warmup trains the proposal path only: no spectral term reaches any parameter."""
    only, joint, bare = (proposal_runs[k] for k in ("only", "joint", "bare"))
    only_names, joint_names = only.grad_names(), joint.grad_names()
    # Intensity head: untouched in the warmup, trained jointly.
    assert_no_intensity_head_update(only)
    assert [n for n in joint_names if "intensity_head" in n]
    intensity_grads = [
        g for k, g in joint.record.items() if k.startswith("grad/") and "intensity_head" in k
    ]
    assert any(g.abs().sum() > 0 for g in intensity_grads)
    # The warmup gradients are a strict subset of the joint ones.
    assert set(only_names) < set(joint_names)
    # Zeroing the remaining spectral weights changes no gradient and no parameter, bit for bit.
    assert_same_update(only, bare)
    ratchet_json(
        GOLDEN,
        "item5a:grad_parameter_names",
        {"proposal_only": only_names, "joint": joint_names},
    )


def test_proposal_only_with_the_ion_mode_term(sodium_data: F.SynthData) -> None:
    """The ion-mode proposal term enters the warmup total; the spectral terms still do not.

    Spectra 5 and 6 are [M+Na]+, the only adduct here with a CMF channel besides CRF, so only
    they have a two-mode target and a positive ion-mode loss.
    """
    steps = {
        "only": run_step("masster", sodium_data, MASSTER_OVERRIDES, PROPOSAL_ONLY, ION_MODE_HEAD),
        "bare": run_step(
            "masster", sodium_data, MASSTER_OVERRIDES, PROPOSAL_ONLY, ION_MODE_HEAD, SPECTRAL_OFF
        ),
    }
    step = steps["only"]
    assert step.config["masster_ion_mode_proposal_loss_weight"] == 0.05
    assert_plain_batch(step)
    ion_mode = step.loss("ion_mode_proposal_loss")
    sodium = th.tensor([i in SODIUM_SPEC_IDS for i in spec_ids(step)])
    assert (ion_mode[sodium] > 0).all() and (ion_mode[~sodium] == 0).all(), ion_mode
    assert_proposal_only_total(step)
    head = [g for k, g in step.record.items() if k.startswith("grad/model.ion_mode_proposal_head.")]
    assert head and any(g.abs().sum() > 0 for g in head)
    assert_no_intensity_head_update(step)
    assert_same_update(step, steps["bare"])
    assert_step_updated_weights(step)
    ratchet_step("item5a:proposal_only_ion_mode", step)


# --------------------------------------------------------------------------------------------
# Item 5b: MASSter joint training with RT and CCS heads
# --------------------------------------------------------------------------------------------


def test_joint_rt_ccs_record(labeled_data: F.SynthData) -> None:
    """RT and CCS heads train jointly with the spectral terms; the record matches golden."""
    step = run_step("masster", labeled_data, MASSTER_OVERRIDES, PROPERTY_HEADS)
    config = step.config
    assert config["masster_proposal_only_until_epoch"] is None
    assert step.batch["spec_rt"].shape == (N_TRAIN,), step.batch["spec_rt"].shape
    assert_plain_batch(step)
    assert_spectral_terms(step)
    assert_reductions(step)
    n = int(step.batch["batch_size"])
    ids = spec_ids(step)
    for term in PROPERTY_LOSS_TERMS:
        assert step.loss(term).shape == (n,), term
        assert th.isfinite(step.loss(term)).all(), term
    # Labels reach the loss in batch order; missing or non-positive labels count as unlabelled.
    for name, labels in (("rt", RT_LABELS), ("ccs", CCS_LABELS)):
        valid = th.tensor([bool(labels[i] > 0) for i in ids])
        fraction = step.loss(f"{name}_label_fraction")
        assert fraction.tolist() == valid.to(fraction.dtype).tolist(), (name, fraction)
        loss, mae = step.loss(f"{name}_loss"), step.loss(f"{name}_mae")
        assert (loss[valid] > 0).all() and (mae[valid] > 0).all(), name
        assert (loss[~valid] == 0).all() and (mae[~valid] == 0).all(), name
    mape = step.loss("ccs_mape")
    assert (mape[step.loss("ccs_label_fraction") == 0] == 0).all()
    assert (mape[step.loss("ccs_label_fraction") > 0] > 0).all()
    # The total adds each head's weighted Smooth-L1 to the item-4 terms.
    extra = config["masster_rt_loss_weight"] * step.loss("rt_loss")
    extra = extra + config["masster_ccs_loss_weight"] * step.loss("ccs_loss")
    assert_joint_total(step, extra=extra)
    # Both heads are trained by the labelled spectra and move in the step.
    for head in ("rt", "ccs"):
        grads = {k: g for k, g in step.record.items() if k.startswith(f"grad/model.{head}_head.")}
        assert grads, head
        assert any(g.abs().sum() > 0 for g in grads.values()), head
        for name in (k.removeprefix("grad/") for k in grads):
            assert not th.equal(step.record[f"param_after/{name}"], step.params_before[name]), name
    assert_step_updated_weights(step)
    ratchet_step("item5b:joint_rt_ccs", step)


def test_deprecated_rt_ccs_aliases_give_the_same_joint_record(labeled_data: F.SynthData) -> None:
    """``masster_predict_rt_ccs`` with ``spec_params.rt_ccs`` is the two heads and both labels."""
    alias = run_step("masster", labeled_data, MASSTER_OVERRIDES, PROPERTY_ALIASES)
    assert alias.config["masster_predict_rt"] is False
    assert alias.config["masster_predict_ccs"] is False
    explicit = run_step("masster", labeled_data, MASSTER_OVERRIDES, PROPERTY_HEADS)
    assert sorted(alias.record) == sorted(explicit.record)
    differing = [k for k in explicit.record if not th.equal(alias.record[k], explicit.record[k])]
    assert not differing, differing


# --------------------------------------------------------------------------------------------
# Item 5c: RT/CCS heads reject proposal-only warmup
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("until_epoch", [0, 2])
@pytest.mark.parametrize("head_flag", sorted(HEAD_FLAG_PROPERTIES))
def test_property_heads_reject_proposal_only_warmup(
    head_flag: str, until_epoch: int, labeled_data: F.SynthData
) -> None:
    """Any RT/CCS head flag with ``masster_proposal_only_until_epoch`` set raises ValueError."""
    heads = {head_flag: True, "spec_params": {"rt": True, "ccs": True}}
    warmup = {**PROPOSAL_ONLY, "masster_proposal_only_until_epoch": until_epoch}
    # Control: the same heads without the warmup build, so only the combination is rejected.
    joint_config = F.make_config("masster", labeled_data, heads, enabled_losses=True)
    assert F.build_module(joint_config).property_names == HEAD_FLAG_PROPERTIES[head_flag]
    config = F.make_config("masster", labeled_data, heads, warmup, enabled_losses=True)
    with pytest.raises(ValueError, match="RT/CCS heads require joint training"):
        F.build_module(config)
